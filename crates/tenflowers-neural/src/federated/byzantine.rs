//! Byzantine-robust aggregation algorithms for federated learning.
//!
//! Provides several aggregation rules that tolerate up to `f` Byzantine
//! (arbitrarily malicious) clients:
//!
//! | Algorithm | Reference |
//! |-----------|-----------|
//! | [`KrumAggregator`] | Blanchard et al., 2017 |
//! | [`FlameAggregator`] | Nguyen et al., 2022 |
//! | [`MedianAggregator`] | coordinate-wise median |
//! | [`TrimmedMeanAggregator`] | trimmed mean |
//! | [`BulyanAggregator`] | El Mhamdi et al., 2018 |

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

use super::types::{
    max_layer_delta_norm, weighted_avg_loss, ClientUpdate, FederatedConfig, FederatedError,
    GlobalUpdate, ModelParams,
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Flatten a `ModelParams` (Vec<Vec<f32>>) into a single `Vec<f32>`.
fn flatten(params: &ModelParams) -> Vec<f32> {
    params.iter().flat_map(|l| l.iter().copied()).collect()
}

/// Squared Euclidean distance between two flat vectors.
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

/// Cosine similarity between two flat vectors (returns 0 if either is zero).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < f32::EPSILON || nb < f32::EPSILON {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Reconstruct a `ModelParams` from a flat vector given the layer shapes.
fn unflatten(flat: &[f32], shapes: &[usize]) -> ModelParams {
    let mut result: ModelParams = Vec::with_capacity(shapes.len());
    let mut offset = 0;
    for &n in shapes {
        result.push(flat[offset..offset + n].to_vec());
        offset += n;
    }
    result
}

/// Extract layer shapes from a `ModelParams`.
fn layer_shapes(params: &ModelParams) -> Vec<usize> {
    params.iter().map(|l| l.len()).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Krum / Multi-Krum
// ─────────────────────────────────────────────────────────────────────────────

/// Krum aggregation rule — Blanchard et al., NIPS 2017.
///
/// For each client `i`, compute its Krum score as the sum of squared distances
/// to its `n − f − 2` nearest neighbors (where `f` is the assumed number of
/// Byzantine clients, `n` is the total number of clients).
///
/// **Krum** selects the client with the minimum score and uses its update as
/// the global model.
///
/// **Multi-Krum** selects the `m` clients with smallest scores and averages
/// their updates, trading off robustness for accuracy.
///
/// # Reference
/// P. Blanchard, R. Guerraoui, J. Stainer, "Machine Learning with Adversaries:
/// Byzantine Tolerant Gradient Descent," NeurIPS 2017.
#[derive(Debug, Clone)]
pub struct KrumAggregator {
    /// Shared configuration.
    pub config: FederatedConfig,
    /// Current global model parameters.
    pub global_params: ModelParams,
    /// Assumed number of Byzantine clients.
    pub f: usize,
    /// Multi-Krum: number of clients to select (1 = standard Krum).
    pub m: usize,
    /// Current round index.
    pub round: usize,
}

impl KrumAggregator {
    /// Create a new Krum aggregator.
    ///
    /// `f` is the assumed Byzantine fault tolerance bound.
    /// `m = 1` gives standard Krum; `m > 1` gives Multi-Krum.
    pub fn new(
        config: FederatedConfig,
        initial_params: ModelParams,
        f: usize,
        m: usize,
    ) -> Result<Self, FederatedError> {
        let n = config.clients_per_round;
        if 2 * f + 2 >= n {
            return Err(FederatedError::InvalidConfig {
                reason: format!("Krum requires n > 2f+2; got n={n}, f={f}"),
            });
        }
        let m_clamped = m.max(1).min(n - f);
        Ok(KrumAggregator {
            config,
            global_params: initial_params,
            f,
            m: m_clamped,
            round: 0,
        })
    }

    /// Aggregate using (Multi-)Krum selection.
    pub fn aggregate(&mut self, updates: &[ClientUpdate]) -> Result<GlobalUpdate, FederatedError> {
        if updates.is_empty() {
            return Err(FederatedError::NoClientUpdates);
        }
        let n = updates.len();
        if n < self.config.min_clients_available {
            return Err(FederatedError::InsufficientClients {
                required: self.config.min_clients_available,
                available: n,
            });
        }
        if 2 * self.f + 2 >= n {
            return Err(FederatedError::InvalidConfig {
                reason: format!("Krum requires n > 2f+2; got n={n}, f={}", self.f),
            });
        }

        let neighbors = n - self.f - 2; // number of nearest neighbors to sum

        // Flatten all parameter vectors
        let flat: Vec<Vec<f32>> = updates.iter().map(|u| flatten(&u.params)).collect();

        // Compute pairwise squared distances
        let mut scores: Vec<f32> = vec![0.0; n];
        for i in 0..n {
            let mut dists: Vec<f32> = (0..n)
                .filter(|&j| j != i)
                .map(|j| sq_dist(&flat[i], &flat[j]))
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            scores[i] = dists[..neighbors].iter().sum();
        }

        // Select top-m clients by lowest score
        let m = self.m.min(n);
        let mut ranked: Vec<usize> = (0..n).collect();
        ranked.sort_by(|&a, &b| {
            scores[a]
                .partial_cmp(&scores[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let selected = &ranked[..m];

        // Average selected clients
        let shapes = layer_shapes(&self.global_params);
        let total_dim: usize = shapes.iter().sum();
        let mut avg = vec![0.0_f32; total_dim];
        let w = 1.0 / m as f32;
        for &idx in selected {
            for (a, &v) in avg.iter_mut().zip(flat[idx].iter()) {
                *a += w * v;
            }
        }

        let old_params = self.global_params.clone();
        self.global_params = unflatten(&avg, &shapes);
        let convergence_metric = max_layer_delta_norm(&old_params, &self.global_params);
        let avg_loss = weighted_avg_loss(updates);
        self.round += 1;

        Ok(GlobalUpdate {
            round: self.round,
            params: self.global_params.clone(),
            participating_clients: m,
            avg_loss,
            convergence_metric,
        })
    }

    /// Return a reference to the current global parameters.
    pub fn distribute(&self) -> &ModelParams {
        &self.global_params
    }

    /// Current communication round.
    pub fn current_round(&self) -> usize {
        self.round
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FLAME
// ─────────────────────────────────────────────────────────────────────────────

/// FLAME aggregation rule — Nguyen et al., 2022.
///
/// Combines three defences:
/// 1. **Cosine-similarity clustering**: clients whose cosine similarity to the
///    majority cluster centroid falls below `cos_threshold` are rejected.
/// 2. **Adaptive norm clipping**: each accepted update is clipped to its own
///    median L2 norm, preventing magnitude attacks.
/// 3. **Gaussian noise addition**: small noise is added to the aggregated
///    model to preserve differential privacy.
///
/// # Reference
/// T. D. Nguyen, P. Rieger, R. De Viti, H. Chen, B. Brandenburg, H. Yalame,
/// H. Möllering, H. Fereidooni, S. Zeitouni, M. Schmitt et al.,
/// "FLAME: Taming Backdoors in Federated Learning," USENIX Security 2022.
#[derive(Debug, Clone)]
pub struct FlameAggregator {
    /// Shared configuration.
    pub config: FederatedConfig,
    /// Current global model parameters.
    pub global_params: ModelParams,
    /// Cosine similarity threshold for cluster membership.
    pub cos_threshold: f32,
    /// Noise multiplier for the final DP noise injection.
    pub noise_multiplier: f32,
    /// Current round.
    pub round: usize,
}

impl FlameAggregator {
    /// Create a new FLAME aggregator.
    ///
    /// `cos_threshold` ∈ [-1, 1]: clients whose cosine similarity to the
    /// cluster mean is below this are rejected.  A typical value is 0.0.
    ///
    /// `noise_multiplier`: standard deviation of the final Gaussian noise
    /// relative to the clipping norm.
    pub fn new(
        config: FederatedConfig,
        initial_params: ModelParams,
        cos_threshold: f32,
        noise_multiplier: f32,
    ) -> Self {
        FlameAggregator {
            config,
            global_params: initial_params,
            cos_threshold,
            noise_multiplier,
            round: 0,
        }
    }

    /// Aggregate using FLAME.
    pub fn aggregate(&mut self, updates: &[ClientUpdate]) -> Result<GlobalUpdate, FederatedError> {
        if updates.is_empty() {
            return Err(FederatedError::NoClientUpdates);
        }
        if updates.len() < self.config.min_clients_available {
            return Err(FederatedError::InsufficientClients {
                required: self.config.min_clients_available,
                available: updates.len(),
            });
        }

        let n = updates.len();
        let flat: Vec<Vec<f32>> = updates.iter().map(|u| flatten(&u.params)).collect();
        let dim = flat[0].len();

        // Step 1: compute naive mean as cluster reference
        let mut centroid = vec![0.0_f32; dim];
        let w = 1.0 / n as f32;
        for v in &flat {
            for (c, &x) in centroid.iter_mut().zip(v.iter()) {
                *c += w * x;
            }
        }

        // Step 2: filter by cosine similarity to centroid
        let accepted: Vec<usize> = (0..n)
            .filter(|&i| cosine_similarity(&flat[i], &centroid) >= self.cos_threshold)
            .collect();

        // Fall back to all clients if none pass the threshold
        let selected_idx = if accepted.is_empty() {
            (0..n).collect::<Vec<_>>()
        } else {
            accepted
        };

        // Step 3: compute median L2 norm for adaptive clipping
        let mut norms: Vec<f32> = selected_idx
            .iter()
            .map(|&i| flat[i].iter().map(|x| x * x).sum::<f32>().sqrt())
            .collect();
        norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_norm = if norms.is_empty() {
            1.0
        } else {
            norms[norms.len() / 2]
        };
        let clip_norm = median_norm.max(f32::EPSILON);

        // Step 4: clip each accepted update to `clip_norm` and average
        let shapes = layer_shapes(&self.global_params);
        let m = selected_idx.len();
        let mut avg = vec![0.0_f32; dim];
        let iw = 1.0 / m as f32;
        for &i in &selected_idx {
            let v = &flat[i];
            let norm = v
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
                .max(f32::EPSILON);
            let scale = (clip_norm / norm).min(1.0);
            for (a, &x) in avg.iter_mut().zip(v.iter()) {
                *a += iw * scale * x;
            }
        }

        // Step 5: add Gaussian noise
        let sigma = self.noise_multiplier * clip_norm;
        if sigma > f32::EPSILON {
            let seed: u64 = self.global_params.iter().flat_map(|l| l.iter()).fold(
                0x517cc1b727220a95_u64,
                |acc, &v| {
                    acc.wrapping_mul(0x9e3779b97f4a7c15)
                        .wrapping_add(v.to_bits() as u64)
                },
            );
            let mut rng = StdRng::seed_from_u64(seed ^ (self.round as u64 * 0x1234567));
            let mut i = 0;
            while i < dim {
                let u1: f64 = rng.random::<f64>().max(1e-10);
                let u2: f64 = rng.random::<f64>();
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = std::f64::consts::TAU * u2;
                avg[i] += (r * theta.cos()) as f32 * sigma;
                i += 1;
                if i < dim {
                    avg[i] += (r * theta.sin()) as f32 * sigma;
                    i += 1;
                }
            }
        }

        let old_params = self.global_params.clone();
        self.global_params = unflatten(&avg, &shapes);
        let convergence_metric = max_layer_delta_norm(&old_params, &self.global_params);
        let avg_loss = weighted_avg_loss(updates);
        self.round += 1;

        Ok(GlobalUpdate {
            round: self.round,
            params: self.global_params.clone(),
            participating_clients: m,
            avg_loss,
            convergence_metric,
        })
    }

    /// Return a reference to the current global parameters.
    pub fn distribute(&self) -> &ModelParams {
        &self.global_params
    }

    /// Current communication round.
    pub fn current_round(&self) -> usize {
        self.round
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Coordinate-wise Median
// ─────────────────────────────────────────────────────────────────────────────

/// Coordinate-wise median aggregation.
///
/// For each parameter coordinate, takes the median value across all client
/// updates.  Provides Byzantine robustness as long as fewer than half the
/// clients are Byzantine (f < n/2).
///
/// # Complexity
/// O(n log n) per coordinate due to sorting.
#[derive(Debug, Clone)]
pub struct MedianAggregator {
    /// Shared configuration.
    pub config: FederatedConfig,
    /// Current global model parameters.
    pub global_params: ModelParams,
    /// Current round.
    pub round: usize,
}

impl MedianAggregator {
    /// Create a new coordinate-wise median aggregator.
    pub fn new(config: FederatedConfig, initial_params: ModelParams) -> Self {
        MedianAggregator {
            config,
            global_params: initial_params,
            round: 0,
        }
    }

    /// Aggregate via coordinate-wise median.
    pub fn aggregate(&mut self, updates: &[ClientUpdate]) -> Result<GlobalUpdate, FederatedError> {
        if updates.is_empty() {
            return Err(FederatedError::NoClientUpdates);
        }
        if updates.len() < self.config.min_clients_available {
            return Err(FederatedError::InsufficientClients {
                required: self.config.min_clients_available,
                available: updates.len(),
            });
        }

        let n = updates.len();
        let num_layers = self.global_params.len();

        let mut new_params: ModelParams = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let layer_len = self.global_params[l].len();
            let mut median_layer = Vec::with_capacity(layer_len);
            for j in 0..layer_len {
                let mut vals: Vec<f32> = updates.iter().map(|u| u.params[l][j]).collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = n / 2;
                let median = if n % 2 == 1 {
                    vals[mid]
                } else {
                    (vals[mid - 1] + vals[mid]) * 0.5
                };
                median_layer.push(median);
            }
            new_params.push(median_layer);
        }

        let old_params = self.global_params.clone();
        self.global_params = new_params;
        let convergence_metric = max_layer_delta_norm(&old_params, &self.global_params);
        let avg_loss = weighted_avg_loss(updates);
        self.round += 1;

        Ok(GlobalUpdate {
            round: self.round,
            params: self.global_params.clone(),
            participating_clients: n,
            avg_loss,
            convergence_metric,
        })
    }

    /// Return a reference to the current global parameters.
    pub fn distribute(&self) -> &ModelParams {
        &self.global_params
    }

    /// Current communication round.
    pub fn current_round(&self) -> usize {
        self.round
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trimmed Mean
// ─────────────────────────────────────────────────────────────────────────────

/// Trimmed mean aggregation.
///
/// For each coordinate, sorts the values from all clients and drops the top
/// and bottom `β` fraction, then averages the remaining values.
///
/// Tolerates up to `β·n` Byzantine clients (roughly).
#[derive(Debug, Clone)]
pub struct TrimmedMeanAggregator {
    /// Shared configuration.
    pub config: FederatedConfig,
    /// Current global model parameters.
    pub global_params: ModelParams,
    /// Fraction to trim from each tail (0 < β < 0.5).
    pub beta: f32,
    /// Current round.
    pub round: usize,
}

impl TrimmedMeanAggregator {
    /// Create a new trimmed mean aggregator.
    ///
    /// `beta` ∈ (0, 0.5): fraction of values dropped from each tail.
    pub fn new(
        config: FederatedConfig,
        initial_params: ModelParams,
        beta: f32,
    ) -> Result<Self, FederatedError> {
        if beta <= 0.0 || beta >= 0.5 {
            return Err(FederatedError::InvalidConfig {
                reason: format!("beta must be in (0, 0.5); got {beta}"),
            });
        }
        Ok(TrimmedMeanAggregator {
            config,
            global_params: initial_params,
            beta,
            round: 0,
        })
    }

    /// Aggregate using trimmed mean.
    pub fn aggregate(&mut self, updates: &[ClientUpdate]) -> Result<GlobalUpdate, FederatedError> {
        if updates.is_empty() {
            return Err(FederatedError::NoClientUpdates);
        }
        let n = updates.len();
        if n < self.config.min_clients_available {
            return Err(FederatedError::InsufficientClients {
                required: self.config.min_clients_available,
                available: n,
            });
        }

        let trim = (n as f32 * self.beta).floor() as usize;
        let keep_start = trim;
        let keep_end = n.saturating_sub(trim);
        if keep_start >= keep_end {
            return Err(FederatedError::InvalidConfig {
                reason: format!("too many trimmed; n={n}, beta={}, trim={trim}", self.beta),
            });
        }

        let num_layers = self.global_params.len();
        let mut new_params: ModelParams = Vec::with_capacity(num_layers);

        for l in 0..num_layers {
            let layer_len = self.global_params[l].len();
            let mut trimmed_layer = Vec::with_capacity(layer_len);
            for j in 0..layer_len {
                let mut vals: Vec<f32> = updates.iter().map(|u| u.params[l][j]).collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let kept = &vals[keep_start..keep_end];
                let mean = kept.iter().sum::<f32>() / kept.len() as f32;
                trimmed_layer.push(mean);
            }
            new_params.push(trimmed_layer);
        }

        let old_params = self.global_params.clone();
        self.global_params = new_params;
        let convergence_metric = max_layer_delta_norm(&old_params, &self.global_params);
        let avg_loss = weighted_avg_loss(updates);
        self.round += 1;

        Ok(GlobalUpdate {
            round: self.round,
            params: self.global_params.clone(),
            participating_clients: keep_end - keep_start,
            avg_loss,
            convergence_metric,
        })
    }

    /// Return a reference to the current global parameters.
    pub fn distribute(&self) -> &ModelParams {
        &self.global_params
    }

    /// Current communication round.
    pub fn current_round(&self) -> usize {
        self.round
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bulyan
// ─────────────────────────────────────────────────────────────────────────────

/// Bulyan aggregation rule — El Mhamdi et al., ICML 2018.
///
/// Two-phase defence:
/// 1. **Selection phase**: run Krum `n − 2f` times to select a set of
///    "Krum-trusted" client updates.
/// 2. **Aggregation phase**: apply coordinate-wise trimmed mean on the
///    selected updates, trimming `f` values from each tail.
///
/// Requires `n ≥ 4f + 3`.
///
/// # Reference
/// E. M. El Mhamdi, R. Guerraoui, S. Rouault,
/// "The Hidden Vulnerability of Distributed Learning in Byzantium," ICML 2018.
#[derive(Debug, Clone)]
pub struct BulyanAggregator {
    /// Shared configuration.
    pub config: FederatedConfig,
    /// Current global model parameters.
    pub global_params: ModelParams,
    /// Assumed number of Byzantine clients.
    pub f: usize,
    /// Current round.
    pub round: usize,
}

impl BulyanAggregator {
    /// Create a new Bulyan aggregator.
    pub fn new(
        config: FederatedConfig,
        initial_params: ModelParams,
        f: usize,
    ) -> Result<Self, FederatedError> {
        let n = config.clients_per_round;
        if n < 4 * f + 3 {
            return Err(FederatedError::InvalidConfig {
                reason: format!("Bulyan requires n >= 4f+3; got n={n}, f={f}"),
            });
        }
        Ok(BulyanAggregator {
            config,
            global_params: initial_params,
            f,
            round: 0,
        })
    }

    /// Aggregate using Bulyan.
    pub fn aggregate(&mut self, updates: &[ClientUpdate]) -> Result<GlobalUpdate, FederatedError> {
        if updates.is_empty() {
            return Err(FederatedError::NoClientUpdates);
        }
        let n = updates.len();
        if n < self.config.min_clients_available {
            return Err(FederatedError::InsufficientClients {
                required: self.config.min_clients_available,
                available: n,
            });
        }
        if n < 4 * self.f + 3 {
            return Err(FederatedError::InvalidConfig {
                reason: format!("Bulyan requires n >= 4f+3; got n={n}, f={}", self.f),
            });
        }

        let flat: Vec<Vec<f32>> = updates.iter().map(|u| flatten(&u.params)).collect();
        let num_select = n.saturating_sub(2 * self.f);
        let neighbors = n.saturating_sub(self.f + 2);

        // Phase 1: iterative Krum selection
        let mut remaining: Vec<usize> = (0..n).collect();
        let mut selected: Vec<usize> = Vec::with_capacity(num_select);

        while selected.len() < num_select && remaining.len() > 2 {
            let nn = remaining.len().min(neighbors + 1);
            let nb = nn.saturating_sub(1);

            // Compute Krum scores within `remaining`
            let mut scores: Vec<(usize, f32)> = remaining
                .iter()
                .map(|&i| {
                    let mut dists: Vec<f32> = remaining
                        .iter()
                        .filter(|&&j| j != i)
                        .map(|&j| sq_dist(&flat[i], &flat[j]))
                        .collect();
                    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let score: f32 = dists[..nb.min(dists.len())].iter().sum();
                    (i, score)
                })
                .collect();

            scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let best = scores[0].0;
            selected.push(best);
            remaining.retain(|&x| x != best);
        }

        // Phase 2: coordinate-wise trimmed mean on selected
        let sel_n = selected.len();
        let trim = self.f.min(sel_n / 2);
        let keep_start = trim;
        let keep_end = sel_n.saturating_sub(trim);
        let keep_count = keep_end.saturating_sub(keep_start).max(1);

        let shapes = layer_shapes(&self.global_params);
        let dim: usize = shapes.iter().sum();
        let mut avg = vec![0.0_f32; dim];

        for coord in 0..dim {
            let mut vals: Vec<f32> = selected.iter().map(|&i| flat[i][coord]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let k_end = keep_end.min(vals.len());
            let k_start = keep_start.min(k_end);
            let kept = &vals[k_start..k_end];
            avg[coord] = if kept.is_empty() {
                vals.iter().sum::<f32>() / vals.len() as f32
            } else {
                kept.iter().sum::<f32>() / keep_count as f32
            };
        }

        let old_params = self.global_params.clone();
        self.global_params = unflatten(&avg, &shapes);
        let convergence_metric = max_layer_delta_norm(&old_params, &self.global_params);
        let avg_loss = weighted_avg_loss(updates);
        self.round += 1;

        Ok(GlobalUpdate {
            round: self.round,
            params: self.global_params.clone(),
            participating_clients: sel_n,
            avg_loss,
            convergence_metric,
        })
    }

    /// Return a reference to the current global parameters.
    pub fn distribute(&self) -> &ModelParams {
        &self.global_params
    }

    /// Current communication round.
    pub fn current_round(&self) -> usize {
        self.round
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Byzantine Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Metrics for evaluating Byzantine-robust aggregation.
#[derive(Debug, Clone, Default)]
pub struct ByzantineMetrics {
    /// Fraction of detected Byzantine clients (TP / (TP + FN)).
    pub detection_rate: f32,
    /// Accuracy of the global model on clean (non-adversarial) data.
    pub clean_accuracy: f32,
    /// Fraction of poisoning attacks that succeeded.
    pub poisoning_success_rate: f32,
    /// Number of rounds evaluated.
    pub num_rounds: usize,
    /// Average L2 distance between aggregated result and true mean.
    pub avg_error_norm: f32,
}

impl ByzantineMetrics {
    /// Create a new zeroed metrics report.
    pub fn new() -> Self {
        ByzantineMetrics::default()
    }

    /// Compute detection rate given Byzantine indices and rejected set.
    ///
    /// `byzantine_ids`: ground-truth Byzantine client indices.
    /// `rejected_ids`: indices rejected by the aggregation rule.
    pub fn compute_detection_rate(&mut self, byzantine_ids: &[usize], rejected_ids: &[usize]) {
        if byzantine_ids.is_empty() {
            self.detection_rate = 1.0; // trivially satisfied
            return;
        }
        let detected = byzantine_ids
            .iter()
            .filter(|id| rejected_ids.contains(id))
            .count();
        self.detection_rate = detected as f32 / byzantine_ids.len() as f32;
    }

    /// Compute the L2 error between the aggregated model and the reference
    /// (e.g., the average of honest client updates).
    pub fn compute_error_norm(&mut self, aggregated: &ModelParams, reference: &ModelParams) {
        let sq_err: f32 = aggregated
            .iter()
            .zip(reference.iter())
            .flat_map(|(al, rl)| al.iter().zip(rl.iter()).map(|(a, r)| (a - r) * (a - r)))
            .sum();
        let err = sq_err.sqrt();
        // Running average
        self.avg_error_norm = if self.num_rounds == 0 {
            err
        } else {
            (self.avg_error_norm * self.num_rounds as f32 + err) / (self.num_rounds + 1) as f32
        };
        self.num_rounds += 1;
    }
}
