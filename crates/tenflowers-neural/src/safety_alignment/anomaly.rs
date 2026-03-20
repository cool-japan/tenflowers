//! Section 5 — Anomaly Detection & OOD
//!
//! - `IsolationForest`
//! - `AutoencoderAnomaly`
//! - `MahalanobisDetector`
//! - `EnergyOodDetector`
//! - `OodBenchmark`

use super::helpers::invert_diagonal_approx;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Isolation Forest
// ─────────────────────────────────────────────────────────────────────────────

const SENTINEL: usize = usize::MAX;

/// A node in an isolation tree.
#[derive(Debug, Clone)]
struct IsoNode {
    feature: usize,
    threshold: f64,
    left: usize,
    right: usize,
    size: usize,
}

/// Average path length normalization factor for a dataset of size `n`.
fn c_factor(n: usize) -> f64 {
    match n {
        0 | 1 => 0.0,
        2 => 1.0,
        n => {
            let hn = (n as f64 - 1.0).ln() + 0.5772156649; // Euler-Mascheroni
            2.0 * hn - 2.0 * (n as f64 - 1.0) / n as f64
        }
    }
}

fn build_isolation_tree(
    data: &[&Vec<f64>],
    depth: usize,
    n_features: usize,
    rng: &mut StdRng,
) -> Vec<IsoNode> {
    let mut nodes: Vec<IsoNode> = Vec::new();
    build_node(data, depth, n_features, 256, rng, &mut nodes);
    nodes
}

fn build_node(
    data: &[&Vec<f64>],
    depth: usize,
    n_features: usize,
    max_depth: usize,
    rng: &mut StdRng,
    nodes: &mut Vec<IsoNode>,
) -> usize {
    let n = data.len();
    let node_idx = nodes.len();

    if n <= 1 || depth >= max_depth || n_features == 0 {
        nodes.push(IsoNode {
            feature: SENTINEL,
            threshold: 0.0,
            left: SENTINEL,
            right: SENTINEL,
            size: n,
        });
        return node_idx;
    }

    let feat = (rng.random::<f64>() * n_features as f64) as usize % n_features;

    let mut feat_vals: Vec<f64> = data
        .iter()
        .map(|x| x.get(feat).copied().unwrap_or(0.0))
        .collect();
    feat_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min_val = feat_vals.first().copied().unwrap_or(0.0);
    let max_val = feat_vals.last().copied().unwrap_or(0.0);

    if (max_val - min_val).abs() < 1e-12 {
        nodes.push(IsoNode {
            feature: SENTINEL,
            threshold: 0.0,
            left: SENTINEL,
            right: SENTINEL,
            size: n,
        });
        return node_idx;
    }

    let threshold = min_val + rng.random::<f64>() * (max_val - min_val);

    let left_data: Vec<&Vec<f64>> = data
        .iter()
        .filter(|x| x.get(feat).copied().unwrap_or(0.0) <= threshold)
        .copied()
        .collect();
    let right_data: Vec<&Vec<f64>> = data
        .iter()
        .filter(|x| x.get(feat).copied().unwrap_or(0.0) > threshold)
        .copied()
        .collect();

    nodes.push(IsoNode {
        feature: feat,
        threshold,
        left: SENTINEL,
        right: SENTINEL,
        size: n,
    });

    let left_idx = build_node(&left_data, depth + 1, n_features, max_depth, rng, nodes);
    let right_idx = build_node(&right_data, depth + 1, n_features, max_depth, rng, nodes);
    nodes[node_idx].left = left_idx;
    nodes[node_idx].right = right_idx;
    node_idx
}

fn path_length(x: &[f64], nodes: &[IsoNode]) -> f64 {
    if nodes.is_empty() {
        return 0.0;
    }
    let mut idx = 0usize;
    let mut depth = 0.0f64;
    loop {
        let node = &nodes[idx];
        if node.feature == SENTINEL {
            return depth + c_factor(node.size);
        }
        let feat_val = x.get(node.feature).copied().unwrap_or(0.0);
        depth += 1.0;
        if feat_val <= node.threshold {
            if node.left == SENTINEL {
                return depth;
            }
            idx = node.left;
        } else {
            if node.right == SENTINEL {
                return depth;
            }
            idx = node.right;
        }
    }
}

/// Isolation Forest anomaly detector.
#[derive(Debug, Clone)]
pub struct IsolationForest {
    /// Number of isolation trees.
    pub n_trees: usize,
    /// Subsample size per tree.
    pub sample_size: usize,
    trees: Vec<Vec<IsoNode>>,
    n_features: usize,
}

impl IsolationForest {
    /// Create an unfitted isolation forest.
    pub fn new(n_trees: usize, sample_size: usize) -> Self {
        Self {
            n_trees: n_trees.max(1),
            sample_size: sample_size.max(2),
            trees: Vec::new(),
            n_features: 0,
        }
    }

    /// Fit on training data.
    pub fn fit(&mut self, x_train: &[Vec<f64>], seed: u64) {
        let n = x_train.len();
        if n == 0 {
            return;
        }
        self.n_features = x_train[0].len();
        let mut rng = StdRng::seed_from_u64(seed);
        let subsample = self.sample_size.min(n);

        self.trees = (0..self.n_trees)
            .map(|_| {
                let indices: Vec<usize> = (0..subsample)
                    .map(|_| (rng.random::<f64>() * n as f64) as usize % n)
                    .collect();
                let sample: Vec<&Vec<f64>> = indices.iter().map(|&i| &x_train[i]).collect();
                build_isolation_tree(&sample, 0, self.n_features, &mut rng)
            })
            .collect();
    }

    /// Compute anomaly score (higher = more anomalous, in `[0, 1]`).
    pub fn anomaly_score(&self, x: &[f64]) -> f64 {
        if self.trees.is_empty() {
            return 0.5;
        }
        let avg_path = self
            .trees
            .iter()
            .map(|tree| path_length(x, tree))
            .sum::<f64>()
            / self.trees.len() as f64;
        let cn = c_factor(self.sample_size);
        2.0_f64.powf(-avg_path / cn)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Autoencoder Anomaly
// ─────────────────────────────────────────────────────────────────────────────

/// Autoencoder-based anomaly detector using reconstruction MSE.
#[derive(Debug, Clone)]
pub struct AutoencoderAnomaly {
    /// Encoder weights.
    pub encoder: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    /// Decoder weights.
    pub decoder: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    /// Anomaly threshold.
    pub threshold: f64,
}

impl AutoencoderAnomaly {
    /// Create with random weights; `layer_sizes` defines encoder architecture.
    pub fn new(layer_sizes: &[usize], seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut encoder = Vec::new();
        let mut decoder = Vec::new();

        for i in 0..layer_sizes.len().saturating_sub(1) {
            let in_dim = layer_sizes[i];
            let out_dim = layer_sizes[i + 1];
            let scale = (2.0 / in_dim as f64).sqrt();
            let weights: Vec<Vec<f64>> = (0..out_dim)
                .map(|_| {
                    (0..in_dim)
                        .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                        .collect()
                })
                .collect();
            encoder.push((weights, vec![0.0; out_dim]));
        }

        let n = layer_sizes.len();
        for i in (1..n).rev() {
            let in_dim = layer_sizes[i];
            let out_dim = layer_sizes[i - 1];
            let scale = (2.0 / in_dim as f64).sqrt();
            let weights: Vec<Vec<f64>> = (0..out_dim)
                .map(|_| {
                    (0..in_dim)
                        .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                        .collect()
                })
                .collect();
            decoder.push((weights, vec![0.0; out_dim]));
        }

        Self {
            encoder,
            decoder,
            threshold: 0.1,
        }
    }

    fn forward(x: &[f64], layers: &[(Vec<Vec<f64>>, Vec<f64>)]) -> Vec<f64> {
        let mut h = x.to_vec();
        for (w, b) in layers {
            let out_dim = w.len();
            let mut new_h = vec![0.0; out_dim];
            for (j, (row, &bias)) in w.iter().zip(b.iter()).enumerate() {
                let s: f64 = h.iter().zip(row.iter()).map(|(x, w)| x * w).sum::<f64>() + bias;
                new_h[j] = s.max(0.0);
            }
            h = new_h;
        }
        h
    }

    /// Compute reconstruction MSE for `x`.
    pub fn score(&self, x: &[f64]) -> f64 {
        let encoded = Self::forward(x, &self.encoder);
        let reconstructed = Self::forward(&encoded, &self.decoder);
        if reconstructed.is_empty() || x.is_empty() {
            return f64::INFINITY;
        }
        let n = x.len().min(reconstructed.len());
        x.iter()
            .zip(reconstructed.iter())
            .take(n)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / n as f64
    }

    /// Returns `true` if reconstruction error exceeds threshold.
    pub fn is_anomalous(&self, x: &[f64]) -> bool {
        self.score(x) > self.threshold
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mahalanobis Detector
// ─────────────────────────────────────────────────────────────────────────────

/// Mahalanobis distance-based OOD detector.
#[derive(Debug, Clone)]
pub struct MahalanobisDetector {
    /// Fitted mean vector.
    pub mean: Vec<f64>,
    /// Fitted precision matrix (diagonal approximation).
    pub precision: Vec<Vec<f64>>,
}

impl MahalanobisDetector {
    /// Create an unfitted detector.
    pub fn new() -> Self {
        Self {
            mean: Vec::new(),
            precision: Vec::new(),
        }
    }

    /// Fit on training data.
    pub fn fit(&mut self, x_train: &[Vec<f64>]) {
        let n = x_train.len();
        if n == 0 {
            return;
        }
        let d = x_train[0].len();
        if d == 0 {
            return;
        }

        let mut mean = vec![0.0; d];
        for x in x_train {
            for (j, &v) in x.iter().enumerate() {
                mean[j] += v;
            }
        }
        mean.iter_mut().for_each(|v| *v /= n as f64);

        let mut cov = vec![vec![0.0; d]; d];
        for x in x_train {
            for i in 0..d {
                for j in 0..d {
                    let xi = x.get(i).copied().unwrap_or(0.0) - mean[i];
                    let xj = x.get(j).copied().unwrap_or(0.0) - mean[j];
                    cov[i][j] += xi * xj;
                }
            }
        }
        for row in &mut cov {
            for v in row.iter_mut() {
                *v /= (n - 1).max(1) as f64;
            }
        }
        for i in 0..d {
            cov[i][i] += 1e-6; // regularize
        }

        self.mean = mean;
        self.precision = invert_diagonal_approx(&cov);
    }

    /// Compute Mahalanobis distance of `x` to the fitted distribution.
    pub fn score(&self, x: &[f64]) -> f64 {
        let d = self.mean.len();
        if d == 0 || x.len() != d {
            return f64::INFINITY;
        }
        let diff: Vec<f64> = x
            .iter()
            .zip(self.mean.iter())
            .map(|(xi, mi)| xi - mi)
            .collect();

        let prec_diff: Vec<f64> = (0..d)
            .map(|i| {
                diff.iter()
                    .zip(self.precision[i].iter())
                    .map(|(dj, pij)| dj * pij)
                    .sum()
            })
            .collect();

        diff.iter()
            .zip(prec_diff.iter())
            .map(|(di, pd)| di * pd)
            .sum::<f64>()
            .max(0.0)
            .sqrt()
    }
}

impl Default for MahalanobisDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Energy OOD Detector
// ─────────────────────────────────────────────────────────────────────────────

/// Energy-based OOD detector: `E(x) = -log Σ_i exp(f_i(x))`.
#[derive(Debug, Clone, Default)]
pub struct EnergyOodDetector {
    /// Temperature scaling factor.
    pub temperature: f64,
}

impl EnergyOodDetector {
    /// Create a new detector.
    pub fn new(temperature: f64) -> Self {
        Self {
            temperature: temperature.max(1e-9),
        }
    }

    /// Compute energy score; higher = OOD.
    pub fn energy_score(&self, logits: &[f64]) -> f64 {
        if logits.is_empty() {
            return 0.0;
        }
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = logits
            .iter()
            .map(|&l| ((l - max_logit) / self.temperature).exp())
            .sum();
        -(max_logit / self.temperature + sum_exp.ln())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OOD Benchmark
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluates AUROC of OOD detection.
#[derive(Debug, Clone, Default)]
pub struct OodBenchmark;

impl OodBenchmark {
    /// Create a new benchmark evaluator.
    pub fn new() -> Self {
        Self
    }

    /// Compute AUROC (ID scores lower, OOD scores higher).
    pub fn auroc(&self, id_scores: &[f64], ood_scores: &[f64]) -> f64 {
        if id_scores.is_empty() || ood_scores.is_empty() {
            return 0.5;
        }
        let n_id = id_scores.len();
        let n_ood = ood_scores.len();
        let mut u_stat = 0.0f64;
        for &ood in ood_scores {
            for &id in id_scores {
                if ood > id {
                    u_stat += 1.0;
                } else if (ood - id).abs() < 1e-12 {
                    u_stat += 0.5;
                }
            }
        }
        u_stat / (n_id * n_ood) as f64
    }
}
