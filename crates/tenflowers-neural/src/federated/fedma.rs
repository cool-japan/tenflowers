//! Federated Matched Averaging (FedMA) — Wang et al., 2020.
//!
//! Addresses the **permutation invariance** problem in neural networks: two
//! models may represent the same function but with neurons in different orders.
//! Naively averaging misaligned neurons produces poor results.  FedMA first
//! matches neurons across clients using a greedy assignment algorithm (maximizing
//! cosine similarity) and then averages the aligned parameters.
//!
//! # Reference
//! H. Wang, M. Yurochkin, Y. Sun, D. Papailiopoulos, Y. Khazaeni,
//! "Federated Learning with Matched Averaging," ICLR 2020.

use super::types::{
    max_layer_delta_norm, weighted_avg_loss, ClientUpdate, FederatedConfig, FederatedError,
    GlobalUpdate, ModelParams,
};

// ─────────────────────────────────────────────────────────────────────────────
// Permutation matrix
// ─────────────────────────────────────────────────────────────────────────────

/// A row-permutation matrix for reordering neurons in a layer.
///
/// `perm[i] = j` means neuron `i` of the reference corresponds to neuron `j`
/// of the target (client) model.
#[derive(Debug, Clone)]
pub struct PermutationMatrix {
    /// The permutation: `perm[i] = j`.
    pub perm: Vec<usize>,
}

impl PermutationMatrix {
    /// Create an identity permutation of size `n`.
    pub fn identity(n: usize) -> Self {
        PermutationMatrix {
            perm: (0..n).collect(),
        }
    }

    /// Apply the permutation to a flat row-major weight matrix of shape
    /// `[num_neurons, fan_in]`, reordering rows according to `perm`.
    ///
    /// Returns a new weight matrix with rows reordered.
    pub fn apply_to_rows(&self, weights: &[f32], num_neurons: usize, fan_in: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; weights.len()];
        for (new_row, &old_row) in self.perm.iter().enumerate() {
            if old_row < num_neurons && new_row < num_neurons {
                let src_start = old_row * fan_in;
                let dst_start = new_row * fan_in;
                let copy_len = fan_in.min(weights.len().saturating_sub(src_start));
                out[dst_start..dst_start + copy_len]
                    .copy_from_slice(&weights[src_start..src_start + copy_len]);
            }
        }
        out
    }

    /// Apply the permutation to select columns of a weight matrix of shape
    /// `[fan_out, num_neurons]`, reordering columns.
    ///
    /// This is used to permute the **input** dimension of the next layer after
    /// permuting the neurons of the current layer.
    pub fn apply_to_cols(&self, weights: &[f32], fan_out: usize, num_neurons: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; weights.len()];
        for row in 0..fan_out {
            for (new_col, &old_col) in self.perm.iter().enumerate() {
                if old_col < num_neurons && new_col < num_neurons {
                    let src_idx = row * num_neurons + old_col;
                    let dst_idx = row * num_neurons + new_col;
                    if src_idx < weights.len() && dst_idx < out.len() {
                        out[dst_idx] = weights[src_idx];
                    }
                }
            }
        }
        out
    }

    /// Size of the permutation.
    pub fn size(&self) -> usize {
        self.perm.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer matching
// ─────────────────────────────────────────────────────────────────────────────

/// Neuron-matching algorithm via greedy column assignment.
///
/// Given a reference layer `A` (shape `[n_neurons, fan_in]`) and a target
/// layer `B` (same shape), finds the assignment `σ` such that
/// `Σ_i cosine_sim(A[i], B[σ(i)])` is maximized.
///
/// Uses a greedy approach: iteratively pick the (unmatched) pair with the
/// highest cosine similarity.  This runs in O(n²) and gives a good
/// approximation to the optimal Hungarian assignment for typical FL settings.
#[derive(Debug, Clone)]
pub struct LayerMatching {
    /// Number of neurons in the layer.
    pub num_neurons: usize,
    /// Fan-in (number of inputs per neuron).
    pub fan_in: usize,
}

impl LayerMatching {
    /// Create a new layer matcher.
    pub fn new(num_neurons: usize, fan_in: usize) -> Self {
        LayerMatching {
            num_neurons,
            fan_in,
        }
    }

    /// Compute a cosine-similarity-based greedy neuron matching.
    ///
    /// `ref_layer`: reference model weights, flat row-major `[num_neurons × fan_in]`.
    /// `client_layer`: client model weights, same shape.
    ///
    /// Returns a [`PermutationMatrix`] mapping reference neuron `i` to client
    /// neuron `perm[i]`.
    pub fn match_neurons(&self, ref_layer: &[f32], client_layer: &[f32]) -> PermutationMatrix {
        let n = self.num_neurons;
        let k = self.fan_in;

        if n == 0 || k == 0 || ref_layer.len() < n * k || client_layer.len() < n * k {
            return PermutationMatrix::identity(n);
        }

        // Precompute norms
        let ref_norms: Vec<f32> = (0..n)
            .map(|i| {
                let row = &ref_layer[i * k..(i + 1) * k];
                row.iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
                    .max(f32::EPSILON)
            })
            .collect();
        let cli_norms: Vec<f32> = (0..n)
            .map(|j| {
                let row = &client_layer[j * k..(j + 1) * k];
                row.iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
                    .max(f32::EPSILON)
            })
            .collect();

        // Compute cosine similarity matrix [n×n]
        let mut sim = vec![0.0_f32; n * n];
        for i in 0..n {
            let a = &ref_layer[i * k..(i + 1) * k];
            let na = ref_norms[i];
            for j in 0..n {
                let b = &client_layer[j * k..(j + 1) * k];
                let nb = cli_norms[j];
                let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
                sim[i * n + j] = dot / (na * nb);
            }
        }

        // Greedy matching
        let mut perm = vec![0usize; n];
        let mut ref_used = vec![false; n];
        let mut cli_used = vec![false; n];

        // Collect all (i, j, sim) triples and sort descending
        let mut candidates: Vec<(usize, usize, f32)> = (0..n)
            .flat_map(|i| {
                let sim_ref: &Vec<f32> = &sim;
                (0..n)
                    .map(move |j| (i, j, sim_ref[i * n + j]))
                    .collect::<Vec<_>>()
            })
            .collect();
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        let mut matched = 0;
        for (i, j, _) in &candidates {
            if matched == n {
                break;
            }
            if !ref_used[*i] && !cli_used[*j] {
                perm[*i] = *j;
                ref_used[*i] = true;
                cli_used[*j] = true;
                matched += 1;
            }
        }

        // Assign any unmatched ref neurons to remaining client neurons
        let mut remaining_cli: Vec<usize> = (0..n).filter(|j| !cli_used[*j]).collect();
        let mut rc_iter = remaining_cli.drain(..);
        for i in 0..n {
            if !ref_used[i] {
                if let Some(j) = rc_iter.next() {
                    perm[i] = j;
                }
            }
        }

        PermutationMatrix { perm }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FedMA Aggregator
// ─────────────────────────────────────────────────────────────────────────────

/// FedMA aggregator — matched averaging across federated clients.
///
/// For each layer, neurons from client models are matched to the reference
/// (first client's) neurons using cosine similarity, then averaged after
/// alignment.  This avoids the "neuron permutation" issue that plagues naive
/// FedAvg when aggregating non-linear networks.
///
/// # Layer shape convention
/// Each layer `l` in `ModelParams` is a flat vector of length `num_neurons * fan_in`.
/// The matcher requires knowing `(num_neurons, fan_in)` per layer, supplied via
/// `layer_shapes` at construction time.  If `layer_shapes` is `None`, the
/// aggregator falls back to FedAvg (no matching).
#[derive(Debug, Clone)]
pub struct FedMaAggregator {
    /// Shared configuration.
    pub config: FederatedConfig,
    /// Current global model parameters.
    pub global_params: ModelParams,
    /// Layer shapes: `(num_neurons, fan_in)` per layer.
    /// `None` means fall back to plain averaging.
    pub layer_shapes: Vec<Option<(usize, usize)>>,
    /// Current round.
    pub round: usize,
}

impl FedMaAggregator {
    /// Create a new FedMA aggregator.
    ///
    /// `layer_shapes`: for each layer, optionally provide `(num_neurons, fan_in)`.
    /// Pass `None` for a layer to skip matching (e.g., bias vectors, BN params).
    pub fn new(
        config: FederatedConfig,
        initial_params: ModelParams,
        layer_shapes: Vec<Option<(usize, usize)>>,
    ) -> Self {
        FedMaAggregator {
            config,
            global_params: initial_params,
            layer_shapes,
            round: 0,
        }
    }

    /// Aggregate client updates using neuron matching.
    ///
    /// Algorithm:
    /// 1. Use the first client's update as the reference.
    /// 2. For each subsequent client, match each layer's neurons to the reference.
    /// 3. Average the aligned parameter vectors.
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

        let num_layers = self.global_params.len();
        let n = updates.len();
        let total_samples: f32 = updates.iter().map(|u| u.num_samples as f32).sum();

        // Use first update as reference for alignment
        let reference = &updates[0].params;

        // For each layer: accumulate aligned values (weighted sum)
        let mut aligned_sum: ModelParams =
            reference.iter().map(|l| vec![0.0_f32; l.len()]).collect();

        for update in updates {
            let wt = update.num_samples as f32 / total_samples;
            for l in 0..num_layers.min(update.params.len()) {
                let layer_len = self.global_params[l].len();
                if update.params[l].len() != layer_len {
                    return Err(FederatedError::DimensionMismatch {
                        layer: l,
                        expected: layer_len,
                        found: update.params[l].len(),
                    });
                }

                let aligned = if let Some(Some((neurons, fan_in))) = self.layer_shapes.get(l) {
                    // Match and align this client's layer to reference
                    let matcher = LayerMatching::new(*neurons, *fan_in);
                    let perm = matcher.match_neurons(&reference[l], &update.params[l]);
                    // Apply permutation to client layer (reorder rows = neurons)
                    perm.apply_to_rows(&update.params[l], *neurons, *fan_in)
                } else {
                    // No matching: use raw values
                    update.params[l].clone()
                };

                for (j, &v) in aligned.iter().enumerate() {
                    if j < aligned_sum[l].len() {
                        aligned_sum[l][j] += wt * v;
                    }
                }
            }
        }

        let old_params = self.global_params.clone();
        self.global_params = aligned_sum;
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

    /// Distribute the current global model.
    pub fn distribute(&self) -> &ModelParams {
        &self.global_params
    }

    /// Match neurons between two layers and return the permutation.
    ///
    /// Useful for debugging or visualization purposes.
    pub fn compute_matching(
        &self,
        layer_idx: usize,
        ref_layer: &[f32],
        client_layer: &[f32],
    ) -> Option<PermutationMatrix> {
        self.layer_shapes.get(layer_idx).and_then(|shape| {
            shape.map(|(neurons, fan_in)| {
                let matcher = LayerMatching::new(neurons, fan_in);
                matcher.match_neurons(ref_layer, client_layer)
            })
        })
    }

    /// Current communication round.
    pub fn current_round(&self) -> usize {
        self.round
    }
}
