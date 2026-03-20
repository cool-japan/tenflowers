//! Advanced sparse learning algorithms: Sparse Transformers, Structured Sparsity,
//! Sparse Coding Layers, and Compressed Sensing utilities.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use super::{box_muller, dot, norm, LassoEncoder};

// ── BigBirdAttention ──────────────────────────────────────────────────────────

/// BigBird attention pattern (Zaheer et al. 2020): block-sparse + random + global.
///
/// Combines three types of attention:
/// - Block-sparse: local sliding window of blocks
/// - Random: random token connections for long-range coverage
/// - Global: special tokens that attend to/from all positions
pub struct BigBirdAttention {
    /// Sequence length (number of tokens).
    pub seq_len: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Block size for block-sparse attention.
    pub block_size: usize,
    /// Number of random attention connections per token.
    pub n_random: usize,
    /// Number of global tokens (e.g., CLS, SEP).
    pub n_global: usize,
}

impl BigBirdAttention {
    /// Generate the BigBird attention mask for a sequence.
    ///
    /// Returns a boolean matrix of shape (seq_len × seq_len) where `true`
    /// means the query at row i can attend to key at column j.
    pub fn compute_attention_mask(&self, rng: &mut impl Rng) -> Vec<Vec<bool>> {
        let n = self.seq_len;
        let mut mask = vec![vec![false; n]; n];

        // 1) Block-sparse: each token attends to its local block neighbours
        let bs = self.block_size.max(1);
        for i in 0..n {
            let block_i = i / bs;
            // Attend to tokens in the adjacent blocks (block_i-1, block_i, block_i+1)
            let start_block = block_i.saturating_sub(1);
            let end_block = (block_i + 2).min((n + bs - 1) / bs);
            for b in start_block..end_block {
                let start_tok = b * bs;
                let end_tok = ((b + 1) * bs).min(n);
                for j in start_tok..end_tok {
                    mask[i][j] = true;
                }
            }
        }

        // 2) Random: each token additionally attends to `n_random` random tokens
        for i in 0..n {
            for _ in 0..self.n_random {
                let j = rng.random_range(0..n);
                mask[i][j] = true;
            }
        }

        // 3) Global: first n_global tokens attend to/from all tokens
        for g in 0..self.n_global.min(n) {
            for j in 0..n {
                mask[g][j] = true;
                mask[j][g] = true;
            }
        }

        mask
    }

    /// Compute sparse attention output for a single head.
    ///
    /// `q`, `k`, `v` are (seq_len × head_dim) matrices stored row-major.
    /// Only positions allowed by `mask` contribute to the attention sum.
    pub fn sparse_attention(
        &self,
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        mask: &[Vec<bool>],
    ) -> Vec<Vec<f32>> {
        let n = self.seq_len.min(q.len());
        let d = self.head_dim as f32;
        let scale = 1.0 / d.sqrt();
        let mut output = vec![vec![0.0_f32; self.head_dim]; n];

        for i in 0..n {
            // Gather allowed keys for query i
            let allowed: Vec<usize> = (0..n).filter(|&j| mask[i].get(j).copied().unwrap_or(false)).collect();
            if allowed.is_empty() {
                continue;
            }

            // Compute scaled dot-product scores
            let scores: Vec<f32> = allowed
                .iter()
                .map(|&j| dot(&q[i], &k[j]) * scale)
                .collect();

            // Softmax over allowed positions
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum::<f32>().max(1e-9);

            // Weighted sum of values
            for (pos, &j) in allowed.iter().enumerate() {
                let weight = exp_scores[pos] / sum_exp;
                for (out, &vval) in output[i].iter_mut().zip(v[j].iter()) {
                    *out += weight * vval;
                }
            }
        }
        output
    }

    /// Count the number of active attention connections in a mask.
    pub fn mask_sparsity(mask: &[Vec<bool>]) -> f32 {
        let total = (mask.len() * mask.first().map(|r| r.len()).unwrap_or(0)) as f32;
        if total < 1.0 {
            return 0.0;
        }
        let active: usize = mask.iter().flat_map(|row| row.iter()).filter(|&&b| b).count();
        active as f32 / total
    }
}

// ── SparseSlidingWindowAttention ──────────────────────────────────────────────

/// Sparse sliding-window attention pattern (Beltagy et al. Longformer 2020): local window + global tokens.
///
/// Local window attention ensures O(n·w) complexity instead of O(n²).
/// Named `SparseSlidingWindowAttention` to avoid conflict with `LongformerAttention` in `moe_scaling`.
pub struct SparseSlidingWindowAttention {
    /// Sequence length.
    pub seq_len: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// One-sided window radius (attends to `window_size` tokens left and right).
    pub window_size: usize,
    /// Indices of global tokens (attend to/from all positions).
    pub global_tokens: Vec<usize>,
}

impl SparseSlidingWindowAttention {
    /// Compute the sliding-window + global attention mask.
    pub fn compute_attention_mask(&self) -> Vec<Vec<bool>> {
        let n = self.seq_len;
        let mut mask = vec![vec![false; n]; n];

        // Sliding window
        for i in 0..n {
            let start = i.saturating_sub(self.window_size);
            let end = (i + self.window_size + 1).min(n);
            for j in start..end {
                mask[i][j] = true;
            }
        }

        // Global tokens
        for &g in &self.global_tokens {
            if g < n {
                for j in 0..n {
                    mask[g][j] = true;
                    mask[j][g] = true;
                }
            }
        }

        mask
    }

    /// Compute the sparse attention output for a single head using the local window.
    ///
    /// `q`, `k`, `v` are (seq_len × head_dim) row vectors.
    pub fn local_attention(
        &self,
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let n = self.seq_len.min(q.len());
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let global_set: std::collections::HashSet<usize> = self.global_tokens.iter().cloned().collect();
        let mut output = vec![vec![0.0_f32; self.head_dim]; n];

        for i in 0..n {
            let mut attend_to: Vec<usize> = Vec::new();

            // Local window
            let start = i.saturating_sub(self.window_size);
            let end = (i + self.window_size + 1).min(n);
            for j in start..end {
                attend_to.push(j);
            }

            // Global tokens (always attend)
            for &g in &self.global_tokens {
                if g < n && !attend_to.contains(&g) {
                    attend_to.push(g);
                }
            }

            // If current token is global, attend to all
            if global_set.contains(&i) {
                attend_to = (0..n).collect();
            }

            let scores: Vec<f32> = attend_to.iter().map(|&j| dot(&q[i], &k[j]) * scale).collect();
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum_e = exps.iter().sum::<f32>().max(1e-9);

            for (pos, &j) in attend_to.iter().enumerate() {
                let w = exps[pos] / sum_e;
                for (o, &vv) in output[i].iter_mut().zip(v[j].iter()) {
                    *o += w * vv;
                }
            }
        }
        output
    }
}

// ── SparseAttentionRouter ─────────────────────────────────────────────────────

/// Learned sparse attention router: selects the top-k keys each query attends to.
///
/// The router learns a per-query scoring function (small MLP) to predict which
/// keys are most relevant, then applies top-k selection.
pub struct SparseAttentionRouter {
    /// Sequence length.
    pub seq_len: usize,
    /// Token/key dimension.
    pub dim: usize,
    /// Number of keys each query attends to.
    pub top_k: usize,
    /// Router MLP weight matrix (dim × dim).
    pub router_w: Vec<Vec<f32>>,
}

impl SparseAttentionRouter {
    /// Initialize the router with random weights.
    pub fn new(seq_len: usize, dim: usize, top_k: usize, rng: &mut impl Rng) -> Self {
        let scale = (2.0 / (dim + dim) as f32).sqrt();
        let router_w: Vec<Vec<f32>> = (0..dim)
            .map(|_| (0..dim).map(|_| box_muller(rng) * scale).collect())
            .collect();
        Self { seq_len, dim, top_k, router_w }
    }

    /// Compute top-k routing: for each query, select the `top_k` most similar keys.
    ///
    /// Returns (selected_indices, attention_weights) for each query.
    pub fn route(&self, queries: &[Vec<f32>], keys: &[Vec<f32>]) -> Vec<(Vec<usize>, Vec<f32>)> {
        let n_q = queries.len().min(self.seq_len);
        let n_k = keys.len();
        let k = self.top_k.min(n_k);

        queries[..n_q].iter().map(|q| {
            // Project query through router weight
            let q_proj: Vec<f32> = self.router_w.iter().map(|row| dot(row, q)).collect();

            // Score each key
            let mut scores: Vec<(usize, f32)> = keys.iter().enumerate().map(|(j, key)| {
                let score = dot(&q_proj, key) / (self.dim as f32).sqrt();
                (j, score)
            }).collect();

            // Top-k selection
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores.truncate(k);

            // Softmax over top-k
            let max_s = scores.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|(_, s)| (s - max_s).exp()).collect();
            let sum_e = exps.iter().sum::<f32>().max(1e-9);
            let weights: Vec<f32> = exps.iter().map(|e| e / sum_e).collect();
            let indices: Vec<usize> = scores.iter().map(|(i, _)| *i).collect();

            (indices, weights)
        }).collect()
    }

    /// Compute sparse attention output using the router.
    pub fn attend(&self, queries: &[Vec<f32>], keys: &[Vec<f32>], values: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let routing = self.route(queries, keys);
        routing.into_iter().map(|(indices, weights)| {
            let mut out = vec![0.0_f32; self.dim];
            for (idx, w) in indices.iter().zip(weights.iter()) {
                if *idx < values.len() {
                    for (o, &v) in out.iter_mut().zip(values[*idx].iter()) {
                        *o += w * v;
                    }
                }
            }
            out
        }).collect()
    }
}

// ── SparsePositionEncoding ────────────────────────────────────────────────────

/// Sparse relative position bias for attention (Raffel et al. T5-style, sparse variant).
///
/// Rather than dense relative position embeddings, only a subset of relative
/// distances receive learned biases (sparsifying the position information).
pub struct SparsePositionEncoding {
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Embedding dimension.
    pub dim: usize,
    /// Number of distinct relative distance buckets.
    pub n_buckets: usize,
    /// Bucket embedding table (n_buckets × dim).
    pub bucket_embeddings: Vec<Vec<f32>>,
}

impl SparsePositionEncoding {
    /// Initialize with random bucket embeddings.
    pub fn new(max_seq_len: usize, dim: usize, n_buckets: usize, rng: &mut impl Rng) -> Self {
        let scale = (1.0 / dim as f32).sqrt();
        let bucket_embeddings = (0..n_buckets)
            .map(|_| (0..dim).map(|_| box_muller(rng) * scale).collect())
            .collect();
        Self { max_seq_len, dim, n_buckets, bucket_embeddings }
    }

    /// Map relative distance to a bucket index (logarithmic bucketing for large gaps).
    pub fn relative_position_bucket(&self, rel_pos: i32) -> usize {
        let abs_pos = rel_pos.unsigned_abs() as usize;
        let half = self.n_buckets / 2;
        if abs_pos < half {
            // Small distances: exact bucket
            abs_pos.min(half.saturating_sub(1))
        } else {
            // Large distances: logarithmic bucket
            let log_bucket = (abs_pos as f32 / half as f32).ln() / (self.max_seq_len as f32 / half as f32).ln().max(1e-8);
            let bucket = half + (log_bucket * half as f32) as usize;
            bucket.min(self.n_buckets - 1)
        }
    }

    /// Compute position bias matrix: shape (seq_len × seq_len × dim).
    /// Returns a flat vector of (seq_len × seq_len) bucket indices.
    pub fn position_bias_indices(&self, seq_len: usize) -> Vec<usize> {
        let n = seq_len.min(self.max_seq_len);
        let mut indices = vec![0usize; n * n];
        for i in 0..n {
            for j in 0..n {
                let rel = j as i32 - i as i32;
                indices[i * n + j] = self.relative_position_bucket(rel);
            }
        }
        indices
    }

    /// Look up position embeddings for a (seq_len × seq_len) attention matrix.
    pub fn get_position_biases(&self, seq_len: usize) -> Vec<Vec<f32>> {
        let indices = self.position_bias_indices(seq_len);
        indices.iter().map(|&b| self.bucket_embeddings[b].clone()).collect()
    }
}

// ── GroupLasso ────────────────────────────────────────────────────────────────

/// Group LASSO: apply L1 penalty over group norms for structured sparsity.
///
/// The penalty is λ * Σ_g ‖w_g‖₂, promoting whole groups to be zero.
/// Uses proximal gradient descent with exact group-prox operator.
pub struct GroupLasso {
    /// Feature groups: list of groups, each containing feature indices.
    pub groups: Vec<Vec<usize>>,
    /// Group L2 regularization weight.
    pub lambda: f32,
    /// Learning rate (proximal gradient step size).
    pub learning_rate: f32,
    /// Maximum optimization iterations.
    pub max_iter: usize,
}

impl GroupLasso {
    /// Fit Group LASSO via proximal gradient descent on squared loss.
    pub fn fit(&self, x_data: &[Vec<f32>], y: &[f32]) -> Vec<f32> {
        if x_data.is_empty() || y.is_empty() {
            return Vec::new();
        }
        let n_features = x_data[0].len();
        let n_samples = x_data.len();
        let mut w = vec![0.0_f32; n_features];

        for _ in 0..self.max_iter {
            // Compute gradient of squared loss: ∇ = (1/n) X^T (Xw - y)
            let preds: Vec<f32> = x_data.iter().map(|row| dot(row, &w)).collect();
            let residuals: Vec<f32> = preds.iter().zip(y.iter()).map(|(&p, &yi)| p - yi).collect();
            let mut grad = vec![0.0_f32; n_features];
            for (i, row) in x_data.iter().enumerate() {
                for (j, &xij) in row.iter().enumerate() {
                    grad[j] += xij * residuals[i] / n_samples as f32;
                }
            }

            // Gradient step
            let mut w_new = w.clone();
            for j in 0..n_features {
                w_new[j] -= self.learning_rate * grad[j];
            }

            // Group proximal operator: prox_{λ‖·‖₂}(w_g) = max(0, 1 - λ/‖w_g‖) w_g
            for group in &self.groups {
                let group_norm: f32 = group.iter()
                    .filter(|&&j| j < n_features)
                    .map(|&j| w_new[j] * w_new[j])
                    .sum::<f32>()
                    .sqrt();
                let threshold = self.lambda * self.learning_rate;
                if group_norm < threshold {
                    for &j in group.iter().filter(|&&j| j < n_features) {
                        w_new[j] = 0.0;
                    }
                } else {
                    let scale = 1.0 - threshold / group_norm;
                    for &j in group.iter().filter(|&&j| j < n_features) {
                        w_new[j] *= scale;
                    }
                }
            }
            w = w_new;
        }
        w
    }

    /// Predict target values using fitted weights.
    pub fn predict(&self, x_data: &[Vec<f32>], w: &[f32]) -> Vec<f32> {
        x_data.iter().map(|row| dot(row, w)).collect()
    }

    /// Compute the fraction of groups that are entirely zero.
    pub fn group_sparsity(&self, w: &[f32]) -> f32 {
        if self.groups.is_empty() {
            return 0.0;
        }
        let zero_groups = self.groups.iter().filter(|group| {
            group.iter().all(|&j| j >= w.len() || w[j].abs() < 1e-8)
        }).count();
        zero_groups as f32 / self.groups.len() as f32
    }
}

// ── StructuredPruningMask ─────────────────────────────────────────────────────

/// N:M sparsity mask for structured sparsity (e.g., 2:4 for Ampere GPUs).
///
/// In N:M sparsity, exactly N out of every M consecutive elements are non-zero.
/// This achieves exactly (1 - N/M) structured sparsity that maps efficiently to
/// hardware accelerators.
pub struct StructuredPruningMask {
    /// Number of non-zero elements to retain per group.
    pub n_keep: usize,
    /// Group size M.
    pub group_size: usize,
}

impl StructuredPruningMask {
    /// Create a 2:4 sparsity mask (default for Ampere/Hopper GPUs).
    pub fn nm_24() -> Self {
        Self { n_keep: 2, group_size: 4 }
    }

    /// Apply N:M sparsity to a weight tensor.
    ///
    /// Within every M consecutive elements, only the N largest by magnitude are kept.
    /// Returns the pruned weight vector and the binary mask.
    pub fn apply(&self, weights: &[f32]) -> (Vec<f32>, Vec<bool>) {
        let n = weights.len();
        let mut pruned = weights.to_vec();
        let mut mask = vec![false; n];
        let m = self.group_size;
        let k = self.n_keep.min(m);

        let n_groups = (n + m - 1) / m;
        for g in 0..n_groups {
            let start = g * m;
            let end = (start + m).min(n);
            let group_len = end - start;

            // Sort indices by magnitude within group
            let mut indices: Vec<usize> = (start..end).collect();
            indices.sort_by(|&a, &b| {
                weights[b].abs().partial_cmp(&weights[a].abs()).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Keep top-k, zero the rest
            let keep_count = k.min(group_len);
            for i in 0..group_len {
                if i < keep_count {
                    mask[indices[i]] = true;
                } else {
                    pruned[indices[i]] = 0.0;
                }
            }
        }
        (pruned, mask)
    }

    /// Compute the actual sparsity ratio of the mask.
    pub fn sparsity(mask: &[bool]) -> f32 {
        if mask.is_empty() {
            return 0.0;
        }
        let zeros = mask.iter().filter(|&&b| !b).count();
        zeros as f32 / mask.len() as f32
    }
}

// ── ChannelPruner ─────────────────────────────────────────────────────────────

/// Iterative channel importance scoring and pruning for CNNs.
///
/// Supports three importance criteria:
/// - L1 norm of filter weights
/// - Taylor first-order approximation (weight × gradient)
/// - FPGM (Filter Pruning via Geometric Median)
#[derive(Debug, Clone)]
pub enum ChannelImportanceCriterion {
    /// Sum of absolute weight values per filter.
    L1Norm,
    /// First-order Taylor: |w| × |∂L/∂w| approximation.
    TaylorExpansion,
    /// Filter Pruning via Geometric Median (He et al. 2019).
    Fpgm,
}

/// Channel pruner that scores and removes channels from a layer.
pub struct ChannelPruner {
    /// Fraction of channels to prune.
    pub prune_ratio: f32,
    /// Importance scoring criterion.
    pub criterion: ChannelImportanceCriterion,
}

impl ChannelPruner {
    /// Compute importance scores for each filter (channel).
    ///
    /// `filters` is a list of filters, each represented as a flat weight vector.
    /// `gradients` is optional; required for Taylor criterion.
    pub fn compute_importance(
        &self,
        filters: &[Vec<f32>],
        gradients: Option<&[Vec<f32>]>,
    ) -> Vec<f32> {
        match &self.criterion {
            ChannelImportanceCriterion::L1Norm => {
                filters.iter().map(|f| f.iter().map(|w| w.abs()).sum::<f32>()).collect()
            }
            ChannelImportanceCriterion::TaylorExpansion => {
                let grads = gradients.unwrap_or(filters);
                filters.iter().zip(grads.iter().chain(std::iter::repeat(&filters[0]))).map(|(f, g)| {
                    f.iter().zip(g.iter()).map(|(w, dw)| (w * dw).abs()).sum::<f32>()
                }).collect()
            }
            ChannelImportanceCriterion::Fpgm => {
                // FPGM: importance = negative of minimum distance to geometric median
                // Approximate via pairwise distances: less important = closer to median cluster
                let n = filters.len();
                (0..n).map(|i| {
                    // Score = inverse of average pairwise L2 distance to other filters
                    // (higher distance from the median cluster → keep)
                    let sum_dist: f32 = (0..n).filter(|&j| j != i).map(|j| {
                        filters[i].iter().zip(filters[j].iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt()
                    }).sum();
                    if n <= 1 { norm(&filters[i]) } else { sum_dist / (n - 1) as f32 }
                }).collect()
            }
        }
    }

    /// Select which channels to prune given importance scores.
    ///
    /// Returns a boolean mask (true = keep, false = prune).
    pub fn prune_mask(&self, importance: &[f32]) -> Vec<bool> {
        let n = importance.len();
        let n_prune = ((n as f32 * self.prune_ratio).round() as usize).min(n);
        let mut indexed: Vec<(usize, f32)> = importance.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut mask = vec![true; n];
        for (idx, _) in indexed.iter().take(n_prune) {
            mask[*idx] = false;
        }
        mask
    }

    /// Apply pruning: zero out pruned channel weights.
    pub fn apply_pruning(&self, filters: &[Vec<f32>], mask: &[bool]) -> Vec<Vec<f32>> {
        filters.iter().zip(mask.iter()).map(|(f, &keep)| {
            if keep { f.clone() } else { vec![0.0_f32; f.len()] }
        }).collect()
    }
}

// ── LayerPruner ───────────────────────────────────────────────────────────────

/// Whole-layer importance estimation via Fisher information approximation.
///
/// Uses the empirical Fisher (squared gradients) to estimate how much each
/// layer contributes to the loss. Layers with near-zero Fisher information
/// can be removed with minimal accuracy impact.
pub struct LayerPruner {
    /// Threshold below which a layer is considered unimportant.
    pub importance_threshold: f32,
}

impl LayerPruner {
    /// Estimate layer importance from weight and gradient vectors.
    ///
    /// Fisher information approximation: I(θ) ≈ (1/n) Σ (∂L/∂θ)²
    /// A layer is deemed prunable if its Fisher trace is below `importance_threshold`.
    pub fn layer_importance(&self, weights: &[f32], gradients: &[f32]) -> f32 {
        if weights.is_empty() {
            return 0.0;
        }
        let n = weights.len().min(gradients.len());
        // Fisher trace: sum of squared gradients, weighted by weight magnitude
        let fisher: f32 = (0..n).map(|i| (gradients[i] * weights[i]).powi(2)).sum::<f32>() / n as f32;
        fisher
    }

    /// Determine which layers should be pruned given their importances.
    ///
    /// Returns a boolean vector (true = keep, false = prune).
    pub fn select_layers_to_prune(&self, importances: &[f32]) -> Vec<bool> {
        importances.iter().map(|&imp| imp >= self.importance_threshold).collect()
    }

    /// Compute sensitivity ranking of layers.
    ///
    /// Returns indices sorted from least to most important.
    pub fn rank_layers(&self, importances: &[f32]) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = importances.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.iter().map(|(i, _)| *i).collect()
    }
}

// ── SparseCodingLayer ─────────────────────────────────────────────────────────

/// Sparse coding layer replacing a dense linear layer.
///
/// Learns a dictionary D and encodes inputs via LISTA (Learned ISTA, Gregor 2010).
/// The forward pass runs a fixed number of ISTA steps with learned step sizes,
/// producing sparse activations.
pub struct SparseCodingLayer {
    /// Input dimension.
    pub input_dim: usize,
    /// Dictionary size (number of atoms / output sparse codes).
    pub dict_size: usize,
    /// Number of unrolled ISTA steps.
    pub n_steps: usize,
    /// L1 regularization weight.
    pub lambda: f32,
    /// Dictionary matrix D: (dict_size × input_dim).
    pub dictionary: Vec<Vec<f32>>,
    /// Learned step size per ISTA iteration (one per step).
    pub step_sizes: Vec<f32>,
    /// Recurrent weight matrix We: (dict_size × dict_size) = I - step * D^T D.
    pub we_matrix: Vec<Vec<f32>>,
}

impl SparseCodingLayer {
    /// Initialize with random dictionary (Xavier) and default step sizes.
    pub fn new(input_dim: usize, dict_size: usize, n_steps: usize, lambda: f32, rng: &mut impl Rng) -> Self {
        let scale = (2.0 / (input_dim + dict_size) as f32).sqrt();
        let dictionary: Vec<Vec<f32>> = (0..dict_size)
            .map(|_| (0..input_dim).map(|_| box_muller(rng) * scale).collect())
            .collect();

        // Normalize dictionary atoms
        let dictionary: Vec<Vec<f32>> = dictionary.into_iter().map(|mut atom| {
            let n = norm(&atom);
            if n > 1e-8 { for v in atom.iter_mut() { *v /= n; } }
            atom
        }).collect();

        // Default step size heuristic: 1 / (largest singular value estimate)
        let default_step = 0.1_f32;
        let step_sizes = vec![default_step; n_steps];

        // We = I - step * D^T D  (approximate; simplified as identity for init)
        let we_matrix = (0..dict_size)
            .map(|i| (0..dict_size).map(|j| if i == j { 1.0 - default_step } else { 0.0 }).collect())
            .collect();

        Self { input_dim, dict_size, n_steps, lambda, dictionary, step_sizes, we_matrix }
    }

    /// Forward pass: encode input via LISTA (unrolled ISTA).
    ///
    /// Returns sparse code z of dimension dict_size.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // Initial response: D^T x
        let mut z: Vec<f32> = self.dictionary.iter()
            .map(|atom| dot(atom, x))
            .collect();

        // Threshold initial response
        let init_thresh = self.lambda * self.step_sizes.first().copied().unwrap_or(0.1);
        for v in z.iter_mut() {
            *v = LassoEncoder::soft_threshold(*v, init_thresh);
        }

        // Unrolled ISTA steps
        for step in 0..self.n_steps {
            let step_size = self.step_sizes.get(step).copied().unwrap_or(0.1);
            let threshold = self.lambda * step_size;

            // Compute We * z + step * D^T * x
            let we_z: Vec<f32> = self.we_matrix.iter().map(|row| dot(row, &z)).collect();
            let dtx: Vec<f32> = self.dictionary.iter().map(|atom| dot(atom, x) * step_size).collect();

            let mut z_new: Vec<f32> = (0..self.dict_size).map(|i| we_z[i] + dtx[i]).collect();
            for v in z_new.iter_mut() {
                *v = LassoEncoder::soft_threshold(*v, threshold);
            }
            z = z_new;
        }
        z
    }

    /// Reconstruct the input from its sparse code.
    pub fn reconstruct(&self, z: &[f32]) -> Vec<f32> {
        let mut x_hat = vec![0.0_f32; self.input_dim];
        for (atom, &zi) in self.dictionary.iter().zip(z.iter()) {
            for (xh, &a) in x_hat.iter_mut().zip(atom.iter()) {
                *xh += zi * a;
            }
        }
        x_hat
    }

    /// Reconstruction loss: ‖x - D z‖₂² / input_dim.
    pub fn reconstruction_loss(&self, x: &[f32]) -> f32 {
        let z = self.forward(x);
        let x_hat = self.reconstruct(&z);
        x.iter().zip(x_hat.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>()
            / self.input_dim as f32
    }
}

// ── ListaNetwork ──────────────────────────────────────────────────────────────

/// Learned ISTA (LISTA) network: unrolled gradient descent for sparse coding.
///
/// Implements the full LISTA architecture (Gregor & LeCun 2010) with
/// learnable forward and recurrent weight matrices Wd and We.
pub struct ListaNetwork {
    /// Input dimension.
    pub input_dim: usize,
    /// Sparse code dimension (number of dictionary atoms).
    pub code_dim: usize,
    /// Number of unrolled iterations (network depth).
    pub n_layers: usize,
    /// L1 threshold per layer.
    pub thresholds: Vec<f32>,
    /// Forward weights Wd: (code_dim × input_dim) — maps input to initial response.
    pub wd: Vec<Vec<f32>>,
    /// Recurrent weights We: (code_dim × code_dim) — maps previous code to next.
    pub we: Vec<Vec<f32>>,
}

impl ListaNetwork {
    /// Initialize LISTA network with Xavier weights.
    pub fn new(input_dim: usize, code_dim: usize, n_layers: usize, lambda: f32, rng: &mut impl Rng) -> Self {
        let scale_wd = (2.0 / (input_dim + code_dim) as f32).sqrt();
        let scale_we = (2.0 / (code_dim + code_dim) as f32).sqrt();
        let wd = (0..code_dim).map(|_| (0..input_dim).map(|_| box_muller(rng) * scale_wd).collect()).collect();
        let we = (0..code_dim).map(|_| (0..code_dim).map(|_| box_muller(rng) * scale_we).collect()).collect();
        let thresholds = vec![lambda; n_layers];
        Self { input_dim, code_dim, n_layers, thresholds, wd, we }
    }

    /// Forward pass through the unrolled ISTA network.
    ///
    /// z_{t+1} = soft_threshold(We z_t + Wd x, θ_t)
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut z = vec![0.0_f32; self.code_dim];
        let init: Vec<f32> = self.wd.iter().map(|row| dot(row, x)).collect();
        let thresh0 = self.thresholds.first().copied().unwrap_or(0.1);
        for (zi, &init_i) in z.iter_mut().zip(init.iter()) {
            *zi = LassoEncoder::soft_threshold(init_i, thresh0);
        }

        for layer in 1..self.n_layers {
            let thresh = self.thresholds.get(layer).copied().unwrap_or(0.1);
            let we_z: Vec<f32> = self.we.iter().map(|row| dot(row, &z)).collect();
            let wd_x: Vec<f32> = self.wd.iter().map(|row| dot(row, x)).collect();
            let mut z_new: Vec<f32> = (0..self.code_dim).map(|i| we_z[i] + wd_x[i]).collect();
            for v in z_new.iter_mut() {
                *v = LassoEncoder::soft_threshold(*v, thresh);
            }
            z = z_new;
        }
        z
    }

    /// Sparsity of the output code (fraction of zero entries).
    pub fn output_sparsity(&self, x: &[f32]) -> f32 {
        let z = self.forward(x);
        let zeros = z.iter().filter(|&&v| v == 0.0).count();
        zeros as f32 / self.code_dim as f32
    }
}

// ── PredictiveCodingLayer ─────────────────────────────────────────────────────

/// Predictive coding layer (Rao & Ballard 1999).
///
/// Maintains two populations of units:
/// - Representation units r: the current estimate of the hidden state
/// - Error units e: the prediction error (actual - predicted)
///
/// Learning minimizes the sum of squared prediction errors across layers.
pub struct PredictiveCodingLayer {
    /// Input dimension.
    pub input_dim: usize,
    /// Hidden representation dimension.
    pub hidden_dim: usize,
    /// Top-down prediction weight matrix (input_dim × hidden_dim).
    pub prediction_w: Vec<Vec<f32>>,
    /// Representation update rate.
    pub r_lr: f32,
    /// Number of inference iterations to converge r.
    pub n_inference_steps: usize,
}

impl PredictiveCodingLayer {
    /// Initialize with random prediction weights.
    pub fn new(input_dim: usize, hidden_dim: usize, rng: &mut impl Rng) -> Self {
        let scale = (1.0 / hidden_dim as f32).sqrt();
        let prediction_w = (0..input_dim)
            .map(|_| (0..hidden_dim).map(|_| box_muller(rng) * scale).collect())
            .collect();
        Self { input_dim, hidden_dim, prediction_w, r_lr: 0.1, n_inference_steps: 20 }
    }

    /// Compute top-down prediction from hidden representation.
    pub fn predict(&self, r: &[f32]) -> Vec<f32> {
        self.prediction_w.iter().map(|row| dot(row, r)).collect()
    }

    /// Compute prediction error: actual - predicted.
    pub fn prediction_error(&self, actual: &[f32], r: &[f32]) -> Vec<f32> {
        let pred = self.predict(r);
        actual.iter().zip(pred.iter()).map(|(a, p)| a - p).collect()
    }

    /// Run inference to find the optimal hidden representation r* for input x.
    ///
    /// Gradient descent on the free energy:
    /// ∂F/∂r = W^T e - r  (with unit precision assumption)
    pub fn infer(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut r = vec![0.0_f32; self.hidden_dim];

        for _ in 0..self.n_inference_steps {
            let e = self.prediction_error(x, &r);
            // Gradient: W^T e
            let grad_r: Vec<f32> = (0..self.hidden_dim).map(|j| {
                self.prediction_w.iter().zip(e.iter()).map(|(row, &ei)| row[j] * ei).sum::<f32>()
            }).collect();
            for (ri, &g) in r.iter_mut().zip(grad_r.iter()) {
                *ri += self.r_lr * g;
            }
        }
        let e_final = self.prediction_error(x, &r);
        (r, e_final)
    }

    /// Compute the free energy (total squared prediction error).
    pub fn free_energy(&self, x: &[f32]) -> f32 {
        let (_, e) = self.infer(x);
        e.iter().map(|v| v * v).sum::<f32>() / self.input_dim as f32
    }
}

// ── CompressedSensingMatrix ───────────────────────────────────────────────────

/// RIP-satisfying random measurement matrices for compressed sensing.
///
/// Provides Gaussian, Bernoulli, and structured (subsampled DFT / circulant)
/// matrices with theoretical RIP guarantees for appropriate (m, n, s) settings.
#[derive(Debug, Clone)]
pub enum CsMatrixType {
    /// i.i.d. Gaussian N(0, 1/m) entries — RIP holds with m = O(s log(n/s)).
    Gaussian,
    /// i.i.d. Bernoulli ±1/sqrt(m) entries — same RIP bound as Gaussian.
    Bernoulli,
    /// Subsampled Hadamard (Walsh-Hadamard) matrix for structured sparsity.
    /// Only valid when n is a power of 2.
    SubsampledHadamard,
}

/// Compressed sensing measurement matrix with RIP analysis.
pub struct CompressedSensingMatrix {
    /// Number of measurements.
    pub m_rows: usize,
    /// Signal length.
    pub n_cols: usize,
    /// Matrix construction type.
    pub matrix_type: CsMatrixType,
    /// The actual measurement matrix stored row-major.
    pub matrix: Vec<Vec<f32>>,
}

impl CompressedSensingMatrix {
    /// Construct a new compressed sensing matrix.
    pub fn new(m_rows: usize, n_cols: usize, matrix_type: CsMatrixType, rng: &mut impl Rng) -> Self {
        let matrix = match &matrix_type {
            CsMatrixType::Gaussian => {
                let scale = 1.0 / (m_rows as f32).sqrt();
                (0..m_rows).map(|_| {
                    (0..n_cols).map(|_| box_muller(rng) * scale).collect()
                }).collect()
            }
            CsMatrixType::Bernoulli => {
                let scale = 1.0 / (m_rows as f32).sqrt();
                (0..m_rows).map(|_| {
                    (0..n_cols).map(|_| if rng.random::<f32>() > 0.5 { scale } else { -scale }).collect()
                }).collect()
            }
            CsMatrixType::SubsampledHadamard => {
                // Build a Hadamard matrix of size n_cols (next power of 2)
                let nh = n_cols.next_power_of_two();
                let mut h = vec![vec![1.0_f32; nh]; nh];
                let mut step = 1usize;
                while step < nh {
                    for i in (0..nh).step_by(2 * step) {
                        for j in i..(i + step).min(nh) {
                            let a = h[j][0..nh].to_vec();
                            let b = h[j + step][0..nh].to_vec();
                            for k in 0..nh {
                                h[j][k] = a[k] + b[k];
                                h[j + step][k] = a[k] - b[k];
                            }
                        }
                    }
                    step *= 2;
                }
                let scale = 1.0 / (m_rows as f32 * nh as f32).sqrt();
                // Subsample m_rows rows at random
                let mut row_indices: Vec<usize> = (0..nh).collect();
                for i in 0..m_rows.min(nh) {
                    let j = i + rng.random_range(0..(nh - i));
                    row_indices.swap(i, j);
                }
                row_indices.truncate(m_rows.min(nh));
                row_indices.iter().map(|&r| {
                    h[r][..n_cols].iter().map(|&v| v * scale).collect()
                }).collect()
            }
        };
        Self { m_rows, n_cols, matrix_type, matrix }
    }

    /// Apply the measurement matrix: y = Φ x.
    pub fn measure(&self, x: &[f32]) -> Vec<f32> {
        self.matrix.iter().map(|row| dot(row, &x[..x.len().min(self.n_cols)])).collect()
    }

    /// Compute Φ^T v.
    pub fn transpose_apply(&self, v: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0_f32; self.n_cols];
        for (row, &vi) in self.matrix.iter().zip(v.iter()) {
            for (r, &a) in result.iter_mut().zip(row.iter()) {
                *r += a * vi;
            }
        }
        result
    }

    /// Theoretical sufficient number of measurements for s-sparse recovery.
    ///
    /// Based on m ≥ C · s · log(n / s) with constant C ≈ 4.
    pub fn sufficient_measurements(n: usize, s: usize) -> usize {
        if s == 0 || n == 0 {
            return 1;
        }
        let c = 4.0_f32;
        (c * s as f32 * (n as f32 / s as f32).ln()).ceil() as usize
    }
}

// ── BasisPursuitDenoise ───────────────────────────────────────────────────────

/// BPDN solver: min ‖x‖₁ s.t. ‖Ax - b‖₂ ≤ σ.
///
/// Implemented via ADMM with an augmented Lagrangian formulation equivalent
/// to LASSO with parameter λ = 1/σ (approximately).
pub struct BasisPursuitDenoise {
    /// Noise tolerance σ.
    pub sigma: f32,
    /// ADMM penalty parameter.
    pub rho: f32,
    /// Maximum ADMM iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f32,
}

impl BasisPursuitDenoise {
    /// Solve BPDN: minimize ‖x‖₁ subject to ‖Ax - b‖₂ ≤ σ.
    ///
    /// Uses ADMM with the Lagrangian: ½‖Ax - b‖₂² + λ‖z‖₁ + ρ/2‖x - z + u‖₂²
    /// where λ = 1 / (sigma * m).
    pub fn solve(&self, a: &[Vec<f32>], b: &[f32]) -> Vec<f32> {
        let m = a.len();
        let n = if m > 0 { a[0].len() } else { 0 };
        if n == 0 || m == 0 {
            return Vec::new();
        }
        let lambda = 1.0 / (self.sigma.max(1e-6) * m as f32);
        let rho = self.rho;

        let mut x = vec![0.0_f32; n];
        let mut z = vec![0.0_f32; n];
        let mut u = vec![0.0_f32; n];

        // A^T b
        let atb: Vec<f32> = (0..n).map(|j| {
            a.iter().zip(b.iter()).map(|(row, &bi)| row[j] * bi).sum::<f32>()
        }).collect();

        for _ in 0..self.max_iter {
            // x-update: (A^T A + rho I) x = A^T b + rho (z - u)
            let rhs: Vec<f32> = (0..n).map(|i| atb[i] + rho * (z[i] - u[i])).collect();
            // CG solve for (A^T A + rho I) x = rhs
            x = bpdn_cg(a, rho, &rhs, &x, 50);

            let z_old = z.clone();
            // z-update: soft threshold with lambda / rho
            for i in 0..n {
                z[i] = LassoEncoder::soft_threshold(x[i] + u[i], lambda / rho);
            }
            // u-update
            for i in 0..n {
                u[i] += x[i] - z[i];
            }

            let primal_res: f32 = (0..n).map(|i| (x[i] - z[i]).powi(2)).sum::<f32>().sqrt();
            let dual_res: f32 = (0..n).map(|i| (rho * (z[i] - z_old[i])).powi(2)).sum::<f32>().sqrt();
            if primal_res < self.tol && dual_res < self.tol {
                break;
            }
        }
        z
    }

    /// Residual norm ‖Ax - b‖₂.
    pub fn residual_norm(a: &[Vec<f32>], b: &[f32], x: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(row, &bi)| {
            let ax_i: f32 = dot(row, x);
            (ax_i - bi).powi(2)
        }).sum::<f32>().sqrt()
    }
}

/// Conjugate gradient for (A^T A + rho I) x = b used in BPDN ADMM.
fn bpdn_cg(a: &[Vec<f32>], rho: f32, b: &[f32], x0: &[f32], max_iter: usize) -> Vec<f32> {
    let n = b.len();
    let mut x = x0.to_vec();
    // Compute r = b - (A^T A + rho I) x0
    let ax: Vec<f32> = a.iter().map(|row| dot(row, &x)).collect();
    let atax: Vec<f32> = (0..n).map(|j| a.iter().zip(ax.iter()).map(|(row, &ai)| row[j] * ai).sum::<f32>()).collect();
    let mut r: Vec<f32> = (0..n).map(|i| b[i] - atax[i] - rho * x[i]).collect();
    let mut p = r.clone();
    let mut rsold: f32 = r.iter().map(|&v| v * v).sum();

    for _ in 0..max_iter {
        if rsold < 1e-12 { break; }
        let ap: Vec<f32> = a.iter().map(|row| dot(row, &p)).collect();
        let atap: Vec<f32> = (0..n).map(|j| a.iter().zip(ap.iter()).map(|(row, &ai)| row[j] * ai).sum::<f32>()).collect();
        let ap_full: Vec<f32> = (0..n).map(|i| atap[i] + rho * p[i]).collect();
        let denom: f32 = p.iter().zip(ap_full.iter()).map(|(&pi, &api)| pi * api).sum();
        if denom.abs() < 1e-14 { break; }
        let alpha = rsold / denom;
        for i in 0..n { x[i] += alpha * p[i]; r[i] -= alpha * ap_full[i]; }
        let rsnew: f32 = r.iter().map(|&v| v * v).sum();
        let beta = rsnew / rsold.max(1e-14);
        for i in 0..n { p[i] = r[i] + beta * p[i]; }
        rsold = rsnew;
    }
    x
}

// ── RecoveryGuarantees ────────────────────────────────────────────────────────

/// Theoretical compressed sensing recovery guarantees and RIP analysis.
pub struct RecoveryGuarantees;

impl RecoveryGuarantees {
    /// Estimate the RIP-s constant δ_s via random s-sparse test vectors.
    ///
    /// δ_s = max over s-sparse unit vectors x: |‖Ax‖₂² - 1|
    pub fn rip_constant(
        a: &[Vec<f32>],
        s: usize,
        n_trials: usize,
        rng: &mut impl Rng,
    ) -> f32 {
        let n = if a.is_empty() { 0 } else { a[0].len() };
        if n == 0 { return 0.0; }
        let mut max_delta = 0.0_f32;

        for _ in 0..n_trials {
            // Random s-sparse unit vector
            let mut support: Vec<usize> = (0..n).collect();
            for i in 0..s.min(n) {
                let j = i + rng.random_range(0..(n - i));
                support.swap(i, j);
            }
            let support = &support[..s.min(n)];

            let mut x = vec![0.0_f32; n];
            let mut sq_sum = 0.0_f32;
            for &i in support {
                let v = box_muller(rng);
                x[i] = v;
                sq_sum += v * v;
            }
            if sq_sum < 1e-10 { continue; }
            let x_norm = sq_sum.sqrt();
            for v in x.iter_mut() { *v /= x_norm; }

            let ax: Vec<f32> = a.iter().map(|row| dot(row, &x)).collect();
            let ax_norm_sq: f32 = ax.iter().map(|&v| v * v).sum();
            let delta = (ax_norm_sq - 1.0).abs();
            if delta > max_delta { max_delta = delta; }
        }
        max_delta
    }

    /// Phase transition boundary: minimum measurement ratio m/n for s/n sparsity ratio.
    ///
    /// Based on Donoho-Tanner phase transition (approximation).
    /// Returns the minimum undersampling ratio m/n needed to recover s-sparse signals.
    pub fn phase_transition(sparsity_ratio: f32) -> f32 {
        // Empirical approximation to the Donoho-Tanner curve
        // δ* ≈ f(ρ) where ρ = s/n, δ = m/n
        let rho = sparsity_ratio.clamp(0.0, 1.0);
        // Gaussian approximation: δ ≈ 2ρ log(1/ρ) + rho*(1 + log(2π)) for small ρ
        if rho < 1e-6 {
            return 0.0;
        }
        let delta = 2.0 * rho * (1.0 / rho).ln() + rho * (1.0 + (2.0 * std::f32::consts::PI).ln());
        delta.min(1.0)
    }

    /// Check if the RIP condition is sufficient for exact recovery.
    ///
    /// Basis Pursuit guarantees exact recovery if δ_{2s} < √2 - 1 ≈ 0.4142.
    pub fn is_rip_sufficient(delta_2s: f32) -> bool {
        delta_2s < (2.0_f32.sqrt() - 1.0)
    }
}

// ── CsMetrics ─────────────────────────────────────────────────────────────────

/// Metrics for evaluating compressed sensing recovery quality.
pub struct CsMetrics;

impl CsMetrics {
    /// Recovery signal-to-noise ratio in dB: 10 log10(‖x‖² / ‖x - x̂‖²).
    pub fn recovery_snr(original: &[f32], recovered: &[f32]) -> f32 {
        let signal_power: f32 = original.iter().map(|&v| v * v).sum();
        let error_power: f32 = original.iter().zip(recovered.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        if error_power < 1e-14 { return 100.0; }
        if signal_power < 1e-14 { return -100.0; }
        10.0 * (signal_power / error_power).log10()
    }

    /// Support recovery rate: fraction of true support indices that are recovered.
    ///
    /// `true_support`: indices of non-zero entries in the original signal.
    /// `recovered`: the recovered signal vector.
    /// `threshold`: values above this are considered non-zero in the recovery.
    pub fn support_recovery_rate(true_support: &[usize], recovered: &[f32], threshold: f32) -> f32 {
        if true_support.is_empty() { return 1.0; }
        let recovered_support: Vec<usize> = recovered.iter().enumerate()
            .filter(|(_, &v)| v.abs() > threshold)
            .map(|(i, _)| i)
            .collect();
        let hits = true_support.iter().filter(|&&i| recovered_support.contains(&i)).count();
        hits as f32 / true_support.len() as f32
    }

    /// Exact support recovery: true if the recovered support exactly matches the true support.
    pub fn exact_support_recovery(true_support: &[usize], recovered: &[f32], threshold: f32) -> bool {
        let mut recovered_support: Vec<usize> = recovered.iter().enumerate()
            .filter(|(_, &v)| v.abs() > threshold)
            .map(|(i, _)| i)
            .collect();
        let mut true_sorted = true_support.to_vec();
        true_sorted.sort_unstable();
        recovered_support.sort_unstable();
        true_sorted == recovered_support
    }

    /// Normalized recovery error ‖x - x̂‖₂ / ‖x‖₂.
    pub fn normalized_error(original: &[f32], recovered: &[f32]) -> f32 {
        let orig_norm: f32 = original.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let err_norm: f32 = original.iter().zip(recovered.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        if orig_norm < 1e-10 { return err_norm; }
        err_norm / orig_norm
    }

    /// Sparsity of the recovered signal (fraction of entries below threshold).
    pub fn recovered_sparsity(recovered: &[f32], threshold: f32) -> f32 {
        if recovered.is_empty() { return 0.0; }
        let zeros = recovered.iter().filter(|&&v| v.abs() <= threshold).count();
        zeros as f32 / recovered.len() as f32
    }
}

// ── Helpers for advanced module ────────────────────────────────────────────────

/// Initialize random key/query/value matrices for attention tests.
pub fn init_attention_matrices(
    seq_len: usize,
    head_dim: usize,
    rng: &mut impl Rng,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let scale = (1.0 / head_dim as f32).sqrt();
    let q: Vec<Vec<f32>> = (0..seq_len)
        .map(|_| (0..head_dim).map(|_| box_muller(rng) * scale).collect())
        .collect();
    let k: Vec<Vec<f32>> = (0..seq_len)
        .map(|_| (0..head_dim).map(|_| box_muller(rng) * scale).collect())
        .collect();
    let v: Vec<Vec<f32>> = (0..seq_len)
        .map(|_| (0..head_dim).map(|_| box_muller(rng) * scale).collect())
        .collect();
    (q, k, v)
}
