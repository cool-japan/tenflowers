//! Graph Attention Network (GAT) layer implementation.

use tenflowers_core::{Result, TensorError};

use super::types::{
    add_bias_inplace, leaky_relu, matvec, softmax_inplace, xavier_init,
};

/// Graph Attention Network (GAT) layer (Veličković et al., 2018).
///
/// For each attention head `k`:
///
/// ```text
/// z_v^k  = W^k · h_v          (linear projection)
/// e_vu^k = LeakyReLU(a^k ⊤ [z_v^k ∥ z_u^k])
/// α_vu^k = softmax_u(e_vu^k)
/// h_v^k  = Σ_u  α_vu^k · z_u^k
/// ```
///
/// Final output is either the concatenation of all head outputs
/// (`concat_heads = true`) or their mean.
///
/// Reference: <https://arxiv.org/abs/1710.10903>
#[derive(Debug, Clone)]
pub struct GatLayer {
    /// Dimensionality of input node features.
    pub in_features: usize,
    /// Dimensionality of output per attention head.
    pub out_features: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Whether to concatenate head outputs (true) or average them (false).
    pub concat_heads: bool,
    /// Dropout probability on attention coefficients during training.
    pub dropout_prob: f32,
    /// Per-head linear weights W^k, shape [num_heads × out_features × in_features].
    w: Vec<f32>,
    /// Per-head attention vectors a^k, shape [num_heads × 2*out_features].
    attn: Vec<f32>,
    /// Optional bias, length = final_out_dim.
    bias: Option<Vec<f32>>,
}

impl GatLayer {
    /// Construct a new GAT layer.
    ///
    /// # Arguments
    ///
    /// * `in_features`  — Dimensionality of input features.
    /// * `out_features` — Dimensionality of each head's output.
    /// * `num_heads`    — Number of parallel attention heads.
    /// * `concat_heads` — Concatenate heads (output dim = `num_heads * out_features`)
    ///   or average them (output dim = `out_features`).
    ///
    /// # Errors
    ///
    /// Returns an error when `in_features`, `out_features`, or `num_heads` is zero.
    pub fn new(
        in_features: usize,
        out_features: usize,
        num_heads: usize,
        concat_heads: bool,
    ) -> Result<Self> {
        if in_features == 0 {
            return Err(TensorError::invalid_argument(
                "GatLayer: in_features must be > 0".to_string(),
            ));
        }
        if out_features == 0 {
            return Err(TensorError::invalid_argument(
                "GatLayer: out_features must be > 0".to_string(),
            ));
        }
        if num_heads == 0 {
            return Err(TensorError::invalid_argument(
                "GatLayer: num_heads must be > 0".to_string(),
            ));
        }

        let w = xavier_init(num_heads * out_features, in_features);
        let attn_flat: Vec<f32> = (0..num_heads * 2 * out_features)
            .map(|_| {
                let r = scirs2_core::random::quick::random_f32();
                r * 2.0 * (6.0_f32 / (2 * out_features) as f32).sqrt()
                    - (6.0_f32 / (2 * out_features) as f32).sqrt()
            })
            .collect();

        let final_out_dim = if concat_heads {
            num_heads * out_features
        } else {
            out_features
        };
        let bias = Some(vec![0.0f32; final_out_dim]);

        Ok(Self {
            in_features,
            out_features,
            num_heads,
            concat_heads,
            dropout_prob: 0.0,
            w,
            attn: attn_flat,
            bias,
        })
    }

    /// Set the dropout probability (clamped to [0, 1]).
    pub fn set_dropout(&mut self, prob: f32) {
        self.dropout_prob = prob.clamp(0.0, 1.0);
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Extract the linear weights for head `k` as a slice of shape
    /// `[out_features, in_features]`.
    fn head_weights(&self, k: usize) -> &[f32] {
        let start = k * self.out_features * self.in_features;
        let end = start + self.out_features * self.in_features;
        &self.w[start..end]
    }

    /// Extract the attention vector for head `k` (length 2*out_features).
    fn head_attn(&self, k: usize) -> &[f32] {
        let start = k * 2 * self.out_features;
        let end = start + 2 * self.out_features;
        &self.attn[start..end]
    }

    /// Project `feat` (length in_features) through head-`k` linear layer.
    fn project(&self, k: usize, feat: &[f32]) -> Vec<f32> {
        matvec(self.head_weights(k), feat, self.out_features, self.in_features)
    }

    /// Compute the un-normalised attention coefficient between node `i` and `j`
    /// for head `k`, given their already-projected features `zi` and `zj`.
    ///
    /// e_ij = LeakyReLU(a ⊤ [zi ∥ zj])
    fn attention_score(&self, k: usize, zi: &[f32], zj: &[f32]) -> f32 {
        let a = self.head_attn(k);
        debug_assert_eq!(a.len(), 2 * self.out_features);
        let dot: f32 = a[..self.out_features]
            .iter()
            .zip(zi.iter())
            .map(|(ai, xi)| ai * xi)
            .sum::<f32>()
            + a[self.out_features..]
                .iter()
                .zip(zj.iter())
                .map(|(ai, xi)| ai * xi)
                .sum::<f32>();
        leaky_relu(dot)
    }

    // ── public forward ────────────────────────────────────────────────────────

    /// Run the GAT forward pass.
    ///
    /// # Arguments
    ///
    /// * `node_features`  — Flat `[num_nodes × in_features]` feature matrix.
    /// * `num_nodes`      — Number of nodes.
    /// * `neighbor_lists` — `neighbor_lists[i]` = neighbour indices of node `i`.
    ///   Self-loops should not be included; the implementation adds the node
    ///   itself to its own attention context automatically.
    /// * `training`       — When `true`, attention dropout is applied if
    ///   `dropout_prob > 0`.
    ///
    /// # Returns
    ///
    /// `(output_flat, out_dim)` where:
    ///
    /// * `out_dim` = `num_heads × out_features` when `concat_heads = true`,
    ///   otherwise `out_features`.
    /// * `output_flat.len()` = `num_nodes × out_dim`.
    ///
    /// # Errors
    ///
    /// * Input slice length mismatch.
    /// * `neighbor_lists.len()` ≠ `num_nodes`.
    /// * Any neighbour index out of range.
    pub fn forward(
        &self,
        node_features: &[f32],
        num_nodes: usize,
        neighbor_lists: &[Vec<usize>],
        training: bool,
    ) -> Result<(Vec<f32>, usize)> {
        // ── validation ──────────────────────────────────────────────────────
        let expected_len = num_nodes * self.in_features;
        if node_features.len() != expected_len {
            return Err(TensorError::invalid_argument(format!(
                "GatLayer::forward: node_features length {} ≠ num_nodes ({}) × in_features ({}) = {}",
                node_features.len(),
                num_nodes,
                self.in_features,
                expected_len,
            )));
        }
        if neighbor_lists.len() != num_nodes {
            return Err(TensorError::invalid_argument(format!(
                "GatLayer::forward: neighbor_lists.len() = {} but num_nodes = {}",
                neighbor_lists.len(),
                num_nodes,
            )));
        }
        for (i, nb_list) in neighbor_lists.iter().enumerate() {
            for &nb in nb_list {
                if nb >= num_nodes {
                    return Err(TensorError::invalid_argument(format!(
                        "GatLayer::forward: node {i} has neighbour index {nb} which is out of \
                         range (num_nodes = {num_nodes})"
                    )));
                }
            }
        }

        // ── pre-project all nodes for all heads ─────────────────────────────
        // projections[k][v] = W^k · h_v  (length out_features)
        let projections: Vec<Vec<Vec<f32>>> = (0..self.num_heads)
            .map(|k| {
                (0..num_nodes)
                    .map(|v| {
                        let feat = &node_features[v * self.in_features..(v + 1) * self.in_features];
                        self.project(k, feat)
                    })
                    .collect()
            })
            .collect();

        let final_out_dim = if self.concat_heads {
            self.num_heads * self.out_features
        } else {
            self.out_features
        };
        let mut output = vec![0.0f32; num_nodes * final_out_dim];

        // ── per-node, per-head attention ─────────────────────────────────────
        for v in 0..num_nodes {
            // Build the neighbourhood context: node v itself + its neighbours
            let mut context: Vec<usize> = vec![v];
            context.extend_from_slice(&neighbor_lists[v]);

            for k in 0..self.num_heads {
                // 1. Compute un-normalised scores e_vu for u ∈ context
                let mut scores: Vec<f32> = context
                    .iter()
                    .map(|&u| self.attention_score(k, &projections[k][v], &projections[k][u]))
                    .collect();

                // 2. Softmax → attention coefficients α_vu
                softmax_inplace(&mut scores);

                // 3. Optional dropout: zero-out some coefficients during training.
                //    Re-normalise afterwards to keep expectations intact.
                if training && self.dropout_prob > 0.0 {
                    let mut alive_sum = 0.0f32;
                    for coeff in scores.iter_mut() {
                        let r = scirs2_core::random::quick::random_f32();
                        if r < self.dropout_prob {
                            *coeff = 0.0;
                        } else {
                            alive_sum += *coeff;
                        }
                    }
                    if alive_sum > 1e-8 {
                        for coeff in scores.iter_mut() {
                            *coeff /= alive_sum;
                        }
                    }
                }

                // 4. Weighted sum: h_v^k = Σ_u α_vu · z_u^k
                let mut h_k = vec![0.0f32; self.out_features];
                for (&alpha, &u) in scores.iter().zip(context.iter()) {
                    let zu = &projections[k][u];
                    for (hki, zui) in h_k.iter_mut().zip(zu.iter()) {
                        *hki += alpha * zui;
                    }
                }

                // 5. Write into output buffer
                if self.concat_heads {
                    let offset = v * final_out_dim + k * self.out_features;
                    output[offset..offset + self.out_features].copy_from_slice(&h_k);
                } else {
                    // Accumulate — divide by num_heads later
                    let offset = v * final_out_dim;
                    for (o, hi) in output[offset..offset + self.out_features]
                        .iter_mut()
                        .zip(h_k.iter())
                    {
                        *o += hi;
                    }
                }
            }

            // When averaging heads, divide by num_heads
            if !self.concat_heads {
                let n = self.num_heads as f32;
                let offset = v * final_out_dim;
                for o in output[offset..offset + self.out_features].iter_mut() {
                    *o /= n;
                }
            }
        }

        // ── add bias ─────────────────────────────────────────────────────────
        if let Some(ref b) = self.bias {
            for v in 0..num_nodes {
                let offset = v * final_out_dim;
                add_bias_inplace(&mut output[offset..offset + final_out_dim], b);
            }
        }

        Ok((output, final_out_dim))
    }

    /// Replace the per-head linear weight tensor.
    ///
    /// Length must equal `num_heads × out_features × in_features`.
    pub fn set_weights(&mut self, w: Vec<f32>) -> Result<()> {
        let expected = self.num_heads * self.out_features * self.in_features;
        if w.len() != expected {
            return Err(TensorError::invalid_argument(format!(
                "GatLayer::set_weights: expected {expected} elements, got {}",
                w.len()
            )));
        }
        self.w = w;
        Ok(())
    }

    /// Replace the per-head attention vectors.
    ///
    /// Length must equal `num_heads × 2 × out_features`.
    pub fn set_attn(&mut self, a: Vec<f32>) -> Result<()> {
        let expected = self.num_heads * 2 * self.out_features;
        if a.len() != expected {
            return Err(TensorError::invalid_argument(format!(
                "GatLayer::set_attn: expected {expected} elements, got {}",
                a.len()
            )));
        }
        self.attn = a;
        Ok(())
    }

    /// Return the output dimension (depends on `concat_heads`).
    pub fn out_dim(&self) -> usize {
        if self.concat_heads {
            self.num_heads * self.out_features
        } else {
            self.out_features
        }
    }
}
