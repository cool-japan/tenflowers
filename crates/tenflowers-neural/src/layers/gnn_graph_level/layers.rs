//! GATv2 and GIN graph neural network layers.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

use super::types::{
    leaky_relu, linear_transform, relu, xavier_uniform, GnnError,
};

// ─────────────────────────────────────────────────────────────────────────────
// GATv2 layer
// ─────────────────────────────────────────────────────────────────────────────

/// GATv2 graph attention layer (Brody, Alon & Yahav, ICLR 2022).
///
/// Dynamic attention: the joint representation of source and target is
/// computed before projecting to a scalar score, enabling the model to attend
/// to *any* pair regardless of the source node:
///
/// ```text
/// e_{ij} = a^T · LeakyReLU(W_l h_i + W_r h_j)
/// α_{ij} = softmax_j(e_{ij})
/// h'_i   = Σ_j α_{ij} W_r h_j    (single-head variant)
/// ```
///
/// The `num_heads` field is stored for multi-head bookkeeping; this implementation
/// applies one head per call to keep weight matrices simple.
#[derive(Debug, Clone)]
pub struct Gatv2Layer {
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension (per head).
    pub out_features: usize,
    /// Number of attention heads (stored; single-head forward implemented here).
    pub num_heads: usize,
    /// Left projection: `[in_features, out_features]`.
    pub w_l: Vec<Vec<f32>>,
    /// Right projection: `[in_features, out_features]`.
    pub w_r: Vec<Vec<f32>>,
    /// Attention vector: `[out_features]`.
    pub a: Vec<f32>,
    /// LeakyReLU negative slope.
    pub negative_slope: f32,
}

impl Gatv2Layer {
    /// Create a new GATv2 layer with Xavier-initialised weights.
    pub fn new(in_features: usize, out_features: usize, num_heads: usize) -> Self {
        let w_l = xavier_uniform(in_features, out_features, 100);
        let w_r = xavier_uniform(in_features, out_features, 101);
        // Attention vector: uniform in [-limit, limit].
        let limit = (6.0_f64 / (out_features + 1) as f64).sqrt() as f32;
        let mut rng = StdRng::seed_from_u64(102);
        let a: Vec<f32> = (0..out_features)
            .map(|_| {
                let u: f32 = rng.random();
                u * 2.0 * limit - limit
            })
            .collect();

        Self {
            in_features,
            out_features,
            num_heads,
            w_l,
            w_r,
            a,
            negative_slope: 0.2,
        }
    }

    /// Compute attention weight matrix `α`: shape `[num_nodes, num_nodes]`.
    ///
    /// `α[i][j]` is the attention weight from node `i` to neighbour `j`
    /// (softmax over adjacency neighbours).
    pub fn attention_scores(
        &self,
        node_features: &[Vec<f32>],
        adj: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let n = node_features.len();
        // Pre-compute W_l h_i for each node: [out_features] per node.
        let wl_h: Vec<Vec<f32>> = node_features
            .iter()
            .map(|h| linear_transform(&self.w_l, h))
            .collect();
        // Pre-compute W_r h_j for each node: [out_features] per node.
        let wr_h: Vec<Vec<f32>> = node_features
            .iter()
            .map(|h| linear_transform(&self.w_r, h))
            .collect();

        let mut alpha = vec![vec![0.0_f32; n]; n];
        for i in 0..n {
            // Raw attention scores for all j that are neighbours (adj[i][j] > 0).
            let mut raw = vec![f32::NEG_INFINITY; n];
            for j in 0..n {
                if adj[i][j] > 0.0 {
                    // joint: LeakyReLU(W_l h_i + W_r h_j)
                    let joint: Vec<f32> = wl_h[i]
                        .iter()
                        .zip(wr_h[j].iter())
                        .map(|(&a, &b)| leaky_relu(a + b, self.negative_slope))
                        .collect();
                    // score: a^T * joint
                    let score: f32 = self.a.iter().zip(joint.iter()).map(|(&a, &x)| a * x).sum();
                    raw[j] = score;
                }
            }
            // Softmax over valid neighbours.
            let max_val = raw
                .iter()
                .cloned()
                .filter(|v| v.is_finite())
                .fold(f32::NEG_INFINITY, f32::max);
            if max_val.is_finite() {
                let mut sum = 0.0_f32;
                for j in 0..n {
                    if raw[j].is_finite() {
                        alpha[i][j] = (raw[j] - max_val).exp();
                        sum += alpha[i][j];
                    } else {
                        alpha[i][j] = 0.0;
                    }
                }
                if sum > 0.0 {
                    for j in 0..n {
                        alpha[i][j] /= sum;
                    }
                }
            }
        }
        alpha
    }

    /// Forward pass: `h'_i = σ( Σ_j α_{ij} W_r h_j )`, ReLU activation.
    ///
    /// Returns `[num_nodes, out_features]`.
    pub fn forward(
        &self,
        node_features: &[Vec<f32>],
        adj: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, GnnError> {
        let n = node_features.len();
        if n == 0 {
            return Err(GnnError::EmptyGraph);
        }
        if adj.len() != n {
            return Err(GnnError::FeatureCountMismatch {
                adj_size: adj.len(),
                feat_count: n,
            });
        }

        let alpha = self.attention_scores(node_features, adj);
        let wr_h: Vec<Vec<f32>> = node_features
            .iter()
            .map(|h| linear_transform(&self.w_r, h))
            .collect();

        let mut out = vec![vec![0.0_f32; self.out_features]; n];
        for i in 0..n {
            for j in 0..n {
                if alpha[i][j] > 0.0 {
                    for f in 0..self.out_features {
                        out[i][f] += alpha[i][j] * wr_h[j][f];
                    }
                }
            }
            // Apply ReLU.
            for f in 0..self.out_features {
                out[i][f] = relu(out[i][f]);
            }
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GIN layer
// ─────────────────────────────────────────────────────────────────────────────

/// Graph Isomorphism Network (GIN) layer (Xu et al., ICLR 2019).
///
/// Aggregate neighbours and update:
/// ```text
/// h'_v = MLP( (1 + ε) · h_v  +  Σ_{u ∈ N(v)} h_u )
/// ```
///
/// Here the MLP is a single linear layer followed by ReLU.
#[derive(Debug, Clone)]
pub struct GinLayer {
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// ε parameter.
    pub epsilon: f32,
    /// MLP weight matrix: `[in_features, out_features]` (row-major).
    pub mlp_weights: Vec<Vec<f32>>,
    /// MLP bias vector: `[out_features]`.
    pub mlp_bias: Vec<f32>,
}

impl GinLayer {
    /// Create a new GIN layer.
    pub fn new(in_features: usize, out_features: usize, epsilon: f32) -> Self {
        let mlp_weights = xavier_uniform(in_features, out_features, 200);
        let mlp_bias = vec![0.0_f32; out_features];
        Self {
            in_features,
            out_features,
            epsilon,
            mlp_weights,
            mlp_bias,
        }
    }

    /// Forward pass.
    ///
    /// Returns `[num_nodes, out_features]`.
    pub fn forward(
        &self,
        node_features: &[Vec<f32>],
        adj: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, GnnError> {
        let n = node_features.len();
        if n == 0 {
            return Err(GnnError::EmptyGraph);
        }
        if adj.len() != n {
            return Err(GnnError::FeatureCountMismatch {
                adj_size: adj.len(),
                feat_count: n,
            });
        }

        let d = self.in_features;

        // Aggregate neighbours.
        let mut agg: Vec<Vec<f32>> = node_features
            .iter()
            .map(|h| h.iter().map(|&x| (1.0 + self.epsilon) * x).collect())
            .collect();

        for i in 0..n {
            for j in 0..n {
                if adj[i][j] > 0.0 {
                    let scale = adj[i][j]; // support weighted edges
                    if node_features[j].len() != d {
                        return Err(GnnError::DimensionMismatch {
                            expected: d,
                            found: node_features[j].len(),
                        });
                    }
                    for f in 0..d {
                        agg[i][f] += scale * node_features[j][f];
                    }
                }
            }
        }

        // Apply MLP: linear + ReLU.
        let mut out = vec![vec![0.0_f32; self.out_features]; n];
        for i in 0..n {
            for o in 0..self.out_features {
                let mut val = self.mlp_bias[o];
                for f in 0..d {
                    val += agg[i][f] * self.mlp_weights[f][o];
                }
                out[i][o] = relu(val);
            }
        }
        Ok(out)
    }
}
