//! Graph Foundation Models and Pre-training for TenfloweRS.
//!
//! This module implements:
//! - **Graph Transformers**: Graphormer, TokenGT, GraphGPS
//! - **Graph Pre-training**: MAE, Edge Prediction, Attribute Masking, Contrastive
//! - **Few-Shot Graph Learning**: ProtoNet, Matching Network, MAML, Task-Aware GNN
//! - **Heterogeneous Graph Transformers**: HGT, R-GCN, CompGCN, HeteroSAGE
//! - **Link Prediction & Graph Generation**: ComplEx, RotatE, GraphRNN, MoleculeGenerator
//! - **Graph Contrastive Pretraining**: GraphCL, GraphMAE, Graph Transformer Pretraining
//! - **Graph Federated Pretraining**: FedGraph, CrossGraphTransfer, GraphMetrics

pub mod extensions;
pub use extensions::*;

pub mod pretraining;
pub use pretraining::*;

#[cfg(test)]
mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Utility helpers
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn xavier_vec(n: usize, fan_in: usize, fan_out: usize, rng: &mut StdRng) -> Vec<f64> {
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
    (0..n)
        .map(|_| rng.random::<f64>() * 2.0 * limit - limit)
        .collect()
}

pub(crate) fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = if b.is_empty() { 0 } else { b[0].len() };
    let inner = b.len();
    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| (0..inner).map(|k| a[i][k] * b[k][j]).sum())
                .collect()
        })
        .collect()
}

pub(crate) fn softmax_1d(v: &[f64]) -> Vec<f64> {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < 1e-30 {
        return vec![1.0 / v.len() as f64; v.len()];
    }
    exps.iter().map(|x| x / sum).collect()
}

pub(crate) fn relu(x: f64) -> f64 {
    x.max(0.0)
}

pub(crate) fn layer_norm_vec(v: &[f64]) -> Vec<f64> {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
    let std = (var + 1e-6).sqrt();
    v.iter().map(|x| (x - mean) / std).collect()
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub(crate) fn bce_loss(pred: f64, target: f64) -> f64 {
    let p = pred.clamp(1e-9, 1.0 - 1e-9);
    -(target * p.ln() + (1.0 - target) * (1.0 - p).ln())
}

pub(crate) fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. GRAPH TRANSFORMERS
// ─────────────────────────────────────────────────────────────────────────────

/// Attention bias derived from graph structure (Graphormer-style).
///
/// Encodes three types of structural information:
/// - **Spatial encoding**: shortest-path distance between node pairs
/// - **Edge encoding**: mean of edge features along the shortest path
/// - **Centrality encoding**: in/out-degree bias per node
#[derive(Debug, Clone)]
pub struct GraphormerBias {
    /// Maximum shortest-path distance to consider (clip beyond this)
    pub max_dist: usize,
    /// Learnable scalar bias per distance bucket (len = max_dist + 1)
    pub spatial_bias: Vec<f64>,
    /// Per-node centrality (degree) bias
    pub centrality_bias: Vec<f64>,
    /// Edge feature dimension
    pub edge_feat_dim: usize,
}

impl GraphormerBias {
    /// Create a new bias block with random initialization.
    pub fn new(max_dist: usize, n_nodes: usize, edge_feat_dim: usize, rng: &mut StdRng) -> Self {
        let spatial_bias = xavier_vec(max_dist + 1, max_dist + 1, 1, rng);
        let centrality_bias = xavier_vec(n_nodes, n_nodes, 1, rng);
        Self {
            max_dist,
            spatial_bias,
            centrality_bias,
            edge_feat_dim,
        }
    }

    /// Compute BFS shortest-path distances from all nodes.
    pub fn bfs_distances(adj: &[Vec<usize>], n: usize) -> Vec<Vec<usize>> {
        let mut dist = vec![vec![usize::MAX; n]; n];
        for src in 0..n {
            dist[src][src] = 0;
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(src);
            while let Some(u) = queue.pop_front() {
                for &v in &adj[u] {
                    if dist[src][v] == usize::MAX {
                        dist[src][v] = dist[src][u] + 1;
                        queue.push_back(v);
                    }
                }
            }
        }
        dist
    }

    /// Compute the additive attention bias matrix of shape [n, n].
    pub fn compute_bias(&self, adj: &[Vec<usize>], degrees: &[usize]) -> Vec<Vec<f64>> {
        let n = adj.len();
        let dists = Self::bfs_distances(adj, n);
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let d = dists[i][j].min(self.max_dist);
                        let spatial = if dists[i][j] == usize::MAX {
                            self.spatial_bias[self.max_dist]
                        } else {
                            self.spatial_bias[d]
                        };
                        let cent_i = if i < self.centrality_bias.len() {
                            self.centrality_bias[i]
                        } else {
                            0.0
                        };
                        let cent_j = if j < self.centrality_bias.len() {
                            self.centrality_bias[j]
                        } else {
                            0.0
                        };
                        let degree_scale = if i < degrees.len() && j < degrees.len() {
                            (degrees[i] as f64 + degrees[j] as f64) * 0.01
                        } else {
                            0.0
                        };
                        spatial + cent_i + cent_j + degree_scale
                    })
                    .collect()
            })
            .collect()
    }
}

/// Single Graphormer layer: multi-head attention with structural biases + FFN.
#[derive(Debug, Clone)]
pub struct GraphormerLayer {
    pub d_model: usize,
    pub num_heads: usize,
    /// W_q, W_k, W_v per head: each [d_model * head_dim]
    pub wq: Vec<Vec<f64>>,
    pub wk: Vec<Vec<f64>>,
    pub wv: Vec<Vec<f64>>,
    /// Output projection [d_model * d_model]
    pub wo: Vec<f64>,
    /// FFN weights
    pub ff1: Vec<Vec<f64>>,
    pub ff2: Vec<Vec<f64>>,
}

impl GraphormerLayer {
    /// Create a new Graphormer layer with the given dimension and number of heads.
    pub fn new(d_model: usize, num_heads: usize, rng: &mut StdRng) -> Result<Self> {
        if d_model % num_heads != 0 {
            return Err(TensorError::invalid_argument(format!(
                "d_model={d_model} not divisible by num_heads={num_heads}"
            )));
        }
        let head_dim = d_model / num_heads;
        let wq = (0..num_heads)
            .map(|_| xavier_vec(d_model * head_dim, d_model, head_dim, rng))
            .collect();
        let wk = (0..num_heads)
            .map(|_| xavier_vec(d_model * head_dim, d_model, head_dim, rng))
            .collect();
        let wv = (0..num_heads)
            .map(|_| xavier_vec(d_model * head_dim, d_model, head_dim, rng))
            .collect();
        let wo = xavier_vec(d_model * d_model, d_model, d_model, rng);
        let ff_dim = d_model * 4;
        let ff1 = (0..ff_dim)
            .map(|_| xavier_vec(d_model, d_model, ff_dim, rng))
            .collect();
        let ff2 = (0..d_model)
            .map(|_| xavier_vec(ff_dim, ff_dim, d_model, rng))
            .collect();
        Ok(Self {
            d_model,
            num_heads,
            wq,
            wk,
            wv,
            wo,
            ff1,
            ff2,
        })
    }

    fn project_head(x: &[Vec<f64>], w: &[f64], d_model: usize, head_dim: usize) -> Vec<Vec<f64>> {
        x.iter()
            .map(|node| {
                (0..head_dim)
                    .map(|j| (0..d_model).map(|k| node[k] * w[k * head_dim + j]).sum())
                    .collect()
            })
            .collect()
    }

    /// Forward pass: node_feats [n, d_model], adj \[n\]\[neighbors\], degrees \[n\]
    pub fn forward(
        &self,
        node_feats: &[Vec<f64>],
        adj: &[Vec<usize>],
        degrees: &[usize],
        bias: &GraphormerBias,
    ) -> Vec<Vec<f64>> {
        let n = node_feats.len();
        let head_dim = self.d_model / self.num_heads;
        let scale = (head_dim as f64).sqrt();
        let attn_bias = bias.compute_bias(adj, degrees);

        // Multi-head attention
        let mut mha_out = vec![vec![0.0f64; self.d_model]; n];

        for h in 0..self.num_heads {
            let q = Self::project_head(node_feats, &self.wq[h], self.d_model, head_dim);
            let k = Self::project_head(node_feats, &self.wk[h], self.d_model, head_dim);
            let v = Self::project_head(node_feats, &self.wv[h], self.d_model, head_dim);

            for i in 0..n {
                let logits: Vec<f64> = (0..n)
                    .map(|j| dot(&q[i], &k[j]) / scale + attn_bias[i][j])
                    .collect();
                let weights = softmax_1d(&logits);
                let head_offset = h * head_dim;
                for d in 0..head_dim {
                    mha_out[i][head_offset + d] += weights
                        .iter()
                        .zip(v.iter())
                        .map(|(w, vj)| w * vj[d])
                        .sum::<f64>();
                }
            }
        }

        // Output projection + residual + layer norm
        let after_attn: Vec<Vec<f64>> = mha_out
            .iter()
            .zip(node_feats.iter())
            .map(|(out, x)| {
                let projected: Vec<f64> = (0..self.d_model)
                    .map(|j| {
                        (0..self.d_model)
                            .map(|k| out[k] * self.wo[k * self.d_model + j])
                            .sum()
                    })
                    .collect();
                layer_norm_vec(
                    &projected
                        .iter()
                        .zip(x.iter())
                        .map(|(a, b)| a + b)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();

        // FFN + residual + layer norm
        after_attn
            .iter()
            .map(|x| {
                let ff_dim = self.ff1.len();
                let h1: Vec<f64> = (0..ff_dim).map(|i| relu(dot(x, &self.ff1[i]))).collect();
                let h2: Vec<f64> = (0..self.d_model).map(|i| dot(&h1, &self.ff2[i])).collect();
                layer_norm_vec(
                    &h2.iter()
                        .zip(x.iter())
                        .map(|(a, b)| a + b)
                        .collect::<Vec<_>>(),
                )
            })
            .collect()
    }
}

/// Full Graphormer model: N layers + virtual node + graph-level readout.
#[derive(Debug, Clone)]
pub struct GraphormerModel {
    /// Stack of Graphormer layers
    pub layers: Vec<GraphormerLayer>,
    /// Structural attention bias
    pub bias: GraphormerBias,
    /// Virtual node embedding
    pub virtual_node: Vec<f64>,
    /// Model dimension
    pub d_model: usize,
}

impl GraphormerModel {
    /// Build a Graphormer model with the given configuration.
    pub fn new(
        n_layers: usize,
        d_model: usize,
        num_heads: usize,
        max_dist: usize,
        n_nodes: usize,
        rng: &mut StdRng,
    ) -> Result<Self> {
        let layers = (0..n_layers)
            .map(|_| GraphormerLayer::new(d_model, num_heads, rng))
            .collect::<Result<Vec<_>>>()?;
        let bias = GraphormerBias::new(max_dist, n_nodes, 0, rng);
        let virtual_node = xavier_vec(d_model, d_model, 1, rng);
        Ok(Self {
            layers,
            bias,
            virtual_node,
            d_model,
        })
    }

    /// Forward: returns graph-level embedding (mean of node embeddings).
    pub fn forward(
        &self,
        node_feats: &[Vec<f64>],
        adj: &[Vec<usize>],
        degrees: &[usize],
    ) -> Vec<f64> {
        let mut x = node_feats.to_vec();
        for layer in &self.layers {
            x = layer.forward(&x, adj, degrees, &self.bias);
        }
        // Mean pooling + virtual node
        let n = x.len();
        if n == 0 {
            return self.virtual_node.clone();
        }
        let mean: Vec<f64> = (0..self.d_model)
            .map(|d| x.iter().map(|v| v[d]).sum::<f64>() / n as f64)
            .collect();
        mean.iter()
            .zip(&self.virtual_node)
            .map(|(a, b)| a + b)
            .collect()
    }
}

/// TokenGT Layer: tokenize subgraph structures as node/edge tokens.
#[derive(Debug, Clone)]
pub struct TokenGtLayer {
    /// Output model dimension
    pub d_model: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Node projection weights
    pub node_proj: Vec<Vec<f64>>,
    /// Edge projection weights
    pub edge_proj: Vec<Vec<f64>>,
    inner: GraphormerLayer,
}

impl TokenGtLayer {
    /// Build a TokenGT layer from input dimension `d_in` to `d_model`.
    pub fn new(d_in: usize, d_model: usize, num_heads: usize, rng: &mut StdRng) -> Result<Self> {
        let node_proj = (0..d_model)
            .map(|_| xavier_vec(d_in, d_in, d_model, rng))
            .collect();
        let edge_proj = (0..d_model)
            .map(|_| xavier_vec(d_in, d_in, d_model, rng))
            .collect();
        let inner = GraphormerLayer::new(d_model, num_heads, rng)?;
        Ok(Self {
            d_model,
            num_heads,
            node_proj,
            edge_proj,
            inner,
        })
    }

    /// Forward: project nodes, create edge tokens, run transformer over all tokens.
    pub fn forward(&self, node_feats: &[Vec<f64>], edges: &[(usize, usize)]) -> Vec<Vec<f64>> {
        // Project node features
        let node_tokens: Vec<Vec<f64>> = node_feats
            .iter()
            .map(|f| {
                (0..self.d_model)
                    .map(|i| dot(f, &self.node_proj[i]))
                    .collect()
            })
            .collect();

        // Create edge tokens as mean of endpoint projections
        let edge_tokens: Vec<Vec<f64>> = edges
            .iter()
            .map(|&(u, v)| {
                let fu = if u < node_feats.len() {
                    &node_feats[u]
                } else {
                    &node_feats[0]
                };
                let fv = if v < node_feats.len() {
                    &node_feats[v]
                } else {
                    &node_feats[0]
                };
                let mean_feat: Vec<f64> = fu
                    .iter()
                    .zip(fv.iter())
                    .map(|(a, b)| (a + b) / 2.0)
                    .collect();
                (0..self.d_model)
                    .map(|i| dot(&mean_feat, &self.edge_proj[i]))
                    .collect()
            })
            .collect();

        // Concatenate all tokens
        let mut all_tokens = node_tokens;
        all_tokens.extend(edge_tokens);

        // Build dummy adj (all tokens attend to each other)
        let t = all_tokens.len();
        let adj: Vec<Vec<usize>> = (0..t)
            .map(|i| (0..t).filter(|&j| j != i).collect())
            .collect();
        let degrees: Vec<usize> = vec![t - 1; t];

        // Dummy bias
        let mut rng = StdRng::seed_from_u64(42);
        let bias = GraphormerBias::new(self.d_model, t, 0, &mut rng);
        self.inner.forward(&all_tokens, &adj, &degrees, &bias)
    }
}

/// GraphGPS Layer: local MPNN + global Transformer in parallel.
#[derive(Debug, Clone)]
pub struct GraphGPSLayer {
    /// Feature dimension
    pub d_model: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Local MPNN message weights [d_model * d_model]
    pub msg_w: Vec<f64>,
    /// Global attention layer
    pub attn: GraphormerLayer,
}

impl GraphGPSLayer {
    /// Build a GraphGPS layer.
    pub fn new(d_model: usize, num_heads: usize, rng: &mut StdRng) -> Result<Self> {
        let msg_w = xavier_vec(d_model * d_model, d_model, d_model, rng);
        let attn = GraphormerLayer::new(d_model, num_heads, rng)?;
        Ok(Self {
            d_model,
            num_heads,
            msg_w,
            attn,
        })
    }

    /// Forward: combines local MPNN aggregation with global attention.
    pub fn forward(&self, x: &[Vec<f64>], edge_idx: &[(usize, usize)]) -> Vec<Vec<f64>> {
        let n = x.len();

        // Local MPNN: aggregate neighbor messages
        let mut local_out = x.to_vec();
        for &(src, dst) in edge_idx {
            if src < n && dst < n {
                let msg: Vec<f64> = (0..self.d_model)
                    .map(|j| {
                        (0..self.d_model)
                            .map(|k| x[src][k] * self.msg_w[k * self.d_model + j])
                            .sum()
                    })
                    .collect();
                for d in 0..self.d_model {
                    local_out[dst][d] += msg[d];
                }
            }
        }
        let local_norm: Vec<Vec<f64>> = local_out.iter().map(|v| layer_norm_vec(v)).collect();

        // Global transformer
        let adj: Vec<Vec<usize>> = (0..n)
            .map(|i| (0..n).filter(|&j| j != i).collect())
            .collect();
        let degrees: Vec<usize> = vec![n.saturating_sub(1); n];
        let mut rng = StdRng::seed_from_u64(99);
        let bias = GraphormerBias::new(8, n, 0, &mut rng);
        let global_out = self.attn.forward(x, &adj, &degrees, &bias);

        // Combine: element-wise sum + layer norm
        local_norm
            .iter()
            .zip(global_out.iter())
            .map(|(l, g)| {
                layer_norm_vec(
                    &l.iter()
                        .zip(g.iter())
                        .map(|(a, b)| a + b)
                        .collect::<Vec<_>>(),
                )
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. GRAPH PRE-TRAINING
// ─────────────────────────────────────────────────────────────────────────────

/// Graph Masked Autoencoder: mask random nodes, reconstruct with GNN decoder.
#[derive(Debug, Clone)]
pub struct GraphMaskedAutoencoder {
    /// Hidden/output dimension
    pub d_model: usize,
    /// Encoder: linear projection
    pub encoder_w: Vec<Vec<f64>>,
    /// Decoder: linear projection back
    pub decoder_w: Vec<Vec<f64>>,
    /// Mask token embedding
    pub mask_token: Vec<f64>,
}

impl GraphMaskedAutoencoder {
    /// Build a Graph MAE with encoder dim `d_model` and input dim `d_in`.
    pub fn new(d_in: usize, d_model: usize, rng: &mut StdRng) -> Self {
        let encoder_w = (0..d_model)
            .map(|_| xavier_vec(d_in, d_in, d_model, rng))
            .collect();
        let decoder_w = (0..d_in)
            .map(|_| xavier_vec(d_model, d_model, d_in, rng))
            .collect();
        let mask_token = xavier_vec(d_in, d_in, 1, rng);
        Self {
            d_model,
            encoder_w,
            decoder_w,
            mask_token,
        }
    }

    /// Randomly mask `ratio` fraction of nodes.
    /// Returns `(visible_indices, masked_indices)`.
    pub fn mask_nodes(
        &self,
        n_nodes: usize,
        ratio: f64,
        rng: &mut StdRng,
    ) -> (Vec<usize>, Vec<usize>) {
        let n_mask = ((n_nodes as f64 * ratio).round() as usize).min(n_nodes);
        let mut indices: Vec<usize> = (0..n_nodes).collect();
        // Fisher-Yates shuffle
        for i in (1..n_nodes).rev() {
            let j = rng.random_range(0..=i);
            indices.swap(i, j);
        }
        let masked = indices[..n_mask].to_vec();
        let visible = indices[n_mask..].to_vec();
        (visible, masked)
    }

    /// Encode visible nodes, replace masked with mask_token, decode all.
    pub fn forward(&self, node_feats: &[Vec<f64>], masked_indices: &[usize]) -> Vec<Vec<f64>> {
        let masked_set: std::collections::HashSet<usize> = masked_indices.iter().cloned().collect();
        let encoded: Vec<Vec<f64>> = node_feats
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let feat = if masked_set.contains(&i) {
                    &self.mask_token
                } else {
                    f
                };
                (0..self.d_model)
                    .map(|j| dot(feat, &self.encoder_w[j]))
                    .collect()
            })
            .collect();
        let d_in = self.decoder_w.len();
        encoded
            .iter()
            .map(|e| (0..d_in).map(|j| dot(e, &self.decoder_w[j])).collect())
            .collect()
    }

    /// MSE reconstruction loss on masked nodes only.
    pub fn reconstruction_loss(
        &self,
        pred_feats: &[Vec<f64>],
        true_feats: &[Vec<f64>],
        masked_indices: &[usize],
    ) -> f64 {
        if masked_indices.is_empty() {
            return 0.0;
        }
        let total: f64 = masked_indices
            .iter()
            .map(|&i| {
                if i >= pred_feats.len() || i >= true_feats.len() {
                    return 0.0;
                }
                pred_feats[i]
                    .iter()
                    .zip(true_feats[i].iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
            })
            .sum();
        total / masked_indices.len() as f64
    }
}

/// Edge prediction pre-training: binary cross-entropy over positive/negative edges.
#[derive(Debug, Clone)]
pub struct EdgePredictionPretraining {
    /// Model dimension
    pub d_model: usize,
    /// Bilinear scorer weights [d_model * d_model]
    pub score_w: Vec<f64>,
}

impl EdgePredictionPretraining {
    /// Build edge prediction pretraining with bilinear scoring.
    pub fn new(d_model: usize, rng: &mut StdRng) -> Self {
        let score_w = xavier_vec(d_model * d_model, d_model, d_model, rng);
        Self { d_model, score_w }
    }

    /// Score a pair of node embeddings (dot product after bilinear).
    pub fn score_edge(&self, h: &[f64], t: &[f64]) -> f64 {
        let wt: Vec<f64> = (0..self.d_model)
            .map(|j| {
                (0..self.d_model)
                    .map(|k| t[k] * self.score_w[k * self.d_model + j])
                    .sum()
            })
            .collect();
        sigmoid(dot(h, &wt))
    }

    /// Compute BCE loss over positive and negative edge samples.
    pub fn loss(
        &self,
        embeddings: &[Vec<f64>],
        pos_edges: &[(usize, usize)],
        neg_edges: &[(usize, usize)],
    ) -> f64 {
        let n = embeddings.len();
        let mut total = 0.0;
        let mut count = 0;
        for &(u, v) in pos_edges {
            if u < n && v < n {
                total += bce_loss(self.score_edge(&embeddings[u], &embeddings[v]), 1.0);
                count += 1;
            }
        }
        for &(u, v) in neg_edges {
            if u < n && v < n {
                total += bce_loss(self.score_edge(&embeddings[u], &embeddings[v]), 0.0);
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f64
        }
    }

    /// Sample negative edges (non-existing pairs).
    pub fn sample_negatives(
        &self,
        n_nodes: usize,
        pos_edges: &[(usize, usize)],
        n_neg: usize,
        rng: &mut StdRng,
    ) -> Vec<(usize, usize)> {
        let pos_set: std::collections::HashSet<(usize, usize)> =
            pos_edges.iter().cloned().collect();
        let mut negs = Vec::new();
        let mut attempts = 0;
        while negs.len() < n_neg && attempts < n_neg * 10 {
            let u = rng.random_range(0..n_nodes);
            let v = rng.random_range(0..n_nodes);
            if u != v && !pos_set.contains(&(u, v)) {
                negs.push((u, v));
            }
            attempts += 1;
        }
        negs
    }
}

/// Attribute masking pre-training: mask node features, predict them.
#[derive(Debug, Clone)]
pub struct AttrMasking {
    /// Feature dimension
    pub d_feat: usize,
    /// Predictor weights [d_feat * d_feat]
    pub pred_w: Vec<f64>,
    /// Value used to mask features
    pub mask_value: f64,
}

impl AttrMasking {
    /// Build attribute masking pretraining.
    pub fn new(d_feat: usize, rng: &mut StdRng) -> Self {
        let pred_w = xavier_vec(d_feat * d_feat, d_feat, d_feat, rng);
        Self {
            d_feat,
            pred_w,
            mask_value: 0.0,
        }
    }

    /// Mask `ratio` fraction of feature dimensions for each node.
    /// Returns `(masked_feats, mask_indices_per_node)`.
    pub fn mask_attrs(
        &self,
        node_feats: &[Vec<f64>],
        ratio: f64,
        rng: &mut StdRng,
    ) -> (Vec<Vec<f64>>, Vec<Vec<usize>>) {
        let n_mask = ((self.d_feat as f64 * ratio).round() as usize).min(self.d_feat);
        let mut masked_feats = node_feats.to_vec();
        let mut mask_indices = Vec::new();
        for feats in masked_feats.iter_mut() {
            let mut indices: Vec<usize> = (0..feats.len()).collect();
            for i in (1..feats.len()).rev() {
                let j = rng.random_range(0..=i);
                indices.swap(i, j);
            }
            let mi: Vec<usize> = indices[..n_mask.min(feats.len())].to_vec();
            for &idx in &mi {
                feats[idx] = self.mask_value;
            }
            mask_indices.push(mi);
        }
        (masked_feats, mask_indices)
    }

    /// Predict original attributes from masked features.
    pub fn predict(&self, masked_feats: &[Vec<f64>]) -> Vec<Vec<f64>> {
        masked_feats
            .iter()
            .map(|f| {
                (0..self.d_feat)
                    .map(|j| {
                        (0..self.d_feat.min(f.len()))
                            .map(|k| f[k] * self.pred_w[k * self.d_feat + j])
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }
}

/// Context prediction: predict local subgraph context (r-hop neighborhood label).
#[derive(Debug, Clone)]
pub struct ContextPrediction {
    /// Number of hops for context radius
    pub hops: usize,
    /// Model dimension
    pub d_model: usize,
    /// Context projection weights
    pub context_w: Vec<Vec<f64>>,
}

impl ContextPrediction {
    /// Build a context prediction pretraining object.
    pub fn new(hops: usize, d_model: usize, rng: &mut StdRng) -> Self {
        let context_w = (0..d_model)
            .map(|_| xavier_vec(d_model, d_model, d_model, rng))
            .collect();
        Self {
            hops,
            d_model,
            context_w,
        }
    }

    /// Get r-hop neighborhood indices for a node.
    pub fn r_hop_neighbors(&self, node: usize, adj: &[Vec<usize>]) -> Vec<usize> {
        let mut visited = std::collections::HashSet::new();
        visited.insert(node);
        let mut frontier = vec![node];
        for _ in 0..self.hops {
            let mut next = Vec::new();
            for &u in &frontier {
                if u < adj.len() {
                    for &v in &adj[u] {
                        if visited.insert(v) {
                            next.push(v);
                        }
                    }
                }
            }
            frontier = next;
        }
        visited.into_iter().filter(|&x| x != node).collect()
    }

    /// Predict context embedding for node from its r-hop neighbors.
    pub fn predict_context(&self, node_feat: &[f64], neighbor_feats: &[Vec<f64>]) -> Vec<f64> {
        if neighbor_feats.is_empty() {
            return (0..self.d_model)
                .map(|j| dot(node_feat, &self.context_w[j]))
                .collect();
        }
        let mean: Vec<f64> = (0..self.d_model.min(node_feat.len()))
            .map(|d| {
                neighbor_feats
                    .iter()
                    .map(|f| if d < f.len() { f[d] } else { 0.0 })
                    .sum::<f64>()
                    / neighbor_feats.len() as f64
            })
            .collect();
        let combined: Vec<f64> = node_feat
            .iter()
            .zip(mean.iter())
            .map(|(a, b)| a + b)
            .collect();
        (0..self.d_model)
            .map(|j| dot(&combined, &self.context_w[j]))
            .collect()
    }
}

/// Graph Contrastive Pre-training: augment graph views, InfoNCE loss.
#[derive(Debug, Clone)]
pub struct GraphContrastivePretraining {
    /// Model dimension
    pub d_model: usize,
    /// InfoNCE temperature
    pub temperature: f64,
    /// Projection head weights
    pub proj_w: Vec<Vec<f64>>,
}

impl GraphContrastivePretraining {
    /// Build a graph contrastive pretraining module.
    pub fn new(d_model: usize, temperature: f64, rng: &mut StdRng) -> Self {
        let proj_w = (0..d_model)
            .map(|_| xavier_vec(d_model, d_model, d_model, rng))
            .collect();
        Self {
            d_model,
            temperature,
            proj_w,
        }
    }

    /// Drop edges randomly to create an augmented view.
    pub fn drop_edges(
        &self,
        edges: &[(usize, usize)],
        drop_ratio: f64,
        rng: &mut StdRng,
    ) -> Vec<(usize, usize)> {
        edges
            .iter()
            .filter(|_| rng.random::<f64>() > drop_ratio)
            .cloned()
            .collect()
    }

    /// Drop node features (zero out random fraction).
    pub fn drop_features(
        &self,
        feats: &[Vec<f64>],
        drop_ratio: f64,
        rng: &mut StdRng,
    ) -> Vec<Vec<f64>> {
        feats
            .iter()
            .map(|f| {
                f.iter()
                    .map(|&x| {
                        if rng.random::<f64>() < drop_ratio {
                            0.0
                        } else {
                            x
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Project embeddings through projection head.
    pub fn project(&self, emb: &[f64]) -> Vec<f64> {
        let d = self.d_model.min(emb.len());
        let h: Vec<f64> = (0..d)
            .map(|j| dot(&emb[..d], &self.proj_w[j][..d]))
            .collect();
        // L2 normalize
        let norm = h.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        h.iter().map(|x| x / norm).collect()
    }

    /// InfoNCE loss between two graph-level embeddings (positive pair).
    /// `others` are negative graph embeddings.
    pub fn info_nce_loss(&self, z1: &[f64], z2: &[f64], negatives: &[Vec<f64>]) -> f64 {
        let pos_sim = dot(z1, z2) / self.temperature;
        let neg_sims: Vec<f64> = negatives
            .iter()
            .map(|z| dot(z1, z) / self.temperature)
            .collect();
        let all_sims: Vec<f64> = std::iter::once(pos_sim)
            .chain(neg_sims.iter().cloned())
            .collect();
        let max = all_sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let log_sum_exp = max + all_sims.iter().map(|&s| (s - max).exp()).sum::<f64>().ln();
        log_sum_exp - pos_sim
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. FEW-SHOT GRAPH LEARNING
// ─────────────────────────────────────────────────────────────────────────────

/// Prototype network for few-shot graph classification.
#[derive(Debug, Clone)]
pub struct GraphProtoNet {
    /// Embedding dimension
    pub d_model: usize,
    /// Embedding network: linear projection
    pub embed_w: Vec<Vec<f64>>,
}

impl GraphProtoNet {
    /// Build a prototype network for few-shot graph classification.
    pub fn new(d_in: usize, d_model: usize, rng: &mut StdRng) -> Self {
        let embed_w = (0..d_model)
            .map(|_| xavier_vec(d_in, d_in, d_model, rng))
            .collect();
        Self { d_model, embed_w }
    }

    /// Embed a graph feature vector.
    pub fn embed(&self, graph_feat: &[f64]) -> Vec<f64> {
        let d = self.embed_w[0].len().min(graph_feat.len());
        (0..self.d_model)
            .map(|j| dot(&graph_feat[..d], &self.embed_w[j][..d]))
            .collect()
    }

    /// Compute class prototypes as mean of support embeddings.
    pub fn compute_prototypes(&self, support: &[(Vec<f64>, usize)]) -> HashMap<usize, Vec<f64>> {
        let mut class_embs: HashMap<usize, Vec<Vec<f64>>> = HashMap::new();
        for (feat, label) in support {
            let emb = self.embed(feat);
            class_embs.entry(*label).or_default().push(emb);
        }
        class_embs
            .into_iter()
            .map(|(cls, embs)| {
                let proto: Vec<f64> = (0..self.d_model)
                    .map(|d| {
                        embs.iter()
                            .map(|e| if d < e.len() { e[d] } else { 0.0 })
                            .sum::<f64>()
                            / embs.len() as f64
                    })
                    .collect();
                (cls, proto)
            })
            .collect()
    }

    /// Predict class for a query graph (nearest prototype).
    pub fn predict(&self, query_feat: &[f64], prototypes: &HashMap<usize, Vec<f64>>) -> usize {
        let q_emb = self.embed(query_feat);
        prototypes
            .iter()
            .min_by(|(_, p1), (_, p2)| {
                let d1: f64 = q_emb
                    .iter()
                    .zip(p1.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                let d2: f64 = q_emb
                    .iter()
                    .zip(p2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                d1.partial_cmp(&d2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(&cls, _)| cls)
            .unwrap_or(0)
    }
}

/// Graph Matching Network: match query graph to support via cross-graph attention.
#[derive(Debug, Clone)]
pub struct GraphMatchingNetwork {
    /// Model dimension
    pub d_model: usize,
    /// Cross-attention query projection
    pub wq: Vec<Vec<f64>>,
    /// Cross-attention key projection
    pub wk: Vec<Vec<f64>>,
}

impl GraphMatchingNetwork {
    /// Build a graph matching network.
    pub fn new(d_model: usize, rng: &mut StdRng) -> Self {
        let wq = (0..d_model)
            .map(|_| xavier_vec(d_model, d_model, d_model, rng))
            .collect();
        let wk = (0..d_model)
            .map(|_| xavier_vec(d_model, d_model, d_model, rng))
            .collect();
        Self { d_model, wq, wk }
    }

    /// Cross-graph attention: query attends to support nodes.
    pub fn cross_attend(&self, query_emb: &[f64], support_embs: &[Vec<f64>]) -> Vec<f64> {
        if support_embs.is_empty() {
            return query_emb.to_vec();
        }
        let d = self.d_model.min(query_emb.len());
        let q: Vec<f64> = (0..d)
            .map(|j| dot(&query_emb[..d], &self.wq[j][..d]))
            .collect();
        let logits: Vec<f64> = support_embs
            .iter()
            .map(|s| {
                let k: Vec<f64> = (0..d)
                    .map(|j| dot(&s[..d.min(s.len())], &self.wk[j][..d]))
                    .collect();
                dot(&q, &k) / (d as f64).sqrt()
            })
            .collect();
        let weights = softmax_1d(&logits);
        let attended: Vec<f64> = (0..d)
            .map(|di| {
                weights
                    .iter()
                    .zip(support_embs.iter())
                    .map(|(w, s)| w * if di < s.len() { s[di] } else { 0.0 })
                    .sum()
            })
            .collect();
        attended
            .iter()
            .zip(query_emb[..d].iter())
            .map(|(a, b)| a + b)
            .collect()
    }

    /// Match score: cosine similarity after cross-attention.
    pub fn match_score(&self, query: &[f64], support: &[Vec<f64>]) -> f64 {
        let q_att = self.cross_attend(query, support);
        let s_mean: Vec<f64> = if support.is_empty() {
            vec![0.0; self.d_model]
        } else {
            (0..self.d_model)
                .map(|d| {
                    support
                        .iter()
                        .map(|s| if d < s.len() { s[d] } else { 0.0 })
                        .sum::<f64>()
                        / support.len() as f64
                })
                .collect()
        };
        let norm_q = q_att.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        let norm_s = s_mean.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        q_att
            .iter()
            .zip(s_mean.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / (norm_q * norm_s)
    }
}

/// MAML for graphs: inner-loop GNN adaptation, outer-loop meta-update.
#[derive(Debug, Clone)]
pub struct MetaGnn {
    /// Feature/model dimension
    pub d_model: usize,
    /// Inner-loop learning rate
    pub inner_lr: f64,
    /// Number of inner adaptation steps
    pub inner_steps: usize,
    /// Base GNN weights (outer loop)
    pub weights: Vec<Vec<f64>>,
}

impl MetaGnn {
    /// Build a meta-learning GNN.
    pub fn new(d_model: usize, inner_lr: f64, inner_steps: usize, rng: &mut StdRng) -> Self {
        let weights = (0..d_model)
            .map(|_| xavier_vec(d_model, d_model, d_model, rng))
            .collect();
        Self {
            d_model,
            inner_lr,
            inner_steps,
            weights,
        }
    }

    /// Simulate inner-loop adaptation (gradient step on task loss).
    pub fn adapt(&self, task_feats: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut adapted = self.weights.clone();
        for _ in 0..self.inner_steps {
            // Simulate gradient: simple gradient descent step
            for (i, row) in adapted.iter_mut().enumerate() {
                if i < task_feats.len() {
                    for (j, w) in row.iter_mut().enumerate() {
                        let grad = if j < task_feats[i].len() {
                            task_feats[i][j] * 0.01
                        } else {
                            0.0
                        };
                        *w -= self.inner_lr * grad;
                    }
                }
            }
        }
        adapted
    }

    /// Run forward with adapted weights.
    pub fn forward(&self, feat: &[f64], weights: &[Vec<f64>]) -> Vec<f64> {
        let d = self.d_model.min(feat.len());
        (0..self.d_model)
            .map(|j| dot(&feat[..d], &weights[j][..d]))
            .collect()
    }
}

/// Task-aware GNN: condition on task embedding from support set.
#[derive(Debug, Clone)]
pub struct TaskAwareGnn {
    /// Node feature dimension
    pub d_model: usize,
    /// Task embedding dimension
    pub task_dim: usize,
    /// Task embedding network
    pub task_w: Vec<Vec<f64>>,
    /// Conditioned GNN weights
    pub gnn_w: Vec<Vec<f64>>,
}

impl TaskAwareGnn {
    /// Build a task-aware GNN.
    pub fn new(d_model: usize, task_dim: usize, rng: &mut StdRng) -> Self {
        let task_w = (0..task_dim)
            .map(|_| xavier_vec(d_model, d_model, task_dim, rng))
            .collect();
        let gnn_w = (0..d_model)
            .map(|_| xavier_vec(d_model + task_dim, d_model + task_dim, d_model, rng))
            .collect();
        Self {
            d_model,
            task_dim,
            task_w,
            gnn_w,
        }
    }

    /// Compute task embedding from support set (mean pooling).
    pub fn encode_task(&self, support_feats: &[Vec<f64>]) -> Vec<f64> {
        if support_feats.is_empty() {
            return vec![0.0; self.task_dim];
        }
        let mean: Vec<f64> = (0..self.d_model)
            .map(|d| {
                support_feats
                    .iter()
                    .map(|f| if d < f.len() { f[d] } else { 0.0 })
                    .sum::<f64>()
                    / support_feats.len() as f64
            })
            .collect();
        (0..self.task_dim)
            .map(|j| dot(&mean, &self.task_w[j]))
            .collect()
    }

    /// Forward: concatenate node feat with task embedding, apply GNN layer.
    pub fn forward(&self, node_feat: &[f64], task_emb: &[f64]) -> Vec<f64> {
        let d_n = self.d_model.min(node_feat.len());
        let d_t = self.task_dim.min(task_emb.len());
        let combined: Vec<f64> = node_feat[..d_n]
            .iter()
            .chain(task_emb[..d_t].iter())
            .cloned()
            .collect();
        let clen = combined.len();
        (0..self.d_model)
            .map(|j| dot(&combined, &self.gnn_w[j][..clen]))
            .collect()
    }
}

/// N-way K-shot episode sampler for few-shot graph learning.
#[derive(Debug, Clone)]
pub struct GraphEpisodeSampler {
    /// Number of classes per episode
    pub n_way: usize,
    /// Number of support examples per class
    pub k_shot: usize,
    /// Number of query examples per class
    pub n_query: usize,
}

impl GraphEpisodeSampler {
    /// Build a few-shot episode sampler.
    pub fn new(n_way: usize, k_shot: usize, n_query: usize) -> Self {
        Self {
            n_way,
            k_shot,
            n_query,
        }
    }

    /// Sample an episode from a labeled dataset.
    /// Returns `(support, query)` as `(Vec<(feat, label)>, Vec<(feat, label)>)`.
    pub fn sample_episode<'a>(
        &self,
        dataset: &'a [(Vec<f64>, usize)],
        rng: &mut StdRng,
    ) -> (Vec<(&'a Vec<f64>, usize)>, Vec<(&'a Vec<f64>, usize)>) {
        // Group by class
        let mut by_class: HashMap<usize, Vec<&'a (Vec<f64>, usize)>> = HashMap::new();
        for item in dataset {
            by_class.entry(item.1).or_default().push(item);
        }
        let classes: Vec<usize> = by_class.keys().cloned().collect();

        // Sample n_way classes
        let mut selected_classes = classes.clone();
        for i in (1..selected_classes.len()).rev() {
            let j = rng.random_range(0..=i);
            selected_classes.swap(i, j);
        }
        selected_classes.truncate(self.n_way);

        let mut support = Vec::new();
        let mut query = Vec::new();

        for cls in &selected_classes {
            if let Some(items) = by_class.get(cls) {
                let mut indices: Vec<usize> = (0..items.len()).collect();
                for i in (1..indices.len()).rev() {
                    let j = rng.random_range(0..=i);
                    indices.swap(i, j);
                }
                for &idx in &indices[..self.k_shot.min(items.len())] {
                    support.push((&items[idx].0, *cls));
                }
                for &idx in &indices[self.k_shot.min(items.len())..]
                    [..self.n_query.min(items.len().saturating_sub(self.k_shot))]
                {
                    query.push((&items[idx].0, *cls));
                }
            }
        }
        (support, query)
    }
}
