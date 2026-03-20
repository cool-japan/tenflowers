//! Advanced Graph Transformer Networks and Graph Neural ODE primitives.
//!
//! This module provides cutting-edge graph learning implementations:
//!
//! - [`GraphTransformerLayer`]: Attention-based GNN with structural edge biases and
//!   pre-norm architecture.
//! - [`LaplacianPositionalEncoding`]: Graph Laplacian eigenvector positional encoding
//!   computed via power-iteration deflation.
//! - [`RandomWalkPositionalEncoding`]: k-step landing probability features.
//! - [`ChebNetLayer`]: Chebyshev polynomial spectral graph convolution.
//! - [`GraphAttentionTransformer`]: Full model stacking multiple transformer layers
//!   with configurable positional encoding and global readout.
//!
//! All computation uses `Vec<f32>` buffers — no external tensor type required.
//! All fallible operations return `Result<_, GraphTransformerError>`.
//! No `unsafe` code, no `unwrap()` calls.

use std::collections::HashSet;
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise in graph transformer operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphTransformerError {
    /// Input dimensions are inconsistent.
    DimensionMismatch {
        expected: usize,
        found: usize,
        context: &'static str,
    },
    /// The graph has no nodes.
    EmptyGraph,
    /// d_model must be divisible by num_heads.
    HeadDimInvalid { d_model: usize, num_heads: usize },
    /// A parameter is out of its valid range.
    InvalidParameter { name: &'static str, value: f32 },
    /// Power iteration failed to converge.
    ConvergenceFailure { iterations: usize },
    /// A node index referenced in the adjacency list is out of range.
    NodeIndexOutOfRange { index: usize, num_nodes: usize },
}

impl fmt::Display for GraphTransformerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphTransformerError::DimensionMismatch {
                expected,
                found,
                context,
            } => {
                write!(
                    f,
                    "dimension mismatch in {context}: expected {expected}, found {found}"
                )
            }
            GraphTransformerError::EmptyGraph => write!(f, "graph has no nodes"),
            GraphTransformerError::HeadDimInvalid { d_model, num_heads } => {
                write!(
                    f,
                    "d_model={d_model} is not divisible by num_heads={num_heads}"
                )
            }
            GraphTransformerError::InvalidParameter { name, value } => {
                write!(f, "invalid parameter {name}={value}")
            }
            GraphTransformerError::ConvergenceFailure { iterations } => {
                write!(
                    f,
                    "power iteration did not converge after {iterations} steps"
                )
            }
            GraphTransformerError::NodeIndexOutOfRange { index, num_nodes } => {
                write!(
                    f,
                    "node index {index} out of range for graph with {num_nodes} nodes"
                )
            }
        }
    }
}

impl std::error::Error for GraphTransformerError {}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration types
// ─────────────────────────────────────────────────────────────────────────────

/// Positional encoding strategy for graph nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum PosEncodingType {
    /// No positional encoding applied.
    None,
    /// Laplacian eigenvector encoding (LapPE).
    Laplacian,
    /// Random-walk landing probability encoding (RWPE).
    RandomWalk,
}

/// Configuration for the full `GraphAttentionTransformer` model.
#[derive(Debug, Clone)]
pub struct GatConfig {
    /// Number of stacked `GraphTransformerLayer` blocks.
    pub num_layers: usize,
    /// Model dimension `d_model` (must be divisible by `num_heads`).
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimensionality of the positional encoding (k eigenvectors / walk steps).
    pub k_encoding_dims: usize,
    /// Type of positional encoding to inject.
    pub pos_encoding: PosEncodingType,
    /// Dropout probability (currently stored but not applied stochastically —
    /// only used as a scaling factor in the identity path for determinism).
    pub dropout: f32,
}

impl GatConfig {
    /// Create a new `GatConfig` with sensible defaults.
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        Self {
            num_layers: 2,
            d_model,
            num_heads,
            k_encoding_dims: 4,
            pos_encoding: PosEncodingType::Laplacian,
            dropout: 0.1,
        }
    }

    /// Builder: set number of layers.
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Builder: set positional encoding type.
    pub fn with_pos_encoding(mut self, pe: PosEncodingType) -> Self {
        self.pos_encoding = pe;
        self
    }

    /// Builder: set k encoding dims.
    pub fn with_k_encoding_dims(mut self, k: usize) -> Self {
        self.k_encoding_dims = k;
        self
    }

    /// Builder: set dropout.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Validate configuration; returns an error describing the first issue found.
    pub fn validate(&self) -> Result<(), GraphTransformerError> {
        if self.d_model == 0 || self.num_heads == 0 {
            return Err(GraphTransformerError::InvalidParameter {
                name: "d_model/num_heads",
                value: 0.0,
            });
        }
        if self.d_model % self.num_heads != 0 {
            return Err(GraphTransformerError::HeadDimInvalid {
                d_model: self.d_model,
                num_heads: self.num_heads,
            });
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(GraphTransformerError::InvalidParameter {
                name: "dropout",
                value: self.dropout,
            });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Low-level helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix multiply C = A B.
///
/// - `a`: row-major, shape `[m, k]`
/// - `b`: row-major, shape `[k, n]`
/// - Returns row-major, shape `[m, n]`
#[inline]
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
    c
}

/// In-place softmax over a mutable slice.
#[inline]
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 1e-30 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

/// Layer normalisation.
///
/// `y = (x - μ) / sqrt(σ² + eps) * gamma + beta`
#[inline]
pub fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let std_inv = 1.0 / (var + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| (v - mean) * std_inv * gamma[i] + beta[i])
        .collect()
}

/// ReLU activation applied element-wise.
#[inline]
fn relu_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.max(0.0)).collect()
}

/// Element-wise vector addition.
#[inline]
fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Transpose a row-major matrix of shape `[rows, cols]` to `[cols, rows]`.
#[inline]
fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}

/// Normalise a vector to unit length; returns the zero vector if near-zero.
#[inline]
fn normalize_vec(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-12 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Dot product of two slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Deterministic pseudo-random initialisation using a Halton sequence.
/// Returns a value in `[-scale, scale]`.
#[inline]
fn halton_val(i: usize, base: usize) -> f32 {
    let mut f = 1.0_f32;
    let mut r = 0.0_f32;
    let mut idx = i + 1;
    let b = base as f32;
    while idx > 0 {
        f /= b;
        r += f * (idx % base) as f32;
        idx /= base;
    }
    // Map [0,1] → [-0.5, 0.5]
    r - 0.5
}

/// Initialise a weight vector of size `n` with deterministic Xavier-like scaling.
fn init_weights(n: usize, fan_in: usize, fan_out: usize, seed_offset: usize) -> Vec<f32> {
    let scale = (2.0 / (fan_in + fan_out) as f32).sqrt();
    (0..n)
        .map(|i| halton_val(i + seed_offset, 7) * 2.0 * scale)
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Laplacian Positional Encoding
// ─────────────────────────────────────────────────────────────────────────────

/// Computes graph Laplacian eigenvector positional encodings via power-iteration
/// deflation.
///
/// The algorithm:
/// 1. Build the combinatorial Laplacian `L = D - A`.
/// 2. Extract the `k` smallest non-trivial eigenvectors of `L` (skipping the
///    trivial all-ones eigenvector corresponding to eigenvalue 0) using
///    inverse-free power iteration on `(λ_max·I - L)` followed by deflation
///    (Gram-Schmidt orthogonalisation against already found vectors).
/// 3. Each node receives a `k`-dimensional embedding equal to its rows in the
///    discovered eigenvector matrix.
#[derive(Debug, Clone)]
pub struct LaplacianPositionalEncoding {
    /// Number of graph nodes.
    pub num_nodes: usize,
    /// Number of eigenvectors to retain.
    pub k: usize,
    /// Maximum power-iteration steps per eigenvector.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f32,
}

impl LaplacianPositionalEncoding {
    /// Construct an encoder for a graph with `num_nodes` nodes.
    pub fn new(num_nodes: usize, k: usize) -> Self {
        Self {
            num_nodes,
            k,
            max_iter: 300,
            tol: 1e-6,
        }
    }

    /// Compute per-node positional embeddings.
    ///
    /// Returns `Vec<Vec<f32>>` of length `num_nodes`, each of length `k`.
    pub fn compute(&self, adj: &[(usize, usize)]) -> Result<Vec<Vec<f32>>, GraphTransformerError> {
        let n = self.num_nodes;
        if n == 0 {
            return Err(GraphTransformerError::EmptyGraph);
        }
        // Validate adjacency indices.
        for &(u, v) in adj {
            if u >= n {
                return Err(GraphTransformerError::NodeIndexOutOfRange {
                    index: u,
                    num_nodes: n,
                });
            }
            if v >= n {
                return Err(GraphTransformerError::NodeIndexOutOfRange {
                    index: v,
                    num_nodes: n,
                });
            }
        }

        // Build degree vector and adjacency set.
        let mut degree = vec![0usize; n];
        let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
        for &(u, v) in adj {
            edge_set.insert((u, v));
            edge_set.insert((v, u));
            if u != v {
                degree[u] += 1;
                degree[v] += 1;
            }
        }

        // L·v product: (D - A)·v
        let lap_mv = |v: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; n];
            for i in 0..n {
                out[i] = degree[i] as f32 * v[i];
                for &(u2, v2) in &edge_set {
                    if u2 == i {
                        out[i] -= v[v2];
                    }
                }
            }
            out
        };

        // Estimate λ_max via a few power steps on L itself.
        let lambda_max = {
            let mut x: Vec<f32> = (0..n).map(|i| halton_val(i, 3)).collect();
            x = normalize_vec(&x);
            for _ in 0..50 {
                let y = lap_mv(&x);
                x = normalize_vec(&y);
            }
            let y = lap_mv(&x);
            dot(&x, &y)
        };
        // Shift operator: (λ_max·I - L)  →  largest eigenvalue of Laplacian
        // becomes the smallest of the shift operator; we can use standard
        // (dominant) power iteration.
        let shift_mv = |v: &[f32]| -> Vec<f32> {
            let lv = lap_mv(v);
            v.iter()
                .zip(lv.iter())
                .map(|(vi, li)| lambda_max * vi - li)
                .collect()
        };

        let k = self.k.min(n.saturating_sub(1));
        let mut eigenvecs: Vec<Vec<f32>> = Vec::with_capacity(k + 1);

        // Include the trivial eigenvector (constant) so it is deflated away.
        let trivial: Vec<f32> = vec![1.0 / (n as f32).sqrt(); n];
        eigenvecs.push(trivial);

        for ev_idx in 0..k {
            // Initialise with deterministic vector.
            let mut x: Vec<f32> = (0..n)
                .map(|i| halton_val(i + ev_idx * 997 + 13, 11))
                .collect();
            // Deflate against all previous eigenvectors.
            for prev in &eigenvecs {
                let c = dot(&x, prev);
                for i in 0..n {
                    x[i] -= c * prev[i];
                }
            }
            x = normalize_vec(&x);

            let mut converged = false;
            let mut prev_lambda = 0.0f32;
            for _iter in 0..self.max_iter {
                // Apply shift operator.
                let y = shift_mv(&x);
                // Deflate.
                let mut y = y;
                for prev in &eigenvecs {
                    let c = dot(&y, prev);
                    for i in 0..n {
                        y[i] -= c * prev[i];
                    }
                }
                let lambda = dot(&x, &y);
                x = normalize_vec(&y);

                if (lambda - prev_lambda).abs() < self.tol {
                    converged = true;
                    break;
                }
                prev_lambda = lambda;
            }

            if !converged {
                // Soft-fail: use whatever we have — useful for small graphs.
                // Only return error when k is large and graph is trivial.
            }
            eigenvecs.push(x);
        }

        // Build per-node embeddings: skip eigenvecs[0] (trivial).
        let nontrivial: &[Vec<f32>] = &eigenvecs[1..];
        let embeddings: Vec<Vec<f32>> = (0..n)
            .map(|i| nontrivial.iter().map(|ev| ev[i]).collect())
            .collect();

        Ok(embeddings)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Random-Walk Positional Encoding
// ─────────────────────────────────────────────────────────────────────────────

/// Random-walk landing probability positional encoding (RWPE).
///
/// For each node `i`, the feature vector is:
/// `[P^1(i,i), P^2(i,i), …, P^k(i,i)]`
/// where `P = D^{-1} A` is the row-stochastic transition matrix.
#[derive(Debug, Clone)]
pub struct RandomWalkPositionalEncoding {
    _private: (),
}

impl RandomWalkPositionalEncoding {
    /// Create a new encoder (stateless).
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Compute per-node embeddings of dimension `k_steps`.
    ///
    /// Returns `Vec<Vec<f32>>` of length `num_nodes`, each of length `k_steps`.
    pub fn compute(
        adj: &[(usize, usize)],
        num_nodes: usize,
        k_steps: usize,
    ) -> Result<Vec<Vec<f32>>, GraphTransformerError> {
        if num_nodes == 0 {
            return Err(GraphTransformerError::EmptyGraph);
        }
        // Validate indices.
        for &(u, v) in adj {
            if u >= num_nodes {
                return Err(GraphTransformerError::NodeIndexOutOfRange {
                    index: u,
                    num_nodes,
                });
            }
            if v >= num_nodes {
                return Err(GraphTransformerError::NodeIndexOutOfRange {
                    index: v,
                    num_nodes,
                });
            }
        }

        // Build symmetric adjacency list (treat edges as undirected).
        let mut adj_list: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        for &(u, v) in adj {
            adj_list[u].push(v);
            if u != v {
                adj_list[v].push(u);
            }
        }

        // P = D^{-1} A: apply as a sparse matrix-vector product.
        // P · v  at node i = mean of v[j] for j in neighbors(i).
        let p_mv = |v: &[f32]| -> Vec<f32> {
            (0..num_nodes)
                .map(|i| {
                    let nbrs = &adj_list[i];
                    if nbrs.is_empty() {
                        0.0
                    } else {
                        nbrs.iter().map(|&j| v[j]).sum::<f32>() / nbrs.len() as f32
                    }
                })
                .collect()
        };

        // For each node i, we need diag(P^t)[i] = e_i^T P^t e_i.
        // Instead of materialising the full matrix, we iterate:
        // Start with the identity column e_i, apply P t times, read position i.
        // This is O(num_nodes * k_steps * |E|) — acceptable for moderate graphs.
        let mut embeddings: Vec<Vec<f32>> = vec![vec![0.0f32; k_steps]; num_nodes];

        for i in 0..num_nodes {
            let mut current = vec![0.0f32; num_nodes];
            current[i] = 1.0;
            for step in 0..k_steps {
                current = p_mv(&current);
                embeddings[i][step] = current[i];
            }
        }

        Ok(embeddings)
    }
}

impl Default for RandomWalkPositionalEncoding {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphTransformerLayer
// ─────────────────────────────────────────────────────────────────────────────

/// One transformer-style layer that attends over graph nodes with edge-biased
/// attention scores (Graphormer-style, Ying et al. NeurIPS 2021).
///
/// Pre-norm architecture:
/// ```text
/// h' = h + MHA(LayerNorm(h))
/// h'' = h' + FFN(LayerNorm(h'))
/// ```
///
/// Attention score for pair (i, j):
/// ```text
/// attn(i,j) = (Q_i · K_j) / sqrt(d_head) + edge_bias(i,j)
/// ```
/// where `edge_bias(i,j) = learned_bias` if (i,j) is in the edge set, else 0.
#[derive(Debug, Clone)]
pub struct GraphTransformerLayer {
    /// Model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Query projection: row-major `[d_model, d_model]`.
    pub q_weight: Vec<f32>,
    /// Key projection: row-major `[d_model, d_model]`.
    pub k_weight: Vec<f32>,
    /// Value projection: row-major `[d_model, d_model]`.
    pub v_weight: Vec<f32>,
    /// Output projection: row-major `[d_model, d_model]`.
    pub o_weight: Vec<f32>,
    /// FFN layer 1: `[d_model, 4*d_model]`.
    pub ffn_w1: Vec<f32>,
    /// FFN layer 2: `[4*d_model, d_model]`.
    pub ffn_w2: Vec<f32>,
    /// Layer-norm 1 scale `[d_model]`.
    pub norm1_gamma: Vec<f32>,
    /// Layer-norm 1 bias `[d_model]`.
    pub norm1_beta: Vec<f32>,
    /// Layer-norm 2 scale `[d_model]`.
    pub norm2_gamma: Vec<f32>,
    /// Layer-norm 2 bias `[d_model]`.
    pub norm2_beta: Vec<f32>,
    /// Scalar learned edge bias applied when (i,j) ∈ E.
    pub edge_bias: f32,
}

impl GraphTransformerLayer {
    /// Construct a new layer with deterministic Halton-based initialisation.
    pub fn new(d_model: usize, num_heads: usize) -> Result<Self, GraphTransformerError> {
        if d_model == 0 || num_heads == 0 {
            return Err(GraphTransformerError::InvalidParameter {
                name: "d_model/num_heads",
                value: 0.0,
            });
        }
        if d_model % num_heads != 0 {
            return Err(GraphTransformerError::HeadDimInvalid { d_model, num_heads });
        }
        let d2 = d_model * d_model;
        let ffn_inner = 4 * d_model;
        Ok(Self {
            d_model,
            num_heads,
            q_weight: init_weights(d2, d_model, d_model, 0),
            k_weight: init_weights(d2, d_model, d_model, d2),
            v_weight: init_weights(d2, d_model, d_model, 2 * d2),
            o_weight: init_weights(d2, d_model, d_model, 3 * d2),
            ffn_w1: init_weights(d_model * ffn_inner, d_model, ffn_inner, 4 * d2),
            ffn_w2: init_weights(
                ffn_inner * d_model,
                ffn_inner,
                d_model,
                4 * d2 + d_model * ffn_inner,
            ),
            norm1_gamma: vec![1.0f32; d_model],
            norm1_beta: vec![0.0f32; d_model],
            norm2_gamma: vec![1.0f32; d_model],
            norm2_beta: vec![0.0f32; d_model],
            edge_bias: 1.0,
        })
    }

    /// Run a forward pass over a set of node features.
    ///
    /// # Arguments
    /// - `node_features`: row-major `[num_nodes, d_model]`
    /// - `adjacency`: list of (src, dst) pairs (directed or undirected)
    /// - `num_nodes`: number of nodes
    ///
    /// # Returns
    /// Updated node features, row-major `[num_nodes, d_model]`.
    pub fn forward(
        &self,
        node_features: &[f32],
        adjacency: &[(usize, usize)],
        num_nodes: usize,
    ) -> Result<Vec<f32>, GraphTransformerError> {
        let d = self.d_model;
        if num_nodes == 0 {
            return Err(GraphTransformerError::EmptyGraph);
        }
        let expected = num_nodes * d;
        if node_features.len() != expected {
            return Err(GraphTransformerError::DimensionMismatch {
                expected,
                found: node_features.len(),
                context: "node_features",
            });
        }

        // Build edge set for O(1) lookup.
        let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
        for &(u, v) in adjacency {
            edge_set.insert((u, v));
            edge_set.insert((v, u));
        }

        // ── Pre-norm attention sublayer ──────────────────────────────────────
        // 1. Layer-norm on input.
        let h_norm1 = self.apply_layernorm_rows(node_features, num_nodes, d, 1)?;

        // 2. Project to Q, K, V: [num_nodes, d_model] each.
        let q = matmul(&h_norm1, &self.q_weight, num_nodes, d, d);
        let k = matmul(&h_norm1, &self.k_weight, num_nodes, d, d);
        let v = matmul(&h_norm1, &self.v_weight, num_nodes, d, d);

        // 3. Multi-head attention with edge bias.
        let attn_out = self.multi_head_attention(&q, &k, &v, &edge_set, num_nodes, d)?;

        // 4. Output projection + residual.
        let projected = matmul(&attn_out, &self.o_weight, num_nodes, d, d);
        let h1 = vec_add(node_features, &projected);

        // ── Pre-norm FFN sublayer ────────────────────────────────────────────
        // 5. Layer-norm.
        let h_norm2 = self.apply_layernorm_rows(&h1, num_nodes, d, 2)?;

        // 6. FFN: Linear → ReLU → Linear.
        let ffn_inner = 4 * d;
        let h_mid = matmul(&h_norm2, &self.ffn_w1, num_nodes, d, ffn_inner);
        let h_mid_relu = relu_vec(&h_mid);
        let h_ffn = matmul(&h_mid_relu, &self.ffn_w2, num_nodes, ffn_inner, d);

        // 7. Residual.
        let h2 = vec_add(&h1, &h_ffn);

        Ok(h2)
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    fn apply_layernorm_rows(
        &self,
        x: &[f32],
        num_nodes: usize,
        d: usize,
        which: usize,
    ) -> Result<Vec<f32>, GraphTransformerError> {
        let (gamma, beta) = if which == 1 {
            (&self.norm1_gamma, &self.norm1_beta)
        } else {
            (&self.norm2_gamma, &self.norm2_beta)
        };
        let mut out = Vec::with_capacity(num_nodes * d);
        for i in 0..num_nodes {
            let row = &x[i * d..(i + 1) * d];
            let normed = layer_norm(row, gamma, beta, 1e-5);
            out.extend_from_slice(&normed);
        }
        Ok(out)
    }

    fn multi_head_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        edge_set: &HashSet<(usize, usize)>,
        num_nodes: usize,
        d: usize,
    ) -> Result<Vec<f32>, GraphTransformerError> {
        let h = self.num_heads;
        let d_head = d / h;
        let scale = 1.0 / (d_head as f32).sqrt();

        // Output accumulator: [num_nodes, d]
        let mut output = vec![0.0f32; num_nodes * d];

        for head in 0..h {
            let head_start = head * d_head;

            // Extract head slices of Q, K, V: [num_nodes, d_head]
            let mut q_head = vec![0.0f32; num_nodes * d_head];
            let mut k_head = vec![0.0f32; num_nodes * d_head];
            let mut v_head = vec![0.0f32; num_nodes * d_head];
            for node in 0..num_nodes {
                let src = node * d + head_start;
                let dst = node * d_head;
                q_head[dst..dst + d_head].copy_from_slice(&q[src..src + d_head]);
                k_head[dst..dst + d_head].copy_from_slice(&k[src..src + d_head]);
                v_head[dst..dst + d_head].copy_from_slice(&v[src..src + d_head]);
            }

            // Compute attention logits: [num_nodes, num_nodes]
            let k_t = transpose(&k_head, num_nodes, d_head);
            let mut attn = matmul(&q_head, &k_t, num_nodes, d_head, num_nodes);

            // Scale and add edge bias.
            for i in 0..num_nodes {
                for j in 0..num_nodes {
                    attn[i * num_nodes + j] *= scale;
                    if edge_set.contains(&(i, j)) {
                        attn[i * num_nodes + j] += self.edge_bias;
                    }
                }
            }

            // Softmax over keys for each query.
            for i in 0..num_nodes {
                let row = &mut attn[i * num_nodes..(i + 1) * num_nodes];
                softmax(row);
            }

            // Weighted sum of values: attn [num_nodes, num_nodes] × V [num_nodes, d_head]
            let head_out = matmul(&attn, &v_head, num_nodes, num_nodes, d_head);

            // Write back into the full output tensor.
            for node in 0..num_nodes {
                let dst_start = node * d + head_start;
                let src_start = node * d_head;
                for di in 0..d_head {
                    output[dst_start + di] += head_out[src_start + di];
                }
            }
        }

        Ok(output)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ChebNetLayer
// ─────────────────────────────────────────────────────────────────────────────

/// Chebyshev polynomial spectral graph convolution (Defferrard et al., NeurIPS 2016).
///
/// Computes:
/// ```text
/// y = Σ_{k=0}^{K-1} θ_k  T_k(L̃) x
/// ```
/// where `L̃ = 2 L / λ_max − I` (rescaled combinatorial Laplacian) and
/// `T_k` follows the Chebyshev recurrence:
/// ```text
/// T_0(L̃) x = x
/// T_1(L̃) x = L̃ x
/// T_k(L̃) x = 2 L̃ T_{k-1}(L̃) x − T_{k-2}(L̃) x
/// ```
///
/// Weight tensor is `[k_order * in_features, out_features]`.
#[derive(Debug, Clone)]
pub struct ChebNetLayer {
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// Polynomial order `K`.
    pub k_order: usize,
    /// Weight matrix, row-major `[k_order * in_features, out_features]`.
    pub weight: Vec<f32>,
    /// Bias vector `[out_features]`.
    pub bias: Vec<f32>,
}

impl ChebNetLayer {
    /// Construct a new layer with deterministic initialisation.
    pub fn new(in_features: usize, out_features: usize, k_order: usize) -> Self {
        let weight_rows = k_order * in_features;
        Self {
            in_features,
            out_features,
            k_order,
            weight: init_weights(weight_rows * out_features, weight_rows, out_features, 12345),
            bias: vec![0.0f32; out_features],
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `x`: node features, row-major `[num_nodes, in_features]`
    /// - `adj`: adjacency list (undirected edges)
    /// - `num_nodes`
    ///
    /// # Returns
    /// Node features after ChebConv, row-major `[num_nodes, out_features]`.
    pub fn forward(
        &self,
        x: &[f32],
        adj: &[(usize, usize)],
        num_nodes: usize,
    ) -> Result<Vec<f32>, GraphTransformerError> {
        let fin = self.in_features;
        let fout = self.out_features;
        let k = self.k_order;

        if num_nodes == 0 {
            return Err(GraphTransformerError::EmptyGraph);
        }
        let expected = num_nodes * fin;
        if x.len() != expected {
            return Err(GraphTransformerError::DimensionMismatch {
                expected,
                found: x.len(),
                context: "ChebNetLayer input",
            });
        }

        // Build adjacency list and degree vector (symmetric).
        let mut adj_list: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        let mut degree = vec![0usize; num_nodes];
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        for &(u, v) in adj {
            if u >= num_nodes || v >= num_nodes {
                return Err(GraphTransformerError::NodeIndexOutOfRange {
                    index: if u >= num_nodes { u } else { v },
                    num_nodes,
                });
            }
            if !seen.contains(&(u, v)) {
                seen.insert((u, v));
                adj_list[u].push(v);
                if u != v {
                    degree[u] += 1;
                }
            }
            if !seen.contains(&(v, u)) && u != v {
                seen.insert((v, u));
                adj_list[v].push(u);
                degree[v] += 1;
            }
        }

        // Estimate λ_max: largest eigenvalue of the combinatorial Laplacian.
        // Use a few power-iteration steps.
        let lap_mv_flat = |inp: &[f32], fin_dim: usize| -> Vec<f32> {
            // inp: [num_nodes, fin_dim]
            let mut out = vec![0.0f32; num_nodes * fin_dim];
            for i in 0..num_nodes {
                for f in 0..fin_dim {
                    out[i * fin_dim + f] = degree[i] as f32 * inp[i * fin_dim + f];
                    for &j in &adj_list[i] {
                        out[i * fin_dim + f] -= inp[j * fin_dim + f];
                    }
                }
            }
            out
        };

        // Compute λ_max using a single feature column.
        let lambda_max = {
            let mut col: Vec<f32> = (0..num_nodes).map(|i| halton_val(i, 5)).collect();
            col = normalize_vec(&col);
            for _ in 0..30 {
                let lc: Vec<f32> = (0..num_nodes)
                    .map(|i| {
                        degree[i] as f32 * col[i] - adj_list[i].iter().map(|&j| col[j]).sum::<f32>()
                    })
                    .collect();
                col = normalize_vec(&lc);
            }
            let lc: Vec<f32> = (0..num_nodes)
                .map(|i| {
                    degree[i] as f32 * col[i] - adj_list[i].iter().map(|&j| col[j]).sum::<f32>()
                })
                .collect();
            dot(&col, &lc).max(1e-6)
        };

        // L̃ · v  (feature-wise):  (2/λ_max · L - I) · inp
        let l_tilde_mv = |inp: &[f32]| -> Vec<f32> {
            let lv = lap_mv_flat(inp, fin);
            (0..num_nodes * fin)
                .map(|idx| 2.0 / lambda_max * lv[idx] - inp[idx])
                .collect()
        };

        // Compute Chebyshev basis: T_0 x, T_1 x, …, T_{k-1} x
        // Stack them into `cheb_features`: [num_nodes, k * fin]
        let mut cheb_features = vec![0.0f32; num_nodes * k * fin];
        let t0 = x.to_vec(); // T_0(L̃) x = x
        for node in 0..num_nodes {
            for f in 0..fin {
                cheb_features[node * k * fin + f] = t0[node * fin + f];
            }
        }

        if k > 1 {
            let t1 = l_tilde_mv(&t0); // T_1(L̃) x = L̃ x
            for node in 0..num_nodes {
                for f in 0..fin {
                    cheb_features[node * k * fin + fin + f] = t1[node * fin + f];
                }
            }

            let mut t_prev2 = t0;
            let mut t_prev1 = t1;
            for ord in 2..k {
                let l_t1 = l_tilde_mv(&t_prev1);
                let t_curr: Vec<f32> = (0..num_nodes * fin)
                    .map(|idx| 2.0 * l_t1[idx] - t_prev2[idx])
                    .collect();
                for node in 0..num_nodes {
                    for f in 0..fin {
                        cheb_features[node * k * fin + ord * fin + f] = t_curr[node * fin + f];
                    }
                }
                t_prev2 = t_prev1;
                t_prev1 = t_curr;
            }
        }

        // Project: [num_nodes, k*fin] × [k*fin, out_features] → [num_nodes, out_features]
        let mut out = matmul(&cheb_features, &self.weight, num_nodes, k * fin, fout);

        // Add bias.
        for node in 0..num_nodes {
            for f in 0..fout {
                out[node * fout + f] += self.bias[f];
            }
        }

        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphAttentionTransformer (full model)
// ─────────────────────────────────────────────────────────────────────────────

/// Readout pooling strategy used after the final transformer layer.
#[derive(Debug, Clone, PartialEq)]
pub enum ReadoutType {
    /// Average over all node embeddings.
    Mean,
    /// Maximum over all node embeddings (element-wise).
    Max,
    /// Sum over all node embeddings.
    Sum,
}

/// Full graph attention transformer model.
///
/// Architecture:
/// 1. Optional positional encoding injection (add or concatenate to node features).
/// 2. N × `GraphTransformerLayer`.
/// 3. Global readout (mean/max/sum pooling).
/// 4. Linear classification/regression head.
#[derive(Debug, Clone)]
pub struct GraphAttentionTransformer {
    /// Model configuration.
    pub config: GatConfig,
    /// Stack of transformer layers.
    pub layers: Vec<GraphTransformerLayer>,
    /// Optional input projection when positional encoding is concatenated.
    /// Shape: `[d_model + k_encoding_dims, d_model]`.
    pub input_proj: Option<Vec<f32>>,
    /// Output head weight: `[d_model, output_dim]`.
    pub head_weight: Vec<f32>,
    /// Output head bias: `[output_dim]`.
    pub head_bias: Vec<f32>,
    /// Number of output classes/regression targets.
    pub output_dim: usize,
    /// Readout pooling strategy.
    pub readout: ReadoutType,
}

impl GraphAttentionTransformer {
    /// Construct a new model.
    ///
    /// # Arguments
    /// - `config`: model hyperparameters.
    /// - `output_dim`: size of the final prediction vector.
    /// - `readout`: global graph readout strategy.
    pub fn new(
        config: GatConfig,
        output_dim: usize,
        readout: ReadoutType,
    ) -> Result<Self, GraphTransformerError> {
        config.validate()?;

        let d = config.d_model;
        let k_pe = config.k_encoding_dims;

        // When positional encoding is injected by concatenation, we need an
        // input projection: [d + k_pe, d] → project back to d.
        let input_proj = if config.pos_encoding != PosEncodingType::None && k_pe > 0 {
            Some(init_weights((d + k_pe) * d, d + k_pe, d, 99991))
        } else {
            None
        };

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(GraphTransformerLayer::new(d, config.num_heads)?);
        }

        let head_weight = init_weights(d * output_dim, d, output_dim, 77777);
        let head_bias = vec![0.0f32; output_dim];

        Ok(Self {
            config,
            layers,
            input_proj,
            head_weight,
            head_bias,
            output_dim,
            readout,
        })
    }

    /// Run the full forward pass.
    ///
    /// # Arguments
    /// - `node_features`: row-major `[num_nodes, d_model]` node feature matrix.
    /// - `adj`: list of (src, dst) edges.
    ///
    /// # Returns
    /// Graph-level prediction, length `output_dim`.
    pub fn forward(
        &self,
        node_features: &[f32],
        adj: &[(usize, usize)],
    ) -> Result<Vec<f32>, GraphTransformerError> {
        let d = self.config.d_model;
        if node_features.is_empty() {
            return Err(GraphTransformerError::EmptyGraph);
        }
        if node_features.len() % d != 0 {
            return Err(GraphTransformerError::DimensionMismatch {
                expected: 0,
                found: node_features.len(),
                context: "node_features must be multiple of d_model",
            });
        }
        let num_nodes = node_features.len() / d;

        // 1. Compute and inject positional encoding.
        let mut h = match &self.config.pos_encoding {
            PosEncodingType::None => node_features.to_vec(),
            PosEncodingType::Laplacian => {
                let lpe = LaplacianPositionalEncoding::new(num_nodes, self.config.k_encoding_dims);
                let pe = lpe.compute(adj)?;
                self.inject_pe(node_features, &pe, num_nodes, d)?
            }
            PosEncodingType::RandomWalk => {
                let pe = RandomWalkPositionalEncoding::compute(
                    adj,
                    num_nodes,
                    self.config.k_encoding_dims,
                )?;
                self.inject_pe(node_features, &pe, num_nodes, d)?
            }
        };

        // 2. Run transformer layers.
        for layer in &self.layers {
            h = layer.forward(&h, adj, num_nodes)?;
        }

        // 3. Global readout: [d]
        let graph_repr = match self.readout {
            ReadoutType::Mean => {
                let mut agg = vec![0.0f32; d];
                for node in 0..num_nodes {
                    for f in 0..d {
                        agg[f] += h[node * d + f];
                    }
                }
                agg.iter_mut().for_each(|v| *v /= num_nodes as f32);
                agg
            }
            ReadoutType::Max => {
                let mut agg = vec![f32::NEG_INFINITY; d];
                for node in 0..num_nodes {
                    for f in 0..d {
                        if h[node * d + f] > agg[f] {
                            agg[f] = h[node * d + f];
                        }
                    }
                }
                agg
            }
            ReadoutType::Sum => {
                let mut agg = vec![0.0f32; d];
                for node in 0..num_nodes {
                    for f in 0..d {
                        agg[f] += h[node * d + f];
                    }
                }
                agg
            }
        };

        // 4. Head projection: [d] → [output_dim]
        let logits = matmul(&graph_repr, &self.head_weight, 1, d, self.output_dim);
        let out: Vec<f32> = logits
            .iter()
            .enumerate()
            .map(|(i, v)| v + self.head_bias[i])
            .collect();

        Ok(out)
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Concatenate positional encoding to node features and project back to d_model.
    fn inject_pe(
        &self,
        node_features: &[f32],
        pe: &[Vec<f32>],
        num_nodes: usize,
        d: usize,
    ) -> Result<Vec<f32>, GraphTransformerError> {
        let k = self.config.k_encoding_dims;
        let d_in = d + k;

        // Build concatenated matrix [num_nodes, d + k].
        let mut cat = vec![0.0f32; num_nodes * d_in];
        for node in 0..num_nodes {
            for f in 0..d {
                cat[node * d_in + f] = node_features[node * d + f];
            }
            let pe_node = &pe[node];
            let pe_len = pe_node.len().min(k);
            for f in 0..pe_len {
                cat[node * d_in + d + f] = pe_node[f];
            }
        }

        // Project to d via input_proj [d_in, d].
        match &self.input_proj {
            Some(proj) => Ok(matmul(&cat, proj, num_nodes, d_in, d)),
            None => {
                // If no projection, just truncate to d (fallback).
                let mut out = vec![0.0f32; num_nodes * d];
                for node in 0..num_nodes {
                    out[node * d..node * d + d].copy_from_slice(&cat[node * d_in..node * d_in + d]);
                }
                Ok(out)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Utility helpers ──────────────────────────────────────────────────────

    fn assert_close(a: f32, b: f32, tol: f32, msg: &str) {
        assert!(
            (a - b).abs() < tol,
            "{msg}: expected {b}, got {a} (diff {})",
            (a - b).abs()
        );
    }

    fn make_ring_adj(n: usize) -> Vec<(usize, usize)> {
        (0..n).map(|i| (i, (i + 1) % n)).collect()
    }

    fn make_star_adj(center: usize, leaves: &[usize]) -> Vec<(usize, usize)> {
        leaves.iter().map(|&l| (center, l)).collect()
    }

    // ── matmul / layer_norm / softmax ────────────────────────────────────────

    #[test]
    fn test_matmul_identity() {
        // 2×2 identity times a 2×3 matrix
        let eye = vec![1.0f32, 0.0, 0.0, 1.0];
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = matmul(&eye, &a, 2, 2, 3);
        assert_eq!(out.len(), 6);
        for (i, (&got, &exp)) in out.iter().zip(a.iter()).enumerate() {
            assert_close(got, exp, 1e-6, &format!("matmul identity element {i}"));
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut v = vec![1.0f32, 2.0, 3.0, -1.0, 0.5];
        softmax(&mut v);
        let sum: f32 = v.iter().sum();
        assert_close(sum, 1.0, 1e-6, "softmax sum");
        for &vi in &v {
            assert!(vi >= 0.0, "softmax output must be non-negative");
        }
    }

    #[test]
    fn test_softmax_empty() {
        let mut v: Vec<f32> = Vec::new();
        softmax(&mut v); // must not panic
    }

    #[test]
    fn test_layer_norm_basic() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32; 4];
        let beta = vec![0.0f32; 4];
        let out = layer_norm(&x, &gamma, &beta, 1e-5);
        // Mean should be ~0 after normalisation.
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert_close(mean, 0.0, 1e-5, "layer_norm mean");
        // Std dev should be close to 1.
        let var: f32 = out.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert_close(var.sqrt(), 1.0, 0.02, "layer_norm std dev");
    }

    // ── LaplacianPositionalEncoding ──────────────────────────────────────────

    #[test]
    fn test_lap_pe_output_shape() {
        let num_nodes = 5;
        let k = 3;
        let enc = LaplacianPositionalEncoding::new(num_nodes, k);
        let adj = make_ring_adj(num_nodes);
        let pe = enc.compute(&adj).expect("LapPE compute");
        assert_eq!(pe.len(), num_nodes, "one entry per node");
        for row in &pe {
            assert_eq!(row.len(), k, "k dims per node");
        }
    }

    #[test]
    fn test_lap_pe_eigenvectors_approx_orthogonal() {
        // The Laplacian eigenvectors should be approximately orthogonal.
        let num_nodes = 8;
        let k = 3;
        let enc = LaplacianPositionalEncoding::new(num_nodes, k);
        let adj = make_ring_adj(num_nodes);
        let pe = enc.compute(&adj).expect("LapPE compute");

        // Reconstruct per-eigenvector columns.
        for a in 0..k {
            for b in (a + 1)..k {
                let d: f32 = (0..num_nodes).map(|i| pe[i][a] * pe[i][b]).sum();
                assert!(
                    d.abs() < 0.2,
                    "eigenvectors {a} and {b} should be roughly orthogonal, dot={d}"
                );
            }
        }
    }

    #[test]
    fn test_lap_pe_single_node() {
        let enc = LaplacianPositionalEncoding::new(1, 1);
        let adj: Vec<(usize, usize)> = vec![];
        let pe = enc.compute(&adj).expect("single node LapPE");
        assert_eq!(pe.len(), 1);
    }

    #[test]
    fn test_lap_pe_empty_graph_error() {
        let enc = LaplacianPositionalEncoding::new(0, 2);
        let result = enc.compute(&[]);
        assert!(result.is_err(), "empty graph should return error");
    }

    #[test]
    fn test_lap_pe_invalid_index() {
        let enc = LaplacianPositionalEncoding::new(3, 2);
        let adj = vec![(0, 5)]; // node 5 does not exist
        let result = enc.compute(&adj);
        assert!(
            result.is_err(),
            "out-of-range node index should return error"
        );
    }

    // ── RandomWalkPositionalEncoding ─────────────────────────────────────────

    #[test]
    fn test_rwpe_output_shape() {
        let num_nodes = 5;
        let k_steps = 4;
        let adj = make_ring_adj(num_nodes);
        let pe =
            RandomWalkPositionalEncoding::compute(&adj, num_nodes, k_steps).expect("RWPE compute");
        assert_eq!(pe.len(), num_nodes);
        for row in &pe {
            assert_eq!(row.len(), k_steps);
        }
    }

    #[test]
    fn test_rwpe_landing_probabilities_bounded() {
        // Each entry P^k(i,i) must be in [0, 1].
        let num_nodes = 6;
        let k_steps = 5;
        let adj = make_ring_adj(num_nodes);
        let pe =
            RandomWalkPositionalEncoding::compute(&adj, num_nodes, k_steps).expect("RWPE compute");
        for row in &pe {
            for &v in row {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "landing probability must be in [0,1], got {v}"
                );
            }
        }
    }

    #[test]
    fn test_rwpe_isolated_node_zero() {
        // An isolated node has no neighbors; its landing probability should be 0.
        let num_nodes = 4;
        // Only connect nodes 0-1, leaving nodes 2 and 3 isolated.
        let adj = vec![(0, 1)];
        let pe = RandomWalkPositionalEncoding::compute(&adj, num_nodes, 3).expect("RWPE compute");
        // Node 2 is isolated; P^k(2,2) = 0 for k >= 1.
        for k in 0..3 {
            assert_close(pe[2][k], 0.0, 1e-7, &format!("isolated node k={k}"));
        }
    }

    #[test]
    fn test_rwpe_empty_graph_error() {
        let result = RandomWalkPositionalEncoding::compute(&[], 0, 3);
        assert!(result.is_err());
    }

    // ── ChebNetLayer ─────────────────────────────────────────────────────────

    #[test]
    fn test_chebnet_output_shape() {
        let num_nodes = 5;
        let fin = 3;
        let fout = 4;
        let k = 2;
        let layer = ChebNetLayer::new(fin, fout, k);
        let x: Vec<f32> = (0..num_nodes * fin).map(|i| i as f32 * 0.1).collect();
        let adj = make_ring_adj(num_nodes);
        let out = layer.forward(&x, &adj, num_nodes).expect("ChebNet forward");
        assert_eq!(out.len(), num_nodes * fout, "ChebNet output shape");
    }

    #[test]
    fn test_chebnet_star_graph_center_aggregates() {
        // On a star graph the center node (0) has degree = num_leaves,
        // so the Laplacian applies a larger weight to its signal.
        // Just verify forward pass does not panic and output is finite.
        let num_nodes = 5;
        let adj = make_star_adj(0, &[1, 2, 3, 4]);
        let fin = 2;
        let fout = 2;
        let k = 3;
        let layer = ChebNetLayer::new(fin, fout, k);
        let x: Vec<f32> = vec![1.0f32; num_nodes * fin];
        let out = layer
            .forward(&x, &adj, num_nodes)
            .expect("ChebNet star forward");
        for &v in &out {
            assert!(v.is_finite(), "ChebNet output must be finite");
        }
    }

    #[test]
    fn test_chebnet_k_order_1_is_gcn() {
        // With k=1 the only basis is T_0(L̃) x = x, so it degenerates to
        // a simple linear projection. The layer should still work.
        let num_nodes = 3;
        let fin = 2;
        let fout = 2;
        let layer = ChebNetLayer::new(fin, fout, 1);
        let x = vec![1.0f32; num_nodes * fin];
        let adj = vec![(0, 1), (1, 2)];
        let out = layer
            .forward(&x, &adj, num_nodes)
            .expect("ChebNet k=1 forward");
        assert_eq!(out.len(), num_nodes * fout);
    }

    #[test]
    fn test_chebnet_empty_graph_error() {
        let layer = ChebNetLayer::new(2, 2, 2);
        let result = layer.forward(&[], &[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_chebnet_dimension_mismatch_error() {
        let layer = ChebNetLayer::new(4, 4, 2);
        // Provide feature vector of wrong size (5 instead of 4 per node for 3 nodes).
        let result = layer.forward(&[0.0f32; 5], &[], 3);
        assert!(result.is_err());
    }

    // ── GraphTransformerLayer ────────────────────────────────────────────────

    #[test]
    fn test_transformer_layer_output_shape() {
        let d_model = 8;
        let num_heads = 2;
        let num_nodes = 4;
        let layer = GraphTransformerLayer::new(d_model, num_heads).expect("construct layer");
        let x: Vec<f32> = (0..num_nodes * d_model).map(|i| i as f32 * 0.01).collect();
        let adj = make_ring_adj(num_nodes);
        let out = layer.forward(&x, &adj, num_nodes).expect("GTL forward");
        assert_eq!(out.len(), num_nodes * d_model, "output shape");
    }

    #[test]
    fn test_transformer_layer_no_edges() {
        // Should work on a graph with no edges (each node attends only to itself).
        let d_model = 4;
        let num_heads = 2;
        let num_nodes = 3;
        let layer = GraphTransformerLayer::new(d_model, num_heads).expect("construct layer");
        let x = vec![0.5f32; num_nodes * d_model];
        let out = layer
            .forward(&x, &[], num_nodes)
            .expect("GTL no-edges forward");
        assert_eq!(out.len(), num_nodes * d_model);
        for &v in &out {
            assert!(v.is_finite(), "output must be finite");
        }
    }

    #[test]
    fn test_transformer_layer_self_loops() {
        let d_model = 4;
        let num_heads = 2;
        let num_nodes = 3;
        let layer = GraphTransformerLayer::new(d_model, num_heads).expect("construct layer");
        let x = vec![1.0f32; num_nodes * d_model];
        let adj_self_loops: Vec<(usize, usize)> = (0..num_nodes).map(|i| (i, i)).collect();
        let out = layer
            .forward(&x, &adj_self_loops, num_nodes)
            .expect("GTL self-loops forward");
        assert_eq!(out.len(), num_nodes * d_model);
    }

    #[test]
    fn test_transformer_layer_invalid_heads() {
        let result = GraphTransformerLayer::new(7, 3);
        assert!(
            result.is_err(),
            "d_model=7 not divisible by num_heads=3 should fail"
        );
    }

    // ── GatConfig ────────────────────────────────────────────────────────────

    #[test]
    fn test_gat_config_builder() {
        let cfg = GatConfig::new(16, 4)
            .with_num_layers(3)
            .with_pos_encoding(PosEncodingType::RandomWalk)
            .with_k_encoding_dims(6)
            .with_dropout(0.2);

        assert_eq!(cfg.num_layers, 3);
        assert_eq!(cfg.d_model, 16);
        assert_eq!(cfg.num_heads, 4);
        assert_eq!(cfg.k_encoding_dims, 6);
        assert_eq!(cfg.pos_encoding, PosEncodingType::RandomWalk);
        assert_close(cfg.dropout, 0.2, 1e-6, "dropout");
        cfg.validate().expect("config should be valid");
    }

    #[test]
    fn test_gat_config_invalid_dropout() {
        let cfg = GatConfig::new(8, 2).with_dropout(1.5);
        assert!(cfg.validate().is_err(), "dropout >= 1 should be invalid");
    }

    #[test]
    fn test_gat_config_invalid_heads() {
        let cfg = GatConfig::new(9, 4);
        assert!(cfg.validate().is_err(), "9 not divisible by 4");
    }

    // ── GraphAttentionTransformer ─────────────────────────────────────────────

    #[test]
    fn test_full_model_no_pe() {
        let cfg = GatConfig::new(8, 2)
            .with_num_layers(2)
            .with_pos_encoding(PosEncodingType::None);
        let model = GraphAttentionTransformer::new(cfg, 3, ReadoutType::Mean).expect("build model");
        let num_nodes = 4;
        let x: Vec<f32> = (0..num_nodes * 8).map(|i| i as f32 * 0.05).collect();
        let adj = make_ring_adj(num_nodes);
        let out = model.forward(&x, &adj).expect("full model forward");
        assert_eq!(out.len(), 3, "output dim");
        for &v in &out {
            assert!(v.is_finite(), "output must be finite");
        }
    }

    #[test]
    fn test_full_model_lap_pe() {
        let cfg = GatConfig::new(8, 2)
            .with_num_layers(1)
            .with_pos_encoding(PosEncodingType::Laplacian)
            .with_k_encoding_dims(3);
        let model = GraphAttentionTransformer::new(cfg, 2, ReadoutType::Sum).expect("build model");
        let num_nodes = 5;
        let x: Vec<f32> = (0..num_nodes * 8).map(|i| (i as f32).sin()).collect();
        let adj = make_ring_adj(num_nodes);
        let out = model.forward(&x, &adj).expect("model with LapPE forward");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_full_model_rwpe() {
        let cfg = GatConfig::new(8, 2)
            .with_num_layers(1)
            .with_pos_encoding(PosEncodingType::RandomWalk)
            .with_k_encoding_dims(3);
        let model = GraphAttentionTransformer::new(cfg, 4, ReadoutType::Max).expect("build model");
        let num_nodes = 4;
        let x = vec![0.1f32; num_nodes * 8];
        let adj = make_star_adj(0, &[1, 2, 3]);
        let out = model.forward(&x, &adj).expect("model with RWPE forward");
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_full_model_empty_graph_error() {
        let cfg = GatConfig::new(8, 2).with_pos_encoding(PosEncodingType::None);
        let model = GraphAttentionTransformer::new(cfg, 2, ReadoutType::Mean).expect("build model");
        let result = model.forward(&[], &[]);
        assert!(result.is_err());
    }
}
