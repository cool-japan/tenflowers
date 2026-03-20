//! Core types for graph-level GNN operations.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

/// Error type for graph-level GNN operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GnnError {
    /// Input graph has no nodes.
    EmptyGraph,
    /// A dimension was expected to be one value but had another.
    DimensionMismatch { expected: usize, found: usize },
    /// Adjacency matrix is not square.
    AdjacencyNotSquare { rows: usize, cols: usize },
    /// Adjacency size does not match the number of node feature vectors.
    FeatureCountMismatch { adj_size: usize, feat_count: usize },
    /// Cluster count `k` is larger than the graph's node count `n`.
    InvalidNumClusters { k: usize, n: usize },
}

impl std::fmt::Display for GnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GnnError::EmptyGraph => write!(f, "graph has no nodes"),
            GnnError::DimensionMismatch { expected, found } => {
                write!(f, "dimension mismatch: expected {expected}, found {found}")
            }
            GnnError::AdjacencyNotSquare { rows, cols } => {
                write!(f, "adjacency matrix is not square: {rows}×{cols}")
            }
            GnnError::FeatureCountMismatch { adj_size, feat_count } => write!(
                f,
                "adjacency size {adj_size} does not match feature count {feat_count}"
            ),
            GnnError::InvalidNumClusters { k, n } => {
                write!(f, "num_clusters {k} > num_nodes {n}")
            }
        }
    }
}

impl std::error::Error for GnnError {}

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers (pub(super) for use by sibling modules)
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix–vector product: y[i] = Σ_j A[i][j] * x[j].
/// A has shape [m, k], x has length k → output has length m.
pub(super) fn mat_vec(a: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    a.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Linear transform: y[o] = Σ_f W[f][o] * x[f].
/// W has shape [in_dim, out_dim] (row = input index, col = output index).
/// x has length in_dim → output has length out_dim.
pub(super) fn linear_transform(w: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    if w.is_empty() {
        return Vec::new();
    }
    let out_dim = w[0].len();
    let mut y = vec![0.0_f32; out_dim];
    for (f, row) in w.iter().enumerate() {
        let xf = if f < x.len() { x[f] } else { 0.0 };
        for (o, &wfo) in row.iter().enumerate() {
            y[o] += wfo * xf;
        }
    }
    y
}

/// Matrix–matrix product C = A × B.
/// A: [m, k], B: [k, n] → C: [m, n].
pub(super) fn mat_mul(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let m = a.len();
    let k = b.len();
    let n = b[0].len();
    let mut c = vec![vec![0.0_f32; n]; m];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i][p];
            for j in 0..n {
                c[i][j] += aip * b[p][j];
            }
        }
    }
    c
}

/// Transpose a 2-D matrix.
pub(super) fn transpose(a: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if a.is_empty() {
        return Vec::new();
    }
    let rows = a.len();
    let cols = a[0].len();
    let mut t = vec![vec![0.0_f32; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// Softmax over a slice — returns a new Vec.
pub(super) fn softmax(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in exps.iter_mut() {
            *e /= sum;
        }
    }
    exps
}

/// LeakyReLU activation.
#[inline]
pub(super) fn leaky_relu(x: f32, negative_slope: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        negative_slope * x
    }
}

/// ReLU activation.
#[inline]
pub(super) fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Row-normalise a weight matrix (Xavier uniform initialisation).
/// `seed` makes it deterministic.
pub(super) fn xavier_uniform(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let limit = (6.0_f64 / (rows + cols) as f64).sqrt() as f32;
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    let u: f32 = rng.random();
                    u * 2.0 * limit - limit
                })
                .collect()
        })
        .collect()
}

/// Frobenius norm of a 2-D matrix.
pub(super) fn frobenius_norm(m: &[Vec<f32>]) -> f32 {
    m.iter()
        .flat_map(|row| row.iter())
        .map(|&x| x * x)
        .sum::<f32>()
        .sqrt()
}

/// L2 norm of a vector.
pub(super) fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Normalise a vector to unit length; returns the zero vector if the norm is
/// below `eps`.
pub(super) fn normalize_vec(v: &[f32]) -> Vec<f32> {
    let n = l2_norm(v);
    if n < 1e-12 {
        v.to_vec()
    } else {
        v.iter().map(|&x| x / n).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph
// ─────────────────────────────────────────────────────────────────────────────

/// Simple adjacency-matrix representation of an attributed graph.
///
/// Node features are stored as a `Vec<Vec<f32>>` with shape `[num_nodes, feat_dim]`.
/// The adjacency matrix `adj` has shape `[num_nodes, num_nodes]` and may contain
/// weighted edges.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Node feature matrix: `[num_nodes, feat_dim]`.
    pub node_features: Vec<Vec<f32>>,
    /// Adjacency matrix (may be weighted): `[num_nodes, num_nodes]`.
    pub adj: Vec<Vec<f32>>,
    /// Number of nodes.
    pub num_nodes: usize,
    /// Feature dimension per node.
    pub feat_dim: usize,
}

impl Graph {
    /// Construct a new graph, validating all dimension constraints.
    pub fn new(
        node_features: Vec<Vec<f32>>,
        adj: Vec<Vec<f32>>,
    ) -> Result<Self, GnnError> {
        let num_nodes = node_features.len();
        if num_nodes == 0 {
            return Err(GnnError::EmptyGraph);
        }

        // Verify adjacency is square.
        let adj_rows = adj.len();
        if adj_rows == 0 {
            return Err(GnnError::AdjacencyNotSquare { rows: 0, cols: 0 });
        }
        let adj_cols = adj[0].len();
        if adj_rows != adj_cols {
            return Err(GnnError::AdjacencyNotSquare {
                rows: adj_rows,
                cols: adj_cols,
            });
        }
        if adj_rows != num_nodes {
            return Err(GnnError::FeatureCountMismatch {
                adj_size: adj_rows,
                feat_count: num_nodes,
            });
        }

        let feat_dim = node_features[0].len();
        // Verify all feature rows have the same dimension.
        for (i, row) in node_features.iter().enumerate() {
            if row.len() != feat_dim {
                return Err(GnnError::DimensionMismatch {
                    expected: feat_dim,
                    found: row.len(),
                });
            }
        }
        // Verify adjacency row lengths.
        for row in adj.iter() {
            if row.len() != adj_cols {
                return Err(GnnError::AdjacencyNotSquare {
                    rows: adj_rows,
                    cols: row.len(),
                });
            }
        }

        Ok(Self {
            node_features,
            adj,
            num_nodes,
            feat_dim,
        })
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Feature dimension.
    pub fn feat_dim(&self) -> usize {
        self.feat_dim
    }

    /// Degree vector: `d[i] = Σ_j A[i][j]`.
    pub fn degree(&self) -> Vec<f32> {
        self.adj
            .iter()
            .map(|row| row.iter().sum::<f32>())
            .collect()
    }

    /// Symmetric normalised adjacency: `Â = D^{-1/2} A D^{-1/2}`.
    ///
    /// Isolated nodes (zero degree) get a factor of 0 to avoid division by zero.
    pub fn normalized_adj(&self) -> Vec<Vec<f32>> {
        let n = self.num_nodes;
        let deg = self.degree();
        // d_inv_sqrt[i] = 1 / sqrt(deg[i])  or 0 if deg == 0
        let d_inv_sqrt: Vec<f32> = deg
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();

        let mut out = vec![vec![0.0_f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                out[i][j] = d_inv_sqrt[i] * self.adj[i][j] * d_inv_sqrt[j];
            }
        }
        out
    }

    /// Graph Laplacian: `L = D - A`.
    pub fn laplacian(&self) -> Vec<Vec<f32>> {
        let n = self.num_nodes;
        let deg = self.degree();
        let mut lap = vec![vec![0.0_f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    lap[i][j] = deg[i] - self.adj[i][j];
                } else {
                    lap[i][j] = -self.adj[i][j];
                }
            }
        }
        lap
    }

    /// Symmetric normalised Laplacian: `L_sym = I - D^{-1/2} A D^{-1/2}`.
    pub fn sym_norm_laplacian(&self) -> Vec<Vec<f32>> {
        let n = self.num_nodes;
        let a_hat = self.normalized_adj();
        let mut l_sym = vec![vec![0.0_f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                let identity = if i == j { 1.0_f32 } else { 0.0_f32 };
                l_sym[i][j] = identity - a_hat[i][j];
            }
        }
        l_sym
    }
}
