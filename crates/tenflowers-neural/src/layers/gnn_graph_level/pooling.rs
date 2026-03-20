//! Global and hierarchical pooling operations for graph-level GNN tasks.

use super::types::{
    frobenius_norm, mat_mul, softmax, transpose, xavier_uniform, GnnError,
};

// ─────────────────────────────────────────────────────────────────────────────
// Global pooling
// ─────────────────────────────────────────────────────────────────────────────

/// Average all node feature vectors into a single graph-level representation.
///
/// Returns a zero vector when `node_features` is empty.
pub fn global_mean_pool(node_features: &[Vec<f32>]) -> Vec<f32> {
    if node_features.is_empty() {
        return Vec::new();
    }
    let n = node_features.len() as f32;
    let d = node_features[0].len();
    let mut out = vec![0.0_f32; d];
    for feat in node_features.iter() {
        for (o, &f) in out.iter_mut().zip(feat.iter()) {
            *o += f;
        }
    }
    for o in out.iter_mut() {
        *o /= n;
    }
    out
}

/// Element-wise maximum over all node feature vectors.
pub fn global_max_pool(node_features: &[Vec<f32>]) -> Vec<f32> {
    if node_features.is_empty() {
        return Vec::new();
    }
    let d = node_features[0].len();
    let mut out = vec![f32::NEG_INFINITY; d];
    for feat in node_features.iter() {
        for (o, &f) in out.iter_mut().zip(feat.iter()) {
            if f > *o {
                *o = f;
            }
        }
    }
    out
}

/// Sum all node feature vectors.
pub fn global_sum_pool(node_features: &[Vec<f32>]) -> Vec<f32> {
    if node_features.is_empty() {
        return Vec::new();
    }
    let d = node_features[0].len();
    let mut out = vec![0.0_f32; d];
    for feat in node_features.iter() {
        for (o, &f) in out.iter_mut().zip(feat.iter()) {
            *o += f;
        }
    }
    out
}

/// Alias for [`global_sum_pool`].
pub fn global_add_pool(node_features: &[Vec<f32>]) -> Vec<f32> {
    global_sum_pool(node_features)
}

// ─────────────────────────────────────────────────────────────────────────────
// DiffPool
// ─────────────────────────────────────────────────────────────────────────────

/// Differentiable hierarchical pooling (Ying et al., NeurIPS 2018).
///
/// DiffPool learns a soft cluster assignment matrix `S ∈ R^{N×K}` via a GNN
/// and uses it to coarsen the graph:
///
/// ```text
/// X_new = S^T X_emb          (K × feat_dim)
/// A_new = S^T A S             (K × K)
/// ```
///
/// The auxiliary losses encourage compact, informative clusters.
#[derive(Debug, Clone)]
pub struct DiffPool {
    /// Number of output (pooled) nodes.
    pub num_output_nodes: usize,
    /// Feature dimension (both input and after the embedding transform).
    pub feat_dim: usize,
    /// Embedding weight matrix: `[feat_dim, feat_dim]`.
    pub embed_weights: Vec<Vec<f32>>,
    /// Assignment weight matrix: `[feat_dim, num_output_nodes]`.
    pub assign_weights: Vec<Vec<f32>>,
}

impl DiffPool {
    /// Create a new DiffPool with Xavier-initialised weights.
    pub fn new(feat_dim: usize, num_output_nodes: usize) -> Self {
        let embed_weights = xavier_uniform(feat_dim, feat_dim, 42);
        let assign_weights = xavier_uniform(feat_dim, num_output_nodes, 43);
        Self {
            num_output_nodes,
            feat_dim,
            embed_weights,
            assign_weights,
        }
    }

    /// Compute soft cluster assignments `S`: shape `[num_nodes, num_output_nodes]`.
    ///
    /// `S[v] = softmax(assign_weights^T h_v)`
    pub fn assignment(&self, node_features: &[Vec<f32>]) -> Vec<Vec<f32>> {
        node_features
            .iter()
            .map(|h_v| {
                // raw score: [num_output_nodes]
                let score: Vec<f32> = (0..self.num_output_nodes)
                    .map(|k| {
                        h_v.iter()
                            .zip(self.assign_weights.iter())
                            .map(|(&x, w_row)| x * w_row[k])
                            .sum()
                    })
                    .collect();
                softmax(&score)
            })
            .collect()
    }

    /// Perform the DiffPool coarsening step.
    ///
    /// Returns `(X_new, A_new)` where
    /// - `X_new: [num_output_nodes, feat_dim]`
    /// - `A_new: [num_output_nodes, num_output_nodes]`
    pub fn pool(
        &self,
        node_features: &[Vec<f32>],
        adj: &[Vec<f32>],
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), GnnError> {
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

        // Embedding transform: X_emb = X * embed_weights.
        let x_emb: Vec<Vec<f32>> = node_features
            .iter()
            .map(|h| {
                let mut y = vec![0.0_f32; self.feat_dim];
                for (f, row) in self.embed_weights.iter().enumerate() {
                    let xf = if f < h.len() { h[f] } else { 0.0 };
                    for (o, &wfo) in row.iter().enumerate() {
                        y[o] += wfo * xf;
                    }
                }
                y
            })
            .collect();

        // Assignment matrix S: [n, K].
        let s = self.assignment(node_features);

        // S^T: [K, n].
        let s_t = transpose(&s);

        // X_new = S^T * X_emb: [K, feat_dim].
        let x_new = mat_mul(&s_t, &x_emb);

        // A_new = S^T * A * S: [K, n] * [n, n] * [n, K] = [K, K].
        let st_a = mat_mul(&s_t, adj);
        let a_new = mat_mul(&st_a, &s);

        Ok((x_new, a_new))
    }

    /// Link-prediction auxiliary loss: `||A - S S^T||_F^2`.
    ///
    /// Penalises assignments that do not reconstruct the adjacency structure.
    pub fn link_prediction_loss(&self, adj: &[Vec<f32>], assignment: &[Vec<f32>]) -> f32 {
        let s_st = mat_mul(assignment, &transpose(assignment));
        let n = adj.len();
        let mut loss = 0.0_f32;
        for i in 0..n.min(s_st.len()) {
            for j in 0..n.min(s_st[i].len()) {
                let diff = adj[i][j] - s_st[i][j];
                loss += diff * diff;
            }
        }
        loss
    }

    /// Entropy auxiliary loss: `Σ_v Σ_k -S[v,k] * log(S[v,k] + ε)`.
    ///
    /// Encourages peaked (low-entropy) cluster assignments.
    pub fn entropy_loss(&self, assignment: &[Vec<f32>]) -> f32 {
        let eps = 1e-10_f32;
        let mut loss = 0.0_f32;
        for row in assignment.iter() {
            for &p in row.iter() {
                loss -= p * (p + eps).ln();
            }
        }
        loss.max(0.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MinCutPool
// ─────────────────────────────────────────────────────────────────────────────

/// MinCut hierarchical pooling (Bianchi et al., ICML 2020).
///
/// Learns cluster assignments by optimising a min-cut objective in the spectral
/// domain, together with an orthogonality regulariser to avoid degenerate
/// solutions.
#[derive(Debug, Clone)]
pub struct MinCutPool {
    /// Number of output clusters.
    pub num_clusters: usize,
    /// Input feature dimension.
    pub feat_dim: usize,
    /// Assignment weight matrix: `[feat_dim, num_clusters]`.
    pub assign_weights: Vec<Vec<f32>>,
}

impl MinCutPool {
    /// Create a new MinCutPool with Xavier-initialised weights.
    pub fn new(feat_dim: usize, num_clusters: usize) -> Self {
        let assign_weights = xavier_uniform(feat_dim, num_clusters, 77);
        Self {
            num_clusters,
            feat_dim,
            assign_weights,
        }
    }

    /// Compute soft assignment matrix `S`: shape `[num_nodes, num_clusters]`.
    pub fn assignment(&self, node_features: &[Vec<f32>]) -> Vec<Vec<f32>> {
        node_features
            .iter()
            .map(|h| {
                let score: Vec<f32> = (0..self.num_clusters)
                    .map(|k| {
                        h.iter()
                            .zip(self.assign_weights.iter())
                            .map(|(&x, w)| x * w[k])
                            .sum()
                    })
                    .collect();
                softmax(&score)
            })
            .collect()
    }

    /// Perform the MinCutPool coarsening step.
    ///
    /// Returns `(X_new, A_new)` where
    /// - `X_new: [num_clusters, feat_dim]`
    /// - `A_new: [num_clusters, num_clusters]`
    pub fn pool(
        &self,
        node_features: &[Vec<f32>],
        adj: &[Vec<f32>],
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), GnnError> {
        let n = node_features.len();
        if n == 0 {
            return Err(GnnError::EmptyGraph);
        }
        if self.num_clusters > n {
            return Err(GnnError::InvalidNumClusters {
                k: self.num_clusters,
                n,
            });
        }
        if adj.len() != n {
            return Err(GnnError::FeatureCountMismatch {
                adj_size: adj.len(),
                feat_count: n,
            });
        }

        let s = self.assignment(node_features);
        let s_t = transpose(&s);

        // X_new = S^T X.
        let x_new = mat_mul(&s_t, node_features);

        // A_new = S^T A S.
        let st_a = mat_mul(&s_t, adj);
        let a_new = mat_mul(&st_a, &s);

        Ok((x_new, a_new))
    }

    /// MinCut loss: `-tr(S^T A S) / tr(S^T D S)`.
    ///
    /// Returns 0 when the denominator is zero.
    pub fn mincut_loss(
        &self,
        adj: &[Vec<f32>],
        degree: &[f32],
        assignment: &[Vec<f32>],
    ) -> f32 {
        let n = adj.len().min(assignment.len());
        let k = self.num_clusters;

        // Numerator: tr(S^T A S).
        // (S^T A S)[p,q] = Σ_i Σ_j S[i,p] * A[i,j] * S[j,q]
        // tr = Σ_p (S^T A S)[p,p]
        let s_t = transpose(assignment);
        let st_a = mat_mul(&s_t, adj);
        let st_a_s = mat_mul(&st_a, assignment);
        let trace_num: f32 = (0..k.min(st_a_s.len()))
            .map(|p| st_a_s[p][p])
            .sum();

        // Denominator: tr(S^T D S).
        // D is diagonal, so (S^T D S)[p,q] = Σ_i S[i,p] * D[i,i] * S[i,q]
        // tr = Σ_p Σ_i S[i,p]^2 * D[i,i]
        let mut trace_den = 0.0_f32;
        for i in 0..n.min(degree.len()) {
            for p in 0..k {
                let sp = assignment[i][p];
                trace_den += sp * sp * degree[i];
            }
        }

        if trace_den.abs() < 1e-12 {
            return 0.0;
        }
        -(trace_num / trace_den)
    }

    /// Orthogonality loss: `||S^T S / ||S^T S||_F  -  I/√K||_F`.
    pub fn orthogonality_loss(&self, assignment: &[Vec<f32>]) -> f32 {
        let k = self.num_clusters;
        let s_t = transpose(assignment);
        let sts = mat_mul(&s_t, assignment); // [K, K]
        let norm = frobenius_norm(&sts);
        if norm < 1e-12 {
            return 0.0;
        }

        // Normalised: sts / norm  vs  I / sqrt(K)
        let inv_sqrt_k = 1.0 / (k as f32).sqrt();
        let mut loss = 0.0_f32;
        for p in 0..k {
            for q in 0..k {
                let target = if p == q { inv_sqrt_k } else { 0.0_f32 };
                let diff = sts[p][q] / norm - target;
                loss += diff * diff;
            }
        }
        loss.sqrt()
    }
}
