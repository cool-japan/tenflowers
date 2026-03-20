//! Extensions: GAT layer, node classifier, and evaluation metrics.
//!
//! - [`GnoGatLayer`]: Multi-head Graph Attention Network layer (Veličković 2018).
//! - [`GnoNodeClassifier`]: Node-classification wrapper around GRAND or GraphODE.
//! - [`GraphOdeMetrics`]: MAE, R², node-classification accuracy, RTE.

use super::{
    matmul, random_matrix, xavier_std, box_muller,
    GnoError, GnoGraph, GrandConfig, GrandModel, GraphOdeConfig, GraphOdeModel,
};
use scirs2_core::random::{rngs::StdRng, SeedableRng, Rng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// §11  GnoGatLayer  (Multi-head Graph Attention Network)
// ─────────────────────────────────────────────────────────────────────────────

/// One multi-head GAT layer.
///
/// α_ij = softmax_j(LeakyReLU(a^T · [Wh_i || Wh_j]))
/// h'_i = || ELU(Σ_j α_ij · Wh_j)  (heads concatenated)
#[derive(Debug, Clone)]
pub struct GnoGatLayer {
    /// Input feature dimension.
    pub in_dim: usize,
    /// Per-head output dimension.
    pub out_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Per-head weight matrices (in_dim × out_dim).
    pub head_weights: Vec<Vec<Vec<f64>>>,
    /// Per-head attention vectors (2*out_dim → 1).
    pub attn_vecs: Vec<Vec<f64>>,
}

impl GnoGatLayer {
    /// Create a new multi-head GAT layer.
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        num_heads: usize,
        seed: u64,
    ) -> Result<Self, GnoError> {
        if in_dim == 0 || out_dim == 0 || num_heads == 0 {
            return Err(GnoError::InvalidConfig(
                "in_dim, out_dim, and num_heads must all be > 0".into(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut head_weights = Vec::with_capacity(num_heads);
        let mut attn_vecs = Vec::with_capacity(num_heads);
        for _ in 0..num_heads {
            let std = xavier_std(in_dim, out_dim);
            head_weights.push(random_matrix(in_dim, out_dim, std, &mut rng));
            // attention vector: 2*out_dim
            let a_std = xavier_std(2 * out_dim, 1);
            let av: Vec<f64> = (0..2 * out_dim)
                .map(|_| {
                    let u1 = rng.random::<f64>();
                    let u2 = rng.random::<f64>();
                    box_muller(u1, u2) * a_std
                })
                .collect();
            attn_vecs.push(av);
        }
        Ok(Self {
            in_dim,
            out_dim,
            num_heads,
            head_weights,
            attn_vecs,
        })
    }

    #[inline]
    fn leaky_relu(x: f64, alpha: f64) -> f64 {
        if x >= 0.0 {
            x
        } else {
            alpha * x
        }
    }

    #[inline]
    fn elu(x: f64) -> f64 {
        if x >= 0.0 {
            x
        } else {
            x.exp() - 1.0
        }
    }

    /// Forward pass: returns (n × num_heads*out_dim) feature matrix.
    pub fn forward(
        &self,
        node_feats: &[Vec<f64>],
        adj: &GnoGraph,
    ) -> Result<Vec<Vec<f64>>, GnoError> {
        let n = node_feats.len();
        if n == 0 {
            return Err(GnoError::EmptyGraph);
        }
        let alpha_neg = 0.2_f64;
        // Per-head output
        let mut all_head_out = vec![vec![0.0f64; self.num_heads * self.out_dim]; n];
        for (head_idx, (w, a)) in self
            .head_weights
            .iter()
            .zip(self.attn_vecs.iter())
            .enumerate()
        {
            // Wh: n × out_dim
            let wh = matmul(node_feats, w)?;
            // Compute attention coefficients for each edge
            // For each node i: α_ij for j ∈ N(i) ∪ {i}
            let mut head_out = vec![vec![0.0f64; self.out_dim]; n];
            for i in 0..n {
                // Neighbours + self
                let mut neighbors: Vec<usize> = adj.adj()[i].iter().map(|(v, _)| *v).collect();
                neighbors.push(i);
                // Compute raw attention scores
                let mut raw: Vec<(usize, f64)> = Vec::with_capacity(neighbors.len());
                for &j in &neighbors {
                    let mut concat = wh[i].clone();
                    concat.extend_from_slice(&wh[j]);
                    let score: f64 = concat.iter().zip(a.iter()).map(|(ci, ai)| ci * ai).sum();
                    raw.push((j, Self::leaky_relu(score, alpha_neg)));
                }
                // Softmax over raw scores
                let max_s = raw
                    .iter()
                    .map(|(_, s)| *s)
                    .fold(f64::NEG_INFINITY, f64::max);
                let sum_exp: f64 = raw.iter().map(|(_, s)| (s - max_s).exp()).sum();
                let sum_exp = sum_exp.max(1e-12);
                // Aggregate
                for (j, score) in &raw {
                    let alpha = (score - max_s).exp() / sum_exp;
                    for k in 0..self.out_dim {
                        head_out[i][k] += alpha * wh[*j][k];
                    }
                }
                // ELU activation
                for v in head_out[i].iter_mut() {
                    *v = Self::elu(*v);
                }
                // Copy into all_head_out
                let offset = head_idx * self.out_dim;
                for k in 0..self.out_dim {
                    all_head_out[i][offset + k] = head_out[i][k];
                }
            }
        }
        Ok(all_head_out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §12  GnoNodeClassifier
// ─────────────────────────────────────────────────────────────────────────────

/// Which backbone to use in the node classifier.
#[derive(Debug, Clone)]
pub enum GnoClassifierBackbone {
    /// Use GRAND as the backbone.
    Grand,
    /// Use GraphODE as the backbone.
    GraphOde,
}

/// Node classification wrapper around GRAND or GraphODE + softmax head.
#[derive(Debug, Clone)]
pub struct GnoNodeClassifier {
    /// The backbone model type.
    pub backbone: GnoClassifierBackbone,
    grand: Option<GrandModel>,
    graph_ode: Option<GraphOdeModel>,
}

impl GnoNodeClassifier {
    /// Create a GRAND-backed node classifier.
    pub fn from_grand(cfg: GrandConfig) -> Result<Self, GnoError> {
        let grand = GrandModel::new(cfg)?;
        Ok(Self {
            backbone: GnoClassifierBackbone::Grand,
            grand: Some(grand),
            graph_ode: None,
        })
    }

    /// Create a GraphODE-backed node classifier.
    pub fn from_graph_ode(cfg: GraphOdeConfig, graph: &GnoGraph) -> Result<Self, GnoError> {
        let gode = GraphOdeModel::new(cfg, graph)?;
        Ok(Self {
            backbone: GnoClassifierBackbone::GraphOde,
            grand: None,
            graph_ode: Some(gode),
        })
    }

    /// Predict class for each node.
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<usize>, GnoError> {
        match &self.backbone {
            GnoClassifierBackbone::Grand => self
                .grand
                .as_ref()
                .ok_or_else(|| GnoError::InvalidConfig("grand model not set".into()))?
                .predict(x),
            GnoClassifierBackbone::GraphOde => self
                .graph_ode
                .as_ref()
                .ok_or_else(|| GnoError::InvalidConfig("graph_ode model not set".into()))?
                .predict(x),
        }
    }

    /// Compute node classification accuracy against ground-truth labels.
    pub fn accuracy(&self, x: &[Vec<f64>], labels: &[usize]) -> Result<f64, GnoError> {
        let preds = self.predict(x)?;
        if preds.len() != labels.len() {
            return Err(GnoError::DimensionMismatch {
                expected: labels.len(),
                found: preds.len(),
                context: "accuracy labels",
            });
        }
        let correct = preds
            .iter()
            .zip(labels.iter())
            .filter(|(p, l)| p == l)
            .count();
        Ok(correct as f64 / labels.len() as f64)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §13  GraphOdeMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluation metrics for graph ODE models.
#[derive(Debug, Clone)]
pub struct GraphOdeMetrics;

impl GraphOdeMetrics {
    /// Mean Absolute Error between two trajectory snapshots at a specific time step.
    pub fn mae_at_t(pred: &[Vec<f64>], target: &[Vec<f64>]) -> Result<f64, GnoError> {
        let n = pred.len();
        if n != target.len() {
            return Err(GnoError::DimensionMismatch {
                expected: target.len(),
                found: n,
                context: "mae_at_t rows",
            });
        }
        if n == 0 {
            return Ok(0.0);
        }
        let d = pred[0].len();
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for i in 0..n {
            for j in 0..d {
                sum += (pred[i][j] - target[i][j]).abs();
                count += 1;
            }
        }
        Ok(if count > 0 { sum / count as f64 } else { 0.0 })
    }

    /// R² (coefficient of determination) for flattened predictions vs targets.
    pub fn r2_score(pred: &[Vec<f64>], target: &[Vec<f64>]) -> Result<f64, GnoError> {
        let n = pred.len();
        if n != target.len() {
            return Err(GnoError::DimensionMismatch {
                expected: target.len(),
                found: n,
                context: "r2_score rows",
            });
        }
        if n == 0 {
            return Ok(0.0);
        }
        let d = pred[0].len();
        let total = (n * d) as f64;
        let mean: f64 = target.iter().flat_map(|r| r.iter()).sum::<f64>() / total;
        let ss_tot: f64 = target
            .iter()
            .flat_map(|r| r.iter())
            .map(|v| (v - mean).powi(2))
            .sum();
        let ss_res: f64 = pred
            .iter()
            .zip(target.iter())
            .flat_map(|(pr, tr)| pr.iter().zip(tr.iter()).map(|(p, t)| (p - t).powi(2)))
            .sum();
        if ss_tot < 1e-15 {
            return Ok(if ss_res < 1e-15 { 1.0 } else { 0.0 });
        }
        Ok(1.0 - ss_res / ss_tot)
    }

    /// Node classification accuracy.
    pub fn node_classification_accuracy(
        pred_labels: &[usize],
        true_labels: &[usize],
    ) -> Result<f64, GnoError> {
        if pred_labels.len() != true_labels.len() {
            return Err(GnoError::DimensionMismatch {
                expected: true_labels.len(),
                found: pred_labels.len(),
                context: "node_classification_accuracy",
            });
        }
        if true_labels.is_empty() {
            return Ok(0.0);
        }
        let correct = pred_labels
            .iter()
            .zip(true_labels.iter())
            .filter(|(p, t)| p == t)
            .count();
        Ok(correct as f64 / true_labels.len() as f64)
    }

    /// Relative Trajectory Error (RTE) between two trajectories.
    ///
    /// RTE = (1/T) Σ_t  ||H_pred(t) - H_true(t)||_F / ||H_true(t)||_F
    pub fn relative_trajectory_error(
        pred_traj: &[Vec<Vec<f64>>],
        true_traj: &[Vec<Vec<f64>>],
    ) -> Result<f64, GnoError> {
        let t_len = pred_traj.len();
        if t_len != true_traj.len() {
            return Err(GnoError::DimensionMismatch {
                expected: true_traj.len(),
                found: t_len,
                context: "rte trajectory length",
            });
        }
        if t_len == 0 {
            return Ok(0.0);
        }
        let mut rte_sum = 0.0f64;
        let mut valid = 0usize;
        for (p_snap, t_snap) in pred_traj.iter().zip(true_traj.iter()) {
            let n = p_snap.len();
            if n != t_snap.len() {
                return Err(GnoError::DimensionMismatch {
                    expected: t_snap.len(),
                    found: n,
                    context: "rte snapshot rows",
                });
            }
            let d = if n > 0 { p_snap[0].len() } else { 0 };
            let mut num = 0.0f64;
            let mut denom = 0.0f64;
            for i in 0..n {
                for j in 0..d {
                    num += (p_snap[i][j] - t_snap[i][j]).powi(2);
                    denom += t_snap[i][j].powi(2);
                }
            }
            if denom > 1e-15 {
                rte_sum += (num / denom).sqrt();
                valid += 1;
            }
        }
        Ok(if valid > 0 {
            rte_sum / valid as f64
        } else {
            0.0
        })
    }
}
