//! Shared types and math utilities for Temporal & Dynamic Graph Neural Networks.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ─────────────────────────────────────────────────────────────────────────────
// Math utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Sigmoid activation.
#[inline]
pub(super) fn sigmoid(x: f64) -> f64 {
    let xc = x.clamp(-500.0, 500.0);
    1.0 / (1.0 + (-xc).exp())
}

/// ReLU activation.
#[inline]
pub(super) fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Tanh activation (wrapped for clarity).
#[inline]
pub(super) fn tanh_act(x: f64) -> f64 {
    x.tanh()
}

/// Numerically stable softmax.
pub(super) fn softmax(v: &[f64]) -> Vec<f64> {
    if v.is_empty() {
        return Vec::new();
    }
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let denom = if sum.abs() < 1e-300 { 1e-300 } else { sum };
    exps.iter().map(|e| e / denom).collect()
}

/// Matrix-vector product: mat [rows × cols] · v [cols] → [rows].
pub(super) fn matvec(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    mat.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum())
        .collect()
}

/// Element-wise addition.
pub(super) fn vecadd(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Element-wise multiplication.
pub(super) fn vecmul(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}

/// Random weight matrix [rows × cols] in [-scale, scale].
pub(super) fn rand_mat(rows: usize, cols: usize, scale: f64, rng: &mut StdRng) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                .collect()
        })
        .collect()
}

/// Random bias vector [dim] initialised to zero.
pub(super) fn zero_vec(dim: usize) -> Vec<f64> {
    vec![0.0; dim]
}

/// Linear transform: W·x + b.
pub(super) fn linear(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    vecadd(&matvec(w, x), b)
}

// ─────────────────────────────────────────────────────────────────────────────
// Core shared types
// ─────────────────────────────────────────────────────────────────────────────

/// A directed temporal edge with optional feature vector.
#[derive(Debug, Clone)]
pub struct TemporalEdge {
    /// Source node index.
    pub src: usize,
    /// Destination node index.
    pub dst: usize,
    /// Event timestamp (seconds / arbitrary units).
    pub time: f64,
    /// Optional edge feature vector.
    pub features: Vec<f64>,
}

impl TemporalEdge {
    /// Create a new temporal edge.
    pub fn new(src: usize, dst: usize, time: f64, features: Vec<f64>) -> Self {
        Self {
            src,
            dst,
            time,
            features,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Heterogeneous graph node/edge type wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// A typed node category.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HeteroNodeType(pub String);

/// A typed edge relation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HeteroEdgeType(pub String);

// ─────────────────────────────────────────────────────────────────────────────
// MessageFunction
// ─────────────────────────────────────────────────────────────────────────────

/// How raw messages are computed from node memories and edge features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageFunction {
    /// Concatenate memories + edge features → message (no projection).
    Identity,
    /// Single-layer MLP projection over the concatenation.
    Mlp,
}

// ─────────────────────────────────────────────────────────────────────────────
// TgnConfig
// ─────────────────────────────────────────────────────────────────────────────

/// TGN configuration.
#[derive(Debug, Clone)]
pub struct TgnConfig {
    pub n_nodes: usize,
    pub node_feat_dim: usize,
    pub edge_feat_dim: usize,
    pub memory_dim: usize,
    pub time_enc_dim: usize,
    pub emb_dim: usize,
    pub n_heads: usize,
    pub message_fn: MessageFunction,
}

impl TgnConfig {
    /// Create a default TGN config.
    pub fn new(
        n_nodes: usize,
        node_feat_dim: usize,
        edge_feat_dim: usize,
        memory_dim: usize,
    ) -> Self {
        Self {
            n_nodes,
            node_feat_dim,
            edge_feat_dim,
            memory_dim,
            time_enc_dim: 32,
            emb_dim: memory_dim,
            n_heads: 4,
            message_fn: MessageFunction::Identity,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PartitionStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// Spatial configuration strategy for ST-GCN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionStrategy {
    /// Single adjacency matrix — no partitioning.
    Uniform,
    /// Partition by graph distance (root, close, further).
    Distance,
    /// Spatial configuration partitioning (Yan 2018 — 3 subsets).
    SpatialConf,
}

impl PartitionStrategy {
    /// Number of subsets in this strategy.
    pub fn n_subsets(&self) -> usize {
        match self {
            PartitionStrategy::Uniform => 1,
            PartitionStrategy::Distance => 2,
            PartitionStrategy::SpatialConf => 3,
        }
    }
}

/// ODE solver method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdeSolver {
    Euler,
    Rk4,
}
