//! Shared types for the advanced GNN layers.

use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers (used by sage.rs, gat.rs, message_passing.rs)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a single pseudo-random f32 in (-1, 1).
///
/// We deliberately avoid the `rand` crate (project policy: use scirs2-core).
pub(super) fn prng_f32() -> f32 {
    scirs2_core::random::quick::random_f32() * 2.0 - 1.0
}

/// Xavier/Glorot uniform initialisation for a weight matrix stored as a flat
/// row-major Vec<f32> of shape `[rows, cols]`.
pub(super) fn xavier_init(rows: usize, cols: usize) -> Vec<f32> {
    let limit = (6.0_f32 / (rows + cols) as f32).sqrt();
    (0..rows * cols)
        .map(|_| prng_f32() * limit)
        .collect()
}

/// Leaky ReLU with negative slope 0.2 (standard in GAT).
pub(super) fn leaky_relu(x: f32) -> f32 {
    if x >= 0.0 { x } else { 0.2 * x }
}

/// L2 norm of a vector.
pub(super) fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// In-place softmax.
pub(super) fn softmax_inplace(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// Row-major matrix-vector product: `W (out_dim × in_dim) · x (in_dim)`.
pub(super) fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);
    (0..out_dim)
        .map(|i| {
            let row = &w[i * in_dim..(i + 1) * in_dim];
            row.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum()
        })
        .collect()
}

/// Add bias vector to a mutable slice in place.
pub(super) fn add_bias_inplace(v: &mut [f32], b: &[f32]) {
    debug_assert_eq!(v.len(), b.len());
    for (vi, bi) in v.iter_mut().zip(b.iter()) {
        *vi += bi;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AggregationMethod
// ─────────────────────────────────────────────────────────────────────────────

/// Neighbourhood aggregation strategy for [`super::sage::GraphSageLayer`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggregationMethod {
    /// Average of neighbour features.
    Mean,
    /// Element-wise maximum over neighbour features.
    Max,
    /// Sum of neighbour features.
    Sum,
}
