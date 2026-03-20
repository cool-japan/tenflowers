//! Internal mathematical helpers for concept learning.

use scirs2_core::random::{rngs::StdRng, Rng};
use scirs2_core::RngExt;

#[inline]
pub(super) fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub(super) fn relu_f64(x: f64) -> f64 {
    x.max(0.0)
}

/// Stable softmax over a slice.
pub(super) fn softmax_f64(v: &[f64]) -> Vec<f64> {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum::<f64>().max(1e-300);
    exps.iter().map(|e| e / sum).collect()
}

/// Dot product.
#[inline]
pub(super) fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm.
#[inline]
pub(super) fn l2_norm_f64(v: &[f64]) -> f64 {
    dot_f64(v, v).sqrt()
}

/// Xavier uniform initialisation for a matrix [rows][cols].
pub(super) fn xavier_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let scale = (6.0_f64 / (rows + cols).max(1) as f64).sqrt();
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.random_range(-scale..scale)).collect())
        .collect()
}
