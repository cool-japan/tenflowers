//! Internal helper functions for mechanistic interpretability computations.
//! All functions here are `pub(super)` — visible only within the MI module.

use scirs2_core::random::{rngs::StdRng, Rng};
use scirs2_core::RngExt;

pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute `mat @ v` where `mat` is `[rows][cols]` and `v` is `[cols]`.
pub(super) fn matvec(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    mat.iter().map(|row| dot(row, v)).collect()
}

/// Compute `v @ mat` where `mat` is `[rows][cols]` and `v` is `[rows]`.
/// Useful for the unembed projection: `residual @ unembed` → `[vocab_size]`.
pub(super) fn vecmat(v: &[f64], mat: &[Vec<f64>]) -> Vec<f64> {
    if mat.is_empty() || v.is_empty() {
        return Vec::new();
    }
    let cols = mat[0].len();
    let mut out = vec![0.0_f64; cols];
    for (i, &vi) in v.iter().enumerate() {
        if i < mat.len() {
            for (j, &m) in mat[i].iter().enumerate() {
                out[j] += vi * m;
            }
        }
    }
    out
}

pub(super) fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max_l).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / logits.len() as f64; logits.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

pub(super) fn layer_norm(x: &[f64], scale: &[f64], bias: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = (var + 1e-5).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, &v)| (v - mean) / std * scale[i] + bias[i])
        .collect()
}

pub(super) fn relu(x: f64) -> f64 {
    x.max(0.0)
}

pub(super) fn entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 1e-30)
        .map(|&p| -p * p.ln())
        .sum()
}

pub(super) fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

pub(super) fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

pub(super) fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let k = b.len();
    let n = if b.is_empty() { 0 } else { b[0].len() };
    let mut out = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                out[i][j] += a[i][l] * b[l][j];
            }
        }
    }
    out
}

pub(super) fn random_matrix(
    rows: usize,
    cols: usize,
    rng: &mut StdRng,
    scale: f64,
) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| (rng.random::<f64>() - 0.5) * 2.0 * scale)
                .collect()
        })
        .collect()
}

pub(super) fn ones_vec(size: usize) -> Vec<f64> {
    vec![1.0; size]
}

pub(super) fn zeros_vec(size: usize) -> Vec<f64> {
    vec![0.0; size]
}
