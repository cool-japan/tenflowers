//! Internal utility functions shared across graph_signal submodules.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn matvec(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    mat.iter().map(|row| dot(row, v)).collect()
}

pub fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b.first().map(|r| r.len()).unwrap_or(0);
    let k = b.len();
    (0..n)
        .map(|i| {
            (0..m)
                .map(|j| {
                    (0..k)
                        .map(|l| a[i].get(l).copied().unwrap_or(0.0) * b[l][j])
                        .sum()
                })
                .collect()
        })
        .collect()
}

pub fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b)
        .map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| x + y).collect())
        .collect()
}

pub fn mat_scale(a: &[Vec<f64>], s: f64) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row| row.iter().map(|x| x * s).collect())
        .collect()
}

pub fn identity(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

pub fn normalize_vec(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-300 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

pub fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub fn rand_weight(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let scale = (2.0 / cols as f64).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                .collect()
        })
        .collect()
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute degree vector (sum of each row), clamped to avoid division by zero.
pub fn degree_vec(adj: &[Vec<f64>]) -> Vec<f64> {
    adj.iter()
        .map(|row| row.iter().sum::<f64>().max(1e-12))
        .collect()
}
