//! Internal math helpers shared across the diffusion_advanced submodules.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::TensorError;

// ─────────────────────────────────────────────────────────────────────────────
// Internal error helper
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
pub(crate) fn make_err(msg: impl Into<String>) -> TensorError {
    TensorError::invalid_argument(msg.into())
}

// ─────────────────────────────────────────────────────────────────────────────
// Math primitives
// ─────────────────────────────────────────────────────────────────────────────

/// Box-Muller: generate `n` samples from N(0, 1) with given seed.
pub(crate) fn normal_samples_f64(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    let mut i = 0_usize;
    while i < n {
        let u1: f64 = (rng.random::<f64>()).max(1e-15);
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        out.push(r * theta.cos());
        i += 1;
        if i < n {
            out.push(r * theta.sin());
            i += 1;
        }
    }
    out
}

/// Dot product of two equal-length slices.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Element-wise `a + alpha * b`.
pub(crate) fn axpy(a: &[f64], alpha: f64, b: &[f64]) -> Vec<f64> {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| ai + alpha * bi)
        .collect()
}

/// Element-wise subtraction.
pub(crate) fn sub_vecs(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai - bi).collect()
}

/// Softmax over a slice.
pub(crate) fn softmax(v: &[f64]) -> Vec<f64> {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum.max(1e-15)).collect()
}

/// L2 squared distance.
pub(crate) fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi).powi(2))
        .sum()
}

/// Simple linear forward pass (no activation).
/// `weight` stored row-major: `weight[i * out_dim + j]`.
pub(crate) fn linear_fwd(input: &[f64], weight: &[f64], bias: &[f64], out_dim: usize) -> Vec<f64> {
    let in_dim = input.len();
    (0..out_dim)
        .map(|j| {
            let s: f64 = (0..in_dim)
                .map(|i| input[i] * weight[i * out_dim + j])
                .sum();
            s + bias[j]
        })
        .collect()
}

/// ReLU activation.
#[inline]
pub(crate) fn relu(x: f64) -> f64 {
    x.max(0.0)
}
