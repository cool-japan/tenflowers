//! Internal helper functions for bayesian_dl.

use scirs2_core::random::{rngs::StdRng, Rng};
use scirs2_core::RngExt;
use std::f64::consts::PI;

/// Box-Muller normal sample from `rng`.
#[inline]
pub(super) fn sample_normal(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(1e-15);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// ReLU activation.
#[inline]
pub(super) fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Softmax over a slice, returns new Vec.
pub(crate) fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_v = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max_v).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum <= 0.0 {
        vec![1.0 / logits.len() as f64; logits.len()]
    } else {
        exps.iter().map(|e| e / sum).collect()
    }
}
