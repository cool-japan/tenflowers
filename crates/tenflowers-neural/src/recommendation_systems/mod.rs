//! Recommendation Systems — comprehensive production-grade implementations.
//!
//! Provides a full suite of collaborative filtering, sequential, and graph-based
//! recommender system components. All computations use `f32`; all weight buffers
//! are plain `Vec<f32>` — no external tensor or autograd dependencies.
//!
//! # Components
//!
//! | Struct | Algorithm |
//! |--------|-----------|
//! | [`MatrixFactorization`] | ALS + SGD matrix factorization |
//! | [`BprLoss`] | Bayesian Personalized Ranking loss |
//! | [`NeuralCF`] | Generalized MF + MLP neural collaborative filtering |
//! | [`SasRec`] | Self-attentive sequential recommendation |
//! | [`BERT4Rec`] | Bidirectional transformer for sequential rec |
//! | [`LightGCN`] | Light graph convolution collaborative filtering |
//! | [`SessionEncoder`] | GRU-based session encoder |
//! | [`RecommendationEvaluator`] | HitRate@K, NDCG@K, MRR, Recall@K |
//! | [`DotProductSimilarity`] | Fast dot-product item similarity |
//! | [`CosineSimilarity`] | Cosine-normalized item similarity |
//! | [`NegativeSampler`] | Uniform and popularity-weighted sampling |
//!
//! # Quick start
//!
//! ```rust,ignore
//! use tenflowers_neural::recommendation_systems::{MatrixFactorization, MfConfig};
//!
//! let cfg = MfConfig { n_users: 100, n_items: 50, emb_dim: 16, ..Default::default() };
//! let mut mf = MatrixFactorization::new(cfg, 42)?;
//! mf.train_step(0, 5, 4.5, 0.01);
//! let pred = mf.predict(0, 5);
//! ```

use std::collections::HashMap;
use scirs2_core::RngExt;

pub mod evaluation;
pub mod graph;
pub mod matrix_factorization;
pub mod neural_cf;
pub mod sequential;
pub mod session;

#[cfg(test)]
mod tests;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by recommendation system operations.
#[derive(Debug, Clone)]
pub enum RecSysError {
    /// Index out of valid user/item range.
    IndexOutOfBounds {
        what: &'static str,
        idx: usize,
        max: usize,
    },
    /// Dimension mismatch between vectors.
    DimensionMismatch { expected: usize, got: usize },
    /// Numerical issue (NaN / Inf).
    NumericalError(String),
    /// Empty input sequence.
    EmptySequence,
}

impl std::fmt::Display for RecSysError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IndexOutOfBounds { what, idx, max } => {
                write!(f, "{what} index {idx} out of bounds (max {max})")
            }
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::NumericalError(msg) => write!(f, "numerical error: {msg}"),
            Self::EmptySequence => write!(f, "empty input sequence"),
        }
    }
}

impl std::error::Error for RecSysError {}

pub(crate) type RecResult<T> = Result<T, RecSysError>;

// ─────────────────────────────────────────────────────────────────────────────
// Low-level math helpers (shared across submodules)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
pub(crate) fn sigmoid_f32(x: f32) -> f32 {
    let x_c = x.clamp(-88.0, 88.0);
    1.0 / (1.0 + (-x_c).exp())
}

#[inline]
pub(crate) fn relu_f32(x: f32) -> f32 {
    x.max(0.0)
}

/// Numerically stable softmax over a mutable slice (in-place).
pub(crate) fn softmax_inplace(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 1e-12 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// Dot product of two equal-length slices.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm.
#[inline]
pub(crate) fn l2_norm_f32(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Xavier (Glorot) uniform initialiser — fills `out` from `[-limit, limit]`.
pub(crate) fn xavier_fill(
    out: &mut [f32],
    fan_in: usize,
    fan_out: usize,
    rng: &mut scirs2_core::random::rngs::StdRng,
) {
    use scirs2_core::random::Rng;
    let limit = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
    for x in out.iter_mut() {
        let u: f32 = rng.random::<f32>();
        *x = u * 2.0 * limit - limit;
    }
}

/// He (Kaiming) normal initialiser for ReLU networks.
pub(crate) fn he_fill(out: &mut [f32], fan_in: usize, rng: &mut scirs2_core::random::rngs::StdRng) {
    use scirs2_core::random::Rng;
    use std::f32::consts::PI;
    let scale = (2.0_f32 / fan_in as f32).sqrt();
    let mut idx = 0;
    while idx + 1 < out.len() {
        let u1 = (rng.random::<f32>() + 1e-10_f32).ln();
        let u1_neg = -u1;
        let u2: f32 = rng.random::<f32>();
        let angle = 2.0_f32 * PI * u2;
        let r = (2.0_f32 * u1_neg).sqrt();
        out[idx] = r * angle.cos() * scale;
        out[idx + 1] = r * angle.sin() * scale;
        idx += 2;
    }
    if idx < out.len() {
        let u1 = (rng.random::<f32>() + 1e-10_f32).ln();
        let u1_neg = -u1;
        let u2: f32 = rng.random::<f32>();
        out[idx] = (2.0_f32 * u1_neg).sqrt() * (2.0_f32 * PI * u2).cos() * scale;
    }
}

/// Matrix–vector multiply: `y = W * x` where W is stored row-major `[out_dim × in_dim]`.
pub(crate) fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0_f32; out_dim];
    for i in 0..out_dim {
        for j in 0..in_dim {
            y[i] += w[i * in_dim + j] * x[j];
        }
    }
    y
}

/// Layer norm: `(x - mean) / sqrt(var + eps) * gamma + beta`.
pub(crate) fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> RecResult<Vec<f32>> {
    if x.len() != gamma.len() || x.len() != beta.len() {
        return Err(RecSysError::DimensionMismatch {
            expected: x.len(),
            got: gamma.len(),
        });
    }
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();
    Ok(x.iter()
        .zip(gamma.iter())
        .zip(beta.iter())
        .map(|((xi, gi), bi)| (xi - mean) * inv_std * gi + bi)
        .collect())
}

/// Gaussian elimination with partial pivoting to solve `A x = b` for `x`.
/// `a_flat` is stored row-major `[n × n]`, `b` has length `n`.
pub(crate) fn solve_linear_system(
    a_flat: &mut [f32],
    b: &mut [f32],
    n: usize,
) -> RecResult<Vec<f32>> {
    for col in 0..n {
        let (pivot_row, _) = (col..n)
            .map(|r| (r, a_flat[r * n + col].abs()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((col, 0.0));
        if a_flat[pivot_row * n + col].abs() < 1e-12 {
            return Err(RecSysError::NumericalError("singular matrix in ALS".into()));
        }
        if pivot_row != col {
            for k in 0..n {
                a_flat.swap(col * n + k, pivot_row * n + k);
            }
            b.swap(col, pivot_row);
        }
        let pivot = a_flat[col * n + col];
        for row in (col + 1)..n {
            let factor = a_flat[row * n + col] / pivot;
            for k in col..n {
                a_flat[row * n + k] -= factor * a_flat[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0_f32; n];
    for row in (0..n).rev() {
        let mut val = b[row];
        for k in (row + 1)..n {
            val -= a_flat[row * n + k] * x[k];
        }
        let diag = a_flat[row * n + row];
        if diag.abs() < 1e-12 {
            return Err(RecSysError::NumericalError(
                "zero diagonal in back-sub".into(),
            ));
        }
        x[row] = val / diag;
    }
    Ok(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Popularity map builder
// ─────────────────────────────────────────────────────────────────────────────

/// Build a popularity map `item_id → interaction_count` from interaction logs.
pub fn build_popularity_map(interactions: &[(usize, usize)]) -> HashMap<usize, usize> {
    let mut map = HashMap::new();
    for &(_, item) in interactions {
        *map.entry(item).or_insert(0) += 1;
    }
    map
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports from submodules
// ─────────────────────────────────────────────────────────────────────────────

pub use evaluation::{
    CosineSimilarity, DotProductSimilarity, NegativeSampler, NegativeSamplingStrategy, RecMetrics,
    RecommendationEvaluator,
};
pub use graph::{LightGCN, LightGcnConfig};
pub use matrix_factorization::{MatrixFactorization, MfConfig};
pub use neural_cf::{BprLoss, NcfConfig, NeuralCF};
pub use sequential::{BERT4Rec, Bert4RecConfig, SasRec, SasRecConfig};
pub use session::{SessionEncoder, SessionEncoderConfig};
