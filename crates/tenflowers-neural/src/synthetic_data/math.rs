//! Shared mathematical primitives for synthetic data generation.

use scirs2_core::random::{rngs::StdRng, Rng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

/// Standard normal log-PDF.
#[inline]
pub(super) fn normal_log_pdf(x: f64) -> f64 {
    -0.5 * x * x - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// Multivariate normal log-PDF with diagonal covariance `var`.
pub(super) fn mvn_log_pdf_diag(x: &[f64], mu: &[f64], var: &[f64]) -> f64 {
    let d = x.len();
    let mut sum = 0.0_f64;
    let log2pi = (2.0 * std::f64::consts::PI).ln();
    for i in 0..d {
        let v = var[i].max(1e-12);
        let diff = x[i] - mu[i];
        sum += -0.5 * (log2pi + v.ln() + diff * diff / v);
    }
    sum
}

/// log-sum-exp for numerical stability.
pub(super) fn log_sum_exp(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return max;
    }
    let sum: f64 = vals.iter().map(|v| (v - max).exp()).sum();
    max + sum.ln()
}

/// Box-Muller transform: one N(0,1) sample from two U(0,1) inputs.
#[inline]
pub(super) fn box_muller(u1: f64, u2: f64) -> f64 {
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Sample a single N(0,1) variate.
pub(super) fn sample_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-300);
    let u2: f64 = rng.random();
    box_muller(u1, u2)
}

/// Cholesky decomposition of an n×n SPD matrix (row-major).
pub(super) fn cholesky(a: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    s = 1e-10;
                }
                l[i * n + j] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Ok(l)
}

/// Multiply lower-triangular L (n×n) by column vector z.
pub(super) fn lower_tri_mv(l: &[f64], z: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..=i {
            out[i] += l[i * n + j] * z[j];
        }
    }
    out
}

/// Empirical CDF at `x` from sorted samples.
pub(super) fn empirical_cdf(sorted_samples: &[f64], x: f64) -> f64 {
    if sorted_samples.is_empty() {
        return 0.5;
    }
    let pos = sorted_samples.partition_point(|&v| v <= x);
    pos as f64 / sorted_samples.len() as f64
}

/// Probit via Abramowitz & Stegun rational approximation.
pub(super) fn probit(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    let c = [2.515517, 0.802853, 0.010328];
    let d = [1.432788, 0.189269, 0.001308];
    let sign = if p < 0.5 { -1.0 } else { 1.0 };
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };
    let num = c[0] + c[1] * t + c[2] * t * t;
    let den = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t;
    sign * (t - num / den)
}

/// Standard normal CDF via error function.
#[inline]
pub(super) fn cdf_normal(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

fn erf_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    sign * (1.0 - poly * (-x * x).exp())
}

/// Empirical quantile from sorted samples.
pub(super) fn empirical_quantile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p * sorted.len() as f64).floor() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Column mean.
pub(super) fn mean_vec(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Biased variance.
pub(super) fn var_vec(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 1.0;
    }
    let m = mean_vec(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
}

/// Euclidean distance.
pub(super) fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Gaussian kernel.
#[inline]
pub(super) fn gauss_kernel(a: &[f64], b: &[f64], sigma: f64) -> f64 {
    let dist2: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    (-dist2 / (2.0 * sigma * sigma)).exp()
}
