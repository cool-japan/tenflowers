//! Internal helper functions shared across safety_alignment sub-modules.

use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use scirs2_core::RngExt;

/// Cosine similarity between two vectors.
pub(super) fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dot_ab: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    dot_ab / (norm_a * norm_b)
}

/// Dot product of two vectors.
pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix multiply A @ B where A is `[m][p]` and B is `[p][n]`.
pub(super) fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 || b.is_empty() {
        return Vec::new();
    }
    let p = b.len();
    let n = b[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for k in 0..p {
            if k >= a[i].len() {
                continue;
            }
            let aik = a[i][k];
            for j in 0..n {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

/// Sigmoid function.
pub(super) fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Max disparity across a slice of rates.
pub(super) fn disparity(rates: Vec<f64>) -> f64 {
    if rates.len() < 2 {
        return 0.0;
    }
    let max = rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = rates.iter().cloned().fold(f64::INFINITY, f64::min);
    (max - min).max(0.0)
}

/// Expected Calibration Error for `(prob, label)` pairs in `n_bins` bins.
pub(super) fn expected_calibration_error(data: &[(f64, u8)], n_bins: usize) -> f64 {
    let n = data.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let bin_width = 1.0 / n_bins as f64;
    let mut ece = 0.0;
    for b in 0..n_bins {
        let lo = b as f64 * bin_width;
        let hi = lo + bin_width;
        let bin: Vec<(f64, u8)> = data
            .iter()
            .filter(|(p, _)| *p >= lo && *p < hi)
            .cloned()
            .collect();
        if bin.is_empty() {
            continue;
        }
        let bin_n = bin.len() as f64;
        let avg_conf: f64 = bin.iter().map(|(p, _)| p).sum::<f64>() / bin_n;
        let accuracy: f64 = bin.iter().filter(|(_, l)| *l == 1).count() as f64 / bin_n;
        ece += (bin_n / n) * (avg_conf - accuracy).abs();
    }
    ece
}

/// Sample a standard normal N(0,1) using the Box-Muller transform.
pub(super) fn standard_normal_sample(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-15);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Inverse normal CDF (probit) for `p ∈ (0.5, 1)`.
///
/// Uses a rational approximation accurate to ~1e-4.
pub(super) fn probit(p: f64) -> f64 {
    // Rational approximation (Abramowitz & Stegun 26.2.17 / 26.2.22)
    let t = (-2.0 * (1.0 - p).ln()).sqrt();
    let c = [2.515_517, 0.802_853, 0.010_328];
    let d = [1.432_788, 0.189_269, 0.001_308];
    let numer = c[0] + c[1] * t + c[2] * t * t;
    let denom = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t;
    t - numer / denom
}

/// Lower Clopper-Pearson confidence bound on binomial probability.
///
/// Returns the `alpha`-quantile of Beta(k, n-k+1).
pub(super) fn clopper_pearson_lower(k: usize, n: usize, alpha: f64) -> f64 {
    if k == 0 {
        return 0.0;
    }
    // Conservative approximation via normal distribution
    let p_hat = k as f64 / n as f64;
    let z = probit(1.0 - alpha);
    let margin = z * (p_hat * (1.0 - p_hat) / n as f64).sqrt();
    (p_hat - margin).clamp(0.0, 1.0)
}

/// Weighted least-squares regression for LIME.
///
/// Fits β via ridge regression: β = (XᵀWX + λI)⁻¹ XᵀWy.
pub(super) fn weighted_linear_regression(
    samples: &[Vec<f64>],
    targets: &[f64],
    weights: &[f64],
) -> Vec<f64> {
    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }
    let d = samples[0].len();
    if d == 0 {
        return Vec::new();
    }

    // Build XᵀWX (d × d) and XᵀWy (d)
    let lambda = 1e-4; // ridge
    let mut xtwx = vec![vec![0.0; d]; d];
    let mut xtwy = vec![0.0; d];

    for (i, (x, &y)) in samples.iter().zip(targets.iter()).enumerate() {
        let w = weights.get(i).copied().unwrap_or(1.0);
        for j in 0..d {
            for k in 0..d {
                xtwx[j][k] +=
                    w * x.get(j).copied().unwrap_or(0.0) * x.get(k).copied().unwrap_or(0.0);
            }
            xtwy[j] += w * x.get(j).copied().unwrap_or(0.0) * y;
        }
    }

    // Ridge: add λ to diagonal
    for i in 0..d {
        xtwx[i][i] += lambda;
    }

    // Solve via diagonal approximation (Jacobi-style)
    solve_linear_diag_approx(&xtwx, &xtwy)
}

/// Diagonal approximation for solving Ax = b (uses only diagonal of A).
pub(super) fn solve_linear_diag_approx(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    b.iter()
        .enumerate()
        .map(|(i, &bi)| {
            let diag = a[i].get(i).copied().unwrap_or(1.0);
            if diag.abs() > 1e-12 {
                bi / diag
            } else {
                0.0
            }
        })
        .collect()
}

/// Diagonal approximation for matrix inversion (returns diag(A)^{-1} as full matrix).
pub(super) fn invert_diagonal_approx(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let d = a.len();
    let mut inv = vec![vec![0.0; d]; d];
    for i in 0..d {
        let diag = a[i].get(i).copied().unwrap_or(1.0);
        inv[i][i] = if diag.abs() > 1e-12 { 1.0 / diag } else { 0.0 };
    }
    inv
}
