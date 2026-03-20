//! Private helper utilities shared across CNF sub-modules.

use scirs2_core::random::{rngs::StdRng, Rng};
use scirs2_core::RngExt;
use std::f64::consts::PI;

/// Sample from a standard normal distribution `N(0, I_d)` using Box-Muller transform.
pub fn sample_standard_normal(d: usize, rng: &mut StdRng) -> Vec<f64> {
    (0..d)
        .map(|_| {
            let u1: f64 = rng.random::<f64>().max(1e-15);
            let u2: f64 = rng.random::<f64>();
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        })
        .collect()
}

/// Log-probability under `N(0, I_d)`.
pub fn standard_normal_log_prob(z: &[f64]) -> f64 {
    let d = z.len();
    let mut lp = -(d as f64) * 0.5 * (2.0 * PI).ln();
    for zi in z {
        lp -= 0.5 * zi * zi;
    }
    lp
}

/// Compute the standard deviation of samples (averaged over dimensions).
pub fn compute_sample_diversity(samples: &[Vec<f64>], z_dim: usize) -> f64 {
    let n = samples.len();
    if n < 2 || z_dim == 0 {
        return 0.0;
    }
    let mut total_var = 0.0_f64;
    for dim in 0..z_dim {
        let values: Vec<f64> = samples.iter().filter_map(|s| s.get(dim).copied()).collect();
        let m = values.len();
        if m < 2 {
            continue;
        }
        let mean = values.iter().sum::<f64>() / m as f64;
        let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (m - 1) as f64;
        total_var += var.sqrt();
    }
    total_var / z_dim as f64
}

/// Overall standard deviation across all dimensions and samples (for KDE bandwidth).
pub fn compute_overall_std(samples: &[Vec<f64>], z_dim: usize) -> f64 {
    let n = samples.len();
    if n == 0 || z_dim == 0 {
        return 1.0;
    }
    let all_vals: Vec<f64> = samples
        .iter()
        .flat_map(|s| s.iter().take(z_dim).copied())
        .collect();
    let m = all_vals.len();
    if m < 2 {
        return 1.0;
    }
    let mean = all_vals.iter().sum::<f64>() / m as f64;
    let var = all_vals
        .iter()
        .map(|v| (v - mean) * (v - mean))
        .sum::<f64>()
        / (m - 1) as f64;
    var.sqrt().max(1e-6)
}

/// Kernel density estimate log-probability of point `x` given samples, bandwidth `bw`.
pub fn kde_log_prob(x: &[f64], samples: &[Vec<f64>], bw: f64, z_dim: usize) -> f64 {
    let n = samples.len();
    if n == 0 {
        return f64::NEG_INFINITY;
    }
    let d = z_dim as f64;
    let log_norm = -(d / 2.0) * (2.0 * PI * bw * bw).ln() - (n as f64).ln();
    let mut log_sum = f64::NEG_INFINITY;

    for s in samples {
        let dist_sq: f64 = x
            .iter()
            .take(z_dim)
            .zip(s.iter().take(z_dim))
            .map(|(xi, si)| (xi - si) * (xi - si))
            .sum();
        let log_k = -0.5 * dist_sq / (bw * bw);
        // Log-sum-exp accumulation
        if log_sum == f64::NEG_INFINITY {
            log_sum = log_k;
        } else {
            let max_v = log_sum.max(log_k);
            log_sum = max_v + ((log_sum - max_v).exp() + (log_k - max_v).exp()).ln();
        }
    }
    log_norm + log_sum
}
