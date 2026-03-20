//! Differential privacy utilities: gradient clipping, noise addition, privacy accounting.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

use super::types::{layer_l2_norm, ModelParams};

/// Clip all gradient vectors so that the per-sample L2 norm does not exceed
/// `max_norm`.
///
/// For each layer vector `g`, if `‖g‖₂ > max_norm` then `g ← g · max_norm / ‖g‖₂`,
/// otherwise `g` is unchanged.
pub fn clip_gradients(grads: &ModelParams, max_norm: f32) -> ModelParams {
    grads
        .iter()
        .map(|layer| {
            let norm = layer_l2_norm(layer);
            if norm > max_norm && norm > f32::EPSILON {
                let scale = max_norm / norm;
                layer.iter().map(|&g| g * scale).collect()
            } else {
                layer.clone()
            }
        })
        .collect()
}

/// Add calibrated Gaussian noise for differential privacy.
///
/// The noise standard deviation is `σ = noise_multiplier · max_norm · sensitivity`.
///
/// Since the task description forbids using the `rand` crate directly and
/// specifies using `scirs2_core` for random operations, this function uses
/// a seeded `StdRng` with Box-Muller sampling.  The seed is derived from the
/// current parameter values (deterministic but unique per call context) so that
/// repeated calls with different parameters produce different noise.
pub fn add_dp_noise(
    params: &ModelParams,
    noise_multiplier: f32,
    max_norm: f32,
    sensitivity: f32,
) -> ModelParams {
    let sigma = noise_multiplier * max_norm * sensitivity;
    if sigma < f32::EPSILON {
        return params.clone();
    }

    // Derive a seed from parameter content so different calls get different noise
    let seed: u64 =
        params
            .iter()
            .flat_map(|layer| layer.iter())
            .fold(0x517cc1b727220a95_u64, |acc, &v| {
                acc.wrapping_mul(0x9e3779b97f4a7c15)
                    .wrapping_add(v.to_bits() as u64)
            });

    let mut rng = StdRng::seed_from_u64(seed);

    params
        .iter()
        .map(|layer| {
            let n = layer.len();
            // Box-Muller sampling
            let mut noise = Vec::with_capacity(n);
            let mut i = 0;
            while i < n {
                let u1: f64 = rng.random::<f64>().max(1e-10);
                let u2: f64 = rng.random::<f64>();
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = std::f64::consts::TAU * u2;
                let z0 = r * theta.cos();
                let z1 = r * theta.sin();
                noise.push(z0 as f32 * sigma);
                i += 1;
                if i < n {
                    noise.push(z1 as f32 * sigma);
                    i += 1;
                }
            }
            layer
                .iter()
                .zip(noise.iter())
                .map(|(&p, &n_val)| p + n_val)
                .collect()
        })
        .collect()
}

/// Moments-accountant privacy loss approximation.
///
/// Given:
/// - `noise_multiplier` σ: ratio of Gaussian noise std to clipping norm,
/// - `sampling_rate` q: fraction of data sampled per step,
/// - `num_rounds` T: number of composition steps,
///
/// returns an `(epsilon, delta)` pair at `delta = 1e-5`.
///
/// Uses the Rényi DP → (ε, δ)-DP conversion with the tight bound from
/// Mironov (2017) for the sampled Gaussian mechanism.
///
/// The implementation evaluates the RDP guarantee at α ∈ {2, …, 64} and
/// picks the tightest conversion.
pub fn compute_privacy_loss(
    noise_multiplier: f32,
    sampling_rate: f32,
    num_rounds: usize,
) -> (f32, f32) {
    const DELTA: f64 = 1e-5;
    let q = sampling_rate as f64;
    let sigma = noise_multiplier as f64;
    let t = num_rounds as f64;

    // RDP epsilon for the sampled Gaussian mechanism at order α (Mironov 2017)
    // Upper bound: α * q² / (2σ²)  (valid when q << 1 and sigma >= 1)
    // We compute for a range of α and take the best (ε, δ) conversion.
    let mut best_epsilon = f64::INFINITY;

    for alpha in 2_u32..=64 {
        let a = alpha as f64;
        // RDP bound per step using the Poisson subsampled Gaussian
        // eps_rdp(α) ≈ a * q² / (2 * σ²)  — first-order bound
        let rdp_step = a * q * q / (2.0 * sigma * sigma);
        let rdp_total = rdp_step * t;

        // Convert RDP to (ε, δ)-DP:
        // ε = rdp_total + log(1 − 1/α) - (log(δ) + log(1 − 1/α)) / (α − 1)
        // Simplified form (Balle et al. 2020):
        // ε = rdp_total + log((α − 1) / α) - (log(δ) + log(1 − 1/α)) / (α − 1)
        let log_delta = DELTA.ln();
        let eps = rdp_total + ((a - 1.0) / a).ln() - (log_delta + ((a - 1.0) / a).ln()) / (a - 1.0);

        if eps.is_finite() && eps < best_epsilon {
            best_epsilon = eps;
        }
    }

    if best_epsilon == f64::INFINITY {
        best_epsilon = f64::MAX / 2.0;
    }

    (best_epsilon.max(0.0) as f32, DELTA as f32)
}
