//! Evaluation metrics for continuous normalizing flow models.

use super::flow_matching::FlowMatchingModel;
use super::mlp::ContinuousNormalizingFlow;
use super::utils::{compute_overall_std, compute_sample_diversity, kde_log_prob};
use scirs2_core::random::{rngs::StdRng, SeedableRng};

/// Evaluation metrics for continuous normalizing flow models.
#[derive(Debug, Clone)]
pub struct CnfMetrics {
    /// Mean log-probability over the test set.
    pub mean_log_prob: f64,
    /// Bits-per-dimension: `−mean_log_prob / (z_dim * ln 2)`.
    pub bits_per_dim: f64,
    /// Standard deviation of sampled points (averaged over dimensions).
    pub sample_diversity: f64,
}

/// Evaluate a `ContinuousNormalizingFlow` on test data.
pub fn evaluate_cnf(
    model: &ContinuousNormalizingFlow,
    test_data: &[Vec<f64>],
    n_steps: usize,
) -> CnfMetrics {
    let n = test_data.len();
    let z_dim = model.dynamics.z_dim;

    if n == 0 || z_dim == 0 {
        return CnfMetrics {
            mean_log_prob: 0.0,
            bits_per_dim: 0.0,
            sample_diversity: 0.0,
        };
    }

    let log_probs: Vec<f64> = test_data
        .iter()
        .map(|x| model.log_prob(x, n_steps))
        .collect();
    let mean_lp = log_probs.iter().sum::<f64>() / n as f64;
    let bpd = -mean_lp / (z_dim as f64 * 2_f64.ln());

    // Sample diversity: collect samples and compute std
    let mut rng = StdRng::seed_from_u64(0xbeefdead_u64);
    let n_samples = n.min(100);
    let samples: Vec<Vec<f64>> = (0..n_samples)
        .map(|_| model.sample(n_steps, &mut rng))
        .collect();
    let diversity = compute_sample_diversity(&samples, z_dim);

    CnfMetrics {
        mean_log_prob: mean_lp,
        bits_per_dim: bpd,
        sample_diversity: diversity,
    }
}

/// Evaluate a `FlowMatchingModel` on test data.
///
/// Computes approximate log-probs via KDE over model samples (Silverman bandwidth).
pub fn evaluate_flow_matching(
    model: &FlowMatchingModel,
    test_data: &[Vec<f64>],
    n_samples: usize,
    n_steps: usize,
    rng: &mut StdRng,
) -> CnfMetrics {
    let n = test_data.len();
    let z_dim = model.config.z_dim;

    if n == 0 || z_dim == 0 {
        return CnfMetrics {
            mean_log_prob: 0.0,
            bits_per_dim: 0.0,
            sample_diversity: 0.0,
        };
    }

    // Approximate log-prob using Gaussian KDE over model samples
    let model_samples = model.sample_batch(n_samples.max(50), n_steps, rng);

    // Bandwidth selection: Silverman's rule h = 1.06 * σ * n^(-1/5)
    let sample_std = compute_overall_std(&model_samples, z_dim);
    let bw = (1.06 * sample_std * (n_samples as f64).powf(-0.2)).max(1e-6);

    let log_probs: Vec<f64> = test_data
        .iter()
        .map(|x| kde_log_prob(x, &model_samples, bw, z_dim))
        .collect();
    let mean_lp = log_probs.iter().sum::<f64>() / n as f64;
    let bpd = (-mean_lp / (z_dim as f64 * 2_f64.ln())).max(0.0);

    let diversity = compute_sample_diversity(&model_samples, z_dim);

    CnfMetrics {
        mean_log_prob: mean_lp,
        bits_per_dim: bpd,
        sample_diversity: diversity,
    }
}
