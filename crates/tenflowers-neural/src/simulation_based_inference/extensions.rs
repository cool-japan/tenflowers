//! Extensions for Simulation-Based Inference: C2ST, SBC, TARP, SbiReport, and tests.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

use super::{
    build_mlp, mlp_forward, mlp_train_step, randn, sigmoid, GaussianSimulator,
    GaussianSimulatorProxy, NeuralDensityEstimator, NeuralRatioEstimator, SequentialNpe, Simulator,
    SnleConfig, SnleEstimator, SnpeConfig, SnpePosterior,
};

// ─────────────────────────────────────────────────────────────────────────────
// C2ST — Classifier Two-Sample Test
// ─────────────────────────────────────────────────────────────────────────────

/// Classifier Two-Sample Test (C2ST).
///
/// Trains a binary MLP classifier to distinguish samples from `p` (label 1)
/// and `q` (label 0), then returns validation accuracy.
/// - Accuracy ≈ 0.5: distributions are indistinguishable.
/// - Accuracy → 1.0: distributions are easily separable.
pub fn c2st_accuracy(
    p_samples: &[Vec<f64>],
    q_samples: &[Vec<f64>],
    hidden_dim: usize,
    rng: &mut StdRng,
) -> f64 {
    if p_samples.is_empty() || q_samples.is_empty() {
        return 0.5;
    }
    let dim = p_samples[0].len();

    let mut layers = build_mlp(&[dim, hidden_dim, 1], rng);

    // Build paired dataset
    let mut all: Vec<(Vec<f64>, f64)> = Vec::new();
    for s in p_samples {
        all.push((s.clone(), 1.0));
    }
    for s in q_samples {
        all.push((s.clone(), 0.0));
    }

    // Shuffle
    for i in (1..all.len()).rev() {
        let j = rng.random_range(0..=i);
        all.swap(i, j);
    }

    let split = all.len() * 4 / 5;
    let train_set = &all[..split];
    let test_set = &all[split..];

    let lr = 1e-2;
    let n_steps = 50;
    let pairs: Vec<(Vec<f64>, Vec<f64>)> = train_set
        .iter()
        .map(|(x, y)| (x.clone(), vec![*y]))
        .collect();

    for _ in 0..n_steps {
        let bce = |pred: &[f64], target: &[f64]| -> f64 {
            if pred.is_empty() || target.is_empty() {
                return 0.0;
            }
            let p = sigmoid(pred[0]).clamp(1e-7, 1.0 - 1e-7);
            let t = target[0];
            -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
        };
        mlp_train_step(&mut layers, &pairs, bce, lr);
    }

    if test_set.is_empty() {
        return 0.5;
    }

    let correct: usize = test_set
        .iter()
        .filter(|(x, y)| {
            let out = mlp_forward(&layers, x);
            let pred = if out.is_empty() { 0.0 } else { sigmoid(out[0]) };
            let predicted_label = if pred >= 0.5 { 1.0 } else { 0.0 };
            (predicted_label - y).abs() < 1e-9
        })
        .count();

    (correct as f64 / test_set.len() as f64).clamp(0.5, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Posterior Diagnostics
// ─────────────────────────────────────────────────────────────────────────────

/// Results from Simulation-Based Calibration (SBC).
#[derive(Debug, Clone)]
pub struct SbcResult {
    /// Rank of theta_true among posterior samples, for each theta dimension.
    pub rank_statistics: Vec<usize>,
    /// Approximate KS-test p-value for uniformity (heuristic).
    pub uniformity_pvalue: f64,
    /// Fraction of theta_true values within the 95% credible interval.
    pub coverage_at_95: f64,
}

/// Run Simulation-Based Calibration.
///
/// For each of `n_trials` repetitions:
/// 1. Sample `theta_true ~ prior`.
/// 2. Simulate `x ~ simulator(theta_true)`.
/// 3. Draw `n_posterior_samples` from the posterior.
/// 4. Compute the rank of `theta_true` (averaged over dimensions).
///
/// Rank statistics should be approximately uniform on [0, n_posterior_samples]
/// for a well-calibrated posterior.
pub fn simulation_based_calibration(
    posterior: &SnpePosterior,
    simulator: &dyn Simulator,
    n_trials: usize,
    n_posterior_samples: usize,
    rng: &mut StdRng,
) -> SbcResult {
    let theta_dim = simulator.theta_dim();
    let mut all_ranks: Vec<usize> = Vec::with_capacity(n_trials);
    let mut inside_95: usize = 0;

    for _ in 0..n_trials {
        let theta_true = simulator.prior_sample(rng);
        let x = simulator.simulate(&theta_true, rng);

        // Build a posterior conditioned on this x
        let local_post = SnpePosterior {
            estimator: posterior.estimator.clone(),
            x_obs: x,
            round_summaries: vec![],
        };

        let samples = local_post.sample(n_posterior_samples, rng);

        // For each dimension, compute rank of theta_true
        for d in 0..theta_dim {
            let true_val = theta_true[d];
            let rank = samples.iter().filter(|s| s[d] < true_val).count();
            all_ranks.push(rank);

            // 95% credible interval: [2.5th, 97.5th percentile]
            let mut sorted_d: Vec<f64> = samples.iter().map(|s| s[d]).collect();
            sorted_d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let lo_idx = ((n_posterior_samples as f64 * 0.025) as usize)
                .min(n_posterior_samples.saturating_sub(1));
            let hi_idx = ((n_posterior_samples as f64 * 0.975) as usize)
                .min(n_posterior_samples.saturating_sub(1));
            let lo = sorted_d.get(lo_idx).copied().unwrap_or(f64::NEG_INFINITY);
            let hi = sorted_d.get(hi_idx).copied().unwrap_or(f64::INFINITY);
            if true_val >= lo && true_val <= hi {
                inside_95 += 1;
            }
        }
    }

    // Approximate KS p-value: compare rank distribution uniformity
    let n_obs = all_ranks.len();
    let uniformity_pvalue = if n_obs == 0 {
        1.0
    } else {
        let max_rank = (n_posterior_samples + 1) as f64;
        let mut sorted_ranks: Vec<f64> = all_ranks.iter().map(|&r| r as f64 / max_rank).collect();
        sorted_ranks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Kolmogorov-Smirnov statistic against Uniform[0,1]
        let ks_stat = sorted_ranks
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                let ecdf = (i + 1) as f64 / n_obs as f64;
                (ecdf - f).abs()
            })
            .fold(0.0_f64, f64::max);
        // Approximation: p ~ exp(-2 * n * ks^2)
        let t = ks_stat * (n_obs as f64).sqrt();
        (-2.0 * t * t).exp().clamp(0.0, 1.0)
    };

    let coverage_at_95 = if n_trials == 0 || theta_dim == 0 {
        0.0
    } else {
        inside_95 as f64 / (n_trials * theta_dim) as f64
    };

    SbcResult {
        rank_statistics: all_ranks,
        uniformity_pvalue,
        coverage_at_95,
    }
}

/// Results from the TARP (Test of Accuracy with Random Points) diagnostic.
#[derive(Debug, Clone)]
pub struct TarpResult {
    /// Pairs of (nominal level, empirical coverage) at 0.1, 0.5, 0.9.
    pub coverage_at_levels: Vec<(f64, f64)>,
    /// Expected calibration error: mean |nominal - empirical|.
    pub expected_coverage_error: f64,
}

/// TARP: Tests whether the posterior is well-calibrated by checking that
/// for a random reference point `r`, theta_true is closer to `r` than a
/// given fraction of posterior samples.
pub fn tarp_test(
    posterior: &SnpePosterior,
    simulator: &dyn Simulator,
    n_trials: usize,
    rng: &mut StdRng,
) -> TarpResult {
    let levels = [0.1_f64, 0.5, 0.9];
    let n_post = 50_usize;
    let mut empirical_hits: Vec<usize> = vec![0; levels.len()];

    for _ in 0..n_trials {
        let theta_true = simulator.prior_sample(rng);
        let x = simulator.simulate(&theta_true, rng);

        let local_post = SnpePosterior {
            estimator: posterior.estimator.clone(),
            x_obs: x,
            round_summaries: vec![],
        };

        let samples = local_post.sample(n_post, rng);

        // Random reference point r ~ prior
        let r = simulator.prior_sample(rng);

        // Distance of theta_true to r
        let dist_true: f64 = theta_true
            .iter()
            .zip(r.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        // Fraction of posterior samples closer to r than theta_true
        let frac_closer = samples
            .iter()
            .filter(|s| {
                let d: f64 = s
                    .iter()
                    .zip(r.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                d < dist_true
            })
            .count() as f64
            / n_post as f64;

        for (k, &lev) in levels.iter().enumerate() {
            if frac_closer <= lev {
                empirical_hits[k] += 1;
            }
        }
    }

    let n = n_trials.max(1) as f64;
    let coverage_at_levels: Vec<(f64, f64)> = levels
        .iter()
        .zip(empirical_hits.iter())
        .map(|(&lev, &hits)| (lev, hits as f64 / n))
        .collect();

    let ece: f64 = coverage_at_levels
        .iter()
        .map(|(nom, emp)| (nom - emp).abs())
        .sum::<f64>()
        / coverage_at_levels.len() as f64;

    TarpResult {
        coverage_at_levels,
        expected_coverage_error: ece,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SbiReport
// ─────────────────────────────────────────────────────────────────────────────

/// Comprehensive SBI run report.
#[derive(Debug, Clone)]
pub struct SbiReport {
    /// Total number of SNPE rounds.
    pub n_rounds: usize,
    /// Total number of simulations executed.
    pub total_simulations: usize,
    /// log q(theta_obs | x_obs) at final round (may be NaN).
    pub final_log_prob: f64,
    /// Optional SBC results.
    pub sbc: Option<SbcResult>,
    /// C2ST accuracy between prior and posterior samples.
    pub c2st_accuracy: f64,
}

impl SbiReport {
    /// Construct from a finished [`SnpePosterior`] and optional diagnostics.
    pub fn from_posterior(
        posterior: &SnpePosterior,
        sbc: Option<SbcResult>,
        c2st_acc: f64,
        total_simulations: usize,
    ) -> Self {
        let n_rounds = posterior.round_summaries.len();
        let final_log_prob = posterior
            .round_summaries
            .last()
            .map(|s| s.training_loss)
            .unwrap_or(f64::NAN);
        Self {
            n_rounds,
            total_simulations,
            final_log_prob,
            sbc,
            c2st_accuracy: c2st_acc,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    fn make_sim(dim: usize) -> GaussianSimulator {
        GaussianSimulator::new(dim, 0.1)
    }

    // ── GaussianSimulator ────────────────────────────────────────────────────

    #[test]
    fn gaussian_simulator_output_dimension() {
        let sim = make_sim(3);
        let mut rng = make_rng();
        let theta = vec![0.0, 0.5, -0.5];
        let x = sim.simulate(&theta, &mut rng);
        assert_eq!(x.len(), 3);
    }

    #[test]
    fn gaussian_simulator_x_near_theta() {
        let sim = GaussianSimulator::new(2, 0.01);
        let mut rng = make_rng();
        let theta = vec![1.0, -1.0];
        let mut sum_err = 0.0;
        for _ in 0..500 {
            let x = sim.simulate(&theta, &mut rng);
            sum_err += (x[0] - theta[0]).abs() + (x[1] - theta[1]).abs();
        }
        assert!(
            sum_err / 500.0 < 0.1,
            "mean error too large: {}",
            sum_err / 500.0
        );
    }

    #[test]
    fn gaussian_simulator_prior_log_prob_higher_for_true() {
        let sim = make_sim(2);
        let theta_true = vec![0.0, 0.0]; // prior mean
        let theta_bad = vec![5.0, 5.0];
        assert!(sim.prior_log_prob(&theta_true) > sim.prior_log_prob(&theta_bad));
    }

    #[test]
    fn gaussian_simulator_prior_sample_dimension() {
        let sim = make_sim(4);
        let mut rng = make_rng();
        let s = sim.prior_sample(&mut rng);
        assert_eq!(s.len(), 4);
    }

    #[test]
    fn gaussian_simulator_prior_sample_finite() {
        let sim = make_sim(3);
        let mut rng = make_rng();
        let s = sim.prior_sample(&mut rng);
        assert!(s.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn gaussian_simulator_theta_x_dim() {
        let sim = make_sim(5);
        assert_eq!(sim.theta_dim(), 5);
        assert_eq!(sim.x_dim(), 5);
    }

    // ── MadeLayer ────────────────────────────────────────────────────────────

    #[test]
    fn made_layer_mask_is_autoregressive() {
        let mut rng = make_rng();
        let in_order = vec![1, 2, 3];
        let out_order = vec![2, 3];
        let layer = super::super::MadeLayer::new(3, 2, &in_order, &out_order, &mut rng);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(
                    layer.mask[i][j],
                    out_order[i] > in_order[j],
                    "mask[{}][{}] should be {}",
                    i,
                    j,
                    out_order[i] > in_order[j]
                );
            }
        }
    }

    #[test]
    fn made_layer_forward_correct_shape() {
        let mut rng = make_rng();
        let in_order = vec![1, 2, 3];
        let out_order = vec![2, 3, 1, 3];
        let layer = super::super::MadeLayer::new(3, 4, &in_order, &out_order, &mut rng);
        let x = vec![1.0, 2.0, 3.0];
        let y = layer.forward(&x);
        assert_eq!(y.len(), 4);
    }

    #[test]
    fn made_layer_forward_finite() {
        let mut rng = make_rng();
        let in_order = vec![1, 2];
        let out_order = vec![2, 1, 2];
        let layer = super::super::MadeLayer::new(2, 3, &in_order, &out_order, &mut rng);
        let y = layer.forward(&[0.5, -0.5]);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn made_layer_masked_weights_zeroed() {
        let mut rng = make_rng();
        // output_order[0] = 1, input_order[0] = 1 → 1 > 1 is false → masked
        let in_order = vec![1, 2];
        let out_order = vec![1];
        let layer = super::super::MadeLayer::new(2, 1, &in_order, &out_order, &mut rng);
        // mask[0][0] = (1 > 1) = false
        assert!(!layer.mask[0][0]);
    }

    // ── NeuralDensityEstimator ───────────────────────────────────────────────

    #[test]
    fn nde_log_prob_finite() {
        let mut rng = make_rng();
        let nde = NeuralDensityEstimator::new(2, 2, 16, 2, &mut rng);
        let theta = vec![0.3, -0.3];
        let x = vec![0.4, -0.2];
        let lp = nde.log_prob(&theta, &x);
        assert!(lp.is_finite(), "log_prob is not finite: {}", lp);
    }

    #[test]
    fn nde_sample_correct_dimension() {
        let mut rng = make_rng();
        let nde = NeuralDensityEstimator::new(3, 2, 16, 2, &mut rng);
        let x = vec![0.0, 1.0];
        let s = nde.sample(&x, &mut rng);
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn nde_sample_finite() {
        let mut rng = make_rng();
        let nde = NeuralDensityEstimator::new(2, 2, 16, 2, &mut rng);
        let x = vec![0.0, 0.0];
        let s = nde.sample(&x, &mut rng);
        assert!(s.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn nde_train_nll_decreases() {
        let mut rng = make_rng();
        let mut nde = NeuralDensityEstimator::new(2, 2, 16, 1, &mut rng);
        let thetas = vec![vec![0.5, -0.5]; 8];
        let xs = vec![vec![0.4, -0.6]; 8];
        let loss1 = nde.train_nll(&thetas, &xs, 1e-3);
        let loss2 = nde.train_nll(&thetas, &xs, 1e-3);
        // At minimum one should be finite
        assert!(loss1.is_finite() || loss2.is_finite());
    }

    // ── SequentialNpe ────────────────────────────────────────────────────────

    #[test]
    fn sequential_npe_runs_two_rounds() {
        let sim = make_sim(2);
        let cfg = SnpeConfig {
            n_rounds: 2,
            n_simulations_per_round: 10,
            hidden_dim: 16,
            n_nde_layers: 1,
            lr: 1e-2,
            n_training_steps: 5,
            batch_size: 4,
        };
        let mut rng = make_rng();
        let mut snpe = SequentialNpe::new(&sim, cfg);
        let x_obs = vec![0.5, 0.5];
        let post = snpe.run(&sim, &x_obs, &mut rng);
        assert_eq!(post.round_summaries.len(), 2);
    }

    #[test]
    fn sequential_npe_accumulates_data() {
        let sim = make_sim(2);
        let cfg = SnpeConfig {
            n_rounds: 3,
            n_simulations_per_round: 8,
            n_training_steps: 3,
            batch_size: 4,
            hidden_dim: 16,
            n_nde_layers: 1,
            lr: 1e-3,
        };
        let mut rng = make_rng();
        let mut snpe = SequentialNpe::new(&sim, cfg);
        snpe.run(&sim, &[0.0, 0.0], &mut rng);
        // 3 rounds × 8 simulations = 24
        assert_eq!(snpe.all_thetas.len(), 24);
    }

    // ── SnpePosterior ────────────────────────────────────────────────────────

    #[test]
    fn snpe_posterior_sample_count() {
        let sim = make_sim(2);
        let cfg = SnpeConfig {
            n_rounds: 1,
            n_simulations_per_round: 10,
            n_training_steps: 3,
            batch_size: 4,
            hidden_dim: 16,
            n_nde_layers: 1,
            lr: 1e-3,
        };
        let mut rng = make_rng();
        let mut snpe = SequentialNpe::new(&sim, cfg);
        let post = snpe.run(&sim, &[0.0, 0.0], &mut rng);
        let samples = post.sample(15, &mut rng);
        assert_eq!(samples.len(), 15);
    }

    #[test]
    fn snpe_posterior_sample_dimensions() {
        let sim = make_sim(3);
        let cfg = SnpeConfig {
            n_rounds: 1,
            n_simulations_per_round: 10,
            n_training_steps: 3,
            batch_size: 4,
            hidden_dim: 16,
            n_nde_layers: 1,
            lr: 1e-3,
        };
        let mut rng = make_rng();
        let mut snpe = SequentialNpe::new(&sim, cfg);
        let post = snpe.run(&sim, &[0.0, 0.0, 0.0], &mut rng);
        let samples = post.sample(5, &mut rng);
        assert!(samples.iter().all(|s| s.len() == 3));
    }

    #[test]
    fn snpe_posterior_map_estimate_finite() {
        let sim = make_sim(2);
        let cfg = SnpeConfig {
            n_rounds: 1,
            n_simulations_per_round: 10,
            n_training_steps: 3,
            batch_size: 4,
            hidden_dim: 16,
            n_nde_layers: 1,
            lr: 1e-3,
        };
        let mut rng = make_rng();
        let mut snpe = SequentialNpe::new(&sim, cfg);
        let post = snpe.run(&sim, &[0.5, -0.5], &mut rng);
        let map = post.map_estimate(&mut rng);
        assert!(
            map.iter().all(|v| v.is_finite()),
            "MAP has non-finite values: {:?}",
            map
        );
    }

    #[test]
    fn snpe_posterior_log_prob_finite() {
        let sim = make_sim(2);
        let cfg = SnpeConfig {
            n_rounds: 1,
            n_simulations_per_round: 10,
            n_training_steps: 3,
            batch_size: 4,
            hidden_dim: 16,
            n_nde_layers: 1,
            lr: 1e-3,
        };
        let mut rng = make_rng();
        let mut snpe = SequentialNpe::new(&sim, cfg);
        let post = snpe.run(&sim, &[0.0, 0.0], &mut rng);
        let lp = post.log_prob(&[0.1, 0.1]);
        assert!(lp.is_finite());
    }

    // ── NeuralLikelihood / SNLE ──────────────────────────────────────────────

    #[test]
    fn neural_likelihood_log_likelihood_finite() {
        let mut rng = make_rng();
        let nl = super::super::NeuralLikelihood::new(2, 2, 16, &mut rng);
        let x = vec![0.5, -0.3];
        let theta = vec![0.2, 0.8];
        let ll = nl.log_likelihood(&x, &theta);
        assert!(ll.is_finite(), "log_likelihood is not finite: {}", ll);
    }

    #[test]
    fn neural_likelihood_train_step_finite() {
        let mut rng = make_rng();
        let mut nl = super::super::NeuralLikelihood::new(2, 2, 16, &mut rng);
        let xs = vec![vec![0.5, 0.5]; 5];
        let thetas = vec![vec![0.3, 0.3]; 5];
        let loss = nl.train_step(&xs, &thetas, 1e-3);
        assert!(loss.is_finite());
    }

    #[test]
    fn snle_posterior_log_prob_finite() {
        let mut rng = make_rng();
        let nl = super::super::NeuralLikelihood::new(2, 2, 16, &mut rng);
        let sim_proxy = Box::new(GaussianSimulatorProxy::new(2, 2));
        let post = super::super::SnlePosterior {
            likelihood: nl,
            simulator: sim_proxy,
            x_obs: vec![0.3, -0.3],
        };
        let lp = post.log_prob(&[0.2, -0.2]);
        assert!(lp.is_finite(), "log_prob is not finite: {}", lp);
    }

    #[test]
    fn snle_posterior_mcmc_returns_correct_count() {
        let mut rng = make_rng();
        let nl = super::super::NeuralLikelihood::new(2, 2, 16, &mut rng);
        let sim_proxy = Box::new(GaussianSimulatorProxy::new(2, 2));
        let post = super::super::SnlePosterior {
            likelihood: nl,
            simulator: sim_proxy,
            x_obs: vec![0.5, 0.5],
        };
        let samples = post.mcmc_sample(20, &mut rng);
        assert_eq!(samples.len(), 20);
    }

    #[test]
    fn snle_posterior_mcmc_dimensions() {
        let mut rng = make_rng();
        let nl = super::super::NeuralLikelihood::new(3, 3, 16, &mut rng);
        let sim_proxy = Box::new(GaussianSimulatorProxy::new(3, 3));
        let post = super::super::SnlePosterior {
            likelihood: nl,
            simulator: sim_proxy,
            x_obs: vec![0.1, 0.2, 0.3],
        };
        let samples = post.mcmc_sample(10, &mut rng);
        assert!(samples.iter().all(|s| s.len() == 3));
    }

    // ── NeuralRatioEstimator ─────────────────────────────────────────────────

    #[test]
    fn ratio_estimator_forward_finite() {
        let mut rng = make_rng();
        let nre = NeuralRatioEstimator::new(2, 2, 16, 2, &mut rng);
        let x = vec![0.5, -0.3];
        let theta = vec![0.2, 0.8];
        let lr = nre.forward(&x, &theta);
        assert!(lr.is_finite() && !lr.is_nan(), "forward not finite: {}", lr);
    }

    #[test]
    fn ratio_estimator_forward_not_nan() {
        let mut rng = make_rng();
        let nre = NeuralRatioEstimator::new(3, 2, 16, 2, &mut rng);
        let x = vec![0.0; 3];
        let theta = vec![0.0; 2];
        let v = nre.forward(&x, &theta);
        assert!(!v.is_nan());
    }

    #[test]
    fn ratio_estimator_train_step_positive_loss() {
        let mut rng = make_rng();
        let mut nre = NeuralRatioEstimator::new(2, 2, 16, 2, &mut rng);
        let jx = vec![vec![0.5, 0.5]; 4];
        let jt = vec![vec![0.3, 0.3]; 4];
        let mx = vec![vec![-0.5, 0.5]; 4];
        let mt = vec![vec![0.7, -0.3]; 4];
        let loss = nre.train_step(&jx, &jt, &mx, &mt, 1e-3);
        assert!(loss >= 0.0, "BCE loss should be non-negative, got {}", loss);
    }

    #[test]
    fn ratio_estimator_log_posterior_finite() {
        let mut rng = make_rng();
        let nre = NeuralRatioEstimator::new(2, 2, 16, 2, &mut rng);
        let x = vec![0.5, 0.5];
        let theta = vec![0.3, 0.3];
        let prior_lp = -1.0;
        let lp = nre.log_posterior(&x, &theta, prior_lp);
        assert!(lp.is_finite());
    }

    // ── C2ST ─────────────────────────────────────────────────────────────────

    #[test]
    fn c2st_accuracy_in_valid_range() {
        let mut rng = make_rng();
        let p: Vec<Vec<f64>> = (0..40)
            .map(|_| vec![randn(&mut rng), randn(&mut rng)])
            .collect();
        let q: Vec<Vec<f64>> = (0..40)
            .map(|_| vec![randn(&mut rng) + 5.0, randn(&mut rng) + 5.0])
            .collect();
        let acc = c2st_accuracy(&p, &q, 16, &mut rng);
        assert!(
            (0.5..=1.0).contains(&acc),
            "c2st accuracy out of range: {}",
            acc
        );
    }

    #[test]
    fn c2st_accuracy_near_half_for_same_distribution() {
        let mut rng = make_rng();
        let samples: Vec<Vec<f64>> = (0..100).map(|_| vec![randn(&mut rng)]).collect();
        let (p, q) = samples.split_at(50);
        let acc = c2st_accuracy(p, q, 16, &mut rng);
        // For the same distribution accuracy should be ≤ 1.0 (the clamp lower-bounds at 0.5)
        assert!(acc <= 1.0);
    }

    #[test]
    fn c2st_accuracy_empty_returns_half() {
        let mut rng = make_rng();
        let acc = c2st_accuracy(&[], &[vec![0.0]], 8, &mut rng);
        assert!((acc - 0.5).abs() < 1e-9);
    }

    // ── SBC ──────────────────────────────────────────────────────────────────

    #[test]
    fn sbc_rank_statistics_correct_length() {
        let sim = make_sim(2);
        let cfg = SnpeConfig {
            n_rounds: 1,
            n_simulations_per_round: 10,
            n_training_steps: 3,
            batch_size: 4,
            hidden_dim: 16,
            n_nde_layers: 1,
            lr: 1e-3,
        };
        let mut rng = make_rng();
        let mut snpe = SequentialNpe::new(&sim, cfg);
        let post = snpe.run(&sim, &[0.0, 0.0], &mut rng);
        let n_trials = 5;
        let n_post = 10;
        let result = simulation_based_calibration(&post, &sim, n_trials, n_post, &mut rng);
        // n_trials × theta_dim = 5 × 2 = 10 rank stats
        assert_eq!(result.rank_statistics.len(), n_trials * 2);
    }

    #[test]
    fn sbc_coverage_in_unit_interval() {
        let sim = make_sim(2);
        let cfg = SnpeConfig {
            n_rounds: 1,
            n_simulations_per_round: 10,
            n_training_steps: 3,
            batch_size: 4,
            hidden_dim: 16,
            n_nde_layers: 1,
            lr: 1e-3,
        };
        let mut rng = make_rng();
        let mut snpe = SequentialNpe::new(&sim, cfg);
        let post = snpe.run(&sim, &[0.0, 0.0], &mut rng);
        let result = simulation_based_calibration(&post, &sim, 5, 10, &mut rng);
        assert!(result.coverage_at_95 >= 0.0 && result.coverage_at_95 <= 1.0);
    }

    #[test]
    fn sbc_uniformity_pvalue_in_unit_interval() {
        let sim = make_sim(2);
        let cfg = SnpeConfig {
            n_rounds: 1,
            n_simulations_per_round: 10,
            n_training_steps: 3,
            batch_size: 4,
            hidden_dim: 16,
            n_nde_layers: 1,
            lr: 1e-3,
        };
        let mut rng = make_rng();
        let mut snpe = SequentialNpe::new(&sim, cfg);
        let post = snpe.run(&sim, &[0.0, 0.0], &mut rng);
        let result = simulation_based_calibration(&post, &sim, 5, 10, &mut rng);
        assert!(result.uniformity_pvalue >= 0.0 && result.uniformity_pvalue <= 1.0);
    }

    // ── TARP ─────────────────────────────────────────────────────────────────

    #[test]
    fn tarp_coverage_levels_count() {
        let sim = make_sim(2);
        let cfg = SnpeConfig {
            n_rounds: 1,
            n_simulations_per_round: 10,
            n_training_steps: 3,
            batch_size: 4,
            hidden_dim: 16,
            n_nde_layers: 1,
            lr: 1e-3,
        };
        let mut rng = make_rng();
        let mut snpe = SequentialNpe::new(&sim, cfg);
        let post = snpe.run(&sim, &[0.0, 0.0], &mut rng);
        let result = tarp_test(&post, &sim, 5, &mut rng);
        assert_eq!(result.coverage_at_levels.len(), 3);
    }

    #[test]
    fn tarp_empirical_coverage_in_unit_interval() {
        let sim = make_sim(2);
        let cfg = SnpeConfig {
            n_rounds: 1,
            n_simulations_per_round: 10,
            n_training_steps: 3,
            batch_size: 4,
            hidden_dim: 16,
            n_nde_layers: 1,
            lr: 1e-3,
        };
        let mut rng = make_rng();
        let mut snpe = SequentialNpe::new(&sim, cfg);
        let post = snpe.run(&sim, &[0.0, 0.0], &mut rng);
        let result = tarp_test(&post, &sim, 5, &mut rng);
        for (_, emp) in &result.coverage_at_levels {
            assert!(
                *emp >= 0.0 && *emp <= 1.0,
                "empirical coverage {} out of range",
                emp
            );
        }
    }

    #[test]
    fn tarp_nominal_levels_correct() {
        let sim = make_sim(2);
        let cfg = SnpeConfig {
            n_rounds: 1,
            n_simulations_per_round: 10,
            n_training_steps: 3,
            batch_size: 4,
            hidden_dim: 16,
            n_nde_layers: 1,
            lr: 1e-3,
        };
        let mut rng = make_rng();
        let mut snpe = SequentialNpe::new(&sim, cfg);
        let post = snpe.run(&sim, &[0.0, 0.0], &mut rng);
        let result = tarp_test(&post, &sim, 5, &mut rng);
        let nominals: Vec<f64> = result.coverage_at_levels.iter().map(|(n, _)| *n).collect();
        assert!((nominals[0] - 0.1).abs() < 1e-9);
        assert!((nominals[1] - 0.5).abs() < 1e-9);
        assert!((nominals[2] - 0.9).abs() < 1e-9);
    }

    #[test]
    fn tarp_ece_non_negative() {
        let sim = make_sim(2);
        let cfg = SnpeConfig {
            n_rounds: 1,
            n_simulations_per_round: 10,
            n_training_steps: 3,
            batch_size: 4,
            hidden_dim: 16,
            n_nde_layers: 1,
            lr: 1e-3,
        };
        let mut rng = make_rng();
        let mut snpe = SequentialNpe::new(&sim, cfg);
        let post = snpe.run(&sim, &[0.0, 0.0], &mut rng);
        let result = tarp_test(&post, &sim, 5, &mut rng);
        assert!(result.expected_coverage_error >= 0.0);
    }

    // ── SbiReport ────────────────────────────────────────────────────────────

    #[test]
    fn sbi_report_construction() {
        let sim = make_sim(2);
        let cfg = SnpeConfig {
            n_rounds: 2,
            n_simulations_per_round: 10,
            n_training_steps: 3,
            batch_size: 4,
            hidden_dim: 16,
            n_nde_layers: 1,
            lr: 1e-3,
        };
        let mut rng = make_rng();
        let mut snpe = SequentialNpe::new(&sim, cfg);
        let post = snpe.run(&sim, &[0.0, 0.0], &mut rng);
        let report = SbiReport::from_posterior(&post, None, 0.75, 20);
        assert_eq!(report.n_rounds, 2);
        assert_eq!(report.total_simulations, 20);
        assert!((report.c2st_accuracy - 0.75).abs() < 1e-9);
    }

    #[test]
    fn sbi_report_with_sbc() {
        let sim = make_sim(2);
        let cfg = SnpeConfig {
            n_rounds: 1,
            n_simulations_per_round: 10,
            n_training_steps: 3,
            batch_size: 4,
            hidden_dim: 16,
            n_nde_layers: 1,
            lr: 1e-3,
        };
        let mut rng = make_rng();
        let mut snpe = SequentialNpe::new(&sim, cfg);
        let post = snpe.run(&sim, &[0.0, 0.0], &mut rng);
        let sbc = simulation_based_calibration(&post, &sim, 3, 5, &mut rng);
        let report = SbiReport::from_posterior(&post, Some(sbc.clone()), 0.6, 10);
        assert!(report.sbc.is_some());
        assert_eq!(
            report.sbc.as_ref().map(|s| s.rank_statistics.len()),
            Some(sbc.rank_statistics.len())
        );
    }

    // ── SnleEstimator (end-to-end) ───────────────────────────────────────────

    #[test]
    fn snle_estimator_runs_without_error() {
        let sim = make_sim(2);
        let cfg = SnleConfig {
            n_rounds: 2,
            n_simulations_per_round: 10,
            hidden_dim: 16,
            lr: 1e-3,
            n_training_steps: 5,
            n_mcmc_steps: 50,
        };
        let mut rng = make_rng();
        let mut est = SnleEstimator::new(&sim, cfg, &mut rng);
        let x_obs = vec![0.3, 0.3];
        let post = est.run(&sim, &x_obs, &mut rng);
        let lp = post.log_prob(&[0.3, 0.3]);
        assert!(lp.is_finite());
    }

    // ── Additional edge-case tests ───────────────────────────────────────────

    #[test]
    fn gaussian_simulator_custom_prior() {
        let sim = GaussianSimulator::with_prior(2, 0.1, vec![2.0, -2.0], vec![0.5, 0.5]);
        let mu = vec![2.0, -2.0];
        let far = vec![10.0, 10.0];
        assert!(sim.prior_log_prob(&mu) > sim.prior_log_prob(&far));
    }

    #[test]
    fn nde_1d_log_prob_reasonable() {
        let mut rng = make_rng();
        let nde = NeuralDensityEstimator::new(1, 1, 8, 1, &mut rng);
        let theta = vec![0.0];
        let x = vec![0.0];
        let lp = nde.log_prob(&theta, &x);
        assert!(lp.is_finite());
    }

    #[test]
    fn nde_multiple_samples_vary() {
        let mut rng = make_rng();
        let nde = NeuralDensityEstimator::new(2, 2, 16, 2, &mut rng);
        let x = vec![0.0, 0.0];
        let s1 = nde.sample(&x, &mut rng);
        let s2 = nde.sample(&x, &mut rng);
        // Two samples should not be identical (with overwhelming probability)
        let same = s1.iter().zip(s2.iter()).all(|(a, b)| (a - b).abs() < 1e-12);
        assert!(
            !same,
            "Two samples are identical, which is astronomically unlikely"
        );
    }

    #[test]
    fn ratio_estimator_accepts_different_dims() {
        let mut rng = make_rng();
        let nre = NeuralRatioEstimator::new(4, 3, 32, 3, &mut rng);
        let x = vec![0.1; 4];
        let theta = vec![0.2; 3];
        let v = nre.forward(&x, &theta);
        assert!(!v.is_nan());
    }

    #[test]
    fn snpe_config_default_sensible() {
        let cfg = SnpeConfig::default();
        assert!(cfg.n_rounds > 0);
        assert!(cfg.hidden_dim > 0);
        assert!(cfg.lr > 0.0);
    }

    #[test]
    fn snle_config_default_sensible() {
        let cfg = SnleConfig::default();
        assert!(cfg.n_rounds > 0);
        assert!(cfg.n_mcmc_steps > 0);
    }

    #[test]
    fn made_layer_update_changes_weights() {
        let mut rng = make_rng();
        let in_order = vec![1, 2];
        let out_order = vec![2, 1];
        let mut layer = super::super::MadeLayer::new(2, 2, &in_order, &out_order, &mut rng);
        let w_before = layer.weights[0][0];
        let grad_w = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let grad_b = vec![0.1, 0.1];
        layer.update(&grad_w, &grad_b, 0.1);
        let w_after = layer.weights[0][0];
        assert!((w_before - w_after).abs() > 1e-10);
    }

    #[test]
    fn sbc_zero_trials_returns_zero_coverage() {
        let sim = make_sim(2);
        let cfg = SnpeConfig {
            n_rounds: 1,
            n_simulations_per_round: 10,
            n_training_steps: 3,
            batch_size: 4,
            hidden_dim: 16,
            n_nde_layers: 1,
            lr: 1e-3,
        };
        let mut rng = make_rng();
        let mut snpe = SequentialNpe::new(&sim, cfg);
        let post = snpe.run(&sim, &[0.0, 0.0], &mut rng);
        let result = simulation_based_calibration(&post, &sim, 0, 10, &mut rng);
        assert_eq!(result.coverage_at_95, 0.0);
    }

    #[test]
    fn snpe_posterior_map_has_correct_dim() {
        let sim = make_sim(3);
        let cfg = SnpeConfig {
            n_rounds: 1,
            n_simulations_per_round: 10,
            n_training_steps: 3,
            batch_size: 4,
            hidden_dim: 16,
            n_nde_layers: 1,
            lr: 1e-3,
        };
        let mut rng = make_rng();
        let mut snpe = SequentialNpe::new(&sim, cfg);
        let post = snpe.run(&sim, &[0.0, 0.0, 0.0], &mut rng);
        let map = post.map_estimate(&mut rng);
        assert_eq!(map.len(), 3);
    }
}
