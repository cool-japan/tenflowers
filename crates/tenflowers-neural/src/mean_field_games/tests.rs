//! Tests for mean_field_games module (core, extensions, advanced).

use super::*;
use super::advanced::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};

// ── helpers ──────────────────────────────────────────────────────────────────

fn softmax_f64(v: &[f64]) -> Vec<f64> {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|x| (x - max).exp()).collect();
    let s: f64 = exps.iter().sum::<f64>().max(1e-15);
    exps.iter().map(|e| e / s).collect()
}

// ── MfgDistribution ──────────────────────────────────────────────────────────

#[test]
fn test_distribution_uniform_sum_to_one() {
    let d = MfgDistribution::uniform(10, -1.0, 1.0).expect("uniform");
    let s: f64 = d.weights.iter().sum();
    assert!((s - 1.0).abs() < 1e-10, "sum={s}");
}

#[test]
fn test_distribution_from_weights_normalises() {
    let d = MfgDistribution::from_weights(vec![1.0, 2.0, 3.0], 0.0, 1.0).expect("fw");
    assert!(((d.weights.iter().sum::<f64>()) - 1.0).abs() < 1e-10);
    assert!((d.weights[2] - 0.5).abs() < 1e-10);
}

#[test]
fn test_distribution_n_bins_zero_error() {
    assert!(MfgDistribution::uniform(0, 0.0, 1.0).is_err());
}

#[test]
fn test_distribution_mean_symmetric() {
    let d = MfgDistribution::uniform(11, -1.0, 1.0).expect("d");
    assert!(d.mean().abs() < 1e-10);
}

#[test]
fn test_distribution_variance_positive() {
    let d = MfgDistribution::uniform(10, -1.0, 1.0).expect("d");
    assert!(d.variance() > 0.0);
}

#[test]
fn test_distribution_skewness_symmetric_zero() {
    let d = MfgDistribution::uniform(11, -1.0, 1.0).expect("d");
    assert!(d.skewness().abs() < 1e-8);
}

#[test]
fn test_distribution_features_length() {
    let d = MfgDistribution::uniform(8, 0.0, 2.0).expect("d");
    assert_eq!(d.features().len(), 3);
}

#[test]
fn test_distribution_wasserstein1_identical() {
    let d1 = MfgDistribution::uniform(8, 0.0, 1.0).expect("d1");
    let d2 = MfgDistribution::uniform(8, 0.0, 1.0).expect("d2");
    assert!(d1.wasserstein1(&d2).expect("w1") < 1e-10);
}

#[test]
fn test_distribution_wasserstein1_different() {
    let d1 = MfgDistribution::from_weights(vec![1.0, 0.0, 0.0, 0.0], 0.0, 3.0).expect("d1");
    let d2 = MfgDistribution::from_weights(vec![0.0, 0.0, 0.0, 1.0], 0.0, 3.0).expect("d2");
    assert!(d1.wasserstein1(&d2).expect("w1") > 0.5);
}

#[test]
fn test_distribution_deposit_and_normalise() {
    let mut d = MfgDistribution {
        weights: vec![0.0; 5],
        x_min: 0.0,
        x_max: 4.0,
    };
    d.deposit_state(2.0);
    d.normalise();
    assert!((d.weights.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}

#[test]
fn test_distribution_grid_point() {
    let d = MfgDistribution::uniform(5, 0.0, 4.0).expect("d");
    assert!((d.grid_point(0) - 0.0).abs() < 1e-10);
    assert!((d.grid_point(4) - 4.0).abs() < 1e-10);
}

// ── McKeanVlasovDynamics ─────────────────────────────────────────────────────

#[test]
fn test_mcv_drift_zero_action_mean_reversion() {
    let dyn_ = McKeanVlasovDynamics::new(MckeanVlasovConfig {
        kappa: 1.0,
        sigma_mf: 0.0,
        sigma: 0.0,
        dt: 0.1,
    });
    let mu = MfgDistribution::uniform(5, -1.0, 1.0).expect("mu");
    assert!((dyn_.drift(MfgState(1.0), MfgAction(0.0), &mu) - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_mcv_deterministic_step() {
    let dyn_ = McKeanVlasovDynamics::new(MckeanVlasovConfig {
        kappa: 0.0,
        sigma_mf: 0.0,
        sigma: 0.0,
        dt: 0.1,
    });
    let mu = MfgDistribution::uniform(5, -1.0, 1.0).expect("mu");
    let next = dyn_.step_det(MfgState(0.0), MfgAction(1.0), &mu);
    assert!((next.0 - 0.1).abs() < 1e-10);
}

#[test]
fn test_mcv_stochastic_step_finite() {
    let dyn_ = McKeanVlasovDynamics::new(MckeanVlasovConfig::default());
    let mu = MfgDistribution::uniform(10, -3.0, 3.0).expect("mu");
    let mut rng = StdRng::seed_from_u64(0);
    assert!(dyn_
        .step(MfgState(0.0), MfgAction(0.0), &mu, &mut rng)
        .0
        .is_finite());
}

// ── LQ-MFG ───────────────────────────────────────────────────────────────────

#[test]
fn test_lq_riccati_terminal_condition() {
    let params = LqMfgParams {
        q_terminal: 3.0,
        ..Default::default()
    };
    let lqmfg = LinearQuadraticMfg::new(params.clone());
    let riccati = lqmfg.solve_riccati();
    assert_eq!(riccati.len(), params.n_steps + 1);
    assert!((riccati[params.n_steps] - 3.0).abs() < 1e-10);
}

#[test]
fn test_lq_riccati_positive() {
    assert!(LinearQuadraticMfg::new(LqMfgParams::default())
        .solve_riccati()
        .iter()
        .all(|&p| p > 0.0));
}

#[test]
fn test_lq_optimal_control_stable() {
    let a = LinearQuadraticMfg::new(LqMfgParams::default())
        .compute_optimal_control(0, 1.0)
        .expect("a");
    assert!(a < 0.0);
}

#[test]
fn test_lq_value_function_nonneg() {
    let v = LinearQuadraticMfg::new(LqMfgParams::default())
        .value_function(0, 1.0)
        .expect("v");
    assert!(v >= 0.0);
}

#[test]
fn test_full_pipeline_metrics() {
    let lqmfg = LinearQuadraticMfg::new(LqMfgParams {
        n_steps: 10,
        horizon: 1.0,
        ..Default::default()
    });
    let (riccati, gains) = lqmfg.compute_equilibrium();
    assert_eq!(riccati.len(), 11);
    assert_eq!(gains.len(), 11);
    let v = lqmfg.value_function(0, 1.0).expect("vf");
    let a = lqmfg.compute_optimal_control(0, 1.0).expect("ctrl");
    assert!(v.is_finite());
    assert!(a.is_finite());
    let poa = MfgMetrics::price_of_anarchy(v, lqmfg.social_optimal_cost(1.0));
    assert!(poa > 0.0);
}

// ── MfgMetrics (extensions) ──────────────────────────────────────────────────

#[test]
fn test_metrics_nash_error_identical_zero() {
    let mu: Vec<_> = (0..3)
        .map(|_| MfgDistribution::uniform(8, -1.0, 1.0).expect("mu"))
        .collect();
    assert!(MfgMetrics::mean_field_nash_error(&mu, &mu.clone()).expect("err") < 1e-10);
}

#[test]
fn test_metrics_social_cost_mean() {
    assert!((MfgMetrics::social_cost(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-10);
}

#[test]
fn test_metrics_price_of_anarchy_gt_one() {
    assert!((MfgMetrics::price_of_anarchy(2.0, 1.0) - 2.0).abs() < 1e-10);
}

// ── MfcCostFunctional ─────────────────────────────────────────────────────────

#[test]
fn test_mfc_cost_functional_nonneg() {
    let cost = MfcCostFunctional::new(&[8], 0).expect("cost");
    let mu = MfgDistribution::uniform(8, -2.0, 2.0).expect("mu");
    let v = cost.evaluate(1.0, 0.5, &mu);
    assert!(v >= 0.0);
}

#[test]
fn test_mfc_cost_functional_running_cost() {
    let cost = MfcCostFunctional::new(&[4], 1).expect("cost");
    let mu = MfgDistribution::uniform(8, -2.0, 2.0).expect("mu");
    let rc = cost.running_cost(MfgState(0.5), MfgAction(-0.3), &mu);
    assert!(rc.is_finite());
}

#[test]
fn test_mfc_cost_functional_wrong_hidden_error() {
    assert!(MfcCostFunctional::new(&[], 0).is_err());
}

// ── MfcValueFunction ──────────────────────────────────────────────────────────

#[test]
fn test_mfc_value_function_forward_finite() {
    let vf = MfcValueFunction::new(&[8], 0).expect("vf");
    let mu = MfgDistribution::uniform(8, -2.0, 2.0).expect("mu");
    assert!(vf.forward(1.0, &mu).is_finite());
}

#[test]
fn test_mfc_value_function_n_params_positive() {
    let vf = MfcValueFunction::new(&[8, 4], 42).expect("vf");
    assert!(vf.n_params() > 0);
}

#[test]
fn test_mfc_value_function_no_hidden_error() {
    assert!(MfcValueFunction::new(&[], 0).is_err());
}

// ── MfcPolicyGradient ─────────────────────────────────────────────────────────

#[test]
fn test_mfc_policy_gradient_train_runs() {
    let cfg = MfcPolicyGradientConfig {
        n_agents: 10,
        n_steps: 5,
        n_episodes: 3,
        n_bins: 8,
        ..Default::default()
    };
    let dyn_ = McKeanVlasovDynamics::new(MckeanVlasovConfig::default());
    let solver = MfcPolicyGradient::new(cfg, dyn_, default_quadratic_cost());
    let result = solver.train().expect("train");
    assert_eq!(result.episode_costs.len(), 3);
}

#[test]
fn test_mfc_policy_gradient_costs_finite() {
    let cfg = MfcPolicyGradientConfig {
        n_agents: 5,
        n_steps: 3,
        n_episodes: 2,
        n_bins: 8,
        ..Default::default()
    };
    let dyn_ = McKeanVlasovDynamics::new(MckeanVlasovConfig::default());
    let solver = MfcPolicyGradient::new(cfg, dyn_, default_quadratic_cost());
    let result = solver.train().expect("train");
    assert!(result.episode_costs.iter().all(|c| c.is_finite()));
}

#[test]
fn test_mfc_policy_gradient_distributions_normalised() {
    let cfg = MfcPolicyGradientConfig {
        n_agents: 10,
        n_steps: 4,
        n_episodes: 2,
        n_bins: 8,
        ..Default::default()
    };
    let dyn_ = McKeanVlasovDynamics::new(MckeanVlasovConfig::default());
    let solver = MfcPolicyGradient::new(cfg, dyn_, default_quadratic_cost());
    let result = solver.train().expect("train");
    for mu in &result.mu_seq {
        assert!((mu.weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }
}

// ── GraphonKernel ─────────────────────────────────────────────────────────────

#[test]
fn test_graphon_kernel_erdos_renyi_eval() {
    let k = GraphonKernel::erdos_renyi(4, 0.5).expect("k");
    assert!((k.eval(0.0, 0.0) - 0.5).abs() < 1e-12);
    assert!((k.eval(0.9, 0.9) - 0.5).abs() < 1e-12);
}

#[test]
fn test_graphon_kernel_block_structure() {
    let k = GraphonKernel::block(8, 2, 0.9, 0.1).expect("k");
    // same block: u=0.1, v=0.2 → both in block 0
    assert!(k.eval(0.1, 0.2) > 0.5);
    // cross block: u=0.1, v=0.8
    assert!(k.eval(0.1, 0.8) < 0.5);
}

#[test]
fn test_graphon_kernel_n_zero_error() {
    assert!(GraphonKernel::erdos_renyi(0, 0.5).is_err());
}

#[test]
fn test_graphon_kernel_wrong_size_error() {
    assert!(GraphonKernel::new(vec![0.5; 5], 3).is_err());
}

#[test]
fn test_graphon_kernel_weighted_mean_finite() {
    let k = GraphonKernel::erdos_renyi(4, 0.5).expect("k");
    let dists: Vec<MfgDistribution> = (0..4)
        .map(|_| MfgDistribution::uniform(8, -2.0, 2.0).expect("mu"))
        .collect();
    assert!(k.weighted_mean(0, &dists).is_finite());
}

// ── GraphonMfgSolver ──────────────────────────────────────────────────────────

#[test]
fn test_graphon_mfg_solver_runs() {
    let kernel = GraphonKernel::erdos_renyi(2, 0.5).expect("k");
    let cfg = GraphonMfgConfig {
        n_nodes: 2,
        n_bins: 8,
        n_steps: 4,
        n_agents_per_node: 10,
        max_iters: 2,
        ..Default::default()
    };
    let dyn_ = McKeanVlasovDynamics::new(MckeanVlasovConfig::default());
    let solver = GraphonMfgSolver::new(kernel, cfg, dyn_);
    let eq = solver.solve().expect("solve");
    assert_eq!(eq.node_distributions.len(), 2);
    assert_eq!(eq.node_distributions[0].len(), 5); // n_steps + 1
}

#[test]
fn test_graphon_mfg_distributions_normalised() {
    let kernel = GraphonKernel::erdos_renyi(3, 0.7).expect("k");
    let cfg = GraphonMfgConfig {
        n_nodes: 3,
        n_bins: 8,
        n_steps: 3,
        n_agents_per_node: 8,
        max_iters: 2,
        ..Default::default()
    };
    let dyn_ = McKeanVlasovDynamics::new(MckeanVlasovConfig::default());
    let solver = GraphonMfgSolver::new(kernel, cfg, dyn_);
    let eq = solver.solve().expect("solve");
    for node_seq in &eq.node_distributions {
        for mu in node_seq {
            assert!((mu.weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        }
    }
}

#[test]
fn test_graphon_mfg_residuals_nonempty() {
    let kernel = GraphonKernel::erdos_renyi(2, 0.5).expect("k");
    let cfg = GraphonMfgConfig {
        n_nodes: 2,
        n_bins: 8,
        n_steps: 3,
        n_agents_per_node: 5,
        max_iters: 3,
        ..Default::default()
    };
    let dyn_ = McKeanVlasovDynamics::new(MckeanVlasovConfig::default());
    let solver = GraphonMfgSolver::new(kernel, cfg, dyn_);
    let eq = solver.solve().expect("solve");
    assert!(!eq.residuals.is_empty());
}

// ── ExponentialUtility ────────────────────────────────────────────────────────

#[test]
fn test_exponential_utility_zero_theta_error() {
    assert!(ExponentialUtility::new(0.0).is_err());
}

#[test]
fn test_exponential_utility_risk_averse_positive_theta() {
    let u = ExponentialUtility::new(1.0).expect("u");
    // higher cost → lower trajectory weight
    let w1 = u.trajectory_weight(1.0);
    let w2 = u.trajectory_weight(2.0);
    assert!(w1 > w2);
}

#[test]
fn test_exponential_utility_aggregate_finite() {
    let u = ExponentialUtility::new(0.5).expect("u");
    let costs = vec![1.0, 2.0, 0.5, 3.0];
    assert!(u.aggregate(&costs).is_finite());
}

#[test]
fn test_exponential_utility_aggregate_empty() {
    let u = ExponentialUtility::new(0.5).expect("u");
    assert_eq!(u.aggregate(&[]), 0.0);
}

// ── RiskSensitiveMfgSolver ────────────────────────────────────────────────────

#[test]
fn test_risk_sensitive_solver_runs() {
    let cfg = RiskSensitiveMfgConfig {
        n_agents: 10,
        n_steps: 4,
        n_bins: 8,
        max_iters: 2,
        ..Default::default()
    };
    let dyn_ = McKeanVlasovDynamics::new(MckeanVlasovConfig::default());
    let solver = RiskSensitiveMfgSolver::new(cfg, dyn_, default_quadratic_cost()).expect("solver");
    let (mu_seq, costs, _) = solver.solve().expect("solve");
    assert_eq!(mu_seq.len(), 5); // n_steps + 1
    assert_eq!(costs.len(), 2);
}

#[test]
fn test_risk_sensitive_solver_costs_finite() {
    let cfg = RiskSensitiveMfgConfig {
        n_agents: 5,
        n_steps: 3,
        n_bins: 8,
        max_iters: 2,
        seed: 7,
        ..Default::default()
    };
    let dyn_ = McKeanVlasovDynamics::new(MckeanVlasovConfig::default());
    let solver = RiskSensitiveMfgSolver::new(cfg, dyn_, default_quadratic_cost()).expect("solver");
    let (_, costs, _) = solver.solve().expect("solve");
    assert!(costs.iter().all(|c| c.is_finite()));
}

// ── CVaRMfgObjective ──────────────────────────────────────────────────────────

#[test]
fn test_cvar_alpha_out_of_range_error() {
    assert!(CVaRMfgObjective::new(0.0).is_err());
    assert!(CVaRMfgObjective::new(1.0).is_err());
}

#[test]
fn test_cvar_compute_basic() {
    let obj = CVaRMfgObjective::new(0.5).expect("obj");
    let costs = vec![1.0, 2.0, 3.0, 4.0];
    let cvar = obj.compute(&costs).expect("cvar");
    // CVaR_0.5 = mean of top 50% = mean of [3.0, 4.0] = 3.5
    assert!((cvar - 3.5).abs() < 0.5); // approximate due to quantile
}

#[test]
fn test_cvar_empty_costs_error() {
    let obj = CVaRMfgObjective::new(0.5).expect("obj");
    assert!(obj.compute(&[]).is_err());
}

#[test]
fn test_cvar_nash_gap_finite() {
    let obj = CVaRMfgObjective::new(0.5).expect("obj");
    let eq_costs = vec![1.0, 2.0, 1.5];
    let dev_costs = vec![1.2, 2.2, 1.7];
    let gap = obj.nash_gap(&eq_costs, &dev_costs).expect("gap");
    assert!(gap.is_finite());
}

#[test]
fn test_cvar_nash_gap_mismatch_error() {
    let obj = CVaRMfgObjective::new(0.5).expect("obj");
    assert!(obj.nash_gap(&[1.0, 2.0], &[1.0]).is_err());
}

// ── StationaryMfgSolver ───────────────────────────────────────────────────────

#[test]
fn test_stationary_solver_runs() {
    let cfg = StationaryMfgConfig {
        n_bins: 8,
        max_iters: 5,
        ..Default::default()
    };
    let dyn_ = McKeanVlasovDynamics::new(MckeanVlasovConfig::default());
    let solver = StationaryMfgSolver::new(cfg, default_quadratic_cost(), dyn_);
    let (mu, v, lambda, residuals, _) = solver.solve().expect("solve");
    assert_eq!(mu.n_bins(), 8);
    assert_eq!(v.len(), 8);
    assert!(lambda.is_finite());
    assert!(!residuals.is_empty());
}

#[test]
fn test_stationary_solver_distribution_normalised() {
    let cfg = StationaryMfgConfig {
        n_bins: 8,
        max_iters: 3,
        ..Default::default()
    };
    let dyn_ = McKeanVlasovDynamics::new(MckeanVlasovConfig::default());
    let solver = StationaryMfgSolver::new(cfg, default_quadratic_cost(), dyn_);
    let (mu, _, _, _, _) = solver.solve().expect("solve");
    assert!((mu.weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);
}

#[test]
fn test_stationary_solver_value_fn_finite() {
    let cfg = StationaryMfgConfig {
        n_bins: 8,
        max_iters: 3,
        ..Default::default()
    };
    let dyn_ = McKeanVlasovDynamics::new(MckeanVlasovConfig::default());
    let solver = StationaryMfgSolver::new(cfg, default_quadratic_cost(), dyn_);
    let (_, v, _, _, _) = solver.solve().expect("solve");
    assert!(v.iter().all(|x| x.is_finite()));
}

// ── ErgodConstantEstimator ────────────────────────────────────────────────────

#[test]
fn test_ergod_constant_estimator_finite() {
    let est = ErgodConstantEstimator::new(8, -3.0, 3.0, 0.05, 10);
    let v: Vec<f64> = (0..8_usize).map(|i| (i as f64 * 0.1 - 0.35).powi(2)).collect();
    let mu = MfgDistribution::uniform(8, -3.0, 3.0).expect("mu");
    let lambda = est.estimate(&v, &mu, &MfgQuadraticCost::default()).expect("lambda");
    assert!(lambda.is_finite());
}

#[test]
fn test_ergod_constant_estimator_bin_mismatch_error() {
    let est = ErgodConstantEstimator::new(8, -3.0, 3.0, 0.05, 10);
    let v = vec![0.0f64; 4]; // wrong size
    let mu = MfgDistribution::uniform(8, -3.0, 3.0).expect("mu");
    assert!(est.estimate(&v, &mu, &MfgQuadraticCost::default()).is_err());
}

// ── MfgMetricsExtended ────────────────────────────────────────────────────────

#[test]
fn test_metrics_extended_from_residuals() {
    let residuals = vec![0.5, 0.25, 0.1, 0.05];
    let m = MfgMetricsExtended::from_residuals(&residuals).expect("m");
    assert!(m.mean_residual > 0.0);
    assert!(m.convergence_rate < 1.0); // decreasing residuals → rate < 1
    assert!(m.final_wasserstein.is_finite());
}

#[test]
fn test_metrics_extended_from_residuals_empty_error() {
    assert!(MfgMetricsExtended::from_residuals(&[]).is_err());
}

#[test]
fn test_metrics_extended_compute() {
    let mu1 = MfgDistribution::uniform(8, -2.0, 2.0).expect("mu1");
    let mu2 = MfgDistribution::from_weights(
        vec![0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05],
        -2.0,
        2.0,
    )
    .expect("mu2");
    let v: Vec<f64> = (0..8).map(|i| -(i as f64 * 0.5 - 1.75).powi(2)).collect();
    let residuals = vec![0.4, 0.2, 0.08];
    let m = MfgMetricsExtended::compute(
        &residuals,
        &mu1,
        &mu2,
        &MfgQuadraticCost::default(),
        &v,
        -2.0,
        2.0,
    )
    .expect("m");
    assert!(m.final_wasserstein >= 0.0);
    assert!(m.nash_gap >= 0.0);
    assert!(m.mean_residual > 0.0);
}

// ── softmax helper used in earlier test ──────────────────────────────────────

#[test]
fn test_softmax_sums_to_one() {
    let s = softmax_f64(&[1.0, 2.0, 3.0]);
    assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}
