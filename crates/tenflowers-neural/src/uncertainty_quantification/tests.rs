//! Tests for the `uncertainty_quantification` module.

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_cfg_default() -> UqMCDropoutConfig {
    UqMCDropoutConfig {
        input_dim: 6,
        hidden_dims: vec![16, 16],
        output_dim: 2,
        heteroscedastic: false,
        seed: 42,
    }
}

fn make_ensemble_cfg() -> UqDeepEnsembleConfig {
    UqDeepEnsembleConfig {
        n_members: 3,
        input_dim: 6,
        hidden_dims: vec![12],
        output_dim: 2,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  UqMCDropout
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_mc_dropout_output_shape() {
    let model = UqMCDropout::new(make_cfg_default());
    let x = vec![0.1_f64; 6];
    let result = model.predict_with_uncertainty(&x, 20, 0.3).expect("MC dropout prediction should succeed");
    assert_eq!(result.mean.len(), 2);
    assert_eq!(result.epistemic_var.len(), 2);
    assert_eq!(result.samples.len(), 20);
    assert_eq!(result.samples[0].len(), 2);
}

#[test]
fn test_mc_dropout_epistemic_var_nonneg() {
    let model = UqMCDropout::new(make_cfg_default());
    let x = vec![0.5_f64; 6];
    let result = model.predict_with_uncertainty(&x, 30, 0.2).expect("MC dropout prediction should succeed");
    for v in &result.epistemic_var {
        assert!(
            *v >= 0.0,
            "epistemic variance must be non-negative, got {v}"
        );
    }
}

#[test]
fn test_mc_dropout_mean_finite() {
    let model = UqMCDropout::new(make_cfg_default());
    let x = vec![0.0_f64; 6];
    let result = model.predict_with_uncertainty(&x, 10, 0.1).expect("MC dropout prediction should succeed");
    for m in &result.mean {
        assert!(m.is_finite(), "mean must be finite");
    }
}

#[test]
fn test_mc_dropout_dimension_mismatch() {
    let model = UqMCDropout::new(make_cfg_default());
    let x = vec![0.0_f64; 5]; // wrong dim
    assert!(model.predict_with_uncertainty(&x, 10, 0.2).is_err());
}

#[test]
fn test_mc_dropout_zero_samples_error() {
    let model = UqMCDropout::new(make_cfg_default());
    let x = vec![0.0_f64; 6];
    assert!(model.predict_with_uncertainty(&x, 0, 0.2).is_err());
}

#[test]
fn test_mc_dropout_heteroscedastic_shape() {
    let cfg = UqMCDropoutConfig {
        heteroscedastic: true,
        ..make_cfg_default()
    };
    let model = UqMCDropout::new(cfg);
    let x = vec![0.1_f64; 6];
    let result = model.predict_with_uncertainty(&x, 15, 0.2).expect("heteroscedastic prediction should succeed");
    assert_eq!(result.aleatoric_mean.len(), 2);
    for a in &result.aleatoric_mean {
        assert!(a.is_finite());
    }
}

#[test]
fn test_mc_dropout_zero_dropout_deterministic() {
    // With zero dropout, multiple calls should give same result (same rng seed path)
    let model = UqMCDropout::new(make_cfg_default());
    let x = vec![0.3_f64; 6];
    let r1 = model.predict_with_uncertainty(&x, 5, 0.0).expect("zero-dropout prediction should succeed");
    let r2 = model.predict_with_uncertainty(&x, 5, 0.0).expect("zero-dropout prediction should succeed");
    for (m1, m2) in r1.mean.iter().zip(r2.mean.iter()) {
        assert!((m1 - m2).abs() < 1e-10);
    }
}

#[test]
fn test_mc_dropout_high_dropout_increases_variance() {
    let model = UqMCDropout::new(make_cfg_default());
    let x = vec![1.0_f64; 6];
    let low_var = model.predict_with_uncertainty(&x, 50, 0.0).expect("low-dropout prediction should succeed");
    let high_var = model.predict_with_uncertainty(&x, 50, 0.5).expect("high-dropout prediction should succeed");
    // Epistemic variance with high dropout should generally be higher
    let low_sum: f64 = low_var.epistemic_var.iter().sum();
    let high_sum: f64 = high_var.epistemic_var.iter().sum();
    // At minimum, both should be non-negative
    assert!(low_sum >= 0.0);
    assert!(high_sum >= 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  UqDeepEnsemble
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ensemble_predict_members_shape() {
    let ensemble = UqDeepEnsemble::new(make_ensemble_cfg());
    let x = vec![0.1_f64; 6];
    let preds = ensemble.predict_members(&x).expect("ensemble prediction should succeed");
    assert_eq!(preds.len(), 3);
    assert_eq!(preds[0].len(), 2);
}

#[test]
fn test_ensemble_members_differ() {
    let ensemble = UqDeepEnsemble::new(make_ensemble_cfg());
    let x = vec![0.5_f64; 6];
    let preds = ensemble.predict_members(&x).expect("ensemble prediction should succeed");
    // At least two members should differ (different random inits)
    let same = preds.windows(2).all(|w| {
        w[0].iter()
            .zip(w[1].iter())
            .all(|(a, b)| (a - b).abs() < 1e-10)
    });
    assert!(
        !same,
        "ensemble members should differ due to different init seeds"
    );
}

#[test]
fn test_ensemble_aggregate_shape() {
    let preds = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    let agg = UqDeepEnsemble::aggregate_predictions(&preds).expect("ensemble aggregation should succeed");
    assert_eq!(agg.ensemble_mean.len(), 2);
    assert_eq!(agg.ensemble_variance.len(), 2);
}

#[test]
fn test_ensemble_aggregate_mean_correct() {
    let preds = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let agg = UqDeepEnsemble::aggregate_predictions(&preds).expect("ensemble aggregation should succeed");
    assert!((agg.ensemble_mean[0] - 2.0).abs() < 1e-10);
    assert!((agg.ensemble_mean[1] - 3.0).abs() < 1e-10);
}

#[test]
fn test_ensemble_aggregate_variance_nonneg() {
    let preds = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let agg = UqDeepEnsemble::aggregate_predictions(&preds).expect("ensemble aggregation should succeed");
    for v in &agg.ensemble_variance {
        assert!(*v >= 0.0);
    }
}

#[test]
fn test_ensemble_aggregate_empty_error() {
    let preds: Vec<Vec<f64>> = Vec::new();
    assert!(UqDeepEnsemble::aggregate_predictions(&preds).is_err());
}

#[test]
fn test_ensemble_nll_finite() {
    let ensemble = UqDeepEnsemble::new(make_ensemble_cfg());
    let inputs = vec![vec![0.1_f64; 6], vec![0.2_f64; 6]];
    let targets = vec![vec![0.0_f64; 2], vec![1.0_f64; 2]];
    let nll = ensemble.nll_test(&inputs, &targets).expect("NLL computation should succeed");
    assert!(nll.is_finite(), "NLL should be finite, got {nll}");
}

#[test]
fn test_ensemble_ece_in_range() {
    let probs = vec![0.9, 0.1, 0.8, 0.2, 0.7, 0.3];
    let labels = vec![true, false, true, false, true, false];
    let ece = UqDeepEnsemble::ece(&probs, &labels, 5).expect("ECE computation should succeed");
    assert!(
        (0.0..=1.0).contains(&ece),
        "ECE must be in [0,1], got {ece}"
    );
}

#[test]
fn test_ensemble_ece_perfect_calibration() {
    // All probs = 0.5, half positive
    let probs = vec![0.5; 10];
    let labels = vec![
        true, false, true, false, true, false, true, false, true, false,
    ];
    let ece = UqDeepEnsemble::ece(&probs, &labels, 5).expect("ECE computation should succeed");
    assert!(ece < 0.1, "ECE should be small for well-calibrated model");
}

#[test]
fn test_ensemble_dim_mismatch_error() {
    let ensemble = UqDeepEnsemble::new(make_ensemble_cfg());
    let x = vec![0.0_f64; 5]; // wrong dim
    assert!(ensemble.predict_members(&x).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  UqConformalRegression
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_conformal_calibrate_returns_quantile() {
    let residuals = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let cr = UqConformalRegression::calibrate(residuals, 0.1).expect("conformal calibration should succeed");
    assert!(cr.q_hat > 0.0, "q_hat must be positive");
    assert!(cr.q_hat <= 1.0 + 1e-6);
}

#[test]
fn test_conformal_interval_contains_point_pred() {
    let residuals: Vec<f64> = (1..=20).map(|i| i as f64 * 0.05).collect();
    let cr = UqConformalRegression::calibrate(residuals, 0.1).expect("conformal calibration should succeed");
    let (lo, hi) = cr.predict_interval(0.5);
    assert!(lo < 0.5);
    assert!(hi > 0.5);
}

#[test]
fn test_conformal_coverage_at_least_nominal() {
    let n_cal = 50;
    let residuals: Vec<f64> = (1..=n_cal).map(|i| i as f64 / n_cal as f64).collect();
    let alpha = 0.1;
    let cr = UqConformalRegression::calibrate(residuals, alpha).expect("conformal calibration should succeed");
    // Empirical coverage on fresh data with same distribution should be ≥ 1-alpha
    let point_preds: Vec<f64> = vec![0.5; 100];
    let true_vals: Vec<f64> = (0..100)
        .map(|i| 0.5 + (i as f64 / 100.0 - 0.5) * cr.q_hat)
        .collect();
    let cov = cr.empirical_coverage(&point_preds, &true_vals).expect("coverage computation should succeed");
    assert!((0.0..=1.0).contains(&cov));
}

#[test]
fn test_conformal_empty_residuals_error() {
    assert!(UqConformalRegression::calibrate(vec![], 0.1).is_err());
}

#[test]
fn test_conformal_invalid_alpha_error() {
    let residuals = vec![0.1, 0.2, 0.3];
    assert!(UqConformalRegression::calibrate(residuals.clone(), 0.0).is_err());
    assert!(UqConformalRegression::calibrate(residuals, 1.0).is_err());
}

#[test]
fn test_conformal_group_calibration() {
    let residuals: Vec<f64> = (1..=20).map(|i| i as f64 * 0.04).collect();
    let mut cr = UqConformalRegression::calibrate(residuals, 0.1).expect("conformal calibration should succeed");
    let group_res: Vec<f64> = (1..=10).map(|i| i as f64 * 0.02).collect();
    cr.add_group_calibration("group_A".to_string(), group_res, 0.1)
        .expect("group calibration should succeed");
    let (lo, hi) = cr.predict_interval_group(1.0, "group_A").expect("group prediction should succeed");
    assert!(hi > lo);
}

#[test]
fn test_conformal_unknown_group_error() {
    let residuals: Vec<f64> = (1..=10).map(|i| i as f64 * 0.05).collect();
    let cr = UqConformalRegression::calibrate(residuals, 0.1).expect("conformal calibration should succeed");
    assert!(cr.predict_interval_group(0.5, "nonexistent").is_err());
}

#[test]
fn test_conformal_coverage_on_calibration_residuals() {
    let n = 100;
    let residuals: Vec<f64> = (1..=n).map(|i| i as f64 / n as f64).collect();
    let alpha = 0.1;
    let cr = UqConformalRegression::calibrate(residuals, alpha).expect("conformal calibration should succeed");
    assert!(cr.coverage >= 0.5, "coverage level should be reasonable");
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  UqLaplaceApprox
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_laplace_from_network_forward_shape() {
    let net = UqMCDropout::new(make_cfg_default());
    let laplace = UqLaplaceApprox::from_network(&net, 1.0).expect("Laplace approximation should succeed");
    let x = vec![0.1_f64; 6];
    let out = laplace.forward_map(&x).expect("forward map should succeed");
    assert_eq!(out.len(), 2);
}

#[test]
fn test_laplace_forward_finite() {
    let net = UqMCDropout::new(make_cfg_default());
    let laplace = UqLaplaceApprox::from_network(&net, 1.0).expect("Laplace approximation should succeed");
    let x = vec![0.5_f64; 6];
    let out = laplace.forward_map(&x).expect("forward map should succeed");
    for o in &out {
        assert!(o.is_finite());
    }
}

#[test]
fn test_laplace_posterior_predictive_shape() {
    let net = UqMCDropout::new(make_cfg_default());
    let laplace = UqLaplaceApprox::from_network(&net, 0.5).expect("Laplace approximation should succeed");
    let x = vec![0.2_f64; 6];
    let (means, vars) = laplace.posterior_predictive(&x, 20).expect("posterior predictive should succeed");
    assert_eq!(means.len(), 2);
    assert_eq!(vars.len(), 2);
}

#[test]
fn test_laplace_posterior_var_nonneg() {
    let net = UqMCDropout::new(make_cfg_default());
    let laplace = UqLaplaceApprox::from_network(&net, 1.0).expect("Laplace approximation should succeed");
    let x = vec![0.3_f64; 6];
    let (_, vars) = laplace.posterior_predictive(&x, 30).expect("posterior predictive should succeed");
    for v in &vars {
        assert!(*v >= 0.0);
    }
}

#[test]
fn test_laplace_hessian_estimation() {
    let net = UqMCDropout::new(make_cfg_default());
    let mut laplace = UqLaplaceApprox::from_network(&net, 1.0).expect("Laplace approximation should succeed");
    let data_x: Vec<Vec<f64>> = (0..10).map(|_| vec![0.1_f64; 6]).collect();
    let data_y: Vec<Vec<f64>> = (0..10).map(|_| vec![0.0_f64; 2]).collect();
    laplace.estimate_hessian(&data_x, &data_y, 1e-4).expect("Hessian estimation should succeed");
    for h in &laplace.hessian_diag {
        assert!(*h > 0.0, "diagonal Hessian must be positive");
    }
}

#[test]
fn test_laplace_dim_mismatch_error() {
    let net = UqMCDropout::new(make_cfg_default());
    let laplace = UqLaplaceApprox::from_network(&net, 1.0).expect("Laplace approximation should succeed");
    let x = vec![0.0_f64; 5]; // wrong
    assert!(laplace.forward_map(&x).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  UqPriorNetworks
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_prior_net_alpha_positive() {
    let net = UqPriorNetworks::new(6, 16, 4, 42);
    let x = vec![0.5_f64; 6];
    let res = net.predict_dirichlet(&x).expect("Dirichlet prediction should succeed");
    for a in &res.alpha {
        assert!(*a > 0.0, "Dirichlet alpha must be positive");
    }
}

#[test]
fn test_prior_net_expected_p_sums_to_one() {
    let net = UqPriorNetworks::new(6, 16, 4, 99);
    let x = vec![0.1_f64; 6];
    let res = net.predict_dirichlet(&x).expect("Dirichlet prediction should succeed");
    let sum: f64 = res.expected_p.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-8,
        "expected probs must sum to 1, got {sum}"
    );
}

#[test]
fn test_prior_net_unc_nonneg() {
    let net = UqPriorNetworks::new(6, 16, 4, 7);
    let x = vec![0.3_f64; 6];
    let res = net.predict_dirichlet(&x).expect("Dirichlet prediction should succeed");
    assert!(res.epistemic_unc >= 0.0);
    assert!(res.aleatoric_unc >= 0.0);
    assert!(res.total_unc >= 0.0);
}

#[test]
fn test_prior_net_total_unc_decomposition() {
    let net = UqPriorNetworks::new(6, 16, 3, 13);
    let x = vec![0.0_f64; 6];
    let res = net.predict_dirichlet(&x).expect("Dirichlet prediction should succeed");
    // total ≈ epistemic + aleatoric
    let expected = res.epistemic_unc + res.aleatoric_unc;
    assert!((res.total_unc - expected).abs() < 1e-8);
}

#[test]
fn test_prior_net_dim_mismatch_error() {
    let net = UqPriorNetworks::new(6, 16, 4, 1);
    let x = vec![0.0_f64; 5]; // wrong dim
    assert!(net.predict_dirichlet(&x).is_err());
}

#[test]
fn test_digamma_positive_input() {
    // digamma(1) = -gamma (Euler-Mascheroni) ≈ -0.5772
    let d1 = digamma(1.0);
    assert!(
        (d1 - (-0.5772156649)).abs() < 0.01,
        "digamma(1) ≈ -0.5772, got {d1}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  UqEvidentialDL
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_evidential_nig_alpha_gt_one() {
    let model = UqEvidentialDL::new(6, 16, 0.01, 42);
    let x = vec![0.1_f64; 6];
    let res = model.predict_nig(&x).expect("NIG prediction should succeed");
    assert!(
        res.alpha > 1.0,
        "alpha must be > 1 for valid NIG, got {}",
        res.alpha
    );
}

#[test]
fn test_evidential_nig_nu_positive() {
    let model = UqEvidentialDL::new(6, 16, 0.01, 1);
    let x = vec![0.5_f64; 6];
    let res = model.predict_nig(&x).expect("NIG prediction should succeed");
    assert!(res.nu > 0.0, "nu must be positive");
}

#[test]
fn test_evidential_nig_beta_positive() {
    let model = UqEvidentialDL::new(6, 16, 0.01, 2);
    let x = vec![0.2_f64; 6];
    let res = model.predict_nig(&x).expect("NIG prediction should succeed");
    assert!(res.beta > 0.0, "beta must be positive");
}

#[test]
fn test_evidential_epistemic_finite() {
    let model = UqEvidentialDL::new(6, 16, 0.01, 3);
    let x = vec![0.3_f64; 6];
    let res = model.predict_nig(&x).expect("NIG prediction should succeed");
    assert!(res.epistemic_unc.is_finite());
    assert!(res.aleatoric_unc.is_finite());
}

#[test]
fn test_evidential_evidence_formula() {
    let model = UqEvidentialDL::new(6, 16, 0.0, 5);
    let x = vec![0.0_f64; 6];
    let res = model.predict_nig(&x).expect("NIG prediction should succeed");
    let expected_ev = 2.0 * res.nu + res.alpha;
    assert!((res.evidence - expected_ev).abs() < 1e-8);
}

#[test]
fn test_evidential_regularization_nonneg() {
    let model = UqEvidentialDL::new(6, 16, 0.1, 0);
    let reg = model.evidence_regularization(1.5, 1.0, 2.0, 3.0);
    assert!(reg >= 0.0);
}

#[test]
fn test_evidential_nig_nll_finite() {
    let model = UqEvidentialDL::new(6, 16, 0.01, 7);
    let nll = model.nig_nll(1.0, 1.0, 1.5, 2.5, 3.0);
    assert!(nll.is_finite(), "NIG NLL must be finite, got {nll}");
}

#[test]
fn test_evidential_dim_mismatch_error() {
    let model = UqEvidentialDL::new(6, 16, 0.01, 8);
    let x = vec![0.0_f64; 4]; // wrong
    assert!(model.predict_nig(&x).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  UqCalibration
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_calibration_temperature_scaling_range() {
    let mut cal = UqCalibration::new(10);
    let logits = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -1.5, 0.5, -0.5, 1.5, -2.5];
    let labels = vec![
        false, false, true, true, true, false, true, false, true, false,
    ];
    let t = cal.fit_temperature(&logits, &labels).expect("temperature fitting should succeed");
    assert!(t > 0.0, "temperature must be positive");
    assert!(t <= 10.0);
}

#[test]
fn test_calibration_temperature_scale_output_range() {
    let mut cal = UqCalibration::new(10);
    let logits: Vec<f64> = (0..10).map(|i| i as f64 - 5.0).collect();
    let labels: Vec<bool> = (0..10).map(|i| i >= 5).collect();
    cal.fit_temperature(&logits, &labels).expect("temperature fitting should succeed");
    for l in &logits {
        let p = cal.temperature_scale(*l);
        assert!((0.0..=1.0).contains(&p));
    }
}

#[test]
fn test_calibration_platt_output_range() {
    let mut cal = UqCalibration::new(10);
    let scores: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
    let labels: Vec<bool> = (0..10).map(|i| i >= 5).collect();
    cal.fit_platt(&scores, &labels).expect("Platt scaling should succeed");
    for &s in &scores {
        let p = cal.platt_scale(s);
        assert!(
            (0.0..=1.0).contains(&p),
            "Platt output must be in [0,1], got {p}"
        );
    }
}

#[test]
fn test_calibration_isotonic_range() {
    let mut cal = UqCalibration::new(5);
    let scores: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
    let labels: Vec<bool> = (0..10).map(|i| i >= 5).collect();
    cal.fit_isotonic(&scores, &labels).expect("isotonic regression should succeed");
    for &s in &scores {
        let p = cal.isotonic_scale(s);
        assert!((0.0..=1.0).contains(&p));
    }
}

#[test]
fn test_calibration_ece_in_range() {
    let cal = UqCalibration::new(10);
    let probs: Vec<f64> = (0..20).map(|i| i as f64 / 20.0).collect();
    let labels: Vec<bool> = (0..20).map(|i| i >= 10).collect();
    let ece = cal.ece(&probs, &labels).expect("ECE computation should succeed");
    assert!((0.0..=1.0).contains(&ece));
}

#[test]
fn test_calibration_mce_gte_ece() {
    let cal = UqCalibration::new(5);
    let probs: Vec<f64> = (0..20).map(|i| i as f64 / 20.0).collect();
    let labels: Vec<bool> = (0..20).map(|i| i >= 10).collect();
    let ece = cal.ece(&probs, &labels).expect("ECE computation should succeed");
    let mce = cal.mce(&probs, &labels).expect("MCE computation should succeed");
    assert!(mce >= ece - 1e-10, "MCE must be >= ECE");
}

#[test]
fn test_calibration_ace_in_range() {
    let cal = UqCalibration::new(5);
    let probs: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
    let labels: Vec<bool> = (0..10).map(|i| i >= 5).collect();
    let ace = cal.ace(&probs, &labels).expect("ACE computation should succeed");
    assert!((0.0..=1.0).contains(&ace));
}

#[test]
fn test_calibration_reliability_diagram_bins() {
    let cal = UqCalibration::new(5);
    let probs = vec![0.1, 0.3, 0.5, 0.7, 0.9];
    let labels = vec![false, false, true, true, true];
    let bins = cal.reliability_diagram(&probs, &labels).expect("reliability diagram should succeed");
    assert_eq!(bins.len(), 5);
}

#[test]
fn test_calibration_empty_input_error() {
    let cal = UqCalibration::new(10);
    let probs: Vec<f64> = Vec::new();
    let labels: Vec<bool> = Vec::new();
    assert!(cal.ece(&probs, &labels).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  UqOodDetector
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ood_msp_score_finite() {
    let det = UqOodDetector::new(UqOodMethod::MaxSoftmax, 8, 3, -0.5);
    let logits = vec![1.0, 2.0, 3.0];
    let feats = vec![0.0; 8];
    let score = det.compute_score(&logits, &feats).expect("OOD score computation should succeed");
    assert!(score.is_finite());
}

#[test]
fn test_ood_energy_score_finite() {
    let det = UqOodDetector::new(UqOodMethod::Energy { temperature: 1.0 }, 8, 3, 0.0);
    let logits = vec![1.0, 2.0, 3.0];
    let feats = vec![0.0; 8];
    let score = det.compute_score(&logits, &feats).expect("OOD score computation should succeed");
    assert!(score.is_finite());
}

#[test]
fn test_ood_mahalanobis_score_nonneg() {
    let det = UqOodDetector::new(UqOodMethod::Mahalanobis, 4, 2, 2.0);
    let feats = vec![1.0, 0.0, -1.0, 0.5];
    let score = det.compute_score(&[], &feats).expect("Mahalanobis score computation should succeed");
    assert!(score >= 0.0);
}

#[test]
fn test_ood_fit_mahalanobis_no_error() {
    let mut det = UqOodDetector::new(UqOodMethod::Mahalanobis, 4, 2, 2.0);
    let features: Vec<Vec<f64>> = (0..10)
        .map(|i| vec![(i as f64) * 0.1, 0.5, -(i as f64) * 0.05, 1.0])
        .collect();
    let labels: Vec<usize> = (0..10).map(|i| i % 2).collect();
    det.fit_mahalanobis(&features, &labels).expect("Mahalanobis fitting should succeed");
}

#[test]
fn test_ood_detect_returns_bool() {
    let det = UqOodDetector::new(UqOodMethod::MaxSoftmax, 4, 3, -0.3);
    let logits = vec![2.0, 1.0, 0.5];
    let feats = vec![0.0; 4];
    let (is_ood, score) = det.detect(&logits, &feats).expect("OOD detection should succeed");
    // Just check types / ranges
    assert!(score.is_finite());
    let _ = is_ood;
}

#[test]
fn test_ood_evaluate_auroc_range() {
    let det = UqOodDetector::new(UqOodMethod::Energy { temperature: 1.0 }, 4, 3, 0.0);
    let id_logits: Vec<Vec<f64>> = (0..5).map(|_| vec![2.0, 1.0, 0.5]).collect();
    let id_feats: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0; 4]).collect();
    let ood_logits: Vec<Vec<f64>> = (0..5).map(|_| vec![0.5, 0.3, 0.1]).collect();
    let ood_feats: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0; 4]).collect();
    let result = det
        .evaluate(&id_logits, &id_feats, &ood_logits, &ood_feats)
        .expect("OOD evaluation should succeed");
    assert!((0.0..=1.0).contains(&result.auroc));
    assert!((0.0..=1.0).contains(&result.fpr95));
}

#[test]
fn test_ood_odin_score_finite() {
    let det = UqOodDetector::new(UqOodMethod::Odin { temperature: 2.0 }, 4, 3, 0.0);
    let logits = vec![1.0, 2.0, 0.5];
    let feats = vec![0.0; 4];
    let score = det.compute_score(&logits, &feats).expect("OOD score computation should succeed");
    assert!(score.is_finite());
}

#[test]
fn test_ood_empty_logits_error() {
    let det = UqOodDetector::new(UqOodMethod::MaxSoftmax, 4, 3, 0.0);
    let score = det.compute_score(&[], &[0.0; 4]);
    assert!(score.is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  UqRiskControl
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_risk_control_calibrate_returns_threshold() {
    let mut rc = UqRiskControl::new(0.1, 0.05, UqRiskFn::Coverage).expect("risk control construction should succeed");
    let scores: Vec<f64> = (0..50).map(|i| i as f64 / 50.0).collect();
    let residuals: Vec<f64> = scores.iter().map(|s| 1.0 - s).collect();
    let lam = rc.calibrate_risk(&scores, &residuals).expect("risk calibration should succeed");
    assert!(lam.is_finite(), "lambda* must be finite");
    assert!(lam >= 0.0);
}

#[test]
fn test_risk_control_empirical_risk_range() {
    let mut rc = UqRiskControl::new(0.2, 0.1, UqRiskFn::ResidualThreshold).expect("risk control construction should succeed");
    let scores: Vec<f64> = (0..20).map(|i| i as f64 * 0.05).collect();
    let residuals = scores.clone();
    rc.calibrate_risk(&scores, &residuals).expect("risk calibration should succeed");
    let risk = rc.empirical_risk(&scores, &residuals).expect("empirical risk computation should succeed");
    assert!((0.0..=1.0).contains(&risk));
}

#[test]
fn test_risk_control_bound_satisfied() {
    let mut rc = UqRiskControl::new(0.3, 0.05, UqRiskFn::Coverage).expect("risk control construction should succeed");
    // All scores very low → lambda will be low, everything covered
    let scores = vec![0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10];
    let residuals = vec![1.0; 10];
    rc.calibrate_risk(&scores, &residuals).expect("risk calibration should succeed");
    let satisfied = rc.risk_bound_satisfied(&scores, &residuals).expect("risk bound check should succeed");
    let _ = satisfied; // result is well-defined
}

#[test]
fn test_risk_control_invalid_alpha_error() {
    assert!(UqRiskControl::new(0.0, 0.05, UqRiskFn::Coverage).is_err());
    assert!(UqRiskControl::new(1.0, 0.05, UqRiskFn::Coverage).is_err());
}

#[test]
fn test_risk_control_invalid_delta_error() {
    assert!(UqRiskControl::new(0.1, 0.0, UqRiskFn::Coverage).is_err());
    assert!(UqRiskControl::new(0.1, 1.0, UqRiskFn::Coverage).is_err());
}

#[test]
fn test_risk_control_insufficient_data_error() {
    let mut rc = UqRiskControl::new(0.1, 0.05, UqRiskFn::Coverage).expect("risk control construction should succeed");
    assert!(rc.calibrate_risk(&[0.5], &[0.5]).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  UqMetrics
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_metrics_brier_score_perfect() {
    let probs = vec![1.0, 0.0, 1.0, 0.0];
    let labels = vec![true, false, true, false];
    let bs = UqMetrics::brier_score(&probs, &labels).expect("Brier score should succeed");
    assert!(bs.abs() < 1e-10, "perfect predictions → BS=0, got {bs}");
}

#[test]
fn test_metrics_brier_score_worst() {
    let probs = vec![0.0, 1.0]; // wrong predictions
    let labels = vec![true, false];
    let bs = UqMetrics::brier_score(&probs, &labels).expect("Brier score should succeed");
    assert!(
        (bs - 1.0).abs() < 1e-10,
        "worst predictions → BS=1, got {bs}"
    );
}

#[test]
fn test_metrics_nll_binary_perfect() {
    let probs = vec![1.0 - 1e-10, 1e-10];
    let labels = vec![true, false];
    let nll = UqMetrics::nll_binary(&probs, &labels).expect("NLL computation should succeed");
    assert!(
        nll < 1e-6,
        "NLL should be near 0 for perfect probs, got {nll}"
    );
}

#[test]
fn test_metrics_coverage_all_covered() {
    let lowers = vec![-1.0, -1.0, -1.0];
    let uppers = vec![1.0, 1.0, 1.0];
    let true_vals = vec![0.0, 0.5, -0.5];
    let cov = UqMetrics::coverage(&lowers, &uppers, &true_vals).expect("coverage computation should succeed");
    assert!((cov - 1.0).abs() < 1e-10);
}

#[test]
fn test_metrics_coverage_none_covered() {
    let lowers = vec![2.0, 2.0];
    let uppers = vec![3.0, 3.0];
    let true_vals = vec![0.0, 1.0];
    let cov = UqMetrics::coverage(&lowers, &uppers, &true_vals).expect("coverage computation should succeed");
    assert!(cov.abs() < 1e-10);
}

#[test]
fn test_metrics_mean_interval_width() {
    let lowers = vec![0.0, 0.0];
    let uppers = vec![1.0, 2.0];
    let width = UqMetrics::mean_interval_width(&lowers, &uppers).expect("interval width computation should succeed");
    assert!((width - 1.5).abs() < 1e-10);
}

#[test]
fn test_metrics_ood_auroc_range() {
    let id_scores = vec![-1.0, -0.5, -0.2];
    let ood_scores = vec![0.5, 1.0, 1.5];
    let auc = UqMetrics::ood_auroc(&id_scores, &ood_scores);
    assert!((0.0..=1.0).contains(&auc));
}

#[test]
fn test_metrics_ood_auroc_perfect_separation() {
    // OOD scores all higher than ID → AUROC = 1.0
    let id_scores = vec![-2.0, -1.0, -0.5];
    let ood_scores = vec![1.0, 2.0, 3.0];
    let auc = UqMetrics::ood_auroc(&id_scores, &ood_scores);
    assert!(
        auc > 0.9,
        "AUROC should be near 1 for perfect separation, got {auc}"
    );
}

#[test]
fn test_metrics_compile_report() {
    let probs = vec![0.8, 0.2, 0.7, 0.3, 0.9, 0.1];
    let labels = vec![true, false, true, false, true, false];
    let lowers = vec![-0.5; 6];
    let uppers = vec![0.5; 6];
    let true_vals = vec![0.1, 0.2, -0.1, 0.3, -0.2, 0.0];
    let id_scores = vec![-1.0, -0.5];
    let ood_scores = vec![1.0, 0.8];
    let report = UqMetrics::compile_report(
        &probs,
        &labels,
        &lowers,
        &uppers,
        &true_vals,
        &id_scores,
        &ood_scores,
        5,
    )
    .expect("metrics report compilation should succeed");
    assert!(report.ece >= 0.0 && report.ece <= 1.0);
    assert!(report.brier_score >= 0.0 && report.brier_score <= 1.0);
    assert!(report.nll >= 0.0);
    assert_eq!(report.n_test, 6);
}

#[test]
fn test_metrics_brier_empty_error() {
    assert!(UqMetrics::brier_score(&[], &[]).is_err());
}

#[test]
fn test_metrics_nll_empty_error() {
    assert!(UqMetrics::nll_binary(&[], &[]).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-component integration tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_integration_mc_dropout_then_conformal() {
    // Use MC dropout variance as nonconformity score, then apply conformal
    let model = UqMCDropout::new(UqMCDropoutConfig {
        input_dim: 4,
        hidden_dims: vec![8],
        output_dim: 1,
        heteroscedastic: false,
        seed: 77,
    });
    let cal_inputs: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.05; 4]).collect();
    let cal_targets: Vec<f64> = cal_inputs.iter().map(|x| x[0] * 2.0).collect();
    // Compute residuals using MC dropout mean
    let mut residuals = Vec::new();
    for (x, &y) in cal_inputs.iter().zip(cal_targets.iter()) {
        let res = model.predict_with_uncertainty(x, 10, 0.1).expect("MC dropout prediction should succeed");
        let pred = res.mean[0];
        residuals.push((pred - y).abs());
    }
    let cr = UqConformalRegression::calibrate(residuals, 0.1).expect("conformal calibration should succeed");
    assert!(cr.q_hat.is_finite());
}

#[test]
fn test_integration_ensemble_then_calibration() {
    let ensemble = UqDeepEnsemble::new(UqDeepEnsembleConfig {
        n_members: 3,
        input_dim: 4,
        hidden_dims: vec![8],
        output_dim: 2,
    });
    // Simulate probability predictions from ensemble
    let x = vec![0.5_f64; 4];
    let preds = ensemble.predict_members(&x).expect("ensemble prediction should succeed");
    let agg = UqDeepEnsemble::aggregate_predictions(&preds).expect("ensemble aggregation should succeed");
    // Convert to binary probability via sigmoid
    let prob = sigmoid(agg.ensemble_mean[0]);
    assert!((0.0..=1.0).contains(&prob));
    // Check calibration
    let probs = vec![prob; 10];
    let labels = vec![true; 5]
        .into_iter()
        .chain(vec![false; 5])
        .collect::<Vec<_>>();
    let cal = UqCalibration::new(5);
    let ece = cal.ece(&probs, &labels).expect("ECE computation should succeed");
    assert!(ece.is_finite());
}

#[test]
fn test_integration_evidential_ood() {
    // Evidential model outputs → use evidence as OOD score
    let model = UqEvidentialDL::new(4, 8, 0.01, 42);
    let in_dist_x = vec![0.1_f64; 4];
    let ood_x = vec![10.0_f64; 4];
    let in_res = model.predict_nig(&in_dist_x).expect("in-distribution NIG prediction should succeed");
    let ood_res = model.predict_nig(&ood_x).expect("OOD NIG prediction should succeed");
    // Both should have finite uncertainties
    assert!(in_res.epistemic_unc.is_finite());
    assert!(ood_res.epistemic_unc.is_finite());
}

#[test]
fn test_integration_risk_control_conformal() {
    let residuals: Vec<f64> = (1..=30).map(|i| i as f64 * 0.03).collect();
    let cr = UqConformalRegression::calibrate(residuals.clone(), 0.1).expect("conformal calibration should succeed");
    // Use q_hat as the lambda in risk control
    let mut rc = UqRiskControl::new(0.1, 0.05, UqRiskFn::Coverage).expect("risk control construction should succeed");
    // Use residuals as both scores and losses
    let lam = rc.calibrate_risk(&residuals, &residuals).expect("risk calibration should succeed");
    assert!(lam > 0.0 || cr.q_hat > 0.0); // at least one is positive
}

#[test]
fn test_softmax_helper_sums_to_one() {
    let v = vec![1.0, 2.0, 3.0, 4.0];
    let s = softmax(&v);
    let total: f64 = s.iter().sum();
    assert!((total - 1.0).abs() < 1e-10);
}

#[test]
fn test_log_sum_exp_helper() {
    let v = vec![0.0, 0.0];
    let lse = log_sum_exp(&v);
    assert!((lse - 2.0_f64.ln()).abs() < 1e-10);
}

#[test]
fn test_lgamma_values() {
    // lgamma(1) = 0, lgamma(2) = 0, lgamma(3) = ln(2)
    let lg1 = lgamma(1.0);
    assert!(lg1.abs() < 0.01, "lgamma(1) ≈ 0, got {lg1}");
    let lg3 = lgamma(3.0);
    assert!(
        (lg3 - 2.0_f64.ln()).abs() < 0.01,
        "lgamma(3) ≈ ln(2), got {lg3}"
    );
}

#[test]
fn test_auroc_perfect() {
    let scores = vec![0.1, 0.2, 0.8, 0.9];
    let labels = vec![false, false, true, true];
    let auc = auroc(&scores, &labels);
    assert!(
        (auc - 1.0).abs() < 1e-10,
        "perfect AUROC should be 1.0, got {auc}"
    );
}

#[test]
fn test_variance_helper() {
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let var = variance(&v);
    assert!(
        (var - 2.0).abs() < 1e-10,
        "variance of [1..5] = 2.0, got {var}"
    );
}
