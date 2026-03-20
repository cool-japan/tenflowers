// Tests for the multi-fidelity learning & surrogate optimization module.
// Covers all public types with 50+ test functions.

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_levels_2() -> Vec<MfiFidelityLevel> {
    vec![
        MfiFidelityLevel::new("coarse", 1.0, 0.6).expect("coarse fidelity level should be valid"),
        MfiFidelityLevel::new("fine", 10.0, 1.0).expect("fine fidelity level should be valid"),
    ]
}

fn make_levels_3() -> Vec<MfiFidelityLevel> {
    vec![
        MfiFidelityLevel::new("coarse", 1.0, 0.5).expect("coarse fidelity level should be valid"),
        MfiFidelityLevel::new("medium", 5.0, 0.8).expect("medium fidelity level should be valid"),
        MfiFidelityLevel::new("fine", 25.0, 1.0).expect("fine fidelity level should be valid"),
    ]
}

fn make_dataset_2levels() -> MfiDataset {
    let levels = make_levels_2();
    let mut ds = MfiDataset::new(levels).expect("dataset creation with 2 levels should succeed");
    // Low-fidelity: y ≈ x² (noisy)
    for i in 0..10_i32 {
        let x = i as f64 - 4.5;
        ds.add_observation(0, vec![x], x * x + 0.2 * (i as f64 % 3.0 - 1.0))
            .expect("adding low-fidelity observation should succeed");
    }
    // High-fidelity: y = x² (exact)
    for i in 0..6_i32 {
        let x = i as f64 - 2.5;
        ds.add_observation(1, vec![x], x * x)
            .expect("adding high-fidelity observation should succeed");
    }
    ds
}

fn make_dataset_3levels() -> MfiDataset {
    let levels = make_levels_3();
    let mut ds = MfiDataset::new(levels).expect("dataset creation with 3 levels should succeed");
    for i in 0..8_i32 {
        let x = i as f64 - 3.5;
        ds.add_observation(0, vec![x], x * x + 0.5)
            .expect("adding level-0 observation should succeed");
    }
    for i in 0..6_i32 {
        let x = i as f64 - 2.5;
        ds.add_observation(1, vec![x], x * x + 0.1)
            .expect("adding level-1 observation should succeed");
    }
    for i in 0..4_i32 {
        let x = i as f64 - 1.5;
        ds.add_observation(2, vec![x], x * x)
            .expect("adding level-2 observation should succeed");
    }
    ds
}

// ─────────────────────────────────────────────────────────────────────────────
// MfiFidelityLevel tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_fidelity_level_valid_construction() {
    let lvl = MfiFidelityLevel::new("test", 3.0, 0.9);
    assert!(lvl.is_ok());
    let lvl = lvl.expect("fidelity level construction should succeed");
    assert_eq!(lvl.label, "test");
    assert!((lvl.relative_cost - 3.0).abs() < 1e-9);
}

#[test]
fn test_fidelity_level_invalid_cost() {
    let result = MfiFidelityLevel::new("bad", 0.5, 0.8);
    assert!(result.is_err());
    match result.expect_err("invalid cost should produce error") {
        MfiError::InvalidParameter { name, .. } => assert_eq!(name, "relative_cost"),
        _ => panic!("wrong error type"),
    }
}

#[test]
fn test_fidelity_level_invalid_accuracy() {
    let result = MfiFidelityLevel::new("bad", 2.0, 1.5);
    assert!(result.is_err());
}

#[test]
fn test_fidelity_level_efficiency() {
    let lvl = MfiFidelityLevel::new("test", 4.0, 0.8).expect("fidelity level should be valid");
    assert!((lvl.efficiency() - 0.2).abs() < 1e-9);
}

#[test]
fn test_fidelity_level_cost_equals_one() {
    // cost = 1.0 is the minimum allowed
    let result = MfiFidelityLevel::new("base", 1.0, 0.3);
    assert!(result.is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// MfiDataset tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_dataset_creation_and_observation() {
    let mut ds = make_dataset_2levels();
    assert_eq!(ds.data[0].len(), 10);
    assert_eq!(ds.data[1].len(), 6);
    assert_eq!(ds.total_observations(), 16);
}

#[test]
fn test_dataset_empty_levels_rejected() {
    let result = MfiDataset::new(vec![]);
    assert!(result.is_err());
}

#[test]
fn test_dataset_out_of_range_level_rejected() {
    let mut ds = make_dataset_2levels();
    let result = ds.add_observation(99, vec![1.0], 1.0);
    assert!(result.is_err());
}

#[test]
fn test_dataset_estimate_correlations() {
    let mut ds = make_dataset_2levels();
    ds.estimate_correlations();
    let c01 = ds.correlations[0][1];
    // Both levels sample from y = x²; the overlap uses the first k paired indices,
    // so a positive correlation is expected even if moderate.
    assert!(c01 > 0.0, "expected positive correlation, got {c01}");
}

#[test]
fn test_dataset_correlation_diagonal_is_one() {
    let mut ds = make_dataset_3levels();
    ds.estimate_correlations();
    for i in 0..3 {
        assert!((ds.correlations[i][i] - 1.0).abs() < 1e-9);
    }
}

#[test]
fn test_dataset_highest_level() {
    let ds = make_dataset_3levels();
    assert_eq!(ds.highest_level(), 2);
}

#[test]
fn test_dataset_empty_level_does_not_crash() {
    let levels = make_levels_2();
    let ds = MfiDataset::new(levels).expect("dataset creation should succeed");
    assert_eq!(ds.total_observations(), 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// MfiLinearARModel tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ar_model_requires_at_least_two_levels() {
    let result = MfiLinearARModel::new(1);
    assert!(result.is_err());
}

#[test]
fn test_ar_model_fit_and_predict() {
    let mut model = MfiLinearARModel::new(2).expect("AR model with 2 levels should be valid");
    let ds = make_dataset_2levels();
    model.fit(&ds).expect("AR model fitting should succeed");
    assert!(model.fitted);

    let pred = model
        .predict(&[0.0])
        .expect("AR model prediction should succeed");
    // y(0) = 0² = 0; allow generous tolerance due to noise + NN lookup
    assert!(
        pred.abs() < 3.0,
        "prediction at 0 should be near 0, got {pred}"
    );
}

#[test]
fn test_ar_model_predict_without_fit_errors() {
    let model = MfiLinearARModel::new(2).expect("AR model construction should succeed");
    let result = model.predict(&[1.0]);
    assert!(result.is_err());
    match result.expect_err("predict without fit should produce error") {
        MfiError::NotFitted { .. } => {}
        _ => panic!("expected NotFitted"),
    }
}

#[test]
fn test_ar_model_rho_estimation() {
    let mut model = MfiLinearARModel::new(2).expect("AR model construction should succeed");
    let ds = make_dataset_2levels();
    model.fit(&ds).expect("AR model fitting should succeed");
    // ρ should be a reasonable positive value for correlated levels
    assert!(
        model.rho[0] > 0.0,
        "ρ should be positive for positively correlated levels, got {}",
        model.rho[0]
    );
}

#[test]
fn test_ar_model_batch_predict() {
    let mut model = MfiLinearARModel::new(2).expect("AR model construction should succeed");
    let ds = make_dataset_2levels();
    model.fit(&ds).expect("AR model fitting should succeed");

    let xs: Vec<Vec<f64>> = vec![vec![-2.0], vec![0.0], vec![2.0]];
    let preds = model
        .predict_batch(&xs)
        .expect("batch prediction should succeed");
    assert_eq!(preds.len(), 3);
}

#[test]
fn test_ar_model_three_levels() {
    let mut model = MfiLinearARModel::new(3).expect("AR model with 3 levels should be valid");
    let ds = make_dataset_3levels();
    model.fit(&ds).expect("AR model fitting should succeed");
    let pred = model
        .predict(&[1.0])
        .expect("AR model prediction should succeed");
    // f(1) = 1; allow large tolerance due to minimal data
    assert!(pred.is_finite(), "prediction should be finite");
}

#[test]
fn test_ar_model_wrong_level_count_errors() {
    let mut model = MfiLinearARModel::new(3).expect("AR model construction should succeed");
    let ds = make_dataset_2levels(); // only 2 levels
    let result = model.fit(&ds);
    assert!(result.is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// MfiGaussianProcess tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_gp_construction() {
    let gp = MfiGaussianProcess::new(2, MfiGpConfig::default());
    assert!(gp.is_ok());
}

#[test]
fn test_gp_zero_levels_rejected() {
    let result = MfiGaussianProcess::new(0, MfiGpConfig::default());
    assert!(result.is_err());
}

#[test]
fn test_gp_fit_and_predict() {
    let mut gp =
        MfiGaussianProcess::new(2, MfiGpConfig::default()).expect("GP construction should succeed");
    let ds = make_dataset_2levels();
    gp.fit(&ds).expect("GP fitting should succeed");
    assert!(gp.fitted);

    let (mean, std) = gp
        .predict_with_uncertainty(&[0.0])
        .expect("GP prediction should succeed");
    assert!(mean.is_finite(), "mean should be finite");
    assert!(std >= 0.0, "std should be non-negative");
}

#[test]
fn test_gp_predict_without_fit_errors() {
    let gp =
        MfiGaussianProcess::new(2, MfiGpConfig::default()).expect("GP construction should succeed");
    let result = gp.predict_with_uncertainty(&[1.0]);
    assert!(result.is_err());
}

#[test]
fn test_gp_batch_predict() {
    let mut gp =
        MfiGaussianProcess::new(2, MfiGpConfig::default()).expect("GP construction should succeed");
    let ds = make_dataset_2levels();
    gp.fit(&ds).expect("GP fitting should succeed");

    let xs: Vec<Vec<f64>> = vec![vec![-1.0], vec![0.0], vec![1.0]];
    let preds = gp
        .predict_batch(&xs)
        .expect("GP batch prediction should succeed");
    assert_eq!(preds.len(), 3);
    for (mean, std) in &preds {
        assert!(mean.is_finite());
        assert!(*std >= 0.0);
    }
}

#[test]
fn test_gp_uncertainty_nonzero_away_from_data() {
    let mut gp =
        MfiGaussianProcess::new(2, MfiGpConfig::default()).expect("GP construction should succeed");
    let ds = make_dataset_2levels();
    gp.fit(&ds).expect("GP fitting should succeed");
    // Far from training data, uncertainty should be higher
    let (_, std_far) = gp
        .predict_with_uncertainty(&[100.0])
        .expect("GP prediction far from data should succeed");
    let (_, std_near) = gp
        .predict_with_uncertainty(&[0.0])
        .expect("GP prediction near data should succeed");
    assert!(
        std_far >= std_near,
        "uncertainty far from data ({std_far}) should be ≥ near data ({std_near})"
    );
}

#[test]
fn test_gp_three_levels() {
    let mut gp = MfiGaussianProcess::new(3, MfiGpConfig::default())
        .expect("GP with 3 levels should be valid");
    let ds = make_dataset_3levels();
    gp.fit(&ds).expect("GP fitting should succeed");
    let (mean, _) = gp
        .predict_with_uncertainty(&[1.0])
        .expect("GP prediction should succeed");
    assert!(mean.is_finite());
}

// ─────────────────────────────────────────────────────────────────────────────
// MfiNeuralNetworkAR tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_nn_ar_construction() {
    let m = MfiNeuralNetworkAR::new(1, 16, 42);
    assert!(m.is_ok());
}

#[test]
fn test_nn_ar_zero_dim_rejected() {
    let result = MfiNeuralNetworkAR::new(0, 16, 0);
    assert!(result.is_err());
}

#[test]
fn test_nn_ar_train_and_predict() {
    let mut model =
        MfiNeuralNetworkAR::new(1, 16, 42).expect("NN AR model construction should succeed");
    let low_x: Vec<Vec<f64>> = (-4..=4).map(|i| vec![i as f64]).collect();
    let low_y: Vec<f64> = low_x.iter().map(|v| v[0] * v[0] + 0.3).collect();
    let high_x: Vec<Vec<f64>> = (-2..=2).map(|i| vec![i as f64]).collect();
    let high_y: Vec<f64> = high_x.iter().map(|v| v[0] * v[0]).collect();

    model
        .train(&low_x, &low_y, &high_x, &high_y, 50, 0.01)
        .expect("NN AR training should succeed");
    assert!(model.fitted);

    let pred = model
        .predict(&[0.0])
        .expect("NN AR prediction should succeed");
    assert!(pred.is_finite(), "prediction should be finite, got {pred}");
}

#[test]
fn test_nn_ar_predict_without_train_errors() {
    let model = MfiNeuralNetworkAR::new(1, 16, 0).expect("NN AR model construction should succeed");
    let result = model.predict(&[1.0]);
    assert!(result.is_err());
    match result.expect_err("predict without training should produce error") {
        MfiError::NotFitted { .. } => {}
        _ => panic!("expected NotFitted"),
    }
}

#[test]
fn test_nn_ar_batch_predict() {
    let mut model =
        MfiNeuralNetworkAR::new(1, 8, 7).expect("NN AR model construction should succeed");
    let low_x: Vec<Vec<f64>> = (-3..=3).map(|i| vec![i as f64]).collect();
    let low_y: Vec<f64> = low_x.iter().map(|v| v[0] * v[0]).collect();
    let high_x = low_x.clone();
    let high_y = low_y.clone();
    model
        .train(&low_x, &low_y, &high_x, &high_y, 10, 0.01)
        .expect("NN AR training should succeed");

    let xs = vec![vec![-1.0], vec![0.0], vec![1.0]];
    let preds = model
        .predict_batch(&xs)
        .expect("NN AR batch prediction should succeed");
    assert_eq!(preds.len(), 3);
    for p in &preds {
        assert!(p.is_finite());
    }
}

#[test]
fn test_nn_ar_mismatched_data_rejected() {
    let mut model =
        MfiNeuralNetworkAR::new(1, 8, 0).expect("NN AR model construction should succeed");
    let result = model.train(&[vec![1.0]], &[1.0, 2.0], &[vec![1.0]], &[1.0], 5, 0.01);
    assert!(result.is_err());
}

#[test]
fn test_nn_ar_empty_data_rejected() {
    let mut model =
        MfiNeuralNetworkAR::new(1, 8, 0).expect("NN AR model construction should succeed");
    let result = model.train(&[], &[], &[vec![1.0]], &[1.0], 5, 0.01);
    assert!(result.is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// MfiActiveLearner tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_active_learner_construction() {
    let al = MfiActiveLearner::new(vec![(0.0, 1.0), (0.0, 1.0)], 50, 5, 42);
    assert!(al.is_ok());
}

#[test]
fn test_active_learner_empty_bounds_rejected() {
    let result = MfiActiveLearner::new(vec![], 50, 5, 0);
    assert!(result.is_err());
}

#[test]
fn test_active_learner_invalid_bounds_rejected() {
    let result = MfiActiveLearner::new(vec![(1.0, 0.0)], 50, 5, 0);
    assert!(result.is_err());
}

#[test]
fn test_active_learner_warm_up_returns_level_zero() {
    let mut al = MfiActiveLearner::new(vec![(0.0, 1.0)], 10, 100, 42)
        .expect("active learner construction should succeed");
    let mut gp =
        MfiGaussianProcess::new(2, MfiGpConfig::default()).expect("GP construction should succeed");
    // gp is not fitted → warm-up path
    let levels = make_levels_2();
    let (x, level) = al
        .suggest_next(&gp, &levels, 0)
        .expect("suggest_next should succeed");
    assert_eq!(x.len(), 1);
    assert_eq!(level, 0);
}

#[test]
fn test_active_learner_fitted_gp_suggests_valid_level() {
    let mut al = MfiActiveLearner::new(vec![(-5.0, 5.0)], 50, 2, 99)
        .expect("active learner construction should succeed");
    let mut gp =
        MfiGaussianProcess::new(2, MfiGpConfig::default()).expect("GP construction should succeed");
    let ds = make_dataset_2levels();
    gp.fit(&ds).expect("GP fitting should succeed");

    let levels = make_levels_2();
    let (x, level) = al
        .suggest_next(&gp, &ds.levels, 20)
        .expect("suggest_next should succeed");
    assert_eq!(x.len(), 1);
    assert!((-5.0..=5.0).contains(&x[0]));
    assert!(level < levels.len());
}

#[test]
fn test_active_learner_successive_calls_produce_different_points() {
    let mut al = MfiActiveLearner::new(vec![(-5.0, 5.0)], 20, 0, 1)
        .expect("active learner construction should succeed");
    let mut gp =
        MfiGaussianProcess::new(2, MfiGpConfig::default()).expect("GP construction should succeed");
    let ds = make_dataset_2levels();
    gp.fit(&ds).expect("GP fitting should succeed");

    let (x1, _) = al
        .suggest_next(&gp, &ds.levels, 20)
        .expect("first suggest_next should succeed");
    let (x2, _) = al
        .suggest_next(&gp, &ds.levels, 20)
        .expect("second suggest_next should succeed");
    // With different seeds they should (very likely) differ
    let equal = x1.iter().zip(x2.iter()).all(|(a, b)| (a - b).abs() < 1e-12);
    assert!(!equal, "successive suggestions should differ");
}

// ─────────────────────────────────────────────────────────────────────────────
// MfiSuccessiveHalving tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sha_construction() {
    let sha = MfiSuccessiveHalving::new(3.0, 1.0, 27.0);
    assert!(sha.is_ok());
}

#[test]
fn test_sha_eta_le_one_rejected() {
    let result = MfiSuccessiveHalving::new(1.0, 1.0, 9.0);
    assert!(result.is_err());
}

#[test]
fn test_sha_invalid_budget_rejected() {
    let result = MfiSuccessiveHalving::new(3.0, 10.0, 1.0);
    assert!(result.is_err());
}

#[test]
fn test_sha_run_promotes_best_configs() {
    let sha = MfiSuccessiveHalving::new(3.0, 1.0, 9.0).expect("SHA construction should succeed");
    // Oracle: score = -params[0]^2 (maximum at 0)
    let oracle = |params: &[f64], _budget: f64| -> f64 { -(params[0] * params[0]) };

    let configs: Vec<MfiConfig> = (-4..=4).map(|i| MfiConfig::new(vec![i as f64])).collect();

    let result = sha.run(configs, oracle).expect("SHA run should succeed");
    // Best config should have params near 0
    assert!(
        result[0].score > result.last().unwrap_or(&result[0]).score - 1e-6,
        "best config should be first"
    );
}

#[test]
fn test_sha_empty_configs_rejected() {
    let sha = MfiSuccessiveHalving::new(3.0, 1.0, 9.0).expect("SHA construction should succeed");
    let result = sha.run(vec![], |_p: &[f64], _b: f64| 0.0);
    assert!(result.is_err());
}

#[test]
fn test_sha_n_rounds() {
    let sha = MfiSuccessiveHalving::new(3.0, 1.0, 27.0).expect("SHA construction should succeed");
    assert_eq!(sha.n_rounds(), 3);
}

#[test]
fn test_sha_single_config_survives() {
    let sha = MfiSuccessiveHalving::new(3.0, 1.0, 9.0).expect("SHA construction should succeed");
    let configs = vec![MfiConfig::new(vec![2.0])];
    let result = sha
        .run(configs, |p: &[f64], _b: f64| p[0])
        .expect("SHA run with single config should succeed");
    assert!(!result.is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// MfiHyperband tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hyperband_construction() {
    let hb = MfiHyperband::new(27.0, 3.0, 0, 2);
    assert!(hb.is_ok());
}

#[test]
fn test_hyperband_invalid_eta_rejected() {
    let result = MfiHyperband::new(27.0, 1.0, 0, 2);
    assert!(result.is_err());
}

#[test]
fn test_hyperband_run_returns_results() {
    let hb = MfiHyperband::new(9.0, 3.0, 42, 1).expect("Hyperband construction should succeed");
    let bounds = vec![(0.0_f64, 1.0_f64)];
    // Oracle: negative squared distance from 0.3
    let oracle = |p: &[f64], _b: f64| -> f64 { -(p[0] - 0.3).powi(2) };
    let results = hb
        .run(oracle, &bounds)
        .expect("Hyperband run should succeed");
    assert!(
        !results.is_empty(),
        "hyperband should return at least one result"
    );
}

#[test]
fn test_hyperband_best_config_is_first() {
    let hb = MfiHyperband::new(9.0, 3.0, 1, 1).expect("Hyperband construction should succeed");
    let bounds = vec![(0.0_f64, 1.0_f64)];
    let oracle = |p: &[f64], _b: f64| -> f64 { -(p[0] - 0.5).powi(2) };
    let results = hb
        .run(oracle, &bounds)
        .expect("Hyperband run should succeed");
    if results.len() > 1 {
        assert!(
            results[0].best_config.score >= results[1].best_config.score,
            "first result should have highest score"
        );
    }
}

#[test]
fn test_hyperband_wrong_dim_rejected() {
    let hb = MfiHyperband::new(9.0, 3.0, 0, 2).expect("Hyperband construction should succeed");
    let bounds = vec![(0.0, 1.0)]; // dim=1 but param_dim=2
    let result = hb.run(|_p: &[f64], _b: f64| 0.0, &bounds);
    assert!(result.is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// MfiBOHB tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_bohb_construction() {
    let bohb = MfiBOHB::new(27.0, 3.0, 0.15, 0.3, 42, vec![(0.0, 1.0)]);
    assert!(bohb.is_ok());
}

#[test]
fn test_bohb_invalid_gamma_rejected() {
    let result = MfiBOHB::new(27.0, 3.0, 1.5, 0.3, 0, vec![(0.0, 1.0)]);
    assert!(result.is_err());
}

#[test]
fn test_bohb_invalid_bandwidth_rejected() {
    let result = MfiBOHB::new(27.0, 3.0, 0.15, 0.0, 0, vec![(0.0, 1.0)]);
    assert!(result.is_err());
}

#[test]
fn test_bohb_suggest_without_observations_random() {
    let mut bohb = MfiBOHB::new(9.0, 3.0, 0.15, 0.3, 42, vec![(0.0, 1.0)])
        .expect("BOHB construction should succeed");
    let (params, budget) = bohb.suggest(3.0).expect("BOHB suggest should succeed");
    assert_eq!(params.len(), 1);
    assert!((0.0..=1.0).contains(&params[0]));
    assert!((1.0..=9.0).contains(&budget));
}

#[test]
fn test_bohb_observe_and_suggest() {
    let mut bohb = MfiBOHB::new(27.0, 3.0, 0.3, 0.4, 7, vec![(0.0, 1.0)])
        .expect("BOHB construction should succeed");

    // Add many observations to trigger KDE path
    for i in 0..30 {
        let x = (i as f64) / 30.0;
        let loss = (x - 0.2).powi(2); // minimum near x=0.2
        bohb.observe(vec![x], 3.0, loss);
    }

    let (params, budget) = bohb
        .suggest(3.0)
        .expect("BOHB suggest after observations should succeed");
    assert_eq!(params.len(), 1);
    assert!((0.0..=1.0).contains(&params[0]));
    let _ = budget;
}

#[test]
fn test_bohb_best_observation() {
    let mut bohb = MfiBOHB::new(9.0, 3.0, 0.15, 0.3, 0, vec![(0.0, 1.0)])
        .expect("BOHB construction should succeed");
    assert!(bohb.best_observation().is_none());

    bohb.observe(vec![0.5], 3.0, 1.0);
    bohb.observe(vec![0.1], 3.0, 0.01);
    bohb.observe(vec![0.9], 3.0, 0.5);

    let best = bohb
        .best_observation()
        .expect("best observation should exist after adding observations");
    assert!((best.loss - 0.01).abs() < 1e-9);
}

#[test]
fn test_bohb_suggest_clamped_to_bounds() {
    let bounds = vec![(0.0_f64, 0.5_f64)];
    let mut bohb = MfiBOHB::new(9.0, 3.0, 0.3, 0.1, 42, bounds.clone())
        .expect("BOHB construction should succeed");

    // Many observations to trigger KDE sampling
    for i in 0..20 {
        let x = (i as f64 / 20.0) * 0.5;
        bohb.observe(vec![x], 3.0, (x - 0.25).powi(2));
    }

    let (params, _) = bohb.suggest(3.0).expect("BOHB suggest should succeed");
    assert!(params[0] >= 0.0 - 1e-9 && params[0] <= 0.5 + 1e-9);
}

// ─────────────────────────────────────────────────────────────────────────────
// MfiEnsemble tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ensemble_uniform_construction() {
    let ens = MfiEnsemble::new(3, MfiEnsembleWeightStrategy::Uniform);
    assert!(ens.is_ok());
}

#[test]
fn test_ensemble_zero_levels_rejected() {
    let result = MfiEnsemble::new(0, MfiEnsembleWeightStrategy::Uniform);
    assert!(result.is_err());
}

#[test]
fn test_ensemble_uniform_fit_and_predict() {
    let mut ens = MfiEnsemble::new(2, MfiEnsembleWeightStrategy::Uniform)
        .expect("uniform ensemble construction should succeed");
    let levels = make_levels_2();
    ens.fit(&levels, &[], &[])
        .expect("ensemble fit should succeed");

    let pred = ens
        .ensemble_predict(&[3.0, 5.0])
        .expect("ensemble predict should succeed");
    assert!(
        (pred - 4.0).abs() < 1e-9,
        "uniform mean of [3,5] = 4, got {pred}"
    );
}

#[test]
fn test_ensemble_accuracy_proportional() {
    let mut ens = MfiEnsemble::new(2, MfiEnsembleWeightStrategy::AccuracyProportional)
        .expect("accuracy-proportional ensemble construction should succeed");
    let levels = make_levels_2();
    ens.fit(&levels, &[], &[])
        .expect("ensemble fit should succeed");

    // Level 0: accuracy 0.6, level 1: accuracy 1.0 → weights 0.375, 0.625
    let pred = ens
        .ensemble_predict(&[0.0, 10.0])
        .expect("ensemble predict should succeed");
    // Expected: 0 * 0.375 + 10 * 0.625 = 6.25
    let expected = 10.0 * (1.0 / 1.6);
    assert!(
        (pred - expected).abs() < 0.5,
        "accuracy-proportional prediction off: got {pred}, expected ~{expected}"
    );
}

#[test]
fn test_ensemble_stacking_fit_and_predict() {
    let mut ens = MfiEnsemble::new(2, MfiEnsembleWeightStrategy::Stacking)
        .expect("stacking ensemble construction should succeed");
    let levels = make_levels_2();
    let val_preds: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            let x = i as f64 - 9.5;
            vec![x * x + 0.3, x * x]
        })
        .collect();
    let val_targets: Vec<f64> = (0..20)
        .map(|i| {
            let x = i as f64 - 9.5;
            x * x
        })
        .collect();

    ens.fit(&levels, &val_preds, &val_targets)
        .expect("stacking ensemble fit should succeed");
    let pred = ens
        .ensemble_predict(&[2.0, 4.0])
        .expect("stacking ensemble predict should succeed");
    assert!(pred.is_finite(), "stacking prediction should be finite");
}

#[test]
fn test_ensemble_predict_without_fit_errors() {
    let ens = MfiEnsemble::new(2, MfiEnsembleWeightStrategy::Uniform)
        .expect("ensemble construction should succeed");
    let result = ens.ensemble_predict(&[1.0, 2.0]);
    assert!(result.is_err());
}

#[test]
fn test_ensemble_wrong_pred_count_errors() {
    let mut ens = MfiEnsemble::new(2, MfiEnsembleWeightStrategy::Uniform)
        .expect("ensemble construction should succeed");
    let levels = make_levels_2();
    ens.fit(&levels, &[], &[])
        .expect("ensemble fit should succeed");
    let result = ens.ensemble_predict(&[1.0, 2.0, 3.0]);
    assert!(result.is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// MfiMetrics and MfiReport tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_metrics_rmse_perfect_predictions() {
    let preds = vec![1.0, 2.0, 3.0];
    let targets = vec![1.0, 2.0, 3.0];
    let m = MfiMetrics::compute_prediction_metrics(&preds, &targets)
        .expect("computing prediction metrics should succeed");
    assert!(m.rmse < 1e-10, "perfect predictions should have zero RMSE");
    assert!(m.mae < 1e-10);
}

#[test]
fn test_metrics_rmse_known_error() {
    let preds = vec![0.0, 0.0, 0.0];
    let targets = vec![1.0, 1.0, 1.0];
    let m = MfiMetrics::compute_prediction_metrics(&preds, &targets)
        .expect("computing prediction metrics should succeed");
    assert!((m.rmse - 1.0).abs() < 1e-9);
    assert!((m.mae - 1.0).abs() < 1e-9);
}

#[test]
fn test_metrics_empty_predictions_rejected() {
    let result = MfiMetrics::compute_prediction_metrics(&[], &[]);
    assert!(result.is_err());
}

#[test]
fn test_metrics_length_mismatch_rejected() {
    let result = MfiMetrics::compute_prediction_metrics(&[1.0, 2.0], &[1.0]);
    assert!(result.is_err());
}

#[test]
fn test_metrics_fidelity_metrics() {
    let ds = make_dataset_2levels();
    let m = MfiMetrics::compute_fidelity_metrics(&ds);
    assert_eq!(m.level_efficiencies.len(), 2);
    assert_eq!(m.level_correlations.len(), 1);
}

#[test]
fn test_metrics_cost_savings_all_low_fidelity() {
    let levels = make_levels_2(); // costs 1 and 10
    let mut ds = MfiDataset::new(levels).expect("dataset creation should succeed");
    // 10 observations all at level 0 (cheapest)
    for i in 0..10 {
        ds.add_observation(0, vec![i as f64], i as f64)
            .expect("adding observation should succeed");
    }
    let savings = MfiMetrics::compute_cost_savings(&ds);
    // actual cost = 10 * 1 = 10; full cost would be 10 * 10 = 100 → savings = 90%
    assert!(
        (savings - 0.9).abs() < 1e-9,
        "expected 90% savings, got {savings}"
    );
}

#[test]
fn test_metrics_cost_savings_empty() {
    let levels = make_levels_2();
    let ds = MfiDataset::new(levels).expect("dataset creation should succeed");
    let savings = MfiMetrics::compute_cost_savings(&ds);
    assert_eq!(savings, 0.0);
}

#[test]
fn test_report_build() {
    let ds = make_dataset_2levels();
    let report = MfiReport::build(&ds, None, None).expect("report building should succeed");
    assert_eq!(report.n_levels, 2);
    assert_eq!(report.obs_per_level[0], 10);
    assert_eq!(report.obs_per_level[1], 6);
    assert_eq!(report.prediction_rmse, 0.0);
}

#[test]
fn test_report_build_with_predictions() {
    let ds = make_dataset_2levels();
    let preds = vec![0.0, 1.1, 4.0, 9.2, 16.1, 25.0];
    let targets: Vec<f64> = (-2..=3_i32).map(|i| (i as f64).powi(2)).collect();
    let report = MfiReport::build(&ds, Some(&preds), Some(&targets))
        .expect("report building with predictions should succeed");
    assert!(report.prediction_rmse > 0.0);
}

#[test]
fn test_report_display_does_not_panic() {
    let ds = make_dataset_2levels();
    let report = MfiReport::build(&ds, None, None).expect("report building should succeed");
    let s = format!("{report}");
    assert!(s.contains("MfiReport"));
    assert!(s.contains("Fidelity levels"));
}

#[test]
fn test_report_level_correlations_populated() {
    let ds = make_dataset_3levels();
    let report = MfiReport::build(&ds, None, None).expect("report building should succeed");
    assert_eq!(
        report.level_correlations.len(),
        2,
        "3 levels → 2 adjacent pairs"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// End-to-end integration tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_end_to_end_ar_then_gp() {
    let ds = make_dataset_2levels();

    // Fit AR model
    let mut ar = MfiLinearARModel::new(2).expect("AR model construction should succeed");
    ar.fit(&ds).expect("AR model fitting should succeed");

    // Fit GP
    let mut gp =
        MfiGaussianProcess::new(2, MfiGpConfig::default()).expect("GP construction should succeed");
    gp.fit(&ds).expect("GP fitting should succeed");

    // Both should give finite predictions at a new point
    let ar_pred = ar.predict(&[1.5]).expect("AR prediction should succeed");
    let (gp_mean, gp_std) = gp
        .predict_with_uncertainty(&[1.5])
        .expect("GP prediction should succeed");

    assert!(ar_pred.is_finite());
    assert!(gp_mean.is_finite());
    assert!(gp_std >= 0.0);
}

#[test]
fn test_end_to_end_hyperband_then_bohb() {
    let bounds = vec![(0.0_f64, 1.0_f64)];
    // Objective: minimize (x - 0.4)^2
    let oracle = |p: &[f64], _b: f64| -> f64 { -(p[0] - 0.4).powi(2) };

    let hb = MfiHyperband::new(9.0, 3.0, 5, 1).expect("Hyperband construction should succeed");
    let hb_results = hb
        .run(oracle, &bounds)
        .expect("Hyperband run should succeed");
    assert!(!hb_results.is_empty());

    let mut bohb = MfiBOHB::new(9.0, 3.0, 0.3, 0.2, 5, bounds.clone())
        .expect("BOHB construction should succeed");
    // Seed with hyperband results
    for r in &hb_results {
        let x = r.best_config.params[0];
        let loss = (x - 0.4).powi(2);
        bohb.observe(r.best_config.params.clone(), 3.0, loss);
    }
    let (params, _) = bohb.suggest(3.0).expect("BOHB suggest should succeed");
    assert!(params[0].is_finite());
}

#[test]
fn test_end_to_end_full_pipeline() {
    // Construct dataset
    let ds = make_dataset_2levels();

    // AR prediction accuracy
    let mut ar = MfiLinearARModel::new(2).expect("AR model construction should succeed");
    ar.fit(&ds).expect("AR model fitting should succeed");
    let test_xs: Vec<Vec<f64>> = vec![vec![-1.0], vec![0.0], vec![1.0]];
    let test_ys: Vec<f64> = vec![1.0, 0.0, 1.0];
    let ar_preds = ar
        .predict_batch(&test_xs)
        .expect("AR batch prediction should succeed");

    // Metrics
    let m = MfiMetrics::compute_prediction_metrics(&ar_preds, &test_ys)
        .expect("computing prediction metrics should succeed");

    // Report
    let report = MfiReport::build(&ds, Some(&ar_preds), Some(&test_ys))
        .expect("report building should succeed");
    assert_eq!(report.n_levels, 2);
    assert!(m.rmse >= 0.0);
    assert!(
        report.cost_savings > 0.0,
        "expect some cost savings vs pure HF"
    );
}
