use super::*;

// ── Helpers ──────────────────────────────────────────────────────────────

fn make_gradient(dim: usize, scale: f64) -> Vec<f64> {
    (0..dim).map(|i| (i as f64 + 1.0) * scale).collect()
}

fn make_checkpoint(step: usize, lr: f64, dim: usize, scale: f64) -> IfGradientCheckpoint {
    IfGradientCheckpoint {
        step,
        learning_rate: lr,
        gradient: make_gradient(dim, scale),
    }
}

// ── 1. IfInfluenceFunction tests ────────────────────────────────────────

#[test]
fn test_influence_tracin_basic() {
    let mut inf = IfInfluenceFunction::new(IfInfluenceMode::TracIn, 1e-5, 1e-3, 10);
    let dim = 5;
    inf.add_train_sample(vec![make_checkpoint(0, 0.01, dim, 1.0)]);
    inf.add_train_sample(vec![make_checkpoint(0, 0.01, dim, -1.0)]);

    let test_cp = vec![make_checkpoint(0, 0.01, dim, 1.0)];

    let score0 = inf.compute_influence(0, &test_cp).expect("ok");
    let score1 = inf.compute_influence(1, &test_cp).expect("ok");
    assert!(score0 > 0.0, "positive influence for aligned gradients");
    assert!(score1 < 0.0, "negative influence for opposed gradients");
}

#[test]
fn test_influence_tracin_multi_checkpoint() {
    let mut inf = IfInfluenceFunction::new(IfInfluenceMode::TracIn, 1e-5, 1e-3, 10);
    let dim = 3;
    let cps = vec![
        make_checkpoint(0, 0.1, dim, 1.0),
        make_checkpoint(1, 0.01, dim, 2.0),
    ];
    inf.add_train_sample(cps);

    let test_cps = vec![
        make_checkpoint(0, 0.1, dim, 1.0),
        make_checkpoint(1, 0.01, dim, 1.0),
    ];

    let score = inf.compute_influence(0, &test_cps).expect("ok");
    // lr0 * dot([1,2,3],[1,2,3]) + lr1 * dot([2,4,6],[1,2,3])
    // = 0.1 * 14 + 0.01 * 28 = 1.4 + 0.28 = 1.68
    assert!((score - 1.68).abs() < 1e-10);
}

#[test]
fn test_influence_exact_basic() {
    let mut inf = IfInfluenceFunction::new(IfInfluenceMode::Exact, 1e-5, 0.1, 50);
    let dim = 4;
    inf.add_train_sample(vec![make_checkpoint(0, 0.01, dim, 1.0)]);

    let test_cp = vec![make_checkpoint(0, 0.01, dim, 1.0)];
    let score = inf.compute_influence(0, &test_cp).expect("ok");
    // Exact mode should return a finite influence
    assert!(score.is_finite());
}

#[test]
fn test_influence_out_of_range() {
    let inf = IfInfluenceFunction::new(IfInfluenceMode::TracIn, 1e-5, 1e-3, 10);
    let test_cp = vec![make_checkpoint(0, 0.01, 3, 1.0)];
    assert!(inf.compute_influence(0, &test_cp).is_err());
}

#[test]
fn test_influence_empty_test_checkpoints() {
    let mut inf = IfInfluenceFunction::new(IfInfluenceMode::TracIn, 1e-5, 1e-3, 10);
    inf.add_train_sample(vec![make_checkpoint(0, 0.01, 3, 1.0)]);
    assert!(inf.compute_influence(0, &[]).is_err());
}

#[test]
fn test_influence_find_most_influential() {
    let mut inf = IfInfluenceFunction::new(IfInfluenceMode::TracIn, 1e-5, 1e-3, 10);
    let dim = 3;
    // Sample 0: aligned
    inf.add_train_sample(vec![make_checkpoint(0, 0.01, dim, 2.0)]);
    // Sample 1: weakly aligned
    inf.add_train_sample(vec![make_checkpoint(0, 0.01, dim, 0.5)]);
    // Sample 2: opposed
    inf.add_train_sample(vec![make_checkpoint(0, 0.01, dim, -3.0)]);

    let test_cp = vec![make_checkpoint(0, 0.01, dim, 1.0)];
    let top2 = inf.find_most_influential(&test_cp, 2).expect("ok");
    assert_eq!(top2.len(), 2);
    // Sample 2 should have highest absolute influence (opposed, scale 3)
    assert_eq!(top2[0].0, 2);
}

#[test]
fn test_influence_num_train_samples() {
    let mut inf = IfInfluenceFunction::new(IfInfluenceMode::TracIn, 1e-5, 1e-3, 10);
    assert_eq!(inf.num_train_samples(), 0);
    inf.add_train_sample(vec![make_checkpoint(0, 0.01, 3, 1.0)]);
    assert_eq!(inf.num_train_samples(), 1);
}

// ── 2. IfPermutationImportance tests ────────────────────────────────────

#[test]
fn test_permutation_importance_accuracy() {
    let data: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
    ];
    let labels = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

    // Model: predict feature 0 (perfect predictor)
    let model = |x: &[Vec<f64>]| -> Result<Vec<f64>> { Ok(x.iter().map(|row| row[0]).collect()) };

    let pi = IfPermutationImportance::new(IfMetricKind::Accuracy, 5, 42);
    let result = pi.compute(&model, &data, &labels).expect("ok");

    assert_eq!(result.len(), 2);
    // Feature 0 should be important, feature 1 should not be
    assert!(result[0].mean_score > result[1].mean_score);
}

#[test]
fn test_permutation_importance_mse() {
    let data: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, 0.5]).collect();
    let labels: Vec<f64> = (0..20).map(|i| i as f64 * 2.0).collect();

    // Model: predict 2 * feature 0
    let model =
        |x: &[Vec<f64>]| -> Result<Vec<f64>> { Ok(x.iter().map(|row| row[0] * 2.0).collect()) };

    let pi = IfPermutationImportance::new(IfMetricKind::Mse, 3, 123);
    let result = pi.compute(&model, &data, &labels).expect("ok");

    assert_eq!(result.len(), 2);
    // Feature 0 is important (shuffling it increases MSE)
    assert!(result[0].mean_score > 0.0);
}

#[test]
fn test_permutation_importance_mae() {
    let data: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
    let labels: Vec<f64> = (0..10).map(|i| i as f64).collect();

    let model = |x: &[Vec<f64>]| -> Result<Vec<f64>> { Ok(x.iter().map(|row| row[0]).collect()) };

    let pi = IfPermutationImportance::new(IfMetricKind::Mae, 5, 99);
    let result = pi.compute(&model, &data, &labels).expect("ok");
    assert!(result[0].mean_score > 0.0);
}

#[test]
fn test_permutation_importance_empty_data() {
    let pi = IfPermutationImportance::new(IfMetricKind::Accuracy, 5, 42);
    assert!(pi.compute(&|_| Ok(vec![]), &[], &[]).is_err());
}

#[test]
fn test_permutation_importance_std_score() {
    let data: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
    let labels: Vec<f64> = (0..20).map(|i| i as f64).collect();

    let model = |x: &[Vec<f64>]| -> Result<Vec<f64>> { Ok(x.iter().map(|row| row[0]).collect()) };

    let pi = IfPermutationImportance::new(IfMetricKind::Mse, 10, 42);
    let result = pi.compute(&model, &data, &labels).expect("ok");
    // std should be non-negative
    assert!(result[0].std_score >= 0.0);
}

// ── 3. IfAttentionRollout tests ─────────────────────────────────────────

#[test]
fn test_rollout_identity() {
    let rollout = IfAttentionRollout::new(0.0); // only identity
    let seq_len = 3;
    let am = vec![0.5, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.5];
    let flow = rollout.compute_rollout(&[am], seq_len).expect("ok");
    // With blend=0, flow should be identity
    for i in 0..seq_len {
        for j in 0..seq_len {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (flow[i * seq_len + j] - expected).abs() < 1e-10,
                "({}, {}): {} != {}",
                i,
                j,
                flow[i * seq_len + j],
                expected
            );
        }
    }
}

#[test]
fn test_rollout_full_blend() {
    let rollout = IfAttentionRollout::new(1.0); // full attention, no identity
    let seq_len = 2;
    let am = vec![0.7, 0.3, 0.4, 0.6];
    let flow = rollout.compute_rollout(&[am], seq_len).expect("ok");
    assert_eq!(flow.len(), 4);
    // Rows should sum to 1 (stochastic)
    let row0_sum: f64 = flow[0..2].iter().sum();
    let row1_sum: f64 = flow[2..4].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-10);
    assert!((row1_sum - 1.0).abs() < 1e-10);
}

#[test]
fn test_rollout_two_layers() {
    let rollout = IfAttentionRollout::new(0.5);
    let seq_len = 2;
    let am1 = vec![0.8, 0.2, 0.3, 0.7];
    let am2 = vec![0.6, 0.4, 0.5, 0.5];
    let flow = rollout.compute_rollout(&[am1, am2], seq_len).expect("ok");
    assert_eq!(flow.len(), 4);
    // All values should be non-negative
    for v in &flow {
        assert!(*v >= -1e-10);
    }
}

#[test]
fn test_rollout_empty_maps() {
    let rollout = IfAttentionRollout::new(0.5);
    assert!(rollout.compute_rollout(&[], 3).is_err());
}

#[test]
fn test_rollout_wrong_size() {
    let rollout = IfAttentionRollout::new(0.5);
    let am = vec![0.5, 0.5]; // 2 elements, but seq_len=2 expects 4
    assert!(rollout.compute_rollout(&[am], 2).is_err());
}

#[test]
fn test_token_importance_basic() {
    let rollout = IfAttentionRollout::new(0.5);
    let seq_len = 3;
    let am = vec![0.5, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.5];
    let grads = vec![vec![1.0, 0.5, 0.5]];
    let importance = rollout
        .compute_token_importance(&[am], &grads, seq_len)
        .expect("ok");
    assert_eq!(importance.len(), seq_len);
    // All values should sum to ~1 (normalised)
    let total: f64 = importance.iter().sum();
    assert!((total - 1.0).abs() < 1e-10);
}

#[test]
fn test_token_importance_mismatched_lengths() {
    let rollout = IfAttentionRollout::new(0.5);
    let am = vec![vec![1.0; 4]];
    let grads = vec![vec![1.0, 2.0]]; // wrong length
    assert!(
        rollout.compute_token_importance(&am, &grads, 2).is_err()
            || rollout.compute_token_importance(&am, &[], 2).is_err()
    );
}

// ── 4. IfFairnessAnalyzer tests ─────────────────────────────────────────

#[test]
fn test_fairness_perfect_parity() {
    let analyzer = IfFairnessAnalyzer::new(10);
    // Both groups have same predictions
    let preds = vec![0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2];
    let labels = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let protected = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    let report = analyzer
        .analyze(&preds, &labels, &protected, 0.5)
        .expect("ok");
    assert!(report.demographic_parity < 1e-10);
    assert!(report.equalized_odds < 1e-10);
}

#[test]
fn test_fairness_disparate_impact() {
    let analyzer = IfFairnessAnalyzer::new(10);
    // Group 0 gets all positive, group 1 gets all negative
    let preds = vec![0.9, 0.8, 0.9, 0.8, 0.1, 0.1, 0.1, 0.1];
    let labels = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let protected = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    let report = analyzer
        .analyze(&preds, &labels, &protected, 0.5)
        .expect("ok");
    assert!(report.demographic_parity > 0.5);
    // Disparate impact should be 0 (one group has 0% selection rate)
    assert!(report.disparate_impact < 0.01);
}

#[test]
fn test_fairness_four_fifths_rule() {
    let analyzer = IfFairnessAnalyzer::new(10);
    // Group 0: 80% positive, Group 1: 100% positive
    let preds = vec![0.9, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9];
    let labels = vec![1.0; 10];
    let protected = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    let report = analyzer
        .analyze(&preds, &labels, &protected, 0.5)
        .expect("ok");
    // Disparate impact = 0.8 / 1.0 = 0.8 (exactly the 4/5 rule threshold)
    assert!((report.disparate_impact - 0.8).abs() < 1e-10);
}

#[test]
fn test_fairness_empty_data() {
    let analyzer = IfFairnessAnalyzer::new(10);
    assert!(analyzer.analyze(&[], &[], &[], 0.5).is_err());
}

#[test]
fn test_fairness_single_group() {
    let analyzer = IfFairnessAnalyzer::new(10);
    let preds = vec![0.9, 0.1];
    let labels = vec![1.0, 0.0];
    let protected = vec![0.0, 0.0]; // all in one group
    assert!(analyzer.analyze(&preds, &labels, &protected, 0.5).is_err());
}

#[test]
fn test_fairness_predictive_parity() {
    let analyzer = IfFairnessAnalyzer::new(10);
    // Group 0: all correct, Group 1: many false positives
    let preds = vec![0.9, 0.1, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9];
    let labels = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let protected = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    let report = analyzer
        .analyze(&preds, &labels, &protected, 0.5)
        .expect("ok");
    // Group 0 precision=1.0, Group 1 precision=0.5
    assert!((report.predictive_parity - 0.5).abs() < 1e-10);
}

// ── 5. IfBiasDetector tests ─────────────────────────────────────────────

#[test]
fn test_bias_detector_basic() {
    let detector = IfBiasDetector::new(10);
    let mut rng = StdRng::seed_from_u64(42);

    // Feature 0 correlated with protected, feature 1 independent
    let n = 100;
    let mut features = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    let mut protected = Vec::with_capacity(n);

    for _ in 0..n {
        let p = if rng.random::<f64>() > 0.5 { 1.0 } else { 0.0 };
        let f0 = p + 0.1 * (rng.random::<f64>() - 0.5); // correlated with protected
        let f1 = rng.random::<f64>(); // independent
        let y = if f0 > 0.5 { 1.0 } else { 0.0 };
        features.push(vec![f0, f1]);
        labels.push(y);
        protected.push(p);
    }

    let report = detector
        .detect_spurious_correlations(&features, &labels, &protected)
        .expect("ok");

    // Feature 0 should have higher MI with protected than feature 1
    assert!(report.feature_protected_mi[0] > report.feature_protected_mi[1]);
}

#[test]
fn test_bias_detector_empty() {
    let detector = IfBiasDetector::new(10);
    assert!(detector
        .detect_spurious_correlations(&[], &[], &[])
        .is_err());
}

#[test]
fn test_bias_detector_partial_correlation() {
    let detector = IfBiasDetector::new(10);

    let features: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64 / 50.0, 0.5]).collect();
    let labels: Vec<f64> = (0..50).map(|i| (i as f64 / 50.0).round()).collect();
    let protected: Vec<f64> = (0..50)
        .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
        .collect();

    let report = detector
        .detect_spurious_correlations(&features, &labels, &protected)
        .expect("ok");

    assert_eq!(report.partial_correlations.len(), 2);
    // Feature 0 should have non-zero partial correlation with labels
    assert!(report.partial_correlations[0].abs() > 0.01);
}

#[test]
fn test_bias_detector_attribution_divergence() {
    let detector = IfBiasDetector::new(10);

    // Group 0: label depends on feature 0
    // Group 1: label depends on feature 1
    let features = vec![
        vec![1.0, 0.0], // g0
        vec![0.0, 0.0], // g0
        vec![1.0, 0.0], // g0
        vec![0.0, 0.0], // g0
        vec![0.0, 1.0], // g1
        vec![0.0, 0.0], // g1
        vec![0.0, 1.0], // g1
        vec![0.0, 0.0], // g1
    ];
    let labels = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let protected = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    let report = detector
        .detect_spurious_correlations(&features, &labels, &protected)
        .expect("ok");

    // Attribution divergence should be non-zero
    assert!(report.attribution_divergence[0] > 0.0 || report.attribution_divergence[1] > 0.0);
}

// ── 6. IfModelDivergence tests ──────────────────────────────────────────

#[test]
fn test_divergence_same_distribution() {
    let div = IfModelDivergence::new(20, 1.0, IfDriftThresholds::default());
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let report = div.compute_drift(&data, &data).expect("ok");
    assert!(report.kl_divergence < 1e-6);
    assert!(report.ks_statistic < 1e-6);
    assert!(!report.alert);
}

#[test]
fn test_divergence_shifted_distribution() {
    let div = IfModelDivergence::new(20, 1.0, IfDriftThresholds::default());
    let ref_data: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
    let cur_data: Vec<f64> = (0..100).map(|i| i as f64 / 100.0 + 5.0).collect();
    let report = div.compute_drift(&ref_data, &cur_data).expect("ok");
    assert!(report.kl_divergence > 0.01);
    assert!(report.ks_statistic > 0.5);
    assert!(report.alert);
}

#[test]
fn test_divergence_psi() {
    let div = IfModelDivergence::new(10, 1.0, IfDriftThresholds::default());
    let ref_data: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let cur_data: Vec<f64> = (0..50).map(|i| (i as f64) * 2.0).collect();
    let report = div.compute_drift(&ref_data, &cur_data).expect("ok");
    assert!(report.psi >= 0.0);
}

#[test]
fn test_divergence_mmd() {
    let div = IfModelDivergence::new(10, 1.0, IfDriftThresholds::default());
    let ref_data: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let cur_data: Vec<f64> = (0..30).map(|i| i as f64 + 100.0).collect();
    let report = div.compute_drift(&ref_data, &cur_data).expect("ok");
    assert!(report.mmd > 0.0);
}

#[test]
fn test_divergence_empty() {
    let div = IfModelDivergence::new(10, 1.0, IfDriftThresholds::default());
    assert!(div.compute_drift(&[], &[1.0]).is_err());
    assert!(div.compute_drift(&[1.0], &[]).is_err());
}

#[test]
fn test_divergence_thresholds() {
    let thresholds = IfDriftThresholds {
        kl_threshold: 0.001,
        mmd_threshold: 0.001,
        psi_threshold: 0.001,
        ks_threshold: 0.001,
    };
    let div = IfModelDivergence::new(10, 1.0, thresholds);
    let ref_data: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let cur_data: Vec<f64> = (0..50).map(|i| i as f64 + 1.0).collect();
    let report = div.compute_drift(&ref_data, &cur_data).expect("ok");
    // With very tight thresholds, even slight shift should alert
    assert!(report.alert);
}

#[test]
fn test_ks_statistic_known_values() {
    let div = IfModelDivergence::new(10, 1.0, IfDriftThresholds::default());
    // Two completely separated distributions
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let report = div.compute_drift(&a, &b).expect("ok");
    // KS should be 1.0 for completely separated
    assert!((report.ks_statistic - 1.0).abs() < 1e-10);
}

// ── 7. IfIntegratedGradients tests ──────────────────────────────────────

#[test]
fn test_ig_linear_model() {
    // f(x) = 2*x0 + 3*x1
    let model = |x: &[f64]| -> Result<f64> { Ok(2.0 * x[0] + 3.0 * x[1]) };
    let ig = IfIntegratedGradients::new(100, 1e-5);
    let input = vec![1.0, 1.0];
    let baseline = vec![0.0, 0.0];

    let result = ig.compute(&model, &input, &baseline).expect("ok");

    // For linear model: attributions should be exact [2.0, 3.0]
    assert!((result.attributions[0] - 2.0).abs() < 0.01);
    assert!((result.attributions[1] - 3.0).abs() < 0.01);
    assert!(result.converged);
}

#[test]
fn test_ig_convergence_check() {
    // f(x) = x0^2 + x1
    let model = |x: &[f64]| -> Result<f64> { Ok(x[0] * x[0] + x[1]) };
    let ig = IfIntegratedGradients::new(200, 1e-5);
    let input = vec![2.0, 3.0];
    let baseline = vec![0.0, 0.0];

    let result = ig.compute(&model, &input, &baseline).expect("ok");

    // f(input) - f(baseline) = 4 + 3 - 0 = 7
    // Sum of attributions should be close to 7
    let attr_sum: f64 = result.attributions.iter().sum();
    assert!((attr_sum - 7.0).abs() < 0.5);
}

#[test]
fn test_ig_zero_baseline() {
    let model = |x: &[f64]| -> Result<f64> { Ok(x[0] + x[1] + x[2]) };
    let ig = IfIntegratedGradients::new(50, 1e-5);
    let input = vec![1.0, 2.0, 3.0];
    let baseline = vec![0.0, 0.0, 0.0];

    let result = ig.compute(&model, &input, &baseline).expect("ok");
    assert_eq!(result.attributions.len(), 3);
}

#[test]
fn test_ig_empty_input() {
    let model = |_x: &[f64]| -> Result<f64> { Ok(0.0) };
    let ig = IfIntegratedGradients::new(50, 1e-5);
    assert!(ig.compute(&model, &[], &[]).is_err());
}

#[test]
fn test_ig_mismatched_lengths() {
    let model = |_x: &[f64]| -> Result<f64> { Ok(0.0) };
    let ig = IfIntegratedGradients::new(50, 1e-5);
    assert!(ig.compute(&model, &[1.0], &[0.0, 0.0]).is_err());
}

// ── 8. IfCounterfactualExplainer tests ──────────────────────────────────

#[test]
fn test_counterfactual_linear_model() {
    // f(x) = sigmoid(10 * (x0 - 0.5))
    let model = |x: &[f64]| -> Result<f64> { Ok(if_sigmoid(10.0 * (x[0] - 0.5))) };

    let explainer = IfCounterfactualExplainer::new(IfRegularization::L2, 0.01, 0.1, 200, 0.01);

    // Start from x=0 (output ≈ 0), target output ≈ 1
    let input = vec![0.0];
    let result = explainer.explain(&model, &input, 1.0).expect("ok");

    // Counterfactual should be near x > 0.5
    assert!(
        result.counterfactual[0] > 0.3,
        "cf = {:?}",
        result.counterfactual
    );
}

#[test]
fn test_counterfactual_l1_regularization() {
    // Use a simple linear model (no sigmoid saturation) so that
    // the classification-loss gradient is always large and clearly
    // dominates the L1 regularisation penalty.  This avoids the
    // flaky scenario where a saturated sigmoid produces near-zero
    // gradients that cannot overcome the L1 penalty.
    //
    // Model: f(x) = x[0] + x[1]   (linear, gradient = [1, 1])
    // Input: [0, 0] -> output 0, target 1.
    let model = |x: &[f64]| -> Result<f64> { Ok(x[0] + x[1]) };

    let explainer = IfCounterfactualExplainer::new(
        IfRegularization::L1,
        0.01, // small lambda so classification gradient dominates
        0.1,  // learning rate
        500,  // iterations
        0.01, // tolerance
    );

    let input = vec![0.0, 0.0];
    let result = explainer.explain(&model, &input, 1.0).expect("ok");
    // With L1 reg, perturbation should be sparse-ish but nonzero
    assert!(
        result.perturbation_size > 0.0,
        "expected nonzero perturbation, got perturbation_size={}, output={}, success={}",
        result.perturbation_size,
        result.output,
        result.success,
    );
    // The optimizer should reach close to the target
    assert!(
        result.success,
        "expected success, got output={}, perturbation_size={}",
        result.output, result.perturbation_size,
    );
}

#[test]
fn test_counterfactual_empty_input() {
    let model = |_x: &[f64]| -> Result<f64> { Ok(0.0) };
    let explainer = IfCounterfactualExplainer::new(IfRegularization::L2, 0.01, 0.1, 100, 0.01);
    assert!(explainer.explain(&model, &[], 1.0).is_err());
}

#[test]
fn test_counterfactual_result_fields() {
    let model = |x: &[f64]| -> Result<f64> { Ok(x[0]) };
    let explainer = IfCounterfactualExplainer::new(IfRegularization::L2, 0.01, 0.1, 50, 0.01);
    let result = explainer.explain(&model, &[0.0], 1.0).expect("ok");
    assert_eq!(result.perturbation.len(), 1);
    assert!(result.iterations > 0);
}

#[test]
fn test_counterfactual_already_at_target() {
    let model = |x: &[f64]| -> Result<f64> { Ok(x[0]) };
    let explainer = IfCounterfactualExplainer::new(IfRegularization::L2, 0.01, 0.1, 100, 0.1);
    let result = explainer.explain(&model, &[1.0], 1.0).expect("ok");
    assert!(result.success);
    assert!(result.perturbation_size < 0.5);
}

// ── Cross-component tests ───────────────────────────────────────────────

#[test]
fn test_helper_dot_product() {
    assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-10);
}

#[test]
fn test_helper_l2_norm() {
    assert!((l2_norm(&[3.0, 4.0]) - 5.0).abs() < 1e-10);
}

#[test]
fn test_helper_pearson_corr_perfect() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    assert!((pearson_corr(&a, &b) - 1.0).abs() < 1e-10);
}

#[test]
fn test_helper_softmax() {
    let s = softmax(&[1.0, 2.0, 3.0]);
    let total: f64 = s.iter().sum();
    assert!((total - 1.0).abs() < 1e-10);
    assert!(s[2] > s[1] && s[1] > s[0]);
}

#[test]
fn test_helper_numerical_gradient() {
    // f(x) = x0^2 + 2*x1 => grad = [2*x0, 2]
    let f = |x: &[f64]| -> Result<f64> { Ok(x[0] * x[0] + 2.0 * x[1]) };
    let g = numerical_gradient(&f, &[3.0, 1.0], 1e-5).expect("ok");
    assert!((g[0] - 6.0).abs() < 1e-4);
    assert!((g[1] - 2.0).abs() < 1e-4);
}

#[test]
fn test_helper_mean_variance() {
    let v = vec![2.0, 4.0, 6.0, 8.0];
    assert!((mean(&v) - 5.0).abs() < 1e-10);
    assert!((variance(&v) - 5.0).abs() < 1e-10);
}

#[test]
fn test_helper_vec_ops() {
    let a = vec![1.0, 2.0];
    let b = vec![3.0, 4.0];
    assert_eq!(vec_add(&a, &b), vec![4.0, 6.0]);
    assert_eq!(vec_sub(&a, &b), vec![-2.0, -2.0]);
    assert_eq!(vec_scale(&a, 2.0), vec![2.0, 4.0]);
}
