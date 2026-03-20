use super::*;
use scirs2_core::random::SeedableRng;

// ── TtaLinear ──────────────────────────────────────────────────────────

#[test]
fn test_tta_linear_output_shape() {
    let layer = TtaLinear::new(4, 3);
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let out = layer.forward(&x);
    assert_eq!(out.len(), 3, "output should have 3 elements");
}

#[test]
fn test_tta_linear_output_finite() {
    let layer = TtaLinear::new(8, 5);
    let x: Vec<f64> = (0..8).map(|i| i as f64 * 0.1).collect();
    let out = layer.forward(&x);
    for &v in &out {
        assert!(v.is_finite(), "output must be finite");
    }
}

#[test]
fn test_tta_linear_update_params() {
    let mut layer = TtaLinear::new(3, 2);
    let w_before: Vec<Vec<f64>> = layer.w.to_vec();
    let grad_w = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
    let grad_b = vec![0.01, 0.02];
    layer.update_params(&grad_w, &grad_b, 0.1);
    // Weights should change.
    let changed = layer
        .w
        .iter()
        .zip(w_before.iter())
        .any(|(r, rb)| r.iter().zip(rb.iter()).any(|(a, b)| (a - b).abs() > 1e-12));
    assert!(changed, "weights should change after update");
}

// ── TtaMlp ────────────────────────────────────────────────────────────

#[test]
fn test_tta_mlp_forward_finite() {
    let mlp = TtaMlp::new(&[4, 8, 3]);
    let x = vec![0.5, -0.3, 1.2, -0.8];
    let out = mlp.forward(&x);
    assert_eq!(out.len(), 3);
    for &v in &out {
        assert!(v.is_finite());
    }
}

#[test]
fn test_tta_mlp_params_flat_length() {
    let mlp = TtaMlp::new(&[4, 8, 3]);
    // Layer 0: 4*8 + 8 = 40; Layer 1: 8*3 + 3 = 27 → total = 67.
    let flat = mlp.params_flat();
    assert_eq!(flat.len(), 67, "flat param count mismatch");
}

#[test]
fn test_tta_mlp_update_with_flat_grad_changes_params() {
    let mut mlp = TtaMlp::new(&[4, 8, 3]);
    let flat_before = mlp.params_flat();
    let n = flat_before.len();
    let grad = vec![1.0f64; n];
    mlp.update_with_flat_grad(&grad, 0.01);
    let flat_after = mlp.params_flat();
    let changed = flat_before
        .iter()
        .zip(flat_after.iter())
        .any(|(a, b)| (a - b).abs() > 1e-12);
    assert!(changed, "params should change after gradient update");
}

#[test]
fn test_tta_mlp_params_flat_roundtrip() {
    let mut mlp = TtaMlp::new(&[2, 4, 2]);
    let flat = mlp.params_flat();
    let n = flat.len();
    // Apply zero gradient — params should stay the same.
    mlp.update_with_flat_grad(&vec![0.0f64; n], 1.0);
    let flat_after = mlp.params_flat();
    for (a, b) in flat.iter().zip(flat_after.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

// ── TtaBatchNormStats ─────────────────────────────────────────────────────

#[test]
fn test_bn_forward_train_output_shape() {
    let mut bn = TtaBatchNormStats::new(4);
    let batch = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 3.0, 4.0, 5.0],
        vec![3.0, 4.0, 5.0, 6.0],
    ];
    let out = bn.forward_train(&batch);
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].len(), 4);
}

#[test]
fn test_bn_forward_train_normalises() {
    let mut bn = TtaBatchNormStats::new(2);
    // All same value → var=0 → output should be beta (0) + gamma*0 = 0.
    let batch = vec![vec![5.0, 5.0], vec![5.0, 5.0], vec![5.0, 5.0]];
    let out = bn.forward_train(&batch);
    for row in &out {
        for &v in row {
            assert!(
                v.abs() < 1e-4,
                "normalised constant batch should be ~0, got {}",
                v
            );
        }
    }
}

#[test]
fn test_bn_forward_eval_shape() {
    let bn = TtaBatchNormStats::new(3);
    let x = vec![1.0, 2.0, 3.0];
    let out = bn.forward_eval(&x);
    assert_eq!(out.len(), 3);
}

#[test]
fn test_bn_forward_eval_uses_running_stats() {
    let bn = TtaBatchNormStats::new(2);
    // running_mean=0, running_var=1, gamma=1, beta=0 → output = x.
    let x = vec![3.0, 7.0];
    let out = bn.forward_eval(&x);
    // x[j] / sqrt(1 + 1e-5) ≈ x[j].
    for (&o, &xi) in out.iter().zip(x.iter()) {
        assert!(
            (o - xi).abs() < 0.01,
            "eval output should match input for default stats"
        );
    }
}

#[test]
fn test_bn_entropy_gradient_shape() {
    let bn = TtaBatchNormStats::new(4);
    let logits = vec![1.0, 0.5, -0.5, 0.0];
    let (gg, gb) = bn.entropy_gradient(&logits);
    assert_eq!(gg.len(), 4);
    assert_eq!(gb.len(), 4);
}

#[test]
fn test_bn_update_affine_changes_params() {
    let mut bn = TtaBatchNormStats::new(3);
    let g_before = bn.gamma.clone();
    let grad_g = vec![0.1, 0.2, 0.3];
    let grad_b = vec![0.01, 0.02, 0.03];
    bn.update_affine(&grad_g, &grad_b, 0.1);
    let changed = bn
        .gamma
        .iter()
        .zip(g_before.iter())
        .any(|(a, b)| (a - b).abs() > 1e-12);
    assert!(changed);
}

// ── TentModel ─────────────────────────────────────────────────────────

#[test]
fn test_tent_predict_sums_to_one() {
    let model = TentModel::new(&[4, 8, 3], TentConfig::default());
    let x = vec![0.1, 0.2, 0.3, 0.4];
    let p = model.predict(&x);
    assert_eq!(p.len(), 3);
    let sum: f64 = p.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "softmax should sum to 1, got {}",
        sum
    );
}

#[test]
fn test_tent_entropy_non_negative() {
    let model = TentModel::new(&[4, 8, 3], TentConfig::default());
    let x = vec![1.0, -1.0, 0.5, -0.5];
    let h = model.entropy(&x);
    assert!(h >= 0.0, "entropy must be non-negative, got {}", h);
}

#[test]
fn test_tent_adapt_runs() {
    let mut model = TentModel::new(
        &[4, 8, 3],
        TentConfig {
            lr: 1e-3,
            n_steps: 2,
            reset_after: false,
        },
    );
    let batch: Vec<Vec<f64>> = (0..5)
        .map(|i| vec![i as f64 * 0.1, 0.2, 0.3, 0.4])
        .collect();
    model.adapt(&batch); // should not panic
}

#[test]
fn test_tent_adapt_and_predict_shape() {
    let mut model = TentModel::new(&[4, 8, 3], TentConfig::default());
    let x = vec![0.1, 0.2, 0.3, 0.4];
    let p = model.adapt_and_predict(&x);
    assert_eq!(p.len(), 3);
    let sum: f64 = p.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_tent_adapt_reset_after() {
    let mut model = TentModel::new(
        &[4, 8, 3],
        TentConfig {
            lr: 1e-2,
            n_steps: 3,
            reset_after: true,
        },
    );
    let orig_gamma: Vec<Vec<f64>> = model.bn_stats.iter().map(|bn| bn.gamma.clone()).collect();
    let batch = vec![vec![1.0, 2.0, 3.0, 4.0]];
    model.adapt(&batch);
    // After reset, gamma should be restored.
    for (bn, og) in model.bn_stats.iter().zip(orig_gamma.iter()) {
        for (&g, &o) in bn.gamma.iter().zip(og.iter()) {
            assert!(
                (g - o).abs() < 1e-12,
                "gamma should be restored after reset"
            );
        }
    }
}

// ── TttModel ──────────────────────────────────────────────────────────

#[test]
fn test_ttt_rotate_r0_is_identity() {
    let x: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let rotated = TttModel::rotate_features(&x, &RotationLabel::R0);
    assert_eq!(x, rotated, "R0 should be identity");
}

#[test]
fn test_ttt_rotate_r180_is_reversal() {
    let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let rotated = TttModel::rotate_features(&x, &RotationLabel::R180);
    let expected: Vec<f64> = x.iter().cloned().rev().collect();
    assert_eq!(rotated, expected, "R180 should be full reversal");
}

#[test]
fn test_ttt_rotate_r90_different_from_identity() {
    let x: Vec<f64> = (1..=8).map(|i| i as f64).collect();
    let rotated = TttModel::rotate_features(&x, &RotationLabel::R90);
    let any_diff = x
        .iter()
        .zip(rotated.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(
        any_diff,
        "R90 should differ from identity for non-uniform input"
    );
}

#[test]
fn test_ttt_rotate_r270_different_from_r180() {
    let x: Vec<f64> = (1..=8).map(|i| i as f64).collect();
    let r180 = TttModel::rotate_features(&x, &RotationLabel::R180);
    let r270 = TttModel::rotate_features(&x, &RotationLabel::R270);
    let any_diff = r180
        .iter()
        .zip(r270.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(any_diff, "R270 should differ from R180");
}

#[test]
fn test_ttt_aux_loss_non_negative() {
    let model = TttModel::new(8, 3, TttConfig::default());
    let x: Vec<f64> = (0..8).map(|i| i as f64 * 0.1).collect();
    for rot in &[
        RotationLabel::R0,
        RotationLabel::R90,
        RotationLabel::R180,
        RotationLabel::R270,
    ] {
        let loss = model.aux_loss(&x, rot);
        assert!(loss >= 0.0, "aux_loss must be non-negative, got {}", loss);
    }
}

#[test]
fn test_ttt_adapt_runs() {
    let mut model = TttModel::new(8, 3, TttConfig::default());
    let x: Vec<f64> = (0..8).map(|i| i as f64 * 0.1).collect();
    let mut rng = StdRng::seed_from_u64(123);
    model.adapt_test(&x, &mut rng);
}

#[test]
fn test_ttt_predict_sums_to_one() {
    let model = TttModel::new(8, 3, TttConfig::default());
    let x: Vec<f64> = (0..8).map(|i| i as f64 * 0.1).collect();
    let p = model.predict(&x);
    assert_eq!(p.len(), 3);
    let sum: f64 = p.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_ttt_fit_runs() {
    let mut model = TttModel::new(4, 2, TttConfig::default());
    let x_train: Vec<Vec<f64>> = (0..5)
        .map(|i| vec![i as f64, 1.0 - i as f64, 0.5, 0.5])
        .collect();
    let y_train = vec![0usize, 1, 0, 1, 0];
    model.fit(&x_train, &y_train, 2, 0.01);
}

// ── OnlineFeatureStats ─────────────────────────────────────────────────

#[test]
fn test_online_feat_stats_update_changes_mean() {
    let mut stats = OnlineFeatureStats::new(3, 0.1);
    let init_mean = stats.running_mean.clone();
    let feat = vec![1.0, 2.0, 3.0];
    stats.update(&feat);
    let changed = stats
        .running_mean
        .iter()
        .zip(init_mean.iter())
        .any(|(a, b)| (a - b).abs() > 1e-12);
    assert!(changed, "running_mean should change after update");
}

#[test]
fn test_online_feat_stats_n_seen_increments() {
    let mut stats = OnlineFeatureStats::new(4, 0.1);
    assert_eq!(stats.n_seen, 0);
    stats.update(&[0.1, 0.2, 0.3, 0.4]);
    assert_eq!(stats.n_seen, 1);
    stats.update(&[0.5, 0.6, 0.7, 0.8]);
    assert_eq!(stats.n_seen, 2);
}

#[test]
fn test_online_feat_stats_alignment_loss_non_negative() {
    let mut stats = OnlineFeatureStats::new(4, 0.1);
    stats.update(&[1.0, 2.0, 3.0, 4.0]);
    let loss = stats.alignment_loss(&[1.5, 2.5, 3.5, 4.5]);
    assert!(loss >= 0.0, "alignment_loss must be non-negative");
}

#[test]
fn test_online_feat_stats_alignment_loss_zero_before_update() {
    let stats = OnlineFeatureStats::new(4, 0.1);
    let loss = stats.alignment_loss(&[1.0, 2.0, 3.0, 4.0]);
    assert!(
        loss.abs() < 1e-10,
        "alignment_loss should be 0 before any update"
    );
}

// ── TttPlusModel ──────────────────────────────────────────────────────

#[test]
fn test_ttt_plus_adapt_test_runs() {
    let mut model = TttPlusModel::new(4, 8, 3, TttPlusConfig::default());
    // Seed the feature stats with one update.
    model.feat_stats.update(&[0.0f64; 8]);
    let x = vec![0.1, 0.2, 0.3, 0.4];
    model.adapt_test(&x); // should not panic
}

#[test]
fn test_ttt_plus_predict_shape() {
    let model = TttPlusModel::new(4, 8, 3, TttPlusConfig::default());
    let p = model.predict(&[0.1, 0.2, 0.3, 0.4]);
    assert_eq!(p.len(), 3);
}

#[test]
fn test_ttt_plus_predict_sums_to_one() {
    let model = TttPlusModel::new(4, 8, 3, TttPlusConfig::default());
    let p = model.predict(&[0.1, 0.2, 0.3, 0.4]);
    let sum: f64 = p.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_ttt_plus_fit_source_runs() {
    let mut model = TttPlusModel::new(4, 8, 3, TttPlusConfig::default());
    let x_train: Vec<Vec<f64>> = (0..4)
        .map(|i| vec![i as f64 * 0.1, 0.2, 0.3, 0.4])
        .collect();
    let y_train = vec![0usize, 1, 2, 0];
    model.fit_source(&x_train, &y_train, 1);
}

// ── OnlineBnLayer ─────────────────────────────────────────────────────

#[test]
fn test_online_bn_normalize_shape() {
    let config = OnlineBnConfig::default();
    let layer = OnlineBnLayer::new(4, config);
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let out = layer.normalize(&x);
    assert_eq!(out.len(), 4);
}

#[test]
fn test_online_bn_normalize_finite() {
    let config = OnlineBnConfig::default();
    let layer = OnlineBnLayer::new(4, config);
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let out = layer.normalize(&x);
    for &v in &out {
        assert!(v.is_finite());
    }
}

#[test]
fn test_online_bn_adapt_step_updates_stats() {
    let mut layer = OnlineBnLayer::new(3, OnlineBnConfig::default());
    let mean_before = layer.current_mean.clone();
    let batch = vec![vec![10.0, 20.0, 30.0], vec![11.0, 21.0, 31.0]];
    layer.adapt_step(&batch);
    let changed = layer
        .current_mean
        .iter()
        .zip(mean_before.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(changed, "current_mean should update after adapt_step");
}

#[test]
fn test_online_bn_n_adapted_increments() {
    let mut layer = OnlineBnLayer::new(2, OnlineBnConfig::default());
    assert_eq!(layer.n_adapted, 0);
    layer.adapt_step(&[vec![1.0, 2.0]]);
    assert_eq!(layer.n_adapted, 1);
}

// ── EpisodicAdapter ───────────────────────────────────────────────────

#[test]
fn test_episodic_adapt_returns_same_architecture() {
    let config = EpisodicAdaptConfig::default();
    let adapter = EpisodicAdapter::new(&[4, 8, 3], config);
    let sup_x: Vec<Vec<f64>> = (0..3).map(|i| vec![i as f64, 1.0, 0.5, -0.5]).collect();
    let sup_y = vec![0usize, 1, 2];
    let adapted = adapter.adapt(&sup_x, &sup_y);
    // Check same number of layers and dimensions.
    assert_eq!(adapted.layers.len(), adapter.model.layers.len());
    for (la, lb) in adapted.layers.iter().zip(adapter.model.layers.iter()) {
        assert_eq!(la.w.len(), lb.w.len());
    }
}

#[test]
fn test_episodic_meta_train_returns_loss_history() {
    let config = EpisodicAdaptConfig {
        inner_lr: 0.01,
        n_inner_steps: 2,
        outer_lr: 0.001,
        n_outer_steps: 5,
    };
    let mut adapter = EpisodicAdapter::new(&[4, 8, 3], config);
    let sup_x: Vec<Vec<f64>> = (0..3).map(|i| vec![i as f64 * 0.1; 4]).collect();
    let sup_y = vec![0usize, 1, 2];
    let qry_x: Vec<Vec<f64>> = (0..3).map(|i| vec![i as f64 * 0.2; 4]).collect();
    let qry_y = vec![0usize, 1, 2];
    let episodes = vec![(sup_x, sup_y, qry_x, qry_y)];
    let mut rng = StdRng::seed_from_u64(7);
    let history = adapter.meta_train(&episodes, &mut rng);
    assert_eq!(
        history.len(),
        5,
        "loss history should have n_outer_steps entries"
    );
    for &l in &history {
        assert!(l.is_finite());
    }
}

#[test]
fn test_episodic_predict_valid_class() {
    let adapter = EpisodicAdapter::new(&[4, 8, 3], EpisodicAdaptConfig::default());
    let pred = adapter.predict(&[0.1, 0.2, 0.3, 0.4]);
    assert!(pred < 3, "predicted class index should be in range");
}

// ── Domain adaptation losses ───────────────────────────────────────────

#[test]
fn test_mmd_non_negative() {
    let src = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let tgt = vec![vec![2.0, 3.0], vec![4.0, 5.0]];
    let mmd = maximum_mean_discrepancy(&src, &tgt, 1.0);
    assert!(mmd >= 0.0, "MMD must be non-negative, got {}", mmd);
}

#[test]
fn test_mmd_zero_for_identical_distributions() {
    let src = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let tgt = src.clone();
    let mmd = maximum_mean_discrepancy(&src, &tgt, 1.0);
    // For identical src and tgt, E[k(xs,xs')] + E[k(xt,xt')] = 2*E[k(xs,xt)]
    // → MMD should be very close to 0.
    assert!(
        mmd < 1e-8,
        "MMD should be ~0 for identical distributions, got {}",
        mmd
    );
}

#[test]
fn test_mmd_larger_for_different_distributions() {
    let src: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, i as f64]).collect();
    let tgt: Vec<Vec<f64>> = (0..10)
        .map(|i| vec![100.0 + i as f64, 100.0 + i as f64])
        .collect();
    let mmd_diff = maximum_mean_discrepancy(&src, &tgt, 1.0);
    let mmd_same = maximum_mean_discrepancy(&src, &src, 1.0);
    assert!(
        mmd_diff >= mmd_same,
        "MMD for shifted distributions should be >= same-distribution MMD"
    );
}

#[test]
fn test_coral_non_negative() {
    let src = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let tgt = vec![vec![2.0, 3.0], vec![4.0, 5.0], vec![6.0, 7.0]];
    let loss = correlation_alignment_loss(&src, &tgt);
    assert!(loss >= 0.0, "CORAL must be non-negative, got {}", loss);
}

#[test]
fn test_coral_zero_for_identical() {
    let src = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let loss = correlation_alignment_loss(&src, &src);
    assert!(
        loss < 1e-10,
        "CORAL should be 0 for identical distributions, got {}",
        loss
    );
}

#[test]
fn test_entropy_loss_non_negative() {
    let logits = vec![1.0, 0.5, -0.5, 0.0];
    let h = entropy_minimization_loss(&logits);
    assert!(h >= 0.0, "entropy loss must be non-negative");
}

#[test]
fn test_entropy_loss_zero_for_one_hot() {
    // A very large logit on one class → near-deterministic → H ≈ 0.
    let mut logits = vec![0.0f64; 5];
    logits[2] = 1000.0;
    let h = entropy_minimization_loss(&logits);
    assert!(
        h < 1e-6,
        "entropy should be near-zero for one-hot logits, got {}",
        h
    );
}

#[test]
fn test_entropy_loss_maximised_for_uniform() {
    // Uniform logits → maximum entropy.
    let n = 4usize;
    let logits_uniform = vec![0.0f64; n];
    let h_uniform = entropy_minimization_loss(&logits_uniform);
    let h_max = (n as f64).ln();
    assert!(
        (h_uniform - h_max).abs() < 1e-6,
        "uniform logits should give max entropy, got {} expected {}",
        h_uniform,
        h_max
    );
}

// ── TtaMetrics & evaluate_tta ──────────────────────────────────────────

#[test]
fn test_tta_metrics_accuracy_in_range() {
    let model = TentModel::new(&[4, 8, 3], TentConfig::default());
    let x_test: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 0.1; 4]).collect();
    let y_test = vec![0usize, 1, 2, 0, 1];
    let metrics = evaluate_tta(&model, &x_test, &y_test);
    assert!(
        metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0,
        "accuracy must be in [0, 1], got {}",
        metrics.accuracy
    );
}

#[test]
fn test_evaluate_tta_returns_finite_accuracy() {
    let model = TentModel::new(&[4, 8, 3], TentConfig::default());
    let x_test: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.05; 4]).collect();
    let y_test: Vec<usize> = (0..10).map(|i| i % 3).collect();
    let metrics = evaluate_tta(&model, &x_test, &y_test);
    assert!(metrics.accuracy.is_finite());
}

#[test]
fn test_evaluate_tta_mean_entropy_non_negative() {
    let model = TentModel::new(&[4, 8, 3], TentConfig::default());
    let x_test: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 0.1; 4]).collect();
    let y_test = vec![0usize, 1, 2, 0, 1];
    let metrics = evaluate_tta(&model, &x_test, &y_test);
    assert!(metrics.mean_entropy >= 0.0);
}

#[test]
fn test_evaluate_tta_mean_confidence_in_range() {
    let model = TentModel::new(&[4, 8, 3], TentConfig::default());
    let x_test: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 0.1; 4]).collect();
    let y_test = vec![0usize, 1, 2, 0, 1];
    let metrics = evaluate_tta(&model, &x_test, &y_test);
    assert!(metrics.mean_confidence >= 0.0 && metrics.mean_confidence <= 1.0);
}

#[test]
fn test_evaluate_tta_empty_set() {
    let model = TentModel::new(&[4, 8, 3], TentConfig::default());
    let metrics = evaluate_tta(&model, &[], &[]);
    assert_eq!(metrics.accuracy, 0.0);
    assert_eq!(metrics.adaptation_time_steps, 0);
}

// ── Additional edge-case tests ─────────────────────────────────────────

#[test]
fn test_softmax_sums_to_one() {
    let logits = vec![1.5, -0.5, 0.3, 2.1];
    let p = softmax(&logits);
    let sum: f64 = p.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12);
}

#[test]
fn test_online_bn_adapt_with_single_sample() {
    let mut layer = OnlineBnLayer::new(2, OnlineBnConfig::default());
    // Single-element batch should not cause division by zero.
    layer.adapt_step(&[vec![3.0, 4.0]]);
    assert!(layer.current_mean[0].is_finite());
    assert!(layer.current_var[0].is_finite());
}

#[test]
fn test_ttt_rotate_preserves_length() {
    let x: Vec<f64> = (0..12).map(|i| i as f64).collect();
    for rot in &[
        RotationLabel::R0,
        RotationLabel::R90,
        RotationLabel::R180,
        RotationLabel::R270,
    ] {
        let out = TttModel::rotate_features(&x, rot);
        assert_eq!(out.len(), x.len(), "rotation must preserve length");
    }
}

#[test]
fn test_tent_model_no_bn_adapts_gracefully() {
    // layer_sizes with only 2 elements → no hidden layers → no BN.
    let mut model = TentModel::new(&[4, 3], TentConfig::default());
    assert_eq!(model.bn_stats.len(), 0);
    // adapt should return without error.
    let batch = vec![vec![1.0, 2.0, 3.0, 4.0]];
    model.adapt(&batch);
}

#[test]
fn test_mmd_empty_inputs() {
    let mmd = maximum_mean_discrepancy(&[], &[], 1.0);
    assert_eq!(mmd, 0.0);
}

#[test]
fn test_coral_small_batch_returns_zero() {
    // n < 2 → should return 0.
    let src = vec![vec![1.0, 2.0]];
    let tgt = vec![vec![3.0, 4.0]];
    let loss = correlation_alignment_loss(&src, &tgt);
    assert_eq!(loss, 0.0);
}

#[test]
fn test_episodic_adapt_changes_params() {
    let config = EpisodicAdaptConfig {
        inner_lr: 0.1,
        n_inner_steps: 5,
        outer_lr: 0.01,
        n_outer_steps: 1,
    };
    let adapter = EpisodicAdapter::new(&[4, 8, 3], config);
    let sup_x: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64, 0.5, -0.3, 1.0]).collect();
    let sup_y = vec![0usize, 1, 2, 0];
    let adapted = adapter.adapt(&sup_x, &sup_y);
    let base_flat = adapter.model.params_flat();
    let adapted_flat = adapted.params_flat();
    let any_diff = base_flat
        .iter()
        .zip(adapted_flat.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(
        any_diff,
        "adapted model should differ from base after adaptation"
    );
}

#[test]
fn test_bn_running_stats_update_after_train() {
    let mut bn = TtaBatchNormStats::new(2);
    let init_mean = bn.running_mean.clone();
    let batch = vec![vec![5.0, 10.0], vec![5.0, 10.0], vec![5.0, 10.0]];
    bn.forward_train(&batch);
    let changed = bn
        .running_mean
        .iter()
        .zip(init_mean.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(changed, "running_mean should update after forward_train");
}
