//! Tests for the Neural Collapse module.

use super::*;

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn make_seed() -> u64 {
    0xDEAD_BEEF_1234_5678u64
}

/// Generate num_classes clusters of n_per_class samples in R^d around ETF-like means.
fn generate_clustered_features(
    num_classes: usize,
    n_per_class: usize,
    d: usize,
    noise: f64,
    seed: &mut u64,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut features = Vec::new();
    let mut labels = Vec::new();
    // Simple class means: class k has mean = e_k (basis vector) if d >= num_classes, else wrap around
    for k in 0..num_classes {
        let center_idx = k % d;
        for _ in 0..n_per_class {
            let mut feat = vec![0.0f64; d];
            feat[center_idx] = 1.0;
            // Add Gaussian noise
            for j in 0..d {
                feat[j] += noise * ncl_randn(seed);
            }
            features.push(feat);
            labels.push(k);
        }
    }
    (features, labels)
}

/// Generate uniformly random features (no cluster structure).
fn generate_random_features(n: usize, d: usize, seed: &mut u64) -> Vec<Vec<f64>> {
    (0..n)
        .map(|_| (0..d).map(|_| ncl_randn(seed)).collect())
        .collect()
}

// ─── §1 NclEquiangularTightFrame ─────────────────────────────────────────────

#[test]
fn test_etf_creation_k2() {
    let mut seed = make_seed();
    let etf = NclEquiangularTightFrame::new(2, 4, &mut seed)
        .expect("ETF num_classes=2 d=4 should succeed");
    assert_eq!(etf.num_classes, 2);
    assert_eq!(etf.d, 4);
    assert_eq!(etf.prototypes.len(), 2);
    assert_eq!(etf.prototypes[0].len(), 4);
}

#[test]
fn test_etf_creation_k3() {
    let mut seed = make_seed();
    let etf = NclEquiangularTightFrame::new(3, 5, &mut seed)
        .expect("ETF num_classes=3 d=5 should succeed");
    assert_eq!(etf.num_classes, 3);
    assert_eq!(etf.d, 5);
}

#[test]
fn test_etf_creation_k5() {
    let mut seed = make_seed();
    let etf = NclEquiangularTightFrame::new(5, 10, &mut seed)
        .expect("ETF num_classes=5 d=10 should succeed");
    assert_eq!(etf.num_classes, 5);
    assert_eq!(etf.d, 10);
    assert_eq!(etf.prototypes.len(), 5);
}

#[test]
fn test_etf_unit_norm() {
    let mut seed = make_seed();
    let etf = NclEquiangularTightFrame::new(4, 8, &mut seed).expect("ETF should succeed");
    for proto in &etf.prototypes {
        let nrm: f64 = proto.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (nrm - 1.0).abs() < 1e-9,
            "Prototype norm should be 1, got {}",
            nrm
        );
    }
}

#[test]
fn test_etf_gram_diagonal_equals_one() {
    let mut seed = make_seed();
    let etf = NclEquiangularTightFrame::new(3, 6, &mut seed).expect("ETF should succeed");
    let gram = etf.gram_matrix();
    for i in 0..etf.num_classes {
        assert!((gram[i][i] - 1.0).abs() < 1e-9, "Gram diagonal should be 1");
    }
}

#[test]
fn test_etf_gram_offdiagonal_etf_property() {
    let mut seed = make_seed();
    let num_classes = 4usize;
    let etf =
        NclEquiangularTightFrame::new(num_classes, 10, &mut seed).expect("ETF should succeed");
    let gram = etf.gram_matrix();
    let target = NclEquiangularTightFrame::theoretical_inner_product(num_classes);
    for i in 0..num_classes {
        for j in 0..num_classes {
            if i != j {
                assert!(
                    (gram[i][j] - target).abs() < 0.15,
                    "Gram[{},{}] = {} should be ≈ {} (target ETF inner product)",
                    i,
                    j,
                    gram[i][j],
                    target
                );
            }
        }
    }
}

#[test]
fn test_etf_verify_property_k2() {
    let mut seed = make_seed();
    let etf = NclEquiangularTightFrame::new(2, 4, &mut seed).expect("ETF should succeed");
    // For num_classes=2, inner product should be -1; check with generous tolerance
    assert!(
        etf.verify_etf_property(0.2),
        "ETF num_classes=2 should verify with tol=0.2"
    );
}

#[test]
fn test_etf_theoretical_inner_product() {
    assert!((NclEquiangularTightFrame::theoretical_inner_product(2) - (-1.0)).abs() < 1e-12);
    assert!((NclEquiangularTightFrame::theoretical_inner_product(3) - (-0.5)).abs() < 1e-12);
    assert!((NclEquiangularTightFrame::theoretical_inner_product(5) - (-0.25)).abs() < 1e-12);
}

#[test]
fn test_etf_nearest_class() {
    let mut seed = make_seed();
    let etf = NclEquiangularTightFrame::new(3, 6, &mut seed).expect("ETF should succeed");
    // Query matching proto 1 exactly should return class 1
    let query = etf.prototypes[1].clone();
    let cls = etf.nearest_class(&query);
    assert_eq!(cls, 1, "Nearest class to prototype 1 should be 1");
}

#[test]
fn test_etf_error_k_less_than_2() {
    let mut seed = make_seed();
    let result = NclEquiangularTightFrame::new(1, 5, &mut seed);
    assert!(result.is_err(), "num_classes=1 should fail");
}

#[test]
fn test_etf_error_d_too_small() {
    let mut seed = make_seed();
    let result = NclEquiangularTightFrame::new(5, 3, &mut seed); // d < num_classes-1
    assert!(result.is_err(), "d < num_classes-1 should fail");
}

#[test]
fn test_etf_gram_matrix_shape() {
    let mut seed = make_seed();
    let num_classes = 4;
    let etf = NclEquiangularTightFrame::new(num_classes, 8, &mut seed).expect("ETF should succeed");
    let gram = etf.gram_matrix();
    assert_eq!(gram.len(), num_classes);
    for row in &gram {
        assert_eq!(row.len(), num_classes);
    }
}

// ─── §2 NclNeuralCollapseMetrics ────────────────────────────────────────────

#[test]
fn test_compute_stats_basic() {
    let mut seed = make_seed();
    let (features, labels) = generate_clustered_features(3, 10, 4, 0.1, &mut seed);
    let stats = NclNeuralCollapseMetrics::compute_stats(&features, &labels, 3)
        .expect("Should compute stats");
    assert_eq!(stats.num_classes, 3);
    assert_eq!(stats.d, 4);
    assert_eq!(stats.class_means.len(), 3);
    assert_eq!(stats.global_mean.len(), 4);
}

#[test]
fn test_compute_stats_within_class_cov_shape() {
    let mut seed = make_seed();
    let (features, labels) = generate_clustered_features(3, 10, 5, 0.1, &mut seed);
    let stats = NclNeuralCollapseMetrics::compute_stats(&features, &labels, 3)
        .expect("Should compute stats");
    assert_eq!(stats.within_class_cov.len(), 5);
    assert_eq!(stats.within_class_cov[0].len(), 5);
    assert_eq!(stats.between_class_cov.len(), 5);
    assert_eq!(stats.between_class_cov[0].len(), 5);
}

#[test]
fn test_compute_stats_error_label_out_of_range() {
    let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let labels = vec![0, 5]; // label 5 out of range for num_classes=3
    let result = NclNeuralCollapseMetrics::compute_stats(&features, &labels, 3);
    assert!(result.is_err(), "Label out of range should fail");
}

#[test]
fn test_nc1_within_class_variability_nonnegative() {
    let mut seed = make_seed();
    let (features, labels) = generate_clustered_features(3, 10, 4, 0.1, &mut seed);
    let stats = NclNeuralCollapseMetrics::compute_stats(&features, &labels, 3)
        .expect("Should compute stats");
    let nc1 = NclNeuralCollapseMetrics::nc1_within_class_variability(&stats);
    assert!(nc1 >= 0.0, "NC1 should be non-negative, got {}", nc1);
}

#[test]
fn test_nc1_low_noise_lower_than_high_noise() {
    let mut seed1 = make_seed();
    let mut seed2 = make_seed();
    let (feats_low, labels_low) = generate_clustered_features(3, 20, 6, 0.01, &mut seed1);
    let (feats_high, labels_high) = generate_clustered_features(3, 20, 6, 2.0, &mut seed2);
    let stats_low = NclNeuralCollapseMetrics::compute_stats(&feats_low, &labels_low, 3)
        .expect("compute_stats for low noise should succeed");
    let stats_high = NclNeuralCollapseMetrics::compute_stats(&feats_high, &labels_high, 3)
        .expect("compute_stats for high noise should succeed");
    let nc1_low = NclNeuralCollapseMetrics::nc1_within_class_variability(&stats_low);
    let nc1_high = NclNeuralCollapseMetrics::nc1_within_class_variability(&stats_high);
    assert!(
        nc1_low <= nc1_high,
        "Low noise NC1 {} should <= high noise NC1 {}",
        nc1_low,
        nc1_high
    );
}

#[test]
fn test_nc2_convergence_to_etf_finite() {
    let mut seed = make_seed();
    let (features, labels) = generate_clustered_features(3, 10, 5, 0.1, &mut seed);
    let stats = NclNeuralCollapseMetrics::compute_stats(&features, &labels, 3)
        .expect("Should compute stats");
    let nc2 = NclNeuralCollapseMetrics::nc2_convergence_to_etf(&stats);
    assert!(nc2.is_finite(), "NC2 should be finite, got {}", nc2);
    assert!(nc2 >= 0.0, "NC2 should be non-negative");
}

#[test]
fn test_nc3_duality_gap_with_matching_weights() {
    let mut seed = make_seed();
    let num_classes = 3;
    let d = 4;
    let (features, labels) = generate_clustered_features(num_classes, 10, d, 0.1, &mut seed);
    let stats = NclNeuralCollapseMetrics::compute_stats(&features, &labels, num_classes)
        .expect("compute_stats should succeed");
    // Use *centered* class means as classifier weights → should give gap ≈ 0
    // nc3_duality_gap normalizes centered means internally, so weights must also be centered
    let centered_weights: Vec<Vec<f64>> = stats
        .class_means
        .iter()
        .map(|mu| (0..d).map(|j| mu[j] - stats.global_mean[j]).collect())
        .collect();
    let nc3_zero = NclNeuralCollapseMetrics::nc3_duality_gap(&stats, &centered_weights);
    assert!(
        nc3_zero.is_finite(),
        "NC3 with matching weights should be finite"
    );
    assert!(nc3_zero < 0.5, "NC3 should be small when W ≈ centered class means");
}

#[test]
fn test_nc4_simple_mnc_range() {
    let mut seed = make_seed();
    let (features, labels) = generate_clustered_features(4, 10, 6, 0.1, &mut seed);
    let stats = NclNeuralCollapseMetrics::compute_stats(&features, &labels, 4)
        .expect("compute_stats should succeed");
    let nc4 = NclNeuralCollapseMetrics::nc4_simple_mnc(&stats);
    assert!(
        (0.0..=1.0).contains(&nc4),
        "NC4 should be in [0,1], got {}",
        nc4
    );
}

#[test]
fn test_collapse_index_range() {
    let mut seed = make_seed();
    let (features, labels) = generate_clustered_features(3, 10, 4, 0.1, &mut seed);
    let stats = NclNeuralCollapseMetrics::compute_stats(&features, &labels, 3)
        .expect("compute_stats should succeed");
    let ci = NclNeuralCollapseMetrics::collapse_index(&stats);
    assert!(
        ci.is_finite() && ci >= 0.0,
        "Collapse index should be finite non-negative"
    );
}

// ─── §3 NclEtfClassifier ──────────────────────────────────────────────────

#[test]
fn test_etf_classifier_creation() {
    let mut seed = make_seed();
    let clf = NclEtfClassifier::new(10, 8, 3, &mut seed).expect("Classifier should be created");
    assert_eq!(clf.input_dim, 10);
    assert_eq!(clf.feat_dim, 8);
    assert_eq!(clf.num_classes, 3);
}

#[test]
fn test_etf_classifier_features_normalized() {
    let mut seed = make_seed();
    let clf = NclEtfClassifier::new(10, 8, 3, &mut seed)
        .expect("ETF classifier creation should succeed");
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let z = clf.extract_features(&x);
    assert_eq!(z.len(), 8);
    let nrm: f64 = z.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        (nrm - 1.0).abs() < 1e-9,
        "Extracted features should be L2-normalized, got norm {}",
        nrm
    );
}

#[test]
fn test_etf_classifier_classify_valid_class() {
    let mut seed = make_seed();
    let num_classes = 4;
    let clf = NclEtfClassifier::new(8, 6, num_classes, &mut seed)
        .expect("ETF classifier creation should succeed");
    let x: Vec<f64> = (0..8).map(|i| i as f64 * 0.1).collect();
    let cls = clf.classify(&x);
    assert!(
        cls < num_classes,
        "Predicted class {} should be in [0, {})",
        cls,
        num_classes
    );
}

#[test]
fn test_etf_classifier_logits_length() {
    let mut seed = make_seed();
    let num_classes = 5;
    let clf = NclEtfClassifier::new(6, 8, num_classes, &mut seed)
        .expect("ETF classifier creation should succeed");
    let x: Vec<f64> = vec![1.0; 6];
    let lgts = clf.logits(&x);
    assert_eq!(
        lgts.len(),
        num_classes,
        "Logits should have num_classes={} elements",
        num_classes
    );
}

#[test]
fn test_etf_classifier_mse_loss_nonnegative() {
    let mut seed = make_seed();
    let clf = NclEtfClassifier::new(6, 8, 3, &mut seed)
        .expect("ETF classifier creation should succeed");
    let x: Vec<f64> = vec![0.5; 6];
    let loss = clf.mse_nc_loss(&x, 0);
    assert!(
        loss >= 0.0,
        "MSE NC loss should be non-negative, got {}",
        loss
    );
}

#[test]
fn test_etf_classifier_cross_entropy_nonnegative() {
    let mut seed = make_seed();
    let clf = NclEtfClassifier::new(6, 8, 3, &mut seed)
        .expect("ETF classifier creation should succeed");
    let x: Vec<f64> = vec![0.3; 6];
    let loss = clf.cross_entropy_loss(&x, 1);
    assert!(loss >= 0.0, "CE loss should be non-negative, got {}", loss);
}

#[test]
fn test_etf_classifier_update_decreases_loss() {
    let mut seed = make_seed();
    let mut clf = NclEtfClassifier::new(4, 6, 3, &mut seed)
        .expect("ETF classifier creation should succeed");
    let x: Vec<f64> = vec![1.0, 0.5, -0.3, 0.8];
    let label = 0;
    let loss_before = clf.mse_nc_loss(&x, label);
    // Multiple gradient steps
    for _ in 0..20 {
        clf.update(&x, label, 0.05);
    }
    let loss_after = clf.mse_nc_loss(&x, label);
    assert!(
        loss_after <= loss_before + 1e-6,
        "Loss should decrease after updates: before={}, after={}",
        loss_before,
        loss_after
    );
}

#[test]
fn test_etf_classifier_error_feat_dim_too_small() {
    let mut seed = make_seed();
    let result = NclEtfClassifier::new(10, 1, 5, &mut seed); // feat_dim=1 < num_classes-1=4
    assert!(result.is_err(), "feat_dim < num_classes-1 should fail");
}

// ─── §4 NclDrLoss ────────────────────────────────────────────────────────────

#[test]
fn test_dr_loss_nonnegative() {
    let logits = vec![
        vec![1.0, 0.2, -0.5],
        vec![-0.3, 0.8, 0.1],
        vec![0.5, -0.1, 0.7],
    ];
    let labels = vec![0, 1, 2];
    let dr = NclDrLoss::new(0.1);
    let loss = dr.compute(&logits, &labels, 1.0);
    assert!(loss >= 0.0, "DR loss should be non-negative, got {}", loss);
}

#[test]
fn test_dr_loss_empty_input() {
    let dr = NclDrLoss::new(0.1);
    let loss = dr.compute(&[], &[], 1.0);
    assert_eq!(loss, 0.0, "Empty input should give 0 loss");
}

#[test]
fn test_gram_loss_nonnegative() {
    let logits = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let labels = vec![0, 1, 2];
    let gl = NclDrLoss::gram_loss(&logits, &labels, 3);
    assert!(gl >= 0.0, "Gram loss should be non-negative, got {}", gl);
}

#[test]
fn test_class_mean_logits_shape() {
    let logits = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
        vec![1.0, 0.5, 0.2],
    ];
    let labels = vec![0, 1, 2, 0];
    let means = NclDrLoss::class_mean_logits(&logits, &labels, 3);
    assert_eq!(
        means.len(),
        3,
        "Should have num_classes=3 class mean logit vectors"
    );
    assert_eq!(means[0].len(), 3);
}

#[test]
fn test_class_mean_logits_correct_mean() {
    // Class 0 has samples at index 0 and 3
    let logits = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![3.0, 0.0]];
    let labels = vec![0, 1, 0];
    let means = NclDrLoss::class_mean_logits(&logits, &labels, 2);
    // Class 0 mean: (logits[0] + logits[2]) / 2 = [2.0, 0.0]
    assert!((means[0][0] - 2.0).abs() < 1e-9);
    assert!((means[0][1] - 0.0).abs() < 1e-9);
    // Class 1 mean: logits[1] = [0.0, 1.0]
    assert!((means[1][0] - 0.0).abs() < 1e-9);
    assert!((means[1][1] - 1.0).abs() < 1e-9);
}

#[test]
fn test_dr_loss_lambda_effect() {
    let logits = vec![vec![1.0, 0.2, -0.5], vec![-0.3, 0.8, 0.1]];
    let labels = vec![0, 1];
    let dr0 = NclDrLoss::new(0.0);
    let dr1 = NclDrLoss::new(1.0);
    let loss0 = dr0.compute(&logits, &labels, 1.0);
    let loss1 = dr1.compute(&logits, &labels, 1.0);
    // Higher lambda = higher total loss (gram term adds to it)
    assert!(loss1 >= loss0 - 1e-9, "Higher lambda should give >= loss");
}

// ─── §5 NclSupcon ─────────────────────────────────────────────────────────

#[test]
fn test_supcon_loss_nonnegative() {
    let mut seed = make_seed();
    let n = 6;
    let d = 4;
    // Generate L2-normalized features
    let mut features = generate_random_features(n, d, &mut seed);
    for f in features.iter_mut() {
        let nrm: f64 = f.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in f.iter_mut() {
            *x /= nrm.max(1e-12);
        }
    }
    let labels = vec![0, 0, 1, 1, 2, 2];
    let sc = NclSupcon::new(0.07);
    let loss = sc
        .loss(&features, &labels)
        .expect("SupCon loss should succeed");
    assert!(
        loss >= 0.0,
        "SupCon loss should be non-negative, got {}",
        loss
    );
}

#[test]
fn test_supcon_loss_error_single_sample() {
    let features = vec![vec![1.0, 0.0, 0.0]];
    let labels = vec![0];
    let sc = NclSupcon::new(0.1);
    let result = sc.loss(&features, &labels);
    assert!(result.is_err(), "Single sample should return error");
}

#[test]
fn test_supcon_loss_length_mismatch() {
    let features = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let labels = vec![0];
    let sc = NclSupcon::new(0.1);
    let result = sc.loss(&features, &labels);
    assert!(result.is_err(), "Length mismatch should return error");
}

#[test]
fn test_nc_supcon_loss_nonnegative() {
    let mut seed = make_seed();
    let num_classes = 3;
    let d = 6;
    let etf = NclEquiangularTightFrame::new(num_classes, d, &mut seed)
        .expect("ETF creation should succeed");
    let mut features = generate_random_features(6, d, &mut seed);
    for f in features.iter_mut() {
        let nrm: f64 = f.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        for x in f.iter_mut() {
            *x /= nrm;
        }
    }
    let labels = vec![0, 1, 2, 0, 1, 2];
    let sc = NclSupcon::new(0.1);
    let loss = sc
        .nc_supcon_loss(&features, &labels, &etf)
        .expect("NC-SupCon should succeed");
    assert!(
        loss >= 0.0,
        "NC-SupCon loss should be non-negative, got {}",
        loss
    );
}

#[test]
fn test_supcon_with_all_same_class() {
    // All anchors same class — positives exist for every anchor
    let mut seed = make_seed();
    let d = 4;
    let mut features = generate_random_features(4, d, &mut seed);
    for f in features.iter_mut() {
        let nrm: f64 = f.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        for x in f.iter_mut() {
            *x /= nrm;
        }
    }
    let labels = vec![0, 0, 0, 0];
    let sc = NclSupcon::new(0.1);
    let loss = sc
        .loss(&features, &labels)
        .expect("All same class should succeed");
    assert!(loss.is_finite(), "Loss should be finite");
}

// ─── §6 NclPrototypeClassifier ───────────────────────────────────────────────

#[test]
fn test_prototype_classifier_creation() {
    let clf = NclPrototypeClassifier::new(4, 8, 0.9);
    assert_eq!(clf.num_classes, 4);
    assert_eq!(clf.d, 8);
    assert_eq!(clf.prototypes.len(), 4);
    assert!((clf.update_momentum - 0.9).abs() < 1e-9);
}

#[test]
fn test_prototype_initialize_from_data() {
    let mut seed = make_seed();
    let (features, labels) = generate_clustered_features(3, 10, 4, 0.01, &mut seed);
    let mut clf = NclPrototypeClassifier::new(3, 4, 0.9);
    clf.initialize_from_data(&features, &labels)
        .expect("Should initialize");
    // Prototypes should not be all zeros
    let all_zero = clf.prototypes.iter().all(|p| p.iter().all(|x| *x == 0.0));
    assert!(
        !all_zero,
        "Prototypes should not all be zero after initialization"
    );
}

#[test]
fn test_prototype_classify_valid_class() {
    let mut seed = make_seed();
    let num_classes = 3;
    let d = 4;
    let (features, labels) = generate_clustered_features(num_classes, 10, d, 0.01, &mut seed);
    let mut clf = NclPrototypeClassifier::new(num_classes, d, 0.9);
    clf.initialize_from_data(&features, &labels)
        .expect("initialize_from_data should succeed");
    let pred = clf.classify(&features[0]);
    assert!(
        pred < num_classes,
        "Predicted class {} should be in [0, {})",
        pred,
        num_classes
    );
}

#[test]
fn test_prototype_classify_correct_for_clean_data() {
    let mut seed = make_seed();
    let num_classes = 3;
    let d = 4;
    let (features, labels) = generate_clustered_features(num_classes, 10, d, 0.01, &mut seed);
    let mut clf = NclPrototypeClassifier::new(num_classes, d, 0.9);
    clf.initialize_from_data(&features, &labels)
        .expect("initialize_from_data should succeed");
    // First sample of class 0 should be classified as 0 with very low noise
    let pred = clf.classify(&features[0]);
    assert_eq!(
        pred, labels[0],
        "With low noise, should predict correct class"
    );
}

#[test]
fn test_prototype_ema_update() {
    let mut clf = NclPrototypeClassifier::new(2, 3, 0.9);
    clf.prototypes[0] = vec![1.0, 0.0, 0.0];
    let new_feat = vec![0.0, 1.0, 0.0];
    clf.update_prototype_ema(&new_feat, 0);
    // After EMA: proto[0] ≈ 0.9 * [1,0,0] + 0.1 * [0,1,0] = [0.9, 0.1, 0.0]
    assert!((clf.prototypes[0][0] - 0.9).abs() < 1e-9);
    assert!((clf.prototypes[0][1] - 0.1).abs() < 1e-9);
}

#[test]
fn test_prototype_distances_shape() {
    let clf = NclPrototypeClassifier::new(4, 6, 0.9);
    let dist = clf.prototype_distances();
    assert_eq!(dist.len(), 4);
    assert_eq!(dist[0].len(), 4);
    // Diagonal should be 0
    for i in 0..4 {
        assert!((dist[i][i]).abs() < 1e-12);
    }
}

#[test]
fn test_prototype_etf_alignment_after_etf_init() {
    let mut seed = make_seed();
    let num_classes = 3;
    let d = 6;
    let etf = NclEquiangularTightFrame::new(num_classes, d, &mut seed)
        .expect("ETF creation should succeed");
    let mut clf = NclPrototypeClassifier::new(num_classes, d, 0.9);
    // Set prototypes to ETF
    clf.prototypes = etf.prototypes.clone();
    let alignment = clf.measure_etf_alignment();
    // Should be near 0 deviation
    assert!(
        alignment < 0.2,
        "ETF-initialized prototypes should have low deviation, got {}",
        alignment
    );
}

// ─── §7 NclLayerAnalysis ─────────────────────────────────────────────────────

#[test]
fn test_layer_analysis_creation() {
    let analysis = NclLayerAnalysis::new();
    assert!(analysis.layer_stats.is_empty());
    assert!(analysis.nc1_per_layer.is_empty());
}

#[test]
fn test_layer_analysis_add_stats() {
    let mut seed = make_seed();
    let mut analysis = NclLayerAnalysis::new();
    let (features, labels) = generate_clustered_features(3, 10, 4, 0.1, &mut seed);
    let stats = NclNeuralCollapseMetrics::compute_stats(&features, &labels, 3)
        .expect("compute_stats should succeed");
    analysis.add_layer_stats(stats, "layer1");
    assert_eq!(analysis.layer_stats.len(), 1);
    assert_eq!(analysis.layer_names[0], "layer1");
}

#[test]
fn test_layer_analysis_compute_progression() {
    let mut seed = make_seed();
    let mut analysis = NclLayerAnalysis::new();
    for i in 0..4 {
        let noise = 1.0 / (i as f64 + 1.0);
        let (features, labels) = generate_clustered_features(3, 10, 4, noise, &mut seed);
        let stats = NclNeuralCollapseMetrics::compute_stats(&features, &labels, 3)
            .expect("compute_stats should succeed");
        analysis.add_layer_stats(stats, &format!("layer{}", i));
    }
    analysis.compute_nc_progression(3);
    assert_eq!(analysis.nc1_per_layer.len(), 4);
    assert_eq!(analysis.nc2_per_layer.len(), 4);
}

#[test]
fn test_layer_analysis_nc_values_finite() {
    let mut seed = make_seed();
    let mut analysis = NclLayerAnalysis::new();
    let (features, labels) = generate_clustered_features(3, 10, 5, 0.1, &mut seed);
    let stats = NclNeuralCollapseMetrics::compute_stats(&features, &labels, 3)
        .expect("compute_stats should succeed");
    analysis.add_layer_stats(stats, "only_layer");
    analysis.compute_nc_progression(3);
    assert!(analysis.nc1_per_layer[0].is_finite());
    assert!(analysis.nc2_per_layer[0].is_finite());
}

#[test]
fn test_layer_analysis_improvement_rate() {
    let mut seed = make_seed();
    let mut analysis = NclLayerAnalysis::new();
    for i in 0..3 {
        let noise = 1.0 / (i as f64 + 1.0);
        let (features, labels) = generate_clustered_features(3, 10, 4, noise, &mut seed);
        let stats = NclNeuralCollapseMetrics::compute_stats(&features, &labels, 3)
            .expect("compute_stats should succeed");
        analysis.add_layer_stats(stats, &format!("l{}", i));
    }
    analysis.compute_nc_progression(3);
    let rates = analysis.nc_improvement_rate();
    assert_eq!(rates.len(), 2, "Should have n-1 improvement rates");
}

#[test]
fn test_layer_analysis_report_nonempty() {
    let mut seed = make_seed();
    let mut analysis = NclLayerAnalysis::new();
    let (features, labels) = generate_clustered_features(3, 10, 4, 0.1, &mut seed);
    let stats = NclNeuralCollapseMetrics::compute_stats(&features, &labels, 3)
        .expect("compute_stats should succeed");
    analysis.add_layer_stats(stats, "layer1");
    analysis.compute_nc_progression(3);
    let report = analysis.report();
    assert!(!report.is_empty(), "Report should not be empty");
    assert!(
        report.contains("layer1"),
        "Report should mention layer name"
    );
}

// ─── §8 NclFewShotNc ─────────────────────────────────────────────────────────

#[test]
fn test_few_shot_creation() {
    let fs = NclFewShotNc::new(8);
    assert_eq!(fs.d, 8);
    assert!(fs.support_means.is_empty());
}

#[test]
fn test_few_shot_fit_support() {
    let mut seed = make_seed();
    let num_classes = 3;
    let d = 6;
    let (support_feats, support_labels) =
        generate_clustered_features(num_classes, 5, d, 0.1, &mut seed);
    let mut fs = NclFewShotNc::new(d);
    fs.fit_support(&support_feats, &support_labels, num_classes)
        .expect("fit_support should succeed");
    assert_eq!(fs.support_means.len(), num_classes);
}

#[test]
fn test_few_shot_classify_valid_class() {
    let mut seed = make_seed();
    let num_classes = 3;
    let d = 5;
    let (support_feats, support_labels) =
        generate_clustered_features(num_classes, 5, d, 0.1, &mut seed);
    let mut fs = NclFewShotNc::new(d);
    fs.fit_support(&support_feats, &support_labels, num_classes)
        .expect("fit_support should succeed");
    let query = vec![0.0f64; d];
    let cls = fs.classify(&query).expect("Classify should succeed");
    assert!(
        cls < num_classes,
        "Class {} should be in [0, {})",
        cls,
        num_classes
    );
}

#[test]
fn test_few_shot_classify_without_fit_fails() {
    let fs = NclFewShotNc::new(4);
    let result = fs.classify(&[1.0, 0.0, 0.0, 0.0]);
    assert!(result.is_err(), "Classify without fit should fail");
}

#[test]
fn test_few_shot_classify_with_nc_prior() {
    let mut seed = make_seed();
    let num_classes = 3;
    let d = 6;
    let etf = NclEquiangularTightFrame::new(num_classes, d, &mut seed)
        .expect("ETF creation should succeed");
    let (support_feats, support_labels) =
        generate_clustered_features(num_classes, 5, d, 0.1, &mut seed);
    let mut fs = NclFewShotNc::new(d);
    fs.fit_support(&support_feats, &support_labels, num_classes)
        .expect("fit_support should succeed");
    let query = vec![0.0f64; d];
    let cls = fs
        .classify_with_nc_prior(&query, &etf, 0.5)
        .expect("NC prior classify should succeed");
    assert!(
        cls < num_classes,
        "Class {} should be in [0, {})",
        cls,
        num_classes
    );
}

#[test]
fn test_few_shot_alpha_0_equals_plain_classify() {
    let mut seed = make_seed();
    let num_classes = 3;
    let d = 5;
    let etf = NclEquiangularTightFrame::new(num_classes, d, &mut seed)
        .expect("ETF creation should succeed");
    let (support_feats, support_labels) =
        generate_clustered_features(num_classes, 5, d, 0.1, &mut seed);
    let mut fs = NclFewShotNc::new(d);
    fs.fit_support(&support_feats, &support_labels, num_classes)
        .expect("fit_support should succeed");
    let query: Vec<f64> = (0..d).map(|i| i as f64 * 0.1).collect();
    let cls_plain = fs.classify(&query).expect("classify should succeed");
    let cls_nc = fs.classify_with_nc_prior(&query, &etf, 0.0)
        .expect("classify_with_nc_prior should succeed");
    assert_eq!(
        cls_plain, cls_nc,
        "alpha=0 NC-prior should equal plain classify"
    );
}

#[test]
fn test_few_shot_k_shot_accuracy_range() {
    let mut seed = make_seed();
    let num_classes = 3;
    let d = 5;
    let (support_feats, support_labels) =
        generate_clustered_features(num_classes, 5, d, 0.01, &mut seed);
    let (query_feats, query_labels) =
        generate_clustered_features(num_classes, 5, d, 0.01, &mut seed);
    let mut fs = NclFewShotNc::new(d);
    fs.fit_support(&support_feats, &support_labels, num_classes)
        .expect("fit_support should succeed");
    let acc = fs.k_shot_accuracy(&query_feats, &query_labels);
    assert!(
        (0.0..=1.0).contains(&acc),
        "Accuracy should be in [0,1], got {}",
        acc
    );
}

// ─── §9 NclFisherRao ─────────────────────────────────────────────────────────

#[test]
fn test_fisher_rao_distance_nonnegative() {
    let mu1 = vec![1.0, 2.0, 3.0];
    let mu2 = vec![4.0, 5.0, 6.0];
    let sigma1 = vec![1.0, 1.0, 1.0];
    let sigma2 = vec![1.0, 1.0, 1.0];
    let d = NclFisherRao::fisher_rao_distance(&mu1, &mu2, &sigma1, &sigma2);
    assert!(
        d >= 0.0,
        "Fisher-Rao distance should be non-negative, got {}",
        d
    );
}

#[test]
fn test_fisher_rao_distance_same_distribution_zero() {
    let mu = vec![1.0, 2.0, 3.0];
    let sigma = vec![1.0, 1.0, 1.0];
    let d = NclFisherRao::fisher_rao_distance(&mu, &mu, &sigma, &sigma);
    assert!(d.abs() < 1e-9, "Distance to itself should be 0, got {}", d);
}

#[test]
fn test_class_separability_nonnegative() {
    let mut seed = make_seed();
    let (features, labels) = generate_clustered_features(3, 10, 4, 0.1, &mut seed);
    let stats = NclNeuralCollapseMetrics::compute_stats(&features, &labels, 3)
        .expect("compute_stats should succeed");
    let sep = NclFisherRao::class_separability(&stats);
    assert!(
        sep >= 0.0,
        "Class separability should be non-negative, got {}",
        sep
    );
}

#[test]
fn test_nc_fisher_index_nonnegative() {
    let mut seed = make_seed();
    let (features, labels) = generate_clustered_features(3, 10, 4, 0.1, &mut seed);
    let stats = NclNeuralCollapseMetrics::compute_stats(&features, &labels, 3)
        .expect("compute_stats should succeed");
    let fi = NclFisherRao::nc_fisher_index(&stats);
    assert!(
        fi >= 0.0,
        "NC Fisher index should be non-negative, got {}",
        fi
    );
}

#[test]
fn test_optimal_transport_distance_nonnegative() {
    let mu1 = vec![0.0, 1.0, 2.0];
    let mu2 = vec![3.0, 4.0, 5.0];
    let d = NclFisherRao::optimal_transport_distance(&mu1, &mu2);
    assert!(d >= 0.0, "OT distance should be non-negative, got {}", d);
}

#[test]
fn test_optimal_transport_distance_correct() {
    let mu1 = vec![0.0, 0.0];
    let mu2 = vec![3.0, 4.0];
    let d = NclFisherRao::optimal_transport_distance(&mu1, &mu2);
    assert!(
        (d - 5.0).abs() < 1e-9,
        "OT distance should be 5.0, got {}",
        d
    );
}

#[test]
fn test_class_separability_increases_with_spread() {
    let mut seed1 = make_seed();
    let mut seed2 = make_seed();
    // More separated classes → higher Fisher-Rao separability
    let (f_close, l_close) = generate_clustered_features(3, 10, 4, 0.01, &mut seed1);
    let (f_far, l_far) = generate_clustered_features(3, 10, 4, 0.01, &mut seed2);
    // Manually shift class means for f_far
    let mut f_far_shifted = f_far.clone();
    for (i, feat) in f_far_shifted.iter_mut().enumerate() {
        let k = l_far[i] as f64;
        feat[0] += k * 5.0; // large shift
    }
    let stats_close = NclNeuralCollapseMetrics::compute_stats(&f_close, &l_close, 3)
        .expect("compute_stats for close features should succeed");
    let stats_far = NclNeuralCollapseMetrics::compute_stats(&f_far_shifted, &l_far, 3)
        .expect("compute_stats for far features should succeed");
    let sep_close = NclFisherRao::class_separability(&stats_close);
    let sep_far = NclFisherRao::class_separability(&stats_far);
    assert!(
        sep_far >= sep_close,
        "More separated classes should have >= separability: close={}, far={}",
        sep_close,
        sep_far
    );
}

// ─── §10 NclMetrics ──────────────────────────────────────────────────────────

#[test]
fn test_accuracy_from_ncm_in_range() {
    let mut seed = make_seed();
    let num_classes = 3;
    let d = 4;
    let (features, labels) = generate_clustered_features(num_classes, 10, d, 0.1, &mut seed);
    let stats = NclNeuralCollapseMetrics::compute_stats(&features, &labels, num_classes)
        .expect("compute_stats should succeed");
    let acc = NclMetrics::accuracy_from_nearest_class_mean(&stats, &features, &labels);
    assert!(
        (0.0..=1.0).contains(&acc),
        "Accuracy should be in [0,1], got {}",
        acc
    );
}

#[test]
fn test_accuracy_high_for_clean_data() {
    let mut seed = make_seed();
    let num_classes = 3;
    let d = 4;
    let (features, labels) = generate_clustered_features(num_classes, 20, d, 0.001, &mut seed);
    let stats = NclNeuralCollapseMetrics::compute_stats(&features, &labels, num_classes)
        .expect("compute_stats should succeed");
    let acc = NclMetrics::accuracy_from_nearest_class_mean(&stats, &features, &labels);
    assert!(
        acc > 0.9,
        "Clean data should give high accuracy, got {}",
        acc
    );
}

#[test]
fn test_variability_collapse_ratio_collapsed() {
    assert_eq!(NclMetrics::variability_collapse_ratio(0.005), "collapsed");
}

#[test]
fn test_variability_collapse_ratio_collapsing() {
    assert_eq!(NclMetrics::variability_collapse_ratio(0.05), "collapsing");
}

#[test]
fn test_variability_collapse_ratio_partial() {
    assert_eq!(NclMetrics::variability_collapse_ratio(0.3), "partial");
}

#[test]
fn test_variability_collapse_ratio_not_collapsed() {
    assert_eq!(NclMetrics::variability_collapse_ratio(1.0), "not_collapsed");
}

#[test]
fn test_etf_angle_deviation_nonnegative() {
    let mut seed = make_seed();
    let num_classes = 4;
    let etf = NclEquiangularTightFrame::new(num_classes, 8, &mut seed)
        .expect("ETF creation should succeed");
    let gram = etf.gram_matrix();
    let dev = NclMetrics::etf_angle_deviation(&gram, num_classes);
    assert!(
        dev >= 0.0,
        "ETF angle deviation should be non-negative, got {}",
        dev
    );
}

#[test]
fn test_etf_angle_deviation_identity_matrix() {
    // Identity matrix → G[i][i]=1, G[i][j]=0; target off-diag = -1/(num_classes-1)
    let num_classes = 3;
    let gram = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let dev = NclMetrics::etf_angle_deviation(&gram, num_classes);
    // Expected: off-diag deviation = |0.0 - (-0.5)| = 0.5 per pair
    let expected_off = 0.5f64;
    // 6 off-diag entries (all i!=j pairs), 3 diag (all 0 deviation), mean = 6*0.5/9
    let expected = (6.0 * expected_off) / 9.0;
    assert!(
        (dev - expected).abs() < 1e-9,
        "Deviation = {}, expected {}",
        dev,
        expected
    );
}

#[test]
fn test_top_k_class_confusion_length() {
    let mut seed = make_seed();
    let num_classes = 4;
    let d = 4;
    let (features, labels) = generate_clustered_features(num_classes, 10, d, 0.1, &mut seed);
    let stats = NclNeuralCollapseMetrics::compute_stats(&features, &labels, num_classes)
        .expect("compute_stats should succeed");
    let confused = NclMetrics::top_k_class_confusion(&stats, 3);
    assert!(confused.len() <= 3, "Should return at most k pairs");
}

#[test]
fn test_top_k_class_confusion_sorted() {
    let mut seed = make_seed();
    let num_classes = 4;
    let d = 4;
    let (features, labels) = generate_clustered_features(num_classes, 10, d, 0.1, &mut seed);
    let stats = NclNeuralCollapseMetrics::compute_stats(&features, &labels, num_classes)
        .expect("compute_stats should succeed");
    let confused = NclMetrics::top_k_class_confusion(&stats, 6);
    // Should be sorted ascending by distance
    for i in 1..confused.len() {
        assert!(
            confused[i].2 >= confused[i - 1].2,
            "Pairs should be sorted by ascending distance"
        );
    }
}

#[test]
fn test_collapse_phase_detection_finds_onset() {
    let nc1_traj = vec![2.0, 1.5, 0.8, 0.3, 0.05, 0.01];
    let onset = NclMetrics::collapse_phase_detection(&nc1_traj);
    assert_eq!(
        onset,
        Some(4),
        "Collapse onset should be at index 4 (0.05 < 0.1)"
    );
}

#[test]
fn test_collapse_phase_detection_no_onset() {
    let nc1_traj = vec![2.0, 1.5, 0.8, 0.3, 0.2];
    let onset = NclMetrics::collapse_phase_detection(&nc1_traj);
    assert_eq!(onset, None, "No onset when NC1 never < 0.1");
}

#[test]
fn test_collapse_phase_detection_empty() {
    let onset = NclMetrics::collapse_phase_detection(&[]);
    assert_eq!(onset, None, "Empty trajectory should give None");
}
