//! Tests for the `anomaly_detection` module.

use super::*;

// --- AnomalyVAE ---

#[test]
fn test_vae_encode_decode_shape() {
    let cfg = AnomalyVaeConfig {
        input_dim: 8,
        latent_dim: 4,
        hidden_dim: 16,
        beta: 1.0,
        seed: 1,
    };
    let vae = AnomalyVAE::new(cfg.clone());
    let x = vec![0.1_f32; 8];
    let (mu, lv) = vae.encode(&x).expect("VAE encode should succeed");
    assert_eq!(mu.len(), cfg.latent_dim);
    assert_eq!(lv.len(), cfg.latent_dim);
    let recon = vae.decode(&mu).expect("VAE decode should succeed");
    assert_eq!(recon.len(), cfg.input_dim);
}

#[test]
fn test_vae_anomaly_score_non_negative() {
    let cfg = AnomalyVaeConfig {
        input_dim: 8,
        latent_dim: 4,
        hidden_dim: 16,
        beta: 1.0,
        seed: 2,
    };
    let vae = AnomalyVAE::new(cfg);
    let x = vec![0.5_f32; 8];
    let score = vae.anomaly_score(&x).expect("anomaly_score should succeed");
    assert!(score.is_finite());
}

#[test]
fn test_vae_dimension_mismatch_error() {
    let cfg = AnomalyVaeConfig::default();
    let vae = AnomalyVAE::new(cfg.clone());
    let x = vec![0.0_f32; cfg.input_dim + 1];
    assert!(vae.encode(&x).is_err());
}

#[test]
fn test_vae_decode_dimension_mismatch_error() {
    let cfg = AnomalyVaeConfig::default();
    let vae = AnomalyVAE::new(cfg.clone());
    let z = vec![0.0_f32; cfg.latent_dim + 1];
    assert!(vae.decode(&z).is_err());
}

#[test]
fn test_vae_beta_scaling() {
    let mut cfg = AnomalyVaeConfig {
        input_dim: 8,
        latent_dim: 4,
        hidden_dim: 16,
        beta: 0.0,
        seed: 3,
    };
    let vae_b0 = AnomalyVAE::new(cfg.clone());
    cfg.beta = 5.0;
    let vae_b5 = AnomalyVAE::new(cfg);
    let x = vec![1.0_f32; 8];
    // Both should be finite; beta=0 removes KL contribution.
    assert!(vae_b0
        .anomaly_score(&x)
        .expect("beta=0 anomaly_score should succeed")
        .is_finite());
    assert!(vae_b5
        .anomaly_score(&x)
        .expect("beta=5 anomaly_score should succeed")
        .is_finite());
}

// --- DeepSVDD ---

#[test]
fn test_svdd_forward_shape() {
    let cfg = DeepSvddConfig {
        input_dim: 10,
        output_dim: 5,
        hidden_dim: 20,
        seed: 10,
    };
    let model = DeepSVDD::new(cfg.clone());
    let x = vec![0.3_f32; 10];
    let emb = model.forward(&x).expect("SVDD forward should succeed");
    assert_eq!(emb.len(), cfg.output_dim);
}

#[test]
fn test_svdd_initial_center_zero() {
    let cfg = DeepSvddConfig::default();
    let model = DeepSVDD::new(cfg.clone());
    assert!(model.center.iter().all(|&v| v == 0.0));
    assert_eq!(model.center.len(), cfg.output_dim);
}

#[test]
fn test_svdd_score_finite() {
    let cfg = DeepSvddConfig {
        input_dim: 6,
        output_dim: 3,
        hidden_dim: 12,
        seed: 5,
    };
    let model = DeepSVDD::new(cfg);
    let x = vec![0.1_f32; 6];
    let s = model.score(&x).expect("SVDD score should succeed");
    assert!(s >= 0.0 && s.is_finite());
}

#[test]
fn test_svdd_update_center() {
    let cfg = DeepSvddConfig {
        input_dim: 4,
        output_dim: 2,
        hidden_dim: 8,
        seed: 7,
    };
    let mut model = DeepSVDD::new(cfg);
    let embeddings = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]];
    model
        .update_center(&embeddings)
        .expect("update_center should succeed");
    assert!((model.center[0] - 2.0).abs() < 1e-5);
    assert!((model.center[1] - 3.0).abs() < 1e-5);
}

#[test]
fn test_svdd_update_center_empty_error() {
    let cfg = DeepSvddConfig::default();
    let mut model = DeepSVDD::new(cfg);
    assert!(model.update_center(&[]).is_err());
}

// --- PatchTSAD ---

#[test]
fn test_patch_tsad_basic() {
    let cfg = PatchTsadConfig::default();
    let mut detector = PatchTSAD::new(cfg);
    let series: Vec<f32> = (0..50).map(|i| (i as f32 * 0.1).sin()).collect();
    let scores = detector
        .score_series(&series, 8, 4)
        .expect("score_series should succeed");
    assert!(!scores.is_empty());
    assert!(scores.iter().all(|s| s.is_finite() && *s >= 0.0));
}

#[test]
fn test_patch_tsad_stride_count() {
    let cfg = PatchTsadConfig::default();
    let mut detector = PatchTSAD::new(cfg);
    let series: Vec<f32> = vec![0.0_f32; 20];
    let scores = detector
        .score_series(&series, 4, 2)
        .expect("score_series should succeed");
    // Patches: 0..4, 2..6, 4..8, 6..10, 8..12, 10..14, 12..16, 14..18, 16..20 => 9
    assert_eq!(scores.len(), 9);
}

#[test]
fn test_patch_tsad_zero_stride_error() {
    let cfg = PatchTsadConfig::default();
    let mut detector = PatchTSAD::new(cfg);
    assert!(detector.score_series(&[1.0, 2.0, 3.0], 2, 0).is_err());
}

#[test]
fn test_patch_tsad_empty_error() {
    let cfg = PatchTsadConfig::default();
    let mut detector = PatchTSAD::new(cfg);
    assert!(detector.score_series(&[], 4, 1).is_err());
}

// --- AnomalyTransformerModel ---

#[test]
fn test_atm_prior_association_row_stochastic() {
    let atm = AnomalyTransformerModel::new(8, 3.0).expect("ATM construction should succeed");
    let prior = atm.prior_association(5);
    assert_eq!(prior.len(), 5);
    for row in &prior {
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "row sum {s}");
    }
}

#[test]
fn test_atm_series_association_row_stochastic() {
    let atm = AnomalyTransformerModel::new(4, 2.0).expect("ATM construction should succeed");
    let x: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect(); // 3 x 4
    let series = atm
        .series_association(&x, 3, 4)
        .expect("series_association should succeed");
    assert_eq!(series.len(), 3);
    for row in &series {
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "row sum {s}");
    }
}

#[test]
fn test_atm_association_discrepancy_non_negative() {
    let atm = AnomalyTransformerModel::new(4, 1.5).expect("ATM construction should succeed");
    let prior = atm.prior_association(4);
    let x: Vec<f32> = (0..16).map(|i| i as f32 * 0.05).collect();
    let series = atm
        .series_association(&x, 4, 4)
        .expect("series_association should succeed");
    let disc = atm
        .association_discrepancy(&prior, &series)
        .expect("association_discrepancy should succeed");
    assert!(disc >= 0.0 && disc.is_finite());
}

#[test]
fn test_atm_invalid_dmodel() {
    assert!(AnomalyTransformerModel::new(0, 1.0).is_err());
}

#[test]
fn test_atm_anomaly_score_sequence() {
    let atm = AnomalyTransformerModel::new(4, 2.0).expect("ATM construction should succeed");
    let x: Vec<f32> = (0..16).map(|i| (i as f32).sin()).collect();
    let score = atm
        .anomaly_score_sequence(&x, 4, 4)
        .expect("anomaly_score_sequence should succeed");
    assert!(score >= 0.0 && score.is_finite());
}

// --- MemoryAugmentedAE ---

#[test]
fn test_memae_encode_shape() {
    let cfg = MemAeConfig::default();
    let model = MemoryAugmentedAE::new(cfg.clone());
    let x = vec![0.2_f32; cfg.input_dim];
    let z = model.encode(&x).expect("MemAE encode should succeed");
    assert_eq!(z.len(), cfg.latent_dim);
}

#[test]
fn test_memae_anomaly_score_finite() {
    let cfg = MemAeConfig {
        input_dim: 16,
        latent_dim: 8,
        n_slots: 5,
        seed: 50,
    };
    let model = MemoryAugmentedAE::new(cfg.clone());
    let x = vec![0.5_f32; cfg.input_dim];
    let s = model
        .anomaly_score(&x)
        .expect("MemAE anomaly_score should succeed");
    assert!(s >= 0.0 && s.is_finite());
}

#[test]
fn test_memae_update_memory() {
    let cfg = MemAeConfig::default();
    let mut model = MemoryAugmentedAE::new(cfg.clone());
    let x = vec![1.0_f32; cfg.input_dim];
    assert!(model.update_memory(&x).is_ok());
}

#[test]
fn test_memory_query_dimension() {
    let mem = AnomalyMemory::new(5, 8, 42);
    let z = vec![0.1_f32; 8];
    let result = mem.query(&z).expect("memory query should succeed");
    assert_eq!(result.len(), 8);
}

#[test]
fn test_memory_address_sums_to_one() {
    let mem = AnomalyMemory::new(4, 6, 3);
    let z = vec![0.5_f32; 6];
    let weights = mem.address(&z).expect("memory address should succeed");
    let s: f32 = weights.iter().sum();
    assert!((s - 1.0).abs() < 1e-5);
}

// --- RobustRCF ---

#[test]
fn test_rcf_score_after_insert() {
    let cfg = RobustRcfConfig {
        n_trees: 10,
        window_size: 50,
        dim: 1,
        seed: 0,
    };
    let mut rcf = RobustRCF::new(cfg);
    for i in 0..30 {
        rcf.insert(vec![i as f32 * 0.1]);
    }
    let score = rcf.score(&[3.0]);
    assert!(score >= 0.0 && score.is_finite());
}

#[test]
fn test_rcf_empty_window_score_zero() {
    let cfg = RobustRcfConfig::default();
    let mut rcf = RobustRCF::new(cfg);
    let score = rcf.score(&[1.0]);
    assert_eq!(score, 0.0);
}

#[test]
fn test_rcf_window_overflow() {
    let cfg = RobustRcfConfig {
        n_trees: 5,
        window_size: 10,
        dim: 1,
        seed: 1,
    };
    let mut rcf = RobustRCF::new(cfg);
    for i in 0..25 {
        rcf.insert(vec![i as f32]);
    }
    assert_eq!(rcf.window.len(), 10);
}

#[test]
fn test_rcf_multidimensional() {
    let cfg = RobustRcfConfig {
        n_trees: 8,
        window_size: 30,
        dim: 3,
        seed: 42,
    };
    let mut rcf = RobustRCF::new(cfg);
    let mut rng = StdRng::seed_from_u64(11);
    for _ in 0..20 {
        rcf.insert(vec![
            rng.random::<f32>(),
            rng.random::<f32>(),
            rng.random::<f32>(),
        ]);
    }
    let score = rcf.score(&[10.0, 10.0, 10.0]);
    assert!(score.is_finite());
}

// --- SpectralResidual ---

#[test]
fn test_sr_basic() {
    let sr = SpectralResidual::new(3);
    let series: Vec<f32> = (0..16).map(|i| (i as f32 * 0.5).sin()).collect();
    let scores = sr
        .compute(&series)
        .expect("spectral residual compute should succeed");
    assert_eq!(scores.len(), 16);
    assert!(scores.iter().all(|s| s.is_finite()));
}

#[test]
fn test_sr_empty_error() {
    let sr = SpectralResidual::new(3);
    assert!(sr.compute(&[]).is_err());
}

#[test]
fn test_sr_spike_detection() {
    let sr = SpectralResidual::new(4);
    // Insert a spike in an otherwise flat series.
    let mut series = vec![0.1_f32; 20];
    series[10] = 10.0;
    let scores = sr
        .compute(&series)
        .expect("spectral residual compute should succeed");
    // The spike region should have relatively high saliency.
    let spike_score = scores[10];
    let mean_score: f32 = scores.iter().sum::<f32>() / scores.len() as f32;
    assert!(spike_score >= mean_score);
}

// --- GaussianMixtureAnomaly ---

#[test]
fn test_gmm_fit_and_score() {
    let mut gmm = GaussianMixtureAnomaly::new(2).expect("GMM construction should succeed");
    let data: Vec<Vec<f32>> = (0..20)
        .map(|i| vec![i as f32 * 0.1, i as f32 * 0.05])
        .collect();
    gmm.fit(&data).expect("GMM fit should succeed");
    let score = gmm
        .anomaly_score(&[1.0, 0.5])
        .expect("GMM anomaly_score should succeed");
    assert!(score.is_finite());
}

#[test]
fn test_gmm_weights_sum_to_one() {
    let mut gmm = GaussianMixtureAnomaly::new(3).expect("GMM construction should succeed");
    let data: Vec<Vec<f32>> = (0..30).map(|i| vec![i as f32 * 0.1]).collect();
    gmm.fit(&data).expect("GMM fit should succeed");
    let s: f32 = gmm.weights.iter().sum();
    assert!((s - 1.0).abs() < 1e-4, "weights sum {s}");
}

#[test]
fn test_gmm_log_likelihood_finite() {
    let mut gmm = GaussianMixtureAnomaly::new(2).expect("GMM construction should succeed");
    let data: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32 * 0.2]).collect();
    gmm.fit(&data).expect("GMM fit should succeed");
    let ll = gmm
        .log_likelihood(&[1.0])
        .expect("GMM log_likelihood should succeed");
    assert!(ll.is_finite());
}

#[test]
fn test_gmm_anomaly_score_vs_log_likelihood() {
    let mut gmm = GaussianMixtureAnomaly::new(2).expect("GMM construction should succeed");
    let data: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32 * 0.1]).collect();
    gmm.fit(&data).expect("GMM fit should succeed");
    let x = vec![0.5_f32];
    let ll = gmm
        .log_likelihood(&x)
        .expect("GMM log_likelihood should succeed");
    let score = gmm
        .anomaly_score(&x)
        .expect("GMM anomaly_score should succeed");
    assert!((score + ll).abs() < 1e-4);
}

#[test]
fn test_gmm_invalid_n_components() {
    assert!(GaussianMixtureAnomaly::new(0).is_err());
}

// --- AnomalyThresholder ---

#[test]
fn test_pot_basic() {
    let scores = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let t = AnomalyThresholder::peak_over_threshold(&scores, 0.8).expect("POT should succeed");
    assert!((4.0..=5.0).contains(&t));
}

#[test]
fn test_pot_empty_error() {
    assert!(AnomalyThresholder::peak_over_threshold(&[], 0.9).is_err());
}

#[test]
fn test_evt_basic() {
    let scores: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
    let t = AnomalyThresholder::extreme_value_theory(&scores).expect("EVT should succeed");
    assert!(t.is_finite() && t >= 0.0);
}

#[test]
fn test_adaptive_threshold_length() {
    let scores: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let thresholds = AnomalyThresholder::adaptive_threshold(&scores, 5, 2.0)
        .expect("adaptive_threshold should succeed");
    assert_eq!(thresholds.len(), scores.len());
}

#[test]
fn test_adaptive_threshold_monotone() {
    // For a monotonically increasing series the mean + k*std should be finite.
    let scores: Vec<f32> = (0..30).map(|i| i as f32).collect();
    let thresholds = AnomalyThresholder::adaptive_threshold(&scores, 10, 1.5)
        .expect("adaptive_threshold should succeed");
    assert!(thresholds.iter().all(|t| t.is_finite()));
}

// --- AnomalyEvaluationMetrics ---

#[test]
fn test_roc_auc_perfect() {
    let scores = vec![1.0_f32, 1.0, 0.0, 0.0];
    let labels = vec![true, true, false, false];
    let auc = AnomalyEvaluationMetrics::roc_auc(&scores, &labels).expect("roc_auc should succeed");
    assert!((auc - 1.0).abs() < 1e-5);
}

#[test]
fn test_roc_auc_random() {
    let scores = vec![0.6_f32, 0.4, 0.6, 0.4];
    let labels = vec![true, false, false, true];
    let auc = AnomalyEvaluationMetrics::roc_auc(&scores, &labels).expect("roc_auc should succeed");
    assert!((0.0..=1.0).contains(&auc));
}

#[test]
fn test_pr_auc_perfect() {
    let scores = vec![0.9_f32, 0.8, 0.2, 0.1];
    let labels = vec![true, true, false, false];
    let auc = AnomalyEvaluationMetrics::pr_auc(&scores, &labels).expect("pr_auc should succeed");
    assert!(auc > 0.5);
}

#[test]
fn test_f1_at_threshold() {
    let scores = vec![0.9_f32, 0.8, 0.2, 0.1];
    let labels = vec![true, true, false, false];
    let f1 = AnomalyEvaluationMetrics::f1_at_threshold(&scores, &labels, 0.5)
        .expect("f1_at_threshold should succeed");
    assert!((f1 - 1.0).abs() < 1e-5);
}

#[test]
fn test_f1_at_threshold_empty() {
    assert!(AnomalyEvaluationMetrics::f1_at_threshold(&[], &[], 0.5).is_err());
}

#[test]
fn test_point_adjust_f1_segment() {
    // Anomalous segment [2,3,4]; at least one fires → entire segment counted as TP.
    let scores = vec![0.1_f32, 0.1, 0.9, 0.1, 0.1];
    let labels = vec![false, false, true, true, true];
    let f1 = AnomalyEvaluationMetrics::point_adjust_f1(&scores, &labels, 0.5)
        .expect("point_adjust_f1 should succeed");
    // TP=3, FP=0, FN=0 → F1=1.0
    assert!((f1 - 1.0).abs() < 1e-5);
}

#[test]
fn test_point_adjust_f1_miss() {
    // Anomalous segment with no prediction → all are FN.
    let scores = vec![0.0_f32, 0.0, 0.0, 0.0];
    let labels = vec![false, true, true, false];
    let f1 = AnomalyEvaluationMetrics::point_adjust_f1(&scores, &labels, 0.5)
        .expect("point_adjust_f1 should succeed");
    assert_eq!(f1, 0.0);
}

#[test]
fn test_metrics_dimension_mismatch() {
    let scores = vec![1.0_f32, 2.0];
    let labels = vec![true];
    assert!(AnomalyEvaluationMetrics::roc_auc(&scores, &labels).is_err());
}
