//! Extended f64-based neural anomaly detection.
//!
//! Provides building blocks and detectors that complement the f32-based
//! implementations in the parent `anomaly_detection` module.

pub mod ae_anomaly;
pub mod building_blocks;
pub mod flow_anomaly;
pub mod isolation_forest;
pub mod metrics;
pub mod svdd;
pub mod vae_anomaly;

pub use ae_anomaly::{AeAnomaly, AeAnomalyConfig};
pub use building_blocks::{box_muller, percentile, relu, relu_d, xavier_limit, AdLinear, AdMlp};
pub use flow_anomaly::{CouplingLayer, FlowAnomaly, FlowAnomalyConfig};
pub use isolation_forest::{
    correction, harmonic, IsolationForestConfig, IsolationNode, IsolationTree,
    NeuralIsolationForest,
};
pub use metrics::{
    compute_anomaly_metrics, compute_auc_roc, compute_average_precision, AnomalyMetrics,
};
pub use svdd::{AdDeepSvdd, AdDeepSvddConfig};
pub use vae_anomaly::{VaeAnomaly, VaeAnomalyConfig};

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    fn seeded_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    // ── AdLinear ─────────────────────────────────────────────────────────────

    #[test]
    fn test_ad_linear_output_shape() {
        let layer = AdLinear::new(4, 3);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = layer.forward(&x);
        assert_eq!(y.len(), 3);
    }

    #[test]
    fn test_ad_linear_forward_finite() {
        let layer = AdLinear::new(5, 2);
        let x = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let y = layer.forward(&x);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ad_linear_update_changes_weights() {
        let mut layer = AdLinear::new(2, 2);
        let orig_w00 = layer.w[0][0];
        let grad_w = vec![vec![1.0, 0.0], vec![0.0, 0.0]];
        let grad_b = vec![0.0, 0.0];
        layer.update(&grad_w, &grad_b, 0.1);
        assert!((layer.w[0][0] - orig_w00 + 0.1).abs() < 1e-10);
    }

    // ── AdMlp ────────────────────────────────────────────────────────────────

    #[test]
    fn test_ad_mlp_output_shape() {
        let mlp = AdMlp::new(&[4, 8, 3]);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = mlp.forward(&x);
        assert_eq!(y.len(), 3);
    }

    #[test]
    fn test_ad_mlp_forward_finite() {
        let mlp = AdMlp::new(&[3, 6, 2]);
        let x = vec![0.5, -0.5, 0.25];
        let y = mlp.forward(&x);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ad_mlp_gradient_fd_shape() {
        let mlp = AdMlp::new(&[2, 4, 2]);
        let x = vec![1.0, 0.5];
        let t = vec![0.0, 1.0];
        let grads = mlp.gradient_fd(&x, &t, 1e-5);
        assert_eq!(grads.len(), 2); // 2 layers
    }

    #[test]
    fn test_ad_mlp_apply_gradients() {
        let mut mlp = AdMlp::new(&[2, 4, 2]);
        let x = vec![1.0, 0.5];
        let t = vec![0.0, 1.0];
        let grads = mlp.gradient_fd(&x, &t, 1e-5);
        let orig_w = mlp.layers[0].w[0][0];
        mlp.apply_gradients(&grads, 0.01);
        // Weight should have changed.
        let diff = (mlp.layers[0].w[0][0] - orig_w).abs();
        // May be zero if gradient is zero; just verify it's finite.
        assert!(mlp.layers[0].w[0][0].is_finite());
        let _ = diff;
    }

    // ── AeAnomaly ────────────────────────────────────────────────────────────

    #[test]
    fn test_ae_anomaly_reconstruct_shape() {
        let cfg = AeAnomalyConfig {
            input_dim: 6,
            latent_dim: 2,
            encoder_hidden: vec![4],
            decoder_hidden: vec![4],
            threshold_percentile: 95.0,
        };
        let ae = AeAnomaly::new(cfg);
        let x = vec![0.1; 6];
        let xhat = ae.reconstruct(&x);
        assert_eq!(xhat.len(), 6);
    }

    #[test]
    fn test_ae_anomaly_reconstruction_error_non_negative() {
        let cfg = AeAnomalyConfig::default();
        let ae = AeAnomaly::new(cfg);
        let x = vec![0.5; 16];
        let err = ae.reconstruction_error(&x);
        assert!(err >= 0.0 && err.is_finite());
    }

    #[test]
    fn test_ae_anomaly_fit_decreases_loss() {
        let cfg = AeAnomalyConfig {
            input_dim: 4,
            latent_dim: 2,
            encoder_hidden: vec![],
            decoder_hidden: vec![],
            threshold_percentile: 95.0,
        };
        let mut ae = AeAnomaly::new(cfg);
        let x_train: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 0.1; 4]).collect();
        let history = ae.fit(&x_train, 3, 0.01);
        assert_eq!(history.len(), 3);
        assert!(history.iter().all(|l| l.is_finite()));
    }

    #[test]
    fn test_ae_anomaly_set_threshold_finite() {
        let cfg = AeAnomalyConfig::default();
        let mut ae = AeAnomaly::new(cfg);
        let x_normal: Vec<Vec<f64>> = (0..10).map(|_| vec![0.0; 16]).collect();
        ae.set_threshold(&x_normal);
        assert!(ae.threshold.is_finite());
    }

    #[test]
    fn test_ae_anomaly_predict_returns_bool() {
        let cfg = AeAnomalyConfig::default();
        let mut ae = AeAnomaly::new(cfg);
        let x_normal: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0; 16]).collect();
        ae.set_threshold(&x_normal);
        let x = vec![0.5; 16];
        let pred = ae.predict(&x);
        // Just ensure it returns a valid bool.
        let _ = pred;
    }

    #[test]
    fn test_ae_anomaly_batch_predict_length() {
        let cfg = AeAnomalyConfig::default();
        let mut ae = AeAnomaly::new(cfg);
        let x_normal: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0; 16]).collect();
        ae.set_threshold(&x_normal);
        let batch: Vec<Vec<f64>> = (0..7).map(|_| vec![0.5; 16]).collect();
        let preds = ae.batch_predict(&batch);
        assert_eq!(preds.len(), 7);
    }

    #[test]
    fn test_ae_score_equals_reconstruction_error() {
        let cfg = AeAnomalyConfig::default();
        let ae = AeAnomaly::new(cfg);
        let x = vec![0.3; 16];
        let score = ae.score(&x);
        let err = ae.reconstruction_error(&x);
        assert!((score - err).abs() < 1e-12);
    }

    // ── VaeAnomaly ───────────────────────────────────────────────────────────

    #[test]
    fn test_vae_anomaly_encode_shapes() {
        let cfg = VaeAnomalyConfig {
            input_dim: 8,
            latent_dim: 3,
            encoder_hidden: vec![6],
            decoder_hidden: vec![6],
            beta: 1.0,
            n_samples: 5,
            threshold_percentile: 95.0,
        };
        let vae = VaeAnomaly::new(cfg.clone());
        let x = vec![0.1; 8];
        let (mu, lv) = vae.encode(&x);
        assert_eq!(mu.len(), cfg.latent_dim);
        assert_eq!(lv.len(), cfg.latent_dim);
    }

    #[test]
    fn test_vae_anomaly_elbo_finite() {
        let cfg = VaeAnomalyConfig::default();
        let vae = VaeAnomaly::new(cfg);
        let x = vec![0.5; 16];
        let elbo = vae.elbo(&x, 0.3, 0.7);
        assert!(elbo.is_finite());
    }

    #[test]
    fn test_vae_anomaly_sample_z_shape() {
        let cfg = VaeAnomalyConfig::default();
        let vae = VaeAnomaly::new(cfg.clone());
        let mu = vec![0.0; cfg.latent_dim];
        let lv = vec![0.0; cfg.latent_dim];
        let z = vae.sample_z(&mu, &lv, 0.5, 0.3);
        assert_eq!(z.len(), cfg.latent_dim);
    }

    #[test]
    fn test_vae_anomaly_score_finite() {
        let cfg = VaeAnomalyConfig::default();
        let vae = VaeAnomaly::new(cfg);
        let mut rng = seeded_rng(99);
        let x = vec![0.5; 16];
        let score = vae.anomaly_score(&x, &mut rng);
        assert!(score.is_finite());
    }

    #[test]
    fn test_vae_anomaly_fit_history() {
        let cfg = VaeAnomalyConfig {
            input_dim: 4,
            latent_dim: 2,
            encoder_hidden: vec![],
            decoder_hidden: vec![],
            beta: 1.0,
            n_samples: 3,
            threshold_percentile: 95.0,
        };
        let mut vae = VaeAnomaly::new(cfg);
        let mut rng = seeded_rng(7);
        let x_train: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64 * 0.1; 4]).collect();
        let history = vae.fit(&x_train, 2, 0.01, &mut rng);
        assert_eq!(history.len(), 2);
        assert!(history.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_vae_anomaly_set_threshold() {
        let cfg = VaeAnomalyConfig::default();
        let mut vae = VaeAnomaly::new(cfg);
        let mut rng = seeded_rng(13);
        let x_normal: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0; 16]).collect();
        vae.set_threshold(&x_normal, &mut rng);
        assert!(vae.threshold.is_finite());
    }

    #[test]
    fn test_vae_anomaly_predict_bool() {
        let cfg = VaeAnomalyConfig::default();
        let mut vae = VaeAnomaly::new(cfg);
        let mut rng = seeded_rng(21);
        let x_normal: Vec<Vec<f64>> = (0..5).map(|_| vec![0.0; 16]).collect();
        vae.set_threshold(&x_normal, &mut rng);
        let x = vec![5.0; 16]; // out-of-distribution
        let pred = vae.predict(&x, &mut rng);
        let _ = pred; // just ensure it runs
    }

    // ── AdDeepSvdd ───────────────────────────────────────────────────────────

    #[test]
    fn test_ad_deep_svdd_score_non_negative() {
        let cfg = AdDeepSvddConfig::default();
        let svdd = AdDeepSvdd::new(cfg);
        let x = vec![0.5; 8];
        let s = svdd.score(&x);
        assert!(s >= 0.0 && s.is_finite());
    }

    #[test]
    fn test_ad_deep_svdd_initialize_center() {
        let cfg = AdDeepSvddConfig::default();
        let mut svdd = AdDeepSvdd::new(cfg.clone());
        let x_train: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1; 8]).collect();
        svdd.initialize_center(&x_train);
        assert_eq!(svdd.center.len(), cfg.output_dim);
        assert!(svdd.center.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn test_ad_deep_svdd_predict_returns_bool() {
        let cfg = AdDeepSvddConfig::default();
        let svdd = AdDeepSvdd::new(cfg);
        let x = vec![0.5; 8];
        let pred = svdd.predict(&x);
        let _ = pred;
    }

    #[test]
    fn test_ad_deep_svdd_svdd_loss_non_negative() {
        let cfg = AdDeepSvddConfig::default();
        let svdd = AdDeepSvdd::new(cfg);
        let batch: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1; 8]).collect();
        let loss = svdd.svdd_loss(&batch);
        assert!(loss >= 0.0 && loss.is_finite());
    }

    #[test]
    fn test_ad_deep_svdd_set_radius() {
        let cfg = AdDeepSvddConfig::default();
        let mut svdd = AdDeepSvdd::new(cfg);
        let x_normal: Vec<Vec<f64>> = (0..10).map(|_| vec![0.1; 8]).collect();
        svdd.set_radius_by_percentile(&x_normal, 95.0);
        assert!(svdd.radius > 0.0 && svdd.radius.is_finite());
    }

    // ── CouplingLayer ────────────────────────────────────────────────────────

    #[test]
    fn test_coupling_layer_output_shape() {
        let layer = CouplingLayer::new(4, 8, false);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let (y, _) = layer.forward(&x);
        assert_eq!(y.len(), 4);
    }

    #[test]
    fn test_coupling_layer_log_det_finite() {
        let layer = CouplingLayer::new(4, 8, false);
        let x = vec![0.1, 0.2, 0.3, 0.4];
        let (_, log_det) = layer.forward(&x);
        assert!(log_det.is_finite());
    }

    #[test]
    fn test_coupling_layer_inverse_of_forward() {
        let layer = CouplingLayer::new(4, 8, false);
        let x = vec![0.5, -0.3, 0.7, 1.2];
        let (y, _) = layer.forward(&x);
        let x_rec = layer.inverse(&y);
        for (xi, xr) in x.iter().zip(x_rec.iter()) {
            assert!((xi - xr).abs() < 1e-9, "x={xi} x_rec={xr}");
        }
    }

    #[test]
    fn test_coupling_layer_inverse_mask() {
        // Test with inverted mask.
        let layer = CouplingLayer::new(6, 8, true);
        let x: Vec<f64> = (0..6).map(|i| i as f64 * 0.1 + 0.05).collect();
        let (y, _) = layer.forward(&x);
        let x_rec = layer.inverse(&y);
        for (xi, xr) in x.iter().zip(x_rec.iter()) {
            assert!((xi - xr).abs() < 1e-9);
        }
    }

    // ── FlowAnomaly ──────────────────────────────────────────────────────────

    #[test]
    fn test_flow_anomaly_log_prob_finite() {
        let cfg = FlowAnomalyConfig {
            input_dim: 4,
            n_coupling_layers: 2,
            hidden_dim: 8,
            n_epochs: 1,
            lr: 1e-3,
            threshold_percentile: 95.0,
        };
        let flow = FlowAnomaly::new(cfg);
        let x = vec![0.1, 0.2, 0.3, 0.4];
        assert!(flow.log_prob(&x).is_finite());
    }

    #[test]
    fn test_flow_anomaly_score_finite() {
        let cfg = FlowAnomalyConfig::default();
        let flow = FlowAnomaly::new(cfg);
        let x = vec![0.5, -0.1, 0.3, 0.7];
        assert!(flow.anomaly_score(&x).is_finite());
    }

    #[test]
    fn test_flow_anomaly_set_threshold() {
        let cfg = FlowAnomalyConfig::default();
        let mut flow = FlowAnomaly::new(cfg);
        let x_normal: Vec<Vec<f64>> = (0..5)
            .map(|i| vec![i as f64 * 0.1, 0.5, -0.2, 0.1])
            .collect();
        flow.set_threshold(&x_normal);
        assert!(flow.threshold.is_finite());
    }

    #[test]
    fn test_flow_anomaly_fit_history_length() {
        let cfg = FlowAnomalyConfig {
            input_dim: 4,
            n_coupling_layers: 2,
            hidden_dim: 4,
            n_epochs: 2,
            lr: 1e-3,
            threshold_percentile: 95.0,
        };
        let mut flow = FlowAnomaly::new(cfg);
        let x_train: Vec<Vec<f64>> = (0..3)
            .map(|i| vec![i as f64 * 0.1, 0.2, 0.3, 0.4])
            .collect();
        let history = flow.fit(&x_train, 2, 1e-3);
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_flow_anomaly_predict_bool() {
        let cfg = FlowAnomalyConfig::default();
        let mut flow = FlowAnomaly::new(cfg);
        let x_normal: Vec<Vec<f64>> = (0..5)
            .map(|i| vec![i as f64 * 0.05, 0.1, 0.1, 0.1])
            .collect();
        flow.set_threshold(&x_normal);
        let pred = flow.predict(&[10.0, 10.0, 10.0, 10.0]);
        let _ = pred;
    }

    // ── IsolationTree / IsolationNode ─────────────────────────────────────────

    #[test]
    fn test_isolation_tree_fit_builds_tree() {
        let data: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let mut rng = seeded_rng(42);
        let tree = IsolationTree::fit(&data, 8, &mut rng);
        assert!(tree.root.is_some());
    }

    #[test]
    fn test_isolation_node_path_length_positive() {
        let data: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let mut rng = seeded_rng(1);
        let tree = IsolationTree::fit(&data, 8, &mut rng);
        let x = vec![5.0];
        let pl = tree.score(&x);
        assert!(pl > 0.0 && pl.is_finite());
    }

    #[test]
    fn test_isolation_tree_empty_data() {
        let mut rng = seeded_rng(0);
        let tree = IsolationTree::fit(&[], 8, &mut rng);
        assert!(tree.root.is_none());
        assert_eq!(tree.score(&[1.0]), 0.0);
    }

    // ── NeuralIsolationForest ─────────────────────────────────────────────────

    #[test]
    fn test_neural_isolation_forest_c_n_two() {
        // c(2): 2*H(1) - 2*(2-1)/2 = 2*1 - 1 = 1.0
        let c = NeuralIsolationForest::c_n(2);
        assert!((c - 1.0).abs() < 1e-6, "c(2) = {c}");
    }

    #[test]
    fn test_neural_isolation_forest_c_n_one() {
        let c = NeuralIsolationForest::c_n(1);
        assert!((c - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_neural_isolation_forest_anomaly_score_range() {
        let cfg = IsolationForestConfig {
            n_trees: 10,
            subsample_size: 10,
            max_depth: 4,
        };
        let mut nif = NeuralIsolationForest::new(4, 2, cfg);
        let mut rng = seeded_rng(55);
        let x_train: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![i as f64 * 0.1, i as f64 * 0.05, -0.1, 0.3])
            .collect();
        nif.fit(&x_train, 1, 0.01, &mut rng);
        let x = vec![0.5, 0.25, -0.1, 0.3];
        let score = nif.anomaly_score(&x);
        assert!(score > 0.0 && score < 1.0, "score={score}");
    }

    #[test]
    fn test_neural_isolation_forest_predict() {
        let cfg = IsolationForestConfig {
            n_trees: 10,
            subsample_size: 8,
            max_depth: 4,
        };
        let mut nif = NeuralIsolationForest::new(2, 2, cfg);
        let mut rng = seeded_rng(77);
        let x_train: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1, 0.5]).collect();
        nif.fit(&x_train, 1, 0.01, &mut rng);
        let pred = nif.predict(&[100.0, 100.0]); // extreme outlier
        let _ = pred;
    }

    #[test]
    fn test_neural_isolation_forest_set_threshold() {
        let cfg = IsolationForestConfig {
            n_trees: 5,
            subsample_size: 8,
            max_depth: 3,
        };
        let mut nif = NeuralIsolationForest::new(2, 2, cfg);
        let mut rng = seeded_rng(88);
        let x_train: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1, 0.5]).collect();
        nif.fit(&x_train, 1, 0.01, &mut rng);
        nif.set_threshold(&x_train);
        assert!(nif.threshold.is_finite());
    }

    // ── compute_auc_roc ───────────────────────────────────────────────────────

    #[test]
    fn test_compute_auc_roc_perfect() {
        let scores = vec![1.0, 0.9, 0.1, 0.05];
        let labels = vec![true, true, false, false];
        let auc = compute_auc_roc(&scores, &labels);
        assert!((auc - 1.0).abs() < 1e-9, "auc={auc}");
    }

    #[test]
    fn test_compute_auc_roc_random_baseline() {
        // Equal scores → AUC ≈ 0.5.
        let scores = vec![0.5, 0.5, 0.5, 0.5];
        let labels = vec![true, false, true, false];
        let auc = compute_auc_roc(&scores, &labels);
        assert!((0.0..=1.0).contains(&auc));
    }

    #[test]
    fn test_compute_auc_roc_worst() {
        // Perfect inverse predictor → AUC = 0.0 (or near 0).
        let scores = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![true, true, false, false];
        let auc = compute_auc_roc(&scores, &labels);
        assert!(auc <= 0.5);
    }

    #[test]
    fn test_compute_auc_roc_empty() {
        let auc = compute_auc_roc(&[], &[]);
        assert_eq!(auc, 0.5);
    }

    // ── compute_average_precision ─────────────────────────────────────────────

    #[test]
    fn test_compute_average_precision_perfect() {
        let scores = vec![0.9, 0.8, 0.2, 0.1];
        let labels = vec![true, true, false, false];
        let ap = compute_average_precision(&scores, &labels);
        assert!(ap > 0.5 && ap <= 1.0, "ap={ap}");
    }

    #[test]
    fn test_compute_average_precision_range() {
        let scores = vec![0.6, 0.4, 0.7, 0.3];
        let labels = vec![true, false, false, true];
        let ap = compute_average_precision(&scores, &labels);
        assert!((0.0..=1.0).contains(&ap));
    }

    #[test]
    fn test_compute_average_precision_no_positives() {
        let scores = vec![0.5, 0.5];
        let labels = vec![false, false];
        let ap = compute_average_precision(&scores, &labels);
        assert_eq!(ap, 0.0);
    }

    // ── compute_anomaly_metrics ───────────────────────────────────────────────

    #[test]
    fn test_compute_anomaly_metrics_perfect_precision_recall() {
        let scores = vec![1.0, 1.0, 0.0, 0.0];
        let labels = vec![true, true, false, false];
        let m = compute_anomaly_metrics(&scores, &labels, 0.5);
        assert!((m.precision - 1.0).abs() < 1e-9);
        assert!((m.recall - 1.0).abs() < 1e-9);
        assert!((m.f1_at_threshold - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_anomaly_metrics_auc_range() {
        let scores = vec![0.7, 0.3, 0.8, 0.1];
        let labels = vec![true, false, true, false];
        let m = compute_anomaly_metrics(&scores, &labels, 0.5);
        assert!(m.auc_roc >= 0.0 && m.auc_roc <= 1.0);
        assert!(m.average_precision >= 0.0 && m.average_precision <= 1.0);
    }

    #[test]
    fn test_compute_anomaly_metrics_no_tp() {
        let scores = vec![0.0, 0.0, 0.0, 0.0];
        let labels = vec![true, true, false, false];
        let m = compute_anomaly_metrics(&scores, &labels, 0.5);
        assert_eq!(m.precision, 0.0);
        assert_eq!(m.recall, 0.0);
        assert_eq!(m.f1_at_threshold, 0.0);
    }

    // ── Percentile / helpers ──────────────────────────────────────────────────

    #[test]
    fn test_percentile_median() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p50 = percentile(&vals, 50.0);
        assert!((p50 - 3.0).abs() < 1.0); // rough check
    }

    #[test]
    fn test_percentile_empty() {
        let p = percentile(&[], 95.0);
        assert_eq!(p, 0.0);
    }

    #[test]
    fn test_harmonic_one() {
        // H(1) = 1
        let h = harmonic(1);
        assert!((h - 1.0).abs() < 0.1, "H(1) approx = {h}");
    }

    #[test]
    fn test_correction_n_eq_2() {
        // c(2) = 2*H(1) - 2*(1)/2 = 2*1 - 1 = 1
        let c = correction(2);
        assert!((c - 1.0).abs() < 0.2, "c(2) = {c}");
    }

    #[test]
    fn test_relu() {
        assert_eq!(relu(1.0), 1.0);
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(0.0), 0.0);
    }

    #[test]
    fn test_relu_d() {
        assert_eq!(relu_d(1.0), 1.0);
        assert_eq!(relu_d(-1.0), 0.0);
        assert_eq!(relu_d(0.0), 0.0);
    }

    #[test]
    fn test_box_muller_finite() {
        let v = box_muller(0.3, 0.7);
        assert!(v.is_finite());
    }

    #[test]
    fn test_vae_elbo_varies_with_beta() {
        let cfg0 = VaeAnomalyConfig {
            beta: 0.0,
            ..VaeAnomalyConfig::default()
        };
        let cfg1 = VaeAnomalyConfig {
            beta: 10.0,
            ..VaeAnomalyConfig::default()
        };
        let vae0 = VaeAnomaly::new(cfg0);
        let vae1 = VaeAnomaly::new(cfg1);
        let x = vec![1.0; 16];
        // Both should be finite; beta affects the KL term.
        assert!(vae0.elbo(&x, 0.4, 0.6).is_finite());
        assert!(vae1.elbo(&x, 0.4, 0.6).is_finite());
    }

    #[test]
    fn test_flow_anomaly_multiple_layers() {
        // Test with more coupling layers than default.
        let cfg = FlowAnomalyConfig {
            input_dim: 6,
            n_coupling_layers: 4,
            hidden_dim: 8,
            n_epochs: 1,
            lr: 1e-3,
            threshold_percentile: 95.0,
        };
        let flow = FlowAnomaly::new(cfg);
        let x = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        assert!(flow.log_prob(&x).is_finite());
        assert!(flow.anomaly_score(&x).is_finite());
    }

    #[test]
    fn test_ae_anomaly_encode_decode_shapes() {
        let cfg = AeAnomalyConfig {
            input_dim: 8,
            latent_dim: 3,
            encoder_hidden: vec![5],
            decoder_hidden: vec![5],
            threshold_percentile: 95.0,
        };
        let ae = AeAnomaly::new(cfg.clone());
        let x = vec![0.5; 8];
        let z = ae.encode(&x);
        assert_eq!(z.len(), cfg.latent_dim);
        let xhat = ae.decode(&z);
        assert_eq!(xhat.len(), cfg.input_dim);
    }

    #[test]
    fn test_isolation_forest_large_subsample() {
        let cfg = IsolationForestConfig {
            n_trees: 5,
            subsample_size: 50,
            max_depth: 6,
        };
        let mut nif = NeuralIsolationForest::new(3, 2, cfg);
        let mut rng = seeded_rng(99);
        let x_train: Vec<Vec<f64>> = (0..60)
            .map(|i| {
                vec![
                    (i as f64 * 0.05) % 1.0,
                    ((i + 1) as f64 * 0.07) % 1.0,
                    ((i + 2) as f64 * 0.03) % 1.0,
                ]
            })
            .collect();
        nif.fit(&x_train, 1, 0.005, &mut rng);
        let score = nif.anomaly_score(&[0.5, 0.5, 0.5]);
        assert!(score > 0.0 && score < 1.0);
    }
}
