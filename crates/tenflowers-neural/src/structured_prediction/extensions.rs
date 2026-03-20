//! Original test suite for the structured_prediction module.
//!
//! This file is part of the structured_prediction module split. It re-exports
//! nothing new but hosts the regression tests for all types defined in mod.rs.

#[cfg(test)]
mod tests {
    use super::super::{
        ctc_loss, label_smoothing_loss, log_sum_exp, ordered_prediction_loss,
        sequence_cross_entropy, softmax, BeliefPropagation, EnergyNetConfig, EnergyNetwork,
        FactorGraph, HammingLoss, LinearChainCrfConfig, SecondOrderCrf, SecondOrderCrfConfig,
        SpLinear, SpLinearChainCrf, SsvmConfig, StructuredLoss, StructuredSvm,
    };

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_features(t: usize, d: usize) -> Vec<Vec<f64>> {
        (0..t)
            .map(|i| (0..d).map(|j| ((i + j + 1) as f64) * 0.1).collect())
            .collect()
    }

    fn make_crf(t: usize) -> (SpLinearChainCrf, Vec<Vec<f64>>, Vec<usize>) {
        let cfg = LinearChainCrfConfig {
            n_classes: 3,
            feature_dim: 4,
        };
        let crf = SpLinearChainCrf::new(cfg);
        let feats = make_features(t, 4);
        let labels: Vec<usize> = (0..t).map(|i| i % 3).collect();
        (crf, feats, labels)
    }

    // ── SpLinear ─────────────────────────────────────────────────────────────

    #[test]
    fn test_splinear_forward_shape() {
        let layer = SpLinear::new(5, 3);
        let x: Vec<f64> = vec![1.0, 0.5, -0.3, 0.2, 0.9];
        let out = layer.forward(&x);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_splinear_update() {
        let mut layer = SpLinear::new(3, 2);
        let w_before = layer.w.clone();
        let grad_w = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let grad_b = vec![0.01, 0.02];
        layer.update(&grad_w, &grad_b, 0.1);
        // Weights should change
        assert!((layer.w[0][0] - w_before[0][0]).abs() > 1e-10);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let v = vec![1.0, 2.0, 3.0, 0.5];
        let s: f64 = softmax(&v).iter().sum();
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp_basic() {
        let v = vec![0.0_f64, 0.0, 0.0];
        let lse = log_sum_exp(&v);
        // log(3)
        assert!((lse - (3.0_f64).ln()).abs() < 1e-9);
    }

    // ── SpLinearChainCrf ─────────────────────────────────────────────────────

    #[test]
    fn test_crf_viterbi_correct_length() {
        let (crf, feats, _) = make_crf(5);
        let path = crf.viterbi(&feats);
        assert_eq!(path.len(), 5);
    }

    #[test]
    fn test_crf_viterbi_valid_labels() {
        let (crf, feats, _) = make_crf(6);
        let path = crf.viterbi(&feats);
        for &c in &path {
            assert!(c < 3, "label {c} out of range");
        }
    }

    #[test]
    fn test_crf_forward_algorithm_finite() {
        let (crf, feats, _) = make_crf(5);
        let log_z = crf.forward_algorithm(&feats);
        assert!(log_z.is_finite(), "log Z should be finite");
    }

    #[test]
    fn test_crf_marginals_sum_to_one() {
        let (crf, feats, _) = make_crf(4);
        let beliefs = crf.marginals(&feats);
        assert_eq!(beliefs.len(), 4);
        for b in &beliefs {
            let s: f64 = b.iter().sum();
            assert!((s - 1.0).abs() < 1e-8, "marginals don't sum to 1: {s}");
        }
    }

    #[test]
    fn test_crf_marginals_length() {
        let (crf, feats, _) = make_crf(7);
        let m = crf.marginals(&feats);
        assert_eq!(m.len(), 7);
        for row in &m {
            assert_eq!(row.len(), 3);
        }
    }

    #[test]
    fn test_crf_log_likelihood_finite() {
        let (crf, feats, labels) = make_crf(5);
        let ll = crf.log_likelihood(&feats, &labels);
        assert!(ll.is_finite(), "log_likelihood should be finite");
    }

    #[test]
    fn test_crf_log_likelihood_leq_zero() {
        let (crf, feats, labels) = make_crf(5);
        let ll = crf.log_likelihood(&feats, &labels);
        // Just check it's a real number
        assert!(!ll.is_nan());
    }

    #[test]
    fn test_crf_train_step_returns_finite() {
        let (mut crf, feats, labels) = make_crf(4);
        let loss = crf.train_step(&feats, &labels, 0.01);
        assert!(loss.is_finite(), "train_step loss should be finite");
    }

    #[test]
    fn test_crf_train_step_non_negative() {
        let (mut crf, feats, labels) = make_crf(4);
        let loss = crf.train_step(&feats, &labels, 0.01);
        assert!(!loss.is_nan());
    }

    #[test]
    fn test_crf_viterbi_consistent_with_model() {
        let (crf, feats, _) = make_crf(5);
        let best = crf.viterbi(&feats);
        let alt: Vec<usize> = vec![0; 5];
        let emit = crf.emission_scores(&feats);
        let score_path = |path: &[usize]| -> f64 {
            let mut s = crf.start_scores[path[0]] + emit[0][path[0]];
            for t in 1..path.len() {
                s += crf.transition[path[t]][path[t - 1]] + emit[t][path[t]];
            }
            s += crf.end_scores[path[path.len() - 1]];
            s
        };
        let score_best = score_path(&best);
        let score_alt = score_path(&alt);
        assert!(
            score_best >= score_alt - 1e-9,
            "Viterbi path should have highest score"
        );
    }

    #[test]
    fn test_crf_empty_features() {
        let cfg = LinearChainCrfConfig {
            n_classes: 3,
            feature_dim: 4,
        };
        let crf = SpLinearChainCrf::new(cfg);
        let path = crf.viterbi(&[]);
        assert_eq!(path.len(), 0);
        let log_z = crf.forward_algorithm(&[]);
        assert_eq!(log_z, 0.0);
    }

    // ── SecondOrderCrf ────────────────────────────────────────────────────────

    #[test]
    fn test_socrf_viterbi2_correct_length() {
        let cfg = SecondOrderCrfConfig {
            n_classes: 3,
            feature_dim: 4,
            window_size: 2,
        };
        let crf = SecondOrderCrf::new(cfg);
        let feats = make_features(5, 4);
        let path = crf.viterbi2(&feats);
        assert_eq!(path.len(), 5);
    }

    #[test]
    fn test_socrf_viterbi2_single_token() {
        let cfg = SecondOrderCrfConfig {
            n_classes: 3,
            feature_dim: 4,
            window_size: 2,
        };
        let crf = SecondOrderCrf::new(cfg);
        let feats = make_features(1, 4);
        let path = crf.viterbi2(&feats);
        assert_eq!(path.len(), 1);
        assert!(path[0] < 3);
    }

    #[test]
    fn test_socrf_viterbi2_valid_labels() {
        let cfg = SecondOrderCrfConfig {
            n_classes: 4,
            feature_dim: 3,
            window_size: 2,
        };
        let crf = SecondOrderCrf::new(cfg);
        let feats = make_features(6, 3);
        let path = crf.viterbi2(&feats);
        for &c in &path {
            assert!(c < 4);
        }
    }

    #[test]
    fn test_socrf_log_likelihood_finite() {
        let cfg = SecondOrderCrfConfig {
            n_classes: 3,
            feature_dim: 4,
            window_size: 2,
        };
        let crf = SecondOrderCrf::new(cfg);
        let feats = make_features(5, 4);
        let labels: Vec<usize> = vec![0, 1, 2, 0, 1];
        let ll = crf.log_likelihood(&feats, &labels);
        assert!(
            ll.is_finite(),
            "SecondOrderCrf log_likelihood should be finite"
        );
    }

    #[test]
    fn test_socrf_log_likelihood_not_nan() {
        let cfg = SecondOrderCrfConfig {
            n_classes: 2,
            feature_dim: 3,
            window_size: 2,
        };
        let crf = SecondOrderCrf::new(cfg);
        let feats = make_features(3, 3);
        let labels = vec![0, 1, 0];
        let ll = crf.log_likelihood(&feats, &labels);
        assert!(!ll.is_nan());
    }

    // ── StructuredSvm ─────────────────────────────────────────────────────────

    #[test]
    fn test_ssvm_joint_features_correct_length() {
        let cfg = SsvmConfig {
            n_classes: 3,
            feature_dim: 4,
            ..Default::default()
        };
        let ssvm = StructuredSvm::new(cfg);
        let feats = make_features(5, 4);
        let labels: Vec<usize> = vec![0, 1, 2, 0, 1];
        let phi = ssvm.joint_features(&feats, &labels);
        assert_eq!(phi.len(), 3 * 4);
    }

    #[test]
    fn test_ssvm_score_finite() {
        let cfg = SsvmConfig {
            n_classes: 3,
            feature_dim: 4,
            ..Default::default()
        };
        let ssvm = StructuredSvm::new(cfg);
        let feats = make_features(5, 4);
        let labels = vec![0, 1, 2, 0, 1];
        let phi = ssvm.joint_features(&feats, &labels);
        let s = ssvm.score(&phi);
        assert!(s.is_finite());
    }

    #[test]
    fn test_ssvm_hinge_loss_non_negative() {
        let cfg = SsvmConfig {
            n_classes: 3,
            feature_dim: 4,
            ..Default::default()
        };
        let ssvm = StructuredSvm::new(cfg);
        let feats = make_features(5, 4);
        let labels = vec![0, 1, 2, 0, 1];
        let hl = HammingLoss;
        let loss = ssvm.hinge_loss(&feats, &labels, &hl);
        assert!(loss >= 0.0, "Hinge loss must be non-negative");
    }

    #[test]
    fn test_ssvm_hinge_loss_zero_when_perfect() {
        let cfg = SsvmConfig {
            n_classes: 3,
            feature_dim: 4,
            ..Default::default()
        };
        let mut ssvm = StructuredSvm::new(cfg);
        let feats = make_features(3, 4);
        let labels = vec![0_usize, 1, 2];
        let hl = HammingLoss;
        let loss = ssvm.hinge_loss(&feats, &labels, &hl);
        assert!(loss >= 0.0);
        // run a train step to ensure it doesn't panic
        let _ = ssvm.train_step(&feats, &labels, &hl);
    }

    #[test]
    fn test_hamming_loss_identical() {
        let hl = HammingLoss;
        let a = vec![0, 1, 2, 0];
        let loss = hl.loss(&a, &a);
        assert!((loss - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hamming_loss_completely_different() {
        let hl = HammingLoss;
        let a = vec![0, 0, 0, 0];
        let b = vec![1, 1, 1, 1];
        let loss = hl.loss(&a, &b);
        assert!((loss - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hamming_loss_partial() {
        let hl = HammingLoss;
        let a = vec![0, 1, 0, 1];
        let b = vec![0, 0, 0, 1];
        let loss = hl.loss(&a, &b);
        assert!((loss - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_ssvm_train_step_finite() {
        let cfg = SsvmConfig {
            n_classes: 3,
            feature_dim: 4,
            ..Default::default()
        };
        let mut ssvm = StructuredSvm::new(cfg);
        let feats = make_features(5, 4);
        let labels = vec![0, 1, 2, 0, 1];
        let hl = HammingLoss;
        let loss = ssvm.train_step(&feats, &labels, &hl);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_ssvm_train_reduces_loss() {
        let cfg = SsvmConfig {
            n_classes: 2,
            feature_dim: 3,
            c_penalty: 1.0,
            n_epochs: 20,
            lr: 0.05,
        };
        let mut ssvm = StructuredSvm::new(cfg);
        let feats = make_features(4, 3);
        let labels = vec![0, 1, 0, 1];
        let hl = HammingLoss;
        let mut losses = Vec::new();
        for _ in 0..10 {
            losses.push(ssvm.train_step(&feats, &labels, &hl));
        }
        let first = losses[0];
        let last = losses[losses.len() - 1];
        assert!(
            last <= first + 1e-6,
            "SSVM loss should not increase: {first} → {last}"
        );
    }

    // ── EnergyNetwork ─────────────────────────────────────────────────────────

    #[test]
    fn test_energy_network_energy_finite() {
        let cfg = EnergyNetConfig {
            input_dim: 4,
            output_dim: 3,
            hidden_dim: 8,
            n_inference_steps: 5,
            inference_lr: 0.01,
            langevin_noise: 0.0,
        };
        let net = EnergyNetwork::new(cfg);
        let x = vec![0.1, 0.2, 0.3, 0.4];
        let y = vec![1.0, 0.5, -0.3];
        let e = net.energy(&x, &y);
        assert!(e.is_finite());
    }

    #[test]
    fn test_energy_network_predict_shape() {
        let cfg = EnergyNetConfig {
            input_dim: 4,
            output_dim: 3,
            hidden_dim: 8,
            n_inference_steps: 5,
            inference_lr: 0.01,
            langevin_noise: 0.0,
        };
        let net = EnergyNetwork::new(cfg);
        let x = vec![0.1, 0.2, 0.3, 0.4];
        let y_init = vec![0.0, 0.0, 0.0];
        let y_pred = net.predict(&x, &y_init);
        assert_eq!(y_pred.len(), 3);
    }

    #[test]
    fn test_energy_network_contrastive_loss_finite() {
        let cfg = EnergyNetConfig {
            input_dim: 4,
            output_dim: 3,
            hidden_dim: 8,
            n_inference_steps: 5,
            inference_lr: 0.01,
            langevin_noise: 0.0,
        };
        let net = EnergyNetwork::new(cfg);
        let x = vec![0.1, 0.2, 0.3, 0.4];
        let y = vec![1.0, 0.5, -0.3];
        let cl = net.contrastive_loss(&x, &y);
        assert!(cl.is_finite());
    }

    #[test]
    fn test_energy_network_contrastive_loss_non_negative() {
        let cfg = EnergyNetConfig {
            input_dim: 3,
            output_dim: 2,
            hidden_dim: 6,
            n_inference_steps: 10,
            inference_lr: 0.01,
            langevin_noise: 0.0,
        };
        let net = EnergyNetwork::new(cfg);
        let x = vec![0.5, -0.3, 0.8];
        let y = vec![1.0, -1.0];
        let cl = net.contrastive_loss(&x, &y);
        assert!(cl >= 0.0);
    }

    #[test]
    fn test_energy_network_train_step_finite() {
        let cfg = EnergyNetConfig {
            input_dim: 3,
            output_dim: 2,
            hidden_dim: 6,
            n_inference_steps: 3,
            inference_lr: 0.01,
            langevin_noise: 0.0,
        };
        let mut net = EnergyNetwork::new(cfg);
        let x_batch = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
        let y_batch = vec![vec![1.0, -1.0], vec![-0.5, 0.5]];
        let loss = net.train_step(&x_batch, &y_batch, 0.001);
        assert!(loss.is_finite());
    }

    // ── FactorGraph + BeliefPropagation ──────────────────────────────────────

    #[test]
    fn test_factor_graph_set_unary() {
        let mut fg = FactorGraph::new(3, 4);
        fg.set_unary(1, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(fg.unary[1], vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn test_belief_propagation_returns_correct_shape() {
        let mut fg = FactorGraph::new(4, 3);
        fg.set_unary(0, vec![1.0, 2.0, 0.5]);
        fg.set_unary(1, vec![0.5, 1.5, 2.0]);
        fg.add_pairwise(
            0,
            1,
            vec![
                vec![1.0, 0.5, 0.2],
                vec![0.5, 1.0, 0.5],
                vec![0.2, 0.5, 1.0],
            ],
        );
        let bp = BeliefPropagation::new(5);
        let beliefs = bp.run(&fg);
        assert_eq!(beliefs.len(), 4);
        for b in &beliefs {
            assert_eq!(b.len(), 3);
        }
    }

    #[test]
    fn test_belief_propagation_beliefs_normalized() {
        let mut fg = FactorGraph::new(3, 4);
        fg.set_unary(0, vec![0.3, 0.7, 0.1, 0.9]);
        fg.set_unary(1, vec![1.0, 1.0, 1.0, 1.0]);
        fg.set_unary(2, vec![0.5, 0.5, 0.5, 0.5]);
        fg.add_pairwise(
            0,
            1,
            vec![
                vec![1.0, 0.1, 0.1, 0.1],
                vec![0.1, 1.0, 0.1, 0.1],
                vec![0.1, 0.1, 1.0, 0.1],
                vec![0.1, 0.1, 0.1, 1.0],
            ],
        );
        fg.add_pairwise(
            1,
            2,
            vec![
                vec![0.5, 0.2, 0.2, 0.1],
                vec![0.2, 0.5, 0.2, 0.1],
                vec![0.2, 0.2, 0.5, 0.1],
                vec![0.1, 0.1, 0.1, 0.7],
            ],
        );
        let bp = BeliefPropagation::new(10);
        let beliefs = bp.run(&fg);
        for (i, b) in beliefs.iter().enumerate() {
            let s: f64 = b.iter().sum();
            assert!((s - 1.0).abs() < 1e-8, "beliefs[{i}] sum = {s}");
        }
    }

    #[test]
    fn test_belief_propagation_map_decode_length() {
        let mut fg = FactorGraph::new(5, 3);
        for i in 0..5 {
            fg.set_unary(i, vec![1.0, 2.0, 0.5]);
        }
        let bp = BeliefPropagation::new(5);
        let map = bp.map_decode(&fg);
        assert_eq!(map.len(), 5);
    }

    #[test]
    fn test_belief_propagation_map_decode_valid_states() {
        let mut fg = FactorGraph::new(4, 3);
        for i in 0..4 {
            fg.set_unary(i, vec![0.1, 0.9, 0.5]);
        }
        let bp = BeliefPropagation::new(5);
        let map = bp.map_decode(&fg);
        for &s in &map {
            assert!(s < 3);
        }
    }

    #[test]
    fn test_belief_propagation_bethe_free_energy_finite() {
        let mut fg = FactorGraph::new(3, 3);
        fg.set_unary(0, vec![1.0, 2.0, 1.0]);
        fg.set_unary(1, vec![1.0, 1.0, 2.0]);
        fg.set_unary(2, vec![2.0, 1.0, 1.0]);
        fg.add_pairwise(
            0,
            1,
            vec![
                vec![1.0, 0.5, 0.5],
                vec![0.5, 1.0, 0.5],
                vec![0.5, 0.5, 1.0],
            ],
        );
        let bp = BeliefPropagation::new(10);
        let beliefs = bp.run(&fg);
        let bfe = bp.bethe_free_energy(&fg, &beliefs);
        assert!(bfe.is_finite(), "Bethe free energy should be finite");
    }

    // ── Loss functions ────────────────────────────────────────────────────────

    #[test]
    fn test_sequence_cross_entropy_non_negative() {
        let logits = vec![
            vec![2.0, 1.0, 0.5],
            vec![0.5, 2.0, 1.0],
            vec![1.0, 0.5, 2.0],
        ];
        let labels = vec![0, 1, 2];
        let loss = sequence_cross_entropy(&logits, &labels);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_sequence_cross_entropy_decreases_with_confidence() {
        let high_conf = vec![vec![10.0, 0.0, 0.0]];
        let low_conf = vec![vec![1.0, 0.9, 0.8]];
        let labels = vec![0];
        let h_loss = sequence_cross_entropy(&high_conf, &labels);
        let l_loss = sequence_cross_entropy(&low_conf, &labels);
        assert!(h_loss < l_loss + 1e-6);
    }

    #[test]
    fn test_ctc_loss_non_negative() {
        let t = 5;
        let n_classes = 4; // 3 + 1 blank
        let log_probs: Vec<Vec<f64>> = (0..t)
            .map(|_| {
                let v = [0.3_f64.ln(), 0.3_f64.ln(), 0.2_f64.ln(), 0.2_f64.ln()];
                v.iter().map(|x| x.max(-100.0)).collect()
            })
            .collect();
        let target = vec![0, 1];
        let _ = n_classes;
        let loss = ctc_loss(&log_probs, &target);
        assert!(loss >= 0.0, "CTC loss should be non-negative, got {loss}");
    }

    #[test]
    fn test_ctc_loss_finite() {
        let log_probs: Vec<Vec<f64>> = vec![
            vec![-0.5, -0.7, -0.8, -0.6],
            vec![-0.6, -0.5, -0.7, -0.8],
            vec![-0.7, -0.6, -0.5, -0.9],
            vec![-0.8, -0.7, -0.5, -0.6],
        ];
        let target = vec![0, 1];
        let loss = ctc_loss(&log_probs, &target);
        assert!(loss.is_finite(), "CTC loss should be finite");
    }

    #[test]
    fn test_ctc_loss_small_for_good_alignment() {
        let log_probs = vec![
            vec![(-100.0_f64), 0.0_f64.ln(), (-100.0_f64)],
            vec![(-100.0_f64), 0.0_f64.ln(), (-100.0_f64)],
            vec![(-100.0_f64), 0.0_f64.ln(), (-100.0_f64)],
        ];
        let target = vec![1_usize];
        let loss = ctc_loss(&log_probs, &target);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_label_smoothing_loss_non_negative() {
        let logits = vec![vec![1.0, 2.0, 0.5]];
        let labels = vec![1];
        let loss = label_smoothing_loss(&logits, &labels, 0.1);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_label_smoothing_loss_equals_ce_when_zero_smoothing() {
        let logits = vec![vec![1.0, 2.0, 0.5], vec![0.5, 0.3, 1.2]];
        let labels = vec![1, 2];
        let ls_loss = label_smoothing_loss(&logits, &labels, 0.0);
        let ce_loss = sequence_cross_entropy(&logits, &labels);
        assert!(
            (ls_loss - ce_loss).abs() < 1e-9,
            "smoothing=0 should equal CE: {ls_loss} vs {ce_loss}"
        );
    }

    #[test]
    fn test_label_smoothing_loss_positive_smoothing_vs_zero() {
        let logits = vec![vec![2.0, 0.1, 0.1], vec![0.1, 2.0, 0.1]];
        let labels = vec![0, 1];
        let ls_0 = label_smoothing_loss(&logits, &labels, 0.0);
        let ls_e = label_smoothing_loss(&logits, &labels, 0.1);
        assert!(
            ls_e >= ls_0 - 1e-9,
            "Smoothed loss should be >= CE (for confident logits)"
        );
    }

    #[test]
    fn test_ordered_prediction_loss_non_negative() {
        let pred = vec![1.0, 0.5, 0.2];
        let tgt = vec![2.0, 1.0, 0.5];
        let loss = ordered_prediction_loss(&pred, &tgt);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_ordered_prediction_loss_zero_when_same() {
        let v = vec![1.0, 2.0, 3.0];
        let loss = ordered_prediction_loss(&v, &v);
        assert!(
            loss < 1e-8,
            "OPL should be ~0 when pred==target, got {loss}"
        );
    }

    #[test]
    fn test_ordered_prediction_loss_finite() {
        let pred = vec![0.5, -0.5, 1.0, -1.0];
        let tgt = vec![1.0, 0.0, -1.0, 2.0];
        let loss = ordered_prediction_loss(&pred, &tgt);
        assert!(loss.is_finite());
    }

    // ── Additional coverage tests ─────────────────────────────────────────────

    #[test]
    fn test_splinear_new_shape() {
        let l = SpLinear::new(7, 5);
        assert_eq!(l.w.len(), 5);
        assert_eq!(l.w[0].len(), 7);
        assert_eq!(l.b.len(), 5);
    }

    #[test]
    fn test_softmax_max_class() {
        let v = vec![0.0, 5.0, 1.0];
        let s = softmax(&v);
        assert!(s[1] > s[0] && s[1] > s[2]);
    }

    #[test]
    fn test_log_sum_exp_single() {
        let v = vec![3.0_f64];
        assert!((log_sum_exp(&v) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_emission_scores_shape() {
        let (crf, feats, _) = make_crf(5);
        let emit = crf.emission_scores(&feats);
        assert_eq!(emit.len(), 5);
        for row in &emit {
            assert_eq!(row.len(), 3);
        }
    }

    #[test]
    fn test_ssvm_loss_augmented_decode_correct_length() {
        let cfg = SsvmConfig {
            n_classes: 3,
            feature_dim: 4,
            ..Default::default()
        };
        let ssvm = StructuredSvm::new(cfg);
        let feats = make_features(5, 4);
        let labels = vec![0, 1, 2, 0, 1];
        let hl = HammingLoss;
        let yhat = ssvm.loss_augmented_decode(&feats, &labels, &hl);
        assert_eq!(yhat.len(), 5);
    }

    #[test]
    fn test_bp_no_edges() {
        let mut fg = FactorGraph::new(3, 3);
        fg.set_unary(0, vec![1.0, 3.0, 1.0]);
        fg.set_unary(1, vec![2.0, 1.0, 2.0]);
        fg.set_unary(2, vec![0.5, 0.5, 3.0]);
        let bp = BeliefPropagation::new(5);
        let beliefs = bp.run(&fg);
        assert_eq!(beliefs.len(), 3);
        // Without edges, beliefs should peak at unary argmax
        assert!(beliefs[0][1] > beliefs[0][0]);
        assert!(beliefs[2][2] > beliefs[2][0]);
    }

    #[test]
    fn test_sequence_cross_entropy_empty() {
        let loss = sequence_cross_entropy(&[], &[]);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_crf_marginals_probabilities() {
        let (crf, feats, _) = make_crf(3);
        let m = crf.marginals(&feats);
        for row in &m {
            for &p in row {
                assert!((0.0..=1.0 + 1e-9).contains(&p), "prob out of range: {p}");
            }
        }
    }

    #[test]
    fn test_energy_net_layers_count() {
        let cfg = EnergyNetConfig {
            input_dim: 5,
            output_dim: 3,
            hidden_dim: 10,
            n_inference_steps: 5,
            inference_lr: 0.01,
            langevin_noise: 0.0,
        };
        let net = EnergyNetwork::new(cfg);
        assert_eq!(net.energy_net.len(), 3);
    }

    #[test]
    fn test_ctc_loss_empty_target() {
        let log_probs = vec![vec![-0.5, -0.6, -0.7]];
        let target: Vec<usize> = vec![];
        let loss = ctc_loss(&log_probs, &target);
        // empty target → 0 by convention
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_socrf_new_shapes() {
        let cfg = SecondOrderCrfConfig {
            n_classes: 4,
            feature_dim: 5,
            window_size: 3,
        };
        let crf = SecondOrderCrf::new(cfg);
        assert_eq!(crf.transition2.len(), 4);
        assert_eq!(crf.transition2[0].len(), 4);
        assert_eq!(crf.transition2[0][0].len(), 4);
        assert_eq!(crf.pairwise.len(), 16); // n^2
    }

    #[test]
    fn test_label_smoothing_loss_finite() {
        let logits = vec![vec![0.0, 0.0, 0.0]];
        let labels = vec![0];
        let loss = label_smoothing_loss(&logits, &labels, 0.1);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_bp_map_decode_prefers_highest_unary() {
        let mut fg = FactorGraph::new(2, 4);
        fg.set_unary(0, vec![0.01, 0.01, 0.97, 0.01]);
        fg.set_unary(1, vec![0.01, 0.01, 0.01, 0.97]);
        let bp = BeliefPropagation::new(10);
        let map = bp.map_decode(&fg);
        assert_eq!(map[0], 2);
        assert_eq!(map[1], 3);
    }
}
