use super::*;
use scirs2_core::random::{rngs::SmallRng, SeedableRng};

fn make_rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

// ── AcError ──────────────────────────────────────────────────────────────

#[test]
fn test_ac_error_display_invalid_input() {
    let e = AcError::InvalidInput("bad dim".to_string());
    let s = e.to_string();
    assert!(s.contains("InvalidInput"));
    assert!(s.contains("bad dim"));
}

#[test]
fn test_ac_error_display_shape_mismatch() {
    let e = AcError::ShapeMismatch("4 != 8".to_string());
    let s = e.to_string();
    assert!(s.contains("ShapeMismatch"));
}

#[test]
fn test_ac_error_display_computation_failed() {
    let e = AcError::ComputationFailed("nan".to_string());
    assert!(e.to_string().contains("ComputationFailed"));
}

#[test]
fn test_ac_error_implements_std_error() {
    let e: Box<dyn std::error::Error> = Box::new(AcError::InvalidInput("x".to_string()));
    assert!(!e.to_string().is_empty());
}

#[test]
fn test_ac_error_equality() {
    assert_eq!(
        AcError::InvalidInput("a".to_string()),
        AcError::InvalidInput("a".to_string())
    );
    assert_ne!(
        AcError::InvalidInput("a".to_string()),
        AcError::ShapeMismatch("a".to_string())
    );
}

// ── AcMlpBlock ───────────────────────────────────────────────────────────

#[test]
fn test_ac_mlp_block_forward_shape() {
    let mut rng = make_rng(42);
    let block = AcMlpBlock::new(8, 16, &mut rng);
    let x = vec![0.1_f64; 8];
    let out = block.forward(&x).expect("MLP block forward should succeed");
    assert_eq!(out.len(), 8);
}

#[test]
fn test_ac_mlp_block_residual() {
    // With zero weights, output = 0 + bias + x; since bias=0 this is just x.
    let block = AcMlpBlock {
        dim: 4,
        hidden: 4,
        w1: vec![0.0; 16],
        b1: vec![0.0; 4],
        w2: vec![0.0; 16],
        b2: vec![0.0; 4],
    };
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let out = block.forward(&x).expect("MLP block forward should succeed");
    // With zero weights: FFN(x)=0 → residual x is preserved.
    for (o, xi) in out.iter().zip(x.iter()) {
        assert!((o - xi).abs() < 1e-10, "residual failed");
    }
}

#[test]
fn test_ac_mlp_block_wrong_dim() {
    let mut rng = make_rng(1);
    let block = AcMlpBlock::new(4, 8, &mut rng);
    let x = vec![0.0_f64; 5];
    assert!(block.forward(&x).is_err());
}

#[test]
fn test_ac_mlp_block_values_finite() {
    let mut rng = make_rng(7);
    let block = AcMlpBlock::new(16, 32, &mut rng);
    let x: Vec<f64> = (0..16).map(|i| i as f64 * 0.01).collect();
    let out = block.forward(&x).expect("MLP block forward should succeed");
    assert!(out.iter().all(|v| v.is_finite()));
}

// ── PonderingState ───────────────────────────────────────────────────────

#[test]
fn test_pondering_state_defaults() {
    let s = PonderingState::new();
    assert_eq!(s.n_steps, 0);
    assert_eq!(s.cumulative_prob, 0.0);
    assert_eq!(s.remainder, 0.0);
    assert_eq!(s.halting_prob, 0.0);
}

#[test]
fn test_pondering_state_clone() {
    let mut s = PonderingState::new();
    s.n_steps = 3;
    s.cumulative_prob = 0.75;
    let s2 = s.clone();
    assert_eq!(s2.n_steps, 3);
    assert!((s2.cumulative_prob - 0.75).abs() < 1e-12);
}

// ── ActLayer ─────────────────────────────────────────────────────────────

#[test]
fn test_act_forward_output_shape() {
    let mut rng = make_rng(10);
    let config = ActConfig {
        dim: 8,
        hidden: 16,
        max_steps: 5,
        threshold: 0.99,
    };
    let layer = ActLayer::new(config, &mut rng);
    let x = vec![0.1_f64; 8];
    let (out, cost) = layer.act_forward(&x).expect("ACT forward should succeed");
    assert_eq!(out.len(), 8);
    assert!(cost >= 1.0);
    assert!(cost.is_finite());
}

#[test]
fn test_act_forward_ponder_cost_range() {
    let mut rng = make_rng(20);
    let config = ActConfig {
        dim: 4,
        hidden: 8,
        max_steps: 10,
        threshold: 0.99,
    };
    let layer = ActLayer::new(config.clone(), &mut rng);
    let x = vec![0.5_f64; 4];
    let (_, cost) = layer.act_forward(&x).expect("ACT forward should succeed");
    // ponder cost is N + R where N ≤ max_steps and 0 ≤ R ≤ 1.
    assert!(cost >= 1.0);
    assert!(cost <= (config.max_steps as f64 + 1.0));
}

#[test]
fn test_act_forward_wrong_dim() {
    let mut rng = make_rng(3);
    let config = ActConfig {
        dim: 4,
        hidden: 8,
        max_steps: 5,
        threshold: 0.99,
    };
    let layer = ActLayer::new(config, &mut rng);
    assert!(layer.act_forward(&[1.0_f64; 5]).is_err());
}

#[test]
fn test_act_forward_output_finite() {
    let mut rng = make_rng(55);
    let config = ActConfig {
        dim: 8,
        hidden: 16,
        max_steps: 6,
        threshold: 0.9,
    };
    let layer = ActLayer::new(config, &mut rng);
    let x = vec![0.2_f64; 8];
    let (out, cost) = layer.act_forward(&x).expect("ACT forward should succeed");
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(cost.is_finite());
}

// ── AdaptiveDepthNetwork ─────────────────────────────────────────────────

#[test]
fn test_adaptive_depth_forward_shape() {
    let mut rng = make_rng(30);
    let net = AdaptiveDepthNetwork::new(8, 16, 4, 0.9, &mut rng);
    let x = vec![0.1_f64; 8];
    let (out, depth) = net.forward(&x, 4).expect("adaptive depth forward should succeed");
    assert_eq!(out.len(), 8);
    assert!((1..=4).contains(&depth));
}

#[test]
fn test_adaptive_depth_budget_respected() {
    let mut rng = make_rng(31);
    let net = AdaptiveDepthNetwork::new(4, 8, 6, 0.99, &mut rng);
    let x = vec![0.0_f64; 4];
    let (_, depth) = net.forward(&x, 3).expect("adaptive depth forward should succeed");
    assert!(depth <= 3, "depth {} exceeds budget 3", depth);
}

#[test]
fn test_adaptive_depth_wrong_dim() {
    let mut rng = make_rng(32);
    let net = AdaptiveDepthNetwork::new(4, 8, 3, 0.9, &mut rng);
    assert!(net.forward(&[1.0_f64; 5], 3).is_err());
}

// ── AcEarlyExitNetwork ───────────────────────────────────────────────────

#[test]
fn test_early_exit_network_shape() {
    let mut rng = make_rng(40);
    let net = AcEarlyExitNetwork::new(8, 16, 8, 4, &mut rng);
    let x = vec![0.1_f64; 8];
    let (probs, exit) = net.forward_with_early_exit(&x, 0.99).expect("early exit forward should succeed");
    assert_eq!(probs.len(), 4);
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-9,
        "softmax sum should be 1, got {}",
        sum
    );
    assert!(exit <= 3);
}

#[test]
fn test_early_exit_network_low_threshold_exits_early() {
    let mut rng = make_rng(41);
    let net = AcEarlyExitNetwork::new(8, 16, 8, 4, &mut rng);
    let x = vec![0.5_f64; 8];
    // Very low threshold: should exit at first opportunity.
    let (_, exit) = net.forward_with_early_exit(&x, 0.0).expect("early exit forward should succeed");
    assert_eq!(exit, 0, "with threshold=0, should exit at first");
}

#[test]
fn test_early_exit_network_wrong_dim() {
    let mut rng = make_rng(42);
    let net = AcEarlyExitNetwork::new(4, 8, 4, 3, &mut rng);
    assert!(net.forward_with_early_exit(&[1.0_f64; 5], 0.9).is_err());
}

#[test]
fn test_early_exit_output_is_valid_distribution() {
    let mut rng = make_rng(43);
    let net = AcEarlyExitNetwork::new(16, 32, 12, 5, &mut rng);
    let x: Vec<f64> = (0..16).map(|i| i as f64 * 0.05).collect();
    let (probs, _) = net.forward_with_early_exit(&x, 0.8).expect("early exit forward should succeed");
    assert!(probs.iter().all(|&p| (0.0..=1.0 + 1e-9).contains(&p)));
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9);
}

// ── ConditionalDepthRouter ───────────────────────────────────────────────

#[test]
fn test_router_score_shape() {
    let mut rng = make_rng(50);
    let router = ConditionalDepthRouter::new(8, &mut rng);
    let tokens = vec![0.1_f64; 8 * 6];
    let scores = router.score_tokens(&tokens, 6).expect("router score_tokens should succeed");
    assert_eq!(scores.len(), 6);
    assert!(scores.iter().all(|&s| (0.0..=1.0).contains(&s)));
}

#[test]
fn test_router_top_k_count() {
    let mut rng = make_rng(51);
    let router = ConditionalDepthRouter::new(4, &mut rng);
    let scores = vec![0.9, 0.1, 0.8, 0.3, 0.7];
    let top3 = router.top_k_routing(&scores, 3);
    assert_eq!(top3.len(), 3);
}

#[test]
fn test_router_straight_through() {
    let mut rng = make_rng(52);
    let router = ConditionalDepthRouter::new(4, &mut rng);
    let scores = vec![0.8, 0.3, 0.6, 0.4];
    let gates = router.straight_through_gate(&scores);
    assert_eq!(gates[0], 1.0);
    assert_eq!(gates[1], 0.0);
    assert_eq!(gates[2], 1.0);
    assert_eq!(gates[3], 0.0);
}

#[test]
fn test_router_wrong_shape() {
    let mut rng = make_rng(53);
    let router = ConditionalDepthRouter::new(4, &mut rng);
    // Passing 4*3=12 floats but n_tokens=3 which needs 12; passing wrong count.
    assert!(router.score_tokens(&[0.0; 13], 3).is_err());
}

// ── AcMixtureOfDepthsLayer ───────────────────────────────────────────────

#[test]
fn test_mod_layer_output_shape() {
    let mut rng = make_rng(60);
    let layer = AcMixtureOfDepthsLayer::new(8, 16, &mut rng);
    let tokens = vec![0.1_f64; 8 * 4];
    let (out, indices) = layer.forward(&tokens, 4, 0.5).expect("MoD forward should succeed");
    assert_eq!(out.len(), 8 * 4);
    assert_eq!(indices.len(), 2); // ceil(4 * 0.5) = 2
}

#[test]
fn test_mod_layer_unrouted_tokens_unchanged() {
    // With capacity_fraction=0 (rounds up to k=1), only 1 token is routed.
    // We can check that unrouted tokens remain identical.
    let mut rng = make_rng(61);
    let layer = AcMixtureOfDepthsLayer::new(4, 8, &mut rng);
    let tokens: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let (out, routed) = layer.forward(&tokens, 4, 0.25).expect("MoD forward should succeed");
    // Non-routed tokens should equal input.
    for t in 0..4 {
        if !routed.contains(&t) {
            let in_slice = &tokens[t * 4..(t + 1) * 4];
            let out_slice = &out[t * 4..(t + 1) * 4];
            assert_eq!(in_slice, out_slice);
        }
    }
}

#[test]
fn test_mod_layer_wrong_shape() {
    let mut rng = make_rng(62);
    let layer = AcMixtureOfDepthsLayer::new(4, 8, &mut rng);
    assert!(layer.forward(&[0.0_f64; 5], 4, 0.5).is_err());
}

// ── SkimmingModel ────────────────────────────────────────────────────────

#[test]
fn test_skimming_forward_shape() {
    let mut rng = make_rng(70);
    let model = SkimmingModel::new(8, 16, &mut rng);
    let tokens = vec![0.1_f64; 8 * 5];
    let (out, mask) = model.skim_forward(&tokens, 5, 0.6).expect("skim forward should succeed");
    assert_eq!(out.len(), 8 * 5);
    assert_eq!(mask.len(), 5);
    let processed = mask.iter().filter(|&&m| m).count();
    assert_eq!(processed, 3); // ceil(5 * 0.6)
}

#[test]
fn test_skimming_non_processed_unchanged() {
    let mut rng = make_rng(71);
    let model = SkimmingModel::new(4, 8, &mut rng);
    let tokens: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
    let (out, mask) = model.skim_forward(&tokens, 5, 0.4).expect("skim forward should succeed");
    for t in 0..5 {
        if !mask[t] {
            assert_eq!(&tokens[t * 4..(t + 1) * 4], &out[t * 4..(t + 1) * 4]);
        }
    }
}

#[test]
fn test_skimming_wrong_shape() {
    let mut rng = make_rng(72);
    let model = SkimmingModel::new(4, 8, &mut rng);
    assert!(model.skim_forward(&[0.0_f64; 9], 3, 0.5).is_err());
}

// ── AcGatingMechanism ────────────────────────────────────────────────────

#[test]
fn test_gating_sigmoid_range() {
    let gate = AcGatingMechanism::new(AcGatingStrategy::Sigmoid);
    let logits = vec![-2.0, 0.0, 2.0, 5.0];
    let w = gate.apply_gate(&logits).expect("sigmoid gate should succeed");
    assert_eq!(w.len(), 4);
    assert!(w.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn test_gating_sparsemax_sums_to_one() {
    let gate = AcGatingMechanism::new(AcGatingStrategy::Sparsemax);
    let logits = vec![1.0, 2.0, 3.0, 0.5];
    let w = gate.apply_gate(&logits).expect("sparsemax gate should succeed");
    let sum: f64 = w.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "sparsemax sum={}", sum);
    assert!(w.iter().all(|&v| v >= 0.0));
}

#[test]
fn test_gating_sparsemax_sparsity() {
    let gate = AcGatingMechanism::new(AcGatingStrategy::Sparsemax);
    // Large gap between first and rest → sparsemax should zero-out lower entries.
    let logits = vec![10.0, -1.0, -2.0, -3.0];
    let w = gate.apply_gate(&logits).expect("sparsemax gate should succeed");
    assert!(w[0] > 0.9, "dominant entry should be close to 1.0");
}

#[test]
fn test_gating_entmax15_sums_to_one() {
    let gate = AcGatingMechanism::new(AcGatingStrategy::Entmax15);
    let logits = vec![0.5, 1.5, 0.3, 0.8];
    let w = gate.apply_gate(&logits).expect("entmax1.5 gate should succeed");
    let sum: f64 = w.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "entmax1.5 sum={}", sum);
    assert!(w.iter().all(|&v| v >= 0.0));
}

#[test]
fn test_gating_soft_top_k_correct_count() {
    let gate = AcGatingMechanism::new(AcGatingStrategy::SoftTopK { k: 2 });
    let logits = vec![1.0, 3.0, 0.5, 2.0, 0.1];
    let w = gate.apply_gate(&logits).expect("soft top-k gate should succeed");
    let nonzero = w.iter().filter(|&&v| v > 1e-12).count();
    assert_eq!(nonzero, 2);
    let sum: f64 = w.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9);
}

#[test]
fn test_gating_empty_logits_error() {
    let gate = AcGatingMechanism::new(AcGatingStrategy::Sigmoid);
    assert!(gate.apply_gate(&[]).is_err());
}

// ── DynamicSlimmingLayer ─────────────────────────────────────────────────

#[test]
fn test_dynamic_slimming_full_width() {
    let mut rng = make_rng(80);
    let layer = DynamicSlimmingLayer::new(8, 16, &mut rng);
    let x = vec![0.1_f64; 8];
    let out = layer.forward(&x).expect("dynamic slimming forward should succeed");
    assert_eq!(out.len(), 8);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn test_dynamic_slimming_half_width() {
    let mut rng = make_rng(81);
    let layer = DynamicSlimmingLayer::new(8, 16, &mut rng);
    let x = vec![0.2_f64; 8];
    let out = layer.forward_slimmed(&x, 0.5).expect("dynamic slimming forward_slimmed should succeed");
    assert_eq!(out.len(), 8);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn test_dynamic_slimming_wrong_dim() {
    let mut rng = make_rng(82);
    let layer = DynamicSlimmingLayer::new(4, 8, &mut rng);
    assert!(layer.forward_slimmed(&[0.0_f64; 5], 0.5).is_err());
}

// ── LayerDropNetwork ─────────────────────────────────────────────────────

#[test]
fn test_layer_drop_inference_shape() {
    let mut net = LayerDropNetwork::new(8, 16, 4, 0.8, 42);
    let x = vec![0.1_f64; 8];
    let out = net.forward(&x, false).expect("layer drop inference forward should succeed");
    assert_eq!(out.len(), 8);
}

#[test]
fn test_layer_drop_training_shape() {
    let mut net = LayerDropNetwork::new(8, 16, 4, 0.8, 43);
    let x = vec![0.2_f64; 8];
    let out = net.forward(&x, true).expect("layer drop training forward should succeed");
    assert_eq!(out.len(), 8);
}

#[test]
fn test_layer_drop_inference_finite() {
    let mut net = LayerDropNetwork::new(16, 32, 6, 0.5, 44);
    let x: Vec<f64> = (0..16).map(|i| i as f64 * 0.01).collect();
    let out = net.forward(&x, false).expect("layer drop inference forward should succeed");
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn test_layer_drop_p_last_one_noop() {
    // p_last=1.0 → survival prob 1 for all layers → training == inference (no drops).
    let mut net = LayerDropNetwork::new(4, 8, 3, 1.0, 99);
    let x = vec![0.5_f64; 4];
    let inf_out = net.forward(&x, false).expect("layer drop inference forward should succeed");
    // With p_last=1, inference and training should give identical outputs.
    let mut net2 = LayerDropNetwork::new(4, 8, 3, 1.0, 99);
    let train_out = net2.forward(&x, true).expect("layer drop training forward should succeed");
    for (a, b) in inf_out.iter().zip(train_out.iter()) {
        assert!((a - b).abs() < 1e-9, "mismatch: {} vs {}", a, b);
    }
}

// ── AdaptiveMixturLayer ──────────────────────────────────────────────────

#[test]
fn test_adaptive_mixtur_output_shape() {
    let mut rng = make_rng(90);
    let layer = AdaptiveMixturLayer::new(8, 16, 4, &mut rng);
    let x = vec![0.1_f64; 8];
    let out = layer.forward(&x).expect("adaptive mixtur forward should succeed");
    assert_eq!(out.len(), 8);
}

#[test]
fn test_adaptive_mixtur_finite() {
    let mut rng = make_rng(91);
    let layer = AdaptiveMixturLayer::new(16, 32, 6, &mut rng);
    let x: Vec<f64> = (0..16).map(|i| i as f64 * 0.02).collect();
    let out = layer.forward(&x).expect("adaptive mixtur forward should succeed");
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn test_adaptive_mixtur_wrong_dim() {
    let mut rng = make_rng(92);
    let layer = AdaptiveMixturLayer::new(4, 8, 3, &mut rng);
    assert!(layer.forward(&[0.0_f64; 5]).is_err());
}

#[test]
fn test_adaptive_mixtur_single_expert() {
    let mut rng = make_rng(93);
    let layer = AdaptiveMixturLayer::new(4, 8, 1, &mut rng);
    let x = vec![0.3_f64; 4];
    let out = layer.forward(&x).expect("adaptive mixtur single expert forward should succeed");
    assert_eq!(out.len(), 4);
    assert!(out.iter().all(|v| v.is_finite()));
}

// ── PonderNetBlock ───────────────────────────────────────────────────────

#[test]
fn test_ponder_net_step_shapes() {
    let block = PonderNetBlock::new(4, 8, 3, 0.2, 100);
    let h = vec![0.0_f64; 8];
    let x = vec![0.1_f64; 4];
    let (h_new, y_n, lambda) = block.step(&h, &x).expect("PonderNet step should succeed");
    assert_eq!(h_new.len(), 8);
    assert_eq!(y_n.len(), 3);
    assert!((0.0..=1.0).contains(&lambda));
}

#[test]
fn test_ponder_net_step_wrong_h() {
    let block = PonderNetBlock::new(4, 8, 3, 0.2, 101);
    let h = vec![0.0_f64; 5]; // wrong
    let x = vec![0.1_f64; 4];
    assert!(block.step(&h, &x).is_err());
}

#[test]
fn test_ponder_net_step_wrong_x() {
    let block = PonderNetBlock::new(4, 8, 3, 0.2, 102);
    let h = vec![0.0_f64; 8];
    let x = vec![0.1_f64; 5]; // wrong
    assert!(block.step(&h, &x).is_err());
}

#[test]
fn test_ponder_net_forward_output_shape() {
    let mut block = PonderNetBlock::new(4, 8, 3, 0.2, 103);
    let x = vec![0.1_f64; 4];
    let (out, kl, steps) = block.ponder_forward(&x, 5).expect("PonderNet forward should succeed");
    assert_eq!(out.len(), 3);
    assert!(kl >= 0.0, "KL should be non-negative");
    assert!((1.0..=5.0 + 1e-9).contains(&steps));
}

#[test]
fn test_ponder_net_forward_kl_finite() {
    let mut block = PonderNetBlock::new(6, 12, 4, 0.3, 104);
    let x = vec![0.2_f64; 6];
    let (_, kl, _) = block.ponder_forward(&x, 8).expect("PonderNet forward should succeed");
    assert!(kl.is_finite());
}

#[test]
fn test_ponder_net_forward_wrong_input() {
    let mut block = PonderNetBlock::new(4, 8, 3, 0.2, 105);
    assert!(block.ponder_forward(&[0.0_f64; 5], 4).is_err());
}

// ── AcMetrics ────────────────────────────────────────────────────────────

#[test]
fn test_mean_ponder_cost_basic() {
    let costs = vec![2.3, 3.1, 1.9, 4.0];
    let mean = AcMetrics::mean_ponder_cost(&costs);
    let expected = (2.3 + 3.1 + 1.9 + 4.0) / 4.0;
    assert!((mean - expected).abs() < 1e-10);
}

#[test]
fn test_mean_ponder_cost_empty() {
    assert_eq!(AcMetrics::mean_ponder_cost(&[]), 0.0);
}

#[test]
fn test_early_exit_distribution_counts() {
    let exits = vec![0, 1, 0, 2, 1, 0, 3];
    let dist = AcMetrics::early_exit_distribution(&exits, 4);
    assert_eq!(dist[0], 3);
    assert_eq!(dist[1], 2);
    assert_eq!(dist[2], 1);
    assert_eq!(dist[3], 1);
}

#[test]
fn test_routing_entropy_uniform() {
    // Uniform distribution has maximum entropy = ln(n).
    let n = 4;
    let p = 1.0 / n as f64;
    let probs: Vec<Vec<f64>> = vec![vec![p; n]; 10];
    let ent = AcMetrics::routing_entropy(&probs);
    let expected = -(p * p.ln()) * n as f64;
    assert!((ent - expected).abs() < 1e-10);
}

#[test]
fn test_routing_entropy_empty() {
    assert_eq!(AcMetrics::routing_entropy(&[]), 0.0);
}

#[test]
fn test_flop_reduction_ratio_all_early() {
    // All samples exit at layer 0, full cost = 4.
    let exits = vec![0, 0, 0];
    let ratio = AcMetrics::flop_reduction_ratio(&exits, 4);
    // actual = 3 * 1 = 3, full = 3 * 4 = 12 → ratio = 0.25.
    assert!((ratio - 0.25).abs() < 1e-10);
}

#[test]
fn test_flop_reduction_ratio_no_early_exit() {
    let exits = vec![3, 3, 3]; // exit at last layer (idx 3), full = 4.
    let ratio = AcMetrics::flop_reduction_ratio(&exits, 4);
    assert!((ratio - 1.0).abs() < 1e-10);
}

#[test]
fn test_flop_reduction_ratio_empty() {
    assert_eq!(AcMetrics::flop_reduction_ratio(&[], 4), 1.0);
}

// ── Integration tests ────────────────────────────────────────────────────

#[test]
fn test_act_deterministic() {
    let mut rng = make_rng(200);
    let config = ActConfig {
        dim: 4,
        hidden: 8,
        max_steps: 5,
        threshold: 0.99,
    };
    let layer = ActLayer::new(config, &mut rng);
    let x = vec![0.1_f64; 4];
    let (out1, cost1) = layer.act_forward(&x).expect("ACT forward should succeed");
    let (out2, cost2) = layer.act_forward(&x).expect("ACT forward should succeed");
    assert_eq!(out1, out2);
    assert_eq!(cost1, cost2);
}

#[test]
fn test_mod_full_capacity_processes_all() {
    let mut rng = make_rng(201);
    let layer = AcMixtureOfDepthsLayer::new(4, 8, &mut rng);
    let tokens = vec![0.1_f64; 4 * 5];
    let (_, routed) = layer.forward(&tokens, 5, 1.0).expect("MoD forward should succeed");
    assert_eq!(routed.len(), 5); // all tokens routed
}

#[test]
fn test_skimming_ratio_one_processes_all() {
    let mut rng = make_rng(202);
    let model = SkimmingModel::new(4, 8, &mut rng);
    let tokens = vec![0.5_f64; 4 * 4];
    let (_, mask) = model.skim_forward(&tokens, 4, 1.0).expect("skim forward should succeed");
    assert!(mask.iter().all(|&m| m));
}

#[test]
fn test_layer_drop_zero_p_last_all_skip_training() {
    // p_last=0 → survival p_L=0 → at training all layers dropped
    // (last layer always drops in training, others may have higher p).
    // Just check it runs without error.
    let mut net = LayerDropNetwork::new(4, 8, 3, 0.0, 300);
    let x = vec![0.3_f64; 4];
    let out = net.forward(&x, true).expect("layer drop training forward should succeed");
    assert_eq!(out.len(), 4);
}

#[test]
fn test_full_pipeline_act_then_metrics() {
    let mut rng = make_rng(300);
    let config = ActConfig {
        dim: 4,
        hidden: 8,
        max_steps: 5,
        threshold: 0.99,
    };
    let layer = ActLayer::new(config, &mut rng);
    let inputs: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1; 4]).collect();
    let costs: Vec<f64> = inputs
        .iter()
        .map(|x| layer.act_forward(x).expect("ACT forward should succeed").1)
        .collect();
    let mean = AcMetrics::mean_ponder_cost(&costs);
    assert!(mean >= 1.0);
    assert!(mean.is_finite());
}
