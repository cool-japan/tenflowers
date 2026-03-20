use super::*;
use scirs2_core::random::{rngs::SmallRng, SeedableRng};

fn make_rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

// ── ModError ──────────────────────────────────────────────────────────────────

#[test]
fn test_mod_error_invalid_input_display() {
    let e = ModError::InvalidInput("bad dim".to_string());
    assert!(e.to_string().contains("InvalidInput"));
    assert!(e.to_string().contains("bad dim"));
}

#[test]
fn test_mod_error_shape_mismatch_display() {
    let e = ModError::ShapeMismatch("4 vs 8".to_string());
    assert!(e.to_string().contains("ShapeMismatch"));
}

#[test]
fn test_mod_error_numerical_error_display() {
    let e = ModError::NumericalError("nan".to_string());
    assert!(e.to_string().contains("NumericalError"));
}

#[test]
fn test_mod_error_implements_std_error() {
    let e: Box<dyn std::error::Error> = Box::new(ModError::InvalidInput("x".to_string()));
    assert!(!e.to_string().is_empty());
}

#[test]
fn test_mod_error_equality() {
    assert_eq!(
        ModError::InvalidInput("a".to_string()),
        ModError::InvalidInput("a".to_string())
    );
    assert_ne!(
        ModError::InvalidInput("a".to_string()),
        ModError::ShapeMismatch("a".to_string())
    );
}

// ── ModRouter ─────────────────────────────────────────────────────────────────

#[test]
fn test_mod_router_score_shape() {
    let mut rng = make_rng(100);
    let router = ModRouter::new(8, 0.5, &mut rng);
    let tokens = vec![0.1_f64; 8 * 6];
    let scores = router.score_tokens(&tokens, 6).expect("token scoring should succeed");
    assert_eq!(scores.len(), 6);
    // sigmoid outputs are in (0, 1).
    assert!(scores.iter().all(|&s| s > 0.0 && s < 1.0));
}

#[test]
fn test_mod_router_score_wrong_shape() {
    let mut rng = make_rng(101);
    let router = ModRouter::new(4, 0.5, &mut rng);
    // n_tokens=3 requires 12 floats; supply 13.
    assert!(router.score_tokens(&[0.0; 13], 3).is_err());
}

#[test]
fn test_mod_router_route_count() {
    let mut rng = make_rng(102);
    let router = ModRouter::new(8, 0.5, &mut rng);
    let tokens = vec![0.2_f64; 8 * 10];
    let (routed, bypassed, _) = router.route(&tokens, 10).expect("token routing should succeed");
    assert_eq!(routed.len(), 5); // ceil(10 * 0.5) = 5
    assert_eq!(bypassed.len(), 5);
    assert_eq!(routed.len() + bypassed.len(), 10);
}

#[test]
fn test_mod_router_route_full_capacity() {
    let mut rng = make_rng(103);
    let router = ModRouter::new(4, 1.0, &mut rng);
    let tokens = vec![0.1_f64; 4 * 5];
    let (routed, bypassed, _) = router.route(&tokens, 5).expect("token routing should succeed");
    assert_eq!(routed.len(), 5);
    assert!(bypassed.is_empty());
}

#[test]
fn test_mod_router_route_minimum_one() {
    let mut rng = make_rng(104);
    let router = ModRouter::new(4, 0.01, &mut rng); // tiny fraction
    let tokens = vec![0.1_f64; 4 * 10];
    let (routed, _, _) = router.route(&tokens, 10).expect("token routing should succeed");
    assert!(!routed.is_empty(), "at least one token must be routed");
}

#[test]
fn test_mod_router_auxiliary_loss_non_negative() {
    let mut rng = make_rng(105);
    let router = ModRouter::new(8, 0.5, &mut rng);
    let tokens = vec![0.3_f64; 8 * 8];
    let scores = router.score_tokens(&tokens, 8).expect("token scoring should succeed");
    let loss = router.auxiliary_loss(&scores);
    assert!(loss >= 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_mod_router_auxiliary_loss_empty() {
    let mut rng = make_rng(106);
    let router = ModRouter::new(4, 0.5, &mut rng);
    assert_eq!(router.auxiliary_loss(&[]), 0.0);
}

#[test]
fn test_mod_router_routed_indices_unique() {
    let mut rng = make_rng(107);
    let router = ModRouter::new(8, 0.6, &mut rng);
    let tokens = vec![0.1_f64; 8 * 8];
    let (routed, _, _) = router.route(&tokens, 8).expect("token routing should succeed");
    let unique: std::collections::HashSet<usize> = routed.iter().cloned().collect();
    assert_eq!(unique.len(), routed.len(), "routed indices must be unique");
}

// ── ModTransformerLayer ───────────────────────────────────────────────────────

#[test]
fn test_mod_transformer_layer_output_shape() {
    let mut rng = make_rng(200);
    let layer = ModTransformerLayer::new(8, 16, 2, 0.5, &mut rng).expect("transformer layer construction should succeed");
    let tokens = vec![0.1_f64; 8 * 4];
    let (out, routed, aux) = layer.forward(&tokens, 4).expect("transformer forward should succeed");
    assert_eq!(out.len(), 8 * 4);
    assert!(!routed.is_empty());
    assert!(aux >= 0.0 && aux.is_finite());
}

#[test]
fn test_mod_transformer_layer_wrong_heads() {
    let mut rng = make_rng(201);
    // dim=7, n_heads=3: 7 % 3 != 0 → error.
    assert!(ModTransformerLayer::new(7, 14, 3, 0.5, &mut rng).is_err());
}

#[test]
fn test_mod_transformer_layer_wrong_input() {
    let mut rng = make_rng(202);
    let layer = ModTransformerLayer::new(8, 16, 2, 0.5, &mut rng).expect("transformer layer construction should succeed");
    // Supply 7*4 floats but n_tokens=4, dim=8 requires 32.
    assert!(layer.forward(&vec![0.0; 7 * 4], 4).is_err());
}

#[test]
fn test_mod_transformer_layer_bypassed_tokens_unchanged() {
    let mut rng = make_rng(203);
    let layer = ModTransformerLayer::new(4, 8, 1, 0.25, &mut rng).expect("transformer layer construction should succeed");
    let tokens: Vec<f64> = (0..4 * 4).map(|i| i as f64 * 0.1).collect();
    let (out, routed, _) = layer.forward(&tokens, 4).expect("transformer forward should succeed");
    // Bypassed tokens should be identical to input.
    for t in 0..4 {
        if !routed.contains(&t) {
            let orig = &tokens[t * 4..(t + 1) * 4];
            let got = &out[t * 4..(t + 1) * 4];
            for (&o, &g) in orig.iter().zip(got.iter()) {
                assert!(
                    (o - g).abs() < 1e-10,
                    "bypassed token {}: {} vs {}",
                    t,
                    o,
                    g
                );
            }
        }
    }
}

#[test]
fn test_mod_transformer_layer_full_capacity() {
    let mut rng = make_rng(204);
    let layer = ModTransformerLayer::new(4, 8, 1, 1.0, &mut rng).expect("transformer layer construction should succeed");
    let tokens = vec![0.1_f64; 4 * 6];
    let (_, routed, _) = layer.forward(&tokens, 6).expect("transformer forward should succeed");
    assert_eq!(routed.len(), 6); // all tokens routed
}

#[test]
fn test_mod_transformer_layer_output_finite() {
    let mut rng = make_rng(205);
    let layer = ModTransformerLayer::new(8, 16, 2, 0.5, &mut rng).expect("transformer layer construction should succeed");
    let tokens: Vec<f64> = (0..8 * 4).map(|i| i as f64 * 0.01).collect();
    let (out, _, _) = layer.forward(&tokens, 4).expect("transformer forward should succeed");
    assert!(out.iter().all(|v| v.is_finite()));
}

// ── ModEarlyExit ──────────────────────────────────────────────────────────────

#[test]
fn test_mod_early_exit_output_shape() {
    let mut rng = make_rng(300);
    let net = ModEarlyExit::new(8, 16, 4, 0.9, 0.01, &mut rng);
    let tokens = vec![0.1_f64; 8 * 5];
    let (out, depths, ponder) = net.forward_with_exit(&tokens, 5).expect("early exit forward should succeed");
    assert_eq!(out.len(), 8 * 5);
    assert_eq!(depths.len(), 5);
    assert!(ponder >= 0.0 && ponder.is_finite());
}

#[test]
fn test_mod_early_exit_wrong_input() {
    let mut rng = make_rng(301);
    let net = ModEarlyExit::new(4, 8, 3, 0.9, 0.1, &mut rng);
    // 4*5 != 5 expected; supply 4*5 but claim n_tokens=6.
    assert!(net.forward_with_exit(&[0.0; 4 * 5], 6).is_err());
}

#[test]
fn test_mod_early_exit_depths_in_range() {
    let mut rng = make_rng(302);
    let n_layers = 6;
    let net = ModEarlyExit::new(8, 16, n_layers, 0.5, 0.01, &mut rng);
    let tokens = vec![0.2_f64; 8 * 4];
    let (_, depths, _) = net.forward_with_exit(&tokens, 4).expect("early exit forward should succeed");
    for &d in &depths {
        assert!(
            d < n_layers,
            "depth {} out of range for n_layers={}",
            d,
            n_layers
        );
    }
}

#[test]
fn test_mod_early_exit_low_threshold_exits_first_layer() {
    let mut rng = make_rng(303);
    // Threshold=0.0: all tokens exit at depth 0.
    let net = ModEarlyExit::new(4, 8, 5, 0.0, 0.01, &mut rng);
    let tokens = vec![0.5_f64; 4 * 3];
    let (_, depths, _) = net.forward_with_exit(&tokens, 3).expect("early exit forward should succeed");
    assert!(depths.iter().all(|&d| d == 0), "all should exit at depth 0");
}

#[test]
fn test_mod_early_exit_ponder_loss_lambda() {
    let mut rng = make_rng(304);
    let lambda = 0.5;
    let net = ModEarlyExit::new(4, 8, 4, 0.0, lambda, &mut rng);
    let tokens = vec![0.1_f64; 4 * 2];
    let (_, depths, ponder) = net.forward_with_exit(&tokens, 2).expect("early exit forward should succeed");
    let mean_d = depths.iter().map(|&d| d as f64).sum::<f64>() / 2.0;
    assert!((ponder - lambda * mean_d).abs() < 1e-10);
}

#[test]
fn test_mod_early_exit_output_finite() {
    let mut rng = make_rng(305);
    let net = ModEarlyExit::new(8, 16, 4, 0.7, 0.01, &mut rng);
    let tokens: Vec<f64> = (0..8 * 4).map(|i| i as f64 * 0.02).collect();
    let (out, _, _) = net.forward_with_exit(&tokens, 4).expect("early exit forward should succeed");
    assert!(out.iter().all(|v| v.is_finite()));
}

// ── ModAdaptiveDepth ──────────────────────────────────────────────────────────

#[test]
fn test_mod_adaptive_depth_output_shape() {
    let mut rng = make_rng(400);
    let net = ModAdaptiveDepth::new(8, 16, 5, 0.01, &mut rng);
    let x = vec![0.1_f64; 8];
    let (out, steps) = net.forward(&x, 5).expect("adaptive depth forward should succeed");
    assert_eq!(out.len(), 8);
    assert!((1..=5).contains(&steps));
}

#[test]
fn test_mod_adaptive_depth_wrong_dim() {
    let mut rng = make_rng(401);
    let net = ModAdaptiveDepth::new(4, 8, 4, 0.01, &mut rng);
    assert!(net.forward(&[0.0_f64; 5], 4).is_err());
}

#[test]
fn test_mod_adaptive_depth_budget_respected() {
    let mut rng = make_rng(402);
    let net = ModAdaptiveDepth::new(4, 8, 8, 0.01, &mut rng);
    let x = vec![0.5_f64; 4];
    let (_, steps) = net.forward(&x, 3).expect("adaptive depth forward should succeed");
    assert!(steps <= 3, "steps {} exceeds budget 3", steps);
}

#[test]
fn test_mod_adaptive_depth_output_finite() {
    let mut rng = make_rng(403);
    let net = ModAdaptiveDepth::new(16, 32, 6, 0.05, &mut rng);
    let x: Vec<f64> = (0..16).map(|i| i as f64 * 0.01).collect();
    let (out, _) = net.forward(&x, 6).expect("adaptive depth forward should succeed");
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn test_mod_adaptive_depth_weighted_output_norm() {
    // The weighted output should have bounded norm (not explode).
    let mut rng = make_rng(404);
    let net = ModAdaptiveDepth::new(8, 16, 4, 0.1, &mut rng);
    let x = vec![1.0_f64; 8];
    let (out, _) = net.forward(&x, 4).expect("adaptive depth forward should succeed");
    let norm: f64 = out.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
    assert!(norm.is_finite() && norm < 1e6, "norm too large: {}", norm);
}

// ── ModUniversalTransformer ───────────────────────────────────────────────────

#[test]
fn test_mod_universal_transformer_output_shape() {
    let mut rng = make_rng(500);
    let ut = ModUniversalTransformer::new(8, 16, 0.01, &mut rng);
    let tokens = vec![0.1_f64; 8 * 4];
    let (out, steps) = ut.forward(&tokens, 4, 4).expect("universal transformer forward should succeed");
    assert_eq!(out.len(), 8 * 4);
    assert!((1..=4).contains(&steps));
}

#[test]
fn test_mod_universal_transformer_wrong_input() {
    let mut rng = make_rng(501);
    let ut = ModUniversalTransformer::new(4, 8, 0.01, &mut rng);
    assert!(ut.forward(&[0.0; 4 * 5], 6, 3).is_err());
}

#[test]
fn test_mod_universal_transformer_output_finite() {
    let mut rng = make_rng(502);
    let ut = ModUniversalTransformer::new(8, 16, 0.01, &mut rng);
    let tokens: Vec<f64> = (0..8 * 3).map(|i| i as f64 * 0.05).collect();
    let (out, _) = ut.forward(&tokens, 3, 5).expect("universal transformer forward should succeed");
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn test_mod_universal_transformer_max_steps_one() {
    let mut rng = make_rng(503);
    let ut = ModUniversalTransformer::new(4, 8, 0.01, &mut rng);
    let tokens = vec![0.2_f64; 4 * 2];
    let (out, steps) = ut.forward(&tokens, 2, 1).expect("universal transformer forward should succeed");
    assert_eq!(out.len(), 4 * 2);
    assert_eq!(steps, 1);
}

#[test]
fn test_mod_universal_transformer_weight_sharing_deterministic() {
    // Same input → same output (no randomness in forward).
    let mut rng = make_rng(504);
    let ut = ModUniversalTransformer::new(4, 8, 0.5, &mut rng);
    let tokens = vec![0.3_f64; 4 * 3];
    let (out1, s1) = ut.forward(&tokens, 3, 4).expect("universal transformer forward should succeed");
    let (out2, s2) = ut.forward(&tokens, 3, 4).expect("universal transformer forward should succeed");
    assert_eq!(out1, out2);
    assert_eq!(s1, s2);
}

// ── ModConditionalComputation ─────────────────────────────────────────────────

#[test]
fn test_mod_conditional_computation_output_shape() {
    let mut rng = make_rng(600);
    let cc = ModConditionalComputation::new(8, 16, 4, &mut rng);
    let x = vec![0.1_f64; 8];
    let (out, frac) = cc.forward(&x, 0.5).expect("conditional computation forward should succeed");
    assert_eq!(out.len(), 8);
    assert!((0.0..=1.0).contains(&frac));
}

#[test]
fn test_mod_conditional_computation_wrong_dim() {
    let mut rng = make_rng(601);
    let cc = ModConditionalComputation::new(4, 8, 3, &mut rng);
    assert!(cc.forward(&[0.0_f64; 5], 0.5).is_err());
}

#[test]
fn test_mod_conditional_computation_high_threshold_skips_all() {
    // Threshold=2.0 (above sigmoid max of 1.0) → all layers skipped → frac=0.
    let mut rng = make_rng(602);
    let cc = ModConditionalComputation::new(4, 8, 5, &mut rng);
    let x = vec![0.5_f64; 4];
    let (out, frac) = cc.forward(&x, 2.0).expect("conditional computation forward should succeed");
    assert_eq!(out.len(), 4);
    assert_eq!(frac, 0.0, "all layers should be skipped");
    // Output should equal input when all layers skipped.
    for (&o, &xi) in out.iter().zip(x.iter()) {
        assert!((o - xi).abs() < 1e-10);
    }
}

#[test]
fn test_mod_conditional_computation_zero_threshold_applies_all() {
    // Threshold=0.0 → all gates fire → frac=1.
    let mut rng = make_rng(603);
    let cc = ModConditionalComputation::new(4, 8, 4, &mut rng);
    let x = vec![0.3_f64; 4];
    let (_, frac) = cc.forward(&x, 0.0).expect("conditional computation forward should succeed");
    assert_eq!(frac, 1.0, "all layers should be active");
}

#[test]
fn test_mod_conditional_computation_output_finite() {
    let mut rng = make_rng(604);
    let cc = ModConditionalComputation::new(8, 16, 5, &mut rng);
    let x: Vec<f64> = (0..8).map(|i| i as f64 * 0.1).collect();
    let (out, _) = cc.forward(&x, 0.5).expect("conditional computation forward should succeed");
    assert!(out.iter().all(|v| v.is_finite()));
}

// ── ModPonderNet ──────────────────────────────────────────────────────────────

#[test]
fn test_mod_ponder_net_step_shapes() {
    let mut rng = make_rng(700);
    let pn = ModPonderNet::new(4, 8, 3, 0.2, 42, &mut rng);
    let h = vec![0.0_f64; 8];
    let x = vec![0.1_f64; 4];
    let (h_new, y, lam) = pn.step(&h, &x).expect("PonderNet step should succeed");
    assert_eq!(h_new.len(), 8);
    assert_eq!(y.len(), 3);
    assert!(lam > 0.0 && lam < 1.0);
}

#[test]
fn test_mod_ponder_net_step_wrong_h() {
    let mut rng = make_rng(701);
    let pn = ModPonderNet::new(4, 8, 3, 0.2, 1, &mut rng);
    assert!(pn.step(&[0.0; 5], &[0.1; 4]).is_err());
}

#[test]
fn test_mod_ponder_net_step_wrong_x() {
    let mut rng = make_rng(702);
    let pn = ModPonderNet::new(4, 8, 3, 0.2, 2, &mut rng);
    assert!(pn.step(&[0.0; 8], &[0.1; 5]).is_err());
}

#[test]
fn test_mod_ponder_net_forward_output_shape() {
    let mut rng = make_rng(703);
    let mut pn = ModPonderNet::new(4, 8, 3, 0.2, 10, &mut rng);
    let x = vec![0.1_f64; 4];
    let (out, p_halt, kl) = pn.forward(&x, 5).expect("PonderNet forward should succeed");
    assert_eq!(out.len(), 3);
    assert_eq!(p_halt.len(), 5);
    assert!(kl >= 0.0 && kl.is_finite());
}

#[test]
fn test_mod_ponder_net_halt_probs_sum_to_one() {
    let mut rng = make_rng(704);
    let mut pn = ModPonderNet::new(4, 8, 3, 0.2, 11, &mut rng);
    let x = vec![0.2_f64; 4];
    let (_, p_halt, _) = pn.forward(&x, 6).expect("PonderNet forward should succeed");
    let sum: f64 = p_halt.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "halt probs sum={}", sum);
}

#[test]
fn test_mod_ponder_net_kl_finite() {
    let mut rng = make_rng(705);
    let mut pn = ModPonderNet::new(6, 12, 4, 0.3, 12, &mut rng);
    let x = vec![0.3_f64; 6];
    let (_, _, kl) = pn.forward(&x, 8).expect("PonderNet forward should succeed");
    assert!(kl.is_finite());
}

#[test]
fn test_mod_ponder_net_forward_zero_max_steps_error() {
    let mut rng = make_rng(706);
    let mut pn = ModPonderNet::new(4, 8, 3, 0.2, 13, &mut rng);
    assert!(pn.forward(&[0.1_f64; 4], 0).is_err());
}

#[test]
fn test_mod_ponder_net_forward_wrong_input() {
    let mut rng = make_rng(707);
    let mut pn = ModPonderNet::new(4, 8, 3, 0.2, 14, &mut rng);
    assert!(pn.forward(&[0.1_f64; 5], 4).is_err());
}

// ── ModEfficiencyMetrics ──────────────────────────────────────────────────────

#[test]
fn test_mod_efficiency_metrics_basic() {
    let depths = vec![2, 3, 4, 2, 3];
    let report = ModEfficiencyMetrics::compute(&depths, 8, &[]).expect("efficiency metrics should succeed");
    assert_eq!(report.n_samples, 5);
    assert!((report.avg_depth - 2.8).abs() < 1e-10);
    assert!((report.flops_fraction - 2.8 / 8.0).abs() < 1e-10);
    assert_eq!(report.token_throughput_ratio, 1.0);
}

#[test]
fn test_mod_efficiency_metrics_with_token_fractions() {
    let depths = vec![2, 4];
    let fractions = vec![0.5, 0.75];
    let report = ModEfficiencyMetrics::compute(&depths, 4, &fractions).expect("efficiency metrics should succeed");
    assert!((report.token_throughput_ratio - 0.625).abs() < 1e-10);
}

#[test]
fn test_mod_efficiency_metrics_empty_depths_error() {
    assert!(ModEfficiencyMetrics::compute(&[], 4, &[]).is_err());
}

#[test]
fn test_mod_efficiency_metrics_zero_max_depth_error() {
    assert!(ModEfficiencyMetrics::compute(&[1, 2], 0, &[]).is_err());
}

#[test]
fn test_mod_efficiency_metrics_flops_saved() {
    let depths = vec![2, 2, 2, 2];
    let report = ModEfficiencyMetrics::compute(&depths, 8, &[]).expect("efficiency metrics should succeed");
    let saved = ModEfficiencyMetrics::flops_saved(&report);
    assert!((saved - 0.75).abs() < 1e-10);
}

#[test]
fn test_mod_efficiency_metrics_mean_ponder_cost() {
    let costs = vec![1.5, 2.5, 3.0];
    let mean = ModEfficiencyMetrics::mean_ponder_cost(&costs);
    assert!((mean - 7.0 / 3.0).abs() < 1e-10);
}

#[test]
fn test_mod_efficiency_metrics_mean_ponder_cost_empty() {
    assert_eq!(ModEfficiencyMetrics::mean_ponder_cost(&[]), 0.0);
}

#[test]
fn test_mod_efficiency_metrics_depth_distribution() {
    let depths = vec![0, 1, 1, 2, 0, 3];
    let hist = ModEfficiencyMetrics::depth_distribution(&depths, 4);
    assert_eq!(hist[0], 2);
    assert_eq!(hist[1], 2);
    assert_eq!(hist[2], 1);
    assert_eq!(hist[3], 1);
}

#[test]
fn test_mod_efficiency_metrics_routing_entropy_uniform() {
    let n = 4;
    let p = 1.0 / n as f64;
    let probs = vec![vec![p; n]; 8];
    let ent = ModEfficiencyMetrics::routing_entropy(&probs);
    let expected = -(p * p.ln()) * n as f64;
    assert!((ent - expected).abs() < 1e-10);
}

#[test]
fn test_mod_efficiency_metrics_routing_entropy_empty() {
    assert_eq!(ModEfficiencyMetrics::routing_entropy(&[]), 0.0);
}

// ── Integration tests ─────────────────────────────────────────────────────────

#[test]
fn test_full_mod_pipeline_router_then_metrics() {
    let mut rng = make_rng(900);
    let router = ModRouter::new(8, 0.5, &mut rng);
    let tokens = vec![0.1_f64; 8 * 10];
    let (routed, _, scores) = router.route(&tokens, 10).expect("token routing should succeed");
    let loss = router.auxiliary_loss(&scores);
    assert!(!routed.is_empty());
    assert!(loss.is_finite());

    // Check efficiency metrics on mock depths.
    let depths: Vec<usize> = (0..10).map(|i| (i % 5) + 1).collect();
    let report = ModEfficiencyMetrics::compute(&depths, 5, &[]).expect("efficiency metrics should succeed");
    assert!(report.flops_fraction > 0.0);
}

#[test]
fn test_full_mod_pipeline_transformer_then_metrics() {
    let mut rng = make_rng(901);
    let layer = ModTransformerLayer::new(4, 8, 1, 0.5, &mut rng).expect("transformer layer construction should succeed");
    let tokens = vec![0.2_f64; 4 * 6];
    let (out, routed, aux) = layer.forward(&tokens, 6).expect("transformer forward should succeed");
    assert_eq!(out.len(), 4 * 6);
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(aux.is_finite());
    // Simulate depth tracking.
    let fracs = vec![routed.len() as f64 / 6.0; 1];
    let depths = vec![1_usize; 1];
    let report = ModEfficiencyMetrics::compute(&depths, 1, &fracs).expect("efficiency metrics should succeed");
    assert!((report.flops_fraction - 1.0).abs() < 1e-10);
}

#[test]
fn test_full_mod_pipeline_ponder_net_then_metrics() {
    let mut rng = make_rng(902);
    let mut pn = ModPonderNet::new(4, 8, 2, 0.2, 99, &mut rng);
    let samples: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 0.1; 4]).collect();
    let mut ponder_costs = Vec::new();
    for x in &samples {
        let (_, p_halt, _) = pn.forward(x, 4).expect("PonderNet forward should succeed");
        let mean_steps: f64 = p_halt
            .iter()
            .enumerate()
            .map(|(n, &p)| (n + 1) as f64 * p)
            .sum();
        ponder_costs.push(mean_steps);
    }
    let mean_cost = ModEfficiencyMetrics::mean_ponder_cost(&ponder_costs);
    assert!(mean_cost >= 1.0);
    assert!(mean_cost.is_finite());
}

#[test]
fn test_early_exit_and_efficiency_report() {
    let mut rng = make_rng(903);
    let net = ModEarlyExit::new(4, 8, 5, 0.7, 0.1, &mut rng);
    let tokens = vec![0.15_f64; 4 * 8];
    let (out, depths, ponder) = net.forward_with_exit(&tokens, 8).expect("early exit forward should succeed");
    assert_eq!(out.len(), 4 * 8);
    assert!(ponder.is_finite());
    let report = ModEfficiencyMetrics::compute(&depths, 5, &[]).expect("efficiency metrics should succeed");
    let saved = ModEfficiencyMetrics::flops_saved(&report);
    assert!((0.0..=1.0).contains(&saved));
}

#[test]
fn test_universal_transformer_consistency() {
    // Same input always produces same output (no stochasticity in forward).
    let mut rng = make_rng(904);
    let ut = ModUniversalTransformer::new(4, 8, 0.5, &mut rng);
    let tokens = vec![0.25_f64; 4 * 3];
    let (out1, _) = ut.forward(&tokens, 3, 5).expect("universal transformer forward should succeed");
    let (out2, _) = ut.forward(&tokens, 3, 5).expect("universal transformer forward should succeed");
    for (&a, &b) in out1.iter().zip(out2.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}
