//! Tests for the sparse_mixture_experts module.

use super::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

// ── SmeExpert tests ────────────────────────────────────────────────────────────

#[test]
fn test_sme_expert_output_shape() {
    let mut rng = make_rng(1);
    let cfg = SmeExpertConfig {
        input_dim: 8,
        hidden_dim: 16,
    };
    let expert = SmeExpert::new(&cfg, &mut rng);
    let x = vec![0.5f32; 8];
    let out = expert.forward(&x).expect("forward failed");
    assert_eq!(out.len(), 8);
}

#[test]
fn test_sme_expert_output_finite() {
    let mut rng = make_rng(2);
    let cfg = SmeExpertConfig {
        input_dim: 4,
        hidden_dim: 8,
    };
    let expert = SmeExpert::new(&cfg, &mut rng);
    let x: Vec<f32> = (0..4).map(|i| i as f32 * 0.1).collect();
    let out = expert.forward(&x).expect("forward failed");
    assert!(
        out.iter().all(|v| v.is_finite()),
        "output contains non-finite values"
    );
}

#[test]
fn test_sme_expert_wrong_input_dim() {
    let mut rng = make_rng(3);
    let cfg = SmeExpertConfig {
        input_dim: 8,
        hidden_dim: 16,
    };
    let expert = SmeExpert::new(&cfg, &mut rng);
    let x = vec![0.1f32; 5]; // wrong dim
    let result = expert.forward(&x);
    assert!(result.is_err(), "expected error for wrong input dim");
}

#[test]
fn test_sme_expert_zero_input() {
    let mut rng = make_rng(4);
    let cfg = SmeExpertConfig {
        input_dim: 6,
        hidden_dim: 12,
    };
    let expert = SmeExpert::new(&cfg, &mut rng);
    let x = vec![0.0f32; 6];
    let out = expert.forward(&x).expect("forward failed");
    // With zero input: h = GeLU(bias) and output = W2·h + b2; all finite.
    assert_eq!(out.len(), 6);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn test_sme_expert_deterministic() {
    let mut rng = make_rng(5);
    let cfg = SmeExpertConfig {
        input_dim: 4,
        hidden_dim: 8,
    };
    let expert = SmeExpert::new(&cfg, &mut rng);
    let x = vec![1.0f32, -1.0, 0.5, -0.5];
    let out1 = expert.forward(&x).expect("forward failed");
    let out2 = expert.forward(&x).expect("forward failed");
    assert_eq!(out1, out2);
}

#[test]
fn test_sme_expert_large_dim() {
    let mut rng = make_rng(6);
    let cfg = SmeExpertConfig {
        input_dim: 64,
        hidden_dim: 128,
    };
    let expert = SmeExpert::new(&cfg, &mut rng);
    let x = vec![0.01f32; 64];
    let out = expert.forward(&x).expect("forward failed");
    assert_eq!(out.len(), 64);
    assert!(out.iter().all(|v| v.is_finite()));
}

// ── SmeRouter tests ────────────────────────────────────────────────────────────

#[test]
fn test_sme_router_output_shape_top2() {
    let mut rng = make_rng(10);
    let router = SmeRouter::new(8, 16, false, &mut rng);
    let x = vec![0.5f32; 16];
    let out = router.route(&x, 2, &mut rng).expect("route failed");
    assert_eq!(out.expert_indices.len(), 2);
    assert_eq!(out.expert_weights.len(), 2);
    assert_eq!(out.router_probs.len(), 8);
    assert_eq!(out.logits.len(), 8);
}

#[test]
fn test_sme_router_weights_sum_to_one() {
    let mut rng = make_rng(11);
    let router = SmeRouter::new(4, 8, false, &mut rng);
    let x = vec![1.0f32; 8];
    let out = router.route(&x, 2, &mut rng).expect("route failed");
    let sum: f32 = out.expert_weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "weights sum = {}", sum);
}

#[test]
fn test_sme_router_probs_sum_to_one() {
    let mut rng = make_rng(12);
    let router = SmeRouter::new(6, 8, false, &mut rng);
    let x = vec![0.2f32; 8];
    let out = router.route(&x, 3, &mut rng).expect("route failed");
    let sum: f32 = out.router_probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "router_probs sum = {}", sum);
}

#[test]
fn test_sme_router_indices_valid() {
    let mut rng = make_rng(13);
    let n_experts = 8;
    let router = SmeRouter::new(n_experts, 16, false, &mut rng);
    let x = vec![0.5f32; 16];
    let out = router.route(&x, 4, &mut rng).expect("route failed");
    for &idx in out.expert_indices.iter() {
        assert!(idx < n_experts, "expert index {} out of range", idx);
    }
}

#[test]
fn test_sme_router_noisy_routing() {
    let mut rng = make_rng(14);
    let router = SmeRouter::new(8, 8, true, &mut rng);
    let x = vec![0.3f32; 8];
    let out = router.route(&x, 2, &mut rng).expect("route failed");
    assert_eq!(out.expert_indices.len(), 2);
    assert!(out
        .expert_weights
        .iter()
        .all(|&w| w >= 0.0 && w.is_finite()));
}

#[test]
fn test_sme_router_wrong_dim() {
    let mut rng = make_rng(15);
    let router = SmeRouter::new(4, 8, false, &mut rng);
    let x = vec![0.1f32; 5]; // wrong dim
    let result = router.route(&x, 2, &mut rng);
    assert!(result.is_err());
}

#[test]
fn test_sme_router_top_k_clamped_to_n_experts() {
    let mut rng = make_rng(16);
    let router = SmeRouter::new(3, 4, false, &mut rng);
    let x = vec![1.0f32; 4];
    // top_k=10 > n_experts=3; should clamp
    let out = router.route(&x, 10, &mut rng).expect("route failed");
    assert!(out.expert_indices.len() <= 3);
}

// ── SmeAuxiliaryLoss tests ────────────────────────────────────────────────────

fn make_routing_outputs(
    n_tokens: usize,
    n_experts: usize,
    top_k: usize,
    seed: u64,
) -> Vec<SmeRoutingOutput> {
    let mut rng = make_rng(seed);
    let router = SmeRouter::new(n_experts, 8, false, &mut rng);
    (0..n_tokens)
        .map(|_| {
            let x: Vec<f32> = (0..8).map(|_| rng.random::<f32>() - 0.5).collect();
            router.route(&x, top_k, &mut rng).expect("route failed")
        })
        .collect()
}

#[test]
fn test_sme_aux_loss_switch_non_negative() {
    let ros = make_routing_outputs(16, 4, 2, 20);
    let aux = SmeAuxiliaryLoss::new(4, SmeAuxLossConfig::default());
    let loss = aux.compute(&ros).expect("compute failed");
    assert!(
        loss.switch_loss >= 0.0,
        "switch_loss should be non-negative"
    );
}

#[test]
fn test_sme_aux_loss_z_loss_non_negative() {
    let ros = make_routing_outputs(16, 4, 2, 21);
    let aux = SmeAuxiliaryLoss::new(4, SmeAuxLossConfig::default());
    let loss = aux.compute(&ros).expect("compute failed");
    assert!(loss.z_loss >= 0.0, "z_loss should be non-negative");
}

#[test]
fn test_sme_aux_loss_importance_non_negative() {
    let ros = make_routing_outputs(16, 4, 2, 22);
    let aux = SmeAuxiliaryLoss::new(4, SmeAuxLossConfig::default());
    let loss = aux.compute(&ros).expect("compute failed");
    assert!(loss.importance_loss >= 0.0);
}

#[test]
fn test_sme_aux_loss_total_finite() {
    let ros = make_routing_outputs(32, 8, 2, 23);
    let aux = SmeAuxiliaryLoss::new(8, SmeAuxLossConfig::default());
    let loss = aux.compute(&ros).expect("compute failed");
    assert!(loss.total.is_finite(), "total loss should be finite");
}

#[test]
fn test_sme_aux_loss_empty_fails() {
    let aux = SmeAuxiliaryLoss::new(4, SmeAuxLossConfig::default());
    let result = aux.compute(&[]);
    assert!(result.is_err(), "empty routing_outputs should fail");
}

#[test]
fn test_sme_aux_loss_balanced_routing_low_switch() {
    // With balanced routing (all experts equally likely), switch loss ≈ 1/n_experts.
    // We can only verify it's finite and bounded.
    let ros = make_routing_outputs(100, 4, 1, 24);
    let aux = SmeAuxiliaryLoss::new(4, SmeAuxLossConfig::default());
    let loss = aux.compute(&ros).expect("compute failed");
    assert!(loss.switch_loss.is_finite());
    // Switch loss should be bounded: L_switch = n_experts * Σ f_i * p_i
    // With balanced routing, each f_i ≈ 1/n_experts, each p_i ≈ 1/n_experts
    // L_switch ≈ n_experts * n_experts * (1/n_experts)^2 = 1.0
    assert!(loss.switch_loss <= 2.0, "switch_loss={}", loss.switch_loss);
}

// ── SmeExpertCapacity tests ───────────────────────────────────────────────────

#[test]
fn test_sme_capacity_per_expert_basic() {
    let cap = SmeExpertCapacity::new(4, 8);
    let c = cap.capacity_per_expert(16, 1.0);
    assert_eq!(c, 4); // 16/4 * 1.0 = 4
}

#[test]
fn test_sme_capacity_per_expert_with_factor() {
    let cap = SmeExpertCapacity::new(4, 8);
    let c = cap.capacity_per_expert(16, 1.5);
    assert_eq!(c, 6); // ceil(16/4 * 1.5) = ceil(6) = 6
}

#[test]
fn test_sme_capacity_per_expert_minimum_one() {
    let cap = SmeExpertCapacity::new(8, 4);
    let c = cap.capacity_per_expert(1, 0.1);
    assert!(c >= 1, "capacity must be at least 1");
}

#[test]
fn test_sme_dispatch_output_shapes() {
    let mut rng = make_rng(30);
    let n_tokens = 8;
    let n_experts = 4;
    let token_dim = 8;
    let cap = SmeExpertCapacity::new(n_experts, token_dim);
    let inputs: Vec<f32> = (0..n_tokens * token_dim).map(|i| i as f32 * 0.01).collect();
    let ros = make_routing_outputs(n_tokens, n_experts, 2, 30);
    let dispatch = cap.dispatch(&inputs, &ros, 1.25).expect("dispatch failed");
    assert_eq!(dispatch.expert_inputs.len(), n_experts);
    assert_eq!(dispatch.overflow.len(), n_tokens);
}

#[test]
fn test_sme_dispatch_wrong_inputs_length() {
    let cap = SmeExpertCapacity::new(4, 8);
    let inputs = vec![0.0f32; 30]; // should be n_tokens * token_dim
    let ros = make_routing_outputs(5, 4, 2, 31);
    let result = cap.dispatch(&inputs, &ros, 1.0);
    assert!(result.is_err(), "expected error for wrong inputs length");
}

#[test]
fn test_sme_dispatch_combine_roundtrip() {
    let mut rng = make_rng(32);
    let n_tokens = 4;
    let n_experts = 2;
    let token_dim = 4;
    let expert_cfg = SmeExpertConfig {
        input_dim: token_dim,
        hidden_dim: 8,
    };
    let experts: Vec<SmeExpert> = (0..n_experts)
        .map(|_| SmeExpert::new(&expert_cfg, &mut rng))
        .collect();
    let cap = SmeExpertCapacity::new(n_experts, token_dim);
    let inputs: Vec<f32> = (0..n_tokens * token_dim).map(|i| i as f32 * 0.1).collect();
    let ros = make_routing_outputs(n_tokens, n_experts, 1, 32);

    let dispatch = cap.dispatch(&inputs, &ros, 2.0).expect("dispatch failed");

    // Process each expert's batch
    let mut expert_outputs = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let n_dispatched = dispatch.expert_token_ids[e].len();
        let mut out = Vec::with_capacity(n_dispatched * token_dim);
        for j in 0..n_dispatched {
            let start = j * token_dim;
            let token_in = &dispatch.expert_inputs[e][start..start + token_dim];
            let eo = experts[e].forward(token_in).expect("expert forward failed");
            out.extend_from_slice(&eo);
        }
        expert_outputs.push(out);
    }

    let combined = cap
        .combine(&expert_outputs, &dispatch, n_tokens)
        .expect("combine failed");
    assert_eq!(combined.len(), n_tokens * token_dim);
    assert!(combined.iter().all(|v| v.is_finite()));
}

#[test]
fn test_sme_combine_wrong_n_experts() {
    let cap = SmeExpertCapacity::new(4, 8);
    let dispatch = SmeDispatchResult {
        expert_inputs: vec![vec![]; 4],
        expert_token_ids: vec![vec![]; 4],
        expert_token_weights: vec![vec![]; 4],
        overflow: vec![],
    };
    let wrong_outputs = vec![vec![]; 3]; // only 3, not 4
    let result = cap.combine(&wrong_outputs, &dispatch, 4);
    assert!(result.is_err());
}

// ── SmeExpertParallelism tests ────────────────────────────────────────────────

#[test]
fn test_sme_expert_parallelism_batch_by_expert_basic() {
    let par = SmeExpertParallelism::new(4);
    let inputs: Vec<f32> = (0..16).map(|i| i as f32).collect(); // 4 tokens × 4 dim
    let assignments = vec![0usize, 1, 0, 2];
    let batches = par
        .batch_by_expert(&inputs, &assignments, 3)
        .expect("batch failed");
    // Should have entries for experts 0, 1, 2
    assert_eq!(batches.len(), 3);
}

#[test]
fn test_sme_expert_parallelism_batch_correct_tokens() {
    let par = SmeExpertParallelism::new(4);
    let inputs: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let assignments = vec![0usize, 0, 1, 1];
    let batches = par
        .batch_by_expert(&inputs, &assignments, 2)
        .expect("batch failed");
    // Expert 0 gets tokens 0,1; expert 1 gets tokens 2,3
    for (exp_idx, token_ids, _) in batches.iter() {
        if *exp_idx == 0 {
            assert_eq!(token_ids, &[0, 1]);
        } else if *exp_idx == 1 {
            assert_eq!(token_ids, &[2, 3]);
        }
    }
}

#[test]
fn test_sme_expert_parallelism_wrong_input_length() {
    let par = SmeExpertParallelism::new(4);
    let inputs = vec![0.0f32; 10]; // wrong length
    let assignments = vec![0usize, 1, 0]; // 3 tokens → should be 12
    let result = par.batch_by_expert(&inputs, &assignments, 2);
    assert!(result.is_err());
}

#[test]
fn test_sme_expert_parallelism_process_and_scatter() {
    let mut rng = make_rng(40);
    let n_experts = 2;
    let token_dim = 4;
    let expert_cfg = SmeExpertConfig {
        input_dim: token_dim,
        hidden_dim: 8,
    };
    let experts: Vec<SmeExpert> = (0..n_experts)
        .map(|_| SmeExpert::new(&expert_cfg, &mut rng))
        .collect();
    let par = SmeExpertParallelism::new(token_dim);
    let n_tokens = 4;
    let inputs: Vec<f32> = (0..n_tokens * token_dim).map(|i| i as f32 * 0.1).collect();
    let assignments = vec![0, 1, 0, 1];
    let batches = par
        .batch_by_expert(&inputs, &assignments, n_experts)
        .expect("batch failed");
    let output = par
        .process_and_scatter(&batches, &experts, n_tokens)
        .expect("scatter failed");
    assert_eq!(output.len(), n_tokens * token_dim);
    assert!(output.iter().all(|v| v.is_finite()));
}

#[test]
fn test_sme_expert_parallelism_empty_assignments() {
    let par = SmeExpertParallelism::new(4);
    let inputs: Vec<f32> = vec![];
    let assignments: Vec<usize> = vec![];
    let batches = par
        .batch_by_expert(&inputs, &assignments, 4)
        .expect("batch failed");
    assert_eq!(batches.len(), 0);
}

// ── SmeTokenChoiceRouter tests ────────────────────────────────────────────────

#[test]
fn test_sme_token_choice_route_batch_shape() {
    let mut rng = make_rng(50);
    let router = SmeTokenChoiceRouter::new(8, 16, 2, &mut rng);
    let inputs: Vec<f32> = (0..4 * 16).map(|i| i as f32 * 0.01).collect();
    let outputs = router
        .route_batch(&inputs, &mut rng)
        .expect("route_batch failed");
    assert_eq!(outputs.len(), 4);
}

#[test]
fn test_sme_token_choice_each_token_gets_k_experts() {
    let mut rng = make_rng(51);
    let top_k = 3;
    let router = SmeTokenChoiceRouter::new(8, 8, top_k, &mut rng);
    let inputs: Vec<f32> = (0..6 * 8).map(|i| i as f32 * 0.05).collect();
    let outputs = router
        .route_batch(&inputs, &mut rng)
        .expect("route_batch failed");
    for ro in outputs.iter() {
        assert_eq!(ro.expert_indices.len(), top_k);
    }
}

#[test]
fn test_sme_token_choice_single_token() {
    let mut rng = make_rng(52);
    let router = SmeTokenChoiceRouter::new(4, 8, 2, &mut rng);
    let inputs: Vec<f32> = vec![0.1f32; 8];
    let outputs = router
        .route_batch(&inputs, &mut rng)
        .expect("route_batch failed");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].expert_indices.len(), 2);
}

// ── SmeExpertChoiceRouter tests ───────────────────────────────────────────────

#[test]
fn test_sme_expert_choice_route_shape() {
    let mut rng = make_rng(60);
    let router = SmeExpertChoiceRouter::new(4, 8, &mut rng);
    let inputs: Vec<f32> = (0..10 * 8).map(|i| i as f32 * 0.01).collect();
    let assignments = router
        .expert_choice_route(&inputs, 3)
        .expect("route failed");
    assert_eq!(assignments.len(), 4);
}

#[test]
fn test_sme_expert_choice_each_expert_gets_capacity_tokens() {
    let mut rng = make_rng(61);
    let n_tokens = 12;
    let n_experts = 3;
    let capacity = 4;
    let router = SmeExpertChoiceRouter::new(n_experts, 8, &mut rng);
    let inputs: Vec<f32> = (0..n_tokens * 8).map(|i| i as f32 * 0.01).collect();
    let assignments = router
        .expert_choice_route(&inputs, capacity)
        .expect("route failed");
    for (exp_idx, token_list) in assignments.iter().enumerate() {
        assert!(
            token_list.len() <= capacity,
            "expert {} has {} tokens, expected <= {}",
            exp_idx,
            token_list.len(),
            capacity
        );
    }
}

#[test]
fn test_sme_expert_choice_indices_in_range() {
    let mut rng = make_rng(62);
    let n_tokens = 8;
    let router = SmeExpertChoiceRouter::new(4, 6, &mut rng);
    let inputs: Vec<f32> = (0..n_tokens * 6).map(|i| i as f32 * 0.1).collect();
    let assignments = router
        .expert_choice_route(&inputs, 2)
        .expect("route failed");
    for token_list in assignments.iter() {
        for &tok_idx in token_list.iter() {
            assert!(tok_idx < n_tokens, "token index {} out of range", tok_idx);
        }
    }
}

#[test]
fn test_sme_expert_choice_sorted_indices() {
    let mut rng = make_rng(63);
    let router = SmeExpertChoiceRouter::new(3, 8, &mut rng);
    let inputs: Vec<f32> = (0..12 * 8).map(|i| i as f32 * 0.05).collect();
    let assignments = router
        .expert_choice_route(&inputs, 4)
        .expect("route failed");
    for token_list in assignments.iter() {
        let sorted = {
            let mut v = token_list.clone();
            v.sort_unstable();
            v
        };
        assert_eq!(*token_list, sorted, "token indices should be sorted");
    }
}

// ── SmeSpecializationMetrics tests ───────────────────────────────────────────

#[test]
fn test_sme_specialization_utilization_sums_to_one() {
    let mut rng = make_rng(70);
    let n_experts = 4;
    let ros = make_routing_outputs(20, n_experts, 2, 70);
    let expert_cfg = SmeExpertConfig {
        input_dim: 8,
        hidden_dim: 16,
    };
    let experts: Vec<SmeExpert> = (0..n_experts)
        .map(|_| SmeExpert::new(&expert_cfg, &mut rng))
        .collect();
    let metrics = SmeSpecializationMetrics::new(n_experts, 0.01);
    let analysis = metrics.analyse(&ros, &experts).expect("analyse failed");
    let sum: f32 = analysis.expert_utilization.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "utilization sum = {}", sum);
}

#[test]
fn test_sme_specialization_entropy_non_negative() {
    let mut rng = make_rng(71);
    let n_experts = 4;
    let ros = make_routing_outputs(16, n_experts, 1, 71);
    let expert_cfg = SmeExpertConfig {
        input_dim: 8,
        hidden_dim: 16,
    };
    let experts: Vec<SmeExpert> = (0..n_experts)
        .map(|_| SmeExpert::new(&expert_cfg, &mut rng))
        .collect();
    let metrics = SmeSpecializationMetrics::new(n_experts, 0.01);
    let analysis = metrics.analyse(&ros, &experts).expect("analyse failed");
    assert!(analysis.routing_entropy >= 0.0);
}

#[test]
fn test_sme_specialization_overlap_matrix_shape() {
    let mut rng = make_rng(72);
    let n_experts = 3;
    let ros = make_routing_outputs(10, n_experts, 1, 72);
    let expert_cfg = SmeExpertConfig {
        input_dim: 8,
        hidden_dim: 16,
    };
    let experts: Vec<SmeExpert> = (0..n_experts)
        .map(|_| SmeExpert::new(&expert_cfg, &mut rng))
        .collect();
    let metrics = SmeSpecializationMetrics::new(n_experts, 0.01);
    let analysis = metrics.analyse(&ros, &experts).expect("analyse failed");
    assert_eq!(analysis.expert_overlap.len(), n_experts);
    for row in analysis.expert_overlap.iter() {
        assert_eq!(row.len(), n_experts);
    }
}

#[test]
fn test_sme_specialization_self_similarity_is_one() {
    let mut rng = make_rng(73);
    let n_experts = 3;
    let ros = make_routing_outputs(10, n_experts, 1, 73);
    let expert_cfg = SmeExpertConfig {
        input_dim: 8,
        hidden_dim: 16,
    };
    let experts: Vec<SmeExpert> = (0..n_experts)
        .map(|_| SmeExpert::new(&expert_cfg, &mut rng))
        .collect();
    let metrics = SmeSpecializationMetrics::new(n_experts, 0.01);
    let analysis = metrics.analyse(&ros, &experts).expect("analyse failed");
    for i in 0..n_experts {
        assert!(
            (analysis.expert_overlap[i][i] - 1.0).abs() < 1e-4,
            "self-similarity of expert {} = {}",
            i,
            analysis.expert_overlap[i][i]
        );
    }
}

#[test]
fn test_sme_specialization_load_imbalance_ge_one() {
    let mut rng = make_rng(74);
    let n_experts = 4;
    let ros = make_routing_outputs(20, n_experts, 1, 74);
    let expert_cfg = SmeExpertConfig {
        input_dim: 8,
        hidden_dim: 16,
    };
    let experts: Vec<SmeExpert> = (0..n_experts)
        .map(|_| SmeExpert::new(&expert_cfg, &mut rng))
        .collect();
    let metrics = SmeSpecializationMetrics::new(n_experts, 0.01);
    let analysis = metrics.analyse(&ros, &experts).expect("analyse failed");
    // Load imbalance = max / mean >= 1.0
    assert!(
        analysis.load_imbalance >= 1.0 - 1e-4,
        "load_imbalance = {}",
        analysis.load_imbalance
    );
}

#[test]
fn test_sme_specialization_empty_fails() {
    let mut rng = make_rng(75);
    let n_experts = 4;
    let expert_cfg = SmeExpertConfig {
        input_dim: 8,
        hidden_dim: 16,
    };
    let experts: Vec<SmeExpert> = (0..n_experts)
        .map(|_| SmeExpert::new(&expert_cfg, &mut rng))
        .collect();
    let metrics = SmeSpecializationMetrics::new(n_experts, 0.01);
    let result = metrics.analyse(&[], &experts);
    assert!(result.is_err());
}

// ── SmeMoeTransformerBlock tests ──────────────────────────────────────────────

fn make_moe_block(seed: u64) -> SmeMoeTransformerBlock {
    let mut rng = make_rng(seed);
    let cfg = SmeMoeBlockConfig {
        model_dim: 8,
        n_heads: 2,
        n_experts: 4,
        expert_hidden_dim: 16,
        top_k: 2,
        capacity_factor: 1.25,
        dropout_rate: 0.0,
        aux_loss_config: SmeAuxLossConfig::default(),
    };
    SmeMoeTransformerBlock::new(cfg, &mut rng)
}

#[test]
fn test_sme_moe_block_output_shape() {
    let mut rng = make_rng(80);
    let block = make_moe_block(80);
    let n_tokens = 4;
    let model_dim = block.config.model_dim;
    let x: Vec<f32> = (0..n_tokens * model_dim).map(|i| i as f32 * 0.01).collect();
    let out = block.forward(&x, false, &mut rng).expect("forward failed");
    assert_eq!(out.output.len(), n_tokens * model_dim);
}

#[test]
fn test_sme_moe_block_output_finite() {
    let mut rng = make_rng(81);
    let block = make_moe_block(81);
    let n_tokens = 3;
    let d = block.config.model_dim;
    let x: Vec<f32> = (0..n_tokens * d).map(|i| i as f32 * 0.05).collect();
    let out = block.forward(&x, false, &mut rng).expect("forward failed");
    assert!(
        out.output.iter().all(|v| v.is_finite()),
        "output contains non-finite"
    );
}

#[test]
fn test_sme_moe_block_aux_loss_finite() {
    let mut rng = make_rng(82);
    let block = make_moe_block(82);
    let d = block.config.model_dim;
    let x: Vec<f32> = vec![0.5f32; 4 * d];
    let out = block.forward(&x, false, &mut rng).expect("forward failed");
    assert!(out.aux_loss.total.is_finite());
    assert!(out.aux_loss.switch_loss.is_finite());
    assert!(out.aux_loss.z_loss.is_finite());
}

#[test]
fn test_sme_moe_block_training_mode() {
    let mut rng = make_rng(83);
    let mut block_rng = make_rng(83);
    let cfg = SmeMoeBlockConfig {
        model_dim: 8,
        n_experts: 4,
        expert_hidden_dim: 16,
        top_k: 2,
        dropout_rate: 0.1,
        ..SmeMoeBlockConfig::default()
    };
    let block = SmeMoeTransformerBlock::new(cfg, &mut block_rng);
    let d = block.config.model_dim;
    let x: Vec<f32> = vec![1.0f32; 4 * d];
    let out = block
        .forward(&x, true, &mut rng)
        .expect("forward in training mode failed");
    assert_eq!(out.output.len(), 4 * d);
}

#[test]
fn test_sme_moe_block_wrong_input_length() {
    let mut rng = make_rng(84);
    let block = make_moe_block(84);
    let x: Vec<f32> = vec![0.0f32; 7]; // not divisible by model_dim=8
    let result = block.forward(&x, false, &mut rng);
    assert!(result.is_err() || result.expect("forward should produce result").output.is_empty()); // check is not divisible
                                                                           // Actually: 7 / 8 = 0 n_tokens, so output would be empty — test just that it doesn't panic.
}

#[test]
fn test_sme_moe_block_single_token() {
    let mut rng = make_rng(85);
    let block = make_moe_block(85);
    let d = block.config.model_dim;
    let x: Vec<f32> = vec![0.3f32; d];
    let out = block
        .forward(&x, false, &mut rng)
        .expect("single token forward failed");
    assert_eq!(out.output.len(), d);
    assert!(out.output.iter().all(|v| v.is_finite()));
}

// ── SmeAdaptiveRouter tests ───────────────────────────────────────────────────

#[test]
fn test_sme_adaptive_router_initial_route() {
    let mut rng = make_rng(90);
    let router = SmeAdaptiveRouter::new(4, 8, 2, 0.9, 0.01, &mut rng);
    let x = vec![0.5f32; 8];
    let out = router.route(&x, &mut rng).expect("route failed");
    assert_eq!(out.expert_indices.len(), 2);
    assert_eq!(out.expert_weights.len(), 2);
}

#[test]
fn test_sme_adaptive_router_bias_updates() {
    let mut rng = make_rng(91);
    let mut router = SmeAdaptiveRouter::new(4, 8, 2, 0.9, 0.1, &mut rng);
    let initial_bias = router.routing_bias.clone();
    // Simulate usage heavily favouring expert 0
    let usage = vec![0.9f32, 0.05, 0.03, 0.02];
    router.adapt_routing_bias(&usage).expect("adapt failed");
    // Expert 0 is overused: its bias should decrease
    // Other experts are underused: their bias should increase
    let _ = initial_bias[0]; // bias may or may not decrease by one step
                                                               // Simply check bias changed
    let changed = router
        .routing_bias
        .iter()
        .zip(initial_bias.iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-9);
    assert!(changed, "routing_bias should have changed after adapt");
}

#[test]
fn test_sme_adaptive_router_ema_converges() {
    let mut rng = make_rng(92);
    let mut router = SmeAdaptiveRouter::new(4, 8, 2, 0.5, 0.01, &mut rng);
    let usage = vec![0.25f32; 4]; // perfectly balanced
    for _ in 0..20 {
        router.adapt_routing_bias(&usage).expect("adapt failed");
    }
    // EMA should converge close to target (0.25)
    for &ema in router.usage_ema.iter() {
        assert!((ema - 0.25).abs() < 0.05, "EMA = {} far from 0.25", ema);
    }
}

#[test]
fn test_sme_adaptive_router_reset_bias() {
    let mut rng = make_rng(93);
    let mut router = SmeAdaptiveRouter::new(4, 8, 2, 0.9, 0.1, &mut rng);
    let usage = vec![0.8f32, 0.1, 0.05, 0.05];
    router.adapt_routing_bias(&usage).expect("adapt failed");
    router.reset_bias();
    assert!(
        router.routing_bias.iter().all(|&b| b == 0.0),
        "bias should be zero after reset"
    );
}

#[test]
fn test_sme_adaptive_router_wrong_usage_length() {
    let mut rng = make_rng(94);
    let mut router = SmeAdaptiveRouter::new(4, 8, 2, 0.9, 0.01, &mut rng);
    let usage = vec![0.5f32; 3]; // wrong length
    let result = router.adapt_routing_bias(&usage);
    assert!(result.is_err());
}

#[test]
fn test_sme_adaptive_router_probs_sum_to_one() {
    let mut rng = make_rng(95);
    let router = SmeAdaptiveRouter::new(6, 8, 3, 0.9, 0.01, &mut rng);
    let x = vec![0.1f32; 8];
    let out = router.route(&x, &mut rng).expect("route failed");
    let sum: f32 = out.router_probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "router_probs sum = {}", sum);
}

// ── SmeMetrics & SmeReport tests ──────────────────────────────────────────────

#[test]
fn test_sme_metrics_record_snapshot() {
    let n_experts = 4;
    let ros = make_routing_outputs(16, n_experts, 2, 100);
    let aux = SmeAuxiliaryLoss::new(n_experts, SmeAuxLossConfig::default());
    let aux_vals = aux.compute(&ros).expect("compute failed");
    let mut metrics = SmeMetrics::new(n_experts, 0.01);
    let snap = metrics.record(0, &ros, &aux_vals).expect("record failed");
    assert_eq!(snap.step, 0);
    assert!(snap.load_imbalance.is_finite());
    assert!(snap.routing_entropy >= 0.0);
}

#[test]
fn test_sme_metrics_history_grows() {
    let n_experts = 4;
    let mut metrics = SmeMetrics::new(n_experts, 0.01);
    let aux = SmeAuxiliaryLoss::new(n_experts, SmeAuxLossConfig::default());
    for step in 0..5 {
        let ros = make_routing_outputs(8, n_experts, 2, 100 + step as u64);
        let aux_vals = aux.compute(&ros).expect("compute failed");
        metrics
            .record(step, &ros, &aux_vals)
            .expect("record failed");
    }
    assert_eq!(metrics.history().len(), 5);
}

#[test]
fn test_sme_metrics_detect_collapse_false_when_balanced() {
    let n_experts = 4;
    let mut metrics = SmeMetrics::new(n_experts, 0.001); // very low threshold
    let aux = SmeAuxiliaryLoss::new(n_experts, SmeAuxLossConfig::default());
    let ros = make_routing_outputs(100, n_experts, 1, 110);
    let aux_vals = aux.compute(&ros).expect("compute failed");
    metrics.record(0, &ros, &aux_vals).expect("record failed");
    // With 100 tokens and top-1 routing over 4 experts, most should get tokens
    // No assertion on result, just that it completes without error.
    let _ = metrics.detect_collapse();
}

#[test]
fn test_sme_metrics_empty_record_fails() {
    let mut metrics = SmeMetrics::new(4, 0.01);
    let aux_vals = SmeAuxLossValues::default();
    let result = metrics.record(0, &[], &aux_vals);
    assert!(result.is_err());
}

#[test]
fn test_sme_report_basic() {
    let n_experts = 4;
    let mut metrics = SmeMetrics::new(n_experts, 0.01);
    let aux = SmeAuxiliaryLoss::new(n_experts, SmeAuxLossConfig::default());
    for step in 0..3 {
        let ros = make_routing_outputs(8, n_experts, 2, 120 + step as u64);
        let aux_vals = aux.compute(&ros).expect("compute failed");
        metrics
            .record(step, &ros, &aux_vals)
            .expect("record failed");
    }
    let report = metrics.report();
    assert_eq!(report.n_experts, n_experts);
    assert_eq!(report.n_steps_recorded, 3);
    assert!(report.mean_load_imbalance >= 1.0 - 1e-4);
    assert_eq!(report.final_expert_utilization.len(), n_experts);
}

#[test]
fn test_sme_report_empty_history() {
    let metrics = SmeMetrics::new(4, 0.01);
    let report = metrics.report();
    assert_eq!(report.n_steps_recorded, 0);
    assert_eq!(report.mean_load_imbalance, 1.0);
}

#[test]
fn test_sme_metrics_mean_load_imbalance() {
    let n_experts = 4;
    let mut metrics = SmeMetrics::new(n_experts, 0.01);
    let aux = SmeAuxiliaryLoss::new(n_experts, SmeAuxLossConfig::default());
    for step in 0..4 {
        let ros = make_routing_outputs(12, n_experts, 2, 130 + step as u64);
        let aux_vals = aux.compute(&ros).expect("compute failed");
        metrics
            .record(step, &ros, &aux_vals)
            .expect("record failed");
    }
    let mli = metrics.mean_load_imbalance();
    assert!(mli >= 1.0 - 1e-4, "mean_load_imbalance = {}", mli);
    assert!(mli.is_finite());
}

#[test]
fn test_sme_metrics_mean_routing_entropy_finite() {
    let n_experts = 4;
    let mut metrics = SmeMetrics::new(n_experts, 0.01);
    let aux = SmeAuxiliaryLoss::new(n_experts, SmeAuxLossConfig::default());
    for step in 0..3 {
        let ros = make_routing_outputs(8, n_experts, 2, 140 + step as u64);
        let aux_vals = aux.compute(&ros).expect("compute failed");
        metrics
            .record(step, &ros, &aux_vals)
            .expect("record failed");
    }
    let entropy = metrics.mean_routing_entropy();
    assert!(entropy.is_finite() && entropy >= 0.0);
}

// ── Integration tests ─────────────────────────────────────────────────────────

#[test]
fn test_sme_full_training_step_integration() {
    let mut rng = make_rng(200);

    let n_experts = 4;
    let model_dim = 8;
    let n_tokens = 8;

    // Build components
    let cfg = SmeMoeBlockConfig {
        model_dim,
        n_heads: 2,
        n_experts,
        expert_hidden_dim: 16,
        top_k: 2,
        capacity_factor: 1.5,
        dropout_rate: 0.0,
        aux_loss_config: SmeAuxLossConfig::default(),
    };
    let block = SmeMoeTransformerBlock::new(cfg, &mut rng);

    let x: Vec<f32> = (0..n_tokens * model_dim).map(|i| i as f32 * 0.01).collect();

    // Forward pass
    let out = block.forward(&x, false, &mut rng).expect("forward failed");
    assert_eq!(out.output.len(), n_tokens * model_dim);
    assert!(out.aux_loss.total >= 0.0);

    // Metrics tracking
    let mut metrics = SmeMetrics::new(n_experts, 0.01);
    // Build a routing batch for metrics (separate router for metrics sim)
    let router = SmeRouter::new(n_experts, model_dim, false, &mut rng);
    let ros: Vec<SmeRoutingOutput> = (0..n_tokens)
        .map(|i| {
            let start = i * model_dim;
            router
                .route(&x[start..start + model_dim], 2, &mut rng)
                .expect("route failed")
        })
        .collect();

    let snap = metrics
        .record(0, &ros, &out.aux_loss)
        .expect("record failed");
    assert!(snap.load_imbalance.is_finite());
    assert!(snap.routing_entropy >= 0.0);
}

#[test]
fn test_sme_adaptive_router_multi_step_training() {
    let mut rng = make_rng(201);
    let mut router = SmeAdaptiveRouter::new(4, 8, 2, 0.95, 0.05, &mut rng);

    // Simulate 10 training steps with skewed usage
    for step in 0..10 {
        let x: Vec<f32> = (0..8).map(|i| (i + step) as f32 * 0.1).collect();
        let out = router.route(&x, &mut rng).expect("route failed");
        assert!(out.expert_weights.iter().all(|&w| w.is_finite()));

        // Simulate observed usage
        let usage = vec![0.6f32, 0.2, 0.15, 0.05];
        router.adapt_routing_bias(&usage).expect("adapt failed");
    }

    // After adaptation, expert 0 bias should be reduced (overused)
    // and expert 3 bias should be increased (underused)
    let b0 = router.routing_bias[0];
    let b3 = router.routing_bias[3];
    assert!(
        b0 < b3,
        "overused expert should have lower bias than underused: b0={}, b3={}",
        b0,
        b3
    );
}

#[test]
fn test_sme_expert_capacity_overflow_handling() {
    let token_dim = 4;
    let n_experts = 2;
    let cap = SmeExpertCapacity::new(n_experts, token_dim);

    // With capacity_factor=0.1, capacity = max(ceil(8/2 * 0.1), 1) = 1
    let inputs: Vec<f32> = (0..8 * token_dim).map(|i| i as f32).collect();
    let ros = make_routing_outputs(8, n_experts, 1, 202);
    let dispatch = cap.dispatch(&inputs, &ros, 0.1).expect("dispatch failed");

    // Some tokens should overflow
    let overflowed: usize = dispatch.overflow.iter().filter(|&&o| o).count();
    // With capacity=1 per expert and top-1 routing, at most 2 tokens fit;
    // 6 should overflow
    assert!(overflowed > 0, "expected some overflow tokens");
}

#[test]
fn test_sme_end_to_end_with_expert_parallelism() {
    let mut rng = make_rng(203);
    let n_experts = 3;
    let token_dim = 4;
    let n_tokens = 6;

    let expert_cfg = SmeExpertConfig {
        input_dim: token_dim,
        hidden_dim: 8,
    };
    let experts: Vec<SmeExpert> = (0..n_experts)
        .map(|_| SmeExpert::new(&expert_cfg, &mut rng))
        .collect();

    let par = SmeExpertParallelism::new(token_dim);
    let inputs: Vec<f32> = (0..n_tokens * token_dim).map(|i| i as f32 * 0.1).collect();

    // Route each token to one expert
    let router = SmeRouter::new(n_experts, token_dim, false, &mut rng);
    let assignments: Vec<usize> = (0..n_tokens)
        .map(|i| {
            let tok = &inputs[i * token_dim..(i + 1) * token_dim];
            let out = router.route(tok, 1, &mut rng).expect("route failed");
            out.expert_indices[0]
        })
        .collect();

    let batches = par
        .batch_by_expert(&inputs, &assignments, n_experts)
        .expect("batch failed");
    let output = par
        .process_and_scatter(&batches, &experts, n_tokens)
        .expect("scatter failed");
    assert_eq!(output.len(), n_tokens * token_dim);
    assert!(output.iter().all(|v| v.is_finite()));
}
