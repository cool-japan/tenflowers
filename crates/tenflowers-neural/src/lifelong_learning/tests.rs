//! Tests for lifelong_learning module.

use super::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

// ── GEM ───────────────────────────────────────────────────────────────────────

#[test]
fn test_episode_memory_add() {
    let mut mem = EpisodeMemory::new(10);
    mem.add_task_memory(
        0,
        vec![
            (vec![1.0, 2.0], vec![0.1, 0.2]),
            (vec![3.0, 4.0], vec![0.3, 0.4]),
        ],
    );
    assert_eq!(mem.get_task_memories(0).len(), 2);
    assert_eq!(mem.get_task_memories(1).len(), 0);
}

#[test]
fn test_episode_memory_ring_overflow() {
    let mut mem = EpisodeMemory::new(3);
    for i in 0..10 {
        let v = vec![i as f64; 2];
        mem.add_task_memory(0, vec![(v.clone(), v)]);
    }
    assert_eq!(mem.get_task_memories(0).len(), 3);
}

#[test]
fn test_gem_projection_nonneg() {
    let constraint = GemConstraint::default();
    let memories = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
    let projected = constraint
        .project_gradient(&[-2.0, 0.5, 1.0], &memories, 1e-7)
        .expect("gradient projection should succeed");
    for m in &memories {
        let dot: f64 = projected.iter().zip(m.iter()).map(|(a, b)| a * b).sum();
        assert!(dot >= -1e-5, "inner product should be >= 0, got {}", dot);
    }
}

#[test]
fn test_gem_projection_identity_when_compatible() {
    let constraint = GemConstraint::default();
    let memories = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let grad = vec![1.0, 1.0];
    let projected = constraint
        .project_gradient(&grad, &memories, 1e-7)
        .expect("gradient projection should succeed");
    assert!((projected[0] - 1.0).abs() < 1e-6);
    assert!((projected[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_gem_projection_preserves_dir() {
    let constraint = GemConstraint::default();
    let grad = vec![3.0, -1.0, 2.0];
    assert_eq!(
        constraint
            .project_gradient(&grad, &[], 1e-7)
            .expect("gradient projection should succeed"),
        grad
    );
}

#[test]
fn test_agem_reference_gradient() {
    let agem = AgemOptimizer::new(4);
    let mut rng = make_rng(42);
    let mems: Vec<(Vec<f64>, Vec<f64>)> = (0..8)
        .map(|i| (vec![i as f64; 4], vec![i as f64 * 0.1; 4]))
        .collect();
    let ref_grad = agem
        .compute_reference(&mems, &mut rng)
        .expect("A-GEM reference computation should succeed");
    assert_eq!(ref_grad.len(), 4);
}

#[test]
fn test_agem_project_no_violation() {
    let agem = AgemOptimizer::new(2);
    let ref_grad = vec![1.0, 0.0, 0.0];
    let grad_ok = vec![0.5, 1.0, -1.0];
    assert_eq!(agem.project(&grad_ok, &ref_grad), grad_ok);
}

#[test]
fn test_agem_project_violation_corrected() {
    let agem = AgemOptimizer::new(2);
    let ref_grad = vec![1.0, 0.0, 0.0];
    let proj = agem.project(&[-1.0, 0.5, 0.5], &ref_grad);
    let dot: f64 = proj.iter().zip(ref_grad.iter()).map(|(a, b)| a * b).sum();
    assert!(dot >= -1e-9, "projected dot should be >= 0, got {}", dot);
}

#[test]
fn test_task_gradient_store() {
    let mut store = TaskGradientStore::new(100);
    store.add_gradient(0, vec![1.0, 2.0, 3.0]);
    store.add_gradient(0, vec![4.0, 5.0, 6.0]);
    store.add_gradient(1, vec![7.0, 8.0, 9.0]);
    assert_eq!(store.total_stored(), 3);
    let sampled = store.sample_past(5, &mut make_rng(0));
    assert_eq!(sampled.len(), 5);
}

#[test]
fn test_gem_trainer_step() {
    let trainer = GemTrainer::new(0.01);
    let mut mem = EpisodeMemory::new(5);
    mem.add_task_memory(0, vec![(vec![1.0], vec![0.5, 0.3])]);
    let mut params = vec![0.5, 0.5];
    let projected = trainer
        .step(&[0.5, -1.0], &mut params, &mem)
        .expect("GEM trainer step should succeed");
    assert_eq!(projected.len(), 2);
}

// ── Replay ────────────────────────────────────────────────────────────────────

#[test]
fn test_experience_replay_balanced() {
    let mut er = ExperienceReplay::new(100);
    let mut rng = make_rng(1);
    for i in 0..20 {
        er.add(vec![i as f64], vec![i as f64 * 2.0], i % 4);
    }
    assert_eq!(er.sample_balanced(3, &mut rng).len(), 12);
}

#[test]
fn test_experience_replay_single_task() {
    let mut er = ExperienceReplay::new(0);
    for i in 0..5 {
        er.add(vec![i as f64], vec![0.0], 0);
    }
    assert_eq!(er.total_examples(), 5);
}

#[test]
fn test_dark_experience_replay_loss() {
    let loss = DarkExperienceReplay::der_loss(&[0.5, 0.3, 0.2], &[0.6, 0.3, 0.1], 1.0);
    assert!(loss > 0.0);
}

#[test]
fn test_der_loss_zero_when_equal() {
    let out = vec![0.5, 0.5];
    assert!(DarkExperienceReplay::der_loss(&out, &out, 1.0) < 1e-10);
}

#[test]
fn test_der_loss_positive() {
    let loss = DarkExperienceReplay::der_loss(&[0.1, 0.9], &[0.9, 0.1], 2.0);
    assert!(loss > 0.5, "expected substantial DER loss, got {}", loss);
}

#[test]
fn test_generative_replay_count() {
    let mut gr = GenerativeReplay::new();
    gr.register_model(0, vec![0.0, 0.0], vec![1.0, 1.0]);
    let samples = gr
        .generate_samples(0, 10, &mut make_rng(7))
        .expect("generative replay should succeed");
    assert_eq!(samples.len(), 10);
    for s in &samples {
        assert_eq!(s.len(), 2);
    }
}

#[test]
fn test_generative_replay_missing_task() {
    assert!(GenerativeReplay::new()
        .generate_samples(99, 5, &mut make_rng(0))
        .is_err());
}

#[test]
fn test_reservoir_update() {
    let mut buf: ReplayBuffer<i32> =
        ReplayBuffer::new(10).expect("replay buffer creation should succeed");
    let mut rng = make_rng(42);
    for i in 0..100 {
        buf.reservoir_update(i, &mut rng);
    }
    assert!(buf.len() <= 10);
    assert_eq!(buf.total_seen(), 100);
}

#[test]
fn test_reservoir_fills_before_reservoir_sampling() {
    let mut buf: ReplayBuffer<u8> =
        ReplayBuffer::new(5).expect("replay buffer creation should succeed");
    let mut rng = make_rng(0);
    for i in 0_u8..5 {
        buf.reservoir_update(i, &mut rng);
    }
    assert_eq!(buf.len(), 5);
}

#[test]
fn test_memory_aware_gradients() {
    let mag =
        MemoryAwareGradients::compute_mag_grad(&[1.0, 1.0], &[2.0, 3.0], &[1.0, 1.0], &[0.5, 2.0]);
    assert!((mag[0] - 0.5).abs() < 1e-9);
    assert!((mag[1] - 4.0).abs() < 1e-9);
}

// ── Architecture ──────────────────────────────────────────────────────────────

#[test]
fn test_progressive_growth_expansion() {
    let mut pg = ProgressiveGrowth::new(10, true);
    pg.expand(0, vec![64, 32]);
    pg.expand(1, vec![64, 32]);
    assert_eq!(pg.num_columns(), 2);
}

#[test]
fn test_progressive_growth_lateral_width() {
    let mut pg = ProgressiveGrowth::new(4, true);
    pg.expand(0, vec![8, 4]);
    pg.expand(1, vec![8, 4]);
    assert_eq!(pg.effective_input_width(0), 12);
}

#[test]
fn test_packnet_mask_sparsity() {
    let mut masker = PackNetMasker::new(10);
    let weights: Vec<f64> = (0..10).map(|i| (i + 1) as f64).collect();
    masker.prune_for_task(0, &weights, 0.5);
    let sparsity = masker
        .achieved_sparsity(0)
        .expect("sparsity computation should succeed");
    assert!(
        (sparsity - 0.5).abs() < 0.1,
        "sparsity ~0.5, got {}",
        sparsity
    );
}

#[test]
fn test_packnet_apply_mask() {
    let mut masker = PackNetMasker::new(4);
    let weights = vec![1.0, 2.0, 3.0, 4.0];
    masker.prune_for_task(0, &weights, 0.5);
    let masked = masker
        .apply_mask(&weights, 0)
        .expect("mask application should succeed");
    assert!(masked.iter().filter(|&&v| v == 0.0).count() >= 1);
}

#[test]
fn test_hat_mask_cumulative() {
    let mut hat = HatMask::new(0.5, 5.0);
    hat.register_embedding(0, vec![1.0, -2.0, 0.0]);
    hat.register_embedding(1, vec![-1.0, 3.0, 0.5]);
    let mask = hat.compute_cumulative_mask(0.5);
    assert_eq!(mask.len(), 3);
    assert_eq!(mask[0], 1.0); // sigmoid(5.0) > 0.5
    assert_eq!(mask[1], 1.0); // sigmoid(15.0) ≈ 1 > 0.5
}

#[test]
fn test_dynamic_expansion() {
    let mut layer = DynamicExpansionLayer::new(64, 1.5, 512);
    assert!(layer.maybe_expand(0.9, 0.8));
    assert!(layer.hidden_size > 64);
}

#[test]
fn test_modular_network_routing() {
    let mn = ModularNetwork::new(4, 8, 2);
    assert_eq!(mn.route(&[1.0; 8], 0).len(), 8);
    assert_eq!(mn.route(&[1.0; 8], 1).len(), 8);
}

#[test]
fn test_modular_network_registered_route() {
    let mut mn = ModularNetwork::new(4, 4, 2);
    mn.register_route(0, vec![0, 2]);
    assert_eq!(mn.route(&[1.0; 4], 0).len(), 4);
}

// ── Regularization ────────────────────────────────────────────────────────────

#[test]
fn test_si_loss_positive() {
    let mut si = SynapticIntelligence::new(4, 1e-3);
    si.omega = vec![1.0; 4];
    si.anchor = vec![0.0; 4];
    assert!(si.si_loss(&[1.0; 4], 1.0) > 0.0);
}

#[test]
fn test_si_update_omega() {
    let mut si = SynapticIntelligence::new(3, 1e-3);
    si.update_omega(&[0.1, -0.2, 0.3], &[0.01, 0.02, -0.01]);
    assert_eq!(si.omega.iter().sum::<f64>(), 0.0); // omega only updated on consolidate
}

#[test]
fn test_si_consolidate() {
    let mut si = SynapticIntelligence::new(2, 1e-3);
    // grad=[-1,-1], delta=[+0.5,+0.5]: w_k = -((-1)*(0.5)) = 0.5 > 0 → importance grows
    si.update_omega(&[-1.0, -1.0], &[0.5, 0.5]);
    si.consolidate(&[0.5, 0.5]);
    assert!(
        si.omega.iter().sum::<f64>() > 0.0,
        "omega should accumulate importance"
    );
}

#[test]
fn test_lwf_loss_positive() {
    let loss = LearningWithoutForgetting::lwf_loss(&[2.0, 0.5, -1.0], &[0.5, 2.0, -1.0], 2.0, 1.0);
    assert!(loss > 0.0);
}

#[test]
fn test_lwf_loss_zero_when_equal() {
    let logits = vec![1.0, 2.0, 0.0];
    let loss = LearningWithoutForgetting::lwf_loss(&logits, &logits, 2.0, 1.0);
    assert!(
        loss < 1e-8,
        "LwF KL loss should be ~0 when logits match, got {}",
        loss
    );
}

#[test]
fn test_lwf_temperature_effect() {
    let new_logits = vec![2.0, 0.0, -2.0];
    let old_logits = vec![0.0, 2.0, -2.0];
    let l1 = LearningWithoutForgetting::lwf_loss(&new_logits, &old_logits, 1.0, 1.0);
    let l4 = LearningWithoutForgetting::lwf_loss(&new_logits, &old_logits, 4.0, 1.0);
    assert_ne!(
        (l1 * 1e6).round(),
        (l4 * 1e6).round(),
        "loss should differ at different temperatures"
    );
}

#[test]
fn test_owm_orthogonal() {
    let mut owm = OWM::new(1e-3);
    owm.add_activation(vec![1.0, 0.0, 0.0]);
    let proj = owm.project_gradient(&[1.0, 0.5, 0.5]);
    assert!(
        proj[0].abs() < 0.2,
        "OWM should suppress past-subspace component"
    );
}

#[test]
fn test_owm_projection_matrix_shape() {
    let owm = OWM::new(1e-2);
    let p = owm.compute_projection_matrix(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
    assert_eq!(p.len(), 2);
    assert_eq!(p[0].len(), 2);
}

#[test]
fn test_functional_reg() {
    let mut fr = FunctionalRegularization::new(1.0);
    fr.register_anchor(vec![1.0, 2.0], vec![0.5, 0.5]);
    assert!(fr.functional_loss(&[vec![0.6, 0.4]]) > 0.0);
}

#[test]
fn test_functional_reg_zero_when_matched() {
    let mut fr = FunctionalRegularization::new(1.0);
    fr.register_anchor(vec![1.0], vec![0.7, 0.3]);
    assert!(fr.functional_loss(&[vec![0.7, 0.3]]) < 1e-10);
}

// ── Metrics ───────────────────────────────────────────────────────────────────

#[test]
fn test_cl_metrics_bwt_no_forgetting() {
    let acc = vec![
        vec![1.0, 0.0, 0.0],
        vec![1.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0],
    ];
    let metrics = ContinualLearningMetrics::compute(&acc);
    assert!((metrics.bwt).abs() < 1e-10);
}

#[test]
fn test_cl_metrics_bwt_with_forgetting() {
    let acc = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.8, 1.0, 0.0],
        vec![0.6, 0.7, 1.0],
    ];
    assert!(ContinualLearningMetrics::compute(&acc).bwt < 0.0);
}

#[test]
fn test_cl_metrics_aia() {
    let acc = vec![vec![0.8, 0.0], vec![0.7, 0.9]];
    assert!((ContinualLearningMetrics::compute(&acc).aia - 0.8).abs() < 1e-9);
}

#[test]
fn test_cl_metrics_final_avg() {
    let acc = vec![vec![0.9, 0.0], vec![0.8, 0.85]];
    assert!((ContinualLearningMetrics::compute(&acc).final_avg - 0.825).abs() < 1e-9);
}

#[test]
fn test_forgetting_measure() {
    let acc = vec![vec![1.0, 0.0], vec![0.7, 1.0]];
    assert!((ForgettingMeasure::compute(&acc) - 0.3).abs() < 1e-9);
}

#[test]
fn test_forgetting_measure_none_when_single_task() {
    assert_eq!(ForgettingMeasure::compute(&[vec![0.9]]), 0.0);
}

#[test]
fn test_plasticity_stability() {
    let metrics = ClMetrics {
        bwt: -0.1,
        fwt: 0.05,
        aia: 0.75,
        final_avg: 0.8,
    };
    let (stability, plasticity) = PlasticityStabilityTradeoff::evaluate(&metrics);
    assert!((stability - 0.9).abs() < 1e-9);
    assert!((plasticity - 0.05).abs() < 1e-9);
}

#[test]
fn test_plasticity_stability_from_matrix() {
    let acc = vec![vec![1.0, 0.0], vec![0.9, 0.95]];
    let (s, p) = PlasticityStabilityTradeoff::evaluate_from_matrix(&acc);
    assert!((0.0..=1.0).contains(&s));
    assert!((-1.0..=1.0).contains(&p));
}

#[test]
fn test_benchmark_suite() {
    let m = BenchmarkSuite::run_permuted_mnist(123);
    assert_eq!(m.len(), 10);
    for i in 0..10 {
        assert!(m[i][i] >= 0.5);
    }
    let c = BenchmarkSuite::run_split_cifar(456);
    assert_eq!(c.len(), 5);
    assert!(ContinualLearningMetrics::compute(&c).final_avg > 0.0);
}

#[test]
fn test_benchmark_suite_metrics_bwt() {
    let m = BenchmarkSuite::run_permuted_mnist(0);
    assert!(ContinualLearningMetrics::compute(&m).bwt <= 0.0);
}

#[test]
fn test_memo_rep_distillation_loss() {
    let mut mr = MemoRep::new(2.0);
    mr.store_representation(0, vec![1.0], vec![0.5, 0.5]);
    assert!(mr.repr_distillation_loss(0, &[vec![0.6, 0.4]]) > 0.0);
}

#[test]
fn test_dark_experience_replay_sample() {
    let mut der = DarkExperienceReplay::new(10);
    der.add(vec![1.0, 2.0], vec![0.9, 0.1], 0);
    assert!(der.sample(0, &mut make_rng(0)).is_some());
}

// ── LllGemModel tests ─────────────────────────────────────────────────────────

#[test]
fn test_lll_gem_model_forward_shape() {
    let model = LllGemModel::new(4, 8, 3, 50, 0.01, 42);
    let out = model
        .forward(&[1.0, 0.0, -1.0, 0.5])
        .expect("forward pass should succeed");
    assert_eq!(out.len(), 3);
}

#[test]
fn test_lll_gem_model_cross_entropy_valid() {
    let model = LllGemModel::new(4, 8, 3, 50, 0.01, 42);
    let logits = vec![1.0, 2.0, 0.0];
    let loss = model
        .cross_entropy_loss(&logits, 1)
        .expect("cross entropy loss should succeed");
    assert!(loss >= 0.0);
}

#[test]
fn test_lll_gem_model_cross_entropy_label_out_of_range() {
    let model = LllGemModel::new(4, 8, 3, 50, 0.01, 42);
    assert!(model.cross_entropy_loss(&[1.0, 2.0, 0.0], 5).is_err());
}

#[test]
fn test_lll_gem_model_add_memory() {
    let mut model = LllGemModel::new(4, 8, 3, 50, 0.01, 42);
    model.add_memory(0, vec![(vec![1.0; 4], vec![0.1; 4])]);
    assert_eq!(model.memory.num_tasks(), 1);
}

#[test]
fn test_lll_gem_model_grad_fd_shape() {
    let model = LllGemModel::new(4, 8, 3, 50, 0.01, 42);
    let grad = model
        .compute_grad_fd(&[1.0, 0.0, -1.0, 0.5], 0, 1e-4)
        .expect("finite-difference gradient should succeed");
    assert!(!grad.is_empty());
}

#[test]
fn test_lll_gem_model_gem_update_no_memory() {
    let mut model = LllGemModel::new(4, 8, 3, 50, 0.01, 42);
    let projected = model
        .gem_update(&[1.0, 0.0, -1.0, 0.5], 0)
        .expect("GEM update should succeed");
    assert!(!projected.is_empty());
}

#[test]
fn test_lll_gem_model_gem_update_with_memory() {
    let mut model = LllGemModel::new(4, 8, 3, 50, 0.01, 42);
    // Add task 0 memory
    let grad_dummy = vec![0.01; 4 * 8 + 8 + 8 * 3 + 3];
    model.add_memory(0, vec![(vec![1.0; 4], grad_dummy)]);
    let projected = model
        .gem_update(&[0.5, -0.5, 0.5, -0.5], 2)
        .expect("GEM update should succeed");
    assert!(!projected.is_empty());
}

// ── LllAGemModel tests ────────────────────────────────────────────────────────

#[test]
fn test_lll_agem_forward_shape() {
    let model = LllAGemModel::new(4, 8, 3, 4, 0.01, 42);
    let out = model
        .forward(&[1.0, 0.0, -1.0, 0.5])
        .expect("forward pass should succeed");
    assert_eq!(out.len(), 3);
}

#[test]
fn test_lll_agem_update_no_memory() {
    let mut model = LllAGemModel::new(4, 8, 3, 4, 0.01, 42);
    let total_params = 4 * 8 + 8 + 8 * 3 + 3;
    let grad = vec![0.01f64; total_params];
    let out = model.agem_update(&grad, &mut make_rng(0));
    assert_eq!(out.len(), total_params);
}

#[test]
fn test_lll_agem_store_episode() {
    let mut model = LllAGemModel::new(4, 8, 3, 4, 0.01, 42);
    model.store_episode(vec![1.0; 4], vec![0.01; 10]);
    assert_eq!(model.episodic_store.len(), 1);
}

#[test]
fn test_lll_agem_update_with_memory_projects() {
    let mut model = LllAGemModel::new(4, 8, 3, 4, 0.01, 42);
    let total_params = 4 * 8 + 8 + 8 * 3 + 3;
    let ref_grad = vec![1.0f64; total_params];
    model.store_episode(vec![1.0; 4], ref_grad.clone());
    // Gradient that violates ref (opposing direction)
    let bad_grad: Vec<f64> = ref_grad.iter().map(|&v| -v * 2.0).collect();
    let projected = model.agem_update(&bad_grad, &mut make_rng(42));
    // After projection, dot product with ref_grad should be >= 0
    let dot: f64 = projected
        .iter()
        .zip(ref_grad.iter())
        .map(|(a, b)| a * b)
        .sum();
    assert!(
        dot >= -1e-6,
        "A-GEM projection should satisfy constraint, got dot={}",
        dot
    );
}

// ── LllER tests ───────────────────────────────────────────────────────────────

#[test]
fn test_lll_er_new_zero_capacity_err() {
    assert!(LllER::new(0).is_err());
}

#[test]
fn test_lll_er_buffer_fills() {
    let mut er = LllER::new(10).expect("ER buffer creation should succeed");
    let mut rng = make_rng(0);
    for i in 0..5 {
        er.update_buffer(
            LllErSample {
                x: vec![i as f64],
                label: i % 3,
                task_id: 0,
            },
            &mut rng,
        );
    }
    assert_eq!(er.buffer_size(), 5);
}

#[test]
fn test_lll_er_reservoir_bounded() {
    let mut er = LllER::new(10).expect("ER buffer creation should succeed");
    let mut rng = make_rng(42);
    for i in 0..100 {
        er.update_buffer(
            LllErSample {
                x: vec![i as f64],
                label: i % 5,
                task_id: i / 20,
            },
            &mut rng,
        );
    }
    assert!(er.buffer_size() <= 10);
    assert_eq!(er.total_seen(), 100);
}

#[test]
fn test_lll_er_sample_replay_empty_err() {
    let er = LllER::new(10).expect("ER buffer creation should succeed");
    assert!(er.sample_replay(4, &mut make_rng(0)).is_err());
}

#[test]
fn test_lll_er_sample_replay_correct_count() {
    let mut er = LllER::new(20).expect("ER buffer creation should succeed");
    let mut rng = make_rng(1);
    for i in 0..10 {
        er.update_buffer(
            LllErSample {
                x: vec![i as f64],
                label: 0,
                task_id: 0,
            },
            &mut rng,
        );
    }
    let samples = er
        .sample_replay(5, &mut rng)
        .expect("replay sampling should succeed");
    assert_eq!(samples.len(), 5);
}

#[test]
fn test_lll_er_train_step_alpha_0() {
    let mixed = LllER::train_step(1.0, 0.5, 0.0);
    assert!((mixed - 0.5).abs() < 1e-9);
}

#[test]
fn test_lll_er_train_step_alpha_1() {
    let mixed = LllER::train_step(1.0, 0.5, 1.0);
    assert!((mixed - 1.0).abs() < 1e-9);
}

#[test]
fn test_lll_er_train_step_mixed() {
    let mixed = LllER::train_step(0.8, 0.4, 0.5);
    assert!((mixed - 0.6).abs() < 1e-9);
}

#[test]
fn test_lll_er_is_full() {
    let mut er = LllER::new(3).expect("ER buffer creation should succeed");
    let mut rng = make_rng(0);
    for i in 0..3 {
        er.update_buffer(
            LllErSample {
                x: vec![i as f64],
                label: 0,
                task_id: 0,
            },
            &mut rng,
        );
    }
    assert!(er.is_full());
}

// ── LllDer tests ──────────────────────────────────────────────────────────────

#[test]
fn test_lll_der_new_zero_capacity_err() {
    assert!(LllDer::new(0, 1.0, 1.0).is_err());
}

#[test]
fn test_lll_der_add_entry() {
    let mut der = LllDer::new(10, 1.0, 0.5).expect("DER buffer creation should succeed");
    let mut rng = make_rng(0);
    der.add_entry(
        LllDerEntry {
            x: vec![1.0, 2.0],
            stored_logits: vec![0.8, 0.2],
            label: 0,
            task_id: 0,
        },
        &mut rng,
    );
    assert_eq!(der.buffer_size(), 1);
}

#[test]
fn test_lll_der_der_loss_zero_when_equal() {
    let der = LllDer::new(10, 1.0, 0.5).expect("DER buffer creation should succeed");
    let entry = LllDerEntry {
        x: vec![1.0],
        stored_logits: vec![0.7, 0.3],
        label: 0,
        task_id: 0,
    };
    let loss = der.der_loss(&[vec![0.7, 0.3]], &[&entry]);
    assert!(loss < 1e-10, "DER loss should be 0 when logits match");
}

#[test]
fn test_lll_der_der_loss_positive() {
    let der = LllDer::new(10, 1.0, 0.5).expect("DER buffer creation should succeed");
    let entry = LllDerEntry {
        x: vec![1.0],
        stored_logits: vec![0.9, 0.1],
        label: 0,
        task_id: 0,
    };
    let loss = der.der_loss(&[vec![0.1, 0.9]], &[&entry]);
    assert!(loss > 0.0, "DER loss should be positive when logits differ");
}

#[test]
fn test_lll_der_der_plus_plus_loss() {
    let der = LllDer::new(10, 1.0, 0.5).expect("DER buffer creation should succeed");
    let entry = LllDerEntry {
        x: vec![1.0],
        stored_logits: vec![0.9, 0.1],
        label: 0,
        task_id: 0,
    };
    let loss = der
        .der_plus_plus_loss(&[vec![0.1, 0.9]], &[&entry])
        .expect("DER++ loss computation should succeed");
    assert!(loss > 0.0);
}

#[test]
fn test_lll_der_plus_plus_bad_label_err() {
    let der = LllDer::new(10, 1.0, 0.5).expect("DER buffer creation should succeed");
    // label=5 but logits has only 2 classes
    let entry = LllDerEntry {
        x: vec![1.0],
        stored_logits: vec![0.5, 0.5],
        label: 5,
        task_id: 0,
    };
    assert!(der
        .der_plus_plus_loss(&[vec![0.5, 0.5]], &[&entry])
        .is_err());
}

#[test]
fn test_lll_der_sample_entries_empty_err() {
    let der = LllDer::new(10, 1.0, 0.5).expect("DER buffer creation should succeed");
    assert!(der.sample_entries(2, &mut make_rng(0)).is_err());
}

#[test]
fn test_lll_der_reservoir_bounded() {
    let mut der = LllDer::new(5, 1.0, 0.5).expect("DER buffer creation should succeed");
    let mut rng = make_rng(7);
    for i in 0..20 {
        der.add_entry(
            LllDerEntry {
                x: vec![i as f64],
                stored_logits: vec![0.5, 0.5],
                label: 0,
                task_id: 0,
            },
            &mut rng,
        );
    }
    assert!(der.buffer_size() <= 5);
}

// ── LllCoPE tests ─────────────────────────────────────────────────────────────

#[test]
fn test_lll_cope_update_prototype() {
    let mut cope = LllCoPE::new(0.9, 1.0);
    cope.update_prototype(&[1.0, 0.0, 0.0], 0);
    let p = cope
        .get_prototype(0)
        .expect("prototype should exist for class 0");
    assert!(p[0] > 0.0);
    assert_eq!(cope.num_classes(), 1);
}

#[test]
fn test_lll_cope_classify_single_class() {
    let mut cope = LllCoPE::new(0.0, 1.0);
    cope.update_prototype(&[1.0, 0.0], 0);
    assert_eq!(
        cope.classify(&[1.0, 0.0])
            .expect("classification should succeed"),
        0
    );
}

#[test]
fn test_lll_cope_classify_nearest() {
    let mut cope = LllCoPE::new(0.0, 1.0);
    cope.update_prototype(&[1.0, 0.0], 0);
    cope.update_prototype(&[0.0, 1.0], 1);
    // [0.9, 0.1] is closer to class 0
    assert_eq!(
        cope.classify(&[0.9, 0.1])
            .expect("classification should succeed"),
        0
    );
    // [0.1, 0.9] is closer to class 1
    assert_eq!(
        cope.classify(&[0.1, 0.9])
            .expect("classification should succeed"),
        1
    );
}

#[test]
fn test_lll_cope_classify_empty_err() {
    let cope = LllCoPE::new(0.9, 1.0);
    assert!(cope.classify(&[1.0, 0.0]).is_err());
}

#[test]
fn test_lll_cope_ema_momentum() {
    let mut cope = LllCoPE::new(0.5, 1.0); // EMA momentum = 0.5
                                           // First update: proto = (1-0.5)*features = 0.5*[2,0]
    cope.update_prototype(&[2.0, 0.0], 0);
    let p = cope
        .get_prototype(0)
        .expect("prototype should exist for class 0")
        .clone();
    // Second update: proto = 0.5*p + 0.5*[0, 2]
    cope.update_prototype(&[0.0, 2.0], 0);
    let p2 = cope
        .get_prototype(0)
        .expect("prototype should exist for class 0");
    // Should interpolate between first proto and new features
    assert!(p2[0] < p[0], "EMA should decay first dimension");
    assert!(p2[1] > 0.0, "EMA should incorporate new features");
}

#[test]
fn test_lll_cope_prototype_drift_no_snapshot() {
    let mut cope = LllCoPE::new(0.9, 1.0);
    cope.update_prototype(&[1.0, 0.0], 0);
    assert!((cope.prototype_drift_loss() - 0.0).abs() < 1e-10);
}

#[test]
fn test_lll_cope_prototype_drift_after_snapshot() {
    let mut cope = LllCoPE::new(0.0, 1.0); // EMA=0 means full replace
    cope.update_prototype(&[1.0, 0.0], 0);
    cope.snapshot_prototypes();
    cope.update_prototype(&[0.0, 1.0], 0); // full replace → prototype moves
    let drift = cope.prototype_drift_loss();
    assert!(drift > 0.0, "Prototype should drift after update");
}

#[test]
fn test_lll_cope_update_prototypes_batch() {
    let mut cope = LllCoPE::new(0.9, 1.0);
    let feats = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]];
    let labels = vec![0, 1, 0];
    cope.update_prototypes(&feats, &labels)
        .expect("batch prototype update should succeed");
    assert_eq!(cope.num_classes(), 2);
}

#[test]
fn test_lll_cope_batch_mismatched_lengths_err() {
    let mut cope = LllCoPE::new(0.9, 1.0);
    assert!(cope.update_prototypes(&[vec![1.0]], &[0, 1]).is_err());
}

// ── LllHAT tests ──────────────────────────────────────────────────────────────

#[test]
fn test_lll_hat_init_task() {
    let mut hat = LllHAT::new(8, 10.0, 0.01);
    hat.init_task(0, &mut make_rng(0));
    assert_eq!(hat.num_tasks(), 1);
}

#[test]
fn test_lll_hat_soft_mask_range() {
    let mut hat = LllHAT::new(8, 5.0, 0.01);
    hat.init_task(0, &mut make_rng(42));
    let mask = hat
        .soft_mask(0)
        .expect("soft mask computation should succeed");
    assert_eq!(mask.len(), 8);
    for &v in &mask {
        assert!((0.0..=1.0).contains(&v), "soft mask must be in [0,1]");
    }
}

#[test]
fn test_lll_hat_hard_mask_binary() {
    let mut hat = LllHAT::new(8, 50.0, 0.01); // very high temperature
    hat.init_task(0, &mut make_rng(1));
    let mask = hat
        .hard_mask(0)
        .expect("hard mask computation should succeed");
    for &v in &mask {
        assert!(v == 0.0 || v == 1.0, "hard mask must be binary");
    }
}

#[test]
fn test_lll_hat_forward_soft() {
    let mut hat = LllHAT::new(4, 5.0, 0.01);
    hat.init_task(0, &mut make_rng(0));
    let out = hat
        .forward(&[1.0, 2.0, 3.0, 4.0], 0, false)
        .expect("HAT forward should succeed");
    assert_eq!(out.len(), 4);
}

#[test]
fn test_lll_hat_forward_hard() {
    let mut hat = LllHAT::new(4, 20.0, 0.01);
    hat.init_task(0, &mut make_rng(0));
    let out = hat
        .forward(&[1.0; 4], 0, true)
        .expect("HAT hard forward should succeed");
    // Hard mask applied: values are either 0 or original
    for &v in &out {
        assert!(v == 0.0 || v == 1.0);
    }
}

#[test]
fn test_lll_hat_unknown_task_err() {
    let hat = LllHAT::new(8, 5.0, 0.01);
    assert!(hat.soft_mask(99).is_err());
}

#[test]
fn test_lll_hat_sparsity_loss_nonneg() {
    let mut hat = LllHAT::new(8, 5.0, 0.1);
    hat.init_task(0, &mut make_rng(0));
    let loss = hat
        .sparsity_loss(0)
        .expect("sparsity loss computation should succeed");
    assert!(loss >= 0.0);
}

#[test]
fn test_lll_hat_consolidate_task() {
    let mut hat = LllHAT::new(8, 100.0, 0.01); // very high temp → hard mask ~ all 1s
    hat.init_task(0, &mut make_rng(0));
    hat.consolidate_task(0)
        .expect("task consolidation should succeed");
    let frozen = hat.frozen_fraction();
    assert!(
        frozen > 0.0,
        "some units should be frozen after consolidation"
    );
}

#[test]
fn test_lll_hat_gradient_mask_sum() {
    let mut hat = LllHAT::new(8, 100.0, 0.01);
    hat.init_task(0, &mut make_rng(0));
    hat.consolidate_task(0)
        .expect("task consolidation should succeed");
    let gmask = hat.gradient_mask();
    // cumulative + gradient mask should sum to layer_size (each unit is 0 or 1)
    let cum_sum: f64 = hat.cumulative_mask.iter().sum();
    let free_sum: f64 = gmask.iter().sum();
    assert!((cum_sum + free_sum - 8.0).abs() < 1e-9);
}

#[test]
fn test_lll_hat_mask_gradient_zeroes_frozen() {
    let mut hat = LllHAT::new(4, 1000.0, 0.01); // very high temp
                                                // Manually set cumulative mask all 1
    hat.cumulative_mask = vec![1.0; 4];
    let masked = hat.mask_gradient(&[1.0; 4]);
    assert!(
        masked.iter().all(|&v| v.abs() < 1e-10),
        "all gradients should be zeroed by mask"
    );
}

// ── LllTaskOracle tests ───────────────────────────────────────────────────────

#[test]
fn test_lll_task_oracle_new_small_window_err() {
    assert!(LllTaskOracle::new(2, 2.0, 0.1).is_err());
}

#[test]
fn test_lll_task_oracle_no_change_stable() {
    let mut oracle = LllTaskOracle::new(10, 3.0, 0.1).expect("task oracle creation should succeed");
    // All constant losses: no change should be detected
    let mut detected = false;
    for _ in 0..20 {
        if oracle.record_loss(0.5) {
            detected = true;
        }
    }
    assert!(!detected, "no task change in stable loss stream");
}

#[test]
fn test_lll_task_oracle_detects_change() {
    let mut oracle = LllTaskOracle::new(10, 1.5, 0.1).expect("task oracle creation should succeed");
    // First fill with low losses, then inject high losses
    for _ in 0..8 {
        oracle.record_loss(0.1);
    }
    let mut detected = false;
    for _ in 0..8 {
        if oracle.record_loss(2.0) {
            detected = true;
        }
    }
    assert!(detected, "task change should be detected when loss spikes");
}

#[test]
fn test_lll_task_oracle_record_increments_step() {
    let mut oracle = LllTaskOracle::new(4, 3.0, 0.1).expect("task oracle creation should succeed");
    for _ in 0..5 {
        oracle.record_loss(0.3);
    }
    assert_eq!(oracle.step, 5);
}

#[test]
fn test_lll_task_oracle_detect_from_slice_no_change() {
    let mut oracle = LllTaskOracle::new(8, 3.0, 0.1).expect("task oracle creation should succeed");
    let losses = vec![0.5f64; 8];
    let changed = oracle.detect_from_slice(&losses);
    assert!(!changed);
}

#[test]
fn test_lll_task_oracle_ema_updates() {
    let mut oracle = LllTaskOracle::new(8, 3.0, 0.5).expect("task oracle creation should succeed");
    oracle.record_loss(1.0);
    assert!((oracle.ema_loss - 0.5).abs() < 1e-9); // 0.5 * 1.0 + 0.5 * 0.0 = 0.5
}

// ── LllMetrics tests ──────────────────────────────────────────────────────────

#[test]
fn test_lll_metrics_empty_err() {
    assert!(LllMetrics::compute(&[], None).is_err());
}

#[test]
fn test_lll_metrics_empty_row_err() {
    assert!(LllMetrics::compute(&[vec![]], None).is_err());
}

#[test]
fn test_lll_metrics_single_task() {
    let acc = vec![vec![0.9]];
    let report = LllMetrics::compute(&acc, None).expect("metrics computation should succeed");
    assert!((report.average_accuracy - 0.9).abs() < 1e-9);
    assert_eq!(report.num_tasks, 1);
}

#[test]
fn test_lll_metrics_backward_transfer_negative() {
    // Task 0 starts at 1.0 but degrades to 0.7 after task 1
    let acc = vec![vec![1.0, 0.0], vec![0.7, 0.9]];
    let report = LllMetrics::compute(&acc, None).expect("metrics computation should succeed");
    assert!(
        report.backward_transfer < 0.0,
        "BWT should be negative (forgetting)"
    );
}

#[test]
fn test_lll_metrics_forward_transfer() {
    let acc = vec![vec![0.8, 0.3], vec![0.7, 0.9]];
    let report = LllMetrics::compute(&acc, None).expect("metrics computation should succeed");
    assert!((report.forward_transfer - 0.3).abs() < 1e-9);
}

#[test]
fn test_lll_metrics_forgetting_nonneg() {
    let acc = vec![vec![1.0, 0.0], vec![0.6, 1.0]];
    let report = LllMetrics::compute(&acc, None).expect("metrics computation should succeed");
    assert!(report.forgetting >= 0.0);
}

#[test]
fn test_lll_metrics_with_oracle() {
    let acc = vec![vec![0.8, 0.0], vec![0.7, 0.85]];
    let oracle = vec![0.95, 0.92];
    let report = LllMetrics::compute(&acc, Some(&oracle))
        .expect("metrics computation with oracle should succeed");
    assert!(
        report.intransigence > 0.0,
        "there should be intransigence vs oracle"
    );
    assert!(report.plasticity <= 1.0 && report.plasticity >= 0.0);
}

#[test]
fn test_lll_metrics_stability_range() {
    let acc = vec![vec![1.0, 0.0], vec![0.8, 1.0]];
    let report = LllMetrics::compute(&acc, None).expect("metrics computation should succeed");
    assert!(report.stability >= 0.0 && report.stability <= 1.0);
}

#[test]
fn test_lll_metrics_plasticity_stability_curve() {
    let acc = vec![vec![0.9, 0.0], vec![0.8, 0.85]];
    let (stability, plasticity) = LllMetrics::plasticity_stability_curve(&acc);
    assert!((0.0..=1.0).contains(&stability));
    assert!((-1.0..=1.0).contains(&plasticity));
}

#[test]
fn test_lll_metrics_per_task_evolution() {
    let acc = vec![
        vec![0.8, 0.0, 0.0],
        vec![0.75, 0.9, 0.0],
        vec![0.7, 0.85, 0.92],
    ];
    let task0_evo = LllMetrics::per_task_evolution(&acc, 0);
    assert_eq!(task0_evo, vec![0.8, 0.75, 0.7]);
}

#[test]
fn test_lll_metrics_area_under_curve() {
    let values = vec![0.5, 0.7, 0.9];
    let auc = LllMetrics::area_under_curve(&values);
    assert!(
        (auc - 0.7).abs() < 1e-9,
        "AUC of linear 0.5→0.7→0.9 = 0.7, got {}",
        auc
    );
}

#[test]
fn test_lll_metrics_area_single_value() {
    assert!((LllMetrics::area_under_curve(&[0.8]) - 0.8).abs() < 1e-9);
}

#[test]
fn test_lll_metrics_area_empty() {
    assert!((LllMetrics::area_under_curve(&[]) - 0.0).abs() < 1e-9);
}

// ── LllLinearLayer tests ──────────────────────────────────────────────────────

#[test]
fn test_lll_linear_layer_forward_relu_shape() {
    let mut rng = make_rng(0);
    let layer = LllLinearLayer::new(4, 8, &mut rng);
    let out = layer
        .forward_relu(&[1.0, -1.0, 0.5, -0.5])
        .expect("ReLU forward should succeed");
    assert_eq!(out.len(), 8);
    for &v in &out {
        assert!(v >= 0.0, "ReLU output must be nonneg");
    }
}

#[test]
fn test_lll_linear_layer_forward_linear_shape() {
    let mut rng = make_rng(1);
    let layer = LllLinearLayer::new(3, 5, &mut rng);
    let out = layer
        .forward_linear(&[1.0, 0.0, -1.0])
        .expect("linear forward should succeed");
    assert_eq!(out.len(), 5);
}

#[test]
fn test_lll_linear_layer_dim_mismatch_err() {
    let mut rng = make_rng(0);
    let layer = LllLinearLayer::new(4, 8, &mut rng);
    assert!(layer.forward_relu(&[1.0, 2.0]).is_err());
}

#[test]
fn test_lll_linear_layer_params_flat_len() {
    let mut rng = make_rng(0);
    let layer = LllLinearLayer::new(4, 8, &mut rng);
    assert_eq!(layer.params_flat().len(), 4 * 8 + 8);
}

#[test]
fn test_lll_linear_layer_apply_grad_updates() {
    let mut rng = make_rng(42);
    let mut layer = LllLinearLayer::new(2, 2, &mut rng);
    let before = layer.params_flat();
    let grad = vec![0.1f64; before.len()];
    layer.apply_grad_flat(&grad, 0.01);
    let after = layer.params_flat();
    for (b, a) in before.iter().zip(after.iter()) {
        assert!(
            (b - a - 0.001).abs() < 1e-9,
            "SGD update: param should decrease by lr*grad"
        );
    }
}
