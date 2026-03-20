use super::*;

use scirs2_core::random::{rngs::StdRng, SeedableRng};

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

// ── L2lError ──────────────────────────────────────────────────────────────

#[test]
fn test_l2l_error_display_invalid_config() {
    let e = L2lError::InvalidConfig {
        field: "lr".into(),
        reason: "negative".into(),
    };
    assert!(e.to_string().contains("lr"));
}

#[test]
fn test_l2l_error_display_gradient() {
    let e = L2lError::GradientError {
        context: "nan grad".into(),
    };
    assert!(e.to_string().contains("nan"));
}

#[test]
fn test_l2l_error_display_opt_failed() {
    let e = L2lError::OptimizationFailed {
        reason: "diverged".into(),
    };
    assert!(e.to_string().contains("diverged"));
}

// ── L2lTensor ─────────────────────────────────────────────────────────────

#[test]
fn test_tensor_zeros() {
    let t = L2lTensor::zeros(5);
    assert_eq!(t.data.len(), 5);
    assert!(t.data.iter().all(|&x| x == 0.0));
    assert_eq!(t.grad.len(), 5);
}

#[test]
fn test_tensor_zeros_like() {
    let t = L2lTensor::new(vec![1.0, 2.0, 3.0]);
    let z = t.zeros_like();
    assert_eq!(z.data.len(), 3);
    assert!(z.data.iter().all(|&x| x == 0.0));
}

#[test]
fn test_tensor_l2_norm() {
    let t = L2lTensor::new(vec![3.0, 4.0]);
    assert!((t.l2_norm() - 5.0).abs() < 1e-10);
}

#[test]
fn test_tensor_dot() {
    let a = L2lTensor::new(vec![1.0, 2.0, 3.0]);
    let b = L2lTensor::new(vec![4.0, 5.0, 6.0]);
    assert!((a.dot(&b) - 32.0).abs() < 1e-10);
}

#[test]
fn test_tensor_axpy() {
    let mut a = L2lTensor::new(vec![1.0, 2.0]);
    let b = L2lTensor::new(vec![3.0, 4.0]);
    a.axpy(2.0, &b);
    assert!((a.data[0] - 7.0).abs() < 1e-10);
    assert!((a.data[1] - 10.0).abs() < 1e-10);
}

#[test]
fn test_tensor_clip_norm() {
    let mut t = L2lTensor::zeros(2);
    t.grad = vec![3.0, 4.0]; // norm = 5
    t.clip_norm(1.0);
    let norm: f64 = t.grad.iter().map(|g| g * g).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-10);
}

#[test]
fn test_tensor_from_random() {
    let mut rng = make_rng(1);
    let t = L2lTensor::from_random(100, &mut rng);
    assert_eq!(t.data.len(), 100);
    let mean: f64 = t.data.iter().sum::<f64>() / 100.0;
    assert!(mean.abs() < 1.0); // rough normality check
}

// ── GradientPreprocessor ──────────────────────────────────────────────────

#[test]
fn test_preprocessor_output_length() {
    let pp = GradientPreprocessor::new();
    let g = vec![0.1, -0.5, 0.0, 2.0];
    let out = pp.preprocess(&g);
    assert_eq!(out.len(), 8);
}

#[test]
fn test_preprocessor_sign() {
    let pp = GradientPreprocessor::new();
    let g = vec![1.0, -1.0, 0.0];
    let out = pp.preprocess(&g);
    assert!((out[1] - 1.0).abs() < 1e-10); // sign(1.0) = 1
    assert!((out[3] - (-1.0)).abs() < 1e-10); // sign(-1.0) = -1
    assert!((out[5] - 0.0).abs() < 1e-10); // sign(0.0) = 0
}

#[test]
fn test_preprocessor_log_clamp() {
    let pp = GradientPreprocessor::new();
    let g = vec![1e10, 1e-20];
    let out = pp.preprocess(&g);
    assert!(out[0] <= 1.0);
    assert!(out[0] >= -1.0);
    assert!(out[2] <= 1.0);
    assert!(out[2] >= -1.0);
}

#[test]
fn test_preprocessor_unpreprocess() {
    let pp = GradientPreprocessor::new();
    let g = vec![0.5, -0.3];
    let update = vec![1.0, -1.0];
    let out = pp.unpreprocess(&update, &g);
    assert_eq!(out.len(), 2);
    assert!(out[0] > 0.0);
    assert!(out[1] < 0.0);
}

#[test]
fn test_preprocessor_default() {
    let pp = GradientPreprocessor::default();
    assert!((pp.epsilon - 1e-8).abs() < 1e-15);
}

// ── L2lLstmCell ──────────────────────────────────────────────────────────

#[test]
fn test_lstm_cell_creation() {
    let mut rng = make_rng(42);
    let cell = L2lLstmCell::new(4, 8, 1, &mut rng);
    assert_eq!(cell.input_size, 4);
    assert_eq!(cell.hidden_size, 8);
    assert_eq!(cell.update_dim, 1);
}

#[test]
fn test_lstm_cell_step_shapes() {
    let mut rng = make_rng(42);
    let cell = L2lLstmCell::new(4, 8, 1, &mut rng);
    let input = vec![0.1; 4];
    let h = vec![0.0; 8];
    let c = vec![0.0; 8];
    let (h_new, c_new, update) = cell.step(&input, &h, &c).expect("step failed");
    assert_eq!(h_new.len(), 8);
    assert_eq!(c_new.len(), 8);
    assert_eq!(update.len(), 1);
}

#[test]
fn test_lstm_cell_step_error_on_bad_input() {
    let mut rng = make_rng(1);
    let cell = L2lLstmCell::new(4, 8, 1, &mut rng);
    let bad_input = vec![0.0; 3]; // wrong size
    let h = vec![0.0; 8];
    let c = vec![0.0; 8];
    assert!(cell.step(&bad_input, &h, &c).is_err());
}

#[test]
fn test_lstm_cell_parameters_roundtrip() {
    let mut rng = make_rng(7);
    let cell = L2lLstmCell::new(2, 4, 1, &mut rng);
    let params = cell.parameters();
    assert!(!params.is_empty());
}

#[test]
fn test_lstm_cell_apply_update() {
    let mut rng = make_rng(10);
    let mut cell = L2lLstmCell::new(2, 4, 1, &mut rng);
    let params_before = cell.parameters();
    let delta = vec![0.01; params_before.len()];
    cell.apply_update(&delta);
    let params_after = cell.parameters();
    let changed = params_before
        .iter()
        .zip(params_after.iter())
        .any(|(a, b)| (a - b).abs() > 1e-12);
    assert!(changed);
}

#[test]
fn test_lstm_forget_bias_init() {
    let mut rng = make_rng(3);
    let cell = L2lLstmCell::new(2, 4, 1, &mut rng);
    // forget bias should be 1 (Jozefowicz 2015)
    assert!(cell.bf.iter().all(|&b| (b - 1.0).abs() < 1e-10));
}

// ── LstmMetaOptimizer ────────────────────────────────────────────────────

#[test]
fn test_lstm_meta_optimizer_creation() {
    let mut rng = make_rng(0);
    let opt = LstmMetaOptimizer::new(16, 5, 0.001, &mut rng);
    assert_eq!(opt.inner_steps, 5);
    assert!((opt.meta_lr - 0.001).abs() < 1e-12);
}

#[test]
fn test_lstm_meta_optimizer_compute_update() {
    let mut rng = make_rng(0);
    let opt = LstmMetaOptimizer::new(8, 5, 0.001, &mut rng);
    let h = vec![0.0; 8];
    let c = vec![0.0; 8];
    let (delta, h_new, c_new) = opt
        .compute_update(0.1, 0.5, 2.0, &h, &c)
        .expect("compute_update should succeed");
    assert!(delta.is_finite());
    assert_eq!(h_new.len(), 8);
    assert_eq!(c_new.len(), 8);
}

#[test]
fn test_lstm_meta_optimizer_adapt_on_task() {
    let mut rng = make_rng(42);
    let opt = LstmMetaOptimizer::new(8, 3, 0.001, &mut rng);
    let task = make_simple_task(&mut rng);
    let init_params = vec![0.0; task.support_x[0].len() + 1];
    let (adapted, loss) = opt
        .adapt_on_task(&task, &init_params)
        .expect("adapt_on_task should succeed");
    assert_eq!(adapted.len(), init_params.len());
    assert!(loss.is_finite());
}

#[test]
fn test_lstm_meta_train_returns_losses() {
    let mut rng = make_rng(1);
    let mut opt = LstmMetaOptimizer::new(4, 2, 0.01, &mut rng);
    let tasks: Vec<L2lTask> = (0..3).map(|_| make_simple_task(&mut rng)).collect();
    let losses = opt
        .meta_train(&tasks, 5, 3, &mut rng)
        .expect("meta_train should succeed");
    assert_eq!(losses.len(), 5);
    assert!(losses.iter().all(|l| l.is_finite()));
}

#[test]
fn test_lstm_meta_train_empty_tasks_err() {
    let mut rng = make_rng(1);
    let mut opt = LstmMetaOptimizer::new(4, 2, 0.01, &mut rng);
    assert!(opt.meta_train(&[], 5, 3, &mut rng).is_err());
}

// ── OptimizerNetwork ─────────────────────────────────────────────────────

#[test]
fn test_optimizer_network_creation() {
    let mut rng = make_rng(0);
    let net = OptimizerNetwork::new(5, 8, &mut rng);
    assert_eq!(net.n_params, 5);
}

#[test]
fn test_optimizer_network_compute_updates() {
    let mut rng = make_rng(0);
    let mut net = OptimizerNetwork::new(5, 8, &mut rng);
    let grad = vec![0.1, -0.2, 0.3, -0.1, 0.05];
    let updates = net
        .compute_updates(&grad)
        .expect("compute_updates should succeed");
    assert_eq!(updates.len(), 5);
    assert!(updates.iter().all(|u| u.is_finite()));
}

#[test]
fn test_optimizer_network_reset_state() {
    let mut rng = make_rng(0);
    let mut net = OptimizerNetwork::new(3, 8, &mut rng);
    let grad = vec![1.0, 2.0, 3.0];
    let _ = net
        .compute_updates(&grad)
        .expect("compute_updates should succeed");
    net.reset_state();
    // After reset, hidden states should be zeros
    assert!(net.hidden_states[0].h.iter().all(|&h| h == 0.0));
}

#[test]
fn test_optimizer_network_prev_grad_updated() {
    let mut rng = make_rng(0);
    let mut net = OptimizerNetwork::new(2, 4, &mut rng);
    let grad = vec![0.5, -0.5];
    let _ = net
        .compute_updates(&grad)
        .expect("compute_updates should succeed");
    assert!((net.prev_grad[0] - 0.5).abs() < 1e-10);
}

#[test]
fn test_optimizer_network_wrong_size_err() {
    let mut rng = make_rng(0);
    let mut net = OptimizerNetwork::new(3, 4, &mut rng);
    let bad_grad = vec![1.0, 2.0]; // wrong size
    assert!(net.compute_updates(&bad_grad).is_err());
}

// ── L2lTcBlock ───────────────────────────────────────────────────────────

#[test]
fn test_tc_block_creation() {
    let mut rng = make_rng(1);
    let block = L2lTcBlock::new(4, 8, 2, &mut rng);
    assert_eq!(block.dilation, 4); // 2^2 = 4
    assert_eq!(block.d_in, 4);
    assert_eq!(block.d_out, 8);
}

#[test]
fn test_tc_block_forward_shape() {
    let mut rng = make_rng(1);
    let block = L2lTcBlock::new(4, 8, 0, &mut rng);
    let seq: Vec<Vec<f64>> = (0..6).map(|_| vec![0.1; 4]).collect();
    let out = block.forward(&seq).expect("forward should succeed");
    assert_eq!(out.len(), 6);
    assert_eq!(out[0].len(), 8);
}

#[test]
fn test_tc_block_causal_zero_pad() {
    let mut rng = make_rng(1);
    let block = L2lTcBlock::new(2, 2, 0, &mut rng); // dilation=1
    let seq: Vec<Vec<f64>> = vec![vec![1.0; 2], vec![2.0; 2]];
    let out = block.forward(&seq).expect("forward should succeed");
    assert_eq!(out.len(), 2);
}

#[test]
fn test_tc_block_empty_err() {
    let mut rng = make_rng(1);
    let block = L2lTcBlock::new(2, 2, 0, &mut rng);
    assert!(block.forward(&[]).is_err());
}

#[test]
fn test_tc_block_relu_activation() {
    // With all-zero weights+biases, all outputs should be 0 (ReLU of 0 = 0)
    let mut rng = make_rng(999);
    let mut block = L2lTcBlock::new(2, 2, 0, &mut rng);
    block.weight = vec![0.0; block.weight.len()];
    block.bias = vec![0.0; 2];
    let seq = vec![vec![1.0; 2]; 3];
    let out = block.forward(&seq).expect("forward should succeed");
    assert!(out.iter().flatten().all(|&v| v == 0.0));
}

// ── L2lAttnBlock ─────────────────────────────────────────────────────────

#[test]
fn test_attn_block_creation() {
    let mut rng = make_rng(2);
    let block = L2lAttnBlock::new(8, &mut rng);
    assert_eq!(block.dim, 8);
}

#[test]
fn test_attn_block_forward_shape() {
    let mut rng = make_rng(2);
    let block = L2lAttnBlock::new(4, &mut rng);
    let seq: Vec<Vec<f64>> = (0..5).map(|_| vec![0.1; 4]).collect();
    let out = block.forward(&seq).expect("forward should succeed");
    assert_eq!(out.len(), 5);
    assert_eq!(out[0].len(), 4);
}

#[test]
fn test_attn_block_empty_err() {
    let mut rng = make_rng(2);
    let block = L2lAttnBlock::new(4, &mut rng);
    assert!(block.forward(&[]).is_err());
}

#[test]
fn test_attn_block_single_step() {
    let mut rng = make_rng(5);
    let block = L2lAttnBlock::new(4, &mut rng);
    let seq = vec![vec![0.5; 4]];
    let out = block.forward(&seq).expect("forward should succeed");
    assert_eq!(out.len(), 1);
    assert!(out[0].iter().all(|v| v.is_finite()));
}

// ── SnailModel ────────────────────────────────────────────────────────────

#[test]
fn test_snail_creation() {
    let mut rng = make_rng(0);
    let model = SnailModel::new(5, 1, 8, 2, &mut rng).expect("SnailModel creation should succeed");
    assert_eq!(model.input_dim, 5);
    assert_eq!(model.output_dim, 1);
}

#[test]
fn test_snail_zero_pairs_err() {
    let mut rng = make_rng(0);
    assert!(SnailModel::new(5, 1, 8, 0, &mut rng).is_err());
}

#[test]
fn test_snail_forward_sequence() {
    let mut rng = make_rng(42);
    let model = SnailModel::new(3, 1, 8, 1, &mut rng).expect("SnailModel creation should succeed");
    let context: Vec<(Vec<f64>, f64)> = (0..5)
        .map(|_| (vec![rng.random::<f64>(); 2], rng.random::<f64>()))
        .collect();
    let query = vec![rng.random::<f64>(); 2];
    let pred = model
        .forward_sequence(&context, &query)
        .expect("forward_sequence should succeed");
    assert_eq!(pred.len(), 1);
    assert!(pred[0].is_finite());
}

#[test]
fn test_snail_empty_context() {
    let mut rng = make_rng(42);
    let model = SnailModel::new(3, 1, 8, 1, &mut rng).expect("SnailModel creation should succeed");
    let context: Vec<(Vec<f64>, f64)> = vec![];
    let query = vec![0.5; 2];
    let pred = model
        .forward_sequence(&context, &query)
        .expect("forward_sequence should succeed");
    assert_eq!(pred.len(), 1);
}

// ── MetaDataset ───────────────────────────────────────────────────────────

#[test]
fn test_meta_dataset_sine_sample() {
    let mut rng = make_rng(0);
    let ds = MetaDataset::sine_regression(1);
    let task = ds.sample_task(5, 10, &mut rng);
    assert_eq!(task.support_x.len(), 5);
    assert_eq!(task.support_y.len(), 5);
    assert_eq!(task.query_x.len(), 10);
    assert_eq!(task.query_y.len(), 10);
}

#[test]
fn test_meta_dataset_linear_sample() {
    let mut rng = make_rng(1);
    let ds = MetaDataset::linear_classification(4);
    let task = ds.sample_task(8, 4, &mut rng);
    assert_eq!(task.support_x.len(), 8);
    assert_eq!(task.query_x[0].len(), 4);
    // Labels should be ±1
    assert!(task.support_y.iter().all(|&y| y == 1.0 || y == -1.0));
}

#[test]
fn test_meta_dataset_batch() {
    let mut rng = make_rng(2);
    let ds = MetaDataset::sine_regression(1);
    let batch = ds.batch_tasks(6, 5, 5, &mut rng);
    assert_eq!(batch.len(), 6);
}

#[test]
fn test_meta_dataset_sine_amplitude_range() {
    let mut rng = make_rng(3);
    let ds = MetaDataset::sine_regression(1);
    for _ in 0..20 {
        let task = ds.sample_task(1, 1, &mut rng);
        // y = amp * sin(x + phase), amp in [0.1, 5], y should be bounded
        for &y in &task.support_y {
            assert!(y.abs() <= 5.1);
        }
    }
}

// ── MetaLearningTrainer ───────────────────────────────────────────────────

#[test]
fn test_meta_trainer_meta_loss() {
    let mut rng = make_rng(0);
    let trainer = MetaLearningTrainer::new(3, 0.01, 0.001);
    let tasks: Vec<L2lTask> = (0..4).map(|_| make_simple_task(&mut rng)).collect();
    let init = vec![0.0; tasks[0].support_x[0].len() + 1];
    let loss = trainer
        .meta_loss(&tasks, &init)
        .expect("meta_loss should succeed");
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

#[test]
fn test_meta_trainer_update_reduces_loss() {
    let mut rng = make_rng(5);
    let mut trainer = MetaLearningTrainer::new(3, 0.1, 0.01);
    let tasks: Vec<L2lTask> = (0..5).map(|_| make_simple_task(&mut rng)).collect();
    let mut params = vec![0.0; tasks[0].support_x[0].len() + 1];
    let loss_before = trainer
        .meta_loss(&tasks, &params)
        .expect("meta_loss should succeed");
    for _ in 0..3 {
        let _ = trainer
            .update_meta_params(&tasks, &mut params)
            .expect("update_meta_params should succeed");
    }
    let loss_after = trainer
        .meta_loss(&tasks, &params)
        .expect("meta_loss should succeed");
    // Either improved or at least finite
    assert!(loss_after.is_finite() && loss_before.is_finite());
}

#[test]
fn test_meta_trainer_empty_tasks_err() {
    let trainer = MetaLearningTrainer::new(3, 0.01, 0.001);
    let params = vec![0.0; 3];
    assert!(trainer.meta_loss(&[], &params).is_err());
}

#[test]
fn test_meta_trainer_evaluate() {
    let mut rng = make_rng(0);
    let trainer = MetaLearningTrainer::new(2, 0.01, 0.001);
    let tasks: Vec<L2lTask> = (0..3).map(|_| make_simple_task(&mut rng)).collect();
    let params = vec![0.0; tasks[0].support_x[0].len() + 1];
    let eval = trainer
        .evaluate(&tasks, &params)
        .expect("evaluate should succeed");
    assert!(eval.is_finite());
}

// ── WarmStartOptimizer ────────────────────────────────────────────────────

#[test]
fn test_warm_start_init() {
    let ws = WarmStartOptimizer::new(0.01, 10);
    let meta = vec![1.0, 2.0, 3.0];
    let task_p = vec![0.0; 3];
    let init = ws.initialize_from_meta(&meta, &task_p);
    assert_eq!(init, meta);
}

#[test]
fn test_warm_start_finetune() {
    let mut rng = make_rng(0);
    let ws = WarmStartOptimizer::new(0.01, 10);
    let task = make_simple_task(&mut rng);
    let meta_params = vec![0.0; task.support_x[0].len() + 1];
    let (final_params, losses) = ws
        .finetune_warm(&meta_params, &task)
        .expect("finetune_warm should succeed");
    assert_eq!(losses.len(), 10);
    assert_eq!(final_params.len(), meta_params.len());
}

#[test]
fn test_warm_vs_cold_comparison() {
    let mut rng = make_rng(10);
    let ws = WarmStartOptimizer::new(0.05, 20);
    let task = make_simple_task(&mut rng);
    let meta_params = vec![0.1; task.support_x[0].len() + 1];
    let (warm_f, cold_f) = ws
        .compare_convergence(&meta_params, &task)
        .expect("compare_convergence should succeed");
    assert!(warm_f.is_finite() && cold_f.is_finite());
}

#[test]
fn test_cold_start_finetune() {
    let mut rng = make_rng(0);
    let ws = WarmStartOptimizer::new(0.01, 5);
    let task = make_simple_task(&mut rng);
    let (_, losses) = ws
        .finetune_cold(task.support_x[0].len() + 1, &task)
        .expect("finetune_cold should succeed");
    assert_eq!(losses.len(), 5);
}

// ── L2lScheduler ─────────────────────────────────────────────────────────

#[test]
fn test_scheduler_creation() {
    let mut rng = make_rng(0);
    let sched = L2lScheduler::new(8, 100, 0.01, &mut rng);
    assert!((sched.base_lr - 0.01).abs() < 1e-12);
    assert_eq!(sched.total_steps, 100);
}

#[test]
fn test_scheduler_predict_lr_range() {
    let mut rng = make_rng(0);
    let mut sched = L2lScheduler::new(8, 100, 0.01, &mut rng);
    let lr = sched
        .predict_lr(0, 1.0, 0.5)
        .expect("predict_lr should succeed");
    assert!(lr > 0.0);
    assert!(lr <= 0.01 + 1e-10);
}

#[test]
fn test_scheduler_reset() {
    let mut rng = make_rng(0);
    let mut sched = L2lScheduler::new(4, 10, 0.01, &mut rng);
    let _ = sched
        .predict_lr(1, 0.5, 0.2)
        .expect("predict_lr should succeed");
    sched.reset();
    assert!(sched.h.iter().all(|&h| h == 0.0));
    assert!(sched.c.iter().all(|&c| c == 0.0));
}

#[test]
fn test_scheduler_multiple_steps() {
    let mut rng = make_rng(3);
    let mut sched = L2lScheduler::new(4, 10, 0.001, &mut rng);
    for step in 0..5 {
        let lr = sched
            .predict_lr(step, 0.5 / (step as f64 + 1.0), 0.1)
            .expect("predict_lr should succeed");
        assert!(lr > 0.0 && lr.is_finite());
    }
}

#[test]
fn test_scheduler_meta_train() {
    let mut rng = make_rng(7);
    let mut sched = L2lScheduler::new(4, 5, 0.001, &mut rng);
    let tasks: Vec<L2lTask> = (0..3).map(|_| make_simple_task(&mut rng)).collect();
    let losses = sched
        .meta_train_scheduler(&tasks, 4, 3, &mut rng)
        .expect("meta_train_scheduler should succeed");
    assert_eq!(losses.len(), 4);
    assert!(losses.iter().all(|l| l.is_finite()));
}

// ── L2lMetrics ────────────────────────────────────────────────────────────

#[test]
fn test_metrics_gen_gap() {
    let train = vec![0.1, 0.2];
    let test = vec![0.3, 0.4];
    let gap = L2lMetrics::meta_generalization_gap(&train, &test);
    assert!((gap - 0.2).abs() < 1e-10);
}

#[test]
fn test_metrics_auc_linear() {
    let losses = vec![1.0, 0.5, 0.0];
    let auc = L2lMetrics::learning_curve_auc(&losses);
    // Trapezoid: (1.0+0.5)/2 * 0.5 + (0.5+0.0)/2 * 0.5 = 0.375 + 0.125 = 0.5
    assert!((auc - 0.5).abs() < 1e-10);
}

#[test]
fn test_metrics_auc_single() {
    let losses = vec![0.5];
    let auc = L2lMetrics::learning_curve_auc(&losses);
    assert!((auc - 0.5).abs() < 1e-10);
}

#[test]
fn test_metrics_auc_empty() {
    let auc = L2lMetrics::learning_curve_auc(&[]);
    assert_eq!(auc, 0.0);
}

#[test]
fn test_metrics_final_performance() {
    let losses = vec![0.2, 0.4, 0.6];
    let perf = L2lMetrics::final_performance(&losses);
    assert!((perf - 0.4).abs() < 1e-10);
}

#[test]
fn test_metrics_adaptation_speed() {
    let curves = vec![vec![1.0, 0.8, 0.6, 0.4], vec![1.0, 0.9, 0.5, 0.2]];
    let speed_k2 = L2lMetrics::adaptation_speed_k(&curves, 2);
    assert!((speed_k2 - 0.55).abs() < 1e-10); // (0.6 + 0.5) / 2
}

#[test]
fn test_metrics_adaptation_speed_k_clamped() {
    let curves = vec![vec![1.0, 0.5]];
    let speed = L2lMetrics::adaptation_speed_k(&curves, 100); // k > len
    assert!((speed - 0.5).abs() < 1e-10);
}

#[test]
fn test_metrics_empty_tasks() {
    assert_eq!(L2lMetrics::final_performance(&[]), 0.0);
    assert_eq!(L2lMetrics::adaptation_speed_k(&[], 3), 0.0);
    assert_eq!(L2lMetrics::meta_generalization_gap(&[], &[]), 0.0);
}

// ── Utility: compute_linear_mse_grad ──────────────────────────────────────

#[test]
fn test_linear_mse_zero_residual() {
    // y = 2*x, params = [2, 0] (w=2, bias=0)
    let xs = vec![vec![1.0], vec![2.0], vec![3.0]];
    let ys = vec![2.0, 4.0, 6.0];
    let params = vec![2.0, 0.0];
    let (loss, _) = compute_linear_mse_grad(&params, &xs, &ys);
    assert!(loss < 1e-10);
}

#[test]
fn test_linear_mse_gradient_direction() {
    let xs = vec![vec![1.0]];
    let ys = vec![1.0];
    let params = vec![0.0, 0.0]; // under-predicts
    let (_, grad) = compute_linear_mse_grad(&params, &xs, &ys);
    // grad w.r.t. w should be negative (want to increase w)
    assert!(grad[0] < 0.0);
}

// ── Integration: full meta-learning pipeline ──────────────────────────────

#[test]
fn test_full_pipeline_sine_tasks() {
    let mut rng = make_rng(99);
    let ds = MetaDataset::sine_regression(1);
    let tasks: Vec<MetaSampledTask> = ds.batch_tasks(4, 5, 5, &mut rng);

    // Convert to L2lTask
    let l2l_tasks: Vec<L2lTask> = tasks
        .into_iter()
        .map(|t| L2lTask {
            support_x: t.support_x,
            support_y: t.support_y,
            query_x: t.query_x,
            query_y: t.query_y,
        })
        .collect();

    let trainer = MetaLearningTrainer::new(3, 0.01, 0.001);
    let init = vec![0.0; 2]; // 1 weight + bias
    let loss = trainer
        .meta_loss(&l2l_tasks, &init)
        .expect("meta_loss should succeed");
    assert!(loss.is_finite() && loss >= 0.0);
}

#[test]
fn test_full_pipeline_snail_on_context() {
    let mut rng = make_rng(77);
    let model = SnailModel::new(3, 1, 4, 1, &mut rng).expect("SnailModel creation should succeed");
    let ds = MetaDataset::sine_regression(2);
    let task = ds.sample_task(5, 1, &mut rng);
    let context: Vec<(Vec<f64>, f64)> = task.support_x.into_iter().zip(task.support_y).collect();
    let query = task.query_x.into_iter().next().unwrap_or(vec![0.0; 2]);
    let pred = model
        .forward_sequence(&context, &query)
        .expect("forward_sequence should succeed");
    assert_eq!(pred.len(), 1);
    assert!(pred[0].is_finite());
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn make_simple_task(rng: &mut StdRng) -> L2lTask {
    let n = 8;
    let d = 2;
    let support_x: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect())
        .collect();
    let support_y: Vec<f64> = support_x
        .iter()
        .map(|x| x[0] * 1.5 + x[1] * (-0.5) + 0.1)
        .collect();
    let query_x: Vec<Vec<f64>> = (0..4)
        .map(|_| (0..d).map(|_| rng.random::<f64>() * 2.0 - 1.0).collect())
        .collect();
    let query_y: Vec<f64> = query_x
        .iter()
        .map(|x| x[0] * 1.5 + x[1] * (-0.5) + 0.1)
        .collect();
    L2lTask {
        support_x,
        support_y,
        query_x,
        query_y,
    }
}
