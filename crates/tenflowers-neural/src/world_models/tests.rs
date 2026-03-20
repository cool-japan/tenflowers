//! Tests for world_models::advanced — TD-MPC2, GWM tokenizer/transformer, evaluation.

use super::*;
use scirs2_core::random::SeedableRng;

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

// --- TdMpc2Config ---

#[test]
fn test_tdmpc2_config_fields() {
    let cfg = TdMpc2Config::new(32, 4, 16, 64, 10);
    assert_eq!(cfg.obs_dim, 32);
    assert_eq!(cfg.action_dim, 4);
    assert_eq!(cfg.latent_dim, 16);
    assert_eq!(cfg.horizon, 10);
    assert!((cfg.gamma - 0.99).abs() < 1e-6);
}

// --- TdMpc2Encoder ---

#[test]
fn test_tdmpc2_encoder_output_dim() {
    let enc = TdMpc2Encoder::new(32, 64, 16, 42);
    let obs = vec![0.1_f32; 32];
    let z = enc.encode(&obs);
    assert_eq!(z.len(), 16, "encoder output should match latent_dim");
}

#[test]
fn test_tdmpc2_encoder_finite() {
    let enc = TdMpc2Encoder::new(16, 32, 8, 7);
    let obs = vec![0.5_f32; 16];
    let z = enc.encode(&obs);
    assert!(z.iter().all(|v| v.is_finite()), "encoder output should be finite");
}

// --- TdMpc2Dynamics ---

#[test]
fn test_tdmpc2_dynamics_output_shapes() {
    let dyn_model = TdMpc2Dynamics::new(16, 4, 32, 10);
    let z = vec![0.1_f32; 16];
    let a = vec![0.5_f32; 4];
    let (z_next, reward) = dyn_model.forward(&z, &a);
    assert_eq!(z_next.len(), 16, "next latent should match latent_dim");
    assert!(reward.is_finite(), "reward should be finite");
}

#[test]
fn test_tdmpc2_dynamics_deterministic() {
    let dyn_model = TdMpc2Dynamics::new(8, 2, 16, 5);
    let z = vec![0.3_f32; 8];
    let a = vec![0.1_f32; 2];
    let (z1, r1) = dyn_model.forward(&z, &a);
    let (z2, r2) = dyn_model.forward(&z, &a);
    assert_eq!(z1, z2, "dynamics should be deterministic");
    assert_eq!(r1, r2);
}

// --- TdMpc2Policy ---

#[test]
fn test_tdmpc2_policy_output_bounded() {
    let pol = TdMpc2Policy::new(16, 32, 4, 42);
    let z = vec![0.5_f32; 16];
    let a = pol.act(&z);
    assert_eq!(a.len(), 4, "policy output should match action_dim");
    assert!(
        a.iter().all(|&v| v > -1.0 && v < 1.0),
        "policy output (tanh) should be in (-1, 1)"
    );
}

// --- TdMpc2Model ---

#[test]
fn test_tdmpc2_model_encode() {
    let cfg = TdMpc2Config::new(24, 3, 12, 32, 5);
    let model = TdMpc2Model::new(cfg, 99);
    let obs = vec![0.2_f32; 24];
    let z = model.encode(&obs);
    assert_eq!(z.len(), 12);
}

#[test]
fn test_tdmpc2_model_step() {
    let cfg = TdMpc2Config::new(16, 4, 8, 32, 5);
    let model = TdMpc2Model::new(cfg, 1);
    let z = vec![0.1_f32; 8];
    let a = vec![0.5_f32; 4];
    let (z_next, r) = model.step(&z, &a);
    assert_eq!(z_next.len(), 8);
    assert!(r.is_finite());
}

#[test]
fn test_tdmpc2_model_act() {
    let cfg = TdMpc2Config::new(16, 4, 8, 32, 5);
    let model = TdMpc2Model::new(cfg, 2);
    let z = vec![0.0_f32; 8];
    let a = model.act(&z);
    assert_eq!(a.len(), 4);
    assert!(a.iter().all(|&v| v > -1.0 && v < 1.0));
}

// --- TdMpc2Planner ---

#[test]
fn test_tdmpc2_planner_output_shape() {
    let cfg = TdMpc2Config::new(16, 4, 8, 32, 5);
    let model = TdMpc2Model::new(cfg, 3);
    let planner = TdMpc2Planner::new(10, 0.1, 1.0);
    let mut rng = make_rng(42);
    let obs = vec![0.1_f32; 16];
    let actions = planner.plan(&model, &obs, 5, &mut rng);
    assert_eq!(actions.len(), 5, "plan should have horizon actions");
    assert_eq!(actions[0].len(), 4, "each action should have action_dim elements");
}

#[test]
fn test_tdmpc2_planner_bounded_actions() {
    let cfg = TdMpc2Config::new(8, 2, 4, 16, 3);
    let model = TdMpc2Model::new(cfg, 4);
    let planner = TdMpc2Planner::new(5, 0.2, 0.5);
    let mut rng = make_rng(77);
    let obs = vec![0.5_f32; 8];
    let actions = planner.plan(&model, &obs, 3, &mut rng);
    for a in &actions {
        assert!(
            a.iter().all(|&v| (-1.0..=1.0).contains(&v)),
            "planned actions should be in [-1, 1]"
        );
    }
}

// --- GwmTokenizer ---

#[test]
fn test_gwm_tokenizer_tokenize_shape() {
    let tok = GwmTokenizer::new(16, 8, 8, 42);
    let obs = vec![0.5_f32; 32]; // 4 patches
    let tokens = tok.tokenize(&obs);
    assert_eq!(tokens.len(), 4, "should produce n_patches tokens");
}

#[test]
fn test_gwm_tokenizer_valid_indices() {
    let tok = GwmTokenizer::new(16, 8, 8, 42);
    let obs = vec![0.3_f32; 40];
    let tokens = tok.tokenize(&obs);
    assert!(
        tokens.iter().all(|&t| t < 16),
        "all tokens should be valid codebook indices"
    );
}

#[test]
fn test_gwm_tokenizer_detokenize_shape() {
    let tok = GwmTokenizer::new(8, 4, 4, 7);
    let tokens = vec![0usize, 3, 5, 2];
    let obs = tok.detokenize(&tokens);
    assert_eq!(obs.len(), 16, "detokenized length should be tokens * patch_size");
}

#[test]
fn test_gwm_tokenizer_roundtrip_shape() {
    let tok = GwmTokenizer::new(16, 4, 4, 99);
    let obs = vec![0.1_f32; 20];
    let tokens = tok.tokenize(&obs);
    let reconstructed = tok.detokenize(&tokens);
    assert_eq!(reconstructed.len(), tokens.len() * 4);
}

// --- GwmTransformer ---

#[test]
fn test_gwm_transformer_forward_shape() {
    let model = GwmTransformer::new(16, 32, 10, 42);
    let tokens = vec![0usize, 3, 7, 2];
    let logits = model.forward(&tokens);
    assert_eq!(logits.len(), 4, "logits should have one entry per token");
    assert_eq!(logits[0].len(), 16, "each entry should have vocab_size logits");
}

#[test]
fn test_gwm_transformer_empty_input() {
    let model = GwmTransformer::new(8, 16, 5, 1);
    let logits = model.forward(&[]);
    assert!(logits.is_empty(), "empty input should give empty output");
}

#[test]
fn test_gwm_transformer_sample_next_token_valid() {
    let model = GwmTransformer::new(16, 32, 10, 42);
    let mut rng = make_rng(5);
    let context = vec![0usize, 1, 2];
    let next = model.sample_next_token(&context, 1.0, &mut rng);
    assert!(next < 16, "sampled token should be a valid codebook index");
}

#[test]
fn test_gwm_transformer_greedy_deterministic() {
    let model = GwmTransformer::new(8, 16, 5, 99);
    let mut rng1 = make_rng(1);
    let mut rng2 = make_rng(2);
    let context = vec![0usize, 1];
    let t1 = model.sample_next_token(&context, 0.0, &mut rng1);
    let t2 = model.sample_next_token(&context, 0.0, &mut rng2);
    assert_eq!(t1, t2, "greedy sampling should be deterministic");
}

#[test]
fn test_gwm_transformer_imagine_length() {
    let model = GwmTransformer::new(16, 32, 20, 42);
    let mut rng = make_rng(10);
    let context = vec![0usize, 3];
    let imagined = model.imagine(&context, 5, 1.0, &mut rng);
    assert_eq!(imagined.len(), 5, "imagined sequence should have horizon length");
}

// --- WorldModelEvaluation ---

#[test]
fn test_wme_open_loop_error() {
    let mut eval = WorldModelEvaluation::new();
    let pred = vec![1.0_f32, 2.0];
    let true_lat = vec![1.1_f32, 2.1];
    eval.record_open_loop_error(&pred, &true_lat);
    let err = eval.mean_open_loop_error();
    assert!(err > 0.0, "error should be positive for imperfect prediction");
    assert!(err < 1.0, "error should be small for close predictions");
}

#[test]
fn test_wme_perfect_prediction_error() {
    let mut eval = WorldModelEvaluation::new();
    let pred = vec![1.0_f32, 2.0, 3.0];
    eval.record_open_loop_error(&pred, &pred);
    let err = eval.mean_open_loop_error();
    assert!(err.abs() < 1e-6, "perfect prediction should have zero error");
}

#[test]
fn test_wme_imagination_horizon() {
    let mut eval = WorldModelEvaluation::new();
    let errors = vec![0.1_f32, 0.2, 0.5, 1.5, 2.0]; // diverges at step 3
    let horizon = eval.record_imagination_horizon(&errors, 1.0);
    assert_eq!(horizon, 3, "horizon should stop at first error > threshold");
}

#[test]
fn test_wme_imagination_horizon_no_diverge() {
    let mut eval = WorldModelEvaluation::new();
    let errors = vec![0.1_f32, 0.2, 0.3]; // never exceeds threshold
    let horizon = eval.record_imagination_horizon(&errors, 1.0);
    assert_eq!(horizon, 3, "if never diverges, horizon = full length");
}

#[test]
fn test_wme_planning_return() {
    let mut eval = WorldModelEvaluation::new();
    eval.record_planning_return(10.5);
    eval.record_planning_return(8.3);
    let mean = eval.mean_planning_return();
    assert!((mean - 9.4).abs() < 1e-5, "mean return should be 9.4, got {mean}");
}

#[test]
fn test_wme_empty_metrics() {
    let eval = WorldModelEvaluation::new();
    assert_eq!(eval.mean_open_loop_error(), 0.0);
    assert_eq!(eval.mean_imagination_horizon(), 0.0);
    assert_eq!(eval.mean_planning_return(), 0.0);
    let (f, l, r) = eval.compounding_error_growth();
    assert_eq!(f, 0.0);
    assert_eq!(l, 0.0);
    assert!((r - 1.0).abs() < 1e-6);
}

#[test]
fn test_wme_compounding_error_growth() {
    let mut eval = WorldModelEvaluation::new();
    eval.record_compounding_errors(vec![0.1_f32, 0.2, 0.4]);
    eval.record_compounding_errors(vec![0.2_f32, 0.4, 0.8]);
    let (first, last, ratio) = eval.compounding_error_growth();
    assert!(first > 0.0, "first step error should be positive");
    assert!(last > first, "last step error should be larger than first");
    assert!(ratio > 1.0, "error should compound (ratio > 1)");
}
