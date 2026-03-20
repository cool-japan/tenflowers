use super::*;
use scirs2_core::random::{rngs::StdRng, SeedableRng};

// ── Utility tests ──

#[test]
fn test_irl_softmax_basic() {
    let logits = vec![1.0, 2.0, 3.0];
    let probs = irl_softmax(&logits);
    assert_eq!(probs.len(), 3);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
    assert!(probs[2] > probs[1]);
    assert!(probs[1] > probs[0]);
}

#[test]
fn test_irl_softmax_empty() {
    let probs = irl_softmax(&[]);
    assert!(probs.is_empty());
}

#[test]
fn test_irl_sigmoid_properties() {
    assert!((irl_sigmoid(0.0) - 0.5).abs() < 1e-10);
    assert!(irl_sigmoid(100.0) > 0.99);
    assert!(irl_sigmoid(-100.0) < 0.01);
}

#[test]
fn test_irl_bce_extremes() {
    let loss = irl_bce(0.9, 1.0);
    assert!(loss < 0.2);
    let loss2 = irl_bce(0.1, 1.0);
    assert!(loss2 > 2.0);
}

// ── IrlMlp tests ──

#[test]
fn test_irl_mlp_forward() {
    let mut rng = StdRng::seed_from_u64(42);
    let mlp = IrlMlp::new(&[3, 8, 2], &mut rng).expect("mlp creation");
    let out = mlp.forward(&[1.0, 0.5, -0.5]);
    assert_eq!(out.len(), 2);
}

#[test]
fn test_irl_mlp_too_few_dims() {
    let mut rng = StdRng::seed_from_u64(42);
    let res = IrlMlp::new(&[3], &mut rng);
    assert!(res.is_err());
}

// ── IrlDemonstration tests ──

#[test]
fn test_irl_demonstration_creation() {
    let demo = IrlDemonstration::new(vec![0, 1, 2], vec![0, 1, 0]);
    assert!(demo.is_ok());
    let demo = demo.expect("valid demo");
    assert_eq!(demo.len(), 3);
    assert!(!demo.is_empty());
}

#[test]
fn test_irl_demonstration_length_mismatch() {
    let res = IrlDemonstration::new(vec![0, 1], vec![0]);
    assert!(res.is_err());
}

#[test]
fn test_irl_demonstration_empty() {
    let res = IrlDemonstration::new(vec![], vec![]);
    assert!(res.is_err());
}

// ── IrlMdp tests ──

#[test]
fn test_irl_mdp_creation() {
    let mdp = IrlMdp::new(4, 2, 3);
    assert!(mdp.is_ok());
    let mdp = mdp.expect("valid mdp");
    assert_eq!(mdp.n_states, 4);
    assert_eq!(mdp.n_actions, 2);
    assert_eq!(mdp.feature_dim, 3);
}

#[test]
fn test_irl_mdp_zero_states() {
    let res = IrlMdp::new(0, 2, 3);
    assert!(res.is_err());
}

#[test]
fn test_irl_mdp_transitions() {
    let mut mdp = IrlMdp::new(3, 2, 2).expect("mdp");
    mdp.set_transition(0, 0, 1, 0.8).expect("set trans");
    mdp.set_transition(0, 0, 0, 0.2).expect("set trans");
    assert!((mdp.get_transition(0, 0, 1) - 0.8).abs() < 1e-10);
    assert!((mdp.get_transition(0, 0, 2) - 0.0).abs() < 1e-10);
}

#[test]
fn test_irl_mdp_invalid_transition() {
    let mut mdp = IrlMdp::new(3, 2, 2).expect("mdp");
    assert!(mdp.set_transition(5, 0, 1, 0.5).is_err());
    assert!(mdp.set_transition(0, 5, 1, 0.5).is_err());
    assert!(mdp.set_transition(0, 0, 1, 1.5).is_err());
}

#[test]
fn test_irl_mdp_features() {
    let mut mdp = IrlMdp::new(3, 2, 2).expect("mdp");
    mdp.set_features(0, vec![1.0, 0.0]).expect("set feats");
    let feats = mdp.get_features(0).expect("get feats");
    assert!((feats[0] - 1.0).abs() < 1e-10);
}

#[test]
fn test_irl_mdp_value_iteration() {
    let mut mdp = IrlMdp::new(3, 2, 2).expect("mdp");
    // Simple chain: 0 -> 1 -> 2 (absorbing)
    mdp.set_transition(0, 0, 1, 1.0).expect("t");
    mdp.set_transition(1, 0, 2, 1.0).expect("t");
    mdp.set_transition(2, 0, 2, 1.0).expect("t");
    // Action 1 stays in place
    mdp.set_transition(0, 1, 0, 1.0).expect("t");
    mdp.set_transition(1, 1, 1, 1.0).expect("t");
    mdp.set_transition(2, 1, 2, 1.0).expect("t");

    let rewards = vec![0.0, 0.0, 1.0];
    let values = mdp.value_iteration(&rewards, 0.9, 100).expect("vi");
    // V(2) should be highest (absorbing reward state)
    assert!(values[2] > values[1]);
    assert!(values[1] > values[0]);
}

#[test]
fn test_irl_mdp_compute_policy() {
    let mut mdp = IrlMdp::new(3, 2, 2).expect("mdp");
    mdp.set_transition(0, 0, 1, 1.0).expect("t");
    mdp.set_transition(1, 0, 2, 1.0).expect("t");
    mdp.set_transition(2, 0, 2, 1.0).expect("t");
    mdp.set_transition(0, 1, 0, 1.0).expect("t");
    mdp.set_transition(1, 1, 1, 1.0).expect("t");
    mdp.set_transition(2, 1, 2, 1.0).expect("t");

    let rewards = vec![0.0, 0.0, 1.0];
    let values = mdp.value_iteration(&rewards, 0.9, 100).expect("vi");
    let policy = mdp.compute_policy(&values, 0.9).expect("policy");
    // State 0 and 1 should prefer action 0 (move towards reward)
    assert_eq!(policy[0], 0);
    assert_eq!(policy[1], 0);
}

#[test]
fn test_irl_mdp_state_visitation() {
    let mut mdp = IrlMdp::new(3, 2, 2).expect("mdp");
    mdp.set_transition(0, 0, 1, 1.0).expect("t");
    mdp.set_transition(1, 0, 2, 1.0).expect("t");
    mdp.set_transition(2, 0, 2, 1.0).expect("t");
    mdp.set_transition(0, 1, 0, 1.0).expect("t");
    mdp.set_transition(1, 1, 1, 1.0).expect("t");
    mdp.set_transition(2, 1, 2, 1.0).expect("t");

    // Deterministic policy: always action 0
    let policy = vec![vec![1.0, 0.0]; 3];
    let init = vec![1.0, 0.0, 0.0]; // start in state 0
    let svf = mdp
        .state_visitation_frequency(&policy, &init, 0.9, 100)
        .expect("svf");
    assert_eq!(svf.len(), 3);
    // All states should have positive visitation
    assert!(svf[0] > 0.0);
    assert!(svf[1] > 0.0);
    assert!(svf[2] > 0.0);
}

#[test]
fn test_irl_mdp_soft_policy() {
    let mut mdp = IrlMdp::new(2, 2, 1).expect("mdp");
    mdp.set_transition(0, 0, 1, 1.0).expect("t");
    mdp.set_transition(0, 1, 0, 1.0).expect("t");
    mdp.set_transition(1, 0, 1, 1.0).expect("t");
    mdp.set_transition(1, 1, 0, 1.0).expect("t");

    let rewards = vec![0.0, 1.0];
    let policy = mdp.soft_policy(&rewards, 0.9, 1.0, 100).expect("soft");
    // At state 0, should prefer action 0 (go to state 1 with reward)
    assert!(policy[0][0] > policy[0][1]);
}

// ── MaxEntIrl tests ──

#[test]
fn test_maxent_irl_creation() {
    let mdp = IrlMdp::new(4, 2, 3).expect("mdp");
    let irl = MaxEntIrl::new(&mdp, 0.9);
    assert_eq!(irl.weights.len(), 3);
    assert_eq!(irl.feature_dim, 3);
}

#[test]
fn test_maxent_irl_expert_features() {
    let mut mdp = IrlMdp::new(3, 2, 2).expect("mdp");
    mdp.set_features(0, vec![1.0, 0.0]).expect("f");
    mdp.set_features(1, vec![0.0, 1.0]).expect("f");
    mdp.set_features(2, vec![0.5, 0.5]).expect("f");

    let irl = MaxEntIrl::new(&mdp, 0.9);
    let demo = IrlDemonstration::new(vec![0, 1, 2], vec![0, 0, 0]).expect("demo");
    let f_exp = irl.expert_feature_expectations(&[demo]).expect("fexp");
    assert_eq!(f_exp.len(), 2);
    // f[0] = 1.0 + 0.0*0.9 + 0.5*0.81 = 1.405
    assert!((f_exp[0] - 1.405).abs() < 1e-6);
}

#[test]
fn test_maxent_irl_compute_rewards() {
    let mut mdp = IrlMdp::new(3, 2, 2).expect("mdp");
    mdp.set_features(0, vec![1.0, 0.0]).expect("f");
    mdp.set_features(1, vec![0.0, 1.0]).expect("f");
    mdp.set_features(2, vec![0.5, 0.5]).expect("f");

    let mut irl = MaxEntIrl::new(&mdp, 0.9);
    irl.weights = vec![1.0, 2.0];
    let rewards = irl.compute_rewards();
    assert!((rewards[0] - 1.0).abs() < 1e-10);
    assert!((rewards[1] - 2.0).abs() < 1e-10);
    // R(2) = 1.0*0.5 + 2.0*0.5 = 1.5
    assert!((rewards[2] - 1.5).abs() < 1e-10);
}

#[test]
fn test_maxent_irl_train() {
    let mut mdp = IrlMdp::new(3, 2, 2).expect("mdp");
    for s in 0..3 {
        for a in 0..2 {
            for sp in 0..3 {
                let _ = mdp.set_transition(s, a, sp, 1.0 / 3.0);
            }
        }
    }
    mdp.set_features(0, vec![1.0, 0.0]).expect("f");
    mdp.set_features(1, vec![0.0, 1.0]).expect("f");
    mdp.set_features(2, vec![0.5, 0.5]).expect("f");

    let mut irl = MaxEntIrl::new(&mdp, 0.9);
    let demo = IrlDemonstration::new(vec![0, 1, 2], vec![0, 0, 0]).expect("demo");
    let losses = irl.train(&[demo], 5, 0.1).expect("train");
    assert_eq!(losses.len(), 5);
    // Weights should have changed from zero
    let w_norm: f64 = irl.weights.iter().map(|w| w * w).sum();
    assert!(w_norm > 0.0);
}

#[test]
fn test_maxent_irl_empty_demos() {
    let mdp = IrlMdp::new(3, 2, 2).expect("mdp");
    let irl = MaxEntIrl::new(&mdp, 0.9);
    let res = irl.expert_feature_expectations(&[]);
    assert!(res.is_err());
}

// ── GailDiscriminator tests ──

#[test]
fn test_gail_discriminator_creation() {
    let disc = GailDiscriminator::new(4, 1, &[16, 16], 10.0, 42);
    assert!(disc.is_ok());
    let disc = disc.expect("disc");
    assert_eq!(disc.input_dim, 5);
}

#[test]
fn test_gail_discriminator_zero_dim() {
    let res = GailDiscriminator::new(0, 1, &[16], 0.0, 42);
    assert!(res.is_err());
}

#[test]
fn test_gail_discriminator_forward() {
    let disc = GailDiscriminator::new(3, 1, &[8], 0.0, 42).expect("disc");
    let sa = IrlStateAction::new(vec![1.0, 0.0, -1.0], vec![0.0]);
    let d = disc.forward(&sa);
    assert!(d > 0.0 && d < 1.0);
}

#[test]
fn test_gail_discriminator_reward() {
    let disc = GailDiscriminator::new(3, 1, &[8], 0.0, 42).expect("disc");
    let sa = IrlStateAction::new(vec![0.5, 0.5, 0.5], vec![1.0]);
    let r = disc.reward(&sa);
    // Reward should be finite
    assert!(r.is_finite());
}

#[test]
fn test_gail_discriminator_loss() {
    let disc = GailDiscriminator::new(2, 1, &[8], 0.0, 42).expect("disc");
    let expert = vec![
        IrlStateAction::new(vec![1.0, 0.0], vec![0.0]),
        IrlStateAction::new(vec![0.0, 1.0], vec![1.0]),
    ];
    let policy = vec![
        IrlStateAction::new(vec![0.5, 0.5], vec![0.0]),
        IrlStateAction::new(vec![-0.5, 0.5], vec![1.0]),
    ];
    let loss = disc.compute_loss(&expert, &policy).expect("loss");
    assert!(loss.is_finite() && loss > 0.0);
}

#[test]
fn test_gail_discriminator_train_step() {
    let mut disc = GailDiscriminator::new(2, 1, &[4], 0.0, 42).expect("disc");
    let expert = vec![IrlStateAction::new(vec![1.0, 0.0], vec![0.0])];
    let policy = vec![IrlStateAction::new(vec![0.0, 1.0], vec![1.0])];
    let loss = disc.train_step(&expert, &policy, 0.01).expect("train");
    assert!(loss.is_finite());
}

#[test]
fn test_gail_discriminator_gradient_penalty() {
    let disc = GailDiscriminator::new(2, 1, &[4], 10.0, 42).expect("disc");
    let expert = vec![IrlStateAction::new(vec![1.0, 0.0], vec![0.0])];
    let policy = vec![IrlStateAction::new(vec![0.0, 1.0], vec![1.0])];
    let loss = disc.compute_loss(&expert, &policy).expect("loss");
    assert!(loss.is_finite());
}

// ── GailTrainer tests ──

#[test]
fn test_gail_trainer_creation() {
    let trainer = GailTrainer::new(4, 3, &[8], &[8], 0.01, 0.2, 42);
    assert!(trainer.is_ok());
    let trainer = trainer.expect("trainer");
    assert_eq!(trainer.state_dim, 4);
    assert_eq!(trainer.n_actions, 3);
}

#[test]
fn test_gail_trainer_select_action() {
    let trainer = GailTrainer::new(3, 2, &[4], &[4], 0.01, 0.2, 42).expect("trainer");
    let mut rng = StdRng::seed_from_u64(123);
    let action = trainer
        .select_action(&[1.0, 0.0, -1.0], &mut rng)
        .expect("action");
    assert!(action < 2);
}

#[test]
fn test_gail_trainer_train_step() {
    let mut trainer = GailTrainer::new(2, 2, &[4], &[4], 0.01, 0.2, 42).expect("trainer");
    let expert = vec![IrlStateAction::new(vec![1.0, 0.0], vec![0.0])];
    let policy = vec![IrlStateAction::new(vec![0.0, 1.0], vec![1.0])];
    let result = trainer
        .train_step(&expert, &policy, 0.01, 0.001)
        .expect("step");
    assert!(result.disc_loss.is_finite());
    assert!(result.policy_loss.is_finite());
    assert!(result.entropy_bonus >= 0.0);
}

#[test]
fn test_gail_trainer_n_params() {
    let trainer = GailTrainer::new(3, 2, &[8], &[8], 0.01, 0.2, 42).expect("trainer");
    assert!(trainer.policy_n_params() > 0);
}

// ── DaggerTrainer tests ──

#[test]
fn test_dagger_creation() {
    let dagger = DaggerTrainer::new(4, 3, &[8], 0.9, 42);
    assert!(dagger.is_ok());
    let dagger = dagger.expect("dagger");
    assert_eq!(dagger.state_dim, 4);
    assert_eq!(dagger.n_actions, 3);
    assert!((dagger.beta - 1.0).abs() < 1e-10);
}

#[test]
fn test_dagger_zero_dim() {
    let res = DaggerTrainer::new(0, 3, &[8], 0.9, 42);
    assert!(res.is_err());
}

#[test]
fn test_dagger_select_action() {
    let dagger = DaggerTrainer::new(3, 2, &[4], 0.5, 42).expect("dagger");
    let mut rng = StdRng::seed_from_u64(99);
    // With beta=1.0, should always return expert action
    let _action = dagger.select_action(&[1.0, 0.0, -1.0], 1, &mut rng);
}

#[test]
fn test_dagger_step() {
    let mut dagger = DaggerTrainer::new(2, 2, &[4], 0.9, 42).expect("dagger");
    let trajs: Vec<(Vec<f64>, usize)> = vec![
        (vec![1.0, 0.0], 0),
        (vec![0.0, 1.0], 1),
        (vec![0.5, 0.5], 0),
    ];
    let loss = dagger.dagger_step(&trajs, 0.01, 2).expect("step");
    assert!(loss.is_finite());
    assert_eq!(dagger.dataset_size(), 3);
    assert!(dagger.beta < 1.0); // decayed
}

#[test]
fn test_dagger_multiple_steps() {
    let mut dagger = DaggerTrainer::new(2, 2, &[4], 0.5, 42).expect("dagger");
    for _ in 0..3 {
        let trajs = vec![(vec![1.0, 0.0], 0), (vec![0.0, 1.0], 1)];
        let _ = dagger.dagger_step(&trajs, 0.01, 1).expect("step");
    }
    assert_eq!(dagger.dataset_size(), 6);
    assert_eq!(dagger.iteration, 3);
    // Beta should have decayed: 1.0 * 0.5^3 = 0.125
    assert!((dagger.beta - 0.125).abs() < 1e-10);
}

#[test]
fn test_dagger_predict() {
    let dagger = DaggerTrainer::new(2, 3, &[4], 0.9, 42).expect("dagger");
    let probs = dagger.predict(&[1.0, -1.0]);
    assert_eq!(probs.len(), 3);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
}

// ── AirlModel tests ──

#[test]
fn test_airl_creation() {
    let airl = AirlModel::new(3, 1, &[8], 0.99, false, 42);
    assert!(airl.is_ok());
    let airl = airl.expect("airl");
    assert_eq!(airl.state_dim, 3);
}

#[test]
fn test_airl_state_only() {
    let airl = AirlModel::new(3, 1, &[8], 0.99, true, 42).expect("airl");
    assert!(airl.state_only);
    let r1 = airl.reward(&[1.0, 0.0, 0.0], &[0.0]);
    let r2 = airl.reward(&[1.0, 0.0, 0.0], &[1.0]);
    // State-only: different actions should give same reward
    assert!((r1 - r2).abs() < 1e-10);
}

#[test]
fn test_airl_f_value() {
    let airl = AirlModel::new(2, 1, &[4], 0.9, true, 42).expect("airl");
    let f = airl.f_value(&[1.0, 0.0], &[0.0], &[0.0, 1.0]);
    assert!(f.is_finite());
}

#[test]
fn test_airl_discriminator() {
    let airl = AirlModel::new(2, 1, &[4], 0.9, false, 42).expect("airl");
    let d = airl.discriminator(&[1.0, 0.0], &[0.0], &[0.0, 1.0], -1.0);
    assert!(d > 0.0 && d < 1.0);
}

#[test]
fn test_airl_train_step() {
    let mut airl = AirlModel::new(2, 1, &[4], 0.9, false, 42).expect("airl");
    let expert = vec![(vec![1.0, 0.0], vec![0.0], vec![0.0, 1.0], -0.5)];
    let policy = vec![(vec![0.5, 0.5], vec![1.0], vec![0.5, -0.5], -1.0)];
    let loss = airl.train_step(&expert, &policy, 0.001).expect("step");
    assert!(loss.is_finite());
}

#[test]
fn test_airl_recover_reward() {
    let airl = AirlModel::new(2, 1, &[4], 0.9, true, 42).expect("airl");
    let states = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
    let rewards = airl.recover_reward(&states);
    assert_eq!(rewards.len(), 3);
    for r in &rewards {
        assert!(r.is_finite());
    }
}

#[test]
fn test_airl_n_params() {
    let airl = AirlModel::new(3, 1, &[8], 0.9, false, 42).expect("airl");
    assert!(airl.n_params() > 0);
}

// ── BehavioralCloning tests ──

#[test]
fn test_bc_creation() {
    let bc = BehavioralCloning::new(4, 3, &[8], true, 0.0, 42);
    assert!(bc.is_ok());
    let bc = bc.expect("bc");
    assert_eq!(bc.state_dim, 4);
    assert_eq!(bc.output_dim, 3);
}

#[test]
fn test_bc_zero_dim() {
    let res = BehavioralCloning::new(0, 3, &[8], true, 0.0, 42);
    assert!(res.is_err());
}

#[test]
fn test_bc_train_discrete() {
    let mut bc = BehavioralCloning::new(2, 2, &[4], true, 0.0, 42).expect("bc");
    let states = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
    let actions = vec![vec![0.0], vec![1.0], vec![0.0]];
    let losses = bc.train(&states, &actions, 3, 0.1, 99).expect("train");
    assert_eq!(losses.len(), 3);
    for loss in &losses {
        assert!(loss.is_finite());
    }
}

#[test]
fn test_bc_train_continuous() {
    let mut bc = BehavioralCloning::new(2, 2, &[4], false, 0.0, 42).expect("bc");
    let states = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let actions = vec![vec![0.5, -0.5], vec![-0.5, 0.5]];
    let losses = bc.train(&states, &actions, 3, 0.01, 99).expect("train");
    assert_eq!(losses.len(), 3);
}

#[test]
fn test_bc_predict() {
    let bc = BehavioralCloning::new(2, 3, &[4], true, 0.0, 42).expect("bc");
    let probs = bc.predict(&[1.0, -1.0]);
    assert_eq!(probs.len(), 3);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
}

#[test]
fn test_bc_predict_action() {
    let bc = BehavioralCloning::new(2, 3, &[4], true, 0.0, 42).expect("bc");
    let action = bc.predict_action(&[0.5, -0.5]);
    assert!(action < 3);
}

#[test]
fn test_bc_noise_augmentation() {
    let mut bc = BehavioralCloning::new(2, 2, &[4], true, 0.1, 42).expect("bc");
    let states = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let actions = vec![vec![0.0], vec![1.0]];
    let losses = bc.train(&states, &actions, 2, 0.1, 99).expect("train");
    assert_eq!(losses.len(), 2);
}

#[test]
fn test_bc_n_params() {
    let bc = BehavioralCloning::new(3, 2, &[8, 8], true, 0.0, 42).expect("bc");
    // (3*8+8) + (8*8+8) + (8*2+2) = 32+8+72+8+18 = 138
    assert!(bc.n_params() > 0);
}

// ── RewardShaping tests ──

#[test]
fn test_reward_shaping_tabular() {
    let rs = RewardShaping::tabular(0.9, vec![0.0, 1.0, 2.0]);
    let shaped = rs.shape_reward(1.0, &[0.0], &[1.0]);
    // F = 0.9*1.0 - 0.0 = 0.9; shaped = 1.0 + 0.9 = 1.9
    assert!((shaped - 1.9).abs() < 1e-10);
}

#[test]
fn test_reward_shaping_linear() {
    let rs = RewardShaping::linear(0.9, vec![1.0, 2.0]);
    // Phi(s) = 1*1 + 2*0 = 1, Phi(s') = 1*0 + 2*1 = 2
    // F = 0.9*2 - 1 = 0.8
    let shaped = rs.shape_reward(1.0, &[1.0, 0.0], &[0.0, 1.0]);
    assert!((shaped - 1.8).abs() < 1e-10);
}

#[test]
fn test_reward_shaping_neural() {
    let rs = RewardShaping::neural(0.9, 2, 4, 42).expect("neural rs");
    let shaped = rs.shape_reward(1.0, &[1.0, 0.0], &[0.0, 1.0]);
    assert!(shaped.is_finite());
}

#[test]
fn test_reward_shaping_trajectory() {
    let rs = RewardShaping::tabular(0.9, vec![0.0, 1.0, 2.0]);
    let rewards = vec![1.0, 2.0];
    let states = vec![vec![0.0], vec![1.0], vec![2.0]];
    let shaped = rs.shape_trajectory(&rewards, &states).expect("traj");
    assert_eq!(shaped.len(), 2);
}

#[test]
fn test_reward_shaping_invariance() {
    // Under potential-based shaping, the optimal policy is invariant
    let rs = RewardShaping::tabular(1.0, vec![0.0, 5.0, 10.0]);
    // If gamma=1: F(s,s') = Phi(s') - Phi(s)
    // Cumulative shaping over a trajectory sums to Phi(final) - Phi(initial)
    let states = vec![vec![0.0], vec![1.0], vec![2.0]];
    let rewards = vec![1.0, 1.0];
    let shaped = rs.shape_trajectory(&rewards, &states).expect("inv");
    // Sum of shaped = sum of rewards + Phi(2) - Phi(0) = 2 + 10 - 0 = 12
    let total_shaped: f64 = shaped.iter().sum();
    assert!((total_shaped - 12.0).abs() < 1e-8);
}

// ── IrlMetrics tests ──

#[test]
fn test_irl_metrics_evd() {
    let evd = IrlMetrics::expected_value_difference(&[10.0, 12.0, 11.0], &[9.0, 10.0, 10.0])
        .expect("evd");
    assert!(evd > 0.0);
    assert!((evd - 1.333333).abs() < 0.01);
}

#[test]
fn test_irl_metrics_evd_empty() {
    let res = IrlMetrics::expected_value_difference(&[], &[1.0]);
    assert!(res.is_err());
}

#[test]
fn test_irl_metrics_fee() {
    let fee = IrlMetrics::feature_expectation_error(&[1.0, 0.0], &[0.0, 1.0]).expect("fee");
    // sqrt(1 + 1) = sqrt(2)
    assert!((fee - std::f64::consts::SQRT_2).abs() < 1e-10);
}

#[test]
fn test_irl_metrics_reward_correlation() {
    // Perfect positive correlation
    let corr = IrlMetrics::reward_correlation(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
        .expect("corr");
    assert!((corr - 1.0).abs() < 1e-10);

    // Perfect negative correlation
    let corr2 =
        IrlMetrics::reward_correlation(&[1.0, 2.0, 3.0], &[-1.0, -2.0, -3.0]).expect("corr2");
    assert!((corr2 - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_irl_metrics_policy_entropy() {
    // Uniform over 2 actions: entropy = ln(2)
    let policy = vec![vec![0.5, 0.5]];
    let ent = IrlMetrics::policy_entropy(&policy);
    assert!((ent - 2.0_f64.ln()).abs() < 1e-10);
}

#[test]
fn test_irl_metrics_success_rate() {
    let sr = IrlMetrics::success_rate(&[10.0, 5.0, 8.0, 12.0], 8.0);
    // 3 out of 4 >= 8.0
    assert!((sr - 0.75).abs() < 1e-10);
}

#[test]
fn test_irl_report_display() {
    let report = IrlReport {
        evd: 0.5,
        feature_expectation_error: 0.1,
        reward_correlation: 0.95,
        policy_entropy: 1.2,
        success_rate: 0.8,
    };
    let s = format!("{}", report);
    assert!(s.contains("EVD"));
    assert!(s.contains("0.95"));
}

#[test]
fn test_irl_metrics_evaluate() {
    let report = IrlMetrics::evaluate(
        &[10.0, 12.0],
        &[9.0, 11.0],
        &[1.0, 0.5],
        &[0.9, 0.6],
        &[1.0, 2.0, 3.0],
        &[1.1, 2.1, 3.1],
        &[vec![0.5, 0.5], vec![0.7, 0.3]],
        8.0,
    )
    .expect("evaluate");
    assert!(report.evd >= 0.0);
    assert!(report.feature_expectation_error >= 0.0);
    assert!(report.reward_correlation >= -1.0);
    assert!(report.reward_correlation <= 1.0);
    assert!(report.policy_entropy >= 0.0);
    assert!(report.success_rate >= 0.0 && report.success_rate <= 1.0);
}

#[test]
fn test_irl_state_action() {
    let sa = IrlStateAction::new(vec![1.0, 2.0], vec![0.0]);
    let feats = sa.as_features();
    assert_eq!(feats.len(), 3);
    assert!((feats[0] - 1.0).abs() < 1e-10);
    assert!((feats[2] - 0.0).abs() < 1e-10);
}
