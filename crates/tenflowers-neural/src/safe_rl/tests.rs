use super::*;

// ── §0 Utility tests ────────────────────────────────────────────────

#[test]
fn test_srl_softmax_basic() {
    let logits = vec![1.0, 2.0, 3.0];
    let probs = srl_softmax(&logits);
    assert_eq!(probs.len(), 3);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
    // probs should be increasing
    assert!(probs[0] < probs[1]);
    assert!(probs[1] < probs[2]);
}

#[test]
fn test_srl_softmax_empty() {
    let probs = srl_softmax(&[]);
    assert!(probs.is_empty());
}

#[test]
fn test_srl_linear_forward() {
    let layer = SrlLinear::new(3, 2, 42);
    let input = vec![1.0, 0.5, -1.0];
    let output = layer.forward(&input).expect("forward should succeed");
    assert_eq!(output.len(), 2);
}

#[test]
fn test_srl_linear_dim_mismatch() {
    let layer = SrlLinear::new(3, 2, 42);
    let input = vec![1.0, 0.5]; // wrong size
    assert!(layer.forward(&input).is_err());
}

#[test]
fn test_srl_mlp_forward() {
    let mlp = SrlMlp::new(4, 16, 3, 99);
    let input = vec![0.1, -0.2, 0.3, 0.4];
    let output = mlp.forward(&input).expect("forward should succeed");
    assert_eq!(output.len(), 3);
}

// ── §1 CMDP environment tests ───────────────────────────────────────

#[test]
fn test_safe_grid_world_reset() {
    let mut env = SrlSafeGridWorld::new(5, vec![(2, 2)]);
    let state = env.reset(42).expect("reset should succeed");
    assert_eq!(state.len(), 4);
    assert_eq!(env.position, (0, 0));
}

#[test]
fn test_safe_grid_world_step() {
    let mut env = SrlSafeGridWorld::new(5, vec![(1, 0)]);
    env.reset(42).expect("reset should succeed");
    // Move down (action 1)
    let result = env.step(1).expect("step should succeed");
    assert_eq!(result.costs.len(), 1);
    assert!(!result.done || env.position == env.goal);
}

#[test]
fn test_safe_grid_world_obstacle_collision() {
    let mut env = SrlSafeGridWorld::new(5, vec![(0, 1)]);
    env.reset(42).expect("reset should succeed");
    // Try to move right into obstacle
    let _result = env.step(3).expect("step should succeed");
    // Should stay at (0,0) since (0,1) is obstacle
    assert_eq!(env.position, (0, 0));
}

#[test]
fn test_safe_cart_pole_reset() {
    let mut env = SrlSafeCartPole::new(0.1);
    let state = env.reset(42).expect("reset should succeed");
    assert_eq!(state.len(), 4);
}

#[test]
fn test_safe_cart_pole_step() {
    let mut env = SrlSafeCartPole::new(0.1);
    env.reset(42).expect("reset should succeed");
    let result = env.step(1).expect("step should succeed");
    assert_eq!(result.next_state.len(), 4);
    assert_eq!(result.costs.len(), 1);
}

#[test]
fn test_safe_cart_pole_cost_generation() {
    let mut env = SrlSafeCartPole::new(0.05);
    env.reset(42).expect("reset should succeed");
    env.theta = 0.06; // above limit
    let result = env.step(0).expect("step should succeed");
    // Cost should be 1.0 since angle exceeds limit
    assert!(result.costs[0] > 0.0);
}

// ── §2 Lagrangian RL tests ──────────────────────────────────────────

#[test]
fn test_lagrangian_rl_creation() {
    let config = SrlLagrangianConfig {
        num_constraints: 2,
        cost_thresholds: vec![10.0, 5.0],
        ..SrlLagrangianConfig::default()
    };
    let agent = SrlLagrangianRl::new(config, 4, 3, 42);
    assert_eq!(agent.lambdas.len(), 2);
    assert_eq!(agent.lambdas[0], 0.0);
}

#[test]
fn test_lagrangian_rl_select_action() {
    let config = SrlLagrangianConfig::default();
    let agent = SrlLagrangianRl::new(config, 4, 3, 42);
    let state = vec![0.1, 0.2, 0.3, 0.4];
    let action = agent
        .select_action(&state, 123)
        .expect("should select action");
    assert!(action < 3);
}

#[test]
fn test_lagrangian_rl_train_step() {
    let config = SrlLagrangianConfig {
        num_constraints: 1,
        cost_thresholds: vec![0.5],
        ..SrlLagrangianConfig::default()
    };
    let mut agent = SrlLagrangianRl::new(config, 4, 2, 42);
    let rewards = vec![1.0, 0.5, 0.8, 1.0, 0.3];
    let costs = vec![vec![0.0, 1.0, 0.0, 1.0, 0.0]];

    let result = agent.train_step(&rewards, &costs).expect("should train");
    assert_eq!(result.constraint_violations.len(), 1);
    assert_eq!(result.lambdas.len(), 1);
}

#[test]
fn test_lagrangian_rl_lambda_increase_on_violation() {
    let config = SrlLagrangianConfig {
        num_constraints: 1,
        cost_thresholds: vec![0.1],
        dual_lr: 0.1,
        pid_kp: 1.0,
        pid_ki: 0.0,
        pid_kd: 0.0,
        ..SrlLagrangianConfig::default()
    };
    let mut agent = SrlLagrangianRl::new(config, 4, 2, 42);

    // High cost exceeds threshold
    let rewards = vec![1.0; 10];
    let costs = vec![vec![1.0; 10]]; // avg cost = 1.0 > threshold 0.1

    let result = agent.train_step(&rewards, &costs).expect("should train");
    assert!(result.lambdas[0] > 0.0); // lambda should increase
}

#[test]
fn test_lagrangian_rl_wrong_constraints() {
    let config = SrlLagrangianConfig {
        num_constraints: 2,
        cost_thresholds: vec![0.5, 0.5],
        ..SrlLagrangianConfig::default()
    };
    let mut agent = SrlLagrangianRl::new(config, 4, 2, 42);
    let rewards = vec![1.0; 5];
    let costs = vec![vec![0.0; 5]]; // only 1 constraint, expected 2
    assert!(agent.train_step(&rewards, &costs).is_err());
}

// ── §3 CPO tests ────────────────────────────────────────────────────

#[test]
fn test_cpo_creation() {
    let config = SrlCpoConfig::default();
    let agent = SrlCpoAgent::new(config, 4, 2, 42);
    assert_eq!(agent.policy.layer1.in_dim, 4);
    assert_eq!(agent.policy.layer2.out_dim, 2);
}

#[test]
fn test_cpo_select_action() {
    let config = SrlCpoConfig::default();
    let agent = SrlCpoAgent::new(config, 4, 2, 42);
    let state = vec![0.1, -0.1, 0.2, 0.0];
    let action = agent
        .select_action(&state, 99)
        .expect("should select action");
    assert!(action < 2);
}

#[test]
fn test_cpo_compute_gae() {
    let config = SrlCpoConfig {
        gamma: 0.99,
        gae_lambda: 0.95,
        ..SrlCpoConfig::default()
    };
    let agent = SrlCpoAgent::new(config, 4, 2, 42);
    let rewards = vec![1.0, 1.0, 1.0, 1.0, 0.0];
    let values = vec![3.5, 2.5, 1.5, 0.5, 0.0];
    let dones = vec![false, false, false, false, true];
    let advantages = agent.compute_gae(&rewards, &values, &dones);
    assert_eq!(advantages.len(), 5);
}

#[test]
fn test_cpo_update() {
    let config = SrlCpoConfig::default();
    let mut agent = SrlCpoAgent::new(config, 4, 2, 42);
    let states: Vec<Vec<f64>> = (0..10)
        .map(|i| vec![i as f64 * 0.1, 0.0, 0.0, 0.0])
        .collect();
    let actions: Vec<usize> = (0..10).map(|i| i % 2).collect();
    let rewards: Vec<f64> = vec![1.0; 10];
    let costs: Vec<f64> = vec![0.1; 10];
    let dones: Vec<bool> = vec![false; 9]
        .into_iter()
        .chain(std::iter::once(true))
        .collect();

    let result = agent
        .update(&states, &actions, &rewards, &costs, &dones)
        .expect("should update");
    assert!(result.kl_divergence >= 0.0);
}

#[test]
fn test_cpo_empty_trajectory() {
    let config = SrlCpoConfig::default();
    let mut agent = SrlCpoAgent::new(config, 4, 2, 42);
    let result = agent.update(&[], &[], &[], &[], &[]);
    assert!(result.is_err());
}

// ── §4 Safety Layer tests ───────────────────────────────────────────

#[test]
fn test_safety_layer_creation() {
    let config = SrlSafetyLayerConfig {
        state_dim: 4,
        action_dim: 2,
        num_constraints: 1,
        lr: 0.01,
        safety_margin: 0.1,
    };
    let layer = SrlSafetyLayer::new(config);
    assert_eq!(layer.constraint_models.len(), 1);
}

#[test]
fn test_safety_layer_correct_safe_action() {
    let config = SrlSafetyLayerConfig {
        state_dim: 2,
        action_dim: 2,
        num_constraints: 1,
        lr: 0.01,
        safety_margin: 0.1,
    };
    let layer = SrlSafetyLayer::new(config);
    // Zero-initialized model means g(s,a)=0, which is > -margin
    // So correction will happen
    let state = vec![1.0, 0.0];
    let action = vec![1.0, 0.0];
    let corrected = layer
        .correct_action(&state, &action)
        .expect("should correct");
    assert_eq!(corrected.len(), 2);
}

#[test]
fn test_safety_layer_update_models() {
    let config = SrlSafetyLayerConfig {
        state_dim: 2,
        action_dim: 2,
        num_constraints: 1,
        lr: 0.01,
        safety_margin: 0.1,
    };
    let mut layer = SrlSafetyLayer::new(config);
    let state = vec![1.0, 0.0];
    let action = vec![0.5, 0.5];
    let costs = vec![1.0];
    layer
        .update_models(&state, &action, &costs)
        .expect("should update");
    // After update, the model should have non-zero weights
    let model = &layer.constraint_models[0];
    let has_nonzero = model.w_state.iter().any(|&w| w.abs() > 1e-15)
        || model.w_action.iter().any(|&w| w.abs() > 1e-15);
    assert!(has_nonzero);
}

#[test]
fn test_safety_layer_wrong_cost_dim() {
    let config = SrlSafetyLayerConfig {
        state_dim: 2,
        action_dim: 2,
        num_constraints: 2,
        lr: 0.01,
        safety_margin: 0.1,
    };
    let mut layer = SrlSafetyLayer::new(config);
    let state = vec![1.0, 0.0];
    let action = vec![0.5, 0.5];
    let costs = vec![1.0]; // should be 2
    assert!(layer.update_models(&state, &action, &costs).is_err());
}

#[test]
fn test_constraint_model_evaluate() {
    let mut model = SrlConstraintModel::new(2, 2);
    model.w_state = vec![1.0, -1.0];
    model.w_action = vec![0.5, 0.5];
    model.bias = 0.1;
    let val = model.evaluate(&[1.0, 0.5], &[0.2, 0.3]);
    // 1*1.0 + (-1)*0.5 + 0.5*0.2 + 0.5*0.3 + 0.1 = 1.0 - 0.5 + 0.1 + 0.15 + 0.1 = 0.85
    assert!((val - 0.85).abs() < 1e-10);
}

// ── §5 Safe Explorer tests ──────────────────────────────────────────

#[test]
fn test_safe_explorer_creation() {
    let config = SrlSafeExplorerConfig {
        ensemble_size: 3,
        state_dim: 4,
        action_dim: 2,
        beta: 2.0,
        cost_threshold: 0.5,
        min_safety_prob: 0.8,
    };
    let explorer = SrlSafeExplorer::new(config, 42);
    assert_eq!(explorer.ensemble.len(), 3);
}

#[test]
fn test_safe_explorer_predict_cost_stats() {
    let config = SrlSafeExplorerConfig {
        ensemble_size: 5,
        state_dim: 2,
        action_dim: 2,
        beta: 1.0,
        cost_threshold: 1.0,
        min_safety_prob: 0.5,
    };
    let explorer = SrlSafeExplorer::new(config, 42);
    let state = vec![0.1, 0.2];
    let (mean, std) = explorer
        .predict_cost_stats(&state, 0)
        .expect("should predict");
    assert!(std >= 0.0);
    // mean can be anything with random init
    let _ = mean; // just check it doesn't panic
}

#[test]
fn test_safe_explorer_is_safe() {
    let config = SrlSafeExplorerConfig {
        ensemble_size: 3,
        state_dim: 2,
        action_dim: 2,
        beta: 0.0,             // no pessimism for easier testing
        cost_threshold: 100.0, // very high threshold
        min_safety_prob: 0.5,
    };
    let explorer = SrlSafeExplorer::new(config, 42);
    let state = vec![0.0, 0.0];
    // With high threshold and no pessimism, should likely be safe
    let result = explorer.is_safe(&state, 0);
    assert!(result.is_ok());
}

#[test]
fn test_safe_explorer_select_action() {
    let config = SrlSafeExplorerConfig {
        ensemble_size: 3,
        state_dim: 2,
        action_dim: 3,
        beta: 0.0,
        cost_threshold: 100.0,
        min_safety_prob: 0.0,
    };
    let explorer = SrlSafeExplorer::new(config, 42);
    let state = vec![0.5, -0.5];
    let prefs = vec![1.0, 2.0, 0.5];
    let action = explorer
        .safe_select_action(&state, &prefs, 99)
        .expect("should select");
    assert!(action < 3);
}

// ── §6 Robust MDP tests ────────────────────────────────────────────

#[test]
fn test_robust_mdp_simple() {
    // 2-state, 2-action MDP
    let config = SrlRobustMdpConfig {
        num_states: 2,
        num_actions: 2,
        gamma: 0.9,
        max_iterations: 100,
        tolerance: 1e-6,
    };
    let transitions = vec![
        vec![vec![0.5, 0.5], vec![0.8, 0.2]], // state 0
        vec![vec![0.3, 0.7], vec![0.6, 0.4]], // state 1
    ];
    let rewards = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
    let radii = vec![vec![0.0, 0.0], vec![0.0, 0.0]]; // no uncertainty

    let mdp = SrlRobustMdp::new(config, transitions, rewards, radii).expect("should create MDP");
    let (values, policy) = mdp.solve().expect("should solve");
    assert_eq!(values.len(), 2);
    assert_eq!(policy.len(), 2);
    assert!(values[0] > 0.0);
    assert!(values[1] > 0.0);
}

#[test]
fn test_robust_mdp_with_uncertainty() {
    let config = SrlRobustMdpConfig {
        num_states: 2,
        num_actions: 2,
        gamma: 0.9,
        max_iterations: 100,
        tolerance: 1e-6,
    };
    let transitions = vec![
        vec![vec![0.9, 0.1], vec![0.1, 0.9]],
        vec![vec![0.5, 0.5], vec![0.5, 0.5]],
    ];
    let rewards = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let radii = vec![vec![0.2, 0.2], vec![0.2, 0.2]]; // moderate uncertainty

    let mdp = SrlRobustMdp::new(config, transitions.clone(), rewards.clone(), radii)
        .expect("should create MDP");
    let (values_robust, _) = mdp.solve().expect("should solve");

    // With uncertainty, values should be lower than without
    let config2 = SrlRobustMdpConfig {
        num_states: 2,
        num_actions: 2,
        gamma: 0.9,
        max_iterations: 100,
        tolerance: 1e-6,
    };
    let radii_zero = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
    let mdp2 =
        SrlRobustMdp::new(config2, transitions, rewards, radii_zero).expect("should create MDP");
    let (values_nominal, _) = mdp2.solve().expect("should solve");

    // Robust values should be <= nominal values (or close)
    for (vr, vn) in values_robust.iter().zip(values_nominal.iter()) {
        assert!(*vr <= *vn + 1e-6);
    }
}

// ── §7 Shielded Policy tests ────────────────────────────────────────

#[test]
fn test_shielded_policy_creation() {
    let config = SrlShieldConfig {
        grid_resolution: 5,
        state_bounds: vec![(0.0, 1.0), (0.0, 1.0)],
        num_actions: 4,
    };
    let shield = SrlShieldedPolicy::new(config);
    assert_eq!(shield.grid_total, 25); // 5^2
    assert!(shield.safe_set.iter().all(|&s| s)); // all initially safe
}

#[test]
fn test_shielded_policy_grid_conversion() {
    let config = SrlShieldConfig {
        grid_resolution: 10,
        state_bounds: vec![(0.0, 1.0), (0.0, 1.0)],
        num_actions: 4,
    };
    let shield = SrlShieldedPolicy::new(config);

    let state = vec![0.25, 0.75];
    let idx = shield.state_to_grid_index(&state);
    assert!(idx < 100);

    let recovered = shield.grid_index_to_state(idx);
    assert_eq!(recovered.len(), 2);
    // Should be approximately correct
    assert!((recovered[0] - 0.25).abs() < 0.15);
    assert!((recovered[1] - 0.75).abs() < 0.15);
}

#[test]
fn test_shielded_policy_compute_safe_set() {
    let config = SrlShieldConfig {
        grid_resolution: 3,
        state_bounds: vec![(0.0, 1.0)],
        num_actions: 2,
    };
    let mut shield = SrlShieldedPolicy::new(config);
    // 3 states: 0, 1, 2
    // Mark state 2 as unsafe
    let unsafe_states = vec![2];
    // Transition: action 0 stays, action 1 moves right
    let transition = |s: usize, a: usize| -> usize {
        if a == 1 {
            (s + 1).min(2)
        } else {
            s
        }
    };
    shield.compute_safe_set(&unsafe_states, transition);

    assert!(shield.safe_set[0]); // state 0 is safe (can stay)
    assert!(shield.safe_set[1]); // state 1 is safe (can go left with action 0)
    assert!(!shield.safe_set[2]); // state 2 is unsafe
}

#[test]
fn test_shielded_policy_shield_action() {
    let config = SrlShieldConfig {
        grid_resolution: 3,
        state_bounds: vec![(0.0, 1.0)],
        num_actions: 2,
    };
    let mut shield = SrlShieldedPolicy::new(config);
    // Remove action 1 from state 1
    shield.allowed_actions[1] = vec![0]; // only action 0 allowed at state 1

    let state = vec![0.5]; // maps to grid index 1
    let safe_action = shield.shield(&state, 1); // proposed action 1 is not allowed
    assert_eq!(safe_action, 0); // should be corrected to 0
}

#[test]
fn test_shielded_policy_is_state_safe() {
    let config = SrlShieldConfig {
        grid_resolution: 5,
        state_bounds: vec![(0.0, 1.0)],
        num_actions: 2,
    };
    let mut shield = SrlShieldedPolicy::new(config);
    shield.safe_set[3] = false;

    let state = vec![0.65]; // maps to index 3
    assert!(!shield.is_state_safe(&state));

    let state2 = vec![0.1]; // maps to index 0
    assert!(shield.is_state_safe(&state2));
}

// ── §8 Barrier Function tests ───────────────────────────────────────

#[test]
fn test_sphere_barrier() {
    let barrier = SrlSphereBarrier {
        center: vec![0.0, 0.0],
        radius: 1.0,
    };
    assert!(barrier.evaluate(&[0.0, 0.0]) > 0.0); // at center, safe
    assert!(barrier.evaluate(&[0.5, 0.5]) > 0.0); // inside sphere
    assert!(barrier.evaluate(&[1.5, 0.0]) < 0.0); // outside sphere
}

#[test]
fn test_halfspace_barrier() {
    let barrier = SrlHalfspaceBarrier {
        normal: vec![1.0, 0.0],
        offset: -0.5,
    };
    assert!(barrier.evaluate(&[1.0, 0.0]) > 0.0); // x > 0.5
    assert!(barrier.evaluate(&[0.0, 0.0]) < 0.0); // x < 0.5
}

#[test]
fn test_barrier_function_safe_control_already_safe() {
    let config = SrlBarrierConfig {
        state_dim: 2,
        control_dim: 2,
        alpha: 1.0,
        barrier_type: SrlBarrierType::Standard,
        fd_step: 1e-5,
        control_bounds: vec![(-10.0, 10.0), (-10.0, 10.0)],
    };
    let barrier = SrlSphereBarrier {
        center: vec![0.0, 0.0],
        radius: 5.0,
    };
    let cbf = SrlBarrierFunction::new(config, Box::new(barrier));

    let state = vec![0.0, 0.0]; // at center, h(x) = 25 >> 0
    let u_nominal = vec![0.1, 0.1];
    let u_safe = cbf
        .compute_safe_control(&state, &u_nominal)
        .expect("should compute");
    assert_eq!(u_safe.len(), 2);
    // Should be close to nominal since far from boundary
    assert!((u_safe[0] - 0.1).abs() < 1.0);
    assert!((u_safe[1] - 0.1).abs() < 1.0);
}

#[test]
fn test_barrier_function_near_boundary() {
    let config = SrlBarrierConfig {
        state_dim: 2,
        control_dim: 2,
        alpha: 1.0,
        barrier_type: SrlBarrierType::Standard,
        fd_step: 1e-5,
        control_bounds: vec![(-10.0, 10.0), (-10.0, 10.0)],
    };
    let barrier = SrlSphereBarrier {
        center: vec![0.0, 0.0],
        radius: 1.0,
    };
    let cbf = SrlBarrierFunction::new(config, Box::new(barrier));

    let state = vec![0.95, 0.0]; // near boundary, h = 1 - 0.9025 = 0.0975
    let u_outward = vec![1.0, 0.0]; // trying to go further out
    let u_safe = cbf
        .compute_safe_control(&state, &u_outward)
        .expect("should compute");
    assert_eq!(u_safe.len(), 2);
}

#[test]
fn test_barrier_function_is_safe() {
    let config = SrlBarrierConfig {
        state_dim: 2,
        control_dim: 2,
        alpha: 1.0,
        barrier_type: SrlBarrierType::Standard,
        fd_step: 1e-5,
        control_bounds: vec![(-1.0, 1.0), (-1.0, 1.0)],
    };
    let barrier = SrlSphereBarrier {
        center: vec![0.0, 0.0],
        radius: 1.0,
    };
    let cbf = SrlBarrierFunction::new(config, Box::new(barrier));
    assert!(cbf.is_safe(&[0.0, 0.0]));
    assert!(!cbf.is_safe(&[2.0, 0.0]));
}

#[test]
fn test_barrier_exponential_type() {
    let config = SrlBarrierConfig {
        state_dim: 2,
        control_dim: 2,
        alpha: 1.0,
        barrier_type: SrlBarrierType::Exponential { gamma: 0.5 },
        fd_step: 1e-5,
        control_bounds: vec![(-10.0, 10.0), (-10.0, 10.0)],
    };
    let barrier = SrlSphereBarrier {
        center: vec![0.0, 0.0],
        radius: 2.0,
    };
    let cbf = SrlBarrierFunction::new(config, Box::new(barrier));
    let state = vec![0.5, 0.5];
    let u = vec![0.1, 0.0];
    let u_safe = cbf
        .compute_safe_control(&state, &u)
        .expect("should compute");
    assert_eq!(u_safe.len(), 2);
}

// ── §9 Metrics tests ────────────────────────────────────────────────

#[test]
fn test_srl_metrics_empty() {
    let metrics = compute_srl_metrics(&[]);
    assert_eq!(metrics.num_episodes, 0);
    assert_eq!(metrics.constraint_violation_rate, 0.0);
}

#[test]
fn test_srl_metrics_basic() {
    let episodes = vec![
        SrlEpisodeData {
            rewards: vec![1.0, 1.0, 1.0, 1.0, 1.0],
            costs: vec![0.0, 0.0, 0.0, 0.0, 0.0],
            violations: vec![false, false, false, false, false],
        },
        SrlEpisodeData {
            rewards: vec![1.0, 0.0, 1.0, 0.0, 1.0],
            costs: vec![0.0, 1.0, 0.0, 1.0, 0.0],
            violations: vec![false, true, false, true, false],
        },
    ];

    let metrics = compute_srl_metrics(&episodes);
    assert_eq!(metrics.num_episodes, 2);
    assert_eq!(metrics.total_steps, 10);
    // 2 violations in 10 steps
    assert!((metrics.constraint_violation_rate - 0.2).abs() < 1e-10);
    // Average reward: (5 + 3) / 2 = 4.0
    assert!((metrics.average_reward - 4.0).abs() < 1e-10);
    // Safety adjusted: 4.0 * (1 - 0.2) = 3.2
    assert!((metrics.safety_adjusted_return - 3.2).abs() < 1e-10);
}

#[test]
fn test_srl_metrics_all_violated() {
    let episodes = vec![SrlEpisodeData {
        rewards: vec![0.0; 5],
        costs: vec![1.0; 5],
        violations: vec![true; 5],
    }];
    let metrics = compute_srl_metrics(&episodes);
    assert!((metrics.constraint_violation_rate - 1.0).abs() < 1e-10);
    assert_eq!(metrics.safety_adjusted_return, 0.0);
}

#[test]
fn test_pareto_front_basic() {
    let points = vec![(1.0, 0.5), (2.0, 0.3), (0.5, 0.9), (1.5, 0.6), (3.0, 0.1)];
    let front = compute_srl_pareto_front(&points);
    // Front should include points not dominated
    assert!(!front.is_empty());
    // First point should have highest reward
    assert!((front[0].0 - 3.0).abs() < 1e-10);
}

#[test]
fn test_pareto_front_empty() {
    let front = compute_srl_pareto_front(&[]);
    assert!(front.is_empty());
}

#[test]
fn test_build_srl_report() {
    let episodes = vec![
        SrlEpisodeData {
            rewards: vec![1.0, 1.0, 1.0],
            costs: vec![0.0, 0.0, 0.0],
            violations: vec![false, false, false],
        },
        SrlEpisodeData {
            rewards: vec![0.5, 0.5],
            costs: vec![1.0, 0.0],
            violations: vec![true, false],
        },
    ];
    let report = build_srl_report(&episodes);
    assert_eq!(report.metrics.num_episodes, 2);
    assert!(!report.pareto_front.is_empty());
}

#[test]
fn test_srl_cost_rate() {
    let episodes = vec![
        SrlEpisodeData {
            rewards: vec![1.0; 10],
            costs: vec![0.0; 10],
            violations: vec![false; 10],
        },
        SrlEpisodeData {
            rewards: vec![1.0; 10],
            costs: vec![1.0; 10],
            violations: (0..10).map(|i| i % 2 == 0).collect(),
        },
    ];
    let metrics = compute_srl_metrics(&episodes);
    // 5 violations in 2 episodes = 2.5 violations per episode
    assert!((metrics.cost_rate - 2.5).abs() < 1e-10);
}

// ── Integration-ish tests ───────────────────────────────────────────

#[test]
fn test_lagrangian_with_grid_world() {
    let mut env = SrlSafeGridWorld::new(4, vec![(1, 1)]);
    let config = SrlLagrangianConfig {
        num_constraints: 1,
        cost_thresholds: vec![5.0],
        ..SrlLagrangianConfig::default()
    };
    let agent = SrlLagrangianRl::new(config, env.state_dim(), env.action_dim(), 42);

    let state = env.reset(42).expect("reset should succeed");
    let action = agent.select_action(&state, 0).expect("should select");
    let result = env.step(action).expect("step should succeed");
    assert_eq!(result.costs.len(), 1);
}

#[test]
fn test_safety_layer_with_learned_model() {
    let config = SrlSafetyLayerConfig {
        state_dim: 2,
        action_dim: 2,
        num_constraints: 1,
        lr: 0.1,
        safety_margin: 0.0,
    };
    let mut layer = SrlSafetyLayer::new(config);

    // Train constraint model to predict cost = state[0] * action[0]
    for i in 0..100 {
        let s = vec![(i as f64) * 0.01, 0.5];
        let a = vec![0.5, 0.3];
        let cost = s[0] * a[0];
        layer.update_models(&s, &a, &[cost]).expect("should update");
    }

    // Now test correction
    let state = vec![2.0, 0.5];
    let action = vec![1.0, 0.5];
    let corrected = layer
        .correct_action(&state, &action)
        .expect("should correct");
    assert_eq!(corrected.len(), 2);
}
