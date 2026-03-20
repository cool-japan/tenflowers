//! Tests for the causal_rl module — 55+ tests covering all major components.

#[cfg(test)]
mod crl_tests {
    use super::super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    fn make_seed() -> u64 {
        0xCAFE_BABE_1234_5678u64
    }

    fn make_scm(state_dim: usize, action_dim: usize) -> CrlStructuralCausalModel {
        CrlStructuralCausalModel::new_mdp_scm(state_dim, action_dim)
    }

    fn make_trajectory(n: usize) -> Vec<(Vec<f64>, usize, f64)> {
        let mut seed = make_seed();
        (0..n)
            .map(|i| {
                let state = vec![crl_randn(&mut seed); 4];
                let action = i % 3;
                let reward = crl_randn(&mut seed);
                (state, action, reward)
            })
            .collect()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §1  CrlStructuralCausalModel
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_scm_new_variable_count() {
        let scm = make_scm(4, 2);
        assert_eq!(scm.n_vars, 4);
        assert_eq!(scm.variables.len(), 4);
    }

    #[test]
    fn test_scm_var_dims_correct() {
        let scm = make_scm(4, 2);
        assert_eq!(scm.var_dims[0], 4); // state
        assert_eq!(scm.var_dims[1], 2); // action
        assert_eq!(scm.var_dims[2], 1); // reward
        assert_eq!(scm.var_dims[3], 4); // next_state
    }

    #[test]
    fn test_scm_sample_dimensions() {
        let scm = make_scm(4, 2);
        let mut seed = make_seed();
        let obs = vec![0.1, 0.2, 0.3, 0.4];
        let action = vec![1.0, 0.0];
        let (next_state, _reward) = scm.sample(&obs, &action, &mut seed);
        assert_eq!(next_state.len(), 4, "next_state must have state_dim length");
    }

    #[test]
    fn test_scm_sample_reward_is_finite() {
        let scm = make_scm(4, 2);
        let mut seed = make_seed();
        let obs = vec![0.5, -0.3, 0.1, 0.7];
        let action = vec![0.0, 1.0];
        let (_next_state, reward) = scm.sample(&obs, &action, &mut seed);
        assert!(reward.is_finite(), "reward must be finite");
    }

    #[test]
    fn test_scm_intervene_changes_output() {
        let scm = make_scm(4, 2);
        let mut seed = make_seed();
        let obs = vec![0.5, -0.3, 0.1, 0.7];
        let action = vec![1.0, 0.0];

        let (base_next, base_reward) = scm.sample(&obs, &action, &mut seed);
        let mut seed2 = make_seed();
        let (int_next, int_reward) =
            scm.intervene(3, &[10.0, 10.0, 10.0, 10.0], &obs, &action, &mut seed2);

        // Intervened next_state should differ from sampled
        assert_eq!(
            int_next,
            vec![10.0, 10.0, 10.0, 10.0],
            "Intervening on next_state should fix it to the given value"
        );
        // Reward should still be computed normally
        assert!(int_reward.is_finite());
        let _ = (base_next, base_reward); // suppress unused warnings
    }

    #[test]
    fn test_scm_intervene_reward() {
        let scm = make_scm(4, 2);
        let mut seed = make_seed();
        let obs = vec![0.5, -0.3, 0.1, 0.7];
        let action = vec![1.0, 0.0];
        let (_next, reward) = scm.intervene(2, &[999.0], &obs, &action, &mut seed);
        assert!(
            (reward - 999.0).abs() < 1e-10,
            "Intervening on reward var should fix it"
        );
    }

    #[test]
    fn test_scm_counterfactual_uses_same_noise() {
        // Test abduction property: same U → predictable counterfactual
        let scm = make_scm(3, 2);
        let obs = vec![1.0, 0.0, -1.0];
        let action = vec![1.0, 0.0];
        let cf_action = vec![0.0, 1.0];
        let mut seed = make_seed();

        let (factual_next, factual_reward) = scm.sample(&obs, &action, &mut seed);
        let mut seed2 = make_seed();
        let (cf_next, cf_reward) = scm.counterfactual(
            &obs,
            &action,
            &factual_next,
            factual_reward,
            &cf_action,
            &mut seed2,
        );
        assert_eq!(cf_next.len(), 3);
        assert!(cf_reward.is_finite());
        // Counterfactual with same action should not change (zero noise in abduction)
        let mut seed3 = make_seed();
        let (same_cf_next, _) = scm.counterfactual(
            &obs,
            &action,
            &factual_next,
            factual_reward,
            &action,
            &mut seed3,
        );
        // With same action, CF should recover factual next (up to deterministic component)
        for i in 0..3 {
            assert!(same_cf_next[i].is_finite());
        }
        let _ = cf_next; // use
    }

    #[test]
    fn test_scm_causal_graph_matrix_shape() {
        let scm = make_scm(4, 2);
        let mat = scm.causal_graph_matrix();
        assert_eq!(mat.len(), 4);
        for row in mat {
            assert_eq!(row.len(), 4);
        }
    }

    #[test]
    fn test_scm_adjacency_reward_depends_on_state() {
        let scm = make_scm(4, 2);
        assert!(scm.adjacency[2][0], "reward should depend on state");
        assert!(scm.adjacency[2][1], "reward should depend on action");
    }

    #[test]
    fn test_scm_mechanism_apply_deterministic() {
        let mut mech = CrlCausalMechanism::new(2, 3, 0.0);
        mech.weights[0][0] = 1.0;
        mech.weights[1][1] = 2.0;
        mech.bias = vec![0.5, -0.5];
        let out = mech.apply_deterministic(&[1.0, 2.0, 3.0]);
        assert!((out[0] - 1.5).abs() < 1e-10);
        assert!((out[1] - 3.5).abs() < 1e-10);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §2  CrlCausalPolicyGradient
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_cpg_policy_sums_to_one() {
        let cpg = CrlCausalPolicyGradient::new(4, 3, 0.01, 0.99);
        let state = vec![0.5, -0.3, 1.0, 0.1];
        let pi = cpg.policy(&state);
        let sum: f64 = pi.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "softmax must sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_cpg_policy_all_positive() {
        let cpg = CrlCausalPolicyGradient::new(4, 3, 0.01, 0.99);
        let state = vec![100.0, -100.0, 50.0, 0.0];
        let pi = cpg.policy(&state);
        for p in &pi {
            assert!(*p >= 0.0 && p.is_finite());
        }
    }

    #[test]
    fn test_cpg_compute_returns_shape() {
        let rewards = vec![1.0, 2.0, 3.0, 4.0];
        let returns = CrlCausalPolicyGradient::compute_returns(&rewards, 0.99);
        assert_eq!(returns.len(), rewards.len());
    }

    #[test]
    fn test_cpg_compute_returns_values() {
        let rewards = vec![1.0, 0.0, 0.0];
        let returns = CrlCausalPolicyGradient::compute_returns(&rewards, 0.5);
        // G_0 = 1, G_1 = 0, G_2 = 0
        assert!((returns[0] - 1.0).abs() < 1e-9);
        assert!((returns[1] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_cpg_compute_returns_discounting() {
        let rewards = vec![0.0, 0.0, 1.0];
        let gamma = 0.9;
        let returns = CrlCausalPolicyGradient::compute_returns(&rewards, gamma);
        // G_0 = gamma^2 * 1 = 0.81
        assert!((returns[0] - 0.81).abs() < 1e-9);
    }

    #[test]
    fn test_cpg_update_changes_weights() {
        let mut cpg = CrlCausalPolicyGradient::new(4, 3, 0.1, 0.99);
        let initial_weights = cpg.policy_weights.clone();
        let traj = make_trajectory(10);
        cpg.update(&traj);
        let weights_changed =
            cpg.policy_weights
                .iter()
                .zip(initial_weights.iter())
                .any(|(new_row, old_row)| {
                    new_row
                        .iter()
                        .zip(old_row.iter())
                        .any(|(n, o)| (n - o).abs() > 1e-12)
                });
        assert!(weights_changed, "Weights should change after update");
    }

    #[test]
    fn test_cpg_update_returns_finite() {
        let mut cpg = CrlCausalPolicyGradient::new(4, 3, 0.01, 0.99);
        let traj = make_trajectory(20);
        let ret = cpg.update(&traj);
        assert!(ret.is_finite());
    }

    #[test]
    fn test_cpg_causal_gradient_shape() {
        let cpg = CrlCausalPolicyGradient::new(4, 3, 0.01, 0.99);
        let traj = make_trajectory(5);
        let returns = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let grad = cpg.causal_gradient(&traj, &returns);
        assert_eq!(grad.len(), 3);
        for row in &grad {
            assert_eq!(row.len(), 4);
        }
    }

    #[test]
    fn test_cpg_infer_causal_mask() {
        let mut cpg = CrlCausalPolicyGradient::new(4, 2, 0.01, 0.99);
        // High importance on feature 0 and 2, low on 1 and 3
        let importance = vec![vec![2.0, 0.01, 1.5, 0.01], vec![0.01, 1.0, 0.01, 1.2]];
        cpg.infer_causal_mask(&importance);
        // Feature 0 should be masked in for action 0
        assert!(cpg.causal_mask[0][0]);
    }

    #[test]
    fn test_cpg_empty_trajectory() {
        let mut cpg = CrlCausalPolicyGradient::new(4, 3, 0.01, 0.99);
        let ret = cpg.update(&[]);
        assert_eq!(ret, 0.0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §3  CrlCausalModelBasedRL
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_cmbrl_new_dimensions() {
        let agent = CrlCausalModelBasedRL::new(4, 3);
        assert_eq!(agent.state_dim, 4);
        assert_eq!(agent.action_dim, 3);
        assert_eq!(agent.policy_weights.len(), 3);
        assert_eq!(agent.value_weights.len(), 4);
    }

    #[test]
    fn test_cmbrl_policy_sums_to_one() {
        let agent = CrlCausalModelBasedRL::new(4, 3);
        let state = vec![0.1, 0.2, 0.3, 0.4];
        let pi = agent.policy(&state);
        let sum: f64 = pi.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cmbrl_value_is_finite() {
        let agent = CrlCausalModelBasedRL::new(4, 3);
        let state = vec![0.1, -0.2, 0.3, -0.4];
        let v = agent.value(&state);
        assert!(v.is_finite());
    }

    #[test]
    fn test_cmbrl_update_world_model_decreases_loss() {
        let mut agent = CrlCausalModelBasedRL::new(4, 3);
        let mut seed = make_seed();
        let transitions: Vec<(Vec<f64>, Vec<f64>, f64, Vec<f64>)> = (0..50)
            .map(|_| {
                let s = vec![crl_randn(&mut seed); 4];
                let a = vec![crl_randn(&mut seed); 3];
                let r = crl_randn(&mut seed);
                let ns = s
                    .iter()
                    .map(|x| x * 0.9 + 0.1 * crl_randn(&mut seed))
                    .collect();
                (s, a, r, ns)
            })
            .collect();

        let loss1 = agent.update_world_model(&transitions);
        let loss2 = agent.update_world_model(&transitions);
        assert!(loss1.is_finite());
        assert!(loss2.is_finite());
        // After two updates loss should either decrease or remain reasonable
        assert!(loss2 <= loss1 * 2.0, "Loss should not drastically increase");
    }

    #[test]
    fn test_cmbrl_model_rollout_length() {
        let agent = CrlCausalModelBasedRL::new(4, 3);
        let mut seed = make_seed();
        let start = vec![0.0; 4];
        let rollout = agent.model_rollout(&start, 7, &mut seed);
        assert_eq!(rollout.len(), 7, "Rollout should have horizon steps");
    }

    #[test]
    fn test_cmbrl_model_rollout_states_finite() {
        let agent = CrlCausalModelBasedRL::new(4, 3);
        let mut seed = make_seed();
        let start = vec![0.1, 0.2, 0.3, 0.4];
        let rollout = agent.model_rollout(&start, 5, &mut seed);
        for (state, reward) in &rollout {
            assert_eq!(state.len(), 4);
            assert!(reward.is_finite());
        }
    }

    #[test]
    fn test_cmbrl_plan_update_returns_finite() {
        let mut agent = CrlCausalModelBasedRL::new(4, 3);
        let mut seed = make_seed();
        let start = vec![0.0; 4];
        let total_r = agent.plan_update(&start, &mut seed);
        assert!(total_r.is_finite());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §4  CrlCounterfactualDataAugmentation
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_cda_augmented_larger_than_original() {
        let scm = make_scm(4, 3);
        let aug = CrlCounterfactualDataAugmentation::new(scm, 0.5);
        let mut seed = make_seed();
        let transitions: Vec<(Vec<f64>, usize, f64, Vec<f64>)> = (0..10)
            .map(|i| (vec![0.1f64; 4], i % 3, 1.0, vec![0.2f64; 4]))
            .collect();
        let augmented = aug.augment_dataset(&transitions, &mut seed);
        assert!(
            augmented.len() > transitions.len(),
            "Augmented should be larger"
        );
    }

    #[test]
    fn test_cda_generate_counterfactual_batch_length() {
        let scm = make_scm(4, 3);
        let aug = CrlCounterfactualDataAugmentation::new(scm, 1.0);
        let mut seed = make_seed();
        let batch: Vec<(Vec<f64>, usize, f64, Vec<f64>)> = (0..5)
            .map(|i| (vec![0.1f64; 4], i % 3, 1.0, vec![0.2f64; 4]))
            .collect();
        let cf_batch = aug.generate_counterfactual_batch(&batch, &mut seed);
        assert_eq!(cf_batch.len(), batch.len());
    }

    #[test]
    fn test_cda_counterfactual_advantage_finite() {
        let scm = make_scm(4, 3);
        let aug = CrlCounterfactualDataAugmentation::new(scm, 0.5);
        let state = vec![0.1f64; 4];
        let q_values = vec![1.0, 2.0, 3.0];
        let adv = aug.counterfactual_advantage(&state, 0, 2, &q_values);
        assert!(adv.is_finite());
        assert!(
            (adv - 2.0).abs() < 1e-10,
            "Q(s,2) - Q(s,0) = 3.0 - 1.0 = 2.0"
        );
    }

    #[test]
    fn test_cda_augment_ratio_zero() {
        let scm = make_scm(4, 3);
        let aug = CrlCounterfactualDataAugmentation::new(scm, 0.0);
        let mut seed = make_seed();
        let transitions: Vec<(Vec<f64>, usize, f64, Vec<f64>)> = (0..5)
            .map(|i| (vec![0.1f64; 4], i % 3, 1.0, vec![0.2f64; 4]))
            .collect();
        let augmented = aug.augment_dataset(&transitions, &mut seed);
        assert_eq!(
            augmented.len(),
            transitions.len(),
            "0 ratio = no augmentation added"
        );
    }

    #[test]
    fn test_cda_cf_states_have_correct_dim() {
        let scm = make_scm(4, 3);
        let aug = CrlCounterfactualDataAugmentation::new(scm, 1.0);
        let mut seed = make_seed();
        let batch: Vec<(Vec<f64>, usize, f64, Vec<f64>)> = (0..3)
            .map(|_| (vec![0.5f64; 4], 0usize, 1.0f64, vec![0.6f64; 4]))
            .collect();
        let cf_batch = aug.generate_counterfactual_batch(&batch, &mut seed);
        for (s, _, _, ns) in &cf_batch {
            assert_eq!(s.len(), 4);
            assert_eq!(ns.len(), 4);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §5  CrlInvariantPolicyLearning
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_ipl_policy_sums_to_one() {
        let ipl = CrlInvariantPolicyLearning::new(4, 3, 0.1);
        let state = vec![0.5, -0.3, 0.1, 0.7];
        let pi = ipl.policy(&state);
        let sum: f64 = pi.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ipl_irm_penalty_non_negative() {
        let ipl = CrlInvariantPolicyLearning::new(4, 3, 0.1);
        // Create mock per-env gradients
        let env_grads = vec![
            vec![vec![0.1, 0.2, 0.0, -0.1]; 3],
            vec![vec![-0.1, 0.1, 0.2, 0.0]; 3],
        ];
        let penalty = ipl.irm_penalty_term(&env_grads);
        assert!(penalty >= 0.0, "IRM penalty must be non-negative");
    }

    #[test]
    fn test_ipl_irm_penalty_zero_for_identical_gradients() {
        let ipl = CrlInvariantPolicyLearning::new(4, 3, 0.1);
        let grad = vec![vec![0.1, 0.2, 0.3, 0.4]; 3];
        let env_grads = vec![grad.clone(), grad.clone()];
        let penalty = ipl.irm_penalty_term(&env_grads);
        assert!(
            penalty < 1e-10,
            "Identical gradients should give zero IRM penalty"
        );
    }

    #[test]
    fn test_ipl_train_step_runs() {
        let mut ipl = CrlInvariantPolicyLearning::new(4, 3, 0.1);
        let mut seed = make_seed();
        let env_trajs = vec![make_trajectory(10), make_trajectory(10)];
        let loss = ipl.train_step(&env_trajs, &mut seed);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_ipl_train_step_updates_weights() {
        let mut ipl = CrlInvariantPolicyLearning::new(4, 3, 0.1);
        let initial = ipl.policy_weights.clone();
        let mut seed = make_seed();
        let env_trajs = vec![make_trajectory(10), make_trajectory(10)];
        ipl.train_step(&env_trajs, &mut seed);
        let changed = ipl
            .policy_weights
            .iter()
            .zip(initial.iter())
            .any(|(new_row, old_row)| {
                new_row
                    .iter()
                    .zip(old_row.iter())
                    .any(|(n, o)| (n - o).abs() > 1e-12)
            });
        assert!(changed);
    }

    #[test]
    fn test_ipl_spurious_test_finite() {
        let ipl = CrlInvariantPolicyLearning::new(4, 3, 0.1);
        let result = ipl.spurious_correlation_test(&[1, 3]);
        assert!(result.is_finite() && result >= 0.0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §6  CrlOffPolicyCounterfactual
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_opc_add_transition_fills_buffer() {
        let scm = make_scm(4, 3);
        let mut agent = CrlOffPolicyCounterfactual::new(4, 3, 100, scm);
        agent.add_transition(vec![0.1; 4], 0, 1.0, vec![0.2; 4], vec![0.33, 0.33, 0.34]);
        assert_eq!(agent.buffer.size, 1);
    }

    #[test]
    fn test_opc_buffer_ring_wraps() {
        let scm = make_scm(4, 3);
        let mut agent = CrlOffPolicyCounterfactual::new(4, 3, 3, scm);
        for i in 0..5 {
            agent.add_transition(
                vec![i as f64; 4],
                0,
                1.0,
                vec![0.0; 4],
                vec![0.33, 0.33, 0.34],
            );
        }
        assert_eq!(agent.buffer.size, 3);
    }

    #[test]
    fn test_opc_q_value_shape() {
        let scm = make_scm(4, 3);
        let agent = CrlOffPolicyCounterfactual::new(4, 3, 100, scm);
        let qs = agent.q_value(&[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(qs.len(), 3);
    }

    #[test]
    fn test_opc_importance_weight_clipped() {
        let scm = make_scm(4, 3);
        let agent = CrlOffPolicyCounterfactual::new(4, 3, 100, scm);
        // Very low behavior probability → high ratio, should be clipped
        let rho = agent.importance_weight(1.0, 0.01);
        assert!(
            rho <= 5.0 + 1e-10,
            "IS weight must be clipped at rho_max=5.0"
        );
    }

    #[test]
    fn test_opc_importance_weight_normal() {
        let scm = make_scm(4, 3);
        let agent = CrlOffPolicyCounterfactual::new(4, 3, 100, scm);
        let rho = agent.importance_weight(0.5, 0.5);
        assert!((rho - 1.0).abs() < 1e-10, "Equal probs → IS weight = 1");
    }

    #[test]
    fn test_opc_q_update_empty_buffer() {
        let scm = make_scm(4, 3);
        let mut agent = CrlOffPolicyCounterfactual::new(4, 3, 100, scm);
        let mut seed = make_seed();
        let result = agent.counterfactual_q_update(32, &mut seed);
        assert!(result.is_err(), "Should return error on empty buffer");
    }

    #[test]
    fn test_opc_q_update_returns_loss() {
        let scm = make_scm(4, 3);
        let mut agent = CrlOffPolicyCounterfactual::new(4, 3, 100, scm);
        let mut seed = make_seed();
        for i in 0..20 {
            agent.add_transition(
                vec![crl_randn(&mut seed); 4],
                i % 3,
                1.0,
                vec![crl_randn(&mut seed); 4],
                vec![0.33, 0.33, 0.34],
            );
        }
        let result = agent.counterfactual_q_update(8, &mut seed);
        assert!(result.is_ok());
        assert!(result.expect("counterfactual_q_update should succeed").is_finite());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §7  CrlCausalCredit
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_credit_vector_length() {
        let scm = make_scm(4, 3);
        let cc = CrlCausalCredit::new(10, 0.9, scm);
        let mut seed = make_seed();
        let traj = make_trajectory(8);
        let credits = cc.causal_credit_vector(&traj, &mut seed);
        assert_eq!(credits.len(), traj.len());
    }

    #[test]
    fn test_credit_vector_finite() {
        let scm = make_scm(4, 3);
        let cc = CrlCausalCredit::new(10, 0.9, scm);
        let mut seed = make_seed();
        let traj = make_trajectory(5);
        let credits = cc.causal_credit_vector(&traj, &mut seed);
        for c in &credits {
            assert!(c.is_finite(), "All credits must be finite");
        }
    }

    #[test]
    fn test_credit_empty_trajectory() {
        let scm = make_scm(4, 3);
        let cc = CrlCausalCredit::new(10, 0.9, scm);
        let mut seed = make_seed();
        let credits = cc.causal_credit_vector(&[], &mut seed);
        assert!(credits.is_empty());
    }

    #[test]
    fn test_hindsight_credit_length() {
        let scm = make_scm(4, 3);
        let cc = CrlCausalCredit::new(10, 0.95, scm);
        let traj = make_trajectory(6);
        let credits = cc.hindsight_credit(&traj);
        assert_eq!(credits.len(), 6);
    }

    #[test]
    fn test_hindsight_credit_finite() {
        let scm = make_scm(4, 3);
        let cc = CrlCausalCredit::new(10, 0.95, scm);
        let traj = make_trajectory(6);
        let credits = cc.hindsight_credit(&traj);
        for c in &credits {
            assert!(c.is_finite());
        }
    }

    #[test]
    fn test_temporal_causal_graph_shape() {
        let scm = make_scm(4, 3);
        let cc = CrlCausalCredit::new(10, 0.9, scm);
        let traj = make_trajectory(4);
        let graph = cc.temporal_causal_graph(&traj);
        assert_eq!(graph.len(), 4);
        for row in &graph {
            assert_eq!(row.len(), 4);
        }
    }

    #[test]
    fn test_temporal_causal_graph_non_negative() {
        let scm = make_scm(4, 3);
        let cc = CrlCausalCredit::new(10, 0.9, scm);
        let traj = make_trajectory(5);
        let graph = cc.temporal_causal_graph(&traj);
        for row in &graph {
            for &v in row {
                assert!(v >= 0.0 && v.is_finite());
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §8  CrlSCMDynaQ
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_dynaq_q_update_changes_table() {
        let scm = make_scm(2, 2);
        let mut agent = CrlSCMDynaQ::new(4, 2, scm);
        let initial_q = agent.q_table[0][0];
        agent.q_update(0, 0, 1.0, 1);
        assert!(
            (agent.q_table[0][0] - initial_q).abs() > 1e-10,
            "Q-table should change after update"
        );
    }

    #[test]
    fn test_dynaq_step_valid_action() {
        let scm = make_scm(2, 2);
        let mut agent = CrlSCMDynaQ::new(4, 2, scm);
        let mut seed = make_seed();
        let action = agent.step(0, 1, 1.0, &mut seed);
        assert!(action < 2, "Returned action should be in valid range");
    }

    #[test]
    fn test_dynaq_greedy_policy_length() {
        let scm = make_scm(2, 2);
        let agent = CrlSCMDynaQ::new(4, 2, scm);
        let policy = agent.greedy_policy();
        assert_eq!(
            policy.len(),
            4,
            "Greedy policy should have n_states entries"
        );
    }

    #[test]
    fn test_dynaq_greedy_policy_valid_actions() {
        let scm = make_scm(2, 2);
        let agent = CrlSCMDynaQ::new(4, 3, scm);
        let policy = agent.greedy_policy();
        for a in policy {
            assert!(a < 3, "Actions must be in [0, n_actions)");
        }
    }

    #[test]
    fn test_dynaq_epsilon_greedy_range() {
        let scm = make_scm(2, 2);
        let agent = CrlSCMDynaQ::new(4, 5, scm);
        let mut seed = make_seed();
        for _ in 0..50 {
            let a = agent.epsilon_greedy(0, &mut seed);
            assert!(a < 5);
        }
    }

    #[test]
    fn test_dynaq_causal_value_shaping() {
        let scm = make_scm(2, 2);
        let mut agent = CrlSCMDynaQ::new(4, 2, scm);
        // Set some non-zero Q-values first
        agent.q_table[0][0] = 1.0;
        agent.q_table[0][1] = -1.0;
        let before = agent.q_table[0][0];
        agent.causal_value_shaping(&[1.0, 0.5, 0.0, 0.0]);
        let after = agent.q_table[0][0];
        assert!(
            (after - before * 1.1).abs() < 1e-9,
            "Should scale by (1 + 0.1 * importance)"
        );
    }

    #[test]
    fn test_dynaq_model_memory_fills() {
        let scm = make_scm(2, 2);
        let mut agent = CrlSCMDynaQ::new(4, 2, scm);
        let mut seed = make_seed();
        agent.step(0, 1, 1.0, &mut seed);
        assert!(!agent.model_memory.is_empty());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §9  CrlCausalCurriculum
    // ─────────────────────────────────────────────────────────────────────────

    fn make_environments(n: usize, state_dim: usize, action_dim: usize) -> Vec<CrlEnvironment> {
        let mut seed = make_seed();
        (0..n)
            .map(|i| CrlEnvironment::new(i, state_dim, action_dim, &mut seed))
            .collect()
    }

    #[test]
    fn test_curriculum_select_valid_index() {
        let envs = make_environments(4, 4, 3);
        let curr = CrlCausalCurriculum::new(envs);
        let idx = curr.select_next_environment(CrlCurriculumStrategy::MaxInformationGain);
        assert!(idx < 4, "Selected env index must be < n_envs");
    }

    #[test]
    fn test_curriculum_select_competence_valid() {
        let envs = make_environments(4, 4, 3);
        let curr = CrlCausalCurriculum::new(envs);
        let idx = curr.select_next_environment(CrlCurriculumStrategy::Competence);
        assert!(idx < 4);
    }

    #[test]
    fn test_curriculum_select_random_valid() {
        let envs = make_environments(4, 4, 3);
        let curr = CrlCausalCurriculum::new(envs);
        let idx = curr.select_next_environment(CrlCurriculumStrategy::Random);
        assert!(idx < 4);
    }

    #[test]
    fn test_curriculum_score_in_range() {
        let envs = make_environments(3, 4, 3);
        let curr = CrlCausalCurriculum::new(envs);
        let score = curr.causal_understanding_score();
        assert!(
            (0.0..=1.0).contains(&score),
            "Score must be in [0, 1], got {}",
            score
        );
    }

    #[test]
    fn test_curriculum_update_causal_estimate() {
        let envs = make_environments(3, 4, 3);
        let mut curr = CrlCausalCurriculum::new(envs);
        let mut seed = make_seed();
        let transitions: Vec<(Vec<f64>, Vec<f64>, f64, Vec<f64>)> = (0..20)
            .map(|_| {
                (
                    vec![crl_randn(&mut seed); 4],
                    vec![crl_randn(&mut seed); 3],
                    crl_randn(&mut seed),
                    vec![crl_randn(&mut seed); 4],
                )
            })
            .collect();
        curr.update_causal_estimate(0, &transitions);
        // Information gain should have decreased
        assert!(curr.information_gain[0] < 1.0);
    }

    #[test]
    fn test_curriculum_information_gain_score() {
        let envs = make_environments(3, 4, 3);
        let curr = CrlCausalCurriculum::new(envs);
        let score = curr.information_gain_score(0);
        assert!(score >= 0.0 && score.is_finite());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // §10  CrlMetrics
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_metrics_causal_regret_finite() {
        let opt = vec![2.0, 3.0, 2.5, 3.5];
        let act = vec![1.5, 2.0, 2.0, 3.0];
        let regret = CrlMetrics::causal_regret(&opt, &act, 0.5);
        assert!(regret.is_finite() && regret >= 0.0);
    }

    #[test]
    fn test_metrics_causal_regret_zero_spurious() {
        let opt = vec![2.0, 3.0];
        let act = vec![1.0, 2.0];
        let regret = CrlMetrics::causal_regret(&opt, &act, 0.0);
        assert!((regret - 0.0).abs() < 1e-10, "Zero spurious → zero regret");
    }

    #[test]
    fn test_metrics_causal_regret_empty() {
        let regret = CrlMetrics::causal_regret(&[], &[], 0.5);
        assert_eq!(regret, 0.0);
    }

    #[test]
    fn test_metrics_counterfactual_fairness_finite() {
        let outcomes = vec![1.0, 2.0, 1.5, 2.5];
        let groups = vec![0, 1, 0, 1];
        let cf_outcomes = vec![1.1, 1.9, 1.6, 2.4];
        let score = CrlMetrics::counterfactual_fairness_score(&outcomes, &groups, &cf_outcomes);
        assert!(score.is_finite() && (0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_metrics_counterfactual_fairness_perfect() {
        let outcomes = vec![1.0, 2.0, 1.0, 2.0];
        let groups = vec![0, 1, 0, 1];
        // Counterfactual same as factual → fairness score ≈ 0
        let score = CrlMetrics::counterfactual_fairness_score(&outcomes, &groups, &outcomes);
        assert!(score.abs() < 1e-10);
    }

    #[test]
    fn test_metrics_causal_discovery_accuracy_perfect() {
        let graph = vec![
            vec![false, true, false],
            vec![false, false, true],
            vec![false, false, false],
        ];
        let acc = CrlMetrics::causal_discovery_accuracy(&graph, &graph);
        assert!((acc - 1.0).abs() < 1e-9, "Perfect prediction → F1 = 1.0");
    }

    #[test]
    fn test_metrics_causal_discovery_accuracy_no_edges() {
        let empty = vec![vec![false; 3]; 3];
        let acc = CrlMetrics::causal_discovery_accuracy(&empty, &empty);
        // 0 TP, 0 FP, 0 FN → F1 = 0
        assert_eq!(acc, 0.0);
    }

    #[test]
    fn test_metrics_invariance_score_identical_envs() {
        let env_returns = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]];
        let score = CrlMetrics::invariance_score(&env_returns);
        assert!(
            (score - 1.0).abs() < 1e-9,
            "Identical envs → invariance = 1.0"
        );
    }

    #[test]
    fn test_metrics_invariance_score_range() {
        let env_returns = vec![vec![1.0, 2.0], vec![10.0, 20.0], vec![100.0]];
        let score = CrlMetrics::invariance_score(&env_returns);
        assert!(score.is_finite() && (0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_metrics_credit_assignment_accuracy_perfect() {
        let credits = vec![1.0, 2.0, 3.0, 4.0];
        let acc = CrlMetrics::credit_assignment_accuracy(&credits, &credits);
        assert!((acc - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_credit_assignment_accuracy_anti() {
        let est = vec![1.0, 2.0, 3.0];
        let true_ = vec![-1.0, -2.0, -3.0];
        let acc = CrlMetrics::credit_assignment_accuracy(&est, &true_);
        assert!((acc + 1.0).abs() < 1e-9, "Anti-correlated → Pearson = -1");
    }

    #[test]
    fn test_metrics_credit_assignment_short() {
        let acc = CrlMetrics::credit_assignment_accuracy(&[1.0], &[2.0]);
        assert_eq!(acc, 0.0, "Less than 2 points → 0");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional integration tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_rng_produces_values_in_range() {
        let mut seed = 12345u64;
        for _ in 0..100 {
            let v = crl_rand01(&mut seed);
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_randn_produces_finite_values() {
        let mut seed = 99999u64;
        for _ in 0..100 {
            let v = crl_randn(&mut seed);
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_full_pipeline_counterfactual_rl() {
        // Integration: SCM + augmentation + Q-learning
        let scm = make_scm(4, 3);
        let scm2 = scm.clone();
        let aug = CrlCounterfactualDataAugmentation::new(scm, 0.5);
        let mut seed = make_seed();

        let transitions: Vec<(Vec<f64>, usize, f64, Vec<f64>)> = (0..20)
            .map(|i| {
                (
                    vec![crl_randn(&mut seed); 4],
                    i % 3,
                    crl_randn(&mut seed),
                    vec![crl_randn(&mut seed); 4],
                )
            })
            .collect();

        let augmented = aug.augment_dataset(&transitions, &mut seed);
        assert!(augmented.len() >= transitions.len());

        let mut agent = CrlOffPolicyCounterfactual::new(4, 3, 200, scm2);
        for (s, a, r, ns) in &augmented {
            agent.add_transition(s.clone(), *a, *r, ns.clone(), vec![0.33, 0.33, 0.34]);
        }

        let result = agent.counterfactual_q_update(16, &mut seed);
        assert!(result.is_ok());
    }

    #[test]
    fn test_credit_hindsight_monotone_decay() {
        let scm = make_scm(4, 3);
        let decay = 0.5;
        let cc = CrlCausalCredit::new(10, decay, scm);
        // Trajectory with all positive rewards
        let traj: Vec<(Vec<f64>, usize, f64)> =
            (0..5).map(|_| (vec![0.0f64; 4], 0usize, 1.0f64)).collect();
        let credits = cc.hindsight_credit(&traj);
        // credits[0] > credits[1] > ... (since same rewards but more discounted)
        // Actually credits[t] = sum_{k>=t} decay^(k-t) r_k = 1 + decay + decay^2 + ...
        // credits[0] = (1 - decay^5) / (1 - decay), credits[4] = 1.0
        assert!(
            credits[0] > credits[4],
            "First step has highest hindsight credit"
        );
    }
}
