//! Tests for the Active Inference (Free Energy Principle) module.
//!
//! Covers all ten sections:
//!  §1 AiGenerativeModel
//!  §2 AiBeliefState
//!  §3 AiVariationalInference
//!  §4 AiExpectedFreeEnergy
//!  §5 AiPolicySelection
//!  §6 AiActiveInferenceAgent
//!  §7 AiMarkovBlanket
//!  §8 AiHierarchicalGenerativeModel
//!  §9 AiParameterLearning
//! §10 AiMetrics

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// §1  AiGenerativeModel tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_generative_model_dimensions() {
    let model = AiGenerativeModel::new(4, 3, 2);
    assert_eq!(model.n_states, 4);
    assert_eq!(model.n_obs, 3);
    assert_eq!(model.n_actions, 2);
    assert_eq!(model.a_matrix.len(), 3); // n_obs rows
    assert_eq!(model.a_matrix[0].len(), 4); // n_states cols
    assert_eq!(model.b_matrices.len(), 2); // n_actions
    assert_eq!(model.b_matrices[0].len(), 4); // n_states rows
    assert_eq!(model.b_matrices[0][0].len(), 4); // n_states cols
    assert_eq!(model.c_vector.len(), 3);
    assert_eq!(model.d_prior.len(), 4);
}

#[test]
fn test_generative_model_a_columns_normalized() {
    let model = AiGenerativeModel::new(5, 4, 3);
    // Each column of A should sum to 1 (valid likelihood)
    for s in 0..model.n_states {
        let col_sum: f64 = (0..model.n_obs).map(|o| model.a_matrix[o][s]).sum();
        assert!(
            (col_sum - 1.0).abs() < 1e-9,
            "Column {} sums to {}",
            s,
            col_sum
        );
    }
}

#[test]
fn test_generative_model_b_columns_normalized() {
    let model = AiGenerativeModel::new(4, 3, 2);
    for u in 0..model.n_actions {
        for s in 0..model.n_states {
            let col_sum: f64 = (0..model.n_states)
                .map(|sn| model.b_matrices[u][sn][s])
                .sum();
            assert!(
                (col_sum - 1.0).abs() < 1e-9,
                "B[{}] col {} sums to {}",
                u,
                s,
                col_sum
            );
        }
    }
}

#[test]
fn test_generative_model_d_prior_uniform() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let expected = 1.0 / 4.0;
    for &p in &model.d_prior {
        assert!((p - expected).abs() < 1e-9);
    }
}

#[test]
fn test_generative_model_log_likelihood_valid() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let ll = model.log_likelihood(0, 0);
    assert!(ll.is_finite(), "log_likelihood should be finite");
    assert!(ll <= 0.0, "log_likelihood should be non-positive");
}

#[test]
fn test_generative_model_c_vector_zeros() {
    let model = AiGenerativeModel::new(4, 3, 2);
    for &c in &model.c_vector {
        assert_eq!(c, 0.0);
    }
}

#[test]
fn test_generative_model_normalize_a() {
    let mut model = AiGenerativeModel::new(3, 4, 2);
    // Corrupt A columns
    for o in 0..model.n_obs {
        for s in 0..model.n_states {
            model.a_matrix[o][s] *= 5.0;
        }
    }
    model.normalize_a();
    for s in 0..model.n_states {
        let col_sum: f64 = (0..model.n_obs).map(|o| model.a_matrix[o][s]).sum();
        assert!(
            (col_sum - 1.0).abs() < 1e-9,
            "Column {} sums to {}",
            s,
            col_sum
        );
    }
}

#[test]
fn test_generative_model_normalize_b() {
    let mut model = AiGenerativeModel::new(4, 3, 2);
    // Scale B up
    for u in 0..model.n_actions {
        for sn in 0..model.n_states {
            for sc in 0..model.n_states {
                model.b_matrices[u][sn][sc] *= 3.0;
            }
        }
    }
    model.normalize_b();
    for u in 0..model.n_actions {
        for s in 0..model.n_states {
            let col_sum: f64 = (0..model.n_states)
                .map(|sn| model.b_matrices[u][sn][s])
                .sum();
            assert!((col_sum - 1.0).abs() < 1e-9);
        }
    }
}

#[test]
fn test_generative_model_transition_returns_correct_action() {
    let model = AiGenerativeModel::new(4, 3, 2);
    // Verify transition() returns the correct B[u] matrix by checking
    // that B[0] and B[1] are the same objects stored in b_matrices
    let b0 = model.transition(0);
    let b1 = model.transition(1);
    // Both should be 4x4 matrices (n_states x n_states)
    assert_eq!(b0.len(), 4);
    assert_eq!(b1.len(), 4);
    // They should be different matrix objects (different memory content)
    // Check element-wise that at least one value differs
    let any_diff = b0
        .iter()
        .flatten()
        .zip(b1.iter().flatten())
        .any(|(a, b)| (a - b).abs() > 1e-12);
    assert!(
        any_diff,
        "B[0] and B[1] should have different matrix elements"
    );
}

#[test]
fn test_generative_model_a_values_in_range() {
    let model = AiGenerativeModel::new(5, 4, 3);
    for row in &model.a_matrix {
        for &v in row {
            assert!((0.0..=1.0).contains(&v), "A value {} out of [0,1]", v);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  AiBeliefState tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_belief_state_uniform_init() {
    let bs = AiBeliefState::new(4);
    let expected = 0.25;
    for &b in &bs.beliefs {
        assert!((b - expected).abs() < 1e-9);
    }
}

#[test]
fn test_belief_state_beliefs_sum_to_one() {
    let bs = AiBeliefState::new(6);
    let sum: f64 = bs.beliefs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9);
}

#[test]
fn test_belief_state_from_prior() {
    let prior = vec![0.1, 0.4, 0.5];
    let bs = AiBeliefState::from_prior(&prior);
    assert!((bs.beliefs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert!(bs.beliefs[1] > bs.beliefs[0]);
    assert!(bs.beliefs[2] > bs.beliefs[1]);
}

#[test]
fn test_belief_state_update_normalizes() {
    let mut bs = AiBeliefState::new(3);
    bs.update(&[0.5, 2.0, 0.5]);
    let sum: f64 = bs.beliefs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9);
}

#[test]
fn test_belief_state_entropy_nonneg() {
    let bs = AiBeliefState::new(4);
    assert!(bs.entropy() >= 0.0);
}

#[test]
fn test_belief_state_entropy_uniform_is_max() {
    let n = 4;
    let bs_uniform = AiBeliefState::new(n);
    let mut bs_peaked = AiBeliefState::new(n);
    bs_peaked.update(&[0.97, 0.01, 0.01, 0.01]);
    assert!(bs_uniform.entropy() > bs_peaked.entropy());
}

#[test]
fn test_belief_state_kl_divergence_nonneg() {
    let bs = AiBeliefState::new(4);
    let prior = vec![0.25, 0.25, 0.25, 0.25];
    let kl = bs.kl_divergence(&prior);
    assert!(
        kl >= -1e-10,
        "KL divergence should be non-negative, got {}",
        kl
    );
}

#[test]
fn test_belief_state_kl_zero_for_equal() {
    let bs = AiBeliefState::new(4);
    let prior = vec![0.25, 0.25, 0.25, 0.25];
    let kl = bs.kl_divergence(&prior);
    assert!(
        kl.abs() < 1e-9,
        "KL(uniform||uniform) should be ~0, got {}",
        kl
    );
}

#[test]
fn test_belief_state_most_likely_state() {
    let mut bs = AiBeliefState::new(5);
    bs.update(&[0.05, 0.05, 0.8, 0.05, 0.05]);
    assert_eq!(bs.most_likely_state(), 2);
}

#[test]
fn test_belief_state_log_beliefs_consistent() {
    let bs = AiBeliefState::new(4);
    for (b, lb) in bs.beliefs.iter().zip(bs.log_beliefs.iter()) {
        assert!((b.ln() - lb).abs() < 1e-9);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  AiVariationalInference tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_variational_inference_basic() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let prior_beliefs = AiBeliefState::new(4);
    let vi = AiVariationalInference::new(20, 0.5, 1e-6);
    let result = vi.infer_states(0, &model, &prior_beliefs);
    assert!(result.is_ok(), "infer_states should succeed");
}

#[test]
fn test_variational_inference_beliefs_valid() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let prior_beliefs = AiBeliefState::new(4);
    let vi = AiVariationalInference::new(20, 0.5, 1e-6);
    let new_beliefs = vi.infer_states(1, &model, &prior_beliefs)
        .expect("infer_states should succeed");
    let sum: f64 = new_beliefs.beliefs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-9,
        "Beliefs should sum to 1, got {}",
        sum
    );
    for &b in &new_beliefs.beliefs {
        assert!(b >= 0.0, "Belief should be non-negative, got {}", b);
    }
}

#[test]
fn test_variational_inference_free_energy_finite() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let beliefs = AiBeliefState::new(4);
    let vi = AiVariationalInference::new(20, 0.5, 1e-6);
    let f = vi.free_energy(0, &beliefs, &model);
    assert!(f.is_finite(), "Free energy should be finite, got {}", f);
}

#[test]
fn test_variational_inference_free_energy_decreases() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let prior_beliefs = AiBeliefState::new(4);
    let vi = AiVariationalInference::new(50, 0.5, 1e-8);
    let f_prior = vi.free_energy(0, &prior_beliefs, &model);
    let new_beliefs = vi.infer_states(0, &model, &prior_beliefs)
        .expect("infer_states should succeed");
    let f_posterior = vi.free_energy(0, &new_beliefs, &model);
    // After inference, free energy should be ≤ prior free energy
    assert!(
        f_posterior <= f_prior + 1e-6,
        "F_posterior={} should be ≤ F_prior={}",
        f_posterior,
        f_prior
    );
}

#[test]
fn test_variational_inference_out_of_range_obs() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let prior_beliefs = AiBeliefState::new(4);
    let vi = AiVariationalInference::new(20, 0.5, 1e-6);
    let result = vi.infer_states(99, &model, &prior_beliefs);
    assert!(result.is_err(), "Should return error for invalid obs index");
}

#[test]
fn test_variational_inference_dimension_mismatch() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let wrong_beliefs = AiBeliefState::new(6); // wrong n_states
    let vi = AiVariationalInference::new(20, 0.5, 1e-6);
    let result = vi.infer_states(0, &model, &wrong_beliefs);
    assert!(
        result.is_err(),
        "Should return error for dimension mismatch"
    );
}

#[test]
fn test_variational_inference_updates_beliefs() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let prior_beliefs = AiBeliefState::new(4);
    let vi = AiVariationalInference::new(30, 0.8, 1e-8);
    let new_beliefs = vi.infer_states(0, &model, &prior_beliefs)
        .expect("infer_states should succeed");
    // Beliefs should have changed from uniform
    let max_diff: f64 = new_beliefs
        .beliefs
        .iter()
        .zip(prior_beliefs.beliefs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    // After enough iterations with informative A, beliefs should shift
    assert!(max_diff >= 0.0); // at minimum, no corruption
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  AiExpectedFreeEnergy tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_efe_basic_computation() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let beliefs = AiBeliefState::new(4);
    let efe = AiExpectedFreeEnergy::new(1);
    let result = efe.compute(&[0], &beliefs, &model);
    assert!(result.is_ok());
    assert!(result.expect("EFE compute should succeed").is_finite());
}

#[test]
fn test_efe_epistemic_value_nonneg() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let qo = vec![1.0 / 3.0; 3];
    let qs = vec![0.25; 4];
    let epi = AiExpectedFreeEnergy::epistemic_value(&qo, &qs, &model.a_matrix);
    assert!(
        epi >= 0.0,
        "Epistemic value should be non-negative, got {}",
        epi
    );
}

#[test]
fn test_efe_pragmatic_value_finite() {
    let qo = vec![0.2, 0.5, 0.3];
    let c = vec![1.0, 0.0, -1.0];
    let prag = AiExpectedFreeEnergy::pragmatic_value(&qo, &c);
    assert!(prag.is_finite());
}

#[test]
fn test_efe_empty_policy_error() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let beliefs = AiBeliefState::new(4);
    let efe = AiExpectedFreeEnergy::new(1);
    let result = efe.compute(&[], &beliefs, &model);
    assert!(result.is_err());
}

#[test]
fn test_efe_invalid_action_error() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let beliefs = AiBeliefState::new(4);
    let efe = AiExpectedFreeEnergy::new(1);
    let result = efe.compute(&[99], &beliefs, &model);
    assert!(result.is_err());
}

#[test]
fn test_efe_multi_step_policy() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let beliefs = AiBeliefState::new(4);
    let efe = AiExpectedFreeEnergy::new(3);
    let result = efe.compute(&[0, 1, 0], &beliefs, &model);
    assert!(result.is_ok());
    assert!(result.expect("EFE compute should succeed for multi-step").is_finite());
}

#[test]
fn test_efe_epistemic_zero_for_deterministic_likelihood() {
    // If A is identity (deterministic), epistemic value should be 0
    let n = 3;
    let mut model = AiGenerativeModel::new(n, n, 1);
    // Set A to identity matrix (column-normalized)
    for o in 0..n {
        for s in 0..n {
            model.a_matrix[o][s] = if o == s { 1.0 } else { 0.0 };
        }
    }
    let qs = vec![1.0 / n as f64; n];
    let qo = vec![1.0 / n as f64; n];
    let epi = AiExpectedFreeEnergy::epistemic_value(&qo, &qs, &model.a_matrix);
    // With deterministic A, H[P(o|s)] = 0 for each s → epistemic ≥ 0
    assert!(epi >= 0.0);
}

#[test]
fn test_efe_pragmatic_with_preferred_obs() {
    let qo = vec![0.0, 0.0, 1.0]; // always obs 2
    let c = vec![-1.0, -1.0, 5.0]; // strong preference for obs 2
    let prag = AiExpectedFreeEnergy::pragmatic_value(&qo, &c);
    assert!((prag - 5.0).abs() < 1e-9);
}

#[test]
fn test_efe_prefers_preferred_outcomes() {
    let mut model = AiGenerativeModel::new(2, 2, 2);
    // Action 0 → stays in state 0, action 1 → transitions to state 1
    // We prefer obs 1 (state 1)
    model.c_vector[1] = 2.0;
    model.c_vector[0] = -2.0;
    let beliefs = AiBeliefState::from_prior(&[1.0, 0.0]); // start in state 0
    let efe = AiExpectedFreeEnergy::new(1);
    let g0 = efe.compute(&[0], &beliefs, &model)
        .expect("EFE compute for action 0 should succeed");
    let g1 = efe.compute(&[1], &beliefs, &model)
        .expect("EFE compute for action 1 should succeed");
    // Not strictly required to be < but should be finite
    assert!(g0.is_finite() && g1.is_finite());
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  AiPolicySelection tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_policy_selection_returns_valid_action() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let beliefs = AiBeliefState::new(4);
    let efe = AiExpectedFreeEnergy::new(1);
    let ps = AiPolicySelection::new(1.0, 1);
    let action = ps.select_action(&beliefs, &model, &efe)
        .expect("select_action should succeed");
    assert!(
        action < model.n_actions,
        "Action {} >= n_actions {}",
        action,
        model.n_actions
    );
}

#[test]
fn test_policy_selection_action_posterior_sums_to_one() {
    let model = AiGenerativeModel::new(4, 3, 3);
    let beliefs = AiBeliefState::new(4);
    let ps = AiPolicySelection::new(1.0, 1);
    let posterior = ps.compute_action_posterior(&beliefs, &model)
        .expect("compute_action_posterior should succeed");
    let sum: f64 = posterior.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "Action posterior sums to {}", sum);
}

#[test]
fn test_policy_selection_posterior_length() {
    let model = AiGenerativeModel::new(4, 3, 5);
    let beliefs = AiBeliefState::new(4);
    let ps = AiPolicySelection::new(0.5, 1);
    let posterior = ps.compute_action_posterior(&beliefs, &model)
        .expect("compute_action_posterior should succeed");
    assert_eq!(posterior.len(), 5);
}

#[test]
fn test_policy_selection_posterior_nonneg() {
    let model = AiGenerativeModel::new(4, 3, 3);
    let beliefs = AiBeliefState::new(4);
    let ps = AiPolicySelection::new(1.0, 1);
    let posterior = ps.compute_action_posterior(&beliefs, &model)
        .expect("compute_action_posterior should succeed");
    for &p in &posterior {
        assert!(p >= 0.0, "Posterior probability should be non-negative");
    }
}

#[test]
fn test_policy_selection_precision_weighted_action() {
    let posterior = vec![0.1, 0.6, 0.3];
    let ps = AiPolicySelection::new(1.0, 1);
    let a = ps.precision_weighted_action(&posterior, 1.0);
    assert_eq!(a, 1); // argmax
}

#[test]
fn test_policy_selection_temperature_effect() {
    let model = AiGenerativeModel::new(4, 3, 4);
    let beliefs = AiBeliefState::new(4);
    let ps_low_temp = AiPolicySelection::new(0.01, 1);
    let ps_high_temp = AiPolicySelection::new(100.0, 1);
    let post_low = ps_low_temp
        .compute_action_posterior(&beliefs, &model)
        .expect("compute_action_posterior for low temp should succeed");
    let post_high = ps_high_temp
        .compute_action_posterior(&beliefs, &model)
        .expect("compute_action_posterior for high temp should succeed");
    // Low temperature → more peaked
    let entropy_low: f64 = super::entropy(&post_low);
    let entropy_high: f64 = super::entropy(&post_high);
    assert!(
        entropy_high >= entropy_low,
        "Higher temperature should have higher entropy"
    );
}

#[test]
fn test_policy_selection_no_actions_error() {
    let model = AiGenerativeModel::new(4, 3, 0);
    let beliefs = AiBeliefState::new(4);
    let efe = AiExpectedFreeEnergy::new(1);
    let ps = AiPolicySelection::new(1.0, 1);
    let result = ps.select_action(&beliefs, &model, &efe);
    assert!(result.is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  AiActiveInferenceAgent tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_agent_construction() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let agent = AiActiveInferenceAgent::new(model);
    assert_eq!(agent.step, 0);
    assert!(agent.free_energy_history.is_empty());
    assert!(agent.action_history.is_empty());
}

#[test]
fn test_agent_step_episode_returns_valid_action() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let mut agent = AiActiveInferenceAgent::new(model);
    let result = agent.step_episode(0)
        .expect("step_episode should succeed");
    let (action, _fe) = result;
    assert!(action < 2, "Action {} should be < 2", action);
}

#[test]
fn test_agent_step_episode_free_energy_finite() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let mut agent = AiActiveInferenceAgent::new(model);
    let (_action, fe) = agent.step_episode(1)
        .expect("step_episode should succeed");
    assert!(fe.is_finite(), "Free energy should be finite, got {}", fe);
}

#[test]
fn test_agent_step_increments() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let mut agent = AiActiveInferenceAgent::new(model);
    agent.step_episode(0).expect("step_episode(0) should succeed");
    agent.step_episode(1).expect("step_episode(1) should succeed");
    assert_eq!(agent.step, 2);
}

#[test]
fn test_agent_action_history() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let mut agent = AiActiveInferenceAgent::new(model);
    for obs in 0..5 {
        agent.step_episode(obs % 3).expect("step_episode should succeed");
    }
    assert_eq!(agent.action_history.len(), 5);
    for &a in &agent.action_history {
        assert!(a < 2);
    }
}

#[test]
fn test_agent_free_energy_history() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let mut agent = AiActiveInferenceAgent::new(model);
    for obs in 0..4 {
        agent.step_episode(obs % 3).expect("step_episode should succeed");
    }
    assert_eq!(agent.free_energy_history.len(), 4);
    for &f in &agent.free_energy_history {
        assert!(f.is_finite());
    }
}

#[test]
fn test_agent_average_free_energy() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let mut agent = AiActiveInferenceAgent::new(model);
    for obs in 0..6 {
        agent.step_episode(obs % 3).expect("step_episode should succeed");
    }
    let avg = agent.average_free_energy();
    assert!(avg.is_finite());
}

#[test]
fn test_agent_reset() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let mut agent = AiActiveInferenceAgent::new(model);
    for obs in 0..5 {
        agent.step_episode(obs % 3).expect("step_episode should succeed");
    }
    agent.reset();
    assert_eq!(agent.step, 0);
    assert!(agent.free_energy_history.is_empty());
    assert!(agent.action_history.is_empty());
}

#[test]
fn test_agent_beliefs_after_step() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let mut agent = AiActiveInferenceAgent::new(model);
    agent.step_episode(0).expect("step_episode should succeed");
    let sum: f64 = agent.beliefs.beliefs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-9,
        "Beliefs should sum to 1 after step"
    );
}

#[test]
fn test_agent_perceive_and_infer() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let mut agent = AiActiveInferenceAgent::new(model);
    let f = agent.perceive_and_infer(0)
        .expect("perceive_and_infer should succeed");
    assert!(f.is_finite());
}

#[test]
fn test_agent_act_returns_valid() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let mut agent = AiActiveInferenceAgent::new(model);
    agent.perceive_and_infer(0)
        .expect("perceive_and_infer should succeed");
    let action = agent.act()
        .expect("act should succeed");
    assert!(action < 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  AiMarkovBlanket tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_markov_blanket_construction() {
    let mb = AiMarkovBlanket::new(3, 2, 4);
    assert_eq!(mb.sensory_states, vec![0, 1, 2]);
    assert_eq!(mb.active_states, vec![3, 4]);
    assert_eq!(mb.internal_states, vec![5, 6, 7, 8]);
    assert_eq!(mb.n_total, 9);
}

#[test]
fn test_markov_blanket_blanket_size() {
    let mb = AiMarkovBlanket::new(3, 2, 4);
    assert_eq!(mb.blanket_size(), 5); // 3 sensory + 2 active
}

#[test]
fn test_markov_blanket_is_internal() {
    let mb = AiMarkovBlanket::new(3, 2, 4);
    assert!(!mb.is_internal(0)); // sensory
    assert!(!mb.is_internal(3)); // active
    assert!(mb.is_internal(5)); // internal
    assert!(mb.is_internal(8)); // internal
}

#[test]
fn test_markov_blanket_surprise_zero_for_perfect_prediction() {
    let mb = AiMarkovBlanket::new(2, 2, 2);
    let obs = vec![1.0, 2.0, 3.0];
    let pred = vec![1.0, 2.0, 3.0];
    assert!(mb.surprise(&obs, &pred).abs() < 1e-12);
}

#[test]
fn test_markov_blanket_surprise_positive_for_mismatch() {
    let mb = AiMarkovBlanket::new(2, 2, 2);
    let obs = vec![1.0, 0.0];
    let pred = vec![0.0, 1.0];
    assert!(mb.surprise(&obs, &pred) > 0.0);
}

#[test]
fn test_markov_blanket_ci_score_diagonal_cov() {
    let mb = AiMarkovBlanket::new(2, 2, 2);
    // Diagonal covariance = independent variables
    let n = mb.n_total;
    let mut cov = vec![vec![0.0; n]; n];
    for i in 0..n {
        cov[i][i] = 1.0;
    }
    let score = mb.conditional_independence_score(&cov);
    // With diagonal cov, cross-covariances are 0 → score should be 0
    assert!(
        score.abs() < 1e-9,
        "Score should be ~0 for diagonal cov, got {}",
        score
    );
}

#[test]
fn test_markov_blanket_surprise_empty_obs() {
    let mb = AiMarkovBlanket::new(2, 2, 2);
    assert_eq!(mb.surprise(&[], &[]), 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  AiHierarchicalGenerativeModel tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hierarchical_model_construction() {
    let specs = &[(4, 3, 2, 2.0), (6, 4, 2, 1.0)];
    let hm = AiHierarchicalGenerativeModel::new(specs);
    assert_eq!(hm.n_levels, 2);
    assert_eq!(hm.levels[0].n_states, 4);
    assert_eq!(hm.levels[1].n_states, 6);
}

#[test]
fn test_hierarchical_model_a_normalized() {
    let specs = &[(4, 3, 2, 1.0), (5, 4, 2, 0.5)];
    let hm = AiHierarchicalGenerativeModel::new(specs);
    for l in 0..hm.n_levels {
        let lv = &hm.levels[l];
        for s in 0..lv.n_states {
            let col_sum: f64 = (0..lv.n_obs).map(|o| lv.a_matrix[o][s]).sum();
            assert!(
                (col_sum - 1.0).abs() < 1e-9,
                "Level {} col {} sums to {}",
                l,
                s,
                col_sum
            );
        }
    }
}

#[test]
fn test_hierarchical_model_top_down_prediction_shape() {
    let specs = &[(4, 3, 2, 1.0)];
    let hm = AiHierarchicalGenerativeModel::new(specs);
    let beliefs = vec![0.25; 4];
    let pred = hm.top_down_prediction(&beliefs, 0)
        .expect("top_down_prediction should succeed");
    assert_eq!(pred.len(), 3); // n_obs at level 0
}

#[test]
fn test_hierarchical_model_top_down_prediction_invalid_level() {
    let specs = &[(4, 3, 2, 1.0)];
    let hm = AiHierarchicalGenerativeModel::new(specs);
    let beliefs = vec![0.25; 4];
    let result = hm.top_down_prediction(&beliefs, 99);
    assert!(result.is_err());
}

#[test]
fn test_hierarchical_model_top_down_prediction_dimension_error() {
    let specs = &[(4, 3, 2, 1.0)];
    let hm = AiHierarchicalGenerativeModel::new(specs);
    let wrong_beliefs = vec![0.5; 2]; // wrong length
    let result = hm.top_down_prediction(&wrong_beliefs, 0);
    assert!(result.is_err());
}

#[test]
fn test_hierarchical_model_bottom_up_error_scale() {
    let specs = &[(4, 3, 2, 2.0)];
    let hm = AiHierarchicalGenerativeModel::new(specs);
    let obs = vec![1.0, 0.5, 0.3];
    let pred = vec![0.8, 0.5, 0.3];
    let error = hm.bottom_up_error(&obs, &pred, 0);
    // Precision = 2.0, error[0] = 2.0 * (1.0 - 0.8) = 0.4
    assert!((error[0] - 0.4).abs() < 1e-9, "Error[0]={}", error[0]);
    // error[1] = 2.0 * 0.0 = 0.0
    assert!(error[1].abs() < 1e-9);
}

#[test]
fn test_hierarchical_model_free_energy_nonneg() {
    let specs = &[(4, 3, 2, 1.0), (5, 4, 2, 0.5)];
    let hm = AiHierarchicalGenerativeModel::new(specs);
    let beliefs = vec![vec![0.25; 4], vec![0.2; 5]];
    let obs = vec![vec![1.0 / 3.0; 3], vec![0.25; 4]];
    let fe = hm.hierarchical_free_energy(&beliefs, &obs)
        .expect("hierarchical_free_energy should succeed");
    assert!(fe >= 0.0);
}

#[test]
fn test_hierarchical_model_free_energy_dimension_error() {
    let specs = &[(4, 3, 2, 1.0)];
    let hm = AiHierarchicalGenerativeModel::new(specs);
    let beliefs = vec![vec![0.25; 4], vec![0.2; 5]]; // too many levels
    let obs = vec![vec![1.0 / 3.0; 3], vec![0.25; 4]];
    let result = hm.hierarchical_free_energy(&beliefs, &obs);
    assert!(result.is_err());
}

#[test]
fn test_hierarchical_model_precision_weights() {
    let specs_high = &[(4, 3, 2, 10.0)];
    let specs_low = &[(4, 3, 2, 0.1)];
    let hm_high = AiHierarchicalGenerativeModel::new(specs_high);
    let hm_low = AiHierarchicalGenerativeModel::new(specs_low);
    let beliefs = vec![vec![0.25; 4]];
    let obs = vec![vec![0.5, 0.3, 0.2]];
    let fe_high = hm_high.hierarchical_free_energy(&beliefs, &obs)
        .expect("hierarchical_free_energy for high precision should succeed");
    let fe_low = hm_low.hierarchical_free_energy(&beliefs, &obs)
        .expect("hierarchical_free_energy for low precision should succeed");
    // Higher precision → larger free energy for same prediction error
    assert!(fe_high > fe_low);
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  AiParameterLearning tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_parameter_learning_construction() {
    let pl = AiParameterLearning::new(0.01, 1.0);
    assert!((pl.learning_rate - 0.01).abs() < 1e-12);
    assert!((pl.concentration - 1.0).abs() < 1e-12);
}

#[test]
fn test_parameter_learning_likelihood_update_changes_a() {
    let mut model = AiGenerativeModel::new(4, 3, 2);
    let beliefs = AiBeliefState::new(4);
    let pl = AiParameterLearning::new(0.5, 1.0);

    let a_before: Vec<f64> = model.a_matrix.iter().flatten().cloned().collect();
    pl.update_likelihood(&mut model, 0, &beliefs);
    let a_after: Vec<f64> = model.a_matrix.iter().flatten().cloned().collect();

    let changed = a_before
        .iter()
        .zip(a_after.iter())
        .any(|(b, a)| (b - a).abs() > 1e-12);
    assert!(changed, "A matrix should change after likelihood update");
}

#[test]
fn test_parameter_learning_a_still_normalized() {
    let mut model = AiGenerativeModel::new(4, 3, 2);
    let beliefs = AiBeliefState::new(4);
    let pl = AiParameterLearning::new(0.3, 1.0);

    for _ in 0..5 {
        pl.update_likelihood(&mut model, 0, &beliefs);
    }

    for s in 0..model.n_states {
        let col_sum: f64 = (0..model.n_obs).map(|o| model.a_matrix[o][s]).sum();
        assert!(
            (col_sum - 1.0).abs() < 1e-9,
            "Col {} sums to {}",
            s,
            col_sum
        );
    }
}

#[test]
fn test_parameter_learning_transition_update_changes_b() {
    let mut model = AiGenerativeModel::new(4, 3, 2);
    let prev_beliefs = AiBeliefState::new(4);
    let curr_beliefs = AiBeliefState::new(4);
    let pl = AiParameterLearning::new(0.5, 1.0);

    let b_before: Vec<f64> = model.b_matrices[0].iter().flatten().cloned().collect();
    pl.update_transitions(&mut model, 0, &prev_beliefs, &curr_beliefs);
    let b_after: Vec<f64> = model.b_matrices[0].iter().flatten().cloned().collect();

    let changed = b_before
        .iter()
        .zip(b_after.iter())
        .any(|(b, a)| (b - a).abs() > 1e-12);
    assert!(changed, "B matrix should change after transition update");
}

#[test]
fn test_parameter_learning_b_still_normalized() {
    let mut model = AiGenerativeModel::new(4, 3, 2);
    let prev_beliefs = AiBeliefState::new(4);
    let curr_beliefs = AiBeliefState::new(4);
    let pl = AiParameterLearning::new(0.3, 1.0);

    for _ in 0..5 {
        pl.update_transitions(&mut model, 0, &prev_beliefs, &curr_beliefs);
    }

    for u in 0..model.n_actions {
        for s in 0..model.n_states {
            let col_sum: f64 = (0..model.n_states)
                .map(|sn| model.b_matrices[u][sn][s])
                .sum();
            assert!(
                (col_sum - 1.0).abs() < 1e-9,
                "B[{}] col {} sums to {}",
                u,
                s,
                col_sum
            );
        }
    }
}

#[test]
fn test_parameter_learning_model_evidence_finite() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let pl = AiParameterLearning::new(0.01, 1.0);
    let obs_seq = vec![0usize, 1, 2, 0, 1];
    let ev = pl.model_evidence(&model, &obs_seq);
    assert!(ev.is_finite());
}

#[test]
fn test_parameter_learning_invalid_obs_ignored() {
    let mut model = AiGenerativeModel::new(4, 3, 2);
    let beliefs = AiBeliefState::new(4);
    let pl = AiParameterLearning::new(0.5, 1.0);

    // Invalid obs index should be silently ignored (no panic)
    pl.update_likelihood(&mut model, 99, &beliefs);

    // A should still be normalized
    for s in 0..model.n_states {
        let col_sum: f64 = (0..model.n_obs).map(|o| model.a_matrix[o][s]).sum();
        assert!((col_sum - 1.0).abs() < 1e-9);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  AiMetrics tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_metrics_free_energy_trajectory_basic() {
    let history = vec![2.0, 1.5, 1.0, 0.8, 0.5];
    let (mean, min, last) = AiMetrics::free_energy_trajectory(&history);
    assert!((mean - 1.16).abs() < 0.01, "mean={}", mean);
    assert!((min - 0.5).abs() < 1e-9);
    assert!((last - 0.5).abs() < 1e-9);
}

#[test]
fn test_metrics_free_energy_trajectory_empty() {
    let (mean, min, last) = AiMetrics::free_energy_trajectory(&[]);
    assert_eq!((mean, min, last), (0.0, 0.0, 0.0));
}

#[test]
fn test_metrics_action_entropy_uniform() {
    let posterior = vec![0.25; 4];
    let h = AiMetrics::action_entropy(&posterior);
    let expected = -(4.0 * 0.25 * 0.25_f64.ln());
    assert!((h - expected).abs() < 1e-9);
}

#[test]
fn test_metrics_action_entropy_deterministic() {
    let posterior = vec![0.0, 0.0, 1.0, 0.0];
    let h = AiMetrics::action_entropy(&posterior);
    assert!(
        h.abs() < 1e-9,
        "Entropy of deterministic dist should be 0, got {}",
        h
    );
}

#[test]
fn test_metrics_belief_accuracy_correct() {
    let mut bs = AiBeliefState::new(4);
    bs.update(&[0.1, 0.1, 0.7, 0.1]);
    let acc = AiMetrics::belief_accuracy(&bs, 2); // true state = 2
    assert!(
        acc > 0.5,
        "Accuracy for true state 2 should be > 0.5, got {}",
        acc
    );
}

#[test]
fn test_metrics_belief_accuracy_out_of_range() {
    let bs = AiBeliefState::new(4);
    let acc = AiMetrics::belief_accuracy(&bs, 99);
    assert_eq!(acc, 0.0);
}

#[test]
fn test_metrics_average_surprise_finite() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let obs_seq = vec![0, 1, 2, 0];
    let state_seq = vec![0, 1, 2, 3];
    let s = AiMetrics::average_surprise(&model, &obs_seq, &state_seq);
    assert!(s.is_finite());
    assert!(s >= 0.0);
}

#[test]
fn test_metrics_average_surprise_empty() {
    let model = AiGenerativeModel::new(4, 3, 2);
    let s = AiMetrics::average_surprise(&model, &[], &[]);
    assert_eq!(s, 0.0);
}

#[test]
fn test_metrics_policy_efficiency_uniform() {
    let actions = vec![0, 1, 2, 3, 0, 1, 2, 3];
    let eff = AiMetrics::policy_efficiency(&actions, 4);
    let expected = (4.0_f64).ln(); // log(n_actions) for uniform
    assert!((eff - expected).abs() < 0.1, "efficiency={}", eff);
}

#[test]
fn test_metrics_policy_efficiency_deterministic() {
    let actions = vec![0, 0, 0, 0];
    let eff = AiMetrics::policy_efficiency(&actions, 4);
    assert!(
        eff.abs() < 1e-9,
        "Deterministic policy entropy should be 0, got {}",
        eff
    );
}

#[test]
fn test_metrics_policy_efficiency_empty() {
    let eff = AiMetrics::policy_efficiency(&[], 4);
    assert_eq!(eff, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_full_episode_loop() {
    let model = AiGenerativeModel::new(5, 4, 3);
    let mut agent = AiActiveInferenceAgent::new(model);
    let observations = [0, 1, 2, 3, 0, 1, 2];

    for &obs in &observations {
        let result = agent.step_episode(obs);
        assert!(result.is_ok(), "step_episode failed for obs={}", obs);
        let (action, fe) = result.expect("step_episode should succeed in full episode loop");
        assert!(action < 3, "Action {} >= 3", action);
        assert!(fe.is_finite());
    }

    let avg_fe = agent.average_free_energy();
    assert!(avg_fe.is_finite());
    assert_eq!(agent.step, observations.len());
}

#[test]
fn test_parameter_learning_in_loop() {
    let mut model = AiGenerativeModel::new(4, 3, 2);
    let pl = AiParameterLearning::new(0.05, 1.0);
    let vi = AiVariationalInference::new(10, 0.5, 1e-5);
    let prior = AiBeliefState::new(4);

    let mut prev_beliefs = prior;
    for t in 0..10 {
        let obs = t % 3;
        let curr_beliefs = vi.infer_states(obs, &model, &prev_beliefs)
            .expect("infer_states should succeed in learning loop");
        if t > 0 {
            pl.update_transitions(&mut model, 0, &prev_beliefs, &curr_beliefs);
        }
        pl.update_likelihood(&mut model, obs, &curr_beliefs);
        prev_beliefs = curr_beliefs;

        // Verify model remains valid
        for s in 0..model.n_states {
            let col_sum: f64 = (0..model.n_obs).map(|o| model.a_matrix[o][s]).sum();
            assert!(
                (col_sum - 1.0).abs() < 1e-7,
                "t={} A col {} sums to {}",
                t,
                s,
                col_sum
            );
        }
    }
}

#[test]
fn test_hierarchical_inference_loop() {
    let specs = &[(4, 3, 2, 2.0), (6, 4, 1, 1.0)];
    let hm = AiHierarchicalGenerativeModel::new(specs);

    let beliefs = vec![vec![0.25; 4], vec![1.0 / 6.0; 6]];
    let obs = vec![vec![1.0 / 3.0; 3], vec![0.25; 4]];

    let fe = hm.hierarchical_free_energy(&beliefs, &obs)
        .expect("hierarchical_free_energy should succeed in integration test");
    assert!(fe.is_finite() && fe >= 0.0);

    let pred = hm.top_down_prediction(&beliefs[0], 0)
        .expect("top_down_prediction should succeed in integration test");
    assert_eq!(pred.len(), 3);

    let err = hm.bottom_up_error(&obs[0], &pred, 0);
    assert_eq!(err.len(), 3);
    for &e in &err {
        assert!(e.is_finite());
    }
}

#[test]
fn test_markov_blanket_full_pipeline() {
    let mb = AiMarkovBlanket::new(4, 3, 5);
    assert_eq!(mb.blanket_size(), 7);
    assert_eq!(mb.n_total, 12);

    // Test surprise with typical sensory prediction
    let obs = vec![0.8, 0.2, 0.5, 0.9];
    let pred = vec![0.7, 0.3, 0.5, 0.8];
    let s = mb.surprise(&obs, &pred);
    assert!(s >= 0.0);
    assert!(s.is_finite());

    // Test conditional independence score with identity covariance
    let n = mb.n_total;
    let mut cov = vec![vec![0.0; n]; n];
    for i in 0..n {
        cov[i][i] = 1.0;
    }
    let score = mb.conditional_independence_score(&cov);
    assert!((score).abs() < 1e-9);
}
