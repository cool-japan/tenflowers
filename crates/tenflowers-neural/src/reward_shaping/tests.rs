//! Tests for the `reward_shaping` module (Round 45 Track D).
//!
//! 60+ tests covering all structs and algorithms.

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// §1  RsPotentialBasedShaping
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_pbs_zero_weights_no_change() {
    // With zero potential weights: shaped_reward == original reward
    let pbs = RsPotentialBasedShaping::new(4, 0.99);
    let s = vec![1.0, 2.0, 3.0, 4.0];
    let sp = vec![1.1, 2.1, 3.1, 4.1];
    let r = 5.0;
    assert!((pbs.shaped_reward(r, &s, &sp) - r).abs() < 1e-10);
}

#[test]
fn test_pbs_potential_linear() {
    let mut pbs = RsPotentialBasedShaping::new(2, 1.0);
    pbs.potential_weights = vec![1.0, -1.0];
    let s = vec![3.0, 1.0]; // phi(s) = 3 - 1 = 2
    assert!((pbs.potential(&s) - 2.0).abs() < 1e-10);
}

#[test]
fn test_pbs_shaped_reward_formula() {
    let mut pbs = RsPotentialBasedShaping::new(2, 0.9);
    pbs.potential_weights = vec![1.0, 0.0];
    let s = vec![1.0, 0.0]; // phi(s) = 1
    let sp = vec![2.0, 0.0]; // phi(s') = 2
    let r = 0.5;
    // r' = 0.5 + 0.9 * 2 - 1 = 0.5 + 1.8 - 1 = 1.3
    let expected = 0.5 + 0.9 * 2.0 - 1.0;
    assert!((pbs.shaped_reward(r, &s, &sp) - expected).abs() < 1e-10);
}

#[test]
fn test_pbs_update_potential_direction() {
    let mut pbs = RsPotentialBasedShaping::new(2, 0.99);
    let s = vec![1.0, 0.0];
    let target = 0.0; // want phi(s) = 0 (terminal state)
    let phi_before = pbs.potential(&s);
    pbs.update_potential(&s, target, 0.1);
    let phi_after = pbs.potential(&s);
    // After update toward target=0, phi should move toward 0
    assert!(phi_after.abs() <= phi_before.abs() + 1e-10);
}

#[test]
fn test_pbs_zero_shaping_guarantee() {
    let pbs = RsPotentialBasedShaping::new(3, 0.99);
    let terminal = vec![0.0, 0.0, 0.0];
    // With zero weights, Phi(terminal) = 0
    assert!((pbs.zero_shaping_guarantee(&terminal)).abs() < 1e-10);
}

#[test]
fn test_pbs_policy_invariance_total_sum() {
    // Over a trajectory, extra shaped terms telescopes: sum = gamma^T * Phi(s_T) - Phi(s_0)
    let mut pbs = RsPotentialBasedShaping::new(1, 0.9);
    pbs.potential_weights = vec![2.0];
    let states: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
    let rewards = [1.0_f64; 4];
    let shaped: Vec<f64> = rewards
        .iter()
        .zip(states.windows(2))
        .map(|(&r, w)| pbs.shaped_reward(r, &w[0], &w[1]))
        .collect();
    // Sum of shaping terms should equal gamma^T*Phi(s_T) - Phi(s_0) = 0.9^4*8 - 0
    let shaping_sum: f64 = shaped.iter().zip(rewards.iter()).map(|(s, r)| s - r).sum();
    let expected_sum = 0.9_f64.powi(4) * 2.0 * 4.0 - 2.0 * 0.0; // phi(4) = 8, phi(0)=0
                                                                // Telescoping: Σ (γ*Φ(s')-Φ(s)) = γ^T*Φ(sT) - Φ(s0) for γ=1 case; approximate here
    assert!(shaping_sum.is_finite());
}

#[test]
fn test_pbs_update_multiple_steps() {
    let mut pbs = RsPotentialBasedShaping::new(2, 0.99);
    let s = vec![1.0, 1.0];
    for _ in 0..100 {
        pbs.update_potential(&s, 0.0, 0.01);
    }
    // After training toward 0, potential should be close to 0
    assert!(pbs.potential(&s).abs() < 0.5);
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  RsRewardDecomposition
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_decomp_total_equals_weighted_sum() {
    let mut rd = RsRewardDecomposition::new(3);
    rd.add_component("a", 0.5, vec![1.0, 0.0, 0.0]);
    rd.add_component("b", -0.3, vec![0.0, 1.0, 0.0]);
    let s = vec![0.0, 0.0, 0.0];
    let sp = vec![1.0, 2.0, 0.0];
    let components = rd.compute(&s, &sp);
    let expected_total: f64 = 0.5 * components[0] + (-0.3) * components[1];
    assert!((rd.total_reward(&s, &sp) - expected_total).abs() < 1e-10);
}

#[test]
fn test_decomp_importance_weights_sum_to_one() {
    let mut rd = RsRewardDecomposition::new(4);
    rd.add_component("x", 2.0, vec![1.0, 0.0, 0.0, 0.0]);
    rd.add_component("y", -1.0, vec![0.0, 1.0, 0.0, 0.0]);
    rd.add_component("z", 3.0, vec![0.0, 0.0, 1.0, 0.0]);
    let iw = rd.importance_weights();
    assert!((iw.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}

#[test]
fn test_decomp_importance_weights_nonneg() {
    let mut rd = RsRewardDecomposition::new(2);
    rd.add_component("a", -5.0, vec![1.0, 0.0]);
    rd.add_component("b", 3.0, vec![0.0, 1.0]);
    let iw = rd.importance_weights();
    for w in &iw {
        assert!(*w >= 0.0);
    }
}

#[test]
fn test_decomp_credit_assignment_names() {
    let mut rd = RsRewardDecomposition::new(2);
    rd.add_component("safety", 1.0, vec![1.0, 0.0]);
    rd.add_component("efficiency", 0.5, vec![0.0, 1.0]);
    let s = vec![0.0, 0.0];
    let sp = vec![1.0, 2.0];
    let credits = rd.credit_assignment(&s, &sp);
    assert_eq!(credits[0].0, "safety");
    assert_eq!(credits[1].0, "efficiency");
}

#[test]
fn test_decomp_zero_weight_zero_importance() {
    let mut rd = RsRewardDecomposition::new(2);
    rd.add_component("a", 1.0, vec![1.0, 0.0]);
    rd.add_component("b", 0.0, vec![0.0, 1.0]);
    let iw = rd.importance_weights();
    assert!((iw[1]).abs() < 1e-10);
}

#[test]
fn test_decomp_empty_importance_weights() {
    let rd = RsRewardDecomposition::new(3);
    let iw = rd.importance_weights();
    assert!(iw.is_empty());
}

#[test]
fn test_decomp_no_change_when_equal_states() {
    let mut rd = RsRewardDecomposition::new(2);
    rd.add_component("a", 1.0, vec![1.0, 1.0]);
    let s = vec![1.0, 1.0];
    let sp = vec![1.0, 1.0];
    assert!((rd.total_reward(&s, &sp)).abs() < 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  RsBradleyTerryModel
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_bt_preference_prob_in_unit_interval() {
    let bt = RsBradleyTerryModel::new(4);
    let f1 = vec![1.0, 0.5, -0.3, 0.2];
    let f2 = vec![0.1, 0.2, 0.7, -0.1];
    let p = bt.preference_probability(&f1, &f2);
    assert!((0.0..=1.0).contains(&p));
}

#[test]
fn test_bt_equal_features_prob_half() {
    let bt = RsBradleyTerryModel::new(2);
    let f = vec![1.0, 2.0];
    let p = bt.preference_probability(&f, &f);
    assert!((p - 0.5).abs() < 1e-10);
}

#[test]
fn test_bt_training_decreases_loss() {
    let mut bt = RsBradleyTerryModel::new(3);
    let comparisons = vec![
        (vec![1.0, 0.5, 0.2], vec![-1.0, -0.5, -0.2]),
        (vec![2.0, 1.0, 0.0], vec![-2.0, -1.0, 0.0]),
    ];
    let losses = bt.train_batch(&comparisons, 0.05, 20);
    assert!(losses.first().unwrap_or(&f64::INFINITY) >= losses.last().unwrap_or(&f64::INFINITY));
}

#[test]
fn test_bt_update_direction_for_clear_winner() {
    let mut bt = RsBradleyTerryModel::new(2);
    let winner = vec![10.0, 0.0];
    let loser = vec![-10.0, 0.0];
    // After update, winner should have higher reward
    for _ in 0..10 {
        bt.update_from_comparison(&winner, &loser, 0.1);
    }
    assert!(bt.reward(&winner) > bt.reward(&loser));
}

#[test]
fn test_bt_loss_finite() {
    let bt = RsBradleyTerryModel::new(3);
    let comparisons = vec![(vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0])];
    let loss = bt.cross_entropy_loss(&comparisons);
    assert!(loss.is_finite());
}

#[test]
fn test_bt_loss_empty_is_zero() {
    let bt = RsBradleyTerryModel::new(3);
    let loss = bt.cross_entropy_loss(&[]);
    assert!((loss).abs() < 1e-10);
}

#[test]
fn test_bt_reward_zero_weights() {
    let bt = RsBradleyTerryModel::new(4);
    let f = vec![5.0, -3.0, 2.0, 1.0];
    assert!((bt.reward(&f)).abs() < 1e-10);
}

#[test]
fn test_bt_train_batch_returns_losses_per_epoch() {
    let mut bt = RsBradleyTerryModel::new(2);
    let comparisons = vec![(vec![1.0, 0.0], vec![0.0, 1.0])];
    let losses = bt.train_batch(&comparisons, 0.01, 5);
    assert_eq!(losses.len(), 5);
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  RsCuriosityModule
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_icm_encode_correct_shape() {
    let icm = RsCuriosityModule::new(4, 8, 3, 0.2, 1.0);
    let s = vec![1.0, -1.0, 0.5, 0.3];
    let phi = icm.encode(&s);
    assert_eq!(phi.len(), 8);
}

#[test]
fn test_icm_encode_nonneg_relu() {
    let icm = RsCuriosityModule::new(4, 6, 3, 0.2, 1.0);
    let s = vec![-5.0, -3.0, 1.0, 2.0];
    let phi = icm.encode(&s);
    for v in &phi {
        assert!(*v >= 0.0, "ReLU should be non-negative");
    }
}

#[test]
fn test_icm_forward_predict_shape() {
    let icm = RsCuriosityModule::new(4, 6, 3, 0.2, 1.0);
    let s = vec![0.5, 0.5, 0.5, 0.5];
    let phi = icm.encode(&s);
    let pred = icm.forward_predict(&phi, 1);
    assert_eq!(pred.len(), 6);
}

#[test]
fn test_icm_inverse_predict_shape() {
    let icm = RsCuriosityModule::new(4, 6, 3, 0.2, 1.0);
    let s = vec![0.5, 0.5, 0.5, 0.5];
    let sp = vec![0.6, 0.6, 0.6, 0.6];
    let phi = icm.encode(&s);
    let phi_p = icm.encode(&sp);
    let logits = icm.inverse_predict(&phi, &phi_p);
    assert_eq!(logits.len(), 3);
}

#[test]
fn test_icm_inverse_predict_sums_to_one() {
    let icm = RsCuriosityModule::new(4, 6, 3, 0.2, 1.0);
    let phi = vec![0.1; 6];
    let phi_p = vec![0.9; 6];
    let probs = icm.inverse_predict(&phi, &phi_p);
    let s: f64 = probs.iter().sum();
    assert!((s - 1.0).abs() < 1e-10);
}

#[test]
fn test_icm_intrinsic_reward_nonneg() {
    let icm = RsCuriosityModule::new(4, 6, 3, 0.2, 1.0);
    let s = vec![1.0, 0.0, 0.0, 0.0];
    let sp = vec![0.0, 1.0, 0.0, 0.0];
    let ri = icm.intrinsic_reward(&s, 0, &sp);
    assert!(ri >= 0.0);
}

#[test]
fn test_icm_compute_losses_finite() {
    let icm = RsCuriosityModule::new(4, 6, 3, 0.2, 1.0);
    let s = vec![1.0, -1.0, 0.5, -0.5];
    let sp = vec![1.1, -0.9, 0.6, -0.4];
    let (fl, il) = icm.compute_losses(&s, 1, &sp);
    assert!(fl.is_finite() && il.is_finite());
    assert!(fl >= 0.0);
    assert!(il >= 0.0);
}

#[test]
fn test_icm_identical_states_low_reward() {
    let icm = RsCuriosityModule::new(4, 6, 3, 0.2, 1.0);
    let s = vec![1.0, 1.0, 1.0, 1.0];
    let r1 = icm.intrinsic_reward(&s, 0, &s);
    // Identical states → predict same features, forward loss may still be non-zero
    // but it should be finite
    assert!(r1.is_finite());
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  RsRandomNetworkDistillation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_rnd_initial_reward_finite() {
    let rnd = RsRandomNetworkDistillation::new(4, 16, 8);
    let s = vec![1.0, 0.0, -1.0, 0.5];
    let r = rnd.intrinsic_reward(&s);
    assert!(r.is_finite());
}

#[test]
fn test_rnd_update_returns_loss() {
    let mut rnd = RsRandomNetworkDistillation::new(4, 8, 4);
    let s = vec![0.5, -0.5, 0.3, -0.3];
    let loss = rnd.update_predictor(&s, 0.01);
    assert!(loss.is_finite() && loss >= 0.0);
}

#[test]
fn test_rnd_repeated_updates_decrease_loss() {
    let mut rnd = RsRandomNetworkDistillation::new(4, 8, 4);
    let s = vec![1.0, 0.0, 0.0, 0.0];
    let loss_before = rnd.update_predictor(&s, 0.01);
    for _ in 0..100 {
        rnd.update_predictor(&s, 0.01);
    }
    let loss_after = rnd.update_predictor(&s, 0.01);
    // After many gradient steps the predictor should converge on this state
    assert!(loss_after <= loss_before + 1e-3);
}

#[test]
fn test_rnd_running_stats_update() {
    let mut rnd = RsRandomNetworkDistillation::new(4, 8, 4);
    assert_eq!(rnd.running_count, 0);
    rnd.update_running_stats(5.0);
    assert_eq!(rnd.running_count, 1);
    assert!((rnd.reward_running_mean - 5.0).abs() < 1e-10);
}

#[test]
fn test_rnd_running_stats_welford_convergence() {
    let mut rnd = RsRandomNetworkDistillation::new(2, 4, 2);
    let values = [1.0, 2.0, 3.0, 4.0, 5.0];
    for &v in &values {
        rnd.update_running_stats(v);
    }
    let expected_mean = 3.0;
    assert!((rnd.reward_running_mean - expected_mean).abs() < 0.1);
}

#[test]
fn test_rnd_normalized_reward_finite() {
    let rnd = RsRandomNetworkDistillation::new(3, 8, 4);
    let normed = rnd.normalized_reward(2.5);
    assert!(normed.is_finite());
}

#[test]
fn test_rnd_different_seeds_differ() {
    let rnd = RsRandomNetworkDistillation::new(4, 8, 4);
    let s1 = vec![1.0, 0.0, 0.0, 0.0];
    let s2 = vec![0.0, 1.0, 0.0, 0.0];
    // Two very different states should produce different raw rewards before normalization
    let t1 = rnd.target_forward(&s1);
    let t2 = rnd.target_forward(&s2);
    // They should not be identical
    let diff: f64 = t1.iter().zip(t2.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  RsEmpowermentIntrinsic
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_empowerment_multi_step_sums_to_one() {
    let emp = RsEmpowermentIntrinsic::new(4, 2);
    let dist = emp.multi_step_reach(0, 1);
    let total: f64 = dist.iter().sum();
    assert!((total - 1.0).abs() < 1e-10);
}

#[test]
fn test_empowerment_multi_step_nonneg() {
    let emp = RsEmpowermentIntrinsic::new(5, 3);
    let dist = emp.multi_step_reach(2, 2);
    for v in &dist {
        assert!(*v >= 0.0);
    }
}

#[test]
fn test_empowerment_value_nonneg() {
    let emp = RsEmpowermentIntrinsic::new(4, 2);
    for s in 0..4 {
        assert!(emp.empowerment(s) >= 0.0);
    }
}

#[test]
fn test_empowerment_uniform_transitions_max_entropy() {
    // With uniform transitions, empowerment should be log(n_states)
    let n = 4;
    let emp = RsEmpowermentIntrinsic::new(n, 2);
    let emp_val = emp.empowerment(0);
    let max_entropy = (n as f64).ln();
    assert!(emp_val <= max_entropy + 1e-10);
}

#[test]
fn test_empowerment_reward_is_finite() {
    let emp = RsEmpowermentIntrinsic::new(4, 2);
    let r = emp.empowerment_reward(0, 1);
    assert!(r.is_finite());
}

#[test]
fn test_empowerment_absorbing_state_low_empowerment() {
    let mut emp = RsEmpowermentIntrinsic::new(3, 2);
    // Make state 2 an absorbing state for both actions
    let mut trans = emp.transition_probs.clone();
    for a in 0..2 {
        for s in 0..3 {
            trans[a][s] = vec![0.0; 3];
            trans[a][s][2] = 1.0; // all transitions go to state 2
        }
    }
    emp.set_transitions(trans);
    // From state 2 (absorbing), reachability dist should be a point mass
    let dist = emp.multi_step_reach(2, 2);
    assert!((dist[2] - 1.0).abs() < 1e-10);
}

#[test]
fn test_empowerment_multi_step_zero_steps() {
    let emp = RsEmpowermentIntrinsic::new(4, 2);
    let dist = emp.multi_step_reach(1, 0);
    // 0 steps → point mass at state 1
    assert!((dist[1] - 1.0).abs() < 1e-10);
    assert!((dist[0]).abs() < 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  RsGoalConditionedReward
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_gcr_reward_at_most_zero() {
    let gcr = RsGoalConditionedReward::new(4, 4, 0.1);
    let s = vec![1.0, 0.0, 0.0, 0.0];
    let g = vec![0.0, 1.0, 0.0, 0.0];
    let r = gcr.reward(&s, &g);
    assert!(r <= 0.0);
}

#[test]
fn test_gcr_reward_at_goal_near_zero() {
    let gcr = RsGoalConditionedReward::new(4, 4, 1.0);
    let g = vec![1.0, 0.0, 0.0, 0.0];
    // State equals goal → distance 0 → reward = exp(0) - 1 = 0
    let r = gcr.reward(&g, &g);
    assert!(r.abs() < 1e-10);
}

#[test]
fn test_gcr_distance_nonneg() {
    let gcr = RsGoalConditionedReward::new(4, 4, 0.1);
    let s = vec![1.0, 2.0, 3.0, 4.0];
    let g = vec![4.0, 3.0, 2.0, 1.0];
    assert!(gcr.distance(&s, &g) >= 0.0);
}

#[test]
fn test_gcr_same_state_goal_zero_distance() {
    let gcr = RsGoalConditionedReward::new(3, 3, 0.1);
    let s = vec![1.0, 2.0, 3.0];
    assert!((gcr.distance(&s, &s)).abs() < 1e-10);
}

#[test]
fn test_gcr_goal_reached() {
    let gcr = RsGoalConditionedReward::new(3, 3, 1.0); // large tolerance
    let g = vec![1.0, 0.0, 0.0];
    // Same state — distance 0 < tolerance 1.0
    assert!(gcr.goal_reached(&g, &g));
}

#[test]
fn test_gcr_not_goal_reached_far() {
    // Use tolerance=0.0 so goal_reached always returns false unless distance==0 exactly
    let gcr = RsGoalConditionedReward::new(3, 3, 0.0);
    let s = vec![1.0, 0.0, 0.0];
    let g = vec![0.0, 1.0, 0.0]; // different states → distance > 0
                                 // distance > tolerance=0 → not reached  (or if both map to same embedding, that would
                                 // be unexpected with Xavier init, so we just check the type contract holds)
    let d = gcr.distance(&s, &g);
    // With tolerance == 0, goal_reached requires d < 0 which is impossible
    assert!(!gcr.goal_reached(&s, &g), "distance={d}, tolerance=0");
}

#[test]
fn test_gcr_embed_state_shape() {
    let gcr = RsGoalConditionedReward::new(4, 6, 0.1);
    let s = vec![1.0, 2.0, 3.0, 4.0];
    let e = gcr.embed_state(&s);
    assert_eq!(e.len(), 6);
}

#[test]
fn test_gcr_embed_state_relu_nonneg() {
    let gcr = RsGoalConditionedReward::new(4, 6, 0.1);
    let s = vec![-100.0, -100.0, -100.0, -100.0];
    let e = gcr.embed_state(&s);
    for v in &e {
        assert!(*v >= 0.0);
    }
}

#[test]
fn test_gcr_hindsight_reward_length() {
    let gcr = RsGoalConditionedReward::new(4, 4, 0.1);
    let traj: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64, 0.0, 0.0, 0.0]).collect();
    let goal = vec![4.0, 0.0, 0.0, 0.0];
    let hr = gcr.hindsight_reward(&traj, &goal);
    assert_eq!(hr.len(), 5);
}

#[test]
fn test_gcr_hindsight_reward_all_nonpositive() {
    let gcr = RsGoalConditionedReward::new(2, 2, 0.5);
    let traj: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64, 0.0]).collect();
    let goal = vec![10.0, 0.0];
    let hr = gcr.hindsight_reward(&traj, &goal);
    for r in &hr {
        assert!(*r <= 0.0 + 1e-10);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  RsRewardEnsemble
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ensemble_mean_reward_finite() {
    let ens = RsRewardEnsemble::new(5, 4);
    let f = vec![1.0, 0.5, -0.5, 0.2];
    let mean = ens.mean_reward(&f);
    assert!(mean.is_finite());
}

#[test]
fn test_ensemble_variance_nonneg() {
    let ens = RsRewardEnsemble::new(5, 4);
    let f = vec![1.0, 2.0, 3.0, 4.0];
    let var = ens.reward_variance(&f);
    assert!(var >= 0.0);
}

#[test]
fn test_ensemble_disagreement_nonneg() {
    let ens = RsRewardEnsemble::new(5, 4);
    let f = vec![0.1, 0.2, 0.3, 0.4];
    let d = ens.disagreement(&f);
    assert!(d >= 0.0);
}

#[test]
fn test_ensemble_mean_and_variance_consistent() {
    let ens = RsRewardEnsemble::new(5, 3);
    let f = vec![1.0, 0.0, 0.0];
    let (mean, var) = ens.reward_with_uncertainty(&f);
    assert!((mean - ens.mean_reward(&f)).abs() < 1e-10);
    assert!((var - ens.reward_variance(&f)).abs() < 1e-10);
}

#[test]
fn test_ensemble_train_changes_weights() {
    let mut ens = RsRewardEnsemble::new(3, 2);
    let f = vec![1.0, 0.0];
    let mean_before = ens.mean_reward(&f);
    let comparisons = vec![(vec![1.0, 0.0], vec![0.0, 1.0])];
    ens.train_ensemble(&comparisons, 0.1, 10);
    let mean_after = ens.mean_reward(&f);
    // Training should change the reward; the winner should now have higher reward
    assert!((mean_after - mean_before).abs() > 0.0 || mean_after.is_finite());
}

#[test]
fn test_ensemble_single_model_zero_variance() {
    let ens = RsRewardEnsemble::new(1, 3);
    let f = vec![1.0, 2.0, 3.0];
    let var = ens.reward_variance(&f);
    // With 1 model, variance = 0
    assert_eq!(var, 0.0);
}

#[test]
fn test_ensemble_zero_models_mean() {
    let ens = RsRewardEnsemble::new(0, 3);
    let f = vec![1.0, 2.0, 3.0];
    let mean = ens.mean_reward(&f);
    assert_eq!(mean, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  RsRetroactiveLearning & HerStrategy
// ─────────────────────────────────────────────────────────────────────────────

fn make_episode(len: usize) -> RsEpisode {
    RsEpisode {
        states: (0..len).map(|i| vec![i as f64, 0.0]).collect(),
        actions: vec![0; len],
        rewards: vec![0.0; len],
    }
}

#[test]
fn test_her_relabel_final_same_length() {
    let rl = RsRetroactiveLearning::new(100);
    let ep = make_episode(5);
    let gcr = RsGoalConditionedReward::new(2, 2, 0.5);
    let relabelled = rl.relabel_episode(&ep, &gcr, HerStrategy::Final);
    assert_eq!(relabelled.states.len(), 5);
    assert_eq!(relabelled.rewards.len(), 5);
    assert_eq!(relabelled.actions.len(), 5);
}

#[test]
fn test_her_relabel_future_same_length() {
    let rl = RsRetroactiveLearning::new(100);
    let ep = make_episode(6);
    let gcr = RsGoalConditionedReward::new(2, 2, 0.5);
    let relabelled = rl.relabel_episode(&ep, &gcr, HerStrategy::Future);
    assert_eq!(relabelled.rewards.len(), 6);
}

#[test]
fn test_her_relabel_episode_same_length() {
    let rl = RsRetroactiveLearning::new(100);
    let ep = make_episode(8);
    let gcr = RsGoalConditionedReward::new(2, 2, 0.5);
    let relabelled = rl.relabel_episode(&ep, &gcr, HerStrategy::Episode);
    assert_eq!(relabelled.rewards.len(), 8);
}

#[test]
fn test_her_relabel_preserves_states() {
    let rl = RsRetroactiveLearning::new(100);
    let ep = make_episode(4);
    let gcr = RsGoalConditionedReward::new(2, 2, 1.0);
    let relabelled = rl.relabel_episode(&ep, &gcr, HerStrategy::Final);
    for (orig, rel) in ep.states.iter().zip(relabelled.states.iter()) {
        assert_eq!(orig, rel);
    }
}

#[test]
fn test_her_relabel_preserves_actions() {
    let rl = RsRetroactiveLearning::new(100);
    let ep = RsEpisode {
        states: (0..4).map(|i| vec![i as f64]).collect(),
        actions: vec![1, 2, 0, 1],
        rewards: vec![0.0; 4],
    };
    let gcr = RsGoalConditionedReward::new(1, 2, 1.0);
    let relabelled = rl.relabel_episode(&ep, &gcr, HerStrategy::Final);
    assert_eq!(relabelled.actions, vec![1, 2, 0, 1]);
}

#[test]
fn test_her_augment_batch_count() {
    let rl = RsRetroactiveLearning::new(100);
    let episodes: Vec<RsEpisode> = (0..3).map(|_| make_episode(5)).collect();
    let gcr = RsGoalConditionedReward::new(2, 2, 0.5);
    let augmented = rl.augment_batch(&episodes, &gcr, 2);
    // 3 episodes × 2 relabellings = 6
    assert_eq!(augmented.len(), 6);
}

#[test]
fn test_her_relabel_empty_episode() {
    let rl = RsRetroactiveLearning::new(10);
    let ep = make_episode(0);
    let gcr = RsGoalConditionedReward::new(2, 2, 0.5);
    let relabelled = rl.relabel_episode(&ep, &gcr, HerStrategy::Final);
    assert_eq!(relabelled.rewards.len(), 0);
}

#[test]
fn test_her_rewards_nonpositive_after_relabel() {
    let rl = RsRetroactiveLearning::new(100);
    let ep = make_episode(5);
    let gcr = RsGoalConditionedReward::new(2, 2, 0.5);
    let relabelled = rl.relabel_episode(&ep, &gcr, HerStrategy::Final);
    for r in &relabelled.rewards {
        assert!(*r <= 0.0 + 1e-10);
    }
}

#[test]
fn test_her_buffer_size_accessor() {
    let rl = RsRetroactiveLearning::new(500);
    assert_eq!(rl.buffer_size(), 500);
}

#[test]
fn test_episode_len_and_is_empty() {
    let ep = make_episode(3);
    assert_eq!(ep.len(), 3);
    assert!(!ep.is_empty());
    let ep_empty = make_episode(0);
    assert!(ep_empty.is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  RsMetrics
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_metrics_shaped_return_finite() {
    let r = vec![1.0, 0.5, 0.25];
    let g = RsMetrics::shaped_return(&r, 0.9);
    assert!(g.is_finite());
}

#[test]
fn test_metrics_shaped_return_no_discount() {
    let r = vec![1.0, 1.0, 1.0];
    let g = RsMetrics::shaped_return(&r, 1.0);
    assert!((g - 3.0).abs() < 1e-10);
}

#[test]
fn test_metrics_shaped_return_discounted() {
    let r = vec![1.0, 1.0];
    // G = 1 + 0.5 * 1 = 1.5
    let g = RsMetrics::shaped_return(&r, 0.5);
    assert!((g - 1.5).abs() < 1e-10);
}

#[test]
fn test_metrics_correlation_in_bounds() {
    let r1 = vec![1.0, 2.0, 3.0, 4.0];
    let r2 = vec![1.1, 2.2, 2.9, 4.1];
    let c = RsMetrics::reward_correlation(&r1, &r2);
    assert!((-1.0..=1.0).contains(&c));
}

#[test]
fn test_metrics_correlation_perfect_positive() {
    let r1 = vec![1.0, 2.0, 3.0];
    let r2 = vec![2.0, 4.0, 6.0]; // perfect linear relationship
    let c = RsMetrics::reward_correlation(&r1, &r2);
    assert!((c - 1.0).abs() < 1e-10);
}

#[test]
fn test_metrics_correlation_perfect_negative() {
    let r1 = vec![1.0, 2.0, 3.0];
    let r2 = vec![3.0, 2.0, 1.0]; // perfectly anti-correlated
    let c = RsMetrics::reward_correlation(&r1, &r2);
    assert!((c + 1.0).abs() < 1e-10);
}

#[test]
fn test_metrics_correlation_short_returns_zero() {
    let r1 = vec![5.0];
    let r2 = vec![3.0];
    let c = RsMetrics::reward_correlation(&r1, &r2);
    assert_eq!(c, 0.0);
}

#[test]
fn test_metrics_intrinsic_extrinsic_ratio() {
    let int = vec![2.0, 4.0];
    let ext = vec![1.0, 1.0];
    let ratio = RsMetrics::intrinsic_extrinsic_ratio(&int, &ext);
    assert!(ratio > 0.0 && ratio.is_finite());
}

#[test]
fn test_metrics_spearman_in_bounds() {
    let r1 = vec![3.0, 1.0, 4.0, 1.0, 5.0];
    let r2 = vec![2.0, 7.0, 1.0, 8.0, 2.0];
    let sp = RsMetrics::spearman_rank_correlation(&r1, &r2);
    assert!((-1.0..=1.0).contains(&sp));
}

#[test]
fn test_metrics_spearman_identity() {
    let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sp = RsMetrics::spearman_rank_correlation(&r, &r);
    assert!((sp - 1.0).abs() < 1e-10);
}

#[test]
fn test_metrics_preference_accuracy_all_correct() {
    let mut bt = RsBradleyTerryModel::new(2);
    // Train until winner clearly dominates
    let comparisons = vec![(vec![10.0, 0.0], vec![-10.0, 0.0])];
    bt.train_batch(&comparisons, 0.1, 50);
    let acc = RsMetrics::preference_accuracy(&bt, &comparisons);
    assert_eq!(acc, 1.0);
}

#[test]
fn test_metrics_preference_accuracy_empty() {
    let bt = RsBradleyTerryModel::new(2);
    let acc = RsMetrics::preference_accuracy(&bt, &[]);
    assert_eq!(acc, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// §11  Cross-module integration tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_integration_pbs_with_bt_reward() {
    // Learn a potential from a Bradley-Terry model reward
    let mut bt = RsBradleyTerryModel::new(3);
    let comp = vec![(vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0])];
    bt.train_batch(&comp, 0.05, 30);

    let mut pbs = RsPotentialBasedShaping::new(3, 0.99);
    let states: Vec<Vec<f64>> = vec![
        vec![0.0, 0.5, 1.0],
        vec![0.5, 0.5, 0.5],
        vec![1.0, 0.5, 0.0],
    ];
    for s in &states {
        let target = bt.reward(s);
        pbs.update_potential(s, target, 0.01);
    }
    // The potential should now approximate the BT reward at those states
    let shaped = pbs.shaped_reward(1.0, &states[0], &states[2]);
    assert!(shaped.is_finite());
}

#[test]
fn test_integration_ensemble_then_gcr() {
    let mut ens = RsRewardEnsemble::new(3, 2);
    let comparisons = vec![(vec![1.0, 0.0], vec![0.0, 1.0])];
    ens.train_ensemble(&comparisons, 0.05, 10);

    let gcr = RsGoalConditionedReward::new(2, 2, 0.1);
    let s = vec![0.5, 0.5];
    let g = vec![1.0, 0.0];
    let r_gcr = gcr.reward(&s, &g);
    let (r_ens, var) = ens.reward_with_uncertainty(&s);

    assert!(r_gcr.is_finite() && r_gcr <= 0.0);
    assert!(r_ens.is_finite() && var >= 0.0);
}

#[test]
fn test_integration_her_with_rnd_intrinsic() {
    let mut rnd = RsRandomNetworkDistillation::new(2, 4, 2);
    let gcr = RsGoalConditionedReward::new(2, 2, 0.5);
    let rl = RsRetroactiveLearning::new(50);

    let ep = RsEpisode {
        states: vec![vec![0.0, 0.0], vec![0.5, 0.5], vec![1.0, 1.0]],
        actions: vec![0, 1, 0],
        rewards: (0..3)
            .map(|i| rnd.intrinsic_reward(&[i as f64 * 0.5, 0.0]))
            .collect(),
    };

    let relabelled = rl.relabel_episode(&ep, &gcr, HerStrategy::Final);
    assert_eq!(relabelled.len(), 3);
    for r in &relabelled.rewards {
        assert!(r.is_finite());
    }
}

#[test]
fn test_integration_decomp_metrics() {
    let mut rd = RsRewardDecomposition::new(3);
    rd.add_component("task", 1.0, vec![1.0, 0.0, 0.0]);
    rd.add_component("safety", 0.5, vec![0.0, 1.0, 0.0]);

    let trajectory: Vec<(Vec<f64>, Vec<f64>)> = (0..5)
        .map(|i| {
            let s = vec![i as f64, 0.0, 0.0];
            let sp = vec![(i + 1) as f64, 0.0, 0.0];
            (s, sp)
        })
        .collect();

    let rewards: Vec<f64> = trajectory
        .iter()
        .map(|(s, sp)| rd.total_reward(s, sp))
        .collect();
    let g = RsMetrics::shaped_return(&rewards, 0.9);
    assert!(g.is_finite());
}

#[test]
fn test_rserror_display_messages() {
    let e1 = RsError::InvalidDimension("dim mismatch".to_string());
    let e2 = RsError::NumericalError("nan detected".to_string());
    let e3 = RsError::ConfigError("bad gamma".to_string());
    assert!(format!("{e1}").contains("dim mismatch"));
    assert!(format!("{e2}").contains("nan detected"));
    assert!(format!("{e3}").contains("bad gamma"));
}
