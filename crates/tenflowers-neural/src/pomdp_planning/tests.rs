//! Tests for the POMDP planning module

#[cfg(test)]
mod tests {
    use scirs2_core::random::rngs::StdRng;
    use scirs2_core::random::{Rng, SeedableRng};
    use scirs2_core::RngExt;

    use crate::pomdp_planning::{
        AlphaVector, BeliefMdpSolver, BeliefState, BtNode, FibAlgorithm, OnlineBeliefTreeSearch,
        PbviSolver, PerseusAlgorithm, PomcpSolver, PomdpGenerator, PomdpMetrics, PomdpModel,
        PomdpPolicyGraph, QmdpApproximation, SarsopAlgorithm,
    };

    // ─── PomdpModel tests ─────────────────────────────────────────────────────

    #[test]
    fn test_pomdp_model_new() {
        let m = PomdpModel::new(3, 2, 4, 0.9);
        assert_eq!(m.n_states, 3);
        assert_eq!(m.n_actions, 2);
        assert_eq!(m.n_obs, 4);
        assert!((m.gamma - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_pomdp_model_set_get() {
        let mut m = PomdpModel::new(2, 2, 2, 0.9);
        m.set_transition(0, 0, 1, 0.7);
        m.set_transition(0, 0, 0, 0.3);
        assert!((m.transition[0][0][1] - 0.7).abs() < 1e-10);
        assert!((m.transition[0][0][0] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_tiger_model_transitions() {
        let tiger = PomdpGenerator::tiger_problem();
        // Listen action: state stays unchanged
        assert!((tiger.transition[0][0][0] - 1.0).abs() < 1e-10);
        assert!((tiger.transition[1][0][1] - 1.0).abs() < 1e-10);
        // Open-Left: uniform reset
        assert!((tiger.transition[0][1][0] - 0.5).abs() < 1e-10);
        assert!((tiger.transition[0][1][1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_tiger_model_observations() {
        let tiger = PomdpGenerator::tiger_problem();
        // Listen action: hear correctly with 0.85
        assert!((tiger.observation[0][0][0] - 0.85).abs() < 1e-10);
        assert!((tiger.observation[0][1][1] - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_tiger_model_rewards() {
        let tiger = PomdpGenerator::tiger_problem();
        assert!((tiger.reward[0][1] - (-100.0)).abs() < 1e-10); // open-left when tiger left
        assert!((tiger.reward[1][1] - 10.0).abs() < 1e-10); // open-left when tiger right
    }

    // ─── BeliefState tests ────────────────────────────────────────────────────

    #[test]
    fn test_belief_uniform() {
        let b = BeliefState::uniform(4);
        assert_eq!(b.prob.len(), 4);
        let sum: f64 = b.prob.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!((b.prob[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_belief_from_vec_normalizes() {
        let b = BeliefState::from_vec(vec![2.0, 2.0, 4.0]);
        let sum: f64 = b.prob.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!((b.prob[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_belief_entropy_uniform() {
        let b = BeliefState::uniform(4);
        let h = b.entropy();
        let expected = (4.0f64).ln();
        assert!(
            (h - expected).abs() < 1e-8,
            "entropy={h}, expected={expected}"
        );
    }

    #[test]
    fn test_belief_entropy_point_mass() {
        let b = BeliefState::point_mass(4, 2);
        let h = b.entropy();
        assert!(h < 1e-10, "point mass should have entropy ~0, got {h}");
    }

    #[test]
    fn test_belief_most_likely_state() {
        let b = BeliefState::from_vec(vec![0.1, 0.7, 0.2]);
        assert_eq!(b.most_likely_state(), 1);
    }

    #[test]
    fn test_belief_update_tiger() {
        let tiger = PomdpGenerator::tiger_problem();
        let b = BeliefState::uniform(2);
        // Update with listen (a=0) and hear-left (o=0)
        let b_new = b.update(0, 0, &tiger);
        assert!(
            b_new.prob[0] > b_new.prob[1],
            "hearing left should increase P(tiger-left)"
        );
        let sum: f64 = b_new.prob.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_belief_update_normalizes() {
        let tiger = PomdpGenerator::tiger_problem();
        let b = BeliefState::uniform(2);
        for a in 0..tiger.n_actions {
            for o in 0..tiger.n_obs {
                let b_new = b.update(a, o, &tiger);
                let sum: f64 = b_new.prob.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-8,
                    "belief sum={sum} for a={a}, o={o}"
                );
            }
        }
    }

    #[test]
    fn test_belief_sample() {
        let mut rng = StdRng::seed_from_u64(123);
        let b = BeliefState::from_vec(vec![0.9, 0.1]);
        let mut counts = [0usize; 2];
        for _ in 0..1000 {
            let s = b.sample(&mut rng);
            counts[s] += 1;
        }
        assert!(
            counts[0] > 800,
            "expected ~900 samples at state 0, got {}",
            counts[0]
        );
    }

    #[test]
    fn test_belief_distance() {
        let b1 = BeliefState::from_vec(vec![1.0, 0.0]);
        let b2 = BeliefState::from_vec(vec![0.0, 1.0]);
        assert!((b1.distance(&b2) - 2.0).abs() < 1e-10);
        assert!((b1.distance(&b1) - 0.0).abs() < 1e-10);
    }

    // ─── AlphaVector tests ────────────────────────────────────────────────────

    #[test]
    fn test_alpha_vector_value() {
        let av = AlphaVector::new(0, vec![1.0, 2.0, 3.0]);
        let b = BeliefState::from_vec(vec![0.5, 0.3, 0.2]);
        let expected = 1.0 * 0.5 + 2.0 * 0.3 + 3.0 * 0.2;
        assert!((av.value(&b) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_alpha_vector_max_value() {
        let alphas = vec![
            AlphaVector::new(0, vec![3.0, 0.0]),
            AlphaVector::new(1, vec![0.0, 3.0]),
        ];
        let b = BeliefState::from_vec(vec![0.9, 0.1]);
        let (val, idx) = AlphaVector::max_value(&alphas, &b);
        assert_eq!(idx, 0);
        assert!((val - 2.7).abs() < 1e-10);
    }

    // ─── QMDP tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_qmdp_tiger_gives_alphas() {
        let tiger = PomdpGenerator::tiger_problem();
        let alphas = QmdpApproximation::compute_qmdp(tiger);
        assert_eq!(alphas.len(), 3); // one per action
        for alpha in &alphas {
            assert_eq!(alpha.coeffs.len(), 2);
        }
    }

    #[test]
    fn test_qmdp_best_action_listen_at_uniform() {
        let tiger = PomdpGenerator::tiger_problem();
        let solver = QmdpApproximation::new(tiger);
        let b = BeliefState::uniform(2);
        let a = solver.best_action(&b);
        // At uniform belief, listening is typically best for Tiger
        assert!(a < 3);
    }

    #[test]
    fn test_qmdp_new_computes_q_values() {
        let tiger = PomdpGenerator::tiger_problem();
        let solver = QmdpApproximation::new(tiger);
        // Q values should be finite
        for row in &solver.q_values {
            for &q in row {
                assert!(q.is_finite(), "Q value should be finite, got {q}");
            }
        }
    }

    // ─── FIB tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_fib_tiger_produces_alphas() {
        let tiger = PomdpGenerator::tiger_problem();
        let alphas = FibAlgorithm::compute_fib(tiger, 10);
        assert_eq!(alphas.len(), 3); // one per action
    }

    #[test]
    fn test_fib_best_action() {
        let tiger = PomdpGenerator::tiger_problem();
        let fib = FibAlgorithm::new(tiger);
        let b = BeliefState::uniform(2);
        let a = fib.best_action(&b);
        assert!(a < 3);
    }

    // ─── PBVI tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_pbvi_tiger_solves() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut solver = PbviSolver::new(tiger);
        let mut rng = StdRng::seed_from_u64(42);
        solver.sample_belief_points(5, 3, &mut rng);
        let alphas = solver.solve(5, 1e-4).to_vec();
        assert!(!alphas.is_empty());
        for av in &alphas {
            assert_eq!(av.coeffs.len(), 2);
        }
    }

    #[test]
    fn test_pbvi_best_action() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut solver = PbviSolver::new(tiger);
        let mut rng = StdRng::seed_from_u64(7);
        solver.sample_belief_points(3, 3, &mut rng);
        solver.solve(3, 1e-3);
        let b = BeliefState::uniform(2);
        let a = solver.best_action(&b);
        assert!(a < 3);
    }

    #[test]
    fn test_pbvi_belief_sampling() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut solver = PbviSolver::new(tiger);
        let mut rng = StdRng::seed_from_u64(99);
        solver.sample_belief_points(5, 4, &mut rng);
        assert!(!solver.belief_points.is_empty());
        for b in &solver.belief_points {
            let sum: f64 = b.prob.iter().sum();
            assert!((sum - 1.0).abs() < 1e-8);
        }
    }

    // ─── Perseus tests ────────────────────────────────────────────────────────

    #[test]
    fn test_perseus_tiger_solves() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut solver = PerseusAlgorithm::new(tiger);
        let mut rng = StdRng::seed_from_u64(11);
        solver.sample_beliefs(5, &mut rng);
        let alphas = solver.solve(5, &mut rng).to_vec();
        assert!(!alphas.is_empty());
    }

    #[test]
    fn test_perseus_best_action() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut solver = PerseusAlgorithm::new(tiger);
        let mut rng = StdRng::seed_from_u64(22);
        solver.sample_beliefs(3, &mut rng);
        solver.solve(3, &mut rng);
        let b = BeliefState::from_vec(vec![0.85, 0.15]);
        let a = solver.best_action(&b);
        assert!(a < 3);
    }

    // ─── SARSOP tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_sarsop_tiger_produces_alphas() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut solver = SarsopAlgorithm::new(tiger, 0.1);
        let alphas = solver.solve(0.1, 10).to_vec();
        assert!(!alphas.is_empty());
    }

    #[test]
    fn test_sarsop_lower_bound_nonempty() {
        let tiger = PomdpGenerator::tiger_problem();
        let solver = SarsopAlgorithm::new(tiger, 0.5);
        assert!(!solver.lower_bound.is_empty());
    }

    // ─── POMCP tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_pomcp_tiger_returns_valid_action() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut solver = PomcpSolver::new(tiger, 1.0, 5);
        let mut rng = StdRng::seed_from_u64(55);
        let particles = vec![0usize, 1, 0, 1, 0, 1]; // uniform particle set
        let a = solver.search(&particles, 50, &mut rng);
        assert!(a < 3);
    }

    #[test]
    fn test_pomcp_root_visits_increase() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut solver = PomcpSolver::new(tiger, 1.0, 3);
        let mut rng = StdRng::seed_from_u64(77);
        let particles = vec![0usize, 1, 0, 1];
        solver.search(&particles, 20, &mut rng);
        assert!(solver.root.visits > 0);
    }

    #[test]
    fn test_pomcp_reinvigorate_particles() {
        let tiger = PomdpGenerator::tiger_problem();
        let solver = PomcpSolver::new(tiger, 1.0, 3);
        let mut rng = StdRng::seed_from_u64(33);
        let mut particles = vec![0usize; 20];
        solver.reinvigorate_particles(&mut particles, 0.1, &mut rng);
        // Some particles may have changed
        assert_eq!(particles.len(), 20);
    }

    // ─── Policy Graph tests ───────────────────────────────────────────────────

    #[test]
    fn test_policy_graph_new() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut rng = StdRng::seed_from_u64(1);
        let pg = PomdpPolicyGraph::new(tiger, 4, &mut rng);
        assert_eq!(pg.nodes.len(), 4);
        for node in &pg.nodes {
            assert!(node.action < 3);
            assert_eq!(node.transitions.len(), 2); // 2 observations
        }
    }

    #[test]
    fn test_policy_graph_evaluate() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut rng = StdRng::seed_from_u64(2);
        let mut pg = PomdpPolicyGraph::new(tiger, 3, &mut rng);
        pg.evaluate_policy_graph(20, 1e-4);
        // Values should be finite
        for row in &pg.node_values {
            for &v in row {
                assert!(v.is_finite(), "value should be finite, got {v}");
            }
        }
    }

    #[test]
    fn test_policy_graph_improve() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut rng = StdRng::seed_from_u64(3);
        let mut pg = PomdpPolicyGraph::new(tiger, 3, &mut rng);
        pg.evaluate_policy_graph(10, 1e-3);
        pg.improve_policy_graph(&mut rng);
        // Should still have same number of nodes
        assert_eq!(pg.nodes.len(), 3);
    }

    #[test]
    fn test_policy_graph_action_at() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut rng = StdRng::seed_from_u64(4);
        let pg = PomdpPolicyGraph::new(tiger, 2, &mut rng);
        let a = pg.action_at(0);
        assert!(a < 3);
        let a2 = pg.action_at(100); // out of bounds → 0
        assert_eq!(a2, 0);
    }

    #[test]
    fn test_policy_graph_next_node() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut rng = StdRng::seed_from_u64(5);
        let pg = PomdpPolicyGraph::new(tiger, 3, &mut rng);
        let next = pg.next_node(0, 0);
        assert!(next < 3);
    }

    // ─── Belief MDP tests ─────────────────────────────────────────────────────

    #[test]
    fn test_belief_mdp_enumerate() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut solver = BeliefMdpSolver::new(tiger);
        let mut rng = StdRng::seed_from_u64(9);
        solver.enumerate_beliefs(5, 5, &mut rng);
        assert!(!solver.beliefs.is_empty());
        for b in &solver.beliefs {
            let sum: f64 = b.prob.iter().sum();
            assert!((sum - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_belief_mdp_solve() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut solver = BeliefMdpSolver::new(tiger);
        let mut rng = StdRng::seed_from_u64(10);
        solver.enumerate_beliefs(3, 3, &mut rng);
        solver.solve(10, 1e-3);
        assert_eq!(solver.policy.len(), solver.beliefs.len());
        for &a in &solver.policy {
            assert!(a < 3);
        }
    }

    #[test]
    fn test_belief_mdp_best_action() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut solver = BeliefMdpSolver::new(tiger);
        let mut rng = StdRng::seed_from_u64(11);
        solver.enumerate_beliefs(3, 3, &mut rng);
        solver.solve(5, 1e-3);
        let b = BeliefState::uniform(2);
        let a = solver.best_action(&b);
        assert!(a < 3);
    }

    // ─── Online Belief Tree Search tests ─────────────────────────────────────

    #[test]
    fn test_ao_star_search_tiger() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut searcher = OnlineBeliefTreeSearch::new(tiger, 3);
        let b = BeliefState::uniform(2);
        let a = searcher.search(b, 10);
        assert!(a < 3);
    }

    #[test]
    fn test_ao_star_expand_node() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut searcher = OnlineBeliefTreeSearch::new(tiger, 3);
        let b = BeliefState::uniform(2);
        let root_val = searcher.heuristic_value(&b);
        searcher.nodes.push(BtNode {
            belief: b,
            depth: 0,
            value: root_val,
            expanded: false,
            children: Vec::new(),
        });
        searcher.expand_node(0);
        assert!(searcher.nodes[0].expanded);
        assert!(!searcher.nodes[0].children.is_empty());
    }

    #[test]
    fn test_ao_star_best_action() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut searcher = OnlineBeliefTreeSearch::new(tiger, 2);
        let b = BeliefState::uniform(2);
        searcher.search(b, 5);
        // Best action should be valid
        let a = searcher.best_action(0);
        assert!(a < 3);
    }

    // ─── PomdpMetrics tests ───────────────────────────────────────────────────

    #[test]
    fn test_expected_discounted_reward_listen_policy() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut rng = StdRng::seed_from_u64(42);
        // Always listen policy
        let listen_policy = |_b: &BeliefState| -> usize { 0 };
        let reward =
            PomdpMetrics::expected_discounted_reward(listen_policy, &tiger, 50, 20, &mut rng);
        // Listening always costs -1, so reward should be around -1/(1-0.95) = -20
        assert!(
            reward < 0.0,
            "listen policy should have negative reward, got {reward}"
        );
    }

    #[test]
    fn test_belief_entropy_metric() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut rng = StdRng::seed_from_u64(7);
        let entropy = PomdpMetrics::belief_state_entropy(&tiger, 10, &mut rng);
        assert!(
            entropy >= 0.0,
            "entropy should be non-negative, got {entropy}"
        );
        assert!(
            entropy.is_finite(),
            "entropy should be finite, got {entropy}"
        );
    }

    #[test]
    fn test_policy_loss_vs_optimal() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut rng = StdRng::seed_from_u64(66);
        // Compare listen vs open-left
        let policy_listen = |_b: &BeliefState| -> usize { 0 };
        let policy_open = |_b: &BeliefState| -> usize { 1 };
        let loss = PomdpMetrics::policy_loss_vs_optimal(
            policy_listen,
            policy_open,
            &tiger,
            20,
            10,
            &mut rng,
        );
        assert!(loss.is_finite());
    }

    // ─── PomdpGenerator tests ─────────────────────────────────────────────────

    #[test]
    fn test_tiger_problem_dimensions() {
        let tiger = PomdpGenerator::tiger_problem();
        assert_eq!(tiger.n_states, 2);
        assert_eq!(tiger.n_actions, 3);
        assert_eq!(tiger.n_obs, 2);
    }

    #[test]
    fn test_grid_maze_pomdp_dimensions() {
        let maze = PomdpGenerator::grid_maze_pomdp(3, 4);
        assert_eq!(maze.n_states, 12);
        assert_eq!(maze.n_actions, 4);
        assert_eq!(maze.n_obs, 4);
    }

    #[test]
    fn test_grid_maze_transitions_valid() {
        let maze = PomdpGenerator::grid_maze_pomdp(2, 2);
        // Each state×action should have transition sum 1.0
        for s in 0..maze.n_states {
            for a in 0..maze.n_actions {
                let sum: f64 = maze.transition[s][a].iter().sum();
                assert!((sum - 1.0).abs() < 1e-8, "T[{s}][{a}] sums to {sum:.6}");
            }
        }
    }

    #[test]
    fn test_grid_maze_observations_valid() {
        let maze = PomdpGenerator::grid_maze_pomdp(2, 2);
        for a in 0..maze.n_actions {
            for sp in 0..maze.n_states {
                let sum: f64 = maze.observation[a][sp].iter().sum();
                assert!((sum - 1.0).abs() < 1e-8, "Z[{a}][{sp}] sums to {sum:.6}");
            }
        }
    }

    #[test]
    fn test_rock_sample_dimensions() {
        let rs = PomdpGenerator::rock_sample_pomdp(2);
        // 4 positions × 4 rock configs = 16 states
        assert_eq!(rs.n_states, 16);
        assert_eq!(rs.n_actions, 5); // left, right, sample, sense_0, sense_1
        assert_eq!(rs.n_obs, 3);
    }

    #[test]
    fn test_rock_sample_transitions_valid() {
        let rs = PomdpGenerator::rock_sample_pomdp(2);
        for s in 0..rs.n_states {
            for a in 0..rs.n_actions {
                let sum: f64 = rs.transition[s][a].iter().sum();
                assert!((sum - 1.0).abs() < 1e-8, "T[{s}][{a}] sums to {sum:.6}");
            }
        }
    }

    // ─── Integration tests ────────────────────────────────────────────────────

    #[test]
    fn test_qmdp_then_belief_update() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut b = BeliefState::uniform(2);
        // Simulate 5 listen steps (action=0) with hear-left observation (o=0)
        // P(tiger-left | hear-left^k) grows with each listen step
        for _ in 0..5 {
            b = b.update(0, 0, &tiger); // listen action=0, hear-left obs=0
        }
        // After 5 "hear-left" observations, P(tiger-left) should be well above 0.5
        assert!(
            b.prob[0] > 0.9,
            "P(tiger-left|5 hear-left)={:.4}",
            b.prob[0]
        );
    }

    #[test]
    fn test_pbvi_then_pomcp() {
        let tiger = PomdpGenerator::tiger_problem();
        // PBVI
        let mut pbvi = PbviSolver::new(tiger.clone());
        let mut rng = StdRng::seed_from_u64(100);
        pbvi.sample_belief_points(3, 3, &mut rng);
        pbvi.solve(3, 1e-3);
        // POMCP
        let mut pomcp = PomcpSolver::new(tiger, 1.0, 5);
        let particles = vec![0usize, 1, 0, 1, 0, 1, 0, 1];
        let a = pomcp.search(&particles, 30, &mut rng);
        assert!(a < 3);
    }

    #[test]
    fn test_full_pipeline_tiger() {
        let tiger = PomdpGenerator::tiger_problem();
        let mut rng = StdRng::seed_from_u64(200);

        // Build PBVI policy
        let mut pbvi = PbviSolver::new(tiger.clone());
        pbvi.sample_belief_points(5, 5, &mut rng);
        pbvi.solve(5, 1e-3);

        // Simulate episode with PBVI policy
        let mut b = BeliefState::uniform(2);
        let mut _s = b.sample(&mut rng);
        let mut total_r = 0.0;

        for step in 0..20 {
            let a = pbvi.best_action(&b);
            total_r += tiger.gamma.powi(step) * tiger.reward[_s][a];

            // Simulate
            let u: f64 = rng.random();
            let mut cum = 0.0;
            let mut sp = 1;
            for (spp, &p) in tiger.transition[_s][a].iter().enumerate() {
                cum += p;
                if u <= cum {
                    sp = spp;
                    break;
                }
            }
            let uo: f64 = rng.random();
            let mut o = 1;
            let mut cum_o = 0.0;
            for (oo, &p) in tiger.observation[a][sp].iter().enumerate() {
                cum_o += p;
                if uo <= cum_o {
                    o = oo;
                    break;
                }
            }
            b = b.update(a, o, &tiger);
            _s = sp;
        }

        assert!(total_r.is_finite());
    }
}
