//! Tests for the cooperative_game_theory module.

#[cfg(test)]
mod tests {
    use super::super::*;

    fn prisoners_dilemma() -> CgtGame {
        CgtGame::new(
            vec![2, 2],
            vec![vec![3.0, 0.0, 5.0, 1.0], vec![3.0, 5.0, 0.0, 1.0]],
        )
        .expect("CgtGame creation should succeed")
    }

    fn matching_pennies() -> CgtGame {
        CgtGame::new(
            vec![2, 2],
            vec![vec![1.0, -1.0, -1.0, 1.0], vec![-1.0, 1.0, 1.0, -1.0]],
        )
        .expect("CgtGame creation should succeed")
    }

    fn three_player_game() -> CgtGame {
        let payoffs: Vec<Vec<f64>> = (0..3)
            .map(|p| {
                (0..8)
                    .map(|i| (p as f64 + 1.0) * (i as f64 + 1.0) / 8.0)
                    .collect()
            })
            .collect();
        CgtGame::new(vec![2, 2, 2], payoffs).expect("CgtGame creation should succeed")
    }

    fn small_coop_game(n: usize) -> CgtCooperativeGame {
        let mut game = CgtCooperativeGame::new(n);
        let total = 1u64 << n;
        for s in 1..total {
            let cnt = s.count_ones() as f64;
            game.set_coalition_value(s, cnt * cnt);
        }
        game
    }

    // ── CgtGame tests ─────────────────────────────────────────────────────

    #[test]
    fn test_cgt_game_creation() {
        let g = prisoners_dilemma();
        assert_eq!(g.num_players(), 2);
        assert_eq!(g.num_actions(0).expect("num_actions should succeed"), 2);
        assert_eq!(g.num_actions(1).expect("num_actions should succeed"), 2);
    }

    #[test]
    fn test_cgt_game_payoff() {
        let g = prisoners_dilemma();
        assert!((g.payoff(0, &[0, 0]).expect("payoff should succeed") - 3.0).abs() < 1e-9);
        assert!((g.payoff(0, &[1, 0]).expect("payoff should succeed") - 5.0).abs() < 1e-9);
        assert!((g.payoff(1, &[0, 1]).expect("payoff should succeed") - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_cgt_game_invalid_player() {
        let g = prisoners_dilemma();
        assert!(g.num_actions(5).is_err());
        assert!(g.payoff(5, &[0, 0]).is_err());
    }

    #[test]
    fn test_cgt_game_all_profiles() {
        let g = prisoners_dilemma();
        let profiles = g.all_profiles();
        assert_eq!(profiles.len(), 4);
        assert!(profiles.contains(&vec![0, 0]));
        assert!(profiles.contains(&vec![1, 1]));
    }

    #[test]
    fn test_cgt_game_three_player() {
        let g = three_player_game();
        assert_eq!(g.num_players(), 3);
        let profiles = g.all_profiles();
        assert_eq!(profiles.len(), 8);
    }

    #[test]
    fn test_cgt_game_invalid_payoff_table() {
        let result = CgtGame::new(vec![2, 2], vec![vec![1.0]; 2]);
        assert!(result.is_err());
    }

    // ── NashEquilibriumSolver tests ────────────────────────────────────────

    #[test]
    fn test_nash_solver_prisoners_dilemma() {
        let g = prisoners_dilemma();
        let ne = NashEquilibriumSolver::solve_2player(&g).expect("Nash solver should succeed");
        assert!(!ne.is_empty());
        let has_dd = ne.iter().any(|(x, y)| x[1] > 0.9 && y[1] > 0.9);
        assert!(has_dd, "Expected pure NE at (D,D)");
    }

    #[test]
    fn test_nash_solver_matching_pennies_mixed() {
        let g = matching_pennies();
        let ne = NashEquilibriumSolver::solve_2player(&g).expect("Nash solver should succeed");
        assert!(!ne.is_empty());
        let (x, y) = &ne[0];
        assert!((x[0] - 0.5).abs() < 0.2, "x[0] ≈ 0.5");
        assert!((y[0] - 0.5).abs() < 0.2, "y[0] ≈ 0.5");
    }

    #[test]
    fn test_nash_solver_probability_sums() {
        let g = prisoners_dilemma();
        let ne = NashEquilibriumSolver::solve_2player(&g).expect("Nash solver should succeed");
        for (x, y) in &ne {
            let sx: f64 = x.iter().sum();
            let sy: f64 = y.iter().sum();
            assert!((sx - 1.0).abs() < 1e-6, "x must sum to 1");
            assert!((sy - 1.0).abs() < 1e-6, "y must sum to 1");
        }
    }

    #[test]
    fn test_nash_solver_wrong_player_count() {
        let g = three_player_game();
        assert!(NashEquilibriumSolver::solve_2player(&g).is_err());
    }

    #[test]
    fn test_nash_find_support_iterated_dominance() {
        let g = prisoners_dilemma();
        let strategies = NashEquilibriumSolver::find_nash_support(&g).expect("find_nash_support should succeed");
        assert_eq!(strategies.len(), 2);
        assert!(strategies[0][1] >= strategies[0][0] - 1e-9);
    }

    // ── CfrSolver tests ───────────────────────────────────────────────────

    #[test]
    fn test_cfr_solver_creation() {
        let solver = CfrSolver::new();
        assert_eq!(solver.n_iterations, 0);
        assert!(solver.nodes.is_empty());
    }

    #[test]
    fn test_cfr_node_uniform_initial() {
        let node = CfrNode::new(3);
        let strat = node.get_strategy(1.0);
        assert_eq!(strat.len(), 3);
        assert!((strat.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        for &s in &strat {
            assert!((s - 1.0 / 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_cfr_node_average_strategy() {
        let mut node = CfrNode::new(2);
        node.strategy_sum = vec![3.0, 1.0];
        let avg = node.get_average_strategy();
        assert!((avg[0] - 0.75).abs() < 1e-9);
        assert!((avg[1] - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_cfr_kuhn_poker_training() {
        let mut solver = CfrSolver::new();
        let exploitability = solver.train_kuhn_poker(1000);
        assert!(
            exploitability < 1.0,
            "exploitability should be small after training"
        );
        assert!(solver.n_iterations == 1000);
        assert!(!solver.nodes.is_empty());
    }

    #[test]
    fn test_cfr_node_regret_update() {
        let mut node = CfrNode::new(2);
        node.update(&[1.0, 0.0], 0.5, 1.0);
        assert!(node.regret_sum[0] > 0.0);
    }

    // ── RegretMinimizer tests ─────────────────────────────────────────────

    #[test]
    fn test_regret_matching_creation() {
        let rm = RegretMinimizer::new(3, RegretAlgorithm::RegretMatching, 0.1);
        assert_eq!(rm.n_actions, 3);
        let d = rm.current_distribution();
        assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_regret_matching_update() {
        let mut rm = RegretMinimizer::new(3, RegretAlgorithm::RegretMatching, 0.1);
        rm.observe_utility(&[1.0, 0.0, 0.5]);
        let d = rm.current_distribution();
        assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_regret_matching_plus() {
        let mut rm = RegretMinimizer::new(2, RegretAlgorithm::RegretMatchingPlus, 0.1);
        for _ in 0..10 {
            rm.observe_utility(&[-1.0, 1.0]);
        }
        let d = rm.current_distribution();
        assert!(d[1] > d[0]);
    }

    #[test]
    fn test_hedge_algorithm() {
        let mut rm = RegretMinimizer::new(3, RegretAlgorithm::Hedge, 0.5);
        rm.observe_utility(&[1.0, 0.0, 0.5]);
        let d = rm.current_distribution();
        assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_online_mirror_descent() {
        let mut rm = RegretMinimizer::new(3, RegretAlgorithm::OnlineMirrorDescent, 1.0);
        rm.observe_utility(&[0.5, 0.2, 0.8]);
        let d = rm.current_distribution();
        assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_regret_bound_scaling() {
        let mut rm = RegretMinimizer::new(4, RegretAlgorithm::RegretMatching, 0.1);
        for _ in 0..100 {
            rm.observe_utility(&[1.0, 0.5, 0.0, 0.8]);
        }
        let bound = rm.regret_bound();
        assert!(bound > 0.0);
    }

    // ── CgtCooperativeGame tests ──────────────────────────────────────────

    #[test]
    fn test_cooperative_game_creation() {
        let mut g = CgtCooperativeGame::new(3);
        g.set_coalition_value(0b111, 10.0);
        assert!((g.grand_coalition_value() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_coalition_value_default_zero() {
        let g = CgtCooperativeGame::new(3);
        assert!((g.compute_coalition_value(0b101) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_superadditive_game() {
        let mut g = CgtCooperativeGame::new(3);
        g.set_coalition_value(0b001, 1.0);
        g.set_coalition_value(0b010, 2.0);
        g.set_coalition_value(0b100, 3.0);
        g.set_coalition_value(0b011, 4.0);
        g.set_coalition_value(0b101, 5.0);
        g.set_coalition_value(0b110, 6.0);
        g.set_coalition_value(0b111, 10.0);
        assert!(g.is_superadditive());
    }

    #[test]
    fn test_not_superadditive_game() {
        let mut g = CgtCooperativeGame::new(2);
        g.set_coalition_value(0b01, 5.0);
        g.set_coalition_value(0b10, 5.0);
        g.set_coalition_value(0b11, 3.0);
        assert!(!g.is_superadditive());
    }

    // ── ShapleyValueCalculator tests ──────────────────────────────────────

    #[test]
    fn test_shapley_exact_3player() {
        let g = small_coop_game(3);
        let shapley = ShapleyValueCalculator::compute_exact(&g);
        assert_eq!(shapley.len(), 3);
        let total: f64 = shapley.iter().sum();
        let gv = g.grand_coalition_value();
        assert!(
            (total - gv).abs() < 1e-6,
            "Shapley values must sum to grand coalition value"
        );
    }

    #[test]
    fn test_shapley_efficiency() {
        let g = small_coop_game(4);
        assert!(CgtMetrics::shapley_efficiency(&g));
    }

    #[test]
    fn test_shapley_approx_close_to_exact() {
        let g = small_coop_game(3);
        let exact = ShapleyValueCalculator::compute_exact(&g);
        let approx = ShapleyValueCalculator::compute_approx(&g, 5000);
        for (e, a) in exact.iter().zip(approx.iter()) {
            assert!(
                (e - a).abs() < 0.5,
                "Approximate Shapley should be close to exact"
            );
        }
    }

    #[test]
    fn test_shapley_dummy_player() {
        let mut g = CgtCooperativeGame::new(2);
        g.set_coalition_value(0b01, 1.0);
        g.set_coalition_value(0b10, 0.0);
        g.set_coalition_value(0b11, 1.0);
        let sv = ShapleyValueCalculator::compute_exact(&g);
        assert!(sv[1].abs() < 1e-9, "Dummy player shapley value should be 0");
        assert!(
            (sv[0] - 1.0).abs() < 1e-9,
            "Non-dummy player gets all value"
        );
    }

    // ── BanzhafIndex tests ────────────────────────────────────────────────

    #[test]
    fn test_banzhaf_index_sum_to_one() {
        let g = small_coop_game(3);
        let bz = BanzhafIndex::compute(&g);
        assert_eq!(bz.len(), 3);
        let sum: f64 = bz.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_banzhaf_weighted_voting() {
        let mut g = CgtCooperativeGame::new(3);
        let winning: Vec<(u64, f64)> = vec![(0b011, 1.0), (0b101, 1.0), (0b111, 1.0)];
        for (mask, val) in winning {
            g.set_coalition_value(mask, val);
        }
        let bz = BanzhafIndex::compute(&g);
        assert!(bz[0] >= bz[1] - 1e-9);
        assert!(bz[1] >= bz[2] - 1e-9);
    }

    // ── CoreSolver tests ──────────────────────────────────────────────────

    #[test]
    fn test_core_solver_in_core() {
        let mut g = CgtCooperativeGame::new(3);
        g.set_coalition_value(0b001, 1.0);
        g.set_coalition_value(0b010, 1.0);
        g.set_coalition_value(0b100, 1.0);
        g.set_coalition_value(0b011, 3.0);
        g.set_coalition_value(0b101, 3.0);
        g.set_coalition_value(0b110, 3.0);
        g.set_coalition_value(0b111, 6.0);
        let alloc = vec![2.0, 2.0, 2.0];
        assert!(CoreSolver::is_in_core(&g, &alloc));
    }

    #[test]
    fn test_core_solver_not_in_core() {
        let mut g = CgtCooperativeGame::new(2);
        g.set_coalition_value(0b01, 1.0);
        g.set_coalition_value(0b10, 1.0);
        g.set_coalition_value(0b11, 3.0);
        let alloc = vec![0.5, 2.5];
        assert!(!CoreSolver::is_in_core(&g, &alloc));
    }

    #[test]
    fn test_find_core_allocation_returns_valid_alloc() {
        let g = small_coop_game(3);
        let alloc = CoreSolver::find_core_allocation(&g).expect("find_core_allocation should succeed");
        assert_eq!(alloc.len(), 3);
        let sum: f64 = alloc.iter().sum();
        let gv = g.grand_coalition_value();
        assert!(
            (sum - gv).abs() < 0.1,
            "Allocation sum should be close to grand coalition value"
        );
    }

    #[test]
    fn test_nucleolus_approximation() {
        let g = small_coop_game(3);
        let nuc = CoreSolver::nucleolus_approximation(&g);
        assert_eq!(nuc.len(), 3);
        for &v in &nuc {
            assert!(v > -1e-6);
        }
    }

    // ── CorrelatedEquilibrium tests ───────────────────────────────────────

    #[test]
    fn test_correlated_equilibrium_pure_ne() {
        let g = prisoners_dilemma();
        let sigma = vec![0.0, 0.0, 0.0, 1.0];
        assert!(CorrelatedEquilibrium::is_correlated_equilibrium(&sigma, &g));
    }

    #[test]
    fn test_correlated_equilibrium_invalid_distribution() {
        let g = matching_pennies();
        let sigma = vec![0.5, 0.5, 0.0, 0.0];
        let _ = CorrelatedEquilibrium::is_correlated_equilibrium(&sigma, &g);
    }

    #[test]
    fn test_correlated_equilibrium_wrong_size() {
        let g = prisoners_dilemma();
        let sigma = vec![0.5, 0.5];
        assert!(!CorrelatedEquilibrium::is_correlated_equilibrium(
            &sigma, &g
        ));
    }

    #[test]
    fn test_find_maximal_welfare_ce() {
        let g = prisoners_dilemma();
        let ce = CorrelatedEquilibrium::find_maximal_welfare_ce(&g).expect("find_maximal_welfare_ce should succeed");
        assert_eq!(ce.len(), 4);
        let sum: f64 = ce.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "CE distribution must sum to 1");
    }

    // ── MechanismDesign tests ─────────────────────────────────────────────

    #[test]
    fn test_vcg_allocation_single_item() {
        let bids = vec![vec![10.0, 5.0], vec![8.0, 3.0]];
        let value_fn = |alloc: &[usize]| -> f64 {
            bids.iter()
                .enumerate()
                .map(|(i, b)| {
                    b.get(alloc.get(i).copied().unwrap_or(0))
                        .copied()
                        .unwrap_or(0.0)
                })
                .sum()
        };
        let (alloc, payments) = MechanismDesign::vcg_allocation(&bids, value_fn).expect("vcg_allocation should succeed");
        assert_eq!(alloc.len(), 2);
        assert_eq!(payments.len(), 2);
    }

    #[test]
    fn test_vcg_allocation_empty_bids() {
        let bids: Vec<Vec<f64>> = vec![];
        let result = MechanismDesign::vcg_allocation(&bids, |_| 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_incentive_compatibility_check() {
        let bids = vec![vec![10.0, 5.0], vec![8.0, 3.0], vec![6.0, 7.0]];
        let ic = MechanismDesign::is_incentive_compatible(&bids);
        assert!(ic);
    }

    // ── EvolutionaryGameDynamics tests ────────────────────────────────────

    #[test]
    fn test_replicator_dynamics_creation() {
        let egd = EvolutionaryGameDynamics::from_symmetric_game(SymmetricGame::RockPaperScissors)
            .expect("from_symmetric_game should succeed");
        assert_eq!(egd.population.len(), 3);
        assert!((egd.population.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_replicator_dynamics_hawk_dove() {
        let mut egd = EvolutionaryGameDynamics::from_symmetric_game(SymmetricGame::HawkDove {
            v: 2.0,
            c: 4.0,
        })
        .expect("from_symmetric_game should succeed");
        let ess = egd.find_evolutionary_stable_strategy();
        assert_eq!(ess.len(), 2);
        assert!((ess.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_replicator_dynamics_prisoners_dilemma() {
        let mut egd =
            EvolutionaryGameDynamics::from_symmetric_game(SymmetricGame::PrisonersDilemma {
                r: 3.0,
                s: 0.0,
                t: 5.0,
                p: 1.0,
            })
            .expect("from_symmetric_game should succeed");
        let ess = egd.find_evolutionary_stable_strategy();
        assert!(ess[1] > ess[0] - 0.1, "Defection should dominate");
    }

    #[test]
    fn test_replicator_dynamics_step_conserves_sum() {
        let mut egd =
            EvolutionaryGameDynamics::from_symmetric_game(SymmetricGame::RockPaperScissors)
                .expect("from_symmetric_game should succeed");
        egd.step();
        assert!((egd.population.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_replicator_dynamics_custom_matrix() {
        let mat = vec![vec![0.0, 1.0], vec![-1.0, 0.0]];
        let mut egd = EvolutionaryGameDynamics::new(mat, 0.01).expect("EvolutionaryGameDynamics creation should succeed");
        let ess = egd.run(1000);
        assert_eq!(ess.len(), 2);
    }

    // ── CgtNeuralNash tests ───────────────────────────────────────────────

    #[test]
    fn test_nash_net_forward_shape() {
        let net = NashNet::new(8, 4, 16);
        let input = vec![0.5f64; 8];
        let out = net.forward(&input);
        assert_eq!(out.len(), 4);
        assert!((out.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_nash_net_softmax_output() {
        let net = NashNet::new(8, 4, 16);
        let input = vec![1.0f64, -1.0, 0.5, 0.0, 0.8, -0.3, 0.2, 0.9];
        let out = net.forward(&input);
        assert!(out.iter().all(|&x| x >= 0.0));
        assert!((out.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_neural_nash_predict() {
        let nn = CgtNeuralNash::new(2, 2);
        let payoff = vec![1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0];
        let (x, y) = nn.predict_nash(&payoff);
        assert_eq!(x.len(), 2);
        assert_eq!(y.len(), 2);
        assert!((x.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!((y.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_neural_nash_training_runs() {
        let mut nn = CgtNeuralNash::new(2, 2);
        nn.train(10, 0.001);
    }

    // ── MeanFieldEquilibrium tests ────────────────────────────────────────

    #[test]
    fn test_mfe_coordination_game() {
        let mfe = MeanFieldEquilibrium::compute_mfe(|a, dist| dist[a], 3, 1e-6);
        assert_eq!(mfe.len(), 3);
        assert!((mfe.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mfe_congestion_game() {
        let mfe = MeanFieldEquilibrium::compute_mfe(|a, dist| 1.0 - dist[a], 3, 1e-5);
        assert_eq!(mfe.len(), 3);
        for &x in &mfe {
            assert!(x > 0.0);
        }
    }

    #[test]
    fn test_mfe_single_action() {
        let mfe = MeanFieldEquilibrium::compute_mfe(|_, _| 1.0, 1, 1e-6);
        assert!((mfe[0] - 1.0).abs() < 1e-6);
    }

    // ── CgtMetrics tests ──────────────────────────────────────────────────

    #[test]
    fn test_exploitability_matching_pennies_mixed_ne() {
        let g = matching_pennies();
        let strategies = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let exp = CgtMetrics::exploitability(&strategies, &g);
        assert!(exp.abs() < 1e-6, "Mixed NE has 0 exploitability");
    }

    #[test]
    fn test_exploitability_pure_strategy_non_ne() {
        let g = matching_pennies();
        let strategies = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let exp = CgtMetrics::exploitability(&strategies, &g);
        assert!(exp > 0.0, "Non-NE should have positive exploitability");
    }

    #[test]
    fn test_social_welfare_computation() {
        let g = prisoners_dilemma();
        let strat = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let sw = CgtMetrics::social_welfare(&strat, &g);
        assert!((sw - 6.0).abs() < 1e-9, "Both cooperate: sw = 3+3 = 6");
    }

    #[test]
    fn test_price_of_anarchy() {
        let g = prisoners_dilemma();
        let poa = CgtMetrics::price_of_anarchy(&g);
        assert!(poa >= 1.0, "PoA should be >= 1");
    }

    #[test]
    fn test_gini_coefficient_equal() {
        let payoffs = vec![1.0, 1.0, 1.0, 1.0];
        let g = CgtMetrics::gini_coefficient(&payoffs);
        assert!(g.abs() < 1e-9, "Equal payoffs → Gini = 0");
    }

    #[test]
    fn test_gini_coefficient_unequal() {
        let payoffs = vec![0.0, 0.0, 0.0, 10.0];
        let g = CgtMetrics::gini_coefficient(&payoffs);
        assert!(g > 0.5, "Very unequal payoffs → Gini > 0.5");
    }

    #[test]
    fn test_gini_coefficient_empty() {
        let payoffs: Vec<f64> = vec![];
        let g = CgtMetrics::gini_coefficient(&payoffs);
        assert_eq!(g, 0.0);
    }

    #[test]
    fn test_shapley_efficiency_verified() {
        let g = small_coop_game(3);
        assert!(CgtMetrics::shapley_efficiency(&g));
    }

    // ── Integration / additional tests ────────────────────────────────────

    #[test]
    fn test_cfr_get_average_strategy_after_training() {
        let mut solver = CfrSolver::new();
        solver.train_kuhn_poker(200);
        for node in solver.nodes.values() {
            let avg = node.get_average_strategy();
            assert!((avg.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_regret_minimizer_convergence_to_optimal() {
        let mut rm = RegretMinimizer::new(2, RegretAlgorithm::RegretMatchingPlus, 0.1);
        for _ in 0..100 {
            rm.observe_utility(&[1.0, 0.0]);
        }
        let d = rm.current_distribution();
        assert!(d[0] > 0.8, "Should mostly play action 0");
    }

    #[test]
    fn test_correlated_eq_uniform_matching_pennies() {
        let g = matching_pennies();
        let sigma = vec![0.25; 4];
        let _result = CorrelatedEquilibrium::is_correlated_equilibrium(&sigma, &g);
    }

    #[test]
    fn test_banzhaf_symmetric_game() {
        let g = small_coop_game(3);
        let bz = BanzhafIndex::compute(&g);
        assert!((bz[0] - bz[1]).abs() < 1e-9);
        assert!((bz[1] - bz[2]).abs() < 1e-9);
    }
}
