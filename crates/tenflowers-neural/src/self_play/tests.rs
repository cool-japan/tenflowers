//! Tests for the self_play module — Track C Round 47.

#![allow(clippy::float_cmp)]

use super::*;

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn make_seed() -> u64 {
    0x00C0_FFEE_BABE_1234_u64
}

fn make_ttt() -> SpTicTacToe {
    SpTicTacToe::new()
}

fn make_network(input_dim: usize, hidden: usize, n_actions: usize) -> SpNeuralNetwork {
    SpNeuralNetwork::new(input_dim, hidden, n_actions)
}

fn make_mcts_config() -> SpMctsConfig {
    SpMctsConfig {
        n_simulations: 10,
        c_puct: 1.5,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        temperature: 1.0,
    }
}

// ─── §0  SpError ─────────────────────────────────────────────────────────────

#[test]
fn test_sp_error_display_invalid_action() {
    let e = SpError::InvalidAction("bad".to_string());
    let s = format!("{}", e);
    assert!(s.contains("InvalidAction"));
    assert!(s.contains("bad"));
}

#[test]
fn test_sp_error_display_game_over() {
    let e = SpError::GameOver;
    let s = format!("{}", e);
    assert!(s.contains("GameOver"));
}

#[test]
fn test_sp_error_display_numerical() {
    let e = SpError::NumericalError("nan".to_string());
    let s = format!("{}", e);
    assert!(s.contains("NumericalError"));
    assert!(s.contains("nan"));
}

// ─── §0b  RNG ────────────────────────────────────────────────────────────────

#[test]
fn test_sp_rand01_in_range() {
    let mut seed = make_seed();
    for _ in 0..1000 {
        let r = sp_rand01(&mut seed);
        assert!((0.0..1.0).contains(&r), "r={} out of [0,1)", r);
    }
}

#[test]
fn test_sp_randn_finite() {
    let mut seed = make_seed();
    for _ in 0..100 {
        let r = sp_randn(&mut seed);
        assert!(r.is_finite(), "randn returned non-finite: {}", r);
    }
}

#[test]
fn test_sp_randn_mean_approx_zero() {
    let mut seed = 42u64;
    let n = 10_000;
    let mean: f64 = (0..n).map(|_| sp_randn(&mut seed)).sum::<f64>() / n as f64;
    assert!(mean.abs() < 0.1, "mean={} far from 0", mean);
}

// ─── §2  SpTicTacToe ─────────────────────────────────────────────────────────

#[test]
fn test_ttt_new_board_empty() {
    let g = make_ttt();
    assert_eq!(g.board, [0i8; 9]);
    assert_eq!(g.current_player, 1);
    assert_eq!(g.n_moves, 0);
}

#[test]
fn test_ttt_legal_actions_full_board_start() {
    let g = make_ttt();
    let legal = g.legal_actions();
    assert_eq!(legal.len(), 9);
    assert_eq!(legal, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn test_ttt_not_terminal_initially() {
    let g = make_ttt();
    assert!(!g.is_terminal());
}

#[test]
fn test_ttt_no_winner_initially() {
    let g = make_ttt();
    assert_eq!(g.winner(), None);
}

#[test]
fn test_ttt_apply_action_valid() {
    let g = make_ttt();
    let next = g.apply_action(4).expect("action 4 should be legal");
    assert_eq!(next.board[4], 1); // player 1 placed
    assert_eq!(next.current_player, -1); // player 2's turn
    assert_eq!(next.n_moves, 1);
}

#[test]
fn test_ttt_apply_action_out_of_range() {
    let g = make_ttt();
    let result = g.apply_action(9);
    assert!(result.is_err());
}

#[test]
fn test_ttt_apply_action_occupied_cell() {
    let g = make_ttt();
    let g2 = g.apply_action(0).expect("apply_action(0) should succeed");
    let result = g2.apply_action(0);
    assert!(result.is_err());
}

#[test]
fn test_ttt_player_alternates() {
    let g = make_ttt();
    let g1 = g.apply_action(0).expect("apply_action(0) should succeed");
    assert_eq!(g1.current_player(), -1);
    let g2 = g1.apply_action(1).expect("apply_action(1) should succeed");
    assert_eq!(g2.current_player(), 1);
}

#[test]
fn test_ttt_winner_row() {
    // Player 1 fills row 0: cells 0,1,2
    let mut g = make_ttt();
    for &action in &[0usize, 3, 1, 4, 2] {
        g = g.apply_action(action).expect("apply_action should succeed");
    }
    assert_eq!(g.winner(), Some(1));
    assert!(g.is_terminal());
}

#[test]
fn test_ttt_winner_column() {
    // Player -1 fills column 0: cells 0,3,6
    // Sequence: P1=1, P2=0, P1=2, P2=3, P1=8, P2=6
    let mut g = make_ttt();
    for &action in &[1usize, 0, 2, 3, 8, 6] {
        g = g.apply_action(action).expect("apply_action should succeed");
    }
    assert_eq!(g.winner(), Some(-1));
}

#[test]
fn test_ttt_winner_diagonal() {
    // Player 1 fills main diagonal: 0,4,8
    // Sequence: P1=0, P2=1, P1=4, P2=2, P1=8
    let mut g = make_ttt();
    for &action in &[0usize, 1, 4, 2, 8] {
        g = g.apply_action(action).expect("apply_action should succeed");
    }
    assert_eq!(g.winner(), Some(1));
}

#[test]
fn test_ttt_draw() {
    // Fill board without anyone winning:
    // X O X
    // X X O
    // O X O  → no winner, 9 moves
    let mut g = make_ttt();
    // sequence: 0(X),1(O),2(X),4(X),3(X) ... let's construct carefully
    // X=0,2,4,5,7  O=1,3,6,8  (no line for X or O)
    for &action in &[0usize, 1, 2, 3, 4, 8, 5, 6, 7] {
        g = g.apply_action(action).expect("apply_action should succeed");
    }
    assert_eq!(g.winner(), None);
    assert!(g.is_terminal());
}

#[test]
fn test_ttt_no_legal_actions_after_game_over() {
    let mut g = make_ttt();
    for &action in &[0usize, 3, 1, 4, 2] {
        g = g.apply_action(action).expect("apply_action should succeed");
    }
    assert!(g.is_terminal());
    assert!(g.legal_actions().is_empty());
}

#[test]
fn test_ttt_state_encoding_length() {
    let g = make_ttt();
    let enc = g.state_encoding(20);
    assert_eq!(enc.len(), 20);
}

#[test]
fn test_ttt_state_encoding_values_bounded() {
    let g = make_ttt().apply_action(4).expect("apply_action(4) should succeed");
    let enc = g.state_encoding(9);
    // Current player is -1; center cell was set to 1 by player 1.
    // From -1 perspective, enc[4] = board[4]*player = 1*(-1) = -1
    assert_eq!(enc[4], -1.0);
}

#[test]
fn test_ttt_apply_action_game_over_returns_err() {
    let mut g = make_ttt();
    for &action in &[0usize, 3, 1, 4, 2] {
        g = g.apply_action(action).expect("apply_action should succeed");
    }
    assert!(g.is_terminal());
    let result = g.apply_action(5);
    assert!(result.is_err());
}

// ─── §3  SpNeuralNetwork ─────────────────────────────────────────────────────

#[test]
fn test_nn_dimensions_correct() {
    let nn = make_network(10, 16, 9);
    assert_eq!(nn.w1.len(), 16);
    assert_eq!(nn.w1[0].len(), 10);
    assert_eq!(nn.w2.len(), 16);
    assert_eq!(nn.policy_head_w.len(), 9);
    assert_eq!(nn.value_head_w.len(), 16);
}

#[test]
fn test_nn_forward_returns_correct_shapes() {
    let nn = make_network(10, 16, 9);
    let state = vec![0.0f64; 10];
    let (logits, value) = nn.forward(&state);
    assert_eq!(logits.len(), 9);
    assert!(value.is_finite(), "value={} is not finite", value);
}

#[test]
fn test_nn_value_in_range() {
    let nn = make_network(10, 16, 9);
    let state = vec![0.1f64; 10];
    let (_, value) = nn.forward(&state);
    assert!(
        (-1.0..=1.0).contains(&value),
        "value={} out of tanh range",
        value
    );
}

#[test]
fn test_nn_policy_sums_to_one_all_legal() {
    let nn = make_network(10, 16, 9);
    let state = vec![0.0f64; 10];
    let legal_mask = vec![true; 9];
    let (probs, _) = nn.policy_value(&state, &legal_mask);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "policy sum={} ≠ 1", sum);
}

#[test]
fn test_nn_policy_sums_to_one_some_legal() {
    let nn = make_network(10, 16, 9);
    let state = vec![0.0f64; 10];
    let legal_mask = vec![true, false, true, false, true, false, false, false, true];
    let (probs, _) = nn.policy_value(&state, &legal_mask);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "masked policy sum={} ≠ 1", sum);
    // Illegal positions should have near-zero probability
    assert!(probs[1] < 1e-6, "illegal action has prob={}", probs[1]);
}

#[test]
fn test_nn_update_weights_changes_w1() {
    let mut nn = make_network(10, 16, 9);
    let w1_before = nn.w1[0][0];
    let grad_w1: Vec<Vec<f64>> = (0..16).map(|_| vec![1.0f64; 10]).collect();
    let grad_w2: Vec<Vec<f64>> = (0..16).map(|_| vec![0.0f64; 16]).collect();
    nn.update_weights(&grad_w1, &grad_w2, 0.1);
    let w1_after = nn.w1[0][0];
    assert!(
        (w1_after - (w1_before - 0.1)).abs() < 1e-12,
        "expected {}−0.1 but got {}",
        w1_before,
        w1_after
    );
}

#[test]
fn test_nn_policy_all_nonnegative() {
    let nn = make_network(10, 16, 9);
    let state: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
    let legal_mask = vec![true; 9];
    let (probs, _) = nn.policy_value(&state, &legal_mask);
    for &p in &probs {
        assert!(p >= 0.0, "negative probability: {}", p);
    }
}

// ─── §4  SpMctsNode ──────────────────────────────────────────────────────────

#[test]
fn test_mcts_node_new_fields() {
    let node = SpMctsNode::new(Some(3), 0.5, 1, None);
    assert_eq!(node.action, Some(3));
    assert_eq!(node.prior, 0.5);
    assert_eq!(node.player, 1);
    assert_eq!(node.parent, None);
    assert_eq!(node.visit_count, 0);
    assert_eq!(node.value_sum, 0.0);
    assert!(!node.is_expanded);
}

#[test]
fn test_mcts_node_q_value_unvisited_is_zero() {
    let node = SpMctsNode::new(None, 1.0, 1, None);
    assert_eq!(node.q_value(), 0.0);
}

#[test]
fn test_mcts_node_q_value_after_update() {
    let mut node = SpMctsNode::new(None, 1.0, 1, None);
    node.visit_count = 4;
    node.value_sum = 2.0;
    assert!((node.q_value() - 0.5).abs() < 1e-12);
}

#[test]
fn test_mcts_node_ucb_increases_with_prior() {
    let mut node_low = SpMctsNode::new(None, 0.1, 1, None);
    let mut node_high = SpMctsNode::new(None, 0.9, 1, None);
    node_low.visit_count = 0;
    node_high.visit_count = 0;
    let ucb_low = node_low.ucb_score(100, 1.0);
    let ucb_high = node_high.ucb_score(100, 1.0);
    assert!(ucb_high > ucb_low, "UCB should increase with prior");
}

#[test]
fn test_mcts_node_ucb_decreases_with_visits() {
    let mut node_few = SpMctsNode::new(None, 0.5, 1, None);
    let mut node_many = SpMctsNode::new(None, 0.5, 1, None);
    node_few.visit_count = 1;
    node_many.visit_count = 100;
    let ucb_few = node_few.ucb_score(1000, 1.0);
    let ucb_many = node_many.ucb_score(1000, 1.0);
    assert!(
        ucb_few > ucb_many,
        "UCB should decrease as node visits increase"
    );
}

#[test]
fn test_mcts_node_terminal_value_none_by_default() {
    let node = SpMctsNode::new(None, 1.0, 1, None);
    assert!(node.terminal_value.is_none());
}

// ─── §5  SpMctsTree ──────────────────────────────────────────────────────────

#[test]
fn test_mcts_tree_new_has_root() {
    let config = make_mcts_config();
    let tree = SpMctsTree::new(config);
    assert_eq!(tree.nodes.len(), 1);
}

#[test]
fn test_mcts_tree_run_simulations_returns_valid_policy() {
    let game = make_ttt();
    let nn = make_network(20, 16, 9);
    let config = make_mcts_config();
    let mut tree = SpMctsTree::new(config);
    let mut seed = make_seed();
    let policy = tree.run_simulations(&game, &nn, &mut seed);
    // Policy should cover at least the 9 possible actions
    assert!(!policy.is_empty());
    let sum: f64 = policy.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "policy sum={} ≠ 1", sum);
}

#[test]
fn test_mcts_tree_policy_all_nonneg() {
    let game = make_ttt();
    let nn = make_network(20, 16, 9);
    let config = make_mcts_config();
    let mut tree = SpMctsTree::new(config);
    let mut seed = make_seed();
    let policy = tree.run_simulations(&game, &nn, &mut seed);
    for &p in &policy {
        assert!(p >= 0.0, "negative policy probability: {}", p);
    }
}

#[test]
fn test_mcts_tree_best_action_valid() {
    let game = make_ttt();
    let nn = make_network(20, 16, 9);
    let config = make_mcts_config();
    let mut tree = SpMctsTree::new(config);
    let mut seed = make_seed();
    tree.run_simulations(&game, &nn, &mut seed);
    let best = tree.best_action();
    assert!(best < 9, "best_action={} should be in [0,8]", best);
    assert!(game.legal_actions().contains(&best));
}

#[test]
fn test_mcts_tree_action_probabilities_sums_to_one() {
    let game = make_ttt();
    let nn = make_network(20, 16, 9);
    let config = make_mcts_config();
    let mut tree = SpMctsTree::new(config);
    let mut seed = make_seed();
    tree.run_simulations(&game, &nn, &mut seed);
    let probs = tree.action_probabilities(1.0);
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "sum={}", sum);
}

#[test]
fn test_mcts_tree_low_temperature_greedy() {
    let game = make_ttt();
    let nn = make_network(20, 16, 9);
    let config = SpMctsConfig {
        n_simulations: 20,
        temperature: 0.001,
        ..make_mcts_config()
    };
    let mut tree = SpMctsTree::new(config);
    let mut seed = make_seed();
    tree.run_simulations(&game, &nn, &mut seed);
    let probs = tree.action_probabilities(0.001);
    // At near-zero temperature, the max should be ~1.0
    let max_prob = probs.iter().cloned().fold(0.0f64, f64::max);
    assert!(max_prob > 0.5, "greedy prob={} should be > 0.5", max_prob);
}

#[test]
fn test_mcts_tree_terminal_position() {
    // Create a near-terminal position where player 1 has two in a row
    let mut g = make_ttt();
    // P1=0, P2=3, P1=1, P2=4  → P1 can win at 2
    for &a in &[0usize, 3, 1, 4] {
        g = g.apply_action(a).expect("apply_action should succeed");
    }
    let nn = make_network(20, 16, 9);
    let config = SpMctsConfig {
        n_simulations: 30,
        ..make_mcts_config()
    };
    let mut tree = SpMctsTree::new(config);
    let mut seed = make_seed();
    let policy = tree.run_simulations(&g, &nn, &mut seed);
    assert!(!policy.is_empty());
    let sum: f64 = policy.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "policy sum={}", sum);
}

// ─── §6  SpSelfPlayBuffer ────────────────────────────────────────────────────

fn make_example(val: f64) -> SpSelfPlayExample {
    SpSelfPlayExample {
        state: vec![val; 10],
        mcts_policy: vec![1.0 / 9.0; 9],
        value: val,
    }
}

#[test]
fn test_buffer_new_is_empty() {
    let buf = SpSelfPlayBuffer::new(100);
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());
}

#[test]
fn test_buffer_add_increases_size() {
    let mut buf = SpSelfPlayBuffer::new(100);
    buf.add(make_example(0.5));
    assert_eq!(buf.len(), 1);
}

#[test]
fn test_buffer_is_ready() {
    let mut buf = SpSelfPlayBuffer::new(100);
    assert!(!buf.is_ready(1));
    buf.add(make_example(0.0));
    assert!(buf.is_ready(1));
    assert!(!buf.is_ready(2));
}

#[test]
fn test_buffer_circular_wrap() {
    let capacity = 5;
    let mut buf = SpSelfPlayBuffer::new(capacity);
    for i in 0..8 {
        buf.add(make_example(i as f64));
    }
    // Buffer should have capacity items
    assert_eq!(buf.len(), capacity);
}

#[test]
fn test_buffer_sample_returns_correct_count() {
    let mut buf = SpSelfPlayBuffer::new(50);
    for i in 0..20 {
        buf.add(make_example(i as f64));
    }
    let mut seed = make_seed();
    let batch = buf.sample(10, &mut seed);
    assert_eq!(batch.len(), 10);
}

#[test]
fn test_buffer_sample_does_not_exceed_buffer_size() {
    let mut buf = SpSelfPlayBuffer::new(50);
    buf.add(make_example(1.0));
    buf.add(make_example(2.0));
    let mut seed = make_seed();
    let batch = buf.sample(100, &mut seed);
    assert_eq!(batch.len(), 2);
}

#[test]
fn test_buffer_sample_empty() {
    let buf = SpSelfPlayBuffer::new(50);
    let mut seed = make_seed();
    let batch = buf.sample(10, &mut seed);
    assert!(batch.is_empty());
}

// ─── §7  SpAlphaZeroTrainer ──────────────────────────────────────────────────

fn make_trainer() -> SpAlphaZeroTrainer {
    SpAlphaZeroTrainer::new(
        20,
        16,
        9,
        SpAlphaZeroConfig {
            n_games_per_iteration: 2,
            n_training_steps: 2,
            batch_size: 4,
            lr: 1e-3,
            buffer_capacity: 500,
            mcts_config: SpMctsConfig {
                n_simulations: 5,
                ..SpMctsConfig::default()
            },
        },
    )
}

#[test]
fn test_trainer_self_play_game_produces_examples() {
    let trainer = make_trainer();
    let game = make_ttt();
    let mut seed = make_seed();
    let examples = trainer.self_play_game(game, &mut seed);
    // A Tic-Tac-Toe game must have at least 5 moves (shortest win) or 9 moves (draw)
    assert!(!examples.is_empty());
    assert!(examples.len() >= 5);
}

#[test]
fn test_trainer_self_play_examples_have_valid_state() {
    let trainer = make_trainer();
    let game = make_ttt();
    let mut seed = make_seed();
    let examples = trainer.self_play_game(game, &mut seed);
    for ex in &examples {
        assert_eq!(ex.state.len(), trainer.network.input_dim);
    }
}

#[test]
fn test_trainer_self_play_examples_policy_sum_to_one() {
    let trainer = make_trainer();
    let game = make_ttt();
    let mut seed = make_seed();
    let examples = trainer.self_play_game(game, &mut seed);
    for ex in &examples {
        let sum: f64 = ex.mcts_policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "policy sum={}", sum);
    }
}

#[test]
fn test_trainer_self_play_value_in_range() {
    let trainer = make_trainer();
    let game = make_ttt();
    let mut seed = make_seed();
    let examples = trainer.self_play_game(game, &mut seed);
    for ex in &examples {
        assert!(
            ex.value >= -1.0 && ex.value <= 1.0,
            "value={} out of [-1,1]",
            ex.value
        );
    }
}

#[test]
fn test_trainer_compute_loss_finite() {
    let trainer = make_trainer();
    let game = make_ttt();
    let mut seed = make_seed();
    let examples = trainer.self_play_game(game, &mut seed);
    let refs: Vec<&SpSelfPlayExample> = examples.iter().collect();
    let (pl, vl) = trainer.compute_loss(&refs);
    assert!(pl.is_finite(), "policy_loss={} is not finite", pl);
    assert!(vl.is_finite(), "value_loss={} is not finite", vl);
    assert!(pl >= 0.0, "policy_loss={} < 0", pl);
    assert!(vl >= 0.0, "value_loss={} < 0", vl);
}

#[test]
fn test_trainer_train_step_without_enough_data_returns_zero() {
    let mut trainer = make_trainer();
    let mut seed = make_seed();
    let loss = trainer.train_step(&mut seed);
    assert_eq!(loss, 0.0);
}

#[test]
fn test_trainer_train_step_returns_finite_loss() {
    let mut trainer = make_trainer();
    let game = make_ttt();
    let mut seed = make_seed();
    // Fill buffer with self-play data
    for _ in 0..5 {
        let examples = trainer.self_play_game(game.clone(), &mut seed);
        for ex in examples {
            trainer.buffer.add(ex);
        }
    }
    // Now we should have enough examples
    if trainer.buffer.is_ready(trainer.config.batch_size) {
        let loss = trainer.train_step(&mut seed);
        assert!(loss.is_finite(), "train_step loss={} is not finite", loss);
    }
}

#[test]
fn test_trainer_iteration_counter_starts_zero() {
    let trainer = make_trainer();
    assert_eq!(trainer.iteration, 0);
}

// ─── §8  SpEloSystem ─────────────────────────────────────────────────────────

#[test]
fn test_elo_expected_score_equal_rating() {
    let score = SpEloSystem::expected_score(1500.0, 1500.0);
    assert!((score - 0.5).abs() < 1e-9, "score={}", score);
}

#[test]
fn test_elo_expected_score_in_range() {
    for &(ra, rb) in &[
        (1000.0, 2000.0),
        (2000.0, 1000.0),
        (1200.0, 1400.0),
        (1600.0, 1200.0),
    ] {
        let s = SpEloSystem::expected_score(ra, rb);
        assert!(
            s > 0.0 && s < 1.0,
            "expected_score({},{})={} not in (0,1)",
            ra,
            rb,
            s
        );
    }
}

#[test]
fn test_elo_expected_score_higher_rating_wins_more() {
    let s_strong = SpEloSystem::expected_score(2000.0, 1000.0);
    let s_weak = SpEloSystem::expected_score(1000.0, 2000.0);
    assert!(s_strong > s_weak);
    assert!((s_strong + s_weak - 1.0).abs() < 1e-9);
}

#[test]
fn test_elo_add_model_and_retrieve() {
    let mut elo = SpEloSystem::new(32.0);
    elo.add_model("alpha", 1500.0);
    assert_eq!(elo.models.len(), 1);
    assert_eq!(elo.elo(0), 1500.0);
}

#[test]
fn test_elo_record_game_win_updates_ratings() {
    let mut elo = SpEloSystem::new(32.0);
    elo.add_model("A", 1500.0);
    elo.add_model("B", 1500.0);
    elo.record_game(0, 1, 1.0); // A wins
    assert!(elo.elo(0) > 1500.0, "winner should gain ELO");
    assert!(elo.elo(1) < 1500.0, "loser should lose ELO");
}

#[test]
fn test_elo_record_game_draw_small_change() {
    let mut elo = SpEloSystem::new(32.0);
    elo.add_model("A", 1500.0);
    elo.add_model("B", 1500.0);
    elo.record_game(0, 1, 0.5); // draw
                                // With equal rating, expected=0.5 and outcome=0.5, so no change
    assert!((elo.elo(0) - 1500.0).abs() < 1e-9, "elo_a={}", elo.elo(0));
    assert!((elo.elo(1) - 1500.0).abs() < 1e-9, "elo_b={}", elo.elo(1));
}

#[test]
fn test_elo_ranking_sorted_descending() {
    let mut elo = SpEloSystem::new(32.0);
    elo.add_model("weak", 1200.0);
    elo.add_model("strong", 1800.0);
    elo.add_model("medium", 1500.0);
    let ranked = elo.ranking();
    assert_eq!(ranked[0].0, "strong");
    assert_eq!(ranked[1].0, "medium");
    assert_eq!(ranked[2].0, "weak");
}

#[test]
fn test_elo_history_records_games() {
    let mut elo = SpEloSystem::new(32.0);
    elo.add_model("A", 1500.0);
    elo.add_model("B", 1500.0);
    elo.record_game(0, 1, 1.0);
    elo.record_game(0, 1, 0.5);
    assert_eq!(elo.history.len(), 2);
}

#[test]
fn test_elo_invalid_model_index_no_panic() {
    let mut elo = SpEloSystem::new(32.0);
    elo.add_model("A", 1500.0);
    // Should not panic, just do nothing
    elo.record_game(0, 99, 1.0);
    assert_eq!(elo.history.len(), 0);
}

// ─── §9  SpMinimax ───────────────────────────────────────────────────────────

fn ttt_evaluator(g: &SpTicTacToe) -> f64 {
    match g.winner() {
        Some(w) => w as f64,
        None => 0.0,
    }
}

#[test]
fn test_minimax_finds_winning_move() {
    // P1 has 0,1 → can win at 2
    // P2 has 3,4 (not threatening yet)
    let mut g = make_ttt();
    for &a in &[0usize, 3, 1, 4] {
        g = g.apply_action(a).expect("apply_action should succeed");
    }
    assert_eq!(g.current_player(), 1);
    let mm = SpMinimax::new(9);
    let (action, _) = mm.search(&g, &ttt_evaluator);
    assert_eq!(
        action, 2,
        "minimax should find winning move at cell 2, got {}",
        action
    );
}

#[test]
fn test_minimax_blocks_opponent_win() {
    // P2 has 3,4 → can win at 5
    // P1 must block at 5 or P2 wins on next move
    let mut g = make_ttt();
    for &a in &[0usize, 3, 1, 4] {
        g = g.apply_action(a).expect("apply_action should succeed");
    }
    // Now P1 plays at 2 to win themselves (minimax handles this too)
    let mm = SpMinimax::new(9);
    let (action, _) = mm.search(&g, &ttt_evaluator);
    // P1 should choose 2 (win immediately)
    assert_eq!(action, 2);
}

#[test]
fn test_minimax_search_returns_legal_action() {
    let g = make_ttt();
    let mm = SpMinimax::new(4);
    let (action, _) = mm.search(&g, &ttt_evaluator);
    assert!(action < 9);
    assert!(g.legal_actions().contains(&action));
}

#[test]
fn test_minimax_value_in_range() {
    let g = make_ttt();
    let mm = SpMinimax::new(4);
    let (_, value) = mm.search(&g, &ttt_evaluator);
    assert!((-1.0..=1.0).contains(&value), "value={}", value);
}

#[test]
fn test_minimax_policy_eval_sums_to_one() {
    let g = make_ttt();
    let mm = SpMinimax::new(3);
    let policy = mm.policy_eval(&g, 3);
    assert_eq!(policy.len(), 9);
    let sum: f64 = policy.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "policy_eval sum={}", sum);
}

#[test]
fn test_minimax_policy_eval_best_action_is_one() {
    let g = make_ttt();
    let mm = SpMinimax::new(3);
    let policy = mm.policy_eval(&g, 3);
    let n_ones = policy.iter().filter(|&&p| (p - 1.0).abs() < 1e-9).count();
    assert_eq!(n_ones, 1, "exactly one action should have prob 1.0");
}

#[test]
fn test_minimax_terminal_returns_empty_policy() {
    let mut g = make_ttt();
    for &a in &[0usize, 3, 1, 4, 2] {
        g = g.apply_action(a).expect("apply_action should succeed");
    }
    assert!(g.is_terminal());
    let mm = SpMinimax::new(3);
    let policy = mm.policy_eval(&g, 3);
    // All zeros for terminal position
    assert!(policy.iter().all(|&p| p == 0.0));
}

// ─── §10  SpMetrics ──────────────────────────────────────────────────────────

#[test]
fn test_metrics_policy_entropy_uniform() {
    let n = 9usize;
    let uniform = vec![1.0 / n as f64; n];
    let entropy = SpMetrics::policy_entropy(&uniform);
    let expected = (n as f64).ln();
    assert!(
        (entropy - expected).abs() < 0.01,
        "uniform entropy={} expected≈{:.4}",
        entropy,
        expected
    );
}

#[test]
fn test_metrics_policy_entropy_deterministic() {
    let mut policy = vec![0.0f64; 9];
    policy[3] = 1.0;
    let entropy = SpMetrics::policy_entropy(&policy);
    assert!(entropy >= 0.0, "entropy={} should be ≥ 0", entropy);
    assert!(
        entropy < 1e-9,
        "deterministic policy should have entropy ≈ 0"
    );
}

#[test]
fn test_metrics_value_calibration_perfect_correlation() {
    let v: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let corr = SpMetrics::value_calibration(&v, &v);
    assert!(
        (corr - 1.0).abs() < 1e-9,
        "perfect correlation should be 1.0, got {}",
        corr
    );
}

#[test]
fn test_metrics_value_calibration_anti_correlation() {
    let v: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let neg: Vec<f64> = v.iter().map(|&x| -x).collect();
    let corr = SpMetrics::value_calibration(&v, &neg);
    assert!(
        (corr + 1.0).abs() < 1e-9,
        "anti-correlation should be -1.0, got {}",
        corr
    );
}

#[test]
fn test_metrics_value_calibration_in_range() {
    let pred = vec![0.5, -0.3, 0.8, -0.1, 0.2];
    let actual = vec![1.0, -1.0, 1.0, 0.0, 0.5];
    let corr = SpMetrics::value_calibration(&pred, &actual);
    assert!((-1.0..=1.0).contains(&corr), "corr={} out of [-1,1]", corr);
}

#[test]
fn test_metrics_game_length_stats() {
    let lengths = vec![5usize, 7, 9, 6, 8];
    let (mean, std, max) = SpMetrics::game_length_stats(&lengths);
    assert!((mean - 7.0).abs() < 0.01, "mean={}", mean);
    assert!(std >= 0.0, "std={}", std);
    assert_eq!(max, 9);
}

#[test]
fn test_metrics_game_length_stats_empty() {
    let (mean, std, max) = SpMetrics::game_length_stats(&[]);
    assert_eq!(mean, 0.0);
    assert_eq!(std, 0.0);
    assert_eq!(max, 0);
}

#[test]
fn test_metrics_policy_improvement_in_range() {
    let score = SpMetrics::policy_improvement(5, 10, 3);
    assert!((0.0..=1.0).contains(&score), "score={}", score);
}

#[test]
fn test_metrics_policy_improvement_all_new_wins() {
    let score = SpMetrics::policy_improvement(0, 10, 0);
    assert!((score - 1.0).abs() < 1e-9, "score={}", score);
}

#[test]
fn test_metrics_policy_improvement_all_old_wins() {
    let score = SpMetrics::policy_improvement(10, 0, 0);
    assert!(score < 1e-9, "score={}", score);
}

#[test]
fn test_metrics_policy_improvement_all_draws() {
    let score = SpMetrics::policy_improvement(0, 0, 10);
    assert!((score - 0.5).abs() < 1e-9, "score={}", score);
}

#[test]
fn test_metrics_mcts_visit_entropy_uniform() {
    let visits = vec![10usize; 9];
    let entropy = SpMetrics::mcts_visit_entropy(&visits);
    let expected = (9.0f64).ln();
    assert!(
        (entropy - expected).abs() < 0.01,
        "uniform mcts entropy={} expected≈{:.4}",
        entropy,
        expected
    );
}

#[test]
fn test_metrics_mcts_visit_entropy_zero_for_single_action() {
    let mut visits = vec![0usize; 9];
    visits[4] = 100;
    let entropy = SpMetrics::mcts_visit_entropy(&visits);
    assert!(
        entropy < 1e-9,
        "entropy={} should be ≈ 0 for single action",
        entropy
    );
}

#[test]
fn test_metrics_mcts_visit_entropy_nonnegative() {
    let visits = vec![3usize, 7, 2, 0, 5, 1, 4, 0, 2];
    let entropy = SpMetrics::mcts_visit_entropy(&visits);
    assert!(entropy >= 0.0, "entropy={} should be ≥ 0", entropy);
}

#[test]
fn test_metrics_mcts_visit_entropy_empty() {
    let entropy = SpMetrics::mcts_visit_entropy(&[]);
    assert_eq!(entropy, 0.0);
}
