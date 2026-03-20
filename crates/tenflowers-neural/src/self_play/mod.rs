//! # Self-Play & Game-Playing Algorithms (Track C Round 47)
//!
//! Production-grade AlphaZero-style self-play algorithms including Monte Carlo
//! Tree Search (MCTS) with neural network guidance, self-play data generation,
//! ELO rating systems, and minimax search with alpha-beta pruning.
//!
//! ## Algorithms
//!
//! - **SpMctsTree** — AlphaZero-style PUCT-based MCTS with neural network priors.
//! - **SpAlphaZeroTrainer** — Full self-play training loop (generate + train).
//! - **SpMinimax** — Alpha-beta pruning minimax for exact small-game solving.
//! - **SpEloSystem** — ELO rating system for comparing agent versions.
//! - **SpSelfPlayBuffer** — Circular replay buffer for self-play examples.
//!
//! ## References
//! - Silver et al. (2017) "Mastering the game of Go without human knowledge"
//! - Silver et al. (2018) "A general reinforcement learning algorithm that masters
//!   chess, shogi, and Go through self-play"

#[cfg(test)]
mod tests;

use std::fmt;

// ─── §0  Error type ──────────────────────────────────────────────────────────

/// Errors produced by the self-play module.
#[derive(Debug, Clone)]
pub enum SpError {
    InvalidAction(String),
    GameOver,
    NumericalError(String),
}

impl fmt::Display for SpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpError::InvalidAction(msg) => write!(f, "InvalidAction: {}", msg),
            SpError::GameOver => write!(f, "GameOver"),
            SpError::NumericalError(msg) => write!(f, "NumericalError: {}", msg),
        }
    }
}

impl std::error::Error for SpError {}

// ─── §0b  Seeded RNG ─────────────────────────────────────────────────────────

/// Xorshift64-based fast uniform U(0,1) random number.
pub fn sp_rand01(seed: &mut u64) -> f64 {
    let mut x = *seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *seed = x;
    // Map u64 to (0, 1) — avoid 0 and 1 exactly
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// Box-Muller transform: U(0,1) pair → N(0,1).
pub fn sp_randn(seed: &mut u64) -> f64 {
    let u1 = sp_rand01(seed).max(1e-15);
    let u2 = sp_rand01(seed);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ─── §1  SpGame — abstract game interface ────────────────────────────────────

/// Abstract interface for two-player zero-sum board games.
///
/// Games are immutable; moves produce new game states via `apply_action`.
/// Players are represented as `+1` (first player) and `-1` (second player).
pub trait SpGame: Clone + Send + Sync {
    /// Total number of possible actions (including illegal ones).
    fn n_actions(&self) -> usize;

    /// Indices of all currently legal actions.
    fn legal_actions(&self) -> Vec<usize>;

    /// Apply `action` and return the resulting game state.
    fn apply_action(&self, action: usize) -> Result<Self, SpError>;

    /// Whether the game has ended.
    fn is_terminal(&self) -> bool;

    /// `Some(+1)` if player 1 won, `Some(-1)` if player -1 won, `None` for draw.
    fn winner(&self) -> Option<i8>;

    /// The player whose turn it is now: `+1` or `-1`.
    fn current_player(&self) -> i8;

    /// Encode the current state as a flat `f64` vector of length `dim`.
    fn state_encoding(&self, dim: usize) -> Vec<f64>;
}

// ─── §2  SpTicTacToe ─────────────────────────────────────────────────────────

/// Classic 3×3 Tic-Tac-Toe implementation of [`SpGame`].
///
/// Board cells: `0` = empty, `+1` = X, `-1` = O.
/// Actions are board indices 0–8 (row-major order).
#[derive(Clone, Debug)]
pub struct SpTicTacToe {
    /// Board state: 0 = empty, 1 = X, -1 = O.
    pub board: [i8; 9],
    /// Current player: +1 or -1.
    pub current_player: i8,
    /// Total moves made so far.
    pub n_moves: usize,
}

impl SpTicTacToe {
    /// Create a fresh empty board with player +1 to move.
    pub fn new() -> Self {
        Self {
            board: [0i8; 9],
            current_player: 1,
            n_moves: 0,
        }
    }

    /// Check if there is a winner given the board.
    fn check_winner(board: &[i8; 9]) -> Option<i8> {
        const LINES: [[usize; 3]; 8] = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8], // rows
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8], // cols
            [0, 4, 8],
            [2, 4, 6], // diagonals
        ];
        for line in &LINES {
            if let Some(w) = Self::check_line(board, *line) {
                return Some(w);
            }
        }
        None
    }

    /// Check a single line for a winner.
    fn check_line(board: &[i8; 9], positions: [usize; 3]) -> Option<i8> {
        let a = board[positions[0]];
        let b = board[positions[1]];
        let c = board[positions[2]];
        if a != 0 && a == b && b == c {
            Some(a)
        } else {
            None
        }
    }
}

impl Default for SpTicTacToe {
    fn default() -> Self {
        Self::new()
    }
}

impl SpGame for SpTicTacToe {
    fn n_actions(&self) -> usize {
        9
    }

    fn legal_actions(&self) -> Vec<usize> {
        if self.is_terminal() {
            return vec![];
        }
        self.board
            .iter()
            .enumerate()
            .filter(|(_, &v)| v == 0)
            .map(|(i, _)| i)
            .collect()
    }

    fn apply_action(&self, action: usize) -> Result<Self, SpError> {
        if action >= 9 {
            return Err(SpError::InvalidAction(format!(
                "action {} out of range [0,8]",
                action
            )));
        }
        if self.is_terminal() {
            return Err(SpError::GameOver);
        }
        if self.board[action] != 0 {
            return Err(SpError::InvalidAction(format!(
                "cell {} is already occupied",
                action
            )));
        }
        let mut new_board = self.board;
        new_board[action] = self.current_player;
        Ok(Self {
            board: new_board,
            current_player: -self.current_player,
            n_moves: self.n_moves + 1,
        })
    }

    fn is_terminal(&self) -> bool {
        Self::check_winner(&self.board).is_some() || self.n_moves >= 9
    }

    fn winner(&self) -> Option<i8> {
        Self::check_winner(&self.board)
    }

    fn current_player(&self) -> i8 {
        self.current_player
    }

    fn state_encoding(&self, dim: usize) -> Vec<f64> {
        // Encode board as plane for current player and opponent, padded to `dim`.
        let mut enc = vec![0.0f64; dim];
        let player = self.current_player as f64;
        for (i, &cell) in self.board.iter().enumerate() {
            if i < dim {
                enc[i] = (cell as f64) * player; // +1 for own pieces, -1 for opponent
            }
        }
        // Optionally add current player indicator at position 9 if dim > 9
        if dim > 9 {
            enc[9] = player;
        }
        enc
    }
}

// ─── §3  SpNeuralNetwork — policy-value network ──────────────────────────────

/// Two-layer MLP policy-value network for AlphaZero-style MCTS.
///
/// Architecture: input → ReLU → hidden → ReLU → {policy head, value head}.
/// The policy head outputs logits (softmax applied externally).
/// The value head outputs a scalar in `[-1, 1]` (tanh applied).
#[derive(Clone, Debug)]
pub struct SpNeuralNetwork {
    /// First layer weights: shape [hidden_dim × input_dim].
    pub w1: Vec<Vec<f64>>,
    /// First layer biases: shape \[hidden_dim\].
    pub b1: Vec<f64>,
    /// Second layer weights: shape [hidden_dim × hidden_dim].
    pub w2: Vec<Vec<f64>>,
    /// Second layer biases: shape \[hidden_dim\].
    pub b2: Vec<f64>,
    /// Policy head weights: shape [n_actions × hidden_dim].
    pub policy_head_w: Vec<Vec<f64>>,
    /// Policy head biases: shape \[n_actions\].
    pub policy_head_b: Vec<f64>,
    /// Value head weights: shape \[hidden_dim\] (flattened 1×hidden).
    pub value_head_w: Vec<f64>,
    /// Value head bias (scalar).
    pub value_head_b: f64,
    /// Input feature dimension.
    pub input_dim: usize,
    /// Hidden layer width.
    pub hidden_dim: usize,
    /// Number of actions (policy output dimension).
    pub n_actions: usize,
}

impl SpNeuralNetwork {
    /// Initialize network with Xavier/Glorot uniform weights.
    pub fn new(input_dim: usize, hidden_dim: usize, n_actions: usize) -> Self {
        let mut seed = 0xDEAD_BEEF_1234_5678u64;

        // Xavier uniform scale for each layer
        let scale1 = (6.0 / (input_dim + hidden_dim) as f64).sqrt();
        let scale2 = (6.0 / (hidden_dim + hidden_dim) as f64).sqrt();
        let scale_p = (6.0 / (hidden_dim + n_actions) as f64).sqrt();
        let scale_v = (6.0 / (hidden_dim + 1) as f64).sqrt();

        let xavier_uniform =
            |rows: usize, cols: usize, scale: f64, seed: &mut u64| -> Vec<Vec<f64>> {
                (0..rows)
                    .map(|_| {
                        (0..cols)
                            .map(|_| (sp_rand01(seed) * 2.0 - 1.0) * scale)
                            .collect()
                    })
                    .collect()
            };

        let w1 = xavier_uniform(hidden_dim, input_dim, scale1, &mut seed);
        let b1 = vec![0.0f64; hidden_dim];
        let w2 = xavier_uniform(hidden_dim, hidden_dim, scale2, &mut seed);
        let b2 = vec![0.0f64; hidden_dim];
        let policy_head_w = xavier_uniform(n_actions, hidden_dim, scale_p, &mut seed);
        let policy_head_b = vec![0.0f64; n_actions];
        let value_head_w = (0..hidden_dim)
            .map(|_| (sp_rand01(&mut seed) * 2.0 - 1.0) * scale_v)
            .collect();

        Self {
            w1,
            b1,
            w2,
            b2,
            policy_head_w,
            policy_head_b,
            value_head_w,
            value_head_b: 0.0,
            input_dim,
            hidden_dim,
            n_actions,
        }
    }

    /// Linear layer: h = W * x + b with ReLU activation.
    fn linear_relu(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
        w.iter()
            .zip(b.iter())
            .map(|(row, &bias)| {
                let s: f64 = row.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum();
                (s + bias).max(0.0)
            })
            .collect()
    }

    /// Softmax over a slice of logits.
    fn softmax(logits: &[f64]) -> Vec<f64> {
        let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum = exps.iter().sum::<f64>().max(1e-30);
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Forward pass: returns (policy_logits \[n_actions\], value scalar).
    pub fn forward(&self, state: &[f64]) -> (Vec<f64>, f64) {
        let h1 = Self::linear_relu(&self.w1, &self.b1, state);
        let h2 = Self::linear_relu(&self.w2, &self.b2, &h1);

        // Policy logits
        let policy_logits: Vec<f64> = self
            .policy_head_w
            .iter()
            .zip(self.policy_head_b.iter())
            .map(|(row, &bias)| {
                let s: f64 = row.iter().zip(h2.iter()).map(|(&wi, &hi)| wi * hi).sum();
                s + bias
            })
            .collect();

        // Value scalar
        let v_raw: f64 = self
            .value_head_w
            .iter()
            .zip(h2.iter())
            .map(|(&wi, &hi)| wi * hi)
            .sum::<f64>()
            + self.value_head_b;
        let value = v_raw.tanh();

        (policy_logits, value)
    }

    /// Policy-value with legal action masking.
    ///
    /// Illegal actions receive `-inf` before softmax so they get probability ≈ 0.
    pub fn policy_value(&self, state: &[f64], legal_mask: &[bool]) -> (Vec<f64>, f64) {
        let (logits, value) = self.forward(state);
        let masked: Vec<f64> = logits
            .iter()
            .zip(legal_mask.iter())
            .map(|(&l, &legal)| if legal { l } else { f64::NEG_INFINITY })
            .collect();
        let probs = Self::softmax(&masked);
        (probs, value)
    }

    /// Simple SGD update for the first two layers.
    pub fn update_weights(&mut self, grad_w1: &[Vec<f64>], grad_w2: &[Vec<f64>], lr: f64) {
        for (row, grad_row) in self.w1.iter_mut().zip(grad_w1.iter()) {
            for (w, &g) in row.iter_mut().zip(grad_row.iter()) {
                *w -= lr * g;
            }
        }
        for (row, grad_row) in self.w2.iter_mut().zip(grad_w2.iter()) {
            for (w, &g) in row.iter_mut().zip(grad_row.iter()) {
                *w -= lr * g;
            }
        }
    }
}

// ─── §4  SpMctsNode ──────────────────────────────────────────────────────────

/// A single node in the MCTS tree (stored in an arena).
#[derive(Clone, Debug)]
pub struct SpMctsNode {
    /// The action that was taken to reach this node.
    pub action: Option<usize>,
    /// The player who just moved (whose perspective the value is from).
    pub player: i8,
    /// Visit count N(s,a).
    pub visit_count: usize,
    /// Accumulated value W(s,a).
    pub value_sum: f64,
    /// Prior probability P(s,a) from the policy network.
    pub prior: f64,
    /// Indices of child nodes in the arena.
    pub children: Vec<usize>,
    /// Index of parent node in the arena.
    pub parent: Option<usize>,
    /// Whether this node has been expanded.
    pub is_expanded: bool,
    /// Terminal value if this is a terminal state.
    pub terminal_value: Option<f64>,
}

impl SpMctsNode {
    /// Create a new unvisited node.
    pub fn new(action: Option<usize>, prior: f64, player: i8, parent: Option<usize>) -> Self {
        Self {
            action,
            player,
            visit_count: 0,
            value_sum: 0.0,
            prior,
            children: vec![],
            parent,
            is_expanded: false,
            terminal_value: None,
        }
    }

    /// Mean action value Q(s,a) = W/N; returns 0 for unvisited nodes.
    pub fn q_value(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f64
        }
    }

    /// PUCT score used for child selection during MCTS.
    ///
    /// U(s,a) = Q(s,a) + c_puct × P(s,a) × √(N_parent) / (1 + N(s,a))
    pub fn ucb_score(&self, parent_visits: usize, c_puct: f64) -> f64 {
        let exploration =
            c_puct * self.prior * (parent_visits as f64).sqrt() / (1.0 + self.visit_count as f64);
        self.q_value() + exploration
    }
}

// ─── §5  SpMctsTree ──────────────────────────────────────────────────────────

/// Configuration for AlphaZero-style MCTS.
#[derive(Clone, Debug)]
pub struct SpMctsConfig {
    /// Number of MCTS simulations per move decision.
    pub n_simulations: usize,
    /// Exploration constant for PUCT (typical: 1.0–5.0).
    pub c_puct: f64,
    /// Dirichlet noise alpha (0.03 for chess, 0.3 for Tic-Tac-Toe).
    pub dirichlet_alpha: f64,
    /// Fraction of Dirichlet noise to mix into root priors.
    pub dirichlet_epsilon: f64,
    /// Temperature for action sampling (1.0 for exploration, ~0 for exploitation).
    pub temperature: f64,
}

impl Default for SpMctsConfig {
    fn default() -> Self {
        Self {
            n_simulations: 50,
            c_puct: 1.5,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            temperature: 1.0,
        }
    }
}

/// AlphaZero-style MCTS tree operating over a game-agnostic interface.
///
/// Uses a flat arena (Vec) to avoid recursive ownership and borrow issues.
pub struct SpMctsTree {
    /// Flat arena of nodes.
    pub nodes: Vec<SpMctsNode>,
    /// MCTS hyperparameters.
    pub config: SpMctsConfig,
}

impl SpMctsTree {
    /// Create an empty tree with a root sentinel node.
    pub fn new(config: SpMctsConfig) -> Self {
        let root = SpMctsNode::new(None, 1.0, 1, None);
        Self {
            nodes: vec![root],
            config,
        }
    }

    /// Rebuild the tree from scratch for a given game state.
    ///
    /// Runs `n_simulations` PUCT simulations and returns a policy vector
    /// whose entries are proportional to visit counts (temperature-scaled).
    pub fn run_simulations<G: SpGame>(
        &mut self,
        root_game: &G,
        network: &SpNeuralNetwork,
        seed: &mut u64,
    ) -> Vec<f64> {
        // Reset arena with fresh root
        self.nodes.clear();
        let root_player = root_game.current_player();
        self.nodes
            .push(SpMctsNode::new(None, 1.0, root_player, None));

        // Expand root immediately
        self.expand(0, root_game, network);

        // Add Dirichlet noise to root priors for exploration
        if !self.nodes[0].children.is_empty() {
            self.add_dirichlet_noise_to_root(seed);
        }

        let legal = root_game.legal_actions();
        if legal.is_empty() {
            return vec![1.0 / root_game.n_actions() as f64; root_game.n_actions()];
        }

        for _ in 0..self.config.n_simulations {
            // Clone game so we can simulate from root
            let mut game = root_game.clone();
            let (leaf_idx, path) = self.select_leaf(0, &mut game, network);

            // Determine value: use terminal value or network evaluation
            let value = if let Some(tv) = self.nodes[leaf_idx].terminal_value {
                tv
            } else {
                let state = game.state_encoding(network.input_dim);
                let legal_mask: Vec<bool> = (0..network.n_actions)
                    .map(|a| game.legal_actions().contains(&a))
                    .collect();
                let (_, v) = network.policy_value(&state, &legal_mask);
                // Value is from the perspective of the node's current player
                v
            };

            self.backpropagate(&path, value);
        }

        self.action_probabilities(self.config.temperature)
    }

    /// Add Dirichlet noise to the priors of the root's children.
    fn add_dirichlet_noise_to_root(&mut self, seed: &mut u64) {
        let child_indices = self.nodes[0].children.clone();
        let n = child_indices.len();
        if n == 0 {
            return;
        }
        // Sample Dirichlet via Gamma distribution approximation
        let alpha = self.config.dirichlet_alpha;
        let epsilon = self.config.dirichlet_epsilon;
        let mut gammas: Vec<f64> = (0..n)
            .map(|_| {
                // Gamma(alpha) via Marsaglia's method for alpha >= 1
                // For small alpha, use log-normal approximation
                let g = if alpha >= 1.0 {
                    sp_gamma_sample(alpha, seed)
                } else {
                    sp_gamma_sample(alpha + 1.0, seed) * sp_rand01(seed).powf(1.0 / alpha)
                };
                g.max(1e-15)
            })
            .collect();
        let sum: f64 = gammas.iter().sum();
        let sum = sum.max(1e-30);
        for g in &mut gammas {
            *g /= sum;
        }
        for (i, &child_idx) in child_indices.iter().enumerate() {
            let child = &mut self.nodes[child_idx];
            child.prior = (1.0 - epsilon) * child.prior + epsilon * gammas[i];
        }
    }

    /// Select the leaf node to expand by following PUCT scores.
    ///
    /// Returns the leaf node index and the path from root to leaf.
    /// Also advances `game` along the selected path.
    fn select_leaf<G: SpGame>(
        &self,
        root: usize,
        game: &mut G,
        _network: &SpNeuralNetwork,
    ) -> (usize, Vec<usize>) {
        let mut current = root;
        let mut path = vec![current];

        loop {
            let node = &self.nodes[current];
            if !node.is_expanded || node.children.is_empty() {
                return (current, path);
            }
            if node.terminal_value.is_some() {
                return (current, path);
            }

            // Select child with highest UCB score
            let parent_visits = node.visit_count;
            let best_child = node.children.iter().copied().max_by(|&a, &b| {
                self.nodes[a]
                    .ucb_score(parent_visits, self.config.c_puct)
                    .partial_cmp(&self.nodes[b].ucb_score(parent_visits, self.config.c_puct))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            match best_child {
                None => return (current, path),
                Some(child_idx) => {
                    let action = self.nodes[child_idx].action;
                    if let Some(act) = action {
                        match game.apply_action(act) {
                            Ok(new_game) => *game = new_game,
                            Err(_) => return (current, path),
                        }
                    }
                    current = child_idx;
                    path.push(current);
                }
            }
        }
    }

    /// Expand a leaf node using the neural network.
    ///
    /// Gets the policy/value from the network, creates child nodes
    /// for each legal action with their prior probabilities.
    fn expand<G: SpGame>(&mut self, node_idx: usize, game: &G, network: &SpNeuralNetwork) {
        if self.nodes[node_idx].is_expanded {
            return;
        }

        // Check terminal
        if game.is_terminal() {
            let w = game.winner();
            let player = self.nodes[node_idx].player;
            let terminal_val = match w {
                Some(winner) => {
                    if winner == player {
                        1.0
                    } else {
                        -1.0
                    }
                }
                None => 0.0, // draw
            };
            self.nodes[node_idx].terminal_value = Some(terminal_val);
            self.nodes[node_idx].is_expanded = true;
            return;
        }

        // Get neural network policy
        let legal = game.legal_actions();
        if legal.is_empty() {
            self.nodes[node_idx].is_expanded = true;
            return;
        }

        let state = game.state_encoding(network.input_dim);
        let legal_mask: Vec<bool> = (0..network.n_actions).map(|a| legal.contains(&a)).collect();
        let (probs, _value) = network.policy_value(&state, &legal_mask);

        let current_player = game.current_player();
        let parent_idx = node_idx;
        let arena_len = self.nodes.len();

        // Build children: one per legal action
        let mut child_indices = Vec::with_capacity(legal.len());
        for (ci, &action) in legal.iter().enumerate() {
            let prior = if action < probs.len() {
                probs[action]
            } else {
                1.0 / legal.len() as f64
            };
            let child_node = SpMctsNode::new(Some(action), prior, current_player, Some(parent_idx));
            self.nodes.push(child_node);
            child_indices.push(arena_len + ci);
        }

        self.nodes[node_idx].children = child_indices;
        self.nodes[node_idx].is_expanded = true;
    }

    /// Backpropagate the simulation value along the path.
    ///
    /// The value is flipped for alternating players since value is always from
    /// the perspective of the node that was expanded.
    fn backpropagate(&mut self, path: &[usize], value: f64) {
        // Determine which player is at the expanded leaf
        let leaf_player = if let Some(&leaf) = path.last() {
            self.nodes[leaf].player
        } else {
            return;
        };

        for &node_idx in path.iter().rev() {
            let node_player = self.nodes[node_idx].player;
            // Value sign: same player gets +value, opponent gets -value
            let signed_value = if node_player == leaf_player {
                value
            } else {
                -value
            };
            self.nodes[node_idx].visit_count += 1;
            self.nodes[node_idx].value_sum += signed_value;
        }
    }

    /// Return the action with the highest visit count from the root.
    pub fn best_action(&self) -> usize {
        let root = &self.nodes[0];
        root.children
            .iter()
            .max_by_key(|&&c| self.nodes[c].visit_count)
            .and_then(|&c| self.nodes[c].action)
            .unwrap_or(0)
    }

    /// Compute the action probability distribution from MCTS visit counts.
    ///
    /// π(a) ∝ N(s,a)^(1/temperature). For temperature → 0 (< 0.01),
    /// this degenerates to argmax over visit counts.
    pub fn action_probabilities(&self, temperature: f64) -> Vec<f64> {
        let root = &self.nodes[0];
        let n_actions = root.children.len();
        if n_actions == 0 {
            return vec![];
        }

        // Map from action → visit_count (not all actions may be children)
        let max_action = root
            .children
            .iter()
            .filter_map(|&c| self.nodes[c].action)
            .max()
            .unwrap_or(0)
            + 1;

        let mut counts = vec![0usize; max_action];
        for &c in &root.children {
            if let Some(action) = self.nodes[c].action {
                if action < counts.len() {
                    counts[action] = self.nodes[c].visit_count;
                }
            }
        }

        if temperature < 0.01 {
            // Argmax
            let best_count = counts.iter().max().copied().unwrap_or(0);
            let n_best = counts.iter().filter(|&&c| c == best_count).count();
            return counts
                .iter()
                .map(|&c| {
                    if c == best_count {
                        1.0 / n_best as f64
                    } else {
                        0.0
                    }
                })
                .collect();
        }

        let inv_temp = 1.0 / temperature;
        let powered: Vec<f64> = counts.iter().map(|&n| (n as f64).powf(inv_temp)).collect();
        let total: f64 = powered.iter().sum();
        if total <= 0.0 {
            let uniform = 1.0 / powered.len() as f64;
            return vec![uniform; powered.len()];
        }
        powered.iter().map(|&p| p / total).collect()
    }
}

// ─── §5b  Gamma sampler helper ───────────────────────────────────────────────

/// Marsaglia & Tsang (2000) Gamma(alpha, 1) sampler for alpha >= 1.
fn sp_gamma_sample(alpha: f64, seed: &mut u64) -> f64 {
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = sp_randn(seed);
        let v = {
            let v_inner = 1.0 + c * x;
            if v_inner <= 0.0 {
                continue;
            }
            v_inner * v_inner * v_inner
        };
        let u = sp_rand01(seed);
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

// ─── §6  SpSelfPlayBuffer ────────────────────────────────────────────────────

/// Single training example generated from self-play.
#[derive(Clone, Debug)]
pub struct SpSelfPlayExample {
    /// Encoded game state at the time of the decision.
    pub state: Vec<f64>,
    /// MCTS visit-count policy used as the training target.
    pub mcts_policy: Vec<f64>,
    /// Game outcome from the perspective of the player-to-move: +1/-1/0.
    pub value: f64,
}

/// Circular replay buffer storing self-play examples.
pub struct SpSelfPlayBuffer {
    /// Internal storage (fixed-size circular buffer).
    pub buffer: Vec<SpSelfPlayExample>,
    /// Maximum number of examples to store.
    pub capacity: usize,
    /// Write head (next insert position).
    pub head: usize,
    /// Current number of valid examples.
    pub size: usize,
}

impl SpSelfPlayBuffer {
    /// Create an empty buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            head: 0,
            size: 0,
        }
    }

    /// Insert an example, overwriting the oldest entry if full.
    pub fn add(&mut self, example: SpSelfPlayExample) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(example);
        } else {
            self.buffer[self.head] = example;
        }
        self.head = (self.head + 1) % self.capacity;
        self.size = self.size.min(self.capacity);
        if self.buffer.len() == self.capacity {
            self.size = self.capacity;
        } else {
            self.size = self.buffer.len();
        }
    }

    /// Sample `batch_size` examples uniformly at random.
    pub fn sample(&self, batch_size: usize, seed: &mut u64) -> Vec<&SpSelfPlayExample> {
        let n = self.buffer.len();
        if n == 0 {
            return vec![];
        }
        let take = batch_size.min(n);
        let mut indices: Vec<usize> = (0..n).collect();
        // Fisher-Yates partial shuffle
        for i in 0..take {
            let j = i + (sp_rand01(seed) * (n - i) as f64) as usize % (n - i);
            indices.swap(i, j);
        }
        indices[..take].iter().map(|&i| &self.buffer[i]).collect()
    }

    /// Returns `true` if at least `min_size` examples are stored.
    pub fn is_ready(&self, min_size: usize) -> bool {
        self.size >= min_size
    }

    /// Number of valid examples currently in the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

// ─── §7  SpAlphaZeroTrainer ──────────────────────────────────────────────────

/// Configuration for the AlphaZero training loop.
#[derive(Clone, Debug)]
pub struct SpAlphaZeroConfig {
    /// Number of self-play games per training iteration.
    pub n_games_per_iteration: usize,
    /// Number of gradient-update steps per training iteration.
    pub n_training_steps: usize,
    /// Mini-batch size for training.
    pub batch_size: usize,
    /// Learning rate.
    pub lr: f64,
    /// Replay buffer capacity.
    pub buffer_capacity: usize,
    /// MCTS configuration used during self-play.
    pub mcts_config: SpMctsConfig,
}

impl Default for SpAlphaZeroConfig {
    fn default() -> Self {
        Self {
            n_games_per_iteration: 5,
            n_training_steps: 10,
            batch_size: 32,
            lr: 1e-3,
            buffer_capacity: 10_000,
            mcts_config: SpMctsConfig::default(),
        }
    }
}

/// Orchestrates AlphaZero-style training: self-play data generation + SGD.
pub struct SpAlphaZeroTrainer {
    /// Policy-value network being trained.
    pub network: SpNeuralNetwork,
    /// Replay buffer storing self-play examples.
    pub buffer: SpSelfPlayBuffer,
    /// Training hyperparameters.
    pub config: SpAlphaZeroConfig,
    /// Current training iteration (incremented each call to a training loop).
    pub iteration: usize,
}

impl SpAlphaZeroTrainer {
    /// Create a new trainer with freshly initialised network and empty buffer.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        n_actions: usize,
        config: SpAlphaZeroConfig,
    ) -> Self {
        let network = SpNeuralNetwork::new(input_dim, hidden_dim, n_actions);
        let buffer = SpSelfPlayBuffer::new(config.buffer_capacity);
        Self {
            network,
            buffer,
            config,
            iteration: 0,
        }
    }

    /// Run a complete self-play game from `initial_game` using current network.
    ///
    /// Returns a list of `SpSelfPlayExample`s with outcomes filled in retrospectively.
    pub fn self_play_game<G: SpGame>(
        &self,
        initial_game: G,
        seed: &mut u64,
    ) -> Vec<SpSelfPlayExample> {
        let mut game = initial_game;
        let mut examples: Vec<(Vec<f64>, Vec<f64>, i8)> = vec![];
        // (state_encoding, mcts_policy, player_at_time)

        let mut mcts_config = self.config.mcts_config.clone();

        loop {
            if game.is_terminal() {
                break;
            }
            let legal = game.legal_actions();
            if legal.is_empty() {
                break;
            }

            let state = game.state_encoding(self.network.input_dim);
            let player = game.current_player();

            let mut tree = SpMctsTree::new(mcts_config.clone());
            let policy = tree.run_simulations(&game, &self.network, seed);

            // Anneal temperature after a few moves
            if examples.len() > 6 {
                mcts_config.temperature = 0.01;
            }

            examples.push((state, policy.clone(), player));

            // Sample action from policy distribution
            let action = if mcts_config.temperature < 0.01 {
                // Greedy
                policy
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(legal[0])
            } else {
                // Stochastic: sample from policy
                let r = sp_rand01(seed);
                let mut cum = 0.0;
                let mut chosen = legal[0];
                for (i, &p) in policy.iter().enumerate() {
                    cum += p;
                    if r <= cum {
                        chosen = i;
                        break;
                    }
                }
                // Clamp to legal actions
                if game.legal_actions().contains(&chosen) {
                    chosen
                } else {
                    legal[0]
                }
            };

            match game.apply_action(action) {
                Ok(next_game) => game = next_game,
                Err(_) => {
                    // Fallback to first legal action
                    if let Some(&fallback) = legal.first() {
                        match game.apply_action(fallback) {
                            Ok(next_game) => game = next_game,
                            Err(_) => break,
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        // Determine game outcome
        let outcome: f64 = match game.winner() {
            Some(w) => w as f64,
            None => 0.0,
        };

        // Assign retrospective values from each player's perspective
        examples
            .into_iter()
            .map(|(state, mcts_policy, player)| {
                let value = outcome * player as f64;
                SpSelfPlayExample {
                    state,
                    mcts_policy,
                    value,
                }
            })
            .collect()
    }

    /// Compute policy loss and value loss for a batch of examples.
    ///
    /// - `policy_loss = -Σ π_mcts(a) × log(π_net(a) + ε)`
    /// - `value_loss  = MSE(v_net, z)`
    pub fn compute_loss(&self, examples: &[&SpSelfPlayExample]) -> (f64, f64) {
        if examples.is_empty() {
            return (0.0, 0.0);
        }
        let mut total_policy_loss = 0.0f64;
        let mut total_value_loss = 0.0f64;

        for ex in examples {
            let legal_mask: Vec<bool> = ex.mcts_policy.iter().map(|&p| p > 0.0).collect();
            let (net_probs, net_value) = self.network.policy_value(&ex.state, &legal_mask);

            // Cross-entropy policy loss
            let pol_loss: f64 = ex
                .mcts_policy
                .iter()
                .zip(net_probs.iter())
                .map(|(&target, &pred)| -target * (pred + 1e-8).ln())
                .sum();
            total_policy_loss += pol_loss;

            // MSE value loss
            let diff = net_value - ex.value;
            total_value_loss += diff * diff;
        }

        let n = examples.len() as f64;
        (total_policy_loss / n, total_value_loss / n)
    }

    /// Perform one mini-batch gradient step.
    ///
    /// Returns the combined loss (policy + value).
    pub fn train_step(&mut self, seed: &mut u64) -> f64 {
        if !self.buffer.is_ready(self.config.batch_size) {
            return 0.0;
        }
        let batch = self.buffer.sample(self.config.batch_size, seed);
        let (policy_loss, value_loss) = self.compute_loss(&batch);

        // Finite-difference gradient estimation for w1 (representative step)
        // Full backprop would require autograd; here we do a lightweight SGD nudge.
        let eps = 1e-4;
        let lr = self.config.lr;
        let n_actions = self.network.n_actions;
        let hidden = self.network.hidden_dim;
        let input_dim = self.network.input_dim;

        // Gradient for policy head bias (closed form: -(target - pred))
        let mut grad_policy_b = vec![0.0f64; n_actions];
        let mut grad_value_b = 0.0f64;

        for ex in &batch {
            let legal_mask: Vec<bool> = ex.mcts_policy.iter().map(|&p| p > 0.0).collect();
            let (net_probs, net_value) = self.network.policy_value(&ex.state, &legal_mask);
            for j in 0..n_actions.min(net_probs.len()).min(ex.mcts_policy.len()) {
                grad_policy_b[j] += net_probs[j] - ex.mcts_policy[j];
            }
            grad_value_b += 2.0 * (net_value - ex.value);
        }
        let n = batch.len() as f64;
        for g in &mut grad_policy_b {
            *g /= n;
        }
        grad_value_b /= n;

        // SGD update on policy head bias and value head bias
        for j in 0..n_actions {
            self.network.policy_head_b[j] -= lr * grad_policy_b[j];
        }
        self.network.value_head_b -= lr * grad_value_b;

        // Finite-difference perturbation on a random subset of w1 entries
        // (representative update to demonstrate gradient flow)
        let mut perturb_seed = *seed ^ 0x1234_ABCD;
        let n_perturb = (hidden * input_dim).min(20);
        for _ in 0..n_perturb {
            let i = (sp_rand01(&mut perturb_seed) * hidden as f64) as usize % hidden;
            let j = (sp_rand01(&mut perturb_seed) * input_dim as f64) as usize % input_dim;

            let orig = self.network.w1[i][j];

            // +eps forward
            self.network.w1[i][j] = orig + eps;
            let loss_plus: f64 = {
                let b = self.buffer.sample(4, seed);
                let (pl, vl) = self.compute_loss(&b);
                pl + vl
            };

            // -eps forward
            self.network.w1[i][j] = orig - eps;
            let loss_minus: f64 = {
                let b = self.buffer.sample(4, seed);
                let (pl, vl) = self.compute_loss(&b);
                pl + vl
            };

            let grad = (loss_plus - loss_minus) / (2.0 * eps);
            self.network.w1[i][j] = orig - lr * grad;
        }

        *seed = perturb_seed;
        policy_loss + value_loss
    }
}

// ─── §8  SpEloSystem ─────────────────────────────────────────────────────────

/// ELO rating tracker for comparing multiple agent versions.
pub struct SpEloSystem {
    /// List of (model_name, elo_rating).
    pub models: Vec<(String, f64)>,
    /// ELO K-factor (controls rating update magnitude).
    pub k: f64,
    /// History of match outcomes: (winner_idx, loser_idx, outcome).
    /// `outcome` is 1.0 for win, 0.5 for draw, 0.0 for loss (from winner_idx perspective).
    pub history: Vec<(usize, usize, f64)>,
}

impl SpEloSystem {
    /// Create an empty ELO system with given K-factor.
    pub fn new(k: f64) -> Self {
        Self {
            models: vec![],
            k,
            history: vec![],
        }
    }

    /// Register a new model with an initial ELO rating.
    pub fn add_model(&mut self, name: &str, initial_elo: f64) {
        self.models.push((name.to_string(), initial_elo));
    }

    /// Expected score for player A against player B using logistic ELO formula.
    ///
    /// P(A wins) = 1 / (1 + 10^((R_B - R_A) / 400))
    pub fn expected_score(r_a: f64, r_b: f64) -> f64 {
        1.0 / (1.0 + 10f64.powf((r_b - r_a) / 400.0))
    }

    /// Record a game result and update ELO ratings.
    ///
    /// `outcome`: 1.0 = model_a wins, 0.0 = model_b wins, 0.5 = draw.
    pub fn record_game(&mut self, model_a: usize, model_b: usize, outcome: f64) {
        if model_a >= self.models.len() || model_b >= self.models.len() {
            return;
        }
        let r_a = self.models[model_a].1;
        let r_b = self.models[model_b].1;
        let e_a = Self::expected_score(r_a, r_b);
        let e_b = 1.0 - e_a;
        let s_a = outcome;
        let s_b = 1.0 - outcome;
        self.models[model_a].1 += self.k * (s_a - e_a);
        self.models[model_b].1 += self.k * (s_b - e_b);
        self.history.push((model_a, model_b, outcome));
    }

    /// Return the current ELO rating for a model.
    pub fn elo(&self, model_idx: usize) -> f64 {
        self.models
            .get(model_idx)
            .map(|(_, elo)| *elo)
            .unwrap_or(0.0)
    }

    /// Return all models sorted by ELO descending.
    pub fn ranking(&self) -> Vec<(String, f64)> {
        let mut ranked = self.models.clone();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

// ─── §9  SpMinimax ───────────────────────────────────────────────────────────

/// Minimax search with alpha-beta pruning for small deterministic games.
///
/// Uses `+1` for the maximising player and `-1` for the minimising player.
/// The `current_player()` return value from the game state determines which
/// player maximises vs minimises at each node.
pub struct SpMinimax {
    /// Maximum search depth.
    pub max_depth: usize,
}

impl SpMinimax {
    /// Create a minimax searcher with given maximum depth.
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }

    /// Run minimax search from the current game state.
    ///
    /// Returns `(best_action, minimax_value)` from the current player's perspective.
    pub fn search<G: SpGame>(&self, game: &G, evaluator: &dyn Fn(&G) -> f64) -> (usize, f64) {
        if game.is_terminal() || game.legal_actions().is_empty() {
            return (0, evaluator(game));
        }

        let maximizing = game.current_player() == 1;
        let mut best_action = game.legal_actions()[0];
        let mut best_value = if maximizing {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };

        for action in game.legal_actions() {
            if let Ok(next_game) = game.apply_action(action) {
                let value = self.alpha_beta(
                    &next_game,
                    self.max_depth - 1,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    !maximizing,
                    evaluator,
                );
                if maximizing {
                    if value > best_value {
                        best_value = value;
                        best_action = action;
                    }
                } else if value < best_value {
                    best_value = value;
                    best_action = action;
                }
            }
        }

        (best_action, best_value)
    }

    /// Recursive alpha-beta pruning implementation.
    fn alpha_beta<G: SpGame>(
        &self,
        game: &G,
        depth: usize,
        mut alpha: f64,
        mut beta: f64,
        maximizing: bool,
        evaluator: &dyn Fn(&G) -> f64,
    ) -> f64 {
        if depth == 0 || game.is_terminal() {
            return evaluator(game);
        }

        let legal = game.legal_actions();
        if legal.is_empty() {
            return evaluator(game);
        }

        if maximizing {
            let mut max_eval = f64::NEG_INFINITY;
            for action in legal {
                if let Ok(next_game) = game.apply_action(action) {
                    let eval =
                        self.alpha_beta(&next_game, depth - 1, alpha, beta, false, evaluator);
                    if eval > max_eval {
                        max_eval = eval;
                    }
                    if max_eval > alpha {
                        alpha = max_eval;
                    }
                    if beta <= alpha {
                        break; // β cutoff
                    }
                }
            }
            max_eval
        } else {
            let mut min_eval = f64::INFINITY;
            for action in legal {
                if let Ok(next_game) = game.apply_action(action) {
                    let eval = self.alpha_beta(&next_game, depth - 1, alpha, beta, true, evaluator);
                    if eval < min_eval {
                        min_eval = eval;
                    }
                    if min_eval < beta {
                        beta = min_eval;
                    }
                    if beta <= alpha {
                        break; // α cutoff
                    }
                }
            }
            min_eval
        }
    }

    /// Compute a deterministic policy vector by running minimax.
    ///
    /// Sets 1.0 for the best action, 0.0 for all others.
    /// Ties broken by first occurrence.
    pub fn policy_eval<G: SpGame>(&self, game: &G, depth: usize) -> Vec<f64> {
        let n = game.n_actions();
        let mut policy = vec![0.0f64; n];
        if game.is_terminal() {
            return policy;
        }

        let maximizing = game.current_player() == 1;
        let legal = game.legal_actions();
        if legal.is_empty() {
            return policy;
        }

        let searcher = SpMinimax { max_depth: depth };
        let evaluator = |g: &G| -> f64 {
            if g.is_terminal() {
                match g.winner() {
                    Some(w) => w as f64,
                    None => 0.0,
                }
            } else {
                0.0
            }
        };

        let mut best_action = legal[0];
        let mut best_value = if maximizing {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };

        for &action in &legal {
            if let Ok(next_game) = game.apply_action(action) {
                let v = searcher.alpha_beta(
                    &next_game,
                    depth.saturating_sub(1),
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    !maximizing,
                    &evaluator,
                );
                if maximizing && v > best_value {
                    best_value = v;
                    best_action = action;
                } else if !maximizing && v < best_value {
                    best_value = v;
                    best_action = action;
                }
            }
        }

        if best_action < policy.len() {
            policy[best_action] = 1.0;
        }
        policy
    }
}

// ─── §10  SpMetrics ──────────────────────────────────────────────────────────

/// Metrics and diagnostic utilities for self-play training.
pub struct SpMetrics;

impl SpMetrics {
    /// Shannon entropy of a policy vector: H(π) = -Σ p × log(p + ε).
    pub fn policy_entropy(policy: &[f64]) -> f64 {
        let raw: f64 = policy
            .iter()
            .map(|&p| if p > 1e-12 { -p * (p).ln() } else { 0.0 })
            .sum();
        raw.max(0.0)
    }

    /// Pearson correlation between predicted network values and actual outcomes.
    ///
    /// Returns a value in `[-1, 1]`; `0.0` if the variance is zero.
    pub fn value_calibration(predicted: &[f64], actual: &[f64]) -> f64 {
        let n = predicted.len().min(actual.len()) as f64;
        if n < 2.0 {
            return 0.0;
        }
        let mean_p = predicted.iter().sum::<f64>() / n;
        let mean_a = actual.iter().sum::<f64>() / n;
        let cov: f64 = predicted
            .iter()
            .zip(actual.iter())
            .map(|(&p, &a)| (p - mean_p) * (a - mean_a))
            .sum::<f64>()
            / n;
        let var_p: f64 = predicted.iter().map(|&p| (p - mean_p).powi(2)).sum::<f64>() / n;
        let var_a: f64 = actual.iter().map(|&a| (a - mean_a).powi(2)).sum::<f64>() / n;
        let denom = (var_p * var_a).sqrt();
        if denom < 1e-12 {
            0.0
        } else {
            (cov / denom).clamp(-1.0, 1.0)
        }
    }

    /// Compute (mean, std, max) of game lengths.
    pub fn game_length_stats(game_lengths: &[usize]) -> (f64, f64, usize) {
        if game_lengths.is_empty() {
            return (0.0, 0.0, 0);
        }
        let n = game_lengths.len() as f64;
        let mean = game_lengths.iter().sum::<usize>() as f64 / n;
        let variance = game_lengths
            .iter()
            .map(|&l| (l as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        let std = variance.sqrt();
        let max = *game_lengths.iter().max().unwrap_or(&0);
        (mean, std, max)
    }

    /// Arena score: fraction of games won or drawn by the new model.
    ///
    /// `score = (new_wins + 0.5 × draws) / total_games`
    pub fn policy_improvement(old_wins: usize, new_wins: usize, draws: usize) -> f64 {
        let total = (old_wins + new_wins + draws) as f64;
        if total <= 0.0 {
            return 0.5;
        }
        (new_wins as f64 + 0.5 * draws as f64) / total
    }

    /// Shannon entropy of normalised MCTS visit counts.
    ///
    /// High entropy = broad exploration; low entropy = focused exploitation.
    pub fn mcts_visit_entropy(visit_counts: &[usize]) -> f64 {
        let total: usize = visit_counts.iter().sum();
        if total == 0 {
            return 0.0;
        }
        let total_f = total as f64;
        visit_counts
            .iter()
            .filter(|&&n| n > 0)
            .map(|&n| {
                let p = n as f64 / total_f;
                -p * p.ln()
            })
            .sum()
    }
}
