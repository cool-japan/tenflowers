//! Counterfactual Regret Minimization (CFR) and online regret minimization.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// §3 CfrSolver — Counterfactual Regret Minimization
// ─────────────────────────────────────────────────────────────────────────────

/// A node in the CFR information set tree.
#[derive(Debug, Clone)]
pub struct CfrNode {
    pub n_actions: usize,
    pub regret_sum: Vec<f64>,
    pub strategy_sum: Vec<f64>,
}

impl CfrNode {
    pub fn new(n_actions: usize) -> Self {
        Self {
            n_actions,
            regret_sum: vec![0.0; n_actions],
            strategy_sum: vec![0.0; n_actions],
        }
    }

    /// Compute current strategy via regret matching.
    pub fn get_strategy(&self, realization_weight: f64) -> Vec<f64> {
        let mut strategy = vec![0.0f64; self.n_actions];
        let pos_sum: f64 = self.regret_sum.iter().map(|&r| r.max(0.0)).sum();
        if pos_sum > 0.0 {
            for (i, s) in strategy.iter_mut().enumerate() {
                *s = self.regret_sum[i].max(0.0) / pos_sum;
            }
        } else {
            let uniform = 1.0 / self.n_actions as f64;
            for s in strategy.iter_mut() {
                *s = uniform;
            }
        }
        let _ = realization_weight; // used externally
        strategy
    }

    /// Return average strategy (Nash approximation).
    pub fn get_average_strategy(&self) -> Vec<f64> {
        let total: f64 = self.strategy_sum.iter().sum();
        if total > 0.0 {
            self.strategy_sum.iter().map(|&s| s / total).collect()
        } else {
            vec![1.0 / self.n_actions as f64; self.n_actions]
        }
    }

    /// Update regret sums and strategy sums.
    pub fn update(&mut self, action_utilities: &[f64], node_utility: f64, reach_prob: f64) {
        let strategy = self.get_strategy(reach_prob);
        for (i, &u) in action_utilities.iter().enumerate().take(self.n_actions) {
            self.regret_sum[i] += reach_prob * (u - node_utility);
        }
        for (i, &s) in strategy.iter().enumerate() {
            self.strategy_sum[i] += reach_prob * s;
        }
    }
}

/// Information set identifier for CFR.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CfrInfoSet {
    pub player: usize,
    pub history: Vec<usize>,
}

impl CfrInfoSet {
    pub fn new(player: usize, history: Vec<usize>) -> Self {
        Self { player, history }
    }
}

/// CFR Solver for extensive-form games.
pub struct CfrSolver {
    pub nodes: HashMap<CfrInfoSet, CfrNode>,
    pub n_iterations: usize,
}

impl CfrSolver {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            n_iterations: 0,
        }
    }

    /// Run CFR for a Kuhn Poker-like two-player zero-sum game.
    /// Returns the exploitability after training.
    pub fn train_kuhn_poker(&mut self, n_iterations: usize) -> f64 {
        let mut rng = StdRng::seed_from_u64(0xbeef);
        let mut total_utility = 0.0f64;
        for _ in 0..n_iterations {
            // Shuffle deck [0,1,2] — deal two cards
            let mut deck = [0usize, 1, 2];
            // Fisher-Yates shuffle
            for i in (1..3).rev() {
                let j = rng.random_range(0_usize..=i);
                deck.swap(i, j);
            }
            let cards = [deck[0], deck[1]];
            total_utility += self.cfr_kuhn(&cards, &[], [1.0, 1.0]);
        }
        self.n_iterations += n_iterations;
        (total_utility / n_iterations as f64).abs()
    }

    fn cfr_kuhn(&mut self, cards: &[usize; 2], history: &[usize], reach_probs: [f64; 2]) -> f64 {
        let n = history.len();
        // Terminal states
        if n >= 2 {
            let last = history[n - 1];
            let second_last = history[n - 2];
            // Both pass
            if last == 0 && second_last == 0 {
                return if cards[0] > cards[1] { 1.0 } else { -1.0 };
            }
            // Both bet
            if last == 1 && second_last == 1 {
                return if cards[0] > cards[1] { 2.0 } else { -2.0 };
            }
            // Pass after bet = fold
            if last == 0 && second_last == 1 {
                return 1.0; // player 1 folds, player 0 wins
            }
            // Bet after pass = call
            if last == 1 && second_last == 0 && n >= 3 {
                return if cards[0] > cards[1] { 2.0 } else { -2.0 };
            }
        }

        let player = n % 2;
        let card = cards[player];
        let info_set = CfrInfoSet::new(player, {
            let mut h = vec![card];
            h.extend_from_slice(history);
            h
        });

        let n_actions = 2usize; // 0=pass, 1=bet
        self.nodes
            .entry(info_set.clone())
            .or_insert_with(|| CfrNode::new(n_actions));

        let strategy = match self.nodes.get(&info_set) {
            Some(node) => node.get_strategy(reach_probs[player]),
            None => vec![0.5; n_actions],
        };

        let mut action_utilities = vec![0.0f64; n_actions];
        for a in 0..n_actions {
            let mut new_history: Vec<usize> = history.to_vec();
            new_history.push(a);
            let mut new_reach = reach_probs;
            new_reach[player] *= strategy[a];
            action_utilities[a] = -self.cfr_kuhn(cards, &new_history, new_reach);
        }
        let node_utility: f64 = action_utilities
            .iter()
            .zip(strategy.iter())
            .map(|(&u, &s)| u * s)
            .sum();

        let opponent = 1 - player;
        if let Some(node) = self.nodes.get_mut(&info_set) {
            for i in 0..n_actions {
                node.regret_sum[i] += reach_probs[opponent] * (action_utilities[i] - node_utility);
            }
            for (i, &s) in strategy.iter().enumerate() {
                node.strategy_sum[i] += reach_probs[player] * s;
            }
        }

        node_utility
    }

    /// Retrieve the average strategy for a given information set.
    pub fn get_average_strategy(&self, info_set: &CfrInfoSet) -> Option<Vec<f64>> {
        self.nodes.get(info_set).map(|n| n.get_average_strategy())
    }
}

impl Default for CfrSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4 RegretMinimizer — Online Regret Minimization
// ─────────────────────────────────────────────────────────────────────────────

/// Algorithm variant for online regret minimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegretAlgorithm {
    RegretMatching,
    RegretMatchingPlus,
    Hedge,
    OnlineMirrorDescent,
}

/// Online regret minimizer with multiple algorithm variants.
pub struct RegretMinimizer {
    pub n_actions: usize,
    pub algorithm: RegretAlgorithm,
    /// η for Hedge / OMD
    pub eta: f64,
    cumulative_regret: Vec<f64>,
    cumulative_weights: Vec<f64>,
    t: usize,
}

impl RegretMinimizer {
    pub fn new(n_actions: usize, algorithm: RegretAlgorithm, eta: f64) -> Self {
        Self {
            n_actions,
            algorithm,
            eta,
            cumulative_regret: vec![0.0; n_actions],
            cumulative_weights: vec![1.0; n_actions],
            t: 0,
        }
    }

    /// Select an action index according to current strategy.
    pub fn select_action(&self) -> usize {
        let weights = self.current_distribution();
        let mut rng = StdRng::seed_from_u64(self.t as u64);
        let u: f64 = rng.random::<f64>();
        let mut cum = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            cum += w;
            if u <= cum {
                return i;
            }
        }
        self.n_actions - 1
    }

    /// Return the current mixed strategy.
    pub fn current_distribution(&self) -> Vec<f64> {
        match self.algorithm {
            RegretAlgorithm::RegretMatching => {
                let pos_sum: f64 = self.cumulative_regret.iter().map(|&r| r.max(0.0)).sum();
                if pos_sum > 0.0 {
                    self.cumulative_regret
                        .iter()
                        .map(|&r| r.max(0.0) / pos_sum)
                        .collect()
                } else {
                    vec![1.0 / self.n_actions as f64; self.n_actions]
                }
            }
            RegretAlgorithm::RegretMatchingPlus => {
                let pos_sum: f64 = self.cumulative_regret.iter().map(|&r| r.max(0.0)).sum();
                if pos_sum > 0.0 {
                    self.cumulative_regret
                        .iter()
                        .map(|&r| r.max(0.0) / pos_sum)
                        .collect()
                } else {
                    vec![1.0 / self.n_actions as f64; self.n_actions]
                }
            }
            RegretAlgorithm::Hedge | RegretAlgorithm::OnlineMirrorDescent => {
                let w_sum: f64 = self.cumulative_weights.iter().sum();
                if w_sum > 0.0 {
                    self.cumulative_weights.iter().map(|&w| w / w_sum).collect()
                } else {
                    vec![1.0 / self.n_actions as f64; self.n_actions]
                }
            }
        }
    }

    /// Observe utility vector and update internal state.
    pub fn observe_utility(&mut self, utilities: &[f64]) {
        if utilities.len() != self.n_actions {
            return;
        }
        self.t += 1;
        let dist = self.current_distribution();
        let expected: f64 = dist
            .iter()
            .zip(utilities.iter())
            .map(|(&d, &u)| d * u)
            .sum();

        match self.algorithm {
            RegretAlgorithm::RegretMatching => {
                for (i, (&u, r)) in utilities
                    .iter()
                    .zip(self.cumulative_regret.iter_mut())
                    .enumerate()
                {
                    let _ = i;
                    *r += u - expected;
                }
            }
            RegretAlgorithm::RegretMatchingPlus => {
                for (&u, r) in utilities.iter().zip(self.cumulative_regret.iter_mut()) {
                    *r = (*r + (u - expected)).max(0.0);
                }
            }
            RegretAlgorithm::Hedge => {
                for (&u, w) in utilities.iter().zip(self.cumulative_weights.iter_mut()) {
                    *w *= (self.eta * u).exp();
                }
            }
            RegretAlgorithm::OnlineMirrorDescent => {
                let step = self.eta / (self.t as f64).sqrt();
                let mut theta: Vec<f64> = self.cumulative_weights.iter().map(|&w| w.ln()).collect();
                for (&u, t) in utilities.iter().zip(theta.iter_mut()) {
                    *t += step * u;
                }
                // Softmax projection
                let max_t = theta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_sum: f64 = theta.iter().map(|&t| (t - max_t).exp()).sum();
                for (w, &t) in self.cumulative_weights.iter_mut().zip(theta.iter()) {
                    *w = (t - max_t).exp() / exp_sum;
                }
            }
        }
    }

    /// Upper bound on cumulative regret.
    pub fn regret_bound(&self) -> f64 {
        let max_utility = 1.0_f64;
        match self.algorithm {
            RegretAlgorithm::RegretMatching | RegretAlgorithm::RegretMatchingPlus => {
                max_utility * ((self.n_actions as f64) * (self.t as f64)).sqrt()
            }
            RegretAlgorithm::Hedge | RegretAlgorithm::OnlineMirrorDescent => {
                (self.n_actions as f64).ln().sqrt() * (self.t as f64).sqrt() / self.eta
            }
        }
    }
}
