//! Cooperative game solution concepts: CgtCooperativeGame, ShapleyValueCalculator,
//! BanzhafIndex, CoreSolver.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;

use super::types::CgtError;

// ─────────────────────────────────────────────────────────────────────────────
// §5 CgtCooperativeGame — Characteristic Function Game
// ─────────────────────────────────────────────────────────────────────────────

/// Characteristic function game v: 2^n → R for N players.
/// Coalition membership is represented by a u64 bitmask.
#[derive(Debug, Clone)]
pub struct CgtCooperativeGame {
    pub n_players: usize,
    coalition_values: HashMap<u64, f64>,
}

impl CgtCooperativeGame {
    pub fn new(n_players: usize) -> Self {
        assert!(n_players <= 63, "At most 63 players supported");
        Self {
            n_players,
            coalition_values: HashMap::new(),
        }
    }

    /// Set the value of a coalition given as bitmask.
    pub fn set_coalition_value(&mut self, coalition_mask: u64, value: f64) {
        self.coalition_values.insert(coalition_mask, value);
    }

    /// Get the value of a coalition given as bitmask.
    pub fn compute_coalition_value(&self, coalition_mask: u64) -> f64 {
        *self.coalition_values.get(&coalition_mask).unwrap_or(&0.0)
    }

    /// Value of the grand coalition (all players).
    pub fn grand_coalition_value(&self) -> f64 {
        let mask = (1u64 << self.n_players) - 1;
        self.compute_coalition_value(mask)
    }

    /// Check superadditivity: v(S ∪ T) >= v(S) + v(T) for all disjoint S, T.
    pub fn is_superadditive(&self) -> bool {
        let n = self.n_players;
        let total = 1u64 << n;
        for s in 1..total {
            for t in 1..total {
                if s & t != 0 {
                    continue;
                } // not disjoint
                let vs = self.compute_coalition_value(s);
                let vt = self.compute_coalition_value(t);
                let vst = self.compute_coalition_value(s | t);
                if vst < vs + vt - 1e-9 {
                    return false;
                }
            }
        }
        true
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6 ShapleyValueCalculator
// ─────────────────────────────────────────────────────────────────────────────

/// Computes Shapley values for cooperative games.
pub struct ShapleyValueCalculator;

impl ShapleyValueCalculator {
    /// Exact Shapley values via all 2^n marginal contribution computations.
    pub fn compute_exact(game: &CgtCooperativeGame) -> Vec<f64> {
        let n = game.n_players;
        let mut shapley = vec![0.0f64; n];
        let total = 1u64 << n;

        for coalition in 0..total {
            let size = coalition.count_ones() as usize;
            let weight_f = marginal_weight(size, n);
            for i in 0..n {
                if (coalition >> i) & 1 == 1 {
                    continue;
                }
                let with_i = coalition | (1u64 << i);
                let v_with = game.compute_coalition_value(with_i);
                let v_without = game.compute_coalition_value(coalition);
                shapley[i] += weight_f * (v_with - v_without);
            }
        }
        shapley
    }

    /// Monte Carlo approximation via random permutation sampling.
    pub fn compute_approx(game: &CgtCooperativeGame, n_samples: usize) -> Vec<f64> {
        let n = game.n_players;
        let mut shapley = vec![0.0f64; n];
        let mut rng = StdRng::seed_from_u64(0xbeef);

        let mut perm: Vec<usize> = (0..n).collect();
        for _ in 0..n_samples {
            // Fisher-Yates shuffle
            for i in (1..n).rev() {
                let j = rng.random_range(0_usize..=i);
                perm.swap(i, j);
            }
            let mut coalition = 0u64;
            for &player in &perm {
                let v_before = game.compute_coalition_value(coalition);
                coalition |= 1u64 << player;
                let v_after = game.compute_coalition_value(coalition);
                shapley[player] += v_after - v_before;
            }
        }
        shapley.iter_mut().for_each(|s| *s /= n_samples as f64);
        shapley
    }
}

fn marginal_weight(s: usize, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if s >= n {
        return 0.0;
    }
    let log_num = log_factorial(s) + log_factorial(n - 1 - s);
    let log_den = log_factorial(n);
    (log_num - log_den).exp()
}

fn log_factorial(n: usize) -> f64 {
    (1..=n).map(|k| (k as f64).ln()).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// §7 BanzhafIndex
// ─────────────────────────────────────────────────────────────────────────────

/// Banzhaf power index for cooperative games.
pub struct BanzhafIndex;

impl BanzhafIndex {
    /// Compute normalised Banzhaf index for all players.
    pub fn compute(game: &CgtCooperativeGame) -> Vec<f64> {
        let n = game.n_players;
        let total = 1u64 << n;
        let mut swings = vec![0.0f64; n];

        for coalition in 0..total {
            for i in 0..n {
                if (coalition >> i) & 1 == 1 {
                    continue;
                }
                let with_i = coalition | (1u64 << i);
                let v_without = game.compute_coalition_value(coalition);
                let v_with = game.compute_coalition_value(with_i);
                if (v_with - v_without).abs() > 1e-9 {
                    swings[i] += 1.0;
                }
            }
        }

        let total_swings: f64 = swings.iter().sum();
        if total_swings > 0.0 {
            swings.iter().map(|&s| s / total_swings).collect()
        } else {
            vec![1.0 / n as f64; n]
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8 CoreSolver
// ─────────────────────────────────────────────────────────────────────────────

/// Solver for the core of a cooperative game.
pub struct CoreSolver;

impl CoreSolver {
    /// Check if a given allocation is in the core.
    /// Core condition: sum(x_i for i in S) >= v(S) for all coalitions S,
    /// and sum(x_i) = v(N).
    pub fn is_in_core(game: &CgtCooperativeGame, allocation: &[f64]) -> bool {
        let n = game.n_players;
        if allocation.len() != n {
            return false;
        }
        let total: f64 = allocation.iter().sum();
        let gv = game.grand_coalition_value();
        if (total - gv).abs() > 1e-6 {
            return false;
        }

        let all_coalitions = 1u64 << n;
        for s in 1..all_coalitions {
            let coalition_val = game.compute_coalition_value(s);
            let coalition_sum: f64 = (0..n)
                .filter(|&i| (s >> i) & 1 == 1)
                .map(|i| allocation[i])
                .sum();
            if coalition_sum < coalition_val - 1e-6 {
                return false;
            }
        }
        true
    }

    /// Find a core allocation using a simplex-like feasibility approach.
    /// Minimises the maximum excess over all coalitions.
    pub fn find_core_allocation(game: &CgtCooperativeGame) -> Result<Vec<f64>, CgtError> {
        let n = game.n_players;
        let gv = game.grand_coalition_value();

        // Start with Shapley values as initial point
        let mut alloc = ShapleyValueCalculator::compute_exact(game);
        let alloc_sum: f64 = alloc.iter().sum();
        if alloc_sum.abs() > 1e-12 {
            let scale = gv / alloc_sum;
            alloc.iter_mut().for_each(|a| *a *= scale);
        } else {
            alloc = vec![gv / n as f64; n];
        }

        if CoreSolver::is_in_core(game, &alloc) {
            return Ok(alloc);
        }

        // Gradient-descent-like projection onto core constraints
        for _ in 0..1000 {
            let all_coalitions = 1u64 << n;
            let mut violated = false;
            for s in 1..all_coalitions {
                let cv = game.compute_coalition_value(s);
                let cs: f64 = (0..n)
                    .filter(|&i| (s >> i) & 1 == 1)
                    .map(|i| alloc[i])
                    .sum();
                if cs < cv - 1e-6 {
                    violated = true;
                    let deficit = cv - cs;
                    let members: Vec<usize> = (0..n).filter(|&i| (s >> i) & 1 == 1).collect();
                    let cnt = members.len() as f64;
                    let share = deficit / cnt;
                    for &i in &members {
                        alloc[i] += share;
                    }
                    let sum: f64 = alloc.iter().sum();
                    if sum.abs() > 1e-12 {
                        let sc = gv / sum;
                        alloc.iter_mut().for_each(|a| *a *= sc);
                    }
                }
            }
            if !violated {
                break;
            }
        }

        // Return best effort even if not strictly in core (empty core game)
        Ok(alloc)
    }

    /// Nucleolus approximation via lexicographic minimax excess.
    pub fn nucleolus_approximation(game: &CgtCooperativeGame) -> Vec<f64> {
        let n = game.n_players;
        let gv = game.grand_coalition_value();
        let mut alloc = vec![gv / n as f64; n];

        for _ in 0..100 {
            let all_coalitions = 1u64 << n;
            let mut max_excess = f64::NEG_INFINITY;
            let mut worst_s = 1u64;
            for s in 1..all_coalitions {
                let cv = game.compute_coalition_value(s);
                let cs: f64 = (0..n)
                    .filter(|&i| (s >> i) & 1 == 1)
                    .map(|i| alloc[i])
                    .sum();
                let excess = cv - cs;
                if excess > max_excess {
                    max_excess = excess;
                    worst_s = s;
                }
            }
            if max_excess <= 0.0 {
                break;
            }
            let members: Vec<usize> = (0..n).filter(|&i| (worst_s >> i) & 1 == 1).collect();
            let cnt = members.len() as f64;
            let step = max_excess / (cnt * 2.0);
            for &i in &members {
                alloc[i] += step;
            }
            let not_members: Vec<usize> = (0..n).filter(|&i| (worst_s >> i) & 1 == 0).collect();
            if !not_members.is_empty() {
                let total_step = step * cnt / not_members.len() as f64;
                for &i in &not_members {
                    alloc[i] -= total_step;
                }
            }
        }
        alloc
    }
}
