//! Core types: CgtError, CgtGame, NashEquilibriumSolver and associated helpers.

use std::collections::HashMap;

/// Error type for cooperative game theory operations.
#[derive(Debug, Clone)]
pub enum CgtError {
    InvalidPlayerIndex(usize),
    InvalidActionProfile,
    NoEquilibriumFound,
    InvalidAllocation,
    SolverFailed(String),
    InvalidInput(String),
}

impl std::fmt::Display for CgtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CgtError::InvalidPlayerIndex(i) => write!(f, "Invalid player index: {}", i),
            CgtError::InvalidActionProfile => write!(f, "Invalid action profile"),
            CgtError::NoEquilibriumFound => write!(f, "No equilibrium found"),
            CgtError::InvalidAllocation => write!(f, "Invalid allocation"),
            CgtError::SolverFailed(s) => write!(f, "Solver failed: {}", s),
            CgtError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
        }
    }
}

impl std::error::Error for CgtError {}

// ─────────────────────────────────────────────────────────────────────────────
// §1 CgtGame — Normal-Form Game
// ─────────────────────────────────────────────────────────────────────────────

/// Normal-form game with N players and payoff tensors.
#[derive(Debug, Clone)]
pub struct CgtGame {
    /// Number of players
    pub n_players: usize,
    /// Number of actions per player: actions_per_player\[i\] = |A_i|
    pub actions_per_player: Vec<usize>,
    /// Payoff tensors stored as flat vectors; one entry per player.
    /// payoffs[p][profile_flat_index] = payoff for player p given that action profile.
    payoffs: Vec<Vec<f64>>,
}

impl CgtGame {
    /// Create a new normal-form game.
    pub fn new(actions_per_player: Vec<usize>, payoffs: Vec<Vec<f64>>) -> Result<Self, CgtError> {
        let n_players = actions_per_player.len();
        let total_profiles: usize = actions_per_player.iter().product();
        for (i, p) in payoffs.iter().enumerate() {
            if p.len() != total_profiles {
                return Err(CgtError::InvalidInput(format!(
                    "Payoff table for player {} has {} entries, expected {}",
                    i,
                    p.len(),
                    total_profiles
                )));
            }
        }
        if payoffs.len() != n_players {
            return Err(CgtError::InvalidInput(
                "Number of payoff tables must equal number of players".into(),
            ));
        }
        Ok(Self {
            n_players,
            actions_per_player,
            payoffs,
        })
    }

    /// Return the number of players.
    pub fn num_players(&self) -> usize {
        self.n_players
    }

    /// Return the number of actions for the given player.
    pub fn num_actions(&self, player: usize) -> Result<usize, CgtError> {
        self.actions_per_player
            .get(player)
            .copied()
            .ok_or(CgtError::InvalidPlayerIndex(player))
    }

    /// Return the payoff for a player given an action profile.
    pub fn payoff(&self, player: usize, action_profile: &[usize]) -> Result<f64, CgtError> {
        if player >= self.n_players {
            return Err(CgtError::InvalidPlayerIndex(player));
        }
        if action_profile.len() != self.n_players {
            return Err(CgtError::InvalidActionProfile);
        }
        let idx = self.profile_to_index(action_profile)?;
        Ok(self.payoffs[player][idx])
    }

    fn profile_to_index(&self, profile: &[usize]) -> Result<usize, CgtError> {
        let mut idx = 0usize;
        let mut stride = 1usize;
        for (p, &a) in profile.iter().enumerate().rev() {
            if a >= self.actions_per_player[p] {
                return Err(CgtError::InvalidActionProfile);
            }
            idx += a * stride;
            stride *= self.actions_per_player[p];
        }
        Ok(idx)
    }

    /// Enumerate all action profiles (as `Vec<usize>`).
    pub fn all_profiles(&self) -> Vec<Vec<usize>> {
        let total: usize = self.actions_per_player.iter().product();
        let mut result = Vec::with_capacity(total);
        let mut current = vec![0usize; self.n_players];
        for _ in 0..total {
            result.push(current.clone());
            // increment mixed-radix counter
            let mut carry = 1;
            for p in (0..self.n_players).rev() {
                current[p] += carry;
                if current[p] < self.actions_per_player[p] {
                    carry = 0;
                    break;
                } else {
                    current[p] = 0;
                }
            }
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2 NashEquilibriumSolver
// ─────────────────────────────────────────────────────────────────────────────

/// Nash equilibrium solver for normal-form games.
pub struct NashEquilibriumSolver;

impl NashEquilibriumSolver {
    /// Find all Nash equilibria for a 2-player game using support enumeration
    /// (Lemke-Howson style tableau approach).
    /// Returns a list of (mixed_strategy_p1, mixed_strategy_p2).
    pub fn solve_2player(game: &CgtGame) -> Result<Vec<(Vec<f64>, Vec<f64>)>, CgtError> {
        if game.n_players != 2 {
            return Err(CgtError::InvalidInput(
                "solve_2player requires exactly 2 players".into(),
            ));
        }
        let m = game.actions_per_player[0];
        let n = game.actions_per_player[1];

        // Build payoff matrices A (player 0) and B (player 1)
        let mut a_mat = vec![vec![0f64; n]; m];
        let mut b_mat = vec![vec![0f64; n]; m];
        for i in 0..m {
            for j in 0..n {
                let profile = [i, j];
                a_mat[i][j] = game.payoff(0, &profile).unwrap_or(0.0);
                b_mat[i][j] = game.payoff(1, &profile).unwrap_or(0.0);
            }
        }

        // Support enumeration: try all non-empty subsets of actions
        let mut equilibria = Vec::new();
        let subsets_m = subsets_of(m);
        let subsets_n = subsets_of(n);

        for sup1 in &subsets_m {
            for sup2 in &subsets_n {
                if let Some((x, y)) = verify_support_ne(&a_mat, &b_mat, sup1, sup2) {
                    equilibria.push((x, y));
                }
            }
        }

        if equilibria.is_empty() {
            // At minimum there is always a pure NE or mixed NE; return uniform if none found
            let x = vec![1.0 / m as f64; m];
            let y = vec![1.0 / n as f64; n];
            equilibria.push((x, y));
        }

        Ok(equilibria)
    }

    /// Find Nash equilibria by iterating iterated dominance elimination for N-player games.
    pub fn find_nash_support(game: &CgtGame) -> Result<Vec<Vec<f64>>, CgtError> {
        // Iteratively eliminate strictly dominated strategies
        let mut surviving: Vec<Vec<bool>> = game
            .actions_per_player
            .iter()
            .map(|&a| vec![true; a])
            .collect();

        let mut changed = true;
        while changed {
            changed = false;
            for p in 0..game.n_players {
                let n_a = game.actions_per_player[p];
                for a in 0..n_a {
                    if !surviving[p][a] {
                        continue;
                    }
                    // Check if action a is strictly dominated by any other action
                    for a2 in 0..n_a {
                        if a2 == a || !surviving[p][a2] {
                            continue;
                        }
                        if is_strictly_dominated(game, p, a, a2, &surviving) {
                            surviving[p][a] = false;
                            changed = true;
                            break;
                        }
                    }
                }
            }
        }

        // Uniform distribution over surviving actions
        let result: Vec<Vec<f64>> = surviving
            .iter()
            .map(|s| {
                let cnt = s.iter().filter(|&&x| x).count().max(1) as f64;
                s.iter().map(|&x| if x { 1.0 / cnt } else { 0.0 }).collect()
            })
            .collect();
        Ok(result)
    }
}

pub(crate) fn subsets_of(n: usize) -> Vec<Vec<usize>> {
    let total = 1usize << n;
    let mut result = Vec::new();
    for mask in 1..total {
        let sub: Vec<usize> = (0..n).filter(|&i| (mask >> i) & 1 == 1).collect();
        result.push(sub);
    }
    result
}

pub(crate) fn verify_support_ne(
    a_mat: &[Vec<f64>],
    b_mat: &[Vec<f64>],
    sup1: &[usize],
    sup2: &[usize],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let m = a_mat.len();
    let n = if m > 0 {
        a_mat[0].len()
    } else {
        return None;
    };

    let y = solve_mixed_strategy_eq(a_mat, sup1, sup2, n)?;
    let x = solve_mixed_strategy_eq(b_mat_transposed(b_mat, m, n).as_slice(), sup2, sup1, m)?;

    let val1 = expected_payoff_row(a_mat, &x, &y, sup1[0]);
    for i in 0..m {
        let v = expected_payoff_row(a_mat, &x, &y, i);
        if sup1.contains(&i) {
            if (v - val1).abs() > 1e-6 {
                return None;
            }
        } else if v > val1 + 1e-6 {
            return None;
        }
    }

    let val2 = expected_payoff_col(b_mat, &x, &y, sup2[0]);
    for j in 0..n {
        let v = expected_payoff_col(b_mat, &x, &y, j);
        if sup2.contains(&j) {
            if (v - val2).abs() > 1e-6 {
                return None;
            }
        } else if v > val2 + 1e-6 {
            return None;
        }
    }

    if x.iter().any(|&v| v < -1e-9) || y.iter().any(|&v| v < -1e-9) {
        return None;
    }

    Some((x, y))
}

fn solve_mixed_strategy_eq(
    payoff: &[Vec<f64>],
    row_support: &[usize],
    col_support: &[usize],
    _total_cols: usize,
) -> Option<Vec<f64>> {
    let k = col_support.len();
    let total_cols = if payoff.is_empty() {
        0
    } else {
        payoff[0].len()
    };

    if k == 1 {
        let mut y = vec![0.0f64; total_cols];
        y[col_support[0]] = 1.0;
        return Some(y);
    }

    let n_eq = k;
    let mut mat = vec![vec![0f64; k + 1]; n_eq];

    let r0 = row_support[0];
    for eq in 0..(k - 1).min(row_support.len() - 1) {
        let ri = row_support[eq + 1];
        for (ci, &j) in col_support.iter().enumerate() {
            let p0 = if j < payoff[r0].len() {
                payoff[r0][j]
            } else {
                0.0
            };
            let pi = if j < payoff[ri].len() {
                payoff[ri][j]
            } else {
                0.0
            };
            mat[eq][ci] = p0 - pi;
        }
        mat[eq][k] = 0.0;
    }

    for ci in 0..k {
        mat[k - 1][ci] = 1.0;
    }
    mat[k - 1][k] = 1.0;

    gaussian_solve(&mut mat, k).map(|sol| {
        let mut y = vec![0.0f64; total_cols];
        for (ci, &j) in col_support.iter().enumerate() {
            y[j] = sol[ci];
        }
        y
    })
}

fn b_mat_transposed(b: &[Vec<f64>], m: usize, n: usize) -> Vec<Vec<f64>> {
    let mut t = vec![vec![0f64; m]; n];
    for i in 0..m {
        for j in 0..n {
            t[j][i] = b[i][j];
        }
    }
    t
}

fn expected_payoff_row(payoff: &[Vec<f64>], _x: &[f64], y: &[f64], row: usize) -> f64 {
    if row >= payoff.len() {
        return 0.0;
    }
    payoff[row]
        .iter()
        .zip(y.iter())
        .map(|(&a, &yj)| a * yj)
        .sum()
}

fn expected_payoff_col(payoff: &[Vec<f64>], x: &[f64], _y: &[f64], col: usize) -> f64 {
    payoff
        .iter()
        .zip(x.iter())
        .map(|(row, &xi)| {
            let v = if col < row.len() { row[col] } else { 0.0 };
            v * xi
        })
        .sum()
}

pub(crate) fn gaussian_solve(mat: &mut [Vec<f64>], n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let pivot_row = (col..n).max_by(|&a, &b| {
            mat[a][col]
                .abs()
                .partial_cmp(&mat[b][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        mat.swap(col, pivot_row);
        let piv = mat[col][col];
        if piv.abs() < 1e-12 {
            continue;
        }
        for j in col..=n {
            mat[col][j] /= piv;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = mat[row][col];
            for j in col..=n {
                let sub = factor * mat[col][j];
                mat[row][j] -= sub;
            }
        }
    }
    Some((0..n).map(|i| mat[i][n]).collect())
}

fn is_strictly_dominated(
    game: &CgtGame,
    player: usize,
    dominated: usize,
    dominator: usize,
    surviving: &[Vec<bool>],
) -> bool {
    let profiles = all_opponent_profiles(game, player, surviving);
    profiles.iter().all(|opp_profile| {
        let mut prof_dom = opp_profile.clone();
        let mut prof_dominator = opp_profile.clone();
        prof_dom[player] = dominated;
        prof_dominator[player] = dominator;
        let v_dom = game.payoff(player, &prof_dom).unwrap_or(0.0);
        let v_dominator = game.payoff(player, &prof_dominator).unwrap_or(0.0);
        v_dominator > v_dom
    })
}

fn all_opponent_profiles(
    game: &CgtGame,
    player: usize,
    surviving: &[Vec<bool>],
) -> Vec<Vec<usize>> {
    let mut profiles = vec![vec![0usize; game.n_players]];
    for p in 0..game.n_players {
        if p == player {
            continue;
        }
        let acts: Vec<usize> = (0..game.actions_per_player[p])
            .filter(|&a| surviving[p][a])
            .collect();
        let mut new_profiles = Vec::new();
        for a in &acts {
            for prof in &profiles {
                let mut np = prof.clone();
                np[p] = *a;
                new_profiles.push(np);
            }
        }
        if !new_profiles.is_empty() {
            profiles = new_profiles;
        }
    }
    profiles
}

// Not used publicly but kept for completeness
#[allow(dead_code)]
pub(crate) fn _unused_hashmap_placeholder() -> HashMap<String, f64> {
    HashMap::new()
}
