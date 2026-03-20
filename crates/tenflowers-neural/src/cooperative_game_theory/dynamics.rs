//! Evolutionary game dynamics, neural Nash finding, and game-theoretic metrics.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

use super::cooperative::{CgtCooperativeGame, ShapleyValueCalculator};
use super::types::{CgtError, CgtGame, NashEquilibriumSolver};

// ─────────────────────────────────────────────────────────────────────────────
// §11 EvolutionaryGameDynamics
// ─────────────────────────────────────────────────────────────────────────────

/// Standard symmetric game payoff matrices.
#[derive(Debug, Clone, Copy)]
pub enum SymmetricGame {
    HawkDove { v: f64, c: f64 },
    RockPaperScissors,
    PrisonersDilemma { r: f64, s: f64, t: f64, p: f64 },
    Custom,
}

/// Replicator dynamics for evolutionary games.
pub struct EvolutionaryGameDynamics {
    pub payoff_matrix: Vec<Vec<f64>>,
    pub dt: f64,
    pub population: Vec<f64>,
}

impl EvolutionaryGameDynamics {
    pub fn new(payoff_matrix: Vec<Vec<f64>>, dt: f64) -> Result<Self, CgtError> {
        let n = payoff_matrix.len();
        if n == 0 {
            return Err(CgtError::InvalidInput("Empty payoff matrix".into()));
        }
        let pop = vec![1.0 / n as f64; n];
        Ok(Self {
            payoff_matrix,
            dt,
            population: pop,
        })
    }

    pub fn from_symmetric_game(game: SymmetricGame) -> Result<Self, CgtError> {
        let mat = match game {
            SymmetricGame::HawkDove { v, c } => vec![vec![(v - c) / 2.0, v], vec![0.0, v / 2.0]],
            SymmetricGame::RockPaperScissors => vec![
                vec![0.0, -1.0, 1.0],
                vec![1.0, 0.0, -1.0],
                vec![-1.0, 1.0, 0.0],
            ],
            SymmetricGame::PrisonersDilemma { r, s, t, p } => vec![vec![r, s], vec![t, p]],
            SymmetricGame::Custom => vec![vec![0.0]],
        };
        Self::new(mat, 0.01)
    }

    /// Compute fitness vector Ax for current population x.
    pub fn fitness(&self) -> Vec<f64> {
        let x = &self.population;
        let n = self.payoff_matrix.len();
        (0..n)
            .map(|i| {
                self.payoff_matrix[i]
                    .iter()
                    .zip(x.iter())
                    .map(|(&a, &xj)| a * xj)
                    .sum::<f64>()
            })
            .collect()
    }

    /// One Euler step of replicator dynamics: dx_i/dt = x_i * (f_i - f_bar).
    pub fn step(&mut self) {
        let f = self.fitness();
        let f_bar: f64 = self
            .population
            .iter()
            .zip(f.iter())
            .map(|(&xi, &fi)| xi * fi)
            .sum();
        let n = self.population.len();
        let mut new_pop = vec![0.0f64; n];
        for i in 0..n {
            new_pop[i] = self.population[i] + self.dt * self.population[i] * (f[i] - f_bar);
        }
        let sum: f64 = new_pop.iter().map(|&x| x.max(0.0)).sum();
        if sum > 0.0 {
            self.population = new_pop.iter().map(|&x| x.max(0.0) / sum).collect();
        }
    }

    /// Run until convergence or max_steps.
    pub fn run(&mut self, max_steps: usize) -> Vec<f64> {
        for _ in 0..max_steps {
            let prev = self.population.clone();
            self.step();
            let diff: f64 = self
                .population
                .iter()
                .zip(prev.iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum();
            if diff < 1e-8 {
                break;
            }
        }
        self.population.clone()
    }

    /// Find evolutionary stable strategy (fixed point of replicator dynamics).
    pub fn find_evolutionary_stable_strategy(&mut self) -> Vec<f64> {
        self.run(10_000)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §12 CgtNeuralNash — Neural Nash Finding
// ─────────────────────────────────────────────────────────────────────────────

/// Simple 3-layer MLP for Nash finding (input: payoff matrix, output: mixed strategies).
#[derive(Debug, Clone)]
pub struct NashNet {
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
    w3: Vec<Vec<f64>>,
    b3: Vec<f64>,
    pub input_dim: usize,
    pub output_dim: usize,
    hidden_dim: usize,
}

impl NashNet {
    pub fn new(input_dim: usize, output_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(12345);
        let scale1 = (2.0f64 / input_dim as f64).sqrt();
        let scale2 = (2.0f64 / hidden_dim as f64).sqrt();
        let w1 = (0..hidden_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| rng.random::<f64>() * 2.0 * scale1 - scale1)
                    .collect()
            })
            .collect();
        let b1 = vec![0.0; hidden_dim];
        let w2 = (0..hidden_dim)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| rng.random::<f64>() * 2.0 * scale2 - scale2)
                    .collect()
            })
            .collect();
        let b2 = vec![0.0; hidden_dim];
        let scale3 = (2.0f64 / hidden_dim as f64).sqrt();
        let w3 = (0..output_dim)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| rng.random::<f64>() * 2.0 * scale3 - scale3)
                    .collect()
            })
            .collect();
        let b3 = vec![0.0; output_dim];
        Self {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            input_dim,
            output_dim,
            hidden_dim,
        }
    }

    fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    fn softmax(v: &[f64]) -> Vec<f64> {
        let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = v.iter().map(|&x| (x - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Forward pass: returns mixed strategy (softmax output).
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let h1: Vec<f64> = (0..self.hidden_dim)
            .map(|i| {
                let z: f64 = self.w1[i]
                    .iter()
                    .zip(input.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<f64>()
                    + self.b1[i];
                Self::relu(z)
            })
            .collect();
        let h2: Vec<f64> = (0..self.hidden_dim)
            .map(|i| {
                let z: f64 = self.w2[i]
                    .iter()
                    .zip(h1.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<f64>()
                    + self.b2[i];
                Self::relu(z)
            })
            .collect();
        let logits: Vec<f64> = (0..self.output_dim)
            .map(|i| {
                self.w3[i]
                    .iter()
                    .zip(h2.iter())
                    .map(|(&w, &x)| w * x)
                    .sum::<f64>()
                    + self.b3[i]
            })
            .collect();
        Self::softmax(&logits)
    }

    /// Compute Nash residual loss for 2-player game.
    fn nash_residual_loss(&self, payoff_flat: &[f64], m: usize, n: usize) -> f64 {
        let strat = self.forward(payoff_flat);
        let x = &strat[..m];
        let y = &strat[m..];
        let mut a = vec![vec![0f64; n]; m];
        let mut b = vec![vec![0f64; n]; m];
        let half = m * n;
        for i in 0..m {
            for j in 0..n {
                a[i][j] = payoff_flat.get(i * n + j).copied().unwrap_or(0.0);
                b[i][j] = payoff_flat.get(half + i * n + j).copied().unwrap_or(0.0);
            }
        }
        let util_x: Vec<f64> = (0..m)
            .map(|i| {
                a[i].iter()
                    .zip(y.iter())
                    .map(|(&aij, &yj)| aij * yj)
                    .sum::<f64>()
            })
            .collect();
        let expected_x: f64 = x.iter().zip(util_x.iter()).map(|(&xi, &ui)| xi * ui).sum();
        let max_x = util_x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let loss_x = (max_x - expected_x).powi(2);

        let util_y: Vec<f64> = (0..n)
            .map(|j| (0..m).map(|i| b[i][j] * x[i]).sum::<f64>())
            .collect();
        let expected_y: f64 = y.iter().zip(util_y.iter()).map(|(&yj, &uj)| yj * uj).sum();
        let max_y = util_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let loss_y = (max_y - expected_y).powi(2);
        loss_x + loss_y
    }
}

/// Neural Nash equilibrium finder.
pub struct CgtNeuralNash {
    pub net: NashNet,
    pub m: usize,
    pub n: usize,
}

impl CgtNeuralNash {
    pub fn new(m: usize, n: usize) -> Self {
        let input_dim = 2 * m * n;
        let output_dim = m + n;
        let net = NashNet::new(input_dim, output_dim, 64);
        Self { net, m, n }
    }

    /// Train via self-play with Nash residual loss (gradient-free coordinate perturbation).
    pub fn train(&mut self, n_games: usize, lr: f64) {
        let mut rng = StdRng::seed_from_u64(0xbabe);
        let m = self.m;
        let n = self.n;
        let input_dim = 2 * m * n;
        let _lr = lr;

        for _ in 0..n_games {
            let payoff_flat: Vec<f64> = (0..input_dim)
                .map(|_| rng.random::<f64>() * 2.0 - 1.0)
                .collect();
            let _loss = self.net.nash_residual_loss(&payoff_flat, m, n);

            let hidden = self.net.hidden_dim;
            let output = self.net.output_dim;
            let input = self.net.input_dim;
            for i in 0..output {
                for j in 0..hidden {
                    let delta = rng.random::<f64>() * 0.01 - 0.005;
                    self.net.w3[i][j] += delta;
                    let new_loss = self.net.nash_residual_loss(&payoff_flat, m, n);
                    if new_loss >= _loss {
                        self.net.w3[i][j] -= delta;
                    }
                }
            }
            let _ = input;
        }
    }

    /// Predict Nash strategy for a given payoff matrix (flat, A then B).
    /// Each returned slice is independently normalised to sum to 1.
    pub fn predict_nash(&self, payoff_flat: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let strat = self.net.forward(payoff_flat);
        let mut x = strat[..self.m].to_vec();
        let mut y = strat[self.m..].to_vec();
        let sx: f64 = x.iter().sum();
        if sx > 1e-12 {
            x.iter_mut().for_each(|v| *v /= sx);
        } else {
            x = vec![1.0 / self.m as f64; self.m];
        }
        let sy: f64 = y.iter().sum();
        if sy > 1e-12 {
            y.iter_mut().for_each(|v| *v /= sy);
        } else {
            y = vec![1.0 / self.n as f64; self.n];
        }
        (x, y)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §14 CgtMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Game-theoretic metrics.
pub struct CgtMetrics;

impl CgtMetrics {
    /// Exploitability of a strategy profile in a 2-player zero-sum game.
    pub fn exploitability(strategies: &[Vec<f64>], game: &CgtGame) -> f64 {
        if game.n_players != 2 || strategies.len() != 2 {
            return f64::NAN;
        }
        let m = game.actions_per_player[0];
        let n = game.actions_per_player[1];
        let x = &strategies[0];
        let y = &strategies[1];

        let br0_val = (0..m)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let u = game.payoff(0, &[i, j]).unwrap_or(0.0);
                        u * y.get(j).copied().unwrap_or(0.0)
                    })
                    .sum::<f64>()
            })
            .fold(f64::NEG_INFINITY, f64::max);

        let ev0: f64 = (0..m)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| {
                game.payoff(0, &[i, j]).unwrap_or(0.0)
                    * x.get(i).copied().unwrap_or(0.0)
                    * y.get(j).copied().unwrap_or(0.0)
            })
            .sum();

        let br1_val = (0..n)
            .map(|j| {
                (0..m)
                    .map(|i| {
                        let u = game.payoff(1, &[i, j]).unwrap_or(0.0);
                        u * x.get(i).copied().unwrap_or(0.0)
                    })
                    .sum::<f64>()
            })
            .fold(f64::NEG_INFINITY, f64::max);

        let ev1: f64 = (0..m)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| {
                game.payoff(1, &[i, j]).unwrap_or(0.0)
                    * x.get(i).copied().unwrap_or(0.0)
                    * y.get(j).copied().unwrap_or(0.0)
            })
            .sum();

        (br0_val - ev0) + (br1_val - ev1)
    }

    /// Social welfare at an equilibrium (sum of all players' expected payoffs).
    pub fn social_welfare(strategies: &[Vec<f64>], game: &CgtGame) -> f64 {
        let profiles = game.all_profiles();
        profiles
            .iter()
            .map(|prof| {
                let prob: f64 = prof
                    .iter()
                    .enumerate()
                    .map(|(p, &a)| {
                        strategies
                            .get(p)
                            .and_then(|s| s.get(a))
                            .copied()
                            .unwrap_or(0.0)
                    })
                    .product();
                let total_payoff: f64 = (0..game.n_players)
                    .map(|p| game.payoff(p, prof).unwrap_or(0.0))
                    .sum();
                prob * total_payoff
            })
            .sum()
    }

    /// Price of anarchy: ratio of worst Nash welfare to social optimum.
    pub fn price_of_anarchy(game: &CgtGame) -> f64 {
        if game.n_players != 2 {
            return f64::NAN;
        }
        let ne = match NashEquilibriumSolver::solve_2player(game) {
            Ok(v) => v,
            Err(_) => return f64::NAN,
        };
        if ne.is_empty() {
            return f64::NAN;
        }

        let worst_ne_sw = ne
            .iter()
            .map(|(x, y)| {
                let strategies = vec![x.clone(), y.clone()];
                Self::social_welfare(&strategies, game)
            })
            .fold(f64::INFINITY, f64::min);

        let profiles = game.all_profiles();
        let opt_sw = profiles
            .iter()
            .map(|prof| {
                (0..game.n_players)
                    .map(|p| game.payoff(p, prof).unwrap_or(0.0))
                    .sum::<f64>()
            })
            .fold(f64::NEG_INFINITY, f64::max);

        if opt_sw.abs() < 1e-9 {
            return 1.0;
        }
        opt_sw / worst_ne_sw.max(1e-9)
    }

    /// Gini coefficient of a payoff vector (measures inequality).
    pub fn gini_coefficient(payoffs: &[f64]) -> f64 {
        let n = payoffs.len();
        if n == 0 {
            return 0.0;
        }
        let mut sorted = payoffs.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let total: f64 = sorted.iter().sum();
        if total.abs() < 1e-12 {
            return 0.0;
        }
        let gini_num: f64 = sorted
            .iter()
            .enumerate()
            .map(|(i, &y)| (2 * (i as i64 + 1) - n as i64 - 1) as f64 * y)
            .sum();
        gini_num / (n as f64 * total)
    }

    /// Shapley efficiency: sum of Shapley values equals grand coalition value.
    pub fn shapley_efficiency(game: &CgtCooperativeGame) -> bool {
        let shapley = ShapleyValueCalculator::compute_exact(game);
        let shapley_sum: f64 = shapley.iter().sum();
        let gv = game.grand_coalition_value();
        (shapley_sum - gv).abs() < 1e-6
    }
}
