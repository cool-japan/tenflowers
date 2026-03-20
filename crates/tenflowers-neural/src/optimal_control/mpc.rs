//! Model Predictive Control variants: CEM, MPPI, Random Shooting.

use super::dynamics::DynamicsModel;
use super::utils::normal_samples_f64;
use scirs2_core::random::{rngs::StdRng, SeedableRng};
use tenflowers_core::{Result, TensorError};

// MPC Common Structures
// ─────────────────────────────────────────────────────────────────────────────

/// A plan produced by an MPC controller.
#[derive(Clone, Debug)]
pub struct MpcPlan {
    /// Planned action sequence (horizon actions).
    pub actions: Vec<Vec<f64>>,
    /// Expected (estimated) total cost of the plan.
    pub expected_cost: f64,
    /// Number of iterations the planner executed.
    pub n_iter: usize,
}

/// Configuration shared across MPC variants.
#[derive(Clone, Debug)]
pub struct MpcConfig {
    /// Planning horizon.
    pub horizon: usize,
    /// Number of candidate trajectories to sample.
    pub n_samples: usize,
    /// Standard deviation of exploration noise.
    pub noise_std: f64,
    /// Number of top trajectories to keep (CEM elite set).
    pub top_k: usize,
    /// Number of CEM refinement iterations.
    pub n_iter: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for MpcConfig {
    fn default() -> Self {
        Self {
            horizon: 20,
            n_samples: 200,
            noise_std: 0.5,
            top_k: 20,
            n_iter: 5,
            seed: 42,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-Entropy Method MPC
// ─────────────────────────────────────────────────────────────────────────────

/// CEM-based MPC controller (Rubinstein, 1997).
///
/// Iteratively samples `n_samples` action sequences from a Gaussian distribution,
/// evaluates their total cost, selects the top-K elite trajectories, and refits
/// the distribution mean and variance.
pub struct CrossEntropyMpc;

impl CrossEntropyMpc {
    /// Create a new `CrossEntropyMpc` planner.
    pub fn new() -> Self {
        Self
    }

    /// Plan an action sequence from state `x0`.
    pub fn plan<F>(
        &self,
        dynamics: &dyn DynamicsModel,
        x0: &[f64],
        cost_fn: &F,
        config: &MpcConfig,
    ) -> Result<MpcPlan>
    where
        F: Fn(&[f64], &[f64]) -> f64,
    {
        let h = config.horizon;
        let m = dynamics.action_dim();
        let n = config.n_samples;
        let k = config.top_k.min(n);

        if k == 0 {
            return Err(TensorError::invalid_argument(
                "CrossEntropyMpc: top_k must be >= 1".to_string(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(config.seed);

        // Initialize distribution: zero mean, noise_std² variance
        let mut mean = vec![vec![0.0f64; m]; h];
        let mut std_dev = vec![vec![config.noise_std; m]; h];

        let mut best_actions = mean.clone();
        let mut best_cost = f64::INFINITY;
        let mut n_iter_done = 0;

        for iter in 0..config.n_iter {
            n_iter_done = iter + 1;

            // Sample n action sequences
            let mut trajs: Vec<(f64, Vec<Vec<f64>>)> = Vec::with_capacity(n);
            for _ in 0..n {
                let mut actions = Vec::with_capacity(h);
                for t in 0..h {
                    let noise = normal_samples_f64(m, &mut rng);
                    let action: Vec<f64> = mean[t]
                        .iter()
                        .zip(std_dev[t].iter())
                        .zip(noise.iter())
                        .map(|((mu, s), eps)| mu + s * eps)
                        .collect();
                    actions.push(action);
                }
                // Evaluate total cost
                let mut x = x0.to_vec();
                let mut cost = 0.0;
                for t in 0..h {
                    cost += cost_fn(&x, &actions[t]);
                    x = dynamics.step(&x, &actions[t]);
                }
                // Terminal cost not provided in CrossEntropyMpc for generality;
                // users include it in the running cost or add a terminal wrapper.
                trajs.push((cost, actions));
            }

            // Sort by cost ascending
            trajs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Update best
            if trajs[0].0 < best_cost {
                best_cost = trajs[0].0;
                best_actions = trajs[0].1.clone();
            }

            // Compute elite mean and std
            let elite = &trajs[..k];
            let mut new_mean = vec![vec![0.0f64; m]; h];
            for (_, actions) in elite {
                for t in 0..h {
                    for j in 0..m {
                        new_mean[t][j] += actions[t][j];
                    }
                }
            }
            for t in 0..h {
                for j in 0..m {
                    new_mean[t][j] /= k as f64;
                }
            }

            let mut new_std = vec![vec![0.0f64; m]; h];
            for (_, actions) in elite {
                for t in 0..h {
                    for j in 0..m {
                        let diff = actions[t][j] - new_mean[t][j];
                        new_std[t][j] += diff * diff;
                    }
                }
            }
            for t in 0..h {
                for j in 0..m {
                    new_std[t][j] = (new_std[t][j] / k as f64).sqrt().max(1e-8);
                }
            }

            mean = new_mean;
            std_dev = new_std;
        }

        Ok(MpcPlan {
            actions: best_actions,
            expected_cost: best_cost,
            n_iter: n_iter_done,
        })
    }

    /// Execute one step: plan from `x0` and return the first action.
    pub fn step<F>(
        &self,
        dynamics: &dyn DynamicsModel,
        x0: &[f64],
        cost_fn: &F,
        config: &MpcConfig,
    ) -> Result<Vec<f64>>
    where
        F: Fn(&[f64], &[f64]) -> f64,
    {
        let plan = self.plan(dynamics, x0, cost_fn, config)?;
        plan.actions
            .into_iter()
            .next()
            .ok_or_else(|| TensorError::invalid_argument("CrossEntropyMpc: empty plan".to_string()))
    }
}

impl Default for CrossEntropyMpc {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MPPI Controller
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the MPPI controller.
#[derive(Clone, Debug)]
pub struct MppiConfig {
    /// Planning horizon.
    pub horizon: usize,
    /// Number of perturbed trajectory samples.
    pub n_samples: usize,
    /// Temperature λ controlling the "peakedness" of the importance weight distribution.
    pub temperature: f64,
    /// Standard deviation of perturbation noise.
    pub noise_std: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for MppiConfig {
    fn default() -> Self {
        Self {
            horizon: 20,
            n_samples: 500,
            temperature: 1.0,
            noise_std: 0.5,
            seed: 0,
        }
    }
}

/// Model Predictive Path Integral (MPPI) controller (Williams et al., 2016).
///
/// Uses importance-weighted Monte Carlo to compute the optimal control update:
/// ```text
/// δu_t = Σ_k w_k * ε_{k,t} / Σ_k w_k,  w_k = exp(-ρ_k / λ)
/// ```
/// where `ρ_k` is the total cost of trajectory `k` under perturbation.
pub struct MppiController {
    /// MPPI configuration.
    pub config: MppiConfig,
    /// Warm-start: previous optimal control sequence.
    prev_controls: Vec<Vec<f64>>,
}

impl MppiController {
    /// Create a new `MppiController` from configuration.
    pub fn new(config: MppiConfig) -> Self {
        Self {
            config,
            prev_controls: Vec::new(),
        }
    }

    /// Plan from `x0`, optionally warm-starting from the previous optimal sequence.
    ///
    /// Returns an `MpcPlan` with the importance-weighted optimal action sequence.
    pub fn plan<F>(
        &mut self,
        dynamics: &dyn DynamicsModel,
        x0: &[f64],
        cost_fn: &F,
    ) -> Result<MpcPlan>
    where
        F: Fn(&[f64], &[f64]) -> f64,
    {
        let h = self.config.horizon;
        let m = dynamics.action_dim();
        let n = self.config.n_samples;
        let lambda = self.config.temperature;
        let sigma = self.config.noise_std;

        // Initialize or warm-start nominal control sequence
        if self.prev_controls.len() != h {
            self.prev_controls = vec![vec![0.0f64; m]; h];
        } else {
            // Shift left: drop first element, append zero at end
            let zero = vec![0.0f64; m];
            self.prev_controls.remove(0);
            self.prev_controls.push(zero);
        }

        let mut rng = StdRng::seed_from_u64(self.config.seed);

        // Sample perturbations and compute trajectory costs
        let total_noise_len = n * h * m;
        let all_noise = normal_samples_f64(total_noise_len, &mut rng);

        let mut costs = vec![0.0f64; n];
        let mut perturbations: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n);

        for k in 0..n {
            let mut eps_k: Vec<Vec<f64>> = Vec::with_capacity(h);
            for t in 0..h {
                let base_idx = (k * h + t) * m;
                let eps: Vec<f64> = (0..m).map(|j| sigma * all_noise[base_idx + j]).collect();
                eps_k.push(eps);
            }

            // Simulate trajectory under u_nom + eps
            let mut x = x0.to_vec();
            let mut traj_cost = 0.0;
            for t in 0..h {
                let u_t: Vec<f64> = self.prev_controls[t]
                    .iter()
                    .zip(eps_k[t].iter())
                    .map(|(un, e)| un + e)
                    .collect();
                traj_cost += cost_fn(&x, &u_t);
                x = dynamics.step(&x, &u_t);
            }
            costs[k] = traj_cost;
            perturbations.push(eps_k);
        }

        // Compute importance weights: w_k = exp(-(ρ_k - ρ_min) / λ)
        let rho_min = costs.iter().cloned().fold(f64::INFINITY, f64::min);
        let weights: Vec<f64> = costs
            .iter()
            .map(|&rho| (-(rho - rho_min) / lambda).exp())
            .collect();
        let w_sum: f64 = weights.iter().sum();

        // Compute optimal control update: δu_t = Σ_k w_k * ε_{k,t} / w_sum
        let mut opt_controls: Vec<Vec<f64>> = Vec::with_capacity(h);
        for t in 0..h {
            let mut delta_u = vec![0.0f64; m];
            for k in 0..n {
                for j in 0..m {
                    delta_u[j] += (weights[k] / w_sum) * perturbations[k][t][j];
                }
            }
            let u_t: Vec<f64> = self.prev_controls[t]
                .iter()
                .zip(delta_u.iter())
                .map(|(un, du)| un + du)
                .collect();
            opt_controls.push(u_t);
        }

        // Compute expected cost
        let expected_cost = costs
            .iter()
            .zip(weights.iter())
            .map(|(c, w)| c * w / w_sum)
            .sum();

        // Update warm-start
        self.prev_controls = opt_controls.clone();

        Ok(MpcPlan {
            actions: opt_controls,
            expected_cost,
            n_iter: 1,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Random Shooting MPC
// ─────────────────────────────────────────────────────────────────────────────

/// Random shooting MPC: sample random action sequences and return the best one.
pub struct RandomShootingMpc;

impl RandomShootingMpc {
    /// Create a new `RandomShootingMpc` planner.
    pub fn new() -> Self {
        Self
    }

    /// Sample `n_samples` random action sequences, simulate trajectories, return the best plan.
    pub fn plan<F>(
        &self,
        dynamics: &dyn DynamicsModel,
        x0: &[f64],
        cost_fn: &F,
        n_samples: usize,
        horizon: usize,
        noise_std: f64,
        seed: u64,
    ) -> Result<MpcPlan>
    where
        F: Fn(&[f64], &[f64]) -> f64,
    {
        let m = dynamics.action_dim();
        let mut rng = StdRng::seed_from_u64(seed);

        let mut best_cost = f64::INFINITY;
        let mut best_actions = vec![vec![0.0f64; m]; horizon];

        for _ in 0..n_samples {
            let noise = normal_samples_f64(horizon * m, &mut rng);
            let actions: Vec<Vec<f64>> = (0..horizon)
                .map(|t| (0..m).map(|j| noise_std * noise[t * m + j]).collect())
                .collect();

            let mut x = x0.to_vec();
            let mut cost = 0.0;
            for t in 0..horizon {
                cost += cost_fn(&x, &actions[t]);
                x = dynamics.step(&x, &actions[t]);
            }

            if cost < best_cost {
                best_cost = cost;
                best_actions = actions;
            }
        }

        Ok(MpcPlan {
            actions: best_actions,
            expected_cost: best_cost,
            n_iter: 1,
        })
    }
}

impl Default for RandomShootingMpc {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
