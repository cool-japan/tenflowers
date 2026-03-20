//! Extensions to the MFG framework: multi-population, common-noise, and metrics.

use super::{mfg_err, sample_normal_f64, clamp_f64, MfgDistribution, MfgState, MfgAction,
            McKeanVlasovDynamics, MfgCostFunction};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use tenflowers_core::Result;

// ── §11 MfgMultiPopulation ────────────────────────────────────────────────────

/// Configuration for one population in a multi-population MFG.
#[derive(Debug, Clone)]
pub struct MfgPopulationConfig {
    /// Human-readable population label.
    pub name: String,
    /// Number of agents in this population.
    pub n_agents: usize,
    /// Own-state quadratic penalty weight.
    pub q: f64,
    /// Control quadratic penalty weight.
    pub r: f64,
    /// Cross-population coupling weights (length = number of populations).
    pub cross_coupling: Vec<f64>,
    /// Mean-reversion coefficient.
    pub kappa: f64,
}

/// Multi-population MFG with K agent types and cross-population coupling.
pub struct MfgMultiPopulation {
    /// Per-population configuration.
    pub populations: Vec<MfgPopulationConfig>,
    /// Number of histogram bins.
    pub n_bins: usize,
    /// Grid lower bound.
    pub x_min: f64,
    /// Grid upper bound.
    pub x_max: f64,
    /// Time horizon.
    pub horizon: f64,
    /// Number of time steps.
    pub n_steps: usize,
    /// RNG seed.
    pub seed: u64,
    /// Individual noise amplitude.
    pub sigma: f64,
}

impl MfgMultiPopulation {
    /// Construct a new multi-population MFG.
    pub fn new(
        populations: Vec<MfgPopulationConfig>,
        n_bins: usize,
        x_min: f64,
        x_max: f64,
        horizon: f64,
        n_steps: usize,
        seed: u64,
        sigma: f64,
    ) -> Self {
        MfgMultiPopulation {
            populations,
            n_bins,
            x_min,
            x_max,
            horizon,
            n_steps,
            seed,
            sigma,
        }
    }

    /// Running cost for population `k`: L_k(x, a, μ) = q_k·(x-μ̄_k)² + r_k·a² + Σ_{j≠k} cc_{kj}·(x-μ̄_j)².
    pub fn running_cost_k(
        &self,
        k: usize,
        x: f64,
        a: f64,
        mu_all: &[MfgDistribution],
    ) -> Result<f64> {
        let pop = self.populations.get(k).ok_or_else(|| {
            mfg_err(
                "MultiPop::running_cost_k",
                &format!("population index {k} out of range"),
            )
        })?;
        let own_mean: f64 = mu_all
            .get(k)
            .ok_or_else(|| mfg_err("MultiPop::running_cost_k", &format!("mu_all[{k}] missing")))?
            .mean();
        let mut cost: f64 = pop.q * (x - own_mean).powi(2) + pop.r * a * a;
        for (j, &cc) in pop.cross_coupling.iter().enumerate() {
            if j == k {
                continue;
            }
            if let Some(mu_j) = mu_all.get(j) {
                let mj_mean: f64 = mu_j.mean();
                cost += cc * (x - mj_mean).powi(2);
            }
        }
        Ok(cost)
    }

    /// Run best-response fixed-point iterations.
    pub fn solve(&self, max_iters: usize) -> Result<Vec<Vec<MfgDistribution>>> {
        let k = self.populations.len();
        let dt = self.horizon / self.n_steps as f64;
        let mut all_mu: Vec<Vec<MfgDistribution>> = (0..k)
            .map(|_| {
                (0..=self.n_steps)
                    .map(|_| MfgDistribution::uniform(self.n_bins, self.x_min, self.x_max))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let mut rng = StdRng::seed_from_u64(self.seed);

        for _iter in 0..max_iters {
            let prev_mu = all_mu.clone();
            for pop_k in 0..k {
                let pop = &self.populations[pop_k];
                let mut states: Vec<f64> = (0..pop.n_agents)
                    .map(|_| sample_normal_f64(&mut rng) * 0.5)
                    .collect();
                for t in 0..=self.n_steps {
                    all_mu[pop_k][t].weights = vec![0.0; self.n_bins];
                }
                for &x in &states {
                    all_mu[pop_k][0].deposit_state(x);
                }
                all_mu[pop_k][0].normalise();
                for t in 0..self.n_steps {
                    let mu_all_t: Vec<MfgDistribution> =
                        (0..k).map(|j| prev_mu[j][t].clone()).collect();
                    for x_ref in states.iter_mut() {
                        let a_best = self.best_response_k(pop_k, *x_ref, &mu_all_t)?;
                        let drift = -pop.kappa * *x_ref + a_best;
                        *x_ref = clamp_f64(
                            *x_ref
                                + drift * dt
                                + self.sigma * dt.sqrt() * sample_normal_f64(&mut rng),
                            self.x_min - 1.0,
                            self.x_max + 1.0,
                        );
                    }
                    for &x in &states {
                        all_mu[pop_k][t + 1].deposit_state(x);
                    }
                    all_mu[pop_k][t + 1].normalise();
                }
            }
        }
        Ok(all_mu)
    }

    fn best_response_k(&self, k: usize, x: f64, mu_all: &[MfgDistribution]) -> Result<f64> {
        let (a_min, a_max, candidates) = (-2.0f64, 2.0f64, 21usize);
        let (mut best_a, mut best_cost) = (0.0f64, f64::INFINITY);
        for c in 0..candidates {
            let a = a_min + (a_max - a_min) * c as f64 / (candidates - 1) as f64;
            let cost = self.running_cost_k(k, x, a, mu_all)?;
            if cost < best_cost {
                best_cost = cost;
                best_a = a;
            }
        }
        Ok(best_a)
    }
}

// ── §12 ExtendedMfgGame ───────────────────────────────────────────────────────

/// MFG with common noise W^0: dx_i = f(x_i,a_i,μ)dt + σ dW^i + σ_0 dW^0.
pub struct ExtendedMfgGame {
    /// Individual agent dynamics.
    pub dynamics: McKeanVlasovDynamics,
    /// Common noise amplitude σ_0.
    pub sigma_common: f64,
    /// Cost functional.
    pub cost: Box<dyn MfgCostFunction>,
    /// Number of agents.
    pub n_agents: usize,
    /// Number of time steps.
    pub n_steps: usize,
    /// Histogram bins.
    pub n_bins: usize,
    /// Grid lower bound.
    pub x_min: f64,
    /// Grid upper bound.
    pub x_max: f64,
    /// Horizon.
    pub horizon: f64,
    /// RNG seed.
    pub seed: u64,
}

impl ExtendedMfgGame {
    /// Construct a new extended MFG with common noise.
    pub fn new(
        dynamics: McKeanVlasovDynamics,
        sigma_common: f64,
        cost: Box<dyn MfgCostFunction>,
        n_agents: usize,
        n_steps: usize,
        n_bins: usize,
        x_min: f64,
        x_max: f64,
        horizon: f64,
        seed: u64,
    ) -> Self {
        ExtendedMfgGame {
            dynamics,
            sigma_common,
            cost,
            n_agents,
            n_steps,
            n_bins,
            x_min,
            x_max,
            horizon,
            seed,
        }
    }

    /// Simulate agents under the given policy and the common noise path.
    /// Returns `(mu_seq, common_signal_path)`.
    pub fn simulate(
        &self,
        policy_fn: &dyn Fn(f64, &MfgDistribution) -> f64,
    ) -> Result<(Vec<MfgDistribution>, Vec<f64>)> {
        let n = self.n_bins;
        let dt = self.horizon / self.n_steps as f64;
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut states: Vec<f64> = (0..self.n_agents)
            .map(|_| sample_normal_f64(&mut rng) * 0.3)
            .collect();
        let mut mu_seq = Vec::with_capacity(self.n_steps + 1);
        let mut signal_path = vec![0.0f64];
        let mut common_signal = 0.0f64;
        let mut current_mu = MfgDistribution {
            weights: vec![0.0; n],
            x_min: self.x_min,
            x_max: self.x_max,
        };
        for &x in &states {
            current_mu.deposit_state(x);
        }
        current_mu.normalise();
        mu_seq.push(current_mu);

        for _t in 0..self.n_steps {
            let dw0 = sample_normal_f64(&mut rng) * dt.sqrt();
            common_signal += self.sigma_common * dw0;
            signal_path.push(common_signal);
            let mu_t = mu_seq
                .last()
                .ok_or_else(|| mfg_err("simulate", "mu_seq is empty"))?;
            for x_ref in states.iter_mut() {
                let a = policy_fn(*x_ref, mu_t);
                let drift = self.dynamics.drift(MfgState(*x_ref), MfgAction(a), mu_t);
                *x_ref = clamp_f64(
                    *x_ref
                        + drift * dt
                        + self.dynamics.config.sigma * sample_normal_f64(&mut rng) * dt.sqrt()
                        + self.sigma_common * dw0,
                    self.x_min - 2.0,
                    self.x_max + 2.0,
                );
            }
            let mut new_mu = MfgDistribution {
                weights: vec![0.0; n],
                x_min: self.x_min,
                x_max: self.x_max,
            };
            for &x in &states {
                new_mu.deposit_state(x);
            }
            new_mu.normalise();
            mu_seq.push(new_mu);
        }
        Ok((mu_seq, signal_path))
    }

    /// Conditional distribution given common signal (Gaussian likelihood approximation).
    pub fn conditional_distribution(&self, mu: &MfgDistribution, signal: f64) -> MfgDistribution {
        let n = mu.n_bins();
        let dx = mu.dx();
        let var = self.sigma_common.powi(2).max(1e-10);
        let mut cond_weights: Vec<f64> = (0..n)
            .map(|i| {
                let xi = mu.x_min + i as f64 * dx;
                mu.weights[i] * (-(signal - xi).powi(2) / (2.0 * var)).exp()
            })
            .collect();
        let s: f64 = cond_weights.iter().sum::<f64>().max(1e-20);
        for w in cond_weights.iter_mut() {
            *w /= s;
        }
        MfgDistribution {
            weights: cond_weights,
            x_min: mu.x_min,
            x_max: mu.x_max,
        }
    }
}

// ── §13 MfgMetrics ────────────────────────────────────────────────────────────

/// Evaluation metrics for mean field games.
#[derive(Debug, Clone)]
pub struct MfgMetrics {
    /// Max Wasserstein-1 error between equilibrium and simulated distributions.
    pub nash_error: f64,
    /// Average deviation incentive (exploitation gap).
    pub exploitation_gap: f64,
    /// Average individual cost at Nash equilibrium.
    pub nash_social_cost: f64,
    /// Social optimum cost (cooperative benchmark).
    pub social_optimum_cost: f64,
    /// Price of anarchy = Nash cost / social optimum cost.
    pub price_of_anarchy: f64,
}

impl MfgMetrics {
    /// Max Wasserstein-1 distance between fixed-point and simulated distributions.
    pub fn mean_field_nash_error(
        mu_fixed: &[MfgDistribution],
        mu_simulated: &[MfgDistribution],
    ) -> Result<f64> {
        if mu_fixed.len() != mu_simulated.len() {
            return Err(mfg_err(
                "MfgMetrics::nash_error",
                "mu_fixed and mu_simulated have different lengths",
            ));
        }
        let mut max_w1 = 0.0f64;
        for (mf, ms) in mu_fixed.iter().zip(mu_simulated.iter()) {
            let w1 = mf.wasserstein1(ms)?;
            if w1 > max_w1 {
                max_w1 = w1;
            }
        }
        Ok(max_w1)
    }

    /// Average |cost_eq - cost_dev| over n_rollouts trajectories.
    pub fn exploitation_gap(
        cost: &dyn MfgCostFunction,
        dynamics: &McKeanVlasovDynamics,
        mu_seq: &[MfgDistribution],
        policy_fn: &dyn Fn(f64, &MfgDistribution) -> f64,
        seed: u64,
        n_rollouts: usize,
        horizon: f64,
    ) -> Result<f64> {
        let n_steps = mu_seq.len().saturating_sub(1);
        if n_steps == 0 {
            return Err(mfg_err(
                "MfgMetrics::exploitation_gap",
                "mu_seq must have at least 2 entries",
            ));
        }
        let dt = horizon / n_steps as f64;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut gap_total = 0.0f64;
        for _ in 0..n_rollouts {
            let x0 = sample_normal_f64(&mut rng) * 0.3;
            let (mut x_eq, mut cost_eq) = (MfgState(x0), 0.0f64);
            for t in 0..n_steps {
                let mu_t = &mu_seq[t];
                let a = MfgAction(policy_fn(x_eq.0, mu_t));
                cost_eq += cost.running_cost(x_eq, a, mu_t) * dt;
                x_eq = dynamics.step(x_eq, a, mu_t, &mut rng);
            }
            let mu_terminal = mu_seq
                .last()
                .ok_or_else(|| mfg_err("rollout", "mu_seq is empty"))?;
            cost_eq += cost.terminal_cost(x_eq, mu_terminal);
            let (mut x_dev, mut cost_dev) = (MfgState(x0), 0.0f64);
            for t in 0..n_steps {
                let mu_t = &mu_seq[t];
                cost_dev += cost.running_cost(x_dev, MfgAction(0.0), mu_t) * dt;
                x_dev = dynamics.step(x_dev, MfgAction(0.0), mu_t, &mut rng);
            }
            cost_dev += cost.terminal_cost(x_dev, mu_terminal);
            gap_total += (cost_eq - cost_dev).abs();
        }
        Ok(gap_total / n_rollouts as f64)
    }

    /// Mean of individual costs.
    pub fn social_cost(costs: &[f64]) -> f64 {
        if costs.is_empty() {
            return 0.0;
        }
        costs.iter().sum::<f64>() / costs.len() as f64
    }

    /// Ratio of Nash cost to social optimum cost.
    pub fn price_of_anarchy(nash_cost: f64, social_optimum: f64) -> f64 {
        if social_optimum.abs() < 1e-12 {
            return 1.0;
        }
        nash_cost / social_optimum
    }

    /// Compute all metrics in one call.
    pub fn compute(
        mu_fixed: &[MfgDistribution],
        mu_simulated: &[MfgDistribution],
        cost: &dyn MfgCostFunction,
        dynamics: &McKeanVlasovDynamics,
        policy_fn: &dyn Fn(f64, &MfgDistribution) -> f64,
        individual_costs: &[f64],
        social_optimum: f64,
        horizon: f64,
        seed: u64,
    ) -> Result<MfgMetrics> {
        let nash_error = Self::mean_field_nash_error(mu_fixed, mu_simulated)?;
        let exploitation_gap =
            Self::exploitation_gap(cost, dynamics, mu_fixed, policy_fn, seed, 20, horizon)?;
        let nash_social_cost = Self::social_cost(individual_costs);
        let price_of_anarchy = Self::price_of_anarchy(nash_social_cost, social_optimum);
        Ok(MfgMetrics {
            nash_error,
            exploitation_gap,
            nash_social_cost,
            social_optimum_cost: social_optimum,
            price_of_anarchy,
        })
    }
}
