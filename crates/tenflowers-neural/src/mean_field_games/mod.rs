//! Mean Field Games (MFG) — comprehensive pure-Rust framework.
//!
//! - McKean–Vlasov stochastic dynamics with mean-field coupling
//! - Fixed-point Nash equilibrium solver (HJB + Fokker–Planck)
//! - Deep MFG solver via neural policy networks
//! - Linear–Quadratic MFG with closed-form Riccati solutions
//! - Multi-population and common-noise extensions
//! - Metrics: Nash error, exploitation gap, social cost, PoA
//! - Advanced: MFC, Graphon MFG, Risk-Sensitive MFG, Stationary MFG

pub mod extensions;
pub use extensions::*;

pub mod advanced;
pub use advanced::*;

#[cfg(test)]
mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ── helpers ──────────────────────────────────────────────────────────────────

#[inline]
pub(crate) fn sample_normal_f64(rng: &mut impl Rng) -> f64 {
    let u1: f64 = (rng.random::<f64>()).max(1e-15);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[inline]
pub(crate) fn clamp_f64(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[inline]
pub(crate) fn mfg_err(op: &str, reason: &str) -> TensorError {
    TensorError::InvalidArgument {
        operation: op.to_string(),
        reason: reason.to_string(),
        context: None,
    }
}

// ── §1 Core types ─────────────────────────────────────────────────────────────

/// Scalar state in a 1-D mean field game.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MfgState(pub f64);

/// Scalar action (control input).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MfgAction(pub f64);

/// Empirical distribution over a discrete state grid; `weights[i]` = P(bin i).
#[derive(Debug, Clone)]
pub struct MfgDistribution {
    pub weights: Vec<f64>,
    pub x_min: f64,
    pub x_max: f64,
}

impl MfgDistribution {
    /// Construct a uniform distribution over `n` bins in `[x_min, x_max]`.
    pub fn uniform(n: usize, x_min: f64, x_max: f64) -> Result<Self> {
        if n == 0 {
            return Err(mfg_err("MfgDistribution::uniform", "n must be > 0"));
        }
        Ok(MfgDistribution {
            weights: vec![1.0 / n as f64; n],
            x_min,
            x_max,
        })
    }

    /// Construct from raw weights (automatically normalised).
    pub fn from_weights(weights: Vec<f64>, x_min: f64, x_max: f64) -> Result<Self> {
        let s: f64 = weights.iter().sum();
        if s <= 0.0 {
            return Err(mfg_err(
                "MfgDistribution::from_weights",
                "weights must have positive sum",
            ));
        }
        Ok(MfgDistribution {
            weights: weights.iter().map(|w| w / s).collect(),
            x_min,
            x_max,
        })
    }

    /// Number of discrete bins.
    pub fn n_bins(&self) -> usize {
        self.weights.len()
    }

    /// Bin spacing.
    pub fn dx(&self) -> f64 {
        if self.weights.len() <= 1 {
            return self.x_max - self.x_min;
        }
        (self.x_max - self.x_min) / (self.weights.len() - 1) as f64
    }

    /// First moment (mean).
    pub fn mean(&self) -> f64 {
        let dx = self.dx();
        self.weights
            .iter()
            .enumerate()
            .map(|(i, w)| (self.x_min + i as f64 * dx) * w)
            .sum()
    }

    /// Second central moment (variance).
    pub fn variance(&self) -> f64 {
        let m = self.mean();
        let dx = self.dx();
        self.weights
            .iter()
            .enumerate()
            .map(|(i, w)| (self.x_min + i as f64 * dx - m).powi(2) * w)
            .sum()
    }

    /// Third standardised moment (skewness).
    pub fn skewness(&self) -> f64 {
        let m = self.mean();
        let var = self.variance();
        if var < 1e-15 {
            return 0.0;
        }
        let std = var.sqrt();
        let dx = self.dx();
        self.weights
            .iter()
            .enumerate()
            .map(|(i, w)| ((self.x_min + i as f64 * dx - m) / std).powi(3) * w)
            .sum()
    }

    /// Returns [mean, variance, skewness].
    pub fn features(&self) -> [f64; 3] {
        [self.mean(), self.variance(), self.skewness()]
    }

    /// Wasserstein-1 distance (same grid required).
    pub fn wasserstein1(&self, other: &MfgDistribution) -> Result<f64> {
        if self.weights.len() != other.weights.len() {
            return Err(mfg_err(
                "wasserstein1",
                "distributions must have same bin count",
            ));
        }
        let dx = self.dx();
        let (mut cs, mut co, mut d) = (0.0f64, 0.0f64, 0.0f64);
        for (ws, wo) in self.weights.iter().zip(other.weights.iter()) {
            cs += ws;
            co += wo;
            d += (cs - co).abs();
        }
        Ok(d * dx)
    }

    /// Grid coordinate for bin `i`.
    pub fn grid_point(&self, i: usize) -> f64 {
        self.x_min + i as f64 * self.dx()
    }

    /// Add 1.0 to the nearest bin for state `x`.
    pub fn deposit_state(&mut self, x: f64) {
        let n = self.weights.len();
        if n == 0 {
            return;
        }
        let idx = ((x - self.x_min) / self.dx()).round() as isize;
        self.weights[idx.max(0).min(n as isize - 1) as usize] += 1.0;
    }

    /// Normalise weights in-place so they sum to 1.
    pub fn normalise(&mut self) {
        let s: f64 = self.weights.iter().sum();
        if s > 1e-20 {
            for w in self.weights.iter_mut() {
                *w /= s;
            }
        }
    }
}

// ── §2 McKean–Vlasov dynamics ─────────────────────────────────────────────────

/// Config for McKean–Vlasov dynamics dx = (-κx + a + σ_mf(mean-x))dt + σ dW.
#[derive(Debug, Clone)]
pub struct MckeanVlasovConfig {
    /// Mean-reversion strength.
    pub kappa: f64,
    /// Mean-field coupling coefficient.
    pub sigma_mf: f64,
    /// Individual noise amplitude.
    pub sigma: f64,
    /// Time step.
    pub dt: f64,
}

impl Default for MckeanVlasovConfig {
    fn default() -> Self {
        MckeanVlasovConfig {
            kappa: 0.5,
            sigma_mf: 0.3,
            sigma: 0.1,
            dt: 0.01,
        }
    }
}

/// McKean–Vlasov stochastic dynamics.
#[derive(Debug, Clone)]
pub struct McKeanVlasovDynamics {
    /// Configuration parameters.
    pub config: MckeanVlasovConfig,
}

impl McKeanVlasovDynamics {
    /// Create with the given config.
    pub fn new(config: MckeanVlasovConfig) -> Self {
        McKeanVlasovDynamics { config }
    }

    /// Drift f(x, a, μ) = -κx + a + σ_mf·(μ_mean - x).
    pub fn drift(&self, x: MfgState, a: MfgAction, mu: &MfgDistribution) -> f64 {
        -self.config.kappa * x.0 + a.0 + self.config.sigma_mf * (mu.mean() - x.0)
    }

    /// Euler–Maruyama step with Gaussian noise.
    pub fn step(
        &self,
        x: MfgState,
        a: MfgAction,
        mu: &MfgDistribution,
        rng: &mut impl Rng,
    ) -> MfgState {
        let f = self.drift(x, a, mu);
        MfgState(
            x.0 + f * self.config.dt
                + self.config.sigma * self.config.dt.sqrt() * sample_normal_f64(rng),
        )
    }

    /// Deterministic (noise-free) Euler step.
    pub fn step_det(&self, x: MfgState, a: MfgAction, mu: &MfgDistribution) -> MfgState {
        MfgState(x.0 + self.drift(x, a, mu) * self.config.dt)
    }
}

// ── §3 Cost functions ──────────────────────────────────────────────────────────

/// Running cost L(x,a,μ) and terminal cost g(x,μ).
pub trait MfgCostFunction: Send + Sync {
    /// Instantaneous running cost.
    fn running_cost(&self, x: MfgState, a: MfgAction, mu: &MfgDistribution) -> f64;
    /// Terminal cost at horizon T.
    fn terminal_cost(&self, x: MfgState, mu: &MfgDistribution) -> f64;
}

/// Quadratic cost: L = q·x² + r·a² + λ·(x-mean)², g = q_T·x².
#[derive(Debug, Clone)]
pub struct MfgQuadraticCost {
    /// State penalty weight.
    pub q: f64,
    /// Control penalty weight.
    pub r: f64,
    /// Mean-field coupling penalty.
    pub lambda: f64,
    /// Terminal state penalty.
    pub q_terminal: f64,
}

impl Default for MfgQuadraticCost {
    fn default() -> Self {
        MfgQuadraticCost {
            q: 1.0,
            r: 0.5,
            lambda: 0.3,
            q_terminal: 2.0,
        }
    }
}

impl MfgCostFunction for MfgQuadraticCost {
    fn running_cost(&self, x: MfgState, a: MfgAction, mu: &MfgDistribution) -> f64 {
        let mean = mu.mean();
        self.q * x.0.powi(2) + self.r * a.0.powi(2) + self.lambda * (x.0 - mean).powi(2)
    }
    fn terminal_cost(&self, x: MfgState, _mu: &MfgDistribution) -> f64 {
        self.q_terminal * x.0.powi(2)
    }
}

// ── §4 MeanFieldNashSolver ────────────────────────────────────────────────────

/// Configuration for the mean-field Nash fixed-point solver.
#[derive(Debug, Clone)]
pub struct MfgNashConfig {
    /// Maximum fixed-point iterations.
    pub max_iters: usize,
    /// Wasserstein convergence tolerance.
    pub tolerance: f64,
    /// Distribution update damping factor ∈ (0,1].
    pub damping: f64,
    /// Number of state-space bins.
    pub n_bins: usize,
    /// Grid lower bound.
    pub x_min: f64,
    /// Grid upper bound.
    pub x_max: f64,
    /// Time steps per episode.
    pub n_steps: usize,
    /// Horizon length T.
    pub horizon: f64,
    /// Number of simulated agents.
    pub n_agents: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for MfgNashConfig {
    fn default() -> Self {
        MfgNashConfig {
            max_iters: 50,
            tolerance: 1e-4,
            damping: 0.5,
            n_bins: 32,
            x_min: -3.0,
            x_max: 3.0,
            n_steps: 20,
            horizon: 1.0,
            n_agents: 200,
            seed: 42,
        }
    }
}

/// Result of the Nash fixed-point solver.
#[derive(Debug, Clone)]
pub struct MfgNashResult {
    /// Mean-field distributions at each time step.
    pub distributions: Vec<MfgDistribution>,
    /// Value function grid at each time step.
    pub value_fn: Vec<Vec<f64>>,
    /// Wasserstein residuals per iteration.
    pub residuals: Vec<f64>,
    /// Whether the solver met the tolerance.
    pub converged: bool,
}

/// Fixed-point Nash equilibrium solver: iterates HJB → simulate → update μ.
pub struct MeanFieldNashSolver {
    /// Stochastic dynamics.
    pub dynamics: McKeanVlasovDynamics,
    /// Cost functional.
    pub cost: Box<dyn MfgCostFunction>,
    /// Solver config.
    pub config: MfgNashConfig,
}

impl MeanFieldNashSolver {
    /// Construct a new solver.
    pub fn new(
        dynamics: McKeanVlasovDynamics,
        cost: Box<dyn MfgCostFunction>,
        config: MfgNashConfig,
    ) -> Self {
        MeanFieldNashSolver {
            dynamics,
            cost,
            config,
        }
    }

    /// Run the fixed-point iteration to Nash equilibrium.
    pub fn solve(&self) -> Result<MfgNashResult> {
        let cfg = &self.config;
        let dt = cfg.horizon / cfg.n_steps as f64;
        let n = cfg.n_bins;
        let dx = (cfg.x_max - cfg.x_min) / (n - 1).max(1) as f64;

        let mut mu_seq: Vec<MfgDistribution> = (0..=cfg.n_steps)
            .map(|_| MfgDistribution::uniform(n, cfg.x_min, cfg.x_max))
            .collect::<Result<Vec<_>>>()?;
        let mut residuals = Vec::new();
        let mut value_fn = vec![vec![0.0f64; n]; cfg.n_steps + 1];

        for iter in 0..cfg.max_iters {
            let mu_terminal = &mu_seq[cfg.n_steps];
            for i in 0..n {
                let xi = cfg.x_min + i as f64 * dx;
                value_fn[cfg.n_steps][i] = self.cost.terminal_cost(MfgState(xi), mu_terminal);
            }
            for t in (0..cfg.n_steps).rev() {
                let mu_t = &mu_seq[t];
                for i in 0..n {
                    let xi = cfg.x_min + i as f64 * dx;
                    let v_x = self.value_gradient(t, i, &value_fn, dx);
                    let a_star = self.optimal_action_fd(xi, v_x, mu_t);
                    let cost_val = self
                        .cost
                        .running_cost(MfgState(xi), MfgAction(a_star), mu_t);
                    let drift = -self.dynamics.config.kappa * xi
                        + a_star
                        + self.dynamics.config.sigma_mf * (mu_t.mean() - xi);
                    let v_xx = self.value_laplacian(t + 1, i, &value_fn, dx);
                    let sigma = self.dynamics.config.sigma;
                    value_fn[t][i] = value_fn[t + 1][i]
                        - dt * (cost_val + v_x * drift + 0.5 * sigma * sigma * v_xx);
                }
            }
            let mut rng = StdRng::seed_from_u64(cfg.seed + iter as u64);
            let new_mu_seq = self.simulate_population(&value_fn, &mu_seq, &mut rng)?;
            let mut max_w1 = 0.0f64;
            for t in 0..=cfg.n_steps {
                let w1 = mu_seq[t].wasserstein1(&new_mu_seq[t])?;
                if w1 > max_w1 {
                    max_w1 = w1;
                }
                let alpha = cfg.damping;
                for i in 0..n {
                    mu_seq[t].weights[i] =
                        (1.0 - alpha) * mu_seq[t].weights[i] + alpha * new_mu_seq[t].weights[i];
                }
                mu_seq[t].normalise();
            }
            residuals.push(max_w1);
            if max_w1 < cfg.tolerance {
                return Ok(MfgNashResult {
                    distributions: mu_seq,
                    value_fn,
                    residuals,
                    converged: true,
                });
            }
        }
        Ok(MfgNashResult {
            distributions: mu_seq,
            value_fn,
            residuals,
            converged: false,
        })
    }

    fn value_gradient(&self, t: usize, i: usize, v: &[Vec<f64>], dx: f64) -> f64 {
        let n = v[t].len();
        if i == 0 {
            (v[t][1] - v[t][0]) / dx
        } else if i == n - 1 {
            (v[t][n - 1] - v[t][n - 2]) / dx
        } else {
            (v[t][i + 1] - v[t][i - 1]) / (2.0 * dx)
        }
    }

    fn value_laplacian(&self, t: usize, i: usize, v: &[Vec<f64>], dx: f64) -> f64 {
        let n = v[t].len();
        if i == 0 || i == n - 1 {
            return 0.0;
        }
        (v[t][i + 1] - 2.0 * v[t][i] + v[t][i - 1]) / (dx * dx)
    }

    fn optimal_action_fd(&self, x: f64, v_x: f64, mu: &MfgDistribution) -> f64 {
        let (a_min, a_max, candidates) = (-2.0f64, 2.0f64, 21usize);
        let (mut best_a, mut best_val) = (0.0f64, f64::INFINITY);
        for k in 0..candidates {
            let a = a_min + (a_max - a_min) * k as f64 / (candidates - 1) as f64;
            let cost = self.cost.running_cost(MfgState(x), MfgAction(a), mu);
            let drift = -self.dynamics.config.kappa * x
                + a
                + self.dynamics.config.sigma_mf * (mu.mean() - x);
            let h = cost + v_x * drift;
            if h < best_val {
                best_val = h;
                best_a = a;
            }
        }
        best_a
    }

    fn simulate_population(
        &self,
        value_fn: &[Vec<f64>],
        mu_seq: &[MfgDistribution],
        rng: &mut impl Rng,
    ) -> Result<Vec<MfgDistribution>> {
        let cfg = &self.config;
        let n = cfg.n_bins;
        let dx = (cfg.x_max - cfg.x_min) / (n - 1).max(1) as f64;
        let mut new_mu_seq: Vec<MfgDistribution> = (0..=cfg.n_steps)
            .map(|_| MfgDistribution {
                weights: vec![0.0; n],
                x_min: cfg.x_min,
                x_max: cfg.x_max,
            })
            .collect();

        let mut states: Vec<f64> = (0..cfg.n_agents)
            .map(|_| {
                let u: f64 = rng.random::<f64>();
                let mut cdf = 0.0f64;
                let mut chosen = cfg.x_min;
                for i in 0..n {
                    cdf += mu_seq[0].weights[i];
                    if u <= cdf {
                        chosen = mu_seq[0].grid_point(i);
                        break;
                    }
                }
                chosen
            })
            .collect();

        for &x in &states {
            new_mu_seq[0].deposit_state(x);
        }
        new_mu_seq[0].normalise();

        for t in 0..cfg.n_steps {
            let mu_t = &mu_seq[t];
            for x_ref in states.iter_mut() {
                let i = (((*x_ref - cfg.x_min) / dx).round() as isize)
                    .max(0)
                    .min(n as isize - 1) as usize;
                let v_x = self.value_gradient(t, i, value_fn, dx);
                let a_star = self.optimal_action_fd(*x_ref, v_x, mu_t);
                let next = self
                    .dynamics
                    .step(MfgState(*x_ref), MfgAction(a_star), mu_t, rng);
                *x_ref = clamp_f64(next.0, cfg.x_min - 1.0, cfg.x_max + 1.0);
            }
            for &x in &states {
                new_mu_seq[t + 1].deposit_state(x);
            }
            new_mu_seq[t + 1].normalise();
        }
        Ok(new_mu_seq)
    }
}

// ── §5 Fokker–Planck solver ───────────────────────────────────────────────────

/// Config for Fokker–Planck solver ∂_t μ + ∂_x(μv) = D·∂_xx μ.
#[derive(Debug, Clone)]
pub struct MfgFokkerPlanckConfig {
    /// Number of state bins.
    pub n_bins: usize,
    /// Grid lower bound.
    pub x_min: f64,
    /// Grid upper bound.
    pub x_max: f64,
    /// Number of time steps.
    pub n_steps: usize,
    /// Time horizon.
    pub horizon: f64,
    /// Diffusion coefficient D.
    pub diffusion: f64,
}

impl Default for MfgFokkerPlanckConfig {
    fn default() -> Self {
        MfgFokkerPlanckConfig {
            n_bins: 32,
            x_min: -3.0,
            x_max: 3.0,
            n_steps: 40,
            horizon: 1.0,
            diffusion: 0.01,
        }
    }
}

/// First-order upwind Fokker–Planck solver.
pub struct MfgFokkerPlanckSolver {
    /// Solver configuration.
    pub config: MfgFokkerPlanckConfig,
}

impl MfgFokkerPlanckSolver {
    /// Construct from config.
    pub fn new(config: MfgFokkerPlanckConfig) -> Self {
        MfgFokkerPlanckSolver { config }
    }

    /// Advance the density `mu0` under the given velocity field for `n_steps`.
    pub fn solve(
        &self,
        mu0: &MfgDistribution,
        velocity: &[Vec<f64>],
    ) -> Result<Vec<MfgDistribution>> {
        let cfg = &self.config;
        let n = cfg.n_bins;
        if mu0.n_bins() != n {
            return Err(mfg_err(
                "FokkerPlanckSolver::solve",
                "mu0 bin count mismatch",
            ));
        }
        if velocity.len() != cfg.n_steps {
            return Err(mfg_err(
                "FokkerPlanckSolver::solve",
                "velocity length mismatch",
            ));
        }
        let dx = (cfg.x_max - cfg.x_min) / (n - 1).max(1) as f64;
        let dt = cfg.horizon / cfg.n_steps as f64;
        let mut distributions = vec![mu0.clone()];
        let mut mu = mu0.weights.clone();

        for t in 0..cfg.n_steps {
            let v = &velocity[t];
            if v.len() != n {
                return Err(mfg_err(
                    "FokkerPlanckSolver::solve",
                    "velocity[t] bin count mismatch",
                ));
            }
            let mut mu_new = mu.clone();
            for i in 0..n {
                let adv = if v[i] >= 0.0 {
                    if i == 0 {
                        v[i] * mu[i] / dx
                    } else {
                        v[i] * (mu[i] - mu[i - 1]) / dx
                    }
                } else if i == n - 1 {
                    v[i] * mu[i] / dx
                } else {
                    v[i] * (mu[i + 1] - mu[i]) / dx
                };
                let diff = if i == 0 || i == n - 1 {
                    0.0
                } else {
                    cfg.diffusion * (mu[i + 1] - 2.0 * mu[i] + mu[i - 1]) / (dx * dx)
                };
                mu_new[i] = (mu[i] - dt * adv + dt * diff).max(0.0);
            }
            let s: f64 = mu_new.iter().sum::<f64>().max(1e-20);
            for m in mu_new.iter_mut() {
                *m /= s;
            }
            mu = mu_new.clone();
            distributions.push(MfgDistribution {
                weights: mu_new,
                x_min: cfg.x_min,
                x_max: cfg.x_max,
            });
        }
        Ok(distributions)
    }

    /// Build a linear (mean-reverting) velocity field v(x) = -κ·x.
    pub fn linear_velocity_field(&self, kappa: f64) -> Vec<Vec<f64>> {
        let cfg = &self.config;
        let n = cfg.n_bins;
        let dx = (cfg.x_max - cfg.x_min) / (n - 1).max(1) as f64;
        (0..cfg.n_steps)
            .map(|_| {
                (0..n)
                    .map(|i| -kappa * (cfg.x_min + i as f64 * dx))
                    .collect()
            })
            .collect()
    }
}

// ── §6 HJB solver ─────────────────────────────────────────────────────────────

/// Config for Hamilton-Jacobi-Bellman backward solver.
#[derive(Debug, Clone)]
pub struct MfgHjbConfig {
    /// Number of state bins.
    pub n_bins: usize,
    /// Grid lower bound.
    pub x_min: f64,
    /// Grid upper bound.
    pub x_max: f64,
    /// Number of time steps.
    pub n_steps: usize,
    /// Time horizon.
    pub horizon: f64,
    /// Diffusion coefficient D.
    pub diffusion: f64,
}

impl Default for MfgHjbConfig {
    fn default() -> Self {
        MfgHjbConfig {
            n_bins: 32,
            x_min: -3.0,
            x_max: 3.0,
            n_steps: 40,
            horizon: 1.0,
            diffusion: 0.01,
        }
    }
}

/// Solves ∂_t V + H(x, ∂_x V, μ) + D·∂_xx V = 0 backward with terminal V(T,x)=g(x,μ_T).
pub struct MfgHjbSolver {
    /// Solver configuration.
    pub config: MfgHjbConfig,
    /// Cost functional.
    pub cost: Box<dyn MfgCostFunction>,
}

impl MfgHjbSolver {
    /// Construct a new HJB solver.
    pub fn new(config: MfgHjbConfig, cost: Box<dyn MfgCostFunction>) -> Self {
        MfgHjbSolver { config, cost }
    }

    /// Solve backward in time given a sequence of mean-field distributions.
    pub fn solve(&self, mu_seq: &[MfgDistribution]) -> Result<Vec<Vec<f64>>> {
        let cfg = &self.config;
        let n = cfg.n_bins;
        let dt = cfg.horizon / cfg.n_steps as f64;
        let dx = (cfg.x_max - cfg.x_min) / (n - 1).max(1) as f64;
        if mu_seq.len() != cfg.n_steps + 1 {
            return Err(mfg_err("HjbSolver::solve", "mu_seq length mismatch"));
        }
        let mut v: Vec<Vec<f64>> = vec![vec![0.0; n]; cfg.n_steps + 1];
        for i in 0..n {
            let xi = cfg.x_min + i as f64 * dx;
            v[cfg.n_steps][i] = self.cost.terminal_cost(MfgState(xi), &mu_seq[cfg.n_steps]);
        }
        for t in (0..cfg.n_steps).rev() {
            let mu_t = &mu_seq[t];
            for i in 0..n {
                let xi = cfg.x_min + i as f64 * dx;
                let v_x = if i == 0 {
                    (v[t + 1][1] - v[t + 1][0]) / dx
                } else if i == n - 1 {
                    (v[t + 1][n - 1] - v[t + 1][n - 2]) / dx
                } else {
                    (v[t + 1][i + 1] - v[t + 1][i - 1]) / (2.0 * dx)
                };
                let v_xx = if i == 0 || i == n - 1 {
                    0.0
                } else {
                    (v[t + 1][i + 1] - 2.0 * v[t + 1][i] + v[t + 1][i - 1]) / (dx * dx)
                };
                let h = self.optimal_hamiltonian(xi, v_x, mu_t);
                v[t][i] = v[t + 1][i] - dt * (h + cfg.diffusion * v_xx);
            }
        }
        Ok(v)
    }

    fn optimal_hamiltonian(&self, x: f64, v_x: f64, mu: &MfgDistribution) -> f64 {
        let (a_min, a_max, candidates) = (-2.0f64, 2.0f64, 21usize);
        let mut best = f64::INFINITY;
        for k in 0..candidates {
            let a = a_min + (a_max - a_min) * k as f64 / (candidates - 1) as f64;
            let h = self.cost.running_cost(MfgState(x), MfgAction(a), mu) + v_x * a;
            if h < best {
                best = h;
            }
        }
        best
    }
}

// ── §7 MfgPolicyNetwork ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct MfgDenseLayer {
    pub(crate) weights: Vec<f64>,
    pub(crate) biases: Vec<f64>,
    pub(crate) in_dim: usize,
    pub(crate) out_dim: usize,
}

impl MfgDenseLayer {
    pub(crate) fn xavier_init(in_dim: usize, out_dim: usize, rng: &mut impl Rng) -> Self {
        let limit = (6.0 / (in_dim + out_dim) as f64).sqrt();
        MfgDenseLayer {
            weights: (0..in_dim * out_dim)
                .map(|_| rng.random::<f64>() * 2.0 * limit - limit)
                .collect(),
            biases: vec![0.0; out_dim],
            in_dim,
            out_dim,
        }
    }
    pub(crate) fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut out = self.biases.clone();
        for j in 0..self.out_dim {
            for i in 0..self.in_dim {
                out[j] += self.weights[j * self.in_dim + i] * x[i];
            }
        }
        out
    }
    pub(crate) fn n_params(&self) -> usize {
        self.weights.len() + self.biases.len()
    }
}

/// MLP policy π(x, μ): input = [x, mean, variance, skewness], output = action ∈ [-bound, +bound].
#[derive(Debug, Clone)]
pub struct MfgPolicyNetwork {
    layers: Vec<MfgDenseLayer>,
    /// Maximum absolute action magnitude.
    pub action_bound: f64,
}

impl MfgPolicyNetwork {
    /// Construct a new policy network with given hidden layer sizes.
    pub fn new(hidden: &[usize], seed: u64) -> Result<Self> {
        if hidden.is_empty() {
            return Err(mfg_err(
                "MfgPolicyNetwork::new",
                "need at least 1 hidden layer",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut dims: Vec<usize> = vec![4];
        dims.extend_from_slice(hidden);
        dims.push(1);
        Ok(MfgPolicyNetwork {
            layers: dims
                .windows(2)
                .map(|w| MfgDenseLayer::xavier_init(w[0], w[1], &mut rng))
                .collect(),
            action_bound: 2.0,
        })
    }

    /// Forward pass: map (x, μ) → action.
    pub fn forward(&self, x: f64, mu: &MfgDistribution) -> f64 {
        let [mean, var, skew] = mu.features();
        let mut input = vec![x, mean, var, skew];
        let n_layers = self.layers.len();
        for (k, layer) in self.layers.iter().enumerate() {
            let out = layer.forward(&input);
            input = if k < n_layers - 1 {
                out.iter().map(|v| v.tanh()).collect()
            } else {
                out.iter().map(|v| v.tanh() * self.action_bound).collect()
            };
        }
        input[0]
    }

    /// Flatten all parameters into a vector.
    pub fn get_params(&self) -> Vec<f64> {
        let mut p = Vec::new();
        for l in &self.layers {
            p.extend_from_slice(&l.weights);
            p.extend_from_slice(&l.biases);
        }
        p
    }

    /// Load flattened parameters.
    pub fn set_params(&mut self, params: &[f64]) -> Result<()> {
        let total: usize = self.layers.iter().map(|l| l.n_params()).sum();
        if params.len() != total {
            return Err(mfg_err(
                "MfgPolicyNetwork::set_params",
                &format!("expected {total} params, got {}", params.len()),
            ));
        }
        let mut offset = 0;
        for l in self.layers.iter_mut() {
            let nw = l.weights.len();
            let nb = l.biases.len();
            l.weights.copy_from_slice(&params[offset..offset + nw]);
            offset += nw;
            l.biases.copy_from_slice(&params[offset..offset + nb]);
            offset += nb;
        }
        Ok(())
    }

    /// Total number of trainable parameters.
    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.n_params()).sum()
    }
}

// ── §8 DeepMfgSolver ──────────────────────────────────────────────────────────

/// Configuration for the deep MFG neural solver.
#[derive(Debug, Clone)]
pub struct MfgDeepConfig {
    /// Hidden layer sizes.
    pub hidden: Vec<usize>,
    /// Adam learning rate.
    pub lr: f64,
    /// Number of outer fixed-point iterations.
    pub outer_iters: usize,
    /// Number of inner gradient steps per outer iteration.
    pub inner_steps: usize,
    /// Number of agents to simulate.
    pub n_agents: usize,
    /// Time steps per episode.
    pub n_steps: usize,
    /// Horizon.
    pub horizon: f64,
    /// State-space bins.
    pub n_bins: usize,
    /// Grid lower bound.
    pub x_min: f64,
    /// Grid upper bound.
    pub x_max: f64,
    /// Finite-difference epsilon for gradient estimation.
    pub fd_eps: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Default for MfgDeepConfig {
    fn default() -> Self {
        MfgDeepConfig {
            hidden: vec![32, 32],
            lr: 1e-3,
            outer_iters: 10,
            inner_steps: 20,
            n_agents: 100,
            n_steps: 10,
            horizon: 1.0,
            n_bins: 32,
            x_min: -3.0,
            x_max: 3.0,
            fd_eps: 1e-4,
            seed: 42,
        }
    }
}

/// Result from DeepMfgSolver.
#[derive(Debug, Clone)]
pub struct MfgDeepResult {
    /// Trained policy network.
    pub policy: MfgPolicyNetwork,
    /// Mean-field distribution sequence.
    pub distributions: Vec<MfgDistribution>,
    /// Per-outer-iteration average loss.
    pub losses: Vec<f64>,
}

/// Neural network MFG solver: train MfgPolicyNetwork via FD policy gradient + fixed-point iteration.
pub struct DeepMfgSolver {
    /// Solver configuration.
    pub config: MfgDeepConfig,
    /// Stochastic dynamics.
    pub dynamics: McKeanVlasovDynamics,
    /// Cost functional.
    pub cost: Box<dyn MfgCostFunction>,
}

impl DeepMfgSolver {
    /// Construct a new deep MFG solver.
    pub fn new(
        config: MfgDeepConfig,
        dynamics: McKeanVlasovDynamics,
        cost: Box<dyn MfgCostFunction>,
    ) -> Self {
        DeepMfgSolver {
            config,
            dynamics,
            cost,
        }
    }

    /// Train the policy network via alternating optimisation.
    pub fn train(&self) -> Result<MfgDeepResult> {
        let cfg = &self.config;
        let mut policy = MfgPolicyNetwork::new(&cfg.hidden, cfg.seed)?;
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let mut mu_seq: Vec<MfgDistribution> = (0..=cfg.n_steps)
            .map(|_| MfgDistribution::uniform(cfg.n_bins, cfg.x_min, cfg.x_max))
            .collect::<Result<Vec<_>>>()?;
        let mut losses = Vec::new();
        let n_params = policy.n_params();
        let (mut m, mut v_adam) = (vec![0.0f64; n_params], vec![0.0f64; n_params]);
        let (beta1, beta2, eps_adam) = (0.9f64, 0.999f64, 1e-8f64);
        let mut step = 0u64;

        for _outer in 0..cfg.outer_iters {
            let mut total_loss = 0.0f64;
            for _inner in 0..cfg.inner_steps {
                step += 1;
                let params = policy.get_params();
                let loss0 = self.rollout_loss(&policy, &mu_seq, &mut rng)?;
                let mut grad = vec![0.0f64; n_params];
                for k in 0..n_params {
                    let mut p_plus = params.clone();
                    p_plus[k] += cfg.fd_eps;
                    let mut policy_plus = policy.clone();
                    policy_plus.set_params(&p_plus)?;
                    grad[k] =
                        (self.rollout_loss(&policy_plus, &mu_seq, &mut rng)? - loss0) / cfg.fd_eps;
                }
                let mut new_params = params.clone();
                for k in 0..n_params {
                    m[k] = beta1 * m[k] + (1.0 - beta1) * grad[k];
                    v_adam[k] = beta2 * v_adam[k] + (1.0 - beta2) * grad[k].powi(2);
                    let m_hat = m[k] / (1.0 - beta1.powi(step as i32));
                    let v_hat = v_adam[k] / (1.0 - beta2.powi(step as i32));
                    new_params[k] -= cfg.lr * m_hat / (v_hat.sqrt() + eps_adam);
                }
                policy.set_params(&new_params)?;
                total_loss += loss0;
            }
            losses.push(total_loss / cfg.inner_steps as f64);
            let new_mu_seq = self.simulate_with_policy(&policy, &mu_seq, &mut rng)?;
            for t in 0..=cfg.n_steps {
                for i in 0..cfg.n_bins {
                    mu_seq[t].weights[i] =
                        0.5 * mu_seq[t].weights[i] + 0.5 * new_mu_seq[t].weights[i];
                }
                mu_seq[t].normalise();
            }
        }
        Ok(MfgDeepResult {
            policy,
            distributions: mu_seq,
            losses,
        })
    }

    fn rollout_loss(
        &self,
        policy: &MfgPolicyNetwork,
        mu_seq: &[MfgDistribution],
        rng: &mut impl Rng,
    ) -> Result<f64> {
        let cfg = &self.config;
        let mut x = MfgState(sample_normal_f64(rng) * 0.5);
        let mut total_cost = 0.0f64;
        let dt = cfg.horizon / cfg.n_steps as f64;
        let n_steps = cfg.n_steps.min(mu_seq.len().saturating_sub(1));
        for t in 0..n_steps {
            let mu_t = &mu_seq[t];
            let a = MfgAction(policy.forward(x.0, mu_t));
            total_cost += self.cost.running_cost(x, a, mu_t) * dt;
            x = self.dynamics.step(x, a, mu_t, rng);
        }
        let mu_terminal = mu_seq
            .last()
            .ok_or_else(|| mfg_err("rollout", "mu_seq is empty"))?;
        total_cost += self.cost.terminal_cost(x, mu_terminal);
        Ok(total_cost)
    }

    fn simulate_with_policy(
        &self,
        policy: &MfgPolicyNetwork,
        mu_seq: &[MfgDistribution],
        rng: &mut impl Rng,
    ) -> Result<Vec<MfgDistribution>> {
        let cfg = &self.config;
        let n = cfg.n_bins;
        let mut new_mu_seq: Vec<MfgDistribution> = (0..=cfg.n_steps)
            .map(|_| MfgDistribution {
                weights: vec![0.0; n],
                x_min: cfg.x_min,
                x_max: cfg.x_max,
            })
            .collect();
        let mut states: Vec<f64> = (0..cfg.n_agents)
            .map(|_| sample_normal_f64(rng) * 0.5)
            .collect();
        for &x in &states {
            new_mu_seq[0].deposit_state(x);
        }
        new_mu_seq[0].normalise();
        let n_steps = cfg.n_steps.min(mu_seq.len().saturating_sub(1));
        for t in 0..n_steps {
            let mu_t = &mu_seq[t];
            for x_ref in states.iter_mut() {
                let a = MfgAction(policy.forward(*x_ref, mu_t));
                let next = self.dynamics.step(MfgState(*x_ref), a, mu_t, rng);
                *x_ref = clamp_f64(next.0, cfg.x_min - 1.0, cfg.x_max + 1.0);
            }
            for &x in &states {
                new_mu_seq[t + 1].deposit_state(x);
            }
            new_mu_seq[t + 1].normalise();
        }
        Ok(new_mu_seq)
    }
}

// ── §9 MfgPopulationSimulator ─────────────────────────────────────────────────

/// Simulates a population of agents under an arbitrary policy closure.
pub struct MfgPopulationSimulator {
    /// Stochastic dynamics.
    pub dynamics: McKeanVlasovDynamics,
    /// Number of agents.
    pub n_agents: usize,
    /// Number of time steps.
    pub n_steps: usize,
    /// Number of histogram bins.
    pub n_bins: usize,
    /// Grid lower bound.
    pub x_min: f64,
    /// Grid upper bound.
    pub x_max: f64,
    /// RNG seed.
    pub seed: u64,
}

impl MfgPopulationSimulator {
    /// Construct a new population simulator.
    pub fn new(
        dynamics: McKeanVlasovDynamics,
        n_agents: usize,
        n_steps: usize,
        n_bins: usize,
        x_min: f64,
        x_max: f64,
        seed: u64,
    ) -> Self {
        MfgPopulationSimulator {
            dynamics,
            n_agents,
            n_steps,
            n_bins,
            x_min,
            x_max,
            seed,
        }
    }

    /// Simulate agents using the given policy closure, starting from `mu0`.
    pub fn simulate(
        &self,
        policy_fn: &dyn Fn(f64, &MfgDistribution) -> f64,
        mu0: &MfgDistribution,
    ) -> Result<Vec<MfgDistribution>> {
        let n = self.n_bins;
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut mu_seq: Vec<MfgDistribution> = Vec::with_capacity(self.n_steps + 1);

        let mut states: Vec<f64> = (0..self.n_agents)
            .map(|_| {
                let u: f64 = rng.random::<f64>();
                let mut cdf = 0.0f64;
                let mut chosen = self.x_min;
                for i in 0..mu0.n_bins() {
                    cdf += mu0.weights[i];
                    if u <= cdf {
                        chosen = mu0.grid_point(i);
                        break;
                    }
                }
                chosen
            })
            .collect();

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
            let mu_t = mu_seq
                .last()
                .ok_or_else(|| mfg_err("simulate", "mu_seq is empty"))?;
            for x_ref in states.iter_mut() {
                let a = policy_fn(*x_ref, mu_t);
                let next = self
                    .dynamics
                    .step(MfgState(*x_ref), MfgAction(a), mu_t, &mut rng);
                *x_ref = clamp_f64(next.0, self.x_min - 1.0, self.x_max + 1.0);
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
        Ok(mu_seq)
    }
}

// ── §10 LinearQuadraticMfg ────────────────────────────────────────────────────

/// Parameters for the LQG mean field game.
#[derive(Debug, Clone)]
pub struct LqMfgParams {
    /// Mean-reversion coefficient.
    pub kappa: f64,
    /// State cost weight.
    pub q: f64,
    /// Control cost weight.
    pub r: f64,
    /// Mean-field interaction weight.
    pub lambda: f64,
    /// Terminal state cost.
    pub q_terminal: f64,
    /// Noise variance σ².
    pub sigma_sq: f64,
    /// Horizon.
    pub horizon: f64,
    /// Number of time steps.
    pub n_steps: usize,
}

impl Default for LqMfgParams {
    fn default() -> Self {
        LqMfgParams {
            kappa: 0.5,
            q: 1.0,
            r: 0.5,
            lambda: 0.3,
            q_terminal: 2.0,
            sigma_sq: 0.01,
            horizon: 1.0,
            n_steps: 100,
        }
    }
}

/// Closed-form Nash equilibrium for the Linear-Quadratic MFG via Riccati equations.
///
/// Riccati solution: -dP/dt = q_eff - P²/r, P(T) = q_T.
/// Closed form: P(t) = α(P_T + α·tanh(α·τ)) / (α + P_T·tanh(α·τ)), τ = T-t, α = √(q_eff/r).
#[derive(Debug, Clone)]
pub struct LinearQuadraticMfg {
    /// LQ-MFG parameters.
    pub params: LqMfgParams,
}

impl LinearQuadraticMfg {
    /// Construct from params.
    pub fn new(params: LqMfgParams) -> Self {
        LinearQuadraticMfg { params }
    }

    /// Solve the Riccati ODE and return P(t) for t = 0..=n_steps.
    pub fn solve_riccati(&self) -> Vec<f64> {
        let p = &self.params;
        let q_eff = p.q + p.lambda;
        let alpha = (q_eff / p.r).sqrt();
        let pt = p.q_terminal;
        let n = p.n_steps;
        let dt = p.horizon / n as f64;
        (0..=n)
            .map(|k| {
                let tau = (n - k) as f64 * dt;
                let th = (alpha * tau).tanh();
                let denom = alpha + pt * th;
                if denom.abs() < 1e-15 {
                    pt
                } else {
                    alpha * (pt + alpha * th) / denom
                }
            })
            .collect()
    }

    /// Compute optimal control u*(t, x) = -P(t)/r · x.
    pub fn compute_optimal_control(&self, t_idx: usize, x: f64) -> Result<f64> {
        let riccati = self.solve_riccati();
        let pk = riccati.get(t_idx).ok_or_else(|| {
            mfg_err(
                "LQ-MFG::compute_optimal_control",
                &format!("t_idx={t_idx} out of range"),
            )
        })?;
        Ok(-pk / self.params.r * x)
    }

    /// Return (Riccati values, feedback gains K(t) = P(t)/r).
    pub fn compute_equilibrium(&self) -> (Vec<f64>, Vec<f64>) {
        let riccati = self.solve_riccati();
        let gains: Vec<f64> = riccati.iter().map(|&p| p / self.params.r).collect();
        (riccati, gains)
    }

    /// Evaluate the value function V(t, x) = P(t)·x².
    pub fn value_function(&self, t_idx: usize, x: f64) -> Result<f64> {
        let riccati = self.solve_riccati();
        let pk = riccati.get(t_idx).ok_or_else(|| {
            mfg_err(
                "LQ-MFG::value_function",
                &format!("t_idx={t_idx} out of range"),
            )
        })?;
        Ok(pk * x * x)
    }

    /// Social optimal cost uses q + 2λ instead of q + λ (cooperative coupling).
    pub fn social_optimal_cost(&self, x0: f64) -> f64 {
        let p = &self.params;
        let q_soc = p.q + 2.0 * p.lambda;
        let alpha = (q_soc / p.r).sqrt();
        let pt = p.q_terminal;
        let big_t = p.horizon;
        let th = (alpha * big_t).tanh();
        let denom = alpha + pt * th;
        let p0 = if denom.abs() < 1e-15 {
            pt
        } else {
            alpha * (pt + alpha * th) / denom
        };
        p0.max(0.0) * x0 * x0
    }
}
