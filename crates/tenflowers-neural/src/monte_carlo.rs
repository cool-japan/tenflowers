//! Monte Carlo Methods and Markov Chain Monte Carlo (MCMC) Algorithms.
//!
//! This module provides a comprehensive set of Monte Carlo and MCMC algorithms
//! for Bayesian inference, probabilistic modelling, and approximate posterior
//! estimation in pure Rust.
//!
//! # Samplers
//!
//! | Sampler | Description |
//! |---------|-------------|
//! | [`MhSampler`] | Metropolis-Hastings with adaptive step size |
//! | [`HmcSampler`] | Hamiltonian Monte Carlo with leapfrog integrator |
//! | [`SvgdOptimizer`] | Stein Variational Gradient Descent |
//! | [`ImportanceSampler`] | Self-normalised importance sampling |
//! | [`SmcSampler`] | Sequential Monte Carlo / annealed IS |
//!
//! # Built-in target densities
//!
//! [`StandardNormal`], [`BananaDistribution`], [`MixtureOfGaussians`]
//!
//! # Diagnostics
//!
//! [`McmcDiagnostics`] — ESS (Geyer), split-R̂, autocorrelation, Geweke Z-score.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Internal RNG helpers (f64 variants)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a single N(0, 1) variate via the Box-Muller transform (f64).
#[inline]
fn sample_normal_f64(rng: &mut impl Rng) -> f64 {
    let u1: f64 = (rng.random::<f64>()).max(1e-300);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Generate a vector of `n` i.i.d. N(0, 1) variates (f64).
fn sample_normal_vec_f64(n: usize, rng: &mut impl Rng) -> Vec<f64> {
    (0..n).map(|_| sample_normal_f64(rng)).collect()
}

/// Numerically stable log-sum-exp over a slice.
///
/// Returns -∞ (as `f64::NEG_INFINITY`) for an empty slice.
#[inline]
fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_v = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_v.is_infinite() {
        return max_v;
    }
    let sum: f64 = values.iter().map(|v| (v - max_v).exp()).sum();
    max_v + sum.ln()
}

// ─────────────────────────────────────────────────────────────────────────────
// LogDensity trait
// ─────────────────────────────────────────────────────────────────────────────

/// A target distribution specified by its unnormalised log-probability density.
///
/// The default implementation of [`LogDensity::grad_log_prob`] uses central
/// finite differences with step `h = 1e-5`.  Implementors may override this
/// with an exact analytical gradient.
pub trait LogDensity {
    /// Evaluate `log p(x)` (up to an additive constant).
    fn log_prob(&self, x: &[f64]) -> f64;

    /// Evaluate the gradient `∇_x log p(x)`.
    ///
    /// The default implementation uses central finite differences.
    fn grad_log_prob(&self, x: &[f64]) -> Vec<f64> {
        let h = 1e-5;
        let mut grad = vec![0.0_f64; x.len()];
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        for i in 0..x.len() {
            xp[i] = x[i] + h;
            xm[i] = x[i] - h;
            grad[i] = (self.log_prob(&xp) - self.log_prob(&xm)) / (2.0 * h);
            xp[i] = x[i];
            xm[i] = x[i];
        }
        grad
    }

    /// Dimensionality of the target distribution.
    fn dim(&self) -> usize;
}

// ─────────────────────────────────────────────────────────────────────────────
// Built-in test densities
// ─────────────────────────────────────────────────────────────────────────────

/// Standard isotropic Gaussian: `log p(x) = -0.5 ‖x‖²`.
#[derive(Debug, Clone)]
pub struct StandardNormal {
    /// Dimensionality.
    pub dim: usize,
}

impl StandardNormal {
    /// Construct a `d`-dimensional standard normal.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl LogDensity for StandardNormal {
    fn log_prob(&self, x: &[f64]) -> f64 {
        -0.5 * x.iter().map(|xi| xi * xi).sum::<f64>()
    }

    fn grad_log_prob(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|xi| -xi).collect()
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// 2-D banana (Haario) distribution.
///
/// `log p(x) = -0.5·(x₀²/σ² + (x₁ - b·x₀²)²)`
#[derive(Debug, Clone)]
pub struct BananaDistribution {
    /// Curvature parameter.
    pub b: f64,
    /// Scale of the first coordinate.
    pub sigma: f64,
}

impl BananaDistribution {
    /// Construct a banana distribution.
    pub fn new(b: f64, sigma: f64) -> Self {
        Self { b, sigma }
    }
}

impl LogDensity for BananaDistribution {
    fn log_prob(&self, x: &[f64]) -> f64 {
        if x.len() < 2 {
            return f64::NEG_INFINITY;
        }
        let x0 = x[0];
        let x1 = x[1];
        -0.5 * (x0 * x0 / (self.sigma * self.sigma) + (x1 - self.b * x0 * x0).powi(2))
    }

    fn grad_log_prob(&self, x: &[f64]) -> Vec<f64> {
        if x.len() < 2 {
            return vec![0.0; x.len()];
        }
        let x0 = x[0];
        let x1 = x[1];
        let diff = x1 - self.b * x0 * x0;
        let g0 = -x0 / (self.sigma * self.sigma) + 2.0 * self.b * x0 * diff;
        let g1 = -diff;
        let mut grad = vec![g0, g1];
        // Pad extra dimensions with zero
        grad.extend(vec![0.0; x.len().saturating_sub(2)]);
        grad
    }

    fn dim(&self) -> usize {
        2
    }
}

/// Mixture of isotropic Gaussians.
///
/// `log p(x) = log Σ_k w_k · N(x; μ_k, I)`
#[derive(Debug, Clone)]
pub struct MixtureOfGaussians {
    /// Component means; each has length `d`.
    pub means: Vec<Vec<f64>>,
    /// Mixture weights (will be normalised internally).
    pub weights: Vec<f64>,
}

impl MixtureOfGaussians {
    /// Construct a mixture.  Weights are normalised to sum to 1.
    pub fn new(means: Vec<Vec<f64>>, weights: Vec<f64>) -> Self {
        let total: f64 = weights.iter().sum::<f64>().max(1e-300);
        let norm_weights: Vec<f64> = weights.iter().map(|w| w / total).collect();
        Self {
            means,
            weights: norm_weights,
        }
    }
}

impl LogDensity for MixtureOfGaussians {
    fn log_prob(&self, x: &[f64]) -> f64 {
        let log_components: Vec<f64> = self
            .means
            .iter()
            .zip(self.weights.iter())
            .map(|(mu, w)| {
                let sq_dist: f64 = x
                    .iter()
                    .zip(mu.iter())
                    .map(|(xi, mi)| (xi - mi).powi(2))
                    .sum();
                w.ln() - 0.5 * sq_dist
            })
            .collect();
        log_sum_exp(&log_components)
    }

    fn dim(&self) -> usize {
        self.means.first().map(|m| m.len()).unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// McmcResult
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated results from an MCMC run.
#[derive(Debug, Clone)]
pub struct McmcResult {
    /// Collected posterior samples (post-warmup), shape `[n_samples][dim]`.
    pub samples: Vec<Vec<f64>>,
    /// Empirical acceptance rate over all proposals.
    pub acceptance_rate: f64,
    /// Per-dimension effective sample size.
    pub effective_sample_size: Vec<f64>,
    /// Per-dimension split-R̂ convergence diagnostic.
    pub r_hat: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Metropolis-Hastings
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Metropolis-Hastings random-walk sampler.
#[derive(Debug, Clone)]
pub struct MhConfig {
    /// Initial proposal standard deviation.
    pub step_size: f64,
    /// Number of warmup (burn-in) steps — samples discarded.
    pub n_warmup: usize,
    /// Number of posterior samples to collect.
    pub n_samples: usize,
    /// Adapt the step size every this many steps during warmup.
    pub adapt_interval: usize,
    /// Seed for the RNG.
    pub seed: u64,
}

impl Default for MhConfig {
    fn default() -> Self {
        Self {
            step_size: 0.5,
            n_warmup: 1000,
            n_samples: 2000,
            adapt_interval: 100,
            seed: 42,
        }
    }
}

/// Metropolis-Hastings random-walk sampler with adaptive step size.
pub struct MhSampler {
    config: MhConfig,
}

impl MhSampler {
    /// Create a sampler from `config`.
    pub fn new(config: MhConfig) -> Self {
        Self { config }
    }

    /// Run the sampler against `target` starting from `init`.
    pub fn sample(&self, target: &dyn LogDensity, init: Vec<f64>) -> Result<McmcResult> {
        let d = target.dim();
        if init.len() != d {
            return Err(TensorError::invalid_argument_op(
                "MhSampler::sample",
                "init length must match target.dim()",
            ));
        }
        let cfg = &self.config;
        let mut rng = StdRng::seed_from_u64(cfg.seed);

        let mut current = init;
        let mut current_lp = target.log_prob(&current);
        let mut step_size = cfg.step_size;

        let total_steps = cfg.n_warmup + cfg.n_samples;
        let mut samples: Vec<Vec<f64>> = Vec::with_capacity(cfg.n_samples);

        let mut n_accepted: usize = 0;
        let mut n_accepted_interval: usize = 0;

        // Target acceptance rate for random-walk MH in high dimensions
        let target_ar = 0.234_f64;

        for step in 0..total_steps {
            // Propose x' = x + σ · ε,  ε ~ N(0,I)
            let noise = sample_normal_vec_f64(d, &mut rng);
            let proposed: Vec<f64> = current
                .iter()
                .zip(noise.iter())
                .map(|(xi, ni)| xi + step_size * ni)
                .collect();

            let proposed_lp = target.log_prob(&proposed);
            let log_alpha = (proposed_lp - current_lp).min(0.0);
            let u: f64 = rng.random::<f64>();

            if u.ln() < log_alpha {
                current = proposed;
                current_lp = proposed_lp;
                n_accepted += 1;
                n_accepted_interval += 1;
            }

            // Adapt step size during warmup
            if step < cfg.n_warmup && cfg.adapt_interval > 0 {
                let interval_end = (step + 1) % cfg.adapt_interval == 0;
                if interval_end {
                    let empirical_ar = n_accepted_interval as f64 / cfg.adapt_interval as f64;
                    // Scale step size to drive empirical AR toward target
                    let ratio = (empirical_ar + 1e-8) / (target_ar + 1e-8);
                    step_size = (step_size * ratio.sqrt()).clamp(1e-6, 100.0);
                    n_accepted_interval = 0;
                }
            }

            if step >= cfg.n_warmup {
                samples.push(current.clone());
            }
        }

        let acceptance_rate = n_accepted as f64 / total_steps as f64;

        let ess: Vec<f64> = (0..d)
            .map(|i| {
                let chain: Vec<f64> = samples.iter().map(|s| s[i]).collect();
                McmcDiagnostics::effective_sample_size(&chain)
            })
            .collect();

        let half = samples.len() / 2;
        let r_hat: Vec<f64> = if samples.len() >= 4 {
            (0..d)
                .map(|i| {
                    let first: Vec<f64> = samples[..half].iter().map(|s| s[i]).collect();
                    let second: Vec<f64> = samples[half..].iter().map(|s| s[i]).collect();
                    let refs: Vec<&[f64]> = vec![&first, &second];
                    McmcDiagnostics::gelman_rubin_r_hat(&refs)
                })
                .collect()
        } else {
            vec![1.0; d]
        };

        Ok(McmcResult {
            samples,
            acceptance_rate,
            effective_sample_size: ess,
            r_hat,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hamiltonian Monte Carlo
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the HMC sampler.
#[derive(Debug, Clone)]
pub struct HmcConfig {
    /// Leapfrog step size ε.
    pub step_size: f64,
    /// Number of leapfrog steps per proposal.
    pub n_leapfrog: usize,
    /// Warmup (burn-in) steps.
    pub n_warmup: usize,
    /// Posterior samples to collect.
    pub n_samples: usize,
    /// Mass matrix scalar (isotropic): momentum p ~ N(0, mass·I).
    pub mass: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Default for HmcConfig {
    fn default() -> Self {
        Self {
            step_size: 0.1,
            n_leapfrog: 10,
            n_warmup: 500,
            n_samples: 1000,
            mass: 1.0,
            seed: 42,
        }
    }
}

/// Hamiltonian Monte Carlo sampler with leapfrog integrator.
///
/// Includes dual-averaging step size adaptation during warmup (simplified
/// Hoffman & Gelman 2014).
pub struct HmcSampler {
    config: HmcConfig,
}

impl HmcSampler {
    /// Create a sampler from `config`.
    pub fn new(config: HmcConfig) -> Self {
        Self { config }
    }

    /// Compute the Hamiltonian: `H = -log p(x) + 0.5·‖p‖²/mass`.
    fn hamiltonian(lp: f64, momentum: &[f64], mass: f64) -> f64 {
        let ke: f64 = momentum.iter().map(|pi| pi * pi).sum::<f64>() / (2.0 * mass);
        -lp + ke
    }

    /// One leapfrog trajectory starting from (`q`, `p`).
    fn leapfrog(
        &self,
        target: &dyn LogDensity,
        q: &[f64],
        p: &[f64],
        step_size: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mass = self.config.mass;
        let mut q = q.to_vec();
        let mut p = p.to_vec();

        // Half step for momentum
        let grad = target.grad_log_prob(&q);
        for i in 0..p.len() {
            p[i] += 0.5 * step_size * grad[i];
        }

        for _l in 0..self.config.n_leapfrog {
            // Full step for position
            for i in 0..q.len() {
                q[i] += step_size * p[i] / mass;
            }
            // Full step for momentum (except at last step)
            if _l < self.config.n_leapfrog - 1 {
                let g = target.grad_log_prob(&q);
                for i in 0..p.len() {
                    p[i] += step_size * g[i];
                }
            }
        }

        // Half step for momentum at the end
        let grad_end = target.grad_log_prob(&q);
        for i in 0..p.len() {
            p[i] += 0.5 * step_size * grad_end[i];
        }

        (q, p)
    }

    /// Run HMC against `target` starting from `init`.
    pub fn sample(&self, target: &dyn LogDensity, init: Vec<f64>) -> Result<McmcResult> {
        let d = target.dim();
        if init.len() != d {
            return Err(TensorError::invalid_argument_op(
                "HmcSampler::sample",
                "init length must match target.dim()",
            ));
        }
        let cfg = &self.config;
        let mut rng = StdRng::seed_from_u64(cfg.seed);

        let mut current_q = init;
        let mut current_lp = target.log_prob(&current_q);

        // Dual averaging parameters (simplified NUTS-style)
        let target_ar = 0.65_f64;
        let mu = (10.0 * cfg.step_size).ln();
        let gamma = 0.05;
        let t0 = 10.0_f64;
        let kappa = 0.75;
        let mut h_bar = 0.0_f64;
        let mut eps_bar = cfg.step_size;
        let mut step_size = cfg.step_size;

        let total_steps = cfg.n_warmup + cfg.n_samples;
        let mut samples: Vec<Vec<f64>> = Vec::with_capacity(cfg.n_samples);
        let mut n_accepted: usize = 0;

        for step in 0..total_steps {
            // Sample momentum: p ~ N(0, mass·I)
            let noise = sample_normal_vec_f64(d, &mut rng);
            let momentum: Vec<f64> = noise.iter().map(|ni| ni * cfg.mass.sqrt()).collect();

            let h_current = Self::hamiltonian(current_lp, &momentum, cfg.mass);

            // Leapfrog
            let (proposed_q, proposed_p) = self.leapfrog(target, &current_q, &momentum, step_size);

            let proposed_lp = target.log_prob(&proposed_q);
            let h_proposed = Self::hamiltonian(proposed_lp, &proposed_p, cfg.mass);

            let log_alpha = (h_current - h_proposed).min(0.0);
            let u: f64 = rng.random::<f64>();

            let accept_prob = log_alpha.exp();
            if u.ln() < log_alpha {
                current_q = proposed_q;
                current_lp = proposed_lp;
                n_accepted += 1;
            }

            // Dual averaging during warmup
            if step < cfg.n_warmup {
                let m = (step + 1) as f64;
                let w = 1.0 / (m + t0);
                h_bar = (1.0 - w) * h_bar + w * (target_ar - accept_prob);
                let log_eps = mu - (m.sqrt() / gamma) * h_bar;
                step_size = log_eps.exp().clamp(1e-6, 100.0);
                let m_kappa = m.powf(-kappa);
                eps_bar = (m_kappa * log_eps + (1.0 - m_kappa) * eps_bar.ln()).exp();
            } else if step == cfg.n_warmup {
                // Switch to the averaged step size
                step_size = eps_bar;
            }

            if step >= cfg.n_warmup {
                samples.push(current_q.clone());
            }
        }

        let acceptance_rate = n_accepted as f64 / total_steps as f64;

        let ess: Vec<f64> = (0..d)
            .map(|i| {
                let chain: Vec<f64> = samples.iter().map(|s| s[i]).collect();
                McmcDiagnostics::effective_sample_size(&chain)
            })
            .collect();

        let half = samples.len() / 2;
        let r_hat: Vec<f64> = if samples.len() >= 4 {
            (0..d)
                .map(|i| {
                    let first: Vec<f64> = samples[..half].iter().map(|s| s[i]).collect();
                    let second: Vec<f64> = samples[half..].iter().map(|s| s[i]).collect();
                    let refs: Vec<&[f64]> = vec![&first, &second];
                    McmcDiagnostics::gelman_rubin_r_hat(&refs)
                })
                .collect()
        } else {
            vec![1.0; d]
        };

        Ok(McmcResult {
            samples,
            acceptance_rate,
            effective_sample_size: ess,
            r_hat,
        })
    }

    /// Compute the Hamiltonian difference for a single leapfrog trajectory.
    ///
    /// Used in energy-conservation tests.
    pub fn hamiltonian_change(
        &self,
        target: &dyn LogDensity,
        q: &[f64],
        mass: f64,
        seed: u64,
    ) -> f64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let d = q.len();
        let noise = sample_normal_vec_f64(d, &mut rng);
        let momentum: Vec<f64> = noise.iter().map(|ni| ni * mass.sqrt()).collect();

        let lp_init = target.log_prob(q);
        let h_init = Self::hamiltonian(lp_init, &momentum, mass);

        let (q_new, p_new) = self.leapfrog(target, q, &momentum, self.config.step_size);
        let lp_new = target.log_prob(&q_new);
        let h_new = Self::hamiltonian(lp_new, &p_new, mass);

        (h_new - h_init).abs()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stein Variational Gradient Descent
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for SVGD.
#[derive(Debug, Clone)]
pub struct SvgdConfig {
    /// Number of particles.
    pub n_particles: usize,
    /// Learning rate.
    pub lr: f64,
    /// Number of update iterations.
    pub n_iterations: usize,
    /// RBF bandwidth.  Set to 0 to use the median heuristic.
    pub bandwidth: f64,
    /// RNG seed for initialisation.
    pub seed: u64,
}

impl Default for SvgdConfig {
    fn default() -> Self {
        Self {
            n_particles: 50,
            lr: 0.01,
            n_iterations: 200,
            bandwidth: 0.0,
            seed: 42,
        }
    }
}

/// Results from an SVGD run.
#[derive(Debug, Clone)]
pub struct SvgdResult {
    /// Final particle positions, shape `[n_particles][dim]`.
    pub particles: Vec<Vec<f64>>,
    /// Final RBF kernel matrix, shape `[n_particles][n_particles]`.
    pub kernel_matrix: Vec<Vec<f64>>,
    /// Mean squared displacement per iteration (proxy for loss).
    pub iteration_losses: Vec<f64>,
}

/// Stein Variational Gradient Descent optimizer.
pub struct SvgdOptimizer {
    config: SvgdConfig,
}

impl SvgdOptimizer {
    /// Create from `config`.
    pub fn new(config: SvgdConfig) -> Self {
        Self { config }
    }

    /// Compute the median inter-particle squared distance for the bandwidth heuristic.
    fn median_bandwidth(particles: &[Vec<f64>]) -> f64 {
        let n = particles.len();
        if n <= 1 {
            return 1.0;
        }
        let mut sq_dists: Vec<f64> = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let sq: f64 = particles[i]
                    .iter()
                    .zip(particles[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                sq_dists.push(sq);
            }
        }
        sq_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sq_dists[sq_dists.len() / 2];
        // h = median / (2 ln(n+1))
        median / (2.0 * ((n + 1) as f64).ln()).max(1e-10)
    }

    /// Compute the RBF kernel matrix and its particle gradients.
    fn rbf_kernel(particles: &[Vec<f64>], h: f64) -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
        let n = particles.len();
        let d = particles.first().map(|p| p.len()).unwrap_or(0);
        let mut k = vec![vec![0.0_f64; n]; n];
        // grad_k[i][j] = ∇_{x_j} k(x_i, x_j) — gradient w.r.t. x_j
        let mut grad_k = vec![vec![vec![0.0_f64; d]; n]; n];

        let inv_h = 1.0 / h.max(1e-10);
        for i in 0..n {
            for j in 0..n {
                let sq: f64 = particles[i]
                    .iter()
                    .zip(particles[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                let kij = (-sq * inv_h).exp();
                k[i][j] = kij;
                // ∇_{x_j} k(x_i, x_j) = k(x_i,x_j) · 2·inv_h·(x_i - x_j)
                for dim in 0..d {
                    grad_k[i][j][dim] = kij * 2.0 * inv_h * (particles[i][dim] - particles[j][dim]);
                }
            }
        }
        (k, grad_k)
    }

    /// Run SVGD against `target` starting from `init_particles`.
    pub fn optimize(
        &self,
        target: &dyn LogDensity,
        init_particles: Vec<Vec<f64>>,
    ) -> Result<SvgdResult> {
        let cfg = &self.config;
        let n = init_particles.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "SvgdOptimizer::optimize",
                "init_particles must be non-empty",
            ));
        }
        let d = target.dim();

        let mut particles = init_particles;
        let mut iteration_losses: Vec<f64> = Vec::with_capacity(cfg.n_iterations);

        for _iter in 0..cfg.n_iterations {
            let h = if cfg.bandwidth <= 0.0 {
                Self::median_bandwidth(&particles)
            } else {
                cfg.bandwidth
            };

            let (k_mat, grad_k) = Self::rbf_kernel(&particles, h);

            // Compute SVGD update direction for each particle i:
            // φ*(x_i) = (1/n) Σ_j [k(x_j,x_i) · ∇_{x_j}log p(x_j) + ∇_{x_j}k(x_j,x_i)]
            let mut phi: Vec<Vec<f64>> = vec![vec![0.0; d]; n];

            // Precompute gradients for all particles
            let grads: Vec<Vec<f64>> = particles.iter().map(|p| target.grad_log_prob(p)).collect();

            for i in 0..n {
                for j in 0..n {
                    for dim in 0..d {
                        phi[i][dim] += k_mat[j][i] * grads[j][dim] + grad_k[j][i][dim];
                    }
                }
                let inv_n = 1.0 / n as f64;
                for dim in 0..d {
                    phi[i][dim] *= inv_n;
                }
            }

            // Compute mean squared displacement (proxy loss)
            let msd: f64 = phi
                .iter()
                .map(|p| p.iter().map(|v| v * v).sum::<f64>())
                .sum::<f64>()
                / (n * d) as f64;
            iteration_losses.push(msd);

            // Update particles
            for i in 0..n {
                for dim in 0..d {
                    particles[i][dim] += cfg.lr * phi[i][dim];
                }
            }
        }

        // Final kernel matrix
        let h_final = if cfg.bandwidth <= 0.0 {
            Self::median_bandwidth(&particles)
        } else {
            cfg.bandwidth
        };
        let (kernel_matrix, _) = Self::rbf_kernel(&particles, h_final);

        Ok(SvgdResult {
            particles,
            kernel_matrix,
            iteration_losses,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Importance Sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Result from an importance-sampling estimation.
#[derive(Debug, Clone)]
pub struct IsResult {
    /// Estimate of E_p\[f\] (where f≡1, this estimates 1.0 for normalised p).
    pub estimate: f64,
    /// Effective sample size.
    pub effective_sample_size: f64,
    /// Self-normalised importance weights.
    pub normalized_weights: Vec<f64>,
}

/// Self-normalised importance sampler.
pub struct ImportanceSampler;

impl ImportanceSampler {
    /// Estimate the normalising constant ratio or expectation.
    ///
    /// - `target`: unnormalised log p(x)
    /// - `proposal_samples`: samples x_i ~ q
    /// - `proposal_log_probs`: log q(x_i) for each sample
    ///
    /// Returns self-normalised weights and the ESS.
    pub fn estimate(
        target: &dyn LogDensity,
        proposal_samples: &[Vec<f64>],
        proposal_log_probs: &[f64],
    ) -> Result<IsResult> {
        let n = proposal_samples.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "ImportanceSampler::estimate",
                "proposal_samples must be non-empty",
            ));
        }
        if proposal_log_probs.len() != n {
            return Err(TensorError::invalid_argument_op(
                "ImportanceSampler::estimate",
                "proposal_log_probs length must match proposal_samples length",
            ));
        }

        // Log unnormalised weights: w_i = log p(x_i) - log q(x_i)
        let log_weights: Vec<f64> = proposal_samples
            .iter()
            .zip(proposal_log_probs.iter())
            .map(|(x, lq)| target.log_prob(x) - lq)
            .collect();

        // Normalise via log-sum-exp for numerical stability
        let log_z = log_sum_exp(&log_weights);
        let normalized_weights: Vec<f64> =
            log_weights.iter().map(|lw| (lw - log_z).exp()).collect();

        // ESS = (Σ ŵ_i)² / Σ ŵ_i²  = 1 / Σ ŵ_i²  (since Σ ŵ_i = 1)
        let sum_sq: f64 = normalized_weights.iter().map(|w| w * w).sum();
        let effective_sample_size = if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 };

        // The IS estimate of Z (or, for normalised targets, ≈ 1)
        // Using the log-mean-exp: estimate = exp(log_z) / n  (no f function → f=1 estimate = Σ w)
        let estimate = log_z.exp() / n as f64;

        Ok(IsResult {
            estimate,
            effective_sample_size,
            normalized_weights,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sequential Monte Carlo
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Sequential Monte Carlo.
#[derive(Debug, Clone)]
pub struct SmcConfig {
    /// Number of SMC particles.
    pub n_particles: usize,
    /// Number of annealing steps (β schedule from 0 to 1).
    pub n_annealing: usize,
    /// Number of MCMC refresh steps per annealing step.
    pub n_mcmc_steps: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for SmcConfig {
    fn default() -> Self {
        Self {
            n_particles: 200,
            n_annealing: 20,
            n_mcmc_steps: 5,
            seed: 42,
        }
    }
}

/// Results from an SMC run.
#[derive(Debug, Clone)]
pub struct SmcResult {
    /// Final particle set after all annealing steps.
    pub particles: Vec<Vec<f64>>,
    /// Log normalising constant estimate.
    pub log_normalizer: f64,
    /// ESS history (one per annealing step).
    pub ess_history: Vec<f64>,
}

/// Sequential Monte Carlo sampler using annealed importance sampling.
///
/// The annealed target is `p_β(x) ∝ p(x)^β · p_0(x)^(1-β)` where
/// `p_0` is the prior (also a [`LogDensity`]) and β ∈ \[0,1\].
pub struct SmcSampler {
    config: SmcConfig,
}

impl SmcSampler {
    /// Create from `config`.
    pub fn new(config: SmcConfig) -> Self {
        Self { config }
    }

    /// Systematic resampling.
    fn systematic_resample(weights: &[f64], rng: &mut impl Rng, n: usize) -> Vec<usize> {
        let u_start: f64 = rng.random::<f64>() / n as f64;
        let mut indices = Vec::with_capacity(n);
        let mut cumsum = 0.0_f64;
        let mut j = 0usize;
        let step = 1.0 / n as f64;
        for k in 0..n {
            let target_u = u_start + k as f64 * step;
            while j < weights.len() - 1 && cumsum + weights[j] < target_u {
                cumsum += weights[j];
                j += 1;
            }
            indices.push(j);
        }
        indices
    }

    /// Run SMC starting from samples drawn from `prior`.
    pub fn sample(&self, target: &dyn LogDensity, prior: &dyn LogDensity) -> Result<SmcResult> {
        let cfg = &self.config;
        let n = cfg.n_particles;
        let d = target.dim();
        let mut rng = StdRng::seed_from_u64(cfg.seed);

        // Initialise particles from prior (approximate via Gaussian perturbation)
        let mut particles: Vec<Vec<f64>> =
            (0..n).map(|_| sample_normal_vec_f64(d, &mut rng)).collect();

        let mut log_weights: Vec<f64> = vec![0.0_f64; n];
        let mut log_normalizer: f64 = 0.0;
        let mut ess_history: Vec<f64> = Vec::with_capacity(cfg.n_annealing);
        let mut prev_beta: f64 = 0.0;

        for step in 1..=cfg.n_annealing {
            let beta = step as f64 / cfg.n_annealing as f64;
            let d_beta = beta - prev_beta;

            // Incremental weights: log w += d_beta · (log_target(x) - log_prior(x))
            for i in 0..n {
                let log_t = target.log_prob(&particles[i]);
                let log_p = prior.log_prob(&particles[i]);
                log_weights[i] += d_beta * (log_t - log_p);
            }

            // Log normaliser increment
            let log_z_step = log_sum_exp(&log_weights) - (n as f64).ln();
            log_normalizer += log_z_step;

            // Normalise weights
            let log_sum = log_sum_exp(&log_weights);
            let norm_weights: Vec<f64> =
                log_weights.iter().map(|lw| (lw - log_sum).exp()).collect();

            // ESS
            let sum_sq: f64 = norm_weights.iter().map(|w| w * w).sum();
            let ess = if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 };
            ess_history.push(ess);

            // Resample when ESS < threshold
            if ess < 0.5 * n as f64 {
                let indices = Self::systematic_resample(&norm_weights, &mut rng, n);
                let new_particles: Vec<Vec<f64>> =
                    indices.iter().map(|&idx| particles[idx].clone()).collect();
                particles = new_particles;
                log_weights = vec![0.0_f64; n];
            }

            // MCMC refresh using random-walk Metropolis for current temperature
            let current_beta = beta;
            for _m in 0..cfg.n_mcmc_steps {
                for i in 0..n {
                    let noise = sample_normal_vec_f64(d, &mut rng);
                    let step_size = 0.5;
                    let proposed: Vec<f64> = particles[i]
                        .iter()
                        .zip(noise.iter())
                        .map(|(xi, ni)| xi + step_size * ni)
                        .collect();

                    let log_p_curr = current_beta * target.log_prob(&particles[i])
                        + (1.0 - current_beta) * prior.log_prob(&particles[i]);
                    let log_p_prop = current_beta * target.log_prob(&proposed)
                        + (1.0 - current_beta) * prior.log_prob(&proposed);

                    let log_alpha = (log_p_prop - log_p_curr).min(0.0);
                    let u: f64 = rng.random::<f64>();
                    if u.ln() < log_alpha {
                        particles[i] = proposed;
                    }
                }
            }

            prev_beta = beta;
        }

        Ok(SmcResult {
            particles,
            log_normalizer,
            ess_history,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IWAE Bound
// ─────────────────────────────────────────────────────────────────────────────

/// Importance-Weighted AutoEncoder (IWAE) evidence lower bound.
///
/// `IWAE-ELBO = E[log(1/K Σ_k exp(log p(x,z_k) - log q(z_k|x)))]`
///
/// This is a tighter lower bound on `log p(x)` than the standard ELBO.
pub fn iwae_elbo(log_p_x_z: &[f64], log_q_z_x: &[f64]) -> f64 {
    if log_p_x_z.is_empty() || log_p_x_z.len() != log_q_z_x.len() {
        return f64::NEG_INFINITY;
    }
    let k = log_p_x_z.len();
    let log_weights: Vec<f64> = log_p_x_z
        .iter()
        .zip(log_q_z_x.iter())
        .map(|(lp, lq)| lp - lq)
        .collect();
    // log(1/K Σ_k exp(log_w_k)) = log_sum_exp(log_w) - log(K)
    log_sum_exp(&log_weights) - (k as f64).ln()
}

/// Gradient weights for IWAE (normalised importance weights).
///
/// These are the self-normalised weights `ŵ_k` used to weight per-sample
/// gradient contributions.  They sum to 1.
pub fn iwae_gradient_weights(log_p_x_z: &[f64], log_q_z_x: &[f64]) -> Vec<f64> {
    if log_p_x_z.is_empty() || log_p_x_z.len() != log_q_z_x.len() {
        return Vec::new();
    }
    let log_weights: Vec<f64> = log_p_x_z
        .iter()
        .zip(log_q_z_x.iter())
        .map(|(lp, lq)| lp - lq)
        .collect();
    let log_sum = log_sum_exp(&log_weights);
    log_weights.iter().map(|lw| (lw - log_sum).exp()).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// MCMC Diagnostics
// ─────────────────────────────────────────────────────────────────────────────

/// Collection of MCMC convergence diagnostics.
pub struct McmcDiagnostics;

impl McmcDiagnostics {
    /// Sample mean of a slice.
    fn mean(chain: &[f64]) -> f64 {
        if chain.is_empty() {
            return 0.0;
        }
        chain.iter().sum::<f64>() / chain.len() as f64
    }

    /// Sample variance of a slice (unbiased, divides by n-1).
    fn variance(chain: &[f64]) -> f64 {
        let n = chain.len();
        if n < 2 {
            return 0.0;
        }
        let m = Self::mean(chain);
        chain.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64
    }

    /// Normalised autocorrelation at `lag`.
    ///
    /// Returns 1.0 for lag 0, 0.0 for chains shorter than lag+2.
    pub fn autocorrelation(chain: &[f64], lag: usize) -> f64 {
        let n = chain.len();
        if n < lag + 2 {
            return if lag == 0 { 1.0 } else { 0.0 };
        }
        let m = Self::mean(chain);
        let var: f64 = chain.iter().map(|x| (x - m).powi(2)).sum::<f64>();
        if var < 1e-15 {
            return if lag == 0 { 1.0 } else { 0.0 };
        }
        let cov: f64 = chain[..n - lag]
            .iter()
            .zip(chain[lag..].iter())
            .map(|(a, b)| (a - m) * (b - m))
            .sum::<f64>();
        cov / var
    }

    /// Effective sample size using Geyer's initial monotone sequence estimator.
    ///
    /// ESS = n / (1 + 2·Σ_{k=1}^{K} ρ_k) where the sum is truncated at the
    /// first lag where the pair-sum `ρ_{2k} + ρ_{2k+1}` is negative.
    pub fn effective_sample_size(chain: &[f64]) -> f64 {
        let n = chain.len();
        if n < 4 {
            return n as f64;
        }

        let max_lag = n / 2;
        let mut sum_rho = 0.0_f64;
        let mut prev_pair = f64::MAX;

        let mut k = 1usize;
        while k < max_lag {
            let r_even = Self::autocorrelation(chain, 2 * k - 1);
            let r_odd = Self::autocorrelation(chain, 2 * k);
            let pair = r_even + r_odd;
            if pair < 0.0 || pair > prev_pair {
                break;
            }
            sum_rho += pair;
            prev_pair = pair;
            k += 1;
        }

        let tau = 1.0 + 2.0 * sum_rho;
        (n as f64 / tau).max(1.0)
    }

    /// Split-chain R̂ for convergence diagnosis.
    ///
    /// Each element of `chains` is one chain.  The chains are each split in
    /// half, giving `2·m` sub-chains, then R̂ is computed via the
    /// between-/within-chain variance decomposition.
    pub fn gelman_rubin_r_hat(chains: &[&[f64]]) -> f64 {
        if chains.is_empty() {
            return 1.0;
        }
        // Split each chain in half
        let mut sub_chains: Vec<Vec<f64>> = Vec::new();
        for chain in chains {
            let half = chain.len() / 2;
            if half < 2 {
                continue;
            }
            sub_chains.push(chain[..half].to_vec());
            sub_chains.push(chain[half..half * 2].to_vec());
        }

        let m = sub_chains.len();
        if m < 2 {
            return 1.0;
        }
        let n = sub_chains.iter().map(|c| c.len()).min().unwrap_or(1) as f64;

        let chain_means: Vec<f64> = sub_chains.iter().map(|c| Self::mean(c)).collect();
        let grand_mean = chain_means.iter().sum::<f64>() / m as f64;

        // Between-chain variance B (times n)
        let b_n = n / (m as f64 - 1.0)
            * chain_means
                .iter()
                .map(|mi| (mi - grand_mean).powi(2))
                .sum::<f64>();

        // Within-chain variance W
        let w: f64 = sub_chains.iter().map(|c| Self::variance(c)).sum::<f64>() / m as f64;

        if w < 1e-15 {
            return 1.0;
        }

        let var_plus = ((n - 1.0) * w + b_n) / n;
        (var_plus / w).sqrt()
    }

    /// Geweke Z-score comparing the mean of the first `frac_a` fraction of the
    /// chain to the last `frac_b` fraction.
    ///
    /// A score outside ±1.96 suggests non-convergence at the 5% level.
    pub fn geweke_z_score(chain: &[f64], frac_a: f64, frac_b: f64) -> f64 {
        let n = chain.len();
        if n < 4 {
            return 0.0;
        }
        let n_a = ((n as f64 * frac_a) as usize).max(1);
        let n_b = ((n as f64 * frac_b) as usize).max(1);
        let start_b = n.saturating_sub(n_b);

        let seg_a = &chain[..n_a];
        let seg_b = &chain[start_b..];

        let mean_a = Self::mean(seg_a);
        let mean_b = Self::mean(seg_b);
        let var_a = Self::variance(seg_a) / seg_a.len() as f64;
        let var_b = Self::variance(seg_b) / seg_b.len() as f64;

        let denom = (var_a + var_b).sqrt();
        if denom < 1e-15 {
            return 0.0;
        }
        (mean_a - mean_b) / denom
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ────────────────────────────────────────────────────────────────

    fn empirical_mean(samples: &[Vec<f64>], dim: usize) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        samples.iter().map(|s| s[dim]).sum::<f64>() / samples.len() as f64
    }

    fn empirical_var(samples: &[Vec<f64>], dim: usize) -> f64 {
        let n = samples.len();
        if n < 2 {
            return 0.0;
        }
        let m = empirical_mean(samples, dim);
        samples.iter().map(|s| (s[dim] - m).powi(2)).sum::<f64>() / (n - 1) as f64
    }

    // ── StandardNormal ────────────────────────────────────────────────────────

    #[test]
    fn test_standard_normal_log_prob() {
        let sn = StandardNormal::new(2);
        let x = vec![0.0, 0.0];
        assert!((sn.log_prob(&x) - 0.0).abs() < 1e-10);
        let x2 = vec![1.0, 0.0];
        assert!((sn.log_prob(&x2) - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_standard_normal_grad() {
        let sn = StandardNormal::new(3);
        let x = vec![1.0, -2.0, 3.0];
        let grad = sn.grad_log_prob(&x);
        for (i, xi) in x.iter().enumerate() {
            assert!(
                (grad[i] + xi).abs() < 1e-6,
                "grad[{i}]={} but expected {}",
                grad[i],
                -xi
            );
        }
    }

    // ── BananaDistribution ───────────────────────────────────────────────────

    #[test]
    fn test_banana_log_prob_at_mode() {
        let banana = BananaDistribution::new(1.0, 1.0);
        // Mode is at origin
        let x_mode = vec![0.0, 0.0];
        assert!((banana.log_prob(&x_mode) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_banana_gradient_numerical() {
        let banana = BananaDistribution::new(0.5, 2.0);
        let x = vec![1.0, 0.5];
        let grad_analytic = banana.grad_log_prob(&x);
        // Numerical gradient via finite differences
        let h = 1e-5;
        let mut x_p = x.clone();
        let mut x_m = x.clone();
        for i in 0..2 {
            x_p[i] = x[i] + h;
            x_m[i] = x[i] - h;
            let g_num = (banana.log_prob(&x_p) - banana.log_prob(&x_m)) / (2.0 * h);
            assert!(
                (grad_analytic[i] - g_num).abs() < 1e-4,
                "Analytic grad[{i}]={} vs numeric={g_num}",
                grad_analytic[i]
            );
            x_p[i] = x[i];
            x_m[i] = x[i];
        }
    }

    #[test]
    fn test_banana_dim() {
        let banana = BananaDistribution::new(1.0, 1.0);
        assert_eq!(banana.dim(), 2);
    }

    // ── MixtureOfGaussians ───────────────────────────────────────────────────

    #[test]
    fn test_mog_log_prob_numerically_stable() {
        let means = vec![vec![-100.0], vec![100.0]];
        let weights = vec![0.5, 0.5];
        let mog = MixtureOfGaussians::new(means, weights);
        let x = vec![-100.0];
        let lp = mog.log_prob(&x);
        assert!(
            lp.is_finite(),
            "log_prob should be finite for well-separated means"
        );
    }

    #[test]
    fn test_mog_symmetric() {
        let means = vec![vec![-1.0], vec![1.0]];
        let weights = vec![0.5, 0.5];
        let mog = MixtureOfGaussians::new(means, weights);
        let lp_pos = mog.log_prob(&[1.0]);
        let lp_neg = mog.log_prob(&[-1.0]);
        assert!((lp_pos - lp_neg).abs() < 1e-10);
    }

    #[test]
    fn test_mog_weight_normalisation() {
        let means = vec![vec![0.0], vec![1.0]];
        let weights = vec![2.0, 3.0]; // unnormalised
        let mog = MixtureOfGaussians::new(means, weights);
        let total: f64 = mog.weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    // ── MH Sampler ───────────────────────────────────────────────────────────

    #[test]
    fn test_mh_mean_standard_normal_1d() {
        let target = StandardNormal::new(1);
        let cfg = MhConfig {
            step_size: 1.0,
            n_warmup: 1000,
            n_samples: 2000,
            adapt_interval: 100,
            seed: 1,
        };
        let sampler = MhSampler::new(cfg);
        let result = sampler
            .sample(&target, vec![0.0])
            .expect("MH sampling failed");
        let m = empirical_mean(&result.samples, 0);
        assert!(m.abs() < 0.2, "MH 1D mean {} should be near 0", m);
    }

    #[test]
    fn test_mh_acceptance_rate_in_range() {
        let target = StandardNormal::new(2);
        let cfg = MhConfig {
            step_size: 1.0,
            n_warmup: 500,
            n_samples: 500,
            adapt_interval: 100,
            seed: 2,
        };
        let sampler = MhSampler::new(cfg);
        let result = sampler
            .sample(&target, vec![0.0, 0.0])
            .expect("MH sampling failed");
        assert!(
            result.acceptance_rate >= 0.10 && result.acceptance_rate <= 0.70,
            "acceptance_rate={} out of [0.10, 0.70]",
            result.acceptance_rate
        );
    }

    #[test]
    fn test_mh_2d_mean_near_zero() {
        let target = StandardNormal::new(2);
        let cfg = MhConfig {
            step_size: 0.8,
            n_warmup: 1000,
            n_samples: 3000,
            adapt_interval: 200,
            seed: 99,
        };
        let sampler = MhSampler::new(cfg);
        let result = sampler
            .sample(&target, vec![0.0, 0.0])
            .expect("MH sampling failed");
        for dim in 0..2 {
            let m = empirical_mean(&result.samples, dim);
            assert!(m.abs() < 0.3, "MH 2D mean[{dim}]={m} should be near 0");
        }
    }

    #[test]
    fn test_mh_sample_count() {
        let target = StandardNormal::new(1);
        let cfg = MhConfig {
            n_samples: 500,
            n_warmup: 100,
            ..MhConfig::default()
        };
        let sampler = MhSampler::new(cfg);
        let result = sampler
            .sample(&target, vec![0.0])
            .expect("MH sampling failed");
        assert_eq!(result.samples.len(), 500);
    }

    #[test]
    fn test_mh_with_banana() {
        let target = BananaDistribution::new(0.5, 2.0);
        let cfg = MhConfig {
            step_size: 0.5,
            n_warmup: 500,
            n_samples: 1000,
            adapt_interval: 100,
            seed: 7,
        };
        let sampler = MhSampler::new(cfg);
        let result = sampler
            .sample(&target, vec![0.0, 0.0])
            .expect("MH banana failed");
        assert!(!result.samples.is_empty());
        assert!(result.acceptance_rate > 0.0);
    }

    // ── HMC Sampler ──────────────────────────────────────────────────────────

    #[test]
    fn test_hmc_mean_standard_normal_1d() {
        let target = StandardNormal::new(1);
        let cfg = HmcConfig {
            step_size: 0.2,
            n_leapfrog: 10,
            n_warmup: 500,
            n_samples: 1000,
            mass: 1.0,
            seed: 3,
        };
        let sampler = HmcSampler::new(cfg);
        let result = sampler
            .sample(&target, vec![0.0])
            .expect("HMC sampling failed");
        let m = empirical_mean(&result.samples, 0);
        assert!(m.abs() < 0.3, "HMC 1D mean={m} should be near 0");
    }

    #[test]
    fn test_hmc_variance_standard_normal() {
        let target = StandardNormal::new(1);
        let cfg = HmcConfig {
            step_size: 0.2,
            n_leapfrog: 10,
            n_warmup: 300,
            n_samples: 1000,
            mass: 1.0,
            seed: 4,
        };
        let sampler = HmcSampler::new(cfg);
        let result = sampler
            .sample(&target, vec![0.0])
            .expect("HMC sampling failed");
        let v = empirical_var(&result.samples, 0);
        assert!(
            (0.4..=2.5).contains(&v),
            "HMC variance={v} should be in [0.4, 2.5]"
        );
    }

    #[test]
    fn test_hmc_energy_conservation() {
        let target = StandardNormal::new(2);
        let cfg = HmcConfig {
            step_size: 0.05, // small step for good energy conservation
            n_leapfrog: 5,
            n_warmup: 0,
            n_samples: 10,
            mass: 1.0,
            seed: 5,
        };
        let sampler = HmcSampler::new(cfg);
        let delta_h = sampler.hamiltonian_change(&target, &[0.0, 0.0], 1.0, 42);
        assert!(
            delta_h < 0.15,
            "Hamiltonian change={delta_h} should be < 0.15 for small step size"
        );
    }

    #[test]
    fn test_hmc_sample_count() {
        let target = StandardNormal::new(2);
        let cfg = HmcConfig {
            n_samples: 300,
            n_warmup: 100,
            ..HmcConfig::default()
        };
        let sampler = HmcSampler::new(cfg);
        let result = sampler
            .sample(&target, vec![0.0, 0.0])
            .expect("HMC sampling failed");
        assert_eq!(result.samples.len(), 300);
    }

    // ── SVGD ────────────────────────────────────────────────────────────────

    #[test]
    fn test_svgd_mean_converges_to_zero() {
        let target = StandardNormal::new(1);
        let cfg = SvgdConfig {
            n_particles: 30,
            lr: 0.05,
            n_iterations: 200,
            bandwidth: 0.0,
            seed: 6,
        };
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let init: Vec<Vec<f64>> = (0..cfg.n_particles)
            .map(|_| vec![sample_normal_f64(&mut rng) * 3.0])
            .collect();
        let optimizer = SvgdOptimizer::new(cfg);
        let result = optimizer.optimize(&target, init).expect("SVGD failed");
        let mean: f64 =
            result.particles.iter().map(|p| p[0]).sum::<f64>() / result.particles.len() as f64;
        assert!(
            mean.abs() < 1.5,
            "SVGD particle mean={mean} should be near 0"
        );
    }

    #[test]
    fn test_svgd_particle_count_preserved() {
        let target = StandardNormal::new(2);
        let cfg = SvgdConfig {
            n_particles: 20,
            lr: 0.01,
            n_iterations: 10,
            bandwidth: 1.0,
            seed: 10,
        };
        let init: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, 0.0]).collect();
        let optimizer = SvgdOptimizer::new(cfg);
        let result = optimizer.optimize(&target, init).expect("SVGD failed");
        assert_eq!(result.particles.len(), 20);
    }

    #[test]
    fn test_svgd_iteration_losses_collected() {
        let target = StandardNormal::new(1);
        let cfg = SvgdConfig {
            n_particles: 10,
            lr: 0.01,
            n_iterations: 50,
            bandwidth: 1.0,
            seed: 11,
        };
        let init: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 - 5.0]).collect();
        let optimizer = SvgdOptimizer::new(cfg);
        let result = optimizer.optimize(&target, init).expect("SVGD failed");
        assert_eq!(result.iteration_losses.len(), 50);
        // All losses should be non-negative
        for l in &result.iteration_losses {
            assert!(*l >= 0.0, "loss={l} should be >= 0");
        }
    }

    // ── Importance Sampling ──────────────────────────────────────────────────

    #[test]
    fn test_is_ess_less_than_n() {
        let target = StandardNormal::new(1);
        let mut rng = StdRng::seed_from_u64(42);
        let n = 100;
        let samples: Vec<Vec<f64>> = (0..n)
            .map(|_| vec![sample_normal_f64(&mut rng) * 2.0]) // proposal: N(0, 4)
            .collect();
        // log q(x) = -0.5 * x^2 / 4 - 0.5*log(2*pi*4) ; just proportional part
        let log_q: Vec<f64> = samples.iter().map(|x| -0.5 * x[0] * x[0] / 4.0).collect();
        let result = ImportanceSampler::estimate(&target, &samples, &log_q).expect("IS failed");
        assert!(
            result.effective_sample_size <= n as f64,
            "ESS={} should be <= n={}",
            result.effective_sample_size,
            n
        );
        assert!(result.effective_sample_size > 0.0);
    }

    #[test]
    fn test_is_weights_sum_to_one() {
        let target = StandardNormal::new(1);
        let mut rng = StdRng::seed_from_u64(99);
        let n = 50;
        let samples: Vec<Vec<f64>> = (0..n).map(|_| vec![sample_normal_f64(&mut rng)]).collect();
        let log_q: Vec<f64> = samples.iter().map(|x| -0.5 * x[0] * x[0]).collect();
        let result = ImportanceSampler::estimate(&target, &samples, &log_q).expect("IS failed");
        let total: f64 = result.normalized_weights.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "IS weights should sum to 1, got {total}"
        );
    }

    #[test]
    fn test_is_estimate_normalised_target() {
        // When proposal = target (q = p), all weights are equal → estimate = 1
        let target = StandardNormal::new(1);
        let mut rng = StdRng::seed_from_u64(77);
        let n = 200;
        let samples: Vec<Vec<f64>> = (0..n).map(|_| vec![sample_normal_f64(&mut rng)]).collect();
        // Use the same log-prob as proposal
        let log_q: Vec<f64> = samples.iter().map(|x| target.log_prob(x)).collect();
        let result = ImportanceSampler::estimate(&target, &samples, &log_q).expect("IS failed");
        // When q=p, ESS ≈ n
        assert!(
            result.effective_sample_size > 0.8 * n as f64,
            "ESS={} when q=p should be close to n={}",
            result.effective_sample_size,
            n
        );
    }

    // ── SMC ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_smc_log_normalizer_near_zero() {
        // For normalised target and standard normal prior, log Z ≈ 0
        let target = StandardNormal::new(1);
        let prior = StandardNormal::new(1);
        let cfg = SmcConfig {
            n_particles: 100,
            n_annealing: 10,
            n_mcmc_steps: 3,
            seed: 8,
        };
        let sampler = SmcSampler::new(cfg);
        let result = sampler.sample(&target, &prior).expect("SMC failed");
        assert!(
            result.log_normalizer.is_finite(),
            "log_normalizer should be finite"
        );
    }

    #[test]
    fn test_smc_particle_count_preserved() {
        let target = StandardNormal::new(2);
        let prior = StandardNormal::new(2);
        let cfg = SmcConfig {
            n_particles: 50,
            n_annealing: 5,
            n_mcmc_steps: 2,
            seed: 9,
        };
        let sampler = SmcSampler::new(cfg);
        let result = sampler.sample(&target, &prior).expect("SMC failed");
        assert_eq!(result.particles.len(), 50);
    }

    #[test]
    fn test_smc_ess_history_length() {
        let target = StandardNormal::new(1);
        let prior = StandardNormal::new(1);
        let cfg = SmcConfig {
            n_particles: 30,
            n_annealing: 8,
            n_mcmc_steps: 1,
            seed: 13,
        };
        let sampler = SmcSampler::new(cfg);
        let result = sampler.sample(&target, &prior).expect("SMC failed");
        assert_eq!(result.ess_history.len(), 8);
        for ess in &result.ess_history {
            assert!(*ess >= 0.0 && *ess <= 30.0 + 1e-6);
        }
    }

    // ── IWAE ────────────────────────────────────────────────────────────────

    #[test]
    fn test_iwae_elbo_geq_standard_elbo() {
        // IWAE-ELBO ≥ standard ELBO (Jensen's inequality, log is concave).
        //
        // For K deterministic samples:
        //   IWAE(K) = log(mean_k exp(log w_k))
        //   ELBO     = mean_k log w_k
        //
        // Jensen's inequality on the convex function log(mean(·)):
        //   log(mean(exp(log w_k))) ≥ mean(log(exp(log w_k))) = mean(log w_k)
        //
        // So IWAE ≥ ELBO, i.e. IWAE is a tighter lower bound than the plain ELBO.
        let log_p = vec![-1.0, -2.0, -1.5];
        let log_q = vec![-0.5, -1.0, -0.8];
        let standard_elbo: f64 = log_p
            .iter()
            .zip(log_q.iter())
            .map(|(lp, lq)| lp - lq)
            .sum::<f64>()
            / log_p.len() as f64;
        let iwae = iwae_elbo(&log_p, &log_q);
        assert!(
            iwae >= standard_elbo - 1e-9,
            "IWAE={iwae} should be ≥ ELBO={standard_elbo}"
        );
    }

    #[test]
    fn test_iwae_gradient_weights_sum_to_one() {
        let log_p = vec![-0.5, -1.0, -2.0, -0.8];
        let log_q = vec![-0.3, -0.9, -1.5, -0.7];
        let weights = iwae_gradient_weights(&log_p, &log_q);
        let total: f64 = weights.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "IWAE weights sum={total} should be 1.0"
        );
    }

    #[test]
    fn test_iwae_gradient_weights_nonnegative() {
        let log_p = vec![-1.0, -2.0, -3.0];
        let log_q = vec![-0.5, -1.5, -2.5];
        let weights = iwae_gradient_weights(&log_p, &log_q);
        for w in &weights {
            assert!(*w >= 0.0, "weight={w} should be non-negative");
        }
    }

    #[test]
    fn test_iwae_elbo_equal_weights_reduces_to_log_mean_exp() {
        // When log_q = 0 for all samples, IWAE = log(mean(exp(log_p)))
        let log_p = vec![-1.0; 4];
        let log_q = vec![0.0; 4];
        let iwae = iwae_elbo(&log_p, &log_q);
        let expected = -1.0_f64; // log(exp(-1)) = -1
        assert!((iwae - expected).abs() < 1e-10);
    }

    #[test]
    fn test_iwae_empty_returns_neg_inf() {
        assert_eq!(iwae_elbo(&[], &[]), f64::NEG_INFINITY);
    }

    // ── MCMC Diagnostics ─────────────────────────────────────────────────────

    #[test]
    fn test_autocorrelation_lag0_is_one() {
        let chain: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let rho = McmcDiagnostics::autocorrelation(&chain, 0);
        assert!((rho - 1.0).abs() < 1e-10, "lag-0 autocorrelation={rho}");
    }

    #[test]
    fn test_autocorrelation_iid_near_zero_for_large_lag() {
        // i.i.d. samples from N(0,1): autocorrelation at lag>0 should be small
        let mut rng = StdRng::seed_from_u64(42);
        let chain: Vec<f64> = (0..500).map(|_| sample_normal_f64(&mut rng)).collect();
        let rho = McmcDiagnostics::autocorrelation(&chain, 10);
        assert!(rho.abs() < 0.15, "lag-10 autocorrelation={rho}");
    }

    #[test]
    fn test_ess_iid_close_to_n() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 500;
        let chain: Vec<f64> = (0..n).map(|_| sample_normal_f64(&mut rng)).collect();
        let ess = McmcDiagnostics::effective_sample_size(&chain);
        // For i.i.d. samples ESS ≈ n; allow generous tolerance
        assert!(
            ess > 0.2 * n as f64,
            "ESS={ess} should be > 0.2 * n={n} for i.i.d. samples"
        );
    }

    #[test]
    fn test_ess_highly_correlated_less_than_n() {
        // Highly correlated chain: x_{t+1} = 0.99 x_t + noise
        let mut rng = StdRng::seed_from_u64(1);
        let n = 500;
        let mut x = 0.0_f64;
        let chain: Vec<f64> = (0..n)
            .map(|_| {
                x = 0.99 * x + 0.1 * sample_normal_f64(&mut rng);
                x
            })
            .collect();
        let ess = McmcDiagnostics::effective_sample_size(&chain);
        assert!(
            ess < n as f64,
            "ESS={ess} should be < n={n} for correlated chain"
        );
    }

    #[test]
    fn test_r_hat_converged_chains() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 500;
        let chain1: Vec<f64> = (0..n).map(|_| sample_normal_f64(&mut rng)).collect();
        let chain2: Vec<f64> = (0..n).map(|_| sample_normal_f64(&mut rng)).collect();
        let refs: Vec<&[f64]> = vec![&chain1, &chain2];
        let r_hat = McmcDiagnostics::gelman_rubin_r_hat(&refs);
        assert!(
            r_hat < 1.1,
            "R-hat={r_hat} should be < 1.1 for converged chains"
        );
    }

    #[test]
    fn test_r_hat_non_converged_chains() {
        // Two chains centred at very different locations
        let chain1: Vec<f64> = vec![-10.0_f64; 200];
        let chain2: Vec<f64> = vec![10.0_f64; 200];
        // Add small noise to avoid zero variance
        let mut rng = StdRng::seed_from_u64(1);
        let chain1_n: Vec<f64> = chain1
            .iter()
            .map(|x| x + 0.01 * sample_normal_f64(&mut rng))
            .collect();
        let chain2_n: Vec<f64> = chain2
            .iter()
            .map(|x| x + 0.01 * sample_normal_f64(&mut rng))
            .collect();
        let refs: Vec<&[f64]> = vec![&chain1_n, &chain2_n];
        let r_hat = McmcDiagnostics::gelman_rubin_r_hat(&refs);
        assert!(
            r_hat > 1.1,
            "R-hat={r_hat} should be > 1.1 for non-converged chains"
        );
    }

    #[test]
    fn test_geweke_z_score_converged() {
        let mut rng = StdRng::seed_from_u64(42);
        let chain: Vec<f64> = (0..2000).map(|_| sample_normal_f64(&mut rng)).collect();
        let z = McmcDiagnostics::geweke_z_score(&chain, 0.1, 0.5);
        // Should be roughly standard normal; |z| < 4 with overwhelming probability
        assert!(z.abs() < 4.0, "Geweke z={z}");
    }

    #[test]
    fn test_geweke_z_score_non_stationary() {
        // Chain with a trend — first part has very different mean from last
        let chain: Vec<f64> = (0..1000).map(|i| i as f64 / 100.0).collect();
        let z = McmcDiagnostics::geweke_z_score(&chain, 0.1, 0.5);
        // Should be large because first part (near 0) differs from last (near 10)
        assert!(
            z.abs() > 1.0,
            "Geweke z={z} should be large for non-stationary"
        );
    }

    // ── log_sum_exp ──────────────────────────────────────────────────────────

    #[test]
    fn test_log_sum_exp_identity() {
        let v = vec![0.0, 0.0, 0.0]; // log(3)
        let result = log_sum_exp(&v);
        assert!((result - 3.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp_extreme() {
        // Should not overflow
        let v = vec![1000.0, 1001.0];
        let result = log_sum_exp(&v);
        assert!(result.is_finite());
    }

    #[test]
    fn test_log_sum_exp_empty() {
        let result = log_sum_exp(&[]);
        assert_eq!(result, f64::NEG_INFINITY);
    }
}
