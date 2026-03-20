//! Robotics extensions: sim-to-real transfer components.
//!
//! This module contains domain randomization, adaptation modules, MMD-based
//! domain adaptation loss, and transfer success evaluation.

use super::{dense_linear, dense_relu, kaiming_uniform};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ═══════════════════ SECTION 5 — SIM-TO-REAL TRANSFER ══════════════════════

/// Physics parameters sampled during domain randomisation.
#[derive(Debug, Clone)]
pub struct PhysicsParams {
    /// Body mass in kg.
    pub mass: f64,
    /// Coulomb friction coefficient.
    pub friction: f64,
    /// Viscous damping coefficient.
    pub damping: f64,
    /// Discrete time-step delay on action application.
    pub action_delay: usize,
}

/// Domain randomiser: samples physics parameters from configurable uniform ranges.
#[derive(Debug, Clone)]
pub struct DomainRandomizer {
    /// Uniform range for mass [min, max].
    pub mass_range: (f64, f64),
    /// Uniform range for friction [min, max].
    pub friction_range: (f64, f64),
    /// Uniform range for damping [min, max].
    pub damping_range: (f64, f64),
    /// Discrete range for action delay [min, max] (inclusive).
    pub action_delay_range: (usize, usize),
}
impl DomainRandomizer {
    /// Create a new domain randomiser with given parameter ranges.
    pub fn new(
        mass_range: (f64, f64),
        friction_range: (f64, f64),
        damping_range: (f64, f64),
        action_delay_range: (usize, usize),
    ) -> Self {
        Self {
            mass_range,
            friction_range,
            damping_range,
            action_delay_range,
        }
    }
    /// Sample a random physics parameter set from the configured ranges.
    pub fn sample_params(&self, rng: &mut StdRng) -> PhysicsParams {
        let mass =
            self.mass_range.0 + rng.random::<f64>() * (self.mass_range.1 - self.mass_range.0);
        let friction = self.friction_range.0
            + rng.random::<f64>() * (self.friction_range.1 - self.friction_range.0);
        let damping = self.damping_range.0
            + rng.random::<f64>() * (self.damping_range.1 - self.damping_range.0);
        let delay_range = (self.action_delay_range.1 - self.action_delay_range.0) + 1;
        let action_delay =
            self.action_delay_range.0 + ((rng.random::<u64>() as usize) % delay_range);
        PhysicsParams {
            mass,
            friction,
            damping,
            action_delay,
        }
    }
}

/// Context-conditioned adaptation module — encodes short trajectory into context embedding.
#[derive(Debug, Clone)]
pub struct AdaptationModule {
    transition_dim: usize,
    context_dim: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}
impl AdaptationModule {
    /// Create a new adaptation module.
    pub fn new(transition_dim: usize, context_dim: usize, seed: u64) -> Result<Self> {
        if transition_dim == 0 || context_dim == 0 {
            return Err(TensorError::invalid_argument_op(
                "AdaptationModule::new",
                "dims must be > 0",
            ));
        }
        let hidden = (transition_dim + context_dim).max(8);
        Ok(Self {
            transition_dim,
            context_dim,
            w1: kaiming_uniform(transition_dim, hidden, seed),
            b1: vec![0.0_f32; hidden],
            w2: kaiming_uniform(hidden, context_dim, seed.wrapping_add(1)),
            b2: vec![0.0_f32; context_dim],
        })
    }
    /// Encode a sequence of recent transitions into a mean-pooled context vector.
    pub fn adapt(&self, recent_transitions: &[Vec<f32>], _rng: &mut StdRng) -> Result<Vec<f32>> {
        if recent_transitions.is_empty() {
            return Ok(vec![0.0_f32; self.context_dim]);
        }
        let hidden = (self.transition_dim + self.context_dim).max(8);
        let mut sum = vec![0.0_f32; self.context_dim];
        for t in recent_transitions {
            if t.len() != self.transition_dim {
                return Err(TensorError::invalid_argument_op(
                    "AdaptationModule::adapt",
                    &format!(
                        "transition dim {} != expected {}",
                        t.len(),
                        self.transition_dim
                    ),
                ));
            }
            let emb = dense_linear(
                &self.w2,
                &self.b2,
                &dense_relu(&self.w1, &self.b1, t, hidden),
                self.context_dim,
            );
            for (s, e) in sum.iter_mut().zip(&emb) {
                *s += e;
            }
        }
        let n = recent_transitions.len() as f32;
        Ok(sum.iter().map(|&v| v / n).collect())
    }
}

/// MMD loss for minimising distributional gap between sim and real feature distributions.
#[derive(Debug, Clone)]
pub struct DomainAdaptationLoss {
    /// RBF kernel bandwidth σ².
    pub sigma_sq: f64,
}
impl DomainAdaptationLoss {
    /// Create a new MMD loss with given RBF bandwidth.
    pub fn new(sigma_sq: f64) -> Self {
        Self {
            sigma_sq: sigma_sq.max(f64::EPSILON),
        }
    }
    fn rbf(&self, x: &[f32], y: &[f32]) -> f64 {
        let d: f64 = x
            .iter()
            .zip(y)
            .map(|(&a, &b)| {
                let d = (a - b) as f64;
                d * d
            })
            .sum();
        (-d / (2.0 * self.sigma_sq)).exp()
    }
    /// Compute the unbiased MMD² between `sim` and `real` feature sets.
    pub fn mmd_loss(&self, sim: &[Vec<f32>], real: &[Vec<f32>]) -> f64 {
        let (n, m) = (sim.len(), real.len());
        if n == 0 || m == 0 {
            return 0.0;
        }
        let mut kxx = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    kxx += self.rbf(&sim[i], &sim[j]);
                }
            }
        }
        kxx /= (n * (n - 1)).max(1) as f64;
        let mut kyy = 0.0_f64;
        for i in 0..m {
            for j in 0..m {
                if i != j {
                    kyy += self.rbf(&real[i], &real[j]);
                }
            }
        }
        kyy /= (m * (m - 1)).max(1) as f64;
        let mut kxy = 0.0_f64;
        for i in 0..n {
            for j in 0..m {
                kxy += self.rbf(&sim[i], &real[j]);
            }
        }
        kxy /= (n * m) as f64;
        (kxx - 2.0 * kxy + kyy).max(0.0)
    }
}

/// Tracks policy transfer success rate and domain shift metrics.
#[derive(Debug, Clone)]
pub struct SimToRealEvaluator {
    sim_rewards: Vec<f64>,
    real_rewards: Vec<f64>,
    mmd_history: Vec<f64>,
}
impl SimToRealEvaluator {
    /// Create a new evaluator with empty history.
    pub fn new() -> Self {
        Self {
            sim_rewards: Vec::new(),
            real_rewards: Vec::new(),
            mmd_history: Vec::new(),
        }
    }
    /// Record a simulated episode reward.
    pub fn record_sim_reward(&mut self, r: f64) {
        self.sim_rewards.push(r);
    }
    /// Record a real-world episode reward.
    pub fn record_real_reward(&mut self, r: f64) {
        self.real_rewards.push(r);
    }
    /// Record an MMD domain gap measurement.
    pub fn record_mmd(&mut self, mmd: f64) {
        self.mmd_history.push(mmd);
    }
    fn mean(v: &[f64]) -> f64 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    }
    /// Fraction of real episodes achieving at least `threshold_fraction` of mean sim reward.
    pub fn transfer_success_rate(&self, threshold_fraction: f64) -> f64 {
        if self.real_rewards.is_empty() || self.sim_rewards.is_empty() {
            return 0.0;
        }
        let threshold = threshold_fraction * Self::mean(&self.sim_rewards);
        self.real_rewards
            .iter()
            .filter(|&&r| r >= threshold)
            .count() as f64
            / self.real_rewards.len() as f64
    }
    /// Mean of recorded MMD domain shift measurements.
    pub fn mean_domain_shift(&self) -> f64 {
        Self::mean(&self.mmd_history)
    }
    /// Mean real-world episode reward.
    pub fn mean_real_reward(&self) -> f64 {
        Self::mean(&self.real_rewards)
    }
}
impl Default for SimToRealEvaluator {
    fn default() -> Self {
        Self::new()
    }
}
