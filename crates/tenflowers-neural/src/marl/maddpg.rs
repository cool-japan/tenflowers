//! MADDPG algorithm: multi-agent DDPG with centralized critic.
//!
//! Contains: MaddpgActor, MaddpgCritic, OuNoise, MaddpgAgent, MaddpgTrainer.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

use super::types::{Agent, Experience, SharedReplayBuffer};

// ─────────────────────────────────────────────────────────────────────────────
// § 5. MADDPG
// ─────────────────────────────────────────────────────────────────────────────

/// Per-agent deterministic policy network (actor).
#[derive(Debug, Clone)]
pub struct MaddpgActor {
    pub(crate) obs_dim: usize,
    pub(crate) action_dim: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) w1: Vec<f64>,
    b1: Vec<f64>,
    pub(crate) w2: Vec<f64>,
    b2: Vec<f64>,
}

impl MaddpgActor {
    /// Create with He initialisation.
    pub fn new(obs_dim: usize, action_dim: usize, hidden_dim: usize, seed: u64) -> Result<Self> {
        if obs_dim == 0 || action_dim == 0 || hidden_dim == 0 {
            return Err(TensorError::invalid_argument(
                "MaddpgActor dimensions must be > 0".into(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let s1 = (2.0_f64 / obs_dim as f64).sqrt();
        let s2 = (2.0_f64 / hidden_dim as f64).sqrt();
        let w1 = (0..hidden_dim * obs_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * s1)
            .collect();
        let b1 = vec![0.0_f64; hidden_dim];
        let w2 = (0..action_dim * hidden_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * s2)
            .collect();
        let b2 = vec![0.0_f64; action_dim];
        Ok(Self {
            obs_dim,
            action_dim,
            hidden_dim,
            w1,
            b1,
            w2,
            b2,
        })
    }

    /// Forward pass: returns actions in [-1, 1] via tanh.
    pub fn forward(&self, obs: &[f64]) -> Result<Vec<f64>> {
        if obs.len() != self.obs_dim {
            return Err(TensorError::invalid_argument(format!(
                "MaddpgActor expected obs_dim={} got {}",
                self.obs_dim,
                obs.len()
            )));
        }
        let mut h = vec![0.0_f64; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut v = self.b1[i];
            for j in 0..self.obs_dim {
                v += self.w1[i * self.obs_dim + j] * obs[j];
            }
            h[i] = v.max(0.0); // ReLU
        }
        let mut a = vec![0.0_f64; self.action_dim];
        for i in 0..self.action_dim {
            let mut v = self.b2[i];
            for j in 0..self.hidden_dim {
                v += self.w2[i * self.hidden_dim + j] * h[j];
            }
            a[i] = v.tanh();
        }
        Ok(a)
    }

    /// Soft-update toward `other`: θ ← (1-τ)θ + τ·θ_other
    pub fn soft_update(&mut self, other: &MaddpgActor, tau: f64) -> Result<()> {
        if self.w1.len() != other.w1.len() {
            return Err(TensorError::invalid_argument(
                "MaddpgActor soft_update: parameter dimension mismatch".into(),
            ));
        }
        for (a, b) in self.w1.iter_mut().zip(&other.w1) {
            *a = (1.0 - tau) * *a + tau * b;
        }
        for (a, b) in self.b1.iter_mut().zip(&other.b1) {
            *a = (1.0 - tau) * *a + tau * b;
        }
        for (a, b) in self.w2.iter_mut().zip(&other.w2) {
            *a = (1.0 - tau) * *a + tau * b;
        }
        for (a, b) in self.b2.iter_mut().zip(&other.b2) {
            *a = (1.0 - tau) * *a + tau * b;
        }
        Ok(())
    }
}

/// Centralized critic observing all agents' observations and actions.
#[derive(Debug, Clone)]
pub struct MaddpgCritic {
    pub(crate) input_dim: usize,
    hidden_dim: usize,
    pub(crate) w1: Vec<f64>,
    pub(crate) b1: Vec<f64>,
    pub(crate) w2: Vec<f64>,
    pub(crate) b2: Vec<f64>,
}

impl MaddpgCritic {
    /// Create with combined observation+action dimension.
    ///
    /// `input_dim = sum(obs_dims) + sum(action_dims)` for all agents.
    pub fn new(input_dim: usize, hidden_dim: usize, seed: u64) -> Result<Self> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err(TensorError::invalid_argument(
                "MaddpgCritic dimensions must be > 0".into(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let s1 = (2.0_f64 / input_dim as f64).sqrt();
        let s2 = (2.0_f64 / hidden_dim as f64).sqrt();
        let w1 = (0..hidden_dim * input_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * s1)
            .collect();
        let b1 = vec![0.0_f64; hidden_dim];
        let w2 = (0..hidden_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * s2)
            .collect();
        let b2 = vec![0.0_f64; 1];
        Ok(Self {
            input_dim,
            hidden_dim,
            w1,
            b1,
            w2,
            b2,
        })
    }

    /// Forward pass: returns scalar Q-value given joint (obs, actions).
    pub fn forward(&self, joint_input: &[f64]) -> Result<f64> {
        if joint_input.len() != self.input_dim {
            return Err(TensorError::invalid_argument(format!(
                "MaddpgCritic expected input_dim={} got {}",
                self.input_dim,
                joint_input.len()
            )));
        }
        let mut h = vec![0.0_f64; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut v = self.b1[i];
            for j in 0..self.input_dim {
                v += self.w1[i * self.input_dim + j] * joint_input[j];
            }
            h[i] = v.max(0.0);
        }
        let mut q = self.b2[0];
        for j in 0..self.hidden_dim {
            q += self.w2[j] * h[j];
        }
        Ok(q)
    }

    /// Soft-update toward `other`.
    pub fn soft_update(&mut self, other: &MaddpgCritic, tau: f64) -> Result<()> {
        if self.w1.len() != other.w1.len() {
            return Err(TensorError::invalid_argument(
                "MaddpgCritic soft_update: parameter dimension mismatch".into(),
            ));
        }
        for (a, b) in self.w1.iter_mut().zip(&other.w1) {
            *a = (1.0 - tau) * *a + tau * b;
        }
        for (a, b) in self.b1.iter_mut().zip(&other.b1) {
            *a = (1.0 - tau) * *a + tau * b;
        }
        for (a, b) in self.w2.iter_mut().zip(&other.w2) {
            *a = (1.0 - tau) * *a + tau * b;
        }
        for (a, b) in self.b2.iter_mut().zip(&other.b2) {
            *a = (1.0 - tau) * *a + tau * b;
        }
        Ok(())
    }
}

/// Ornstein-Uhlenbeck noise process for continuous action-space exploration.
#[derive(Debug, Clone)]
pub struct OuNoise {
    pub mu: Vec<f64>,
    pub theta: f64,
    pub sigma: f64,
    pub state: Vec<f64>,
    seed: u64,
}

impl OuNoise {
    /// Construct with zero mean, θ=0.15, σ=0.2.
    pub fn new(dim: usize, seed: u64) -> Self {
        Self {
            mu: vec![0.0; dim],
            theta: 0.15,
            sigma: 0.2,
            state: vec![0.0; dim],
            seed,
        }
    }

    /// Sample next noise vector using Euler-Maruyama discretization.
    pub fn sample(&mut self) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        self.seed = self.seed.wrapping_add(1);
        let n: Vec<f64> = (0..self.state.len())
            .map(|_| {
                // Box-Muller for standard normal
                let u1: f64 = rng.random::<f64>().max(1e-12);
                let u2: f64 = rng.random::<f64>();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect();
        self.state = self
            .state
            .iter()
            .zip(self.mu.iter())
            .zip(n.iter())
            .map(|((s, m), ni)| s + self.theta * (m - s) + self.sigma * ni)
            .collect();
        self.state.clone()
    }

    /// Reset state to zero.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }
}

/// Full MADDPG agent: actor + critic + target networks + exploration noise.
#[derive(Debug, Clone)]
pub struct MaddpgAgent {
    pub actor: MaddpgActor,
    pub actor_target: MaddpgActor,
    pub critic: MaddpgCritic,
    pub critic_target: MaddpgCritic,
    pub noise: OuNoise,
    pub agent_id: usize,
    pub tau: f64,
}

impl MaddpgAgent {
    /// Create with τ=0.01 soft-update rate.
    pub fn new(
        agent_id: usize,
        obs_dim: usize,
        action_dim: usize,
        hidden_dim: usize,
        critic_input_dim: usize,
        seed: u64,
    ) -> Result<Self> {
        let actor = MaddpgActor::new(obs_dim, action_dim, hidden_dim, seed)?;
        let actor_target = actor.clone();
        let critic = MaddpgCritic::new(critic_input_dim, hidden_dim, seed + 1000)?;
        let critic_target = critic.clone();
        let noise = OuNoise::new(action_dim, seed + 2000);
        Ok(Self {
            actor,
            actor_target,
            critic,
            critic_target,
            noise,
            agent_id,
            tau: 0.01,
        })
    }

    /// Act with exploration noise.
    pub fn act_with_noise(&mut self, obs: &[f64]) -> Result<Vec<f64>> {
        let mut a = self.actor.forward(obs)?;
        let noise = self.noise.sample();
        for (ai, ni) in a.iter_mut().zip(&noise) {
            *ai = (*ai + ni).clamp(-1.0, 1.0);
        }
        Ok(a)
    }

    /// Perform soft target-network updates.
    pub fn update_targets(&mut self) -> Result<()> {
        self.actor_target.soft_update(&self.actor, self.tau)?;
        self.critic_target.soft_update(&self.critic, self.tau)
    }
}

impl Agent for MaddpgAgent {
    fn act(&self, obs: &[f64]) -> Vec<f64> {
        self.actor
            .forward(obs)
            .unwrap_or_else(|_| vec![0.0; self.actor.action_dim])
    }

    fn update(&mut self, _experience: &Experience) {}

    fn id(&self) -> usize {
        self.agent_id
    }
}

/// Coordinates centralized training / decentralized execution for MADDPG.
#[derive(Debug)]
pub struct MaddpgTrainer {
    pub agents: Vec<MaddpgAgent>,
    pub replay: SharedReplayBuffer,
    pub gamma: f64,
    pub batch_size: usize,
    pub(crate) step: u64,
}

impl MaddpgTrainer {
    /// Construct trainer.
    pub fn new(
        agents: Vec<MaddpgAgent>,
        buffer_capacity: usize,
        gamma: f64,
        batch_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            agents,
            replay: SharedReplayBuffer::new(buffer_capacity)?,
            gamma,
            batch_size,
            step: 0,
        })
    }

    /// Compute DDPG-style TD loss for one agent given a joint batch.
    ///
    /// The joint input for the critic is `[all_obs || all_actions]`.
    pub fn critic_loss(
        &self,
        agent_idx: usize,
        batch: &[Experience],
        all_next_actions: &[Vec<f64>],
    ) -> Result<f64> {
        let agent = self
            .agents
            .get(agent_idx)
            .ok_or_else(|| TensorError::invalid_argument(format!("No agent {agent_idx}")))?;
        let mut loss = 0.0_f64;
        for (exp, next_action_row) in batch.iter().zip(all_next_actions) {
            let joint_curr: Vec<f64> = exp.obs.iter().chain(exp.action.iter()).cloned().collect();
            if joint_curr.len() != agent.critic.input_dim {
                return Err(TensorError::invalid_argument(format!(
                    "Critic input_dim mismatch: expected {} got {}",
                    agent.critic.input_dim,
                    joint_curr.len()
                )));
            }
            let q_curr = agent.critic.forward(&joint_curr)?;
            let joint_next: Vec<f64> = exp
                .next_obs
                .iter()
                .chain(next_action_row.iter())
                .cloned()
                .collect();
            if joint_next.len() != agent.critic_target.input_dim {
                return Err(TensorError::invalid_argument(
                    "Critic target input_dim mismatch".into(),
                ));
            }
            let q_next = agent.critic_target.forward(&joint_next)?;
            let target = exp.reward + if exp.done { 0.0 } else { self.gamma * q_next };
            loss += (q_curr - target).powi(2);
        }
        Ok(loss / batch.len() as f64)
    }

    /// Update all target networks.
    pub fn update_targets(&mut self) -> Result<()> {
        for agent in &mut self.agents {
            agent.update_targets()?;
        }
        self.step += 1;
        Ok(())
    }
}
