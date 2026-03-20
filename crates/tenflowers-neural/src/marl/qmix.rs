//! QMIX algorithm: per-agent Q-networks + hypernetwork mixing.
//!
//! Contains: AgentQNetwork, MixingNetwork, QmixAgent, QmixTrainer.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

use super::types::Experience;

// ─────────────────────────────────────────────────────────────────────────────
// § 4. QMIX
// ─────────────────────────────────────────────────────────────────────────────

/// Per-agent Q-network: linear model with optional GRU-style recurrent state.
///
/// Implements a two-layer MLP: `obs → hidden → Q(obs, a)` for each
/// discrete action, plus a hidden state vector simulating GRU memory.
#[derive(Debug, Clone)]
pub struct AgentQNetwork {
    pub(crate) obs_dim: usize,
    pub(crate) action_dim: usize,
    pub(crate) hidden_dim: usize,
    /// Weights for layer 1: (hidden_dim × obs_dim)
    pub(crate) w1: Vec<f64>,
    pub(crate) b1: Vec<f64>,
    /// Weights for layer 2: (action_dim × hidden_dim)
    pub(crate) w2: Vec<f64>,
    pub(crate) b2: Vec<f64>,
    /// GRU-style recurrent weights: (hidden_dim × hidden_dim)
    pub(crate) w_h: Vec<f64>,
    /// Current hidden state.
    pub hidden: Vec<f64>,
}

impl AgentQNetwork {
    /// Create a randomly-initialised Q-network.
    pub fn new(obs_dim: usize, action_dim: usize, hidden_dim: usize, seed: u64) -> Result<Self> {
        if obs_dim == 0 || action_dim == 0 || hidden_dim == 0 {
            return Err(TensorError::invalid_argument(
                "AgentQNetwork dimensions must be > 0".into(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let scale1 = (2.0_f64 / obs_dim as f64).sqrt();
        let scale2 = (2.0_f64 / hidden_dim as f64).sqrt();
        let w1 = (0..hidden_dim * obs_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale1)
            .collect();
        let b1 = vec![0.0_f64; hidden_dim];
        let w2 = (0..action_dim * hidden_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale2)
            .collect();
        let b2 = vec![0.0_f64; action_dim];
        let w_h = (0..hidden_dim * hidden_dim)
            .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale2)
            .collect();
        Ok(Self {
            obs_dim,
            action_dim,
            hidden_dim,
            w1,
            b1,
            w2,
            b2,
            w_h,
            hidden: vec![0.0_f64; hidden_dim],
        })
    }

    /// Forward pass: returns Q-values for each action, updating hidden state.
    pub fn forward(&mut self, obs: &[f64]) -> Result<Vec<f64>> {
        if obs.len() != self.obs_dim {
            return Err(TensorError::invalid_argument(format!(
                "AgentQNetwork expected obs_dim={} got {}",
                self.obs_dim,
                obs.len()
            )));
        }
        // GRU-style hidden update: h_new = tanh(W1 * obs + Wh * h + b1)
        let mut h = vec![0.0_f64; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut val = self.b1[i];
            for j in 0..self.obs_dim {
                val += self.w1[i * self.obs_dim + j] * obs[j];
            }
            for j in 0..self.hidden_dim {
                val += self.w_h[i * self.hidden_dim + j] * self.hidden[j];
            }
            h[i] = val.tanh();
        }
        self.hidden = h.clone();
        // Output layer: Q = W2 * h + b2
        let mut q = vec![0.0_f64; self.action_dim];
        for i in 0..self.action_dim {
            let mut val = self.b2[i];
            for j in 0..self.hidden_dim {
                val += self.w2[i * self.hidden_dim + j] * h[j];
            }
            q[i] = val;
        }
        Ok(q)
    }

    /// Reset the recurrent hidden state.
    pub fn reset_hidden(&mut self) {
        self.hidden.fill(0.0);
    }
}

/// Hypernetwork-based mixing network enforcing monotonicity.
///
/// Given the global state `s`, two hypernetworks generate positive mixing
/// weights and biases so that the joint Q-value is monotonically non-decreasing
/// in each agent's Q-value.
///
/// Architecture:
/// - Hyper-W1: state → abs(W1) of shape (n_agents × embed_dim)
/// - Hyper-B1: state → B1 of shape (embed_dim,)
/// - Hyper-W2: state → abs(W2) of shape (embed_dim × 1)
/// - Hyper-B2: state → scalar bias via V-network
#[derive(Debug, Clone)]
pub struct MixingNetwork {
    pub(crate) n_agents: usize,
    pub(crate) state_dim: usize,
    pub(crate) embed_dim: usize,
    /// Hyper-W1 weights: (n_agents * embed_dim) × state_dim
    hyper_w1: Vec<f64>,
    hyper_b1: Vec<f64>,
    /// Hyper-W2 weights: embed_dim × state_dim
    hyper_w2: Vec<f64>,
    hyper_b2: Vec<f64>,
    /// V-network for final bias (2-layer MLP)
    v_w1: Vec<f64>,
    v_b1: Vec<f64>,
    v_w2: Vec<f64>,
    v_b2: Vec<f64>,
}

impl MixingNetwork {
    /// Construct with He initialisation.
    pub fn new(n_agents: usize, state_dim: usize, embed_dim: usize, seed: u64) -> Result<Self> {
        if n_agents == 0 || state_dim == 0 || embed_dim == 0 {
            return Err(TensorError::invalid_argument(
                "MixingNetwork dimensions must be > 0".into(),
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let s1 = (2.0_f64 / state_dim as f64).sqrt();
        let s2 = (2.0_f64 / embed_dim as f64).sqrt();
        let mut rand_vec = |n: usize, scale: f64| -> Vec<f64> {
            let mut r = StdRng::seed_from_u64(rng.random::<u64>());
            let v: Vec<f64> = (0..n)
                .map(|_| (r.random::<f64>() * 2.0 - 1.0) * scale)
                .collect();
            v
        };
        let hyper_w1 = rand_vec(n_agents * embed_dim * state_dim, s1);
        let hyper_b1 = rand_vec(embed_dim * state_dim, s1);
        let hyper_w2 = rand_vec(embed_dim * state_dim, s2);
        let hyper_b2 = rand_vec(state_dim, s1);
        // V-network: state → embed_dim → 1
        let v_w1 = rand_vec(embed_dim * state_dim, s1);
        let v_b1 = rand_vec(embed_dim, 0.0);
        let v_w2 = rand_vec(embed_dim, s2);
        let v_b2 = rand_vec(1, 0.0);
        Ok(Self {
            n_agents,
            state_dim,
            embed_dim,
            hyper_w1,
            hyper_b1,
            hyper_w2,
            hyper_b2,
            v_w1,
            v_b1,
            v_w2,
            v_b2,
        })
    }

    /// Forward pass: mix individual Q-values using global state.
    ///
    /// Monotonicity is guaranteed by taking abs(W) in each mixing layer.
    pub fn forward(&self, agent_qs: &[f64], state: &[f64]) -> Result<f64> {
        if agent_qs.len() != self.n_agents {
            return Err(TensorError::invalid_argument(format!(
                "MixingNetwork expected {} agent Q-values, got {}",
                self.n_agents,
                agent_qs.len()
            )));
        }
        if state.len() != self.state_dim {
            return Err(TensorError::invalid_argument(format!(
                "MixingNetwork expected state_dim={} got {}",
                self.state_dim,
                state.len()
            )));
        }
        // Layer 1 weights from hyper_w1: shape = (n_agents * embed_dim, state_dim)
        // We compute W1 * state → shape (n_agents * embed_dim,), then take abs
        let w1_size = self.n_agents * self.embed_dim;
        let mut w1 = vec![0.0_f64; w1_size];
        for i in 0..w1_size {
            for j in 0..self.state_dim {
                w1[i] += self.hyper_w1[i * self.state_dim + j] * state[j];
            }
            w1[i] = w1[i].abs();
        }
        // b1: shape (embed_dim,) from hyper_b1 * state
        let mut b1 = vec![0.0_f64; self.embed_dim];
        for i in 0..self.embed_dim {
            for j in 0..self.state_dim {
                b1[i] += self.hyper_b1[i * self.state_dim + j] * state[j];
            }
        }
        // hidden = elu(W1 @ agent_qs + b1), W1 is (embed_dim, n_agents)
        let mut hidden = vec![0.0_f64; self.embed_dim];
        for i in 0..self.embed_dim {
            let mut val = b1[i];
            for j in 0..self.n_agents {
                val += w1[i * self.n_agents + j] * agent_qs[j];
            }
            hidden[i] = if val >= 0.0 { val } else { val.exp() - 1.0 }; // ELU
        }
        // Layer 2 weights: shape (embed_dim,) from hyper_w2 * state, take abs
        let mut w2 = vec![0.0_f64; self.embed_dim];
        for i in 0..self.embed_dim {
            for j in 0..self.state_dim {
                w2[i] += self.hyper_w2[i * self.state_dim + j] * state[j];
            }
            w2[i] = w2[i].abs();
        }
        // scalar output before bias
        let mix: f64 = w2.iter().zip(hidden.iter()).map(|(a, b)| a * b).sum();
        // V-network bias
        let mut v_h = vec![0.0_f64; self.embed_dim];
        for i in 0..self.embed_dim {
            let mut val = self.v_b1[i];
            for j in 0..self.state_dim {
                val += self.v_w1[i * self.state_dim + j] * state[j];
            }
            v_h[i] = if val >= 0.0 { val } else { val.exp() - 1.0 };
        }
        let v_bias: f64 = self
            .v_w2
            .iter()
            .zip(v_h.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            + self.v_b2[0];
        Ok(mix + v_bias)
    }
}

/// Agent for QMIX: holds a local Q-network and uses ε-greedy exploration.
#[derive(Debug, Clone)]
pub struct QmixAgent {
    pub q_net: AgentQNetwork,
    pub epsilon: f64,
    agent_id: usize,
    rng_seed: u64,
}

impl QmixAgent {
    /// Create a QMIX agent.
    pub fn new(
        agent_id: usize,
        obs_dim: usize,
        action_dim: usize,
        hidden_dim: usize,
        epsilon: f64,
        seed: u64,
    ) -> Result<Self> {
        Ok(Self {
            q_net: AgentQNetwork::new(obs_dim, action_dim, hidden_dim, seed)?,
            epsilon,
            agent_id,
            rng_seed: seed,
        })
    }

    /// ε-greedy action selection.
    pub fn select_action(&mut self, obs: &[f64]) -> Result<usize> {
        let mut rng = StdRng::seed_from_u64(self.rng_seed);
        self.rng_seed = self.rng_seed.wrapping_add(1);
        if rng.random::<f64>() < self.epsilon {
            Ok(rng.random_range(0..self.q_net.action_dim))
        } else {
            let q_vals = self.q_net.forward(obs)?;
            let best = q_vals
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            Ok(best)
        }
    }
}

impl super::types::Agent for QmixAgent {
    fn act(&self, obs: &[f64]) -> Vec<f64> {
        let mut agent = self.clone();
        match agent.q_net.forward(obs) {
            Ok(q) => {
                let best = q
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                vec![best as f64]
            }
            Err(_) => vec![0.0],
        }
    }

    fn update(&mut self, _experience: &Experience) {
        // Weight update handled by QmixTrainer externally.
    }

    fn id(&self) -> usize {
        self.agent_id
    }
}

/// Trainer that jointly optimises agents via QMIX.
#[derive(Debug)]
pub struct QmixTrainer {
    pub agents: Vec<QmixAgent>,
    pub mixer: MixingNetwork,
    pub gamma: f64,
    pub(crate) step: u64,
}

impl QmixTrainer {
    /// Create from pre-built agents and mixer.
    pub fn new(agents: Vec<QmixAgent>, mixer: MixingNetwork, gamma: f64) -> Self {
        Self {
            agents,
            mixer,
            gamma,
            step: 0,
        }
    }

    /// Compute the QMIX TD loss for a batch of joint experiences.
    ///
    /// `joint_exps[t]` is a vector of one `Experience` per agent at time `t`.
    /// `states[t]` / `next_states[t]` are the global state vectors.
    pub fn compute_loss(
        &mut self,
        joint_exps: &[Vec<Experience>],
        states: &[Vec<f64>],
        next_states: &[Vec<f64>],
    ) -> Result<f64> {
        if joint_exps.len() != states.len() || states.len() != next_states.len() {
            return Err(TensorError::invalid_argument(
                "QmixTrainer: joint_exps, states, and next_states must have equal length".into(),
            ));
        }
        let mut total_loss = 0.0_f64;
        for (t, exps) in joint_exps.iter().enumerate() {
            // Current Q-values
            let mut qs_curr = Vec::with_capacity(self.agents.len());
            for (i, agent) in self.agents.iter_mut().enumerate() {
                let exp = exps.get(i).ok_or_else(|| {
                    TensorError::invalid_argument(format!(
                        "Missing experience for agent {i} at t={t}"
                    ))
                })?;
                let q_vals = agent.q_net.forward(&exp.obs)?;
                let action_idx = exp.action[0] as usize;
                let q = q_vals.get(action_idx).copied().unwrap_or(0.0);
                qs_curr.push(q);
            }
            let q_tot = self.mixer.forward(&qs_curr, &states[t])?;
            // Target Q-values (max over next actions)
            let team_reward: f64 = exps.iter().map(|e| e.reward).sum::<f64>() / exps.len() as f64;
            let done = exps.first().map(|e| e.done).unwrap_or(false);
            let mut qs_next = Vec::with_capacity(self.agents.len());
            for (i, agent) in self.agents.iter_mut().enumerate() {
                let exp = exps.get(i).ok_or_else(|| {
                    TensorError::invalid_argument(format!(
                        "Missing experience for agent {i} at t={t}"
                    ))
                })?;
                let q_next = agent.q_net.forward(&exp.next_obs)?;
                let max_q = q_next.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                qs_next.push(max_q);
            }
            let q_tot_next = self.mixer.forward(&qs_next, &next_states[t])?;
            let target = team_reward + if done { 0.0 } else { self.gamma * q_tot_next };
            total_loss += (q_tot - target).powi(2);
        }
        self.step += 1;
        Ok(total_loss / joint_exps.len() as f64)
    }
}
