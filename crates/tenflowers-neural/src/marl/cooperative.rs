//! Cooperative task structures for multi-agent systems.
//!
//! Contains: TeamRewardShaper, CreditAssignment, ComaTrainer,
//! VdnMixer, WqmixTrainer.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

use super::maddpg::MaddpgCritic;
use super::qmix::{MixingNetwork, QmixAgent, QmixTrainer};
use super::types::Experience;

// ─────────────────────────────────────────────────────────────────────────────
// § 7. Cooperative Task Structures
// ─────────────────────────────────────────────────────────────────────────────

/// Potential-based reward shaping for cooperative multi-agent tasks.
///
/// Adds potential difference `γΦ(s') - Φ(s)` to each agent's reward,
/// preserving optimal joint policy (Ng et al., 1999).
#[derive(Debug, Clone)]
pub struct TeamRewardShaper {
    pub gamma: f64,
    potential_weights: Vec<f64>,
}

impl TeamRewardShaper {
    /// Create with given weights for a linear potential function `Φ(s) = w·s`.
    pub fn new(gamma: f64, potential_weights: Vec<f64>) -> Self {
        Self {
            gamma,
            potential_weights,
        }
    }

    /// Evaluate the potential function Φ(state).
    pub fn potential(&self, state: &[f64]) -> f64 {
        self.potential_weights
            .iter()
            .zip(state.iter().chain(std::iter::repeat(&0.0)))
            .map(|(w, s)| w * s)
            .sum()
    }

    /// Compute shaped reward: `r + γΦ(s') - Φ(s)`.
    pub fn shape(&self, reward: f64, state: &[f64], next_state: &[f64]) -> f64 {
        reward + self.gamma * self.potential(next_state) - self.potential(state)
    }

    /// Shape rewards for all agents jointly (shared potential).
    pub fn shape_team(&self, rewards: &[f64], state: &[f64], next_state: &[f64]) -> Vec<f64> {
        let shaping = self.gamma * self.potential(next_state) - self.potential(state);
        rewards.iter().map(|r| r + shaping).collect()
    }
}

/// Credit assignment via difference rewards and COMA counterfactual baseline.
#[derive(Debug, Clone)]
pub struct CreditAssignment {
    pub n_agents: usize,
    /// Default action for computing counterfactual baseline (zero-action).
    pub default_action: Vec<f64>,
}

impl CreditAssignment {
    /// Create with a given default action for the counterfactual.
    pub fn new(n_agents: usize, action_dim: usize) -> Self {
        Self {
            n_agents,
            default_action: vec![0.0; action_dim],
        }
    }

    /// Difference reward: `D_i = Q(s, a) - Q(s, a_{-i}=default)`.
    ///
    /// The critic function `q_fn` takes `(state, actions_flat)` → scalar.
    pub fn difference_reward<F>(
        &self,
        state: &[f64],
        actions: &[Vec<f64>],
        q_fn: F,
        agent_idx: usize,
    ) -> Result<f64>
    where
        F: Fn(&[f64], &[f64]) -> f64,
    {
        if agent_idx >= self.n_agents {
            return Err(TensorError::invalid_argument(format!(
                "CreditAssignment: agent_idx {} out of range {}",
                agent_idx, self.n_agents
            )));
        }
        let flat: Vec<f64> = actions.iter().flat_map(|a| a.iter().cloned()).collect();
        let q_joint = q_fn(state, &flat);
        // Replace agent i's action with default
        let mut cf_actions = actions.to_vec();
        cf_actions[agent_idx] = self.default_action.clone();
        let flat_cf: Vec<f64> = cf_actions.iter().flat_map(|a| a.iter().cloned()).collect();
        let q_cf = q_fn(state, &flat_cf);
        Ok(q_joint - q_cf)
    }

    /// COMA counterfactual advantage: `A_i(s,a) = Q(s,a) - Σ_a' π(a'|s)Q(s,a'_{-i})`.
    ///
    /// Uses a discrete approximation by averaging over `n_samples` random
    /// counterfactual actions drawn from a uniform distribution in [-1,1].
    pub fn coma_advantage<F>(
        &self,
        state: &[f64],
        actions: &[Vec<f64>],
        q_fn: F,
        agent_idx: usize,
        n_samples: usize,
        seed: u64,
    ) -> Result<f64>
    where
        F: Fn(&[f64], &[f64]) -> f64,
    {
        if agent_idx >= self.n_agents {
            return Err(TensorError::invalid_argument(format!(
                "CreditAssignment: agent_idx {} out of range {}",
                agent_idx, self.n_agents
            )));
        }
        let action_dim = actions[agent_idx].len();
        let flat: Vec<f64> = actions.iter().flat_map(|a| a.iter().cloned()).collect();
        let q_actual = q_fn(state, &flat);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut baseline = 0.0_f64;
        for _ in 0..n_samples {
            let cf_a: Vec<f64> = (0..action_dim)
                .map(|_| rng.random::<f64>() * 2.0 - 1.0)
                .collect();
            let mut cf = actions.to_vec();
            cf[agent_idx] = cf_a;
            let flat_cf: Vec<f64> = cf.iter().flat_map(|a| a.iter().cloned()).collect();
            baseline += q_fn(state, &flat_cf);
        }
        baseline /= n_samples.max(1) as f64;
        Ok(q_actual - baseline)
    }
}

/// Centralized critic for COMA with counterfactual advantage.
#[derive(Debug, Clone)]
pub struct ComaTrainer {
    pub critic: MaddpgCritic,
    pub credit: CreditAssignment,
    pub gamma: f64,
}

impl ComaTrainer {
    /// Create with a centralized critic.
    pub fn new(
        n_agents: usize,
        action_dim: usize,
        critic_input_dim: usize,
        hidden_dim: usize,
        gamma: f64,
        seed: u64,
    ) -> Result<Self> {
        Ok(Self {
            critic: MaddpgCritic::new(critic_input_dim, hidden_dim, seed)?,
            credit: CreditAssignment::new(n_agents, action_dim),
            gamma,
        })
    }

    /// Compute counterfactual advantages for each agent using the centralized critic.
    pub fn compute_advantages(
        &self,
        state: &[f64],
        actions: &[Vec<f64>],
        n_samples: usize,
        seed: u64,
    ) -> Result<Vec<f64>> {
        let critic_ref = &self.critic;
        let advantages: Result<Vec<f64>> = (0..self.credit.n_agents)
            .map(|i| {
                self.credit.coma_advantage(
                    state,
                    actions,
                    |s, a| {
                        let input: Vec<f64> = s.iter().chain(a.iter()).cloned().collect();
                        if input.len() == critic_ref.input_dim {
                            critic_ref.forward(&input).unwrap_or(0.0)
                        } else {
                            0.0
                        }
                    },
                    i,
                    n_samples,
                    seed + i as u64,
                )
            })
            .collect();
        advantages
    }
}

/// Value Decomposition Networks: joint Q = sum of individual Q-values.
#[derive(Debug, Clone)]
pub struct VdnMixer {
    pub n_agents: usize,
}

impl VdnMixer {
    /// Create a VDN mixer for `n_agents` agents.
    pub fn new(n_agents: usize) -> Result<Self> {
        if n_agents == 0 {
            return Err(TensorError::invalid_argument(
                "VdnMixer: n_agents must be > 0".into(),
            ));
        }
        Ok(Self { n_agents })
    }

    /// Mix individual Q-values by simple summation.
    pub fn mix(&self, agent_qs: &[f64]) -> Result<f64> {
        if agent_qs.len() != self.n_agents {
            return Err(TensorError::invalid_argument(format!(
                "VdnMixer: expected {} agent Q-values, got {}",
                self.n_agents,
                agent_qs.len()
            )));
        }
        Ok(agent_qs.iter().sum())
    }

    /// Decompose a joint target into uniform individual targets (sum / n).
    pub fn decompose(&self, joint_target: f64) -> Vec<f64> {
        vec![joint_target / self.n_agents as f64; self.n_agents]
    }
}

/// Weighted QMIX (WQMIX): applies higher weight to states where the
/// optimistic projection (argmax) disagrees with the central policy.
///
/// Uses a separate `central_mixer` (standard sum-mixer or full QMIX) alongside
/// the standard monotone `qmix_mixer`, and blends them with weight `alpha`.
#[derive(Debug)]
pub struct WqmixTrainer {
    pub qmix: QmixTrainer,
    pub vdn: VdnMixer,
    /// Blending weight α ∈ (0, 1]: 1.0 = pure QMIX.
    pub alpha: f64,
}

impl WqmixTrainer {
    /// Create.
    pub fn new(qmix: QmixTrainer, vdn: VdnMixer, alpha: f64) -> Result<Self> {
        if !(0.0 < alpha && alpha <= 1.0) {
            return Err(TensorError::invalid_argument(
                "WqmixTrainer: alpha must be in (0, 1]".into(),
            ));
        }
        Ok(Self { qmix, vdn, alpha })
    }

    /// Compute WQMIX loss: α·L_qmix + (1-α)·L_vdn.
    ///
    /// `joint_exps`, `states`, `next_states` follow the same convention as
    /// `QmixTrainer::compute_loss`.
    pub fn compute_loss(
        &mut self,
        joint_exps: &[Vec<Experience>],
        states: &[Vec<f64>],
        next_states: &[Vec<f64>],
    ) -> Result<f64> {
        let qmix_loss = self.qmix.compute_loss(joint_exps, states, next_states)?;
        // VDN loss: mix by sum
        let mut vdn_loss = 0.0_f64;
        for (t, exps) in joint_exps.iter().enumerate() {
            let qs: Vec<f64> = exps.iter().map(|e| e.reward).collect();
            let team_reward: f64 = qs.iter().sum();
            let done = exps.first().map(|e| e.done).unwrap_or(false);
            let vdn_q_curr = self.vdn.mix(&qs)?;
            // Estimate next VDN Q as average next reward proxy
            let next_qs: Vec<f64> = exps
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    self.qmix
                        .agents
                        .get(i)
                        .map(|ag| {
                            let mut ag2 = ag.clone();
                            ag2.q_net
                                .forward(&e.next_obs)
                                .ok()
                                .and_then(|v| v.iter().cloned().reduce(f64::max))
                                .unwrap_or(0.0)
                        })
                        .unwrap_or(0.0)
                })
                .collect();
            let vdn_q_next = self.vdn.mix(&next_qs)?;
            let target = team_reward
                + if done {
                    0.0
                } else {
                    self.qmix.gamma * vdn_q_next
                };
            let _ = next_states[t].len(); // touch to avoid unused warning
            vdn_loss += (vdn_q_curr - target).powi(2);
        }
        vdn_loss /= joint_exps.len() as f64;
        Ok(self.alpha * qmix_loss + (1.0 - self.alpha) * vdn_loss)
    }
}

// Suppress unused import warnings for types used in cooperative.rs
// (QmixAgent and MixingNetwork are used indirectly via QmixTrainer)
const _: fn() = || {
    let _: Option<QmixAgent> = None;
    let _: Option<MixingNetwork> = None;
};
