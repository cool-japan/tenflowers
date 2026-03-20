//! Deep Q-Network (DQN) variants and utilities.
//!
//! Provides configuration, action-selection, TD-target computation, Huber
//! loss, soft/hard parameter updates, and the Dueling-network Q-head.
//!
//! Supported variants:
//! - Standard DQN (Mnih et al., 2015)
//! - Double DQN (DDQN) (van Hasselt et al., 2016)
//! - Dueling Network Architectures (Wang et al., 2016)

use scirs2_core::random::Random;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// QConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameter configuration for a DQN-family agent.
#[derive(Debug, Clone)]
pub struct QConfig {
    /// Learning rate for the online Q-network.
    pub learning_rate: f32,
    /// Discount factor γ ∈ [0, 1].
    pub gamma: f32,
    /// Starting ε for ε-greedy exploration.
    pub epsilon_start: f32,
    /// Terminal ε after decay.
    pub epsilon_end: f32,
    /// Number of steps over which ε is linearly decayed.
    pub epsilon_decay: u64,
    /// How many steps between hard target-network copies (when `tau == 1`).
    pub target_update_freq: u64,
    /// Mini-batch size for each gradient update.
    pub batch_size: usize,
    /// Soft-update coefficient τ (1.0 → hard copy, 0.0 → no update).
    pub tau: f32,
    /// Whether to use Double-DQN action selection.
    pub double_dqn: bool,
    /// Whether to use a Dueling network architecture.
    pub dueling: bool,
}

impl Default for QConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay: 10_000,
            target_update_freq: 1_000,
            batch_size: 32,
            tau: 1.0,
            double_dqn: false,
            dueling: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DqnAgent
// ─────────────────────────────────────────────────────────────────────────────

/// Stateful DQN agent that tracks exploration decay and episode counts.
///
/// The agent does **not** own or update neural-network parameters — it
/// provides pure functions that operate on `&[f32]` Q-value slices,
/// matching the TenfloweRS convention of separating parameter storage from
/// algorithmic logic.
#[derive(Debug, Clone)]
pub struct DqnAgent {
    /// Hyper-parameters.
    pub config: QConfig,
    /// Total number of environment steps taken.
    pub step_count: u64,
    /// Total number of episodes completed.
    pub episode_count: u64,
}

impl DqnAgent {
    /// Create a new `DqnAgent` with the given configuration.
    pub fn new(config: QConfig) -> Self {
        Self {
            config,
            step_count: 0,
            episode_count: 0,
        }
    }

    /// Compute the current exploration probability at the given step.
    ///
    /// Uses linear annealing:
    /// ε(t) = ε_end + (ε_start − ε_end) · max(0, 1 − t / decay_steps)
    pub fn epsilon(&self, step: u64) -> f32 {
        let frac = if self.config.epsilon_decay == 0 {
            1.0_f32
        } else {
            (step as f32 / self.config.epsilon_decay as f32).min(1.0)
        };
        self.config.epsilon_end
            + (self.config.epsilon_start - self.config.epsilon_end) * (1.0 - frac)
    }

    /// ε-greedy action selection with per-step decayed exploration.
    ///
    /// With probability `ε(step_count)` a uniformly random action is returned;
    /// otherwise the action with the highest Q-value is selected.
    ///
    /// # Errors
    /// Returns an error if `q_values` is empty.
    pub fn epsilon_greedy(&self, q_values: &[f32], seed: u64) -> Result<usize> {
        if q_values.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "DqnAgent::epsilon_greedy",
                "q_values must not be empty",
            ));
        }

        let eps = self.epsilon(self.step_count);
        let mut rng = Random::seed(seed);
        let p: f64 = rng.gen_range(0.0..1.0);

        if p < eps as f64 {
            let idx = rng.gen_range(0.0..q_values.len() as f64) as usize;
            Ok(idx.min(q_values.len() - 1))
        } else {
            // Greedy: argmax Q.
            let best = q_values
                .iter()
                .enumerate()
                .fold((0_usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                })
                .0;
            Ok(best)
        }
    }

    /// Compute the standard DQN Bellman target.
    ///
    /// target = r + γ · max Q(s', ·) · (1 − done)
    ///
    /// When `done == true` the next-state bootstrap is zeroed out.
    pub fn compute_td_target(&self, reward: f32, next_q_values: &[f32], done: bool) -> Result<f32> {
        if next_q_values.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "DqnAgent::compute_td_target",
                "next_q_values must not be empty",
            ));
        }
        let max_next_q = next_q_values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let bootstrap = if done {
            0.0
        } else {
            self.config.gamma * max_next_q
        };
        Ok(reward + bootstrap)
    }

    /// Compute the Double-DQN (DDQN) Bellman target.
    ///
    /// DDQN decorrelates action selection from action evaluation:
    /// 1. Select the best action using the **online** network:
    ///    `a* = argmax_a Q_online(s', a)`
    /// 2. Evaluate it with the **target** network:
    ///    `target = r + γ · Q_target(s', a*) · (1 − done)`
    ///
    /// # Errors
    /// Returns an error if either Q-value slice is empty or has different lengths.
    pub fn compute_double_td_target(
        &self,
        reward: f32,
        next_q_values_online: &[f32],
        next_q_values_target: &[f32],
        done: bool,
    ) -> Result<f32> {
        if next_q_values_online.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "DqnAgent::compute_double_td_target",
                "next_q_values_online must not be empty",
            ));
        }
        if next_q_values_online.len() != next_q_values_target.len() {
            return Err(TensorError::invalid_argument_op(
                "DqnAgent::compute_double_td_target",
                &format!(
                    "online Q length {} != target Q length {}",
                    next_q_values_online.len(),
                    next_q_values_target.len()
                ),
            ));
        }

        // Action selection with online network.
        let best_action = next_q_values_online
            .iter()
            .enumerate()
            .fold((0_usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            })
            .0;

        // Evaluation with target network.
        let target_q = next_q_values_target[best_action];
        let bootstrap = if done {
            0.0
        } else {
            self.config.gamma * target_q
        };
        Ok(reward + bootstrap)
    }

    /// Huber (smooth-L1) loss between a predicted Q-value and its TD target.
    ///
    /// L(δ) = ½δ² if |δ| ≤ 1,  else |δ| − ½
    ///
    /// Averaged over a batch.
    ///
    /// # Errors
    /// Returns an error if the slices are empty or have different lengths.
    pub fn td_loss(&self, predicted_q: &[f32], target_q: &[f32]) -> Result<f32> {
        if predicted_q.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "DqnAgent::td_loss",
                "predicted_q must not be empty",
            ));
        }
        if predicted_q.len() != target_q.len() {
            return Err(TensorError::invalid_argument_op(
                "DqnAgent::td_loss",
                &format!(
                    "predicted_q length {} != target_q length {}",
                    predicted_q.len(),
                    target_q.len()
                ),
            ));
        }
        let n = predicted_q.len() as f32;
        let loss: f32 = predicted_q
            .iter()
            .zip(target_q.iter())
            .map(|(&p, &t)| {
                let delta = (p - t).abs();
                if delta <= 1.0 {
                    0.5 * delta * delta
                } else {
                    delta - 0.5
                }
            })
            .sum::<f32>()
            / n;
        Ok(loss)
    }

    /// Soft-update target network parameters towards online network parameters.
    ///
    /// θ_target ← τ·θ_online + (1−τ)·θ_target
    ///
    /// Returns a new `Vec<f32>` of updated target parameters.
    ///
    /// # Errors
    /// Returns an error if the parameter slices have different lengths.
    pub fn soft_update_params(
        &self,
        online_params: &[f32],
        target_params: &[f32],
    ) -> Result<Vec<f32>> {
        if online_params.len() != target_params.len() {
            return Err(TensorError::invalid_argument_op(
                "DqnAgent::soft_update_params",
                &format!(
                    "online_params length {} != target_params length {}",
                    online_params.len(),
                    target_params.len()
                ),
            ));
        }
        let tau = self.config.tau;
        let updated: Vec<f32> = online_params
            .iter()
            .zip(target_params.iter())
            .map(|(&o, &t)| tau * o + (1.0 - tau) * t)
            .collect();
        Ok(updated)
    }

    /// Check whether a hard target-network update is due at the given step.
    ///
    /// Returns `true` when `step` is a positive multiple of `target_update_freq`.
    pub fn hard_update_needed(&self, step: u64) -> bool {
        self.config.target_update_freq > 0 && step > 0 && step % self.config.target_update_freq == 0
    }

    /// Increment the step counter and return the new value.
    pub fn step(&mut self) -> u64 {
        self.step_count += 1;
        self.step_count
    }

    /// Increment the episode counter and return the new value.
    pub fn episode_done(&mut self) -> u64 {
        self.episode_count += 1;
        self.episode_count
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DuelingHeads
// ─────────────────────────────────────────────────────────────────────────────

/// Dueling network output head.
///
/// Combines a scalar value stream V(s) and a per-action advantage stream
/// A(s, a) into Q-values using the mean-centering identity:
///
/// Q(s, a) = V(s) + A(s, a) − mean_a'[A(s, a')]
///
/// This avoids the unidentifiability that arises when V and A are estimated
/// independently.
///
/// Reference: Wang et al., "Dueling Network Architectures for Deep Reinforcement
/// Learning", ICML 2016.
#[derive(Debug, Clone)]
pub struct DuelingHeads;

impl DuelingHeads {
    /// Create a new `DuelingHeads` combinator (stateless).
    pub fn new() -> Self {
        Self
    }

    /// Combine `value` and `advantages` into Q-values.
    ///
    /// Q(s, a) = V(s) + A(s, a) − mean(A(s, ·))
    ///
    /// # Errors
    /// Returns an error if `advantages` is empty.
    pub fn forward(&self, value: f32, advantages: &[f32]) -> Result<Vec<f32>> {
        if advantages.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "DuelingHeads::forward",
                "advantages must not be empty",
            ));
        }
        let mean_adv = advantages.iter().sum::<f32>() / advantages.len() as f32;
        let q: Vec<f32> = advantages.iter().map(|&a| value + a - mean_adv).collect();
        Ok(q)
    }
}

impl Default for DuelingHeads {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_agent() -> DqnAgent {
        DqnAgent::new(QConfig::default())
    }

    // ── epsilon decay ─────────────────────────────────────────────────────────

    #[test]
    fn test_epsilon_at_step_zero_is_start() {
        let agent = default_agent(); // epsilon_start=1.0
        assert!((agent.epsilon(0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_epsilon_at_full_decay_is_end() {
        let agent = default_agent(); // epsilon_end=0.05, decay=10_000
        let eps = agent.epsilon(10_000);
        assert!((eps - 0.05).abs() < 1e-6, "eps={}", eps);
    }

    #[test]
    fn test_epsilon_beyond_decay_stays_at_end() {
        let agent = default_agent();
        assert!((agent.epsilon(99_999) - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_epsilon_midpoint() {
        // At step 5000 (half of 10_000): eps = 0.05 + 0.95*0.5 = 0.525
        let agent = default_agent();
        let eps = agent.epsilon(5_000);
        assert!((eps - 0.525).abs() < 1e-5, "eps={}", eps);
    }

    // ── epsilon_greedy ────────────────────────────────────────────────────────

    #[test]
    fn test_epsilon_greedy_zero_eps_is_argmax() {
        let mut agent = default_agent();
        // Force step past decay so eps ≈ 0.05; then also ensure via agent config.
        agent.config.epsilon_start = 0.0;
        agent.config.epsilon_end = 0.0;
        let q = [0.1_f32, 0.9, 0.3];
        for seed in 0..20_u64 {
            let action = agent.epsilon_greedy(&q, seed).expect("valid");
            assert_eq!(action, 1, "seed={seed}");
        }
    }

    #[test]
    fn test_epsilon_greedy_full_eps_is_random() {
        let mut agent = default_agent();
        agent.config.epsilon_start = 1.0;
        agent.config.epsilon_end = 1.0;
        let q = [1.0_f32, 0.0, 0.0, 0.0];
        let actions: std::collections::HashSet<usize> = (0..50_u64)
            .map(|s| agent.epsilon_greedy(&q, s).expect("valid"))
            .collect();
        assert!(actions.len() > 1, "expected diversity, got {:?}", actions);
    }

    #[test]
    fn test_epsilon_greedy_empty_q_fails() {
        let agent = default_agent();
        assert!(agent.epsilon_greedy(&[], 0).is_err());
    }

    // ── compute_td_target ─────────────────────────────────────────────────────

    #[test]
    fn test_td_target_non_terminal() {
        let agent = default_agent(); // gamma=0.99
                                     // r=1, max_next_q=2: target = 1 + 0.99*2 = 2.98
        let target = agent
            .compute_td_target(1.0, &[1.0, 2.0, 0.5], false)
            .expect("ok");
        assert!((target - 2.98).abs() < 1e-5, "target={}", target);
    }

    #[test]
    fn test_td_target_terminal_ignores_next() {
        let agent = default_agent();
        let target = agent.compute_td_target(1.0, &[999.0], true).expect("ok");
        assert!((target - 1.0).abs() < 1e-5, "target={}", target);
    }

    #[test]
    fn test_td_target_empty_fails() {
        let agent = default_agent();
        assert!(agent.compute_td_target(1.0, &[], false).is_err());
    }

    // ── compute_double_td_target ──────────────────────────────────────────────

    #[test]
    fn test_double_td_target_differs_from_standard() {
        let agent = default_agent();
        // Online Q: [0.1, 5.0, 0.3] → argmax=1
        // Target Q: [0.2, 0.8, 0.9]  → target Q at action 1 = 0.8
        // standard target = r + 0.99 * max([0.2, 0.8, 0.9]) = 1 + 0.99*0.9 = 1.891
        // double  target = r + 0.99 * 0.8 = 1 + 0.792 = 1.792
        let online = [0.1_f32, 5.0, 0.3];
        let target_net = [0.2_f32, 0.8, 0.9];
        let ddqn = agent
            .compute_double_td_target(1.0, &online, &target_net, false)
            .expect("ok");
        let standard = agent
            .compute_td_target(1.0, &target_net, false)
            .expect("ok");
        assert!((ddqn - 1.792).abs() < 1e-4, "ddqn={}", ddqn);
        assert!((standard - 1.891).abs() < 1e-4, "standard={}", standard);
        // They must differ.
        assert!((ddqn - standard).abs() > 1e-3);
    }

    #[test]
    fn test_double_td_target_done_zeroes_bootstrap() {
        let agent = default_agent();
        let result = agent
            .compute_double_td_target(2.0, &[10.0], &[10.0], true)
            .expect("ok");
        assert!((result - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_double_td_target_length_mismatch_fails() {
        let agent = default_agent();
        assert!(agent
            .compute_double_td_target(1.0, &[1.0, 2.0], &[1.0], false)
            .is_err());
    }

    // ── td_loss (Huber) ───────────────────────────────────────────────────────

    #[test]
    fn test_td_loss_small_delta_is_quadratic() {
        // |δ| ≤ 1: L = ½δ²
        let agent = default_agent();
        let pred = [1.5_f32];
        let tgt = [1.0_f32]; // δ = 0.5
        let loss = agent.td_loss(&pred, &tgt).expect("ok");
        assert!((loss - 0.125).abs() < 1e-6, "loss={}", loss);
    }

    #[test]
    fn test_td_loss_large_delta_is_linear() {
        // |δ| > 1: L = |δ| - 0.5
        let agent = default_agent();
        let pred = [3.0_f32]; // δ = 2.0
        let tgt = [1.0_f32];
        let loss = agent.td_loss(&pred, &tgt).expect("ok");
        assert!((loss - 1.5).abs() < 1e-6, "loss={}", loss);
    }

    #[test]
    fn test_td_loss_perfect_prediction_is_zero() {
        let agent = default_agent();
        let v = [1.0_f32, 2.0, 3.0];
        let loss = agent.td_loss(&v, &v).expect("ok");
        assert!(loss.abs() < 1e-7);
    }

    #[test]
    fn test_td_loss_empty_fails() {
        let agent = default_agent();
        assert!(agent.td_loss(&[], &[]).is_err());
    }

    #[test]
    fn test_td_loss_length_mismatch_fails() {
        let agent = default_agent();
        assert!(agent.td_loss(&[1.0, 2.0], &[1.0]).is_err());
    }

    // ── soft_update_params ────────────────────────────────────────────────────

    #[test]
    fn test_soft_update_tau1_full_copy() {
        let cfg = QConfig {
            tau: 1.0,
            ..QConfig::default()
        };
        let agent = DqnAgent::new(cfg);
        let online = [1.0_f32, 2.0, 3.0];
        let target = [0.0_f32, 0.0, 0.0];
        let result = agent.soft_update_params(&online, &target).expect("ok");
        for (&r, &o) in result.iter().zip(online.iter()) {
            assert!((r - o).abs() < 1e-6);
        }
    }

    #[test]
    fn test_soft_update_tau0_unchanged() {
        let cfg = QConfig {
            tau: 0.0,
            ..QConfig::default()
        };
        let agent = DqnAgent::new(cfg);
        let online = [1.0_f32, 2.0];
        let target = [5.0_f32, 6.0];
        let result = agent.soft_update_params(&online, &target).expect("ok");
        assert!((result[0] - 5.0).abs() < 1e-6);
        assert!((result[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_soft_update_midpoint() {
        let cfg = QConfig {
            tau: 0.5,
            ..QConfig::default()
        };
        let agent = DqnAgent::new(cfg);
        let online = [2.0_f32];
        let target = [0.0_f32];
        let result = agent.soft_update_params(&online, &target).expect("ok");
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_soft_update_length_mismatch_fails() {
        let agent = default_agent();
        assert!(agent.soft_update_params(&[1.0, 2.0], &[1.0]).is_err());
    }

    // ── hard_update_needed ────────────────────────────────────────────────────

    #[test]
    fn test_hard_update_needed_at_freq() {
        let agent = default_agent(); // target_update_freq=1000
        assert!(agent.hard_update_needed(1_000));
        assert!(agent.hard_update_needed(2_000));
        assert!(!agent.hard_update_needed(999));
        assert!(!agent.hard_update_needed(1_001));
    }

    #[test]
    fn test_hard_update_needed_step_zero_is_false() {
        let agent = default_agent();
        assert!(!agent.hard_update_needed(0));
    }

    // ── DuelingHeads ──────────────────────────────────────────────────────────

    #[test]
    fn test_dueling_heads_q_equals_v_plus_a_minus_mean() {
        let head = DuelingHeads::new();
        let value = 2.0_f32;
        let advantages = [0.0_f32, 1.0, -1.0]; // mean = 0
        let q = head.forward(value, &advantages).expect("ok");
        // Q = V + A - mean(A) = 2 + A - 0
        assert!((q[0] - 2.0).abs() < 1e-6);
        assert!((q[1] - 3.0).abs() < 1e-6);
        assert!((q[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dueling_heads_constant_advantages_all_equal_to_v() {
        // A(s,a) = c for all a → Q(s,a) = V(s) + c − c = V(s)
        let head = DuelingHeads::new();
        let value = 3.0_f32;
        let advantages = [5.0_f32; 4];
        let q = head.forward(value, &advantages).expect("ok");
        for &qi in &q {
            assert!((qi - value).abs() < 1e-6, "qi={}", qi);
        }
    }

    #[test]
    fn test_dueling_heads_empty_advantages_fails() {
        let head = DuelingHeads::new();
        assert!(head.forward(1.0, &[]).is_err());
    }

    #[test]
    fn test_dueling_heads_single_action() {
        // Single action: mean(A) == A → Q = V
        let head = DuelingHeads::new();
        let q = head.forward(4.0, &[7.0]).expect("ok");
        assert_eq!(q.len(), 1);
        assert!((q[0] - 4.0).abs() < 1e-6);
    }

    // ── step / episode_done counters ──────────────────────────────────────────

    #[test]
    fn test_step_increments_counter() {
        let mut agent = default_agent();
        assert_eq!(agent.step(), 1);
        assert_eq!(agent.step(), 2);
        assert_eq!(agent.step_count, 2);
    }

    #[test]
    fn test_episode_done_increments_counter() {
        let mut agent = default_agent();
        assert_eq!(agent.episode_done(), 1);
        assert_eq!(agent.episode_done(), 2);
        assert_eq!(agent.episode_count, 2);
    }
}
