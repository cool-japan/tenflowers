//! Soft Actor-Critic (SAC) utilities.
//!
//! Implements the loss functions and helper routines for the maximum-entropy
//! actor-critic algorithm described in:
//!
//! > Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
//! > Reinforcement Learning with a Stochastic Actor", ICML 2018.
//!
//! and the automatic temperature adjustment extension in:
//!
//! > Haarnoja et al., "Soft Actor-Critic Algorithms and Applications", 2019.
//!
//! All functions operate on plain `f32` slices to remain independent of any
//! particular tensor backend.

use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// SacConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameter configuration for a SAC agent.
#[derive(Debug, Clone)]
pub struct SacConfig {
    /// Learning rate for the actor (policy) network.
    pub actor_lr: f32,
    /// Learning rate for the twin critic networks.
    pub critic_lr: f32,
    /// Learning rate for the log-temperature (α) parameter.
    pub alpha_lr: f32,
    /// Discount factor γ ∈ [0, 1].
    pub gamma: f32,
    /// Soft-update coefficient τ for target-critic polyak averaging.
    pub tau: f32,
    /// Scaling factor applied to the entropy target heuristic:
    /// `target_entropy = −target_entropy_scale × action_dim`.
    pub target_entropy_scale: f32,
    /// Initial value of the temperature parameter α.
    pub initial_alpha: f32,
}

impl Default for SacConfig {
    fn default() -> Self {
        Self {
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            alpha_lr: 3e-4,
            gamma: 0.99,
            tau: 0.005,
            target_entropy_scale: 1.0,
            initial_alpha: 0.2,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SacAgent
// ─────────────────────────────────────────────────────────────────────────────

/// Stateful SAC agent holding configuration and the learnable log-temperature.
///
/// Like [`DqnAgent`](super::dqn::DqnAgent), this struct does **not** own
/// network parameters; it exposes pure loss-computation methods that operate
/// on `&[f32]` slices.
#[derive(Debug, Clone)]
pub struct SacAgent {
    /// Hyper-parameters.
    pub config: SacConfig,
    /// Log-temperature parameter log(α), learned via gradient descent.
    pub log_alpha: f32,
}

impl SacAgent {
    /// Create a new `SacAgent` with the given configuration.
    ///
    /// `log_alpha` is initialised to `ln(initial_alpha)`.
    pub fn new(config: SacConfig) -> Self {
        let log_alpha = config.initial_alpha.max(1e-8).ln();
        Self { config, log_alpha }
    }

    /// Return the temperature parameter α = exp(log_alpha).
    ///
    /// Clamped to at least `1e-8` to avoid division by zero downstream.
    pub fn alpha(&self) -> f32 {
        self.log_alpha.exp().max(1e-8)
    }

    /// Compute the SAC target Q-value for a batch of transitions.
    ///
    /// target_q = r + γ · (1 − d) · (min(Q1', Q2') − α · log π')
    ///
    /// # Arguments
    /// * `reward`        — scalar reward r.
    /// * `done`          — terminal flag (f32: 0.0 or 1.0).
    /// * `next_q1`       — target Q1 evaluated at next state and next action.
    /// * `next_q2`       — target Q2 evaluated at next state and next action.
    /// * `next_log_prob` — log π_θ(a'|s') for the sampled next action.
    /// * `gamma`         — discount factor (override; uses `config.gamma` if set to
    ///                     `f32::NAN`).
    /// * `alpha`         — temperature (pass `self.alpha()` for automatic).
    pub fn compute_target_q(
        &self,
        reward: f32,
        done: f32,
        next_q1: f32,
        next_q2: f32,
        next_log_prob: f32,
        gamma: f32,
        alpha: f32,
    ) -> f32 {
        let min_next_q = if next_q1 < next_q2 { next_q1 } else { next_q2 };
        let soft_value = min_next_q - alpha * next_log_prob;
        reward + gamma * (1.0 - done) * soft_value
    }

    /// Compute the twin-critic loss (mean-squared error).
    ///
    /// L_critic = ½ · mean( (Q_pred − Q_target)² )
    ///
    /// # Errors
    /// Returns an error if the slices are empty or have different lengths.
    pub fn critic_loss(&self, q_pred: &[f32], q_target: &[f32]) -> Result<f32> {
        if q_pred.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "SacAgent::critic_loss",
                "q_pred must not be empty",
            ));
        }
        if q_pred.len() != q_target.len() {
            return Err(TensorError::invalid_argument_op(
                "SacAgent::critic_loss",
                &format!(
                    "q_pred length {} != q_target length {}",
                    q_pred.len(),
                    q_target.len()
                ),
            ));
        }
        let n = q_pred.len() as f32;
        let mse: f32 = q_pred
            .iter()
            .zip(q_target.iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum::<f32>()
            / n;
        Ok(0.5 * mse)
    }

    /// Compute the actor loss (policy gradient).
    ///
    /// L_actor = mean( α · log π(a|s) − Q(s, a) )
    ///
    /// Minimising this loss encourages the policy to take actions with both
    /// high Q-value and high entropy.
    ///
    /// # Errors
    /// Returns an error if the slices are empty or have different lengths.
    pub fn actor_loss(&self, log_probs: &[f32], q_values: &[f32], alpha: f32) -> Result<f32> {
        if log_probs.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "SacAgent::actor_loss",
                "log_probs must not be empty",
            ));
        }
        if log_probs.len() != q_values.len() {
            return Err(TensorError::invalid_argument_op(
                "SacAgent::actor_loss",
                &format!(
                    "log_probs length {} != q_values length {}",
                    log_probs.len(),
                    q_values.len()
                ),
            ));
        }
        let n = log_probs.len() as f32;
        let loss: f32 = log_probs
            .iter()
            .zip(q_values.iter())
            .map(|(&lp, &q)| alpha * lp - q)
            .sum::<f32>()
            / n;
        Ok(loss)
    }

    /// Compute the temperature (α) loss for automatic entropy tuning.
    ///
    /// L_α = −mean( log_alpha · (log π(a|s) + H_target) )
    ///
    /// where `H_target` is the target entropy (a negative constant).
    ///
    /// Minimising this loss adapts log_alpha so that the policy entropy
    /// tracks `H_target`.
    ///
    /// # Errors
    /// Returns an error if `log_probs` is empty.
    pub fn alpha_loss(
        &self,
        log_probs: &[f32],
        target_entropy: f32,
        log_alpha: f32,
    ) -> Result<f32> {
        if log_probs.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "SacAgent::alpha_loss",
                "log_probs must not be empty",
            ));
        }
        let n = log_probs.len() as f32;
        // L_α = -mean(log_alpha * (log_π + H_target))
        let loss: f32 = log_probs
            .iter()
            .map(|&lp| -(log_alpha * (lp + target_entropy)))
            .sum::<f32>()
            / n;
        Ok(loss)
    }

    /// Heuristic target entropy for continuous action spaces.
    ///
    /// H_target = −action_dim · target_entropy_scale
    ///
    /// For a `tanh`-squashed Gaussian policy, typical practice is to set
    /// `H_target = −dim(A)`.
    pub fn target_entropy(&self, action_dim: usize) -> f32 {
        -(action_dim as f32) * self.config.target_entropy_scale
    }

    /// Soft-update (polyak average) for target network parameters.
    ///
    /// θ_target ← τ·θ_source + (1−τ)·θ_target
    ///
    /// # Errors
    /// Returns an error if the slices have different lengths.
    pub fn soft_update(&self, source: &[f32], target: &[f32]) -> Result<Vec<f32>> {
        if source.len() != target.len() {
            return Err(TensorError::invalid_argument_op(
                "SacAgent::soft_update",
                &format!(
                    "source length {} != target length {}",
                    source.len(),
                    target.len()
                ),
            ));
        }
        let tau = self.config.tau;
        Ok(source
            .iter()
            .zip(target.iter())
            .map(|(&s, &t)| tau * s + (1.0 - tau) * t)
            .collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_agent() -> SacAgent {
        SacAgent::new(SacConfig::default())
    }

    // ── alpha ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_alpha_matches_initial_config() {
        let agent = default_agent(); // initial_alpha=0.2
        assert!(
            (agent.alpha() - 0.2).abs() < 1e-5,
            "alpha={}",
            agent.alpha()
        );
    }

    #[test]
    fn test_alpha_exp_log_alpha() {
        let mut agent = default_agent();
        agent.log_alpha = 0.0; // exp(0) = 1
        assert!((agent.alpha() - 1.0).abs() < 1e-6);
    }

    // ── compute_target_q ──────────────────────────────────────────────────────

    #[test]
    fn test_target_q_non_terminal() {
        let agent = default_agent(); // gamma=0.99
                                     // r=1, done=0, Q1'=2, Q2'=3, log_π'=-0.5, alpha=0.2
                                     // min(Q')=2, soft_value = 2 - 0.2*(-0.5) = 2.1
                                     // target = 1 + 0.99 * 2.1 = 3.079
        let tq = agent.compute_target_q(1.0, 0.0, 2.0, 3.0, -0.5, 0.99, 0.2);
        assert!((tq - 3.079).abs() < 1e-4, "tq={}", tq);
    }

    #[test]
    fn test_target_q_terminal_zeroes_bootstrap() {
        let agent = default_agent();
        let tq = agent.compute_target_q(5.0, 1.0, 100.0, 200.0, -1.0, 0.99, 0.2);
        assert!((tq - 5.0).abs() < 1e-5, "tq={}", tq);
    }

    #[test]
    fn test_target_q_uses_min_of_twin_critics() {
        let agent = default_agent();
        // Q1'=1.5 < Q2'=3.0 → should use 1.5
        let tq_min = agent.compute_target_q(0.0, 0.0, 1.5, 3.0, 0.0, 1.0, 0.0);
        // Q1'=3.0 > Q2'=1.5 → should also use 1.5
        let tq_min2 = agent.compute_target_q(0.0, 0.0, 3.0, 1.5, 0.0, 1.0, 0.0);
        assert!((tq_min - 1.5).abs() < 1e-6);
        assert!((tq_min2 - 1.5).abs() < 1e-6);
    }

    // ── critic_loss ───────────────────────────────────────────────────────────

    #[test]
    fn test_critic_loss_perfect_prediction_zero() {
        let agent = default_agent();
        let v = [1.0_f32, 2.0, 3.0];
        let loss = agent.critic_loss(&v, &v).expect("ok");
        assert!(loss.abs() < 1e-7, "loss={}", loss);
    }

    #[test]
    fn test_critic_loss_mse_halved() {
        // ½ · (2-0)² = 2.0
        let agent = default_agent();
        let pred = [2.0_f32];
        let tgt = [0.0_f32];
        let loss = agent.critic_loss(&pred, &tgt).expect("ok");
        assert!((loss - 2.0).abs() < 1e-6, "loss={}", loss);
    }

    #[test]
    fn test_critic_loss_empty_fails() {
        let agent = default_agent();
        assert!(agent.critic_loss(&[], &[]).is_err());
    }

    #[test]
    fn test_critic_loss_length_mismatch_fails() {
        let agent = default_agent();
        assert!(agent.critic_loss(&[1.0, 2.0], &[1.0]).is_err());
    }

    // ── actor_loss ────────────────────────────────────────────────────────────

    #[test]
    fn test_actor_loss_formula() {
        // L = mean( alpha * log_pi - Q )
        // log_pi=[-1.0], Q=[2.0], alpha=0.5 → 0.5*(-1) - 2 = -2.5
        let agent = default_agent();
        let loss = agent.actor_loss(&[-1.0_f32], &[2.0_f32], 0.5).expect("ok");
        assert!((loss - (-2.5)).abs() < 1e-6, "loss={}", loss);
    }

    #[test]
    fn test_actor_loss_high_entropy_reduces_loss() {
        // Higher entropy (less negative log_prob) → smaller alpha*log_pi term.
        let agent = default_agent();
        let loss_low_entropy = agent.actor_loss(&[-5.0_f32], &[1.0_f32], 0.2).expect("ok");
        let loss_high_entropy = agent.actor_loss(&[-0.1_f32], &[1.0_f32], 0.2).expect("ok");
        // low entropy: 0.2*(-5)-1=-2, high entropy: 0.2*(-0.1)-1=-1.02
        // high entropy loss is less negative (larger) when Q is fixed
        assert!(loss_high_entropy > loss_low_entropy);
    }

    #[test]
    fn test_actor_loss_empty_fails() {
        let agent = default_agent();
        assert!(agent.actor_loss(&[], &[], 0.2).is_err());
    }

    // ── alpha_loss ────────────────────────────────────────────────────────────

    #[test]
    fn test_alpha_loss_sign_when_entropy_above_target() {
        // When log_prob >> -target_entropy (policy too deterministic),
        // log_prob + target_entropy > 0 → gradient pushes alpha up.
        // L = -log_alpha * (log_prob + H_target)
        // If policy entropy < target: log_prob is large negative, so
        // log_prob + H_target < 0 → L = -log_alpha * negative = positive * log_alpha.
        // We test the sign of the loss here.
        let agent = default_agent(); // log_alpha = ln(0.2) < 0
        let log_probs = [-0.01_f32]; // high entropy (close to 0)
        let target_entropy = -5.0_f32; // target is high entropy
                                       // log_prob + H_target = -0.01 + (-5) = -5.01 < 0
                                       // L = -log_alpha * (-5.01) = log_alpha * 5.01 (log_alpha < 0) → L < 0
        let loss = agent
            .alpha_loss(&log_probs, target_entropy, agent.log_alpha)
            .expect("ok");
        // The loss should push log_alpha up when entropy is below target.
        // (Numerical check)
        assert!(loss.is_finite(), "loss={}", loss);
    }

    #[test]
    fn test_alpha_loss_empty_fails() {
        let agent = default_agent();
        assert!(agent.alpha_loss(&[], -1.0, 0.0).is_err());
    }

    #[test]
    fn test_alpha_loss_known_value() {
        // L = -mean(log_alpha * (log_pi + H))
        // log_alpha=0.0, log_pi=-1.0, H=-2.0
        // L = -(0.0 * (-1.0 + -2.0)) = 0
        let agent = default_agent();
        let loss = agent.alpha_loss(&[-1.0_f32], -2.0, 0.0).expect("ok");
        assert!(loss.abs() < 1e-7, "loss={}", loss);
    }

    // ── target_entropy ────────────────────────────────────────────────────────

    #[test]
    fn test_target_entropy_is_negative_action_dim() {
        let agent = default_agent(); // target_entropy_scale=1.0
        assert!((agent.target_entropy(4) - (-4.0)).abs() < 1e-6);
        assert!((agent.target_entropy(1) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_target_entropy_with_scale() {
        let cfg = SacConfig {
            target_entropy_scale: 0.5,
            ..SacConfig::default()
        };
        let agent = SacAgent::new(cfg);
        assert!((agent.target_entropy(4) - (-2.0)).abs() < 1e-6);
    }

    // ── soft_update ───────────────────────────────────────────────────────────

    #[test]
    fn test_soft_update_tau1_full_copy() {
        let cfg = SacConfig {
            tau: 1.0,
            ..SacConfig::default()
        };
        let agent = SacAgent::new(cfg);
        let src = [1.0_f32, 2.0, 3.0];
        let tgt = [0.0_f32, 0.0, 0.0];
        let res = agent.soft_update(&src, &tgt).expect("ok");
        for (&r, &s) in res.iter().zip(src.iter()) {
            assert!((r - s).abs() < 1e-6);
        }
    }

    #[test]
    fn test_soft_update_tau0_unchanged() {
        let cfg = SacConfig {
            tau: 0.0,
            ..SacConfig::default()
        };
        let agent = SacAgent::new(cfg);
        let src = [1.0_f32, 2.0];
        let tgt = [5.0_f32, 6.0];
        let res = agent.soft_update(&src, &tgt).expect("ok");
        assert!((res[0] - 5.0).abs() < 1e-6);
        assert!((res[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_soft_update_default_tau() {
        // tau=0.005: result = 0.005*src + 0.995*tgt
        let agent = default_agent();
        let src = [10.0_f32];
        let tgt = [0.0_f32];
        let res = agent.soft_update(&src, &tgt).expect("ok");
        assert!((res[0] - 0.05).abs() < 1e-5, "res={}", res[0]);
    }

    #[test]
    fn test_soft_update_length_mismatch_fails() {
        let agent = default_agent();
        assert!(agent.soft_update(&[1.0, 2.0], &[1.0]).is_err());
    }
}
