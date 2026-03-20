//! Policy-gradient utilities for REINFORCE, PPO, and DQN-style algorithms.
//!
//! All functions operate on plain `f32` slices, making them independent of any
//! particular tensor backend.

use scirs2_core::random::Random;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// REINFORCE
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the REINFORCE policy-gradient loss.
///
/// L = -mean( log_prob_t * G_t )
///
/// A *lower* loss means higher log-probability of high-return actions.
///
/// # Arguments
/// * `log_probs` — log probabilities of the actions taken, shape `[batch]`.
/// * `returns`   — discounted returns G_t, shape `[batch]`.
///
/// # Errors
/// Returns an error if the slices have different lengths or are empty.
pub fn reinforce_loss(log_probs: &[f32], returns: &[f32]) -> Result<f32> {
    if log_probs.len() != returns.len() {
        return Err(TensorError::invalid_argument_op(
            "reinforce_loss",
            &format!(
                "log_probs length {} != returns length {}",
                log_probs.len(),
                returns.len()
            ),
        ));
    }
    if log_probs.is_empty() {
        return Err(TensorError::invalid_argument_op(
            "reinforce_loss",
            "log_probs and returns must not be empty",
        ));
    }

    let n = log_probs.len() as f32;
    let sum: f32 = log_probs
        .iter()
        .zip(returns.iter())
        .map(|(&lp, &g)| lp * g)
        .sum();

    // Negate to turn maximisation into minimisation.
    Ok(-sum / n)
}

// ─────────────────────────────────────────────────────────────────────────────
// Advantage normalisation
// ─────────────────────────────────────────────────────────────────────────────

/// Normalise a mutable advantage vector to zero mean and unit standard deviation.
///
/// If the standard deviation is below `1e-8` the advantages are left unchanged
/// to avoid division by near-zero.
pub fn normalize_advantages(advantages: &mut [f32]) {
    if advantages.is_empty() {
        return;
    }

    let n = advantages.len() as f32;
    let mean = advantages.iter().sum::<f32>() / n;
    let var = advantages.iter().map(|&a| (a - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();

    if std < 1e-8 {
        return;
    }

    for a in advantages.iter_mut() {
        *a = (*a - mean) / std;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Entropy
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Shannon entropy of a categorical probability distribution.
///
/// H = -Σ p_i · log(p_i)   (zero-probability bins are skipped)
///
/// # Arguments
/// * `probs` — probability distribution over actions; must sum to ≈ 1.
///
/// # Errors
/// Returns an error if `probs` is empty or contains a negative value.
pub fn categorical_entropy(probs: &[f32]) -> Result<f32> {
    if probs.is_empty() {
        return Err(TensorError::invalid_argument_op(
            "categorical_entropy",
            "probs must not be empty",
        ));
    }
    for &p in probs {
        if p < 0.0 {
            return Err(TensorError::invalid_argument_op(
                "categorical_entropy",
                "all probabilities must be non-negative",
            ));
        }
    }

    let entropy = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum::<f32>();

    Ok(entropy)
}

// ─────────────────────────────────────────────────────────────────────────────
// PPO clip objective
// ─────────────────────────────────────────────────────────────────────────────

/// Single-sample clipped PPO surrogate objective (clipped policy loss term).
///
/// L_CLIP(t) = min( ratio · A, clip(ratio, 1−ε, 1+ε) · A )
///
/// Note: this returns the *un-negated* value.  When using as a loss to
/// *minimise*, negate the mean of `ppo_clip_loss` values.
///
/// # Arguments
/// * `ratio`     — π_θ(a|s) / π_θ_old(a|s).
/// * `advantage` — GAE advantage estimate.
/// * `epsilon`   — clip range (e.g. 0.2).
pub fn ppo_clip_loss(ratio: f32, advantage: f32, epsilon: f32) -> f32 {
    let clipped_ratio = ratio.clamp(1.0 - epsilon, 1.0 + epsilon);
    f32::min(ratio * advantage, clipped_ratio * advantage)
}

// ─────────────────────────────────────────────────────────────────────────────
// PPO full loss
// ─────────────────────────────────────────────────────────────────────────────

/// Components of the combined PPO loss.
#[derive(Debug, Clone)]
pub struct PpoLossComponents {
    /// Mean clipped policy loss (to be minimised → typically negated surrogate).
    pub policy_loss: f32,
    /// Mean value-function MSE loss.
    pub value_loss: f32,
    /// Mean entropy bonus (to be maximised → subtracted from total).
    pub entropy_loss: f32,
    /// Combined scalar: policy_loss + value_coef · value_loss − entropy_bonus · entropy.
    pub total_loss: f32,
    /// Fraction of ratios that were clipped (diagnostic).
    pub clip_fraction: f32,
}

/// Compute the full batched PPO loss.
///
/// total_loss = policy_loss + value_coef · value_loss − entropy_bonus · mean(entropy)
///
/// # Arguments
/// * `ratios`        — π_new / π_old, shape `[batch]`.
/// * `advantages`    — GAE advantages, shape `[batch]`.
/// * `epsilon`       — PPO clip range.
/// * `value_preds`   — value-function predictions, shape `[batch]`.
/// * `value_targets` — TD-λ value targets, shape `[batch]`.
/// * `value_coef`    — scalar weight for the value loss.
/// * `entropy_bonus` — scalar weight for the entropy regularisation.
/// * `entropies`     — per-sample policy entropies, shape `[batch]`.
///
/// # Errors
/// Returns an error if any slice lengths are inconsistent or slices are empty.
pub fn ppo_loss(
    ratios: &[f32],
    advantages: &[f32],
    epsilon: f32,
    value_preds: &[f32],
    value_targets: &[f32],
    value_coef: f32,
    entropy_bonus: f32,
    entropies: &[f32],
) -> Result<PpoLossComponents> {
    let n = ratios.len();

    if n == 0 {
        return Err(TensorError::invalid_argument_op(
            "ppo_loss",
            "input slices must not be empty",
        ));
    }
    if advantages.len() != n {
        return Err(TensorError::invalid_argument_op(
            "ppo_loss",
            &format!(
                "advantages length {} != ratios length {}",
                advantages.len(),
                n
            ),
        ));
    }
    if value_preds.len() != n {
        return Err(TensorError::invalid_argument_op(
            "ppo_loss",
            &format!(
                "value_preds length {} != ratios length {}",
                value_preds.len(),
                n
            ),
        ));
    }
    if value_targets.len() != n {
        return Err(TensorError::invalid_argument_op(
            "ppo_loss",
            &format!(
                "value_targets length {} != ratios length {}",
                value_targets.len(),
                n
            ),
        ));
    }
    if entropies.len() != n {
        return Err(TensorError::invalid_argument_op(
            "ppo_loss",
            &format!(
                "entropies length {} != ratios length {}",
                entropies.len(),
                n
            ),
        ));
    }

    let nf = n as f32;

    // Policy loss (negate clip objective so it can be minimised).
    let mut clipped_count = 0_usize;
    let policy_sum: f32 = ratios
        .iter()
        .zip(advantages.iter())
        .map(|(&r, &a)| {
            let clipped_r = r.clamp(1.0 - epsilon, 1.0 + epsilon);
            if (r - clipped_r).abs() > 1e-7 {
                clipped_count += 1;
            }
            f32::min(r * a, clipped_r * a)
        })
        .sum();

    let policy_loss = -policy_sum / nf;

    // Value loss: MSE between predictions and targets.
    let value_sum: f32 = value_preds
        .iter()
        .zip(value_targets.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum();
    let value_loss = value_sum / nf;

    // Entropy (maximise → add negative to total).
    let entropy_mean: f32 = entropies.iter().sum::<f32>() / nf;
    let entropy_loss = entropy_mean;

    let total_loss = policy_loss + value_coef * value_loss - entropy_bonus * entropy_mean;
    let clip_fraction = clipped_count as f32 / nf;

    Ok(PpoLossComponents {
        policy_loss,
        value_loss,
        entropy_loss,
        total_loss,
        clip_fraction,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Action selection
// ─────────────────────────────────────────────────────────────────────────────

/// Epsilon-greedy action selection.
///
/// With probability `epsilon` a random action is selected; otherwise the
/// action with the highest Q-value is returned.
///
/// # Arguments
/// * `q_values` — Q(s, a) for each action.
/// * `epsilon`  — exploration probability ∈ [0, 1].
/// * `seed`     — RNG seed.
///
/// # Panics
/// Panics if `q_values` is empty.
pub fn epsilon_greedy(q_values: &[f32], epsilon: f32, seed: u64) -> usize {
    assert!(!q_values.is_empty(), "q_values must not be empty");

    let mut rng = Random::seed(seed);
    let p: f64 = rng.gen_range(0.0..1.0);

    if p < epsilon as f64 {
        // Random action.
        let idx = rng.gen_range(0.0..q_values.len() as f64) as usize;
        idx.min(q_values.len() - 1)
    } else {
        // Greedy action.
        q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// Boltzmann (softmax) action selection.
///
/// Actions are sampled proportionally to exp(Q(s,a) / temperature).
///
/// # Arguments
/// * `q_values`    — Q-values for each action.
/// * `temperature` — controls exploration (higher → more uniform).
/// * `seed`        — RNG seed.
///
/// # Errors
/// Returns an error if `q_values` is empty or `temperature ≤ 0`.
pub fn boltzmann_action(q_values: &[f32], temperature: f32, seed: u64) -> Result<usize> {
    if q_values.is_empty() {
        return Err(TensorError::invalid_argument_op(
            "boltzmann_action",
            "q_values must not be empty",
        ));
    }
    if temperature <= 0.0 {
        return Err(TensorError::invalid_argument_op(
            "boltzmann_action",
            "temperature must be strictly positive",
        ));
    }

    // Numerically stable softmax: subtract max before exponentiating.
    let max_q = q_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let exps: Vec<f32> = q_values
        .iter()
        .map(|&q| ((q - max_q) / temperature).exp())
        .collect();

    let total: f32 = exps.iter().sum();

    let mut rng = Random::seed(seed);
    let mut threshold = rng.gen_range(0.0..total as f64) as f32;

    for (i, &e) in exps.iter().enumerate() {
        threshold -= e;
        if threshold <= 0.0 {
            return Ok(i);
        }
    }

    // Fallback due to floating-point rounding.
    Ok(q_values.len() - 1)
}

// ─────────────────────────────────────────────────────────────────────────────
// DQN utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a single DQN TD target.
///
/// target = r + γ · max Q(s', ·) · (1 − done)
///
/// When `done == true`, the bootstrapped next-state value is zeroed out.
pub fn dqn_td_target(reward: f32, next_q_max: f32, gamma: f32, done: bool) -> f32 {
    let bootstrap = if done { 0.0 } else { gamma * next_q_max };
    reward + bootstrap
}

/// Compute mean-squared-error loss between Q-predictions and TD targets.
///
/// # Errors
/// Returns an error if the slices differ in length or are empty.
pub fn dqn_loss(q_predictions: &[f32], td_targets: &[f32]) -> Result<f32> {
    if q_predictions.len() != td_targets.len() {
        return Err(TensorError::invalid_argument_op(
            "dqn_loss",
            &format!(
                "q_predictions length {} != td_targets length {}",
                q_predictions.len(),
                td_targets.len()
            ),
        ));
    }
    if q_predictions.is_empty() {
        return Err(TensorError::invalid_argument_op(
            "dqn_loss",
            "q_predictions and td_targets must not be empty",
        ));
    }

    let n = q_predictions.len() as f32;
    let mse: f32 = q_predictions
        .iter()
        .zip(td_targets.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum::<f32>()
        / n;

    Ok(mse)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── reinforce_loss ────────────────────────────────────────────────────────

    #[test]
    fn test_reinforce_loss_basic() {
        // High return + high log_prob (closer to 0) → low (negative) loss means
        // the training signal is strong. Loss = -mean(log_prob * return).
        let log_probs = [-0.1_f32, -0.2]; // close to 0 → high probability
        let returns = [10.0_f32, 10.0];
        let loss = reinforce_loss(&log_probs, &returns).expect("valid inputs");
        // loss = -(-0.1*10 + -0.2*10)/2 = -((-1 + -2)/2) = -(-1.5) = 1.5
        assert!((loss - 1.5).abs() < 1e-5, "loss={}", loss);
    }

    #[test]
    fn test_reinforce_loss_sign_high_return_high_prob() {
        // Large log_prob magnitude (negative) + positive return → larger positive loss
        // than small magnitude + positive return → stronger gradient.
        let log_probs_hi = [-0.01_f32]; // nearly certain action
        let log_probs_lo = [-5.0_f32]; // very uncertain action
        let returns = [1.0_f32];

        let loss_hi = reinforce_loss(&log_probs_hi, &returns).expect("ok");
        let loss_lo = reinforce_loss(&log_probs_lo, &returns).expect("ok");

        // High-prob action: -(-0.01 * 1) / 1 = 0.01 (small gradient)
        // Low-prob action:  -(-5.0  * 1) / 1 = 5.0  (large gradient)
        assert!(
            loss_lo > loss_hi,
            "low-prob action should produce larger loss; lo={}, hi={}",
            loss_lo,
            loss_hi
        );
    }

    #[test]
    fn test_reinforce_loss_length_mismatch() {
        assert!(reinforce_loss(&[0.1], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_reinforce_loss_empty() {
        assert!(reinforce_loss(&[], &[]).is_err());
    }

    // ── normalize_advantages ──────────────────────────────────────────────────

    #[test]
    fn test_normalize_advantages_zero_mean_unit_std() {
        let mut adv = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        normalize_advantages(&mut adv);

        let n = adv.len() as f32;
        let mean = adv.iter().sum::<f32>() / n;
        let var = adv.iter().map(|&a| (a - mean).powi(2)).sum::<f32>() / n;
        let std = var.sqrt();

        assert!(mean.abs() < 1e-5, "mean not ≈ 0: {}", mean);
        assert!((std - 1.0).abs() < 1e-5, "std not ≈ 1: {}", std);
    }

    #[test]
    fn test_normalize_advantages_constant_unchanged() {
        // All-equal values → std ≈ 0, should not divide by zero.
        let mut adv = vec![3.0_f32; 5];
        normalize_advantages(&mut adv); // should not panic
                                        // Values remain unchanged because std < 1e-8.
        for &a in &adv {
            assert!((a - 3.0).abs() < 1e-5, "unexpected change: {}", a);
        }
    }

    #[test]
    fn test_normalize_advantages_empty_noop() {
        let mut adv: Vec<f32> = Vec::new();
        normalize_advantages(&mut adv); // should not panic
        assert!(adv.is_empty());
    }

    // ── categorical_entropy ───────────────────────────────────────────────────

    #[test]
    fn test_categorical_entropy_uniform_is_max() {
        // For n=4 uniform distribution, H = ln(4) ≈ 1.3863.
        let n = 4;
        let uniform = vec![0.25_f32; n];
        let h_uniform = categorical_entropy(&uniform).expect("valid");

        // One-hot distribution has H = 0.
        let mut one_hot = vec![0.0_f32; n];
        one_hot[0] = 1.0;
        let h_onehot = categorical_entropy(&one_hot).expect("valid");

        assert!(
            h_uniform > h_onehot,
            "uniform entropy should exceed one-hot"
        );
        assert!((h_onehot).abs() < 1e-6, "one-hot entropy should be 0");
        assert!(
            (h_uniform - (n as f32).ln()).abs() < 1e-5,
            "uniform entropy={}, expected ln(4)={}",
            h_uniform,
            (n as f32).ln()
        );
    }

    #[test]
    fn test_categorical_entropy_negative_prob_fails() {
        assert!(categorical_entropy(&[0.5, -0.1, 0.6]).is_err());
    }

    #[test]
    fn test_categorical_entropy_empty_fails() {
        assert!(categorical_entropy(&[]).is_err());
    }

    // ── ppo_clip_loss ─────────────────────────────────────────────────────────

    #[test]
    fn test_ppo_clip_loss_clips_at_epsilon() {
        // ratio = 1.5, advantage = 1.0, epsilon = 0.2
        // clipped_ratio = 1.2
        // min(1.5, 1.2) = 1.2
        let result = ppo_clip_loss(1.5, 1.0, 0.2);
        assert!((result - 1.2).abs() < 1e-6, "result={}", result);
    }

    #[test]
    fn test_ppo_clip_loss_unclipped_when_within_range() {
        // ratio = 1.1, advantage = 1.0, epsilon = 0.2
        // clipped_ratio = 1.1 (within [0.8, 1.2])
        // min(1.1, 1.1) = 1.1
        let result = ppo_clip_loss(1.1, 1.0, 0.2);
        assert!((result - 1.1).abs() < 1e-6, "result={}", result);
    }

    #[test]
    fn test_ppo_clip_loss_negative_advantage() {
        // ratio = 0.5, advantage = -1.0, epsilon = 0.2
        // clipped_ratio = 0.8
        // ratio*adv = -0.5, clipped*adv = -0.8
        // min(-0.5, -0.8) = -0.8
        let result = ppo_clip_loss(0.5, -1.0, 0.2);
        assert!((result - (-0.8)).abs() < 1e-6, "result={}", result);
    }

    // ── ppo_loss ──────────────────────────────────────────────────────────────

    #[test]
    fn test_ppo_loss_components_shapes() {
        let ratios = [1.0_f32, 1.1, 0.9, 1.2];
        let advantages = [0.5_f32, 0.3, -0.2, 0.8];
        let value_preds = [1.0_f32; 4];
        let value_targets = [1.5_f32; 4];
        let entropies = [0.6_f32; 4];

        let result = ppo_loss(
            &ratios,
            &advantages,
            0.2,
            &value_preds,
            &value_targets,
            0.5,
            0.01,
            &entropies,
        )
        .expect("valid inputs");

        // Value loss = MSE(1.0, 1.5) = 0.25.
        assert!(
            (result.value_loss - 0.25).abs() < 1e-5,
            "value_loss={}",
            result.value_loss
        );
        // Entropy loss == mean entropies.
        assert!(
            (result.entropy_loss - 0.6).abs() < 1e-5,
            "entropy_loss={}",
            result.entropy_loss
        );
        // Clip fraction: only ratio=1.2 is clipped (exceeds 1.2 = boundary; clip starts > 1.2).
        // ratio=1.2 is exactly at boundary — not clipped. Expect 0.
        assert_eq!(result.clip_fraction, 0.0);
    }

    #[test]
    fn test_ppo_loss_clip_fraction() {
        let ratios = [1.5_f32, 0.5, 1.0, 1.0];
        let advantages = [1.0_f32; 4];
        let value_preds = [0.0_f32; 4];
        let value_targets = [0.0_f32; 4];
        let entropies = [0.0_f32; 4];

        let result = ppo_loss(
            &ratios,
            &advantages,
            0.2,
            &value_preds,
            &value_targets,
            0.0,
            0.0,
            &entropies,
        )
        .expect("valid");
        // ratio=1.5 and ratio=0.5 should be clipped (outside [0.8, 1.2]).
        assert!(
            (result.clip_fraction - 0.5).abs() < 1e-5,
            "clip_fraction={}",
            result.clip_fraction
        );
    }

    // ── epsilon_greedy ────────────────────────────────────────────────────────

    #[test]
    fn test_epsilon_greedy_zero_epsilon_returns_argmax() {
        let q = [0.1_f32, 0.5, 0.3, 0.9, 0.2];
        // With epsilon=0 and any seed, the greedy action should always be 3.
        for seed in 0..10_u64 {
            let action = epsilon_greedy(&q, 0.0, seed);
            assert_eq!(action, 3, "seed={}", seed);
        }
    }

    #[test]
    fn test_epsilon_greedy_full_epsilon_is_random() {
        // With epsilon=1 all choices should be random; across many seeds we
        // expect to see more than one distinct action.
        let q = [0.1_f32, 0.9, 0.2];
        let actions: std::collections::HashSet<usize> =
            (0..30_u64).map(|s| epsilon_greedy(&q, 1.0, s)).collect();
        assert!(
            actions.len() > 1,
            "expected random diversity, got {:?}",
            actions
        );
    }

    // ── boltzmann_action ──────────────────────────────────────────────────────

    #[test]
    fn test_boltzmann_action_valid() {
        let q = [0.0_f32, 1.0, 2.0];
        let action = boltzmann_action(&q, 1.0, 42).expect("valid");
        assert!(action < 3);
    }

    #[test]
    fn test_boltzmann_action_zero_temperature_fails() {
        assert!(boltzmann_action(&[1.0, 2.0], 0.0, 0).is_err());
    }

    #[test]
    fn test_boltzmann_action_empty_fails() {
        assert!(boltzmann_action(&[], 1.0, 0).is_err());
    }

    // ── dqn_td_target ─────────────────────────────────────────────────────────

    #[test]
    fn test_dqn_td_target_non_terminal() {
        // target = 1.0 + 0.99 * 2.0 = 2.98
        let target = dqn_td_target(1.0, 2.0, 0.99, false);
        assert!((target - 2.98).abs() < 1e-5, "target={}", target);
    }

    #[test]
    fn test_dqn_td_target_terminal_ignores_next_q() {
        // When done=true the next-state value should be zeroed.
        let target = dqn_td_target(1.0, 999.0, 0.99, true);
        assert!((target - 1.0).abs() < 1e-5, "target={}", target);
    }

    // ── dqn_loss ──────────────────────────────────────────────────────────────

    #[test]
    fn test_dqn_loss_is_mse() {
        let preds = [2.0_f32, 4.0];
        let targets = [1.0_f32, 3.0];
        // MSE = ((2-1)^2 + (4-3)^2) / 2 = (1+1)/2 = 1.0
        let loss = dqn_loss(&preds, &targets).expect("valid");
        assert!((loss - 1.0).abs() < 1e-6, "loss={}", loss);
    }

    #[test]
    fn test_dqn_loss_perfect_predictions() {
        let v = [1.0_f32, 2.0, 3.0];
        let loss = dqn_loss(&v, &v).expect("valid");
        assert!(loss.abs() < 1e-7, "loss should be 0, got {}", loss);
    }

    #[test]
    fn test_dqn_loss_length_mismatch_fails() {
        assert!(dqn_loss(&[1.0, 2.0], &[1.0]).is_err());
    }

    #[test]
    fn test_dqn_loss_empty_fails() {
        assert!(dqn_loss(&[], &[]).is_err());
    }
}
