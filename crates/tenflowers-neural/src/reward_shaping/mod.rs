//! # Advanced Reward Modeling & Intrinsic Motivation (Round 45 Track D)
//!
//! Implements a comprehensive suite of reward shaping and intrinsic motivation
//! algorithms for reinforcement learning and RLHF pipelines.
//!
//! ## Algorithms
//!
//! | Struct | Reference |
//! |--------|-----------|
//! | [`RsPotentialBasedShaping`] | Ng, Harada & Russell (1999) — policy-invariant shaping |
//! | [`RsRewardDecomposition`] | Juozapaitis et al. (2019) — interpretable decomposition |
//! | [`RsBradleyTerryModel`] | Bradley-Terry / Christiano et al. (2017) RLHF |
//! | [`RsCuriosityModule`] | ICM — Pathak et al. (2017) |
//! | [`RsRandomNetworkDistillation`] | RND — Burda et al. (2018) |
//! | [`RsEmpowermentIntrinsic`] | Salge et al. (2014) — channel capacity |
//! | [`RsGoalConditionedReward`] | UVFA — Schaul et al. (2015) + HER |
//! | [`RsRewardEnsemble`] | Ensemble uncertainty for active reward learning |
//! | [`RsRetroactiveLearning`] | HER — Andrychowicz et al. (2017) |
//! | [`RsMetrics`] | Evaluation utilities |
//!
//! All RNG uses `scirs2_core::random`; no `rand` crate dependency.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::fmt;

#[cfg(test)]
mod tests;

// ─────────────────────────────────────────────────────────────────────────────
// §0 Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors specific to reward-shaping operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RsError {
    /// Mismatched or illegal vector dimensions.
    InvalidDimension(String),
    /// Numerical instability (NaN, Inf, singular matrix, …).
    NumericalError(String),
    /// Bad hyperparameter configuration.
    ConfigError(String),
}

impl fmt::Display for RsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RsError::InvalidDimension(msg) => write!(f, "RsError::InvalidDimension: {msg}"),
            RsError::NumericalError(msg) => write!(f, "RsError::NumericalError: {msg}"),
            RsError::ConfigError(msg) => write!(f, "RsError::ConfigError: {msg}"),
        }
    }
}

impl std::error::Error for RsError {}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable sigmoid.
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// ReLU activation.
#[inline]
fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Dot product of two equal-length slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |acc, (&x, &y)| acc + x * y)
}

/// L2 norm of a slice.
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |acc, &x| acc + x * x).sqrt()
}

/// L2 squared distance between two equal-length slices.
fn l2_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).fold(0.0_f64, |acc, (&x, &y)| {
        let d = x - y;
        acc + d * d
    })
}

/// Xavier-uniform initialisation for a 2-D weight matrix.
/// Returns a flat row-major Vec of shape `[rows × cols]`.
fn xavier_init(rows: usize, cols: usize, seed: u64, rng_offset: u64) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(rng_offset));
    let limit = (6.0_f64 / (rows + cols) as f64).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    let u: f64 = rng.random();
                    u * 2.0 * limit - limit
                })
                .collect()
        })
        .collect()
}

/// Apply a linear layer with bias, no activation.
fn linear_fwd(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    w.iter()
        .zip(b.iter())
        .map(|(row, &bias)| dot(row, x) + bias)
        .collect()
}

/// Apply a linear layer followed by ReLU.
fn linear_relu(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    w.iter()
        .zip(b.iter())
        .map(|(row, &bias)| relu(dot(row, x) + bias))
        .collect()
}

/// Softmax in-place over a Vec.
fn softmax_vec(v: &mut [f64]) {
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0_f64;
    for x in v.iter_mut() {
        *x = (*x - max_v).exp();
        sum += *x;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Shannon entropy of a probability distribution.
fn entropy(p: &[f64]) -> f64 {
    p.iter().fold(
        0.0_f64,
        |acc, &pi| {
            if pi > 1e-15 {
                acc - pi * pi.ln()
            } else {
                acc
            }
        },
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  RsPotentialBasedShaping
// ─────────────────────────────────────────────────────────────────────────────

/// Potential-based reward shaping (Ng, Harada & Russell 1999).
///
/// The shaped reward is:
/// ```text
/// r'(s, a, s') = r(s, a, s') + γ · Φ(s') − Φ(s)
/// ```
/// where Φ is a linear potential: `Φ(s) = w · φ(s)`.
///
/// This shaping is provably policy-invariant: any optimal policy under the
/// original reward is also optimal under the shaped reward.
#[derive(Debug, Clone)]
pub struct RsPotentialBasedShaping {
    /// Discount factor for the shaping term.
    pub gamma: f64,
    /// Linear weights for the potential function.
    potential_weights: Vec<f64>,
    /// Dimension of the state feature vector.
    feature_dim: usize,
}

impl RsPotentialBasedShaping {
    /// Create a new shaping module with zero-initialised potential weights.
    pub fn new(feature_dim: usize, gamma: f64) -> Self {
        Self {
            gamma,
            potential_weights: vec![0.0; feature_dim],
            feature_dim,
        }
    }

    /// Compute the linear potential `Φ(s) = w · φ(s)`.
    pub fn potential(&self, state_features: &[f64]) -> f64 {
        dot(
            &self.potential_weights,
            &state_features[..self.feature_dim.min(state_features.len())],
        )
    }

    /// Compute the shaped reward:
    /// `r'(s, a, s') = r + γ · Φ(s') − Φ(s)`.
    pub fn shaped_reward(&self, reward: f64, state: &[f64], next_state: &[f64]) -> f64 {
        let phi_s = self.potential(state);
        let phi_sp = self.potential(next_state);
        reward + self.gamma * phi_sp - phi_s
    }

    /// Gradient step on the potential weights:
    /// `w ← w − lr · (Φ(s) − target) · φ(s)`.
    pub fn update_potential(&mut self, state: &[f64], target: f64, lr: f64) {
        let phi = self.potential(state);
        let err = phi - target;
        let len = self.feature_dim.min(state.len());
        for i in 0..len {
            self.potential_weights[i] -= lr * err * state[i];
        }
    }

    /// Return the current potential for a state (useful for verifying
    /// Φ(terminal) ≈ 0 after training).
    pub fn zero_shaping_guarantee(&self, state: &[f64]) -> f64 {
        self.potential(state)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  RsRewardDecomposition
// ─────────────────────────────────────────────────────────────────────────────

/// A single interpretable reward component.
#[derive(Debug, Clone)]
pub struct RsRewardComponent {
    /// Human-readable name (e.g. "safety", "efficiency").
    pub name: String,
    /// Combination weight `w_k`.
    pub weight: f64,
    /// Per-component scale factor (applied before weighting).
    pub scale: f64,
}

/// Interpretable reward decomposition (Juozapaitis et al. 2019).
///
/// The total reward is `r(s, s') = Σ_k w_k · r_k(s, s')` where
/// `r_k(s, s') = scale_k · (extractor_k · (s' − s))`.
#[derive(Debug, Clone)]
pub struct RsRewardDecomposition {
    components: Vec<RsRewardComponent>,
    /// One linear extractor per component, each of length `feature_dim`.
    feature_extractors: Vec<Vec<f64>>,
    feature_dim: usize,
}

impl RsRewardDecomposition {
    /// Create an empty decomposition for a given feature dimension.
    pub fn new(feature_dim: usize) -> Self {
        Self {
            components: Vec::new(),
            feature_extractors: Vec::new(),
            feature_dim,
        }
    }

    /// Register a named reward component with its extractor vector.
    pub fn add_component(&mut self, name: &str, weight: f64, extractor: Vec<f64>) {
        self.components.push(RsRewardComponent {
            name: name.to_owned(),
            weight,
            scale: 1.0,
        });
        // Truncate or zero-pad extractor to feature_dim
        let mut ext = extractor;
        ext.resize(self.feature_dim, 0.0);
        self.feature_extractors.push(ext);
    }

    /// Compute the reward for each component given transition `(s, s')`.
    ///
    /// Component reward: `r_k = scale_k * extractor_k · (s' - s)`.
    pub fn compute(&self, state: &[f64], next_state: &[f64]) -> Vec<f64> {
        let dim = self.feature_dim.min(state.len()).min(next_state.len());
        self.feature_extractors
            .iter()
            .zip(self.components.iter())
            .map(|(ext, comp)| {
                let delta: f64 =
                    (0..dim).fold(0.0, |acc, i| acc + ext[i] * (next_state[i] - state[i]));
                comp.scale * delta
            })
            .collect()
    }

    /// Total weighted reward `Σ_k w_k · r_k(s, s')`.
    pub fn total_reward(&self, state: &[f64], next_state: &[f64]) -> f64 {
        let raw = self.compute(state, next_state);
        raw.iter()
            .zip(self.components.iter())
            .fold(0.0_f64, |acc, (&rk, comp)| acc + comp.weight * rk)
    }

    /// Normalised absolute weights `|w_k| / Σ |w_j|` for interpretability.
    pub fn importance_weights(&self) -> Vec<f64> {
        let abs_sum: f64 = self.components.iter().map(|c| c.weight.abs()).sum();
        if abs_sum < 1e-15 {
            return vec![0.0; self.components.len()];
        }
        self.components
            .iter()
            .map(|c| c.weight.abs() / abs_sum)
            .collect()
    }

    /// Return named `(name, contribution)` pairs for a single transition.
    pub fn credit_assignment(&self, state: &[f64], next_state: &[f64]) -> Vec<(String, f64)> {
        let raw = self.compute(state, next_state);
        self.components
            .iter()
            .zip(raw.iter())
            .map(|(comp, &rk)| (comp.name.clone(), comp.weight * rk))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  RsBradleyTerryModel
// ─────────────────────────────────────────────────────────────────────────────

/// Linear Bradley-Terry reward model for learning from human preferences
/// (Christiano et al. 2017, RLHF).
///
/// Models `P(τ_i > τ_j) = σ(r(τ_i) − r(τ_j))` where
/// `r(features) = w · features`.
#[derive(Debug, Clone)]
pub struct RsBradleyTerryModel {
    /// Linear reward weights.
    reward_weights: Vec<f64>,
    feature_dim: usize,
}

impl RsBradleyTerryModel {
    /// Initialise with zero weights.
    pub fn new(feature_dim: usize) -> Self {
        Self {
            reward_weights: vec![0.0; feature_dim],
            feature_dim,
        }
    }

    /// Scalar reward `r = w · features`.
    pub fn reward(&self, features: &[f64]) -> f64 {
        dot(
            &self.reward_weights,
            &features[..self.feature_dim.min(features.len())],
        )
    }

    /// `P(i ≻ j) = σ(r(i) − r(j))`.
    pub fn preference_probability(&self, feat_i: &[f64], feat_j: &[f64]) -> f64 {
        sigmoid(self.reward(feat_i) - self.reward(feat_j))
    }

    /// One MLE gradient step maximising `log P(winner ≻ loser)`:
    ///
    /// `∇w = (1 − P(win ≻ lose)) · (φ_win − φ_lose)`
    pub fn update_from_comparison(&mut self, feat_winner: &[f64], feat_loser: &[f64], lr: f64) {
        let p = self.preference_probability(feat_winner, feat_loser);
        let scale = lr * (1.0 - p);
        let len = self
            .feature_dim
            .min(feat_winner.len())
            .min(feat_loser.len());
        for i in 0..len {
            self.reward_weights[i] += scale * (feat_winner[i] - feat_loser[i]);
        }
    }

    /// Train on a batch of `(winner_feat, loser_feat)` pairs for `epochs` epochs.
    /// Returns a Vec of per-epoch average cross-entropy loss.
    pub fn train_batch(
        &mut self,
        comparisons: &[(Vec<f64>, Vec<f64>)],
        lr: f64,
        epochs: usize,
    ) -> Vec<f64> {
        let mut losses = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            for (w, l) in comparisons {
                self.update_from_comparison(w, l, lr);
            }
            losses.push(self.cross_entropy_loss(comparisons));
        }
        losses
    }

    /// Average binary cross-entropy over all comparisons:
    /// `−(1/N) Σ log P(winner_i ≻ loser_i)`.
    pub fn cross_entropy_loss(&self, comparisons: &[(Vec<f64>, Vec<f64>)]) -> f64 {
        if comparisons.is_empty() {
            return 0.0;
        }
        const EPS: f64 = 1e-12;
        let sum: f64 = comparisons
            .iter()
            .map(|(w, l)| {
                let p = self.preference_probability(w, l).max(EPS);
                -p.ln()
            })
            .sum();
        sum / comparisons.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  RsCuriosityModule  (ICM — Pathak et al. 2017)
// ─────────────────────────────────────────────────────────────────────────────

/// Intrinsic Curiosity Module (ICM).
///
/// Architecture:
/// - **Encoder**: `φ(s) = ReLU(W_e · s + b_e)` — maps state to features.
/// - **Forward model**: `φ̂(s') = f(φ(s), a_onehot)` — predicts next features.
/// - **Inverse model**: `â = g(φ(s), φ(s'))` — predicts action from features.
///
/// Intrinsic reward: `r_i = (η/2) · ‖φ̂(s') − φ(s')‖²`.
#[derive(Debug, Clone)]
pub struct RsCuriosityModule {
    // Encoder weights [feature_dim × state_dim]
    encoder_w: Vec<Vec<f64>>,
    encoder_b: Vec<f64>,
    // Forward model [feature_dim × (feature_dim + n_actions)]
    forward_w: Vec<Vec<f64>>,
    forward_b: Vec<f64>,
    // Inverse model [n_actions × (2 * feature_dim)]
    inverse_w: Vec<Vec<f64>>,
    inverse_b: Vec<f64>,
    /// State space dimension.
    pub state_dim: usize,
    /// Internal feature space dimension.
    pub feature_dim: usize,
    /// Number of discrete actions.
    pub n_actions: usize,
    /// Weight balancing forward vs inverse loss: `L = (1-β)L_inv + β L_fwd`.
    pub beta: f64,
    /// Intrinsic reward scaling coefficient.
    pub eta: f64,
}

impl RsCuriosityModule {
    /// Construct an ICM with Xavier-initialised networks.
    pub fn new(
        state_dim: usize,
        feature_dim: usize,
        n_actions: usize,
        beta: f64,
        eta: f64,
    ) -> Self {
        let encoder_w = xavier_init(feature_dim, state_dim, 42, 0);
        let encoder_b = vec![0.0; feature_dim];
        let forward_w = xavier_init(feature_dim, feature_dim + n_actions, 42, 1);
        let forward_b = vec![0.0; feature_dim];
        let inverse_w = xavier_init(n_actions, 2 * feature_dim, 42, 2);
        let inverse_b = vec![0.0; n_actions];
        Self {
            encoder_w,
            encoder_b,
            forward_w,
            forward_b,
            inverse_w,
            inverse_b,
            state_dim,
            feature_dim,
            n_actions,
            beta,
            eta,
        }
    }

    /// Encode a state vector to internal features: `ReLU(W_e · s + b_e)`.
    pub fn encode(&self, state: &[f64]) -> Vec<f64> {
        linear_relu(&self.encoder_w, &self.encoder_b, state)
    }

    /// Predict next features from current features and a discrete action (one-hot).
    pub fn forward_predict(&self, features: &[f64], action: usize) -> Vec<f64> {
        // Concatenate features with action one-hot encoding
        let mut inp = Vec::with_capacity(self.feature_dim + self.n_actions);
        inp.extend_from_slice(features);
        let mut onehot = vec![0.0_f64; self.n_actions];
        if action < self.n_actions {
            onehot[action] = 1.0;
        }
        inp.extend_from_slice(&onehot);
        linear_relu(&self.forward_w, &self.forward_b, &inp)
    }

    /// Predict action logits from the pair `(φ(s), φ(s'))`.
    pub fn inverse_predict(&self, feat_curr: &[f64], feat_next: &[f64]) -> Vec<f64> {
        let mut inp = Vec::with_capacity(2 * self.feature_dim);
        inp.extend_from_slice(feat_curr);
        inp.extend_from_slice(feat_next);
        let logits = linear_fwd(&self.inverse_w, &self.inverse_b, &inp);
        let mut out = logits;
        softmax_vec(&mut out);
        out
    }

    /// Compute intrinsic reward: `r_i = (η/2) · ‖φ̂(s') − φ(s')‖²`.
    pub fn intrinsic_reward(&self, state: &[f64], action: usize, next_state: &[f64]) -> f64 {
        let phi_s = self.encode(state);
        let phi_sp = self.encode(next_state);
        let phi_hat_sp = self.forward_predict(&phi_s, action);
        let mse = l2_sq(&phi_hat_sp, &phi_sp);
        0.5 * self.eta * mse
    }

    /// Compute `(forward_loss, inverse_loss)` for a single transition.
    ///
    /// - `forward_loss  = (1/2) · ‖φ̂(s') − φ(s')‖²`
    /// - `inverse_loss  = −log P̂(a | φ(s), φ(s'))` (cross-entropy)
    pub fn compute_losses(&self, state: &[f64], action: usize, next_state: &[f64]) -> (f64, f64) {
        let phi_s = self.encode(state);
        let phi_sp = self.encode(next_state);
        let phi_hat_sp = self.forward_predict(&phi_s, action);
        let forward_loss = 0.5 * l2_sq(&phi_hat_sp, &phi_sp);
        let action_probs = self.inverse_predict(&phi_s, &phi_sp);
        let p_action = if action < action_probs.len() {
            action_probs[action].max(1e-15)
        } else {
            1e-15
        };
        let inverse_loss = -p_action.ln();
        (forward_loss, inverse_loss)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  RsRandomNetworkDistillation  (RND — Burda et al. 2018)
// ─────────────────────────────────────────────────────────────────────────────

/// Random Network Distillation (RND) intrinsic motivation.
///
/// A **fixed** random target network `T` and a **trained** predictor `P`
/// both map states to fixed-dimensional feature vectors.
///
/// Intrinsic reward: `r_i = ‖T(s) − P(s)‖²`, normalised by a running
/// estimate of the reward's mean and variance (Welford's algorithm).
#[derive(Debug, Clone)]
pub struct RsRandomNetworkDistillation {
    // Target (frozen) — two-layer MLP
    target_w1: Vec<Vec<f64>>,
    target_b1: Vec<f64>,
    target_w2: Vec<Vec<f64>>,
    target_b2: Vec<f64>,
    // Predictor (trained)
    predictor_w1: Vec<Vec<f64>>,
    predictor_b1: Vec<f64>,
    predictor_w2: Vec<Vec<f64>>,
    predictor_b2: Vec<f64>,
    /// State dimension.
    pub state_dim: usize,
    /// Hidden layer dimension.
    pub hidden_dim: usize,
    /// Output embedding dimension.
    pub output_dim: usize,
    /// Running mean for Welford normalisation.
    pub reward_running_mean: f64,
    /// Running variance for Welford normalisation.
    pub reward_running_var: f64,
    /// Number of updates seen (for Welford).
    running_count: u64,
}

impl RsRandomNetworkDistillation {
    /// Construct RND; both networks share Xavier init but predictor is mutable.
    pub fn new(state_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let target_w1 = xavier_init(hidden_dim, state_dim, 1337, 0);
        let target_b1 = vec![0.0; hidden_dim];
        let target_w2 = xavier_init(output_dim, hidden_dim, 1337, 1);
        let target_b2 = vec![0.0; output_dim];
        // Predictor starts with different init so there is prediction error from the start
        let predictor_w1 = xavier_init(hidden_dim, state_dim, 9999, 0);
        let predictor_b1 = vec![0.0; hidden_dim];
        let predictor_w2 = xavier_init(output_dim, hidden_dim, 9999, 1);
        let predictor_b2 = vec![0.0; output_dim];
        Self {
            target_w1,
            target_b1,
            target_w2,
            target_b2,
            predictor_w1,
            predictor_b1,
            predictor_w2,
            predictor_b2,
            state_dim,
            hidden_dim,
            output_dim,
            reward_running_mean: 0.0,
            reward_running_var: 1.0,
            running_count: 0,
        }
    }

    /// Forward pass through the frozen target network.
    fn target_forward(&self, state: &[f64]) -> Vec<f64> {
        let h = linear_relu(&self.target_w1, &self.target_b1, state);
        linear_fwd(&self.target_w2, &self.target_b2, &h)
    }

    /// Forward pass through the trained predictor network.
    fn predictor_forward(&self, state: &[f64]) -> Vec<f64> {
        let h = linear_relu(&self.predictor_w1, &self.predictor_b1, state);
        linear_fwd(&self.predictor_w2, &self.predictor_b2, &h)
    }

    /// Raw intrinsic reward `‖T(s) − P(s)‖²` normalised by running stats.
    pub fn intrinsic_reward(&self, state: &[f64]) -> f64 {
        let t = self.target_forward(state);
        let p = self.predictor_forward(state);
        let raw = l2_sq(&t, &p);
        self.normalized_reward(raw)
    }

    /// One gradient-descent step on the predictor MSE loss.
    /// Returns the raw (unnormalised) MSE loss before the update.
    pub fn update_predictor(&mut self, state: &[f64], lr: f64) -> f64 {
        let t = self.target_forward(state);
        let h1 = linear_relu(&self.predictor_w1, &self.predictor_b1, state);
        let p = linear_fwd(&self.predictor_w2, &self.predictor_b2, &h1);

        let raw_loss = l2_sq(&t, &p);

        // Gradient of MSE w.r.t. output layer pre-activation: δ_out = 2*(p - t) / output_dim
        let out_dim = self.output_dim;
        let mut delta_out: Vec<f64> = p
            .iter()
            .zip(t.iter())
            .map(|(&pi, &ti)| 2.0 * (pi - ti) / out_dim as f64)
            .collect();

        // Update W2 and b2
        for (i, row) in self.predictor_w2.iter_mut().enumerate() {
            let d = delta_out[i];
            self.predictor_b2[i] -= lr * d;
            for (j, w) in row.iter_mut().enumerate() {
                *w -= lr * d * h1[j];
            }
        }

        // Backprop through ReLU at hidden layer
        let hidden_dim = self.hidden_dim;
        let mut delta_h1 = vec![0.0_f64; hidden_dim];
        for i in 0..hidden_dim {
            let grad_relu = if h1[i] > 0.0 { 1.0 } else { 0.0 };
            let back: f64 = (0..out_dim)
                .map(|j| delta_out[j] * self.predictor_w2[j][i])
                .sum();
            delta_h1[i] = back * grad_relu;
        }

        // Update W1 and b1
        for (i, row) in self.predictor_w1.iter_mut().enumerate() {
            let d = delta_h1[i];
            self.predictor_b1[i] -= lr * d;
            let state_len = state.len().min(self.state_dim);
            for j in 0..state_len {
                row[j] -= lr * d * state[j];
            }
        }

        // Suppress unused warning on delta_out
        let _ = delta_out.as_mut_slice();

        raw_loss
    }

    /// Update Welford running mean and variance with a new raw reward value.
    pub fn update_running_stats(&mut self, reward: f64) {
        self.running_count += 1;
        let n = self.running_count as f64;
        let delta = reward - self.reward_running_mean;
        self.reward_running_mean += delta / n;
        let delta2 = reward - self.reward_running_mean;
        // Welford M2 trick — store var directly as approximate M2/n
        self.reward_running_var = ((n - 1.0) * self.reward_running_var + delta * delta2) / n;
    }

    /// Normalise: `(r − μ) / sqrt(σ² + ε)`.
    pub fn normalized_reward(&self, raw_reward: f64) -> f64 {
        const EPS: f64 = 1e-8;
        let std = (self.reward_running_var + EPS).sqrt();
        (raw_reward - self.reward_running_mean) / std
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  RsEmpowermentIntrinsic
// ─────────────────────────────────────────────────────────────────────────────

/// Tabular empowerment — intrinsic motivation based on the channel capacity
/// of the agent's influence over its environment (Salge et al. 2014).
///
/// Empowerment is approximated as the Shannon entropy of the reachable state
/// distribution under a uniform policy after `n_steps` steps.
#[derive(Debug, Clone)]
pub struct RsEmpowermentIntrinsic {
    /// Transition probabilities `P(s' | s, a)`, shape `[n_actions][n_states][n_states]`.
    pub transition_probs: Vec<Vec<Vec<f64>>>,
    /// Number of environment states.
    pub n_states: usize,
    /// Number of discrete actions.
    pub n_actions: usize,
    /// Look-ahead horizon for empowerment.
    pub n_steps: usize,
}

impl RsEmpowermentIntrinsic {
    /// Create a new module with **uniform** transition probabilities.
    pub fn new(n_states: usize, n_actions: usize) -> Self {
        let uniform = 1.0 / n_states as f64;
        let transition_probs = (0..n_actions)
            .map(|_| (0..n_states).map(|_| vec![uniform; n_states]).collect())
            .collect();
        Self {
            transition_probs,
            n_states,
            n_actions,
            n_steps: 1,
        }
    }

    /// Replace the current transition table.
    pub fn set_transitions(&mut self, transitions: Vec<Vec<Vec<f64>>>) {
        self.transition_probs = transitions;
    }

    /// Compute `P(s' | state, uniform_policy)` after `n` steps.
    ///
    /// Uses the marginalised n-step distribution:
    /// `P_n(s' | s) = (1/|A|) Σ_a [T_a^n](s, s')`.
    pub fn multi_step_reach(&self, state: usize, n: usize) -> Vec<f64> {
        if self.n_states == 0 {
            return Vec::new();
        }
        // Initialise as a point mass at `state`
        let mut dist = vec![0.0_f64; self.n_states];
        if state < self.n_states {
            dist[state] = 1.0;
        }
        for _ in 0..n {
            let mut next_dist = vec![0.0_f64; self.n_states];
            let inv_a = 1.0 / self.n_actions.max(1) as f64;
            for a in 0..self.n_actions {
                let trans = &self.transition_probs[a];
                for s in 0..self.n_states {
                    if dist[s].abs() < 1e-15 {
                        continue;
                    }
                    for sp in 0..self.n_states {
                        next_dist[sp] += dist[s] * trans[s][sp] * inv_a;
                    }
                }
            }
            dist = next_dist;
        }
        // Normalise to correct floating-point drift
        let total: f64 = dist.iter().sum();
        if total > 1e-15 {
            for v in dist.iter_mut() {
                *v /= total;
            }
        }
        dist
    }

    /// Empowerment at `state`: entropy of the `n_steps`-step reachability distribution.
    pub fn empowerment(&self, state: usize) -> f64 {
        let dist = self.multi_step_reach(state, self.n_steps);
        entropy(&dist)
    }

    /// Empowerment-based shaping reward: `E(s') − E(s)`.
    pub fn empowerment_reward(&self, state: usize, next_state: usize) -> f64 {
        let e_next = self.empowerment(next_state);
        let e_curr = self.empowerment(state);
        e_next - e_curr
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  RsGoalConditionedReward
// ─────────────────────────────────────────────────────────────────────────────

/// Universal Value Function Approximator–style goal-conditioned reward
/// (Schaul et al. 2015) with Hindsight Experience Replay support
/// (Andrychowicz et al. 2017).
///
/// Reward: `r(s, g) = exp(−d²(s,g) / (2·tol²)) − 1 ∈ [−1, 0]`.
#[derive(Debug, Clone)]
pub struct RsGoalConditionedReward {
    /// Goal embedding weight matrix, shape `[embed_dim × state_dim]`.
    goal_embed_w: Vec<Vec<f64>>,
    /// Goal embedding bias, length `embed_dim`.
    goal_embed_b: Vec<f64>,
    /// Output dimension of the goal embedding.
    pub embed_dim: usize,
    /// Input state dimension.
    pub state_dim: usize,
    /// Distance threshold below which a goal is considered "reached".
    pub tolerance: f64,
}

impl RsGoalConditionedReward {
    /// Construct with Xavier-initialised embedding.
    pub fn new(state_dim: usize, embed_dim: usize, tolerance: f64) -> Self {
        let goal_embed_w = xavier_init(embed_dim, state_dim, 7777, 0);
        let goal_embed_b = vec![0.0; embed_dim];
        Self {
            goal_embed_w,
            goal_embed_b,
            embed_dim,
            state_dim,
            tolerance,
        }
    }

    /// Embed a state: `ReLU(W · s + b)`.
    pub fn embed_state(&self, state: &[f64]) -> Vec<f64> {
        linear_relu(&self.goal_embed_w, &self.goal_embed_b, state)
    }

    /// L2 distance in embedding space: `‖φ(s) − φ(g)‖₂`.
    pub fn distance(&self, state: &[f64], goal: &[f64]) -> f64 {
        let phi_s = self.embed_state(state);
        let phi_g = self.embed_state(goal);
        l2_sq(&phi_s, &phi_g).sqrt()
    }

    /// Gaussian-shaped goal reward: `exp(−d²/(2·tol²)) − 1 ∈ [−1, 0]`.
    pub fn reward(&self, state: &[f64], goal: &[f64]) -> f64 {
        let d = self.distance(state, goal);
        let tol = self.tolerance.max(1e-8);
        (-(d * d) / (2.0 * tol * tol)).exp() - 1.0
    }

    /// Check whether the state is within `tolerance` of the goal.
    pub fn goal_reached(&self, state: &[f64], goal: &[f64]) -> bool {
        self.distance(state, goal) < self.tolerance
    }

    /// Hindsight relabelling: recompute rewards for a trajectory using an
    /// achieved goal as the new goal (HER, Andrychowicz et al. 2017).
    pub fn hindsight_reward(&self, trajectory: &[Vec<f64>], achieved_goal: &[f64]) -> Vec<f64> {
        trajectory
            .iter()
            .map(|state| self.reward(state, achieved_goal))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  RsRewardEnsemble
// ─────────────────────────────────────────────────────────────────────────────

/// Ensemble of linear Bradley-Terry reward models for uncertainty-aware
/// reward learning (Ibarz et al. 2018 / active reward querying).
#[derive(Debug, Clone)]
pub struct RsRewardEnsemble {
    models: Vec<RsBradleyTerryModel>,
    n_models: usize,
    feature_dim: usize,
}

impl RsRewardEnsemble {
    /// Create an ensemble of `n_models` independent Bradley-Terry models.
    pub fn new(n_models: usize, feature_dim: usize) -> Self {
        Self {
            models: (0..n_models)
                .map(|_| RsBradleyTerryModel::new(feature_dim))
                .collect(),
            n_models,
            feature_dim,
        }
    }

    /// Mean reward across ensemble members.
    pub fn mean_reward(&self, features: &[f64]) -> f64 {
        if self.n_models == 0 {
            return 0.0;
        }
        let sum: f64 = self.models.iter().map(|m| m.reward(features)).sum();
        sum / self.n_models as f64
    }

    /// Sample variance of reward predictions.
    pub fn reward_variance(&self, features: &[f64]) -> f64 {
        if self.n_models < 2 {
            return 0.0;
        }
        let mean = self.mean_reward(features);
        let var: f64 = self
            .models
            .iter()
            .map(|m| {
                let r = m.reward(features);
                (r - mean) * (r - mean)
            })
            .sum::<f64>()
            / (self.n_models - 1) as f64;
        var
    }

    /// Return `(mean, variance)` jointly.
    pub fn reward_with_uncertainty(&self, features: &[f64]) -> (f64, f64) {
        (self.mean_reward(features), self.reward_variance(features))
    }

    /// Train each model on a bootstrapped (with replacement) subset of comparisons.
    pub fn train_ensemble(&mut self, comparisons: &[(Vec<f64>, Vec<f64>)], lr: f64, epochs: usize) {
        if comparisons.is_empty() {
            return;
        }
        let n = comparisons.len();
        for (idx, model) in self.models.iter_mut().enumerate() {
            // Deterministic bootstrap: model idx uses a fixed seed offset
            let mut rng = StdRng::seed_from_u64((idx as u64).wrapping_mul(12345).wrapping_add(1));
            let subset: Vec<(Vec<f64>, Vec<f64>)> = (0..n)
                .map(|_| {
                    let i: usize = (rng.random::<u64>() as usize) % n;
                    comparisons[i].clone()
                })
                .collect();
            model.train_batch(&subset, lr, epochs);
        }
    }

    /// Standard deviation of individual reward predictions — used as an
    /// uncertainty measure for exploration / active querying.
    pub fn disagreement(&self, features: &[f64]) -> f64 {
        self.reward_variance(features).sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  RsRetroactiveLearning  (HER — Andrychowicz et al. 2017)
// ─────────────────────────────────────────────────────────────────────────────

/// Hindsight Experience Replay (HER) goal relabelling strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HerStrategy {
    /// Use the **final** state of the episode as the hindsight goal.
    Final,
    /// Use a **uniformly random future** state from the same episode.
    Future,
    /// Use a **uniformly random** state from anywhere in the episode.
    Episode,
}

/// Episode data: parallel arrays of states, actions, and rewards.
#[derive(Debug, Clone)]
pub struct RsEpisode {
    /// State observations at each time step.
    pub states: Vec<Vec<f64>>,
    /// Actions taken at each time step.
    pub actions: Vec<usize>,
    /// Rewards received at each time step.
    pub rewards: Vec<f64>,
}

impl RsEpisode {
    /// Length of the episode (number of time steps).
    pub fn len(&self) -> usize {
        self.rewards.len()
    }

    /// Whether the episode is empty.
    pub fn is_empty(&self) -> bool {
        self.rewards.is_empty()
    }
}

/// Hindsight Experience Replay augmentation utility.
pub struct RsRetroactiveLearning {
    /// Maximum number of stored episodes (not enforced here — for metadata).
    buffer_size: usize,
}

impl RsRetroactiveLearning {
    /// Create a new retroactive learning helper.
    pub fn new(buffer_size: usize) -> Self {
        Self { buffer_size }
    }

    /// Relabel a single episode with a new goal determined by `her_strategy`,
    /// then recompute rewards using `reward_fn`.
    pub fn relabel_episode(
        &self,
        episode: &RsEpisode,
        reward_fn: &RsGoalConditionedReward,
        her_strategy: HerStrategy,
    ) -> RsEpisode {
        let t = episode.states.len();
        if t == 0 {
            return RsEpisode {
                states: Vec::new(),
                actions: episode.actions.clone(),
                rewards: Vec::new(),
            };
        }
        let goal: &Vec<f64> = match her_strategy {
            HerStrategy::Final => episode.states.last().unwrap_or(&episode.states[0]),
            HerStrategy::Future => {
                // Use a seeded pseudo-random pick to remain deterministic
                let seed_idx = t / 2;
                &episode.states[seed_idx]
            }
            HerStrategy::Episode => {
                let seed_idx = t / 3;
                &episode.states[seed_idx]
            }
        };
        let new_rewards: Vec<f64> = episode
            .states
            .iter()
            .map(|s| reward_fn.reward(s, goal))
            .collect();
        RsEpisode {
            states: episode.states.clone(),
            actions: episode.actions.clone(),
            rewards: new_rewards,
        }
    }

    /// For each episode, produce `k` additional hindsight-relabelled episodes.
    pub fn augment_batch(
        &self,
        episodes: &[RsEpisode],
        reward_fn: &RsGoalConditionedReward,
        k: usize,
    ) -> Vec<RsEpisode> {
        let strategies = [
            HerStrategy::Final,
            HerStrategy::Future,
            HerStrategy::Episode,
        ];
        let mut result = Vec::new();
        for episode in episodes {
            for relabel_idx in 0..k {
                let strategy = strategies[relabel_idx % strategies.len()];
                let relabelled = self.relabel_episode(episode, reward_fn, strategy);
                result.push(relabelled);
            }
        }
        result
    }

    /// Accessor for the configured buffer size.
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  RsMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluation utilities for reward modelling and shaping.
pub struct RsMetrics;

impl RsMetrics {
    /// Discounted sum of rewards: `G = Σ_t γ^t · r_t`.
    pub fn shaped_return(rewards: &[f64], gamma: f64) -> f64 {
        let mut g = 0.0_f64;
        let mut discount = 1.0_f64;
        for &r in rewards {
            g += discount * r;
            discount *= gamma;
        }
        g
    }

    /// Pearson correlation between two reward sequences.
    pub fn reward_correlation(r1: &[f64], r2: &[f64]) -> f64 {
        let n = r1.len().min(r2.len());
        if n < 2 {
            return 0.0;
        }
        let mean1 = r1[..n].iter().sum::<f64>() / n as f64;
        let mean2 = r2[..n].iter().sum::<f64>() / n as f64;
        let mut cov = 0.0_f64;
        let mut var1 = 0.0_f64;
        let mut var2 = 0.0_f64;
        for i in 0..n {
            let d1 = r1[i] - mean1;
            let d2 = r2[i] - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }
        let denom = (var1 * var2).sqrt();
        if denom < 1e-15 {
            0.0
        } else {
            (cov / denom).clamp(-1.0, 1.0)
        }
    }

    /// Ratio `mean|intrinsic| / (mean|extrinsic| + ε)`.
    pub fn intrinsic_extrinsic_ratio(intrinsic: &[f64], extrinsic: &[f64]) -> f64 {
        const EPS: f64 = 1e-8;
        let mean_int = if intrinsic.is_empty() {
            0.0
        } else {
            intrinsic.iter().map(|x| x.abs()).sum::<f64>() / intrinsic.len() as f64
        };
        let mean_ext = if extrinsic.is_empty() {
            0.0
        } else {
            extrinsic.iter().map(|x| x.abs()).sum::<f64>() / extrinsic.len() as f64
        };
        mean_int / (mean_ext + EPS)
    }

    /// Spearman rank correlation between two reward sequences.
    pub fn spearman_rank_correlation(r1: &[f64], r2: &[f64]) -> f64 {
        let n = r1.len().min(r2.len());
        if n < 2 {
            return 0.0;
        }
        let rank = |v: &[f64]| -> Vec<f64> {
            let mut indexed: Vec<(usize, f64)> = v[..n].iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut ranks = vec![0.0_f64; n];
            let mut i = 0;
            while i < n {
                let mut j = i;
                while j < n - 1 && (indexed[j + 1].1 - indexed[j].1).abs() < 1e-15 {
                    j += 1;
                }
                let avg_rank = (i + j) as f64 / 2.0 + 1.0;
                for k in i..=j {
                    ranks[indexed[k].0] = avg_rank;
                }
                i = j + 1;
            }
            ranks
        };
        let rk1 = rank(r1);
        let rk2 = rank(r2);
        Self::reward_correlation(&rk1, &rk2)
    }

    /// Fraction of comparisons for which the model correctly predicts the winner.
    pub fn preference_accuracy(
        model: &RsBradleyTerryModel,
        comparisons: &[(Vec<f64>, Vec<f64>)],
    ) -> f64 {
        if comparisons.is_empty() {
            return 0.0;
        }
        let correct = comparisons
            .iter()
            .filter(|(w, l)| model.preference_probability(w, l) >= 0.5)
            .count();
        correct as f64 / comparisons.len() as f64
    }
}
