//! Section 1 — Constitutional AI & RLHF
//!
//! - `ConstitutionalPrinciple` + `ConstitutionalAiFilter`
//! - `PpoWithKl` (PPO with KL penalty)
//! - `DpoTrainer` (Direct Preference Optimization)
//! - `SycophancyDetector`

use super::helpers::cosine_similarity;

/// A single constitutional principle used to evaluate model outputs.
///
/// The principle is represented as a natural-language string together with a
/// scalar `weight` (relative importance) and a `violation_penalty` that is
/// subtracted from the overall safety score when the principle is violated.
#[derive(Debug, Clone)]
pub struct ConstitutionalPrinciple {
    /// Human-readable description of the principle.
    pub principle: String,
    /// Relative weight (≥ 0); higher means the principle matters more.
    pub weight: f64,
    /// Penalty subtracted from the safety score on violation (≥ 0).
    pub violation_penalty: f64,
}

impl ConstitutionalPrinciple {
    /// Create a new constitutional principle.
    pub fn new(principle: impl Into<String>, weight: f64, violation_penalty: f64) -> Self {
        Self {
            principle: principle.into(),
            weight: weight.max(0.0),
            violation_penalty: violation_penalty.max(0.0),
        }
    }
}

/// Scores model output features against a set of constitutional principles.
#[derive(Debug, Clone, Default)]
pub struct ConstitutionalAiFilter;

impl ConstitutionalAiFilter {
    /// Create a new filter instance.
    pub fn new() -> Self {
        Self
    }

    /// Evaluate `output_features` against `principle` and return a weighted
    /// safety score.
    pub fn evaluate(&self, output_features: &[f64], principle: &ConstitutionalPrinciple) -> f64 {
        if output_features.is_empty() {
            return -principle.violation_penalty * principle.weight;
        }
        let mean = output_features.iter().sum::<f64>() / output_features.len() as f64;
        let raw = principle.weight * mean;
        if mean <= 0.0 {
            raw - principle.violation_penalty
        } else {
            raw
        }
    }
}

/// PPO with a KL-divergence penalty from a reference (frozen) model.
///
/// ```text
/// L = -min(r·A, clip(r, 1-ε, 1+ε)·A) + β·KL(π ‖ π_ref)
/// ```
#[derive(Debug, Clone)]
pub struct PpoWithKl {
    /// PPO clipping range `ε`.
    pub clip_epsilon: f64,
}

impl PpoWithKl {
    /// Create a new PPO-KL trainer.
    pub fn new(clip_epsilon: f64) -> Self {
        Self {
            clip_epsilon: clip_epsilon.clamp(1e-6, 1.0),
        }
    }

    /// Compute the PPO-KL loss for a single transition.
    pub fn compute_loss(&self, ratio: f64, advantage: f64, kl_div: f64, kl_coef: f64) -> f64 {
        let eps = self.clip_epsilon;
        let clipped = ratio.clamp(1.0 - eps, 1.0 + eps);
        let ppo_term = (ratio * advantage).min(clipped * advantage);
        -ppo_term + kl_coef * kl_div.max(0.0)
    }
}

/// Direct Preference Optimization (DPO) trainer (Rafailov et al., 2023).
///
/// ```text
/// L_DPO = -log σ(β · (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x))))
/// ```
#[derive(Debug, Clone, Default)]
pub struct DpoTrainer;

impl DpoTrainer {
    /// Create a new DPO trainer.
    pub fn new() -> Self {
        Self
    }

    /// Compute the DPO loss.
    pub fn dpo_loss(
        &self,
        log_prob_chosen: f64,
        log_prob_rejected: f64,
        ref_log_prob_chosen: f64,
        ref_log_prob_rejected: f64,
        beta: f64,
    ) -> f64 {
        let delta =
            (log_prob_chosen - ref_log_prob_chosen) - (log_prob_rejected - ref_log_prob_rejected);
        let logit = beta * delta;
        // -log σ(logit) = log(1 + exp(-logit)) [numerically stable]
        if logit >= 0.0 {
            (-logit).exp().ln_1p()
        } else {
            (-logit) + logit.exp().ln_1p()
        }
    }
}

/// Detects sycophantic responses by comparing output feature vectors against
/// stored preference patterns using cosine similarity.
#[derive(Debug, Clone)]
pub struct SycophancyDetector {
    /// Reference sycophancy patterns (each is a feature vector).
    pub patterns: Vec<Vec<f64>>,
    /// Cosine-similarity threshold above which a response is flagged.
    pub threshold: f64,
}

impl SycophancyDetector {
    /// Create a detector with the given reference patterns and detection threshold.
    pub fn new(patterns: Vec<Vec<f64>>, threshold: f64) -> Self {
        Self {
            patterns,
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Returns `true` if `features` is considered sycophantic.
    pub fn is_sycophantic(&self, features: &[f64]) -> bool {
        self.max_similarity(features) >= self.threshold
    }

    /// Returns the maximum cosine similarity between `features` and any stored pattern.
    pub fn max_similarity(&self, features: &[f64]) -> f64 {
        self.patterns
            .iter()
            .map(|p| cosine_similarity(features, p))
            .fold(f64::NEG_INFINITY, f64::max)
    }
}
