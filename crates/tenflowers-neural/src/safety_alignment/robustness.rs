//! Section 2 — Robustness & Adversarial Defense
//!
//! - `CertifiedRobustness` (randomized smoothing, Cohen et al. 2019)
//! - `AdversarialTrainingAugmenter` (PGD with random restart)
//! - `InputSmoothing` (majority vote over noisy copies)
//! - `FeatureSqueezing` (bit-depth reduction detection)
//! - `DefenseEnsemble` (majority voting over multiple defenses)

use super::helpers::{clopper_pearson_lower, probit, standard_normal_sample};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

/// Certified L₂ robustness via randomized smoothing (Cohen et al., 2019).
#[derive(Debug, Clone)]
pub struct CertifiedRobustness {
    /// Gaussian noise standard deviation σ.
    pub sigma: f64,
    /// Number of Monte-Carlo samples for probability estimation.
    pub n_samples: usize,
    /// Confidence level α for the one-sided Clopper-Pearson interval.
    pub alpha: f64,
}

impl CertifiedRobustness {
    /// Create a new certifier.
    pub fn new(sigma: f64, n_samples: usize, alpha: f64) -> Self {
        Self {
            sigma: sigma.max(1e-9),
            n_samples: n_samples.max(1),
            alpha: alpha.clamp(1e-9, 0.5),
        }
    }

    /// Certify robustness of `predict_fn` at point `x`.
    ///
    /// Returns `(predicted_class, certified_radius)`.
    /// If the abstain condition is triggered, `certified_radius` is `0.0`.
    pub fn certify(
        &self,
        x: &[f64],
        seed: u64,
        predict_fn: &dyn Fn(&[f64]) -> usize,
    ) -> (usize, f64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut vote_counts: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();

        for _ in 0..self.n_samples {
            let noisy: Vec<f64> = x
                .iter()
                .map(|&xi| xi + self.sigma * standard_normal_sample(&mut rng))
                .collect();
            let class = predict_fn(&noisy);
            *vote_counts.entry(class).or_insert(0) += 1;
        }

        let (top_class, top_count) = vote_counts
            .iter()
            .max_by_key(|(_, &v)| v)
            .map(|(&k, &v)| (k, v))
            .unwrap_or((0, 0));

        let p_a_lower = clopper_pearson_lower(top_count, self.n_samples, self.alpha);

        if p_a_lower <= 0.5 {
            return (top_class, 0.0);
        }

        let radius = self.sigma * probit(p_a_lower);
        (top_class, radius.max(0.0))
    }
}

/// Generates adversarial examples via PGD for data augmentation.
#[derive(Debug, Clone)]
pub struct AdversarialTrainingAugmenter {
    /// L∞ perturbation budget ε.
    pub epsilon: f64,
    /// Per-step size α.
    pub step_size: f64,
    /// Number of PGD iterations.
    pub n_steps: usize,
    /// Number of random restarts.
    pub n_restarts: usize,
}

impl AdversarialTrainingAugmenter {
    /// Create a new augmenter.
    pub fn new(epsilon: f64, step_size: f64, n_steps: usize, n_restarts: usize) -> Self {
        Self {
            epsilon: epsilon.max(0.0),
            step_size: step_size.max(0.0),
            n_steps: n_steps.max(1),
            n_restarts: n_restarts.max(1),
        }
    }

    /// Generate an adversarial example for `x`.
    pub fn augment(
        &self,
        x: &[f64],
        seed: u64,
        gradient_fn: &dyn Fn(&[f64]) -> Vec<f64>,
    ) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = x.len();
        let mut best_adv = x.to_vec();
        let mut best_loss = f64::NEG_INFINITY;

        for _ in 0..self.n_restarts {
            let mut x_adv: Vec<f64> = x
                .iter()
                .map(|&xi| {
                    let r: f64 = rng.random::<f64>() * 2.0 - 1.0;
                    xi + r * self.epsilon
                })
                .collect();

            for _ in 0..self.n_steps {
                let grad = gradient_fn(&x_adv);
                for i in 0..n {
                    let sign = if grad[i] > 0.0 {
                        1.0
                    } else if grad[i] < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    x_adv[i] += self.step_size * sign;
                    x_adv[i] = (x_adv[i] - x[i]).clamp(-self.epsilon, self.epsilon) + x[i];
                }
            }

            let g = gradient_fn(&x_adv);
            let loss: f64 = g.iter().map(|v| v.abs()).sum();
            if loss > best_loss {
                best_loss = loss;
                best_adv = x_adv;
            }
        }
        best_adv
    }
}

/// Randomized input smoothing via majority vote.
#[derive(Debug, Clone)]
pub struct InputSmoothing {
    /// Gaussian noise standard deviation.
    pub sigma: f64,
    /// Number of noisy copies to evaluate.
    pub n_smoothing: usize,
}

impl InputSmoothing {
    /// Create a new smoother.
    pub fn new(sigma: f64, n_smoothing: usize) -> Self {
        Self {
            sigma: sigma.max(0.0),
            n_smoothing: n_smoothing.max(1),
        }
    }

    /// Predict with smoothing.
    pub fn predict(&self, x: &[f64], seed: u64, predict_fn: &dyn Fn(&[f64]) -> usize) -> usize {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut vote_counts: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();

        for _ in 0..self.n_smoothing {
            let noisy: Vec<f64> = x
                .iter()
                .map(|&xi| xi + self.sigma * standard_normal_sample(&mut rng))
                .collect();
            let class = predict_fn(&noisy);
            *vote_counts.entry(class).or_insert(0) += 1;
        }

        vote_counts
            .into_iter()
            .max_by_key(|(_, v)| *v)
            .map(|(k, _)| k)
            .unwrap_or(0)
    }
}

/// Feature squeezing: detect adversarial inputs by checking if bit-depth
/// reduction changes the prediction.
#[derive(Debug, Clone)]
pub struct FeatureSqueezing {
    /// Number of bits for quantization.
    pub bits: u32,
    /// Feature value range (min, max).
    pub feature_range: (f64, f64),
}

impl FeatureSqueezing {
    /// Create a new feature squeezer.
    pub fn new(bits: u32, feature_range: (f64, f64)) -> Self {
        let (lo, hi) = feature_range;
        let (lo, hi) = if lo < hi { (lo, hi) } else { (hi, lo) };
        Self {
            bits: bits.clamp(1, 32),
            feature_range: (lo, hi),
        }
    }

    /// Quantize `x` to `self.bits` bits within `self.feature_range`.
    pub fn squeeze(&self, x: &[f64]) -> Vec<f64> {
        let (lo, hi) = self.feature_range;
        let levels = (1u64 << self.bits) as f64 - 1.0;
        let range = (hi - lo).max(1e-12);
        x.iter()
            .map(|&v| {
                let normalized = (v - lo) / range;
                let quantized = (normalized * levels).round() / levels;
                quantized * range + lo
            })
            .collect()
    }

    /// Returns `true` if prediction changes after squeezing.
    pub fn detect(&self, x: &[f64], predict_fn: &dyn Fn(&[f64]) -> usize) -> bool {
        let original_class = predict_fn(x);
        let squeezed = self.squeeze(x);
        let squeezed_class = predict_fn(&squeezed);
        original_class != squeezed_class
    }
}

/// Combines multiple defenses via majority voting.
#[derive(Debug, Clone)]
pub struct DefenseEnsemble {
    /// Defense names (for bookkeeping).
    pub defense_names: Vec<String>,
}

impl DefenseEnsemble {
    /// Create a new ensemble.
    pub fn new(defense_names: Vec<String>) -> Self {
        Self { defense_names }
    }

    /// Aggregate `votes` by majority vote; ties broken by smallest class index.
    pub fn vote(&self, votes: &[usize]) -> usize {
        if votes.is_empty() {
            return 0;
        }
        let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &v in votes {
            *counts.entry(v).or_insert(0) += 1;
        }
        let max_count = *counts.values().max().unwrap_or(&0);
        let mut best = usize::MAX;
        for (&class, &count) in &counts {
            if count == max_count && class < best {
                best = class;
            }
        }
        if best == usize::MAX {
            0
        } else {
            best
        }
    }
}
