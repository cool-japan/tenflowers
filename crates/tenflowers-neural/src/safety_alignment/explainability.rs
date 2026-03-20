//! Section 4 — Interpretability & Explainability
//!
//! - `LimeExplainer` (local linear surrogate)
//! - `ShapValues` (Shapley sampling approximation)
//! - `CounterfactualExplainer` (gradient-based minimal change)
//! - `ConceptBottleneck` (TCAV concept activation vectors)
//! - `AttentionExplainer` (attention rollout)

use super::helpers::{dot, mat_mul, standard_normal_sample, weighted_linear_regression};
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

/// LIME: locally fit a linear model to explain predictions.
#[derive(Debug, Clone)]
pub struct LimeExplainer {
    /// Number of perturbed samples.
    pub n_samples: usize,
    /// Kernel bandwidth.
    pub sigma: f64,
}

impl LimeExplainer {
    /// Create a new LIME explainer.
    pub fn new(n_samples: usize, sigma: f64) -> Self {
        Self {
            n_samples: n_samples.max(1),
            sigma: sigma.max(1e-9),
        }
    }

    /// Explain prediction for `x`; returns feature importances.
    pub fn explain(&self, x: &[f64], seed: u64, predict_fn: &dyn Fn(&[f64]) -> f64) -> Vec<f64> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let mut rng = StdRng::seed_from_u64(seed);

        let mut samples: Vec<Vec<f64>> = Vec::with_capacity(self.n_samples);
        let mut predictions: Vec<f64> = Vec::with_capacity(self.n_samples);
        let mut weights: Vec<f64> = Vec::with_capacity(self.n_samples);

        for _ in 0..self.n_samples {
            let perturbed: Vec<f64> = x
                .iter()
                .map(|&xi| xi + self.sigma * standard_normal_sample(&mut rng))
                .collect();
            let pred = predict_fn(&perturbed);
            let sq_dist: f64 = perturbed
                .iter()
                .zip(x.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            let w = (-sq_dist / (self.sigma * self.sigma)).exp();
            predictions.push(pred);
            samples.push(perturbed);
            weights.push(w);
        }

        weighted_linear_regression(&samples, &predictions, &weights)
    }
}

/// SHAP value approximation via Shapley sampling.
#[derive(Debug, Clone)]
pub struct ShapValues {
    /// Number of background samples for Shapley estimation.
    pub n_background: usize,
}

impl ShapValues {
    /// Create a new SHAP estimator.
    pub fn new(n_background: usize) -> Self {
        Self {
            n_background: n_background.max(1),
        }
    }

    /// Compute SHAP values; `sum(shap) ≈ f(x) - base_value`.
    pub fn compute_shap(
        &self,
        x: &[f64],
        base_value: f64,
        seed: u64,
        predict_fn: &dyn Fn(&[f64]) -> f64,
    ) -> Vec<f64> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut shap = vec![0.0f64; n];

        for _ in 0..self.n_background {
            let mut perm: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (rng.random::<f64>() * (i + 1) as f64) as usize;
                perm.swap(i, j);
            }

            let mut prev_pred = base_value;
            let mut coalition: Vec<f64> = vec![base_value; n];

            for &feat in &perm {
                coalition[feat] = x[feat];
                let new_pred = predict_fn(&coalition);
                shap[feat] += new_pred - prev_pred;
                prev_pred = new_pred;
            }
        }

        for s in &mut shap {
            *s /= self.n_background as f64;
        }
        shap
    }
}

/// Gradient-based counterfactual explainer.
#[derive(Debug, Clone)]
pub struct CounterfactualExplainer {
    /// Learning rate.
    pub step_size: f64,
    /// Maximum gradient steps.
    pub max_steps: usize,
    /// Proximity regularization weight λ.
    pub lambda: f64,
}

impl CounterfactualExplainer {
    /// Create a new counterfactual explainer.
    pub fn new(step_size: f64, max_steps: usize, lambda: f64) -> Self {
        Self {
            step_size: step_size.max(0.0),
            max_steps: max_steps.max(1),
            lambda,
        }
    }

    /// Find a counterfactual for `x` predicted as `target_class`.
    pub fn find_counterfactual(
        &self,
        x: &[f64],
        target_class: usize,
        predict_fn: &dyn Fn(&[f64]) -> usize,
        score_fn: &dyn Fn(&[f64], usize) -> f64,
    ) -> Vec<f64> {
        let n = x.len();
        let mut x_cf = x.to_vec();
        let eps = 1e-4;

        for _ in 0..self.max_steps {
            if predict_fn(&x_cf) == target_class {
                break;
            }
            let mut grad = vec![0.0f64; n];
            let base_score = score_fn(&x_cf, target_class);
            for i in 0..n {
                let mut x_plus = x_cf.clone();
                x_plus[i] += eps;
                let score_plus = score_fn(&x_plus, target_class);
                let d_class = (score_plus - base_score) / eps;
                let d_prox = 2.0 * self.lambda * (x_cf[i] - x[i]);
                grad[i] = -d_class + d_prox;
            }
            for i in 0..n {
                x_cf[i] -= self.step_size * grad[i];
            }
        }
        x_cf
    }
}

/// Concept Activation Vector (CAV) for TCAV-style interpretability.
#[derive(Debug, Clone, Default)]
pub struct ConceptBottleneck;

impl ConceptBottleneck {
    /// Create a new concept bottleneck explainer.
    pub fn new() -> Self {
        Self
    }

    /// Compute CAV: train linear SVM on concept vs random examples.
    ///
    /// Returns a unit-length direction vector.
    pub fn compute_cav(
        &self,
        concept_examples: &[Vec<f64>],
        random_examples: &[Vec<f64>],
    ) -> Vec<f64> {
        let n_concept = concept_examples.len();
        let n_random = random_examples.len();
        let n = n_concept + n_random;
        if n == 0 {
            return Vec::new();
        }
        let feat_dim = concept_examples
            .first()
            .or_else(|| random_examples.first())
            .map(|v| v.len())
            .unwrap_or(0);
        if feat_dim == 0 {
            return Vec::new();
        }

        let mut weights = vec![0.0f64; feat_dim];
        let lr = 0.01;
        let n_epochs = 50;

        let mut all_data: Vec<(&Vec<f64>, f64)> = Vec::with_capacity(n);
        for ex in concept_examples {
            all_data.push((ex, 1.0));
        }
        for ex in random_examples {
            all_data.push((ex, -1.0));
        }

        for _ in 0..n_epochs {
            for (feat, label) in &all_data {
                let label = *label;
                let score: f64 = feat.iter().zip(weights.iter()).map(|(f, w)| f * w).sum();
                let margin = label * score;
                if margin < 1.0 {
                    for (j, w) in weights.iter_mut().enumerate() {
                        *w += lr * label * feat.get(j).copied().unwrap_or(0.0);
                    }
                }
            }
        }

        let norm: f64 = weights.iter().map(|w| w * w).sum::<f64>().sqrt();
        if norm > 1e-12 {
            weights.iter_mut().for_each(|w| *w /= norm);
        }
        weights
    }

    /// Compute the TCAV score: fraction of concept examples where directional
    /// derivative along the CAV is positive.
    pub fn tcav_score(
        &self,
        concept_examples: &[Vec<f64>],
        cav: &[f64],
        gradient_fn: &dyn Fn(&[f64]) -> Vec<f64>,
    ) -> f64 {
        let n = concept_examples.len();
        if n == 0 {
            return 0.0;
        }
        let positives = concept_examples
            .iter()
            .filter(|ex| {
                let grad = gradient_fn(ex);
                dot(&grad, cav) > 0.0
            })
            .count();
        positives as f64 / n as f64
    }
}

/// Attention rollout visualization (Abnar & Zuidema, 2020).
#[derive(Debug, Clone, Default)]
pub struct AttentionExplainer;

impl AttentionExplainer {
    /// Create a new attention explainer.
    pub fn new() -> Self {
        Self
    }

    /// Compute attention rollout across transformer layers.
    ///
    /// `attention_weights`: list of `[seq_len][seq_len]` per-layer matrices.
    /// Returns a vector of length `seq_len`.
    pub fn rollout(&self, attention_weights: &[Vec<Vec<f64>>]) -> Vec<f64> {
        if attention_weights.is_empty() {
            return Vec::new();
        }
        let seq_len = attention_weights[0].len();
        if seq_len == 0 {
            return Vec::new();
        }

        let mut rollout: Vec<Vec<f64>> = (0..seq_len)
            .map(|i| {
                let mut row = vec![0.0; seq_len];
                row[i] = 1.0;
                row
            })
            .collect();

        for layer_attn in attention_weights {
            let augmented: Vec<Vec<f64>> = layer_attn
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    row.iter()
                        .enumerate()
                        .map(|(j, &v)| {
                            let identity = if i == j { 1.0 } else { 0.0 };
                            0.5 * v + 0.5 * identity
                        })
                        .collect()
                })
                .collect();

            let normalized: Vec<Vec<f64>> = augmented
                .iter()
                .map(|row| {
                    let s: f64 = row.iter().sum();
                    if s > 1e-12 {
                        row.iter().map(|&v| v / s).collect()
                    } else {
                        row.clone()
                    }
                })
                .collect();

            rollout = mat_mul(&normalized, &rollout);
        }

        rollout
            .last()
            .cloned()
            .unwrap_or_else(|| vec![0.0; seq_len])
    }
}
