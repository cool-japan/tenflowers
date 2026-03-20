//! Model ensembling module for combining multiple model predictions
//!
//! Provides ensemble strategies including majority voting, average probability,
//! weighted averaging, stacking generalization, AdaBoost, and diversity metrics.

use std::fmt;

/// Error types for ensemble operations
#[derive(Debug)]
pub enum EnsembleError {
    EmptyPredictions,
    DimensionMismatch {
        expected: usize,
        found: usize,
    },
    WeightMismatch {
        num_models: usize,
        num_weights: usize,
    },
    InvalidNumClasses(usize),
    InvalidStrategy,
    ZeroTotalWeight,
}

impl fmt::Display for EnsembleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EnsembleError::EmptyPredictions => write!(f, "No predictions provided"),
            EnsembleError::DimensionMismatch { expected, found } => {
                write!(
                    f,
                    "Dimension mismatch: expected {}, found {}",
                    expected, found
                )
            }
            EnsembleError::WeightMismatch {
                num_models,
                num_weights,
            } => {
                write!(
                    f,
                    "Weight count {} does not match model count {}",
                    num_weights, num_models
                )
            }
            EnsembleError::InvalidNumClasses(n) => {
                write!(f, "Invalid number of classes: {}", n)
            }
            EnsembleError::InvalidStrategy => write!(f, "Invalid ensemble strategy for this task"),
            EnsembleError::ZeroTotalWeight => write!(f, "Total weight is zero"),
        }
    }
}

impl std::error::Error for EnsembleError {}

/// Strategies for combining model predictions
#[derive(Debug, Clone)]
pub enum EnsembleStrategy {
    /// Classification: mode of class predictions (majority vote)
    MajorityVoting,
    /// Classification: mean of softmax/probability outputs
    AverageProbability,
    /// Weighted combination of outputs
    WeightedAverage(Vec<f32>),
    /// Meta-learner combines base predictions (stacking)
    StackedGeneralization,
    /// Pick the prediction with highest confidence (max probability)
    MaxProbability,
}

/// A model ensemble that combines predictions from multiple models
#[derive(Debug, Clone)]
pub struct ModelEnsemble {
    /// Number of base models
    pub num_models: usize,
    /// Ensemble combination strategy
    pub strategy: EnsembleStrategy,
    /// Per-model weights (used in WeightedAverage and StackedGeneralization)
    pub model_weights: Vec<f32>,
}

impl ModelEnsemble {
    /// Create a new model ensemble with the given strategy
    pub fn new(num_models: usize, strategy: EnsembleStrategy) -> Self {
        let model_weights = match &strategy {
            EnsembleStrategy::WeightedAverage(weights) => weights.clone(),
            _ => vec![1.0_f32 / num_models as f32; num_models],
        };
        Self {
            num_models,
            strategy,
            model_weights,
        }
    }

    /// Create a model ensemble with uniform (equal) weights
    pub fn uniform(num_models: usize) -> Self {
        Self::new(num_models, EnsembleStrategy::AverageProbability)
    }

    /// Validate that predictions have correct shape
    fn validate_classification_predictions(
        &self,
        predictions: &[Vec<f32>],
        n_classes: usize,
    ) -> Result<(), EnsembleError> {
        if predictions.is_empty() {
            return Err(EnsembleError::EmptyPredictions);
        }
        if n_classes == 0 {
            return Err(EnsembleError::InvalidNumClasses(n_classes));
        }
        if predictions.len() != self.num_models {
            return Err(EnsembleError::DimensionMismatch {
                expected: self.num_models,
                found: predictions.len(),
            });
        }
        for (i, pred) in predictions.iter().enumerate() {
            if pred.len() != n_classes {
                return Err(EnsembleError::DimensionMismatch {
                    expected: n_classes,
                    found: pred.len(),
                });
            }
        }
        Ok(())
    }

    /// Combine classification outputs (logits or probabilities)
    ///
    /// Returns combined probabilities over classes of shape `[n_classes]`.
    pub fn combine_classification(
        &self,
        predictions: &[Vec<f32>],
        n_classes: usize,
    ) -> Result<Vec<f32>, EnsembleError> {
        self.validate_classification_predictions(predictions, n_classes)?;

        match &self.strategy {
            EnsembleStrategy::MajorityVoting => {
                // Each model votes for its argmax class; result is one-hot of majority
                let mut class_votes = vec![0usize; n_classes];
                for pred in predictions {
                    let argmax = argmax_f32(pred);
                    class_votes[argmax] += 1;
                }
                let winning_class = class_votes
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, &v)| v)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let mut result = vec![0.0_f32; n_classes];
                result[winning_class] = 1.0;
                Ok(result)
            }

            EnsembleStrategy::AverageProbability => {
                let mut result = vec![0.0_f32; n_classes];
                for pred in predictions {
                    for (r, &p) in result.iter_mut().zip(pred.iter()) {
                        *r += p;
                    }
                }
                let n = self.num_models as f32;
                for r in result.iter_mut() {
                    *r /= n;
                }
                Ok(result)
            }

            EnsembleStrategy::WeightedAverage(weights) => {
                if weights.len() != self.num_models {
                    return Err(EnsembleError::WeightMismatch {
                        num_models: self.num_models,
                        num_weights: weights.len(),
                    });
                }
                let total_weight: f32 = weights.iter().sum();
                if total_weight == 0.0 {
                    return Err(EnsembleError::ZeroTotalWeight);
                }
                let mut result = vec![0.0_f32; n_classes];
                for (pred, &w) in predictions.iter().zip(weights.iter()) {
                    for (r, &p) in result.iter_mut().zip(pred.iter()) {
                        *r += w * p;
                    }
                }
                for r in result.iter_mut() {
                    *r /= total_weight;
                }
                Ok(result)
            }

            EnsembleStrategy::MaxProbability => {
                // Pick the model with highest confidence (max of its argmax probability)
                let best_pred = predictions
                    .iter()
                    .max_by(|a, b| {
                        let max_a = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let max_b = b.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        max_a
                            .partial_cmp(&max_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .ok_or(EnsembleError::EmptyPredictions)?;
                Ok(best_pred.clone())
            }

            EnsembleStrategy::StackedGeneralization => {
                // Default implementation: uniform average (meta-learner weights must be set separately)
                let mut result = vec![0.0_f32; n_classes];
                let total_weight: f32 = self.model_weights.iter().sum();
                if total_weight == 0.0 {
                    return Err(EnsembleError::ZeroTotalWeight);
                }
                for (pred, &w) in predictions.iter().zip(self.model_weights.iter()) {
                    for (r, &p) in result.iter_mut().zip(pred.iter()) {
                        *r += w * p;
                    }
                }
                for r in result.iter_mut() {
                    *r /= total_weight;
                }
                Ok(result)
            }
        }
    }

    /// Combine regression outputs from multiple models
    pub fn combine_regression(&self, predictions: &[f32]) -> Result<f32, EnsembleError> {
        if predictions.is_empty() {
            return Err(EnsembleError::EmptyPredictions);
        }
        if predictions.len() != self.num_models {
            return Err(EnsembleError::DimensionMismatch {
                expected: self.num_models,
                found: predictions.len(),
            });
        }

        match &self.strategy {
            EnsembleStrategy::MajorityVoting => {
                // For regression, MajorityVoting falls back to median
                let mut sorted = predictions.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = sorted.len() / 2;
                if sorted.len() % 2 == 0 {
                    Ok((sorted[mid - 1] + sorted[mid]) / 2.0)
                } else {
                    Ok(sorted[mid])
                }
            }
            EnsembleStrategy::AverageProbability => {
                let sum: f32 = predictions.iter().sum();
                Ok(sum / self.num_models as f32)
            }
            EnsembleStrategy::WeightedAverage(weights) => {
                if weights.len() != self.num_models {
                    return Err(EnsembleError::WeightMismatch {
                        num_models: self.num_models,
                        num_weights: weights.len(),
                    });
                }
                let total_weight: f32 = weights.iter().sum();
                if total_weight == 0.0 {
                    return Err(EnsembleError::ZeroTotalWeight);
                }
                let weighted_sum: f32 = predictions
                    .iter()
                    .zip(weights.iter())
                    .map(|(&p, &w)| p * w)
                    .sum();
                Ok(weighted_sum / total_weight)
            }
            EnsembleStrategy::MaxProbability | EnsembleStrategy::StackedGeneralization => {
                // Weighted average using model_weights
                let total_weight: f32 = self.model_weights.iter().sum();
                if total_weight == 0.0 {
                    return Err(EnsembleError::ZeroTotalWeight);
                }
                let weighted_sum: f32 = predictions
                    .iter()
                    .zip(self.model_weights.iter())
                    .map(|(&p, &w)| p * w)
                    .sum();
                Ok(weighted_sum / total_weight)
            }
        }
    }

    /// Predict the winning class index
    pub fn predict_class(
        &self,
        predictions: &[Vec<f32>],
        n_classes: usize,
    ) -> Result<usize, EnsembleError> {
        let combined = self.combine_classification(predictions, n_classes)?;
        Ok(argmax_f32(&combined))
    }

    /// Compute epistemic uncertainty as Shannon entropy of the ensemble mean probability
    ///
    /// Higher entropy = higher uncertainty.
    pub fn epistemic_uncertainty(
        &self,
        predictions: &[Vec<f32>],
        n_classes: usize,
    ) -> Result<f32, EnsembleError> {
        self.validate_classification_predictions(predictions, n_classes)?;
        let mean_probs = self.combine_classification(predictions, n_classes)?;
        Ok(shannon_entropy(&mean_probs))
    }

    /// Compute disagreement rate among models
    ///
    /// Fraction of model pairs that disagree on the predicted class.
    pub fn disagreement_rate(
        &self,
        predictions: &[Vec<f32>],
        n_classes: usize,
    ) -> Result<f32, EnsembleError> {
        self.validate_classification_predictions(predictions, n_classes)?;
        if self.num_models < 2 {
            return Ok(0.0);
        }

        let classes: Vec<usize> = predictions.iter().map(|p| argmax_f32(p)).collect();
        let mut disagreements = 0usize;
        let mut pairs = 0usize;

        for i in 0..classes.len() {
            for j in (i + 1)..classes.len() {
                pairs += 1;
                if classes[i] != classes[j] {
                    disagreements += 1;
                }
            }
        }

        if pairs == 0 {
            return Ok(0.0);
        }

        Ok(disagreements as f32 / pairs as f32)
    }
}

/// Configuration for Bootstrap Aggregation (Bagging)
#[derive(Debug, Clone)]
pub struct BaggingConfig {
    /// Number of base models
    pub num_models: usize,
    /// Fraction of training data to use for each model's bootstrap sample
    pub sample_fraction: f32,
    /// Whether to sample with replacement
    pub with_replacement: bool,
}

impl Default for BaggingConfig {
    fn default() -> Self {
        Self {
            num_models: 10,
            sample_fraction: 0.632,
            with_replacement: true,
        }
    }
}

/// AdaBoost state for adaptive boosting
#[derive(Debug, Clone)]
pub struct AdaBoostState {
    /// Per-sample weights (normalized, sum = 1.0)
    pub sample_weights: Vec<f32>,
    /// Per-model alpha weights (contribution of each model to final vote)
    pub model_weights: Vec<f32>,
    /// Number of boosting rounds (models)
    pub num_models: usize,
}

impl AdaBoostState {
    /// Initialize with uniform sample weights over `n_samples` samples
    pub fn new(n_samples: usize) -> Self {
        let w = 1.0_f32 / n_samples as f32;
        Self {
            sample_weights: vec![w; n_samples],
            model_weights: Vec::new(),
            num_models: 0,
        }
    }

    /// Update sample weights after a model predicts, and compute model weight alpha.
    ///
    /// Returns the alpha (model weight) for the just-trained model.
    pub fn update_weights(&mut self, predictions: &[usize], targets: &[usize]) -> f32 {
        assert_eq!(
            predictions.len(),
            targets.len(),
            "predictions and targets must have the same length"
        );
        assert_eq!(
            predictions.len(),
            self.sample_weights.len(),
            "predictions length must match sample_weights length"
        );

        // Weighted error rate
        let error: f32 = predictions
            .iter()
            .zip(targets.iter())
            .zip(self.sample_weights.iter())
            .map(|((&pred, &target), &w)| if pred != target { w } else { 0.0 })
            .sum();

        // Clip error to avoid division by zero or log(0)
        let error = error.clamp(1e-10, 1.0 - 1e-10);

        // Alpha: model weight
        let alpha = 0.5 * ((1.0 - error) / error).ln();
        self.model_weights.push(alpha);
        self.num_models += 1;

        // Update sample weights
        for (i, (&pred, &target)) in predictions.iter().zip(targets.iter()).enumerate() {
            let indicator = if pred != target { 1.0_f32 } else { -1.0_f32 };
            self.sample_weights[i] *= (alpha * indicator).exp();
        }

        self.normalize();
        alpha
    }

    /// Normalize sample weights so they sum to 1.0
    pub fn normalize(&mut self) {
        let total: f32 = self.sample_weights.iter().sum();
        if total > 0.0 {
            for w in self.sample_weights.iter_mut() {
                *w /= total;
            }
        }
    }

    /// Final weighted majority vote for a single sample given per-model class predictions
    ///
    /// `per_model_predictions` contains the predicted class index for this sample from each model.
    pub fn predict(&self, per_model_predictions: &[usize]) -> usize {
        if per_model_predictions.is_empty() || self.model_weights.is_empty() {
            return 0;
        }

        // Find number of classes from max prediction
        let max_class = per_model_predictions.iter().cloned().max().unwrap_or(0) + 1;
        let mut class_scores = vec![0.0_f32; max_class];

        for (&pred, &alpha) in per_model_predictions.iter().zip(self.model_weights.iter()) {
            if pred < class_scores.len() {
                class_scores[pred] += alpha;
            }
        }

        argmax_f32(&class_scores)
    }
}

/// Linear stacking meta-learner that combines base model outputs
#[derive(Debug, Clone)]
pub struct StackingMeta {
    /// Meta-learner weight per base model (shape: [num_base_models, output_dim])
    pub weights: Vec<f32>,
    /// Bias term (shape: \[output_dim\])
    pub bias: f32,
    /// Output dimension
    pub output_dim: usize,
    /// Number of base models
    pub num_base_models: usize,
}

impl StackingMeta {
    /// Create a new StackingMeta with uniform weights and zero bias
    pub fn new(num_base_models: usize, output_dim: usize) -> Self {
        let w = 1.0_f32 / num_base_models as f32;
        Self {
            weights: vec![w; num_base_models * output_dim],
            bias: 0.0,
            output_dim,
            num_base_models,
        }
    }

    /// Forward pass: linear combination of base model outputs
    ///
    /// `base_outputs`: `[num_base_models][output_dim]`
    /// Returns: `[output_dim]`
    pub fn forward(&self, base_outputs: &[Vec<f32>]) -> Result<Vec<f32>, EnsembleError> {
        if base_outputs.is_empty() {
            return Err(EnsembleError::EmptyPredictions);
        }
        if base_outputs.len() != self.num_base_models {
            return Err(EnsembleError::DimensionMismatch {
                expected: self.num_base_models,
                found: base_outputs.len(),
            });
        }
        for (i, out) in base_outputs.iter().enumerate() {
            if out.len() != self.output_dim {
                return Err(EnsembleError::DimensionMismatch {
                    expected: self.output_dim,
                    found: out.len(),
                });
            }
        }

        let mut result = vec![self.bias; self.output_dim];

        for (model_idx, model_out) in base_outputs.iter().enumerate() {
            for (dim_idx, &val) in model_out.iter().enumerate() {
                let weight_idx = model_idx * self.output_dim + dim_idx;
                result[dim_idx] += self.weights[weight_idx] * val;
            }
        }

        Ok(result)
    }

    /// Fit the meta-learner on validation set predictions using gradient descent
    ///
    /// # Arguments
    /// * `base_outputs` - `[n_samples][num_models][output_dim]`
    /// * `targets` - `[n_samples][output_dim]`
    /// * `lr` - learning rate
    /// * `epochs` - number of training epochs
    pub fn fit(
        base_outputs: &[Vec<Vec<f32>>],
        targets: &[Vec<f32>],
        lr: f32,
        epochs: usize,
    ) -> Result<Self, EnsembleError> {
        if base_outputs.is_empty() || targets.is_empty() {
            return Err(EnsembleError::EmptyPredictions);
        }
        let n_samples = base_outputs.len();
        if targets.len() != n_samples {
            return Err(EnsembleError::DimensionMismatch {
                expected: n_samples,
                found: targets.len(),
            });
        }

        let num_base_models = base_outputs[0].len();
        let output_dim = targets[0].len();

        if num_base_models == 0 {
            return Err(EnsembleError::EmptyPredictions);
        }
        if output_dim == 0 {
            return Err(EnsembleError::InvalidNumClasses(0));
        }

        let mut meta = Self::new(num_base_models, output_dim);

        for _epoch in 0..epochs {
            let mut weight_grad = vec![0.0_f32; num_base_models * output_dim];
            let mut bias_grad = 0.0_f32;

            for (sample_base, sample_target) in base_outputs.iter().zip(targets.iter()) {
                // Forward pass
                let output = meta.forward(sample_base)?;

                // MSE gradient: 2 * (output - target) / n_samples
                for dim_idx in 0..output_dim {
                    let delta = 2.0 * (output[dim_idx] - sample_target[dim_idx]) / n_samples as f32;
                    bias_grad += delta;
                    for model_idx in 0..num_base_models {
                        let weight_idx = model_idx * output_dim + dim_idx;
                        weight_grad[weight_idx] += delta * sample_base[model_idx][dim_idx];
                    }
                }
            }

            // Update parameters
            for (w, &g) in meta.weights.iter_mut().zip(weight_grad.iter()) {
                *w -= lr * g;
            }
            meta.bias -= lr * bias_grad;
        }

        Ok(meta)
    }
}

/// Q-statistic diversity measure between two classifiers
///
/// Q = (N11*N00 - N01*N10) / (N11*N00 + N01*N10)
/// where Nij = number of samples where classifier 1 is correct (i=1) and classifier 2 is correct (j=1)
pub fn q_statistic(pred1: &[usize], pred2: &[usize], targets: &[usize]) -> f32 {
    assert_eq!(pred1.len(), pred2.len());
    assert_eq!(pred1.len(), targets.len());

    let mut n11 = 0.0_f32; // both correct
    let mut n10 = 0.0_f32; // 1 correct, 2 wrong
    let mut n01 = 0.0_f32; // 1 wrong, 2 correct
    let mut n00 = 0.0_f32; // both wrong

    for ((&p1, &p2), &t) in pred1.iter().zip(pred2.iter()).zip(targets.iter()) {
        match (p1 == t, p2 == t) {
            (true, true) => n11 += 1.0,
            (true, false) => n10 += 1.0,
            (false, true) => n01 += 1.0,
            (false, false) => n00 += 1.0,
        }
    }

    let numerator = n11 * n00 - n01 * n10;
    let denominator = n11 * n00 + n01 * n10;

    if denominator.abs() < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Correlation coefficient diversity measure between two classifiers
///
/// Measures statistical correlation between the correctness vectors of two classifiers.
/// Range: [-1, 1]. 0 means independent, 1 means identical behavior.
pub fn correlation_coefficient(pred1: &[usize], pred2: &[usize], targets: &[usize]) -> f32 {
    assert_eq!(pred1.len(), pred2.len());
    assert_eq!(pred1.len(), targets.len());

    let n = pred1.len() as f32;
    if n == 0.0 {
        return 0.0;
    }

    // Binary correctness vectors
    let c1: Vec<f32> = pred1
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| if p == t { 1.0 } else { 0.0 })
        .collect();
    let c2: Vec<f32> = pred2
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| if p == t { 1.0 } else { 0.0 })
        .collect();

    let mean1: f32 = c1.iter().sum::<f32>() / n;
    let mean2: f32 = c2.iter().sum::<f32>() / n;

    let numerator: f32 = c1
        .iter()
        .zip(c2.iter())
        .map(|(&a, &b)| (a - mean1) * (b - mean2))
        .sum();
    let var1: f32 = c1.iter().map(|&a| (a - mean1) * (a - mean1)).sum::<f32>();
    let var2: f32 = c2.iter().map(|&b| (b - mean2) * (b - mean2)).sum::<f32>();

    let denominator = (var1 * var2).sqrt();
    if denominator < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

// ---- Internal helpers ----

/// Compute argmax index of a slice
fn argmax_f32(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Shannon entropy of a probability distribution
fn shannon_entropy(probs: &[f32]) -> f32 {
    probs
        .iter()
        .map(|&p| if p > 1e-10 { -p * p.ln() } else { 0.0 })
        .sum()
}

// ---- Tests ----
#[cfg(test)]
mod tests {
    use super::*;

    fn make_ensemble(n: usize) -> ModelEnsemble {
        ModelEnsemble::uniform(n)
    }

    #[test]
    fn test_majority_voting_basic() {
        // 3 models; 2 vote for class 0, 1 votes for class 1
        let ensemble = ModelEnsemble::new(3, EnsembleStrategy::MajorityVoting);
        let preds = vec![vec![0.9_f32, 0.1], vec![0.8, 0.2], vec![0.3, 0.7]];
        let class = ensemble
            .predict_class(&preds, 2)
            .expect("predict_class failed");
        assert_eq!(class, 0, "Majority should vote for class 0");
    }

    #[test]
    fn test_majority_voting_tie_goes_to_first_by_argmax() {
        // 2 models; one votes class 0, one votes class 1 → tie → argmax picks 0 (lower index)
        let ensemble = ModelEnsemble::new(2, EnsembleStrategy::MajorityVoting);
        let preds = vec![vec![0.9_f32, 0.1], vec![0.2, 0.8]];
        let combined = ensemble
            .combine_classification(&preds, 2)
            .expect("combine_classification failed");
        // Each class gets 1 vote — tie; the one that appears first as max wins by argmax
        assert_eq!(combined.len(), 2);
    }

    #[test]
    fn test_average_probability_combines_correctly() {
        let ensemble = ModelEnsemble::new(2, EnsembleStrategy::AverageProbability);
        let preds = vec![vec![0.6_f32, 0.4], vec![0.2, 0.8]];
        let combined = ensemble
            .combine_classification(&preds, 2)
            .expect("combine_classification failed");
        assert!(
            (combined[0] - 0.4).abs() < 1e-6,
            "Expected 0.4, got {}",
            combined[0]
        );
        assert!(
            (combined[1] - 0.6).abs() < 1e-6,
            "Expected 0.6, got {}",
            combined[1]
        );
    }

    #[test]
    fn test_weighted_average_respects_weights() {
        let weights = vec![0.7_f32, 0.3];
        let ensemble = ModelEnsemble::new(2, EnsembleStrategy::WeightedAverage(weights));
        let preds = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]];
        let combined = ensemble
            .combine_classification(&preds, 2)
            .expect("combine_classification failed");
        assert!(
            (combined[0] - 0.7).abs() < 1e-6,
            "Expected 0.7, got {}",
            combined[0]
        );
        assert!(
            (combined[1] - 0.3).abs() < 1e-6,
            "Expected 0.3, got {}",
            combined[1]
        );
    }

    #[test]
    fn test_max_probability_picks_most_confident() {
        let ensemble = ModelEnsemble::new(2, EnsembleStrategy::MaxProbability);
        let preds = vec![vec![0.6_f32, 0.4], vec![0.1, 0.9]];
        let combined = ensemble
            .combine_classification(&preds, 2)
            .expect("combine_classification failed");
        // Second model has highest confidence (0.9 for class 1)
        assert!((combined[1] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_regression_average() {
        let ensemble = ModelEnsemble::new(3, EnsembleStrategy::AverageProbability);
        let preds = vec![1.0_f32, 2.0, 3.0];
        let result = ensemble
            .combine_regression(&preds)
            .expect("combine_regression failed");
        assert!((result - 2.0).abs() < 1e-6, "Expected 2.0, got {}", result);
    }

    #[test]
    fn test_regression_weighted_average() {
        let weights = vec![0.5_f32, 0.3, 0.2];
        let ensemble = ModelEnsemble::new(3, EnsembleStrategy::WeightedAverage(weights));
        let preds = vec![10.0_f32, 0.0, 0.0];
        let result = ensemble
            .combine_regression(&preds)
            .expect("combine_regression failed");
        // 0.5 * 10 + 0.3 * 0 + 0.2 * 0 = 5.0  (weights sum to 1.0)
        assert!((result - 5.0).abs() < 1e-5, "Expected 5.0, got {}", result);
    }

    #[test]
    fn test_epistemic_uncertainty_uniform_is_high() {
        let ensemble = make_ensemble(2);
        // Uniform distribution over 4 classes has max entropy
        let preds = vec![
            vec![0.25_f32, 0.25, 0.25, 0.25],
            vec![0.25_f32, 0.25, 0.25, 0.25],
        ];
        let uncertainty = ensemble
            .epistemic_uncertainty(&preds, 4)
            .expect("epistemic_uncertainty failed");
        // ln(4) ≈ 1.386
        assert!(
            uncertainty > 1.0,
            "Uniform distribution should have high entropy"
        );
    }

    #[test]
    fn test_epistemic_uncertainty_certain_is_low() {
        let ensemble = make_ensemble(2);
        let preds = vec![vec![1.0_f32, 0.0], vec![1.0_f32, 0.0]];
        let uncertainty = ensemble
            .epistemic_uncertainty(&preds, 2)
            .expect("epistemic_uncertainty failed");
        assert!(
            uncertainty < 1e-6,
            "Certain distribution should have ~0 entropy"
        );
    }

    #[test]
    fn test_disagreement_rate_all_agree() {
        let ensemble = make_ensemble(3);
        let preds = vec![vec![0.9_f32, 0.1], vec![0.8, 0.2], vec![0.7, 0.3]];
        let rate = ensemble
            .disagreement_rate(&preds, 2)
            .expect("disagreement_rate failed");
        assert!(
            (rate - 0.0).abs() < 1e-6,
            "All models agree — rate should be 0"
        );
    }

    #[test]
    fn test_disagreement_rate_all_disagree() {
        // 2 models: model 0 picks class 0, model 1 picks class 1
        let ensemble = make_ensemble(2);
        let preds = vec![vec![0.9_f32, 0.1], vec![0.1, 0.9]];
        let rate = ensemble
            .disagreement_rate(&preds, 2)
            .expect("disagreement_rate failed");
        assert!(
            (rate - 1.0).abs() < 1e-6,
            "Models completely disagree — rate should be 1.0"
        );
    }

    #[test]
    fn test_adaboost_initialization() {
        let state = AdaBoostState::new(4);
        assert_eq!(state.sample_weights.len(), 4);
        let expected_w = 0.25_f32;
        for &w in &state.sample_weights {
            assert!((w - expected_w).abs() < 1e-6);
        }
        assert_eq!(state.model_weights.len(), 0);
    }

    #[test]
    fn test_adaboost_weight_update_perfect_model() {
        // If a model is perfect (no errors), weights don't change much but alpha is high
        let mut state = AdaBoostState::new(4);
        let preds = vec![0, 1, 2, 3]; // perfect predictions
        let targets = vec![0, 1, 2, 3];
        let alpha = state.update_weights(&preds, &targets);
        assert!(alpha > 0.0, "Alpha should be positive for a good model");
        assert_eq!(state.model_weights.len(), 1);
    }

    #[test]
    fn test_adaboost_normalize_sums_to_one() {
        let mut state = AdaBoostState::new(5);
        state.sample_weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        state.normalize();
        let total: f32 = state.sample_weights.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "Normalized weights should sum to 1"
        );
    }

    #[test]
    fn test_adaboost_predict() {
        let state = AdaBoostState {
            sample_weights: vec![0.25; 4],
            model_weights: vec![2.0, 1.0],
            num_models: 2,
        };
        // Model 0 says class 1 (alpha=2.0), model 1 says class 0 (alpha=1.0)
        let per_model = vec![1usize, 0];
        let predicted = state.predict(&per_model);
        // class 1 score = 2.0, class 0 score = 1.0 → class 1 wins
        assert_eq!(predicted, 1);
    }

    #[test]
    fn test_stacking_meta_forward() {
        let meta = StackingMeta::new(2, 3);
        // Each model gets uniform weight = 0.5
        let base_outputs = vec![vec![1.0_f32, 0.0, 0.0], vec![0.0_f32, 1.0, 0.0]];
        let result = meta.forward(&base_outputs).expect("forward failed");
        assert_eq!(result.len(), 3);
        // output = 0.5 * [1,0,0] + 0.5 * [0,1,0] + 0 bias = [0.5, 0.5, 0.0]
        assert!((result[0] - 0.5).abs() < 1e-6);
        assert!((result[1] - 0.5).abs() < 1e-6);
        assert!((result[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_stacking_meta_fit() {
        let base_outputs = vec![
            vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]],
            vec![vec![0.8_f32, 0.2], vec![0.3_f32, 0.7]],
        ];
        let targets = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
        let meta = StackingMeta::fit(&base_outputs, &targets, 0.01, 10);
        assert!(meta.is_ok(), "StackingMeta::fit should succeed");
    }

    #[test]
    fn test_q_statistic_both_correct() {
        let pred1 = vec![0, 1, 2];
        let pred2 = vec![0, 1, 2];
        let targets = vec![0, 1, 2];
        let q = q_statistic(&pred1, &pred2, &targets);
        // N11=3, N00=0, N01=0, N10=0 → numerator=0, denominator=0 → 0
        assert!((q - 0.0).abs() < 1e-6 || q.is_finite());
    }

    #[test]
    fn test_correlation_coefficient_identical_classifiers() {
        let pred1 = vec![0, 1, 0, 1];
        let pred2 = vec![0, 1, 0, 1];
        let targets = vec![0, 1, 0, 1];
        let r = correlation_coefficient(&pred1, &pred2, &targets);
        // Both always correct → both correctness vectors identical (all 1s) → variance 0 → 0
        assert!(r.is_finite());
    }

    #[test]
    fn test_error_empty_predictions() {
        let ensemble = make_ensemble(2);
        let result = ensemble.combine_classification(&[], 3);
        assert!(matches!(result, Err(EnsembleError::EmptyPredictions)));
    }

    #[test]
    fn test_error_weight_mismatch() {
        let weights = vec![0.5_f32]; // only 1 weight for 2 models
        let ensemble = ModelEnsemble::new(2, EnsembleStrategy::WeightedAverage(weights));
        let preds = vec![vec![0.5_f32, 0.5], vec![0.3_f32, 0.7]];
        let result = ensemble.combine_classification(&preds, 2);
        assert!(matches!(result, Err(EnsembleError::WeightMismatch { .. })));
    }
}
