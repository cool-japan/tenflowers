//! Conformal Prediction Module — Round 13 Track B.
//!
//! This module provides a comprehensive, production-grade implementation of
//! conformal prediction methods for uncertainty quantification in ML systems.
//!
//! # Methods Provided
//!
//! - **Split (Inductive) Conformal Prediction**: Simple, computationally efficient
//!   approach with marginal coverage guarantee P(Y ∈ Ĉ(X)) ≥ 1−α.
//! - **Conformalized Quantile Regression (CQR)**: Adapts quantile regression to
//!   produce adaptive-width intervals via conformalization.
//! - **Adaptive Prediction Sets (APS)**: Classification analog of CP — produces
//!   minimal prediction sets containing the true label with probability ≥ 1−α.
//! - **Cross-Conformal Prediction**: K-fold cross-validation variant for improved
//!   calibration set utilization.
//! - **Coverage Diagnostics**: Full suite of empirical coverage and interval-width
//!   statistics.
//!
//! # Design Principles
//!
//! * No `unwrap()` — all fallible paths surface a `TensorError`.
//! * No `unsafe` code anywhere.
//! * Randomness via `scirs2_core::random` (never the `rand` crate directly).
//! * File stays below 2 000 lines per the project refactoring policy.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::TensorError;

// ─────────────────────────────────────────────────────────────────────────────
// Section 1 — Score Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Nonconformity score function used to measure how unusual a prediction is.
pub trait ScoreFunction: Send + Sync {
    /// Compute the nonconformity score for one (prediction, true_value) pair.
    fn score(&self, prediction: f64, true_value: f64) -> f64;
}

/// Absolute residual: `|y − ŷ|`. The most common score for regression CP.
pub struct AbsoluteResidual;

impl ScoreFunction for AbsoluteResidual {
    #[inline]
    fn score(&self, prediction: f64, true_value: f64) -> f64 {
        (true_value - prediction).abs()
    }
}

/// Signed residual: `y − ŷ`. Useful for asymmetric interval construction.
pub struct SignedResidual;

impl ScoreFunction for SignedResidual {
    #[inline]
    fn score(&self, prediction: f64, true_value: f64) -> f64 {
        true_value - prediction
    }
}

/// Normalized residual: `|y − ŷ| / σ̂` where σ̂ is a per-point uncertainty estimate.
///
/// Because the score function API is stateless (receives only the scalar
/// prediction/true_value pair), normalization is approximated by the global mean σ̂.
/// For index-aware normalization use the raw scores and divide externally.
pub struct NormalizedResidual {
    /// Per-point uncertainty estimates (standard deviations).
    pub sigma: Vec<f64>,
    /// Mean σ̂ pre-computed for the stateless `score()` path.
    mean_sigma: f64,
}

impl NormalizedResidual {
    /// Create from a vector of per-point uncertainty estimates.
    ///
    /// Returns an error if `sigma` is empty or contains non-positive values.
    pub fn new(sigma: Vec<f64>) -> Result<Self, TensorError> {
        if sigma.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "NormalizedResidual::new",
                "sigma vector must not be empty",
            ));
        }
        for &s in &sigma {
            if s <= 0.0 {
                return Err(TensorError::invalid_argument_op(
                    "NormalizedResidual::new",
                    "all sigma values must be strictly positive",
                ));
            }
        }
        let mean_sigma = sigma.iter().sum::<f64>() / sigma.len() as f64;
        Ok(Self { sigma, mean_sigma })
    }

    /// Compute the normalized score using the index-specific σ̂.
    ///
    /// Returns an error if `idx` is out of range.
    pub fn score_indexed(
        &self,
        prediction: f64,
        true_value: f64,
        idx: usize,
    ) -> Result<f64, TensorError> {
        if idx >= self.sigma.len() {
            return Err(TensorError::invalid_argument_op(
                "NormalizedResidual::score_indexed",
                &format!(
                    "index {} out of range for sigma of length {}",
                    idx,
                    self.sigma.len()
                ),
            ));
        }
        Ok((true_value - prediction).abs() / self.sigma[idx])
    }
}

impl ScoreFunction for NormalizedResidual {
    /// Stateless score: uses the mean σ̂ for normalization.
    #[inline]
    fn score(&self, prediction: f64, true_value: f64) -> f64 {
        (true_value - prediction).abs() / self.mean_sigma
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 2 — Split (Inductive) Conformal Prediction
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`SplitConformal`].
#[derive(Debug, Clone)]
pub struct SplitConformalConfig {
    /// Target miscoverage rate α ∈ (0, 1). Guarantees coverage ≥ 1 − α.
    pub alpha: f64,
    /// Fraction of data reserved for calibration when the caller does not
    /// provide a pre-split calibration set.
    pub calibration_ratio: f64,
    /// RNG seed (for any internal shuffling).
    pub seed: u64,
}

impl Default for SplitConformalConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            calibration_ratio: 0.2,
            seed: 42,
        }
    }
}

/// A conformal prediction interval `[lower, upper]` together with metadata.
#[derive(Debug, Clone)]
pub struct ConformalPredictionInterval {
    /// Lower bound of the prediction interval.
    pub lower: f64,
    /// Upper bound of the prediction interval.
    pub upper: f64,
    /// Interval half-width: `upper − lower`.
    pub width: f64,
    /// Theoretical marginal coverage guarantee: `1 − α`.
    pub coverage_guarantee: f64,
}

/// Split (inductive) conformal predictor for regression.
///
/// After [`SplitConformal::calibrate`] is called, [`SplitConformal::predict_interval`]
/// produces intervals with marginal coverage ≥ 1 − α.
pub struct SplitConformal {
    config: SplitConformalConfig,
    /// Calibration quantile q̂ = ⌈(1−α)(n+1)⌉/n quantile of calibration scores.
    quantile: Option<f64>,
}

impl SplitConformal {
    /// Create a new `SplitConformal` predictor with the given configuration.
    pub fn new(config: SplitConformalConfig) -> Self {
        Self {
            config,
            quantile: None,
        }
    }

    /// Calibrate the predictor using held-out calibration predictions and
    /// their corresponding true values.
    ///
    /// Computes nonconformity scores `s_i = |y_i − ŷ_i|` and stores the
    /// `⌈(1−α)(n+1)⌉/n` empirical quantile (treating ∞ as the (n+1)-th score).
    ///
    /// Returns the calibration quantile q̂.
    pub fn calibrate(
        &mut self,
        predictions: &[f64],
        true_values: &[f64],
    ) -> Result<f64, TensorError> {
        if predictions.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "SplitConformal::calibrate",
                "calibration set must not be empty",
            ));
        }
        if predictions.len() != true_values.len() {
            return Err(TensorError::invalid_argument_op(
                "SplitConformal::calibrate",
                &format!(
                    "predictions length {} != true_values length {}",
                    predictions.len(),
                    true_values.len()
                ),
            ));
        }

        let n = predictions.len();
        let mut scores: Vec<f64> = predictions
            .iter()
            .zip(true_values.iter())
            .map(|(&p, &y)| (y - p).abs())
            .collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // q_hat = ceil((1-α)(n+1)) / n quantile
        // Equivalently: take the index k = ceil((1-α)(n+1)) - 1 (0-based) into
        // the sorted scores. If k >= n, q_hat = +∞.
        let level = self.config.alpha;
        let k_float = ((1.0 - level) * (n as f64 + 1.0)).ceil();
        let k = k_float as usize; // 1-based index

        let q_hat = if k == 0 {
            // Extremely small α — use the smallest score
            scores[0]
        } else if k > n {
            f64::INFINITY
        } else {
            scores[k - 1] // convert 1-based to 0-based
        };

        self.quantile = Some(q_hat);
        Ok(q_hat)
    }

    /// Produce the conformal interval for a single new prediction.
    ///
    /// Requires that [`SplitConformal::calibrate`] has been called first.
    pub fn predict_interval(
        &self,
        prediction: f64,
    ) -> Result<ConformalPredictionInterval, TensorError> {
        let q = self.quantile.ok_or_else(|| {
            TensorError::invalid_argument_op(
                "SplitConformal::predict_interval",
                "calibrate() must be called before predict_interval()",
            )
        })?;

        let lower = prediction - q;
        let upper = prediction + q;
        let width = upper - lower;

        Ok(ConformalPredictionInterval {
            lower,
            upper,
            width,
            coverage_guarantee: 1.0 - self.config.alpha,
        })
    }

    /// Produce conformal intervals for a batch of new predictions.
    pub fn predict_intervals(
        &self,
        predictions: &[f64],
    ) -> Result<Vec<ConformalPredictionInterval>, TensorError> {
        predictions
            .iter()
            .map(|&p| self.predict_interval(p))
            .collect()
    }

    /// Compute the empirical coverage of the predictor on a test set.
    ///
    /// Returns the fraction of test points whose true value falls inside the
    /// conformal interval.
    pub fn empirical_coverage(
        &self,
        predictions: &[f64],
        true_values: &[f64],
    ) -> Result<f64, TensorError> {
        if predictions.len() != true_values.len() {
            return Err(TensorError::invalid_argument_op(
                "SplitConformal::empirical_coverage",
                &format!(
                    "predictions length {} != true_values length {}",
                    predictions.len(),
                    true_values.len()
                ),
            ));
        }
        if predictions.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "SplitConformal::empirical_coverage",
                "test set must not be empty",
            ));
        }

        let intervals = self.predict_intervals(predictions)?;
        let covered = intervals
            .iter()
            .zip(true_values.iter())
            .filter(|(interval, &y)| y >= interval.lower && y <= interval.upper)
            .count();

        Ok(covered as f64 / predictions.len() as f64)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3 — Conformalized Quantile Regression (CQR)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`ConformalizingQr`].
#[derive(Debug, Clone)]
pub struct CqrConfig {
    /// Target miscoverage rate α ∈ (0, 1).
    pub alpha: f64,
    /// Lower quantile level α/2.
    pub lower_quantile: f64,
    /// Upper quantile level 1−α/2.
    pub upper_quantile: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Default for CqrConfig {
    fn default() -> Self {
        let alpha = 0.1_f64;
        Self {
            alpha,
            lower_quantile: alpha / 2.0,
            upper_quantile: 1.0 - alpha / 2.0,
            seed: 42,
        }
    }
}

/// A CQR prediction interval with provenance from the quantile models.
#[derive(Debug, Clone)]
pub struct CqrInterval {
    /// Final lower bound after conformalization.
    pub lower: f64,
    /// Final upper bound after conformalization.
    pub upper: f64,
    /// Interval width.
    pub width: f64,
    /// Raw lower quantile prediction (before adjustment).
    pub lower_quantile_pred: f64,
    /// Raw upper quantile prediction (before adjustment).
    pub upper_quantile_pred: f64,
}

/// Linear quantile regression model trained via pinball-loss gradient descent.
///
/// The model predicts quantile `τ` by minimising the pinball (check) loss.
pub struct QuantileModel {
    weights: Vec<f64>,
    bias: f64,
    /// Quantile level τ ∈ (0, 1).
    pub quantile: f64,
}

impl QuantileModel {
    /// Create a new (untrained) model for the given feature dimension and quantile.
    pub fn new(n_features: usize, quantile: f64) -> Self {
        Self {
            weights: vec![0.0; n_features],
            bias: 0.0,
            quantile,
        }
    }

    /// Pinball (check) loss gradient w.r.t. a scalar residual `u = y − ŷ`.
    ///
    /// `∂L_τ/∂ŷ = τ − 𝟙[u < 0]`
    #[inline]
    fn pinball_gradient(residual: f64, quantile: f64) -> f64 {
        if residual >= 0.0 {
            -quantile
        } else {
            1.0 - quantile
        }
    }

    /// Pinball loss value for monitoring (not used during training directly).
    pub fn pinball_loss(residual: f64, quantile: f64) -> f64 {
        if residual >= 0.0 {
            quantile * residual
        } else {
            (quantile - 1.0) * residual
        }
    }

    /// Fit the quantile model on `(x, y)` data via gradient descent.
    ///
    /// `n_iter` — number of stochastic gradient descent passes over the data.
    /// `lr`     — learning rate.
    pub fn fit(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
        n_iter: usize,
        lr: f64,
    ) -> Result<(), TensorError> {
        if x.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "QuantileModel::fit",
                "training data must not be empty",
            ));
        }
        if x.len() != y.len() {
            return Err(TensorError::invalid_argument_op(
                "QuantileModel::fit",
                &format!("x has {} rows but y has {} elements", x.len(), y.len()),
            ));
        }

        let n_features = x[0].len();
        if n_features != self.weights.len() {
            return Err(TensorError::invalid_argument_op(
                "QuantileModel::fit",
                &format!(
                    "model has {} features but data has {}",
                    self.weights.len(),
                    n_features
                ),
            ));
        }

        let n = x.len();
        let quantile = self.quantile;

        for _ in 0..n_iter {
            for i in 0..n {
                let xi = &x[i];
                let yi = y[i];

                // Forward: ŷ = w·x + b
                let y_hat: f64 = xi
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(&xi_j, &w_j)| xi_j * w_j)
                    .sum::<f64>()
                    + self.bias;

                let residual = yi - y_hat;
                let grad_scale = Self::pinball_gradient(residual, quantile);

                // Gradient descent step: w -= lr * grad_scale * x (negative grad of pinball)
                // The gradient w.r.t. ŷ is -grad_scale, so gradient w.r.t. w is grad_scale * x
                // We minimise, so w := w - lr * ∂L/∂w = w - lr * grad_scale * x
                for j in 0..n_features {
                    self.weights[j] -= lr * grad_scale * xi[j];
                }
                self.bias -= lr * grad_scale;
            }
        }

        Ok(())
    }

    /// Predict the conditional quantile for each row in `x`.
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, TensorError> {
        if x.is_empty() {
            return Ok(Vec::new());
        }

        let n_features = self.weights.len();
        x.iter()
            .enumerate()
            .map(|(i, xi)| {
                if xi.len() != n_features {
                    return Err(TensorError::invalid_argument_op(
                        "QuantileModel::predict",
                        &format!(
                            "row {} has {} features, expected {}",
                            i,
                            xi.len(),
                            n_features
                        ),
                    ));
                }
                let y_hat: f64 = xi
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(&xi_j, &w_j)| xi_j * w_j)
                    .sum::<f64>()
                    + self.bias;
                Ok(y_hat)
            })
            .collect()
    }
}

/// Conformalized Quantile Regression predictor.
///
/// Trains two [`QuantileModel`]s (lower and upper quantiles) and then
/// conformally adjusts their outputs using a held-out calibration set.
pub struct ConformalizingQr {
    config: CqrConfig,
    lower_model: QuantileModel,
    upper_model: QuantileModel,
    /// Conformal adjustment q̂ added symmetrically to the quantile bounds.
    quantile_adjustment: Option<f64>,
}

impl ConformalizingQr {
    /// Create a new `ConformalizingQr` with the given configuration.
    ///
    /// `n_features` — number of input features (determines model size).
    pub fn new(config: CqrConfig, n_features: usize) -> Self {
        let lower_model = QuantileModel::new(n_features, config.lower_quantile);
        let upper_model = QuantileModel::new(n_features, config.upper_quantile);
        Self {
            config,
            lower_model,
            upper_model,
            quantile_adjustment: None,
        }
    }

    /// Fit the two quantile models on the training data.
    pub fn fit(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
        n_iter: usize,
        lr: f64,
    ) -> Result<(), TensorError> {
        self.lower_model.fit(x, y, n_iter, lr)?;
        self.upper_model.fit(x, y, n_iter, lr)?;
        Ok(())
    }

    /// Calibrate the conformal adjustment using a held-out calibration set.
    ///
    /// CQR nonconformity score:
    /// `s_i = max(q_{α/2}(x_i) − y_i,  y_i − q_{1−α/2}(x_i))`
    pub fn calibrate(&mut self, x_cal: &[Vec<f64>], y_cal: &[f64]) -> Result<(), TensorError> {
        if x_cal.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "ConformalizingQr::calibrate",
                "calibration set must not be empty",
            ));
        }
        if x_cal.len() != y_cal.len() {
            return Err(TensorError::invalid_argument_op(
                "ConformalizingQr::calibrate",
                &format!(
                    "x_cal has {} rows but y_cal has {} elements",
                    x_cal.len(),
                    y_cal.len()
                ),
            ));
        }

        let lower_preds = self.lower_model.predict(x_cal)?;
        let upper_preds = self.upper_model.predict(x_cal)?;

        let n = x_cal.len();
        let mut scores: Vec<f64> = (0..n)
            .map(|i| {
                let lo = lower_preds[i];
                let hi = upper_preds[i];
                let y = y_cal[i];
                f64::max(lo - y, y - hi)
            })
            .collect();

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let level = self.config.alpha;
        let k_float = ((1.0 - level) * (n as f64 + 1.0)).ceil();
        let k = k_float as usize;

        let q_hat = if k == 0 {
            scores[0]
        } else if k > n {
            f64::INFINITY
        } else {
            scores[k - 1]
        };

        self.quantile_adjustment = Some(q_hat);
        Ok(())
    }

    /// Produce a conformalized interval for a single input vector.
    pub fn predict_interval(&self, x: &[f64]) -> Result<CqrInterval, TensorError> {
        let q_hat = self.quantile_adjustment.ok_or_else(|| {
            TensorError::invalid_argument_op(
                "ConformalizingQr::predict_interval",
                "calibrate() must be called before predict_interval()",
            )
        })?;

        let x_row = vec![x.to_vec()];
        let lo_preds = self.lower_model.predict(&x_row)?;
        let hi_preds = self.upper_model.predict(&x_row)?;

        let lo_raw = lo_preds[0];
        let hi_raw = hi_preds[0];

        let lower = lo_raw - q_hat;
        let upper = hi_raw + q_hat;
        let width = upper - lower;

        Ok(CqrInterval {
            lower,
            upper,
            width,
            lower_quantile_pred: lo_raw,
            upper_quantile_pred: hi_raw,
        })
    }

    /// Produce conformalized intervals for a batch of input vectors.
    pub fn predict_intervals(&self, x: &[Vec<f64>]) -> Result<Vec<CqrInterval>, TensorError> {
        x.iter().map(|xi| self.predict_interval(xi)).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4 — Adaptive Prediction Sets (Classification)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`AdaptivePredictionSets`].
#[derive(Debug, Clone)]
pub struct AdaptivePredictionSetConfig {
    /// Target miscoverage rate α.
    pub alpha: f64,
    /// Total number of classes.
    pub n_classes: usize,
    /// RAPS regularization parameter λ (Angelopoulos et al., 2021).
    pub regularization: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Default for AdaptivePredictionSetConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            n_classes: 10,
            regularization: 0.01,
            seed: 42,
        }
    }
}

/// A conformal prediction set for a classification problem.
#[derive(Debug, Clone)]
pub struct PredictionSet {
    /// Class indices included in the prediction set.
    pub classes: Vec<usize>,
    /// Theoretical marginal coverage guarantee: `1 − α`.
    pub coverage_guarantee: f64,
    /// Number of classes in the set.
    pub set_size: usize,
}

/// Adaptive prediction sets (APS) conformal predictor for classification.
///
/// Uses cumulative softmax thresholding (THR) as the nonconformity score:
/// `s_i = cumsum(sorted_probs)` up to (and including) the true label's rank.
pub struct AdaptivePredictionSets {
    config: AdaptivePredictionSetConfig,
    /// Calibration quantile for cumulative softmax threshold.
    quantile: Option<f64>,
}

impl AdaptivePredictionSets {
    /// Create a new `AdaptivePredictionSets` predictor.
    pub fn new(config: AdaptivePredictionSetConfig) -> Self {
        Self {
            config,
            quantile: None,
        }
    }

    /// Calibrate using softmax score vectors and their true class labels.
    ///
    /// Score for sample `i`:
    /// `s_i = Σ_{j : π_j ≤ rank(y_i)} softmax_j(x_i) + u_i * softmax_{y_i}(x_i)`
    /// where `u_i ~ Uniform[0, 1]` introduces randomness to break ties.
    ///
    /// Here we use a deterministic (u_i = 0) approximation for reproducibility,
    /// controlled by the seed. With regularization λ we add
    /// `λ * (rank(y_i) - 1)` to the score (RAPS variant).
    pub fn calibrate(
        &mut self,
        softmax_scores: &[Vec<f64>],
        true_labels: &[usize],
    ) -> Result<(), TensorError> {
        if softmax_scores.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "AdaptivePredictionSets::calibrate",
                "calibration set must not be empty",
            ));
        }
        if softmax_scores.len() != true_labels.len() {
            return Err(TensorError::invalid_argument_op(
                "AdaptivePredictionSets::calibrate",
                &format!(
                    "softmax_scores has {} rows but true_labels has {} elements",
                    softmax_scores.len(),
                    true_labels.len()
                ),
            ));
        }

        let n_classes = self.config.n_classes;
        let lambda = self.config.regularization;
        let mut rng = StdRng::seed_from_u64(self.config.seed);

        let n = softmax_scores.len();
        let mut scores: Vec<f64> = Vec::with_capacity(n);

        for i in 0..n {
            let probs = &softmax_scores[i];
            let true_label = true_labels[i];

            if probs.len() < n_classes {
                return Err(TensorError::invalid_argument_op(
                    "AdaptivePredictionSets::calibrate",
                    &format!(
                        "row {} has {} probabilities but n_classes={}",
                        i,
                        probs.len(),
                        n_classes
                    ),
                ));
            }
            if true_label >= n_classes {
                return Err(TensorError::invalid_argument_op(
                    "AdaptivePredictionSets::calibrate",
                    &format!(
                        "true_label {} >= n_classes {} at index {}",
                        true_label, n_classes, i
                    ),
                ));
            }

            // Sort class indices by descending probability (π ordering).
            let mut order: Vec<usize> = (0..n_classes).collect();
            order.sort_by(|&a, &b| {
                probs[b]
                    .partial_cmp(&probs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Find rank of true label in sorted order (1-based).
            let true_rank = order
                .iter()
                .position(|&c| c == true_label)
                .unwrap_or(n_classes - 1);

            // Cumulative probability up to and including the true label's rank.
            let cum_prob: f64 = order[..=true_rank].iter().map(|&c| probs[c]).sum();

            // Randomised score (u breaks ties, controlling set size coverage).
            let u: f64 = rng.random::<f64>();
            let raps_penalty = lambda * true_rank as f64;
            let score = cum_prob - probs[true_label] * u + raps_penalty;

            scores.push(score);
        }

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let level = self.config.alpha;
        let k_float = ((1.0 - level) * (n as f64 + 1.0)).ceil();
        let k = k_float as usize;

        let q_hat = if k == 0 {
            scores[0]
        } else if k > n {
            f64::INFINITY
        } else {
            scores[k - 1]
        };

        self.quantile = Some(q_hat);
        Ok(())
    }

    /// Predict the conformal prediction set for a single softmax vector.
    ///
    /// Includes classes in descending probability order until the cumulative
    /// sum exceeds the calibration threshold q̂.
    pub fn predict_set(&self, softmax_scores: &[f64]) -> Result<PredictionSet, TensorError> {
        let q_hat = self.quantile.ok_or_else(|| {
            TensorError::invalid_argument_op(
                "AdaptivePredictionSets::predict_set",
                "calibrate() must be called before predict_set()",
            )
        })?;

        let n_classes = self.config.n_classes;
        if softmax_scores.len() < n_classes {
            return Err(TensorError::invalid_argument_op(
                "AdaptivePredictionSets::predict_set",
                &format!(
                    "softmax_scores has {} elements but n_classes={}",
                    softmax_scores.len(),
                    n_classes
                ),
            ));
        }

        // Sort by descending probability.
        let mut order: Vec<usize> = (0..n_classes).collect();
        order.sort_by(|&a, &b| {
            softmax_scores[b]
                .partial_cmp(&softmax_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let lambda = self.config.regularization;
        let mut cum_prob = 0.0_f64;
        let mut classes = Vec::new();

        for (rank, &cls) in order.iter().enumerate() {
            cum_prob += softmax_scores[cls];
            let score = cum_prob + lambda * rank as f64;
            classes.push(cls);
            if score >= q_hat {
                break;
            }
        }

        let set_size = classes.len();
        Ok(PredictionSet {
            classes,
            coverage_guarantee: 1.0 - self.config.alpha,
            set_size,
        })
    }

    /// Predict conformal prediction sets for a batch of softmax vectors.
    pub fn predict_sets(
        &self,
        softmax_scores: &[Vec<f64>],
    ) -> Result<Vec<PredictionSet>, TensorError> {
        softmax_scores.iter().map(|s| self.predict_set(s)).collect()
    }

    /// Compute the average prediction set size across a collection of sets.
    pub fn average_set_size(&self, sets: &[PredictionSet]) -> f64 {
        if sets.is_empty() {
            return 0.0;
        }
        sets.iter().map(|s| s.set_size as f64).sum::<f64>() / sets.len() as f64
    }

    /// Compute the empirical coverage: fraction of sets containing the true label.
    pub fn empirical_coverage(&self, sets: &[PredictionSet], true_labels: &[usize]) -> f64 {
        if sets.is_empty() || sets.len() != true_labels.len() {
            return 0.0;
        }
        let covered = sets
            .iter()
            .zip(true_labels.iter())
            .filter(|(s, &label)| s.classes.contains(&label))
            .count();
        covered as f64 / sets.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5 — Cross-Conformal Prediction
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`CrossConformal`].
#[derive(Debug, Clone)]
pub struct CrossConformalConfig {
    /// Target miscoverage rate α.
    pub alpha: f64,
    /// Number of folds K (typically 5 or 10).
    pub n_folds: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for CrossConformalConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            n_folds: 5,
            seed: 42,
        }
    }
}

/// Cross-conformal predictor.
///
/// In K-fold cross-conformal prediction the data is split into K folds. For
/// each fold k, a model is trained on the other K−1 folds and nonconformity
/// scores are computed on fold k. The final quantile is the average of the K
/// per-fold quantiles.
pub struct CrossConformal {
    config: CrossConformalConfig,
    /// Per-fold calibration quantiles.
    fold_quantiles: Vec<f64>,
}

impl CrossConformal {
    /// Create a new `CrossConformal` predictor.
    pub fn new(config: CrossConformalConfig) -> Self {
        Self {
            config,
            fold_quantiles: Vec::new(),
        }
    }

    /// Calibrate from pre-computed fold predictions.
    ///
    /// `predictions_by_fold[k]` is a `Vec<(prediction, true_value)>` for fold `k`.
    ///
    /// Returns the mean calibration quantile across folds.
    pub fn calibrate(
        &mut self,
        predictions_by_fold: &[Vec<(f64, f64)>],
    ) -> Result<f64, TensorError> {
        if predictions_by_fold.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "CrossConformal::calibrate",
                "predictions_by_fold must not be empty",
            ));
        }

        let level = self.config.alpha;
        self.fold_quantiles.clear();

        for (k, fold) in predictions_by_fold.iter().enumerate() {
            if fold.is_empty() {
                return Err(TensorError::invalid_argument_op(
                    "CrossConformal::calibrate",
                    &format!("fold {} is empty", k),
                ));
            }

            let n = fold.len();
            let mut scores: Vec<f64> = fold
                .iter()
                .map(|&(pred, truth)| (truth - pred).abs())
                .collect();
            scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let k_float = ((1.0 - level) * (n as f64 + 1.0)).ceil();
            let k_idx = k_float as usize;

            let q_hat = if k_idx == 0 {
                scores[0]
            } else if k_idx > n {
                f64::INFINITY
            } else {
                scores[k_idx - 1]
            };

            self.fold_quantiles.push(q_hat);
        }

        // Combined quantile = mean of fold quantiles (standard cross-CP aggregation).
        let mean_q = self.fold_quantiles.iter().sum::<f64>() / self.fold_quantiles.len() as f64;
        Ok(mean_q)
    }

    /// Produce the conformal interval for a single prediction using the mean
    /// cross-fold quantile.
    pub fn predict_interval(
        &self,
        prediction: f64,
    ) -> Result<ConformalPredictionInterval, TensorError> {
        if self.fold_quantiles.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "CrossConformal::predict_interval",
                "calibrate() must be called before predict_interval()",
            ));
        }

        let q = self.fold_quantiles.iter().sum::<f64>() / self.fold_quantiles.len() as f64;
        let lower = prediction - q;
        let upper = prediction + q;
        let width = upper - lower;

        Ok(ConformalPredictionInterval {
            lower,
            upper,
            width,
            coverage_guarantee: 1.0 - self.config.alpha,
        })
    }

    /// Per-fold quantiles computed during calibration.
    pub fn fold_quantiles(&self) -> &[f64] {
        &self.fold_quantiles
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6 — Coverage Diagnostics & Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Comprehensive coverage and interval-width statistics.
#[derive(Debug, Clone)]
pub struct CoverageDiagnostics {
    /// Fraction of test points whose true value falls inside the predicted interval.
    pub empirical_coverage: f64,
    /// Target coverage `1 − α`.
    pub target_coverage: f64,
    /// `empirical_coverage − target_coverage` (positive ⟹ over-covered).
    pub coverage_gap: f64,
    /// Mean interval width across all test points.
    pub mean_interval_width: f64,
    /// Median interval width.
    pub median_interval_width: f64,
    /// Standard deviation of interval widths.
    pub interval_width_std: f64,
    /// Empirical conditional coverage within 10 equal-width bins of sorted
    /// predictions (decile conditional coverage).
    pub coverage_by_bin: Vec<f64>,
}

/// Compute comprehensive coverage diagnostics from a collection of prediction
/// intervals and their corresponding true values.
pub fn compute_coverage_diagnostics(
    intervals: &[ConformalPredictionInterval],
    true_values: &[f64],
    target_coverage: f64,
) -> Result<CoverageDiagnostics, TensorError> {
    if intervals.is_empty() {
        return Err(TensorError::invalid_argument_op(
            "compute_coverage_diagnostics",
            "intervals must not be empty",
        ));
    }
    if intervals.len() != true_values.len() {
        return Err(TensorError::invalid_argument_op(
            "compute_coverage_diagnostics",
            &format!(
                "intervals length {} != true_values length {}",
                intervals.len(),
                true_values.len()
            ),
        ));
    }

    let n = intervals.len();

    // Empirical coverage
    let covered = intervals
        .iter()
        .zip(true_values.iter())
        .filter(|(iv, &y)| y >= iv.lower && y <= iv.upper)
        .count();
    let empirical_coverage = covered as f64 / n as f64;
    let coverage_gap = empirical_coverage - target_coverage;

    // Interval widths
    let widths: Vec<f64> = intervals.iter().map(|iv| iv.width).collect();
    let mean_width = widths.iter().sum::<f64>() / n as f64;

    let mut sorted_widths = widths.clone();
    sorted_widths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_width = if n % 2 == 0 {
        (sorted_widths[n / 2 - 1] + sorted_widths[n / 2]) / 2.0
    } else {
        sorted_widths[n / 2]
    };

    let variance = widths
        .iter()
        .map(|&w| (w - mean_width).powi(2))
        .sum::<f64>()
        / n as f64;
    let width_std = variance.sqrt();

    // Conditional coverage by decile (bin on interval midpoint)
    let n_bins = 10_usize;
    let midpoints: Vec<f64> = intervals
        .iter()
        .map(|iv| (iv.lower + iv.upper) / 2.0)
        .collect();
    let mut mid_sorted: Vec<(f64, usize)> = midpoints
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, m)| (m, i))
        .collect();
    mid_sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let bin_size = (n + n_bins - 1) / n_bins; // ceil division
    let mut coverage_by_bin = Vec::with_capacity(n_bins);
    let mut chunk_start = 0;

    for _bin in 0..n_bins {
        let chunk_end = (chunk_start + bin_size).min(n);
        if chunk_start >= chunk_end {
            break;
        }
        let bin_covered = mid_sorted[chunk_start..chunk_end]
            .iter()
            .filter(|&&(_, orig_idx)| {
                let y = true_values[orig_idx];
                let iv = &intervals[orig_idx];
                y >= iv.lower && y <= iv.upper
            })
            .count();
        coverage_by_bin.push(bin_covered as f64 / (chunk_end - chunk_start) as f64);
        chunk_start = chunk_end;
    }

    Ok(CoverageDiagnostics {
        empirical_coverage,
        target_coverage,
        coverage_gap,
        mean_interval_width: mean_width,
        median_interval_width: median_width,
        interval_width_std: width_std,
        coverage_by_bin,
    })
}

/// Compute the conformal p-value for a test point given calibration scores.
///
/// `p_val = (#{s_i ≥ s_test} + 1) / (n + 1)`
///
/// Under the null hypothesis of exchangeability the p-value is super-uniform.
pub fn conformal_p_value(calibration_scores: &[f64], test_score: f64) -> f64 {
    let n = calibration_scores.len();
    if n == 0 {
        return 1.0;
    }
    let count = calibration_scores
        .iter()
        .filter(|&&s| s >= test_score)
        .count();
    (count as f64 + 1.0) / (n as f64 + 1.0)
}

/// Apply Bonferroni correction to a vector of conformal p-values.
///
/// Returns a boolean mask where `true` indicates the null hypothesis is
/// rejected at the family-wise error rate `alpha`.
///
/// Each individual test uses threshold `alpha / m` where `m` is the number of tests.
pub fn bonferroni_correction(p_values: &[f64], alpha: f64) -> Vec<bool> {
    let m = p_values.len();
    if m == 0 {
        return Vec::new();
    }
    let threshold = alpha / m as f64;
    p_values.iter().map(|&p| p <= threshold).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Build a synthetic regression dataset: y = 2*x + noise.
    fn synthetic_regression(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 2.0 * xi + (rng.random::<f64>() - 0.5) * 0.2)
            .collect();
        (x, y)
    }

    /// Build a synthetic multi-feature dataset: y = w·x + noise.
    fn synthetic_multi_regression(n: usize, d: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let x: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..d).map(|_| rng.random::<f64>()).collect())
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|xi| xi.iter().sum::<f64>() / d as f64 + (rng.random::<f64>() - 0.5) * 0.1)
            .collect();
        (x, y)
    }

    /// Perfect softmax (one-hot) scores.
    fn one_hot_scores(n_classes: usize, true_label: usize) -> Vec<f64> {
        let mut v = vec![0.01 / (n_classes as f64); n_classes];
        v[true_label] = 1.0 - 0.01 * (n_classes as f64 - 1.0) / n_classes as f64;
        // renormalise
        let sum: f64 = v.iter().sum();
        v.iter_mut().for_each(|x| *x /= sum);
        v
    }

    /// Uniform softmax scores.
    fn uniform_scores(n_classes: usize) -> Vec<f64> {
        vec![1.0 / n_classes as f64; n_classes]
    }

    // ── AbsoluteResidual ─────────────────────────────────────────────────────

    #[test]
    fn test_absolute_residual_zero() {
        let sf = AbsoluteResidual;
        assert_eq!(sf.score(3.0, 3.0), 0.0);
    }

    #[test]
    fn test_absolute_residual_positive() {
        let sf = AbsoluteResidual;
        assert!((sf.score(1.0, 4.0) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_absolute_residual_negative() {
        let sf = AbsoluteResidual;
        assert!((sf.score(4.0, 1.0) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_signed_residual() {
        let sf = SignedResidual;
        assert!((sf.score(1.0, 4.0) - 3.0).abs() < 1e-12);
        assert!((sf.score(4.0, 1.0) - (-3.0)).abs() < 1e-12);
    }

    #[test]
    fn test_normalized_residual_new_empty() {
        assert!(NormalizedResidual::new(vec![]).is_err());
    }

    #[test]
    fn test_normalized_residual_new_non_positive() {
        assert!(NormalizedResidual::new(vec![1.0, 0.0, 1.0]).is_err());
        assert!(NormalizedResidual::new(vec![1.0, -1.0, 1.0]).is_err());
    }

    #[test]
    fn test_normalized_residual_score() {
        let nr = NormalizedResidual::new(vec![2.0, 4.0]).expect("valid");
        // mean sigma = 3.0; score = |4-1| / 3.0 = 1.0
        let s = nr.score(1.0, 4.0);
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_residual_score_indexed() {
        let nr = NormalizedResidual::new(vec![2.0, 4.0]).expect("valid");
        let s = nr.score_indexed(1.0, 5.0, 1).expect("ok");
        assert!((s - 1.0).abs() < 1e-10); // |5-1| / 4.0 = 1.0
    }

    #[test]
    fn test_normalized_residual_score_indexed_oob() {
        let nr = NormalizedResidual::new(vec![2.0, 4.0]).expect("valid");
        assert!(nr.score_indexed(1.0, 5.0, 5).is_err());
    }

    // ── SplitConformal ────────────────────────────────────────────────────────

    #[test]
    fn test_split_conformal_basic_calibration() {
        // Use alpha=0.5 so that k = ceil(0.5*(5+1)) = 3 <= n=5 → finite q̂
        let mut cp = SplitConformal::new(SplitConformalConfig {
            alpha: 0.5,
            ..Default::default()
        });
        let preds = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let truth = vec![0.1, 1.2, 2.3, 3.4, 4.5];
        let q = cp.calibrate(&preds, &truth).expect("calibration ok");
        assert!(q > 0.0, "quantile must be positive");
        assert!(
            q.is_finite(),
            "quantile must be finite with alpha=0.5 and n=5"
        );
        // Scores are |truth-preds| = [0.1, 0.2, 0.3, 0.4, 0.5]; sorted → same order
        // k = ceil(0.5 * 6) = 3; q̂ = scores[2] = 0.3
        assert!((q - 0.3).abs() < 1e-10, "expected q̂=0.3, got {}", q);
    }

    #[test]
    fn test_split_conformal_predict_interval_width() {
        let mut cp = SplitConformal::new(SplitConformalConfig {
            alpha: 0.1,
            ..Default::default()
        });
        let preds: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let truth: Vec<f64> = preds.iter().map(|&p| p + 0.05).collect();
        let q = cp.calibrate(&preds, &truth).expect("ok");
        let interval = cp.predict_interval(5.0).expect("ok");
        assert!((interval.width - 2.0 * q).abs() < 1e-10);
    }

    #[test]
    fn test_split_conformal_empirical_coverage() {
        // Build predictions that are close to truth: ŷ_i = y_i + small_noise.
        // This simulates a near-perfect model; conformal calibration then learns
        // a tight q̂ and the resulting intervals have marginal coverage >= 1-α.
        let mut rng = StdRng::seed_from_u64(123);
        let n_total = 400;
        let truth_all: Vec<f64> = (0..n_total)
            .map(|i| 2.0 * (i as f64 / n_total as f64))
            .collect();
        // Predictions = truth + IID noise from Uniform[-0.05, 0.05]
        let preds_all: Vec<f64> = truth_all
            .iter()
            .map(|&y| y + (rng.random::<f64>() - 0.5) * 0.1)
            .collect();

        let (preds_cal, truth_cal) = (preds_all[..200].to_vec(), truth_all[..200].to_vec());
        let (preds_test, truth_test) = (preds_all[200..].to_vec(), truth_all[200..].to_vec());

        let mut cp = SplitConformal::new(SplitConformalConfig {
            alpha: 0.1,
            ..Default::default()
        });
        cp.calibrate(&preds_cal, &truth_cal).expect("ok");

        let coverage = cp.empirical_coverage(&preds_test, &truth_test).expect("ok");
        // Conformal guarantee: coverage >= 1 - alpha = 0.9 (marginal).
        // Allow slack for finite sample variability; theoretical min is ceil((n+1)(1-α))/n.
        assert!(
            coverage >= 0.80,
            "expected coverage >= 0.80 but got {:.3}",
            coverage
        );
    }

    #[test]
    fn test_split_conformal_uncalibrated_error() {
        let cp = SplitConformal::new(SplitConformalConfig::default());
        assert!(cp.predict_interval(1.0).is_err());
    }

    #[test]
    fn test_split_conformal_empty_calibration_error() {
        let mut cp = SplitConformal::new(SplitConformalConfig::default());
        assert!(cp.calibrate(&[], &[]).is_err());
    }

    #[test]
    fn test_split_conformal_mismatched_lengths_error() {
        let mut cp = SplitConformal::new(SplitConformalConfig::default());
        assert!(cp.calibrate(&[1.0, 2.0], &[1.0]).is_err());
    }

    #[test]
    fn test_split_conformal_batch_intervals() {
        let mut cp = SplitConformal::new(SplitConformalConfig {
            alpha: 0.1,
            ..Default::default()
        });
        let preds: Vec<f64> = (0..50).map(|i| i as f64 * 0.02).collect();
        let truth: Vec<f64> = preds.iter().map(|&p| p + 0.01).collect();
        cp.calibrate(&preds, &truth).expect("ok");

        let test_preds = vec![0.5, 1.0, 1.5, 2.0];
        let intervals = cp.predict_intervals(&test_preds).expect("ok");
        assert_eq!(intervals.len(), 4);
        for iv in &intervals {
            assert!(iv.lower < iv.upper);
            assert!(iv.width > 0.0);
        }
    }

    #[test]
    fn test_split_conformal_small_calibration_coverage() {
        // Even with 10 calibration points, CP guarantees marginal coverage
        let mut cp = SplitConformal::new(SplitConformalConfig {
            alpha: 0.2,
            ..Default::default()
        });
        let preds: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let truth: Vec<f64> = preds.iter().map(|&p| p + 0.5).collect();
        let q = cp.calibrate(&preds, &truth).expect("ok");
        assert!(q >= 0.5); // Must capture the 0.5 residual for 80% coverage
    }

    // ── QuantileModel ─────────────────────────────────────────────────────────

    #[test]
    fn test_quantile_model_fit_and_predict() {
        let (x, y) = synthetic_multi_regression(100, 2, 99);
        let mut model = QuantileModel::new(2, 0.5);
        model.fit(&x, &y, 200, 0.01).expect("fit ok");
        let preds = model.predict(&x).expect("predict ok");
        assert_eq!(preds.len(), 100);
    }

    #[test]
    fn test_quantile_model_pinball_loss_decreases() {
        let (x, y) = synthetic_multi_regression(50, 2, 77);
        let mut model = QuantileModel::new(2, 0.5);

        // Compute initial loss
        let initial_preds = model.predict(&x).expect("ok");
        let initial_loss: f64 = initial_preds
            .iter()
            .zip(y.iter())
            .map(|(&p, &yi)| QuantileModel::pinball_loss(yi - p, 0.5))
            .sum::<f64>();

        model.fit(&x, &y, 500, 0.05).expect("fit ok");

        let final_preds = model.predict(&x).expect("ok");
        let final_loss: f64 = final_preds
            .iter()
            .zip(y.iter())
            .map(|(&p, &yi)| QuantileModel::pinball_loss(yi - p, 0.5))
            .sum::<f64>();

        assert!(
            final_loss < initial_loss,
            "loss should decrease: initial={:.4} final={:.4}",
            initial_loss,
            final_loss
        );
    }

    #[test]
    fn test_quantile_model_pinball_loss_values() {
        // Positive residual: L_τ(u) = τ * u
        assert!((QuantileModel::pinball_loss(2.0, 0.75) - 1.5).abs() < 1e-12);
        // Negative residual: L_τ(u) = (τ-1) * u = (1-τ) * (-u)
        assert!((QuantileModel::pinball_loss(-2.0, 0.75) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_model_empty_x_error() {
        let mut model = QuantileModel::new(2, 0.5);
        assert!(model.fit(&[], &[], 10, 0.01).is_err());
    }

    #[test]
    fn test_quantile_model_mismatched_lengths_error() {
        let (x, _) = synthetic_multi_regression(10, 2, 1);
        let y = vec![0.0; 5];
        let mut model = QuantileModel::new(2, 0.5);
        assert!(model.fit(&x, &y, 10, 0.01).is_err());
    }

    // ── ConformalizingQr ──────────────────────────────────────────────────────

    #[test]
    fn test_cqr_end_to_end() {
        let (x, y) = synthetic_multi_regression(200, 2, 42);
        let (x_train, y_train) = (x[..100].to_vec(), y[..100].to_vec());
        let (x_cal, y_cal) = (x[100..150].to_vec(), y[100..150].to_vec());
        let (x_test, y_test) = (x[150..].to_vec(), y[150..].to_vec());

        let config = CqrConfig {
            alpha: 0.1,
            lower_quantile: 0.05,
            upper_quantile: 0.95,
            seed: 42,
        };
        let mut cqr = ConformalizingQr::new(config, 2);
        cqr.fit(&x_train, &y_train, 300, 0.02).expect("fit ok");
        cqr.calibrate(&x_cal, &y_cal).expect("calibrate ok");

        let intervals = cqr.predict_intervals(&x_test).expect("predict ok");
        assert_eq!(intervals.len(), x_test.len());

        let covered = intervals
            .iter()
            .zip(y_test.iter())
            .filter(|(iv, &y)| y >= iv.lower && y <= iv.upper)
            .count();
        let coverage = covered as f64 / x_test.len() as f64;
        assert!(
            coverage >= 0.8,
            "CQR coverage {:.3} should be >= 0.8",
            coverage
        );
    }

    #[test]
    fn test_cqr_uncalibrated_error() {
        let cqr = ConformalizingQr::new(CqrConfig::default(), 2);
        assert!(cqr.predict_interval(&[0.5, 0.5]).is_err());
    }

    #[test]
    fn test_cqr_interval_structure() {
        let (x, y) = synthetic_multi_regression(100, 2, 13);
        let mut cqr = ConformalizingQr::new(CqrConfig::default(), 2);
        cqr.fit(&x[..80], &y[..80], 200, 0.02).expect("fit ok");
        cqr.calibrate(&x[80..], &y[80..]).expect("calibrate ok");

        let iv = cqr.predict_interval(&x[0]).expect("ok");
        assert!(iv.lower < iv.upper);
        assert!(iv.width > 0.0);
        assert!((iv.width - (iv.upper - iv.lower)).abs() < 1e-10);
    }

    // ── AdaptivePredictionSets ────────────────────────────────────────────────

    #[test]
    fn test_aps_calibration_and_coverage() {
        let n_classes = 5;
        let n_cal = 200;
        let mut rng = StdRng::seed_from_u64(7);

        // Calibration: generate random softmax + true label
        let softmax_cal: Vec<Vec<f64>> = (0..n_cal)
            .map(|_| {
                let raw: Vec<f64> = (0..n_classes).map(|_| rng.random::<f64>()).collect();
                let sum: f64 = raw.iter().sum();
                raw.iter().map(|&v| v / sum).collect()
            })
            .collect();
        let true_labels: Vec<usize> = (0..n_cal)
            .map(|_| (rng.random::<u64>() as usize) % n_classes)
            .collect();

        let config = AdaptivePredictionSetConfig {
            alpha: 0.1,
            n_classes,
            regularization: 0.0,
            seed: 42,
        };
        let mut aps = AdaptivePredictionSets::new(config);
        aps.calibrate(&softmax_cal, &true_labels).expect("ok");

        // Test set
        let softmax_test: Vec<Vec<f64>> = (0..100)
            .map(|_| {
                let raw: Vec<f64> = (0..n_classes).map(|_| rng.random::<f64>()).collect();
                let sum: f64 = raw.iter().sum();
                raw.iter().map(|&v| v / sum).collect()
            })
            .collect();
        let true_labels_test: Vec<usize> = (0..100)
            .map(|_| (rng.random::<u64>() as usize) % n_classes)
            .collect();

        let sets = aps.predict_sets(&softmax_test).expect("ok");
        let coverage = aps.empirical_coverage(&sets, &true_labels_test);
        assert!(
            coverage >= 0.75,
            "APS coverage {:.3} should be >= 0.75",
            coverage
        );
    }

    #[test]
    fn test_aps_uniform_softmax_set_size() {
        let n_classes = 10;
        let n_cal = 500;
        let alpha = 0.1;

        // All uniform softmax → sets should include ~90% of classes
        let softmax_cal: Vec<Vec<f64>> = vec![uniform_scores(n_classes); n_cal];
        let true_labels: Vec<usize> = (0..n_cal).map(|i| i % n_classes).collect();

        let config = AdaptivePredictionSetConfig {
            alpha,
            n_classes,
            regularization: 0.0,
            seed: 1,
        };
        let mut aps = AdaptivePredictionSets::new(config);
        aps.calibrate(&softmax_cal, &true_labels).expect("ok");

        let softmax_test: Vec<Vec<f64>> = vec![uniform_scores(n_classes); 100];
        let sets = aps.predict_sets(&softmax_test).expect("ok");
        let avg_size = aps.average_set_size(&sets);

        // With uniform softmax and 10 classes, sets should be non-trivial (>= 1)
        assert!(
            avg_size >= 1.0,
            "average set size {:.2} should be >= 1",
            avg_size
        );
    }

    #[test]
    fn test_aps_perfect_classifier_small_sets() {
        let n_classes = 5;
        // With one-hot predictions the true label always has highest prob → small sets
        let softmax_cal: Vec<Vec<f64>> = (0..100)
            .map(|i| one_hot_scores(n_classes, i % n_classes))
            .collect();
        let true_labels: Vec<usize> = (0..100).map(|i| i % n_classes).collect();

        let config = AdaptivePredictionSetConfig {
            alpha: 0.1,
            n_classes,
            regularization: 0.0,
            seed: 3,
        };
        let mut aps = AdaptivePredictionSets::new(config);
        aps.calibrate(&softmax_cal, &true_labels).expect("ok");

        let test_scores = vec![one_hot_scores(n_classes, 2)];
        let sets = aps.predict_sets(&test_scores).expect("ok");
        // Should include class 2 with high confidence
        assert!(sets[0].classes.contains(&2));
    }

    #[test]
    fn test_aps_uncalibrated_error() {
        let aps = AdaptivePredictionSets::new(AdaptivePredictionSetConfig::default());
        let scores = uniform_scores(10);
        assert!(aps.predict_set(&scores).is_err());
    }

    #[test]
    fn test_aps_empty_calibration_error() {
        let mut aps = AdaptivePredictionSets::new(AdaptivePredictionSetConfig::default());
        assert!(aps.calibrate(&[], &[]).is_err());
    }

    #[test]
    fn test_aps_mismatched_lengths_error() {
        let mut aps = AdaptivePredictionSets::new(AdaptivePredictionSetConfig {
            n_classes: 5,
            ..Default::default()
        });
        let scores = vec![uniform_scores(5); 10];
        let labels = vec![0usize; 5];
        assert!(aps.calibrate(&scores, &labels).is_err());
    }

    #[test]
    fn test_aps_average_set_size_empty() {
        let aps = AdaptivePredictionSets::new(AdaptivePredictionSetConfig::default());
        assert_eq!(aps.average_set_size(&[]), 0.0);
    }

    #[test]
    fn test_aps_empirical_coverage_mismatched() {
        let aps = AdaptivePredictionSets::new(AdaptivePredictionSetConfig::default());
        let sets = vec![PredictionSet {
            classes: vec![0],
            coverage_guarantee: 0.9,
            set_size: 1,
        }];
        let labels = vec![0, 1]; // different length
        assert_eq!(aps.empirical_coverage(&sets, &labels), 0.0);
    }

    // ── CrossConformal ────────────────────────────────────────────────────────

    #[test]
    fn test_cross_conformal_calibration() {
        let fold1 = vec![(1.0_f64, 1.1_f64), (2.0, 2.2), (3.0, 3.3)];
        let fold2 = vec![(4.0_f64, 4.4_f64), (5.0, 5.1), (6.0, 6.2)];
        let folds = vec![fold1, fold2];

        let config = CrossConformalConfig {
            alpha: 0.1,
            n_folds: 2,
            seed: 1,
        };
        let mut cc = CrossConformal::new(config);
        let mean_q = cc.calibrate(&folds).expect("ok");
        assert!(mean_q > 0.0);
        assert_eq!(cc.fold_quantiles().len(), 2);
    }

    #[test]
    fn test_cross_conformal_predict_interval() {
        let folds: Vec<Vec<(f64, f64)>> = (0..5)
            .map(|k| {
                (0..10)
                    .map(|i| {
                        let p = (k * 10 + i) as f64;
                        (p, p + 0.5)
                    })
                    .collect()
            })
            .collect();

        let config = CrossConformalConfig {
            alpha: 0.1,
            n_folds: 5,
            seed: 0,
        };
        let mut cc = CrossConformal::new(config);
        cc.calibrate(&folds).expect("ok");

        let iv = cc.predict_interval(10.0).expect("ok");
        assert!(iv.lower < iv.upper);
        assert!((iv.coverage_guarantee - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_cross_conformal_uncalibrated_error() {
        let cc = CrossConformal::new(CrossConformalConfig::default());
        assert!(cc.predict_interval(1.0).is_err());
    }

    #[test]
    fn test_cross_conformal_empty_folds_error() {
        let mut cc = CrossConformal::new(CrossConformalConfig::default());
        assert!(cc.calibrate(&[]).is_err());
    }

    #[test]
    fn test_cross_conformal_empty_fold_error() {
        let mut cc = CrossConformal::new(CrossConformalConfig::default());
        let folds: Vec<Vec<(f64, f64)>> = vec![vec![(1.0, 1.0)], vec![]];
        assert!(cc.calibrate(&folds).is_err());
    }

    // ── CoverageDiagnostics ───────────────────────────────────────────────────

    #[test]
    fn test_coverage_diagnostics_perfect_coverage() {
        // All intervals contain the true value
        let intervals: Vec<ConformalPredictionInterval> = (0..100)
            .map(|_| ConformalPredictionInterval {
                lower: -1.0,
                upper: 1.0,
                width: 2.0,
                coverage_guarantee: 0.9,
            })
            .collect();
        let true_values = vec![0.0_f64; 100];

        let diag = compute_coverage_diagnostics(&intervals, &true_values, 0.9).expect("ok");
        assert_eq!(diag.empirical_coverage, 1.0);
        assert!((diag.mean_interval_width - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_diagnostics_zero_coverage() {
        let intervals: Vec<ConformalPredictionInterval> = (0..50)
            .map(|_| ConformalPredictionInterval {
                lower: 10.0,
                upper: 20.0,
                width: 10.0,
                coverage_guarantee: 0.9,
            })
            .collect();
        let true_values = vec![0.0_f64; 50];

        let diag = compute_coverage_diagnostics(&intervals, &true_values, 0.9).expect("ok");
        assert_eq!(diag.empirical_coverage, 0.0);
        assert!((diag.coverage_gap - (-0.9)).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_diagnostics_stats() {
        let widths = [1.0, 2.0, 3.0, 4.0, 5.0];
        let intervals: Vec<ConformalPredictionInterval> = widths
            .iter()
            .map(|&w| ConformalPredictionInterval {
                lower: 0.0,
                upper: w,
                width: w,
                coverage_guarantee: 0.9,
            })
            .collect();
        let true_values = vec![0.5; 5]; // all inside

        let diag = compute_coverage_diagnostics(&intervals, &true_values, 0.9).expect("ok");
        assert!((diag.mean_interval_width - 3.0).abs() < 1e-10);
        assert!((diag.median_interval_width - 3.0).abs() < 1e-10);
        assert!(diag.interval_width_std > 0.0);
    }

    #[test]
    fn test_coverage_diagnostics_empty_error() {
        assert!(compute_coverage_diagnostics(&[], &[], 0.9).is_err());
    }

    #[test]
    fn test_coverage_diagnostics_mismatched_error() {
        let iv = ConformalPredictionInterval {
            lower: 0.0,
            upper: 1.0,
            width: 1.0,
            coverage_guarantee: 0.9,
        };
        assert!(compute_coverage_diagnostics(&[iv], &[0.0, 0.0], 0.9).is_err());
    }

    #[test]
    fn test_coverage_diagnostics_bin_count() {
        let intervals: Vec<ConformalPredictionInterval> = (0..100)
            .map(|_| ConformalPredictionInterval {
                lower: -1.0,
                upper: 1.0,
                width: 2.0,
                coverage_guarantee: 0.9,
            })
            .collect();
        let true_values = vec![0.0; 100];
        let diag = compute_coverage_diagnostics(&intervals, &true_values, 0.9).expect("ok");
        // Should have at most 10 bins
        assert!(diag.coverage_by_bin.len() <= 10);
        assert!(!diag.coverage_by_bin.is_empty());
    }

    // ── conformal_p_value ─────────────────────────────────────────────────────

    #[test]
    fn test_conformal_p_value_basic() {
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // test_score = 3.0: #{s >= 3.0} = 3 → p = (3+1)/(5+1) = 4/6 ≈ 0.667
        let p = conformal_p_value(&scores, 3.0);
        assert!((p - 4.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_conformal_p_value_extreme_low() {
        let scores = vec![1.0, 2.0, 3.0];
        // test_score = 10.0: #{s >= 10.0} = 0 → p = 1/4 = 0.25
        let p = conformal_p_value(&scores, 10.0);
        assert!((p - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_conformal_p_value_extreme_high() {
        let scores = vec![1.0, 2.0, 3.0];
        // test_score = 0.5: #{s >= 0.5} = 3 → p = 4/4 = 1.0
        let p = conformal_p_value(&scores, 0.5);
        assert!((p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_conformal_p_value_empty_calibration() {
        let p = conformal_p_value(&[], 1.0);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn test_conformal_p_value_super_uniform() {
        // Under exchangeability, P(p-value <= t) <= t. Test this empirically.
        let mut rng = StdRng::seed_from_u64(55);
        let n_cal = 100;
        let n_test = 1000;
        let cal_scores: Vec<f64> = (0..n_cal).map(|_| rng.random::<f64>()).collect();

        let mut p_vals: Vec<f64> = Vec::with_capacity(n_test);
        for _ in 0..n_test {
            let test_s: f64 = rng.random::<f64>(); // same distribution
            p_vals.push(conformal_p_value(&cal_scores, test_s));
        }

        // Check super-uniformity at t=0.1: fraction(p <= 0.1) should be <= 0.1 + slack
        let frac_low = p_vals.iter().filter(|&&p| p <= 0.1).count() as f64 / n_test as f64;
        assert!(
            frac_low <= 0.15,
            "super-uniformity violated: frac(p<=0.1) = {:.3}",
            frac_low
        );
    }

    // ── bonferroni_correction ─────────────────────────────────────────────────

    #[test]
    fn test_bonferroni_basic() {
        let p_values = vec![0.001, 0.05, 0.5, 0.01];
        // α=0.05, m=4 → threshold = 0.05/4 = 0.0125
        let rejected = bonferroni_correction(&p_values, 0.05);
        assert_eq!(rejected, vec![true, false, false, true]);
    }

    #[test]
    fn test_bonferroni_all_rejected() {
        let p_values = vec![0.0001, 0.0002];
        let rejected = bonferroni_correction(&p_values, 0.05);
        assert_eq!(rejected, vec![true, true]);
    }

    #[test]
    fn test_bonferroni_none_rejected() {
        let p_values = vec![0.1, 0.2, 0.3];
        let rejected = bonferroni_correction(&p_values, 0.05);
        assert_eq!(rejected, vec![false, false, false]);
    }

    #[test]
    fn test_bonferroni_empty() {
        let rejected = bonferroni_correction(&[], 0.05);
        assert!(rejected.is_empty());
    }

    #[test]
    fn test_bonferroni_fwer_control() {
        // Under H0: all p-values uniform; Bonferroni FWER <= alpha
        let mut rng = StdRng::seed_from_u64(99);
        let alpha = 0.05;
        let m = 20;
        let n_simulations = 1000;
        let mut family_wise_errors = 0;

        for _ in 0..n_simulations {
            let p_values: Vec<f64> = (0..m).map(|_| rng.random::<f64>()).collect();
            let rejected = bonferroni_correction(&p_values, alpha);
            if rejected.iter().any(|&r| r) {
                family_wise_errors += 1;
            }
        }

        let fwer = family_wise_errors as f64 / n_simulations as f64;
        assert!(
            fwer <= alpha + 0.03,
            "FWER {:.3} exceeds alpha + 0.03 = {:.3}",
            fwer,
            alpha + 0.03
        );
    }
}
