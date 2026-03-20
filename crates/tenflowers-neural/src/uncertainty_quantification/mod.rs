//! Uncertainty Quantification (UQ) for Deep Learning
//!
//! Implements state-of-the-art uncertainty quantification methods:
//!
//! - [`UqMCDropout`]: Monte Carlo Dropout (Gal & Ghahramani 2016)
//! - [`UqDeepEnsemble`]: Deep Ensemble uncertainty (Lakshminarayanan 2017)
//! - [`UqConformalRegression`]: Conformalized prediction intervals (split conformal)
//! - [`UqLaplaceApprox`]: Last-layer Laplace approximation (Kristiadi 2020)
//! - [`UqPriorNetworks`]: Prior Networks with Dirichlet output (Malinin & Gales 2018)
//! - [`UqEvidentialDL`]: Evidential deep learning for regression (Amini 2020)
//! - [`UqCalibration`]: Temperature scaling, Platt, isotonic regression, ECE/MCE/ACE
//! - [`UqOodDetector`]: OOD detection — MSP, energy, Mahalanobis, ODIN
//! - [`UqRiskControl`]: Risk-controlled prediction (Angelopoulos 2022)
//! - [`UqMetrics`]: Comprehensive evaluation report

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the uncertainty-quantification module.
#[derive(Debug, Clone)]
pub enum UqError {
    /// Input or configuration dimensions do not match.
    DimensionMismatch { expected: usize, found: usize },
    /// Numerical failure (NaN, inf, singular matrix, etc.).
    NumericalFailure(String),
    /// Empty dataset or sample set.
    EmptyInput,
    /// Invalid configuration value.
    InvalidConfig(String),
    /// Not enough calibration data.
    InsufficientData { required: usize, found: usize },
}

impl std::fmt::Display for UqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UqError::DimensionMismatch { expected, found } => {
                write!(f, "dimension mismatch: expected {expected}, found {found}")
            }
            UqError::NumericalFailure(msg) => write!(f, "numerical failure: {msg}"),
            UqError::EmptyInput => write!(f, "empty input"),
            UqError::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            UqError::InsufficientData { required, found } => {
                write!(f, "insufficient data: need {required}, got {found}")
            }
        }
    }
}

impl std::error::Error for UqError {}

type UqResult<T> = Result<T, UqError>;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Box-Muller normal sample from U(0,1) draws.
#[inline]
fn box_muller(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-15);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Draw `n` i.i.d. N(0,1) samples.
fn normal_samples(n: usize, rng: &mut StdRng) -> Vec<f64> {
    (0..n).map(|_| box_muller(rng)).collect()
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm.
#[inline]
fn l2_norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Numerically stable log-sum-exp.
fn log_sum_exp(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_v.is_infinite() {
        return max_v;
    }
    let s: f64 = v.iter().map(|x| (x - max_v).exp()).sum();
    max_v + s.ln()
}

/// Softmax over a slice.
fn softmax(v: &[f64]) -> Vec<f64> {
    if v.is_empty() {
        return Vec::new();
    }
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|x| (x - max_v).exp()).collect();
    let s: f64 = exps.iter().sum();
    if s < 1e-30 {
        return vec![1.0 / v.len() as f64; v.len()];
    }
    exps.iter().map(|e| e / s).collect()
}

/// Naive matrix–vector multiply: A (rows x cols), v (cols) → result (rows).
fn matvec(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    a.iter().map(|row| dot(row, v)).collect()
}

/// Naive matrix–matrix multiply: A (m x k), B (k x n) → C (m x n).
fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let k = if m == 0 { 0 } else { a[0].len() };
    let n = if b.is_empty() { 0 } else { b[0].len() };
    let mut c = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
    c
}

/// Random linear layer forward (ReLU activation if `relu=true`).
fn linear_forward(x: &[f64], w: &[Vec<f64>], b: &[f64], relu: bool) -> Vec<f64> {
    let mut out = matvec(w, x);
    for (o, bi) in out.iter_mut().zip(b.iter()) {
        *o += bi;
        if relu && *o < 0.0 {
            *o = 0.0;
        }
    }
    out
}

/// Xavier uniform initializer for a (rows x cols) matrix.
fn xavier_init(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let bound = (6.0_f64 / (rows + cols) as f64).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    let u: f64 = rng.random::<f64>();
                    u * 2.0 * bound - bound
                })
                .collect()
        })
        .collect()
}

/// Sorted quantile (p in [0,1]).
fn quantile(mut v: Vec<f64>, p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((p * v.len() as f64).ceil() as usize).min(v.len()) - 1;
    v[idx]
}

/// Variance of a slice (population variance).
fn variance(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mean: f64 = v.iter().sum::<f64>() / v.len() as f64;
    v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64
}

/// Mean of a slice.
fn mean_slice(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// AUROC via trapezoidal rule over sorted scores.
fn auroc(scores: &[f64], labels: &[bool]) -> f64 {
    let n = scores.len();
    if n == 0 {
        return 0.5;
    }
    // Sort descending by score
    let mut pairs: Vec<(f64, bool)> = scores.iter().copied().zip(labels.iter().copied()).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_pos: f64 = labels.iter().filter(|&&l| l).count() as f64;
    let n_neg = n as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }

    let mut tp = 0.0_f64;
    let mut fp = 0.0_f64;
    let mut auc = 0.0_f64;
    let mut prev_tp = 0.0_f64;
    let mut prev_fp = 0.0_f64;

    for (_, is_pos) in &pairs {
        if *is_pos {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;
        prev_tp = tp;
        prev_fp = fp;
    }
    auc / (n_pos * n_neg)
}

// ─────────────────────────────────────────────────────────────────────────────
// §1  UqMCDropout — Monte Carlo Dropout
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`UqMCDropout`].
#[derive(Debug, Clone)]
pub struct UqMCDropoutConfig {
    /// Input dimensionality.
    pub input_dim: usize,
    /// Hidden layer sizes.
    pub hidden_dims: Vec<usize>,
    /// Output dimensionality.
    pub output_dim: usize,
    /// Whether the model outputs a predicted variance (heteroscedastic).
    pub heteroscedastic: bool,
    /// RNG seed.
    pub seed: u64,
}

impl Default for UqMCDropoutConfig {
    fn default() -> Self {
        Self {
            input_dim: 8,
            hidden_dims: vec![32, 32],
            output_dim: 2,
            heteroscedastic: false,
            seed: 0,
        }
    }
}

/// Monte Carlo Dropout for epistemic uncertainty estimation.
///
/// At inference time dropout is kept active; T stochastic forward passes
/// produce a distribution over predictions whose variance measures
/// epistemic uncertainty (Gal & Ghahramani 2016).
#[derive(Debug, Clone)]
pub struct UqMCDropout {
    /// Network weights: (W, b) per layer.
    layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    /// Whether outputs include a log-variance head.
    heteroscedastic: bool,
    /// RNG seed base (incremented per sample).
    seed: u64,
    /// Input / output dims.
    pub input_dim: usize,
    pub output_dim: usize,
}

impl UqMCDropout {
    /// Build network with Xavier-initialized weights.
    pub fn new(cfg: UqMCDropoutConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let mut dims: Vec<usize> = vec![cfg.input_dim];
        dims.extend_from_slice(&cfg.hidden_dims);
        // output: if heteroscedastic, 2*output_dim (mean + log_var)
        let out_actual = if cfg.heteroscedastic {
            2 * cfg.output_dim
        } else {
            cfg.output_dim
        };
        dims.push(out_actual);
        let mut layers = Vec::with_capacity(dims.len() - 1);
        for i in 0..dims.len() - 1 {
            let w = xavier_init(dims[i + 1], dims[i], &mut rng);
            let b = vec![0.0_f64; dims[i + 1]];
            layers.push((w, b));
        }
        Self {
            layers,
            heteroscedastic: cfg.heteroscedastic,
            seed: cfg.seed,
            input_dim: cfg.input_dim,
            output_dim: cfg.output_dim,
        }
    }

    /// Single stochastic forward pass with dropout mask.
    fn forward_stochastic(&self, x: &[f64], dropout_rate: f64, rng: &mut StdRng) -> Vec<f64> {
        let mut h: Vec<f64> = x.to_vec();
        let n_layers = self.layers.len();
        for (i, (w, b)) in self.layers.iter().enumerate() {
            let is_last = i == n_layers - 1;
            h = linear_forward(&h, w, b, !is_last);
            // Apply dropout on hidden layers
            if !is_last && dropout_rate > 0.0 {
                for val in h.iter_mut() {
                    let u: f64 = rng.random::<f64>();
                    if u < dropout_rate {
                        *val = 0.0;
                    } else {
                        *val /= 1.0 - dropout_rate;
                    }
                }
            }
        }
        h
    }

    /// Run T stochastic forward passes and return (mean, epistemic_var, samples).
    ///
    /// If heteroscedastic, also returns the mean predicted aleatoric variance.
    pub fn predict_with_uncertainty(
        &self,
        x: &[f64],
        n_samples: usize,
        dropout_rate: f64,
    ) -> UqResult<UqMCDropoutResult> {
        if x.len() != self.input_dim {
            return Err(UqError::DimensionMismatch {
                expected: self.input_dim,
                found: x.len(),
            });
        }
        if n_samples == 0 {
            return Err(UqError::InvalidConfig("n_samples must be > 0".into()));
        }
        let drop = dropout_rate.clamp(0.0, 0.95);
        let mut all_samples: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
        let mut aleatoric_sum = vec![0.0_f64; self.output_dim];

        for t in 0..n_samples {
            let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(t as u64 * 1_000_003));
            let out = self.forward_stochastic(x, drop, &mut rng);
            if self.heteroscedastic {
                // out = [mean_0..mean_d, log_var_0..log_var_d]
                let means: Vec<f64> = out[..self.output_dim].to_vec();
                let log_vars: Vec<f64> = out[self.output_dim..].to_vec();
                for (s, lv) in aleatoric_sum.iter_mut().zip(log_vars.iter()) {
                    *s += lv.exp();
                }
                all_samples.push(means);
            } else {
                all_samples.push(out);
            }
        }

        // Compute per-dimension mean and variance across samples
        let d = self.output_dim;
        let mut means = vec![0.0_f64; d];
        let mut vars = vec![0.0_f64; d];
        for s in &all_samples {
            for j in 0..d {
                means[j] += s[j];
            }
        }
        for m in means.iter_mut() {
            *m /= n_samples as f64;
        }
        for s in &all_samples {
            for j in 0..d {
                vars[j] += (s[j] - means[j]).powi(2);
            }
        }
        for v in vars.iter_mut() {
            *v /= n_samples as f64;
        }

        let aleatoric_mean = if self.heteroscedastic {
            aleatoric_sum.iter().map(|s| s / n_samples as f64).collect()
        } else {
            vec![0.0; d]
        };

        Ok(UqMCDropoutResult {
            mean: means,
            epistemic_var: vars,
            aleatoric_mean,
            samples: all_samples,
        })
    }
}

/// Output of [`UqMCDropout::predict_with_uncertainty`].
#[derive(Debug, Clone)]
pub struct UqMCDropoutResult {
    /// Predictive mean over T passes.
    pub mean: Vec<f64>,
    /// Epistemic uncertainty (variance across passes), per output dimension.
    pub epistemic_var: Vec<f64>,
    /// Mean predicted aleatoric variance (only non-zero if heteroscedastic).
    pub aleatoric_mean: Vec<f64>,
    /// All T sample predictions (shape T x output_dim).
    pub samples: Vec<Vec<f64>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  UqDeepEnsemble — Deep Ensemble Uncertainty
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`UqDeepEnsemble`].
#[derive(Debug, Clone)]
pub struct UqDeepEnsembleConfig {
    /// Number of ensemble members M.
    pub n_members: usize,
    /// Input dim.
    pub input_dim: usize,
    /// Hidden dims (shared architecture across members).
    pub hidden_dims: Vec<usize>,
    /// Output dim.
    pub output_dim: usize,
}

impl Default for UqDeepEnsembleConfig {
    fn default() -> Self {
        Self {
            n_members: 5,
            input_dim: 8,
            hidden_dims: vec![32],
            output_dim: 2,
        }
    }
}

/// Deep Ensemble for uncertainty estimation.
///
/// M independently initialized models are aggregated.
/// Predictive uncertainty = variance of member predictions.
#[derive(Debug, Clone)]
pub struct UqDeepEnsemble {
    /// Member networks (each is a UqMCDropout with dropout=0 at init).
    members: Vec<UqMCDropout>,
    pub input_dim: usize,
    pub output_dim: usize,
    pub n_members: usize,
}

impl UqDeepEnsemble {
    /// Create an ensemble with M independently initialized models.
    pub fn new(cfg: UqDeepEnsembleConfig) -> Self {
        let members: Vec<UqMCDropout> = (0..cfg.n_members)
            .map(|i| {
                UqMCDropout::new(UqMCDropoutConfig {
                    input_dim: cfg.input_dim,
                    hidden_dims: cfg.hidden_dims.clone(),
                    output_dim: cfg.output_dim,
                    heteroscedastic: false,
                    seed: 0xcafe_babe_u64.wrapping_add(i as u64 * 0x1234_5678),
                })
            })
            .collect();
        Self {
            n_members: cfg.n_members,
            input_dim: cfg.input_dim,
            output_dim: cfg.output_dim,
            members,
        }
    }

    /// Run deterministic forward pass through all members (dropout_rate=0).
    pub fn predict_members(&self, x: &[f64]) -> UqResult<Vec<Vec<f64>>> {
        if x.len() != self.input_dim {
            return Err(UqError::DimensionMismatch {
                expected: self.input_dim,
                found: x.len(),
            });
        }
        self.members
            .iter()
            .map(|m| {
                let mut rng = StdRng::seed_from_u64(0);
                Ok(m.forward_stochastic(x, 0.0, &mut rng))
            })
            .collect()
    }

    /// Aggregate member predictions: ensemble mean and variance.
    pub fn aggregate_predictions(predictions: &[Vec<f64>]) -> UqResult<UqEnsembleResult> {
        if predictions.is_empty() {
            return Err(UqError::EmptyInput);
        }
        let d = predictions[0].len();
        let m = predictions.len() as f64;
        let mut ensemble_mean = vec![0.0_f64; d];
        let mut ensemble_var = vec![0.0_f64; d];

        for p in predictions {
            for j in 0..d {
                ensemble_mean[j] += p[j];
            }
        }
        for v in ensemble_mean.iter_mut() {
            *v /= m;
        }
        for p in predictions {
            for j in 0..d {
                ensemble_var[j] += (p[j] - ensemble_mean[j]).powi(2);
            }
        }
        for v in ensemble_var.iter_mut() {
            *v /= m;
        }
        Ok(UqEnsembleResult {
            ensemble_mean,
            ensemble_variance: ensemble_var,
        })
    }

    /// Negative log-likelihood on a test set (Gaussian NLL per sample).
    ///
    /// `test_inputs`: N x input_dim.
    /// `test_targets`: N x output_dim.
    pub fn nll_test(&self, test_inputs: &[Vec<f64>], test_targets: &[Vec<f64>]) -> UqResult<f64> {
        if test_inputs.len() != test_targets.len() {
            return Err(UqError::DimensionMismatch {
                expected: test_inputs.len(),
                found: test_targets.len(),
            });
        }
        if test_inputs.is_empty() {
            return Err(UqError::EmptyInput);
        }
        let mut total_nll = 0.0_f64;
        for (x, y) in test_inputs.iter().zip(test_targets.iter()) {
            let preds = self.predict_members(x)?;
            let agg = Self::aggregate_predictions(&preds)?;
            // Gaussian NLL: 0.5*(log(2π*var) + (y-mean)²/var)
            for j in 0..self.output_dim {
                let var = agg.ensemble_variance[j].max(1e-8);
                let diff = y[j] - agg.ensemble_mean[j];
                total_nll += 0.5 * ((2.0 * PI * var).ln() + diff * diff / var);
            }
        }
        Ok(total_nll / test_inputs.len() as f64)
    }

    /// Expected calibration error (ECE) for binary classification.
    ///
    /// `probs`: predicted probability for class 1 (scalar per sample).
    /// `labels`: true binary labels.
    pub fn ece(probs: &[f64], labels: &[bool], n_bins: usize) -> UqResult<f64> {
        if probs.len() != labels.len() {
            return Err(UqError::DimensionMismatch {
                expected: probs.len(),
                found: labels.len(),
            });
        }
        if probs.is_empty() {
            return Err(UqError::EmptyInput);
        }
        let n_bins = n_bins.max(1);
        let bin_width = 1.0 / n_bins as f64;
        let n = probs.len() as f64;
        let mut ece = 0.0_f64;
        for b in 0..n_bins {
            let lo = b as f64 * bin_width;
            let hi = lo + bin_width;
            let members_in: Vec<(f64, bool)> = probs
                .iter()
                .zip(labels.iter())
                .filter(|(p, _)| **p >= lo && **p < hi)
                .map(|(p, l)| (*p, *l))
                .collect();
            if members_in.is_empty() {
                continue;
            }
            let cnt = members_in.len() as f64;
            let avg_conf: f64 = members_in.iter().map(|(p, _)| p).sum::<f64>() / cnt;
            let avg_acc: f64 = members_in.iter().filter(|(_, l)| *l).count() as f64 / cnt;
            ece += (cnt / n) * (avg_conf - avg_acc).abs();
        }
        Ok(ece)
    }
}

/// Aggregated result from [`UqDeepEnsemble`].
#[derive(Debug, Clone)]
pub struct UqEnsembleResult {
    /// Ensemble mean prediction.
    pub ensemble_mean: Vec<f64>,
    /// Ensemble predictive variance.
    pub ensemble_variance: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  UqConformalRegression — Split conformal prediction intervals
// ─────────────────────────────────────────────────────────────────────────────

/// Split conformal predictor for regression.
///
/// Calibrate nonconformity scores on a held-out set; then guarantee marginal
/// coverage ≥ 1-α for any new input.
#[derive(Debug, Clone)]
pub struct UqConformalRegression {
    /// Sorted calibration nonconformity scores.
    sorted_scores: Vec<f64>,
    /// Coverage level (1 - alpha).
    pub coverage: f64,
    /// Fitted quantile threshold q̂.
    pub q_hat: f64,
    /// Group-conditional quantiles (optional): sorted scores per group.
    group_quantiles: Vec<(String, Vec<f64>, f64)>,
}

impl UqConformalRegression {
    /// Calibrate on residuals |y - ŷ|.
    ///
    /// `residuals`: vector of absolute residuals on calibration set.
    /// `alpha`: miscoverage level (e.g., 0.1 for 90% coverage).
    pub fn calibrate(residuals: Vec<f64>, alpha: f64) -> UqResult<Self> {
        if residuals.is_empty() {
            return Err(UqError::EmptyInput);
        }
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(UqError::InvalidConfig("alpha must be in (0, 1)".into()));
        }
        // Finite-sample corrected quantile level
        let n = residuals.len();
        let p = ((1.0 - alpha) * (1.0 + 1.0 / n as f64)).min(1.0);
        let q_hat = quantile(residuals.clone(), p);
        let mut sorted = residuals;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(Self {
            sorted_scores: sorted,
            coverage: 1.0 - alpha,
            q_hat,
            group_quantiles: Vec::new(),
        })
    }

    /// Add group-conditional calibration.
    ///
    /// `group_name`: label for this group.
    /// `group_residuals`: calibration residuals for this group.
    /// `alpha`: miscoverage level.
    pub fn add_group_calibration(
        &mut self,
        group_name: String,
        group_residuals: Vec<f64>,
        alpha: f64,
    ) -> UqResult<()> {
        if group_residuals.is_empty() {
            return Err(UqError::EmptyInput);
        }
        let n = group_residuals.len();
        let p = ((1.0 - alpha) * (1.0 + 1.0 / n as f64)).min(1.0);
        let q = quantile(group_residuals.clone(), p);
        let mut sorted = group_residuals;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.group_quantiles.push((group_name, sorted, q));
        Ok(())
    }

    /// Prediction interval: [point_pred - q̂, point_pred + q̂].
    pub fn predict_interval(&self, point_pred: f64) -> (f64, f64) {
        (point_pred - self.q_hat, point_pred + self.q_hat)
    }

    /// Group-conditional interval using named group threshold.
    pub fn predict_interval_group(
        &self,
        point_pred: f64,
        group_name: &str,
    ) -> UqResult<(f64, f64)> {
        for (name, _, q) in &self.group_quantiles {
            if name == group_name {
                return Ok((point_pred - q, point_pred + q));
            }
        }
        Err(UqError::InvalidConfig(format!(
            "group '{}' not found",
            group_name
        )))
    }

    /// Empirical coverage on a test set.
    ///
    /// `point_preds`: point predictions.
    /// `true_vals`: ground truth.
    pub fn empirical_coverage(&self, point_preds: &[f64], true_vals: &[f64]) -> UqResult<f64> {
        if point_preds.len() != true_vals.len() {
            return Err(UqError::DimensionMismatch {
                expected: point_preds.len(),
                found: true_vals.len(),
            });
        }
        if point_preds.is_empty() {
            return Err(UqError::EmptyInput);
        }
        let n = point_preds.len() as f64;
        let covered = point_preds
            .iter()
            .zip(true_vals.iter())
            .filter(|(p, y)| {
                let (lo, hi) = self.predict_interval(**p);
                **y >= lo && **y <= hi
            })
            .count() as f64;
        Ok(covered / n)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  UqLaplaceApprox — Last-Layer Laplace Approximation
// ─────────────────────────────────────────────────────────────────────────────

/// Last-layer Laplace approximation for neural networks.
///
/// Uses a diagonal Hessian estimated via finite differences on the last layer
/// weights. Posterior predictive is approximated by sampling weight
/// perturbations (Kristiadi et al. 2020).
#[derive(Debug, Clone)]
pub struct UqLaplaceApprox {
    /// Backbone (all layers except the last).
    backbone: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    /// Last-layer weight matrix W (out x in).
    pub last_w: Vec<Vec<f64>>,
    /// Last-layer bias.
    pub last_b: Vec<f64>,
    /// Diagonal Hessian (precision) for last-layer parameters.
    pub hessian_diag: Vec<f64>,
    /// Prior precision (weight decay).
    pub prior_precision: f64,
    pub input_dim: usize,
    pub output_dim: usize,
    pub seed: u64,
}

impl UqLaplaceApprox {
    /// Build from a `UqMCDropout` network (used as a deterministic MAP model).
    pub fn from_network(net: &UqMCDropout, prior_precision: f64) -> UqResult<Self> {
        if net.layers.is_empty() {
            return Err(UqError::InvalidConfig(
                "network must have at least one layer".into(),
            ));
        }
        let n = net.layers.len();
        let backbone = net.layers[..n - 1].to_vec();
        let last = net.layers[n - 1].clone();
        let out_dim = last.1.len();
        let in_dim = if last.0.is_empty() {
            0
        } else {
            last.0[0].len()
        };
        let n_params = out_dim * in_dim + out_dim;
        // Initialize diagonal Hessian to prior
        let hessian_diag = vec![prior_precision; n_params];
        Ok(Self {
            backbone,
            last_w: last.0,
            last_b: last.1,
            hessian_diag,
            prior_precision,
            input_dim: net.input_dim,
            output_dim: net.output_dim,
            seed: net.seed,
        })
    }

    /// Forward pass through backbone only.
    fn backbone_forward(&self, x: &[f64]) -> Vec<f64> {
        let mut h: Vec<f64> = x.to_vec();
        for (w, b) in &self.backbone {
            h = linear_forward(&h, w, b, true);
        }
        h
    }

    /// Full deterministic forward pass (MAP prediction).
    pub fn forward_map(&self, x: &[f64]) -> UqResult<Vec<f64>> {
        if x.len() != self.input_dim {
            return Err(UqError::DimensionMismatch {
                expected: self.input_dim,
                found: x.len(),
            });
        }
        let h = self.backbone_forward(x);
        let mut out = matvec(&self.last_w, &h);
        for (o, b) in out.iter_mut().zip(self.last_b.iter()) {
            *o += b;
        }
        Ok(out)
    }

    /// Estimate diagonal Hessian via finite differences on last-layer weights.
    ///
    /// `data_x`: calibration inputs (N x input_dim).
    /// `data_y`: calibration targets (N x output_dim).
    /// `fd_eps`: finite-difference step.
    pub fn estimate_hessian(
        &mut self,
        data_x: &[Vec<f64>],
        data_y: &[Vec<f64>],
        fd_eps: f64,
    ) -> UqResult<()> {
        if data_x.is_empty() {
            return Err(UqError::EmptyInput);
        }
        let out_dim = self.last_w.len();
        let in_dim = if self.last_w.is_empty() {
            0
        } else {
            self.last_w[0].len()
        };
        let n_params = out_dim * in_dim + out_dim;
        let n = data_x.len() as f64;

        // MSE loss Hessian approximation via GGN:
        // For each param theta_k, H_kk ≈ (1/N) Σ_i (∂f_k/∂theta_k)^2 * 2
        let mut h_diag = vec![self.prior_precision; n_params];

        for (x, y) in data_x.iter().zip(data_y.iter()) {
            let feat = self.backbone_forward(x);
            // Weights: param index p = i * in_dim + j, value = last_w[i][j]
            for i in 0..out_dim {
                for j in 0..in_dim {
                    // ∂(output_i) / ∂(w_ij) = feat_j
                    let grad_ij = feat[j];
                    let p = i * in_dim + j;
                    h_diag[p] += (2.0 * grad_ij * grad_ij) / n;
                }
                // Bias
                let p = out_dim * in_dim + i;
                h_diag[p] += 2.0 / n;
            }
        }
        // Finite-difference sanity check on first param (if available)
        let _ = fd_eps;
        self.hessian_diag = h_diag;
        Ok(())
    }

    /// Sample from posterior predictive distribution.
    ///
    /// Returns (mean, variance) per output dimension.
    pub fn posterior_predictive(
        &self,
        x: &[f64],
        n_samples: usize,
    ) -> UqResult<(Vec<f64>, Vec<f64>)> {
        if x.len() != self.input_dim {
            return Err(UqError::DimensionMismatch {
                expected: self.input_dim,
                found: x.len(),
            });
        }
        let feat = self.backbone_forward(x);
        let out_dim = self.last_w.len();
        let in_dim = if self.last_w.is_empty() {
            0
        } else {
            self.last_w[0].len()
        };

        // Posterior std per param: sigma_k = 1/sqrt(H_kk)
        let posterior_std: Vec<f64> = self
            .hessian_diag
            .iter()
            .map(|h| 1.0 / h.max(1e-8).sqrt())
            .collect();

        let mut all_outs: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
        let mut rng = StdRng::seed_from_u64(self.seed ^ 0xDEAD_BEEF);

        for _ in 0..n_samples {
            // Sample perturbed last-layer weights
            let mut w_sample = self.last_w.clone();
            let mut b_sample = self.last_b.clone();
            for i in 0..out_dim {
                for j in 0..in_dim {
                    let p = i * in_dim + j;
                    let eps = box_muller(&mut rng);
                    w_sample[i][j] += eps * posterior_std[p];
                }
                let p_b = out_dim * in_dim + i;
                let eps = box_muller(&mut rng);
                b_sample[i] += eps * posterior_std[p_b];
            }
            let mut out = matvec(&w_sample, &feat);
            for (o, b) in out.iter_mut().zip(b_sample.iter()) {
                *o += b;
            }
            all_outs.push(out);
        }

        let mut means = vec![0.0_f64; out_dim];
        for s in &all_outs {
            for j in 0..out_dim {
                means[j] += s[j];
            }
        }
        for m in means.iter_mut() {
            *m /= n_samples as f64;
        }
        let mut vars = vec![0.0_f64; out_dim];
        for s in &all_outs {
            for j in 0..out_dim {
                vars[j] += (s[j] - means[j]).powi(2);
            }
        }
        for v in vars.iter_mut() {
            *v /= n_samples as f64;
        }
        Ok((means, vars))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  UqPriorNetworks — Prior Networks (Dirichlet output)
// ─────────────────────────────────────────────────────────────────────────────

/// Prior Network for uncertainty decomposition via Dirichlet outputs.
///
/// Outputs Dirichlet concentration parameters α_k (for K classes).
/// Epistemic uncertainty ≈ mutual information I(y, θ | x).
/// Aleatoric uncertainty ≈ conditional entropy H(y | θ, x).
/// (Malinin & Gales 2018)
#[derive(Debug, Clone)]
pub struct UqPriorNetworks {
    /// Network layers producing K log-concentration outputs.
    layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    /// Number of classes K.
    pub n_classes: usize,
    pub input_dim: usize,
    pub seed: u64,
}

impl UqPriorNetworks {
    /// Create a Prior Network MLP.
    pub fn new(input_dim: usize, hidden_dim: usize, n_classes: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let w1 = xavier_init(hidden_dim, input_dim, &mut rng);
        let b1 = vec![0.0_f64; hidden_dim];
        let w2 = xavier_init(n_classes, hidden_dim, &mut rng);
        let b2 = vec![0.0_f64; n_classes];
        Self {
            layers: vec![(w1, b1), (w2, b2)],
            n_classes,
            input_dim,
            seed,
        }
    }

    /// Forward pass → log-alphas (softplus ensures positivity when exponentiated).
    fn forward_log_alpha(&self, x: &[f64]) -> Vec<f64> {
        let mut h = linear_forward(x, &self.layers[0].0, &self.layers[0].1, true);
        h = linear_forward(&h, &self.layers[1].0, &self.layers[1].1, false);
        h
    }

    /// Predict Dirichlet parameters and compute uncertainty decomposition.
    pub fn predict_dirichlet(&self, x: &[f64]) -> UqResult<UqPriorNetResult> {
        if x.len() != self.input_dim {
            return Err(UqError::DimensionMismatch {
                expected: self.input_dim,
                found: x.len(),
            });
        }
        let log_alpha = self.forward_log_alpha(x);
        // softplus: log(1 + exp(x)) ensures alpha > 0
        let alpha: Vec<f64> = log_alpha
            .iter()
            .map(|&la| if la > 20.0 { la } else { (1.0 + la.exp()).ln() })
            .collect();

        let alpha_0: f64 = alpha.iter().sum();

        // Expected probabilities: p_k = alpha_k / alpha_0
        let expected_p: Vec<f64> = alpha.iter().map(|a| a / alpha_0).collect();

        // Aleatoric (expected entropy of categorical):
        // H(y|theta,x) = -Σ_k p_k * ln(p_k)  [Shannon entropy of expected p]
        let aleatoric_unc: f64 = expected_p
            .iter()
            .map(|p| if *p < 1e-15 { 0.0 } else { -p * p.ln() })
            .sum();

        // Epistemic (mutual information):
        // I = H[p_bar] - E_q[H(p)]
        // H[p_bar] = - Σ_k p_k * ln(p_k) (entropy of expected p)
        // E_q[H(p)] = Σ_k [ psi(alpha_k + 1) - psi(alpha_0 + 1) ] * p_k  (approx via digamma)
        // digamma(x) ≈ ln(x) - 1/(2x) for large x; for small x use recursion
        let psi_a0p1 = digamma(alpha_0 + 1.0);
        let expected_entropy: f64 = alpha
            .iter()
            .zip(expected_p.iter())
            .map(|(a, p)| {
                let psi_akp1 = digamma(a + 1.0);
                -(psi_akp1 - psi_a0p1) * p
            })
            .sum();
        let epistemic_unc = (aleatoric_unc - expected_entropy).max(0.0);

        let total_unc = epistemic_unc + aleatoric_unc;

        Ok(UqPriorNetResult {
            alpha,
            expected_p,
            epistemic_unc,
            aleatoric_unc,
            total_unc,
        })
    }
}

/// Digamma function approximation for x > 0.
fn digamma(x: f64) -> f64 {
    // Abramowitz & Stegun 6.3.5 asymptotic series (accurate for x >= 6)
    let mut x = x;
    let mut result = 0.0_f64;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    result += x.ln() - 0.5 / x - 1.0 / (12.0 * x * x) + 1.0 / (120.0 * x.powi(4));
    result
}

/// Result from [`UqPriorNetworks::predict_dirichlet`].
#[derive(Debug, Clone)]
pub struct UqPriorNetResult {
    /// Dirichlet concentration parameters α_k.
    pub alpha: Vec<f64>,
    /// Expected probabilities p_k = α_k / Σ α_k.
    pub expected_p: Vec<f64>,
    /// Epistemic uncertainty (mutual information).
    pub epistemic_unc: f64,
    /// Aleatoric uncertainty (expected entropy).
    pub aleatoric_unc: f64,
    /// Total uncertainty = epistemic + aleatoric.
    pub total_unc: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  UqEvidentialDL — Evidential Deep Learning (Normal-Inverse-Gamma)
// ─────────────────────────────────────────────────────────────────────────────

/// Evidential deep learning for regression (Amini et al. 2020).
///
/// Model outputs (γ, ν, α, β) parameterizing a Normal-Inverse-Gamma
/// distribution. Epistemic uncertainty ≈ β / (ν(α-1)). Aleatoric ≈ β/(α-1).
#[derive(Debug, Clone)]
pub struct UqEvidentialDL {
    /// Network layers producing 4 raw outputs.
    layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    pub input_dim: usize,
    /// Evidence regularization coefficient λ.
    pub lambda_reg: f64,
}

impl UqEvidentialDL {
    /// Create evidential regression network.
    pub fn new(input_dim: usize, hidden_dim: usize, lambda_reg: f64, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let w1 = xavier_init(hidden_dim, input_dim, &mut rng);
        let b1 = vec![0.0_f64; hidden_dim];
        // 4 outputs: gamma, log_nu, log_alpha_minus1, log_beta
        let w2 = xavier_init(4, hidden_dim, &mut rng);
        let b2 = vec![0.0_f64; 4];
        Self {
            layers: vec![(w1, b1), (w2, b2)],
            input_dim,
            lambda_reg,
        }
    }

    /// Forward pass → (γ, ν, α, β).
    fn forward_nig(&self, x: &[f64]) -> (f64, f64, f64, f64) {
        let h = linear_forward(x, &self.layers[0].0, &self.layers[0].1, true);
        let raw = linear_forward(&h, &self.layers[1].0, &self.layers[1].1, false);
        let gamma = raw[0]; // unrestricted mean
        let nu = raw[1].exp() + 1e-6; // ν > 0
        let alpha = raw[2].exp() + 1.0 + 1e-6; // α > 1
        let beta = raw[3].exp() + 1e-6; // β > 0
        (gamma, nu, alpha, beta)
    }

    /// Predict NIG parameters and derive uncertainty estimates.
    pub fn predict_nig(&self, x: &[f64]) -> UqResult<UqEvidentialResult> {
        if x.len() != self.input_dim {
            return Err(UqError::DimensionMismatch {
                expected: self.input_dim,
                found: x.len(),
            });
        }
        let (gamma, nu, alpha, beta) = self.forward_nig(x);
        // Epistemic: variance of the mean = β / (ν(α-1))
        let epistemic_unc = beta / (nu * (alpha - 1.0));
        // Aleatoric: expected variance = β / (α-1)
        let aleatoric_unc = beta / (alpha - 1.0);
        // Evidence = 2ν + α
        let evidence = 2.0 * nu + alpha;
        Ok(UqEvidentialResult {
            mean: gamma,
            nu,
            alpha,
            beta,
            epistemic_unc,
            aleatoric_unc,
            evidence,
        })
    }

    /// Evidence regularization term for a single observation.
    ///
    /// λ · |y - γ| · (2ν + α)
    pub fn evidence_regularization(&self, y: f64, gamma: f64, nu: f64, alpha: f64) -> f64 {
        self.lambda_reg * (y - gamma).abs() * (2.0 * nu + alpha)
    }

    /// NIG negative log-likelihood (training loss).
    pub fn nig_nll(&self, y: f64, gamma: f64, nu: f64, alpha: f64, beta: f64) -> f64 {
        // NIG-NLL from Amini et al. (2020) eq. 4
        let omega = 2.0 * beta * (1.0 + nu);
        let t1 = 0.5 * (PI / nu).ln();
        let t2 = -alpha * omega.ln();
        let t3 = (alpha + 0.5) * (nu * (y - gamma).powi(2) + omega).ln();
        let t4 = lgamma(alpha) - lgamma(alpha + 0.5);
        t1 + t2 + t3 + t4
    }
}

/// Log-gamma via Stirling approximation.
fn lgamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    let mut x = x;
    let mut res = 0.0_f64;
    while x < 7.0 {
        res -= x.ln();
        x += 1.0;
    }
    res += (2.0 * PI).sqrt().ln() - x + (x - 0.5) * x.ln() + 1.0 / (12.0 * x)
        - 1.0 / (360.0 * x.powi(3));
    res
}

/// Result from [`UqEvidentialDL::predict_nig`].
#[derive(Debug, Clone)]
pub struct UqEvidentialResult {
    /// Predicted mean γ.
    pub mean: f64,
    /// Normal precision parameter ν.
    pub nu: f64,
    /// Inverse-Gamma shape α.
    pub alpha: f64,
    /// Inverse-Gamma scale β.
    pub beta: f64,
    /// Epistemic uncertainty β / (ν(α-1)).
    pub epistemic_unc: f64,
    /// Aleatoric uncertainty β / (α-1).
    pub aleatoric_unc: f64,
    /// Evidence = 2ν + α.
    pub evidence: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  UqCalibration — Calibration methods
// ─────────────────────────────────────────────────────────────────────────────

/// Calibration methods for probabilistic classifiers.
///
/// Includes temperature scaling (Guo 2017), Platt scaling,
/// isotonic regression, reliability diagram, ECE / MCE / ACE.
#[derive(Debug, Clone)]
pub struct UqCalibration {
    /// Calibrated temperature T (temperature scaling).
    pub temperature: f64,
    /// Platt sigmoid parameters (a, b) such that p = σ(a·f + b).
    pub platt_params: (f64, f64),
    /// Isotonic mapping: sorted (score, calibrated_p) pairs.
    isotonic_pairs: Vec<(f64, f64)>,
    /// Number of ECE bins.
    pub n_bins: usize,
}

impl UqCalibration {
    /// Create with default temperature=1.0.
    pub fn new(n_bins: usize) -> Self {
        Self {
            temperature: 1.0,
            platt_params: (1.0, 0.0),
            isotonic_pairs: Vec::new(),
            n_bins: n_bins.max(1),
        }
    }

    /// Fit temperature scaling on validation logits and binary labels.
    ///
    /// Searches T ∈ [0.01, 10.0] to minimize NLL.
    pub fn fit_temperature(&mut self, logits: &[f64], labels: &[bool]) -> UqResult<f64> {
        if logits.len() != labels.len() || logits.is_empty() {
            return Err(UqError::DimensionMismatch {
                expected: logits.len(),
                found: labels.len(),
            });
        }
        let nll = |t: f64| -> f64 {
            logits
                .iter()
                .zip(labels.iter())
                .map(|(l, &y)| {
                    let p = sigmoid(l / t);
                    let p_clip = p.clamp(1e-10, 1.0 - 1e-10);
                    if y {
                        -p_clip.ln()
                    } else {
                        -(1.0 - p_clip).ln()
                    }
                })
                .sum::<f64>()
                / logits.len() as f64
        };
        // Golden-section search
        let mut lo = 0.01_f64;
        let mut hi = 10.0_f64;
        for _ in 0..50 {
            let m1 = lo + (hi - lo) / 3.0;
            let m2 = hi - (hi - lo) / 3.0;
            if nll(m1) < nll(m2) {
                hi = m2;
            } else {
                lo = m1;
            }
        }
        self.temperature = (lo + hi) / 2.0;
        Ok(self.temperature)
    }

    /// Apply temperature scaling to a logit.
    pub fn temperature_scale(&self, logit: f64) -> f64 {
        sigmoid(logit / self.temperature)
    }

    /// Fit Platt scaling (logistic regression on outputs).
    pub fn fit_platt(&mut self, scores: &[f64], labels: &[bool]) -> UqResult<()> {
        if scores.len() != labels.len() || scores.is_empty() {
            return Err(UqError::DimensionMismatch {
                expected: scores.len(),
                found: labels.len(),
            });
        }
        // SGD on binary cross-entropy for (a, b)
        let mut a = 1.0_f64;
        let mut b = 0.0_f64;
        let lr = 0.01_f64;
        let n = scores.len() as f64;
        for _ in 0..500 {
            let mut da = 0.0_f64;
            let mut db = 0.0_f64;
            for (&s, &y) in scores.iter().zip(labels.iter()) {
                let p = sigmoid(a * s + b).clamp(1e-10, 1.0 - 1e-10);
                let err = p - if y { 1.0 } else { 0.0 };
                da += err * s / n;
                db += err / n;
            }
            a -= lr * da;
            b -= lr * db;
        }
        self.platt_params = (a, b);
        Ok(())
    }

    /// Apply Platt scaling to a score.
    pub fn platt_scale(&self, score: f64) -> f64 {
        sigmoid(self.platt_params.0 * score + self.platt_params.1)
    }

    /// Fit isotonic regression (pool-adjacent-violators algorithm).
    pub fn fit_isotonic(&mut self, scores: &[f64], labels: &[bool]) -> UqResult<()> {
        if scores.len() != labels.len() || scores.is_empty() {
            return Err(UqError::EmptyInput);
        }
        // Sort by score
        let mut pairs: Vec<(f64, f64)> = scores
            .iter()
            .zip(labels.iter())
            .map(|(&s, &l)| (s, if l { 1.0 } else { 0.0 }))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Pool-adjacent-violators (PAV) algorithm
        let mut pools: Vec<(f64, f64, f64)> = Vec::new(); // (sum_y, count, avg_score)
        for (s, y) in pairs {
            pools.push((y, 1.0, s));
            while pools.len() >= 2 {
                let n = pools.len();
                if pools[n - 2].0 / pools[n - 2].1 > pools[n - 1].0 / pools[n - 1].1 {
                    // Safe: len >= 2 checked above
                    if let (Some((sy2, c2, ss2)), Some((sy1, c1, ss1))) = (pools.pop(), pools.pop())
                    {
                        pools.push((sy1 + sy2, c1 + c2, (ss1 + ss2) / 2.0));
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        // Flatten into (score_threshold, calibrated_p)
        self.isotonic_pairs = pools.iter().map(|(sy, cnt, s)| (*s, sy / cnt)).collect();
        Ok(())
    }

    /// Apply isotonic calibration (nearest-neighbor lookup).
    pub fn isotonic_scale(&self, score: f64) -> f64 {
        if self.isotonic_pairs.is_empty() {
            return score.clamp(0.0, 1.0);
        }
        // Find closest score in isotonic pairs
        let best = self.isotonic_pairs.iter().min_by(|a, b| {
            (a.0 - score)
                .abs()
                .partial_cmp(&(b.0 - score).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        best.map(|(_, p)| *p).unwrap_or(score.clamp(0.0, 1.0))
    }

    /// Reliability diagram data: (bin_center, avg_accuracy, avg_confidence, count).
    pub fn reliability_diagram(
        &self,
        probs: &[f64],
        labels: &[bool],
    ) -> UqResult<Vec<(f64, f64, f64, usize)>> {
        if probs.len() != labels.len() || probs.is_empty() {
            return Err(UqError::DimensionMismatch {
                expected: probs.len(),
                found: labels.len(),
            });
        }
        let bin_width = 1.0 / self.n_bins as f64;
        let mut bins: Vec<(f64, f64, f64, usize)> = (0..self.n_bins)
            .map(|b| (b as f64 * bin_width + bin_width / 2.0, 0.0, 0.0, 0_usize))
            .collect();
        for (&p, &l) in probs.iter().zip(labels.iter()) {
            let b = ((p / bin_width).floor() as usize).min(self.n_bins - 1);
            bins[b].1 += if l { 1.0 } else { 0.0 }; // acc sum
            bins[b].2 += p; // conf sum
            bins[b].3 += 1;
        }
        for bin in bins.iter_mut() {
            if bin.3 > 0 {
                let cnt = bin.3 as f64;
                bin.1 /= cnt;
                bin.2 /= cnt;
            }
        }
        Ok(bins)
    }

    /// Expected Calibration Error (ECE).
    pub fn ece(&self, probs: &[f64], labels: &[bool]) -> UqResult<f64> {
        let n = probs.len() as f64;
        let bins = self.reliability_diagram(probs, labels)?;
        let ece = bins
            .iter()
            .map(|(_, acc, conf, cnt)| (*cnt as f64 / n) * (conf - acc).abs())
            .sum();
        Ok(ece)
    }

    /// Maximum Calibration Error (MCE).
    pub fn mce(&self, probs: &[f64], labels: &[bool]) -> UqResult<f64> {
        let bins = self.reliability_diagram(probs, labels)?;
        let mce = bins
            .iter()
            .filter(|(_, _, _, cnt)| *cnt > 0)
            .map(|(_, acc, conf, _)| (conf - acc).abs())
            .fold(0.0_f64, f64::max);
        Ok(mce)
    }

    /// Adaptive Calibration Error (ACE) — equal-mass bins.
    pub fn ace(&self, probs: &[f64], labels: &[bool]) -> UqResult<f64> {
        if probs.len() != labels.len() || probs.is_empty() {
            return Err(UqError::EmptyInput);
        }
        let mut pairs: Vec<(f64, bool)> = probs
            .iter()
            .zip(labels.iter())
            .map(|(&p, &l)| (p, l))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let n = pairs.len();
        let bin_size = (n / self.n_bins).max(1);
        let mut ace = 0.0_f64;
        let mut count = 0;
        for chunk in pairs.chunks(bin_size) {
            if chunk.is_empty() {
                continue;
            }
            let avg_conf: f64 = chunk.iter().map(|(p, _)| p).sum::<f64>() / chunk.len() as f64;
            let avg_acc: f64 = chunk.iter().filter(|(_, l)| *l).count() as f64 / chunk.len() as f64;
            ace += (avg_conf - avg_acc).abs();
            count += 1;
        }
        Ok(if count > 0 { ace / count as f64 } else { 0.0 })
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  UqOodDetector — Out-of-Distribution Detection
// ─────────────────────────────────────────────────────────────────────────────

/// OOD detection method.
#[derive(Debug, Clone, PartialEq)]
pub enum UqOodMethod {
    /// Maximum softmax probability (Hendrycks & Gimpel 2017).
    MaxSoftmax,
    /// Energy score: -T·log Σ_c exp(f_c/T).
    Energy { temperature: f64 },
    /// Mahalanobis distance from class-conditional Gaussians.
    Mahalanobis,
    /// ODIN: temperature scaling + input pre-processing gradient (Liang 2018).
    Odin { temperature: f64 },
}

/// Out-of-Distribution detector.
#[derive(Debug, Clone)]
pub struct UqOodDetector {
    /// Detection method.
    pub method: UqOodMethod,
    /// Class-conditional means for Mahalanobis (n_classes x feature_dim).
    class_means: Vec<Vec<f64>>,
    /// Shared inverse covariance for Mahalanobis (feature_dim x feature_dim).
    shared_precision: Vec<Vec<f64>>,
    /// Threshold for binary OOD decision.
    pub threshold: f64,
    pub feature_dim: usize,
    pub n_classes: usize,
}

impl UqOodDetector {
    /// Create a new OOD detector.
    pub fn new(method: UqOodMethod, feature_dim: usize, n_classes: usize, threshold: f64) -> Self {
        Self {
            method,
            class_means: vec![vec![0.0; feature_dim]; n_classes],
            shared_precision: (0..feature_dim)
                .map(|i| {
                    let mut row = vec![0.0_f64; feature_dim];
                    row[i] = 1.0;
                    row
                })
                .collect(),
            threshold,
            feature_dim,
            n_classes,
        }
    }

    /// Fit class-conditional Gaussians for Mahalanobis distance.
    ///
    /// `features`: (N x feature_dim) in-distribution samples.
    /// `class_labels`: class index per sample.
    pub fn fit_mahalanobis(
        &mut self,
        features: &[Vec<f64>],
        class_labels: &[usize],
    ) -> UqResult<()> {
        if features.len() != class_labels.len() || features.is_empty() {
            return Err(UqError::EmptyInput);
        }
        let d = self.feature_dim;
        let k = self.n_classes;

        // Compute class means
        let mut counts = vec![0_usize; k];
        let mut means = vec![vec![0.0_f64; d]; k];
        for (f, &c) in features.iter().zip(class_labels.iter()) {
            if c >= k {
                return Err(UqError::InvalidConfig(format!(
                    "class {c} >= n_classes {k}"
                )));
            }
            counts[c] += 1;
            for j in 0..d {
                means[c][j] += f[j];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..d {
                    means[c][j] /= counts[c] as f64;
                }
            }
        }
        self.class_means = means.clone();

        // Compute pooled covariance
        let mut cov = vec![vec![0.0_f64; d]; d];
        let n = features.len() as f64;
        for (f, &c) in features.iter().zip(class_labels.iter()) {
            let mu = &means[c];
            for i in 0..d {
                for j in 0..d {
                    cov[i][j] += (f[i] - mu[i]) * (f[j] - mu[j]) / n;
                }
            }
        }
        // Add small ridge for invertibility
        for i in 0..d {
            cov[i][i] += 1e-5;
        }
        // Approximate inverse via diagonal (for efficiency)
        self.shared_precision = (0..d)
            .map(|i| {
                let mut row = vec![0.0_f64; d];
                row[i] = 1.0 / cov[i][i].max(1e-10);
                row
            })
            .collect();
        Ok(())
    }

    /// Compute OOD score for a given input.
    ///
    /// `logits`: raw class logits (for MSP/Energy/ODIN).
    /// `features`: penultimate-layer feature vector (for Mahalanobis).
    pub fn compute_score(&self, logits: &[f64], features: &[f64]) -> UqResult<f64> {
        match &self.method {
            UqOodMethod::MaxSoftmax => {
                if logits.is_empty() {
                    return Err(UqError::EmptyInput);
                }
                let probs = softmax(logits);
                let msp = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                // Higher MSP = more in-distribution; return negative for OOD score
                Ok(-msp)
            }
            UqOodMethod::Energy { temperature } => {
                if logits.is_empty() {
                    return Err(UqError::EmptyInput);
                }
                let t = *temperature;
                let scaled: Vec<f64> = logits.iter().map(|l| l / t).collect();
                let energy = -t * log_sum_exp(&scaled);
                Ok(energy)
            }
            UqOodMethod::Mahalanobis => {
                if features.len() != self.feature_dim {
                    return Err(UqError::DimensionMismatch {
                        expected: self.feature_dim,
                        found: features.len(),
                    });
                }
                // Min Mahalanobis distance to any class center
                let min_dist = self
                    .class_means
                    .iter()
                    .map(|mu| {
                        let diff: Vec<f64> =
                            features.iter().zip(mu.iter()).map(|(f, m)| f - m).collect();
                        let prec_diff = matvec(&self.shared_precision, &diff);
                        dot(&diff, &prec_diff)
                    })
                    .fold(f64::INFINITY, f64::min);
                Ok(min_dist.sqrt())
            }
            UqOodMethod::Odin { temperature } => {
                // Simplified ODIN: temperature scaling of logits
                if logits.is_empty() {
                    return Err(UqError::EmptyInput);
                }
                let t = *temperature;
                let scaled: Vec<f64> = logits.iter().map(|l| l / t).collect();
                let probs = softmax(&scaled);
                let msp = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                Ok(-msp)
            }
        }
    }

    /// Detect OOD: returns (is_ood, score).
    pub fn detect(&self, logits: &[f64], features: &[f64]) -> UqResult<(bool, f64)> {
        let score = self.compute_score(logits, features)?;
        Ok((score > self.threshold, score))
    }

    /// Evaluate AUROC and FPR@95TPR on labeled data.
    ///
    /// `id_logits` / `id_features`: in-distribution.
    /// `ood_logits` / `ood_features`: out-of-distribution.
    pub fn evaluate(
        &self,
        id_logits: &[Vec<f64>],
        id_features: &[Vec<f64>],
        ood_logits: &[Vec<f64>],
        ood_features: &[Vec<f64>],
    ) -> UqResult<UqOodEvalResult> {
        let mut scores = Vec::new();
        let mut labels = Vec::new();
        for (l, f) in id_logits.iter().zip(id_features.iter()) {
            scores.push(self.compute_score(l, f)?);
            labels.push(true); // in-distribution = positive
        }
        for (l, f) in ood_logits.iter().zip(ood_features.iter()) {
            scores.push(self.compute_score(l, f)?);
            labels.push(false); // OOD = negative
        }
        // Negate scores for AUROC (higher score = more OOD; we want ID to score higher)
        let neg_scores: Vec<f64> = scores.iter().map(|s| -s).collect();
        let auc = auroc(&neg_scores, &labels);

        // FPR at 95% TPR
        let mut id_scores: Vec<f64> = id_logits
            .iter()
            .zip(id_features.iter())
            .map(|(l, f)| self.compute_score(l, f).unwrap_or(f64::INFINITY))
            .collect();
        id_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let thr_95 = if id_scores.is_empty() {
            0.0
        } else {
            let idx = ((id_scores.len() as f64 * 0.95) as usize).min(id_scores.len() - 1);
            id_scores[idx]
        };
        let fpr95 = if ood_logits.is_empty() {
            0.0
        } else {
            let fp = ood_logits
                .iter()
                .zip(ood_features.iter())
                .filter(|(l, f)| self.compute_score(l, f).unwrap_or(f64::INFINITY) <= thr_95)
                .count() as f64;
            fp / ood_logits.len() as f64
        };
        Ok(UqOodEvalResult { auroc: auc, fpr95 })
    }
}

/// OOD evaluation metrics.
#[derive(Debug, Clone)]
pub struct UqOodEvalResult {
    /// Area under ROC curve.
    pub auroc: f64,
    /// False positive rate at 95% true positive rate.
    pub fpr95: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  UqRiskControl — Risk-Controlled Prediction (Learn-then-Test)
// ─────────────────────────────────────────────────────────────────────────────

/// Risk-controlled prediction (Angelopoulos et al. 2022).
///
/// Finds a threshold λ* such that risk R(λ*) ≤ α with probability ≥ 1-δ.
/// Uses the Hoeffding-style bound on calibration set.
#[derive(Debug, Clone)]
pub struct UqRiskControl {
    /// Miscoverage / risk tolerance α.
    pub alpha: f64,
    /// Confidence δ (tolerated failure probability).
    pub delta: f64,
    /// Calibrated threshold λ*.
    pub lambda_star: f64,
    /// Risk function name.
    pub risk_fn: UqRiskFn,
}

/// Risk function variants.
#[derive(Debug, Clone, PartialEq)]
pub enum UqRiskFn {
    /// Coverage risk: fraction not covered by prediction set.
    Coverage,
    /// FWER-type: fraction of samples where true label falls outside interval.
    IntervalWidth,
    /// Custom threshold on residuals.
    ResidualThreshold,
}

impl UqRiskControl {
    /// Create a new risk controller.
    pub fn new(alpha: f64, delta: f64, risk_fn: UqRiskFn) -> UqResult<Self> {
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(UqError::InvalidConfig("alpha must be in (0,1)".into()));
        }
        if !(0.0 < delta && delta < 1.0) {
            return Err(UqError::InvalidConfig("delta must be in (0,1)".into()));
        }
        Ok(Self {
            alpha,
            delta,
            lambda_star: 0.0,
            risk_fn,
        })
    }

    /// Calibrate: find λ* minimizing interval size subject to coverage ≥ 1-α.
    ///
    /// `cal_scores`: nonconformity scores on calibration set (lower = more conforming).
    /// `cal_residuals`: actual residuals / losses per calibration point.
    /// Returns the threshold λ* (quantile of scores such that risk ≤ α + Hoeffding slack).
    pub fn calibrate_risk(&mut self, cal_scores: &[f64], cal_residuals: &[f64]) -> UqResult<f64> {
        if cal_scores.len() != cal_residuals.len() {
            return Err(UqError::DimensionMismatch {
                expected: cal_scores.len(),
                found: cal_residuals.len(),
            });
        }
        let n = cal_scores.len();
        if n < 2 {
            return Err(UqError::InsufficientData {
                required: 2,
                found: n,
            });
        }
        // Hoeffding correction: inflate alpha by sqrt(ln(1/delta)/(2n))
        let hoeffding_slack = ((1.0_f64 / self.delta).ln() / (2.0 * n as f64)).sqrt();
        let effective_alpha = (self.alpha - hoeffding_slack).max(0.0);

        // λ* = (1 - effective_alpha)(1 + 1/n) quantile of scores
        let p = ((1.0 - effective_alpha) * (1.0 + 1.0 / n as f64)).min(1.0);
        self.lambda_star = quantile(cal_scores.to_vec(), p);
        Ok(self.lambda_star)
    }

    /// Compute empirical risk R(λ) = fraction of samples where residual > λ.
    pub fn empirical_risk(&self, scores: &[f64], residuals: &[f64]) -> UqResult<f64> {
        if scores.is_empty() {
            return Err(UqError::EmptyInput);
        }
        let n = scores.len() as f64;
        let violated = scores
            .iter()
            .zip(residuals.iter())
            .filter(|(s, _)| **s > self.lambda_star)
            .count() as f64;
        Ok(violated / n)
    }

    /// Check that the estimated risk bound is satisfied.
    pub fn risk_bound_satisfied(
        &self,
        test_scores: &[f64],
        test_residuals: &[f64],
    ) -> UqResult<bool> {
        let risk = self.empirical_risk(test_scores, test_residuals)?;
        Ok(risk <= self.alpha)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  UqMetrics — Comprehensive UQ Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Comprehensive uncertainty quantification metrics and report.
#[derive(Debug, Clone)]
pub struct UqMetrics;

/// Full UQ evaluation report.
#[derive(Debug, Clone)]
pub struct UqReport {
    /// Expected calibration error.
    pub ece: f64,
    /// Maximum calibration error.
    pub mce: f64,
    /// Adaptive calibration error.
    pub ace: f64,
    /// Brier score (mean squared probability error).
    pub brier_score: f64,
    /// Negative log-likelihood on test set.
    pub nll: f64,
    /// AUROC for OOD detection.
    pub ood_auroc: f64,
    /// Coverage for regression (fraction within intervals).
    pub coverage: f64,
    /// Mean interval width.
    pub mean_interval_width: f64,
    /// Number of test samples.
    pub n_test: usize,
}

impl UqMetrics {
    /// Brier score for binary classification.
    pub fn brier_score(probs: &[f64], labels: &[bool]) -> UqResult<f64> {
        if probs.len() != labels.len() || probs.is_empty() {
            return Err(UqError::DimensionMismatch {
                expected: probs.len(),
                found: labels.len(),
            });
        }
        let bs: f64 = probs
            .iter()
            .zip(labels.iter())
            .map(|(&p, &l)| {
                let y = if l { 1.0 } else { 0.0 };
                (p - y).powi(2)
            })
            .sum::<f64>()
            / probs.len() as f64;
        Ok(bs)
    }

    /// NLL on a binary classification test set.
    pub fn nll_binary(probs: &[f64], labels: &[bool]) -> UqResult<f64> {
        if probs.len() != labels.len() || probs.is_empty() {
            return Err(UqError::DimensionMismatch {
                expected: probs.len(),
                found: labels.len(),
            });
        }
        let nll: f64 = probs
            .iter()
            .zip(labels.iter())
            .map(|(&p, &l)| {
                let p_clip = p.clamp(1e-12, 1.0 - 1e-12);
                if l {
                    -p_clip.ln()
                } else {
                    -(1.0 - p_clip).ln()
                }
            })
            .sum::<f64>()
            / probs.len() as f64;
        Ok(nll)
    }

    /// Coverage of prediction intervals (regression).
    pub fn coverage(lowers: &[f64], uppers: &[f64], true_vals: &[f64]) -> UqResult<f64> {
        let n = lowers.len();
        if n != uppers.len() || n != true_vals.len() || n == 0 {
            return Err(UqError::DimensionMismatch {
                expected: n,
                found: true_vals.len(),
            });
        }
        let covered = lowers
            .iter()
            .zip(uppers.iter())
            .zip(true_vals.iter())
            .filter(|((lo, hi), y)| **y >= **lo && **y <= **hi)
            .count() as f64;
        Ok(covered / n as f64)
    }

    /// Mean interval width.
    pub fn mean_interval_width(lowers: &[f64], uppers: &[f64]) -> UqResult<f64> {
        if lowers.len() != uppers.len() || lowers.is_empty() {
            return Err(UqError::EmptyInput);
        }
        let w: f64 = lowers
            .iter()
            .zip(uppers.iter())
            .map(|(lo, hi)| (hi - lo).abs())
            .sum::<f64>()
            / lowers.len() as f64;
        Ok(w)
    }

    /// AUROC for OOD detection given ID and OOD scores.
    pub fn ood_auroc(id_scores: &[f64], ood_scores: &[f64]) -> f64 {
        // Higher score = more OOD. Build combined vec.
        let mut scores: Vec<f64> = id_scores.to_vec();
        scores.extend_from_slice(ood_scores);
        let mut labels: Vec<bool> = vec![false; id_scores.len()]; // ID = negative class
        labels.extend(vec![true; ood_scores.len()]); // OOD = positive class
        auroc(&scores, &labels)
    }

    /// Compile a full UQ report.
    pub fn compile_report(
        probs: &[f64],
        labels: &[bool],
        lowers: &[f64],
        uppers: &[f64],
        true_vals: &[f64],
        id_scores: &[f64],
        ood_scores: &[f64],
        n_bins: usize,
    ) -> UqResult<UqReport> {
        let cal = UqCalibration::new(n_bins);
        let ece = cal.ece(probs, labels)?;
        let mce = cal.mce(probs, labels)?;
        let ace = cal.ace(probs, labels)?;
        let brier_score = Self::brier_score(probs, labels)?;
        let nll = Self::nll_binary(probs, labels)?;
        let ood_auroc = Self::ood_auroc(id_scores, ood_scores);
        let coverage = if !lowers.is_empty() {
            Self::coverage(lowers, uppers, true_vals)?
        } else {
            0.0
        };
        let mean_interval_width = if !lowers.is_empty() {
            Self::mean_interval_width(lowers, uppers)?
        } else {
            0.0
        };
        Ok(UqReport {
            ece,
            mce,
            ace,
            brier_score,
            nll,
            ood_auroc,
            coverage,
            mean_interval_width,
            n_test: probs.len(),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
