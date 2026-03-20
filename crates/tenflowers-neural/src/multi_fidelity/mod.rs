//! Multi-Fidelity Learning & Surrogate Optimization.
//!
//! Implements multi-fidelity modelling, surrogate-based optimization, and
//! bandit-style hyperparameter optimizers that exploit cheap low-fidelity
//! approximations before committing budget to expensive high-fidelity
//! evaluations.
//!
//! ## Main components
//!
//! | Type | Description |
//! |------|-------------|
//! | [`MfiFidelityLevel`] | Abstract fidelity level with cost model |
//! | [`MfiDataset`] | Paired samples at each fidelity level |
//! | [`MfiLinearARModel`] | Kennedy & O'Hagan (2000) linear AR model |
//! | [`MfiGaussianProcess`] | Nonlinear AR GP (Perdikaris 2017) |
//! | [`MfiNeuralNetworkAR`] | Neural AR model (Meng & Karniadakis 2020) |
//! | [`MfiActiveLearner`] | Uncertainty-to-cost ratio acquisition |
//! | [`MfiSuccessiveHalving`] | Successive halving bracket for HPO |
//! | [`MfiHyperband`] | Hyperband (Li 2017) over multiple brackets |
//! | [`MfiBOHB`] | BOHB: KDE-guided hyperband (Falkner 2018) |
//! | [`MfiEnsemble`] | Stacked multi-fidelity ensemble |
//! | [`MfiMetrics`] / [`MfiReport`] | Evaluation utilities |
//!
//! All randomness is sourced from `scirs2_core::random` — no `rand` crate.
//! No `unsafe`. No `unwrap()`.

#[cfg(test)]
mod tests;

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Internal error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the multi-fidelity module.
#[derive(Debug)]
pub enum MfiError {
    /// Input slice was empty or had mismatched lengths.
    InvalidInput { context: String },
    /// A numerical operation failed (e.g. singular matrix, Cholesky failure).
    NumericalError { context: String },
    /// Model was queried before being trained.
    NotFitted { model: String },
    /// A parameter value is outside its valid range.
    InvalidParameter { name: String, reason: String },
}

impl fmt::Display for MfiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { context } => write!(f, "invalid input: {context}"),
            Self::NumericalError { context } => write!(f, "numerical error: {context}"),
            Self::NotFitted { model } => write!(f, "model '{model}' is not fitted"),
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter '{name}': {reason}")
            }
        }
    }
}

impl std::error::Error for MfiError {}

type Result<T> = std::result::Result<T, MfiError>;

// ─────────────────────────────────────────────────────────────────────────────
// 1. MfiFidelityLevel — fidelity abstraction with cost model
// ─────────────────────────────────────────────────────────────────────────────

/// A single fidelity level with an associated cost model.
///
/// Levels are ordered 0 (cheapest) … L-1 (most expensive / ground truth).
#[derive(Debug, Clone)]
pub struct MfiFidelityLevel {
    /// Human-readable label (e.g. "coarse", "medium", "fine").
    pub label: String,
    /// Relative cost compared to the cheapest level (must be ≥ 1.0).
    pub relative_cost: f64,
    /// Estimated accuracy relative to the highest fidelity (0 … 1).
    pub accuracy_estimate: f64,
}

impl MfiFidelityLevel {
    /// Create a new fidelity level.
    ///
    /// `relative_cost` must be ≥ 1.0 and `accuracy_estimate` in [0, 1].
    pub fn new(
        label: impl Into<String>,
        relative_cost: f64,
        accuracy_estimate: f64,
    ) -> Result<Self> {
        if relative_cost < 1.0 {
            return Err(MfiError::InvalidParameter {
                name: "relative_cost".to_string(),
                reason: format!("must be ≥ 1.0, got {relative_cost}"),
            });
        }
        if !(0.0..=1.0).contains(&accuracy_estimate) {
            return Err(MfiError::InvalidParameter {
                name: "accuracy_estimate".to_string(),
                reason: format!("must be in [0,1], got {accuracy_estimate}"),
            });
        }
        Ok(Self {
            label: label.into(),
            relative_cost,
            accuracy_estimate,
        })
    }

    /// Information-per-cost ratio: higher is better.
    pub fn efficiency(&self) -> f64 {
        self.accuracy_estimate / self.relative_cost
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. MfiDataset — samples at each fidelity level
// ─────────────────────────────────────────────────────────────────────────────

/// Observed dataset at a single fidelity level.
#[derive(Debug, Clone, Default)]
pub struct MfiFidelityData {
    /// Input vectors.
    pub x: Vec<Vec<f64>>,
    /// Corresponding observed outputs.
    pub y: Vec<f64>,
}

impl MfiFidelityData {
    /// Append a single (x, y) observation.
    pub fn push(&mut self, x_point: Vec<f64>, y_val: f64) {
        self.x.push(x_point);
        self.y.push(y_val);
    }

    /// Number of observations.
    pub fn len(&self) -> usize {
        self.y.len()
    }

    /// Whether there are no observations.
    pub fn is_empty(&self) -> bool {
        self.y.is_empty()
    }
}

/// Multi-fidelity dataset holding observations at every fidelity level.
///
/// `levels[0]` is the cheapest; `levels[n_levels-1]` is the most expensive.
#[derive(Debug, Clone)]
pub struct MfiDataset {
    /// Fidelity-level definitions.
    pub levels: Vec<MfiFidelityLevel>,
    /// Observations at each level (same length as `levels`).
    pub data: Vec<MfiFidelityData>,
    /// Empirical inter-level correlation matrix (optional; shape L×L).
    pub correlations: Vec<Vec<f64>>,
}

impl MfiDataset {
    /// Construct an empty multi-fidelity dataset.
    pub fn new(levels: Vec<MfiFidelityLevel>) -> Result<Self> {
        if levels.is_empty() {
            return Err(MfiError::InvalidInput {
                context: "MfiDataset must have at least one fidelity level".to_string(),
            });
        }
        let n = levels.len();
        let data = (0..n).map(|_| MfiFidelityData::default()).collect();
        let correlations = vec![vec![1.0; n]; n];
        Ok(Self {
            levels,
            data,
            correlations,
        })
    }

    /// Add an observation at the given fidelity `level_idx`.
    pub fn add_observation(&mut self, level_idx: usize, x: Vec<f64>, y: f64) -> Result<()> {
        if level_idx >= self.data.len() {
            return Err(MfiError::InvalidInput {
                context: format!("level_idx {level_idx} out of range"),
            });
        }
        self.data[level_idx].push(x, y);
        Ok(())
    }

    /// Estimate Pearson correlations between every pair of fidelity levels using
    /// all observations that share the same input point (matched by index).
    pub fn estimate_correlations(&mut self) {
        let n = self.levels.len();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    self.correlations[i][j] = 1.0;
                    continue;
                }
                let n_i = self.data[i].len();
                let n_j = self.data[j].len();
                let k = n_i.min(n_j);
                if k < 2 {
                    self.correlations[i][j] = 0.0;
                    continue;
                }
                let yi: &[f64] = &self.data[i].y[..k];
                let yj: &[f64] = &self.data[j].y[..k];
                self.correlations[i][j] = pearson_correlation(yi, yj);
            }
        }
    }

    /// Return the index of the highest (most expensive) fidelity level.
    pub fn highest_level(&self) -> usize {
        self.levels.len() - 1
    }

    /// Total number of observations across all levels.
    pub fn total_observations(&self) -> usize {
        self.data.iter().map(|d| d.len()).sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. MfiLinearARModel — Kennedy & O'Hagan (2000)
// ─────────────────────────────────────────────────────────────────────────────

/// Linear auto-regressive multi-fidelity model (Kennedy & O'Hagan 2000).
///
/// At each level l > 0:
/// ```text
/// y_l(x) = ρ_l · y_{l-1}(x) + δ_l(x)
/// ```
/// where `ρ_l` is estimated via linear regression on paired samples.
#[derive(Debug, Clone)]
pub struct MfiLinearARModel {
    /// Number of fidelity levels.
    pub n_levels: usize,
    /// ρ scaling factors (length n_levels - 1); rho\[l\] multiplies level l.
    pub rho: Vec<f64>,
    /// Residual means δ_l (length n_levels).
    pub delta_mean: Vec<f64>,
    /// Whether the model has been fitted.
    pub fitted: bool,
    /// Training x at each level.
    x_train: Vec<Vec<Vec<f64>>>,
    /// Training y at each level.
    y_train: Vec<Vec<f64>>,
}

impl MfiLinearARModel {
    /// Create an unfitted model for `n_levels` fidelity levels.
    pub fn new(n_levels: usize) -> Result<Self> {
        if n_levels < 2 {
            return Err(MfiError::InvalidParameter {
                name: "n_levels".to_string(),
                reason: "must be at least 2 for AR model".to_string(),
            });
        }
        Ok(Self {
            n_levels,
            rho: vec![1.0; n_levels - 1],
            delta_mean: vec![0.0; n_levels],
            fitted: false,
            x_train: vec![Vec::new(); n_levels],
            y_train: vec![Vec::new(); n_levels],
        })
    }

    /// Fit the AR model using the provided dataset.
    ///
    /// For each level l ≥ 1, the ρ_l is estimated as the OLS slope of y_l vs
    /// y_{l-1} evaluated at the first `k = min(n_l, n_{l-1})` paired samples.
    pub fn fit(&mut self, dataset: &MfiDataset) -> Result<()> {
        if dataset.data.len() != self.n_levels {
            return Err(MfiError::InvalidInput {
                context: format!(
                    "dataset has {} levels but model expects {}",
                    dataset.data.len(),
                    self.n_levels
                ),
            });
        }
        // Store training data for prediction later
        for l in 0..self.n_levels {
            self.x_train[l] = dataset.data[l].x.clone();
            self.y_train[l] = dataset.data[l].y.clone();
        }

        // Level 0: compute residual mean
        if !self.y_train[0].is_empty() {
            let mean0: f64 = self.y_train[0].iter().sum::<f64>() / self.y_train[0].len() as f64;
            self.delta_mean[0] = mean0;
        }

        // Levels 1..n_levels: estimate rho via OLS slope
        for l in 1..self.n_levels {
            let n_paired = self.y_train[l].len().min(self.y_train[l - 1].len());
            if n_paired < 2 {
                self.rho[l - 1] = 1.0;
                self.delta_mean[l] = if self.y_train[l].is_empty() {
                    0.0
                } else {
                    self.y_train[l].iter().sum::<f64>() / self.y_train[l].len() as f64
                };
                continue;
            }
            let y_low = &self.y_train[l - 1][..n_paired];
            let y_high = &self.y_train[l][..n_paired];

            // OLS: rho = cov(y_low, y_high) / var(y_low)
            let mean_low: f64 = y_low.iter().sum::<f64>() / n_paired as f64;
            let mean_high: f64 = y_high.iter().sum::<f64>() / n_paired as f64;

            let cov: f64 = y_low
                .iter()
                .zip(y_high.iter())
                .map(|(a, b)| (a - mean_low) * (b - mean_high))
                .sum::<f64>();
            let var_low: f64 = y_low.iter().map(|a| (a - mean_low).powi(2)).sum::<f64>();

            self.rho[l - 1] = if var_low < 1e-12 { 1.0 } else { cov / var_low };

            // Residual mean: δ_l = mean(y_l - ρ_l · y_{l-1})
            let delta_sum: f64 = y_high
                .iter()
                .zip(y_low.iter())
                .map(|(h, lo)| h - self.rho[l - 1] * lo)
                .sum::<f64>();
            self.delta_mean[l] = delta_sum / n_paired as f64;
        }

        self.fitted = true;
        Ok(())
    }

    /// Predict the highest-fidelity output at `x` by recursive AR propagation.
    ///
    /// Uses nearest-neighbour lookup within each level for the missing values.
    pub fn predict(&self, x: &[f64]) -> Result<f64> {
        if !self.fitted {
            return Err(MfiError::NotFitted {
                model: "MfiLinearARModel".to_string(),
            });
        }

        // Start from level 0 prediction (nearest-neighbour or mean)
        let mut pred = nearest_neighbour_predict(&self.x_train[0], &self.y_train[0], x)
            .unwrap_or(self.delta_mean[0]);

        // Recursively apply AR: y_l = rho_l * y_{l-1} + delta_l
        for l in 1..self.n_levels {
            let nn_high = nearest_neighbour_predict(&self.x_train[l], &self.y_train[l], x);
            pred = match nn_high {
                Some(v) => v,
                None => self.rho[l - 1] * pred + self.delta_mean[l],
            };
        }

        Ok(pred)
    }

    /// Predict a batch of points.
    pub fn predict_batch(&self, xs: &[Vec<f64>]) -> Result<Vec<f64>> {
        xs.iter().map(|x| self.predict(x)).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. MfiGaussianProcess — Nonlinear AR GP (Perdikaris 2017)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the per-level GP surrogate.
#[derive(Debug, Clone)]
pub struct MfiGpConfig {
    /// RBF length scale.
    pub length_scale: f64,
    /// Signal standard deviation σ.
    pub signal_std: f64,
    /// Observation noise σ_n.
    pub noise_std: f64,
}

impl Default for MfiGpConfig {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            signal_std: 1.0,
            noise_std: 1e-2,
        }
    }
}

/// Per-level GP surrogate used inside MfiGaussianProcess.
#[derive(Debug, Clone)]
struct LevelGp {
    config: MfiGpConfig,
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    /// Pre-computed (K + σ_n² I)⁻¹.
    k_inv: Vec<Vec<f64>>,
    /// Pre-computed K_inv @ y (normalized).
    alpha: Vec<f64>,
    y_mean: f64,
    y_std: f64,
    fitted: bool,
}

impl LevelGp {
    fn new(config: MfiGpConfig) -> Self {
        Self {
            config,
            x_train: Vec::new(),
            y_train: Vec::new(),
            k_inv: Vec::new(),
            alpha: Vec::new(),
            y_mean: 0.0,
            y_std: 1.0,
            fitted: false,
        }
    }

    fn rbf_kernel(&self, a: &[f64], b: &[f64]) -> f64 {
        let l2 = l2_sq(a, b);
        let ls = self.config.length_scale;
        self.config.signal_std.powi(2) * (-l2 / (2.0 * ls * ls)).exp()
    }

    fn kernel_matrix(&self, xs: &[Vec<f64>], xt: &[Vec<f64>]) -> Vec<Vec<f64>> {
        xs.iter()
            .map(|a| xt.iter().map(|b| self.rbf_kernel(a, b)).collect())
            .collect()
    }

    fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<()> {
        if x.is_empty() {
            return Err(MfiError::InvalidInput {
                context: "LevelGp::fit called with empty data".to_string(),
            });
        }
        let n = y.len();
        self.x_train = x.to_vec();
        self.y_train = y.to_vec();

        let sum: f64 = y.iter().sum();
        self.y_mean = sum / n as f64;
        let var: f64 = y.iter().map(|v| (v - self.y_mean).powi(2)).sum::<f64>() / n as f64;
        self.y_std = if var < 1e-12 { 1.0 } else { var.sqrt() };

        let y_norm: Vec<f64> = y.iter().map(|v| (v - self.y_mean) / self.y_std).collect();

        let mut k = self.kernel_matrix(x, x);
        let noise_var = self.config.noise_std * self.config.noise_std;
        for i in 0..n {
            k[i][i] += noise_var;
        }

        self.k_inv = mat_inv_sym(&k)?;
        self.alpha = mat_vec_mul(&self.k_inv, &y_norm);
        self.fitted = true;
        Ok(())
    }

    fn predict_single(&self, x: &[f64]) -> Result<(f64, f64)> {
        if !self.fitted {
            return Err(MfiError::NotFitted {
                model: "LevelGp".to_string(),
            });
        }
        let k_star: Vec<f64> = self
            .x_train
            .iter()
            .map(|xt| self.rbf_kernel(x, xt))
            .collect();

        let mean_norm: f64 = dot(&k_star, &self.alpha);
        let mean = mean_norm * self.y_std + self.y_mean;

        let k_inv_k_star = mat_vec_mul(&self.k_inv, &k_star);
        let var_reduction = dot(&k_star, &k_inv_k_star);
        let k_xx = self.rbf_kernel(x, x);
        let var = (k_xx - var_reduction).max(0.0) * self.y_std * self.y_std;

        Ok((mean, var.sqrt()))
    }
}

/// Multi-fidelity Gaussian Process following the nonlinear auto-regressive
/// scheme of Perdikaris et al. (2017).
///
/// Each level trains a separate GP on a composite input `[x, y_{l-1}(x)]`.
#[derive(Debug, Clone)]
pub struct MfiGaussianProcess {
    /// Number of fidelity levels.
    pub n_levels: usize,
    /// GP configuration (same for all levels).
    pub config: MfiGpConfig,
    /// One GP per level.
    gps: Vec<LevelGp>,
    /// Whether fit has been called.
    pub fitted: bool,
}

impl MfiGaussianProcess {
    /// Construct a new unfitted model.
    pub fn new(n_levels: usize, config: MfiGpConfig) -> Result<Self> {
        if n_levels < 1 {
            return Err(MfiError::InvalidParameter {
                name: "n_levels".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        let gps = (0..n_levels)
            .map(|_| LevelGp::new(config.clone()))
            .collect();
        Ok(Self {
            n_levels,
            config,
            gps,
            fitted: false,
        })
    }

    /// Fit the multi-fidelity GP.
    ///
    /// `data_per_level[l]` holds the (x, y) pairs at level l.  Level 0 is
    /// fitted directly; level l > 0 gets the composite input [x, ŷ_{l-1}(x)].
    pub fn fit(&mut self, dataset: &MfiDataset) -> Result<()> {
        if dataset.data.len() != self.n_levels {
            return Err(MfiError::InvalidInput {
                context: format!(
                    "dataset has {} levels but model expects {}",
                    dataset.data.len(),
                    self.n_levels
                ),
            });
        }

        for l in 0..self.n_levels {
            let d = &dataset.data[l];
            if d.is_empty() {
                continue;
            }
            if l == 0 {
                self.gps[0].fit(&d.x, &d.y)?;
            } else {
                // Composite inputs: [x, y_{l-1}(x)]
                let composite: Vec<Vec<f64>> =
                    d.x.iter()
                        .map(|xi| {
                            let prev_pred = self.gps[l - 1]
                                .predict_single(xi)
                                .map(|(m, _)| m)
                                .unwrap_or(0.0);
                            let mut comp = xi.clone();
                            comp.push(prev_pred);
                            comp
                        })
                        .collect();
                self.gps[l].fit(&composite, &d.y)?;
            }
        }

        self.fitted = true;
        Ok(())
    }

    /// Predict mean and standard deviation at the highest fidelity level.
    pub fn predict_with_uncertainty(&self, x: &[f64]) -> Result<(f64, f64)> {
        if !self.fitted {
            return Err(MfiError::NotFitted {
                model: "MfiGaussianProcess".to_string(),
            });
        }
        let mut mean = 0.0_f64;
        let mut std = 0.0_f64;

        for l in 0..self.n_levels {
            if !self.gps[l].fitted {
                continue;
            }
            let inp: Vec<f64> = if l == 0 {
                x.to_vec()
            } else {
                let mut v = x.to_vec();
                v.push(mean);
                v
            };
            let (m, s) = self.gps[l].predict_single(&inp)?;
            mean = m;
            std = s;
        }

        Ok((mean, std))
    }

    /// Predict a batch of points.
    pub fn predict_batch(&self, xs: &[Vec<f64>]) -> Result<Vec<(f64, f64)>> {
        xs.iter()
            .map(|x| self.predict_with_uncertainty(x))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. MfiNeuralNetworkAR — Neural multi-fidelity (Meng & Karniadakis 2020)
// ─────────────────────────────────────────────────────────────────────────────

/// A simple MLP with one hidden layer.
#[derive(Debug, Clone)]
struct MfiMlp {
    w1: Vec<Vec<f64>>, // [hidden x input]
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>, // [output x hidden]
    b2: Vec<f64>,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
}

impl MfiMlp {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, rng: &mut StdRng) -> Self {
        let scale1 = (2.0 / input_dim as f64).sqrt();
        let scale2 = (2.0 / hidden_dim as f64).sqrt();

        let w1: Vec<Vec<f64>> = (0..hidden_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| rng.random::<f64>() * 2.0 * scale1 - scale1)
                    .collect()
            })
            .collect();
        let b1 = vec![0.0; hidden_dim];
        let w2: Vec<Vec<f64>> = (0..output_dim)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| rng.random::<f64>() * 2.0 * scale2 - scale2)
                    .collect()
            })
            .collect();
        let b2 = vec![0.0; output_dim];

        Self {
            w1,
            b1,
            w2,
            b2,
            input_dim,
            hidden_dim,
            output_dim,
        }
    }

    fn forward(&self, x: &[f64]) -> Vec<f64> {
        // Hidden layer + ReLU
        let h: Vec<f64> = self
            .w1
            .iter()
            .zip(self.b1.iter())
            .map(|(row, &b)| (dot(row, x) + b).max(0.0))
            .collect();
        // Output layer (linear)
        self.w2
            .iter()
            .zip(self.b2.iter())
            .map(|(row, &b)| dot(row, &h) + b)
            .collect()
    }

    /// Single SGD update step.  Returns the squared loss.
    fn train_step(&mut self, x: &[f64], target: f64, lr: f64) -> f64 {
        // Forward
        let h_pre: Vec<f64> = self
            .w1
            .iter()
            .zip(self.b1.iter())
            .map(|(row, &b)| dot(row, x) + b)
            .collect();
        let h: Vec<f64> = h_pre.iter().map(|&v| v.max(0.0)).collect();
        let out: Vec<f64> = self
            .w2
            .iter()
            .zip(self.b2.iter())
            .map(|(row, &b)| dot(row, &h) + b)
            .collect();

        let pred = out[0];
        let loss = (pred - target).powi(2);
        let d_out = 2.0 * (pred - target); // gradient of MSE

        // Backprop through w2, b2
        for (j, row) in self.w2.iter_mut().enumerate() {
            let d = d_out * (if j == 0 { 1.0 } else { 0.0 });
            for (k, w) in row.iter_mut().enumerate() {
                *w -= lr * d * h[k];
            }
        }
        self.b2[0] -= lr * d_out;

        // Backprop through w1, b1
        let d_h: Vec<f64> = (0..self.hidden_dim)
            .map(|k| d_out * self.w2[0][k] * if h_pre[k] > 0.0 { 1.0 } else { 0.0 })
            .collect();

        for (i, row) in self.w1.iter_mut().enumerate() {
            for (j, w) in row.iter_mut().enumerate() {
                *w -= lr * d_h[i] * x[j];
            }
            self.b1[i] -= lr * d_h[i];
        }

        loss
    }
}

/// Neural network multi-fidelity AR model (Meng & Karniadakis 2020).
///
/// * `low_net`: direct MLP for the low-fidelity output.
/// * `high_net`: MLP receiving `[y_low(x), x]` and predicting y_high.
#[derive(Debug, Clone)]
pub struct MfiNeuralNetworkAR {
    /// Input dimensionality.
    pub input_dim: usize,
    /// Hidden layer width.
    pub hidden_dim: usize,
    /// Random seed.
    pub seed: u64,
    /// Low-fidelity MLP.
    low_net: Option<MfiMlp>,
    /// High-fidelity MLP (input: [y_low, x]).
    high_net: Option<MfiMlp>,
    /// Whether both networks are trained.
    pub fitted: bool,
}

impl MfiNeuralNetworkAR {
    /// Create an unfitted model.
    pub fn new(input_dim: usize, hidden_dim: usize, seed: u64) -> Result<Self> {
        if input_dim == 0 {
            return Err(MfiError::InvalidParameter {
                name: "input_dim".to_string(),
                reason: "must be > 0".to_string(),
            });
        }
        Ok(Self {
            input_dim,
            hidden_dim,
            seed,
            low_net: None,
            high_net: None,
            fitted: false,
        })
    }

    /// Train both networks.
    ///
    /// `low_data`: (x, y) pairs at low fidelity.
    /// `high_data`: (x, y) pairs at high fidelity.
    pub fn train(
        &mut self,
        low_x: &[Vec<f64>],
        low_y: &[f64],
        high_x: &[Vec<f64>],
        high_y: &[f64],
        epochs: usize,
        lr: f64,
    ) -> Result<()> {
        if low_x.len() != low_y.len() || high_x.len() != high_y.len() {
            return Err(MfiError::InvalidInput {
                context: "x and y lengths must match".to_string(),
            });
        }
        if low_x.is_empty() || high_x.is_empty() {
            return Err(MfiError::InvalidInput {
                context: "training data must not be empty".to_string(),
            });
        }

        let mut rng = StdRng::seed_from_u64(self.seed);

        // Train low-fidelity net
        let mut low_net = MfiMlp::new(self.input_dim, self.hidden_dim, 1, &mut rng);
        for _ in 0..epochs {
            for (x, &y) in low_x.iter().zip(low_y.iter()) {
                low_net.train_step(x, y, lr);
            }
        }

        // Train high-fidelity net with composite input [y_low(x), x]
        let high_input_dim = 1 + self.input_dim;
        let mut high_net = MfiMlp::new(high_input_dim, self.hidden_dim, 1, &mut rng);

        for _ in 0..epochs {
            for (x, &y_h) in high_x.iter().zip(high_y.iter()) {
                let y_low_pred = low_net.forward(x)[0];
                let mut comp = vec![y_low_pred];
                comp.extend_from_slice(x);
                high_net.train_step(&comp, y_h, lr);
            }
        }

        self.low_net = Some(low_net);
        self.high_net = Some(high_net);
        self.fitted = true;
        Ok(())
    }

    /// Predict high-fidelity output at `x`.
    pub fn predict(&self, x: &[f64]) -> Result<f64> {
        let low_net = self.low_net.as_ref().ok_or_else(|| MfiError::NotFitted {
            model: "MfiNeuralNetworkAR (low_net)".to_string(),
        })?;
        let high_net = self.high_net.as_ref().ok_or_else(|| MfiError::NotFitted {
            model: "MfiNeuralNetworkAR (high_net)".to_string(),
        })?;

        let y_low = low_net.forward(x)[0];
        let mut comp = vec![y_low];
        comp.extend_from_slice(x);
        Ok(high_net.forward(&comp)[0])
    }

    /// Predict a batch of points.
    pub fn predict_batch(&self, xs: &[Vec<f64>]) -> Result<Vec<f64>> {
        xs.iter().map(|x| self.predict(x)).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. MfiActiveLearner — active learning with uncertainty-to-cost ratio
// ─────────────────────────────────────────────────────────────────────────────

/// Active learning strategy for multi-fidelity data collection.
///
/// At each step the learner selects the candidate point and fidelity level
/// that maximise `uncertainty / cost`.
#[derive(Debug, Clone)]
pub struct MfiActiveLearner {
    /// Search space bounds: [(lo, hi); dim].
    pub search_bounds: Vec<(f64, f64)>,
    /// Number of candidates to sample at each suggestion step.
    pub n_candidates: usize,
    /// Minimum number of observations needed before active querying starts.
    pub warm_up: usize,
    /// Random seed.
    pub seed: u64,
    /// Internal counter for reproducibility across calls.
    call_count: u64,
}

impl MfiActiveLearner {
    /// Create an active learner over the given search space.
    pub fn new(
        search_bounds: Vec<(f64, f64)>,
        n_candidates: usize,
        warm_up: usize,
        seed: u64,
    ) -> Result<Self> {
        if search_bounds.is_empty() {
            return Err(MfiError::InvalidInput {
                context: "search_bounds must not be empty".to_string(),
            });
        }
        for &(lo, hi) in &search_bounds {
            if lo >= hi {
                return Err(MfiError::InvalidParameter {
                    name: "search_bounds".to_string(),
                    reason: format!("lower bound {lo} must be < upper bound {hi}"),
                });
            }
        }
        Ok(Self {
            search_bounds,
            n_candidates,
            warm_up,
            seed,
            call_count: 0,
        })
    }

    /// Suggest the next (x, level_idx) to evaluate given the fitted GP.
    ///
    /// Returns `(x_new, level_idx)` that maximises `uncertainty / cost`.
    pub fn suggest_next(
        &mut self,
        gp: &MfiGaussianProcess,
        levels: &[MfiFidelityLevel],
        total_obs: usize,
    ) -> Result<(Vec<f64>, usize)> {
        if levels.is_empty() {
            return Err(MfiError::InvalidInput {
                context: "levels must not be empty".to_string(),
            });
        }

        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(self.call_count));
        self.call_count += 1;

        let dim = self.search_bounds.len();
        let candidates: Vec<Vec<f64>> = (0..self.n_candidates)
            .map(|_| {
                self.search_bounds
                    .iter()
                    .map(|&(lo, hi)| lo + rng.random::<f64>() * (hi - lo))
                    .collect()
            })
            .collect();

        // During warm-up, return a random candidate at level 0
        if total_obs < self.warm_up || !gp.fitted {
            let idx = (rng.random::<f64>() * candidates.len() as f64) as usize;
            let idx = idx.min(candidates.len() - 1);
            return Ok((candidates[idx].clone(), 0));
        }

        let _ = dim; // used above

        // Score each (candidate, level) pair by uncertainty / cost
        let mut best_score = f64::NEG_INFINITY;
        let mut best_x = candidates[0].clone();
        let mut best_level = 0_usize;

        for x in &candidates {
            let (_mean, std) = gp.predict_with_uncertainty(x).unwrap_or((0.0, 0.0));
            for (l_idx, level) in levels.iter().enumerate() {
                let score = std / level.relative_cost;
                if score > best_score {
                    best_score = score;
                    best_x = x.clone();
                    best_level = l_idx;
                }
            }
        }

        Ok((best_x, best_level))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. MfiSuccessiveHalving — SHA bracket for HPO
// ─────────────────────────────────────────────────────────────────────────────

/// A hyperparameter configuration with its current performance score.
#[derive(Debug, Clone)]
pub struct MfiConfig {
    /// Hyperparameter values (continuous representation).
    pub params: Vec<f64>,
    /// Best observed score at the current budget (higher is better).
    pub score: f64,
    /// Total budget consumed so far.
    pub budget_used: f64,
}

impl MfiConfig {
    /// Create a new configuration with an initial score of negative infinity.
    pub fn new(params: Vec<f64>) -> Self {
        Self {
            params,
            score: f64::NEG_INFINITY,
            budget_used: 0.0,
        }
    }
}

/// Successive halving bracket.
///
/// Starts `n` configs at `min_budget`, promotes the top `1/eta` fraction
/// to `eta * budget`, and repeats until `max_budget`.
#[derive(Debug, Clone)]
pub struct MfiSuccessiveHalving {
    /// Halving factor η (e.g. 3 means top 1/3 are promoted).
    pub eta: f64,
    /// Minimum per-config budget in this bracket.
    pub min_budget: f64,
    /// Maximum per-config budget.
    pub max_budget: f64,
}

impl MfiSuccessiveHalving {
    /// Create a new bracket.
    pub fn new(eta: f64, min_budget: f64, max_budget: f64) -> Result<Self> {
        if eta <= 1.0 {
            return Err(MfiError::InvalidParameter {
                name: "eta".to_string(),
                reason: "must be > 1.0".to_string(),
            });
        }
        if min_budget <= 0.0 || max_budget <= 0.0 || min_budget > max_budget {
            return Err(MfiError::InvalidParameter {
                name: "budget".to_string(),
                reason: "min_budget must be in (0, max_budget]".to_string(),
            });
        }
        Ok(Self {
            eta,
            min_budget,
            max_budget,
        })
    }

    /// Number of SHA rounds in this bracket.
    pub fn n_rounds(&self) -> usize {
        ((self.max_budget / self.min_budget).ln() / self.eta.ln()).ceil() as usize
    }

    /// Run the SHA bracket, evaluating configs with the provided oracle.
    ///
    /// `oracle(config, budget)` returns a performance score (higher is better).
    pub fn run<F>(&self, mut configs: Vec<MfiConfig>, oracle: F) -> Result<Vec<MfiConfig>>
    where
        F: Fn(&[f64], f64) -> f64,
    {
        if configs.is_empty() {
            return Err(MfiError::InvalidInput {
                context: "MfiSuccessiveHalving::run requires at least one config".to_string(),
            });
        }

        let mut budget = self.min_budget;
        let mut survivors = configs.len();

        while budget <= self.max_budget && survivors > 0 {
            // Evaluate all remaining configs at current budget
            for cfg in configs.iter_mut().take(survivors) {
                cfg.score = oracle(&cfg.params, budget);
                cfg.budget_used = budget;
            }

            // Sort descending by score (best first)
            configs[..survivors].sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Promote top 1/eta
            survivors = ((survivors as f64) / self.eta).floor() as usize;
            budget *= self.eta;
        }

        Ok(configs)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. MfiHyperband — Hyperband (Li 2017)
// ─────────────────────────────────────────────────────────────────────────────

/// Result from one Hyperband bracket.
#[derive(Debug, Clone)]
pub struct MfiHyperbandResult {
    /// Index of the bracket (0 = maximum number of configs).
    pub bracket: usize,
    /// Best configuration found in this bracket.
    pub best_config: MfiConfig,
}

/// Hyperband: runs multiple SHA brackets in parallel, each with a different
/// trade-off between number of initial configs and minimum budget.
#[derive(Debug, Clone)]
pub struct MfiHyperband {
    /// Maximum per-config budget (B).
    pub max_budget: f64,
    /// Halving factor η.
    pub eta: f64,
    /// Random seed for config generation.
    pub seed: u64,
    /// Dimension of the config parameter space.
    pub param_dim: usize,
}

impl MfiHyperband {
    /// Create a new Hyperband optimizer.
    pub fn new(max_budget: f64, eta: f64, seed: u64, param_dim: usize) -> Result<Self> {
        if eta <= 1.0 {
            return Err(MfiError::InvalidParameter {
                name: "eta".to_string(),
                reason: "must be > 1.0".to_string(),
            });
        }
        if max_budget <= 0.0 {
            return Err(MfiError::InvalidParameter {
                name: "max_budget".to_string(),
                reason: "must be > 0".to_string(),
            });
        }
        Ok(Self {
            max_budget,
            eta,
            seed,
            param_dim,
        })
    }

    /// Number of brackets s_max + 1.
    fn n_brackets(&self) -> usize {
        ((self.max_budget.ln() / self.eta.ln()).floor() as usize) + 1
    }

    /// Run all Hyperband brackets and return results sorted best-first.
    pub fn run<F>(&self, oracle: F, search_bounds: &[(f64, f64)]) -> Result<Vec<MfiHyperbandResult>>
    where
        F: Fn(&[f64], f64) -> f64,
    {
        if search_bounds.len() != self.param_dim {
            return Err(MfiError::InvalidInput {
                context: format!(
                    "search_bounds dim {} != param_dim {}",
                    search_bounds.len(),
                    self.param_dim
                ),
            });
        }
        let s_max = self.n_brackets().saturating_sub(1);
        let mut results = Vec::new();
        let mut rng = StdRng::seed_from_u64(self.seed);

        for s in (0..=s_max).rev() {
            // n_s configs, r_s minimum budget
            let n_s =
                ((s_max + 1) as f64 / self.eta.powi(s as i32) / (s as f64 + 1.0)).ceil() as usize;
            let r_s = self.max_budget / self.eta.powi(s as i32);
            let r_s = r_s.max(1.0);

            // Sample n_s random configurations
            let configs: Vec<MfiConfig> = (0..n_s.max(1))
                .map(|_| {
                    let params: Vec<f64> = search_bounds
                        .iter()
                        .map(|&(lo, hi)| lo + rng.random::<f64>() * (hi - lo))
                        .collect();
                    MfiConfig::new(params)
                })
                .collect();

            let sha = MfiSuccessiveHalving::new(self.eta, r_s, self.max_budget)?;
            let mut evaluated = sha.run(configs, &oracle)?;

            // Best config is first after sorting in SHA
            if !evaluated.is_empty() {
                let best = evaluated.remove(0);
                results.push(MfiHyperbandResult {
                    bracket: s,
                    best_config: best,
                });
            }
        }

        // Sort all results by best score descending
        results.sort_by(|a, b| {
            b.best_config
                .score
                .partial_cmp(&a.best_config.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. MfiBOHB — BOHB with multi-fidelity (Falkner 2018)
// ─────────────────────────────────────────────────────────────────────────────

/// KDE-based density model used by BOHB to model good vs bad configurations.
#[derive(Debug, Clone)]
struct MfiKde {
    /// Bandwidth.
    bw: f64,
    /// Observed configuration vectors.
    observations: Vec<Vec<f64>>,
}

impl MfiKde {
    fn new(bw: f64) -> Self {
        Self {
            bw,
            observations: Vec::new(),
        }
    }

    fn add(&mut self, x: Vec<f64>) {
        self.observations.push(x);
    }

    /// Evaluate the KDE log-density at `x`.
    fn log_density(&self, x: &[f64]) -> f64 {
        if self.observations.is_empty() {
            return 0.0;
        }
        let n = self.observations.len() as f64;
        let log_n = n.ln();
        let sum: f64 = self
            .observations
            .iter()
            .map(|xi| {
                let d2 = l2_sq(x, xi);
                -d2 / (2.0 * self.bw * self.bw)
            })
            .map(|lk| lk.exp())
            .sum();
        if sum <= 0.0 {
            return f64::NEG_INFINITY;
        }
        sum.ln() - log_n
    }
}

/// Configuration with fitness history across budgets.
#[derive(Debug, Clone)]
pub struct MfiBohbObservation {
    /// Hyperparameter vector.
    pub params: Vec<f64>,
    /// Budget at which it was evaluated.
    pub budget: f64,
    /// Observed loss (lower is better for BOHB).
    pub loss: f64,
}

/// BOHB: combines KDE-based Bayesian Optimization with Hyperband scheduling
/// (Falkner, Klein & Hutter 2018).
///
/// Maintains separate `l(x)` (good) and `g(x)` (bad) KDE models;
/// suggestions from `l(x)` are preferred.
#[derive(Debug, Clone)]
pub struct MfiBOHB {
    /// Maximum budget.
    pub max_budget: f64,
    /// Halving factor η.
    pub eta: f64,
    /// Fraction of observations used for the good model.
    pub gamma: f64,
    /// KDE bandwidth.
    pub bandwidth: f64,
    /// Random seed.
    pub seed: u64,
    /// Parameter search bounds.
    pub search_bounds: Vec<(f64, f64)>,
    /// All observations so far.
    observations: Vec<MfiBohbObservation>,
    /// Call counter for reproducibility.
    call_count: u64,
}

impl MfiBOHB {
    /// Create a new BOHB optimizer.
    pub fn new(
        max_budget: f64,
        eta: f64,
        gamma: f64,
        bandwidth: f64,
        seed: u64,
        search_bounds: Vec<(f64, f64)>,
    ) -> Result<Self> {
        if !(0.0..1.0).contains(&gamma) {
            return Err(MfiError::InvalidParameter {
                name: "gamma".to_string(),
                reason: "must be in [0, 1)".to_string(),
            });
        }
        if bandwidth <= 0.0 {
            return Err(MfiError::InvalidParameter {
                name: "bandwidth".to_string(),
                reason: "must be > 0".to_string(),
            });
        }
        Ok(Self {
            max_budget,
            eta,
            gamma,
            bandwidth,
            seed,
            search_bounds,
            observations: Vec::new(),
            call_count: 0,
        })
    }

    /// Register an evaluation result.
    pub fn observe(&mut self, params: Vec<f64>, budget: f64, loss: f64) {
        self.observations.push(MfiBohbObservation {
            params,
            budget,
            loss,
        });
    }

    /// Suggest a new (config, budget) pair.
    ///
    /// When enough observations exist, samples from the good KDE `l(x)` and
    /// selects the candidate maximising `log l(x) - log g(x)`.
    pub fn suggest(&mut self, current_budget: f64) -> Result<(Vec<f64>, f64)> {
        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(self.call_count));
        self.call_count += 1;

        let budget = current_budget.min(self.max_budget).max(1.0);

        // Collect observations at this budget level
        let mut obs_at_budget: Vec<&MfiBohbObservation> = self
            .observations
            .iter()
            .filter(|o| (o.budget - budget).abs() < budget * 0.5 + 1.0)
            .collect();

        // If fewer than 2 * (1/gamma) observations, fall back to random
        let n_good_min = (2.0 / (1.0 - self.gamma)).ceil() as usize;
        if obs_at_budget.len() < n_good_min {
            let params: Vec<f64> = self
                .search_bounds
                .iter()
                .map(|&(lo, hi)| lo + rng.random::<f64>() * (hi - lo))
                .collect();
            return Ok((params, budget));
        }

        // Sort by loss ascending (lower loss = better)
        obs_at_budget.sort_by(|a, b| {
            a.loss
                .partial_cmp(&b.loss)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n_good = ((obs_at_budget.len() as f64 * self.gamma).ceil() as usize).max(1);

        // Build l(x) and g(x) KDEs
        let mut l_kde = MfiKde::new(self.bandwidth);
        let mut g_kde = MfiKde::new(self.bandwidth);

        for (i, obs) in obs_at_budget.iter().enumerate() {
            if i < n_good {
                l_kde.add(obs.params.clone());
            } else {
                g_kde.add(obs.params.clone());
            }
        }

        // Sample n_candidates from l(x) and pick best log(l/g)
        let n_cand = 24;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_params: Vec<f64> = self
            .search_bounds
            .iter()
            .map(|&(lo, hi)| lo + rng.random::<f64>() * (hi - lo))
            .collect();

        for _ in 0..n_cand {
            // Sample from l(x) observations with noise
            let src_idx = (rng.random::<f64>() * n_good as f64) as usize;
            let src_idx = src_idx.min(n_good - 1);
            let src = &obs_at_budget[src_idx].params;
            let cand: Vec<f64> = src
                .iter()
                .zip(self.search_bounds.iter())
                .map(|(&v, &(lo, hi))| {
                    let noise: f64 = rng.random::<f64>() * 2.0 - 1.0;
                    (v + noise * self.bandwidth).clamp(lo, hi)
                })
                .collect();

            let log_l = l_kde.log_density(&cand);
            let log_g = g_kde.log_density(&cand);
            let score = log_l - log_g;

            if score > best_score {
                best_score = score;
                best_params = cand;
            }
        }

        Ok((best_params, budget))
    }

    /// Return the best observation seen so far (lowest loss).
    pub fn best_observation(&self) -> Option<&MfiBohbObservation> {
        self.observations.iter().min_by(|a, b| {
            a.loss
                .partial_cmp(&b.loss)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. MfiEnsemble — Multi-fidelity ensemble prediction
// ─────────────────────────────────────────────────────────────────────────────

/// Weighting strategy for ensemble members.
#[derive(Debug, Clone)]
pub enum MfiEnsembleWeightStrategy {
    /// Uniform: all models weighted equally.
    Uniform,
    /// Weight proportional to fidelity accuracy estimate.
    AccuracyProportional,
    /// Stacking: train a meta-linear model on held-out predictions.
    Stacking,
}

/// Multi-fidelity ensemble that combines predictions from an AR model and a GP.
#[derive(Debug, Clone)]
pub struct MfiEnsemble {
    /// Number of fidelity levels.
    pub n_levels: usize,
    /// Weighting strategy.
    pub strategy: MfiEnsembleWeightStrategy,
    /// Per-level weights (set after fit).
    weights: Vec<f64>,
    /// Stacking meta-model coefficients (for Stacking strategy).
    meta_coef: Vec<f64>,
    /// Whether the ensemble is fitted.
    pub fitted: bool,
}

impl MfiEnsemble {
    /// Create a new ensemble.
    pub fn new(n_levels: usize, strategy: MfiEnsembleWeightStrategy) -> Result<Self> {
        if n_levels == 0 {
            return Err(MfiError::InvalidParameter {
                name: "n_levels".to_string(),
                reason: "must be > 0".to_string(),
            });
        }
        Ok(Self {
            n_levels,
            strategy,
            weights: vec![1.0 / n_levels as f64; n_levels],
            meta_coef: vec![0.0; n_levels + 1],
            fitted: false,
        })
    }

    /// Fit the ensemble weights from the dataset and level accuracy estimates.
    ///
    /// For Stacking, uses OLS on validation predictions supplied as
    /// `val_preds[sample][level]` and `val_targets`.
    pub fn fit(
        &mut self,
        levels: &[MfiFidelityLevel],
        val_preds: &[Vec<f64>],
        val_targets: &[f64],
    ) -> Result<()> {
        if levels.len() != self.n_levels {
            return Err(MfiError::InvalidInput {
                context: "levels count must match n_levels".to_string(),
            });
        }
        match &self.strategy {
            MfiEnsembleWeightStrategy::Uniform => {
                self.weights = vec![1.0 / self.n_levels as f64; self.n_levels];
            }
            MfiEnsembleWeightStrategy::AccuracyProportional => {
                let total: f64 = levels.iter().map(|l| l.accuracy_estimate).sum::<f64>();
                if total < 1e-12 {
                    self.weights = vec![1.0 / self.n_levels as f64; self.n_levels];
                } else {
                    self.weights = levels.iter().map(|l| l.accuracy_estimate / total).collect();
                }
            }
            MfiEnsembleWeightStrategy::Stacking => {
                if val_preds.len() != val_targets.len() || val_preds.is_empty() {
                    return Err(MfiError::InvalidInput {
                        context: "val_preds and val_targets must be non-empty and match length"
                            .to_string(),
                    });
                }
                // Fit ridge regression: minimise ||X β - y||² + λ||β||²
                // X is [n_samples x (n_levels+1)] with a bias column
                let n = val_targets.len();
                let p = self.n_levels + 1;
                let lambda = 1e-4;

                // Build X
                let mut xt_x = vec![vec![0.0_f64; p]; p];
                let mut xt_y = vec![0.0_f64; p];

                for (i, preds) in val_preds.iter().enumerate() {
                    let mut row = preds[..self.n_levels.min(preds.len())].to_vec();
                    while row.len() < self.n_levels {
                        row.push(0.0);
                    }
                    row.push(1.0); // bias

                    let y = val_targets[i];
                    for j in 0..p {
                        xt_y[j] += row[j] * y;
                        for k in 0..p {
                            xt_x[j][k] += row[j] * row[k];
                        }
                    }
                }

                // Ridge: (XᵀX + λI)β = Xᵀy
                for i in 0..p {
                    xt_x[i][i] += lambda * n as f64;
                }

                let inv = mat_inv_sym(&xt_x)?;
                self.meta_coef = mat_vec_mul(&inv, &xt_y);

                // Derive weights from meta_coef (ignoring bias)
                let coef_sum: f64 = self.meta_coef[..self.n_levels]
                    .iter()
                    .map(|c| c.abs())
                    .sum();
                if coef_sum < 1e-12 {
                    self.weights = vec![1.0 / self.n_levels as f64; self.n_levels];
                } else {
                    self.weights = self.meta_coef[..self.n_levels]
                        .iter()
                        .map(|c| c.abs() / coef_sum)
                        .collect();
                }
            }
        }
        self.fitted = true;
        Ok(())
    }

    /// Combine per-level predictions into a single ensemble prediction.
    ///
    /// `level_preds` must have exactly `n_levels` entries.
    pub fn ensemble_predict(&self, level_preds: &[f64]) -> Result<f64> {
        if !self.fitted {
            return Err(MfiError::NotFitted {
                model: "MfiEnsemble".to_string(),
            });
        }
        if level_preds.len() != self.n_levels {
            return Err(MfiError::InvalidInput {
                context: format!(
                    "expected {} level predictions, got {}",
                    self.n_levels,
                    level_preds.len()
                ),
            });
        }

        match &self.strategy {
            MfiEnsembleWeightStrategy::Stacking => {
                // Full meta-model: dot(level_preds, coef[..n_levels]) + bias
                let pred: f64 = level_preds
                    .iter()
                    .zip(self.meta_coef.iter())
                    .map(|(p, c)| p * c)
                    .sum::<f64>()
                    + *self.meta_coef.last().unwrap_or(&0.0);
                Ok(pred)
            }
            _ => {
                let pred: f64 = level_preds
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(p, w)| p * w)
                    .sum();
                Ok(pred)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. MfiMetrics & MfiReport — evaluation utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluation metrics for a multi-fidelity system.
#[derive(Debug, Clone, Default)]
pub struct MfiMetrics {
    /// Root Mean Squared Error of the surrogate at the test fidelity.
    pub rmse: f64,
    /// Mean Absolute Error.
    pub mae: f64,
    /// Fraction of total cost saved vs evaluating everything at the highest fidelity.
    pub cost_savings_fraction: f64,
    /// Pearson correlation between fidelity levels (pairs of adjacent levels).
    pub level_correlations: Vec<f64>,
    /// Efficiency: accuracy per unit cost for each level.
    pub level_efficiencies: Vec<f64>,
}

impl MfiMetrics {
    /// Compute prediction accuracy metrics for predictions vs ground truth.
    pub fn compute_prediction_metrics(predictions: &[f64], targets: &[f64]) -> Result<Self> {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return Err(MfiError::InvalidInput {
                context: "predictions and targets must be non-empty with equal length".to_string(),
            });
        }
        let n = predictions.len() as f64;
        let mse: f64 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / n;
        let mae: f64 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>()
            / n;

        Ok(Self {
            rmse: mse.sqrt(),
            mae,
            ..Default::default()
        })
    }

    /// Compute inter-level correlation and efficiency metrics.
    pub fn compute_fidelity_metrics(dataset: &MfiDataset) -> Self {
        let n_levels = dataset.levels.len();
        let mut corrs = Vec::new();
        for l in 0..n_levels.saturating_sub(1) {
            let n_l = dataset.data[l].len().min(dataset.data[l + 1].len());
            if n_l >= 2 {
                let c =
                    pearson_correlation(&dataset.data[l].y[..n_l], &dataset.data[l + 1].y[..n_l]);
                corrs.push(c);
            } else {
                corrs.push(0.0);
            }
        }

        let efficiencies = dataset.levels.iter().map(|lv| lv.efficiency()).collect();

        Self {
            level_correlations: corrs,
            level_efficiencies: efficiencies,
            ..Default::default()
        }
    }

    /// Compute the cost savings fraction.
    ///
    /// Compares actual cost (weighted by level usage) against using only the
    /// highest fidelity for every observation.
    pub fn compute_cost_savings(dataset: &MfiDataset) -> f64 {
        let n_levels = dataset.levels.len();
        let total_obs: usize = dataset.data.iter().map(|d| d.len()).sum();
        if total_obs == 0 || n_levels == 0 {
            return 0.0;
        }
        let max_cost = dataset
            .levels
            .last()
            .map(|l| l.relative_cost)
            .unwrap_or(1.0);
        let actual_cost: f64 = dataset
            .data
            .iter()
            .zip(dataset.levels.iter())
            .map(|(d, lv)| d.len() as f64 * lv.relative_cost)
            .sum();
        let full_cost = total_obs as f64 * max_cost;
        if full_cost < 1e-12 {
            return 0.0;
        }
        1.0 - actual_cost / full_cost
    }
}

/// Human-readable report from a multi-fidelity run.
#[derive(Debug, Clone)]
pub struct MfiReport {
    /// Number of fidelity levels.
    pub n_levels: usize,
    /// Total observations at each level.
    pub obs_per_level: Vec<usize>,
    /// Prediction RMSE at the highest level.
    pub prediction_rmse: f64,
    /// MAE.
    pub prediction_mae: f64,
    /// Fraction of budget saved.
    pub cost_savings: f64,
    /// Adjacent-level correlations.
    pub level_correlations: Vec<f64>,
    /// Per-level efficiency.
    pub level_efficiencies: Vec<f64>,
}

impl MfiReport {
    /// Build a full report from a dataset and optional held-out evaluation.
    pub fn build(
        dataset: &MfiDataset,
        predictions: Option<&[f64]>,
        targets: Option<&[f64]>,
    ) -> Result<Self> {
        let n_levels = dataset.levels.len();
        let obs_per_level = dataset.data.iter().map(|d| d.len()).collect();
        let fid_metrics = MfiMetrics::compute_fidelity_metrics(dataset);
        let cost_savings = MfiMetrics::compute_cost_savings(dataset);

        let (rmse, mae) = match (predictions, targets) {
            (Some(p), Some(t)) => {
                let m = MfiMetrics::compute_prediction_metrics(p, t)?;
                (m.rmse, m.mae)
            }
            _ => (0.0, 0.0),
        };

        Ok(Self {
            n_levels,
            obs_per_level,
            prediction_rmse: rmse,
            prediction_mae: mae,
            cost_savings,
            level_correlations: fid_metrics.level_correlations,
            level_efficiencies: fid_metrics.level_efficiencies,
        })
    }
}

impl fmt::Display for MfiReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== MfiReport ===")?;
        writeln!(f, "  Fidelity levels : {}", self.n_levels)?;
        for (i, &n) in self.obs_per_level.iter().enumerate() {
            writeln!(f, "    Level {i}: {n} observations")?;
        }
        writeln!(f, "  Prediction RMSE : {:.6}", self.prediction_rmse)?;
        writeln!(f, "  Prediction MAE  : {:.6}", self.prediction_mae)?;
        writeln!(f, "  Cost savings    : {:.2}%", self.cost_savings * 100.0)?;
        writeln!(f, "  Level correlations: {:?}", self.level_correlations)?;
        writeln!(f, "  Level efficiencies: {:?}", self.level_efficiencies)?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal numerical helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Squared Euclidean distance.
fn l2_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Dot product.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix-vector product.
fn mat_vec_mul(mat: &[Vec<f64>], vec: &[f64]) -> Vec<f64> {
    mat.iter().map(|row| dot(row, vec)).collect()
}

/// Symmetric positive-definite matrix inversion via Cholesky decomposition.
fn mat_inv_sym(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = a.len();
    // Cholesky: A = L Lᵀ
    let l = cholesky(a)?;

    // Invert L by forward substitution
    let mut l_inv = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        l_inv[i][i] = 1.0 / l[i][i].max(1e-15);
        for j in 0..i {
            let s: f64 = (j..i).map(|k| l[i][k] * l_inv[k][j]).sum();
            l_inv[i][j] = -s / l[i][i];
        }
    }

    // A⁻¹ = L⁻ᵀ L⁻¹
    let mut inv = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0_f64;
            for k in i..n {
                s += l_inv[k][i] * l_inv[k][j];
            }
            inv[i][j] = s;
            inv[j][i] = s;
        }
    }
    Ok(inv)
}

/// Cholesky decomposition (lower triangular).
fn cholesky(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let d = a[i][i] - s;
                if d < 0.0 {
                    return Err(MfiError::NumericalError {
                        context: format!("Cholesky failed at ({i},{i}): diagonal value {d} < 0"),
                    });
                }
                l[i][i] = d.sqrt().max(1e-15);
            } else {
                l[i][j] = (a[i][j] - s) / l[j][j].max(1e-15);
            }
        }
    }
    Ok(l)
}

/// Pearson correlation coefficient.
fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let mean_a: f64 = a.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = b.iter().sum::<f64>() / n as f64;
    let num: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - mean_a) * (y - mean_b))
        .sum();
    let den_a: f64 = a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>().sqrt();
    let den_b: f64 = b.iter().map(|y| (y - mean_b).powi(2)).sum::<f64>().sqrt();
    let den = den_a * den_b;
    if den < 1e-12 {
        0.0
    } else {
        num / den
    }
}

/// Nearest-neighbour prediction (returns the y of the closest x in training data).
fn nearest_neighbour_predict(x_train: &[Vec<f64>], y_train: &[f64], x: &[f64]) -> Option<f64> {
    if x_train.is_empty() {
        return None;
    }
    let (_, y) = x_train
        .iter()
        .zip(y_train.iter())
        .min_by(|(a, _), (b, _)| {
            l2_sq(a, x)
                .partial_cmp(&l2_sq(b, x))
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
    Some(*y)
}
