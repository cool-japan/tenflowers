//! Influence Functions and Model Interpretability
//!
//! Comprehensive tools for understanding model behaviour, detecting bias, and
//! attributing predictions to training examples and input features.
//!
//! # Components
//!
//! - **[`IfInfluenceFunction`]** — TracIn-style training-example attribution with
//!   exact (Hessian-vector product via finite differences) and approximate modes.
//! - **[`IfPermutationImportance`]** — Feature importance via repeated
//!   Fisher-Yates shuffling and metric degradation.
//! - **[`IfAttentionRollout`]** — Multi-layer attention flow computation with
//!   optional gradient weighting.
//! - **[`IfFairnessAnalyzer`]** — Demographic parity, equalised odds,
//!   calibration gap, predictive parity, and disparate impact (4/5 rule).
//! - **[`IfBiasDetector`]** — Spurious-correlation detection via mutual
//!   information, partial correlation, and conditioned feature attribution.
//! - **[`IfModelDivergence`]** — Distribution-shift tracking: KL divergence,
//!   MMD, PSI, Kolmogorov-Smirnov, with configurable alert thresholds.
//! - **[`IfIntegratedGradients`]** — Riemann-sum integrated gradients
//!   along a straight-line path with convergence checking.
//! - **[`IfCounterfactualExplainer`]** — Gradient-based minimal-perturbation
//!   counterfactual search with L1/L2 regularisation.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use tenflowers_neural::influence_functions::{IfPermutationImportance, IfMetricKind};
//!
//! let imp = IfPermutationImportance::new(IfMetricKind::Accuracy, 10, 42);
//! let scores = imp.compute(
//!     |x| { /* model inference */ Ok(x.to_vec()) },
//!     &data,
//!     &labels,
//! )?;
//! ```

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Mathematical helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable sigmoid: `1 / (1 + exp(-x))`.
#[inline]
fn if_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// L2 norm of a slice.
#[inline]
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// L1 norm of a slice.
#[inline]
fn l1_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).sum()
}

/// Element-wise subtraction: `a - b`.
fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai - bi).collect()
}

/// Element-wise addition: `a + b`.
fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai + bi).collect()
}

/// Scalar multiplication: `alpha * v`.
fn vec_scale(v: &[f64], alpha: f64) -> Vec<f64> {
    v.iter().map(|x| x * alpha).collect()
}

/// Numerical gradient of a scalar-valued function `f` at point `x`
/// using central finite differences with step `h`.
fn numerical_gradient(f: &dyn Fn(&[f64]) -> Result<f64>, x: &[f64], h: f64) -> Result<Vec<f64>> {
    let d = x.len();
    let mut grad = vec![0.0; d];
    for i in 0..d {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += h;
        xm[i] -= h;
        let fp = f(&xp)?;
        let fm = f(&xm)?;
        grad[i] = (fp - fm) / (2.0 * h);
    }
    Ok(grad)
}

/// Hessian-vector product via finite differences:
/// `H(x) * v ≈ (∇f(x + h*v) - ∇f(x - h*v)) / (2h)`.
fn hvp_fd(
    grad_fn: &dyn Fn(&[f64]) -> Result<Vec<f64>>,
    x: &[f64],
    v: &[f64],
    h: f64,
) -> Result<Vec<f64>> {
    let xpv = vec_add(x, &vec_scale(v, h));
    let xmv = vec_sub(x, &vec_scale(v, h));
    let gp = grad_fn(&xpv)?;
    let gm = grad_fn(&xmv)?;
    Ok(vec_sub(&gp, &gm).iter().map(|gi| gi / (2.0 * h)).collect())
}

/// RBF (Gaussian) kernel: `exp(-||x - y||^2 / (2 * bandwidth^2))`.
#[inline]
fn rbf_kernel(x: &[f64], y: &[f64], bandwidth: f64) -> f64 {
    let sq_dist: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    (-sq_dist / (2.0 * bandwidth * bandwidth)).exp()
}

/// Mean of a slice.
#[inline]
fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Variance (population) of a slice.
#[inline]
fn variance(v: &[f64]) -> f64 {
    let m = mean(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len().max(1) as f64
}

/// Standard deviation (population) of a slice.
#[inline]
fn std_dev(v: &[f64]) -> f64 {
    variance(v).sqrt()
}

/// Pearson correlation coefficient between two slices.
fn pearson_corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let ma = mean(a);
    let mb = mean(b);
    let mut cov = 0.0_f64;
    let mut va = 0.0_f64;
    let mut vb = 0.0_f64;
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    let denom = (va * vb).sqrt();
    if denom < 1e-30 {
        return 0.0;
    }
    cov / denom
}

/// Softmax of a vector (numerically stable).
fn softmax(v: &[f64]) -> Vec<f64> {
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|x| (x - max_v).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < 1e-30 {
        return vec![1.0 / v.len() as f64; v.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

/// Clamp a value to `[lo, hi]`.
#[inline]
fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. IfInfluenceFunction — TracIn-style training-example attribution
// ─────────────────────────────────────────────────────────────────────────────

/// Mode for influence computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IfInfluenceMode {
    /// Approximate mode (TracIn): dot products of per-checkpoint gradients.
    TracIn,
    /// Exact mode: inverse-Hessian-vector product via finite differences
    /// (Koh & Liang, 2017).
    Exact,
}

/// A single gradient checkpoint snapshot — gradients of the loss at a particular
/// training step for a particular sample.
#[derive(Debug, Clone)]
pub struct IfGradientCheckpoint {
    /// Training step at which this checkpoint was taken.
    pub step: usize,
    /// Learning rate at this step (used as weight in TracIn).
    pub learning_rate: f64,
    /// Gradient vector of the loss w.r.t. model parameters at this step.
    pub gradient: Vec<f64>,
}

/// TracIn / exact influence-function computer.
///
/// Stores per-sample gradient checkpoints and computes pairwise influences
/// via either TracIn (sum of lr-weighted gradient dot products over checkpoints)
/// or exact inverse-Hessian-vector products via finite differences.
#[derive(Debug, Clone)]
pub struct IfInfluenceFunction {
    /// Mode of computation.
    mode: IfInfluenceMode,
    /// Per-sample checkpoints: `train_checkpoints[sample_idx]` is a vec of
    /// gradient snapshots across training steps.
    train_checkpoints: Vec<Vec<IfGradientCheckpoint>>,
    /// Finite-difference step for Hessian-vector products (exact mode).
    fd_step: f64,
    /// Damping factor for inverse Hessian approximation (exact mode).
    damping: f64,
    /// Number of CG iterations for inverse-Hessian solve (exact mode).
    cg_iters: usize,
}

impl IfInfluenceFunction {
    /// Create a new influence function computer.
    ///
    /// # Arguments
    /// * `mode` — TracIn or Exact.
    /// * `fd_step` — Finite-difference step for Hessian-vector products.
    /// * `damping` — Damping for inverse Hessian (regularisation).
    /// * `cg_iters` — Number of conjugate gradient iterations.
    pub fn new(mode: IfInfluenceMode, fd_step: f64, damping: f64, cg_iters: usize) -> Self {
        Self {
            mode,
            train_checkpoints: Vec::new(),
            fd_step: fd_step.max(1e-8),
            damping: damping.max(1e-10),
            cg_iters: cg_iters.max(1),
        }
    }

    /// Add gradient checkpoints for a single training sample.
    pub fn add_train_sample(&mut self, checkpoints: Vec<IfGradientCheckpoint>) {
        self.train_checkpoints.push(checkpoints);
    }

    /// Number of stored training samples.
    pub fn num_train_samples(&self) -> usize {
        self.train_checkpoints.len()
    }

    /// Compute the influence of training sample `train_idx` on a test sample
    /// whose gradient checkpoints are provided.
    ///
    /// **TracIn mode**: `I = Σ_t η_t * <g_train_t, g_test_t>`
    /// **Exact mode**: `I = -<∇L_test, H^{-1} ∇L_train>` using CG with
    /// Hessian-vector products approximated by finite differences.
    pub fn compute_influence(
        &self,
        train_idx: usize,
        test_checkpoints: &[IfGradientCheckpoint],
    ) -> Result<f64> {
        if train_idx >= self.train_checkpoints.len() {
            return Err(TensorError::invalid_argument_op(
                "IfInfluenceFunction::compute_influence",
                &format!(
                    "train_idx {} out of range (have {} samples)",
                    train_idx,
                    self.train_checkpoints.len()
                ),
            ));
        }
        if test_checkpoints.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "IfInfluenceFunction::compute_influence",
                "test_checkpoints must not be empty",
            ));
        }

        let train_cps = &self.train_checkpoints[train_idx];
        if train_cps.is_empty() {
            return Ok(0.0);
        }

        match self.mode {
            IfInfluenceMode::TracIn => self.compute_tracin(train_cps, test_checkpoints),
            IfInfluenceMode::Exact => self.compute_exact(train_cps, test_checkpoints),
        }
    }

    /// TracIn: `I = Σ_t η_t * <g_train_t, g_test_t>` summed over matching
    /// steps.  When step counts differ we zip them by index.
    fn compute_tracin(
        &self,
        train_cps: &[IfGradientCheckpoint],
        test_cps: &[IfGradientCheckpoint],
    ) -> Result<f64> {
        let n = train_cps.len().min(test_cps.len());
        let mut influence = 0.0_f64;
        for i in 0..n {
            let lr = train_cps[i].learning_rate;
            let d = dot(&train_cps[i].gradient, &test_cps[i].gradient);
            influence += lr * d;
        }
        Ok(influence)
    }

    /// Exact influence via damped inverse-Hessian approximation.
    ///
    /// We aggregate: `g_train = mean of train gradients`, `g_test = mean of
    /// test gradients`, then solve `(H + λI) s = g_test` with CG using
    /// Hessian-vector products estimated from the stored checkpoints.
    /// Influence = `-<g_train, s>`.
    fn compute_exact(
        &self,
        train_cps: &[IfGradientCheckpoint],
        test_cps: &[IfGradientCheckpoint],
    ) -> Result<f64> {
        let dim = train_cps[0].gradient.len();

        // Aggregate gradient
        let g_train = Self::mean_gradient(train_cps, dim);
        let g_test = Self::mean_gradient(test_cps, dim);

        // Approximate inverse-Hessian-vector product using CG.
        // We approximate `H * v` by averaging hvp estimates from all
        // checkpoint pairs.  Since we don't have the actual loss function
        // here, we use a Gauss-Newton-like diagonal approximation from
        // the stored gradients: `H ≈ (1/T) Σ_t g_t g_t^T + λI`.
        // So `H v = (1/T) Σ_t <g_t, v> g_t + λ v`.
        let all_grads: Vec<&[f64]> = train_cps.iter().map(|cp| cp.gradient.as_slice()).collect();

        let hv = |v: &[f64]| -> Vec<f64> {
            let t = all_grads.len() as f64;
            let mut result = vec_scale(v, self.damping);
            for g in &all_grads {
                let coeff = dot(g, v) / t.max(1.0);
                for j in 0..dim {
                    result[j] += coeff * g[j];
                }
            }
            result
        };

        // CG solve: (H + λI) s = g_test
        let mut s = vec![0.0; dim];
        let mut r = g_test.clone();
        let mut p = r.clone();
        let mut rs_old = dot(&r, &r);

        for _ in 0..self.cg_iters {
            let ap = hv(&p);
            let pap = dot(&p, &ap);
            if pap.abs() < 1e-30 {
                break;
            }
            let alpha = rs_old / pap;
            for j in 0..dim {
                s[j] += alpha * p[j];
                r[j] -= alpha * ap[j];
            }
            let rs_new = dot(&r, &r);
            if rs_new < 1e-20 {
                break;
            }
            let beta = rs_new / rs_old.max(1e-30);
            for j in 0..dim {
                p[j] = r[j] + beta * p[j];
            }
            rs_old = rs_new;
        }

        // Influence = -<g_train, s>
        Ok(-dot(&g_train, &s))
    }

    /// Find the top-k most influential training samples for a given test sample.
    ///
    /// Returns a sorted vec of `(train_idx, influence_score)` in descending
    /// order of absolute influence.
    pub fn find_most_influential(
        &self,
        test_checkpoints: &[IfGradientCheckpoint],
        k: usize,
    ) -> Result<Vec<(usize, f64)>> {
        let mut scores: Vec<(usize, f64)> = Vec::with_capacity(self.train_checkpoints.len());
        for idx in 0..self.train_checkpoints.len() {
            let score = self.compute_influence(idx, test_checkpoints)?;
            scores.push((idx, score));
        }
        scores.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(k);
        Ok(scores)
    }

    /// Mean gradient over a set of checkpoints.
    fn mean_gradient(cps: &[IfGradientCheckpoint], dim: usize) -> Vec<f64> {
        let n = cps.len() as f64;
        let mut g = vec![0.0; dim];
        for cp in cps {
            for (j, gj) in cp.gradient.iter().enumerate().take(dim) {
                g[j] += gj / n;
            }
        }
        g
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. IfPermutationImportance — Feature importance by permutation
// ─────────────────────────────────────────────────────────────────────────────

/// Which metric to use when measuring performance degradation after
/// permuting a feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IfMetricKind {
    /// Classification accuracy.
    Accuracy,
    /// Mean Squared Error (lower is better).
    Mse,
    /// Mean Absolute Error (lower is better).
    Mae,
}

/// Result of a single feature's permutation importance.
#[derive(Debug, Clone)]
pub struct IfFeatureImportance {
    /// Feature index.
    pub feature_idx: usize,
    /// Mean importance score (mean decrease in metric over repeats).
    pub mean_score: f64,
    /// Standard deviation of the importance score over repeats.
    pub std_score: f64,
}

/// Permutation importance computer.
///
/// Estimates feature importance by repeatedly shuffling each feature column
/// using a Fisher-Yates shuffle and measuring the resulting degradation in
/// model performance.
#[derive(Debug, Clone)]
pub struct IfPermutationImportance {
    metric: IfMetricKind,
    n_repeats: usize,
    seed: u64,
}

impl IfPermutationImportance {
    /// Create a new permutation importance estimator.
    ///
    /// * `metric` — Which metric to evaluate.
    /// * `n_repeats` — How many shuffles per feature.
    /// * `seed` — RNG seed for reproducibility.
    pub fn new(metric: IfMetricKind, n_repeats: usize, seed: u64) -> Self {
        Self {
            metric,
            n_repeats: n_repeats.max(1),
            seed,
        }
    }

    /// Compute permutation importance for all features.
    ///
    /// * `model_fn` — A function `(features: &[Vec<f64>]) -> Result<Vec<f64>>`
    ///   that takes `n_samples` feature vectors (each of length `n_features`) and
    ///   returns predictions.
    /// * `data` — `n_samples` feature vectors, each of length `n_features`.
    /// * `labels` — Ground-truth labels of length `n_samples`.
    pub fn compute(
        &self,
        model_fn: &dyn Fn(&[Vec<f64>]) -> Result<Vec<f64>>,
        data: &[Vec<f64>],
        labels: &[f64],
    ) -> Result<Vec<IfFeatureImportance>> {
        let n = data.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "IfPermutationImportance::compute",
                "data must not be empty",
            ));
        }
        if labels.len() != n {
            return Err(TensorError::invalid_argument_op(
                "IfPermutationImportance::compute",
                "data and labels must have the same length",
            ));
        }
        let n_features = data[0].len();
        if n_features == 0 {
            return Err(TensorError::invalid_argument_op(
                "IfPermutationImportance::compute",
                "feature vectors must not be empty",
            ));
        }

        // Baseline metric
        let base_preds = model_fn(data)?;
        let base_score = self.evaluate_metric(&base_preds, labels)?;

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut results = Vec::with_capacity(n_features);

        for feat in 0..n_features {
            let mut repeat_scores = Vec::with_capacity(self.n_repeats);

            for _ in 0..self.n_repeats {
                // Copy data and shuffle column `feat`
                let mut shuffled = data.to_vec();
                // Fisher-Yates shuffle on column
                let mut col: Vec<f64> = shuffled.iter().map(|row| row[feat]).collect();
                for i in (1..col.len()).rev() {
                    let j = (rng.random::<f64>() * (i + 1) as f64) as usize;
                    let j = j.min(i); // safety clamp
                    col.swap(i, j);
                }
                for (row_idx, row) in shuffled.iter_mut().enumerate() {
                    row[feat] = col[row_idx];
                }

                let preds = model_fn(&shuffled)?;
                let score = self.evaluate_metric(&preds, labels)?;
                repeat_scores.push(score);
            }

            let mean_perm = mean(&repeat_scores);
            let importance = match self.metric {
                // For accuracy, higher is better → importance = base - permuted
                IfMetricKind::Accuracy => base_score - mean_perm,
                // For MSE/MAE, lower is better → importance = permuted - base
                IfMetricKind::Mse | IfMetricKind::Mae => mean_perm - base_score,
            };

            let std_imp = if self.n_repeats > 1 {
                let scores_diff: Vec<f64> = repeat_scores
                    .iter()
                    .map(|s| match self.metric {
                        IfMetricKind::Accuracy => base_score - s,
                        IfMetricKind::Mse | IfMetricKind::Mae => s - base_score,
                    })
                    .collect();
                std_dev(&scores_diff)
            } else {
                0.0
            };

            results.push(IfFeatureImportance {
                feature_idx: feat,
                mean_score: importance,
                std_score: std_imp,
            });
        }

        Ok(results)
    }

    /// Evaluate the chosen metric.
    fn evaluate_metric(&self, predictions: &[f64], labels: &[f64]) -> Result<f64> {
        let n = predictions.len();
        if n == 0 {
            return Err(TensorError::invalid_argument_op(
                "evaluate_metric",
                "empty predictions",
            ));
        }
        match self.metric {
            IfMetricKind::Accuracy => {
                let correct = predictions
                    .iter()
                    .zip(labels.iter())
                    .filter(|(p, l)| (p.round() - l.round()).abs() < 1e-6)
                    .count();
                Ok(correct as f64 / n as f64)
            }
            IfMetricKind::Mse => {
                let mse: f64 = predictions
                    .iter()
                    .zip(labels.iter())
                    .map(|(p, l)| (p - l).powi(2))
                    .sum::<f64>()
                    / n as f64;
                Ok(mse)
            }
            IfMetricKind::Mae => {
                let mae: f64 = predictions
                    .iter()
                    .zip(labels.iter())
                    .map(|(p, l)| (p - l).abs())
                    .sum::<f64>()
                    / n as f64;
                Ok(mae)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. IfAttentionRollout — Full attention flow computation
// ─────────────────────────────────────────────────────────────────────────────

/// Attention rollout computer for multi-layer transformer models.
///
/// Given attention matrices from each layer, computes the total attention
/// flow from the input tokens to the output using the rollout method
/// (Abnar & Zuidema, 2020):
///
/// `A_flow = Π_{i=1}^{L} (0.5 * I + 0.5 * Ā_i)`
///
/// where `Ā_i` is the row-normalised (or head-averaged) attention matrix
/// at layer `i`.
#[derive(Debug, Clone)]
pub struct IfAttentionRollout {
    /// Identity blending factor (typically 0.5).
    blend_factor: f64,
}

impl IfAttentionRollout {
    /// Create a new attention rollout computer.
    ///
    /// * `blend_factor` — Weight of identity in the blending
    ///   `(1 - blend) * I + blend * A`.  Default is 0.5.
    pub fn new(blend_factor: f64) -> Self {
        Self {
            blend_factor: clamp(blend_factor, 0.0, 1.0),
        }
    }

    /// Compute the attention rollout across layers.
    ///
    /// * `attention_maps` — A vec of `n_layers` attention matrices, each of
    ///   shape `[seq_len, seq_len]` stored row-major.  If multiple heads
    ///   exist, pass the head-averaged matrix.
    /// * `seq_len` — Sequence length (number of tokens).
    ///
    /// Returns the accumulated attention flow matrix `[seq_len, seq_len]`.
    pub fn compute_rollout(&self, attention_maps: &[Vec<f64>], seq_len: usize) -> Result<Vec<f64>> {
        if attention_maps.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "IfAttentionRollout::compute_rollout",
                "attention_maps must not be empty",
            ));
        }
        if seq_len == 0 {
            return Err(TensorError::invalid_argument_op(
                "IfAttentionRollout::compute_rollout",
                "seq_len must be > 0",
            ));
        }
        let expected = seq_len * seq_len;
        for (i, am) in attention_maps.iter().enumerate() {
            if am.len() != expected {
                return Err(TensorError::invalid_argument_op(
                    "IfAttentionRollout::compute_rollout",
                    &format!(
                        "attention_maps[{}] has length {} but expected {}",
                        i,
                        am.len(),
                        expected
                    ),
                ));
            }
        }

        // Start with identity matrix
        let mut flow = vec![0.0; expected];
        for i in 0..seq_len {
            flow[i * seq_len + i] = 1.0;
        }

        let blend = self.blend_factor;
        let id_weight = 1.0 - blend;

        for am in attention_maps {
            // Row-normalise the attention matrix
            let norm_am = self.row_normalise(am, seq_len);

            // Blended: B = id_weight * I + blend * norm_am
            let mut blended = vec![0.0; expected];
            for r in 0..seq_len {
                for c in 0..seq_len {
                    let idx = r * seq_len + c;
                    blended[idx] = blend * norm_am[idx];
                    if r == c {
                        blended[idx] += id_weight;
                    }
                }
            }

            // flow = flow * blended (matrix multiply)
            flow = self.matmul(&flow, &blended, seq_len);
        }

        Ok(flow)
    }

    /// Gradient-weighted attention: multiply attention maps by gradient
    /// magnitudes before rollout.
    ///
    /// * `attention_maps` — Per-layer `[seq_len, seq_len]` attention matrices.
    /// * `gradients` — Per-layer gradient magnitude vectors `[seq_len]`.
    /// * `seq_len` — Sequence length.
    ///
    /// Returns per-token importance scores `[seq_len]`.
    pub fn compute_token_importance(
        &self,
        attention_maps: &[Vec<f64>],
        gradients: &[Vec<f64>],
        seq_len: usize,
    ) -> Result<Vec<f64>> {
        if attention_maps.len() != gradients.len() {
            return Err(TensorError::invalid_argument_op(
                "IfAttentionRollout::compute_token_importance",
                "attention_maps and gradients must have the same length",
            ));
        }
        for (i, g) in gradients.iter().enumerate() {
            if g.len() != seq_len {
                return Err(TensorError::invalid_argument_op(
                    "IfAttentionRollout::compute_token_importance",
                    &format!(
                        "gradients[{}] has length {} but expected {}",
                        i,
                        g.len(),
                        seq_len
                    ),
                ));
            }
        }

        // Weight each attention row by the gradient magnitude at that position
        let weighted_maps: Vec<Vec<f64>> = attention_maps
            .iter()
            .zip(gradients.iter())
            .map(|(am, grad)| {
                let mut wam = am.clone();
                for r in 0..seq_len {
                    let g_mag = grad[r].abs();
                    for c in 0..seq_len {
                        wam[r * seq_len + c] *= g_mag;
                    }
                }
                wam
            })
            .collect();

        // Compute rollout on the weighted maps
        let flow = self.compute_rollout(&weighted_maps, seq_len)?;

        // Extract importance: sum of the first row (CLS token) or mean of rows
        // We use the mean column sums as the token importance
        let mut importance = vec![0.0; seq_len];
        for r in 0..seq_len {
            for c in 0..seq_len {
                importance[c] += flow[r * seq_len + c];
            }
        }
        let total: f64 = importance.iter().sum();
        if total > 1e-30 {
            for v in &mut importance {
                *v /= total;
            }
        }

        Ok(importance)
    }

    /// Row-normalise an attention matrix so each row sums to 1.
    fn row_normalise(&self, am: &[f64], seq_len: usize) -> Vec<f64> {
        let mut result = am.to_vec();
        for r in 0..seq_len {
            let row_sum: f64 = (0..seq_len).map(|c| result[r * seq_len + c]).sum();
            if row_sum > 1e-30 {
                for c in 0..seq_len {
                    result[r * seq_len + c] /= row_sum;
                }
            }
        }
        result
    }

    /// Dense matrix multiply `A * B` where both are `[n, n]` row-major.
    fn matmul(&self, a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut c = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..n {
                let aik = a[i * n + k];
                if aik.abs() < 1e-30 {
                    continue;
                }
                for j in 0..n {
                    c[i * n + j] += aik * b[k * n + j];
                }
            }
        }
        c
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. IfFairnessAnalyzer — Bias detection and fairness metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Complete fairness evaluation report.
#[derive(Debug, Clone)]
pub struct IfFairnessReport {
    /// Demographic parity: |P(Ŷ=1|A=0) - P(Ŷ=1|A=1)|.
    pub demographic_parity: f64,
    /// Equalised odds: max(|TPR gap|, |FPR gap|).
    pub equalized_odds: f64,
    /// TPR gap: TPR(A=1) - TPR(A=0).
    pub tpr_gap: f64,
    /// FPR gap: FPR(A=1) - FPR(A=0).
    pub fpr_gap: f64,
    /// Calibration gap: |E[Y|Ŷ≈p, A=0] - E[Y|Ŷ≈p, A=1]| averaged over bins.
    pub calibration_gap: f64,
    /// Predictive parity: |Precision(A=0) - Precision(A=1)|.
    pub predictive_parity: f64,
    /// Disparate impact ratio: min(rate_0 / rate_1, rate_1 / rate_0).
    /// The 4/5 (80%) rule threshold.
    pub disparate_impact: f64,
}

/// Fairness analyser for binary classification with a binary protected
/// attribute.
///
/// Computes a comprehensive set of group fairness metrics following the
/// taxonomy in Verma & Rubin (2018).
#[derive(Debug)]
pub struct IfFairnessAnalyzer {
    /// Number of calibration bins.
    n_bins: usize,
}

impl IfFairnessAnalyzer {
    /// Create a new fairness analyser.
    ///
    /// * `n_bins` — Number of calibration bins for the calibration gap metric.
    pub fn new(n_bins: usize) -> Self {
        Self {
            n_bins: n_bins.max(2),
        }
    }

    /// Analyse fairness.
    ///
    /// * `predictions` — Model predictions (probabilities in [0, 1]).
    /// * `labels` — Ground-truth binary labels (0 or 1).
    /// * `protected` — Protected attribute (0 or 1) for each sample.
    /// * `threshold` — Classification threshold for converting probabilities
    ///   to binary predictions.
    pub fn analyze(
        &self,
        predictions: &[f64],
        labels: &[f64],
        protected: &[f64],
        threshold: f64,
    ) -> Result<IfFairnessReport> {
        let n = predictions.len();
        if n == 0 || labels.len() != n || protected.len() != n {
            return Err(TensorError::invalid_argument_op(
                "IfFairnessAnalyzer::analyze",
                "predictions, labels, and protected must have the same non-zero length",
            ));
        }

        // Split into groups
        let mut group0_preds = Vec::new();
        let mut group0_labels = Vec::new();
        let mut group1_preds = Vec::new();
        let mut group1_labels = Vec::new();

        for i in 0..n {
            if protected[i] < 0.5 {
                group0_preds.push(predictions[i]);
                group0_labels.push(labels[i]);
            } else {
                group1_preds.push(predictions[i]);
                group1_labels.push(labels[i]);
            }
        }

        if group0_preds.is_empty() || group1_preds.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "IfFairnessAnalyzer::analyze",
                "both protected groups must have at least one sample",
            ));
        }

        // --- Demographic Parity ---
        let rate0 = group0_preds.iter().filter(|p| **p >= threshold).count() as f64
            / group0_preds.len() as f64;
        let rate1 = group1_preds.iter().filter(|p| **p >= threshold).count() as f64
            / group1_preds.len() as f64;
        let demographic_parity = (rate0 - rate1).abs();

        // --- Disparate Impact ---
        let disparate_impact = if rate0.max(rate1) < 1e-30 {
            1.0
        } else {
            rate0.min(rate1) / rate0.max(rate1)
        };

        // --- TPR / FPR per group ---
        let (tpr0, fpr0) = Self::tpr_fpr(&group0_preds, &group0_labels, threshold);
        let (tpr1, fpr1) = Self::tpr_fpr(&group1_preds, &group1_labels, threshold);
        let tpr_gap = tpr1 - tpr0;
        let fpr_gap = fpr1 - fpr0;
        let equalized_odds = tpr_gap.abs().max(fpr_gap.abs());

        // --- Predictive Parity ---
        let prec0 = Self::precision(&group0_preds, &group0_labels, threshold);
        let prec1 = Self::precision(&group1_preds, &group1_labels, threshold);
        let predictive_parity = (prec0 - prec1).abs();

        // --- Calibration Gap ---
        let calibration_gap =
            self.calibration_gap(&group0_preds, &group0_labels, &group1_preds, &group1_labels);

        Ok(IfFairnessReport {
            demographic_parity,
            equalized_odds,
            tpr_gap,
            fpr_gap,
            calibration_gap,
            predictive_parity,
            disparate_impact,
        })
    }

    /// Compute TPR and FPR for a group.
    fn tpr_fpr(preds: &[f64], labels: &[f64], threshold: f64) -> (f64, f64) {
        let mut tp = 0usize;
        let mut fn_ = 0usize;
        let mut fp = 0usize;
        let mut tn = 0usize;
        for (p, l) in preds.iter().zip(labels.iter()) {
            let pred_pos = *p >= threshold;
            let actual_pos = *l >= 0.5;
            match (pred_pos, actual_pos) {
                (true, true) => tp += 1,
                (false, true) => fn_ += 1,
                (true, false) => fp += 1,
                (false, false) => tn += 1,
            }
        }
        let tpr = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };
        let fpr = if fp + tn > 0 {
            fp as f64 / (fp + tn) as f64
        } else {
            0.0
        };
        (tpr, fpr)
    }

    /// Precision for a group.
    fn precision(preds: &[f64], labels: &[f64], threshold: f64) -> f64 {
        let mut tp = 0usize;
        let mut fp = 0usize;
        for (p, l) in preds.iter().zip(labels.iter()) {
            if *p >= threshold {
                if *l >= 0.5 {
                    tp += 1;
                } else {
                    fp += 1;
                }
            }
        }
        if tp + fp == 0 {
            return 0.0;
        }
        tp as f64 / (tp + fp) as f64
    }

    /// Calibration gap: mean absolute difference in calibration across bins.
    fn calibration_gap(
        &self,
        preds0: &[f64],
        labels0: &[f64],
        preds1: &[f64],
        labels1: &[f64],
    ) -> f64 {
        let mut total_gap = 0.0_f64;
        let mut n_bins_used = 0usize;
        let bin_width = 1.0 / self.n_bins as f64;

        for b in 0..self.n_bins {
            let lo = b as f64 * bin_width;
            let hi = lo + bin_width;

            let (sum0, cnt0) = Self::bin_calibration(preds0, labels0, lo, hi);
            let (sum1, cnt1) = Self::bin_calibration(preds1, labels1, lo, hi);

            if cnt0 > 0 && cnt1 > 0 {
                let cal0 = sum0 / cnt0 as f64;
                let cal1 = sum1 / cnt1 as f64;
                total_gap += (cal0 - cal1).abs();
                n_bins_used += 1;
            }
        }

        if n_bins_used == 0 {
            return 0.0;
        }
        total_gap / n_bins_used as f64
    }

    /// Sum and count of true labels in a prediction bin.
    fn bin_calibration(preds: &[f64], labels: &[f64], lo: f64, hi: f64) -> (f64, usize) {
        let mut sum = 0.0_f64;
        let mut count = 0usize;
        for (p, l) in preds.iter().zip(labels.iter()) {
            if *p >= lo && *p < hi {
                sum += l;
                count += 1;
            }
        }
        (sum, count)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. IfBiasDetector — Spurious correlation detection
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a bias / spurious-correlation analysis.
#[derive(Debug, Clone)]
pub struct IfBiasReport {
    /// Mutual information between each feature and the protected attribute.
    pub feature_protected_mi: Vec<f64>,
    /// Partial correlation between each feature and labels, controlling for
    /// the protected attribute.
    pub partial_correlations: Vec<f64>,
    /// Feature attributions conditioned on protected=0.
    pub attribution_group0: Vec<f64>,
    /// Feature attributions conditioned on protected=1.
    pub attribution_group1: Vec<f64>,
    /// Attribution divergence between groups per feature.
    pub attribution_divergence: Vec<f64>,
}

/// Bias detector: finds spurious correlations between features and a
/// protected attribute that may cause unfair predictions.
///
/// Uses:
/// 1. Mutual information estimation (histogram-based) between each feature
///    and the protected attribute.
/// 2. Partial correlation: Pearson correlation between feature and label
///    controlling for the protected attribute (residual method).
/// 3. Feature attribution divergence between protected groups.
#[derive(Debug, Clone)]
pub struct IfBiasDetector {
    /// Number of bins for MI estimation.
    n_bins: usize,
}

impl IfBiasDetector {
    /// Create a new bias detector.
    pub fn new(n_bins: usize) -> Self {
        Self {
            n_bins: n_bins.max(2),
        }
    }

    /// Detect spurious correlations.
    ///
    /// * `features` — `n_samples` feature vectors.
    /// * `labels` — Ground-truth labels.
    /// * `protected` — Protected attribute (binary: 0 or 1).
    pub fn detect_spurious_correlations(
        &self,
        features: &[Vec<f64>],
        labels: &[f64],
        protected: &[f64],
    ) -> Result<IfBiasReport> {
        let n = features.len();
        if n == 0 || labels.len() != n || protected.len() != n {
            return Err(TensorError::invalid_argument_op(
                "IfBiasDetector::detect_spurious_correlations",
                "features, labels, protected must have the same non-zero length",
            ));
        }
        let d = features[0].len();
        if d == 0 {
            return Err(TensorError::invalid_argument_op(
                "IfBiasDetector::detect_spurious_correlations",
                "feature vectors must have non-zero dimension",
            ));
        }

        // 1. MI(feature_j, protected)
        let mut mi_scores = Vec::with_capacity(d);
        for j in 0..d {
            let col: Vec<f64> = features.iter().map(|row| row[j]).collect();
            let mi = self.histogram_mi(&col, protected);
            mi_scores.push(mi);
        }

        // 2. Partial correlation: corr(feature_j, label | protected)
        let mut partial_corrs = Vec::with_capacity(d);
        for j in 0..d {
            let col: Vec<f64> = features.iter().map(|row| row[j]).collect();
            let pc = Self::partial_correlation(&col, labels, protected);
            partial_corrs.push(pc);
        }

        // 3. Feature attribution per group
        // Use simple linear regression coefficient as a proxy attribution
        let mut attr_g0 = vec![0.0; d];
        let mut attr_g1 = vec![0.0; d];
        let mut attr_div = vec![0.0; d];

        let (feats0, labels0): (Vec<&Vec<f64>>, Vec<f64>) = features
            .iter()
            .zip(labels.iter())
            .zip(protected.iter())
            .filter(|((_, _), p)| **p < 0.5)
            .map(|((f, l), _)| (f, *l))
            .unzip();

        let (feats1, labels1): (Vec<&Vec<f64>>, Vec<f64>) = features
            .iter()
            .zip(labels.iter())
            .zip(protected.iter())
            .filter(|((_, _), p)| **p >= 0.5)
            .map(|((f, l), _)| (f, *l))
            .unzip();

        for j in 0..d {
            if !feats0.is_empty() {
                let col: Vec<f64> = feats0.iter().map(|row| row[j]).collect();
                attr_g0[j] = pearson_corr(&col, &labels0).abs();
            }
            if !feats1.is_empty() {
                let col: Vec<f64> = feats1.iter().map(|row| row[j]).collect();
                attr_g1[j] = pearson_corr(&col, &labels1).abs();
            }
            attr_div[j] = (attr_g0[j] - attr_g1[j]).abs();
        }

        Ok(IfBiasReport {
            feature_protected_mi: mi_scores,
            partial_correlations: partial_corrs,
            attribution_group0: attr_g0,
            attribution_group1: attr_g1,
            attribution_divergence: attr_div,
        })
    }

    /// Histogram-based mutual information estimation between two
    /// real-valued vectors.
    fn histogram_mi(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        if n < 2 {
            return 0.0;
        }

        let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let x_range = (x_max - x_min).max(1e-10);
        let y_range = (y_max - y_min).max(1e-10);

        let nb = self.n_bins;
        let mut joint = vec![0usize; nb * nb];
        let mut marginal_x = vec![0usize; nb];
        let mut marginal_y = vec![0usize; nb];

        for i in 0..n {
            let bx = ((x[i] - x_min) / x_range * (nb - 1) as f64).round() as usize;
            let by = ((y[i] - y_min) / y_range * (nb - 1) as f64).round() as usize;
            let bx = bx.min(nb - 1);
            let by = by.min(nb - 1);
            joint[bx * nb + by] += 1;
            marginal_x[bx] += 1;
            marginal_y[by] += 1;
        }

        let nf = n as f64;
        let mut mi = 0.0_f64;
        for bx in 0..nb {
            for by in 0..nb {
                let pxy = joint[bx * nb + by] as f64 / nf;
                let px = marginal_x[bx] as f64 / nf;
                let py = marginal_y[by] as f64 / nf;
                if pxy > 1e-30 && px > 1e-30 && py > 1e-30 {
                    mi += pxy * (pxy / (px * py)).ln();
                }
            }
        }
        mi.max(0.0)
    }

    /// Partial correlation of `x` and `y` controlling for `z`,
    /// computed via the residual method.
    fn partial_correlation(x: &[f64], y: &[f64], z: &[f64]) -> f64 {
        let rxy = pearson_corr(x, y);
        let rxz = pearson_corr(x, z);
        let ryz = pearson_corr(y, z);
        let denom = ((1.0 - rxz * rxz) * (1.0 - ryz * ryz)).sqrt();
        if denom < 1e-30 {
            return rxy;
        }
        (rxy - rxz * ryz) / denom
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. IfModelDivergence — Distribution shift tracking
// ─────────────────────────────────────────────────────────────────────────────

/// Drift detection report with multiple divergence metrics and alert status.
#[derive(Debug, Clone)]
pub struct IfDriftReport {
    /// Histogram-based KL divergence: KL(P || Q).
    pub kl_divergence: f64,
    /// Maximum Mean Discrepancy with RBF kernel.
    pub mmd: f64,
    /// Population Stability Index.
    pub psi: f64,
    /// Kolmogorov-Smirnov statistic.
    pub ks_statistic: f64,
    /// Whether any metric exceeds its alert threshold.
    pub alert: bool,
    /// Per-metric alert details.
    pub alerts: HashMap<String, bool>,
}

/// Alert thresholds for distribution drift metrics.
#[derive(Debug, Clone)]
pub struct IfDriftThresholds {
    /// KL divergence threshold (default: 0.1).
    pub kl_threshold: f64,
    /// MMD threshold (default: 0.1).
    pub mmd_threshold: f64,
    /// PSI threshold (default: 0.2 — "moderate shift").
    pub psi_threshold: f64,
    /// KS statistic threshold (default: 0.1).
    pub ks_threshold: f64,
}

impl Default for IfDriftThresholds {
    fn default() -> Self {
        Self {
            kl_threshold: 0.1,
            mmd_threshold: 0.1,
            psi_threshold: 0.2,
            ks_threshold: 0.1,
        }
    }
}

/// Distribution-shift tracker using multiple divergence measures.
///
/// Supports:
/// - Histogram-based KL divergence (symmetric).
/// - Maximum Mean Discrepancy with RBF kernel.
/// - Population Stability Index (PSI).
/// - Kolmogorov-Smirnov statistic.
#[derive(Debug, Clone)]
pub struct IfModelDivergence {
    /// Number of histogram bins.
    n_bins: usize,
    /// RBF bandwidth for MMD computation.
    bandwidth: f64,
    /// Alert thresholds.
    thresholds: IfDriftThresholds,
}

impl IfModelDivergence {
    /// Create a new divergence tracker.
    pub fn new(n_bins: usize, bandwidth: f64, thresholds: IfDriftThresholds) -> Self {
        Self {
            n_bins: n_bins.max(2),
            bandwidth: bandwidth.max(1e-6),
            thresholds,
        }
    }

    /// Compute all divergence metrics between two distributions.
    ///
    /// * `reference` — Reference (training) distribution samples.
    /// * `current` — Current (production) distribution samples.
    pub fn compute_drift(&self, reference: &[f64], current: &[f64]) -> Result<IfDriftReport> {
        if reference.is_empty() || current.is_empty() {
            return Err(TensorError::invalid_argument_op(
                "IfModelDivergence::compute_drift",
                "reference and current must not be empty",
            ));
        }

        let kl = self.kl_divergence(reference, current);
        let mmd_val = self.mmd(reference, current);
        let psi = self.psi(reference, current);
        let ks = self.ks_statistic(reference, current);

        let mut alerts = HashMap::new();
        alerts.insert("kl".to_string(), kl > self.thresholds.kl_threshold);
        alerts.insert("mmd".to_string(), mmd_val > self.thresholds.mmd_threshold);
        alerts.insert("psi".to_string(), psi > self.thresholds.psi_threshold);
        alerts.insert("ks".to_string(), ks > self.thresholds.ks_threshold);

        let alert = alerts.values().any(|v| *v);

        Ok(IfDriftReport {
            kl_divergence: kl,
            mmd: mmd_val,
            psi,
            ks_statistic: ks,
            alert,
            alerts,
        })
    }

    /// Histogram-based KL divergence: KL(P || Q).
    fn kl_divergence(&self, p: &[f64], q: &[f64]) -> f64 {
        let (hist_p, hist_q) = self.compute_histograms(p, q);
        let np = p.len() as f64;
        let nq = q.len() as f64;
        let eps = 1e-10;

        let mut kl = 0.0_f64;
        for i in 0..self.n_bins {
            let pi = (hist_p[i] as f64 / np) + eps;
            let qi = (hist_q[i] as f64 / nq) + eps;
            kl += pi * (pi / qi).ln();
        }
        kl.max(0.0)
    }

    /// Maximum Mean Discrepancy with RBF kernel.
    fn mmd(&self, x: &[f64], y: &[f64]) -> f64 {
        let nx = x.len() as f64;
        let ny = y.len() as f64;

        // E[k(x, x')]
        let mut kxx = 0.0_f64;
        for i in 0..x.len() {
            for j in (i + 1)..x.len() {
                kxx += rbf_kernel(&[x[i]], &[x[j]], self.bandwidth);
            }
        }
        kxx = if x.len() > 1 {
            2.0 * kxx / (nx * (nx - 1.0))
        } else {
            0.0
        };

        // E[k(y, y')]
        let mut kyy = 0.0_f64;
        for i in 0..y.len() {
            for j in (i + 1)..y.len() {
                kyy += rbf_kernel(&[y[i]], &[y[j]], self.bandwidth);
            }
        }
        kyy = if y.len() > 1 {
            2.0 * kyy / (ny * (ny - 1.0))
        } else {
            0.0
        };

        // E[k(x, y)]
        let mut kxy = 0.0_f64;
        for xi in x {
            for yj in y {
                kxy += rbf_kernel(&[*xi], &[*yj], self.bandwidth);
            }
        }
        kxy /= nx * ny;

        (kxx + kyy - 2.0 * kxy).max(0.0).sqrt()
    }

    /// Population Stability Index.
    fn psi(&self, reference: &[f64], current: &[f64]) -> f64 {
        let (hist_r, hist_c) = self.compute_histograms(reference, current);
        let nr = reference.len() as f64;
        let nc = current.len() as f64;
        let eps = 1e-10;

        let mut psi_val = 0.0_f64;
        for i in 0..self.n_bins {
            let pr = (hist_r[i] as f64 / nr) + eps;
            let pc = (hist_c[i] as f64 / nc) + eps;
            psi_val += (pc - pr) * (pc / pr).ln();
        }
        psi_val.max(0.0)
    }

    /// Kolmogorov-Smirnov statistic: max |F_ref(x) - F_cur(x)|.
    fn ks_statistic(&self, reference: &[f64], current: &[f64]) -> f64 {
        let mut ref_sorted = reference.to_vec();
        let mut cur_sorted = current.to_vec();
        ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        cur_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let nr = reference.len() as f64;
        let nc = current.len() as f64;

        // Merge both sorted arrays and track CDFs
        let mut ri = 0usize;
        let mut ci = 0usize;
        let mut max_d = 0.0_f64;

        while ri < ref_sorted.len() || ci < cur_sorted.len() {
            let rv = if ri < ref_sorted.len() {
                ref_sorted[ri]
            } else {
                f64::INFINITY
            };
            let cv = if ci < cur_sorted.len() {
                cur_sorted[ci]
            } else {
                f64::INFINITY
            };

            if rv <= cv {
                ri += 1;
            }
            if cv <= rv {
                ci += 1;
            }

            let cdf_r = ri as f64 / nr;
            let cdf_c = ci as f64 / nc;
            let d = (cdf_r - cdf_c).abs();
            if d > max_d {
                max_d = d;
            }
        }
        max_d
    }

    /// Compute aligned histograms for two distributions using the same bin
    /// boundaries.
    fn compute_histograms(&self, a: &[f64], b: &[f64]) -> (Vec<usize>, Vec<usize>) {
        let all_min = a
            .iter()
            .chain(b.iter())
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let all_max = a
            .iter()
            .chain(b.iter())
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let range = (all_max - all_min).max(1e-10);

        let nb = self.n_bins;
        let mut hist_a = vec![0usize; nb];
        let mut hist_b = vec![0usize; nb];

        for x in a {
            let bin = ((*x - all_min) / range * (nb - 1) as f64).round() as usize;
            hist_a[bin.min(nb - 1)] += 1;
        }
        for x in b {
            let bin = ((*x - all_min) / range * (nb - 1) as f64).round() as usize;
            hist_b[bin.min(nb - 1)] += 1;
        }

        (hist_a, hist_b)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. IfIntegratedGradients — Integrated gradients attribution
// ─────────────────────────────────────────────────────────────────────────────

/// Result of an integrated gradients computation.
#[derive(Debug, Clone)]
pub struct IfIgResult {
    /// Per-feature attribution scores.
    pub attributions: Vec<f64>,
    /// Convergence delta: `|sum(attributions) - (f(input) - f(baseline))|`.
    pub convergence_delta: f64,
    /// Whether the approximation converged (delta < 5% of output difference).
    pub converged: bool,
}

/// Integrated Gradients attribution method (Sundararajan et al., 2017).
///
/// Computes the path integral of gradients along a straight line from a
/// baseline to the input, using a Riemann sum approximation.
///
/// The attribution for feature `i` is:
/// `IG_i = (x_i - x'_i) * Σ_{k=1}^{n} ∂F/∂x_i(x' + k/n * (x - x')) / n`
#[derive(Debug, Clone)]
pub struct IfIntegratedGradients {
    /// Number of interpolation steps.
    n_steps: usize,
    /// Finite-difference step for gradient computation.
    fd_step: f64,
}

impl IfIntegratedGradients {
    /// Create a new integrated gradients computer.
    ///
    /// * `n_steps` — Number of Riemann sum steps (higher = more accurate).
    /// * `fd_step` — Step size for numerical gradient.
    pub fn new(n_steps: usize, fd_step: f64) -> Self {
        Self {
            n_steps: n_steps.max(1),
            fd_step: fd_step.max(1e-8),
        }
    }

    /// Compute integrated gradients.
    ///
    /// * `model_fn` — Scalar-valued model function `f: &[f64] -> Result<f64>`.
    /// * `input` — The input to explain.
    /// * `baseline` — The baseline (reference) input (e.g., all zeros).
    pub fn compute(
        &self,
        model_fn: &dyn Fn(&[f64]) -> Result<f64>,
        input: &[f64],
        baseline: &[f64],
    ) -> Result<IfIgResult> {
        let d = input.len();
        if d == 0 {
            return Err(TensorError::invalid_argument_op(
                "IfIntegratedGradients::compute",
                "input must not be empty",
            ));
        }
        if baseline.len() != d {
            return Err(TensorError::invalid_argument_op(
                "IfIntegratedGradients::compute",
                "input and baseline must have the same length",
            ));
        }

        let f_input = model_fn(input)?;
        let f_baseline = model_fn(baseline)?;
        let delta = vec_sub(input, baseline);

        // Riemann sum: average gradients along the path
        let mut avg_grad = vec![0.0; d];
        for k in 1..=self.n_steps {
            let alpha = k as f64 / self.n_steps as f64;
            let point: Vec<f64> = baseline
                .iter()
                .zip(delta.iter())
                .map(|(b, di)| b + alpha * di)
                .collect();
            let grad = numerical_gradient(model_fn, &point, self.fd_step)?;
            for j in 0..d {
                avg_grad[j] += grad[j] / self.n_steps as f64;
            }
        }

        // Attributions = (x - baseline) * avg_grad
        let attributions: Vec<f64> = delta
            .iter()
            .zip(avg_grad.iter())
            .map(|(di, gi)| di * gi)
            .collect();

        // Convergence check
        let attr_sum: f64 = attributions.iter().sum();
        let output_diff = f_input - f_baseline;
        let convergence_delta = (attr_sum - output_diff).abs();
        let converged = if output_diff.abs() < 1e-10 {
            convergence_delta < 1e-6
        } else {
            convergence_delta / output_diff.abs() < 0.05
        };

        Ok(IfIgResult {
            attributions,
            convergence_delta,
            converged,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. IfCounterfactualExplainer — Counterfactual explanations
// ─────────────────────────────────────────────────────────────────────────────

/// Regularisation type for counterfactual search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IfRegularization {
    /// L1 norm (sparse perturbation).
    L1,
    /// L2 norm (smooth perturbation).
    L2,
}

/// Result of a counterfactual explanation search.
#[derive(Debug, Clone)]
pub struct IfCounterfactualResult {
    /// The counterfactual input.
    pub counterfactual: Vec<f64>,
    /// The perturbation applied: `counterfactual - original`.
    pub perturbation: Vec<f64>,
    /// The model output at the counterfactual.
    pub output: f64,
    /// Whether the target class was achieved.
    pub success: bool,
    /// Number of iterations used.
    pub iterations: usize,
    /// Perturbation size (L1 or L2 norm depending on config).
    pub perturbation_size: f64,
}

/// Gradient-based counterfactual explanation finder.
///
/// Searches for the smallest perturbation that changes the model's prediction
/// to a target class/value, using gradient descent with L1 or L2 regularisation.
///
/// Objective: `min_δ loss(f(x+δ), target) + λ * ||δ||_{1 or 2}`
#[derive(Debug, Clone)]
pub struct IfCounterfactualExplainer {
    /// Regularisation type.
    regularization: IfRegularization,
    /// Regularisation strength.
    lambda: f64,
    /// Learning rate for gradient descent.
    learning_rate: f64,
    /// Maximum number of iterations.
    max_iters: usize,
    /// Convergence tolerance for the loss.
    tolerance: f64,
    /// Finite-difference step for gradients.
    fd_step: f64,
}

impl IfCounterfactualExplainer {
    /// Create a new counterfactual explainer.
    ///
    /// * `regularization` — L1 or L2.
    /// * `lambda` — Regularisation strength.
    /// * `learning_rate` — Step size for gradient descent.
    /// * `max_iters` — Maximum number of optimisation steps.
    /// * `tolerance` — Stop when loss < tolerance.
    pub fn new(
        regularization: IfRegularization,
        lambda: f64,
        learning_rate: f64,
        max_iters: usize,
        tolerance: f64,
    ) -> Self {
        Self {
            regularization,
            lambda: lambda.max(0.0),
            learning_rate: learning_rate.max(1e-8),
            max_iters: max_iters.max(1),
            tolerance: tolerance.max(1e-12),
            fd_step: 1e-5,
        }
    }

    /// Find a counterfactual explanation.
    ///
    /// * `model_fn` — Scalar-valued model function.
    /// * `input` — Original input.
    /// * `target` — Target output value (e.g., 1.0 for positive class).
    pub fn explain(
        &self,
        model_fn: &dyn Fn(&[f64]) -> Result<f64>,
        input: &[f64],
        target: f64,
    ) -> Result<IfCounterfactualResult> {
        let d = input.len();
        if d == 0 {
            return Err(TensorError::invalid_argument_op(
                "IfCounterfactualExplainer::explain",
                "input must not be empty",
            ));
        }

        let mut cf = input.to_vec();
        let mut best_cf = cf.clone();
        let mut best_loss = f64::INFINITY;
        let mut success = false;
        let mut iters_used = 0;

        for iter in 0..self.max_iters {
            iters_used = iter + 1;

            let output = model_fn(&cf)?;

            // Classification loss: (output - target)^2
            let class_loss = (output - target).powi(2);

            // Regularisation
            let delta = vec_sub(&cf, input);
            let reg_loss = match self.regularization {
                IfRegularization::L1 => l1_norm(&delta),
                IfRegularization::L2 => l2_norm(&delta).powi(2),
            };

            let total_loss = class_loss + self.lambda * reg_loss;

            if total_loss < best_loss {
                best_loss = total_loss;
                best_cf = cf.clone();
            }

            // Check convergence on classification loss
            if class_loss < self.tolerance {
                success = true;
                best_cf = cf.clone();
                break;
            }

            // Compute gradient of loss w.r.t. cf via FD
            let loss_fn = |x: &[f64]| -> Result<f64> {
                let o = model_fn(x)?;
                let cl = (o - target).powi(2);
                let d_x = vec_sub(x, input);
                let rl = match self.regularization {
                    IfRegularization::L1 => l1_norm(&d_x),
                    IfRegularization::L2 => l2_norm(&d_x).powi(2),
                };
                Ok(cl + self.lambda * rl)
            };

            let grad = numerical_gradient(&loss_fn, &cf, self.fd_step)?;

            // Gradient descent step
            for j in 0..d {
                cf[j] -= self.learning_rate * grad[j];
            }
        }

        let final_output = model_fn(&best_cf)?;
        let perturbation = vec_sub(&best_cf, input);
        let perturbation_size = match self.regularization {
            IfRegularization::L1 => l1_norm(&perturbation),
            IfRegularization::L2 => l2_norm(&perturbation),
        };

        // Re-check success
        if (final_output - target).powi(2) < self.tolerance {
            success = true;
        }

        Ok(IfCounterfactualResult {
            counterfactual: best_cf,
            perturbation,
            output: final_output,
            success,
            iterations: iters_used,
            perturbation_size,
        })
    }
}

#[cfg(test)]
mod tests;
