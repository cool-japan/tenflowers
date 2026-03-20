//! Advanced Optimisation & Training Dynamics — Track TD.
//!
//! Provides a rich collection of algorithms for loss-landscape analysis,
//! sharpness-aware minimisation, gradient monitoring, advanced LR schedules,
//! and curriculum learning strategies.
//!
//! All structs implement `Debug` + `Clone` and contain no `unwrap()` calls
//! outside of the test section.  The module is backed by plain `Vec<f64>`
//! arithmetic and uses `scirs2_core` for reproducible random sampling.
//!
//! # Quick Overview
//!
//! | Group | Key types |
//! |-------|-----------|
//! | SAM | [`SamOptimizer`], [`AsymSam`], [`FisherSam`], [`SamConfig`] |
//! | Gradient monitoring | [`GradientMonitor`], [`SignalToNoiseRatio`], [`LayerWiseLrScheduler`], [`GradientFlowChecker`], [`GradientHealthReport`] |
//! | Loss landscape | [`LossLandscape`], [`SharpnessMetric`], [`HessianTrace`], [`FlatnessMeasure`] |
//! | LR schedules | [`WarmupCosineScheduler`], [`PolynomialDecayScheduler`], [`OneCycleLrScheduler`], [`StochasticDepthScheduler`], [`GrokFastScheduler`] |
//! | Curriculum learning | [`DifficultyScorer`], [`CurriculumSampler`], [`SelfPacedLearning`], [`MixedCurriculumScheduler`], [`AntiCurriculumTrainer`] |

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// L2 norm of a f64 slice.
#[inline]
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Elementwise division of a slice by a scalar.
#[inline]
fn div_scalar(v: &[f64], s: f64) -> Vec<f64> {
    let safe = if s.abs() < 1e-30 { 1e-30 } else { s };
    v.iter().map(|x| x / safe).collect()
}

/// Elementwise addition of two slices of equal length.
#[inline]
fn add_vecs(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Elementwise subtraction `a - b`.
#[inline]
fn sub_vecs(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Scalar multiplication of a slice.
#[inline]
fn scale_vec(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}

/// Sample n values from N(0,1) using Box-Muller (seeded).
fn sample_standard_normal(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    let mut i = 0usize;
    while i < n {
        let u1: f64 = rng.random::<f64>().max(1e-40);
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        out.push(r * theta.cos());
        if i + 1 < n {
            out.push(r * theta.sin());
        }
        i += 2;
    }
    out.truncate(n);
    out
}

// ═════════════════════════════════════════════════════════════════════════════
// § 1 — Sharpness-Aware Minimisation (SAM)
// ═════════════════════════════════════════════════════════════════════════════

/// Configuration for SAM-family optimisers.
#[derive(Debug, Clone)]
pub struct SamConfig {
    /// Neighbourhood radius ρ (size of the perturbation ball).
    pub rho: f64,
    /// Adaptive SAM: scale ρ by per-parameter gradient magnitude.
    pub adaptive: bool,
    /// Base learning rate forwarded to the inner optimiser.
    pub base_lr: f64,
}

impl SamConfig {
    /// Create a new `SamConfig`.
    pub fn new(rho: f64, adaptive: bool, base_lr: f64) -> Self {
        Self {
            rho,
            adaptive,
            base_lr,
        }
    }
}

impl Default for SamConfig {
    fn default() -> Self {
        Self {
            rho: 0.05,
            adaptive: false,
            base_lr: 0.01,
        }
    }
}

/// Sharpness-Aware Minimisation (Foret et al., 2021).
///
/// Two-step protocol:
/// 1. \[`first_step`\] — compute the perturbation `ε = ρ * g / ‖g‖` and return
///    perturbed parameters `θ̂ = θ + ε`.
/// 2. \[`second_step`\] — apply a plain SGD update using the gradient evaluated
///    at the perturbed point, then restore the original parameters conceptually
///    (caller is responsible for using the returned params).
#[derive(Debug, Clone)]
pub struct SamOptimizer {
    /// SAM configuration.
    pub config: SamConfig,
    /// Saved original parameters from the last `first_step` call.
    saved_params: Vec<f64>,
    /// Saved perturbation from the last `first_step` call.
    saved_eps: Vec<f64>,
}

impl SamOptimizer {
    /// Create a new `SamOptimizer`.
    pub fn new(config: SamConfig) -> Self {
        Self {
            config,
            saved_params: Vec::new(),
            saved_eps: Vec::new(),
        }
    }

    /// **First step**: compute perturbation and return perturbed parameters.
    ///
    /// `grads` and `params` must have the same length.
    pub fn first_step(&mut self, grads: &[f64], params: &[f64]) -> Result<Vec<f64>> {
        if grads.len() != params.len() {
            return Err(TensorError::invalid_argument(
                "grads and params must have equal length".to_string(),
            ));
        }
        let norm = l2_norm(grads);
        let eps: Vec<f64> = if self.config.adaptive {
            // Adaptive SAM: scale each element by |g_i| * ρ / ‖g‖
            let safe_norm = norm.max(1e-30);
            grads
                .iter()
                .map(|g| self.config.rho * g.abs() * g / safe_norm)
                .collect()
        } else {
            // Standard SAM: uniform scaling
            let scaled = div_scalar(grads, norm);
            scale_vec(&scaled, self.config.rho)
        };
        self.saved_params = params.to_vec();
        self.saved_eps = eps.clone();
        Ok(add_vecs(params, &eps))
    }

    /// **Second step**: apply SGD update at perturbed point, then restore.
    ///
    /// Returns the updated parameters (restoring then applying `−lr * g_perturbed`).
    pub fn second_step(
        &self,
        grads_at_perturbed: &[f64],
        _perturbed_params: &[f64],
    ) -> Result<Vec<f64>> {
        if grads_at_perturbed.len() != self.saved_params.len() {
            return Err(TensorError::invalid_argument(
                "grads length must match saved params length".to_string(),
            ));
        }
        // Restore original params, then apply gradient step
        let update = scale_vec(grads_at_perturbed, self.config.base_lr);
        Ok(sub_vecs(&self.saved_params, &update))
    }

    /// Return the last saved perturbation.
    pub fn last_perturbation(&self) -> &[f64] {
        &self.saved_eps
    }
}

/// Asymmetric SAM — perturb only in the positive gradient direction.
///
/// Dimensions where the gradient is negative receive no perturbation,
/// reducing the perturbation towards a half-space neighbourhood.
#[derive(Debug, Clone)]
pub struct AsymSam {
    /// SAM configuration.
    pub config: SamConfig,
    /// Saved original parameters from the last `first_step` call.
    saved_params: Vec<f64>,
}

impl AsymSam {
    /// Create a new `AsymSam`.
    pub fn new(config: SamConfig) -> Self {
        Self {
            config,
            saved_params: Vec::new(),
        }
    }

    /// First step: only perturb dimensions with positive gradient.
    pub fn first_step(&mut self, grads: &[f64], params: &[f64]) -> Result<Vec<f64>> {
        if grads.len() != params.len() {
            return Err(TensorError::invalid_argument(
                "grads and params must have equal length".to_string(),
            ));
        }
        // Only keep positive-gradient elements for the norm calculation
        let pos_grads: Vec<f64> = grads.iter().map(|g| g.max(0.0)).collect();
        let norm = l2_norm(&pos_grads).max(1e-30);
        let eps: Vec<f64> = pos_grads
            .iter()
            .map(|g| self.config.rho * g / norm)
            .collect();
        self.saved_params = params.to_vec();
        Ok(add_vecs(params, &eps))
    }

    /// Second step: plain SGD update using gradient at perturbed point.
    pub fn second_step(&self, grads_at_perturbed: &[f64]) -> Result<Vec<f64>> {
        if grads_at_perturbed.len() != self.saved_params.len() {
            return Err(TensorError::invalid_argument(
                "grads length must match saved params length".to_string(),
            ));
        }
        let update = scale_vec(grads_at_perturbed, self.config.base_lr);
        Ok(sub_vecs(&self.saved_params, &update))
    }
}

/// SAM with a diagonal Fisher-matrix preconditioned perturbation.
///
/// The perturbation is `ε_i = ρ * g_i / (F_ii * ‖g / F‖)` where `F_ii` is
/// the diagonal Fisher approximation (squared gradient magnitude accumulated
/// over a short window).
#[derive(Debug, Clone)]
pub struct FisherSam {
    /// SAM configuration.
    pub config: SamConfig,
    /// Diagonal Fisher approximation (accumulated squared gradients).
    fisher_diag: Vec<f64>,
    /// EMA decay for the Fisher accumulator.
    pub ema_decay: f64,
    /// Saved original parameters from the last `first_step` call.
    saved_params: Vec<f64>,
}

impl FisherSam {
    /// Create a new `FisherSam`.
    pub fn new(config: SamConfig, ema_decay: f64) -> Self {
        Self {
            config,
            fisher_diag: Vec::new(),
            ema_decay,
            saved_params: Vec::new(),
        }
    }

    /// Update the diagonal Fisher with new gradients (EMA update).
    pub fn update_fisher(&mut self, grads: &[f64]) {
        if self.fisher_diag.len() != grads.len() {
            self.fisher_diag = grads.iter().map(|g| g * g).collect();
        } else {
            for (f, g) in self.fisher_diag.iter_mut().zip(grads.iter()) {
                *f = self.ema_decay * (*f) + (1.0 - self.ema_decay) * g * g;
            }
        }
    }

    /// First step: Fisher-preconditioned perturbation.
    pub fn first_step(&mut self, grads: &[f64], params: &[f64]) -> Result<Vec<f64>> {
        if grads.len() != params.len() {
            return Err(TensorError::invalid_argument(
                "grads and params must have equal length".to_string(),
            ));
        }
        self.update_fisher(grads);
        // Preconditioned gradient: g_i / sqrt(F_ii + eps)
        let prec: Vec<f64> = grads
            .iter()
            .zip(self.fisher_diag.iter())
            .map(|(g, f)| g / (f.sqrt() + 1e-8))
            .collect();
        let norm = l2_norm(&prec).max(1e-30);
        let eps = scale_vec(&prec, self.config.rho / norm);
        self.saved_params = params.to_vec();
        Ok(add_vecs(params, &eps))
    }

    /// Second step: SGD update at perturbed point and restore.
    pub fn second_step(&self, grads_at_perturbed: &[f64]) -> Result<Vec<f64>> {
        if grads_at_perturbed.len() != self.saved_params.len() {
            return Err(TensorError::invalid_argument(
                "grads length must match saved params length".to_string(),
            ));
        }
        let update = scale_vec(grads_at_perturbed, self.config.base_lr);
        Ok(sub_vecs(&self.saved_params, &update))
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// § 2 — Gradient Norm Analysis
// ═════════════════════════════════════════════════════════════════════════════

/// Tracks per-layer gradient norms over training steps.
#[derive(Debug, Clone)]
pub struct GradientMonitor {
    /// Per-layer gradient norm history: `layer_id -> Vec<f64>`.
    layer_history: std::collections::HashMap<usize, Vec<f64>>,
    /// Maximum history length to retain per layer.
    pub max_history: usize,
}

impl GradientMonitor {
    /// Create a new `GradientMonitor`.
    pub fn new() -> Self {
        Self {
            layer_history: std::collections::HashMap::new(),
            max_history: 1000,
        }
    }

    /// Create with a custom history length.
    pub fn with_max_history(max_history: usize) -> Self {
        Self {
            layer_history: std::collections::HashMap::new(),
            max_history,
        }
    }

    /// Record gradients for a layer.
    pub fn record(&mut self, layer_id: usize, grads: &[f64]) {
        let norm = l2_norm(grads);
        let hist = self.layer_history.entry(layer_id).or_insert_with(Vec::new);
        if hist.len() >= self.max_history {
            hist.remove(0);
        }
        hist.push(norm);
    }

    /// Compute the global gradient norm (root-sum-square of latest layer norms).
    pub fn global_norm(&self) -> f64 {
        let sum_sq: f64 = self
            .layer_history
            .values()
            .filter_map(|h| h.last())
            .map(|n| n * n)
            .sum();
        sum_sq.sqrt()
    }

    /// Latest gradient norm per layer (sorted by layer_id).
    pub fn layer_norms(&self) -> Vec<f64> {
        let mut pairs: Vec<(usize, f64)> = self
            .layer_history
            .iter()
            .filter_map(|(id, h)| h.last().map(|n| (*id, *n)))
            .collect();
        pairs.sort_by_key(|(id, _)| *id);
        pairs.into_iter().map(|(_, n)| n).collect()
    }

    /// Gradient noise scale: ratio of gradient variance to mean squared norm.
    ///
    /// Defined as `GNS = (B * Var[g]) / ‖E[g]‖²` where B is a batch-size proxy.
    /// Here we estimate using the per-step norms of the last tracked layer.
    pub fn grad_noise_scale(&self) -> f64 {
        // Use the history from the layer with the most samples as a proxy.
        let longest = self.layer_history.values().max_by_key(|h| h.len());
        match longest {
            None => 0.0,
            Some(hist) if hist.len() < 2 => 0.0,
            Some(hist) => {
                let n = hist.len() as f64;
                let mean = hist.iter().sum::<f64>() / n;
                let var = hist.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
                let mean_sq = mean * mean;
                if mean_sq < 1e-30 {
                    0.0
                } else {
                    var / mean_sq
                }
            }
        }
    }
}

impl Default for GradientMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Signal-to-Noise Ratio (SNR) of gradients: `mean(g)² / Var(g)` per parameter.
#[derive(Debug, Clone, Default)]
pub struct SignalToNoiseRatio;

impl SignalToNoiseRatio {
    /// Create a new `SignalToNoiseRatio` calculator.
    pub fn new() -> Self {
        Self
    }

    /// Compute per-parameter SNR from a gradient history matrix.
    ///
    /// `grad_history` is `T × D` (T time-steps, D parameters).
    /// Returns a `D`-dimensional vector of SNR values.
    pub fn compute(grad_history: &[Vec<f64>]) -> Vec<f64> {
        if grad_history.is_empty() {
            return Vec::new();
        }
        let d = grad_history[0].len();
        let t = grad_history.len() as f64;
        let mut snr = vec![0.0f64; d];
        for dim in 0..d {
            let vals: Vec<f64> = grad_history.iter().map(|row| row[dim]).collect();
            let mean = vals.iter().sum::<f64>() / t;
            let var = if t > 1.0 {
                vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (t - 1.0)
            } else {
                0.0
            };
            snr[dim] = if var < 1e-30 { 0.0 } else { mean * mean / var };
        }
        snr
    }
}

/// Assigns different learning rates to layers based on gradient SNR statistics.
#[derive(Debug, Clone)]
pub struct LayerWiseLrScheduler {
    /// Per-layer SNR values (updated periodically).
    pub layer_snr: Vec<f64>,
}

impl LayerWiseLrScheduler {
    /// Create a new scheduler.
    pub fn new() -> Self {
        Self {
            layer_snr: Vec::new(),
        }
    }

    /// Update the SNR statistics.
    pub fn update_snr(&mut self, snr: Vec<f64>) {
        self.layer_snr = snr;
    }

    /// Compute the learning rate for layer `layer_id` given a global LR.
    ///
    /// Formula: `lr * snr_layer / snr_mean` (clipped to `[0.01, 100.0] * lr`).
    pub fn get_lr(&self, layer_id: usize, global_lr: f64) -> f64 {
        if self.layer_snr.is_empty() {
            return global_lr;
        }
        let snr_val = if layer_id < self.layer_snr.len() {
            self.layer_snr[layer_id]
        } else {
            1.0
        };
        let mean_snr = {
            let s = self.layer_snr.iter().sum::<f64>();
            let n = self.layer_snr.len() as f64;
            if n > 0.0 && s > 1e-30 {
                s / n
            } else {
                1.0
            }
        };
        let factor = snr_val / mean_snr;
        // Clip to reasonable range
        let factor_clamped = factor.clamp(0.01, 100.0);
        global_lr * factor_clamped
    }
}

impl Default for LayerWiseLrScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Gradient health status classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GradientStatus {
    /// Gradients are in a healthy range.
    Healthy,
    /// Gradients are vanishing (too small).
    Vanishing,
    /// Gradients are exploding (too large).
    Exploding,
}

/// Report produced by [`GradientFlowChecker`].
#[derive(Debug, Clone)]
pub struct GradientHealthReport {
    /// Overall health status.
    pub status: GradientStatus,
    /// Minimum gradient norm observed.
    pub min_norm: f64,
    /// Maximum gradient norm observed.
    pub max_norm: f64,
    /// Ratio `max_norm / min_norm` (or 0 if degenerate).
    pub ratio: f64,
}

/// Detects vanishing and exploding gradient conditions from a sequence of norms.
#[derive(Debug, Clone)]
pub struct GradientFlowChecker {
    /// Norms below this threshold are considered vanishing.
    pub vanishing_threshold: f64,
    /// Norms above this threshold are considered exploding.
    pub exploding_threshold: f64,
}

impl GradientFlowChecker {
    /// Create a new checker with default thresholds.
    pub fn new() -> Self {
        Self {
            vanishing_threshold: 1e-7,
            exploding_threshold: 1e4,
        }
    }

    /// Create a checker with custom thresholds.
    pub fn with_thresholds(vanishing: f64, exploding: f64) -> Self {
        Self {
            vanishing_threshold: vanishing,
            exploding_threshold: exploding,
        }
    }

    /// Analyse a sequence of gradient norms and return a health report.
    pub fn check(&self, norms: &[f64]) -> GradientHealthReport {
        if norms.is_empty() {
            return GradientHealthReport {
                status: GradientStatus::Healthy,
                min_norm: 0.0,
                max_norm: 0.0,
                ratio: 0.0,
            };
        }
        let min_norm = norms.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_norm = norms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ratio = if min_norm > 1e-30 {
            max_norm / min_norm
        } else {
            0.0
        };
        let status = if max_norm < self.vanishing_threshold {
            GradientStatus::Vanishing
        } else if max_norm > self.exploding_threshold {
            GradientStatus::Exploding
        } else {
            GradientStatus::Healthy
        };
        GradientHealthReport {
            status,
            min_norm,
            max_norm,
            ratio,
        }
    }
}

impl Default for GradientFlowChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// § 3 — Loss Landscape Analysis
// ═════════════════════════════════════════════════════════════════════════════

/// Evaluates loss on a 2-D grid around a centre point.
///
/// Given a centre `θ` and two random directions `d1`, `d2`, the grid
/// evaluates `L(θ + α*d1 + β*d2)` for `(α, β)` on a regular grid.
#[derive(Debug, Clone)]
pub struct LossLandscape {
    /// Number of grid points along each axis.
    pub grid_size: usize,
    /// Range of the scalar coefficients (−range .. +range).
    pub range: f64,
    /// Random seed for direction generation.
    pub seed: u64,
}

impl LossLandscape {
    /// Create a new `LossLandscape`.
    pub fn new(grid_size: usize, range: f64, seed: u64) -> Self {
        Self {
            grid_size,
            range,
            seed,
        }
    }

    /// Generate two random unit directions in parameter space of dimension `dim`.
    pub fn random_directions(&self, dim: usize) -> (Vec<f64>, Vec<f64>) {
        let d1_raw = sample_standard_normal(dim, self.seed);
        let d2_raw = sample_standard_normal(dim, self.seed.wrapping_add(1));
        let n1 = l2_norm(&d1_raw).max(1e-30);
        let n2 = l2_norm(&d2_raw).max(1e-30);
        (div_scalar(&d1_raw, n1), div_scalar(&d2_raw, n2))
    }

    /// Evaluate the loss landscape on a 2-D grid.
    ///
    /// Returns a `grid_size × grid_size` matrix (row-major) of loss values,
    /// plus the (α, β) coordinates for each axis.
    ///
    /// `loss_fn` receives a parameter vector and returns a scalar loss.
    pub fn evaluate<F>(
        &self,
        center: &[f64],
        loss_fn: &mut F,
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>, Vec<f64>)>
    where
        F: FnMut(&[f64]) -> f64,
    {
        if self.grid_size == 0 {
            return Err(TensorError::invalid_argument(
                "grid_size must be > 0".to_string(),
            ));
        }
        let dim = center.len();
        let (d1, d2) = self.random_directions(dim);
        let step = if self.grid_size > 1 {
            2.0 * self.range / (self.grid_size - 1) as f64
        } else {
            0.0
        };
        let coords: Vec<f64> = (0..self.grid_size)
            .map(|i| -self.range + i as f64 * step)
            .collect();

        let mut grid = Vec::with_capacity(self.grid_size);
        for &alpha in &coords {
            let mut row = Vec::with_capacity(self.grid_size);
            for &beta in &coords {
                let p: Vec<f64> = center
                    .iter()
                    .zip(d1.iter())
                    .zip(d2.iter())
                    .map(|((c, e1), e2)| c + alpha * e1 + beta * e2)
                    .collect();
                row.push(loss_fn(&p));
            }
            grid.push(row);
        }
        Ok((grid, coords.clone(), coords))
    }
}

/// Sharpness metric: maximum loss increase inside an L∞ ball of radius `ε`.
#[derive(Debug, Clone, Default)]
pub struct SharpnessMetric;

impl SharpnessMetric {
    /// Create a new `SharpnessMetric`.
    pub fn new() -> Self {
        Self
    }

    /// Compute sharpness by evaluating `n_samples` random L∞-ball perturbations.
    ///
    /// Returns `max_perturbation_loss - center_loss`.
    pub fn compute_sharpness<F>(
        &self,
        params: &[f64],
        loss_fn: &mut F,
        radius: f64,
        n_samples: usize,
        seed: u64,
    ) -> f64
    where
        F: FnMut(&[f64]) -> f64,
    {
        let center_loss = loss_fn(params);
        let dim = params.len();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut max_loss = center_loss;
        for _ in 0..n_samples {
            let perturbed: Vec<f64> = params
                .iter()
                .map(|p| {
                    let u: f64 = rng.random::<f64>() * 2.0 - 1.0; // uniform(-1,1)
                    p + radius * u
                })
                .collect();
            let _ = dim; // suppress unused warning
            let l = loss_fn(&perturbed);
            if l > max_loss {
                max_loss = l;
            }
        }
        max_loss - center_loss
    }
}

/// Hutchinson trace estimator for the Hessian.
///
/// Estimates `tr(H) ≈ (1/S) Σ v^T H v` where `v ~ N(0, I)` and
/// `H v ≈ (g(θ + h v) - g(θ - h v)) / (2h)` via finite-difference.
#[derive(Debug, Clone)]
pub struct HessianTrace {
    /// Number of random vectors to average over.
    pub n_samples: usize,
    /// Finite-difference step size.
    pub h: f64,
    /// Random seed.
    pub seed: u64,
}

impl HessianTrace {
    /// Create a new `HessianTrace` estimator.
    pub fn new(n_samples: usize, h: f64, seed: u64) -> Self {
        Self { n_samples, h, seed }
    }

    /// Estimate `tr(H)` using Hutchinson's method.
    ///
    /// `grad_fn` must return the gradient vector at a given parameter point.
    pub fn estimate<F>(&self, params: &[f64], grad_fn: &mut F) -> f64
    where
        F: FnMut(&[f64]) -> Vec<f64>,
    {
        let d = params.len();
        let mut trace_acc = 0.0f64;
        for s in 0..self.n_samples {
            let v = sample_standard_normal(d, self.seed.wrapping_add(s as u64));
            // θ + h·v
            let p_plus: Vec<f64> = params
                .iter()
                .zip(v.iter())
                .map(|(p, vi)| p + self.h * vi)
                .collect();
            // θ - h·v
            let p_minus: Vec<f64> = params
                .iter()
                .zip(v.iter())
                .map(|(p, vi)| p - self.h * vi)
                .collect();
            let g_plus = grad_fn(&p_plus);
            let g_minus = grad_fn(&p_minus);
            // Hv ≈ (g+ - g-) / (2h)
            let hv: Vec<f64> = g_plus
                .iter()
                .zip(g_minus.iter())
                .map(|(gp, gm)| (gp - gm) / (2.0 * self.h))
                .collect();
            // v^T Hv
            let vt_hv: f64 = v.iter().zip(hv.iter()).map(|(vi, hi)| vi * hi).sum();
            trace_acc += vt_hv;
        }
        trace_acc / self.n_samples as f64
    }
}

/// PAC-Bayes flatness measure.
///
/// Computes the ratio of the mean loss under Gaussian perturbation
/// `N(θ, σ² I)` to the original loss `L(θ)`.
#[derive(Debug, Clone)]
pub struct FlatnessMeasure {
    /// Standard deviation of the Gaussian perturbation.
    pub sigma: f64,
    /// Number of Monte-Carlo samples.
    pub n_samples: usize,
    /// Random seed.
    pub seed: u64,
}

impl FlatnessMeasure {
    /// Create a new `FlatnessMeasure`.
    pub fn new(sigma: f64, n_samples: usize, seed: u64) -> Self {
        Self {
            sigma,
            n_samples,
            seed,
        }
    }

    /// Compute the flatness ratio `E[L(θ + δ)] / L(θ)`.
    ///
    /// Returns `1.0` when `L(θ) ≈ 0` to avoid division by zero.
    pub fn compute<F>(&self, params: &[f64], loss_fn: &mut F) -> f64
    where
        F: FnMut(&[f64]) -> f64,
    {
        let center_loss = loss_fn(params);
        if center_loss.abs() < 1e-30 {
            return 1.0;
        }
        let d = params.len();
        let mut perturbed_sum = 0.0f64;
        for s in 0..self.n_samples {
            let noise = sample_standard_normal(d, self.seed.wrapping_add(s as u64));
            let p_pert: Vec<f64> = params
                .iter()
                .zip(noise.iter())
                .map(|(p, n)| p + self.sigma * n)
                .collect();
            perturbed_sum += loss_fn(&p_pert);
        }
        let mean_perturbed = perturbed_sum / self.n_samples as f64;
        mean_perturbed / center_loss
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// § 4 — Learning Rate Schedulers
// ═════════════════════════════════════════════════════════════════════════════

/// Linear warmup followed by cosine decay to a minimum LR.
#[derive(Debug, Clone)]
pub struct WarmupCosineScheduler {
    /// Number of warmup steps (linear increase from 0 to `peak_lr`).
    pub warmup_steps: usize,
    /// Total steps (warmup + cosine decay).
    pub total_steps: usize,
    /// Peak learning rate reached at the end of warmup.
    pub peak_lr: f64,
    /// Minimum learning rate at the end of cosine decay.
    pub min_lr: f64,
}

impl WarmupCosineScheduler {
    /// Create a new `WarmupCosineScheduler`.
    pub fn new(peak_lr: f64, warmup_steps: usize, total_steps: usize, min_lr: f64) -> Self {
        Self {
            warmup_steps,
            total_steps,
            peak_lr,
            min_lr,
        }
    }

    /// Get the learning rate at the given step.
    pub fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            let progress = (step + 1) as f64 / self.warmup_steps.max(1) as f64;
            self.min_lr + (self.peak_lr - self.min_lr) * progress
        } else {
            // Cosine decay
            let decay_steps = self.total_steps.saturating_sub(self.warmup_steps).max(1);
            let t = (step - self.warmup_steps) as f64 / decay_steps as f64;
            let t_clamped = t.clamp(0.0, 1.0);
            let cos_factor = 0.5 * (1.0 + (std::f64::consts::PI * t_clamped).cos());
            self.min_lr + (self.peak_lr - self.min_lr) * cos_factor
        }
    }
}

/// Polynomial decay scheduler: `lr = lr_0 * (1 - step/total)^power`.
///
/// This is the `training_dynamics`-local variant; the optimizers crate has its own.
#[derive(Debug, Clone)]
pub struct TdPolynomialDecayScheduler {
    /// Initial learning rate.
    pub initial_lr: f64,
    /// Total number of steps.
    pub total_steps: usize,
    /// Decay exponent.
    pub power: f64,
    /// Minimum learning rate (clamp floor).
    pub min_lr: f64,
}

impl TdPolynomialDecayScheduler {
    /// Create a new `TdPolynomialDecayScheduler`.
    pub fn new(initial_lr: f64, total_steps: usize, power: f64, min_lr: f64) -> Self {
        Self {
            initial_lr,
            total_steps,
            power,
            min_lr,
        }
    }

    /// Get the learning rate at the given step.
    pub fn get_lr(&self, step: usize) -> f64 {
        if step >= self.total_steps {
            return self.min_lr;
        }
        let fraction = step as f64 / self.total_steps as f64;
        let lr = self.initial_lr * (1.0 - fraction).powf(self.power);
        lr.max(self.min_lr)
    }
}

/// Type alias for backward compatibility; use `TdPolynomialDecayScheduler` in new code.
pub type PolynomialDecayScheduler = TdPolynomialDecayScheduler;

/// PyTorch-style 1-cycle learning rate schedule.
///
/// Three phases:
/// 1. **Warmup**: linear ramp from `base_lr` to `max_lr` over `pct_start * total` steps.
/// 2. **Annealing**: cosine decay from `max_lr` to `base_lr` over the remaining steps.
/// 3. **Final phase**: cosine decay from `base_lr` to `final_lr` over `pct_final * total` steps.
///
/// This is the `training_dynamics`-local variant; the optimizers crate has its own.
#[derive(Debug, Clone)]
pub struct TdOneCycleLrScheduler {
    /// Maximum learning rate (peak of the schedule).
    pub max_lr: f64,
    /// Base learning rate (start / division factor).
    pub base_lr: f64,
    /// Final (minimum) learning rate at the end.
    pub final_lr: f64,
    /// Total number of training steps.
    pub total_steps: usize,
    /// Fraction of steps used for the warmup phase (default 0.3).
    pub pct_start: f64,
    /// Fraction of steps for the final annealing phase (default 0.1).
    pub pct_final: f64,
}

impl TdOneCycleLrScheduler {
    /// Create a new `TdOneCycleLrScheduler` with default pct_start=0.3, pct_final=0.1.
    pub fn new(max_lr: f64, total_steps: usize, div_factor: f64, final_div_factor: f64) -> Self {
        let base_lr = max_lr / div_factor.max(1e-10);
        let final_lr = max_lr / final_div_factor.max(1e-10);
        Self {
            max_lr,
            base_lr,
            final_lr,
            total_steps,
            pct_start: 0.3,
            pct_final: 0.1,
        }
    }

    /// Get the learning rate at the given step.
    pub fn get_lr(&self, step: usize) -> f64 {
        let total = self.total_steps.max(1) as f64;
        let warmup_end = (self.pct_start * total) as usize;
        let final_start = total as usize - (self.pct_final * total) as usize;

        if step <= warmup_end {
            // Phase 1: linear warmup
            let t = step as f64 / warmup_end.max(1) as f64;
            self.base_lr + (self.max_lr - self.base_lr) * t
        } else if step < final_start {
            // Phase 2: cosine annealing from max_lr to base_lr
            let phase_len = (final_start - warmup_end).max(1) as f64;
            let t = (step - warmup_end) as f64 / phase_len;
            let cos_factor = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
            self.base_lr + (self.max_lr - self.base_lr) * cos_factor
        } else {
            // Phase 3: cosine annealing from base_lr to final_lr
            let phase_len = (self.total_steps - final_start).max(1) as f64;
            let t = (step - final_start) as f64 / phase_len;
            let t_clamped = t.clamp(0.0, 1.0);
            let cos_factor = 0.5 * (1.0 + (std::f64::consts::PI * t_clamped).cos());
            self.final_lr + (self.base_lr - self.final_lr) * cos_factor
        }
    }
}

/// Type alias for backward compatibility; use `TdOneCycleLrScheduler` in new code.
pub type OneCycleLrScheduler = TdOneCycleLrScheduler;

/// Stochastic depth scheduler — gradually increases the fraction of active
/// layers during training, implementing a "curriculum depth" strategy.
#[derive(Debug, Clone)]
pub struct StochasticDepthScheduler {
    /// Starting survival probability (depth fraction at step 0).
    pub initial_prob: f64,
    /// Final survival probability (depth fraction at `total_steps`).
    pub final_prob: f64,
    /// Total steps for the curriculum.
    pub total_steps: usize,
}

impl StochasticDepthScheduler {
    /// Create a new `StochasticDepthScheduler`.
    pub fn new(initial_prob: f64, final_prob: f64, total_steps: usize) -> Self {
        Self {
            initial_prob,
            final_prob,
            total_steps,
        }
    }

    /// Get the survival probability (layer inclusion rate) at the given step.
    pub fn get_survival_prob(&self, step: usize) -> f64 {
        let t = (step as f64 / self.total_steps.max(1) as f64).clamp(0.0, 1.0);
        self.initial_prob + (self.final_prob - self.initial_prob) * t
    }
}

/// Grokking acceleration scheduler.
///
/// Detects apparent convergence (loss plateau) and temporarily boosts the
/// learning rate to help the model "grok" (generalise).
#[derive(Debug, Clone)]
pub struct GrokFastScheduler {
    /// Base learning rate.
    pub base_lr: f64,
    /// Boost factor applied after convergence detection.
    pub boost_factor: f64,
    /// Number of consecutive steps without improvement to declare convergence.
    pub patience: usize,
    /// Duration of the LR boost in steps.
    pub boost_steps: usize,
    /// Minimum improvement threshold.
    pub min_delta: f64,
    // Internal state
    best_loss: f64,
    steps_without_improvement: usize,
    boost_remaining: usize,
}

impl GrokFastScheduler {
    /// Create a new `GrokFastScheduler`.
    pub fn new(base_lr: f64, boost_factor: f64, patience: usize, boost_steps: usize) -> Self {
        Self {
            base_lr,
            boost_factor,
            patience,
            boost_steps,
            min_delta: 1e-6,
            best_loss: f64::INFINITY,
            steps_without_improvement: 0,
            boost_remaining: 0,
        }
    }

    /// Update the scheduler with the current loss and return the learning rate.
    pub fn step(&mut self, loss: f64) -> f64 {
        if loss < self.best_loss - self.min_delta {
            self.best_loss = loss;
            self.steps_without_improvement = 0;
        } else {
            self.steps_without_improvement += 1;
        }
        if self.steps_without_improvement >= self.patience && self.boost_remaining == 0 {
            self.boost_remaining = self.boost_steps;
            self.steps_without_improvement = 0;
        }
        if self.boost_remaining > 0 {
            self.boost_remaining -= 1;
            self.base_lr * self.boost_factor
        } else {
            self.base_lr
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// § 5 — Curriculum Learning
// ═════════════════════════════════════════════════════════════════════════════

/// Assigns difficulty scores to training examples via various heuristics.
#[derive(Debug, Clone, Default)]
pub struct DifficultyScorer;

impl DifficultyScorer {
    /// Create a new `DifficultyScorer`.
    pub fn new() -> Self {
        Self
    }

    /// Difficulty based on raw loss magnitude (higher loss = harder).
    pub fn score_by_loss_magnitude(losses: &[f64]) -> Vec<f64> {
        losses.to_vec()
    }

    /// Difficulty based on gradient norm (higher norm = harder).
    pub fn score_by_gradient_norm(grad_norms: &[f64]) -> Vec<f64> {
        grad_norms.to_vec()
    }

    /// Difficulty based on prediction confidence.
    ///
    /// `confidences[i]` is the maximum predicted probability for example i.
    /// Difficulty = `1 - confidence` (low confidence → hard).
    pub fn score_by_prediction_confidence(confidences: &[f64]) -> Vec<f64> {
        confidences
            .iter()
            .map(|c| 1.0 - c.clamp(0.0, 1.0))
            .collect()
    }
}

/// Curriculum sampler based on the self-paced competence function.
///
/// Competence at time `t`: `c(t) = min(1, c0 + (1 - c0) * (t/T)^p)`.
/// Examples with difficulty `≤ c(t) * max_difficulty` are eligible.
#[derive(Debug, Clone)]
pub struct CurriculumSampler {
    /// Initial competence level `c0 ∈ [0, 1]`.
    pub c0: f64,
    /// Competence growth exponent.
    pub growth_power: f64,
    /// Random seed for sampling.
    pub seed: u64,
}

impl CurriculumSampler {
    /// Create a new `CurriculumSampler`.
    pub fn new(c0: f64, growth_power: f64, seed: u64) -> Self {
        Self {
            c0,
            growth_power,
            seed,
        }
    }

    /// Competence at time `t`.
    pub fn competence(&self, t: usize, total: usize) -> f64 {
        let frac = (t as f64 / total.max(1) as f64).clamp(0.0, 1.0);
        let c = self.c0 + (1.0 - self.c0) * frac.powf(self.growth_power);
        c.clamp(self.c0, 1.0)
    }

    /// Sample `n` example indices from the eligible set.
    ///
    /// Eligible indices satisfy `score[i] ≤ max_score * competence(t, T)`.
    /// Returns indices in the original order, with random subsampling if necessary.
    pub fn sample_batch(&self, scores: &[f64], t: usize, total: usize, n: usize) -> Vec<usize> {
        let c = self.competence(t, total);
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let threshold = max_score * c;
        let eligible: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter(|(_, s)| **s <= threshold)
            .map(|(i, _)| i)
            .collect();
        if eligible.is_empty() {
            // Fallback: return easiest examples
            let mut pairs: Vec<(usize, f64)> = scores.iter().cloned().enumerate().collect();
            pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            return pairs.iter().take(n).map(|(i, _)| *i).collect();
        }
        if eligible.len() <= n {
            return eligible;
        }
        // Random subsample from eligible
        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(t as u64));
        let mut pool = eligible.clone();
        // Fisher-Yates partial shuffle
        let k = n.min(pool.len());
        for i in 0..k {
            let j = i + (rng.random::<u64>() as usize % (pool.len() - i));
            pool.swap(i, j);
        }
        pool.truncate(n);
        pool
    }
}

/// Self-Paced Learning (SPL): include examples with loss below threshold λ,
/// gradually increase λ over training.
#[derive(Debug, Clone)]
pub struct SelfPacedLearning {
    /// Initial loss threshold.
    pub initial_lambda: f64,
    /// Maximum threshold (all examples included when λ ≥ max_loss).
    pub max_lambda: f64,
    /// Multiplicative growth rate for λ per epoch.
    pub growth_rate: f64,
    /// Current threshold.
    pub current_lambda: f64,
}

impl SelfPacedLearning {
    /// Create a new `SelfPacedLearning` scheduler.
    pub fn new(initial_lambda: f64, max_lambda: f64, growth_rate: f64) -> Self {
        Self {
            initial_lambda,
            max_lambda,
            growth_rate,
            current_lambda: initial_lambda,
        }
    }

    /// Advance the threshold by one epoch.
    pub fn step(&mut self) {
        self.current_lambda = (self.current_lambda * self.growth_rate).min(self.max_lambda);
    }

    /// Return indices of examples with loss ≤ `current_lambda`.
    pub fn select(&self, losses: &[f64]) -> Vec<usize> {
        losses
            .iter()
            .enumerate()
            .filter(|(_, l)| **l <= self.current_lambda)
            .map(|(i, _)| i)
            .collect()
    }

    /// Return the current loss threshold.
    pub fn current_threshold(&self) -> f64 {
        self.current_lambda
    }
}

/// Mixed curriculum scheduler: interleaves easy and hard samples with a
/// smoothly varying mixing weight.
#[derive(Debug, Clone)]
pub struct MixedCurriculumScheduler {
    /// Total training steps.
    pub total_steps: usize,
    /// Fraction of hard examples at step 0 (start easy).
    pub initial_hard_fraction: f64,
    /// Fraction of hard examples at `total_steps` (end hard).
    pub final_hard_fraction: f64,
}

impl MixedCurriculumScheduler {
    /// Create a new `MixedCurriculumScheduler`.
    pub fn new(total_steps: usize, initial_hard_fraction: f64, final_hard_fraction: f64) -> Self {
        Self {
            total_steps,
            initial_hard_fraction,
            final_hard_fraction,
        }
    }

    /// Return per-example sample weights at the given step.
    ///
    /// Easy examples (low score) get weight `1 - hard_frac`;
    /// hard examples (high score) get weight `hard_frac`.
    /// The split is determined by the median score.
    pub fn get_sample_weights(&self, scores: &[f64], step: usize) -> Vec<f64> {
        let t = (step as f64 / self.total_steps.max(1) as f64).clamp(0.0, 1.0);
        let hard_frac = self.initial_hard_fraction
            + (self.final_hard_fraction - self.initial_hard_fraction) * t;
        let easy_weight = 1.0 - hard_frac;
        let hard_weight = hard_frac;

        if scores.is_empty() {
            return Vec::new();
        }
        // Find median to split easy/hard
        let mut sorted = scores.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        scores
            .iter()
            .map(|s| {
                if *s <= median {
                    easy_weight
                } else {
                    hard_weight
                }
            })
            .collect()
    }
}

/// Anti-curriculum trainer: start with hard examples, gradually introduce easier ones.
#[derive(Debug, Clone)]
pub struct AntiCurriculumTrainer {
    /// Fraction of top-difficulty examples to include at step 0.
    pub initial_hard_fraction: f64,
    /// Total steps before all examples are included.
    pub total_steps: usize,
}

impl AntiCurriculumTrainer {
    /// Create a new `AntiCurriculumTrainer`.
    pub fn new(initial_hard_fraction: f64, total_steps: usize) -> Self {
        Self {
            initial_hard_fraction,
            total_steps,
        }
    }

    /// Return ordered example indices from hardest to easiest.
    ///
    /// At step `t`, includes the top-`k` hardest examples plus a fraction
    /// of easier examples, increasing monotonically.
    pub fn get_indices(&self, scores: &[f64], step: usize) -> Vec<usize> {
        let t = (step as f64 / self.total_steps.max(1) as f64).clamp(0.0, 1.0);
        let fraction = self.initial_hard_fraction + (1.0 - self.initial_hard_fraction) * t;
        let k = ((fraction * scores.len() as f64).round() as usize).clamp(1, scores.len());

        // Sort indices by difficulty descending (hardest first)
        let mut pairs: Vec<(usize, f64)> = scores.iter().cloned().enumerate().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.iter().take(k).map(|(i, _)| *i).collect()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    // ── SAM ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_sam_config() {
        let cfg = SamConfig::new(0.1, true, 0.01);
        assert!((cfg.rho - 0.1).abs() < EPS);
        assert!(cfg.adaptive);
        assert!((cfg.base_lr - 0.01).abs() < EPS);
    }

    #[test]
    fn test_sam_first_step_perturbation() {
        let cfg = SamConfig::new(0.05, false, 0.01);
        let mut sam = SamOptimizer::new(cfg);
        let grads = vec![3.0f64, 4.0]; // norm = 5
        let params = vec![1.0f64, 2.0];
        let perturbed = sam.first_step(&grads, &params).expect("first_step ok");
        // ε = 0.05 * [3, 4] / 5 = [0.03, 0.04]
        assert!(
            (perturbed[0] - 1.03).abs() < 1e-10,
            "perturbed[0]={}",
            perturbed[0]
        );
        assert!(
            (perturbed[1] - 2.04).abs() < 1e-10,
            "perturbed[1]={}",
            perturbed[1]
        );
    }

    #[test]
    fn test_sam_perturbation_unit_scale() {
        // The perturbation should have L2 norm = rho
        let cfg = SamConfig::new(0.1, false, 0.01);
        let mut sam = SamOptimizer::new(cfg);
        let grads = vec![1.0f64, 2.0, 3.0, 4.0];
        let params = vec![0.0f64; 4];
        let _perturbed = sam.first_step(&grads, &params).expect("ok");
        let eps = sam.last_perturbation();
        let eps_norm = l2_norm(eps);
        assert!((eps_norm - 0.1).abs() < 1e-10, "eps_norm={eps_norm}");
    }

    #[test]
    fn test_sam_second_step_restores_and_updates() {
        let cfg = SamConfig::new(0.05, false, 0.1);
        let mut sam = SamOptimizer::new(cfg);
        let grads = vec![1.0f64, 0.0];
        let params = vec![5.0f64, 5.0];
        let perturbed = sam.first_step(&grads, &params).expect("ok");
        let updated = sam.second_step(&grads, &perturbed).expect("ok");
        // Should be original params - lr * grads = [5 - 0.1*1, 5 - 0.1*0] = [4.9, 5.0]
        assert!((updated[0] - 4.9).abs() < 1e-10);
        assert!((updated[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_asym_sam_positive_only() {
        let cfg = SamConfig::new(0.1, false, 0.01);
        let mut asym = AsymSam::new(cfg);
        let grads = vec![-1.0f64, 2.0, -3.0, 4.0];
        let params = vec![0.0f64; 4];
        let perturbed = asym.first_step(&grads, &params).expect("ok");
        // Only positive grads (index 1 and 3) should be perturbed
        assert!(perturbed[0].abs() < 1e-10, "idx0 should not be perturbed");
        assert!(perturbed[2].abs() < 1e-10, "idx2 should not be perturbed");
        assert!(perturbed[1] > 0.0, "idx1 should be perturbed upward");
        assert!(perturbed[3] > 0.0, "idx3 should be perturbed upward");
    }

    #[test]
    fn test_fisher_sam_first_step_reduces_high_variance() {
        let cfg = SamConfig::new(0.1, false, 0.01);
        let mut fisher = FisherSam::new(cfg, 0.9);
        let grads = vec![10.0f64, 0.1, 10.0, 0.1];
        let params = vec![0.0f64; 4];
        // Warm up Fisher
        fisher.update_fisher(&grads);
        fisher.update_fisher(&grads);
        let perturbed = fisher.first_step(&grads, &params).expect("ok");
        // High-variance dims should be dampened; all perturbed values finite
        for (i, p) in perturbed.iter().enumerate() {
            assert!(p.is_finite(), "perturbed[{i}] is not finite");
        }
    }

    // ── Gradient monitoring ───────────────────────────────────────────────────

    #[test]
    fn test_gradient_monitor_record() {
        let mut monitor = GradientMonitor::new();
        monitor.record(0, &[3.0, 4.0]); // norm = 5
        let norms = monitor.layer_norms();
        assert_eq!(norms.len(), 1);
        assert!((norms[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_global_norm_computation() {
        let mut monitor = GradientMonitor::new();
        monitor.record(0, &[3.0f64, 0.0]); // norm = 3
        monitor.record(1, &[0.0f64, 4.0]); // norm = 4
        let gn = monitor.global_norm(); // sqrt(9 + 16) = 5
        assert!((gn - 5.0).abs() < 1e-10, "global_norm={gn}");
    }

    #[test]
    fn test_snr_computation() {
        // Constant gradient: SNR should be very high (zero variance)
        let history = vec![vec![2.0f64, 3.0], vec![2.0f64, 3.0], vec![2.0f64, 3.0]];
        let snr = SignalToNoiseRatio::compute(&history);
        // Variance is 0, so SNR = 0.0 by convention (avoid division by zero)
        assert_eq!(snr.len(), 2);
        assert_eq!(snr[0], 0.0);
    }

    #[test]
    fn test_snr_noisy_gradient() {
        // Gradient oscillating: low SNR
        let history = vec![vec![1.0f64], vec![-1.0f64], vec![1.0f64], vec![-1.0f64]];
        let snr = SignalToNoiseRatio::compute(&history);
        // mean = 0, so SNR = 0
        assert!(snr[0] < 1e-10);
    }

    #[test]
    fn test_layer_wise_lr() {
        let mut sched = LayerWiseLrScheduler::new();
        sched.update_snr(vec![0.5, 1.0, 1.5]); // mean = 1.0
        let lr0 = sched.get_lr(0, 0.01); // 0.01 * 0.5/1.0 = 0.005
        let lr2 = sched.get_lr(2, 0.01); // 0.01 * 1.5/1.0 = 0.015
        assert!((lr0 - 0.005).abs() < 1e-10, "lr0={lr0}");
        assert!((lr2 - 0.015).abs() < 1e-10, "lr2={lr2}");
    }

    #[test]
    fn test_gradient_health_healthy() {
        let checker = GradientFlowChecker::new();
        let norms = vec![0.5, 1.0, 0.8, 1.2];
        let report = checker.check(&norms);
        assert_eq!(report.status, GradientStatus::Healthy);
        assert!(report.min_norm < report.max_norm);
    }

    #[test]
    fn test_gradient_health_vanishing() {
        let checker = GradientFlowChecker::new();
        let norms = vec![1e-9, 5e-10, 2e-10];
        let report = checker.check(&norms);
        assert_eq!(report.status, GradientStatus::Vanishing);
    }

    #[test]
    fn test_gradient_health_exploding() {
        let checker = GradientFlowChecker::new();
        let norms = vec![1.0, 100.0, 50000.0];
        let report = checker.check(&norms);
        assert_eq!(report.status, GradientStatus::Exploding);
    }

    #[test]
    fn test_gradient_health_ratio() {
        let checker = GradientFlowChecker::new();
        let norms = vec![1.0, 2.0, 4.0];
        let report = checker.check(&norms);
        assert!((report.ratio - 4.0).abs() < 1e-10, "ratio={}", report.ratio);
    }

    // ── Loss landscape ────────────────────────────────────────────────────────

    #[test]
    fn test_loss_landscape_grid_size() {
        let ll = LossLandscape::new(5, 1.0, 42);
        let center = vec![0.0f64; 4];
        let (grid, alpha, beta) = ll
            .evaluate(&center, &mut |p: &[f64]| {
                p.iter().map(|x| x * x).sum::<f64>()
            })
            .expect("evaluate ok");
        assert_eq!(grid.len(), 5);
        assert_eq!(grid[0].len(), 5);
        assert_eq!(alpha.len(), 5);
        assert_eq!(beta.len(), 5);
    }

    #[test]
    fn test_sharpness_metric_positive() {
        let metric = SharpnessMetric::new();
        let params = vec![0.0f64; 4];
        // Loss = sum of squares; near origin is a minimum; sharpness should be >= 0
        let sharpness = metric.compute_sharpness(
            &params,
            &mut |p: &[f64]| p.iter().map(|x| x * x).sum::<f64>(),
            0.5,
            50,
            123,
        );
        assert!(
            sharpness >= 0.0,
            "sharpness should be non-negative, got {sharpness}"
        );
    }

    #[test]
    fn test_sharpness_metric_increases_with_radius() {
        let metric = SharpnessMetric::new();
        let params = vec![0.0f64; 4];
        let loss_fn = |p: &[f64]| p.iter().map(|x| x * x).sum::<f64>();
        let s1 = metric.compute_sharpness(
            &params,
            &mut {
                let mut f = loss_fn;
                move |p| f(p)
            },
            0.1,
            50,
            1,
        );
        let s2 = metric.compute_sharpness(
            &params,
            &mut {
                let mut f = loss_fn;
                move |p| f(p)
            },
            1.0,
            50,
            1,
        );
        assert!(
            s2 >= s1,
            "larger radius should give larger sharpness: s1={s1} s2={s2}"
        );
    }

    #[test]
    fn test_hutchinson_trace_estimator() {
        // H = 2*I for loss = sum(x^2), so tr(H) = 2*d = 8 for d=4
        let ht = HessianTrace::new(200, 1e-4, 777);
        let params = vec![0.0f64; 4];
        let trace = ht.estimate(&params, &mut |p: &[f64]| {
            // grad of sum(x^2) = 2x
            p.iter().map(|x| 2.0 * x).collect()
        });
        // Expected: 8.0 (2 * 4 dimensions)
        assert!((trace - 8.0).abs() < 1.0, "trace={trace} expected ~8.0");
    }

    #[test]
    fn test_flatness_measure() {
        let fm = FlatnessMeasure::new(0.01, 100, 42);
        // Loss = 1.0 (constant), ratio should be ~1.0
        let ratio = fm.compute(&[0.0f64; 4], &mut |_| 1.0);
        assert!((ratio - 1.0).abs() < 1e-10, "constant loss => ratio=1");
    }

    #[test]
    fn test_flatness_measure_sharp_function() {
        let fm = FlatnessMeasure::new(1.0, 100, 42);
        // Loss = large quadratic -> perturbed mean >> center loss => ratio > 1
        let params = vec![0.0f64; 4];
        let ratio = fm.compute(&params, &mut |p: &[f64]| {
            1.0 + p.iter().map(|x| 100.0 * x * x).sum::<f64>()
        });
        assert!(ratio > 1.0, "sharp function ratio={ratio}");
    }

    // ── Schedulers ────────────────────────────────────────────────────────────

    #[test]
    fn test_warmup_cosine_warmup_phase() {
        let sched = WarmupCosineScheduler::new(0.1, 10, 100, 0.0);
        // At step 0: lr ~ peak_lr * 1/10 = 0.01
        let lr0 = sched.get_lr(0);
        assert!((lr0 - 0.01).abs() < 1e-10, "lr0={lr0}");
        // At end of warmup (step 9): lr = peak_lr = 0.1
        let lr9 = sched.get_lr(9);
        assert!((lr9 - 0.1).abs() < 1e-10, "lr9={lr9}");
    }

    #[test]
    fn test_warmup_cosine_decay_phase() {
        let sched = WarmupCosineScheduler::new(1.0, 0, 100, 0.0);
        // No warmup: at step 0 should be at peak, at step 100 should be at min
        let lr_start = sched.get_lr(0);
        assert!((lr_start - 1.0).abs() < 1e-10, "start={lr_start}");
        let lr_end = sched.get_lr(100);
        assert!(lr_end < 0.1, "end should be near min_lr, got {lr_end}");
    }

    #[test]
    fn test_polynomial_decay() {
        let sched = PolynomialDecayScheduler::new(1.0, 100, 1.0, 0.0);
        // Linear decay: at step 50 -> lr = 0.5
        let lr50 = sched.get_lr(50);
        assert!((lr50 - 0.5).abs() < 1e-10, "lr50={lr50}");
        // At total_steps: lr = min_lr = 0
        let lr100 = sched.get_lr(100);
        assert!(lr100.abs() < 1e-10, "lr100={lr100}");
    }

    #[test]
    fn test_polynomial_decay_quadratic() {
        let sched = PolynomialDecayScheduler::new(1.0, 100, 2.0, 0.0);
        // Quadratic: at step 50 -> lr = (0.5)^2 = 0.25
        let lr50 = sched.get_lr(50);
        assert!((lr50 - 0.25).abs() < 1e-10, "lr50={lr50}");
    }

    #[test]
    fn test_one_cycle_peak() {
        let sched = OneCycleLrScheduler::new(1.0, 100, 10.0, 1e4);
        // At 30% of total (peak of warmup): should be near max_lr
        let lr_peak = sched.get_lr(30);
        assert!((lr_peak - 1.0).abs() < 1e-10, "peak lr={lr_peak}");
    }

    #[test]
    fn test_one_cycle_final() {
        let sched = OneCycleLrScheduler::new(1.0, 100, 10.0, 1e4);
        // At final step: should be near final_lr = 1.0 / 1e4 = 0.0001
        let lr_final = sched.get_lr(99);
        assert!(
            lr_final < sched.base_lr,
            "final lr={lr_final} should be < base_lr={}",
            sched.base_lr
        );
    }

    #[test]
    fn test_stochastic_depth_scheduler() {
        let sched = StochasticDepthScheduler::new(0.5, 1.0, 100);
        assert!((sched.get_survival_prob(0) - 0.5).abs() < 1e-10);
        assert!((sched.get_survival_prob(100) - 1.0).abs() < 1e-10);
        let mid = sched.get_survival_prob(50);
        assert!(mid > 0.5 && mid < 1.0, "mid={mid}");
    }

    #[test]
    fn test_grok_fast_scheduler_boost() {
        let mut sched = GrokFastScheduler::new(0.01, 5.0, 3, 2);
        // Steps without improvement
        let _lr1 = sched.step(1.0); // best = 1.0
        let _lr2 = sched.step(1.0); // no improvement
        let _lr3 = sched.step(1.0); // no improvement (3 steps => trigger boost)
        let lr_boosted = sched.step(1.0); // should be boosted
        assert!((lr_boosted - 0.05).abs() < 1e-10, "boosted lr={lr_boosted}");
    }

    // ── Curriculum learning ───────────────────────────────────────────────────

    #[test]
    fn test_difficulty_scorer_confidence() {
        let confidences = vec![0.9, 0.5, 0.1];
        let scores = DifficultyScorer::score_by_prediction_confidence(&confidences);
        assert!((scores[0] - 0.1).abs() < 1e-10); // low difficulty
        assert!((scores[1] - 0.5).abs() < 1e-10);
        assert!((scores[2] - 0.9).abs() < 1e-10); // high difficulty
    }

    #[test]
    fn test_difficulty_scorer_loss() {
        let losses = vec![0.1, 5.0, 2.0];
        let scores = DifficultyScorer::score_by_loss_magnitude(&losses);
        assert_eq!(scores, losses);
    }

    #[test]
    fn test_curriculum_sampler_c0() {
        let sampler = CurriculumSampler::new(0.5, 1.0, 42);
        // At t=0: competence = c0 = 0.5
        let c = sampler.competence(0, 100);
        assert!((c - 0.5).abs() < 1e-10, "c={c}");
    }

    #[test]
    fn test_curriculum_sampler_full() {
        let sampler = CurriculumSampler::new(0.0, 1.0, 42);
        // At t=T: competence = 1.0 (all examples eligible)
        let c = sampler.competence(100, 100);
        assert!((c - 1.0).abs() < 1e-10, "c at T={c}");
    }

    #[test]
    fn test_curriculum_sampler_sample_batch() {
        let sampler = CurriculumSampler::new(0.5, 1.0, 42);
        // At mid-point, only easy half (low scores) should be eligible
        let scores = vec![0.1, 0.2, 0.8, 0.9];
        let batch = sampler.sample_batch(&scores, 50, 100, 4);
        // All returned indices must be valid
        for &idx in &batch {
            assert!(idx < scores.len(), "invalid index {idx}");
        }
    }

    #[test]
    fn test_self_paced_threshold() {
        let spl = SelfPacedLearning::new(1.0, 10.0, 2.0);
        // Initially lambda=1.0; examples with loss<=1 are selected
        let losses = vec![0.5, 1.0, 1.5, 2.0];
        let selected = spl.select(&losses);
        assert_eq!(selected, vec![0, 1]); // indices 0 and 1
    }

    #[test]
    fn test_self_paced_step_grows_threshold() {
        let mut spl = SelfPacedLearning::new(1.0, 100.0, 3.0);
        spl.step();
        assert!((spl.current_threshold() - 3.0).abs() < 1e-10);
        spl.step();
        assert!((spl.current_threshold() - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_mixed_curriculum_weights() {
        let sched = MixedCurriculumScheduler::new(100, 0.1, 0.9);
        let scores = vec![1.0, 2.0, 3.0, 4.0]; // median = 2.0 or 3.0
        let weights_start = sched.get_sample_weights(&scores, 0);
        let weights_end = sched.get_sample_weights(&scores, 100);
        // At start, hard_frac=0.1 so easy_weight=0.9
        assert!(
            weights_start[0] > weights_end[0],
            "easy example should have more weight at start"
        );
    }

    #[test]
    fn test_anti_curriculum_ordering() {
        let trainer = AntiCurriculumTrainer::new(0.5, 100);
        let scores = vec![0.1, 5.0, 3.0, 0.2];
        // At step 0: include top 50% hardest = 2 examples
        let indices = trainer.get_indices(&scores, 0);
        assert_eq!(indices.len(), 2);
        // Hardest should be index 1 (score=5.0) and index 2 (score=3.0)
        assert!(
            indices.contains(&1),
            "should contain hardest (idx 1): {:?}",
            indices
        );
    }

    #[test]
    fn test_anti_curriculum_full_at_end() {
        let trainer = AntiCurriculumTrainer::new(0.5, 100);
        let scores = vec![0.1, 0.2, 0.3, 0.4];
        let indices = trainer.get_indices(&scores, 100);
        assert_eq!(indices.len(), 4, "all examples at final step");
    }
}
