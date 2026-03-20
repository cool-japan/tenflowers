//! # Simulation ML
//!
//! Machine learning methods for scientific simulation and surrogate modeling.
//!
//! This module provides:
//! - **NeuralSurrogate** — neural network surrogate for expensive simulations
//! - **EnsembleSurrogate** — uncertainty-aware surrogate via deep ensemble
//! - **PhysicsInformedSurrogate** — surrogate with linear physics residual constraints
//! - **GaussianProcessSurrogate** — GP surrogate with RBF kernel (Cholesky solve)
//! - **LatinHypercubeSampler** — space-filling design for simulation experiments
//! - **AdaptiveSampler** — sequential experimental design via Expected Improvement
//! - **DifferentiableSimulator** — spring-mass system with Verlet integration
//! - **TurbulenceModel** — Smagorinsky and ML-augmented RANS closure
//! - **SimulationDataAugmenter** — noise injection, symmetry, interpolation
//! - **SimulationMetrics** — Q², relative L2, coverage probability, etc.

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by simulation-ML routines.
#[derive(Debug, Clone)]
pub enum SimError {
    /// Dimension mismatch between operands.
    DimensionMismatch { expected: usize, got: usize },
    /// Model has no training data yet.
    NoTrainingData,
    /// Cholesky decomposition failed (matrix not positive-definite).
    CholeskyFailed,
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// General numerical error.
    NumericalError(String),
}

impl std::fmt::Display for SimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            SimError::NoTrainingData => write!(f, "surrogate has no training data"),
            SimError::CholeskyFailed => {
                write!(
                    f,
                    "Cholesky decomposition failed: matrix not positive-definite"
                )
            }
            SimError::InvalidConfig(s) => write!(f, "invalid config: {s}"),
            SimError::NumericalError(s) => write!(f, "numerical error: {s}"),
        }
    }
}

impl std::error::Error for SimError {}

type Result<T> = std::result::Result<T, SimError>;

// ---------------------------------------------------------------------------
// 1. NeuralSurrogate
// ---------------------------------------------------------------------------

/// Activation function choices for surrogate networks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurrogateActivation {
    ReLU,
    Tanh,
    GELU,
    Swish,
}

impl SurrogateActivation {
    /// Apply the activation element-wise in place.
    fn apply(self, x: f32) -> f32 {
        match self {
            SurrogateActivation::ReLU => x.max(0.0),
            SurrogateActivation::Tanh => x.tanh(),
            SurrogateActivation::GELU => {
                // Approximate GELU: x * Φ(x) ≈ 0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))
                let inner = 0.797_884_6_f32 * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            }
            SurrogateActivation::Swish => x * (1.0 / (1.0 + (-x).exp())),
        }
    }
}

/// Configuration for a neural surrogate model.
#[derive(Debug, Clone)]
pub struct SurrogateConfig {
    pub input_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub output_dim: usize,
    pub activation: SurrogateActivation,
}

/// A dense (fully-connected) layer stored as (weight_matrix_row_major, bias).
/// Weight matrix shape: [out_dim × in_dim] (row-major).
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Vec<f32>, // [out_dim * in_dim]
    pub bias: Vec<f32>,    // [out_dim]
    pub in_dim: usize,
    pub out_dim: usize,
}

impl DenseLayer {
    /// Initialise with Xavier-uniform weights.
    fn new(in_dim: usize, out_dim: usize, rng: &mut StdRng) -> Self {
        let limit = (6.0_f32 / (in_dim + out_dim) as f32).sqrt();
        let weights: Vec<f32> = (0..in_dim * out_dim)
            .map(|_| {
                let u: f32 = rng.random::<f32>();
                -limit + 2.0 * limit * u
            })
            .collect();
        let bias = vec![0.0_f32; out_dim];
        DenseLayer {
            weights,
            bias,
            in_dim,
            out_dim,
        }
    }

    /// Forward pass: y = W x + b.
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out = self.bias.clone();
        for o in 0..self.out_dim {
            for i in 0..self.in_dim {
                out[o] += self.weights[o * self.in_dim + i] * x[i];
            }
        }
        out
    }
}

/// Neural network surrogate for expensive simulations.
///
/// Architecture: input → [hidden layers with activation] → output (linear).
/// Training uses finite-difference gradient descent (MSE loss).
#[derive(Debug, Clone)]
pub struct NeuralSurrogate {
    pub layers: Vec<DenseLayer>,
    pub activation: SurrogateActivation,
    pub config: SurrogateConfig,
}

impl NeuralSurrogate {
    /// Build a new surrogate from a `SurrogateConfig`, initialised with seed.
    pub fn new(config: SurrogateConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut dims: Vec<usize> = vec![config.input_dim];
        dims.extend_from_slice(&config.hidden_dims);
        dims.push(config.output_dim);

        let layers: Vec<DenseLayer> = dims
            .windows(2)
            .map(|w| DenseLayer::new(w[0], w[1], &mut rng))
            .collect();

        NeuralSurrogate {
            layers,
            activation: config.activation,
            config,
        }
    }

    /// Forward pass through all layers.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let n_layers = self.layers.len();
        let mut h: Vec<f32> = x.to_vec();
        for (idx, layer) in self.layers.iter().enumerate() {
            let pre = layer.forward(&h);
            // Apply activation to all layers except the last (linear output).
            h = if idx < n_layers - 1 {
                pre.iter().map(|&v| self.activation.apply(v)).collect()
            } else {
                pre
            };
        }
        h
    }

    /// MSE loss between prediction and target.
    fn mse_loss(&self, x: &[f32], y: &[f32]) -> f32 {
        let pred = self.forward(x);
        pred.iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t) * (p - t))
            .sum::<f32>()
            / y.len() as f32
    }

    /// One step of finite-difference gradient descent on MSE loss.
    /// Returns the loss before the step.
    pub fn train_step(&mut self, x: &[f32], y: &[f32], lr: f32) -> f32 {
        let eps = 1e-4_f32;
        let base_loss = self.mse_loss(x, y);

        // Iterate over every parameter in every layer and compute FD gradient.
        for layer_idx in 0..self.layers.len() {
            // Weights
            let n_w = self.layers[layer_idx].weights.len();
            let mut w_grads = vec![0.0_f32; n_w];
            for wi in 0..n_w {
                let orig = self.layers[layer_idx].weights[wi];
                self.layers[layer_idx].weights[wi] = orig + eps;
                let loss_plus = self.mse_loss(x, y);
                self.layers[layer_idx].weights[wi] = orig;
                w_grads[wi] = (loss_plus - base_loss) / eps;
            }
            // Apply weight updates
            for wi in 0..n_w {
                self.layers[layer_idx].weights[wi] -= lr * w_grads[wi];
            }

            // Biases
            let n_b = self.layers[layer_idx].bias.len();
            let mut b_grads = vec![0.0_f32; n_b];
            for bi in 0..n_b {
                let orig = self.layers[layer_idx].bias[bi];
                self.layers[layer_idx].bias[bi] = orig + eps;
                let loss_plus = self.mse_loss(x, y);
                self.layers[layer_idx].bias[bi] = orig;
                b_grads[bi] = (loss_plus - base_loss) / eps;
            }
            for bi in 0..n_b {
                self.layers[layer_idx].bias[bi] -= lr * b_grads[bi];
            }
        }

        base_loss
    }
}

// ---------------------------------------------------------------------------
// 2. EnsembleSurrogate
// ---------------------------------------------------------------------------

/// Uncertainty-aware surrogate via deep ensemble.
///
/// Produces mean and variance (epistemic uncertainty) by averaging
/// predictions across multiple independently initialised networks.
#[derive(Debug, Clone)]
pub struct EnsembleSurrogate {
    pub models: Vec<NeuralSurrogate>,
}

impl EnsembleSurrogate {
    /// Create an ensemble of `n_models` surrogates, each with a unique seed.
    pub fn new(config: SurrogateConfig, n_models: usize, base_seed: u64) -> Self {
        let models = (0..n_models)
            .map(|i| NeuralSurrogate::new(config.clone(), base_seed + i as u64))
            .collect();
        EnsembleSurrogate { models }
    }

    /// Predict, returning element-wise (mean, variance) across ensemble members.
    pub fn predict_with_uncertainty(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let preds: Vec<Vec<f32>> = self.models.iter().map(|m| m.forward(x)).collect();
        if preds.is_empty() {
            return (vec![], vec![]);
        }
        let out_dim = preds[0].len();
        let n = preds.len() as f32;

        let mean: Vec<f32> = (0..out_dim)
            .map(|j| preds.iter().map(|p| p[j]).sum::<f32>() / n)
            .collect();

        let variance: Vec<f32> = (0..out_dim)
            .map(|j| {
                let m = mean[j];
                preds.iter().map(|p| (p[j] - m) * (p[j] - m)).sum::<f32>() / n
            })
            .collect();

        (mean, variance)
    }

    /// Epistemic uncertainty score = mean variance across output dimensions.
    pub fn active_learning_score(&self, x: &[f32]) -> f32 {
        let (_, var) = self.predict_with_uncertainty(x);
        if var.is_empty() {
            return 0.0;
        }
        var.iter().sum::<f32>() / var.len() as f32
    }

    /// Train all ensemble members for one step on a single data point.
    pub fn train_step_all(&mut self, x: &[f32], y: &[f32], lr: f32) -> Vec<f32> {
        self.models
            .iter_mut()
            .map(|m| m.train_step(x, y, lr))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// 3. PhysicsInformedSurrogate (PiSurrogate)
// ---------------------------------------------------------------------------

/// A linear physics constraint of the form A·pred ≈ b.
/// `constraint_matrix` is stored row-major with shape [n_constraints × output_dim].
#[derive(Debug, Clone)]
pub struct PhysicsResidual {
    /// Matrix A, shape [n_constraints × output_dim] in row-major order.
    pub constraint_matrix: Vec<f32>,
    /// Right-hand side b, shape \[n_constraints\].
    pub rhs: Vec<f32>,
    pub n_constraints: usize,
    pub output_dim: usize,
}

impl PhysicsResidual {
    /// Construct a physics residual from the given matrix (row-major) and rhs.
    pub fn new(
        constraint_matrix: Vec<f32>,
        rhs: Vec<f32>,
        n_constraints: usize,
        output_dim: usize,
    ) -> Self {
        PhysicsResidual {
            constraint_matrix,
            rhs,
            n_constraints,
            output_dim,
        }
    }

    /// Compute ||A·pred - b||² / n_constraints.
    pub fn physics_loss(&self, pred: &[f32]) -> f32 {
        let mut total = 0.0_f32;
        for ci in 0..self.n_constraints {
            let mut row_val = 0.0_f32;
            for j in 0..self.output_dim.min(pred.len()) {
                row_val += self.constraint_matrix[ci * self.output_dim + j] * pred[j];
            }
            let diff = row_val - self.rhs[ci];
            total += diff * diff;
        }
        if self.n_constraints > 0 {
            total / self.n_constraints as f32
        } else {
            0.0
        }
    }
}

/// Physics-informed surrogate combining data-driven and physics residual losses.
///
/// `total_loss = data_loss + lambda * physics_loss`
#[derive(Debug, Clone)]
pub struct PiSurrogate {
    pub base: NeuralSurrogate,
    pub constraints: Vec<PhysicsResidual>,
    /// Weighting for the physics penalty term.
    pub lambda: f32,
}

impl PiSurrogate {
    /// Create a new physics-informed surrogate.
    pub fn new(
        config: SurrogateConfig,
        constraints: Vec<PhysicsResidual>,
        lambda: f32,
        seed: u64,
    ) -> Self {
        PiSurrogate {
            base: NeuralSurrogate::new(config, seed),
            constraints,
            lambda,
        }
    }

    /// Forward pass delegates to base surrogate.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        self.base.forward(x)
    }

    /// Compute total loss: MSE data term + λ Σ physics residuals.
    pub fn total_loss(&self, x: &[f32], y: &[f32]) -> f32 {
        let pred = self.base.forward(x);

        // MSE data loss
        let data_loss = pred
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t) * (p - t))
            .sum::<f32>()
            / y.len().max(1) as f32;

        // Physics penalty
        let phys_loss: f32 = self
            .constraints
            .iter()
            .map(|c| c.physics_loss(&pred))
            .sum::<f32>();

        data_loss + self.lambda * phys_loss
    }

    /// One training step using total loss for the finite-difference gradient.
    pub fn train_step(&mut self, x: &[f32], y: &[f32], lr: f32) -> f32 {
        let eps = 1e-4_f32;

        // Capture current total loss by delegating through a helper closure-friendly approach.
        let base_loss = self.total_loss(x, y);

        for layer_idx in 0..self.base.layers.len() {
            // Weight gradients
            let n_w = self.base.layers[layer_idx].weights.len();
            let mut w_grads = vec![0.0_f32; n_w];
            for wi in 0..n_w {
                let orig = self.base.layers[layer_idx].weights[wi];
                self.base.layers[layer_idx].weights[wi] = orig + eps;
                let loss_plus = self.total_loss(x, y);
                self.base.layers[layer_idx].weights[wi] = orig;
                w_grads[wi] = (loss_plus - base_loss) / eps;
            }
            for wi in 0..n_w {
                self.base.layers[layer_idx].weights[wi] -= lr * w_grads[wi];
            }

            // Bias gradients
            let n_b = self.base.layers[layer_idx].bias.len();
            let mut b_grads = vec![0.0_f32; n_b];
            for bi in 0..n_b {
                let orig = self.base.layers[layer_idx].bias[bi];
                self.base.layers[layer_idx].bias[bi] = orig + eps;
                let loss_plus = self.total_loss(x, y);
                self.base.layers[layer_idx].bias[bi] = orig;
                b_grads[bi] = (loss_plus - base_loss) / eps;
            }
            for bi in 0..n_b {
                self.base.layers[layer_idx].bias[bi] -= lr * b_grads[bi];
            }
        }

        base_loss
    }
}

// ---------------------------------------------------------------------------
// 4. GaussianProcessSurrogate
// ---------------------------------------------------------------------------

/// Squared-exponential (RBF) covariance kernel.
#[derive(Debug, Clone)]
pub struct RbfKernel {
    /// Characteristic length-scale.
    pub length_scale: f32,
    /// Signal (output) variance σ_f².
    pub signal_var: f32,
    /// Observation noise variance σ_n².
    pub noise_var: f32,
}

impl RbfKernel {
    /// Create a new RBF kernel.
    pub fn new(length_scale: f32, signal_var: f32, noise_var: f32) -> Self {
        RbfKernel {
            length_scale,
            signal_var,
            noise_var,
        }
    }

    /// k(x₁, x₂) = σ_f² exp(-||x₁-x₂||² / (2 l²))
    pub fn kernel(&self, x1: &[f32], x2: &[f32]) -> f32 {
        let sq_dist: f32 = x1
            .iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        self.signal_var * (-sq_dist / (2.0 * self.length_scale * self.length_scale)).exp()
    }
}

/// Gaussian Process surrogate trained on small datasets.
///
/// Solves (K + σ_n² I) α = y via Cholesky decomposition for exact GP inference.
#[derive(Debug, Clone)]
pub struct GpSurrogate {
    pub x_train: Vec<Vec<f32>>,
    pub y_train: Vec<f32>,
    pub kernel: RbfKernel,
    /// α = (K + σ_n² I)⁻¹ y, computed after fitting.
    pub alpha: Vec<f32>,
    /// Lower Cholesky factor L of (K + σ_n² I), stored row-major.
    chol: Vec<f32>,
    n: usize,
}

impl GpSurrogate {
    /// Create an empty GP surrogate.
    pub fn new(kernel: RbfKernel) -> Self {
        GpSurrogate {
            x_train: vec![],
            y_train: vec![],
            kernel,
            alpha: vec![],
            chol: vec![],
            n: 0,
        }
    }

    /// Fit the GP to training data by computing the Cholesky factorisation.
    pub fn fit(&mut self, x_train: Vec<Vec<f32>>, y_train: Vec<f32>) -> Result<()> {
        if x_train.len() != y_train.len() {
            return Err(SimError::DimensionMismatch {
                expected: x_train.len(),
                got: y_train.len(),
            });
        }
        self.n = x_train.len();
        self.x_train = x_train;
        self.y_train = y_train.clone();

        // Build kernel matrix K + σ_n² I
        let n = self.n;
        let mut k = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                k[i * n + j] = self.kernel.kernel(&self.x_train[i], &self.x_train[j]);
            }
            k[i * n + i] += self.kernel.noise_var;
        }

        // Cholesky decomposition: L L^T = K + noise*I
        let l = cholesky_decompose(&k, n)?;

        // Solve L L^T α = y by forward/back substitution
        let ly = forward_substitution(&l, &y_train, n);
        self.alpha = back_substitution_transpose(&l, &ly, n);
        self.chol = l;
        Ok(())
    }

    /// Predict mean and variance at a new point x*.
    pub fn predict(&self, x: &[f32]) -> Result<(f32, f32)> {
        if self.n == 0 {
            return Err(SimError::NoTrainingData);
        }
        // k* = [k(x*, x_1), ..., k(x*, x_n)]
        let k_star: Vec<f32> = self
            .x_train
            .iter()
            .map(|xi| self.kernel.kernel(x, xi))
            .collect();

        // Mean: k*^T α
        let mean: f32 = k_star
            .iter()
            .zip(self.alpha.iter())
            .map(|(a, b)| a * b)
            .sum();

        // Variance: k(x*,x*) - k*^T (K+σI)^{-1} k*
        //          = k** - ||L^{-1} k*||²
        let k_self = self.kernel.signal_var; // k(x*,x*) when sq_dist=0
        let v = forward_substitution(&self.chol, &k_star, self.n);
        let v_sq: f32 = v.iter().map(|vi| vi * vi).sum();
        let variance = (k_self - v_sq).max(0.0);

        Ok((mean, variance))
    }
}

// ---------------------------------------------------------------------------
// Cholesky helpers
// ---------------------------------------------------------------------------

/// Compute the lower Cholesky factor L such that A = L L^T.
/// A must be symmetric positive-definite. A stored row-major, size n×n.
fn cholesky_decompose(a: &[f32], n: usize) -> Result<Vec<f32>> {
    let mut l = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0_f32;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a[i * n + i] - sum;
                if diag <= 0.0 {
                    return Err(SimError::CholeskyFailed);
                }
                l[i * n + j] = diag.sqrt();
            } else {
                if l[j * n + j].abs() < 1e-12 {
                    return Err(SimError::CholeskyFailed);
                }
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
            }
        }
    }
    Ok(l)
}

/// Forward substitution: solve L y = b where L is lower triangular.
fn forward_substitution(l: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut y = vec![0.0_f32; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * y[j];
        }
        let diag = l[i * n + i];
        y[i] = if diag.abs() > 1e-12 { s / diag } else { 0.0 };
    }
    y
}

/// Back substitution: solve L^T x = y where L is lower triangular.
fn back_substitution_transpose(l: &[f32], y: &[f32], n: usize) -> Vec<f32> {
    let mut x = vec![0.0_f32; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in i + 1..n {
            s -= l[j * n + i] * x[j];
        }
        let diag = l[i * n + i];
        x[i] = if diag.abs() > 1e-12 { s / diag } else { 0.0 };
    }
    x
}

// ---------------------------------------------------------------------------
// 5. LatinHypercubeSampler
// ---------------------------------------------------------------------------

/// Latin Hypercube Sampler for space-filling experimental design.
///
/// Stratifies each dimension into `n_samples` equal-probability intervals
/// and samples one point per stratum per dimension, then applies a random
/// permutation across dimensions to avoid column correlations.
#[derive(Debug, Clone)]
pub struct LatinHypercubeSampler {
    pub n_samples: usize,
    pub n_dims: usize,
}

impl LatinHypercubeSampler {
    /// Create a new LHS sampler.
    pub fn new(n_samples: usize, n_dims: usize) -> Result<Self> {
        if n_samples == 0 || n_dims == 0 {
            return Err(SimError::InvalidConfig(
                "n_samples and n_dims must be positive".to_string(),
            ));
        }
        Ok(LatinHypercubeSampler { n_samples, n_dims })
    }

    /// Generate a stratified LHS design and scale to `bounds`.
    ///
    /// Returns `n_samples` points, each of dimension `n_dims`.
    pub fn sample(&self, bounds: &[(f32, f32)], rng: &mut StdRng) -> Result<Vec<Vec<f32>>> {
        if bounds.len() != self.n_dims {
            return Err(SimError::DimensionMismatch {
                expected: self.n_dims,
                got: bounds.len(),
            });
        }
        let n = self.n_samples;
        // For each dimension, generate a stratified permutation.
        // Each stratum i contributes one sample in [i/n, (i+1)/n].
        let mut design = vec![vec![0.0_f32; self.n_dims]; n];

        for d in 0..self.n_dims {
            // Generate strata indices 0..n and shuffle them.
            let mut perm: Vec<usize> = (0..n).collect();
            fisher_yates_shuffle(&mut perm, rng);

            let (lo, hi) = bounds[d];
            let range = hi - lo;
            for (i, &stratum) in perm.iter().enumerate() {
                let u: f32 = rng.random::<f32>();
                let unit = (stratum as f32 + u) / n as f32;
                design[i][d] = lo + unit * range;
            }
        }
        Ok(design)
    }

    /// Maximin LHS: generate `n_candidates` LHS designs and return the one
    /// with the largest minimum pairwise distance (best space-filling).
    pub fn maximin_lhs(
        &self,
        bounds: &[(f32, f32)],
        n_candidates: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<Vec<f32>>> {
        if n_candidates == 0 {
            return Err(SimError::InvalidConfig(
                "n_candidates must be positive".to_string(),
            ));
        }
        let mut best: Vec<Vec<f32>> = vec![];
        let mut best_mindist = -1.0_f32;

        for _ in 0..n_candidates {
            let design = self.sample(bounds, rng)?;
            let md = min_pairwise_distance(&design);
            if md > best_mindist {
                best_mindist = md;
                best = design;
            }
        }
        Ok(best)
    }
}

/// Fisher-Yates shuffle.
fn fisher_yates_shuffle(v: &mut [usize], rng: &mut StdRng) {
    let n = v.len();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        v.swap(i, j);
    }
}

/// Minimum pairwise Euclidean distance among a set of points.
fn min_pairwise_distance(points: &[Vec<f32>]) -> f32 {
    let n = points.len();
    if n < 2 {
        return f32::INFINITY;
    }
    let mut min_d = f32::INFINITY;
    for i in 0..n {
        for j in i + 1..n {
            let d: f32 = points[i]
                .iter()
                .zip(points[j].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            if d < min_d {
                min_d = d;
            }
        }
    }
    min_d
}

// ---------------------------------------------------------------------------
// 6. AdaptiveSampler
// ---------------------------------------------------------------------------

/// Approximate the standard normal CDF using Hart's rational approximation.
/// Valid for z in (−∞, +∞) with absolute error < 7.5×10⁻⁸.
fn standard_normal_cdf(z: f32) -> f32 {
    // Abramowitz & Stegun approximation (7.1.26)
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let poly = t
        * (0.319_381_54
            + t * (-0.356_563_78 + t * (1.781_477_9 + t * (-1.821_255_9 + t * 1.330_274_5))));
    let pdf = (-z * z / 2.0).exp() / (2.0 * std::f32::consts::PI).sqrt();
    if z >= 0.0 {
        1.0 - pdf * poly
    } else {
        pdf * poly
    }
}

/// Approximate the standard normal PDF.
fn standard_normal_pdf(z: f32) -> f32 {
    (-z * z / 2.0).exp() / (2.0 * std::f32::consts::PI).sqrt()
}

/// Sequential experimental design using Expected Improvement (EI) over a GP surrogate.
#[derive(Debug, Clone)]
pub struct AdaptiveSampler {
    pub surrogate: GpSurrogate,
}

impl AdaptiveSampler {
    /// Create an adaptive sampler backed by the given GP surrogate.
    pub fn new(surrogate: GpSurrogate) -> Self {
        AdaptiveSampler { surrogate }
    }

    /// Expected Improvement at `x` given the current best observed value `y_best`.
    /// EI(x) = (μ - y_best) Φ(Z) + σ φ(Z),  Z = (μ - y_best) / σ
    pub fn expected_improvement(&self, x: &[f32], y_best: f32) -> f32 {
        match self.surrogate.predict(x) {
            Ok((mean, var)) => {
                let std_dev = var.sqrt().max(1e-8);
                let z = (mean - y_best) / std_dev;
                let ei =
                    (mean - y_best) * standard_normal_cdf(z) + std_dev * standard_normal_pdf(z);
                ei.max(0.0)
            }
            Err(_) => 0.0,
        }
    }

    /// Return the index of the candidate that maximises EI.
    pub fn next_point(&self, candidates: &[Vec<f32>], y_best: f32) -> usize {
        candidates
            .iter()
            .enumerate()
            .map(|(i, x)| (i, self.expected_improvement(x, y_best)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// 7. DifferentiableSimulator — Spring-Mass System
// ---------------------------------------------------------------------------

/// A spring connecting two mass nodes in a 1D spring-mass chain.
/// Fields: (node_i, node_j, stiffness_k, rest_length).
#[derive(Debug, Clone)]
pub struct Spring {
    pub node_i: usize,
    pub node_j: usize,
    pub stiffness: f32,
    pub rest_length: f32,
}

/// Simple 1D spring-mass system supporting Verlet integration.
///
/// Positions and velocities are 1D arrays of length `n_masses`.
#[derive(Debug, Clone)]
pub struct SpringMassSystem {
    pub masses: Vec<f32>,
    pub springs: Vec<Spring>,
}

impl SpringMassSystem {
    /// Create a spring-mass system.
    pub fn new(masses: Vec<f32>, springs: Vec<(usize, usize, f32, f32)>) -> Result<Self> {
        if masses.is_empty() {
            return Err(SimError::InvalidConfig(
                "masses must be non-empty".to_string(),
            ));
        }
        let springs = springs
            .into_iter()
            .map(|(i, j, k, r)| Spring {
                node_i: i,
                node_j: j,
                stiffness: k,
                rest_length: r,
            })
            .collect();
        Ok(SpringMassSystem { masses, springs })
    }

    /// Compute spring forces on each mass.
    fn compute_forces(&self, positions: &[f32]) -> Vec<f32> {
        let n = self.masses.len();
        let mut forces = vec![0.0_f32; n];
        for s in &self.springs {
            if s.node_i >= n || s.node_j >= n {
                continue;
            }
            let dx = positions[s.node_j] - positions[s.node_i];
            let current_len = dx.abs();
            let extension = current_len - s.rest_length;
            let force_mag = s.stiffness * extension;
            // Direction: positive means spring pulls i toward j.
            let dir = if dx >= 0.0 { 1.0_f32 } else { -1.0_f32 };
            forces[s.node_i] += force_mag * dir;
            forces[s.node_j] -= force_mag * dir;
        }
        forces
    }

    /// Velocity Verlet integration step.
    /// Returns (new_positions, new_velocities).
    pub fn simulate_step(
        &self,
        positions: &[f32],
        velocities: &[f32],
        dt: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = self.masses.len();
        if positions.len() != n || velocities.len() != n {
            return Err(SimError::DimensionMismatch {
                expected: n,
                got: positions.len(),
            });
        }

        let forces = self.compute_forces(positions);

        // Half-step velocity, full position update.
        let mut acc = vec![0.0_f32; n];
        for i in 0..n {
            acc[i] = forces[i] / self.masses[i].max(1e-12);
        }

        let new_pos: Vec<f32> = (0..n)
            .map(|i| positions[i] + velocities[i] * dt + 0.5 * acc[i] * dt * dt)
            .collect();

        let forces2 = self.compute_forces(&new_pos);
        let new_vel: Vec<f32> = (0..n)
            .map(|i| {
                let acc2 = forces2[i] / self.masses[i].max(1e-12);
                velocities[i] + 0.5 * (acc[i] + acc2) * dt
            })
            .collect();

        Ok((new_pos, new_vel))
    }

    /// KE = Σ 0.5 mᵢ vᵢ²
    pub fn kinetic_energy(&self, velocities: &[f32]) -> f32 {
        self.masses
            .iter()
            .zip(velocities.iter())
            .map(|(m, v)| 0.5 * m * v * v)
            .sum()
    }

    /// PE = Σ springs 0.5 k (|x_j - x_i| - rest_length)²
    pub fn potential_energy(&self, positions: &[f32]) -> f32 {
        let n = positions.len();
        self.springs
            .iter()
            .filter(|s| s.node_i < n && s.node_j < n)
            .map(|s| {
                let dx = (positions[s.node_j] - positions[s.node_i]).abs();
                let ext = dx - s.rest_length;
                0.5 * s.stiffness * ext * ext
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// 8. TurbulenceModel
// ---------------------------------------------------------------------------

/// Symmetric Reynolds stress tensor (upper-triangle storage).
#[derive(Debug, Clone)]
pub struct ReynoldsStressTensor {
    pub s11: f32,
    pub s12: f32,
    pub s13: f32,
    pub s22: f32,
    pub s23: f32,
    pub s33: f32,
}

impl ReynoldsStressTensor {
    /// Frobenius norm |S| = sqrt(2 Sᵢⱼ Sᵢⱼ).
    pub fn frobenius_norm(&self) -> f32 {
        let trace_sq = self.s11 * self.s11
            + self.s22 * self.s22
            + self.s33 * self.s33
            + 2.0 * (self.s12 * self.s12 + self.s13 * self.s13 + self.s23 * self.s23);
        (2.0 * trace_sq).sqrt()
    }
}

/// Smagorinsky sub-grid-scale model for LES turbulence closure.
///
/// Computes the eddy viscosity ν_t = (Cs δ)² |S|.
#[derive(Debug, Clone)]
pub struct SmagorinskyModel {
    /// Smagorinsky constant (typical: 0.1–0.2).
    pub cs: f32,
    /// Filter width (grid resolution).
    pub delta: f32,
}

impl SmagorinskyModel {
    /// Create a new Smagorinsky model.
    pub fn new(cs: f32, delta: f32) -> Self {
        SmagorinskyModel { cs, delta }
    }

    /// Eddy viscosity: ν_t = (Cs δ)² |S|.
    pub fn eddy_viscosity(&self, strain_rate: &ReynoldsStressTensor) -> f32 {
        let cs_delta = self.cs * self.delta;
        cs_delta * cs_delta * strain_rate.frobenius_norm()
    }
}

/// ML correction model that adds learned residuals to RANS outputs.
///
/// Implements a single linear layer: correction = W · features + b.
/// `weights` shape: [output_dim × feature_dim] row-major.
#[derive(Debug, Clone)]
pub struct MlCorrectionModel {
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub feature_dim: usize,
    pub output_dim: usize,
}

impl MlCorrectionModel {
    /// Create a zero-initialised correction model.
    pub fn new_zeros(feature_dim: usize, output_dim: usize) -> Self {
        MlCorrectionModel {
            weights: vec![0.0_f32; output_dim * feature_dim],
            bias: vec![0.0_f32; output_dim],
            feature_dim,
            output_dim,
        }
    }

    /// Apply the learned correction: corrected = rans_output + W · features + b.
    pub fn apply_correction(&self, rans_output: &[f32], features: &[f32]) -> Vec<f32> {
        let out_dim = self.output_dim.min(rans_output.len());
        (0..out_dim)
            .map(|o| {
                let corr: f32 = (0..self.feature_dim.min(features.len()))
                    .map(|fi| self.weights[o * self.feature_dim + fi] * features[fi])
                    .sum::<f32>()
                    + self.bias[o];
                rans_output[o] + corr
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// 9. SimulationDataAugmenter
// ---------------------------------------------------------------------------

/// Augmentation utilities for simulation datasets.
pub struct SimulationDataAugmenter;

impl SimulationDataAugmenter {
    /// Add independent Gaussian noise with standard deviation `noise_std` to each element.
    pub fn noise_injection(x: &[f32], noise_std: f32, rng: &mut StdRng) -> Vec<f32> {
        x.iter()
            .map(|&v| {
                let noise = sample_standard_normal_f32(rng) * noise_std;
                v + noise
            })
            .collect()
    }

    /// Generate `n` symmetrically-transformed copies of `x` via `symmetry_matrix`.
    ///
    /// `symmetry_matrix` is row-major, shape [dim × dim].
    /// For each copy `k`, the output is `M^k · x`.
    pub fn symmetry_augment(x: &[f32], symmetry_matrix: &[f32], n: usize) -> Vec<Vec<f32>> {
        let dim = x.len();
        if dim == 0 || n == 0 {
            return vec![];
        }
        let mat_dim = (symmetry_matrix.len() as f32).sqrt().round() as usize;
        if mat_dim * mat_dim != symmetry_matrix.len() || mat_dim != dim {
            // Matrix/vector dimension mismatch — return copies of x unchanged.
            return (0..n).map(|_| x.to_vec()).collect();
        }

        let mut results = Vec::with_capacity(n);
        let mut current = x.to_vec();
        for _ in 0..n {
            current = mat_vec_mul(symmetry_matrix, &current, dim);
            results.push(current.clone());
        }
        results
    }

    /// Generate `n_interp` linearly-interpolated states between `x1` and `x2`.
    /// Endpoints are excluded; only interior points are returned.
    pub fn interpolate_between(x1: &[f32], x2: &[f32], n_interp: usize) -> Vec<Vec<f32>> {
        if n_interp == 0 {
            return vec![];
        }
        let len = x1.len().min(x2.len());
        (1..=n_interp)
            .map(|k| {
                let t = k as f32 / (n_interp + 1) as f32;
                (0..len).map(|i| x1[i] + t * (x2[i] - x1[i])).collect()
            })
            .collect()
    }
}

/// Box-Muller transform to sample N(0,1).
fn sample_standard_normal_f32(rng: &mut StdRng) -> f32 {
    let u1: f32 = rng.random::<f32>().max(1e-37);
    let u2: f32 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Matrix-vector multiplication y = A x, A is [n × n] row-major.
fn mat_vec_mul(a: &[f32], x: &[f32], n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| a[i * n + j] * x.get(j).copied().unwrap_or(0.0))
                .sum::<f32>()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 10. SimulationMetrics
// ---------------------------------------------------------------------------

/// Evaluation metrics for assessing surrogate quality.
pub struct SimulationMetrics;

impl SimulationMetrics {
    /// Relative L2 error: ||pred - true||₂ / ||true||₂.
    pub fn relative_l2_error(pred: &[f32], true_vals: &[f32]) -> f32 {
        let n = pred.len().min(true_vals.len());
        if n == 0 {
            return 0.0;
        }
        let num: f32 = (0..n)
            .map(|i| (pred[i] - true_vals[i]).powi(2))
            .sum::<f32>()
            .sqrt();
        let den: f32 = (0..n).map(|i| true_vals[i].powi(2)).sum::<f32>().sqrt();
        if den < 1e-12 {
            num
        } else {
            num / den
        }
    }

    /// Maximum absolute error: max_i |pred_i - true_i|.
    pub fn max_absolute_error(pred: &[f32], true_vals: &[f32]) -> f32 {
        pred.iter()
            .zip(true_vals.iter())
            .map(|(p, t)| (p - t).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Predictive Q² score (analogous to R² but on test data):
    /// Q² = 1 - SS_res / SS_tot, where SS_tot uses the training-data mean.
    /// Here we compute it as Q² = 1 - ||pred - true||² / ||true - mean(true)||².
    pub fn q2_score(pred: &[f32], true_vals: &[f32]) -> f32 {
        let n = pred.len().min(true_vals.len());
        if n == 0 {
            return 0.0;
        }
        let mean: f32 = true_vals[..n].iter().sum::<f32>() / n as f32;
        let ss_res: f32 = (0..n).map(|i| (pred[i] - true_vals[i]).powi(2)).sum();
        let ss_tot: f32 = (0..n).map(|i| (true_vals[i] - mean).powi(2)).sum();
        if ss_tot < 1e-12 {
            if ss_res < 1e-12 {
                1.0
            } else {
                0.0
            }
        } else {
            1.0 - ss_res / ss_tot
        }
    }

    /// Coverage probability: fraction of true values within the z·σ interval
    /// [mean - z·std, mean + z·std].
    pub fn coverage_probability(
        pred_mean: &[f32],
        pred_std: &[f32],
        true_vals: &[f32],
        z: f32,
    ) -> f32 {
        let n = pred_mean.len().min(pred_std.len()).min(true_vals.len());
        if n == 0 {
            return 0.0;
        }
        let covered = (0..n)
            .filter(|&i| {
                let lo = pred_mean[i] - z * pred_std[i];
                let hi = pred_mean[i] + z * pred_std[i];
                true_vals[i] >= lo && true_vals[i] <= hi
            })
            .count();
        covered as f32 / n as f32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------- SurrogateActivation -------

    #[test]
    fn test_activation_relu_positive() {
        assert!((SurrogateActivation::ReLU.apply(2.0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_activation_relu_negative() {
        assert!((SurrogateActivation::ReLU.apply(-1.5)).abs() < 1e-6);
    }

    #[test]
    fn test_activation_tanh() {
        let v = SurrogateActivation::Tanh.apply(0.0);
        assert!(v.abs() < 1e-6);
    }

    #[test]
    fn test_activation_gelu_zero() {
        let v = SurrogateActivation::GELU.apply(0.0);
        assert!(v.abs() < 1e-5);
    }

    #[test]
    fn test_activation_swish_positive() {
        let v = SurrogateActivation::Swish.apply(1.0);
        assert!(v > 0.0);
    }

    // ------- NeuralSurrogate -------

    #[test]
    fn test_neural_surrogate_forward_shape() {
        let config = SurrogateConfig {
            input_dim: 4,
            hidden_dims: vec![8, 8],
            output_dim: 2,
            activation: SurrogateActivation::ReLU,
        };
        let model = NeuralSurrogate::new(config, 42);
        let x = vec![1.0_f32; 4];
        let y = model.forward(&x);
        assert_eq!(y.len(), 2);
    }

    #[test]
    fn test_neural_surrogate_train_step_returns_finite_loss() {
        let config = SurrogateConfig {
            input_dim: 2,
            hidden_dims: vec![4],
            output_dim: 1,
            activation: SurrogateActivation::Tanh,
        };
        let mut model = NeuralSurrogate::new(config, 7);
        let x = vec![0.5_f32, -0.3];
        let y = vec![1.0_f32];
        // Run several steps; loss returned should be a valid finite float.
        let initial_loss = model.mse_loss(&x, &y);
        let mut final_loss = initial_loss;
        for _ in 0..10 {
            final_loss = model.train_step(&x, &y, 0.001);
        }
        assert!(initial_loss.is_finite());
        assert!(final_loss.is_finite());
        assert!(initial_loss >= 0.0);
    }

    #[test]
    fn test_neural_surrogate_gelu_activation() {
        let config = SurrogateConfig {
            input_dim: 3,
            hidden_dims: vec![6],
            output_dim: 2,
            activation: SurrogateActivation::GELU,
        };
        let model = NeuralSurrogate::new(config, 99);
        let x = vec![0.1, 0.2, 0.3];
        let y = model.forward(&x);
        assert_eq!(y.len(), 2);
    }

    #[test]
    fn test_neural_surrogate_swish_activation() {
        let config = SurrogateConfig {
            input_dim: 2,
            hidden_dims: vec![4, 4],
            output_dim: 1,
            activation: SurrogateActivation::Swish,
        };
        let model = NeuralSurrogate::new(config, 13);
        let out = model.forward(&[1.0, -1.0]);
        assert_eq!(out.len(), 1);
    }

    // ------- EnsembleSurrogate -------

    #[test]
    fn test_ensemble_predict_mean_shape() {
        let config = SurrogateConfig {
            input_dim: 3,
            hidden_dims: vec![4],
            output_dim: 2,
            activation: SurrogateActivation::ReLU,
        };
        let ens = EnsembleSurrogate::new(config, 5, 0);
        let (mean, var) = ens.predict_with_uncertainty(&[0.1, 0.2, 0.3]);
        assert_eq!(mean.len(), 2);
        assert_eq!(var.len(), 2);
    }

    #[test]
    fn test_ensemble_variance_non_negative() {
        let config = SurrogateConfig {
            input_dim: 2,
            hidden_dims: vec![4],
            output_dim: 1,
            activation: SurrogateActivation::Tanh,
        };
        let ens = EnsembleSurrogate::new(config, 3, 42);
        let (_, var) = ens.predict_with_uncertainty(&[0.5, -0.5]);
        assert!(var.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_ensemble_active_learning_score() {
        let config = SurrogateConfig {
            input_dim: 2,
            hidden_dims: vec![4],
            output_dim: 1,
            activation: SurrogateActivation::ReLU,
        };
        let ens = EnsembleSurrogate::new(config, 4, 1);
        let score = ens.active_learning_score(&[0.0, 1.0]);
        assert!(score >= 0.0);
    }

    #[test]
    fn test_ensemble_train_step_all() {
        let config = SurrogateConfig {
            input_dim: 2,
            hidden_dims: vec![4],
            output_dim: 1,
            activation: SurrogateActivation::ReLU,
        };
        let mut ens = EnsembleSurrogate::new(config, 3, 5);
        let losses = ens.train_step_all(&[0.1, 0.2], &[1.0], 0.01);
        assert_eq!(losses.len(), 3);
    }

    // ------- PhysicsResidual / PiSurrogate -------

    #[test]
    fn test_physics_residual_zero_loss_when_satisfied() {
        // A = I, b = [1, 1], pred = [1, 1] => loss = 0
        let a = vec![1.0_f32, 0.0, 0.0, 1.0];
        let b = vec![1.0_f32, 1.0];
        let pr = PhysicsResidual::new(a, b, 2, 2);
        let loss = pr.physics_loss(&[1.0, 1.0]);
        assert!(loss.abs() < 1e-5);
    }

    #[test]
    fn test_physics_residual_nonzero_loss() {
        let a = vec![1.0_f32, 0.0, 0.0, 1.0];
        let b = vec![0.0_f32, 0.0];
        let pr = PhysicsResidual::new(a, b, 2, 2);
        let loss = pr.physics_loss(&[1.0, 1.0]);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_pi_surrogate_total_loss_non_negative() {
        let config = SurrogateConfig {
            input_dim: 2,
            hidden_dims: vec![4],
            output_dim: 2,
            activation: SurrogateActivation::ReLU,
        };
        let pr = PhysicsResidual::new(vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0], 2, 2);
        let mut pi = PiSurrogate::new(config, vec![pr], 0.1, 77);
        let loss = pi.total_loss(&[0.1, 0.2], &[0.5, 0.5]);
        assert!(loss >= 0.0);
        // Train once and check loss is a valid float
        let tl = pi.train_step(&[0.1, 0.2], &[0.5, 0.5], 0.001);
        assert!(tl.is_finite());
    }

    // ------- RbfKernel -------

    #[test]
    fn test_rbf_kernel_self_distance() {
        let k = RbfKernel::new(1.0, 1.0, 0.01);
        let x = vec![1.0_f32, 2.0, 3.0];
        let val = k.kernel(&x, &x);
        // k(x,x) = signal_var
        assert!((val - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rbf_kernel_decreases_with_distance() {
        let k = RbfKernel::new(1.0, 1.0, 0.01);
        let x1 = vec![0.0_f32];
        let x2 = vec![1.0_f32];
        let x3 = vec![5.0_f32];
        let k12 = k.kernel(&x1, &x2);
        let k13 = k.kernel(&x1, &x3);
        assert!(k12 > k13);
    }

    // ------- GpSurrogate -------

    #[test]
    fn test_gp_surrogate_fit_and_predict() {
        let kernel = RbfKernel::new(1.0, 1.0, 0.1);
        let mut gp = GpSurrogate::new(kernel);
        let x_train: Vec<Vec<f32>> = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y_train = vec![0.0_f32, 1.0, 0.0];
        gp.fit(x_train, y_train).expect("GP fit should succeed");

        let (mean, var) = gp.predict(&[1.0]).expect("predict should succeed");
        // Mean near training point should be close to y_train[1] = 1.0
        assert!((mean - 1.0).abs() < 0.5);
        assert!(var >= 0.0);
    }

    #[test]
    fn test_gp_surrogate_variance_zero_at_training_points() {
        let kernel = RbfKernel::new(1.0, 1.0, 1e-6);
        let mut gp = GpSurrogate::new(kernel);
        let x_train: Vec<Vec<f32>> = vec![vec![0.0], vec![2.0]];
        let y_train = vec![1.0_f32, -1.0];
        gp.fit(x_train, y_train).expect("fit");
        let (_m, v) = gp.predict(&[0.0]).expect("predict");
        // With tiny noise, variance at training point should be very small
        assert!(v < 0.1);
    }

    #[test]
    fn test_gp_surrogate_no_training_data_error() {
        let kernel = RbfKernel::new(1.0, 1.0, 0.1);
        let gp = GpSurrogate::new(kernel);
        let result = gp.predict(&[0.0]);
        assert!(result.is_err());
    }

    // ------- LatinHypercubeSampler -------

    #[test]
    fn test_lhs_sample_shape() {
        let mut rng = StdRng::seed_from_u64(42);
        let lhs = LatinHypercubeSampler::new(10, 3).expect("new");
        let bounds = vec![(0.0_f32, 1.0), (0.0, 2.0), (-1.0, 1.0)];
        let design = lhs.sample(&bounds, &mut rng).expect("sample");
        assert_eq!(design.len(), 10);
        assert_eq!(design[0].len(), 3);
    }

    #[test]
    fn test_lhs_sample_within_bounds() {
        let mut rng = StdRng::seed_from_u64(7);
        let lhs = LatinHypercubeSampler::new(20, 2).expect("new");
        let bounds = vec![(1.0_f32, 3.0), (-2.0_f32, -0.5)];
        let design = lhs.sample(&bounds, &mut rng).expect("sample");
        for p in &design {
            assert!(p[0] >= 1.0 && p[0] <= 3.0);
            assert!(p[1] >= -2.0 && p[1] <= -0.5);
        }
    }

    #[test]
    fn test_lhs_invalid_config() {
        assert!(LatinHypercubeSampler::new(0, 2).is_err());
        assert!(LatinHypercubeSampler::new(5, 0).is_err());
    }

    #[test]
    fn test_maximin_lhs_returns_design() {
        let mut rng = StdRng::seed_from_u64(99);
        let lhs = LatinHypercubeSampler::new(5, 2).expect("new");
        let bounds = vec![(0.0_f32, 1.0), (0.0_f32, 1.0)];
        let best = lhs.maximin_lhs(&bounds, 5, &mut rng).expect("maximin");
        assert_eq!(best.len(), 5);
    }

    // ------- AdaptiveSampler -------

    #[test]
    fn test_adaptive_sampler_ei_positive() {
        let kernel = RbfKernel::new(1.0, 1.0, 0.1);
        let mut gp = GpSurrogate::new(kernel);
        let xt = vec![vec![0.0_f32], vec![1.0], vec![2.0]];
        let yt = vec![0.0_f32, 1.0, 0.5];
        gp.fit(xt, yt).expect("fit");
        let sampler = AdaptiveSampler::new(gp);
        let ei = sampler.expected_improvement(&[1.5], 0.5);
        assert!(ei >= 0.0);
    }

    #[test]
    fn test_adaptive_sampler_next_point() {
        let kernel = RbfKernel::new(1.0, 1.0, 0.1);
        let mut gp = GpSurrogate::new(kernel);
        let xt = vec![vec![0.0_f32], vec![2.0]];
        let yt = vec![0.5_f32, 1.0];
        gp.fit(xt, yt).expect("fit");
        let sampler = AdaptiveSampler::new(gp);
        let candidates = vec![vec![0.5_f32], vec![1.0], vec![1.5], vec![2.5]];
        let idx = sampler.next_point(&candidates, 0.5);
        assert!(idx < candidates.len());
    }

    // ------- SpringMassSystem -------

    #[test]
    fn test_spring_mass_simulate_step() {
        let masses = vec![1.0_f32, 1.0];
        let springs = vec![(0usize, 1usize, 10.0_f32, 1.0_f32)];
        let sys = SpringMassSystem::new(masses, springs).expect("new");
        let pos = vec![0.0_f32, 2.0];
        let vel = vec![0.0_f32, 0.0];
        let (new_pos, new_vel) = sys.simulate_step(&pos, &vel, 0.01).expect("step");
        assert_eq!(new_pos.len(), 2);
        assert_eq!(new_vel.len(), 2);
    }

    #[test]
    fn test_spring_mass_kinetic_energy() {
        let masses = vec![2.0_f32, 1.0];
        let springs = vec![];
        let sys = SpringMassSystem::new(masses, springs).expect("new");
        let vel = vec![1.0_f32, 2.0];
        let ke = sys.kinetic_energy(&vel);
        // KE = 0.5*2*1 + 0.5*1*4 = 1 + 2 = 3
        assert!((ke - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_spring_mass_potential_energy_at_rest() {
        let masses = vec![1.0_f32, 1.0];
        let springs = vec![(0usize, 1usize, 5.0_f32, 1.0_f32)];
        let sys = SpringMassSystem::new(masses, springs).expect("new");
        // Distance == rest_length => PE = 0
        let pos = vec![0.0_f32, 1.0];
        let pe = sys.potential_energy(&pos);
        assert!(pe.abs() < 1e-5);
    }

    #[test]
    fn test_spring_mass_energy_conservation_approx() {
        let masses = vec![1.0_f32, 1.0];
        let springs = vec![(0usize, 1usize, 50.0_f32, 1.0_f32)];
        let sys = SpringMassSystem::new(masses, springs).expect("new");
        let mut pos = vec![0.0_f32, 2.0]; // extension = 1.0
        let mut vel = vec![0.0_f32, 0.0];
        let e0 = sys.kinetic_energy(&vel) + sys.potential_energy(&pos);
        for _ in 0..200 {
            let (np, nv) = sys.simulate_step(&pos, &vel, 0.001).expect("step");
            pos = np;
            vel = nv;
        }
        let e1 = sys.kinetic_energy(&vel) + sys.potential_energy(&pos);
        // Energy should be approximately conserved (small Verlet drift).
        assert!((e1 - e0).abs() / (e0.abs() + 1e-6) < 0.05);
    }

    // ------- TurbulenceModel -------

    #[test]
    fn test_reynolds_stress_frobenius_norm() {
        let rst = ReynoldsStressTensor {
            s11: 1.0,
            s12: 0.0,
            s13: 0.0,
            s22: 1.0,
            s23: 0.0,
            s33: 1.0,
        };
        let norm = rst.frobenius_norm();
        // |S|² = 2 * Sij Sij (Einstein sum over all i,j)
        // For diagonal S with s11=s22=s33=1: Sij Sij = 1+1+1 = 3
        // |S|² = 2*3 = 6 => |S| = sqrt(6) ≈ 2.449
        assert!((norm - (6.0_f32).sqrt()).abs() < 1e-4);
    }

    #[test]
    fn test_smagorinsky_eddy_viscosity() {
        let model = SmagorinskyModel::new(0.1, 0.5);
        let rst = ReynoldsStressTensor {
            s11: 1.0,
            s12: 0.0,
            s13: 0.0,
            s22: 1.0,
            s23: 0.0,
            s33: 1.0,
        };
        let nu = model.eddy_viscosity(&rst);
        // nu = (Cs*delta)^2 * |S| = (0.1*0.5)^2 * sqrt(6) = 0.0025 * sqrt(6)
        let expected = 0.0025 * (6.0_f32).sqrt();
        assert!((nu - expected).abs() < 1e-5);
    }

    #[test]
    fn test_ml_correction_model_zero_weights() {
        let model = MlCorrectionModel::new_zeros(3, 2);
        let rans = vec![1.0_f32, 2.0];
        let features = vec![0.5_f32, -0.5, 1.0];
        let corrected = model.apply_correction(&rans, &features);
        // Zero weights => correction = 0 => output == rans_output
        assert!((corrected[0] - 1.0).abs() < 1e-6);
        assert!((corrected[1] - 2.0).abs() < 1e-6);
    }

    // ------- SimulationDataAugmenter -------

    #[test]
    fn test_noise_injection_shape() {
        let mut rng = StdRng::seed_from_u64(1);
        let x = vec![1.0_f32, 2.0, 3.0];
        let noisy = SimulationDataAugmenter::noise_injection(&x, 0.1, &mut rng);
        assert_eq!(noisy.len(), 3);
    }

    #[test]
    fn test_noise_injection_zero_noise() {
        let mut rng = StdRng::seed_from_u64(0);
        let x = vec![5.0_f32, -3.0, 0.0];
        // With noise_std = 0, output should equal input exactly (noise = 0 * N(0,1)).
        let noisy = SimulationDataAugmenter::noise_injection(&x, 0.0, &mut rng);
        for (a, b) in x.iter().zip(noisy.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_symmetry_augment_count() {
        let x = vec![1.0_f32, 0.0];
        // Rotation by 90°: [[0,-1],[1,0]]
        let mat = vec![0.0_f32, -1.0, 1.0, 0.0];
        let copies = SimulationDataAugmenter::symmetry_augment(&x, &mat, 4);
        assert_eq!(copies.len(), 4);
    }

    #[test]
    fn test_symmetry_augment_rotation_cycle() {
        let x = vec![1.0_f32, 0.0];
        // 90° rotation: after 4 applications should return to x.
        let mat = vec![0.0_f32, -1.0, 1.0, 0.0];
        let copies = SimulationDataAugmenter::symmetry_augment(&x, &mat, 4);
        // copies[3] = M^4 x ≈ x (since M^4 = I for 90° rotation)
        assert!((copies[3][0] - 1.0).abs() < 1e-4);
        assert!(copies[3][1].abs() < 1e-4);
    }

    #[test]
    fn test_interpolate_between_count() {
        let x1 = vec![0.0_f32, 0.0];
        let x2 = vec![1.0_f32, 1.0];
        let interp = SimulationDataAugmenter::interpolate_between(&x1, &x2, 4);
        assert_eq!(interp.len(), 4);
    }

    #[test]
    fn test_interpolate_between_values() {
        let x1 = vec![0.0_f32];
        let x2 = vec![10.0_f32];
        let interp = SimulationDataAugmenter::interpolate_between(&x1, &x2, 4);
        // t = 1/5, 2/5, 3/5, 4/5 => 2, 4, 6, 8
        let expected = [2.0_f32, 4.0, 6.0, 8.0];
        for (v, e) in interp.iter().zip(expected.iter()) {
            assert!((v[0] - e).abs() < 1e-5);
        }
    }

    // ------- SimulationMetrics -------

    #[test]
    fn test_relative_l2_error_perfect_prediction() {
        let pred = vec![1.0_f32, 2.0, 3.0];
        let true_vals = vec![1.0_f32, 2.0, 3.0];
        let err = SimulationMetrics::relative_l2_error(&pred, &true_vals);
        assert!(err.abs() < 1e-6);
    }

    #[test]
    fn test_relative_l2_error_nonzero() {
        let pred = vec![1.1_f32, 2.0];
        let true_vals = vec![1.0_f32, 2.0];
        let err = SimulationMetrics::relative_l2_error(&pred, &true_vals);
        assert!(err > 0.0 && err < 1.0);
    }

    #[test]
    fn test_max_absolute_error() {
        let pred = vec![1.0_f32, 3.0, 5.0];
        let true_vals = vec![1.5_f32, 2.5, 5.0];
        let err = SimulationMetrics::max_absolute_error(&pred, &true_vals);
        assert!((err - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_q2_score_perfect() {
        let pred = vec![1.0_f32, 2.0, 3.0];
        let true_vals = vec![1.0_f32, 2.0, 3.0];
        let q2 = SimulationMetrics::q2_score(&pred, &true_vals);
        assert!((q2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_q2_score_constant_prediction() {
        let n = 10;
        let true_vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mean = true_vals.iter().sum::<f32>() / n as f32;
        let pred = vec![mean; n];
        let q2 = SimulationMetrics::q2_score(&pred, &true_vals);
        assert!((q2 - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_coverage_probability_full_coverage() {
        let means = vec![0.0_f32, 1.0, 2.0];
        let stds = vec![10.0_f32, 10.0, 10.0];
        let true_vals = vec![0.1_f32, 0.9, 2.1];
        let cov = SimulationMetrics::coverage_probability(&means, &stds, &true_vals, 1.0);
        assert!((cov - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_coverage_probability_no_coverage() {
        let means = vec![0.0_f32];
        let stds = vec![0.0_f32];
        let true_vals = vec![100.0_f32];
        let cov = SimulationMetrics::coverage_probability(&means, &stds, &true_vals, 1.0);
        assert!(cov.abs() < 1e-5);
    }

    // ------- Cholesky helpers -------

    #[test]
    fn test_cholesky_2x2() {
        // A = [[4, 2], [2, 3]]  => L = [[2, 0], [1, sqrt(2)]]
        let a = vec![4.0_f32, 2.0, 2.0, 3.0];
        let l = cholesky_decompose(&a, 2).expect("cholesky");
        assert!((l[0] - 2.0).abs() < 1e-5); // L[0,0]
        assert!((l[2] - 1.0).abs() < 1e-5); // L[1,0]
        assert!((l[3] - (2.0_f32).sqrt()).abs() < 1e-4); // L[1,1]
    }

    #[test]
    fn test_cholesky_non_pd_fails() {
        // A = [[-1, 0],[0, 1]] is not PD.
        let a = vec![-1.0_f32, 0.0, 0.0, 1.0];
        let result = cholesky_decompose(&a, 2);
        assert!(result.is_err());
    }
}
