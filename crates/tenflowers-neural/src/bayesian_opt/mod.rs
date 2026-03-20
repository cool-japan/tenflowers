//! Bayesian Optimization and CMA-ES Module — Round 13 Track D.
//!
//! This module provides:
//! - **Gaussian Process** surrogate models (RBF, Matérn 5/2, Periodic kernels)
//! - **Acquisition Functions**: UCB, EI, PI, Thompson Sampling
//! - **Bayesian Optimization** loop with Latin Hypercube initialization
//! - **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy, Hansen 2016)
//! - **Multi-Fidelity Optimization** using layered Bayesian optimizers
//!
//! All randomness is sourced from `scirs2_core::random` — no `rand` crate is used.
//! No `unsafe` code. No `unwrap()`.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use tenflowers_neural::bayesian_opt::{
//!     BayesianOptimizer, BayesOptConfig, BayesSearchSpace,
//!     AcquisitionFunction, GpConfig, KernelType,
//! };
//!
//! let space = BayesSearchSpace::new(vec![(-5.0, 5.0), (-5.0, 5.0)]).unwrap();
//! let config = BayesOptConfig {
//!     n_initial: 5,
//!     n_iterations: 20,
//!     acquisition: AcquisitionFunction::ExpectedImprovement { xi: 0.01 },
//!     gp_config: GpConfig::default(),
//!     n_candidates: 1000,
//!     seed: 42,
//! };
//! let mut opt = BayesianOptimizer::new(config, space, true);
//! let result = opt.optimize(|x| x[0].powi(2) + x[1].powi(2)).unwrap();
//! assert!(result.best_y < 1.0);
//! ```

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// 1. Kernel definitions
// ─────────────────────────────────────────────────────────────────────────────

/// Kernel function types for the Gaussian Process surrogate.
#[derive(Debug, Clone)]
pub enum KernelType {
    /// Squared Exponential (RBF): k(x,x') = σ² · exp(−‖x−x'‖²/(2l²))
    SquaredExponential { length_scale: f64, signal_std: f64 },
    /// Matérn 5/2: k(x,x') = σ²·(1+√5·r/l+5r²/(3l²))·exp(−√5·r/l)
    Matern52 { length_scale: f64, signal_std: f64 },
    /// Periodic: k(x,x') = σ²·exp(−2·sin²(π‖x−x'‖/p)/l²)
    Periodic {
        length_scale: f64,
        period: f64,
        signal_std: f64,
    },
}

impl Default for KernelType {
    fn default() -> Self {
        Self::SquaredExponential {
            length_scale: 1.0,
            signal_std: 1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. GP configuration and implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Gaussian Process surrogate.
#[derive(Debug, Clone)]
pub struct GpConfig {
    /// Kernel function to use.
    pub kernel: KernelType,
    /// Observation noise standard deviation σ_n.
    pub noise_std: f64,
    /// Number of random restarts for hyperparameter optimization (currently unused — reserved).
    pub n_restarts: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for GpConfig {
    fn default() -> Self {
        Self {
            kernel: KernelType::default(),
            noise_std: 1e-2,
            n_restarts: 3,
            seed: 0,
        }
    }
}

/// Gaussian Process surrogate model.
///
/// Supports fitting, prediction (mean + variance), and log marginal likelihood
/// computation for hyperparameter selection.
#[derive(Debug, Clone)]
pub struct GaussianProcess {
    config: GpConfig,
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    /// Output mean used for normalization.
    pub y_mean: f64,
    /// Output std used for normalization.
    pub y_std: f64,
    /// (K + σ_n² I)⁻¹ — precomputed for fast prediction.
    k_inv: Vec<Vec<f64>>,
    /// K_inv @ y_normalized — precomputed alpha vector.
    alpha: Vec<f64>,
    /// Cholesky lower-triangular L such that K + σ_n² I = L Lᵀ.
    chol_l: Vec<Vec<f64>>,
    /// Whether the GP has been fitted.
    pub fitted: bool,
}

impl GaussianProcess {
    /// Create a new, unfitted Gaussian Process.
    pub fn new(config: GpConfig) -> Self {
        Self {
            config,
            x_train: Vec::new(),
            y_train: Vec::new(),
            y_mean: 0.0,
            y_std: 1.0,
            k_inv: Vec::new(),
            alpha: Vec::new(),
            chol_l: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the GP to training data `(x, y)`.
    ///
    /// Normalizes outputs, builds the kernel matrix, performs Cholesky
    /// decomposition and precomputes the alpha vector.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<()> {
        if x.is_empty() || y.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "GaussianProcess::fit".to_string(),
                reason: "training data must be non-empty".to_string(),
                context: None,
            });
        }
        if x.len() != y.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "GaussianProcess::fit".to_string(),
                expected: format!("x.len()={}", x.len()),
                got: format!("y.len()={}", y.len()),
                context: None,
            });
        }

        self.x_train = x.to_vec();
        self.y_train = y.to_vec();

        // Normalize y
        let n = y.len();
        let y_sum: f64 = y.iter().sum();
        self.y_mean = y_sum / n as f64;
        let var = y.iter().map(|v| (v - self.y_mean).powi(2)).sum::<f64>() / n as f64;
        self.y_std = if var < 1e-12 { 1.0 } else { var.sqrt() };

        let y_norm: Vec<f64> = y.iter().map(|v| (v - self.y_mean) / self.y_std).collect();

        // Build kernel matrix K
        let mut k_mat = self.kernel_matrix(x, x);

        // Add noise: K += σ_n² I
        let noise_var = self.config.noise_std * self.config.noise_std;
        for i in 0..n {
            k_mat[i][i] += noise_var;
        }

        // Cholesky: K + σ_n² I = L Lᵀ
        let l = self.cholesky(&k_mat)?;
        self.chol_l = l.clone();

        // alpha = K⁻¹ y = L⁻ᵀ (L⁻¹ y)
        let v = self.solve_lower_triangular(&l, &y_norm)?;
        let alpha = self.solve_lower_triangular_transpose(&l, &v)?;
        self.alpha = alpha;

        // K_inv = (L Lᵀ)⁻¹
        self.k_inv = self.invert_via_cholesky(&k_mat)?;

        self.fitted = true;
        Ok(())
    }

    /// Predict mean and standard deviation for each test point.
    pub fn predict(&self, x_test: &[Vec<f64>]) -> Result<Vec<(f64, f64)>> {
        if !self.fitted {
            return Err(TensorError::InvalidOperation {
                operation: "GaussianProcess::predict".to_string(),
                reason: "GP must be fitted before prediction".to_string(),
                context: None,
            });
        }

        // k_star: (n_test, n_train)
        let k_star = self.kernel_matrix(x_test, &self.x_train);

        // k_star_star: prior variance at test points
        let k_ss_diag: Vec<f64> = x_test.iter().map(|x| self.kernel(x, x)).collect();

        let n_test = x_test.len();
        let mut results = Vec::with_capacity(n_test);

        for i in 0..n_test {
            // posterior mean: k_star[i] @ alpha
            let mean_norm: f64 = k_star[i]
                .iter()
                .zip(self.alpha.iter())
                .map(|(ks, a)| ks * a)
                .sum();
            let mean = mean_norm * self.y_std + self.y_mean;

            // posterior variance: k_ss - k_star[i]ᵀ K_inv k_star[i]
            // v = L⁻¹ k_star[i]
            let v = self.solve_lower_triangular(&self.chol_l, &k_star[i])?;
            let var_reduction: f64 = v.iter().map(|vi| vi * vi).sum();
            let var = (k_ss_diag[i] - var_reduction).max(0.0);
            let std = (var * self.y_std * self.y_std).sqrt();

            results.push((mean, std));
        }

        Ok(results)
    }

    /// Compute log marginal likelihood: −½ yᵀ K⁻¹ y − ½ log|K| − n/2 log(2π)
    ///
    /// Uses the Cholesky factorization for numerical stability.
    pub fn log_marginal_likelihood(&self) -> Result<f64> {
        if !self.fitted {
            return Err(TensorError::InvalidOperation {
                operation: "GaussianProcess::log_marginal_likelihood".to_string(),
                reason: "GP must be fitted first".to_string(),
                context: None,
            });
        }
        let n = self.x_train.len();
        let y_norm: Vec<f64> = self
            .y_train
            .iter()
            .map(|v| (v - self.y_mean) / self.y_std)
            .collect();

        // Data fit term: −½ αᵀ y_norm  (α = K⁻¹ y)
        let data_fit: f64 = -0.5
            * self
                .alpha
                .iter()
                .zip(y_norm.iter())
                .map(|(a, y)| a * y)
                .sum::<f64>();

        // Log determinant term: −½ sum log(L_ii)² = −sum log(L_ii)
        let log_det: f64 = self
            .chol_l
            .iter()
            .enumerate()
            .map(|(i, row)| {
                if row[i] > 0.0 {
                    row[i].ln()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .sum();
        let complexity = -log_det;

        // Normalization constant
        let norm_const = -0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln();

        Ok(data_fit + complexity + norm_const)
    }

    /// Compute kernel matrix K\[i,j\] = k(x1\[i\], x2\[j\]).
    pub fn kernel_matrix(&self, x1: &[Vec<f64>], x2: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x1.iter()
            .map(|xi| x2.iter().map(|xj| self.kernel(xi, xj)).collect())
            .collect()
    }

    /// Evaluate the kernel function between two points.
    pub fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let sq_dist: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        let dist = sq_dist.sqrt();

        match &self.config.kernel {
            KernelType::SquaredExponential {
                length_scale,
                signal_std,
            } => {
                let l = *length_scale;
                let s = *signal_std;
                s * s * (-sq_dist / (2.0 * l * l)).exp()
            }
            KernelType::Matern52 {
                length_scale,
                signal_std,
            } => {
                let l = *length_scale;
                let s = *signal_std;
                let r = dist / l;
                let sqrt5_r = 5.0_f64.sqrt() * r;
                s * s * (1.0 + sqrt5_r + 5.0 * r * r / 3.0) * (-sqrt5_r).exp()
            }
            KernelType::Periodic {
                length_scale,
                period,
                signal_std,
            } => {
                let l = *length_scale;
                let p = *period;
                let s = *signal_std;
                let sin_val = (std::f64::consts::PI * dist / p).sin();
                s * s * (-2.0 * sin_val * sin_val / (l * l)).exp()
            }
        }
    }

    /// Cholesky decomposition: returns lower-triangular L such that A = L Lᵀ.
    /// Uses the Cholesky-Banachiewicz algorithm with diagonal regularization.
    pub fn cholesky(&self, matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = matrix.len();
        let mut l = vec![vec![0.0_f64; n]; n];

        for i in 0..n {
            for j in 0..=i {
                let sum: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
                if i == j {
                    let d = matrix[i][i] - sum;
                    if d < 0.0 {
                        return Err(TensorError::NumericalError {
                            operation: "GaussianProcess::cholesky".to_string(),
                            details: format!(
                                "Matrix is not positive definite at index {}: d={:.3e}",
                                i, d
                            ),
                            suggestions: vec![
                                "Increase noise_std".to_string(),
                                "Check for duplicate training points".to_string(),
                            ],
                            context: None,
                        });
                    }
                    l[i][j] = d.max(0.0).sqrt();
                } else {
                    let lj = l[j][j];
                    if lj.abs() < 1e-15 {
                        l[i][j] = 0.0;
                    } else {
                        l[i][j] = (matrix[i][j] - sum) / lj;
                    }
                }
            }
        }
        Ok(l)
    }

    /// Forward substitution: solve L x = b for lower-triangular L.
    pub fn solve_lower_triangular(&self, l: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
        let n = l.len();
        if b.len() != n {
            return Err(TensorError::ShapeMismatch {
                operation: "solve_lower_triangular".to_string(),
                expected: format!("b.len()={}", n),
                got: format!("{}", b.len()),
                context: None,
            });
        }
        let mut x = vec![0.0_f64; n];
        for i in 0..n {
            let sum: f64 = (0..i).map(|j| l[i][j] * x[j]).sum();
            let diag = l[i][i];
            if diag.abs() < 1e-15 {
                x[i] = 0.0;
            } else {
                x[i] = (b[i] - sum) / diag;
            }
        }
        Ok(x)
    }

    /// Back-substitution: solve Lᵀ x = b for lower-triangular L.
    pub fn solve_lower_triangular_transpose(&self, l: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
        let n = l.len();
        if b.len() != n {
            return Err(TensorError::ShapeMismatch {
                operation: "solve_lower_triangular_transpose".to_string(),
                expected: format!("b.len()={}", n),
                got: format!("{}", b.len()),
                context: None,
            });
        }
        let mut x = vec![0.0_f64; n];
        for i in (0..n).rev() {
            let sum: f64 = ((i + 1)..n).map(|j| l[j][i] * x[j]).sum();
            let diag = l[i][i];
            if diag.abs() < 1e-15 {
                x[i] = 0.0;
            } else {
                x[i] = (b[i] - sum) / diag;
            }
        }
        Ok(x)
    }

    /// Invert a positive definite matrix via its Cholesky factorization.
    ///
    /// Solves K X = I column-by-column.
    pub fn invert_via_cholesky(&self, matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = matrix.len();
        let l = self.cholesky(matrix)?;
        let mut inv = vec![vec![0.0_f64; n]; n];

        for col in 0..n {
            let mut e = vec![0.0_f64; n];
            e[col] = 1.0;
            let v = self.solve_lower_triangular(&l, &e)?;
            let col_x = self.solve_lower_triangular_transpose(&l, &v)?;
            for row in 0..n {
                inv[row][col] = col_x[row];
            }
        }
        Ok(inv)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Acquisition functions
// ─────────────────────────────────────────────────────────────────────────────

/// Acquisition function to decide where to evaluate next.
#[derive(Debug, Clone)]
pub enum AcquisitionFunction {
    /// Upper Confidence Bound: μ(x) + κ·σ(x)
    UpperConfidenceBound { kappa: f64 },
    /// Expected Improvement with exploration parameter ξ.
    ExpectedImprovement { xi: f64 },
    /// Probability of Improvement with exploration parameter ξ.
    ProbabilityOfImprovement { xi: f64 },
    /// Thompson Sampling: draw from GP posterior.
    ThompsonSampling,
}

/// Result of acquisition function evaluation.
#[derive(Debug, Clone)]
pub struct AcquisitionResult {
    pub value: f64,
    pub point: Vec<f64>,
}

impl AcquisitionFunction {
    /// Evaluate the acquisition function at a point with given GP statistics.
    ///
    /// `rng` is used only for Thompson Sampling.
    pub fn evaluate(&self, mean: f64, std: f64, best_y: f64, rng: &mut StdRng) -> f64 {
        match self {
            Self::UpperConfidenceBound { kappa } => mean + kappa * std,
            Self::ExpectedImprovement { xi } => Self::expected_improvement(mean, std, best_y, *xi),
            Self::ProbabilityOfImprovement { xi } => {
                if std < 1e-10 {
                    if mean > best_y + xi {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    let z = (mean - best_y - xi) / std;
                    Self::normal_cdf(z)
                }
            }
            Self::ThompsonSampling => {
                // Sample from N(mean, std²) using Box-Muller
                let u1: f64 = (rng.random::<f64>()).max(1e-10);
                let u2: f64 = rng.random::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                mean + std * z
            }
        }
    }

    /// Expected Improvement: EI(x) = (μ(x) − y* − ξ)·Φ(Z) + σ(x)·φ(Z)
    /// where Z = (μ(x) − y* − ξ) / σ(x).
    fn expected_improvement(mean: f64, std: f64, best_y: f64, xi: f64) -> f64 {
        if std < 1e-10 {
            return 0.0;
        }
        let improvement = mean - best_y - xi;
        let z = improvement / std;
        improvement * Self::normal_cdf(z) + std * Self::normal_pdf(z)
    }

    /// Normal CDF approximation using Abramowitz & Stegun 26.2.17.
    pub fn normal_cdf(z: f64) -> f64 {
        if z > 8.0 {
            return 1.0;
        }
        if z < -8.0 {
            return 0.0;
        }
        // Use rational approximation for |z| <= 8
        let t = 1.0 / (1.0 + 0.2316419 * z.abs());
        let poly = t
            * (0.319_381_53
                + t * (-0.356_563_782
                    + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
        let phi = Self::normal_pdf(z);
        let cdf_neg = phi * poly;
        if z >= 0.0 {
            1.0 - cdf_neg
        } else {
            cdf_neg
        }
    }

    /// Normal PDF: φ(z) = exp(−z²/2) / √(2π)
    pub fn normal_pdf(z: f64) -> f64 {
        (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Search space
// ─────────────────────────────────────────────────────────────────────────────

/// Continuous search space defined by per-dimension (lower, upper) bounds.
///
/// Named `BayesSearchSpace` to avoid conflict with `nas::SearchSpace`.
#[derive(Debug, Clone)]
pub struct BayesSearchSpace {
    /// Per-dimension (lower, upper) bounds.
    pub bounds: Vec<(f64, f64)>,
    /// Dimensionality of the space.
    pub n_dims: usize,
}

impl BayesSearchSpace {
    /// Create a new search space from (lower, upper) bound pairs.
    pub fn new(bounds: Vec<(f64, f64)>) -> Result<Self> {
        if bounds.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "BayesSearchSpace::new".to_string(),
                reason: "bounds must be non-empty".to_string(),
                context: None,
            });
        }
        for (i, &(lo, hi)) in bounds.iter().enumerate() {
            if lo >= hi {
                return Err(TensorError::InvalidArgument {
                    operation: "BayesSearchSpace::new".to_string(),
                    reason: format!("bound[{}]: lower ({}) must be < upper ({})", i, lo, hi),
                    context: None,
                });
            }
        }
        let n_dims = bounds.len();
        Ok(Self { bounds, n_dims })
    }

    /// Draw a uniform random sample within bounds.
    pub fn random_sample(&self, rng: &mut StdRng) -> Vec<f64> {
        self.bounds
            .iter()
            .map(|&(lo, hi)| {
                let u: f64 = rng.random();
                lo + u * (hi - lo)
            })
            .collect()
    }

    /// Latin Hypercube Sampling: divide each dimension into `n` strata and
    /// take one sample per stratum, then permute across dimensions.
    pub fn latin_hypercube_sample(&self, n: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
        let d = self.n_dims;
        // For each dimension, generate n stratified samples
        let mut lhs = vec![vec![0.0_f64; d]; n];

        for dim in 0..d {
            let (lo, hi) = self.bounds[dim];
            let range = hi - lo;
            // Stratified positions [0, n)
            let mut positions: Vec<usize> = (0..n).collect();
            // Fisher-Yates shuffle
            for i in (1..n).rev() {
                let j: usize = (rng.random::<f64>() * (i + 1) as f64) as usize;
                let j = j.min(i);
                positions.swap(i, j);
            }
            for (row, &pos) in positions.iter().enumerate() {
                let u: f64 = rng.random();
                lhs[row][dim] = lo + ((pos as f64 + u) / n as f64) * range;
            }
        }
        lhs
    }

    /// Clamp a point to the search space bounds.
    pub fn clamp(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(self.bounds.iter())
            .map(|(&xi, &(lo, hi))| xi.clamp(lo, hi))
            .collect()
    }

    /// Check whether a point lies within the bounds.
    pub fn contains(&self, x: &[f64]) -> bool {
        x.iter()
            .zip(self.bounds.iter())
            .all(|(&xi, &(lo, hi))| xi >= lo && xi <= hi)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Bayesian Optimization
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Bayesian Optimization loop.
#[derive(Debug, Clone)]
pub struct BayesOptConfig {
    /// Number of initial random evaluations (Latin Hypercube).
    pub n_initial: usize,
    /// Number of BO iterations after initialization.
    pub n_iterations: usize,
    /// Acquisition function to guide sampling.
    pub acquisition: AcquisitionFunction,
    /// Gaussian Process surrogate configuration.
    pub gp_config: GpConfig,
    /// Number of random candidates for acquisition maximization.
    pub n_candidates: usize,
    /// Random seed.
    pub seed: u64,
}

/// A single observation recorded during optimization.
#[derive(Debug, Clone)]
pub struct ObservationRecord {
    /// The point evaluated.
    pub x: Vec<f64>,
    /// Objective value at x.
    pub y: f64,
    /// Iteration index (0 = initialization phase).
    pub iteration: usize,
}

/// Final result returned by the optimizer.
#[derive(Debug, Clone)]
pub struct BayesOptResult {
    /// Best point found.
    pub best_x: Vec<f64>,
    /// Best objective value.
    pub best_y: f64,
    /// All recorded observations.
    pub observations: Vec<ObservationRecord>,
    /// Total number of objective evaluations.
    pub n_evaluations: usize,
}

/// Bayesian Optimizer wrapping a GP surrogate and an acquisition function.
pub struct BayesianOptimizer {
    config: BayesOptConfig,
    space: BayesSearchSpace,
    gp: GaussianProcess,
    observations: Vec<ObservationRecord>,
    rng: StdRng,
    best_y: f64,
    best_x: Vec<f64>,
    /// If true, minimize the objective; if false, maximize.
    minimize: bool,
}

impl BayesianOptimizer {
    /// Create a new Bayesian Optimizer.
    pub fn new(config: BayesOptConfig, space: BayesSearchSpace, minimize: bool) -> Self {
        let seed = config.seed;
        let gp = GaussianProcess::new(config.gp_config.clone());
        let init_best = if minimize {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        Self {
            config,
            space,
            gp,
            observations: Vec::new(),
            rng: StdRng::seed_from_u64(seed),
            best_y: init_best,
            best_x: Vec::new(),
            minimize,
        }
    }

    /// Run the full optimization loop with `objective: &[f64] -> f64`.
    pub fn optimize<F>(&mut self, objective: F) -> Result<BayesOptResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n_initial = self.config.n_initial;
        let n_iterations = self.config.n_iterations;

        // Phase 1: Latin Hypercube initialization
        let init_points = self.space.latin_hypercube_sample(n_initial, &mut self.rng);
        for x in init_points {
            let y_raw = objective(&x);
            self.register_internal(x, y_raw, 0)?;
        }

        // Phase 2: BO iterations
        for iter in 1..=n_iterations {
            // Fit GP on all observations
            let (xs, ys) = self.observation_data();
            self.gp.fit(&xs, &ys)?;

            // Suggest next point
            let x_next = {
                let gp_clone = self.gp.clone();
                let mut rng_clone = StdRng::seed_from_u64(self.rng.random::<u64>());
                self.maximize_acquisition(&gp_clone, &mut rng_clone)?
            };

            let y_raw = objective(&x_next);
            self.register_internal(x_next, y_raw, iter)?;
        }

        Ok(BayesOptResult {
            best_x: self.best_x.clone(),
            best_y: self.best_y,
            observations: self.observations.clone(),
            n_evaluations: self.observations.len(),
        })
    }

    /// Suggest the next point to evaluate (without running the objective).
    ///
    /// Requires at least `n_initial` observations already registered.
    pub fn suggest(&mut self) -> Result<Vec<f64>> {
        if self.observations.is_empty() {
            // Cold start: random sample
            return Ok(self.space.random_sample(&mut self.rng));
        }
        let (xs, ys) = self.observation_data();
        self.gp.fit(&xs, &ys)?;
        let gp_clone = self.gp.clone();
        let mut rng_clone = StdRng::seed_from_u64(self.rng.random::<u64>());
        self.maximize_acquisition(&gp_clone, &mut rng_clone)
    }

    /// Register a manual observation (for use in external optimization loops).
    pub fn register(&mut self, x: Vec<f64>, y: f64) -> Result<()> {
        let iter = self.observations.len();
        self.register_internal(x, y, iter)
    }

    // Internal registration: stores observation and updates best.
    fn register_internal(&mut self, x: Vec<f64>, y_raw: f64, iteration: usize) -> Result<()> {
        // For minimize, negate so internally we always maximize
        let y_for_acq = if self.minimize { -y_raw } else { y_raw };

        let improved = if self.best_x.is_empty() {
            true
        } else if self.minimize {
            y_raw < self.best_y
        } else {
            y_raw > self.best_y
        };

        if improved {
            self.best_y = y_raw;
            self.best_x = x.clone();
        }

        self.observations.push(ObservationRecord {
            x,
            y: y_for_acq,
            iteration,
        });
        Ok(())
    }

    // Collect (x_train, y_train) for GP fitting (internally normalized direction).
    fn observation_data(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let xs: Vec<Vec<f64>> = self.observations.iter().map(|o| o.x.clone()).collect();
        let ys: Vec<f64> = self.observations.iter().map(|o| o.y).collect();
        (xs, ys)
    }

    /// Maximize acquisition function by evaluating `n_candidates` random points.
    fn maximize_acquisition(&self, gp: &GaussianProcess, rng: &mut StdRng) -> Result<Vec<f64>> {
        let n = self.config.n_candidates;
        let best_acq_y: f64 = self
            .observations
            .iter()
            .map(|o| o.y)
            .fold(f64::NEG_INFINITY, f64::max);

        let candidates: Vec<Vec<f64>> = (0..n).map(|_| self.space.random_sample(rng)).collect();

        let preds = gp.predict(&candidates)?;

        let mut best_val = f64::NEG_INFINITY;
        let mut best_pt = candidates[0].clone();

        for (pt, (mean, std)) in candidates.into_iter().zip(preds.iter()) {
            let acq = self
                .config
                .acquisition
                .evaluate(*mean, *std, best_acq_y, rng);
            if acq > best_val {
                best_val = acq;
                best_pt = pt;
            }
        }
        Ok(best_pt)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. CMA-ES
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the CMA-ES optimizer.
#[derive(Debug, Clone)]
pub struct CmaEsConfig {
    /// Problem dimensionality.
    pub n_dims: usize,
    /// Population size λ.
    pub population_size: usize,
    /// Number of parents μ (best solutions used for update).
    pub n_parents: usize,
    /// Initial distribution mean.
    pub initial_mean: Vec<f64>,
    /// Initial step size σ.
    pub initial_sigma: f64,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on function values.
    pub tol_fun: f64,
    /// Convergence tolerance on step size.
    pub tol_x: f64,
    /// Random seed.
    pub seed: u64,
}

impl CmaEsConfig {
    /// Construct with Hansen (2016) defaults.
    pub fn new(n_dims: usize, initial_mean: Vec<f64>, seed: u64) -> Self {
        let lambda = 4 + (3.0 * (n_dims as f64).ln()) as usize;
        let mu = lambda / 2;
        Self {
            n_dims,
            population_size: lambda,
            n_parents: mu,
            initial_mean,
            initial_sigma: 0.5,
            max_iterations: 1000,
            tol_fun: 1e-11,
            tol_x: 1e-11,
            seed,
        }
    }
}

/// Mutable state of the CMA-ES optimizer.
#[derive(Debug, Clone)]
pub struct CmaEsState {
    /// Current distribution mean.
    pub mean: Vec<f64>,
    /// Current step size σ.
    pub sigma: f64,
    /// Covariance matrix C (n × n).
    pub cov_matrix: Vec<Vec<f64>>,
    /// Evolution path for step-size control.
    pub p_sigma: Vec<f64>,
    /// Evolution path for covariance matrix update.
    pub p_c: Vec<f64>,
    /// Current iteration index.
    pub iteration: usize,
    /// Best function value seen.
    pub best_y: f64,
    /// Best point seen.
    pub best_x: Vec<f64>,
    /// Eigenvalues D² of C (for sampling).
    pub eigenvalues: Vec<f64>,
    /// Eigenvector matrix B (columns are eigenvectors).
    pub eigenvectors: Vec<Vec<f64>>,
}

/// Final result returned by CMA-ES.
#[derive(Debug, Clone)]
pub struct CmaEsResult {
    /// Best point found.
    pub best_x: Vec<f64>,
    /// Best function value.
    pub best_y: f64,
    /// Total number of function evaluations.
    pub n_evaluations: usize,
    /// Total number of iterations performed.
    pub n_iterations: usize,
    /// Whether convergence was detected.
    pub converged: bool,
    /// Final step size.
    pub final_sigma: f64,
}

/// CMA-ES optimizer (Hansen 2016 formulation).
///
/// Maintains a multivariate normal distribution N(m, σ²C) and adapts it based
/// on the ranked fitness of sampled solutions.
pub struct CmaEs {
    config: CmaEsConfig,
    state: CmaEsState,
    rng: StdRng,
    // --- CMA-ES constants (computed once in new()) ---
    /// Normalized recombination weights w_i (sums to 1).
    weights: Vec<f64>,
    /// Effective selection mass μ_eff = 1/∑w_i².
    mu_eff: f64,
    /// Step-size control learning rate c_σ.
    c_sigma: f64,
    /// Step-size damping d_σ.
    d_sigma: f64,
    /// Covariance matrix cumulation rate c_c.
    c_c: f64,
    /// Rank-1 update learning rate c_1.
    c_1: f64,
    /// Rank-μ update learning rate c_μ.
    c_mu: f64,
    /// E[‖N(0,I)‖] ≈ √n·(1 − 1/(4n) + 1/(21n²)).
    chi_n: f64,
    /// Eigensystem update counter (updated every n_dims iterations for efficiency).
    eigen_update_counter: usize,
}

impl CmaEs {
    /// Create a new CMA-ES optimizer and compute all strategy constants.
    pub fn new(config: CmaEsConfig) -> Self {
        let n = config.n_dims;
        let lambda = config.population_size;
        let mu = config.n_parents;

        // --- Recombination weights ---
        let mu_half = mu as f64 + 0.5;
        let raw_weights: Vec<f64> = (0..mu)
            .map(|i| (mu_half - (i + 1) as f64 + 1.0).ln().max(0.0))
            .collect();
        let w_sum: f64 = raw_weights.iter().sum();
        let weights: Vec<f64> = raw_weights.iter().map(|w| w / w_sum).collect();

        // --- mu_eff ---
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // --- Strategy parameters (Hansen 2016) ---
        let c_sigma = (mu_eff + 2.0) / (n as f64 + mu_eff + 5.0);
        let d_sigma =
            1.0 + 2.0 * (0.0_f64.max(((mu_eff - 1.0) / (n as f64 + 1.0)).sqrt() - 1.0)) + c_sigma;
        let c_c = (4.0 + mu_eff / n as f64) / (n as f64 + 4.0 + 2.0 * mu_eff / n as f64);
        let c_1 = 2.0 / ((n as f64 + 1.3).powi(2) + mu_eff);
        let c_mu = (1.0 - c_1)
            .min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n as f64 + 2.0).powi(2) + mu_eff));
        let chi_n =
            (n as f64).sqrt() * (1.0 - 1.0 / (4.0 * n as f64) + 1.0 / (21.0 * (n as f64).powi(2)));

        // --- Initial state ---
        let mean = config.initial_mean.clone();
        let sigma = config.initial_sigma;

        // Identity covariance
        let mut cov = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            cov[i][i] = 1.0;
        }
        // Identity eigensystem
        let eigenvalues = vec![1.0_f64; n];
        let mut eigenvectors = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            eigenvectors[i][i] = 1.0;
        }

        let state = CmaEsState {
            mean: mean.clone(),
            sigma,
            cov_matrix: cov,
            p_sigma: vec![0.0_f64; n],
            p_c: vec![0.0_f64; n],
            iteration: 0,
            best_y: f64::INFINITY,
            best_x: mean,
            eigenvalues,
            eigenvectors,
        };

        let seed = config.seed;
        Self {
            config,
            state,
            rng: StdRng::seed_from_u64(seed),
            weights,
            mu_eff,
            c_sigma,
            d_sigma,
            c_c,
            c_1,
            c_mu,
            chi_n,
            eigen_update_counter: 0,
        }
    }

    /// Run the full optimization loop. Minimizes the objective.
    pub fn optimize<F>(&mut self, objective: F) -> Result<CmaEsResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let max_iter = self.config.max_iterations;
        let mut n_evals = 0usize;

        for _iter in 0..max_iter {
            let best = self.step(&objective)?;
            n_evals += self.config.population_size;

            if self.converged() {
                return Ok(CmaEsResult {
                    best_x: self.state.best_x.clone(),
                    best_y: self.state.best_y,
                    n_evaluations: n_evals,
                    n_iterations: self.state.iteration,
                    converged: true,
                    final_sigma: self.state.sigma,
                });
            }

            let _ = best; // suppress unused warning
        }

        Ok(CmaEsResult {
            best_x: self.state.best_x.clone(),
            best_y: self.state.best_y,
            n_evaluations: n_evals,
            n_iterations: self.state.iteration,
            converged: false,
            final_sigma: self.state.sigma,
        })
    }

    /// Perform a single CMA-ES iteration: sample → evaluate → update.
    ///
    /// Returns the best fitness in this generation.
    pub fn step<F>(&mut self, objective: &F) -> Result<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let population = self.sample_population()?;

        // Evaluate all individuals
        let mut fitness: Vec<(f64, usize)> = population
            .iter()
            .enumerate()
            .map(|(i, x)| (objective(x), i))
            .collect();

        // Sort ascending (minimize)
        fitness.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Update best
        if fitness[0].0 < self.state.best_y {
            self.state.best_y = fitness[0].0;
            self.state.best_x = population[fitness[0].1].clone();
        }

        // Reorder population by rank
        let ranked: Vec<Vec<f64>> = fitness
            .iter()
            .map(|(_, idx)| population[*idx].clone())
            .collect();

        let fitness_vals: Vec<f64> = fitness.iter().map(|(f, _)| *f).collect();

        self.update_distribution(&ranked, &fitness_vals)?;
        self.state.iteration += 1;

        Ok(fitness[0].0)
    }

    /// Sample λ points from N(mean, σ² C).
    pub fn sample_population(&mut self) -> Result<Vec<Vec<f64>>> {
        let lambda = self.config.population_size;
        let mut pop = Vec::with_capacity(lambda);

        for _ in 0..lambda {
            let mut sample_rng = StdRng::seed_from_u64(self.rng.random::<u64>());
            let z = self.sample_from_normal(&mut sample_rng)?;

            let x: Vec<f64> = self
                .state
                .mean
                .iter()
                .zip(z.iter())
                .map(|(m, zi)| m + self.state.sigma * zi)
                .collect();
            pop.push(x);
        }
        Ok(pop)
    }

    /// Update mean, evolution paths, step size, and covariance matrix.
    pub fn update_distribution(&mut self, population: &[Vec<f64>], _fitness: &[f64]) -> Result<()> {
        let n = self.config.n_dims;
        let mu = self.config.n_parents;

        if population.len() < mu {
            return Err(TensorError::InvalidArgument {
                operation: "CmaEs::update_distribution".to_string(),
                reason: format!("population size {} < n_parents {}", population.len(), mu),
                context: None,
            });
        }

        let old_mean = self.state.mean.clone();

        // New mean: weighted sum of best mu individuals
        let mut new_mean = vec![0.0_f64; n];
        for (i, w) in self.weights.iter().enumerate() {
            for d in 0..n {
                new_mean[d] += w * population[i][d];
            }
        }

        // Step (new_mean - old_mean) / sigma in original coordinates
        let step: Vec<f64> = new_mean
            .iter()
            .zip(old_mean.iter())
            .map(|(nm, om)| (nm - om) / self.state.sigma)
            .collect();

        // C^{-1/2} step = B D^{-1} Bᵀ step
        let c_invsqrt_step = self.cov_invsqrt_mult(&step)?;

        // Update evolution path p_sigma
        let ps_factor = (1.0 - self.c_sigma).sqrt();
        let ps_update_factor = (self.mu_eff * self.c_sigma * (2.0 - self.c_sigma)).sqrt();
        let p_sigma_new: Vec<f64> = self
            .state
            .p_sigma
            .iter()
            .zip(c_invsqrt_step.iter())
            .map(|(ps, s)| ps_factor * ps + ps_update_factor * s)
            .collect();

        // Indicator h_sigma (stagnation check)
        let ps_norm: f64 = p_sigma_new.iter().map(|v| v * v).sum::<f64>().sqrt();
        let threshold = (1.4 + 2.0 / (n as f64 + 1.0)) * self.chi_n;
        let h_sigma = if ps_norm
            / (1.0 - (1.0 - self.c_sigma).powi(2 * (self.state.iteration + 1) as i32)).sqrt()
            < threshold
        {
            1.0
        } else {
            0.0
        };

        // Update evolution path p_c
        let pc_factor = 1.0 - self.c_c;
        let pc_update_factor = h_sigma * (self.mu_eff * self.c_c * (2.0 - self.c_c)).sqrt();
        let p_c_new: Vec<f64> = self
            .state
            .p_c
            .iter()
            .zip(step.iter())
            .map(|(pc, s)| pc_factor * pc + pc_update_factor * s)
            .collect();

        // Rank-1 update outer product
        let rank1: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| p_c_new[i] * p_c_new[j]).collect())
            .collect();

        // Rank-mu update: sum_i w_i * y_i y_iᵀ  where y_i = (x_i - m_old)/sigma
        let mut rank_mu = vec![vec![0.0_f64; n]; n];
        for (i, w) in self.weights.iter().enumerate() {
            let y: Vec<f64> = population[i]
                .iter()
                .zip(old_mean.iter())
                .map(|(xi, mi)| (xi - mi) / self.state.sigma)
                .collect();
            for r in 0..n {
                for c in 0..n {
                    rank_mu[r][c] += w * y[r] * y[c];
                }
            }
        }

        // Update covariance
        let c_h_delta = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c);
        let base_factor = 1.0 - self.c_1 - self.c_mu + c_h_delta * self.c_1;
        for r in 0..n {
            for c in 0..n {
                self.state.cov_matrix[r][c] = base_factor * self.state.cov_matrix[r][c]
                    + self.c_1 * rank1[r][c]
                    + self.c_mu * rank_mu[r][c];
            }
        }

        // Enforce symmetry
        for r in 0..n {
            for c in 0..r {
                let avg = (self.state.cov_matrix[r][c] + self.state.cov_matrix[c][r]) / 2.0;
                self.state.cov_matrix[r][c] = avg;
                self.state.cov_matrix[c][r] = avg;
            }
        }

        // Update step size
        let sigma_update = (self.c_sigma / self.d_sigma * (ps_norm / self.chi_n - 1.0)).exp();
        self.state.sigma *= sigma_update;
        self.state.sigma = self.state.sigma.max(1e-20);

        // Commit updates
        self.state.mean = new_mean;
        self.state.p_sigma = p_sigma_new;
        self.state.p_c = p_c_new;

        // Update eigensystem every n_dims iterations (or each iteration for small n)
        let update_freq = self.config.n_dims.max(1);
        self.eigen_update_counter += 1;
        if self.eigen_update_counter >= update_freq {
            self.eigen_update_counter = 0;
            self.update_eigensystem()?;
        }

        Ok(())
    }

    /// Compute C^{-1/2} v = B D^{-1} Bᵀ v using the current eigensystem.
    fn cov_invsqrt_mult(&self, v: &[f64]) -> Result<Vec<f64>> {
        let n = self.config.n_dims;
        // Bᵀ v
        let bt_v: Vec<f64> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| self.state.eigenvectors[j][i] * v[j])
                    .sum::<f64>()
            })
            .collect();
        // D^{-1} (Bᵀ v)
        let d_inv_bt_v: Vec<f64> = bt_v
            .iter()
            .zip(self.state.eigenvalues.iter())
            .map(|(bi, &ev)| {
                let d = ev.max(1e-20).sqrt();
                bi / d
            })
            .collect();
        // B (D^{-1} Bᵀ v)
        let result: Vec<f64> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| self.state.eigenvectors[i][j] * d_inv_bt_v[j])
                    .sum::<f64>()
            })
            .collect();
        Ok(result)
    }

    /// Compute C^{1/2} v = B D Bᵀ v for sampling.
    fn cov_sqrt_mult(&self, v: &[f64]) -> Result<Vec<f64>> {
        let n = self.config.n_dims;
        let bt_v: Vec<f64> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| self.state.eigenvectors[j][i] * v[j])
                    .sum::<f64>()
            })
            .collect();
        let d_bt_v: Vec<f64> = bt_v
            .iter()
            .zip(self.state.eigenvalues.iter())
            .map(|(bi, &ev)| bi * ev.max(0.0).sqrt())
            .collect();
        let result: Vec<f64> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| self.state.eigenvectors[i][j] * d_bt_v[j])
                    .sum::<f64>()
            })
            .collect();
        Ok(result)
    }

    /// Update the eigendecomposition of C using the Jacobi method.
    ///
    /// Implements the Jacobi eigendecomposition for symmetric matrices.
    /// C = B D² Bᵀ where D² contains eigenvalues and B is orthonormal.
    pub fn update_eigensystem(&mut self) -> Result<()> {
        let n = self.config.n_dims;

        // Copy C into a working matrix
        let mut a = self.state.cov_matrix.clone();
        // Start with identity for eigenvectors
        let mut v = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            v[i][i] = 1.0;
        }

        // Jacobi iterations (max sweeps = 100 * n)
        let max_sweeps = 100 * n;
        for _ in 0..max_sweeps {
            // Find off-diagonal element with largest absolute value
            let mut max_off = 0.0_f64;
            let mut p = 0usize;
            let mut q = 1usize;
            for i in 0..n {
                for j in (i + 1)..n {
                    if a[i][j].abs() > max_off {
                        max_off = a[i][j].abs();
                        p = i;
                        q = j;
                    }
                }
            }

            // Converged when off-diagonal elements are negligible
            if max_off < 1e-14 {
                break;
            }

            // Compute Jacobi rotation angle
            let app = a[p][p];
            let aqq = a[q][q];
            let apq = a[p][q];

            let tau = (aqq - app) / (2.0 * apq);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let cos = 1.0 / (1.0 + t * t).sqrt();
            let sin = t * cos;

            // Apply Givens rotation to rows/columns p, q
            a[p][p] = app - t * apq;
            a[q][q] = aqq + t * apq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;

            for r in 0..n {
                if r != p && r != q {
                    let arp = a[r][p];
                    let arq = a[r][q];
                    a[r][p] = cos * arp - sin * arq;
                    a[p][r] = a[r][p];
                    a[r][q] = sin * arp + cos * arq;
                    a[q][r] = a[r][q];
                }
                // Update eigenvector matrix
                let vrp = v[r][p];
                let vrq = v[r][q];
                v[r][p] = cos * vrp - sin * vrq;
                v[r][q] = sin * vrp + cos * vrq;
            }
        }

        // Eigenvalues are the diagonal of A after convergence
        let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i].max(1e-20)).collect();

        // Sort eigenvalues in descending order and permute eigenvectors accordingly
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| {
            eigenvalues[j]
                .partial_cmp(&eigenvalues[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_eigenvalues: Vec<f64> = order.iter().map(|&i| eigenvalues[i]).collect();
        let mut sorted_eigenvectors = vec![vec![0.0_f64; n]; n];
        for (new_col, &old_col) in order.iter().enumerate() {
            for row in 0..n {
                sorted_eigenvectors[row][new_col] = v[row][old_col];
            }
        }

        eigenvalues = sorted_eigenvalues;
        self.state.eigenvalues = eigenvalues;
        self.state.eigenvectors = sorted_eigenvectors;

        Ok(())
    }

    /// Sample z ~ N(0, C) using z = B √D u, u ~ N(0, I).
    pub fn sample_from_normal(&self, rng: &mut StdRng) -> Result<Vec<f64>> {
        let n = self.config.n_dims;
        // Sample u ~ N(0, I) via Box-Muller
        let mut u = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let r1: f64 = (rng.random::<f64>()).max(1e-300);
            let r2: f64 = rng.random::<f64>();
            let mag = (-2.0 * r1.ln()).sqrt();
            let angle = 2.0 * std::f64::consts::PI * r2;
            u.push(mag * angle.cos());
            i += 1;
            if i < n {
                u.push(mag * angle.sin());
                i += 1;
            }
        }
        u.truncate(n);

        // z = B √D u
        self.cov_sqrt_mult(&u)
    }

    /// Check whether convergence criteria are met.
    pub fn converged(&self) -> bool {
        self.state.sigma < self.config.tol_x
    }

    /// Get a reference to the current state.
    pub fn state(&self) -> &CmaEsState {
        &self.state
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Multi-Fidelity Optimization
// ─────────────────────────────────────────────────────────────────────────────

/// A fidelity level for multi-fidelity optimization.
#[derive(Debug, Clone)]
pub struct FidelityLevel {
    /// Unique identifier (typically 0 = lowest, N = highest).
    pub id: usize,
    /// Relative cost (1.0 = full / highest fidelity).
    pub cost: f64,
    /// Systematic bias versus full fidelity (for simulation).
    pub bias: f64,
}

/// Configuration for multi-fidelity optimization.
#[derive(Debug, Clone)]
pub struct MultiFidelityConfig {
    /// Ordered fidelity levels (index 0 = cheapest).
    pub fidelity_levels: Vec<FidelityLevel>,
    /// Total budget in cost units.
    pub budget: f64,
    /// Base BO configuration (reused per fidelity level).
    pub base_config: BayesOptConfig,
}

/// Multi-Fidelity Optimizer.
///
/// Uses low-fidelity evaluations to narrow the search space, then evaluates
/// the most promising candidates at higher fidelity.
pub struct MultiFidelityOptimizer {
    config: MultiFidelityConfig,
    /// One BayesianOptimizer per fidelity level.
    optimizers: Vec<BayesianOptimizer>,
}

impl MultiFidelityOptimizer {
    /// Create a new multi-fidelity optimizer.
    ///
    /// Constructs one `BayesianOptimizer` per fidelity level, each seeded
    /// differently to avoid correlation.
    pub fn new(config: MultiFidelityConfig, space: BayesSearchSpace) -> Result<Self> {
        if config.fidelity_levels.is_empty() {
            return Err(TensorError::InvalidArgument {
                operation: "MultiFidelityOptimizer::new".to_string(),
                reason: "fidelity_levels must be non-empty".to_string(),
                context: None,
            });
        }
        if config.budget <= 0.0 {
            return Err(TensorError::InvalidArgument {
                operation: "MultiFidelityOptimizer::new".to_string(),
                reason: "budget must be positive".to_string(),
                context: None,
            });
        }

        let mut optimizers = Vec::with_capacity(config.fidelity_levels.len());
        for (idx, level) in config.fidelity_levels.iter().enumerate() {
            let mut bo_config = config.base_config.clone();
            bo_config.seed = config.base_config.seed.wrapping_add(level.id as u64 + 1);
            // Scale iterations by fidelity cost (more cheap, fewer expensive)
            let cost_factor = if level.cost > 0.0 {
                1.0 / level.cost
            } else {
                1.0
            };
            let scaled_iters =
                (config.base_config.n_iterations as f64 * cost_factor).min(500.0) as usize;
            bo_config.n_iterations = scaled_iters.max(1);
            let bo_space = space.clone();
            // All levels minimize
            optimizers.push(BayesianOptimizer::new(bo_config, bo_space, true));
            let _ = idx; // suppress unused warning
        }

        Ok(Self { config, optimizers })
    }

    /// Run multi-fidelity optimization.
    ///
    /// Strategy:
    /// 1. Warm up each fidelity level with cheap evaluations.
    /// 2. Identify the top candidates from the lowest fidelity.
    /// 3. Evaluate top candidates at progressively higher fidelities.
    pub fn optimize<F>(&mut self, objective: F) -> Result<BayesOptResult>
    where
        F: Fn(&[f64], usize) -> f64,
    {
        let n_levels = self.config.fidelity_levels.len();
        let mut remaining_budget = self.config.budget;

        let mut all_observations: Vec<ObservationRecord> = Vec::new();
        let mut global_best_y = f64::INFINITY;
        let mut global_best_x = Vec::new();

        // Phase 1: explore using cheapest fidelity
        let cheapest_id = self.config.fidelity_levels[0].id;
        let cheapest_cost = self.config.fidelity_levels[0].cost.max(1e-10);

        // Budget allocation: 60% for cheapest, rest shared among higher fidelities
        let cheap_budget = (self.config.budget * 0.6 / cheapest_cost) as usize;
        let cheap_budget = cheap_budget.max(self.config.base_config.n_initial);

        let level0_obj = |x: &[f64]| objective(x, cheapest_id);
        // Run cheapest-level optimizer
        let result0 = self.optimizers[0].optimize(level0_obj)?;
        let eval_cost = result0.n_evaluations as f64 * cheapest_cost;
        remaining_budget -= eval_cost;

        for obs in &result0.observations {
            all_observations.push(ObservationRecord {
                x: obs.x.clone(),
                y: obs.y,
                iteration: obs.iteration,
            });
        }

        // Collect best candidates from cheapest level
        let mut sorted_obs = result0.observations.clone();
        sorted_obs.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal));
        let top_k = (sorted_obs.len() / 4).clamp(1, 5);
        let candidates: Vec<Vec<f64>> =
            sorted_obs.iter().take(top_k).map(|o| o.x.clone()).collect();

        // Phase 2: evaluate top candidates at each higher fidelity
        for level_idx in 1..n_levels {
            if remaining_budget <= 0.0 {
                break;
            }
            let fid = self.config.fidelity_levels[level_idx].id;
            let cost = self.config.fidelity_levels[level_idx].cost.max(1e-10);

            for (cand_idx, x) in candidates.iter().enumerate() {
                if remaining_budget < cost {
                    break;
                }
                let y = objective(x, fid);
                remaining_budget -= cost;

                if y < global_best_y {
                    global_best_y = y;
                    global_best_x = x.clone();
                }

                all_observations.push(ObservationRecord {
                    x: x.clone(),
                    y,
                    iteration: level_idx * 1000 + cand_idx,
                });

                // Register with the corresponding optimizer
                self.optimizers[level_idx].register(x.clone(), y)?;
            }

            // Run additional BO at this fidelity with remaining budget
            if remaining_budget > cost * 2.0 {
                let fid_copy = fid;
                let obj_for_level = |x: &[f64]| objective(x, fid_copy);
                let extra_result = self.optimizers[level_idx].optimize(obj_for_level)?;
                let eval_cost = extra_result.n_evaluations as f64 * cost;
                remaining_budget -= eval_cost;

                if extra_result.best_y < global_best_y {
                    global_best_y = extra_result.best_y;
                    global_best_x = extra_result.best_x.clone();
                }
                for obs in &extra_result.observations {
                    all_observations.push(ObservationRecord {
                        x: obs.x.clone(),
                        y: obs.y,
                        iteration: obs.iteration + level_idx * 10000,
                    });
                }
            }
        }

        // If no high-fidelity evaluations ran, fall back to cheapest result
        if global_best_x.is_empty() {
            global_best_x = result0.best_x.clone();
            global_best_y = result0.best_y;
        }

        let n_evals = all_observations.len();
        Ok(BayesOptResult {
            best_x: global_best_x,
            best_y: global_best_y,
            observations: all_observations,
            n_evaluations: n_evals,
        })
    }
}

#[cfg(test)]
mod tests;
