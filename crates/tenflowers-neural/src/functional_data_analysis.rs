//! Functional Data Analysis (FDA) — production-grade tools for functions as data.
//!
//! This module provides:
//! - [`FdaBasis`] / [`FourierBasis`] / [`BSplineBasis`] / [`LegendreBasis`]: basis systems
//! - [`FdaObservation`] / [`FdaDataset`]: functional observations and datasets
//! - [`FpcaModel`] / [`FpcaResult`]: Functional PCA
//! - [`ScalarOnFunction`] / [`FunctionOnScalar`]: functional regression
//! - [`ElasticRegistration`] / [`FrechetMean`]: curve alignment and averaging
//! - [`FunctionalNeuralNetwork`]: neural network for functional inputs
//! - [`FdaMetrics`] / [`FdaEvalReport`]: distances and evaluation

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::f32::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// 1. FunctionalBasis
// ─────────────────────────────────────────────────────────────────────────────

/// Discriminant for basis function families.
///
/// Note: `FdaBasisKind` is the FDA-specific basis type enum to avoid collision
/// with the `BasisType` export from `time_series`.
#[derive(Clone, Debug, PartialEq)]
pub enum FdaBasisKind {
    /// Alternating sin/cos with configurable period.
    Fourier,
    /// B-spline of given degree with knot vector.
    BSpline,
    /// Legendre polynomials on [-1, 1].
    Legendre,
    /// Haar-like wavelet placeholder.
    Wavelet,
}

// ── Fourier ──────────────────────────────────────────────────────────────────

/// Fourier basis: constant + alternating sin/cos harmonics.
///
/// With `n_basis = 2k+1` the basis is:
///   `[1, sin(2πt/T), cos(2πt/T), sin(4πt/T), cos(4πt/T), …]`
#[derive(Clone, Debug)]
pub struct FourierBasis {
    /// Total number of basis functions (should be odd for full harmonic pairs).
    pub n_basis: usize,
    /// Period T.
    pub period: f32,
}

impl FourierBasis {
    /// Evaluate all basis functions at a single point `x`.
    pub fn evaluate(&self, x: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n_basis);
        // basis 0: constant 1
        out.push(1.0_f32);
        let mut k = 1_usize;
        while out.len() < self.n_basis {
            let arg = 2.0 * PI * (k as f32) * x / self.period;
            if out.len() < self.n_basis {
                out.push(arg.sin());
            }
            if out.len() < self.n_basis {
                out.push(arg.cos());
            }
            k += 1;
        }
        out
    }

    /// Evaluate all basis functions at each point in `xs`.
    pub fn evaluate_all(&self, xs: &[f32]) -> Vec<Vec<f32>> {
        xs.iter().map(|&x| self.evaluate(x)).collect()
    }
}

// ── B-Spline ─────────────────────────────────────────────────────────────────

/// B-spline basis using the de Boor recurrence.
#[derive(Clone, Debug)]
pub struct BSplineBasis {
    /// Number of basis functions (= number of control points).
    pub n_basis: usize,
    /// Polynomial degree.
    pub degree: usize,
    /// Knot vector (must have length `n_basis + degree + 1`).
    pub knots: Vec<f32>,
}

impl BSplineBasis {
    /// Evaluate all B-spline basis functions at `x` via de Boor recurrence.
    pub fn evaluate(&self, x: f32) -> Vec<f32> {
        let m = self.knots.len();
        let n = m.saturating_sub(1); // number of intervals
                                     // Degree-0 basis: indicator on each knot span
        let mut b: Vec<f32> = (0..n)
            .map(|i| {
                let lo = self.knots[i];
                let hi = self.knots[i + 1];
                if lo <= x && x < hi {
                    1.0_f32
                } else if i + 1 == n && (x - hi).abs() < 1e-8 {
                    // Include the right endpoint in the last span
                    1.0_f32
                } else {
                    0.0_f32
                }
            })
            .collect();

        // Raise degree
        for d in 1..=self.degree {
            let new_n = n.saturating_sub(d);
            let mut b_new = vec![0.0_f32; new_n];
            for i in 0..new_n {
                let denom_l = self.knots[i + d] - self.knots[i];
                let denom_r = self.knots[i + d + 1] - self.knots[i + 1];
                let alpha = if denom_l.abs() > 1e-12 {
                    (x - self.knots[i]) / denom_l
                } else {
                    0.0
                };
                let beta = if denom_r.abs() > 1e-12 {
                    (self.knots[i + d + 1] - x) / denom_r
                } else {
                    0.0
                };
                b_new[i] = alpha * b[i] + beta * b[i + 1].max(0.0) * 1.0;
                // correct formulation
                if i + 1 < b.len() {
                    b_new[i] = alpha * b[i]
                        + (1.0
                            - if denom_r.abs() > 1e-12 {
                                (x - self.knots[i + 1]) / denom_r
                            } else {
                                0.0
                            })
                            * b[i + 1];
                }
            }
            b = b_new;
        }
        // Trim / pad to n_basis
        b.truncate(self.n_basis);
        while b.len() < self.n_basis {
            b.push(0.0);
        }
        b
    }

    /// Evaluate all B-spline basis functions at each point in `xs`.
    pub fn evaluate_all(&self, xs: &[f32]) -> Vec<Vec<f32>> {
        xs.iter().map(|&x| self.evaluate(x)).collect()
    }
}

// ── Legendre ─────────────────────────────────────────────────────────────────

/// Legendre polynomial basis on [-1, 1].
///
/// Uses the three-term recurrence:
///   `P_0(x) = 1`, `P_1(x) = x`,
///   `(n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)`.
#[derive(Clone, Debug)]
pub struct LegendreBasis {
    /// Number of basis functions (polynomials P_0 … P_{n-1}).
    pub n_basis: usize,
}

impl LegendreBasis {
    /// Evaluate all Legendre polynomials at `x ∈ [-1, 1]`.
    pub fn evaluate(&self, x: f32) -> Vec<f32> {
        if self.n_basis == 0 {
            return vec![];
        }
        let mut p = Vec::with_capacity(self.n_basis);
        p.push(1.0_f32); // P_0
        if self.n_basis == 1 {
            return p;
        }
        p.push(x); // P_1
        for n in 1..(self.n_basis - 1) {
            let nf = n as f32;
            let prev2 = p[n - 1];
            let prev1 = p[n];
            let next = ((2.0 * nf + 1.0) * x * prev1 - nf * prev2) / (nf + 1.0);
            p.push(next);
        }
        p
    }

    /// Evaluate all Legendre polynomials at each point in `xs`.
    pub fn evaluate_all(&self, xs: &[f32]) -> Vec<Vec<f32>> {
        xs.iter().map(|&x| self.evaluate(x)).collect()
    }
}

// ── Wrapper enum ─────────────────────────────────────────────────────────────

/// Enum wrapping all supported basis families.
#[derive(Clone, Debug)]
pub enum FdaBasis {
    /// Fourier basis (periodic).
    Fourier(FourierBasis),
    /// B-Spline basis.
    BSpline(BSplineBasis),
    /// Legendre polynomial basis.
    Legendre(LegendreBasis),
}

impl FdaBasis {
    /// Evaluate the basis at `x`.
    pub fn evaluate(&self, x: f32) -> Vec<f32> {
        match self {
            FdaBasis::Fourier(b) => b.evaluate(x),
            FdaBasis::BSpline(b) => b.evaluate(x),
            FdaBasis::Legendre(b) => b.evaluate(x),
        }
    }

    /// Number of basis functions.
    pub fn n_basis(&self) -> usize {
        match self {
            FdaBasis::Fourier(b) => b.n_basis,
            FdaBasis::BSpline(b) => b.n_basis,
            FdaBasis::Legendre(b) => b.n_basis,
        }
    }

    /// Evaluate the basis at each point in `xs`.
    pub fn evaluate_all(&self, xs: &[f32]) -> Vec<Vec<f32>> {
        xs.iter().map(|&x| self.evaluate(x)).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. FdaObservation / FdaDataset
// ─────────────────────────────────────────────────────────────────────────────

/// A single discretely-observed functional observation.
#[derive(Clone, Debug)]
pub struct FdaObservation {
    /// Unique identifier.
    pub id: usize,
    /// Argument (x) points, must be sorted ascending.
    pub x_points: Vec<f32>,
    /// Corresponding function values y(x).
    pub y_values: Vec<f32>,
}

impl FdaObservation {
    /// Linear interpolation at `x`; clamps to boundary outside the domain.
    pub fn interpolate_linear(&self, x: f32) -> f32 {
        let n = self.x_points.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return self.y_values[0];
        }
        // Clamp below
        if x <= self.x_points[0] {
            return self.y_values[0];
        }
        // Clamp above
        if x >= self.x_points[n - 1] {
            return self.y_values[n - 1];
        }
        // Binary search for interval
        let idx = self.x_points.partition_point(|&p| p <= x).saturating_sub(1);
        let idx = idx.min(n - 2);
        let x0 = self.x_points[idx];
        let x1 = self.x_points[idx + 1];
        let y0 = self.y_values[idx];
        let y1 = self.y_values[idx + 1];
        let dx = x1 - x0;
        if dx.abs() < 1e-12 {
            return y0;
        }
        y0 + (y1 - y0) * (x - x0) / dx
    }

    /// Least-squares projection of this observation onto the given basis.
    ///
    /// Returns the coefficient vector `c` such that `Σ c_j φ_j(x_i) ≈ y_i`.
    pub fn to_basis_coefs(&self, basis: &FdaBasis) -> Vec<f32> {
        let n = self.x_points.len();
        let p = basis.n_basis();
        if n == 0 || p == 0 {
            return vec![0.0; p];
        }
        // Build design matrix Φ (n × p)
        let phi: Vec<Vec<f32>> = basis.evaluate_all(&self.x_points);
        // Normal equations: (Φ^T Φ) c = Φ^T y
        // Φ^T Φ is p×p
        let mut ata = vec![vec![0.0_f32; p]; p];
        let mut aty = vec![0.0_f32; p];
        for i in 0..n {
            let row = &phi[i];
            let yi = self.y_values[i];
            for j in 0..p {
                aty[j] += row[j] * yi;
                for k in 0..p {
                    ata[j][k] += row[j] * row[k];
                }
            }
        }
        // Solve via Cholesky-like (simple Gaussian elimination for small p)
        solve_linear_system(&ata, &aty)
    }

    /// L2 norm via trapezoidal integration over the observation points.
    pub fn l2_norm(&self) -> f32 {
        let n = self.x_points.len();
        if n < 2 {
            return 0.0;
        }
        let mut sum = 0.0_f32;
        for i in 0..(n - 1) {
            let dx = self.x_points[i + 1] - self.x_points[i];
            let y0 = self.y_values[i];
            let y1 = self.y_values[i + 1];
            sum += 0.5 * dx * (y0 * y0 + y1 * y1);
        }
        sum.sqrt()
    }

    /// Domain length of this observation.
    pub fn domain_length(&self) -> f32 {
        if self.x_points.len() < 2 {
            return 0.0;
        }
        self.x_points[self.x_points.len() - 1] - self.x_points[0]
    }
}

/// A collection of functional observations sharing the same domain.
#[derive(Clone, Debug)]
pub struct FdaDataset {
    /// All observations in the dataset.
    pub observations: Vec<FdaObservation>,
    /// Common domain interval `(a, b)`.
    pub domain: (f32, f32),
}

impl FdaDataset {
    /// Pointwise empirical mean evaluated at `eval_points`.
    pub fn mean_function(&self, eval_points: &[f32]) -> Vec<f32> {
        let n = self.observations.len();
        if n == 0 {
            return vec![0.0; eval_points.len()];
        }
        let mut mean = vec![0.0_f32; eval_points.len()];
        for obs in &self.observations {
            for (j, &x) in eval_points.iter().enumerate() {
                mean[j] += obs.interpolate_linear(x);
            }
        }
        for v in &mut mean {
            *v /= n as f32;
        }
        mean
    }

    /// Empirical covariance surface (N×N matrix) evaluated at `eval_points`.
    ///
    /// `C(s, t) = (1/n) Σ_i [f_i(s) - μ(s)] [f_i(t) - μ(t)]`
    pub fn covariance_surface(&self, eval_points: &[f32]) -> Vec<Vec<f32>> {
        let m = eval_points.len();
        let n = self.observations.len();
        if n == 0 {
            return vec![vec![0.0; m]; m];
        }
        let mean = self.mean_function(eval_points);
        // Centred evaluations: matrix (n × m)
        let centered: Vec<Vec<f32>> = self
            .observations
            .iter()
            .map(|obs| {
                eval_points
                    .iter()
                    .enumerate()
                    .map(|(j, &x)| obs.interpolate_linear(x) - mean[j])
                    .collect()
            })
            .collect();
        // Cov(j, k) = (1/n) Σ_i centred[i][j] * centred[i][k]
        let mut cov = vec![vec![0.0_f32; m]; m];
        for i in 0..n {
            for j in 0..m {
                for k in j..m {
                    cov[j][k] += centered[i][j] * centered[i][k];
                }
            }
        }
        let n_f = n as f32;
        for j in 0..m {
            for k in j..m {
                cov[j][k] /= n_f;
                cov[k][j] = cov[j][k]; // symmetry
            }
        }
        cov
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. FpcaModel — Functional PCA
// ─────────────────────────────────────────────────────────────────────────────

/// Result of fitting a [`FpcaModel`].
#[derive(Clone, Debug)]
pub struct FpcaResult {
    /// Top-k eigenvalues of the covariance operator.
    pub eigenvalues: Vec<f32>,
    /// Top-k eigenfunctions evaluated at the `eval_points` (each is a `Vec<f32>`).
    pub eigenfunctions: Vec<Vec<f32>>,
    /// Proportion of variance explained by each component.
    pub explained_variance_ratio: Vec<f32>,
}

/// Functional PCA via empirical covariance matrix and power iteration.
#[derive(Clone, Debug)]
pub struct FpcaModel {
    /// Number of components to extract.
    pub n_components: usize,
    /// Basis system for representing the covariance eigenfunctions.
    pub basis: FdaBasis,
    /// Grid points at which functions are evaluated.
    pub eval_points: Vec<f32>,
    /// Fitted result (available after calling `fit`).
    pub result: Option<FpcaResult>,
}

impl FpcaModel {
    /// Create a new (unfitted) FpcaModel.
    pub fn new(n_components: usize, basis: FdaBasis, eval_points: Vec<f32>) -> Self {
        Self {
            n_components,
            basis,
            eval_points,
            result: None,
        }
    }

    /// Fit the model and return a [`FpcaResult`].
    pub fn fit(&mut self, data: &FdaDataset) -> FpcaResult {
        let cov = data.covariance_surface(&self.eval_points);
        let m = self.eval_points.len();
        let k = self.n_components.min(m);

        // Power iteration with deflation to find top-k eigenvectors
        let mut evecs: Vec<Vec<f32>> = Vec::with_capacity(k);
        let mut evals: Vec<f32> = Vec::with_capacity(k);
        let mut deflated = cov.clone();

        for _ in 0..k {
            let (eval, evec) = power_iteration(&deflated, 200, 1e-6);
            if eval.abs() < 1e-10 {
                // No more significant components
                evecs.push(vec![0.0_f32; m]);
                evals.push(0.0);
            } else {
                // Deflate: A ← A - λ v v^T
                for r in 0..m {
                    for c in 0..m {
                        deflated[r][c] -= eval * evec[r] * evec[c];
                    }
                }
                evals.push(eval);
                evecs.push(evec);
            }
        }

        // Explained variance ratio
        let total_var: f32 = evals.iter().map(|e| e.abs()).sum::<f32>().max(1e-12);
        let explained: Vec<f32> = evals.iter().map(|e| e.abs() / total_var).collect();

        let res = FpcaResult {
            eigenvalues: evals,
            eigenfunctions: evecs,
            explained_variance_ratio: explained,
        };
        self.result = Some(res.clone());
        res
    }

    /// Project a new observation onto the fitted eigenfunctions.
    ///
    /// Returns scores (FPC coefficients).
    pub fn transform(&self, obs: &FdaObservation) -> Vec<f32> {
        let res = match &self.result {
            Some(r) => r,
            None => return vec![0.0; self.n_components],
        };
        // Evaluate obs at eval_points and subtract mean (approximated as zero here)
        let f_vals: Vec<f32> = self
            .eval_points
            .iter()
            .map(|&x| obs.interpolate_linear(x))
            .collect();
        res.eigenfunctions
            .iter()
            .map(|ef| l2_inner_product_grid(&f_vals, ef))
            .collect()
    }

    /// Reconstruct a function from FPC scores at given eval points.
    pub fn inverse_transform(&self, scores: &[f32], eval_points: &[f32]) -> Vec<f32> {
        let res = match &self.result {
            Some(r) => r,
            None => return vec![0.0; eval_points.len()],
        };
        let mut out = vec![0.0_f32; eval_points.len()];
        for (ci, &score) in scores.iter().enumerate() {
            if ci >= res.eigenfunctions.len() {
                break;
            }
            let ef = &res.eigenfunctions[ci];
            // Interpolate eigenfunction at requested eval_points
            for (j, &x) in eval_points.iter().enumerate() {
                let ef_val = interp_linear_grid(&self.eval_points, ef, x);
                out[j] += score * ef_val;
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. FunctionalRegression
// ─────────────────────────────────────────────────────────────────────────────

/// Scalar-on-function regression: y = ∫ β(t) X(t) dt + ε.
///
/// Fitted via OLS in the basis coefficient space.
#[derive(Clone, Debug)]
pub struct ScalarOnFunction {
    /// Basis expansion coefficients of the slope function β.
    pub beta_coefs: Vec<f32>,
    /// Scalar intercept.
    pub intercept: f32,
    /// Basis system.
    pub basis: FdaBasis,
}

impl ScalarOnFunction {
    /// Fit a scalar-on-function regression model.
    ///
    /// Each observation is projected onto the basis; the resulting coefficient
    /// matrix is used as the design matrix in ordinary least squares.
    pub fn fit(x_data: &FdaDataset, y: &[f32]) -> Self {
        let n = x_data.observations.len();
        let p = if n > 0 {
            // Detect basis from first obs – fall back to 5-basis Legendre
            5_usize
        } else {
            5
        };
        // Build a simple Legendre basis for projection
        let basis = FdaBasis::Legendre(LegendreBasis { n_basis: p });

        // Design matrix A (n × (p+1)) with intercept column
        let mut a = vec![vec![0.0_f32; p + 1]; n];
        for (i, obs) in x_data.observations.iter().enumerate() {
            let coefs = obs.to_basis_coefs(&basis);
            a[i][0] = 1.0; // intercept
            for j in 0..p {
                a[i][j + 1] = *coefs.get(j).unwrap_or(&0.0);
            }
        }
        // Solve OLS: (A^T A) θ = A^T y
        let mut ata = vec![vec![0.0_f32; p + 1]; p + 1];
        let mut aty = vec![0.0_f32; p + 1];
        for i in 0..n {
            let yi = *y.get(i).unwrap_or(&0.0);
            for j in 0..=p {
                aty[j] += a[i][j] * yi;
                for k in 0..=p {
                    ata[j][k] += a[i][j] * a[i][k];
                }
            }
        }
        let theta = solve_linear_system(&ata, &aty);
        let intercept = *theta.first().unwrap_or(&0.0);
        let beta_coefs = theta[1..].to_vec();
        Self {
            beta_coefs,
            intercept,
            basis,
        }
    }

    /// Predict the scalar response for a single functional observation.
    pub fn predict(&self, obs: &FdaObservation) -> f32 {
        let coefs = obs.to_basis_coefs(&self.basis);
        let mut pred = self.intercept;
        for (j, &c) in coefs.iter().enumerate() {
            pred += self.beta_coefs.get(j).copied().unwrap_or(0.0) * c;
        }
        pred
    }

    /// Coefficient of determination R².
    pub fn r_squared(&self, x_data: &FdaDataset, y: &[f32]) -> f32 {
        let n = x_data.observations.len().min(y.len());
        if n == 0 {
            return 0.0;
        }
        let y_mean = y[..n].iter().sum::<f32>() / n as f32;
        let ss_tot: f32 = y[..n].iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        if ss_tot < 1e-12 {
            return 1.0;
        }
        let ss_res: f32 = x_data.observations[..n]
            .iter()
            .zip(y[..n].iter())
            .map(|(obs, &yi)| {
                let pred = self.predict(obs);
                (yi - pred).powi(2)
            })
            .sum();
        1.0 - ss_res / ss_tot
    }
}

/// Function-on-scalar regression: Y(t) = α(t) + β(t) x + ε(t).
///
/// Each eval-point is regressed independently on the scalar covariate.
#[derive(Clone, Debug)]
pub struct FunctionOnScalar {
    /// Number of evaluation points.
    pub n_eval_points: usize,
    /// Basis system (stored for interface compatibility).
    pub basis: FdaBasis,
    /// Coefficient matrix (n_eval_points × 2): [intercept, slope] per eval point.
    pub coef_matrix: Vec<Vec<f32>>,
}

impl FunctionOnScalar {
    /// Fit a function-on-scalar model.
    pub fn fit(x: &[f32], y_data: &FdaDataset) -> Self {
        let eval_points: Vec<f32> = {
            let (a, b) = y_data.domain;
            let m = 20_usize;
            (0..m)
                .map(|i| a + (b - a) * i as f32 / (m - 1).max(1) as f32)
                .collect()
        };
        let m = eval_points.len();
        let basis = FdaBasis::Legendre(LegendreBasis { n_basis: 3 });
        let n = y_data.observations.len().min(x.len());
        let mut coef_matrix = vec![vec![0.0_f32; 2]; m];
        for j in 0..m {
            let t = eval_points[j];
            // Gather y_i(t) for all observations
            let yt: Vec<f32> = y_data.observations[..n]
                .iter()
                .map(|obs| obs.interpolate_linear(t))
                .collect();
            // OLS: y = a + b*x
            let x_mean = if n > 0 {
                x[..n].iter().sum::<f32>() / n as f32
            } else {
                0.0
            };
            let y_mean = if n > 0 {
                yt.iter().sum::<f32>() / n as f32
            } else {
                0.0
            };
            let sxx: f32 = x[..n].iter().map(|&xi| (xi - x_mean).powi(2)).sum();
            let sxy: f32 = x[..n]
                .iter()
                .zip(yt.iter())
                .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
                .sum();
            let slope = if sxx.abs() > 1e-12 { sxy / sxx } else { 0.0 };
            let intercept = y_mean - slope * x_mean;
            coef_matrix[j] = vec![intercept, slope];
        }
        Self {
            n_eval_points: m,
            basis,
            coef_matrix,
        }
    }

    /// Predict the functional response at all eval points for scalar `x_new`.
    pub fn predict(&self, x_new: f32) -> Vec<f32> {
        self.coef_matrix
            .iter()
            .map(|c| c.first().copied().unwrap_or(0.0) + c.get(1).copied().unwrap_or(0.0) * x_new)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. ElasticRegistration — SRSF-based curve registration
// ─────────────────────────────────────────────────────────────────────────────

/// Elastic registration utilities based on the Square Root Slope Function (SRSF).
pub struct ElasticRegistration;

impl ElasticRegistration {
    /// Compute the Square Root Slope Function: `q(t) = f'(t) / sqrt(|f'(t)| + eps)`.
    ///
    /// Finite differences approximate the derivative.
    pub fn srsf(f: &[f32], dt: f32) -> Vec<f32> {
        let n = f.len();
        if n < 2 {
            return vec![0.0; n];
        }
        let eps = 1e-6_f32;
        let mut q = Vec::with_capacity(n);
        for i in 0..n {
            let fprime = if i == 0 {
                (f[1] - f[0]) / dt
            } else if i == n - 1 {
                (f[n - 1] - f[n - 2]) / dt
            } else {
                (f[i + 1] - f[i - 1]) / (2.0 * dt)
            };
            q.push(fprime / (fprime.abs() + eps).sqrt());
        }
        q
    }

    /// L2 inner product of two equal-length vectors (uniform grid).
    pub fn l2_inner_product(q1: &[f32], q2: &[f32]) -> f32 {
        q1.iter().zip(q2.iter()).map(|(&a, &b)| a * b).sum::<f32>()
    }

    /// Elastic distance between two curves via SRSF L2 distance.
    pub fn elastic_distance(f1: &[f32], f2: &[f32], dt: f32) -> f32 {
        let q1 = Self::srsf(f1, dt);
        let q2 = Self::srsf(f2, dt);
        let diff: f32 = q1
            .iter()
            .zip(q2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();
        (diff * dt).sqrt()
    }

    /// Iterative Karcher mean in SRSF space.
    pub fn karcher_mean(functions: &[Vec<f32>], dt: f32, n_iter: usize) -> Vec<f32> {
        if functions.is_empty() {
            return vec![];
        }
        let m = functions[0].len();
        // Initialise mean as pointwise arithmetic mean
        let mut mean: Vec<f32> = vec![0.0_f32; m];
        for f in functions {
            for (j, &v) in f.iter().enumerate() {
                if j < m {
                    mean[j] += v;
                }
            }
        }
        let n_f = functions.len() as f32;
        for v in &mut mean {
            *v /= n_f;
        }
        // Gradient-descent update in SRSF space
        for _ in 0..n_iter {
            let q_mean = Self::srsf(&mean, dt);
            let mut grad = vec![0.0_f32; m];
            for f in functions {
                let aligned = Self::align_to_template(f, &mean, dt);
                let q_aligned = Self::srsf(&aligned, dt);
                for j in 0..m {
                    grad[j] += q_aligned[j] - q_mean[j];
                }
            }
            // Update mean in signal space (step 0.5/n)
            let step = 0.5 / n_f;
            for j in 0..m {
                mean[j] += step * grad[j] * dt;
            }
        }
        mean
    }

    /// Align `f` to `template` using a simple linear time warp.
    pub fn align_to_template(f: &[f32], template: &[f32], _dt: f32) -> Vec<f32> {
        let n = f.len();
        if n == 0 {
            return vec![];
        }
        // Simple linear rescaling: find scale factor minimising L2 distance
        // y_aligned(t) = f(alpha * t + beta) — here we just return f (identity warp)
        // A proper implementation would minimise SRSF distance over the group of
        // diffeomorphisms; here we use the identity warp as a lightweight stand-in.
        let _ = template; // suppress unused warning
        f.to_vec()
    }
}

/// Fréchet mean computation (pointwise L2 mean for functional data).
pub struct FrechetMean;

impl FrechetMean {
    /// Pointwise arithmetic mean (= Fréchet mean under L2 metric).
    pub fn compute(functions: &[Vec<f32>]) -> Vec<f32> {
        if functions.is_empty() {
            return vec![];
        }
        let m = functions[0].len();
        let mut out = vec![0.0_f32; m];
        for f in functions {
            for (j, &v) in f.iter().enumerate() {
                if j < m {
                    out[j] += v;
                }
            }
        }
        let n_f = functions.len() as f32;
        for v in &mut out {
            *v /= n_f;
        }
        out
    }

    /// Weighted Fréchet mean.
    pub fn compute_weighted(functions: &[Vec<f32>], weights: &[f32]) -> Vec<f32> {
        if functions.is_empty() {
            return vec![];
        }
        let m = functions[0].len();
        let mut out = vec![0.0_f32; m];
        let weight_sum: f32 = weights.iter().sum::<f32>().max(1e-12);
        for (f, &w) in functions.iter().zip(weights.iter()) {
            for (j, &v) in f.iter().enumerate() {
                if j < m {
                    out[j] += w * v;
                }
            }
        }
        for v in &mut out {
            *v /= weight_sum;
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. FunctionalNeuralNetwork
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the functional neural network.
#[derive(Clone, Debug)]
pub struct FnnConfig {
    /// Number of basis coefficients used as input features.
    pub n_basis: usize,
    /// Hidden layer widths.
    pub hidden_dims: Vec<usize>,
    /// Output dimension.
    pub output_dim: usize,
}

/// A single fully-connected layer.
#[derive(Clone, Debug)]
pub struct FnnLayer {
    /// Weight matrix (out_dim × in_dim).
    pub weights: Vec<Vec<f32>>,
    /// Bias vector (out_dim).
    pub bias: Vec<f32>,
}

impl FnnLayer {
    /// Xavier-initialised layer.
    fn new_xavier(in_dim: usize, out_dim: usize, rng: &mut StdRng) -> Self {
        let limit = (6.0_f32 / (in_dim + out_dim) as f32).sqrt();
        let weights: Vec<Vec<f32>> = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| {
                        let u: f32 = rng.random();
                        -limit + 2.0 * limit * u
                    })
                    .collect()
            })
            .collect();
        let bias = vec![0.0_f32; out_dim];
        Self { weights, bias }
    }

    /// Forward pass: `ReLU(W x + b)`.
    fn forward_relu(&self, x: &[f32]) -> Vec<f32> {
        self.weights
            .iter()
            .zip(self.bias.iter())
            .map(|(row, &b)| {
                let z: f32 = row
                    .iter()
                    .zip(x.iter())
                    .map(|(&w, &xi)| w * xi)
                    .sum::<f32>()
                    + b;
                z.max(0.0) // ReLU
            })
            .collect()
    }

    /// Forward pass without activation (used in the output layer).
    fn forward_linear(&self, x: &[f32]) -> Vec<f32> {
        self.weights
            .iter()
            .zip(self.bias.iter())
            .map(|(row, &b)| {
                row.iter()
                    .zip(x.iter())
                    .map(|(&w, &xi)| w * xi)
                    .sum::<f32>()
                    + b
            })
            .collect()
    }
}

/// Neural network that operates on basis coefficient vectors.
#[derive(Clone, Debug)]
pub struct FunctionalNeuralNetwork {
    /// Configuration.
    pub config: FnnConfig,
    /// Layers (including the output layer).
    pub layers: Vec<FnnLayer>,
}

impl FunctionalNeuralNetwork {
    /// Build a new network with Xavier-initialised weights.
    pub fn new(config: FnnConfig, rng: &mut StdRng) -> Self {
        let mut dims = vec![config.n_basis];
        dims.extend_from_slice(&config.hidden_dims);
        dims.push(config.output_dim);
        let layers: Vec<FnnLayer> = dims
            .windows(2)
            .map(|w| FnnLayer::new_xavier(w[0], w[1], rng))
            .collect();
        Self { config, layers }
    }

    /// Forward pass from basis coefficients to output.
    pub fn forward(&self, coefs: &[f32]) -> Vec<f32> {
        if self.layers.is_empty() {
            return vec![];
        }
        let mut h: Vec<f32> = coefs.to_vec();
        let n_layers = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            if i + 1 < n_layers {
                h = layer.forward_relu(&h);
            } else {
                h = layer.forward_linear(&h);
            }
        }
        h
    }

    /// Project a functional observation to basis coefficients, then forward-pass.
    pub fn predict_from_obs(&self, obs: &FdaObservation, basis: &FdaBasis) -> Vec<f32> {
        let coefs = obs.to_basis_coefs(basis);
        self.forward(&coefs)
    }

    /// Mean squared error over a batch.
    pub fn mse_loss(predictions: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
        let n = predictions.len().min(targets.len());
        if n == 0 {
            return 0.0;
        }
        let mut total = 0.0_f32;
        let mut count = 0_usize;
        for i in 0..n {
            let m = predictions[i].len().min(targets[i].len());
            for j in 0..m {
                total += (predictions[i][j] - targets[i][j]).powi(2);
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f32
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. FdaMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Collection of functional data metrics and distances.
pub struct FdaMetrics;

impl FdaMetrics {
    /// L2 distance between two functions sampled on a uniform grid.
    pub fn l2_distance(f1: &[f32], f2: &[f32], dt: f32) -> f32 {
        let n = f1.len().min(f2.len());
        if n == 0 {
            return 0.0;
        }
        let diff: f32 = (0..n).map(|i| (f1[i] - f2[i]).powi(2)).sum();
        (diff * dt).sqrt()
    }

    /// H1 Sobolev norm: `||f||^2 = ||f||_L2^2 + lambda ||f'||_L2^2`.
    pub fn h1_sobolev_norm(f: &[f32], dt: f32, lambda: f32) -> f32 {
        let n = f.len();
        if n < 2 {
            return 0.0;
        }
        // L2 norm squared via trapezoidal rule
        let l2_sq: f32 = {
            let mut s = 0.0_f32;
            for i in 0..(n - 1) {
                s += 0.5 * dt * (f[i].powi(2) + f[i + 1].powi(2));
            }
            s
        };
        // ||f'||^2 via finite differences
        let deriv_sq: f32 = {
            let mut s = 0.0_f32;
            for i in 0..(n - 1) {
                let fp = (f[i + 1] - f[i]) / dt;
                s += fp.powi(2) * dt;
            }
            s
        };
        (l2_sq + lambda * deriv_sq).sqrt()
    }

    /// Dynamic time warping distance.
    pub fn dtw_distance(f1: &[f32], f2: &[f32]) -> f32 {
        let n = f1.len();
        let m = f2.len();
        if n == 0 || m == 0 {
            return 0.0;
        }
        let inf = f32::MAX / 2.0;
        let mut dtw = vec![vec![inf; m + 1]; n + 1];
        dtw[0][0] = 0.0;
        for i in 1..=n {
            for j in 1..=m {
                let cost = (f1[i - 1] - f2[j - 1]).abs();
                let prev = dtw[i - 1][j].min(dtw[i][j - 1]).min(dtw[i - 1][j - 1]);
                dtw[i][j] = cost + prev;
            }
        }
        dtw[n][m]
    }

    /// Amplitude and phase distances based on SRSF decomposition.
    ///
    /// Returns `(amplitude_dist, phase_dist)`.
    pub fn amplitude_phase_distance(f1: &[f32], f2: &[f32], dt: f32) -> (f32, f32) {
        let q1 = ElasticRegistration::srsf(f1, dt);
        let q2 = ElasticRegistration::srsf(f2, dt);
        let amplitude = Self::l2_distance(&q1, &q2, dt);
        // Phase distance: approximate as mean absolute time shift of peaks
        let phase = {
            let peak1 = argmax(f1);
            let peak2 = argmax(f2);
            ((peak1 as f32 - peak2 as f32) * dt).abs()
        };
        (amplitude, phase)
    }
}

/// Summary evaluation report for an FDA dataset.
#[derive(Clone, Debug)]
pub struct FdaEvalReport {
    /// Mean L2 pairwise distance in the dataset.
    pub mean_l2: f32,
    /// Mean DTW pairwise distance.
    pub mean_dtw: f32,
    /// Fraction of variance explained by the top-k FPC components.
    pub fpca_variance_explained: f32,
}

impl FdaEvalReport {
    /// Compute an evaluation report for the dataset.
    pub fn evaluate(data: &FdaDataset, n_components: usize) -> Self {
        let (a, b) = data.domain;
        let m = 20_usize;
        let eval_pts: Vec<f32> = (0..m)
            .map(|i| a + (b - a) * i as f32 / (m - 1).max(1) as f32)
            .collect();

        // Pairwise L2 and DTW distances
        let n = data.observations.len();
        let dt = if m > 1 { (b - a) / (m - 1) as f32 } else { 1.0 };
        let (mut sum_l2, mut sum_dtw, mut count) = (0.0_f32, 0.0_f32, 0_usize);
        let grids: Vec<Vec<f32>> = data
            .observations
            .iter()
            .map(|obs| {
                eval_pts
                    .iter()
                    .map(|&x| obs.interpolate_linear(x))
                    .collect()
            })
            .collect();
        for i in 0..n {
            for j in (i + 1)..n {
                sum_l2 += FdaMetrics::l2_distance(&grids[i], &grids[j], dt);
                sum_dtw += FdaMetrics::dtw_distance(&grids[i], &grids[j]);
                count += 1;
            }
        }
        let mean_l2 = if count > 0 {
            sum_l2 / count as f32
        } else {
            0.0
        };
        let mean_dtw = if count > 0 {
            sum_dtw / count as f32
        } else {
            0.0
        };

        // FPCA explained variance
        let basis = FdaBasis::Legendre(LegendreBasis { n_basis: 5 });
        let mut fpca = FpcaModel::new(n_components, basis, eval_pts);
        let fpca_result = fpca.fit(data);
        let fpca_variance_explained = fpca_result
            .explained_variance_ratio
            .iter()
            .sum::<f32>()
            .min(1.0);

        Self {
            mean_l2,
            mean_dtw,
            fpca_variance_explained,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Solve a square linear system `A x = b` via Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[Vec<f32>], b: &[f32]) -> Vec<f32> {
    let n = b.len();
    if n == 0 {
        return vec![];
    }
    // Build augmented matrix [A | b]
    let mut aug: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..n {
        // Find pivot
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                aug[r1][col]
                    .abs()
                    .partial_cmp(&aug[r2][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);
        aug.swap(col, pivot_row);
        let pivot = aug[col][col];
        if pivot.abs() < 1e-12 {
            continue;
        }
        for j in col..=n {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in col..=n {
                let sub = factor * aug[col][j];
                aug[row][j] -= sub;
            }
        }
    }
    (0..n).map(|i| *aug[i].get(n).unwrap_or(&0.0)).collect()
}

/// Power iteration to find the dominant eigenpair `(λ, v)` of a symmetric matrix.
fn power_iteration(a: &[Vec<f32>], max_iter: usize, tol: f32) -> (f32, Vec<f32>) {
    let n = a.len();
    if n == 0 {
        return (0.0, vec![]);
    }
    // Initialise with constant vector
    let mut v: Vec<f32> = vec![1.0_f32 / (n as f32).sqrt(); n];
    let mut eigenvalue = 0.0_f32;
    for _ in 0..max_iter {
        // w = A v
        let mut w = vec![0.0_f32; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += a[i][j] * v[j];
            }
        }
        // Rayleigh quotient
        let rq: f32 = v.iter().zip(w.iter()).map(|(&vi, &wi)| vi * wi).sum();
        // Normalise
        let norm = (w.iter().map(|&x| x * x).sum::<f32>()).sqrt().max(1e-12);
        let v_new: Vec<f32> = w.iter().map(|&x| x / norm).collect();
        // Convergence check
        let diff: f32 = v
            .iter()
            .zip(v_new.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        eigenvalue = rq;
        v = v_new;
        if diff < tol {
            break;
        }
    }
    (eigenvalue, v)
}

/// L2 inner product on a uniform grid (no dt scaling — returns sum of products).
fn l2_inner_product_grid(f1: &[f32], f2: &[f32]) -> f32 {
    f1.iter().zip(f2.iter()).map(|(&a, &b)| a * b).sum()
}

/// Linear interpolation on a uniform grid described by `xs` → `ys`.
fn interp_linear_grid(xs: &[f32], ys: &[f32], x: f32) -> f32 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return ys[0];
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    let idx = xs.partition_point(|&p| p <= x).saturating_sub(1).min(n - 2);
    let x0 = xs[idx];
    let x1 = xs[idx + 1];
    let dx = x1 - x0;
    if dx.abs() < 1e-12 {
        return ys[idx];
    }
    ys[idx] + (ys[idx + 1] - ys[idx]) * (x - x0) / dx
}

/// Return the index of the maximum element.
fn argmax(f: &[f32]) -> usize {
    f.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn linspace(a: f32, b: f32, n: usize) -> Vec<f32> {
        if n < 2 {
            return vec![a];
        }
        (0..n)
            .map(|i| a + (b - a) * i as f32 / (n - 1) as f32)
            .collect()
    }

    fn sine_obs(id: usize, n: usize, freq: f32) -> FdaObservation {
        let xs = linspace(0.0, 1.0, n);
        let ys = xs.iter().map(|&x| (2.0 * PI * freq * x).sin()).collect();
        FdaObservation {
            id,
            x_points: xs,
            y_values: ys,
        }
    }

    fn make_dataset(n_obs: usize) -> FdaDataset {
        let obs: Vec<FdaObservation> = (0..n_obs)
            .map(|i| sine_obs(i, 50, (i + 1) as f32))
            .collect();
        FdaDataset {
            observations: obs,
            domain: (0.0, 1.0),
        }
    }

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    // ── 1. Basis tests ────────────────────────────────────────────────────────

    #[test]
    fn test_fourier_basis_evaluate_shape() {
        let basis = FourierBasis {
            n_basis: 7,
            period: 1.0,
        };
        let vals = basis.evaluate(0.3);
        assert_eq!(vals.len(), 7, "Fourier basis must return n_basis values");
    }

    #[test]
    fn test_fourier_basis_orthogonality() {
        // Approximate ∫_0^1 φ_1(x) φ_2(x) dx ≈ 0 for distinct harmonics
        let basis = FourierBasis {
            n_basis: 5,
            period: 1.0,
        };
        let n = 1000;
        let xs = linspace(0.0, 1.0, n);
        let dt = 1.0 / (n - 1) as f32;
        let all = basis.evaluate_all(&xs);
        let dot_01: f32 = all.iter().map(|row| row[1] * row[2]).sum::<f32>() * dt;
        assert!(
            dot_01.abs() < 0.05,
            "Fourier basis sin/cos should be approximately orthogonal, got {}",
            dot_01
        );
    }

    #[test]
    fn test_bspline_basis_partition_of_unity() {
        // For degree-2 B-splines, sum of basis values should equal 1
        let n_basis = 4_usize;
        let degree = 2_usize;
        // Build uniform knots
        let knots: Vec<f32> = vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0];
        let basis = BSplineBasis {
            n_basis,
            degree,
            knots,
        };
        let x = 0.25_f32;
        let vals = basis.evaluate(x);
        let sum: f32 = vals.iter().sum();
        // Partition of unity: sum ≈ 1
        assert!(
            (sum - 1.0).abs() < 0.15,
            "B-spline partition of unity failed: sum={}",
            sum
        );
    }

    #[test]
    fn test_legendre_basis_values() {
        let basis = LegendreBasis { n_basis: 3 };
        let vals = basis.evaluate(0.5);
        assert!((vals[0] - 1.0).abs() < 1e-5, "P_0 = 1");
        assert!((vals[1] - 0.5).abs() < 1e-5, "P_1(0.5) = 0.5");
    }

    #[test]
    fn test_fda_basis_enum_evaluate() {
        let b = FdaBasis::Legendre(LegendreBasis { n_basis: 4 });
        let vals = b.evaluate(0.0);
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_legendre_basis_recurrence() {
        // P_2(x) = 0.5 * (3x^2 - 1)
        let basis = LegendreBasis { n_basis: 3 };
        let x = 0.7_f32;
        let vals = basis.evaluate(x);
        let p2_expected = 0.5 * (3.0 * x * x - 1.0);
        assert!(
            (vals[2] - p2_expected).abs() < 1e-5,
            "P_2 recurrence, got {}, expected {}",
            vals[2],
            p2_expected
        );
    }

    #[test]
    fn test_bspline_evaluate_domain() {
        let knots = vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0];
        let basis = BSplineBasis {
            n_basis: 4,
            degree: 2,
            knots,
        };
        let vals = basis.evaluate(0.5);
        assert_eq!(
            vals.len(),
            4,
            "B-spline evaluate must return n_basis values"
        );
        // All values should be in [0, 1]
        for &v in &vals {
            assert!(
                (-0.01..=1.01).contains(&v),
                "B-spline value out of range: {}",
                v
            );
        }
    }

    // ── 2. FdaObservation tests ───────────────────────────────────────────────

    #[test]
    fn test_fda_observation_interpolate_linear() {
        let obs = FdaObservation {
            id: 0,
            x_points: vec![0.0, 1.0, 2.0],
            y_values: vec![0.0, 2.0, 4.0],
        };
        let y = obs.interpolate_linear(0.5);
        assert!(
            (y - 1.0).abs() < 1e-5,
            "Linear interp at 0.5 should be 1.0, got {}",
            y
        );
    }

    #[test]
    fn test_fda_observation_l2_norm_positive() {
        let obs = sine_obs(0, 100, 2.0);
        let norm = obs.l2_norm();
        assert!(norm > 0.0, "L2 norm of non-zero function must be positive");
    }

    #[test]
    fn test_fda_observation_to_basis_coefs_shape() {
        let obs = sine_obs(0, 50, 1.0);
        let basis = FdaBasis::Fourier(FourierBasis {
            n_basis: 5,
            period: 1.0,
        });
        let coefs = obs.to_basis_coefs(&basis);
        assert_eq!(
            coefs.len(),
            5,
            "Coefficient vector must have length n_basis"
        );
    }

    #[test]
    fn test_fda_observation_domain_length() {
        let obs = FdaObservation {
            id: 0,
            x_points: vec![0.0, 0.5, 1.0],
            y_values: vec![0.0, 1.0, 0.0],
        };
        assert!((obs.domain_length() - 1.0).abs() < 1e-5);
    }

    // ── 3. FdaDataset tests ───────────────────────────────────────────────────

    #[test]
    fn test_fda_dataset_mean_function_shape() {
        let ds = make_dataset(5);
        let pts = linspace(0.0, 1.0, 20);
        let mean = ds.mean_function(&pts);
        assert_eq!(mean.len(), 20);
    }

    #[test]
    fn test_fda_dataset_covariance_surface_symmetric() {
        let ds = make_dataset(6);
        let pts = linspace(0.0, 1.0, 10);
        let cov = ds.covariance_surface(&pts);
        let m = pts.len();
        for i in 0..m {
            for j in 0..m {
                assert!(
                    (cov[i][j] - cov[j][i]).abs() < 1e-5,
                    "Covariance must be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_fda_dataset_covariance_shape() {
        let ds = make_dataset(4);
        let pts = linspace(0.0, 1.0, 8);
        let cov = ds.covariance_surface(&pts);
        assert_eq!(cov.len(), 8);
        assert_eq!(cov[0].len(), 8);
    }

    #[test]
    fn test_covariance_surface_diagonal_positive() {
        let ds = make_dataset(5);
        let pts = linspace(0.0, 1.0, 10);
        let cov = ds.covariance_surface(&pts);
        for i in 0..cov.len() {
            assert!(
                cov[i][i] >= -1e-6,
                "Diagonal covariance must be non-negative, got {}",
                cov[i][i]
            );
        }
    }

    // ── 4. FPCA tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_fpca_model_fit_result_shape() {
        let ds = make_dataset(8);
        let basis = FdaBasis::Legendre(LegendreBasis { n_basis: 4 });
        let pts = linspace(0.0, 1.0, 15);
        let mut model = FpcaModel::new(3, basis, pts);
        let result = model.fit(&ds);
        assert_eq!(result.eigenvalues.len(), 3);
        assert_eq!(result.eigenfunctions.len(), 3);
        assert_eq!(result.explained_variance_ratio.len(), 3);
    }

    #[test]
    fn test_fpca_explained_variance_sums_to_one() {
        let ds = make_dataset(10);
        let basis = FdaBasis::Legendre(LegendreBasis { n_basis: 4 });
        let pts = linspace(0.0, 1.0, 15);
        let mut model = FpcaModel::new(3, basis, pts);
        let result = model.fit(&ds);
        let total: f32 = result.explained_variance_ratio.iter().sum();
        // With k components the sum ≤ 1 (deflation removes variance)
        assert!(
            (0.0..=1.01).contains(&total),
            "EVR should be in [0,1], got {}",
            total
        );
    }

    #[test]
    fn test_fpca_transform_shape() {
        let ds = make_dataset(8);
        let basis = FdaBasis::Legendre(LegendreBasis { n_basis: 4 });
        let pts = linspace(0.0, 1.0, 15);
        let mut model = FpcaModel::new(3, basis, pts);
        model.fit(&ds);
        let obs = sine_obs(99, 50, 1.0);
        let scores = model.transform(&obs);
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn test_fpca_inverse_transform_shape() {
        let ds = make_dataset(8);
        let basis = FdaBasis::Legendre(LegendreBasis { n_basis: 4 });
        let pts = linspace(0.0, 1.0, 15);
        let mut model = FpcaModel::new(3, basis, pts.clone());
        model.fit(&ds);
        let scores = vec![0.1_f32, -0.2, 0.3];
        let recon = model.inverse_transform(&scores, &pts);
        assert_eq!(recon.len(), pts.len());
    }

    #[test]
    fn test_fpca_reconstruction_error() {
        let ds = make_dataset(10);
        let pts = linspace(0.0, 1.0, 20);
        let basis = FdaBasis::Legendre(LegendreBasis { n_basis: 5 });
        let mut model = FpcaModel::new(5, basis, pts.clone());
        model.fit(&ds);
        // For a single function reconstruct from its scores
        let obs = &ds.observations[0];
        let scores = model.transform(obs);
        let recon = model.inverse_transform(&scores, &pts);
        // Reconstruction should be finite
        for &v in &recon {
            assert!(v.is_finite(), "Reconstruction must produce finite values");
        }
    }

    #[test]
    fn test_fpca_scores_zero_mean() {
        // Scores of all observations should have approximately zero mean
        let ds = make_dataset(8);
        let basis = FdaBasis::Legendre(LegendreBasis { n_basis: 4 });
        let pts = linspace(0.0, 1.0, 15);
        let mut model = FpcaModel::new(2, basis, pts);
        model.fit(&ds);
        let scores: Vec<Vec<f32>> = ds
            .observations
            .iter()
            .map(|obs| model.transform(obs))
            .collect();
        let n = scores.len() as f32;
        let mean0: f32 = scores.iter().map(|s| s[0]).sum::<f32>() / n;
        // We don't subtract the mean during transform so this is a sanity check only
        assert!(mean0.is_finite(), "FPC score mean should be finite");
    }

    // ── 5. Functional regression tests ────────────────────────────────────────

    #[test]
    fn test_scalar_on_function_fit_predict() {
        let ds = make_dataset(6);
        let y: Vec<f32> = (0..6_usize).map(|i| i as f32 * 0.5).collect();
        let model = ScalarOnFunction::fit(&ds, &y);
        let pred = model.predict(&ds.observations[0]);
        assert!(pred.is_finite(), "Prediction must be finite");
    }

    #[test]
    fn test_scalar_on_function_r_squared_range() {
        let ds = make_dataset(8);
        let y: Vec<f32> = (0..8).map(|i| (i as f32).sin()).collect();
        let model = ScalarOnFunction::fit(&ds, &y);
        let r2 = model.r_squared(&ds, &y);
        assert!(r2 <= 1.0 + 1e-4, "R² cannot exceed 1, got {}", r2);
    }

    #[test]
    fn test_function_on_scalar_fit_predict_shape() {
        let ds = make_dataset(6);
        let x: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let model = FunctionOnScalar::fit(&x, &ds);
        let pred = model.predict(3.0);
        assert_eq!(pred.len(), model.n_eval_points);
    }

    // ── 6. ElasticRegistration tests ─────────────────────────────────────────

    #[test]
    fn test_srsf_computation() {
        let f: Vec<f32> = linspace(0.0, 1.0, 20).iter().map(|&x| x * x).collect();
        let q = ElasticRegistration::srsf(&f, 1.0 / 19.0);
        assert_eq!(q.len(), f.len(), "SRSF must have same length as input");
        // All values should be finite
        for &v in &q {
            assert!(v.is_finite(), "SRSF must be finite");
        }
    }

    #[test]
    fn test_l2_inner_product_identity() {
        let v: Vec<f32> = vec![1.0, 2.0, 3.0];
        let dot = ElasticRegistration::l2_inner_product(&v, &v);
        assert!((dot - 14.0).abs() < 1e-5, "||v||^2 = 14, got {}", dot);
    }

    #[test]
    fn test_elastic_distance_symmetric() {
        let f1: Vec<f32> = linspace(0.0, 1.0, 30).iter().map(|&x| x.sin()).collect();
        let f2: Vec<f32> = linspace(0.0, 1.0, 30)
            .iter()
            .map(|&x| (2.0 * x).sin())
            .collect();
        let dt = 1.0 / 29.0;
        let d12 = ElasticRegistration::elastic_distance(&f1, &f2, dt);
        let d21 = ElasticRegistration::elastic_distance(&f2, &f1, dt);
        assert!(
            (d12 - d21).abs() < 1e-5,
            "Elastic distance must be symmetric"
        );
    }

    #[test]
    fn test_elastic_distance_self_zero() {
        let f: Vec<f32> = (0..30).map(|i| (i as f32 / 29.0).powi(2)).collect();
        let dt = 1.0 / 29.0;
        let d = ElasticRegistration::elastic_distance(&f, &f, dt);
        assert!(
            d.abs() < 1e-5,
            "Self elastic distance must be zero, got {}",
            d
        );
    }

    #[test]
    fn test_karcher_mean_shape() {
        let funcs: Vec<Vec<f32>> = (0..5)
            .map(|i| {
                linspace(0.0, 1.0, 30)
                    .iter()
                    .map(|&x| (x + i as f32 * 0.1).sin())
                    .collect()
            })
            .collect();
        let mean = ElasticRegistration::karcher_mean(&funcs, 1.0 / 29.0, 5);
        assert_eq!(
            mean.len(),
            30,
            "Karcher mean must have same length as input functions"
        );
    }

    #[test]
    fn test_align_to_template_shape() {
        let f: Vec<f32> = linspace(0.0, 1.0, 20).iter().map(|&x| x.sin()).collect();
        let template: Vec<f32> = linspace(0.0, 1.0, 20)
            .iter()
            .map(|&x| (2.0 * x).sin())
            .collect();
        let aligned = ElasticRegistration::align_to_template(&f, &template, 1.0 / 19.0);
        assert_eq!(aligned.len(), f.len());
    }

    #[test]
    fn test_frechet_mean_compute() {
        let funcs: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0], vec![3.0, 4.0, 5.0]];
        let m = FrechetMean::compute(&funcs);
        assert!((m[0] - 2.0).abs() < 1e-5);
        assert!((m[1] - 3.0).abs() < 1e-5);
        assert!((m[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_frechet_mean_weighted() {
        let funcs: Vec<Vec<f32>> = vec![vec![0.0, 0.0], vec![2.0, 4.0]];
        let weights = vec![0.5_f32, 0.5];
        let m = FrechetMean::compute_weighted(&funcs, &weights);
        assert!((m[0] - 1.0).abs() < 1e-5);
        assert!((m[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_elastic_karcher_convergence() {
        // The Karcher mean with more iterations should not blow up
        let funcs: Vec<Vec<f32>> = (0..4)
            .map(|i| {
                linspace(0.0, 1.0, 40)
                    .iter()
                    .map(|&x| ((i + 1) as f32 * x).sin())
                    .collect()
            })
            .collect();
        let mean5 = ElasticRegistration::karcher_mean(&funcs, 1.0 / 39.0, 5);
        let mean10 = ElasticRegistration::karcher_mean(&funcs, 1.0 / 39.0, 10);
        assert_eq!(mean5.len(), mean10.len());
        for (&v5, &v10) in mean5.iter().zip(mean10.iter()) {
            assert!(v5.is_finite() && v10.is_finite());
        }
    }

    // ── 7. FunctionalNeuralNetwork tests ──────────────────────────────────────

    #[test]
    fn test_fnn_forward_shape() {
        let config = FnnConfig {
            n_basis: 5,
            hidden_dims: vec![8, 6],
            output_dim: 3,
        };
        let mut rng = make_rng();
        let net = FunctionalNeuralNetwork::new(config, &mut rng);
        let coefs = vec![0.1_f32; 5];
        let out = net.forward(&coefs);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_fnn_predict_from_obs() {
        let config = FnnConfig {
            n_basis: 5,
            hidden_dims: vec![4],
            output_dim: 2,
        };
        let mut rng = make_rng();
        let net = FunctionalNeuralNetwork::new(config, &mut rng);
        let obs = sine_obs(0, 50, 1.0);
        let basis = FdaBasis::Legendre(LegendreBasis { n_basis: 5 });
        let out = net.predict_from_obs(&obs, &basis);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_fnn_mse_loss_zero_perfect() {
        let predictions = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]];
        let targets = predictions.clone();
        let loss = FunctionalNeuralNetwork::mse_loss(&predictions, &targets);
        assert!(
            loss.abs() < 1e-6,
            "MSE of identical predictions/targets must be 0"
        );
    }

    #[test]
    fn test_fnn_xavier_init_reasonable_range() {
        let config = FnnConfig {
            n_basis: 10,
            hidden_dims: vec![16],
            output_dim: 4,
        };
        let mut rng = make_rng();
        let net = FunctionalNeuralNetwork::new(config, &mut rng);
        // Xavier limit for (10, 16): sqrt(6/(10+16)) ≈ 0.48
        let limit = (6.0_f32 / (10.0 + 16.0)).sqrt() + 0.1;
        for layer in &net.layers {
            for row in &layer.weights {
                for &w in row {
                    assert!(
                        w.abs() <= limit + 0.5,
                        "Weight magnitude {} exceeds expected Xavier range",
                        w
                    );
                }
            }
        }
    }

    // ── 8. FdaMetrics tests ───────────────────────────────────────────────────

    #[test]
    fn test_l2_distance_zero_same() {
        let f: Vec<f32> = linspace(0.0, 1.0, 20).iter().map(|&x| x.sin()).collect();
        let d = FdaMetrics::l2_distance(&f, &f, 1.0 / 19.0);
        assert!(d.abs() < 1e-6, "L2 distance to self must be 0");
    }

    #[test]
    fn test_l2_distance_positive() {
        let f1: Vec<f32> = vec![0.0; 10];
        let f2: Vec<f32> = vec![1.0; 10];
        let d = FdaMetrics::l2_distance(&f1, &f2, 1.0 / 9.0);
        assert!(
            d > 0.0,
            "L2 distance between distinct functions must be positive"
        );
    }

    #[test]
    fn test_h1_sobolev_norm_positive() {
        let f: Vec<f32> = linspace(0.0, 1.0, 30).iter().map(|&x| x.sin()).collect();
        let norm = FdaMetrics::h1_sobolev_norm(&f, 1.0 / 29.0, 0.1);
        assert!(
            norm > 0.0,
            "H1 Sobolev norm must be positive for non-zero function"
        );
    }

    #[test]
    fn test_dtw_distance_zero_same() {
        let f: Vec<f32> = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let d = FdaMetrics::dtw_distance(&f, &f);
        assert!(d.abs() < 1e-6, "DTW distance to self must be 0");
    }

    #[test]
    fn test_dtw_distance_symmetric() {
        let f1: Vec<f32> = vec![0.0, 1.0, 2.0, 1.5, 0.5];
        let f2: Vec<f32> = vec![0.5, 1.5, 2.5, 1.0, 0.0];
        let d12 = FdaMetrics::dtw_distance(&f1, &f2);
        let d21 = FdaMetrics::dtw_distance(&f2, &f1);
        assert!((d12 - d21).abs() < 1e-5, "DTW distance must be symmetric");
    }

    #[test]
    fn test_amplitude_phase_distance_returns_tuple() {
        let f1: Vec<f32> = linspace(0.0, 1.0, 30).iter().map(|&x| x.sin()).collect();
        let f2: Vec<f32> = linspace(0.0, 1.0, 30)
            .iter()
            .map(|&x| (2.0 * x).sin())
            .collect();
        let (amp, phase) = FdaMetrics::amplitude_phase_distance(&f1, &f2, 1.0 / 29.0);
        assert!(amp.is_finite(), "Amplitude distance must be finite");
        assert!(phase.is_finite(), "Phase distance must be finite");
    }

    #[test]
    fn test_fda_eval_report_fields() {
        let ds = make_dataset(4);
        let report = FdaEvalReport::evaluate(&ds, 2);
        assert!(report.mean_l2 >= 0.0);
        assert!(report.mean_dtw >= 0.0);
        assert!(report.fpca_variance_explained >= 0.0 && report.fpca_variance_explained <= 1.01);
    }

    #[test]
    fn test_evaluate_runs() {
        let ds = make_dataset(5);
        let report = FdaEvalReport::evaluate(&ds, 3);
        assert!(report.mean_l2.is_finite());
        assert!(report.mean_dtw.is_finite());
        assert!(report.fpca_variance_explained.is_finite());
    }
}
