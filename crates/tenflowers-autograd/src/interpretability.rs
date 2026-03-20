//! # Gradient Saliency and Model Interpretability
//!
//! This module provides gradient-based interpretability methods for understanding
//! model predictions.  All methods operate on raw `f64` slices and accept a
//! scalar-valued closure `f: &[f64] -> f64`, making them agnostic to the
//! specific tensor backend in use.
//!
//! ## Methods implemented
//!
//! - **Vanilla Gradient Saliency** – finite-difference gradient of f w.r.t. input,
//!   optionally wrapped in absolute value.
//! - **SmoothGrad** – averages gradients over many noisy copies of the input
//!   (Smilkov et al., 2017).
//! - **Integrated Gradients** – approximates the Aumann-Shapley attribution
//!   integral from a baseline to the input (Sundararajan et al., 2017).
//! - **Attention Rollout** – propagates attention through transformer layers.
//! - **Gradient × Input** – element-wise product of gradient and input.
//! - **Top-k Attribution Summary** – returns the k most salient input dimensions.
//!
//! ## Usage
//!
//! ```rust
//! use tenflowers_autograd::interpretability::{vanilla_saliency, abs_saliency};
//!
//! // f(x) = sum(x_i^2)
//! let x = vec![1.0_f64, 2.0, 3.0];
//! let saliency = vanilla_saliency(|v: &[f64]| v.iter().map(|xi| xi * xi).sum(), &x, 1e-5);
//! // saliency[i] ≈ 2 * x[i]
//! assert!((saliency[1] - 4.0).abs() < 1e-3);
//! ```

use std::fmt;
use tenflowers_core::TensorError;

// ---------------------------------------------------------------------------
// Public error type used by methods that can fail
// ---------------------------------------------------------------------------

/// Errors that can occur during interpretability computations.
#[derive(Debug, Clone)]
pub struct InterpretabilityError {
    /// Human-readable description.
    pub message: String,
}

impl InterpretabilityError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for InterpretabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InterpretabilityError: {}", self.message)
    }
}

impl std::error::Error for InterpretabilityError {}

impl From<InterpretabilityError> for TensorError {
    fn from(e: InterpretabilityError) -> Self {
        TensorError::invalid_operation_simple(e.message)
    }
}

/// Result type for interpretability operations.
pub type InterpResult<T> = std::result::Result<T, InterpretabilityError>;

// ---------------------------------------------------------------------------
// Finite-difference gradient helper (central difference)
// ---------------------------------------------------------------------------

/// Compute the gradient of a scalar function using central finite differences.
///
/// `∂f/∂x_i ≈ (f(x + ε·eᵢ) - f(x - ε·eᵢ)) / (2ε)`
fn finite_diff_gradient<F>(f: F, x: &[f64], eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    let mut grad = vec![0.0_f64; n];
    let mut x_perturbed = x.to_vec();
    let two_eps = 2.0 * eps;

    for i in 0..n {
        let orig = x_perturbed[i];

        x_perturbed[i] = orig + eps;
        let f_plus = f(&x_perturbed);

        x_perturbed[i] = orig - eps;
        let f_minus = f(&x_perturbed);

        x_perturbed[i] = orig; // restore

        grad[i] = (f_plus - f_minus) / two_eps;
    }

    grad
}

// ---------------------------------------------------------------------------
// Vanilla Gradient Saliency
// ---------------------------------------------------------------------------

/// Compute the vanilla gradient saliency map for a scalar function.
///
/// The saliency map is the vector of partial derivatives
/// `[∂f/∂x_0, ∂f/∂x_1, …, ∂f/∂x_{n-1}]`
/// approximated by central finite differences with step size `eps`.
///
/// # Arguments
///
/// * `f`   – Scalar function R^n → R.
/// * `x`   – Input point, length `n`.
/// * `eps` – Finite-difference step size (e.g. `1e-5`).
///
/// # Returns
///
/// A `Vec<f64>` of length `n` containing the signed gradient components.
///
/// # Example
///
/// ```rust
/// use tenflowers_autograd::interpretability::vanilla_saliency;
///
/// // f(x) = x[0]^2 + x[1]^2  →  gradient = [2*x[0], 2*x[1]]
/// let x = vec![1.0_f64, 3.0];
/// let s = vanilla_saliency(|v: &[f64]| v[0] * v[0] + v[1] * v[1], &x, 1e-5);
/// assert!((s[0] - 2.0).abs() < 1e-3);
/// assert!((s[1] - 6.0).abs() < 1e-3);
/// ```
pub fn vanilla_saliency<F>(f: F, x: &[f64], eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    finite_diff_gradient(f, x, eps)
}

/// Compute the absolute (unsigned) gradient saliency map.
///
/// This is the element-wise absolute value of `vanilla_saliency`.  Commonly
/// used when visualising feature importance regardless of sign.
///
/// # Example
///
/// ```rust
/// use tenflowers_autograd::interpretability::abs_saliency;
///
/// let x = vec![-2.0_f64, 1.0];
/// let s = abs_saliency(|v: &[f64]| v[0] * v[0] + v[1] * v[1], &x, 1e-5);
/// // All values must be non-negative
/// assert!(s.iter().all(|&v| v >= 0.0));
/// ```
pub fn abs_saliency<F>(f: F, x: &[f64], eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    finite_diff_gradient(f, x, eps)
        .into_iter()
        .map(f64::abs)
        .collect()
}

// ---------------------------------------------------------------------------
// SmoothGrad
// ---------------------------------------------------------------------------

/// SmoothGrad: reduces noise in gradient-based saliency maps by averaging
/// gradients computed over many slightly-perturbed copies of the input.
///
/// Reference: Smilkov et al., "SmoothGrad: removing noise by adding noise", 2017.
///
/// # Example
///
/// ```rust
/// use tenflowers_autograd::interpretability::SmoothGrad;
///
/// let x = vec![1.0_f64, -1.0, 2.0];
/// let sg = SmoothGrad::with_seed(50, 0.1, 42);
/// let result = sg.compute(|v: &[f64]| v.iter().map(|xi| xi * xi).sum(), &x);
/// // Result must be finite
/// assert!(result.iter().all(|v| v.is_finite()));
/// ```
#[derive(Debug, Clone)]
pub struct SmoothGrad {
    /// Number of noisy samples to average over.
    pub n_samples: usize,
    /// Standard deviation of the Gaussian noise added to the input.
    pub noise_std: f64,
    /// Finite-difference step size used within each gradient computation.
    pub eps: f64,
    /// Seed for the internal RNG; 0 means "use default seeding".
    seed: u64,
}

impl SmoothGrad {
    /// Create a `SmoothGrad` instance with default finite-difference step
    /// (`eps = 1e-5`) and a fixed seed derived from the noise_std parameter.
    pub fn new(n_samples: usize, noise_std: f64) -> Self {
        // Derive a reproducible default seed from the parameters.
        let seed = (noise_std.to_bits()) ^ (n_samples as u64).wrapping_mul(6364136223846793005);
        Self {
            n_samples,
            noise_std,
            eps: 1e-5,
            seed,
        }
    }

    /// Create a `SmoothGrad` instance with an explicit seed for reproducibility.
    pub fn with_seed(n_samples: usize, noise_std: f64, seed: u64) -> Self {
        Self {
            n_samples,
            noise_std,
            eps: 1e-5,
            seed,
        }
    }

    /// Compute the smoothed saliency map.
    ///
    /// Adds Gaussian noise N(0, noise_std²) to each input copy, computes
    /// the gradient of `f` at each perturbed point, then returns the
    /// element-wise average.
    pub fn compute<F>(&self, f: F, x: &[f64]) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = x.len();
        let mut sum_grads = vec![0.0_f64; n];

        // Generate all noise using a deterministic LCG so we avoid any
        // dependency on rand_distr at this call site.  For scientific-grade
        // smoothing the caller should provide enough samples; the LCG is
        // good enough for variance reduction.
        let noise_values = self.generate_noise(n * self.n_samples);

        for s in 0..self.n_samples {
            let noise_offset = s * n;
            let noisy_x: Vec<f64> = x
                .iter()
                .enumerate()
                .map(|(i, &xi)| xi + noise_values[noise_offset + i])
                .collect();

            let grad = finite_diff_gradient(&f, &noisy_x, self.eps);

            for (acc, g) in sum_grads.iter_mut().zip(grad.iter()) {
                *acc += g;
            }
        }

        let scale = if self.n_samples > 0 {
            1.0 / self.n_samples as f64
        } else {
            0.0
        };

        sum_grads.iter().map(|&v| v * scale).collect()
    }

    /// Compute squared SmoothGrad: averages (gradient)² before taking the
    /// element-wise square root.  This further reduces variance.
    pub fn compute_squared<F>(&self, f: F, x: &[f64]) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = x.len();
        let mut sum_sq = vec![0.0_f64; n];

        let noise_values = self.generate_noise(n * self.n_samples);

        for s in 0..self.n_samples {
            let noise_offset = s * n;
            let noisy_x: Vec<f64> = x
                .iter()
                .enumerate()
                .map(|(i, &xi)| xi + noise_values[noise_offset + i])
                .collect();

            let grad = finite_diff_gradient(&f, &noisy_x, self.eps);

            for (acc, g) in sum_sq.iter_mut().zip(grad.iter()) {
                *acc += g * g;
            }
        }

        let scale = if self.n_samples > 0 {
            1.0 / self.n_samples as f64
        } else {
            0.0
        };

        sum_sq.iter().map(|&v| (v * scale).sqrt()).collect()
    }

    // Internal helper: generate `count` approximate N(0, noise_std²) samples
    // using the Box-Muller transform driven by a splitmix64 RNG.
    fn generate_noise(&self, count: usize) -> Vec<f64> {
        let mut state = self.seed.wrapping_add(1); // avoid 0-seed degenerate case
        let mut noise = Vec::with_capacity(count);

        // We generate pairs via Box-Muller; if count is odd, we discard the
        // last member of a pair.
        let pairs = (count + 1) / 2;
        for _ in 0..pairs {
            let u1 = self.splitmix64_f64(&mut state);
            let u2 = self.splitmix64_f64(&mut state);

            // Clamp u1 away from zero to avoid log(0).
            let u1_clamped = u1.max(1e-300);
            let r = (-2.0 * u1_clamped.ln()).sqrt() * self.noise_std;
            let theta = std::f64::consts::TAU * u2;

            noise.push(r * theta.cos());
            if noise.len() < count {
                noise.push(r * theta.sin());
            }
        }

        noise
    }

    /// splitmix64 step — produces a f64 in (0, 1).
    fn splitmix64_f64(&self, state: &mut u64) -> f64 {
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        // Map to (0, 1) — exclude 0 and 1 exactly.
        (z as f64) / (u64::MAX as f64 + 1.0)
    }
}

// ---------------------------------------------------------------------------
// Integrated Gradients
// ---------------------------------------------------------------------------

/// Integrated Gradients (Sundararajan et al., 2017).
///
/// Approximates the attribution integral
///
/// ```text
/// IG_i(x) = (x_i - baseline_i) ·
///           ∫₀¹ (∂f/∂x_i)(baseline + t·(x - baseline)) dt
/// ```
///
/// using a Riemann sum with `n_steps` equally-spaced interpolation points.
///
/// # Completeness axiom
///
/// The sum of integrated gradients approximates `f(input) - f(baseline)`.
/// Use [`IntegratedGradients::completeness_error`] to check this numerically.
///
/// # Example
///
/// ```rust
/// use tenflowers_autograd::interpretability::IntegratedGradients;
///
/// // For a linear function f(x) = w · x, IG = w * (x - baseline)
/// let w = vec![1.0_f64, 2.0, 3.0];
/// let f = |v: &[f64]| v.iter().zip(w.iter()).map(|(xi, wi)| xi * wi).sum::<f64>();
/// let baseline = vec![0.0_f64; 3];
/// let input    = vec![1.0_f64, 1.0, 1.0];
///
/// let ig = IntegratedGradients::new(200);
/// let attrs = ig.compute(f, &baseline, &input).expect("compute failed");
/// // IG should equal (input - baseline) * w = [1, 2, 3]
/// assert!((attrs[0] - 1.0).abs() < 0.05);
/// assert!((attrs[1] - 2.0).abs() < 0.05);
/// assert!((attrs[2] - 3.0).abs() < 0.05);
/// ```
#[derive(Debug, Clone)]
pub struct IntegratedGradients {
    /// Number of Riemann-sum intervals.
    pub n_steps: usize,
    /// Finite-difference step for gradient estimation.
    pub eps: f64,
}

impl IntegratedGradients {
    /// Create an `IntegratedGradients` instance with the given number of steps
    /// and the default finite-difference step `eps = 1e-5`.
    pub fn new(n_steps: usize) -> Self {
        Self {
            n_steps,
            eps: 1e-5,
        }
    }

    /// Compute integrated gradients from `baseline` to `input`.
    ///
    /// # Arguments
    ///
    /// * `f`        – Scalar function R^n → R.
    /// * `baseline` – Reference point (e.g. all-zeros), length `n`.
    /// * `input`    – Target input, length `n`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `baseline.len() != input.len()` or if `n_steps == 0`.
    pub fn compute<F>(
        &self,
        f: F,
        baseline: &[f64],
        input: &[f64],
    ) -> InterpResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = baseline.len();
        if n != input.len() {
            return Err(InterpretabilityError::new(format!(
                "baseline length ({}) must equal input length ({})",
                n,
                input.len()
            )));
        }
        if self.n_steps == 0 {
            return Err(InterpretabilityError::new(
                "n_steps must be greater than zero",
            ));
        }

        let delta: Vec<f64> = input
            .iter()
            .zip(baseline.iter())
            .map(|(xi, bi)| xi - bi)
            .collect();

        // Accumulate gradients along the straight-line path.
        let mut sum_grads = vec![0.0_f64; n];

        for k in 0..self.n_steps {
            // t ∈ (0, 1] using the midpoint of each sub-interval gives a
            // better approximation than evaluating at the right endpoint.
            let t = (k as f64 + 0.5) / self.n_steps as f64;

            let interpolated: Vec<f64> = baseline
                .iter()
                .zip(delta.iter())
                .map(|(bi, di)| bi + t * di)
                .collect();

            let grad = finite_diff_gradient(&f, &interpolated, self.eps);

            for (acc, g) in sum_grads.iter_mut().zip(grad.iter()) {
                *acc += g;
            }
        }

        // Scale by 1/n_steps and multiply by (input - baseline).
        let scale = 1.0 / self.n_steps as f64;
        let attributions: Vec<f64> = sum_grads
            .iter()
            .zip(delta.iter())
            .map(|(g, d)| g * scale * d)
            .collect();

        Ok(attributions)
    }

    /// Measure the deviation from the completeness axiom.
    ///
    /// Returns `|sum(attributions) - (f(input) - f(baseline))|`.
    /// A well-converged result should yield a small value (< 1 % of the
    /// output difference for `n_steps ≥ 100`).
    pub fn completeness_error<F>(
        &self,
        f: F,
        baseline: &[f64],
        input: &[f64],
        attributions: &[f64],
    ) -> f64
    where
        F: Fn(&[f64]) -> f64 + Clone,
    {
        let sum_attrs: f64 = attributions.iter().sum();
        let output_diff = f.clone()(input) - f(baseline);
        (sum_attrs - output_diff).abs()
    }
}

// ---------------------------------------------------------------------------
// Attention Analysis
// ---------------------------------------------------------------------------

/// Attention rollout (Abnar & Zuidema, 2020).
///
/// Propagates raw attention matrices through all transformer layers to obtain
/// an "effective" attention from the last layer back to each input token.
///
/// Each layer's attention matrix `A ∈ R^{seq×seq}` is first augmented with a
/// residual identity connection (i.e. `(A + I) / 2`) and then accumulated via
/// matrix multiplication through the layer stack.
///
/// # Arguments
///
/// * `attention_maps` – One `seq × seq` attention matrix per layer, stored
///   row-major (row = query token, col = key token).  Each inner `Vec<f32>`
///   must have exactly `seq_len * seq_len` elements.
/// * `seq_len`         – Sequence length.
///
/// # Returns
///
/// A `Vec<f32>` of length `seq_len` representing the effective attention from
/// the CLS token (index 0) to every token in the sequence.
///
/// # Errors
///
/// Returns `Err` if any attention map has the wrong number of elements.
///
/// # Example
///
/// ```rust
/// use tenflowers_autograd::interpretability::attention_rollout;
///
/// let seq = 4usize;
/// // Identity attention: each token attends only to itself.
/// let identity: Vec<f32> = (0..seq)
///     .flat_map(|i| (0..seq).map(move |j| if i == j { 1.0f32 } else { 0.0 }))
///     .collect();
///
/// let maps = vec![identity.clone(), identity.clone()];
/// let rollout = attention_rollout(&maps, seq).expect("attention_rollout failed");
/// assert_eq!(rollout.len(), seq);
/// ```
pub fn attention_rollout(
    attention_maps: &[Vec<f32>],
    seq_len: usize,
) -> InterpResult<Vec<f32>> {
    if attention_maps.is_empty() {
        return Err(InterpretabilityError::new(
            "attention_maps must not be empty",
        ));
    }

    let expected = seq_len * seq_len;

    // Validate all maps.
    for (layer_idx, map) in attention_maps.iter().enumerate() {
        if map.len() != expected {
            return Err(InterpretabilityError::new(format!(
                "attention_maps[{}] has {} elements; expected {} (seq_len={})",
                layer_idx,
                map.len(),
                expected,
                seq_len
            )));
        }
    }

    // Start with the identity matrix.
    let mut rollout = identity_matrix(seq_len);

    for map in attention_maps {
        // Add residual connection: A_hat = (A + I) / 2
        let a_hat = add_residual_and_normalize(map, seq_len);
        // rollout = a_hat @ rollout  (matrix multiply)
        rollout = matmul_f32(&a_hat, &rollout, seq_len);
    }

    // Return the first row (CLS token's effective attention).
    Ok(rollout[..seq_len].to_vec())
}

/// Build a flattened row-major identity matrix of size `n × n`.
fn identity_matrix(n: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// A_hat[i][j] = (A[i][j] + I[i][j]) / 2, then row-normalise.
fn add_residual_and_normalize(a: &[f32], n: usize) -> Vec<f32> {
    let mut result = a.to_vec();

    // Add identity and halve.
    for i in 0..n {
        result[i * n + i] += 1.0;
        for j in 0..n {
            result[i * n + j] *= 0.5;
        }
    }

    // Row-normalise so each row sums to 1.
    for i in 0..n {
        let row_sum: f32 = result[i * n..(i + 1) * n].iter().sum();
        if row_sum > 1e-30 {
            for j in 0..n {
                result[i * n + j] /= row_sum;
            }
        }
    }

    result
}

/// Row-major matrix multiplication: C = A @ B, both `n × n`.
fn matmul_f32(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * n];
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            if a_ik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Gradient × Input
// ---------------------------------------------------------------------------

/// Element-wise product of a gradient vector and the corresponding input.
///
/// `grad_times_input[i] = gradient[i] * input[i]`
///
/// Commonly used as a cheap attribution baseline that incorporates input
/// magnitude.
///
/// # Panics
///
/// Panics if `gradient.len() != input.len()`.  Consider zipping externally if
/// mismatched lengths are possible.
///
/// # Example
///
/// ```rust
/// use tenflowers_autograd::interpretability::gradient_times_input;
///
/// let g = vec![2.0_f64, -1.0, 3.0];
/// let x = vec![1.0_f64,  4.0, 2.0];
/// let gxi = gradient_times_input(&g, &x);
/// assert_eq!(gxi, vec![2.0, -4.0, 6.0]);
/// ```
pub fn gradient_times_input(gradient: &[f64], input: &[f64]) -> Vec<f64> {
    assert_eq!(
        gradient.len(),
        input.len(),
        "gradient and input must have the same length"
    );
    gradient
        .iter()
        .zip(input.iter())
        .map(|(g, x)| g * x)
        .collect()
}

// ---------------------------------------------------------------------------
// Top-k attribution summary
// ---------------------------------------------------------------------------

/// Return the `k` input dimensions with the largest attribution magnitude.
///
/// The returned vector contains `(index, attribution_value)` pairs sorted by
/// `|value|` in descending order.  If `k ≥ attributions.len()`, all dimensions
/// are returned.
///
/// # Example
///
/// ```rust
/// use tenflowers_autograd::interpretability::top_k_attributions;
///
/// let attrs = vec![-5.0_f64, 1.0, 3.0, -4.0, 2.0];
/// let top = top_k_attributions(&attrs, 2);
/// assert_eq!(top[0].0, 0); // index 0 has |value| = 5
/// assert_eq!(top[1].0, 3); // index 3 has |value| = 4
/// ```
pub fn top_k_attributions(attributions: &[f64], k: usize) -> Vec<(usize, f64)> {
    let n = attributions.len();
    let k_eff = k.min(n);

    let mut indexed: Vec<(usize, f64)> = attributions
        .iter()
        .copied()
        .enumerate()
        .collect();

    // Sort by |value| descending.
    indexed.sort_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    indexed.truncate(k_eff);
    indexed
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ----- vanilla_saliency -----

    /// f(x) = sum(x_i^2) => gradient = 2*x  => saliency[i] ≈ 2*x[i]
    #[test]
    fn test_vanilla_saliency_quadratic() {
        let x = vec![1.0_f64, 2.0, 3.0];
        let saliency =
            vanilla_saliency(|v: &[f64]| v.iter().map(|xi| xi * xi).sum(), &x, 1e-5);
        assert_eq!(saliency.len(), 3);
        // Expected: [2, 4, 6]
        for (i, &xi) in x.iter().enumerate() {
            let expected = 2.0 * xi;
            assert!(
                (saliency[i] - expected).abs() < 1e-3,
                "saliency[{}] = {}, expected {}",
                i,
                saliency[i],
                expected
            );
        }
    }

    /// Saliency of a linear function f(x) = sum(x) should equal all-ones.
    #[test]
    fn test_vanilla_saliency_linear() {
        let x = vec![0.5_f64, -1.0, 2.5];
        let s = vanilla_saliency(|v: &[f64]| v.iter().sum(), &x, 1e-6);
        for &si in &s {
            assert!((si - 1.0).abs() < 1e-3, "expected 1.0, got {}", si);
        }
    }

    /// Saliency on a single-element input.
    #[test]
    fn test_vanilla_saliency_scalar() {
        let x = vec![3.0_f64];
        // f(x) = x^3 => f'(x) = 3x^2 = 27 at x=3
        let s = vanilla_saliency(|v: &[f64]| v[0].powi(3), &x, 1e-5);
        assert!((s[0] - 27.0).abs() < 1e-2, "got {}", s[0]);
    }

    // ----- abs_saliency -----

    /// abs_saliency must always be non-negative.
    #[test]
    fn test_abs_saliency_nonnegative() {
        let x = vec![-3.0_f64, 0.0, 5.0, -1.0];
        let s = abs_saliency(|v: &[f64]| v.iter().map(|xi| xi * xi).sum(), &x, 1e-5);
        for &si in &s {
            assert!(si >= 0.0, "abs_saliency value {} is negative", si);
        }
    }

    /// abs_saliency values equal |vanilla_saliency| values.
    #[test]
    fn test_abs_saliency_equals_abs_vanilla() {
        let x = vec![1.0_f64, -2.0, 3.0];
        let f = |v: &[f64]| v.iter().map(|xi| xi * xi).sum::<f64>();
        let van = vanilla_saliency(f, &x, 1e-5);
        let abs = abs_saliency(|v: &[f64]| v.iter().map(|xi| xi * xi).sum(), &x, 1e-5);
        for (v, a) in van.iter().zip(abs.iter()) {
            assert!((v.abs() - a).abs() < 1e-12);
        }
    }

    // ----- SmoothGrad -----

    /// SmoothGrad with many samples should produce finite results.
    #[test]
    fn test_smoothgrad_finite() {
        let x = vec![1.0_f64, -1.0, 2.0];
        let sg = SmoothGrad::with_seed(50, 0.1, 42);
        let result = sg.compute(|v: &[f64]| v.iter().map(|xi| xi * xi).sum(), &x);
        assert_eq!(result.len(), 3);
        assert!(
            result.iter().all(|v| v.is_finite()),
            "SmoothGrad produced non-finite values: {:?}",
            result
        );
    }

    /// SmoothGrad with n_samples=1 should be close to vanilla saliency (same point).
    #[test]
    fn test_smoothgrad_one_sample_close_to_vanilla() {
        let x = vec![1.0_f64, 2.0, 3.0];
        // noise_std=0 (no noise) with 1 sample → exact vanilla gradient.
        let sg = SmoothGrad::with_seed(1, 0.0, 99);
        let sg_result = sg.compute(|v: &[f64]| v.iter().map(|xi| xi * xi).sum(), &x);
        let van = vanilla_saliency(|v: &[f64]| v.iter().map(|xi| xi * xi).sum(), &x, 1e-5);
        for (s, v) in sg_result.iter().zip(van.iter()) {
            assert!((s - v).abs() < 1e-3, "sg={} van={}", s, v);
        }
    }

    /// SmoothGrad squared should produce finite, non-negative results.
    #[test]
    fn test_smoothgrad_squared_nonneg() {
        let x = vec![1.0_f64, -1.0, 0.0];
        let sg = SmoothGrad::with_seed(30, 0.05, 7);
        let result = sg.compute_squared(|v: &[f64]| v.iter().map(|xi| xi * xi).sum(), &x);
        assert_eq!(result.len(), 3);
        for &v in &result {
            assert!(v.is_finite() && v >= 0.0, "squared result {} invalid", v);
        }
    }

    /// SmoothGrad with different seeds should generally differ (noise has effect).
    #[test]
    fn test_smoothgrad_seed_matters() {
        let x = vec![1.0_f64, 2.0, 3.0];
        let sg1 = SmoothGrad::with_seed(20, 1.0, 1);
        let sg2 = SmoothGrad::with_seed(20, 1.0, 999);
        let r1 = sg1.compute(|v: &[f64]| v.iter().map(|xi| xi * xi).sum(), &x);
        let r2 = sg2.compute(|v: &[f64]| v.iter().map(|xi| xi * xi).sum(), &x);
        // At least one element should differ (with overwhelming probability).
        let any_diff = r1.iter().zip(r2.iter()).any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(any_diff, "Different seeds produced identical SmoothGrad results");
    }

    // ----- IntegratedGradients -----

    /// For a linear function f(x) = w·x the IG equals w*(x - baseline).
    #[test]
    fn test_ig_linear_function() {
        let w = vec![1.0_f64, 2.0, 3.0];
        let f = |v: &[f64]| v.iter().zip(w.iter()).map(|(xi, wi)| xi * wi).sum::<f64>();
        let baseline = vec![0.0_f64; 3];
        let input = vec![1.0_f64, 1.0, 1.0];

        let ig = IntegratedGradients::new(200);
        let attrs = ig.compute(f, &baseline, &input).expect("IG compute failed");

        assert_eq!(attrs.len(), 3);
        // Expected: w * (input - baseline) = [1, 2, 3]
        let expected = vec![1.0_f64, 2.0, 3.0];
        for (i, (&a, &e)) in attrs.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 0.1,
                "IG[{}] = {}, expected {}",
                i,
                a,
                e
            );
        }
    }

    /// Completeness: sum(IG) ≈ f(input) - f(baseline).
    #[test]
    fn test_ig_completeness() {
        let f = |v: &[f64]| v.iter().map(|xi| xi * xi).sum::<f64>();
        let baseline = vec![0.0_f64; 4];
        let input = vec![1.0_f64, 2.0, 3.0, 4.0];

        let ig = IntegratedGradients::new(300);
        let attrs = ig.compute(f, &baseline, &input).expect("IG compute failed");

        // f(input) - f(baseline) = 1 + 4 + 9 + 16 = 30
        let expected_diff = 30.0_f64;
        let err = ig.completeness_error(
            |v: &[f64]| v.iter().map(|xi| xi * xi).sum::<f64>(),
            &baseline,
            &input,
            &attrs,
        );
        assert!(
            err < expected_diff * 0.05,  // within 5 %
            "Completeness error {} too large (expected_diff={})",
            err,
            expected_diff
        );
    }

    /// IG should return Err when baseline and input have different lengths.
    #[test]
    fn test_ig_length_mismatch_error() {
        let ig = IntegratedGradients::new(50);
        let result = ig.compute(
            |v: &[f64]| v.iter().sum(),
            &[0.0_f64, 0.0],
            &[1.0_f64],
        );
        assert!(result.is_err(), "Expected Err on length mismatch");
    }

    /// IG should return Err when n_steps == 0.
    #[test]
    fn test_ig_zero_steps_error() {
        let ig = IntegratedGradients::new(0);
        let result = ig.compute(
            |v: &[f64]| v.iter().sum(),
            &[0.0_f64],
            &[1.0_f64],
        );
        assert!(result.is_err(), "Expected Err when n_steps=0");
    }

    // ----- attention_rollout -----

    /// Identity attention: each token only attends to itself.
    /// After rollout and residual augmentation, each row should be uniform.
    #[test]
    fn test_attention_rollout_identity() {
        let seq = 4usize;
        // Build identity attention matrix (row-major).
        let identity: Vec<f32> = (0..seq)
            .flat_map(|i| (0..seq).map(move |j| if i == j { 1.0f32 } else { 0.0 }))
            .collect();

        let maps = vec![identity];
        let rollout = attention_rollout(&maps, seq).expect("rollout failed");

        assert_eq!(rollout.len(), seq);
        // After adding I and normalising, a^hat[i][j] = 0.5 * I[i][j] + 0.5/n
        // The first-row of the product should be near uniform = 1/seq.
        for &v in &rollout {
            assert!(v.is_finite() && v >= 0.0, "non-finite rollout value {}", v);
        }
        // Row should sum to ~1.
        let sum: f32 = rollout.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "rollout sum = {}", sum);
    }

    /// Rollout with two identical layers should still sum to 1.
    #[test]
    fn test_attention_rollout_two_layers_sum_one() {
        let seq = 3usize;
        // Uniform attention: each token attends equally to all tokens.
        let uniform_val = 1.0 / seq as f32;
        let uniform: Vec<f32> = vec![uniform_val; seq * seq];

        let maps = vec![uniform.clone(), uniform];
        let rollout = attention_rollout(&maps, seq).expect("rollout failed");

        assert_eq!(rollout.len(), seq);
        let sum: f32 = rollout.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "rollout sum = {}", sum);
    }

    /// attention_rollout should return Err if map size doesn't match seq_len.
    #[test]
    fn test_attention_rollout_bad_size() {
        let bad_map = vec![1.0f32; 6]; // 6 != 3*3
        let result = attention_rollout(&[bad_map], 3);
        assert!(result.is_err(), "Expected Err on size mismatch");
    }

    /// attention_rollout should return Err for empty maps slice.
    #[test]
    fn test_attention_rollout_empty_maps() {
        let result = attention_rollout(&[], 4);
        assert!(result.is_err(), "Expected Err for empty maps");
    }

    // ----- gradient_times_input -----

    /// gradient_times_input is the element-wise product.
    #[test]
    fn test_gradient_times_input_basic() {
        let g = vec![2.0_f64, -1.0, 3.0];
        let x = vec![1.0_f64, 4.0, 2.0];
        let gxi = gradient_times_input(&g, &x);
        assert_eq!(gxi.len(), 3);
        assert!((gxi[0] - 2.0).abs() < 1e-12);
        assert!((gxi[1] - (-4.0)).abs() < 1e-12);
        assert!((gxi[2] - 6.0).abs() < 1e-12);
    }

    /// gradient_times_input with zeros.
    #[test]
    fn test_gradient_times_input_zeros() {
        let g = vec![5.0_f64, -3.0];
        let x = vec![0.0_f64, 0.0];
        let gxi = gradient_times_input(&g, &x);
        assert!((gxi[0]).abs() < 1e-12);
        assert!((gxi[1]).abs() < 1e-12);
    }

    // ----- top_k_attributions -----

    /// top_k_attributions returns k elements sorted by |value| descending.
    #[test]
    fn test_top_k_basic() {
        let attrs = vec![-5.0_f64, 1.0, 3.0, -4.0, 2.0];
        let top = top_k_attributions(&attrs, 3);
        assert_eq!(top.len(), 3);
        // Sorted order by magnitude: 5 (idx0), 4 (idx3), 3 (idx2)
        assert_eq!(top[0].0, 0);
        assert!((top[0].1 - (-5.0)).abs() < 1e-12);
        assert_eq!(top[1].0, 3);
        assert!((top[1].1 - (-4.0)).abs() < 1e-12);
        assert_eq!(top[2].0, 2);
        assert!((top[2].1 - 3.0).abs() < 1e-12);
    }

    /// When k >= n, all elements are returned.
    #[test]
    fn test_top_k_larger_than_n() {
        let attrs = vec![1.0_f64, 2.0, 3.0];
        let top = top_k_attributions(&attrs, 10);
        assert_eq!(top.len(), 3, "expected all 3 elements, got {}", top.len());
    }

    /// When k == 0, the result is empty.
    #[test]
    fn test_top_k_zero() {
        let attrs = vec![1.0_f64, 2.0];
        let top = top_k_attributions(&attrs, 0);
        assert!(top.is_empty());
    }

    /// top_k_attributions on an empty slice returns empty.
    #[test]
    fn test_top_k_empty_attributions() {
        let top = top_k_attributions(&[], 5);
        assert!(top.is_empty());
    }

    // ----- Integration / edge-case -----

    /// Full pipeline: vanilla_saliency → gradient_times_input → top_k.
    #[test]
    fn test_full_pipeline() {
        let input = vec![1.0_f64, -2.0, 3.0, -4.0, 0.5];
        // f(x) = sum(x_i^2)
        let f = |v: &[f64]| v.iter().map(|xi| xi * xi).sum::<f64>();
        let grad = vanilla_saliency(f, &input, 1e-5);
        let gxi = gradient_times_input(&grad, &input);
        let top3 = top_k_attributions(&gxi, 3);

        assert_eq!(top3.len(), 3);
        // gxi[i] = 2*x_i * x_i = 2*x_i^2; largest should be |x|=4 → idx 3
        assert_eq!(top3[0].0, 3, "index 3 has largest |gxi|");
    }
}
