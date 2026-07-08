//! Gradient parity validation: compare autodiff gradients against numeric finite differences.
//!
//! This module provides numeric gradient checking to validate that analytically computed
//! gradients (e.g., from an autodiff engine) match numerical finite-difference approximations.
//! It is used to catch bugs in gradient computation during development and testing.
//!
//! # Central vs. Forward Differences
//!
//! **Central differences** (default):
//!
//! ```text
//! df/dx_i ≈ (f(x + h·eᵢ) - f(x - h·eᵢ)) / (2h)   — O(h²) error
//! ```
//!
//! **Forward differences** (cheaper, less accurate):
//!
//! ```text
//! df/dx_i ≈ (f(x + h·eᵢ) - f(x)) / h              — O(h) error
//! ```
//!
//! # Usage
//!
//! ```rust,no_run
//! use tenflowers::gradient_parity::GradientParityChecker;
//!
//! let checker = GradientParityChecker::new(1e-5, 1e-4);
//! let result = checker.check_scalar_function(
//!     |x| x.iter().map(|&v| v * v).sum::<f32>(),   // f(x) = sum(x²)
//!     &[1.0f32, 2.0f32, 3.0f32],                    // input x
//!     &[2.0f32, 4.0f32, 6.0f32],                    // expected gradient: df/dx_i = 2·x_i
//! ).unwrap();
//! assert!(result.is_passing());
//! ```

use tenflowers_core::{Result, TensorError};

// ─── Result type ──────────────────────────────────────────────────────────────

/// Result of a gradient parity check.
///
/// Captures absolute errors, relative errors, RMS error, a pass/fail verdict,
/// a human-readable summary, per-element error values, and indices of elements
/// that individually failed the tolerance test.
#[derive(Debug, Clone)]
pub struct GradientParityResult {
    /// Maximum absolute error: `max |analytical_i - numeric_i|`
    pub max_abs_error: f32,
    /// Maximum relative error: `max |analytical_i - numeric_i| / max(|analytical_i|, |numeric_i|, ε)`
    pub max_rel_error: f32,
    /// Root-mean-square error across all elements
    pub rms_error: f32,
    /// `true` when `max_abs_error < atol` **or** `max_rel_error < rtol`
    pub passed: bool,
    /// Human-readable one-line summary
    pub summary: String,
    /// Per-element signed errors: `analytical_i - numeric_i`
    pub element_errors: Vec<f32>,
    /// Indices where both `|error| >= atol` and `rel_error >= rtol`
    pub failed_indices: Vec<usize>,
}

impl GradientParityResult {
    /// Returns `true` when the check passed.
    pub fn is_passing(&self) -> bool {
        self.passed
    }

    /// Returns a multi-line human-readable report string.
    pub fn format_report(&self) -> String {
        let status = if self.passed { "PASS" } else { "FAIL" };
        let n_failed = self.failed_indices.len();
        let n_total = self.element_errors.len();
        format!(
            "[{}] gradient parity\n\
             max_abs_error = {:.3e}  max_rel_error = {:.3e}  rms_error = {:.3e}\n\
             failed elements: {}/{}\n\
             failed indices: {:?}",
            status,
            self.max_abs_error,
            self.max_rel_error,
            self.rms_error,
            n_failed,
            n_total,
            self.failed_indices,
        )
    }
}

// ─── Checker ──────────────────────────────────────────────────────────────────

/// Configuration and entry-point for gradient parity checking.
///
/// An element passes when:
/// ```text
///   |analytical_i - numeric_i| < atol
///   OR
///   |analytical_i - numeric_i| / max(|analytical_i|, |numeric_i|, 1e-8) < rtol
/// ```
///
/// The overall check passes when **all** elements individually pass.
pub struct GradientParityChecker {
    /// Absolute tolerance.
    pub atol: f32,
    /// Relative tolerance.
    pub rtol: f32,
    /// Finite-difference step size `h`.
    pub h: f32,
    /// When `true`, use central differences `(f(x+h) - f(x-h)) / 2h` (default).
    /// When `false`, use forward differences `(f(x+h) - f(x)) / h`.
    pub use_central_diff: bool,
}

impl GradientParityChecker {
    /// Create a checker with the supplied tolerances.
    ///
    /// Defaults: `h = 1e-3`, central differences enabled.
    pub fn new(atol: f32, rtol: f32) -> Self {
        Self {
            atol,
            rtol,
            h: 1e-3,
            use_central_diff: true,
        }
    }

    /// Override the finite-difference step size.
    pub fn with_h(mut self, h: f32) -> Self {
        self.h = h;
        self
    }

    /// Switch to forward differences: `(f(x+h) - f(x)) / h`.
    ///
    /// Forward differences are cheaper (one fewer function evaluation per
    /// dimension) but introduce `O(h)` truncation error rather than `O(h²)`.
    pub fn with_forward_diff(mut self) -> Self {
        self.use_central_diff = false;
        self
    }

    /// Check the gradient of a scalar-output function using central or forward
    /// differences (depending on [`use_central_diff`](Self::use_central_diff)).
    ///
    /// # Arguments
    ///
    /// * `f` — A pure function mapping `&[f32]` → `f32`.
    /// * `input` — The point at which to evaluate the gradient.
    /// * `analytical_grad` — The analytically computed gradient to validate.
    ///   Must have the same length as `input`.
    ///
    /// # Errors
    ///
    /// Returns `Err` when `input` and `analytical_grad` have different lengths.
    pub fn check_scalar_function<F>(
        &self,
        f: F,
        input: &[f32],
        analytical_grad: &[f32],
    ) -> Result<GradientParityResult>
    where
        F: Fn(&[f32]) -> f32,
    {
        validate_lengths(input, analytical_grad, "check_scalar_function")?;
        let numeric_grad = if self.use_central_diff {
            compute_central_diff_gradient(&f, input, self.h)
        } else {
            compute_forward_diff_gradient(&f, input, self.h)
        };
        Ok(gradients_are_close(
            analytical_grad,
            &numeric_grad,
            self.atol,
            self.rtol,
        ))
    }

    /// Check the gradient of a scalar-output function using **forward** differences,
    /// regardless of the `use_central_diff` setting.
    ///
    /// # Arguments
    ///
    /// * `f` — A pure function mapping `&[f32]` → `f32`.
    /// * `input` — The point at which to evaluate the gradient.
    /// * `analytical_grad` — The analytically computed gradient to validate.
    ///   Must have the same length as `input`.
    ///
    /// # Errors
    ///
    /// Returns `Err` when lengths differ.
    pub fn check_forward_diff<F>(
        &self,
        f: F,
        input: &[f32],
        analytical_grad: &[f32],
    ) -> Result<GradientParityResult>
    where
        F: Fn(&[f32]) -> f32,
    {
        validate_lengths(input, analytical_grad, "check_forward_diff")?;
        let numeric_grad = compute_forward_diff_gradient(&f, input, self.h);
        Ok(gradients_are_close(
            analytical_grad,
            &numeric_grad,
            self.atol,
            self.rtol,
        ))
    }

    /// Convenience wrapper that returns `true` on pass, `false` on failure or error.
    ///
    /// Uses the same difference method as [`check_scalar_function`](Self::check_scalar_function).
    pub fn quick_check<F>(&self, f: F, input: &[f32], analytical_grad: &[f32]) -> bool
    where
        F: Fn(&[f32]) -> f32,
    {
        self.check_scalar_function(f, input, analytical_grad)
            .map(|r| r.passed)
            .unwrap_or(false)
    }
}

// ─── Jacobian ─────────────────────────────────────────────────────────────────

/// Compute the numeric Jacobian of a vector-to-vector function.
///
/// Returns a matrix of shape `[output_size, input_size]` where entry `[i, j]`
/// approximates `∂f_i / ∂x_j`.
///
/// # Arguments
///
/// * `f` — A pure function mapping `&[f32]` → `Vec<f32>`.
/// * `input` — The point at which to evaluate the Jacobian.
/// * `output_size` — The number of output elements of `f`.
/// * `h` — The finite-difference step size.
/// * `use_central` — When `true`, use central differences `O(h²)`.
///   When `false`, use forward differences `O(h)`.
///
/// # Errors
///
/// Returns `Err` when `output_size` is zero but `input` is non-empty,
/// or when `f` returns a slice of length other than `output_size`.
pub fn numeric_jacobian<F>(
    f: F,
    input: &[f32],
    output_size: usize,
    h: f32,
    use_central: bool,
) -> Result<Vec<Vec<f32>>>
where
    F: Fn(&[f32]) -> Vec<f32>,
{
    let n = input.len();

    // Validate: call once to confirm output_size matches what f actually returns.
    if n > 0 || output_size > 0 {
        let f_x = f(input);
        if f_x.len() != output_size {
            return Err(TensorError::InvalidArgument {
                operation: "numeric_jacobian".to_string(),
                reason: format!(
                    "output_size is {output_size} but f returned {} elements",
                    f_x.len()
                ),
                context: None,
            });
        }
    }

    // Initialise Jacobian rows.  jacobian[i][j] = ∂f_i/∂x_j
    let mut jacobian: Vec<Vec<f32>> = (0..output_size).map(|_| vec![0.0f32; n]).collect();

    let mut x_perturbed = input.to_vec();

    for j in 0..n {
        let orig = x_perturbed[j];

        if use_central {
            // f(x + h·eⱼ)
            x_perturbed[j] = orig + h;
            let f_plus = f(&x_perturbed);

            // f(x - h·eⱼ)
            x_perturbed[j] = orig - h;
            let f_minus = f(&x_perturbed);

            let inv_2h = 1.0 / (2.0 * h);
            for i in 0..output_size {
                jacobian[i][j] = (f_plus[i] - f_minus[i]) * inv_2h;
            }
        } else {
            // f(x)  — we already evaluated it above for validation, but
            // to keep things simple and avoid caching logic we re-evaluate.
            let f_x = f(input);

            // f(x + h·eⱼ)
            x_perturbed[j] = orig + h;
            let f_plus = f(&x_perturbed);

            let inv_h = 1.0 / h;
            for i in 0..output_size {
                jacobian[i][j] = (f_plus[i] - f_x[i]) * inv_h;
            }
        }

        // Restore the element.
        x_perturbed[j] = orig;
    }

    Ok(jacobian)
}

// ─── Standalone comparison helper ─────────────────────────────────────────────

/// Check whether two gradient vectors are approximately equal under the given
/// absolute and relative tolerances.
///
/// An element passes when:
/// ```text
///   |analytical_i - numeric_i| < atol
///   OR
///   |analytical_i - numeric_i| / max(|analytical_i|, |numeric_i|, 1e-8) < rtol
/// ```
///
/// The overall [`GradientParityResult::passed`] field is `true` when **all**
/// elements pass.
///
/// # Panics
///
/// Does not panic; if the slices have different lengths the comparison is
/// performed only over the shorter one (the summary will note the length
/// mismatch).
pub fn gradients_are_close(
    analytical: &[f32],
    numeric: &[f32],
    atol: f32,
    rtol: f32,
) -> GradientParityResult {
    let len = analytical.len().min(numeric.len());
    let length_mismatch = analytical.len() != numeric.len();

    let mut element_errors = Vec::with_capacity(len);
    let mut failed_indices = Vec::new();
    let mut max_abs_error = 0.0f32;
    let mut max_rel_error = 0.0f32;
    let mut sum_sq_error = 0.0f32;

    const EPS: f32 = 1e-8;

    for i in 0..len {
        let a = analytical[i];
        let n = numeric[i];
        let abs_err = (a - n).abs();
        let rel_err = abs_err / a.abs().max(n.abs()).max(EPS);

        element_errors.push(a - n);

        if abs_err > max_abs_error {
            max_abs_error = abs_err;
        }
        if rel_err > max_rel_error {
            max_rel_error = rel_err;
        }
        sum_sq_error += abs_err * abs_err;

        // Element fails when it satisfies neither tolerance.
        if abs_err >= atol && rel_err >= rtol {
            failed_indices.push(i);
        }
    }

    let rms_error = if len > 0 {
        (sum_sq_error / len as f32).sqrt()
    } else {
        0.0
    };

    // Overall pass: no individually failing elements AND no length mismatch.
    let passed = failed_indices.is_empty() && !length_mismatch;

    let status = if passed { "PASS" } else { "FAIL" };
    let summary = if length_mismatch {
        format!(
            "[{}] length mismatch: analytical={} numeric={} \
             max_abs={:.3e} max_rel={:.3e} rms={:.3e} failed={}/{}",
            status,
            analytical.len(),
            numeric.len(),
            max_abs_error,
            max_rel_error,
            rms_error,
            failed_indices.len(),
            len,
        )
    } else {
        format!(
            "[{}] max_abs={:.3e} max_rel={:.3e} rms={:.3e} failed={}/{}",
            status,
            max_abs_error,
            max_rel_error,
            rms_error,
            failed_indices.len(),
            len,
        )
    };

    GradientParityResult {
        max_abs_error,
        max_rel_error,
        rms_error,
        passed,
        summary,
        element_errors,
        failed_indices,
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Validate that `input` and `grad` have the same length, returning a
/// `TensorError::InvalidArgument` when they differ.
fn validate_lengths(input: &[f32], grad: &[f32], operation: &str) -> Result<()> {
    if input.len() != grad.len() {
        return Err(TensorError::InvalidArgument {
            operation: operation.to_string(),
            reason: format!(
                "input length ({}) differs from analytical_grad length ({})",
                input.len(),
                grad.len()
            ),
            context: None,
        });
    }
    Ok(())
}

/// Compute the gradient of `f: &[f32] -> f32` at `input` using central differences.
///
/// `df/dx_i ≈ (f(x + h·eᵢ) - f(x - h·eᵢ)) / (2h)`
fn compute_central_diff_gradient<F>(f: &F, input: &[f32], h: f32) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    let n = input.len();
    let mut grad = vec![0.0f32; n];
    let mut x = input.to_vec();
    let inv_2h = 1.0 / (2.0 * h);

    for i in 0..n {
        let orig = x[i];

        x[i] = orig + h;
        let f_plus = f(&x);

        x[i] = orig - h;
        let f_minus = f(&x);

        grad[i] = (f_plus - f_minus) * inv_2h;

        x[i] = orig; // restore
    }

    grad
}

/// Compute the gradient of `f: &[f32] -> f32` at `input` using forward differences.
///
/// `df/dx_i ≈ (f(x + h·eᵢ) - f(x)) / h`
fn compute_forward_diff_gradient<F>(f: &F, input: &[f32], h: f32) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    let n = input.len();
    let f_x = f(input);
    let mut grad = vec![0.0f32; n];
    let mut x = input.to_vec();
    let inv_h = 1.0 / h;

    for i in 0..n {
        let orig = x[i];

        x[i] = orig + h;
        let f_plus = f(&x);

        grad[i] = (f_plus - f_x) * inv_h;

        x[i] = orig; // restore
    }

    grad
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Default tolerances used across most tests.
    const ATOL: f32 = 1e-3;
    const RTOL: f32 = 1e-2;

    // ── 1 ── f(x) = x², grad = 2x, single-element inputs ─────────────────

    #[test]
    fn test_check_quadratic() {
        let checker = GradientParityChecker::new(ATOL, RTOL);
        // Test at three separate single-element inputs.
        for x_val in [1.0f32, 2.0f32, 3.0f32] {
            let input = vec![x_val];
            let analytical = vec![2.0 * x_val];
            let result = checker
                .check_scalar_function(|x| x[0] * x[0], &input, &analytical)
                .expect("check must succeed");
            assert!(
                result.is_passing(),
                "quadratic at x={x_val}: {}",
                result.summary
            );
        }
    }

    // ── 2 ── f(x) = sum(x_i²), grad = 2·x_i ─────────────────────────────

    #[test]
    fn test_check_sum_of_squares() {
        let checker = GradientParityChecker::new(ATOL, RTOL);
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let analytical: Vec<f32> = input.iter().map(|&v| 2.0 * v).collect();

        let result = checker
            .check_scalar_function(
                |x| x.iter().map(|&v| v * v).sum::<f32>(),
                &input,
                &analytical,
            )
            .expect("check must succeed");

        assert!(result.is_passing(), "sum_of_squares: {}", result.summary);
        assert!(result.max_abs_error < ATOL * 10.0);
    }

    // ── 3 ── f(x) = x[0]*x[1], grad = [x[1], x[0]] ──────────────────────

    #[test]
    fn test_check_product() {
        let checker = GradientParityChecker::new(ATOL, RTOL);
        let input = vec![3.0f32, 7.0f32];
        // grad[0] = x[1] = 7, grad[1] = x[0] = 3
        let analytical = vec![7.0f32, 3.0f32];

        let result = checker
            .check_scalar_function(|x| x[0] * x[1], &input, &analytical)
            .expect("check must succeed");

        assert!(result.is_passing(), "product: {}", result.summary);
    }

    // ── 4 ── relu gradient for positive inputs ────────────────────────────

    #[test]
    fn test_check_relu_gradient() {
        let checker = GradientParityChecker::new(ATOL, RTOL);
        // relu(x) = x for x > 0, grad = 1.
        let input: Vec<f32> = vec![0.5, 1.0, 2.5, 3.3];
        let analytical: Vec<f32> = vec![1.0; input.len()]; // all positive

        let result = checker
            .check_scalar_function(
                |x| x.iter().map(|&v| v.max(0.0)).sum::<f32>(),
                &input,
                &analytical,
            )
            .expect("check must succeed");

        assert!(result.is_passing(), "relu: {}", result.summary);
    }

    // ── 5 ── wrong gradient is detected ──────────────────────────────────

    #[test]
    fn test_check_fails_on_wrong_grad() {
        let checker = GradientParityChecker::new(1e-6, 1e-5); // tight tolerances
        let input = vec![2.0f32];
        // Correct grad = 2*2 = 4, but we pass 100.
        let wrong_grad = vec![100.0f32];

        let result = checker
            .check_scalar_function(|x| x[0] * x[0], &input, &wrong_grad)
            .expect("check must not error");

        assert!(
            !result.is_passing(),
            "should have failed but passed: {}",
            result.summary
        );
        assert!(!result.failed_indices.is_empty());
    }

    // ── 6 ── forward diff and central diff give similar results ───────────

    #[test]
    fn test_forward_diff_vs_central() {
        let central = GradientParityChecker::new(ATOL, RTOL);
        let forward = GradientParityChecker::new(ATOL, RTOL).with_forward_diff();

        let input: Vec<f32> = vec![1.0, 2.0, 3.0];
        let analytical: Vec<f32> = input.iter().map(|&v| 2.0 * v).collect();

        let r_central = central
            .check_scalar_function(
                |x| x.iter().map(|&v| v * v).sum::<f32>(),
                &input,
                &analytical,
            )
            .expect("central check must succeed");

        let r_forward = forward
            .check_forward_diff(
                |x| x.iter().map(|&v| v * v).sum::<f32>(),
                &input,
                &analytical,
            )
            .expect("forward check must succeed");

        assert!(r_central.is_passing(), "central: {}", r_central.summary);
        assert!(r_forward.is_passing(), "forward: {}", r_forward.summary);

        // Central should be more accurate (smaller max_abs_error) or at worst equal.
        // With h=1e-3, central O(h²)≈1e-6, forward O(h)≈1e-3.
        assert!(
            r_central.max_abs_error <= r_forward.max_abs_error + 1e-4,
            "central ({}) should not be worse than forward ({})",
            r_central.max_abs_error,
            r_forward.max_abs_error
        );
    }

    // ── 7 ── Jacobian of f(x) = x is identity matrix ─────────────────────

    #[test]
    fn test_numeric_jacobian_identity() {
        let n = 4usize;
        let input: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();

        let jacobian =
            numeric_jacobian(|x| x.to_vec(), &input, n, 1e-3, true).expect("jacobian must succeed");

        assert_eq!(jacobian.len(), n, "Jacobian row count");
        for (i, row) in jacobian.iter().enumerate().take(n) {
            assert_eq!(row.len(), n, "Jacobian col count");
            for (j, &got) in row.iter().enumerate().take(n) {
                let expected = if i == j { 1.0f32 } else { 0.0f32 };
                assert!(
                    (got - expected).abs() < 1e-2,
                    "J[{i}][{j}] = {got:.4}, expected {expected}"
                );
            }
        }
    }

    // ── 8 ── Jacobian of f(x) = A*x for known A ──────────────────────────

    #[test]
    fn test_numeric_jacobian_linear() {
        // A = [[1,2],[3,4]], x = [1,1]  → f(x) = [3,7]
        //   → J = [[1,2],[3,4]]
        let a = [[1.0f32, 2.0], [3.0, 4.0]];
        let input = vec![1.0f32, 1.0f32];

        let jacobian = numeric_jacobian(
            |x| {
                vec![
                    a[0][0] * x[0] + a[0][1] * x[1],
                    a[1][0] * x[0] + a[1][1] * x[1],
                ]
            },
            &input,
            2,
            1e-3,
            true,
        )
        .expect("jacobian must succeed");

        for i in 0..2 {
            for j in 0..2 {
                let expected = a[i][j];
                let got = jacobian[i][j];
                assert!(
                    (got - expected).abs() < 1e-2,
                    "J[{i}][{j}] = {got:.4}, expected {expected}"
                );
            }
        }
    }

    // ── 9 ── gradients_are_close passes on identical inputs ───────────────

    #[test]
    fn test_gradients_are_close_pass() {
        let analytical = vec![1.0f32, 2.0, 3.0];
        let numeric = analytical.clone();
        let result = gradients_are_close(&analytical, &numeric, ATOL, RTOL);
        assert!(result.passed, "identical gradients should pass");
        assert_eq!(result.max_abs_error, 0.0);
        assert!(result.failed_indices.is_empty());
    }

    // ── 10 ── gradients_are_close fails on very different inputs ──────────

    #[test]
    fn test_gradients_are_close_fail() {
        let analytical = vec![1.0f32, 0.0, -1.0];
        let numeric = vec![100.0f32, -100.0, 100.0];
        let result = gradients_are_close(&analytical, &numeric, 1e-6, 1e-6);
        assert!(!result.passed, "very different gradients should fail");
        assert_eq!(result.failed_indices.len(), 3);
    }

    // ── 11 ── empty input handled gracefully ──────────────────────────────

    #[test]
    fn test_empty_input() {
        let checker = GradientParityChecker::new(ATOL, RTOL);
        let input: Vec<f32> = vec![];
        let analytical: Vec<f32> = vec![];

        let result = checker
            .check_scalar_function(|_x| 0.0f32, &input, &analytical)
            .expect("empty input must succeed");

        // Zero-element check trivially passes.
        assert!(result.is_passing(), "empty: {}", result.summary);
        assert_eq!(result.rms_error, 0.0);
        assert!(result.failed_indices.is_empty());
    }

    // ── 12 ── large input performance ─────────────────────────────────────
    //
    // This test validates that the checker processes 1000 elements without
    // error and returns the correct number of per-element results.
    //
    // Tolerance note: with 1000 f32 elements summed in the objective function,
    // floating-point rounding accumulates across the sum, causing the
    // finite-difference estimate to diverge from the analytical gradient by
    // up to ~5e-2 for the larger x values.  We therefore use tolerances
    // appropriate for a large-scale f32 sum rather than the tight defaults.
    #[test]
    fn test_large_input() {
        let n = 1000usize;
        // Use tolerances that account for f32 summation rounding over 1000 elements.
        let checker = GradientParityChecker::new(0.1, 0.15);
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.001 + 0.1).collect();
        let analytical: Vec<f32> = input.iter().map(|&v| 2.0 * v).collect();

        let result = checker
            .check_scalar_function(
                |x| x.iter().map(|&v| v * v).sum::<f32>(),
                &input,
                &analytical,
            )
            .expect("large input check must succeed");

        assert!(result.is_passing(), "large input: {}", result.summary);
        assert_eq!(result.element_errors.len(), n);
    }
}
