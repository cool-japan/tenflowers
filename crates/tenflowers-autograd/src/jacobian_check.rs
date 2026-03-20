//! # Jacobian Checker and Advanced Gradient Verification
//!
//! This module extends the numerical gradient checking infrastructure with:
//!
//! - **Jacobian Checker**: Full numerical Jacobian computation and comparison for
//!   vector-to-vector functions (R^n → R^m).
//! - **Per-element gradient verification**: Element-wise gradient checking for
//!   scalar functions with detailed per-element error reporting.
//! - **Higher-order numerical derivatives**: Numerical Hessian diagonal and
//!   Hessian-vector products (HVP) via central finite differences.
//! - **Assertion helper**: Panic-on-failure convenience function for test code.
//!
//! ## Design
//!
//! All computations operate on raw `f64` slices to avoid coupling to the `Tensor`
//! abstraction, making the checker easy to apply to any function signature.  No
//! `unwrap()` calls appear in production code; errors are propagated via the
//! dedicated error types or returned as `Result`.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use tenflowers_autograd::jacobian_check::{JacobianChecker, check_gradient_per_element};
//!
//! // Verify Jacobian of a linear map y = W * x
//! let w = vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]];
//! let x = vec![1.0_f64, 0.5];
//!
//! let checker = JacobianChecker::new();
//! let f = |x: &[f64]| -> Vec<f64> {
//!     let y0 = w[0][0] * x[0] + w[0][1] * x[1];
//!     let y1 = w[1][0] * x[0] + w[1][1] * x[1];
//!     vec![y0, y1]
//! };
//!
//! // Analytical Jacobian is just W
//! let analytical = w.clone();
//! let report = checker.check(f, &x, &analytical).expect("check failed");
//! assert!(report.passed);
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Report returned when a Jacobian check passes.
///
/// Contains aggregate statistics as well as the count of per-element failures.
#[derive(Debug, Clone)]
pub struct JacobianCheckReport {
    /// Maximum absolute difference across all Jacobian entries.
    pub max_abs_error: f64,
    /// Maximum relative difference across all Jacobian entries.
    pub max_rel_error: f64,
    /// Total number of Jacobian elements that were compared.
    pub num_elements_checked: usize,
    /// Number of elements that exceeded the tolerance thresholds.
    pub num_failures: usize,
    /// `true` when `num_failures == 0`.
    pub passed: bool,
}

impl fmt::Display for JacobianCheckReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "JacobianCheckReport {{")?;
        writeln!(
            f,
            "  passed: {}, failures: {}/{}",
            self.passed, self.num_failures, self.num_elements_checked
        )?;
        writeln!(f, "  max_abs_error: {:.3e}", self.max_abs_error)?;
        writeln!(f, "  max_rel_error: {:.3e}", self.max_rel_error)?;
        write!(f, "}}")
    }
}

/// Error returned when a Jacobian check fails.
///
/// Carries the full [`JacobianCheckReport`] plus details about the single worst
/// mismatching element.
#[derive(Debug, Clone)]
pub struct JacobianCheckError {
    /// Aggregate statistics for the failed check.
    pub report: JacobianCheckReport,
    /// `(output_idx, input_idx)` of the element with the largest absolute error.
    pub worst_element: (usize, usize),
    /// Analytical Jacobian value at the worst element.
    pub analytical_value: f64,
    /// Numerical Jacobian value at the worst element.
    pub numerical_value: f64,
}

impl fmt::Display for JacobianCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Jacobian check failed: {} element(s) exceeded tolerance. \
             Worst mismatch at ({}, {}): analytical={:.6e}, numerical={:.6e}",
            self.report.num_failures,
            self.worst_element.0,
            self.worst_element.1,
            self.analytical_value,
            self.numerical_value,
        )
    }
}

impl std::error::Error for JacobianCheckError {}

// ---------------------------------------------------------------------------
// Gradient error types (per-element check)
// ---------------------------------------------------------------------------

/// Report from [`check_gradient_per_element`].
#[derive(Debug, Clone)]
pub struct GradientReport {
    /// Maximum absolute error across all gradient elements.
    pub max_error: f64,
    /// Mean absolute error across all gradient elements.
    pub mean_error: f64,
    /// Number of input elements checked.
    pub num_elements: usize,
    /// Number of elements whose error exceeded `atol`.
    pub num_failures: usize,
    /// Per-element absolute errors in the same order as the input.
    pub element_errors: Vec<f64>,
}

impl fmt::Display for GradientReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GradientReport {{ max_error: {:.3e}, mean_error: {:.3e}, \
             failures: {}/{} }}",
            self.max_error, self.mean_error, self.num_failures, self.num_elements
        )
    }
}

/// Error returned when [`check_gradient_per_element`] detects mismatches.
#[derive(Debug, Clone)]
pub struct GradientError {
    /// Aggregate report for the failed check.
    pub report: GradientReport,
    /// Index of the element with the largest absolute error.
    pub worst_index: usize,
    /// Analytical gradient value at `worst_index`.
    pub analytical_value: f64,
    /// Numerical gradient value at `worst_index`.
    pub numerical_value: f64,
}

impl fmt::Display for GradientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Gradient check failed: {} element(s) exceeded tolerance. \
             Worst mismatch at index {}: analytical={:.6e}, numerical={:.6e}",
            self.report.num_failures,
            self.worst_index,
            self.analytical_value,
            self.numerical_value,
        )
    }
}

impl std::error::Error for GradientError {}

// ---------------------------------------------------------------------------
// JacobianChecker
// ---------------------------------------------------------------------------

/// Verify the Jacobian of a vector-valued function f: R^n → R^m numerically.
///
/// Uses central finite differences with configurable step size and tolerance.
///
/// # Example
///
/// ```rust
/// use tenflowers_autograd::jacobian_check::JacobianChecker;
///
/// let checker = JacobianChecker::new();
/// let f = |x: &[f64]| vec![x[0] * x[0], x[1] * x[1]];
/// let x = vec![2.0_f64, 3.0];
///
/// // Analytical Jacobian of [x0^2, x1^2]: diagonal [[2*x0, 0], [0, 2*x1]]
/// let analytical = vec![
///     vec![4.0, 0.0],
///     vec![0.0, 6.0],
/// ];
/// let report = checker.check(f, &x, &analytical).expect("check failed");
/// assert!(report.passed);
/// ```
#[derive(Debug, Clone)]
pub struct JacobianChecker {
    /// Finite difference step size (default 1e-5).
    pub eps: f64,
    /// Absolute tolerance for element comparison (default 1e-4).
    pub atol: f64,
    /// Relative tolerance for element comparison (default 1e-3).
    pub rtol: f64,
}

impl Default for JacobianChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl JacobianChecker {
    /// Create a checker with sensible defaults: `eps=1e-5`, `atol=1e-4`, `rtol=1e-3`.
    pub fn new() -> Self {
        Self {
            eps: 1e-5,
            atol: 1e-4,
            rtol: 1e-3,
        }
    }

    /// Create a checker with a custom finite-difference step size.
    ///
    /// `atol` and `rtol` retain their defaults.
    pub fn with_eps(eps: f64) -> Self {
        Self {
            eps,
            atol: 1e-4,
            rtol: 1e-3,
        }
    }

    /// Compute the numerical Jacobian of `f: R^n → R^m`.
    ///
    /// Returns an `m × n` matrix (outer Vec = output dimension, inner Vec = input
    /// dimension) of central-difference partial derivatives:
    ///
    /// ```text
    /// J[i][j] = (f(x + ε e_j)[i] - f(x - ε e_j)[i]) / (2ε)
    /// ```
    ///
    /// # Panics
    ///
    /// Does not panic; returns an empty outer Vec if `x` is empty.
    pub fn numerical_jacobian<F>(&self, f: F, x: &[f64]) -> Vec<Vec<f64>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }

        // Determine output dimension from a single evaluation.
        let f0 = f(x);
        let m = f0.len();
        if m == 0 {
            return Vec::new();
        }

        // Allocate J as m × n filled with 0.
        let mut jacobian: Vec<Vec<f64>> = (0..m).map(|_| vec![0.0_f64; n]).collect();

        let mut x_perturbed = x.to_vec();
        let two_eps = 2.0 * self.eps;

        for j in 0..n {
            let orig = x_perturbed[j];

            // f(x + ε e_j)
            x_perturbed[j] = orig + self.eps;
            let f_plus = f(&x_perturbed);

            // f(x - ε e_j)
            x_perturbed[j] = orig - self.eps;
            let f_minus = f(&x_perturbed);

            // Restore
            x_perturbed[j] = orig;

            for i in 0..m {
                jacobian[i][j] = (f_plus[i] - f_minus[i]) / two_eps;
            }
        }

        jacobian
    }

    /// Compare an analytical Jacobian against a freshly computed numerical one.
    ///
    /// Returns `Ok(JacobianCheckReport)` when every element satisfies:
    ///
    /// ```text
    /// |analytical[i][j] - numerical[i][j]| <= atol
    ///   OR
    /// |analytical[i][j] - numerical[i][j]| / max(|analytical|, |numerical|, 1e-12) <= rtol
    /// ```
    ///
    /// Returns `Err(JacobianCheckError)` otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`JacobianCheckError`] when any element exceeds the configured
    /// tolerance thresholds.
    pub fn check<F>(
        &self,
        f: F,
        x: &[f64],
        analytical_jacobian: &[Vec<f64>],
    ) -> Result<JacobianCheckReport, JacobianCheckError>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let numerical = self.numerical_jacobian(&f, x);

        let m = analytical_jacobian.len();
        let n = if m > 0 { analytical_jacobian[0].len() } else { 0 };

        let mut max_abs_error = 0.0_f64;
        let mut max_rel_error = 0.0_f64;
        let mut num_failures = 0_usize;
        let mut worst_element = (0_usize, 0_usize);
        let mut worst_analytical = 0.0_f64;
        let mut worst_numerical = 0.0_f64;
        let mut worst_abs_error = f64::NEG_INFINITY;

        for i in 0..m {
            let num_row_len = if i < numerical.len() { numerical[i].len() } else { 0 };
            for j in 0..n {
                let a = analytical_jacobian[i][j];
                let num_val = if i < numerical.len() && j < num_row_len {
                    numerical[i][j]
                } else {
                    0.0
                };

                let abs_err = (a - num_val).abs();
                let denom = a.abs().max(num_val.abs()).max(1e-12);
                let rel_err = abs_err / denom;

                if abs_err > max_abs_error {
                    max_abs_error = abs_err;
                }
                if rel_err > max_rel_error {
                    max_rel_error = rel_err;
                }

                // An element fails if it exceeds BOTH abs and rel tolerance.
                let passes = abs_err <= self.atol || rel_err <= self.rtol;
                if !passes {
                    num_failures += 1;
                    if abs_err > worst_abs_error {
                        worst_abs_error = abs_err;
                        worst_element = (i, j);
                        worst_analytical = a;
                        worst_numerical = num_val;
                    }
                }
            }
        }

        let num_elements_checked = m * n;
        let passed = num_failures == 0;

        let report = JacobianCheckReport {
            max_abs_error,
            max_rel_error,
            num_elements_checked,
            num_failures,
            passed,
        };

        if passed {
            Ok(report)
        } else {
            Err(JacobianCheckError {
                report,
                worst_element,
                analytical_value: worst_analytical,
                numerical_value: worst_numerical,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Per-element gradient verification
// ---------------------------------------------------------------------------

/// Verify the gradient of a scalar function at each input element independently.
///
/// Uses central finite differences:
///
/// ```text
/// numerical_grad[i] = (f(x + ε e_i) - f(x - ε e_i)) / (2ε)
/// ```
///
/// Each element's absolute error is compared to `atol`.
///
/// # Errors
///
/// Returns [`GradientError`] when any element's error exceeds `atol`.
pub fn check_gradient_per_element<F>(
    f: F,
    x: &[f64],
    analytical_grad: &[f64],
    eps: f64,
    atol: f64,
) -> Result<GradientReport, GradientError>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    let mut element_errors = Vec::with_capacity(n);
    let mut num_failures = 0_usize;
    let mut worst_index = 0_usize;
    let mut worst_abs_error = f64::NEG_INFINITY;
    let mut worst_analytical = 0.0_f64;
    let mut worst_numerical = 0.0_f64;

    let mut x_perturbed = x.to_vec();
    let two_eps = 2.0 * eps;

    for i in 0..n {
        let orig = x_perturbed[i];

        x_perturbed[i] = orig + eps;
        let f_plus = f(&x_perturbed);

        x_perturbed[i] = orig - eps;
        let f_minus = f(&x_perturbed);

        x_perturbed[i] = orig;

        let numerical = (f_plus - f_minus) / two_eps;
        let a = analytical_grad[i];
        let abs_err = (a - numerical).abs();

        element_errors.push(abs_err);

        if abs_err > atol {
            num_failures += 1;
            if abs_err > worst_abs_error {
                worst_abs_error = abs_err;
                worst_index = i;
                worst_analytical = a;
                worst_numerical = numerical;
            }
        }
    }

    let max_error = element_errors
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_error = if n > 0 {
        element_errors.iter().sum::<f64>() / n as f64
    } else {
        0.0
    };

    let report = GradientReport {
        max_error,
        mean_error,
        num_elements: n,
        num_failures,
        element_errors,
    };

    if num_failures == 0 {
        Ok(report)
    } else {
        Err(GradientError {
            report,
            worst_index,
            analytical_value: worst_analytical,
            numerical_value: worst_numerical,
        })
    }
}

// ---------------------------------------------------------------------------
// Higher-order numerical derivatives
// ---------------------------------------------------------------------------

/// Compute the diagonal of the numerical Hessian via central differences.
///
/// For each index `i`, this uses the second-order central difference formula:
///
/// ```text
/// H_ii = (f(x + ε e_i) - 2·f(x) + f(x - ε e_i)) / ε²
/// ```
///
/// # Arguments
///
/// - `f`   – scalar objective.
/// - `x`   – evaluation point.
/// - `eps` – finite-difference step size (e.g. `1e-4`).
///
/// # Returns
///
/// A `Vec<f64>` of length `n = x.len()`.
pub fn numerical_hessian_diagonal<F>(f: F, x: &[f64], eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    let f0 = f(x);
    let eps_sq = eps * eps;

    let mut diag = Vec::with_capacity(n);
    let mut x_perturbed = x.to_vec();

    for i in 0..n {
        let orig = x_perturbed[i];

        x_perturbed[i] = orig + eps;
        let f_plus = f(&x_perturbed);

        x_perturbed[i] = orig - eps;
        let f_minus = f(&x_perturbed);

        x_perturbed[i] = orig;

        diag.push((f_plus - 2.0 * f0 + f_minus) / eps_sq);
    }

    diag
}

/// Compute the Hessian-vector product (HVP) H·v via two-point finite difference.
///
/// Uses the formula:
///
/// ```text
/// (H·v)[i] = (∇f(x + ε·v)[i] - ∇f(x - ε·v)[i]) / (2ε)
/// ```
///
/// where each gradient component is itself approximated with a central difference.
/// This avoids materialising the full Hessian matrix.
///
/// # Arguments
///
/// - `f`   – scalar objective.
/// - `x`   – evaluation point (length n).
/// - `v`   – direction vector (length n).
/// - `eps` – outer finite-difference step size.
///
/// # Returns
///
/// `H·v` as a `Vec<f64>` of length `n`.
pub fn numerical_hvp<F>(f: F, x: &[f64], v: &[f64], eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    let inner_eps = 1e-5_f64;
    let two_inner_eps = 2.0 * inner_eps;

    // Helper: compute gradient of f at point p via central differences.
    let grad = |p: &[f64]| -> Vec<f64> {
        let mut g = Vec::with_capacity(n);
        let mut pp = p.to_vec();
        for i in 0..n {
            let orig = pp[i];
            pp[i] = orig + inner_eps;
            let fp = f(&pp);
            pp[i] = orig - inner_eps;
            let fm = f(&pp);
            pp[i] = orig;
            g.push((fp - fm) / two_inner_eps);
        }
        g
    };

    // x_plus  = x + ε·v
    // x_minus = x - ε·v
    let x_plus: Vec<f64> = x.iter().zip(v.iter()).map(|(xi, vi)| xi + eps * vi).collect();
    let x_minus: Vec<f64> = x.iter().zip(v.iter()).map(|(xi, vi)| xi - eps * vi).collect();

    let g_plus = grad(&x_plus);
    let g_minus = grad(&x_minus);

    let two_eps = 2.0 * eps;
    g_plus
        .iter()
        .zip(g_minus.iter())
        .map(|(gp, gm)| (gp - gm) / two_eps)
        .collect()
}

// ---------------------------------------------------------------------------
// Assertion convenience helper
// ---------------------------------------------------------------------------

/// Assert that the analytical gradient of a scalar function is numerically correct.
///
/// This function panics with a descriptive message if the check fails.  It is
/// intended for use inside unit tests where panicking on failure is the desired
/// behaviour.
///
/// # Panics
///
/// Panics when the gradient check fails.
pub fn assert_gradient_correct<F>(f: F, x: &[f64], analytical_grad: &[f64], eps: f64)
where
    F: Fn(&[f64]) -> f64,
{
    let atol = eps.sqrt() * 10.0; // generous tolerance relative to eps
    match check_gradient_per_element(f, x, analytical_grad, eps, atol) {
        Ok(report) => {
            // Sanity: if passed, max_error must be within atol.
            assert!(
                report.max_error <= atol,
                "assert_gradient_correct: unexpected max_error {:.3e} with atol {:.3e}",
                report.max_error,
                atol
            );
        }
        Err(err) => {
            panic!(
                "assert_gradient_correct: {err}\n  report: {}",
                err.report
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helper constants / closures
    // ------------------------------------------------------------------

    /// f(x) = sum(x[i]^2).  Analytical gradient: 2*x[i].
    fn sum_squares(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    fn sum_squares_grad(x: &[f64]) -> Vec<f64> {
        x.iter().map(|xi| 2.0 * xi).collect()
    }

    // ------------------------------------------------------------------
    // Jacobian tests
    // ------------------------------------------------------------------

    /// Test 1: numerical_jacobian on a linear map y = W*x (Jacobian = W).
    #[test]
    fn test_numerical_jacobian_linear_map() {
        // W = [[1, 2], [3, 4]]
        let w = [[1.0_f64, 2.0], [3.0, 4.0]];
        let x = vec![0.5_f64, -1.0];

        let checker = JacobianChecker::new();
        let f = |xv: &[f64]| -> Vec<f64> {
            vec![
                w[0][0] * xv[0] + w[0][1] * xv[1],
                w[1][0] * xv[0] + w[1][1] * xv[1],
            ]
        };

        let jac = checker.numerical_jacobian(f, &x);

        assert_eq!(jac.len(), 2, "output dimension should be 2");
        assert_eq!(jac[0].len(), 2, "input dimension should be 2");

        // Jacobian of linear map equals the weight matrix.
        let tol = 1e-9_f64;
        assert!((jac[0][0] - 1.0).abs() < tol, "J[0][0] should be 1");
        assert!((jac[0][1] - 2.0).abs() < tol, "J[0][1] should be 2");
        assert!((jac[1][0] - 3.0).abs() < tol, "J[1][0] should be 3");
        assert!((jac[1][1] - 4.0).abs() < tol, "J[1][1] should be 4");
    }

    /// Test 2: JacobianChecker::check passes for a correct analytical Jacobian.
    #[test]
    fn test_jacobian_check_passes_for_correct_jacobian() {
        let checker = JacobianChecker::new();
        // f(x) = [x0^2, x1^2]  =>  J = diag(2*x0, 2*x1)
        let x = vec![2.0_f64, 3.0];
        let f = |xv: &[f64]| vec![xv[0] * xv[0], xv[1] * xv[1]];

        let analytical = vec![
            vec![2.0 * x[0], 0.0],
            vec![0.0, 2.0 * x[1]],
        ];

        let result = checker.check(f, &x, &analytical);
        assert!(result.is_ok(), "Correct Jacobian should pass: {:?}", result);
        let report = result.expect("check should return Ok for correct Jacobian");
        assert!(report.passed);
        assert_eq!(report.num_failures, 0);
    }

    /// Test 3: JacobianChecker::check fails for a wrong analytical Jacobian.
    #[test]
    fn test_jacobian_check_fails_for_wrong_jacobian() {
        let checker = JacobianChecker::new();
        let x = vec![2.0_f64, 3.0];
        let f = |xv: &[f64]| vec![xv[0] * xv[0], xv[1] * xv[1]];

        // Intentionally wrong: supply identity instead of 2x diagonal.
        let wrong_analytical = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];

        let result = checker.check(f, &x, &wrong_analytical);
        assert!(result.is_err(), "Wrong Jacobian should fail");
        let err = result.expect_err("Expected Err for wrong Jacobian");
        assert!(err.report.num_failures > 0);
    }

    /// Test 4: Non-square Jacobian (3 outputs, 2 inputs).
    #[test]
    fn test_numerical_jacobian_non_square() {
        // f: R^2 -> R^3
        // f(x) = [x0+x1, x0*x1, x0^2]
        // J = [[1, 1], [x1, x0], [2*x0, 0]]
        let x = vec![1.0_f64, 2.0];
        let checker = JacobianChecker::new();

        let f = |xv: &[f64]| vec![
            xv[0] + xv[1],
            xv[0] * xv[1],
            xv[0] * xv[0],
        ];

        let analytical = vec![
            vec![1.0, 1.0],
            vec![x[1], x[0]],
            vec![2.0 * x[0], 0.0],
        ];

        let result = checker.check(f, &x, &analytical);
        assert!(result.is_ok(), "Non-square Jacobian check failed: {:?}", result);
    }

    /// Test 5: Jacobian of a sine-based function.
    #[test]
    fn test_numerical_jacobian_sine_function() {
        // f(x) = [sin(x0), cos(x1)]
        // J = [[cos(x0), 0], [0, -sin(x1)]]
        let x = vec![0.5_f64, 1.0];
        let checker = JacobianChecker::new();

        let f = |xv: &[f64]| vec![xv[0].sin(), xv[1].cos()];

        let analytical = vec![
            vec![x[0].cos(), 0.0],
            vec![0.0, -x[1].sin()],
        ];

        let result = checker.check(f, &x, &analytical);
        assert!(result.is_ok(), "Sine Jacobian check failed: {:?}", result);
    }

    /// Test 6: Jacobian of an exp-based function.
    #[test]
    fn test_numerical_jacobian_exp_function() {
        // f(x) = [exp(x0), exp(x1)]
        // J = diag(exp(x0), exp(x1))
        let x = vec![0.2_f64, -0.3];
        let checker = JacobianChecker::new();

        let f = |xv: &[f64]| vec![xv[0].exp(), xv[1].exp()];

        let analytical = vec![
            vec![x[0].exp(), 0.0],
            vec![0.0, x[1].exp()],
        ];

        let result = checker.check(f, &x, &analytical);
        assert!(result.is_ok(), "Exp Jacobian check failed: {:?}", result);
    }

    /// Test 7: Tolerance behaviour — pass with loose tolerance, fail with tight.
    #[test]
    fn test_jacobian_tolerance_behavior() {
        // f(x) = [x0^2]   at x=[3.0]
        // Analytical Jacobian deliberately off by 0.1: [[5.9]] instead of [[6.0]]
        let x = vec![3.0_f64];
        let f = |xv: &[f64]| vec![xv[0] * xv[0]];

        // Off-by-0.1 analytical entry
        let slightly_wrong = vec![vec![5.9_f64]];

        // Loose checker: atol=1.0 — should pass
        let loose = JacobianChecker {
            eps: 1e-5,
            atol: 1.0,
            rtol: 1.0,
        };
        assert!(
            loose.check(f.clone(), &x, &slightly_wrong).is_ok(),
            "Loose tolerance should pass"
        );

        // Tight checker: atol=1e-8 — should fail
        let tight = JacobianChecker {
            eps: 1e-5,
            atol: 1e-8,
            rtol: 1e-8,
        };
        assert!(
            tight.check(f, &x, &slightly_wrong).is_err(),
            "Tight tolerance should fail"
        );
    }

    // ------------------------------------------------------------------
    // Per-element gradient tests
    // ------------------------------------------------------------------

    /// Test 8: check_gradient_per_element on sum_squares, grad = 2*x.
    #[test]
    fn test_check_gradient_per_element_sum_squares() {
        let x = vec![1.0_f64, 2.0, 3.0, -1.5, 0.0];
        let grad = sum_squares_grad(&x);

        let result = check_gradient_per_element(sum_squares, &x, &grad, 1e-5, 1e-4);
        assert!(result.is_ok(), "sum_squares gradient check failed: {:?}", result);

        let report = result.expect("Expected Ok for sum_squares gradient");
        assert_eq!(report.num_elements, x.len());
        assert_eq!(report.num_failures, 0);
        assert!(report.max_error < 1e-4);
    }

    /// Test 9: check_gradient_per_element fails for wrong gradient.
    #[test]
    fn test_check_gradient_per_element_fails_for_wrong_gradient() {
        let x = vec![1.0_f64, 2.0, 3.0];
        let wrong_grad = vec![0.0, 0.0, 0.0]; // all zeros — clearly wrong

        let result = check_gradient_per_element(sum_squares, &x, &wrong_grad, 1e-5, 1e-6);
        assert!(result.is_err(), "Wrong gradient should fail");
        let err = result.expect_err("Expected Err for wrong gradient");
        assert!(err.report.num_failures > 0);
    }

    /// Test 10: Gradient of f(x) = sum(sin(x)), grad = cos(x).
    #[test]
    fn test_check_gradient_per_element_sine() {
        let x = vec![0.0_f64, 0.5, 1.0, 1.5, std::f64::consts::PI / 4.0];
        let analytical_grad: Vec<f64> = x.iter().map(|xi| xi.cos()).collect();
        let f = |xv: &[f64]| xv.iter().map(|xi| xi.sin()).sum::<f64>();

        let result = check_gradient_per_element(f, &x, &analytical_grad, 1e-5, 1e-4);
        assert!(result.is_ok(), "Sine gradient check failed: {:?}", result);
    }

    /// Test 11: Gradient of f(x) = sum(exp(x)), grad = exp(x).
    #[test]
    fn test_check_gradient_per_element_exp() {
        let x = vec![-1.0_f64, 0.0, 1.0, 2.0];
        let analytical_grad: Vec<f64> = x.iter().map(|xi| xi.exp()).collect();
        let f = |xv: &[f64]| xv.iter().map(|xi| xi.exp()).sum::<f64>();

        let result = check_gradient_per_element(f, &x, &analytical_grad, 1e-5, 1e-4);
        assert!(result.is_ok(), "Exp gradient check failed: {:?}", result);
    }

    // ------------------------------------------------------------------
    // Hessian diagonal tests
    // ------------------------------------------------------------------

    /// Test 12: Hessian diagonal of sum_squares should be all 2.
    #[test]
    fn test_numerical_hessian_diagonal_sum_squares() {
        let x = vec![1.0_f64, 2.0, -1.0, 0.5];
        let diag = numerical_hessian_diagonal(sum_squares, &x, 1e-4);

        assert_eq!(diag.len(), x.len());
        for (i, &h_ii) in diag.iter().enumerate() {
            assert!(
                (h_ii - 2.0).abs() < 1e-3,
                "H[{i},{i}] = {h_ii:.6e}, expected ~2.0"
            );
        }
    }

    /// Test 13: Hessian diagonal of f(x) = sum(x^4/4) should be sum(x^2).
    #[test]
    fn test_numerical_hessian_diagonal_quartic() {
        // f(x) = sum(x_i^4 / 4)
        // f''(x_i) = 3 * x_i^2
        let x = vec![1.0_f64, 2.0, -1.0];
        let f = |xv: &[f64]| xv.iter().map(|xi| xi.powi(4) / 4.0).sum::<f64>();

        let diag = numerical_hessian_diagonal(f, &x, 1e-4);
        for (i, (&xi, &h_ii)) in x.iter().zip(diag.iter()).enumerate() {
            let expected = 3.0 * xi * xi;
            assert!(
                (h_ii - expected).abs() < 1e-3,
                "H[{i},{i}] = {h_ii:.6e}, expected {expected:.6e}"
            );
        }
    }

    /// Test 14: Hessian diagonal of a composite function f(x) = sum(sin(x)).
    #[test]
    fn test_numerical_hessian_diagonal_sine() {
        // f(x) = sum(sin(x_i))
        // f''(x_i) = -sin(x_i)
        let x = vec![0.3_f64, 1.0, -0.5];
        let f = |xv: &[f64]| xv.iter().map(|xi| xi.sin()).sum::<f64>();

        let diag = numerical_hessian_diagonal(f, &x, 1e-4);
        for (i, (&xi, &h_ii)) in x.iter().zip(diag.iter()).enumerate() {
            let expected = -xi.sin();
            assert!(
                (h_ii - expected).abs() < 1e-3,
                "H[{i},{i}] = {h_ii:.6e}, expected {expected:.6e} (sin)"
            );
        }
    }

    // ------------------------------------------------------------------
    // HVP tests
    // ------------------------------------------------------------------

    /// Test 15: HVP on quadratic f(x) = 0.5*xTx.  H = I so H*v = v.
    #[test]
    fn test_numerical_hvp_identity_hessian() {
        // f(x) = 0.5 * (x0^2 + x1^2 + x2^2)  =>  H = I
        let x = vec![1.0_f64, -2.0, 0.5];
        let v = vec![1.0_f64, 2.0, -1.0];
        let f = |xv: &[f64]| 0.5 * xv.iter().map(|xi| xi * xi).sum::<f64>();

        let hvp = numerical_hvp(f, &x, &v, 1e-3);
        // H*v = I*v = v
        for (i, (&vi, &hv_i)) in v.iter().zip(hvp.iter()).enumerate() {
            assert!(
                (hv_i - vi).abs() < 1e-3,
                "HVP[{i}] = {hv_i:.6e}, expected {vi:.6e}"
            );
        }
    }

    /// Test 16: HVP on f(x) = 0.5*(a0*x0^2 + a1*x1^2).  H = diag(a), H*v = a*v.
    #[test]
    fn test_numerical_hvp_diagonal_hessian() {
        let a = [2.0_f64, 5.0];
        let x = vec![1.0_f64, 1.0];
        let v = vec![1.0_f64, 1.0];

        let f = |xv: &[f64]| 0.5 * (a[0] * xv[0] * xv[0] + a[1] * xv[1] * xv[1]);
        let hvp = numerical_hvp(f, &x, &v, 1e-3);

        assert!((hvp[0] - a[0] * v[0]).abs() < 1e-3, "hvp[0] wrong");
        assert!((hvp[1] - a[1] * v[1]).abs() < 1e-3, "hvp[1] wrong");
    }

    // ------------------------------------------------------------------
    // assert_gradient_correct tests
    // ------------------------------------------------------------------

    /// Test 17: assert_gradient_correct should not panic for a correct gradient.
    #[test]
    fn test_assert_gradient_correct_passes() {
        let x = vec![1.0_f64, 2.0, -1.0];
        let grad = sum_squares_grad(&x);
        // Must not panic.
        assert_gradient_correct(sum_squares, &x, &grad, 1e-5);
    }

    /// Test 18: assert_gradient_correct should panic for a wrong gradient.
    #[test]
    #[should_panic(expected = "assert_gradient_correct")]
    fn test_assert_gradient_correct_panics_for_wrong_grad() {
        let x = vec![1.0_f64, 2.0];
        let wrong_grad = vec![0.0_f64, 0.0];
        assert_gradient_correct(sum_squares, &x, &wrong_grad, 1e-5);
    }

    // ------------------------------------------------------------------
    // Tolerance / edge-case tests
    // ------------------------------------------------------------------

    /// Test 19: Single-element inputs work correctly.
    #[test]
    fn test_single_element_gradient() {
        let x = vec![3.0_f64];
        let grad = vec![6.0_f64]; // 2*3
        let result = check_gradient_per_element(sum_squares, &x, &grad, 1e-5, 1e-4);
        assert!(result.is_ok());
    }

    /// Test 20: Composite function: f(x) = sum(sin(x^2)), grad_i = 2*x_i*cos(x_i^2).
    #[test]
    fn test_check_gradient_composite_function() {
        let x = vec![0.5_f64, 1.0, -0.3];
        let analytical_grad: Vec<f64> = x.iter().map(|xi| 2.0 * xi * (xi * xi).cos()).collect();
        let f = |xv: &[f64]| xv.iter().map(|xi| (xi * xi).sin()).sum::<f64>();

        let result = check_gradient_per_element(f, &x, &analytical_grad, 1e-5, 1e-4);
        assert!(result.is_ok(), "Composite function gradient failed: {:?}", result);
    }

    /// Test 21: Verify GradientReport fields are populated correctly.
    #[test]
    fn test_gradient_report_fields() {
        let x = vec![1.0_f64, 2.0];
        let grad = sum_squares_grad(&x);
        let report = check_gradient_per_element(sum_squares, &x, &grad, 1e-5, 1e-4)
            .expect("Expected Ok for sum_squares gradient report");

        assert_eq!(report.num_elements, 2);
        assert_eq!(report.element_errors.len(), 2);
        assert!(report.max_error >= report.mean_error - 1e-15);
    }

    /// Test 22: Verify JacobianCheckReport display.
    #[test]
    fn test_jacobian_report_display() {
        let report = JacobianCheckReport {
            max_abs_error: 1e-6,
            max_rel_error: 1e-5,
            num_elements_checked: 4,
            num_failures: 0,
            passed: true,
        };
        let s = format!("{report}");
        assert!(s.contains("passed: true"), "Display should mention 'passed: true'");
    }
}
