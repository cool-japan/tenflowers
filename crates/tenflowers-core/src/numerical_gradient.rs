//! Numerical Gradient Validation Utilities
//!
//! This module provides tools for validating automatic differentiation implementations
//! by comparing analytical gradients against numerical gradients computed via finite
//! differences. This is essential for ensuring correctness of custom operations and
//! gradient implementations.
//!
//! # Overview
//!
//! Numerical gradient checking uses the definition of a derivative:
//!
//! ```text
//! f'(x) ≈ [f(x + ε) - f(x - ε)] / (2ε)
//! ```
//!
//! By comparing this numerical approximation with the analytical gradient computed
//! via automatic differentiation, we can detect bugs in gradient implementations.
//!
//! # Example
//!
//! ```rust
//! use tenflowers_core::numerical_gradient::{check_gradients, GradientCheckConfig};
//! use tenflowers_core::Tensor;
//!
//! // Define a function to test
//! fn square(x: &Tensor<f32>) -> Tensor<f32> {
//!     x * x
//! }
//!
//! // Analytical gradient: 2x
//! fn square_grad(x: &Tensor<f32>) -> Tensor<f32> {
//!     x * 2.0
//! }
//!
//! let config = GradientCheckConfig::default();
//! let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
//!
//! // Verify gradient correctness
//! let result = check_gradients(&x, square, square_grad, &config);
//! assert!(result.is_ok());
//! ```

use crate::{Result, Tensor, TensorError};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::marker::PhantomData;

/// Configuration for numerical gradient checking
#[derive(Debug, Clone)]
pub struct GradientCheckConfig {
    /// Epsilon for finite difference computation
    pub epsilon: f64,
    /// Relative tolerance for gradient comparison
    pub rtol: f64,
    /// Absolute tolerance for gradient comparison
    pub atol: f64,
    /// Check gradients element-wise (slower but more detailed)
    pub check_elementwise: bool,
    /// Use central differences (more accurate but 2x slower)
    pub use_central_difference: bool,
    /// Maximum number of samples to check (for large tensors)
    pub max_samples: Option<usize>,
    /// Random seed for sampling (if max_samples is set)
    pub random_seed: Option<u64>,
}

impl Default for GradientCheckConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-5,
            rtol: 1e-3,
            atol: 1e-5,
            check_elementwise: false,
            use_central_difference: true,
            max_samples: None,
            random_seed: None,
        }
    }
}

impl GradientCheckConfig {
    /// Create a configuration for strict checking (tighter tolerances)
    pub fn strict() -> Self {
        Self {
            epsilon: 1e-6,
            rtol: 1e-4,
            atol: 1e-6,
            check_elementwise: true,
            use_central_difference: true,
            max_samples: None,
            random_seed: None,
        }
    }

    /// Create a configuration for relaxed checking (looser tolerances)
    pub fn relaxed() -> Self {
        Self {
            epsilon: 1e-4,
            rtol: 1e-2,
            atol: 1e-4,
            check_elementwise: false,
            use_central_difference: true,
            max_samples: Some(100),
            random_seed: Some(42),
        }
    }

    /// Create a configuration for fast checking (forward differences, sampling)
    pub fn fast() -> Self {
        Self {
            epsilon: 1e-5,
            rtol: 1e-3,
            atol: 1e-5,
            check_elementwise: false,
            use_central_difference: false,
            max_samples: Some(50),
            random_seed: Some(42),
        }
    }
}

/// Result of gradient checking
#[derive(Debug, Clone)]
pub struct GradientCheckResult {
    /// Whether the gradient check passed
    pub passed: bool,
    /// Maximum relative error found
    pub max_relative_error: f64,
    /// Maximum absolute error found
    pub max_absolute_error: f64,
    /// Number of elements checked
    pub num_elements_checked: usize,
    /// Number of elements that failed tolerance check
    pub num_failures: usize,
    /// Indices of failed elements (if check_elementwise is true)
    pub failed_indices: Vec<usize>,
    /// Detailed error message if check failed
    pub error_message: Option<String>,
}

impl GradientCheckResult {
    /// Check if the gradient validation passed
    pub fn is_ok(&self) -> bool {
        self.passed
    }

    /// Get failure rate (percentage of elements that failed)
    pub fn failure_rate(&self) -> f64 {
        if self.num_elements_checked == 0 {
            0.0
        } else {
            (self.num_failures as f64 / self.num_elements_checked as f64) * 100.0
        }
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        if self.passed {
            format!(
                "✓ Gradient check passed\n\
                 Elements checked: {}\n\
                 Max relative error: {:.2e}\n\
                 Max absolute error: {:.2e}",
                self.num_elements_checked, self.max_relative_error, self.max_absolute_error
            )
        } else {
            format!(
                "✗ Gradient check FAILED\n\
                 Elements checked: {}\n\
                 Failures: {} ({:.2}%)\n\
                 Max relative error: {:.2e}\n\
                 Max absolute error: {:.2e}\n\
                 {}",
                self.num_elements_checked,
                self.num_failures,
                self.failure_rate(),
                self.max_relative_error,
                self.max_absolute_error,
                self.error_message.as_deref().unwrap_or("")
            )
        }
    }
}

/// Numerical gradient checker
pub struct NumericalGradientChecker<T> {
    config: GradientCheckConfig,
    _phantom: PhantomData<T>,
}

impl<T> NumericalGradientChecker<T>
where
    T: Float + FromPrimitive + Clone + Send + Sync + Default + 'static,
{
    /// Create a new gradient checker with the given configuration
    pub fn new(config: GradientCheckConfig) -> Self {
        Self {
            config,
            _phantom: PhantomData,
        }
    }

    /// Compute numerical gradient using finite differences
    pub fn compute_numerical_gradient<F>(&self, input: &Tensor<T>, func: F) -> Result<Tensor<T>>
    where
        F: Fn(&Tensor<T>) -> Result<Tensor<T>>,
    {
        let input_data = input.data();
        let input_shape = input.shape();
        let mut gradient_data = Vec::with_capacity(input_data.len());

        let epsilon = T::from_f64(self.config.epsilon).ok_or_else(|| {
            TensorError::invalid_operation_simple("Failed to convert epsilon".to_string())
        })?;

        for i in 0..input_data.len() {
            let grad = if self.config.use_central_difference {
                // Central difference: [f(x + ε) - f(x - ε)] / (2ε)
                let mut input_plus = input_data.to_vec();
                let mut input_minus = input_data.to_vec();

                input_plus[i] = input_plus[i] + epsilon;
                input_minus[i] = input_minus[i] - epsilon;

                let x_plus = Tensor::from_array(
                    scirs2_core::ndarray::Array::from_shape_vec(
                        input_shape.dims().to_vec(),
                        input_plus.to_vec(),
                    )
                    .map_err(|e| TensorError::invalid_argument(format!("Shape mismatch: {}", e)))?
                    .into_dyn(),
                );

                let x_minus = Tensor::from_array(
                    scirs2_core::ndarray::Array::from_shape_vec(
                        input_shape.dims().to_vec(),
                        input_minus.to_vec(),
                    )
                    .map_err(|e| TensorError::invalid_argument(format!("Shape mismatch: {}", e)))?
                    .into_dyn(),
                );

                let f_plus = func(&x_plus)?;
                let f_minus = func(&x_minus)?;

                // For element-wise functions, derivative at position i
                let two_epsilon = epsilon + epsilon;
                let diff = f_plus.data()[i] - f_minus.data()[i];
                diff / two_epsilon
            } else {
                // Forward difference: [f(x + ε) - f(x)] / ε
                let mut input_plus = input_data.to_vec();
                input_plus[i] = input_plus[i] + epsilon;

                let x_plus = Tensor::from_array(
                    scirs2_core::ndarray::Array::from_shape_vec(
                        input_shape.dims().to_vec(),
                        input_plus.to_vec(),
                    )
                    .map_err(|e| TensorError::invalid_argument(format!("Shape mismatch: {}", e)))?
                    .into_dyn(),
                );

                let f_plus = func(&x_plus)?;
                let f_x = func(input)?;

                // For element-wise functions, derivative at position i
                let diff = f_plus.data()[i] - f_x.data()[i];
                diff / epsilon
            };

            gradient_data.push(grad);
        }

        let gradient_array =
            scirs2_core::ndarray::Array::from_shape_vec(input_shape.dims().to_vec(), gradient_data)
                .map_err(|e| TensorError::invalid_argument(format!("Shape mismatch: {}", e)))?
                .into_dyn();

        Ok(Tensor::from_array(gradient_array))
    }

    /// Compare analytical and numerical gradients
    pub fn compare_gradients(
        &self,
        numerical: &Tensor<T>,
        analytical: &Tensor<T>,
    ) -> Result<GradientCheckResult> {
        if numerical.shape() != analytical.shape() {
            return Err(TensorError::invalid_argument(format!(
                "Shape mismatch: numerical {:?} vs analytical {:?}",
                numerical.shape(),
                analytical.shape()
            )));
        }

        let num_data = numerical.data();
        let ana_data = analytical.data();

        let rtol = self.config.rtol;
        let atol = self.config.atol;

        let mut max_rel_error = 0.0;
        let mut max_abs_error = 0.0;
        let mut num_failures = 0;
        let mut failed_indices = Vec::new();

        for i in 0..num_data.len() {
            let num_val = num_data[i].to_f64().unwrap_or(0.0);
            let ana_val = ana_data[i].to_f64().unwrap_or(0.0);

            let abs_error = (num_val - ana_val).abs();
            let rel_error = if ana_val.abs() > 1e-10 {
                abs_error / ana_val.abs()
            } else {
                abs_error
            };

            max_rel_error = max_rel_error.max(rel_error);
            max_abs_error = max_abs_error.max(abs_error);

            if rel_error > rtol && abs_error > atol {
                num_failures += 1;
                if self.config.check_elementwise {
                    failed_indices.push(i);
                }
            }
        }

        let passed = num_failures == 0;
        let error_message = if !passed {
            Some(format!(
                "Gradient mismatch: {} of {} elements exceed tolerance (rtol={}, atol={})",
                num_failures,
                num_data.len(),
                rtol,
                atol
            ))
        } else {
            None
        };

        Ok(GradientCheckResult {
            passed,
            max_relative_error: max_rel_error,
            max_absolute_error: max_abs_error,
            num_elements_checked: num_data.len(),
            num_failures,
            failed_indices,
            error_message,
        })
    }

    /// Full gradient check: compute numerical gradient and compare
    pub fn check<F, G>(
        &self,
        input: &Tensor<T>,
        forward: F,
        gradient: G,
    ) -> Result<GradientCheckResult>
    where
        F: Fn(&Tensor<T>) -> Result<Tensor<T>>,
        G: Fn(&Tensor<T>) -> Result<Tensor<T>>,
    {
        let numerical_grad = self.compute_numerical_gradient(input, forward)?;
        let analytical_grad = gradient(input)?;

        self.compare_gradients(&numerical_grad, &analytical_grad)
    }
}

/// Convenience function for quick gradient checking
pub fn check_gradients<T, F, G>(
    input: &Tensor<T>,
    forward: F,
    gradient: G,
    config: &GradientCheckConfig,
) -> Result<GradientCheckResult>
where
    T: Float + FromPrimitive + Clone + Send + Sync + Default + 'static,
    F: Fn(&Tensor<T>) -> Result<Tensor<T>>,
    G: Fn(&Tensor<T>) -> Result<Tensor<T>>,
{
    let checker = NumericalGradientChecker::new(config.clone());
    checker.check(input, forward, gradient)
}

/// Convenience function with default configuration
pub fn quick_check_gradients<T, F, G>(
    input: &Tensor<T>,
    forward: F,
    gradient: G,
) -> Result<GradientCheckResult>
where
    T: Float + FromPrimitive + Clone + Send + Sync + Default + 'static,
    F: Fn(&Tensor<T>) -> Result<Tensor<T>>,
    G: Fn(&Tensor<T>) -> Result<Tensor<T>>,
{
    check_gradients(input, forward, gradient, &GradientCheckConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_gradient_check_linear() {
        // Test f(x) = 2x, f'(x) = 2
        let input = Tensor::from_array(array![1.0, 2.0, 3.0].into_dyn());

        let forward = |x: &Tensor<f32>| {
            let data: Vec<f32> = x.data().iter().map(|&v| v * 2.0).collect();
            let result_array = scirs2_core::ndarray::Array::from_vec(data).into_dyn();
            Ok(Tensor::from_array(result_array))
        };

        let gradient = |_x: &Tensor<f32>| {
            let grad = array![2.0, 2.0, 2.0].into_dyn();
            Ok(Tensor::from_array(grad))
        };

        // Use relaxed config due to f32 precision limits
        let config = GradientCheckConfig::relaxed();
        let result = check_gradients(&input, forward, gradient, &config).unwrap();

        assert!(
            result.passed,
            "Gradient check should pass for linear function: {}",
            result.summary()
        );
    }

    #[test]
    fn test_gradient_check_square() {
        // Test f(x) = x^2, f'(x) = 2x
        let input = Tensor::from_array(array![1.0, 2.0, 3.0].into_dyn());

        let forward = |x: &Tensor<f32>| {
            let data: Vec<f32> = x.data().iter().map(|&v| v * v).collect();
            let result_array = scirs2_core::ndarray::Array::from_vec(data).into_dyn();
            Ok(Tensor::from_array(result_array))
        };

        let gradient = |x: &Tensor<f32>| {
            let data: Vec<f32> = x.data().iter().map(|&v| 2.0 * v).collect();
            let grad_array = scirs2_core::ndarray::Array::from_vec(data).into_dyn();
            Ok(Tensor::from_array(grad_array))
        };

        // Use relaxed config due to f32 precision limits
        let config = GradientCheckConfig::relaxed();
        let result = check_gradients(&input, forward, gradient, &config).unwrap();

        assert!(
            result.passed,
            "Gradient check should pass for square function: {}",
            result.summary()
        );
    }

    #[test]
    fn test_gradient_check_incorrect_gradient() {
        // Test with intentionally incorrect gradient
        let input = Tensor::from_array(array![1.0, 2.0, 3.0].into_dyn());

        let forward = |x: &Tensor<f32>| {
            let data: Vec<f32> = x.data().iter().map(|&v| v * v).collect();
            let result_array = scirs2_core::ndarray::Array::from_vec(data).into_dyn();
            Ok(Tensor::from_array(result_array))
        };

        let wrong_gradient = |x: &Tensor<f32>| {
            // Incorrect: should be 2x, but we return 3x
            let data: Vec<f32> = x.data().iter().map(|&v| 3.0 * v).collect();
            let grad_array = scirs2_core::ndarray::Array::from_vec(data).into_dyn();
            Ok(Tensor::from_array(grad_array))
        };

        let config = GradientCheckConfig::default();
        let result = check_gradients(&input, forward, wrong_gradient, &config).unwrap();

        assert!(
            !result.passed,
            "Gradient check should fail for incorrect gradient"
        );
        assert!(result.num_failures > 0);
    }

    #[test]
    fn test_gradient_check_config_tolerances() {
        let input = Tensor::from_array(array![1.0].into_dyn());

        let forward = |x: &Tensor<f32>| {
            let data: Vec<f32> = x.data().iter().map(|&v| v * v).collect();
            let result_array = scirs2_core::ndarray::Array::from_vec(data).into_dyn();
            Ok(Tensor::from_array(result_array))
        };

        let slightly_off_gradient = |x: &Tensor<f32>| {
            // Slightly incorrect: 2x * 1.01
            let data: Vec<f32> = x.data().iter().map(|&v| 2.0 * v * 1.01).collect();
            let grad_array = scirs2_core::ndarray::Array::from_vec(data).into_dyn();
            Ok(Tensor::from_array(grad_array))
        };

        // Should pass with relaxed config
        let relaxed = GradientCheckConfig::relaxed();
        let result = check_gradients(&input, forward, slightly_off_gradient, &relaxed).unwrap();
        assert!(result.passed, "Should pass with relaxed tolerances");

        // Should fail with strict config
        let strict = GradientCheckConfig::strict();
        let result = check_gradients(&input, forward, slightly_off_gradient, &strict).unwrap();
        assert!(!result.passed, "Should fail with strict tolerances");
    }

    #[test]
    fn test_gradient_check_result_summary() {
        let result = GradientCheckResult {
            passed: false,
            max_relative_error: 0.05,
            max_absolute_error: 0.01,
            num_elements_checked: 100,
            num_failures: 10,
            failed_indices: vec![],
            error_message: Some("Test error".to_string()),
        };

        let summary = result.summary();
        assert!(summary.contains("FAILED"));
        assert!(summary.contains("10.00%"));

        assert_eq!(result.failure_rate(), 10.0);
    }
}
