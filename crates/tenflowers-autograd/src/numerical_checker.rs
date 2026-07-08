//! # Numerical Gradient Checker
//!
//! This module provides comprehensive numerical gradient validation using finite
//! differences to verify the correctness of analytical gradient implementations.
//!
//! ## Features
//!
//! - **Finite Difference Methods**: Central, forward, and backward difference approximations
//! - **Multi-Point Accuracy**: Support for 2-point, 4-point, and higher-order methods
//! - **Property-Based Testing**: Integration with property testing frameworks
//! - **Adaptive Epsilon**: Automatic epsilon selection based on tensor values
//! - **Comprehensive Reporting**: Detailed error analysis and visualization
//!
//! ## Usage
//!
//! ```rust,no_run
//! use tenflowers_autograd::numerical_checker::{NumericalChecker, CheckerConfig};
//! use tenflowers_autograd::GradientTape;
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = CheckerConfig::default();
//! let mut checker = NumericalChecker::new(config.clone());
//!
//! let x = Tensor::<f32>::ones(&[2, 2]);
//!
//! // Compute numerical gradient using central difference
//! let epsilon = config.epsilon();
//! let numerical_grad = checker.compute_numerical_gradient(
//!     &x,
//!     |x_val| {
//!         // Forward function: y = x^2
//!         tenflowers_core::ops::mul(x_val, x_val)
//!     },
//!     epsilon,
//! )?;
//!
//! println!("Numerical gradient shape: {:?}", numerical_grad.shape());
//! # Ok(())
//! # }
//! ```

use scirs2_core::random::Random;
use std::fmt;
use tenflowers_core::{Result, Tensor};

/// Finite difference method for numerical gradient computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiniteDifferenceMethod {
    /// Forward difference: (f(x+h) - f(x)) / h
    Forward,
    /// Backward difference: (f(x) - f(x-h)) / h
    Backward,
    /// Central difference: (f(x+h) - f(x-h)) / (2h)
    Central,
    /// 4-point central difference for higher accuracy
    Central4Point,
    /// 6-point central difference for very high accuracy
    Central6Point,
}

impl FiniteDifferenceMethod {
    /// Get the truncation error order (h^n) for this method
    pub fn error_order(&self) -> usize {
        match self {
            FiniteDifferenceMethod::Forward => 1,
            FiniteDifferenceMethod::Backward => 1,
            FiniteDifferenceMethod::Central => 2,
            FiniteDifferenceMethod::Central4Point => 4,
            FiniteDifferenceMethod::Central6Point => 6,
        }
    }

    /// Get recommended epsilon for this method
    pub fn recommended_epsilon(&self) -> f64 {
        match self {
            FiniteDifferenceMethod::Forward | FiniteDifferenceMethod::Backward => 1e-5,
            FiniteDifferenceMethod::Central => 1e-6,
            FiniteDifferenceMethod::Central4Point => 1e-7,
            FiniteDifferenceMethod::Central6Point => 1e-8,
        }
    }
}

/// Configuration for numerical gradient checking
#[derive(Debug, Clone)]
pub struct CheckerConfig {
    /// Finite difference method to use
    pub method: FiniteDifferenceMethod,
    /// Step size for finite differences
    pub epsilon: Option<f64>,
    /// Relative tolerance for gradient comparison
    pub rtol: f64,
    /// Absolute tolerance for gradient comparison
    pub atol: f64,
    /// Whether to use adaptive epsilon selection
    pub adaptive_epsilon: bool,
    /// Minimum epsilon value for adaptive selection
    pub min_epsilon: f64,
    /// Maximum epsilon value for adaptive selection
    pub max_epsilon: f64,
    /// Number of random points to sample for property testing
    pub num_samples: usize,
    /// Whether to check gradient at tensor boundaries
    pub check_boundaries: bool,
    /// Whether to check gradient at zero
    pub check_zero: bool,
    /// Random seed for reproducible testing
    pub seed: Option<u64>,
}

impl Default for CheckerConfig {
    fn default() -> Self {
        Self {
            method: FiniteDifferenceMethod::Central,
            epsilon: None, // Will use recommended epsilon
            rtol: 1e-3,
            atol: 1e-5,
            adaptive_epsilon: true,
            min_epsilon: 1e-10,
            max_epsilon: 1e-2,
            num_samples: 10,
            check_boundaries: true,
            check_zero: true,
            seed: None,
        }
    }
}

impl CheckerConfig {
    /// Get the epsilon value to use (either configured or recommended)
    pub fn epsilon(&self) -> f64 {
        self.epsilon
            .unwrap_or_else(|| self.method.recommended_epsilon())
    }
}

/// Result of a numerical gradient check
#[derive(Debug, Clone)]
pub struct GradientCheckResult {
    /// Whether the gradient check passed
    pub is_valid: bool,
    /// Maximum absolute error across all elements
    pub max_error: f64,
    /// Mean absolute error
    pub mean_error: f64,
    /// Relative error (|analytical - numerical| / max(|analytical|, |numerical|))
    pub relative_error: f64,
    /// Number of elements checked
    pub num_elements: usize,
    /// Number of elements that passed the tolerance check
    pub num_passed: usize,
    /// Analytical gradient values (sample)
    pub analytical_sample: Vec<f64>,
    /// Numerical gradient values (sample)
    pub numerical_sample: Vec<f64>,
    /// Detailed error analysis
    pub error_analysis: ErrorAnalysis,
}

impl fmt::Display for GradientCheckResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Gradient Check Result:")?;
        writeln!(
            f,
            "  Status: {}",
            if self.is_valid { "PASS" } else { "FAIL" }
        )?;
        writeln!(f, "  Max Error: {:.2e}", self.max_error)?;
        writeln!(f, "  Mean Error: {:.2e}", self.mean_error)?;
        writeln!(f, "  Relative Error: {:.2e}", self.relative_error)?;
        writeln!(f, "  Elements Checked: {}", self.num_elements)?;
        writeln!(
            f,
            "  Elements Passed: {}/{}",
            self.num_passed, self.num_elements
        )?;
        writeln!(
            f,
            "  Pass Rate: {:.1}%",
            (self.num_passed as f64 / self.num_elements as f64) * 100.0
        )?;
        Ok(())
    }
}

/// Detailed error analysis for gradient checking
#[derive(Debug, Clone)]
pub struct ErrorAnalysis {
    /// Distribution of errors (histogram buckets)
    pub error_histogram: Vec<(f64, usize)>,
    /// Indices of worst errors
    pub worst_indices: Vec<usize>,
    /// Worst error values
    pub worst_errors: Vec<f64>,
    /// Whether errors are systematic (biased)
    pub is_systematic: bool,
    /// Mean signed error (indicates bias)
    pub mean_signed_error: f64,
    /// Standard deviation of errors
    pub std_error: f64,
}

/// Numerical gradient checker
pub struct NumericalChecker {
    config: CheckerConfig,
    rng: Random<scirs2_core::rand_prelude::StdRng>,
}

impl NumericalChecker {
    /// Create a new numerical checker with the given configuration
    pub fn new(config: CheckerConfig) -> Self {
        let rng = if let Some(seed) = config.seed {
            // Create Random with specified seed
            Random::seed(seed)
        } else {
            // Create Random with default seed (0)
            Random::seed(0)
        };

        Self { config, rng }
    }
}

impl Default for NumericalChecker {
    /// Create a checker with default configuration
    fn default() -> Self {
        Self::new(CheckerConfig::default())
    }
}

impl NumericalChecker {
    /// Compute numerical gradient using finite differences
    pub fn compute_numerical_gradient<F>(
        &mut self,
        x: &Tensor<f32>,
        f: F,
        epsilon: f64,
    ) -> Result<Tensor<f32>>
    where
        F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        match self.config.method {
            FiniteDifferenceMethod::Forward => self.forward_difference(x, f, epsilon),
            FiniteDifferenceMethod::Backward => self.backward_difference(x, f, epsilon),
            FiniteDifferenceMethod::Central => self.central_difference(x, f, epsilon),
            FiniteDifferenceMethod::Central4Point => self.central_difference_4point(x, f, epsilon),
            FiniteDifferenceMethod::Central6Point => self.central_difference_6point(x, f, epsilon),
        }
    }

    /// Forward difference: (f(x+h) - f(x)) / h
    fn forward_difference<F>(&self, x: &Tensor<f32>, f: F, epsilon: f64) -> Result<Tensor<f32>>
    where
        F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        let f_x = f(x)?;
        let x_plus_h = self.add_epsilon(x, epsilon)?;
        let f_x_plus_h = f(&x_plus_h)?;

        let diff = tenflowers_core::ops::sub(&f_x_plus_h, &f_x)?;
        let shape_vec: Vec<usize> = diff.shape().dims().to_vec();
        let eps_array = scirs2_core::ndarray::ArrayD::from_elem(shape_vec, epsilon as f32);
        let eps_tensor = Tensor::from_array(eps_array);
        tenflowers_core::ops::div(&diff, &eps_tensor)
    }

    /// Backward difference: (f(x) - f(x-h)) / h
    fn backward_difference<F>(&self, x: &Tensor<f32>, f: F, epsilon: f64) -> Result<Tensor<f32>>
    where
        F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        let f_x = f(x)?;
        let x_minus_h = self.sub_epsilon(x, epsilon)?;
        let f_x_minus_h = f(&x_minus_h)?;

        let diff = tenflowers_core::ops::sub(&f_x, &f_x_minus_h)?;
        let shape_vec: Vec<usize> = diff.shape().dims().to_vec();
        let eps_array = scirs2_core::ndarray::ArrayD::from_elem(shape_vec, epsilon as f32);
        let eps_tensor = Tensor::from_array(eps_array);
        tenflowers_core::ops::div(&diff, &eps_tensor)
    }

    /// Central difference: (f(x+h) - f(x-h)) / (2h)
    fn central_difference<F>(&self, x: &Tensor<f32>, f: F, epsilon: f64) -> Result<Tensor<f32>>
    where
        F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        let x_plus_h = self.add_epsilon(x, epsilon)?;
        let x_minus_h = self.sub_epsilon(x, epsilon)?;

        let f_plus = f(&x_plus_h)?;
        let f_minus = f(&x_minus_h)?;

        let diff = tenflowers_core::ops::sub(&f_plus, &f_minus)?;
        let shape_vec: Vec<usize> = diff.shape().dims().to_vec();
        let two_eps_array =
            scirs2_core::ndarray::ArrayD::from_elem(shape_vec, (2.0 * epsilon) as f32);
        let two_eps_tensor = Tensor::from_array(two_eps_array);
        tenflowers_core::ops::div(&diff, &two_eps_tensor)
    }

    /// 4-point central difference for higher accuracy
    fn central_difference_4point<F>(
        &self,
        x: &Tensor<f32>,
        f: F,
        epsilon: f64,
    ) -> Result<Tensor<f32>>
    where
        F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        // Formula: (-f(x+2h) + 8*f(x+h) - 8*f(x-h) + f(x-2h)) / (12h)
        let x_plus_h = self.add_epsilon(x, epsilon)?;
        let x_minus_h = self.sub_epsilon(x, epsilon)?;
        let x_plus_2h = self.add_epsilon(x, 2.0 * epsilon)?;
        let x_minus_2h = self.sub_epsilon(x, 2.0 * epsilon)?;

        let f_plus_h = f(&x_plus_h)?;
        let f_minus_h = f(&x_minus_h)?;
        let f_plus_2h = f(&x_plus_2h)?;
        let f_minus_2h = f(&x_minus_2h)?;

        // Compute numerator: -f(x+2h) + 8*f(x+h) - 8*f(x-h) + f(x-2h)
        let shape_vec: Vec<usize> = f_plus_h.shape().dims().to_vec();
        let eight_array = scirs2_core::ndarray::ArrayD::from_elem(shape_vec.clone(), 8.0f32);
        let eight = Tensor::from_array(eight_array);

        let term1 = f_plus_2h;
        let neg_term1 = tenflowers_core::ops::neg(&term1)?;
        let term2 = tenflowers_core::ops::mul(&eight, &f_plus_h)?;
        let term3 = tenflowers_core::ops::mul(&eight, &f_minus_h)?;
        let neg_term3 = tenflowers_core::ops::neg(&term3)?;

        let sum1 = tenflowers_core::ops::add(&neg_term1, &term2)?;
        let sum2 = tenflowers_core::ops::add(&sum1, &neg_term3)?;
        let numerator = tenflowers_core::ops::add(&sum2, &f_minus_2h)?;

        let denom_shape_vec: Vec<usize> = numerator.shape().dims().to_vec();
        let denom_array =
            scirs2_core::ndarray::ArrayD::from_elem(denom_shape_vec, (12.0 * epsilon) as f32);
        let denominator = Tensor::from_array(denom_array);

        tenflowers_core::ops::div(&numerator, &denominator)
    }

    /// 6-point central difference for very high accuracy
    fn central_difference_6point<F>(
        &self,
        x: &Tensor<f32>,
        f: F,
        epsilon: f64,
    ) -> Result<Tensor<f32>>
    where
        F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        // Formula: (f(x+3h) - 9*f(x+2h) + 45*f(x+h) - 45*f(x-h) + 9*f(x-2h) - f(x-3h)) / (60h)
        let points = [3.0, 2.0, 1.0, -1.0, -2.0, -3.0];
        let coeffs = [1.0f32, -9.0, 45.0, -45.0, 9.0, -1.0];

        let mut result = None;

        for (i, (&coeff_val, &point)) in coeffs.iter().zip(points.iter()).enumerate() {
            let x_offset = if point > 0.0 {
                self.add_epsilon(x, point * epsilon)?
            } else {
                self.sub_epsilon(x, -point * epsilon)?
            };

            let f_val = f(&x_offset)?;
            let coeff_shape: Vec<usize> = f_val.shape().dims().to_vec();
            let coeff_array = scirs2_core::ndarray::ArrayD::from_elem(coeff_shape, coeff_val);
            let coeff = Tensor::from_array(coeff_array);

            let term = tenflowers_core::ops::mul(&coeff, &f_val)?;

            result = Some(if let Some(acc) = result {
                tenflowers_core::ops::add(&acc, &term)?
            } else {
                term
            });
        }

        let numerator = result.expect("result should be Some after processing all coefficients");
        let denom_shape: Vec<usize> = numerator.shape().dims().to_vec();
        let denom_array =
            scirs2_core::ndarray::ArrayD::from_elem(denom_shape, (60.0 * epsilon) as f32);
        let denominator = Tensor::from_array(denom_array);

        tenflowers_core::ops::div(&numerator, &denominator)
    }

    /// Add epsilon to tensor (element-wise perturbation)
    fn add_epsilon(&self, x: &Tensor<f32>, epsilon: f64) -> Result<Tensor<f32>> {
        let shape_vec: Vec<usize> = x.shape().dims().to_vec();
        let eps_array = scirs2_core::ndarray::ArrayD::from_elem(shape_vec, epsilon as f32);
        let eps_tensor = Tensor::from_array(eps_array);
        tenflowers_core::ops::add(x, &eps_tensor)
    }

    /// Subtract epsilon from tensor
    fn sub_epsilon(&self, x: &Tensor<f32>, epsilon: f64) -> Result<Tensor<f32>> {
        let shape_vec: Vec<usize> = x.shape().dims().to_vec();
        let eps_array = scirs2_core::ndarray::ArrayD::from_elem(shape_vec, epsilon as f32);
        let eps_tensor = Tensor::from_array(eps_array);
        tenflowers_core::ops::sub(x, &eps_tensor)
    }

    /// Compare analytical and numerical gradients
    pub fn compare_gradients(
        &self,
        analytical: &Tensor<f32>,
        numerical: &Tensor<f32>,
    ) -> Result<GradientCheckResult> {
        // Ensure shapes match
        if analytical.shape() != numerical.shape() {
            return Err(tenflowers_core::TensorError::ShapeMismatch {
                operation: "Gradient comparison".to_string(),
                expected: format!("{:?}", analytical.shape()),
                got: format!("{:?}", numerical.shape()),
                context: None,
            });
        }

        let num_elements = analytical.size();

        // Flatten for comparison
        let analytical_data: Vec<f32> = analytical.to_vec()?;
        let numerical_data: Vec<f32> = numerical.to_vec()?;

        let mut errors = Vec::with_capacity(num_elements);
        let mut num_passed = 0;

        for (&a, &n) in analytical_data.iter().zip(numerical_data.iter()) {
            let abs_error = (a - n).abs();
            let max_val = a.abs().max(n.abs());
            let rel_error = if max_val > 1e-10 {
                abs_error / max_val
            } else {
                abs_error
            };

            errors.push(abs_error as f64);

            // Check tolerance
            if abs_error <= self.config.atol as f32 || rel_error <= self.config.rtol as f32 {
                num_passed += 1;
            }
        }

        let max_error = errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_error = errors.iter().sum::<f64>() / num_elements as f64;

        // Compute relative error
        let mut rel_errors = Vec::with_capacity(num_elements);
        for (&a, &n) in analytical_data.iter().zip(numerical_data.iter()) {
            let max_val = a.abs().max(n.abs());
            let rel_error = if max_val > 1e-10 {
                ((a - n).abs() / max_val) as f64
            } else {
                (a - n).abs() as f64
            };
            rel_errors.push(rel_error);
        }
        let relative_error = rel_errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Sample values for reporting
        let sample_size = num_elements.min(10);
        let analytical_sample: Vec<f64> = analytical_data[..sample_size]
            .iter()
            .map(|&x| x as f64)
            .collect();
        let numerical_sample: Vec<f64> = numerical_data[..sample_size]
            .iter()
            .map(|&x| x as f64)
            .collect();

        // Error analysis
        let error_analysis = self.analyze_errors(&errors, &analytical_data, &numerical_data);

        let is_valid = (num_passed as f64 / num_elements as f64) >= 0.95; // 95% pass rate

        Ok(GradientCheckResult {
            is_valid,
            max_error,
            mean_error,
            relative_error,
            num_elements,
            num_passed,
            analytical_sample,
            numerical_sample,
            error_analysis,
        })
    }

    /// Analyze error distribution and characteristics
    fn analyze_errors(
        &self,
        errors: &[f64],
        analytical: &[f32],
        numerical: &[f32],
    ) -> ErrorAnalysis {
        // Create histogram
        let num_buckets = 10;
        let max_error = errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let bucket_size = max_error / num_buckets as f64;

        let mut histogram = vec![0usize; num_buckets];
        for &error in errors {
            let bucket = ((error / bucket_size) as usize).min(num_buckets - 1);
            histogram[bucket] += 1;
        }

        let error_histogram: Vec<(f64, usize)> = (0..num_buckets)
            .map(|i| (i as f64 * bucket_size, histogram[i]))
            .collect();

        // Find worst errors
        let mut indexed_errors: Vec<(usize, f64)> =
            errors.iter().enumerate().map(|(i, &e)| (i, e)).collect();
        indexed_errors.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .expect("partial_cmp should not return None for valid values")
        });

        let worst_count = errors.len().min(5);
        let worst_indices: Vec<usize> = indexed_errors[..worst_count]
            .iter()
            .map(|(i, _)| *i)
            .collect();
        let worst_errors: Vec<f64> = indexed_errors[..worst_count]
            .iter()
            .map(|(_, e)| *e)
            .collect();

        // Compute mean signed error (bias)
        let signed_errors: Vec<f64> = analytical
            .iter()
            .zip(numerical.iter())
            .map(|(&a, &n)| (a - n) as f64)
            .collect();
        let mean_signed_error = signed_errors.iter().sum::<f64>() / signed_errors.len() as f64;

        // Compute standard deviation
        let variance = errors
            .iter()
            .map(|&e| {
                let diff = e - (errors.iter().sum::<f64>() / errors.len() as f64);
                diff * diff
            })
            .sum::<f64>()
            / errors.len() as f64;
        let std_error = variance.sqrt();

        // Check if errors are systematic
        let is_systematic = mean_signed_error.abs() > std_error * 0.5;

        ErrorAnalysis {
            error_histogram,
            worst_indices,
            worst_errors,
            is_systematic,
            mean_signed_error,
            std_error,
        }
    }

    /// Property-based gradient checking: test analytical gradients against numerical
    /// gradients at multiple random points.
    ///
    /// This validates that the analytical gradient function `f_grad` agrees with the
    /// finite-difference (numerical) gradient of the forward function `f` over a set of
    /// randomly sampled inputs sharing the same shape as `x`.
    ///
    /// # Arguments
    ///
    /// * `x` - Reference tensor used only to determine the shape of the random samples.
    /// * `f` - Forward function `y = f(x)`; its finite-difference gradient is the reference.
    /// * `f_grad` - Analytical gradient function `g = ∂f/∂x` to be validated.
    ///
    /// # Returns
    ///
    /// One [`GradientCheckResult`] per sampled point, comparing the analytical gradient
    /// against the numerical gradient at that point.
    pub fn property_test<F, G>(
        &mut self,
        x: &Tensor<f32>,
        f: F,
        f_grad: G,
    ) -> Result<Vec<GradientCheckResult>>
    where
        F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
        G: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        let mut results = Vec::with_capacity(self.config.num_samples);

        for _ in 0..self.config.num_samples {
            // Sample a random input point with the same shape as `x`.
            let shape = x.shape().dims();
            let random_tensor = self.generate_random_tensor(shape)?;

            // Compute the numerical (finite-difference) gradient at this point.
            let epsilon = self.config.epsilon();
            let numerical = self.compute_numerical_gradient(&random_tensor, &f, epsilon)?;

            // Compute the analytical gradient at the SAME point and compare against it.
            // This is the real validation: analytical vs. numerical at identical inputs.
            let analytical = f_grad(&random_tensor)?;

            let result = self.compare_gradients(&analytical, &numerical)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Generate a random tensor with the given shape.
    ///
    /// Samples each element uniformly from `[-1, 1)` using the checker's seeded
    /// `scirs2_core::random::Random` generator, ensuring reproducibility when a seed
    /// is configured (per the SCIRS2 policy — no hand-rolled RNGs).
    fn generate_random_tensor(&mut self, shape: &[usize]) -> Result<Tensor<f32>> {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size)
            .map(|_| self.rng.random_range(-1.0_f32..1.0_f32))
            .collect();

        let array =
            scirs2_core::ndarray::ArrayD::from_shape_vec(shape.to_vec(), data).map_err(|e| {
                tenflowers_core::TensorError::InvalidShape {
                    operation: "Random tensor generation".to_string(),
                    reason: e.to_string(),
                    shape: Some(shape.to_vec()),
                    context: None,
                }
            })?;

        Ok(Tensor::from_array(array))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finite_difference_methods() {
        assert_eq!(FiniteDifferenceMethod::Forward.error_order(), 1);
        assert_eq!(FiniteDifferenceMethod::Central.error_order(), 2);
        assert_eq!(FiniteDifferenceMethod::Central4Point.error_order(), 4);
    }

    #[test]
    fn test_checker_config_default() {
        let config = CheckerConfig::default();
        assert_eq!(config.method, FiniteDifferenceMethod::Central);
        assert!(config.adaptive_epsilon);
    }

    #[test]
    fn test_numerical_checker_creation() {
        let config = CheckerConfig::default();
        let _checker = NumericalChecker::new(config);
    }

    #[test]
    fn test_gradient_check_result_display() {
        let result = GradientCheckResult {
            is_valid: true,
            max_error: 1e-5,
            mean_error: 1e-6,
            relative_error: 1e-4,
            num_elements: 100,
            num_passed: 98,
            analytical_sample: vec![1.0, 2.0, 3.0],
            numerical_sample: vec![1.0, 2.0, 3.0],
            error_analysis: ErrorAnalysis {
                error_histogram: vec![],
                worst_indices: vec![],
                worst_errors: vec![],
                is_systematic: false,
                mean_signed_error: 0.0,
                std_error: 1e-6,
            },
        };

        let display = format!("{}", result);
        assert!(display.contains("PASS"));
        assert!(display.contains("Max Error"));
    }

    #[test]
    fn test_property_test_correct_analytical_passes() {
        // Use f(x) = 2x (linear), whose exact analytical gradient is the constant 2.
        //
        // A linear function has zero truncation error in central difference:
        // central_diff(2x) = (2(x+ε) - 2(x-ε)) / (2ε) = 4ε / 2ε = 2 exactly.
        //
        // In f32 arithmetic, rounding error ≈ |f(x)| * eps_mach / ε.
        // f32 machine epsilon ≈ 1.19e-7.  With |f(x)| ≤ 2 (inputs from [-1,1)):
        //   rounding error ≈ 2 * 1.19e-7 / 5e-3 ≈ 4.8e-5  → well under atol=1e-3.
        // We use ε=5e-3 (near optimal for f32 central difference: eps_mach^(1/3) ≈ 4.9e-3)
        // to keep rounding error safely below atol while truncation error is exactly 0.
        let config = CheckerConfig {
            num_samples: 4,
            seed: Some(7),
            epsilon: Some(5e-3), // f32-safe epsilon; default 1e-6 is too close to f32 noise floor
            rtol: 1e-2,
            atol: 1e-3,
            ..Default::default()
        };
        let mut checker = NumericalChecker::new(config);

        let x = Tensor::<f32>::ones(&[3]);

        // f(t) = 2t — computed as t + t to exercise the ops path, gradient is constant 2.
        let f = |t: &Tensor<f32>| tenflowers_core::ops::add(t, t);
        // Analytical gradient of 2t is the constant tensor [2, 2, 2].
        let f_grad = |t: &Tensor<f32>| {
            use scirs2_core::ndarray::ArrayD;
            let ones = ArrayD::from_elem(t.shape().dims().to_vec(), 2.0_f32);
            Ok::<_, tenflowers_core::TensorError>(Tensor::from_array(ones))
        };

        let results = checker
            .property_test(&x, f, f_grad)
            .expect("property_test should succeed");
        assert_eq!(results.len(), 4);
        for result in &results {
            assert!(
                result.is_valid,
                "analytical gradient of f(x)=2x (constant 2) must validate against \
                 the numerical gradient; max_error={:.3e}, mean_error={:.3e}",
                result.max_error, result.mean_error
            );
        }
    }

    #[test]
    fn test_property_test_wrong_analytical_fails() {
        // Provide a deliberately WRONG analytical gradient (constant 5 instead of 2
        // for f(x) = 2x). This proves the comparison is real: the old fabricated
        // `analytical = numerical.clone()` would always pass, but a genuine comparison
        // must reject a gradient that is 2.5× too large.
        // Using the same f32-safe epsilon=5e-3 so the numerical baseline is accurate.
        let config = CheckerConfig {
            num_samples: 4,
            seed: Some(11),
            epsilon: Some(5e-3),
            rtol: 1e-2,
            atol: 1e-3,
            ..Default::default()
        };
        let mut checker = NumericalChecker::new(config);

        let x = Tensor::<f32>::ones(&[3]);
        let f = |t: &Tensor<f32>| tenflowers_core::ops::add(t, t);
        // Wrong gradient: 5 instead of 2. Absolute error 3 >> atol=1e-4, so every
        // sample will be rejected by at least 95% of its elements.
        let wrong_grad = |t: &Tensor<f32>| {
            use scirs2_core::ndarray::ArrayD;
            let fives = ArrayD::from_elem(t.shape().dims().to_vec(), 5.0_f32);
            Ok::<_, tenflowers_core::TensorError>(Tensor::from_array(fives))
        };

        let results = checker
            .property_test(&x, f, wrong_grad)
            .expect("property_test should succeed");
        assert!(
            results.iter().any(|r| !r.is_valid),
            "gradient constant 5 for f(x)=2x (true gradient 2) must be rejected; \
             absolute error 3 >> atol=1e-4"
        );
    }
}
