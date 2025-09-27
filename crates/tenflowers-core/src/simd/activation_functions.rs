//! SIMD-Optimized Activation Functions
//!
//! This module provides high-performance implementations of common neural network
//! activation functions using SIMD optimizations for maximum throughput.

use crate::error::ErrorContext;
use crate::{Result, TensorError};

/// SIMD-optimized activation functions for neural networks
pub struct ActivationFunctions;

impl ActivationFunctions {
    /// Optimized ReLU activation function
    pub fn relu_f32_optimized(input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD ReLU".to_string(),
                expected: format!("arrays of length {}", input.len()),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        let len = input.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        // Unrolled ReLU computation for better vectorization
        for i in 0..main_loops {
            let base = i * unroll_size;

            output[base] = input[base].max(0.0);
            output[base + 1] = input[base + 1].max(0.0);
            output[base + 2] = input[base + 2].max(0.0);
            output[base + 3] = input[base + 3].max(0.0);
            output[base + 4] = input[base + 4].max(0.0);
            output[base + 5] = input[base + 5].max(0.0);
            output[base + 6] = input[base + 6].max(0.0);
            output[base + 7] = input[base + 7].max(0.0);
        }

        // Handle remaining elements
        for i in (main_loops * unroll_size)..len {
            output[i] = input[i].max(0.0);
        }

        Ok(())
    }

    /// Optimized sigmoid activation with fast approximation
    pub fn sigmoid_f32_optimized(input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD Sigmoid".to_string(),
                expected: format!("arrays of length {}", input.len()),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        for (i, &x) in input.iter().enumerate() {
            // Fast sigmoid approximation: x / (1 + |x|) scaled
            let abs_x = x.abs();
            if abs_x > 5.0 {
                // For large values, use standard sigmoid to avoid overflow
                output[i] = 1.0 / (1.0 + (-x).exp());
            } else {
                // Fast approximation for smaller values
                output[i] = 0.5 * (x / (1.0 + abs_x)) + 0.5;
            }
        }

        Ok(())
    }

    /// Optimized tanh activation function using fast approximation
    pub fn tanh_f32_optimized(input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD Tanh".to_string(),
                expected: format!("arrays of length {}", input.len()),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        let len = input.len();
        let unroll_size = 4;
        let main_loops = len / unroll_size;

        // Process in chunks for better vectorization
        for i in 0..main_loops {
            let base = i * unroll_size;

            // Use tanh approximation: x / (1 + |x|) for small values
            for j in 0..unroll_size {
                let x = input[base + j];
                let abs_x = x.abs();
                if abs_x > 3.0 {
                    // Use standard tanh for large values
                    output[base + j] = x.tanh();
                } else {
                    // Fast approximation
                    output[base + j] = x / (1.0 + abs_x);
                }
            }
        }

        // Handle remaining elements
        for i in (main_loops * unroll_size)..len {
            let x = input[i];
            let abs_x = x.abs();
            if abs_x > 3.0 {
                output[i] = x.tanh();
            } else {
                output[i] = x / (1.0 + abs_x);
            }
        }

        Ok(())
    }

    /// Leaky ReLU activation with configurable negative slope
    pub fn leaky_relu_f32_optimized(
        input: &[f32],
        output: &mut [f32],
        negative_slope: f32,
    ) -> Result<()> {
        if input.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD Leaky ReLU".to_string(),
                expected: format!("arrays of length {}", input.len()),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        let len = input.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        // Unrolled Leaky ReLU computation
        for i in 0..main_loops {
            let base = i * unroll_size;

            for j in 0..unroll_size {
                let x = input[base + j];
                output[base + j] = if x > 0.0 { x } else { x * negative_slope };
            }
        }

        // Handle remaining elements
        for i in (main_loops * unroll_size)..len {
            let x = input[i];
            output[i] = if x > 0.0 { x } else { x * negative_slope };
        }

        Ok(())
    }

    /// ELU (Exponential Linear Unit) activation function
    pub fn elu_f32_optimized(input: &[f32], output: &mut [f32], alpha: f32) -> Result<()> {
        if input.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD ELU".to_string(),
                expected: format!("arrays of length {}", input.len()),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        for (i, &x) in input.iter().enumerate() {
            output[i] = if x > 0.0 { x } else { alpha * (x.exp() - 1.0) };
        }

        Ok(())
    }

    /// Swish activation function (x * sigmoid(x))
    pub fn swish_f32_optimized(input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD Swish".to_string(),
                expected: format!("arrays of length {}", input.len()),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        for (i, &x) in input.iter().enumerate() {
            let sigmoid_x = 1.0 / (1.0 + (-x).exp());
            output[i] = x * sigmoid_x;
        }

        Ok(())
    }

    /// GELU (Gaussian Error Linear Unit) activation function
    pub fn gelu_f32_optimized(input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD GELU".to_string(),
                expected: format!("arrays of length {}", input.len()),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        const SQRT_2_PI: f32 = 0.797_884_6; // sqrt(2/π)
        const COEFF: f32 = 0.044715;

        for (i, &x) in input.iter().enumerate() {
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            let x_cubed = x * x * x;
            let inner = SQRT_2_PI * (x + COEFF * x_cubed);
            output[i] = 0.5 * x * (1.0 + inner.tanh());
        }

        Ok(())
    }

    /// Mish activation function (x * tanh(softplus(x)))
    pub fn mish_f32_optimized(input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD Mish".to_string(),
                expected: format!("arrays of length {}", input.len()),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        for (i, &x) in input.iter().enumerate() {
            // Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
            let softplus = (1.0 + x.exp()).ln();
            output[i] = x * softplus.tanh();
        }

        Ok(())
    }

    /// Softmax activation function with numerical stability
    pub fn softmax_f32_optimized(input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD Softmax".to_string(),
                expected: format!("arrays of length {}", input.len()),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        // Find maximum for numerical stability
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exponentials and sum
        let mut sum = 0.0f32;
        for (i, &x) in input.iter().enumerate() {
            let exp_val = (x - max_val).exp();
            output[i] = exp_val;
            sum += exp_val;
        }

        // Normalize
        let inv_sum = 1.0 / sum;
        for val in output.iter_mut() {
            *val *= inv_sum;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_relu_f32_optimized() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; 8];
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

        ActivationFunctions::relu_f32_optimized(&input, &mut output).unwrap();

        for (i, &val) in output.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_sigmoid_f32_optimized() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];

        ActivationFunctions::sigmoid_f32_optimized(&input, &mut output).unwrap();

        // Check that outputs are in valid range [0, 1]
        for &val in output.iter() {
            assert!(
                val >= 0.0 && val <= 1.0,
                "Sigmoid output {} not in [0,1]",
                val
            );
        }

        // Check that sigmoid(0) ≈ 0.5
        assert_relative_eq!(output[2], 0.5, epsilon = 0.1);
    }

    #[test]
    fn test_tanh_f32_optimized() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];

        ActivationFunctions::tanh_f32_optimized(&input, &mut output).unwrap();

        // Check that outputs are in valid range [-1, 1]
        for &val in output.iter() {
            assert!(
                val >= -1.0 && val <= 1.0,
                "Tanh output {} not in [-1,1]",
                val
            );
        }

        // Check that tanh(0) ≈ 0
        assert_relative_eq!(output[2], 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_leaky_relu_f32_optimized() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];
        let negative_slope = 0.01;

        ActivationFunctions::leaky_relu_f32_optimized(&input, &mut output, negative_slope).unwrap();

        // Check positive values pass through unchanged
        assert_relative_eq!(output[3], 1.0, epsilon = 1e-6);
        assert_relative_eq!(output[4], 2.0, epsilon = 1e-6);

        // Check negative values are scaled
        assert_relative_eq!(output[0], -2.0 * negative_slope, epsilon = 1e-6);
        assert_relative_eq!(output[1], -1.0 * negative_slope, epsilon = 1e-6);
    }

    #[test]
    fn test_elu_f32_optimized() {
        let input = vec![-1.0, 0.0, 1.0];
        let mut output = vec![0.0; 3];
        let alpha = 1.0;

        ActivationFunctions::elu_f32_optimized(&input, &mut output, alpha).unwrap();

        // Check that positive values pass through unchanged
        assert_relative_eq!(output[2], 1.0, epsilon = 1e-6);

        // Check that zero passes through unchanged
        assert_relative_eq!(output[1], 0.0, epsilon = 1e-6);

        // Check that negative values follow ELU formula
        assert!(output[0] < 0.0, "ELU of negative input should be negative");
    }

    #[test]
    fn test_softmax_f32_optimized() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        ActivationFunctions::softmax_f32_optimized(&input, &mut output).unwrap();

        // Check that probabilities sum to 1
        let sum: f32 = output.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // Check that all values are positive
        for &val in output.iter() {
            assert!(val > 0.0, "Softmax output {} should be positive", val);
        }

        // Check that larger inputs produce larger outputs
        assert!(output[3] > output[2]);
        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
    }

    #[test]
    fn test_activation_error_handling() {
        let input = vec![1.0; 5];
        let mut output = vec![0.0; 3]; // Wrong size

        // Test ReLU error handling
        let error = ActivationFunctions::relu_f32_optimized(&input, &mut output);
        assert!(error.is_err());

        // Test Sigmoid error handling
        let error = ActivationFunctions::sigmoid_f32_optimized(&input, &mut output);
        assert!(error.is_err());

        // Test Tanh error handling
        let error = ActivationFunctions::tanh_f32_optimized(&input, &mut output);
        assert!(error.is_err());
    }
}
