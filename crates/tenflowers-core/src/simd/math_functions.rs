//! SIMD-Optimized Mathematical Functions
//!
//! This module provides high-performance implementations of mathematical functions
//! including exponential, logarithmic, trigonometric, and power functions using SIMD optimizations.

use crate::error::ErrorContext;
use crate::{Result, TensorError};

/// SIMD-optimized mathematical functions
pub struct MathFunctions;

impl MathFunctions {
    /// Optimized exponential function using fast approximation
    pub fn exp_f32_optimized(input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD exp".to_string(),
                expected: format!("arrays of length {}", input.len()),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        let len = input.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        // Use fast approximation for moderate values, accurate exp for extremes
        for i in 0..main_loops {
            let base = i * unroll_size;

            for j in 0..8 {
                let x = input[base + j];
                output[base + j] = if x.abs() > 5.0 {
                    // Use standard exp for extreme values
                    x.exp()
                } else {
                    // Fast polynomial approximation: e^x ≈ 1 + x + x²/2 + x³/6 + x⁴/24
                    let x2 = x * x;
                    let x3 = x2 * x;
                    let x4 = x2 * x2;
                    1.0 + x + x2 * 0.5 + x3 / 6.0 + x4 / 24.0
                };
            }
        }

        // Handle remaining elements
        for i in (main_loops * unroll_size)..len {
            let x = input[i];
            output[i] = if x.abs() > 5.0 {
                x.exp()
            } else {
                let x2 = x * x;
                let x3 = x2 * x;
                let x4 = x2 * x2;
                1.0 + x + x2 * 0.5 + x3 / 6.0 + x4 / 24.0
            };
        }

        Ok(())
    }

    /// Element-wise square root with SIMD optimization hints
    pub fn sqrt_f32_optimized(input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD sqrt_f32".to_string(),
                expected: format!("arrays of length {}", input.len()),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        let len = input.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        // Unrolled sqrt computation for better vectorization
        for i in 0..main_loops {
            let base = i * unroll_size;

            output[base] = input[base].sqrt();
            output[base + 1] = input[base + 1].sqrt();
            output[base + 2] = input[base + 2].sqrt();
            output[base + 3] = input[base + 3].sqrt();
            output[base + 4] = input[base + 4].sqrt();
            output[base + 5] = input[base + 5].sqrt();
            output[base + 6] = input[base + 6].sqrt();
            output[base + 7] = input[base + 7].sqrt();
        }

        // Handle remaining elements
        for i in (main_loops * unroll_size)..len {
            output[i] = input[i].sqrt();
        }

        Ok(())
    }

    /// Element-wise natural logarithm with SIMD optimization hints
    pub fn log_f32_optimized(input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD log_f32".to_string(),
                expected: format!("arrays of length {}", input.len()),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        let len = input.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        // Unrolled ln computation for better vectorization
        for i in 0..main_loops {
            let base = i * unroll_size;

            output[base] = input[base].ln();
            output[base + 1] = input[base + 1].ln();
            output[base + 2] = input[base + 2].ln();
            output[base + 3] = input[base + 3].ln();
            output[base + 4] = input[base + 4].ln();
            output[base + 5] = input[base + 5].ln();
            output[base + 6] = input[base + 6].ln();
            output[base + 7] = input[base + 7].ln();
        }

        // Handle remaining elements
        for i in (main_loops * unroll_size)..len {
            output[i] = input[i].ln();
        }

        Ok(())
    }

    /// Element-wise power operation (x^y) with SIMD optimization hints
    pub fn pow_f32_optimized(base: &[f32], exponent: &[f32], output: &mut [f32]) -> Result<()> {
        if base.len() != exponent.len() || base.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD pow_f32".to_string(),
                expected: format!("arrays of length {}", base.len()),
                got: format!(
                    "base: {}, exponent: {}, output: {}",
                    base.len(),
                    exponent.len(),
                    output.len()
                ),
                context: Some(ErrorContext::new()),
            });
        }

        let len = base.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        // Unrolled power computation for better vectorization
        for i in 0..main_loops {
            let idx = i * unroll_size;

            output[idx] = base[idx].powf(exponent[idx]);
            output[idx + 1] = base[idx + 1].powf(exponent[idx + 1]);
            output[idx + 2] = base[idx + 2].powf(exponent[idx + 2]);
            output[idx + 3] = base[idx + 3].powf(exponent[idx + 3]);
            output[idx + 4] = base[idx + 4].powf(exponent[idx + 4]);
            output[idx + 5] = base[idx + 5].powf(exponent[idx + 5]);
            output[idx + 6] = base[idx + 6].powf(exponent[idx + 6]);
            output[idx + 7] = base[idx + 7].powf(exponent[idx + 7]);
        }

        // Handle remaining elements
        for i in (main_loops * unroll_size)..len {
            output[i] = base[i].powf(exponent[i]);
        }

        Ok(())
    }

    /// Element-wise sine function with SIMD optimization hints
    pub fn sin_f32_optimized(input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD sin_f32".to_string(),
                expected: format!("arrays of length {}", input.len()),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        let len = input.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        // Unrolled sine computation for better vectorization
        for i in 0..main_loops {
            let base = i * unroll_size;

            output[base] = input[base].sin();
            output[base + 1] = input[base + 1].sin();
            output[base + 2] = input[base + 2].sin();
            output[base + 3] = input[base + 3].sin();
            output[base + 4] = input[base + 4].sin();
            output[base + 5] = input[base + 5].sin();
            output[base + 6] = input[base + 6].sin();
            output[base + 7] = input[base + 7].sin();
        }

        // Handle remaining elements
        for i in (main_loops * unroll_size)..len {
            output[i] = input[i].sin();
        }

        Ok(())
    }

    /// Element-wise cosine function with SIMD optimization hints
    pub fn cos_f32_optimized(input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD cos_f32".to_string(),
                expected: format!("arrays of length {}", input.len()),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        let len = input.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        // Unrolled cosine computation for better vectorization
        for i in 0..main_loops {
            let base = i * unroll_size;

            output[base] = input[base].cos();
            output[base + 1] = input[base + 1].cos();
            output[base + 2] = input[base + 2].cos();
            output[base + 3] = input[base + 3].cos();
            output[base + 4] = input[base + 4].cos();
            output[base + 5] = input[base + 5].cos();
            output[base + 6] = input[base + 6].cos();
            output[base + 7] = input[base + 7].cos();
        }

        // Handle remaining elements
        for i in (main_loops * unroll_size)..len {
            output[i] = input[i].cos();
        }

        Ok(())
    }

    /// Element-wise absolute value with SIMD optimization hints
    pub fn abs_f32_optimized(input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD abs_f32".to_string(),
                expected: format!("arrays of length {}", input.len()),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        let len = input.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        // Unrolled absolute value computation for better vectorization
        for i in 0..main_loops {
            let base = i * unroll_size;

            output[base] = input[base].abs();
            output[base + 1] = input[base + 1].abs();
            output[base + 2] = input[base + 2].abs();
            output[base + 3] = input[base + 3].abs();
            output[base + 4] = input[base + 4].abs();
            output[base + 5] = input[base + 5].abs();
            output[base + 6] = input[base + 6].abs();
            output[base + 7] = input[base + 7].abs();
        }

        // Handle remaining elements
        for i in (main_loops * unroll_size)..len {
            output[i] = input[i].abs();
        }

        Ok(())
    }

    /// Element-wise subtraction with SIMD optimization hints
    pub fn sub_f32_optimized(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD sub_f32".to_string(),
                expected: format!("arrays of length {}", a.len()),
                got: format!("a: {}, b: {}, result: {}", a.len(), b.len(), result.len()),
                context: Some(ErrorContext::new()),
            });
        }

        let len = a.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        // Process 8 elements at a time for better vectorization
        for i in 0..main_loops {
            let base = i * unroll_size;

            result[base] = a[base] - b[base];
            result[base + 1] = a[base + 1] - b[base + 1];
            result[base + 2] = a[base + 2] - b[base + 2];
            result[base + 3] = a[base + 3] - b[base + 3];
            result[base + 4] = a[base + 4] - b[base + 4];
            result[base + 5] = a[base + 5] - b[base + 5];
            result[base + 6] = a[base + 6] - b[base + 6];
            result[base + 7] = a[base + 7] - b[base + 7];
        }

        // Handle remaining elements
        for i in (main_loops * unroll_size)..len {
            result[i] = a[i] - b[i];
        }

        Ok(())
    }

    /// Element-wise division with SIMD optimization hints and division by zero protection
    pub fn div_f32_optimized(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD div_f32".to_string(),
                expected: format!("arrays of length {}", a.len()),
                got: format!("a: {}, b: {}, result: {}", a.len(), b.len(), result.len()),
                context: Some(ErrorContext::new()),
            });
        }

        let len = a.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;
        const EPSILON: f32 = 1e-8;

        // Process 8 elements at a time for better vectorization
        for i in 0..main_loops {
            let base = i * unroll_size;

            result[base] = a[base] / (b[base] + EPSILON);
            result[base + 1] = a[base + 1] / (b[base + 1] + EPSILON);
            result[base + 2] = a[base + 2] / (b[base + 2] + EPSILON);
            result[base + 3] = a[base + 3] / (b[base + 3] + EPSILON);
            result[base + 4] = a[base + 4] / (b[base + 4] + EPSILON);
            result[base + 5] = a[base + 5] / (b[base + 5] + EPSILON);
            result[base + 6] = a[base + 6] / (b[base + 6] + EPSILON);
            result[base + 7] = a[base + 7] / (b[base + 7] + EPSILON);
        }

        // Handle remaining elements
        for i in (main_loops * unroll_size)..len {
            result[i] = a[i] / (b[i] + EPSILON);
        }

        Ok(())
    }

    /// Fast reciprocal (1/x) function with SIMD optimization
    pub fn reciprocal_f32_optimized(input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD reciprocal".to_string(),
                expected: format!("arrays of length {}", input.len()),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        const EPSILON: f32 = 1e-8;
        let len = input.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        // Unrolled reciprocal computation for better vectorization
        for i in 0..main_loops {
            let base = i * unroll_size;

            output[base] = 1.0 / (input[base] + EPSILON);
            output[base + 1] = 1.0 / (input[base + 1] + EPSILON);
            output[base + 2] = 1.0 / (input[base + 2] + EPSILON);
            output[base + 3] = 1.0 / (input[base + 3] + EPSILON);
            output[base + 4] = 1.0 / (input[base + 4] + EPSILON);
            output[base + 5] = 1.0 / (input[base + 5] + EPSILON);
            output[base + 6] = 1.0 / (input[base + 6] + EPSILON);
            output[base + 7] = 1.0 / (input[base + 7] + EPSILON);
        }

        // Handle remaining elements
        for i in (main_loops * unroll_size)..len {
            output[i] = 1.0 / (input[i] + EPSILON);
        }

        Ok(())
    }

    /// Element-wise clamp operation (min/max bounds)
    pub fn clamp_f32_optimized(
        input: &[f32],
        output: &mut [f32],
        min_val: f32,
        max_val: f32,
    ) -> Result<()> {
        if input.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD clamp".to_string(),
                expected: format!("arrays of length {}", input.len()),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        let len = input.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        // Unrolled clamp computation for better vectorization
        for i in 0..main_loops {
            let base = i * unroll_size;

            output[base] = input[base].clamp(min_val, max_val);
            output[base + 1] = input[base + 1].clamp(min_val, max_val);
            output[base + 2] = input[base + 2].clamp(min_val, max_val);
            output[base + 3] = input[base + 3].clamp(min_val, max_val);
            output[base + 4] = input[base + 4].clamp(min_val, max_val);
            output[base + 5] = input[base + 5].clamp(min_val, max_val);
            output[base + 6] = input[base + 6].clamp(min_val, max_val);
            output[base + 7] = input[base + 7].clamp(min_val, max_val);
        }

        // Handle remaining elements
        for i in (main_loops * unroll_size)..len {
            output[i] = input[i].clamp(min_val, max_val);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_exp_f32_optimized() {
        let input = vec![0.0, 1.0, -1.0, 2.0];
        let mut output = vec![0.0; 4];

        MathFunctions::exp_f32_optimized(&input, &mut output).unwrap();

        // Check that exp(0) ≈ 1
        assert_relative_eq!(output[0], 1.0, epsilon = 0.1);

        // Check that exp(1) ≈ e ≈ 2.718
        assert_relative_eq!(output[1], std::f32::consts::E, epsilon = 0.1);

        // Check that all outputs are positive
        for &val in output.iter() {
            assert!(val > 0.0, "Exponential output {} should be positive", val);
        }
    }

    #[test]
    fn test_sqrt_f32_optimized() {
        let input = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let mut output = vec![0.0; 5];
        let expected = vec![0.0, 1.0, 2.0, 3.0, 4.0];

        MathFunctions::sqrt_f32_optimized(&input, &mut output).unwrap();

        for (i, &val) in output.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_log_f32_optimized() {
        let input = vec![1.0, std::f32::consts::E, 10.0];
        let mut output = vec![0.0; 3];

        MathFunctions::log_f32_optimized(&input, &mut output).unwrap();

        // Check that ln(1) = 0
        assert_relative_eq!(output[0], 0.0, epsilon = 1e-6);

        // Check that ln(e) = 1
        assert_relative_eq!(output[1], 1.0, epsilon = 1e-6);

        // Check that ln(10) ≈ 2.303
        assert_relative_eq!(output[2], 10.0f32.ln(), epsilon = 1e-6);
    }

    #[test]
    fn test_pow_f32_optimized() {
        let base = vec![2.0, 3.0, 4.0, 5.0];
        let exponent = vec![2.0, 3.0, 0.5, 0.0];
        let mut output = vec![0.0; 4];
        let expected = vec![4.0, 27.0, 2.0, 1.0]; // 2², 3³, √4, 5⁰

        MathFunctions::pow_f32_optimized(&base, &exponent, &mut output).unwrap();

        for (i, &val) in output.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_sin_f32_optimized() {
        let input = vec![
            0.0,
            std::f32::consts::PI / 6.0,
            std::f32::consts::PI / 4.0,
            std::f32::consts::PI / 2.0,
        ];
        let mut output = vec![0.0; 4];
        let expected = vec![0.0, 0.5, std::f32::consts::FRAC_1_SQRT_2, 1.0];

        MathFunctions::sin_f32_optimized(&input, &mut output).unwrap();

        for (i, &val) in output.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_cos_f32_optimized() {
        let input = vec![
            0.0,
            std::f32::consts::PI / 6.0,
            std::f32::consts::PI / 4.0,
            std::f32::consts::PI / 3.0,
            std::f32::consts::PI / 2.0,
        ];
        let mut output = vec![0.0; 5];
        let expected = vec![
            1.0,
            (3.0_f32).sqrt() / 2.0,
            std::f32::consts::FRAC_1_SQRT_2,
            0.5,
            0.0,
        ];

        MathFunctions::cos_f32_optimized(&input, &mut output).unwrap();

        for (i, &val) in output.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_abs_f32_optimized() {
        let input = vec![-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, -7.0];
        let mut output = vec![0.0; 8];
        let expected = vec![5.0, 3.0, 1.0, 0.0, 1.0, 3.0, 5.0, 7.0];

        MathFunctions::abs_f32_optimized(&input, &mut output).unwrap();

        for (i, &val) in output.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_sub_f32_optimized() {
        let a = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut result = vec![0.0; 5];
        let expected = vec![4.0, 2.0, 0.0, -2.0, -4.0];

        MathFunctions::sub_f32_optimized(&a, &b, &mut result).unwrap();

        for (i, &val) in result.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_div_f32_optimized() {
        let a = vec![6.0, 8.0, 10.0, 12.0];
        let b = vec![2.0, 4.0, 5.0, 3.0];
        let mut result = vec![0.0; 4];
        let expected = vec![3.0, 2.0, 2.0, 4.0];

        MathFunctions::div_f32_optimized(&a, &b, &mut result).unwrap();

        for (i, &val) in result.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_reciprocal_f32_optimized() {
        let input = vec![1.0, 2.0, 4.0, 5.0, 10.0];
        let mut output = vec![0.0; 5];
        let expected = vec![1.0, 0.5, 0.25, 0.2, 0.1];

        MathFunctions::reciprocal_f32_optimized(&input, &mut output).unwrap();

        for (i, &val) in output.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_clamp_f32_optimized() {
        let input = vec![-5.0, -1.0, 0.0, 1.0, 5.0, 10.0];
        let mut output = vec![0.0; 6];
        let min_val = -2.0;
        let max_val = 3.0;
        let expected = vec![-2.0, -1.0, 0.0, 1.0, 3.0, 3.0];

        MathFunctions::clamp_f32_optimized(&input, &mut output, min_val, max_val).unwrap();

        for (i, &val) in output.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_mathematical_functions_error_handling() {
        let input = vec![1.0; 5];
        let mut output = vec![0.0; 3]; // Wrong size

        // Test sqrt error handling
        let error = MathFunctions::sqrt_f32_optimized(&input, &mut output);
        assert!(error.is_err());

        // Test log error handling
        let error = MathFunctions::log_f32_optimized(&input, &mut output);
        assert!(error.is_err());

        // Test abs error handling
        let error = MathFunctions::abs_f32_optimized(&input, &mut output);
        assert!(error.is_err());
    }

    #[test]
    fn test_pow_error_handling() {
        let base = vec![1.0; 5];
        let exponent = vec![2.0; 3]; // Wrong size
        let mut output = vec![0.0; 5];

        let error = MathFunctions::pow_f32_optimized(&base, &exponent, &mut output);
        assert!(error.is_err());
    }
}
