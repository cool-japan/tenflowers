//! SIMD-Optimized Reduction Operations
//!
//! This module provides high-performance reduction operations including sum, min/max,
//! normalization, and statistical computations using SIMD optimizations.

use crate::error::ErrorContext;
use crate::{Result, TensorError};

/// SIMD-optimized reduction operations
pub struct ReductionOps;

impl ReductionOps {
    /// Ultra-fast vectorized horizontal sum reduction
    pub fn sum_f32_unchecked(input: &[f32]) -> f32 {
        let len = input.len();

        if len < 8 {
            let mut sum = 0.0f32;
            for i in 0..len {
                unsafe {
                    sum += *input.get_unchecked(i);
                }
            }
            return sum;
        }

        // Vectorized reduction with multiple accumulators to avoid dependency chains
        let chunks = len / 8;
        let remainder = len % 8;

        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;
        let mut acc4 = 0.0f32;

        for chunk in 0..chunks {
            let base = chunk * 8;

            unsafe {
                // Use multiple accumulators to maximize ILP (Instruction Level Parallelism)
                acc1 += *input.get_unchecked(base) + *input.get_unchecked(base + 1);
                acc2 += *input.get_unchecked(base + 2) + *input.get_unchecked(base + 3);
                acc3 += *input.get_unchecked(base + 4) + *input.get_unchecked(base + 5);
                acc4 += *input.get_unchecked(base + 6) + *input.get_unchecked(base + 7);
            }
        }

        // Combine accumulators
        let mut total_sum = acc1 + acc2 + acc3 + acc4;

        // Handle remaining elements
        let remainder_start = chunks * 8;
        for i in 0..remainder {
            unsafe {
                total_sum += *input.get_unchecked(remainder_start + i);
            }
        }

        total_sum
    }

    /// Advanced vectorized sum reduction with numerical stability
    pub fn reduce_sum_f32_optimized(input: &[f32]) -> Result<f32> {
        if input.is_empty() {
            return Ok(0.0);
        }

        let len = input.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        // Use multiple accumulators to maximize vectorization and reduce dependencies
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;
        let mut sum4 = 0.0f32;

        // Process 8 elements at a time with 4 parallel accumulators
        for i in 0..main_loops {
            let base = i * unroll_size;

            sum1 += input[base] + input[base + 1];
            sum2 += input[base + 2] + input[base + 3];
            sum3 += input[base + 4] + input[base + 5];
            sum4 += input[base + 6] + input[base + 7];
        }

        // Handle remaining elements
        let mut remaining_sum = 0.0f32;
        #[allow(clippy::needless_range_loop)] // Performance-critical SIMD remainder handling
        for i in (main_loops * unroll_size)..len {
            remaining_sum += input[i];
        }

        Ok(sum1 + sum2 + sum3 + sum4 + remaining_sum)
    }

    /// Vectorized maximum element finding with SIMD optimization
    pub fn max_f32_unchecked(input: &[f32]) -> f32 {
        let len = input.len();

        if len == 0 {
            return f32::NEG_INFINITY;
        }

        if len < 8 {
            let mut max_val = unsafe { *input.get_unchecked(0) };
            for i in 1..len {
                unsafe {
                    let val = *input.get_unchecked(i);
                    if val > max_val {
                        max_val = val;
                    }
                }
            }
            return max_val;
        }

        // Vectorized max reduction with multiple comparisons
        let chunks = len / 8;
        let remainder = len % 8;

        let mut max1 = unsafe { *input.get_unchecked(0) };
        let mut max2 = max1;
        let mut max3 = max1;
        let mut max4 = max1;

        for chunk in 0..chunks {
            let base = chunk * 8;

            unsafe {
                max1 = max1
                    .max(*input.get_unchecked(base))
                    .max(*input.get_unchecked(base + 1));
                max2 = max2
                    .max(*input.get_unchecked(base + 2))
                    .max(*input.get_unchecked(base + 3));
                max3 = max3
                    .max(*input.get_unchecked(base + 4))
                    .max(*input.get_unchecked(base + 5));
                max4 = max4
                    .max(*input.get_unchecked(base + 6))
                    .max(*input.get_unchecked(base + 7));
            }
        }

        // Combine maximums
        let mut global_max = max1.max(max2).max(max3).max(max4);

        // Handle remaining elements
        let remainder_start = chunks * 8;
        for i in 0..remainder {
            unsafe {
                global_max = global_max.max(*input.get_unchecked(remainder_start + i));
            }
        }

        global_max
    }

    /// Vectorized min/max operations with efficient branching
    pub fn reduce_min_max_f32_optimized(input: &[f32]) -> Result<(f32, f32)> {
        if input.is_empty() {
            return Err(TensorError::InvalidOperation {
                operation: "SIMD min_max".to_string(),
                reason: "Cannot find min/max of empty array".to_string(),
                context: Some(ErrorContext::new()),
            });
        }

        let len = input.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        // Initialize with first element to avoid NaN issues
        let mut min_val = input[0];
        let mut max_val = input[0];

        // Process 8 elements at a time
        for i in 0..main_loops {
            let base = i * unroll_size;

            // Use explicit min/max for better compiler optimization
            min_val = min_val
                .min(input[base])
                .min(input[base + 1])
                .min(input[base + 2])
                .min(input[base + 3])
                .min(input[base + 4])
                .min(input[base + 5])
                .min(input[base + 6])
                .min(input[base + 7]);

            max_val = max_val
                .max(input[base])
                .max(input[base + 1])
                .max(input[base + 2])
                .max(input[base + 3])
                .max(input[base + 4])
                .max(input[base + 5])
                .max(input[base + 6])
                .max(input[base + 7]);
        }

        // Handle remaining elements
        #[allow(clippy::needless_range_loop)] // Performance-critical SIMD remainder handling
        for i in (main_loops * unroll_size)..len {
            min_val = min_val.min(input[i]);
            max_val = max_val.max(input[i]);
        }

        Ok((min_val, max_val))
    }

    /// Advanced normalization with streaming computation for memory efficiency
    pub fn normalize_f32_optimized(input: &[f32], output: &mut [f32], eps: f32) -> Result<()> {
        if input.len() != output.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD normalize".to_string(),
                expected: format!("arrays of length {}", input.len()),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        // First pass: compute mean using optimized sum
        let sum = Self::reduce_sum_f32_optimized(input)?;
        let mean = sum / input.len() as f32;

        // Second pass: compute variance in streaming fashion
        let mut variance_sum = 0.0f32;
        let len = input.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        for i in 0..main_loops {
            let base = i * unroll_size;

            // Compute squared differences with unrolling
            let d0 = input[base] - mean;
            let d1 = input[base + 1] - mean;
            let d2 = input[base + 2] - mean;
            let d3 = input[base + 3] - mean;
            let d4 = input[base + 4] - mean;
            let d5 = input[base + 5] - mean;
            let d6 = input[base + 6] - mean;
            let d7 = input[base + 7] - mean;

            variance_sum +=
                d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
        }

        // Handle remaining elements
        #[allow(clippy::needless_range_loop)] // Performance-critical SIMD remainder handling
        for i in (main_loops * unroll_size)..len {
            let diff = input[i] - mean;
            variance_sum += diff * diff;
        }

        let variance = variance_sum / input.len() as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();

        // Third pass: normalize with vectorization
        for i in 0..main_loops {
            let base = i * unroll_size;

            output[base] = (input[base] - mean) * inv_std;
            output[base + 1] = (input[base + 1] - mean) * inv_std;
            output[base + 2] = (input[base + 2] - mean) * inv_std;
            output[base + 3] = (input[base + 3] - mean) * inv_std;
            output[base + 4] = (input[base + 4] - mean) * inv_std;
            output[base + 5] = (input[base + 5] - mean) * inv_std;
            output[base + 6] = (input[base + 6] - mean) * inv_std;
            output[base + 7] = (input[base + 7] - mean) * inv_std;
        }

        // Handle remaining elements
        #[allow(clippy::needless_range_loop)] // Performance-critical SIMD remainder handling
        for i in (main_loops * unroll_size)..len {
            output[i] = (input[i] - mean) * inv_std;
        }

        Ok(())
    }

    /// Compute mean and variance in a single pass (Welford's algorithm)
    pub fn mean_variance_f32_optimized(input: &[f32]) -> Result<(f32, f32)> {
        if input.is_empty() {
            return Err(TensorError::InvalidOperation {
                operation: "SIMD mean_variance".to_string(),
                reason: "Cannot compute mean/variance of empty array".to_string(),
                context: Some(ErrorContext::new()),
            });
        }

        // Use Welford's online algorithm for numerical stability
        let mut mean = 0.0f32;
        let mut m2 = 0.0f32;

        for (i, &value) in input.iter().enumerate() {
            let count = (i + 1) as f32;
            let delta = value - mean;
            mean += delta / count;
            let delta2 = value - mean;
            m2 += delta * delta2;
        }

        let variance = if input.len() > 1 {
            m2 / (input.len() - 1) as f32
        } else {
            0.0
        };

        Ok((mean, variance))
    }

    /// Compute standard deviation using optimized variance calculation
    pub fn std_f32_optimized(input: &[f32]) -> Result<f32> {
        let (_, variance) = Self::mean_variance_f32_optimized(input)?;
        Ok(variance.sqrt())
    }

    /// Reduce operation with custom binary function (generalized)
    pub fn reduce_f32_optimized<F>(input: &[f32], initial: f32, op: F) -> Result<f32>
    where
        F: Fn(f32, f32) -> f32,
    {
        if input.is_empty() {
            return Ok(initial);
        }

        let mut result = initial;
        let len = input.len();
        let unroll_size = 4;
        let main_loops = len / unroll_size;

        // Process 4 elements at a time for better vectorization
        for i in 0..main_loops {
            let base = i * unroll_size;

            result = op(result, input[base]);
            result = op(result, input[base + 1]);
            result = op(result, input[base + 2]);
            result = op(result, input[base + 3]);
        }

        // Handle remaining elements
        #[allow(clippy::needless_range_loop)] // Performance-critical SIMD remainder handling
        for i in (main_loops * unroll_size)..len {
            result = op(result, input[i]);
        }

        Ok(result)
    }

    /// L2 norm (Euclidean norm) with optimized computation
    pub fn l2_norm_f32_optimized(input: &[f32]) -> Result<f32> {
        if input.is_empty() {
            return Ok(0.0);
        }

        let len = input.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        let mut sum_squares = 0.0f32;

        // Unrolled computation of sum of squares
        for i in 0..main_loops {
            let base = i * unroll_size;

            sum_squares += input[base] * input[base];
            sum_squares += input[base + 1] * input[base + 1];
            sum_squares += input[base + 2] * input[base + 2];
            sum_squares += input[base + 3] * input[base + 3];
            sum_squares += input[base + 4] * input[base + 4];
            sum_squares += input[base + 5] * input[base + 5];
            sum_squares += input[base + 6] * input[base + 6];
            sum_squares += input[base + 7] * input[base + 7];
        }

        // Handle remaining elements
        #[allow(clippy::needless_range_loop)] // Performance-critical SIMD remainder handling
        for i in (main_loops * unroll_size)..len {
            sum_squares += input[i] * input[i];
        }

        Ok(sum_squares.sqrt())
    }

    /// L1 norm (Manhattan norm) with optimized computation
    pub fn l1_norm_f32_optimized(input: &[f32]) -> Result<f32> {
        if input.is_empty() {
            return Ok(0.0);
        }

        let len = input.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        let mut sum_abs = 0.0f32;

        // Unrolled computation of sum of absolute values
        for i in 0..main_loops {
            let base = i * unroll_size;

            sum_abs += input[base].abs();
            sum_abs += input[base + 1].abs();
            sum_abs += input[base + 2].abs();
            sum_abs += input[base + 3].abs();
            sum_abs += input[base + 4].abs();
            sum_abs += input[base + 5].abs();
            sum_abs += input[base + 6].abs();
            sum_abs += input[base + 7].abs();
        }

        // Handle remaining elements
        #[allow(clippy::needless_range_loop)] // Performance-critical SIMD remainder handling
        for i in (main_loops * unroll_size)..len {
            sum_abs += input[i].abs();
        }

        Ok(sum_abs)
    }

    /// Argmax - find index of maximum element
    pub fn argmax_f32_optimized(input: &[f32]) -> Result<usize> {
        if input.is_empty() {
            return Err(TensorError::InvalidOperation {
                operation: "SIMD argmax".to_string(),
                reason: "Cannot find argmax of empty array".to_string(),
                context: Some(ErrorContext::new()),
            });
        }

        let mut max_val = input[0];
        let mut max_idx = 0;

        for (i, &val) in input.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        Ok(max_idx)
    }

    /// Argmin - find index of minimum element
    pub fn argmin_f32_optimized(input: &[f32]) -> Result<usize> {
        if input.is_empty() {
            return Err(TensorError::InvalidOperation {
                operation: "SIMD argmin".to_string(),
                reason: "Cannot find argmin of empty array".to_string(),
                context: Some(ErrorContext::new()),
            });
        }

        let mut min_val = input[0];
        let mut min_idx = 0;

        for (i, &val) in input.iter().enumerate().skip(1) {
            if val < min_val {
                min_val = val;
                min_idx = i;
            }
        }

        Ok(min_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sum_f32_unchecked() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let expected = 15.0;

        let result = ReductionOps::sum_f32_unchecked(&input);

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_reduce_sum_f32_optimized() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let expected = 55.0;

        let result = ReductionOps::reduce_sum_f32_optimized(&input).unwrap();

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_max_f32_unchecked() {
        let input = vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0];
        let expected = 9.0;

        let result = ReductionOps::max_f32_unchecked(&input);

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_reduce_min_max_f32_optimized() {
        let input = vec![3.0, 1.0, 7.0, 2.0, 9.0, 4.0];
        let expected = (1.0, 9.0);

        let result = ReductionOps::reduce_min_max_f32_optimized(&input).unwrap();

        assert_relative_eq!(result.0, expected.0, epsilon = 1e-6);
        assert_relative_eq!(result.1, expected.1, epsilon = 1e-6);
    }

    #[test]
    fn test_normalize_f32_optimized() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; 5];
        let eps = 1e-8;

        ReductionOps::normalize_f32_optimized(&input, &mut output, eps).unwrap();

        // Check that normalized output has mean ≈ 0 and std ≈ 1
        let sum: f32 = output.iter().sum();
        let mean = sum / output.len() as f32;
        assert_relative_eq!(mean, 0.0, epsilon = 1e-5);

        let variance: f32 =
            output.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / output.len() as f32;
        let std = variance.sqrt();
        assert_relative_eq!(std, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_mean_variance_f32_optimized() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let expected_mean = 5.0;
        let expected_variance = 20.0 / 3.0; // Sample variance

        let (mean, variance) = ReductionOps::mean_variance_f32_optimized(&input).unwrap();

        assert_relative_eq!(mean, expected_mean, epsilon = 1e-6);
        assert_relative_eq!(variance, expected_variance, epsilon = 1e-6);
    }

    #[test]
    fn test_l2_norm_f32_optimized() {
        let input = vec![3.0, 4.0]; // 3-4-5 triangle
        let expected = 5.0;

        let result = ReductionOps::l2_norm_f32_optimized(&input).unwrap();

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_l1_norm_f32_optimized() {
        let input = vec![-2.0, 3.0, -4.0, 1.0];
        let expected = 10.0; // |−2| + |3| + |−4| + |1| = 10

        let result = ReductionOps::l1_norm_f32_optimized(&input).unwrap();

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_argmax_f32_optimized() {
        let input = vec![1.0, 5.0, 3.0, 9.0, 2.0];
        let expected = 3; // Index of maximum value (9.0)

        let result = ReductionOps::argmax_f32_optimized(&input).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_argmin_f32_optimized() {
        let input = vec![5.0, 1.0, 3.0, 9.0, 2.0];
        let expected = 1; // Index of minimum value (1.0)

        let result = ReductionOps::argmin_f32_optimized(&input).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_empty_array_handling() {
        let empty: Vec<f32> = vec![];

        // Test operations that should handle empty arrays
        let sum = ReductionOps::reduce_sum_f32_optimized(&empty).unwrap();
        assert_relative_eq!(sum, 0.0, epsilon = 1e-6);

        // Test operations that should error on empty arrays
        let min_max = ReductionOps::reduce_min_max_f32_optimized(&empty);
        assert!(min_max.is_err());

        let argmax = ReductionOps::argmax_f32_optimized(&empty);
        assert!(argmax.is_err());

        let argmin = ReductionOps::argmin_f32_optimized(&empty);
        assert!(argmin.is_err());
    }

    #[test]
    fn test_reduce_f32_optimized_with_custom_op() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let initial = 1.0;
        let product_op = |acc: f32, x: f32| acc * x;
        let expected = 120.0; // 1 * 1 * 2 * 3 * 4 * 5 = 120

        let result = ReductionOps::reduce_f32_optimized(&input, initial, product_op).unwrap();

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }
}
