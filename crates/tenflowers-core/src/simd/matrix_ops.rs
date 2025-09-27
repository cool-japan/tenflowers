//! SIMD-Optimized Matrix Operations
//!
//! This module provides high-performance matrix operations including matrix multiplication,
//! dot products, and matrix-vector operations using SIMD optimizations and cache-friendly algorithms.

use crate::error::ErrorContext;
use crate::{Result, TensorError};

/// SIMD-optimized matrix operations
pub struct MatrixOps;

impl MatrixOps {
    /// Advanced cache-friendly matrix multiplication using blocking
    pub fn matmul_f32_blocked(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        block_size: usize,
    ) -> Result<()> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD blocked matmul".to_string(),
                expected: format!("matrices A({m}x{k}), B({k}x{n}), C({m}x{n})"),
                got: format!("A: {}, B: {}, C: {}", a.len(), b.len(), c.len()),
                context: Some(ErrorContext::new()),
            });
        }

        // Initialize result matrix to zero
        for elem in c.iter_mut() {
            *elem = 0.0;
        }

        // Block-wise multiplication for better cache locality
        for i0 in (0..m).step_by(block_size) {
            for j0 in (0..n).step_by(block_size) {
                for k0 in (0..k).step_by(block_size) {
                    // Process block
                    let i_max = (i0 + block_size).min(m);
                    let j_max = (j0 + block_size).min(n);
                    let k_max = (k0 + block_size).min(k);

                    for i in i0..i_max {
                        for j in j0..j_max {
                            let mut sum = 0.0f32;

                            // Unrolled inner loop for vectorization
                            let mut kk = k0;
                            while kk + 4 <= k_max {
                                sum += a[i * k + kk] * b[kk * n + j];
                                sum += a[i * k + kk + 1] * b[(kk + 1) * n + j];
                                sum += a[i * k + kk + 2] * b[(kk + 2) * n + j];
                                sum += a[i * k + kk + 3] * b[(kk + 3) * n + j];
                                kk += 4;
                            }

                            // Handle remaining elements
                            while kk < k_max {
                                sum += a[i * k + kk] * b[kk * n + j];
                                kk += 1;
                            }

                            c[i * n + j] += sum;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Optimized dot product for f32 vectors using Kahan summation
    pub fn dot_product_f32_optimized(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD dot product".to_string(),
                expected: format!("vectors of length {}", a.len()),
                got: format!("a: {}, b: {}", a.len(), b.len()),
                context: Some(ErrorContext::new()),
            });
        }

        let len = a.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        let mut sum = 0.0f32;
        let mut compensation = 0.0f32; // Kahan summation for better numerical stability

        // Process in unrolled chunks
        for i in 0..main_loops {
            let base = i * unroll_size;

            // Compute partial products
            let mut chunk_sum = 0.0f32;
            chunk_sum += a[base] * b[base];
            chunk_sum += a[base + 1] * b[base + 1];
            chunk_sum += a[base + 2] * b[base + 2];
            chunk_sum += a[base + 3] * b[base + 3];
            chunk_sum += a[base + 4] * b[base + 4];
            chunk_sum += a[base + 5] * b[base + 5];
            chunk_sum += a[base + 6] * b[base + 6];
            chunk_sum += a[base + 7] * b[base + 7];

            // Kahan summation
            let y = chunk_sum - compensation;
            let t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }

        // Handle remaining elements
        for i in (main_loops * unroll_size)..len {
            let y = (a[i] * b[i]) - compensation;
            let t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }

        Ok(sum)
    }

    /// Ultra-high-performance matrix-vector multiplication with SIMD
    #[inline(always)]
    pub fn matvec_f32_unchecked(
        matrix: &[f32],
        vector: &[f32],
        result: &mut [f32],
        rows: usize,
        cols: usize,
    ) {
        // Clear result vector
        result[..rows].fill(0.0);

        // Optimized matrix-vector multiplication
        #[allow(clippy::needless_range_loop)]
        // Performance-critical matrix operations need indexing
        for i in 0..rows {
            let row_start = i * cols;
            let mut sum = 0.0f32;

            // Use vectorized loop with unrolling
            let unroll_size = 8;
            let main_loops = cols / unroll_size;

            // Process 8 elements at a time
            for j in 0..main_loops {
                let base = j * unroll_size;
                sum += matrix[row_start + base] * vector[base];
                sum += matrix[row_start + base + 1] * vector[base + 1];
                sum += matrix[row_start + base + 2] * vector[base + 2];
                sum += matrix[row_start + base + 3] * vector[base + 3];
                sum += matrix[row_start + base + 4] * vector[base + 4];
                sum += matrix[row_start + base + 5] * vector[base + 5];
                sum += matrix[row_start + base + 6] * vector[base + 6];
                sum += matrix[row_start + base + 7] * vector[base + 7];
            }

            // Handle remaining elements
            for j in (main_loops * unroll_size)..cols {
                sum += matrix[row_start + j] * vector[j];
            }

            result[i] = sum;
        }
    }

    /// Safe matrix-vector multiplication with error checking
    pub fn matvec_f32_optimized(
        matrix: &[f32],
        vector: &[f32],
        result: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if matrix.len() != rows * cols {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD matvec".to_string(),
                expected: format!("matrix with {} elements", rows * cols),
                got: format!("matrix with {} elements", matrix.len()),
                context: Some(ErrorContext::new()),
            });
        }

        if vector.len() != cols {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD matvec".to_string(),
                expected: format!("vector with {} elements", cols),
                got: format!("vector with {} elements", vector.len()),
                context: Some(ErrorContext::new()),
            });
        }

        if result.len() != rows {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD matvec".to_string(),
                expected: format!("result vector with {} elements", rows),
                got: format!("result vector with {} elements", result.len()),
                context: Some(ErrorContext::new()),
            });
        }

        Self::matvec_f32_unchecked(matrix, vector, result, rows, cols);
        Ok(())
    }

    /// Optimized matrix transpose with cache-friendly blocking
    pub fn transpose_f32_blocked(
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
        block_size: usize,
    ) -> Result<()> {
        if input.len() != rows * cols || output.len() != rows * cols {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD transpose".to_string(),
                expected: format!("matrices with {} elements", rows * cols),
                got: format!("input: {}, output: {}", input.len(), output.len()),
                context: Some(ErrorContext::new()),
            });
        }

        // Block-wise transpose for better cache locality
        for i0 in (0..rows).step_by(block_size) {
            for j0 in (0..cols).step_by(block_size) {
                let i_max = (i0 + block_size).min(rows);
                let j_max = (j0 + block_size).min(cols);

                // Transpose within the block
                for i in i0..i_max {
                    for j in j0..j_max {
                        output[j * rows + i] = input[i * cols + j];
                    }
                }
            }
        }

        Ok(())
    }

    /// Optimized outer product computation
    pub fn outer_product_f32_optimized(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        let m = a.len();
        let n = b.len();

        if result.len() != m * n {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD outer product".to_string(),
                expected: format!("result matrix with {} elements", m * n),
                got: format!("result matrix with {} elements", result.len()),
                context: Some(ErrorContext::new()),
            });
        }

        // Compute outer product: result[i,j] = a[i] * b[j]
        for (i, &a_val) in a.iter().enumerate() {
            let row_start = i * n;
            let unroll_size = 8;
            let main_loops = n / unroll_size;

            // Unrolled computation for better vectorization
            for j in 0..main_loops {
                let base = j * unroll_size;
                result[row_start + base] = a_val * b[base];
                result[row_start + base + 1] = a_val * b[base + 1];
                result[row_start + base + 2] = a_val * b[base + 2];
                result[row_start + base + 3] = a_val * b[base + 3];
                result[row_start + base + 4] = a_val * b[base + 4];
                result[row_start + base + 5] = a_val * b[base + 5];
                result[row_start + base + 6] = a_val * b[base + 6];
                result[row_start + base + 7] = a_val * b[base + 7];
            }

            // Handle remaining elements
            for j in (main_loops * unroll_size)..n {
                result[row_start + j] = a_val * b[j];
            }
        }

        Ok(())
    }

    /// Fast matrix addition with SIMD optimization
    pub fn matrix_add_f32_optimized(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD matrix add".to_string(),
                expected: format!("matrices of length {}", a.len()),
                got: format!("a: {}, b: {}, result: {}", a.len(), b.len(), result.len()),
                context: Some(ErrorContext::new()),
            });
        }

        // Use the basic_ops add function for element-wise addition
        super::basic_ops::BasicOps::add_f32_unchecked(a, b, result);

        Ok(())
    }

    /// Optimized matrix scaling (scalar multiplication)
    pub fn matrix_scale_f32_optimized(
        matrix: &[f32],
        scalar: f32,
        result: &mut [f32],
    ) -> Result<()> {
        if matrix.len() != result.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD matrix scale".to_string(),
                expected: format!("matrices of length {}", matrix.len()),
                got: format!("matrix: {}, result: {}", matrix.len(), result.len()),
                context: Some(ErrorContext::new()),
            });
        }

        let len = matrix.len();
        let unroll_size = 8;
        let main_loops = len / unroll_size;

        // Unrolled scalar multiplication
        for i in 0..main_loops {
            let base = i * unroll_size;
            result[base] = matrix[base] * scalar;
            result[base + 1] = matrix[base + 1] * scalar;
            result[base + 2] = matrix[base + 2] * scalar;
            result[base + 3] = matrix[base + 3] * scalar;
            result[base + 4] = matrix[base + 4] * scalar;
            result[base + 5] = matrix[base + 5] * scalar;
            result[base + 6] = matrix[base + 6] * scalar;
            result[base + 7] = matrix[base + 7] * scalar;
        }

        // Handle remaining elements
        for i in (main_loops * unroll_size)..len {
            result[i] = matrix[i] * scalar;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dot_product_f32_optimized() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let expected = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0; // 40.0

        let result = MatrixOps::dot_product_f32_optimized(&a, &b).unwrap();

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_matmul_f32_blocked() {
        // Test 2x2 matrix multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1, 2], [3, 4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5, 6], [7, 8]]
        let mut c = vec![0.0; 4];
        let expected = vec![19.0, 22.0, 43.0, 50.0]; // [[19, 22], [43, 50]]

        MatrixOps::matmul_f32_blocked(&a, &b, &mut c, 2, 2, 2, 2).unwrap();

        for (i, &val) in c.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_matvec_f32_optimized() {
        // Test 2x3 matrix times 3x1 vector
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1, 2, 3], [4, 5, 6]]
        let vector = vec![1.0, 2.0, 3.0];
        let mut result = vec![0.0; 2];
        let expected = vec![14.0, 32.0]; // [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3]

        MatrixOps::matvec_f32_optimized(&matrix, &vector, &mut result, 2, 3).unwrap();

        for (i, &val) in result.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_transpose_f32_blocked() {
        // Test 2x3 matrix transpose to 3x2
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1, 2, 3], [4, 5, 6]]
        let mut output = vec![0.0; 6];
        let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // [[1, 4], [2, 5], [3, 6]]

        MatrixOps::transpose_f32_blocked(&input, &mut output, 2, 3, 2).unwrap();

        for (i, &val) in output.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_outer_product_f32_optimized() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0];
        let mut result = vec![0.0; 6];
        let expected = vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]; // [[3, 4, 5], [6, 8, 10]]

        MatrixOps::outer_product_f32_optimized(&a, &b, &mut result).unwrap();

        for (i, &val) in result.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_matrix_add_f32_optimized() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 4];
        let expected = vec![6.0, 8.0, 10.0, 12.0];

        MatrixOps::matrix_add_f32_optimized(&a, &b, &mut result).unwrap();

        for (i, &val) in result.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_matrix_scale_f32_optimized() {
        let matrix = vec![1.0, 2.0, 3.0, 4.0];
        let scalar = 2.5;
        let mut result = vec![0.0; 4];
        let expected = vec![2.5, 5.0, 7.5, 10.0];

        MatrixOps::matrix_scale_f32_optimized(&matrix, scalar, &mut result).unwrap();

        for (i, &val) in result.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_matrix_ops_error_handling() {
        // Test dot product size mismatch
        let a = vec![1.0; 3];
        let b = vec![1.0; 4];
        let error = MatrixOps::dot_product_f32_optimized(&a, &b);
        assert!(error.is_err());

        // Test matmul size mismatch
        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let mut c = vec![0.0; 3]; // Wrong size
        let error = MatrixOps::matmul_f32_blocked(&a, &b, &mut c, 2, 2, 2, 2);
        assert!(error.is_err());

        // Test matvec size mismatch
        let matrix = vec![1.0; 6];
        let vector = vec![1.0; 2]; // Wrong size for 2x3 matrix
        let mut result = vec![0.0; 2];
        let error = MatrixOps::matvec_f32_optimized(&matrix, &vector, &mut result, 2, 3);
        assert!(error.is_err());
    }
}
