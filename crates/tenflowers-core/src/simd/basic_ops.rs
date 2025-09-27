//! Basic SIMD Operations
//!
//! This module provides optimized element-wise operations using SIMD instructions
//! for fundamental tensor operations like addition, multiplication, and fused multiply-add.

use crate::error::ErrorContext;
use crate::{Result, TensorError};

/// Basic SIMD-optimized element-wise operations
pub struct BasicOps;

impl BasicOps {
    /// Fast inline element-wise addition without bounds checking (for hot paths)
    #[inline(always)]
    pub fn add_f32_unchecked(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();

        // For very small arrays, use simple vectorizable loop
        if len < 32 {
            for i in 0..len {
                unsafe {
                    *result.get_unchecked_mut(i) = a.get_unchecked(i) + b.get_unchecked(i);
                }
            }
            return;
        }

        // For larger arrays, use optimal chunk size for SIMD (8 elements = 256 bits for AVX)
        let chunks = len / 8;
        let remainder = len % 8;

        // Process 8 elements at a time - optimal for most SIMD instruction sets
        for chunk in 0..chunks {
            let base = chunk * 8;

            // Compact unrolling that's more likely to vectorize efficiently
            unsafe {
                *result.get_unchecked_mut(base) = a.get_unchecked(base) + b.get_unchecked(base);
                *result.get_unchecked_mut(base + 1) =
                    a.get_unchecked(base + 1) + b.get_unchecked(base + 1);
                *result.get_unchecked_mut(base + 2) =
                    a.get_unchecked(base + 2) + b.get_unchecked(base + 2);
                *result.get_unchecked_mut(base + 3) =
                    a.get_unchecked(base + 3) + b.get_unchecked(base + 3);
                *result.get_unchecked_mut(base + 4) =
                    a.get_unchecked(base + 4) + b.get_unchecked(base + 4);
                *result.get_unchecked_mut(base + 5) =
                    a.get_unchecked(base + 5) + b.get_unchecked(base + 5);
                *result.get_unchecked_mut(base + 6) =
                    a.get_unchecked(base + 6) + b.get_unchecked(base + 6);
                *result.get_unchecked_mut(base + 7) =
                    a.get_unchecked(base + 7) + b.get_unchecked(base + 7);
            }
        }

        // Handle remaining elements
        let remainder_start = chunks * 8;
        for i in 0..remainder {
            unsafe {
                *result.get_unchecked_mut(remainder_start + i) =
                    a.get_unchecked(remainder_start + i) + b.get_unchecked(remainder_start + i);
            }
        }
    }

    /// Element-wise addition with SciRS2-Core SIMD auto-vectorization (safe version)
    pub fn add_f32_optimized(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD add_f32".to_string(),
                expected: format!("arrays of length {}", a.len()),
                got: format!("a: {}, b: {}, result: {}", a.len(), b.len(), result.len()),
                context: Some(ErrorContext::new()),
            });
        }

        // Use basic SIMD when available (advanced SIMD features pending in scirs2_core)
        #[cfg(feature = "simd")]
        {
            // For now, use basic vectorization - advanced SIMD will be added when available
            if a.len() >= 64 {
                // Fallback to optimized manual implementation
            }
        }

        Self::add_f32_unchecked(a, b, result);
        Ok(())
    }

    /// Fast inline element-wise multiplication without bounds checking (for hot paths)
    #[inline(always)]
    pub fn mul_f32_unchecked(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();

        // For very small arrays, use simple vectorizable loop
        if len < 32 {
            for i in 0..len {
                unsafe {
                    *result.get_unchecked_mut(i) = a.get_unchecked(i) * b.get_unchecked(i);
                }
            }
            return;
        }

        // For larger arrays, use optimal chunk size for SIMD (8 elements = 256 bits for AVX)
        let chunks = len / 8;
        let remainder = len % 8;

        // Process 8 elements at a time - optimal for most SIMD instruction sets
        for chunk in 0..chunks {
            let base = chunk * 8;

            unsafe {
                *result.get_unchecked_mut(base) = a.get_unchecked(base) * b.get_unchecked(base);
                *result.get_unchecked_mut(base + 1) =
                    a.get_unchecked(base + 1) * b.get_unchecked(base + 1);
                *result.get_unchecked_mut(base + 2) =
                    a.get_unchecked(base + 2) * b.get_unchecked(base + 2);
                *result.get_unchecked_mut(base + 3) =
                    a.get_unchecked(base + 3) * b.get_unchecked(base + 3);
                *result.get_unchecked_mut(base + 4) =
                    a.get_unchecked(base + 4) * b.get_unchecked(base + 4);
                *result.get_unchecked_mut(base + 5) =
                    a.get_unchecked(base + 5) * b.get_unchecked(base + 5);
                *result.get_unchecked_mut(base + 6) =
                    a.get_unchecked(base + 6) * b.get_unchecked(base + 6);
                *result.get_unchecked_mut(base + 7) =
                    a.get_unchecked(base + 7) * b.get_unchecked(base + 7);
            }
        }

        // Handle remaining elements
        let remainder_start = chunks * 8;
        for i in 0..remainder {
            unsafe {
                *result.get_unchecked_mut(remainder_start + i) =
                    a.get_unchecked(remainder_start + i) * b.get_unchecked(remainder_start + i);
            }
        }
    }

    /// Element-wise multiplication with optimization hints for f32 tensors (safe version)
    pub fn mul_f32_optimized(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD mul_f32".to_string(),
                expected: format!("arrays of length {}", a.len()),
                got: format!("a: {}, b: {}, result: {}", a.len(), b.len(), result.len()),
                context: Some(ErrorContext::new()),
            });
        }

        Self::mul_f32_unchecked(a, b, result);
        Ok(())
    }

    /// Fast inline element-wise subtraction without bounds checking (for hot paths)
    #[inline(always)]
    pub fn sub_f32_unchecked(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        // For very small arrays, use simple vectorizable loop
        if len < 32 {
            for i in 0..len {
                unsafe {
                    *result.get_unchecked_mut(i) = a.get_unchecked(i) - b.get_unchecked(i);
                }
            }
            return;
        }
        // For larger arrays, use optimal chunk size for SIMD (8 elements = 256 bits for AVX)
        let chunks = len / 8;
        let remainder = len % 8;
        // Process 8 elements at a time - optimal for most SIMD instruction sets
        for chunk in 0..chunks {
            let base = chunk * 8;
            unsafe {
                *result.get_unchecked_mut(base) = a.get_unchecked(base) - b.get_unchecked(base);
                *result.get_unchecked_mut(base + 1) =
                    a.get_unchecked(base + 1) - b.get_unchecked(base + 1);
                *result.get_unchecked_mut(base + 2) =
                    a.get_unchecked(base + 2) - b.get_unchecked(base + 2);
                *result.get_unchecked_mut(base + 3) =
                    a.get_unchecked(base + 3) - b.get_unchecked(base + 3);
                *result.get_unchecked_mut(base + 4) =
                    a.get_unchecked(base + 4) - b.get_unchecked(base + 4);
                *result.get_unchecked_mut(base + 5) =
                    a.get_unchecked(base + 5) - b.get_unchecked(base + 5);
                *result.get_unchecked_mut(base + 6) =
                    a.get_unchecked(base + 6) - b.get_unchecked(base + 6);
                *result.get_unchecked_mut(base + 7) =
                    a.get_unchecked(base + 7) - b.get_unchecked(base + 7);
            }
        }
        // Handle remaining elements
        let remainder_start = chunks * 8;
        for i in 0..remainder {
            unsafe {
                *result.get_unchecked_mut(remainder_start + i) =
                    a.get_unchecked(remainder_start + i) - b.get_unchecked(remainder_start + i);
            }
        }
    }

    /// Element-wise subtraction with optimization hints for f32 tensors (safe version)
    pub fn sub_f32_optimized(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "SIMD sub_f32".to_string(),
                expected: format!("arrays of length {}", a.len()),
                got: format!("a: {}, b: {}, result: {}", a.len(), b.len(), result.len()),
                context: Some(ErrorContext::new()),
            });
        }
        Self::sub_f32_unchecked(a, b, result);
        Ok(())
    }

    /// Ultra-high-performance fused multiply-add operation (a * b + c)
    #[inline(always)]
    pub fn fma_f32_unchecked(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        let len = a.len();

        // Process elements in chunks of 8 for optimal SIMD performance
        let chunk_size = 8;
        let full_chunks = len / chunk_size;

        // Main processing loop with unrolling for SIMD optimization
        for chunk in 0..full_chunks {
            let base = chunk * chunk_size;

            // Manually unrolled FMA operations for better vectorization
            unsafe {
                // Load and compute chunk
                *result.get_unchecked_mut(base) =
                    a.get_unchecked(base) * b.get_unchecked(base) + c.get_unchecked(base);
                *result.get_unchecked_mut(base + 1) = a.get_unchecked(base + 1)
                    * b.get_unchecked(base + 1)
                    + c.get_unchecked(base + 1);
                *result.get_unchecked_mut(base + 2) = a.get_unchecked(base + 2)
                    * b.get_unchecked(base + 2)
                    + c.get_unchecked(base + 2);
                *result.get_unchecked_mut(base + 3) = a.get_unchecked(base + 3)
                    * b.get_unchecked(base + 3)
                    + c.get_unchecked(base + 3);
                *result.get_unchecked_mut(base + 4) = a.get_unchecked(base + 4)
                    * b.get_unchecked(base + 4)
                    + c.get_unchecked(base + 4);
                *result.get_unchecked_mut(base + 5) = a.get_unchecked(base + 5)
                    * b.get_unchecked(base + 5)
                    + c.get_unchecked(base + 5);
                *result.get_unchecked_mut(base + 6) = a.get_unchecked(base + 6)
                    * b.get_unchecked(base + 6)
                    + c.get_unchecked(base + 6);
                *result.get_unchecked_mut(base + 7) = a.get_unchecked(base + 7)
                    * b.get_unchecked(base + 7)
                    + c.get_unchecked(base + 7);
            }
        }

        // Handle remaining elements
        let remainder_start = full_chunks * chunk_size;
        for i in remainder_start..len {
            unsafe {
                *result.get_unchecked_mut(i) =
                    a.get_unchecked(i) * b.get_unchecked(i) + c.get_unchecked(i);
            }
        }
    }

    /// Batch vector operations with memory-friendly access patterns
    pub fn batch_add_f32(batches: &[(&[f32], &[f32])], results: &mut [&mut [f32]]) -> Result<()> {
        if batches.len() != results.len() {
            return Err(TensorError::ShapeMismatch {
                operation: "Batch SIMD add".to_string(),
                expected: format!("{} result arrays", batches.len()),
                got: format!("{} result arrays", results.len()),
                context: Some(ErrorContext::new()),
            });
        }

        for (i, ((a, b), result)) in batches.iter().zip(results.iter_mut()).enumerate() {
            if a.len() != b.len() || a.len() != result.len() {
                return Err(TensorError::ShapeMismatch {
                    operation: format!("Batch SIMD add (batch {})", i),
                    expected: format!("arrays of length {}", a.len()),
                    got: format!("a: {}, b: {}, result: {}", a.len(), b.len(), result.len()),
                    context: Some(ErrorContext::new()),
                });
            }

            Self::add_f32_unchecked(a, b, result);
        }

        Ok(())
    }

    /// Auto-select the best implementation based on input characteristics
    pub fn add_f32_auto(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        let size = a.len();

        if size < 64 {
            // Use simple implementation for small arrays to avoid overhead
            if a.len() != b.len() || a.len() != result.len() {
                return Err(TensorError::ShapeMismatch {
                    operation: "Simple add_f32".to_string(),
                    expected: format!("arrays of length {}", a.len()),
                    got: format!("a: {}, b: {}, result: {}", a.len(), b.len(), result.len()),
                    context: Some(ErrorContext::new()),
                });
            }

            for i in 0..size {
                result[i] = a[i] + b[i];
            }
        } else {
            // Use optimized implementation for larger arrays
            Self::add_f32_optimized(a, b, result)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_add_f32_unchecked() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let mut result = vec![0.0; 10];
        let expected = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0];

        BasicOps::add_f32_unchecked(&a, &b, &mut result);

        for (i, &val) in result.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_add_f32_optimized() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let mut result = vec![0.0; 10];
        let expected = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0];

        BasicOps::add_f32_optimized(&a, &b, &mut result).unwrap();

        for (i, &val) in result.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_mul_f32_optimized() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut result = vec![0.0; 8];
        let expected = vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0];

        BasicOps::mul_f32_optimized(&a, &b, &mut result).unwrap();

        for (i, &val) in result.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_fma_f32_unchecked() {
        let a = vec![2.0, 3.0, 4.0, 5.0];
        let b = vec![3.0, 4.0, 5.0, 6.0];
        let c = vec![1.0, 2.0, 3.0, 4.0];
        let mut result = vec![0.0; 4];
        let expected = vec![7.0, 14.0, 23.0, 34.0]; // a*b+c

        BasicOps::fma_f32_unchecked(&a, &b, &c, &mut result);

        for (i, &val) in result.iter().enumerate() {
            assert_relative_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_shape_mismatch_errors() {
        let a = vec![1.0; 5];
        let b = vec![1.0; 3]; // Wrong size
        let mut result = vec![0.0; 5];

        let error = BasicOps::add_f32_optimized(&a, &b, &mut result);
        assert!(error.is_err());

        let error = BasicOps::mul_f32_optimized(&a, &b, &mut result);
        assert!(error.is_err());
    }
}
