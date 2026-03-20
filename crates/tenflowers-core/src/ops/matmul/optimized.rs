//! CPU-Optimized Matrix Multiplication Implementations
//!
//! This module contains various optimized CPU implementations including
//! BLAS integration, blocked algorithms, and cache-friendly variants.

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::numeric::Zero;

/// Optimized 2D matrix multiplication dispatcher
pub fn matmul_2d_optimized<T>(a: ArrayView2<T>, b: ArrayView2<T>) -> Array2<T>
where
    T: Clone + Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Default,
{
    let (m, k) = a.dim();
    let (_, n) = b.dim();

    // For small matrices, use optimized simple implementation
    if m <= 64 && n <= 64 && k <= 64 {
        matmul_simple_optimized(a, b)
    } else {
        // For larger matrices, use blocked implementation
        matmul_blocked(a, b)
    }
}

/// BLAS-optimized matrix multiplication for f32
#[cfg(feature = "blas")]
pub fn matmul_blas_f32<T>(a: ArrayView2<T>, b: ArrayView2<T>) -> Array2<T>
where
    T: Clone + Default + Into<f32> + From<f32>,
{
    use ndarray_linalg::*;

    // Convert to f32 for BLAS operations
    let a_f32: Array2<f32> = a.mapv(|x| x.clone().into());
    let b_f32: Array2<f32> = b.mapv(|x| x.clone().into());

    // Use BLAS general matrix multiply
    let result_f32 = a_f32.dot(&b_f32);

    // Convert back to original type
    result_f32.mapv(|x| x.into())
}

/// BLAS-optimized matrix multiplication for f64
#[cfg(feature = "blas")]
pub fn matmul_blas_f64<T>(a: ArrayView2<T>, b: ArrayView2<T>) -> Array2<T>
where
    T: Clone + Default + Into<f64> + From<f64>,
{
    use ndarray_linalg::*;

    // Convert to f64 for BLAS operations
    let a_f64: Array2<f64> = a.mapv(|x| x.clone().into());
    let b_f64: Array2<f64> = b.mapv(|x| x.clone().into());

    // Use BLAS general matrix multiply
    let result_f64 = a_f64.dot(&b_f64);

    // Convert back to original type
    result_f64.mapv(|x| x.into())
}

/// Blocked matrix multiplication for better cache performance
pub fn matmul_blocked<T>(a: ArrayView2<T>, b: ArrayView2<T>) -> Array2<T>
where
    T: Clone + Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Default,
{
    let (m, k) = a.dim();
    let (_, n) = b.dim();

    const BLOCK_SIZE: usize = 64; // Optimized for L1 cache

    let mut result = Array2::<T>::zeros((m, n));

    // Blocked algorithm - improves cache locality
    for i_block in (0..m).step_by(BLOCK_SIZE) {
        for j_block in (0..n).step_by(BLOCK_SIZE) {
            for k_block in (0..k).step_by(BLOCK_SIZE) {
                let i_end = (i_block + BLOCK_SIZE).min(m);
                let j_end = (j_block + BLOCK_SIZE).min(n);
                let k_end = (k_block + BLOCK_SIZE).min(k);

                for i in i_block..i_end {
                    for j in j_block..j_end {
                        let mut sum = result[[i, j]].clone();
                        for k_idx in k_block..k_end {
                            sum = sum + (a[[i, k_idx]].clone() * b[[k_idx, j]].clone());
                        }
                        result[[i, j]] = sum;
                    }
                }
            }
        }
    }

    result
}

/// Simple optimized matrix multiplication for small matrices
pub fn matmul_simple_optimized<T>(a: ArrayView2<T>, b: ArrayView2<T>) -> Array2<T>
where
    T: Clone + Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Default,
{
    let (m, k) = a.dim();
    let (_, n) = b.dim();

    let mut result = Array2::zeros((m, n));

    // Simple IJK order for small matrices - better cache locality than IKJ
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for k_idx in 0..k {
                sum = sum + (a[[i, k_idx]].clone() * b[[k_idx, j]].clone());
            }
            result[[i, j]] = sum;
        }
    }

    result
}

/// Cache-optimized matrix multiplication
pub fn matmul_cache_optimized<T>(a: &Array2<T>, b: &Array2<T>) -> Array2<T>
where
    T: Clone + Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Default,
{
    let (m, k) = a.dim();
    let (_, n) = b.dim();

    // Fallback to cache-friendly implementation for other types
    // Use blocked implementation for large matrices, simple cache-optimized for small ones
    if m > 128 || n > 128 || k > 128 {
        // Use blocked implementation for better cache locality on large matrices
        matmul_blocked_optimized(a, b, 64)
    } else {
        // For smaller matrices, simple algorithm with good cache behavior
        let mut result = Array2::<T>::zeros((m, n));

        for i in 0..m {
            for k_idx in 0..k {
                let a_ik = a[[i, k_idx]].clone();
                for j in 0..n {
                    result[[i, j]] =
                        result[[i, j]].clone() + (a_ik.clone() * b[[k_idx, j]].clone());
                }
            }
        }

        result
    }
}

/// Blocked matrix multiplication with configurable block size
pub fn matmul_blocked_optimized<T>(a: &Array2<T>, b: &Array2<T>, block_size: usize) -> Array2<T>
where
    T: Clone + Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Default,
{
    let (m, k) = a.dim();
    let (_, n) = b.dim();

    let mut result = Array2::<T>::zeros((m, n));

    for i in (0..m).step_by(block_size) {
        for j in (0..n).step_by(block_size) {
            for k_idx in (0..k).step_by(block_size) {
                let i_end = (i + block_size).min(m);
                let j_end = (j + block_size).min(n);
                let k_end = (k_idx + block_size).min(k);

                // Multiply blocks
                for ii in i..i_end {
                    for jj in j..j_end {
                        let mut sum = result[[ii, jj]].clone();
                        for kk in k_idx..k_end {
                            sum = sum + (a[[ii, kk]].clone() * b[[kk, jj]].clone());
                        }
                        result[[ii, jj]] = sum;
                    }
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_matmul_simple_optimized() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let result = matmul_simple_optimized(a.view(), b.view());
        let expected = array![[19.0, 22.0], [43.0, 50.0]];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_blocked() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];

        let result = matmul_blocked(a.view(), b.view());
        let expected = array![[58.0, 64.0], [139.0, 154.0]];

        assert_eq!(result, expected);
    }
}
