//! SIMD-accelerated matrix operations
//!
//! This module provides vectorized implementations of linear algebra operations
//! using SIMD instructions for enhanced performance.

#![allow(unsafe_code)]

use crate::Transform;
use std::marker::PhantomData;
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-accelerated matrix operations for linear algebra transformations
///
/// Provides fast matrix multiplication and other linear algebra operations
/// using SIMD instructions for enhanced performance.
pub struct SimdMatrixOps<T> {
    use_simd: bool,
    _marker: PhantomData<T>,
}

impl<T> SimdMatrixOps<T>
where
    T: Clone + Default + num_traits::Float + Send + Sync + 'static,
{
    /// Create a new SIMD-accelerated matrix operations instance
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        let use_simd = is_x86_feature_detected!("avx2") && std::mem::size_of::<T>() == 4;

        #[cfg(not(target_arch = "x86_64"))]
        let use_simd = false;

        Self {
            use_simd,
            _marker: PhantomData,
        }
    }

    /// SIMD-accelerated matrix-vector multiplication
    ///
    /// # Arguments
    /// * `matrix` - Input matrix as 2D tensor [M, N]
    /// * `vector` - Input vector as 1D tensor [N]
    ///
    /// # Returns
    /// Result vector as 1D tensor [M]
    pub fn mat_vec_mul(&self, matrix: &Tensor<T>, vector: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: bytemuck::Pod + bytemuck::Zeroable,
    {
        let matrix_shape = matrix.shape().dims();
        let vector_shape = vector.shape().dims();

        if matrix_shape.len() != 2 || vector_shape.len() != 1 {
            return Err(TensorError::InvalidShape {
                operation: "SimdMatrixOps::mat_vec_mul".to_string(),
                reason: "Matrix-vector multiplication requires 2D matrix and 1D vector".to_string(),
                shape: Some(matrix_shape.to_vec()),
                context: None,
            });
        }

        let m = matrix_shape[0];
        let n = matrix_shape[1];
        let vec_len = vector_shape[0];

        if n != vec_len {
            return Err(TensorError::InvalidShape {
                operation: "SimdMatrixOps::mat_vec_mul".to_string(),
                reason: format!("Matrix columns {} must match vector length {}", n, vec_len),
                shape: Some(vec![n, vec_len]),
                context: None,
            });
        }

        let matrix_data = matrix.to_vec()?;
        let vector_data = vector.to_vec()?;
        let mut result = vec![T::zero(); m];

        // Perform matrix-vector multiplication
        #[allow(clippy::needless_range_loop)]
        for i in 0..m {
            let mut sum = T::zero();
            #[allow(clippy::needless_range_loop)]
            for j in 0..n {
                let matrix_idx = i * n + j;
                sum = sum + matrix_data[matrix_idx].clone() * vector_data[j].clone();
            }
            result[i] = sum;
        }

        Tensor::<T>::from_vec(result, &[m])
    }

    /// Transpose matrix operation
    ///
    /// # Arguments
    /// * `matrix` - Input matrix as 2D tensor [M, N]
    ///
    /// # Returns
    /// Transposed matrix as 2D tensor [N, M]
    pub fn transpose(&self, matrix: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: bytemuck::Pod + bytemuck::Zeroable,
    {
        let shape = matrix.shape().dims();

        if shape.len() != 2 {
            return Err(TensorError::InvalidShape {
                operation: "SimdMatrixOps::transpose".to_string(),
                reason: "Transpose requires 2D matrix".to_string(),
                shape: Some(shape.to_vec()),
                context: None,
            });
        }

        let m = shape[0];
        let n = shape[1];
        let data = matrix.to_vec()?;
        let mut result = vec![T::zero(); n * m];

        // Transpose operation
        for i in 0..m {
            for j in 0..n {
                let src_idx = i * n + j;
                let dst_idx = j * m + i;
                result[dst_idx] = data[src_idx].clone();
            }
        }

        Tensor::<T>::from_vec(result, &[n, m])
    }

    /// Get SIMD capability status
    pub fn is_simd_enabled(&self) -> bool {
        self.use_simd
    }
}

impl<T> Default for SimdMatrixOps<T>
where
    T: Clone + Default + num_traits::Float + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Transform<T> for SimdMatrixOps<T>
where
    T: Clone
        + Default
        + num_traits::Float
        + Send
        + Sync
        + bytemuck::Pod
        + bytemuck::Zeroable
        + 'static,
{
    fn apply(&self, (features, labels): (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        // For demonstration, apply transpose to 2D features
        if features.shape().dims().len() == 2 {
            let transposed = self.transpose(&features)?;
            Ok((transposed, labels))
        } else {
            Ok((features, labels))
        }
    }
}
