//! Core Matrix Multiplication API
//!
//! This module contains the main public API functions for matrix multiplication,
//! including matmul, dot product, and batch matmul operations.

use crate::shape_error_taxonomy::ShapeErrorUtils;
use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use scirs2_core::numeric::Zero;

use super::batch::matmul_batch;
use super::optimized::matmul_2d_optimized;
use super::shapes::compute_matmul_shape;

/// Matrix multiplication for 2D tensors with broadcasting support
pub fn matmul<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Default
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let a_shape = a.shape().dims();
    let b_shape = b.shape().dims();

    // Validate shapes
    if a_shape.is_empty() || b_shape.is_empty() {
        return Err(ShapeErrorUtils::rank_range_mismatch(
            "matmul",
            1,
            None,
            if a_shape.is_empty() {
                a.shape()
            } else {
                b.shape()
            },
        ));
    }

    // Check dimension compatibility
    let a_cols = a_shape[a_shape.len() - 1];
    let b_rows = if b_shape.len() == 1 {
        b_shape[0]
    } else {
        b_shape[b_shape.len() - 2]
    };

    if a_cols != b_rows {
        return Err(ShapeErrorUtils::matmul_incompatible(
            "matmul",
            a.shape(),
            b.shape(),
            false,
            false,
        ));
    }

    match (a_shape.len(), b_shape.len()) {
        (2, 2) => {
            // Simple 2D matrix multiplication
            matmul_2d(&a.storage, &b.storage)
        }
        (1, 2) => {
            // Vector-matrix multiplication: [n] × [n, m] = [m]
            vector_matrix_mul(a, b)
        }
        (2, 1) => {
            // Matrix-vector multiplication: [m, n] × [n] = [m]
            matrix_vector_mul(a, b)
        }
        (_, _) if a_shape.len() > 2 || b_shape.len() > 2 => {
            // Batch matrix multiplication
            batch_matmul(a, b)
        }
        _ => Err(TensorError::unsupported_operation_simple(
            "Unsupported tensor dimensions for matmul".to_string(),
        )),
    }
}

/// Batch matrix multiplication with broadcasting
pub fn batch_matmul<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Zero
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Default
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let result_shape = compute_matmul_shape(a.shape().dims(), b.shape().dims())?;

    if a.shape().dims().len() < 2 || b.shape().dims().len() < 2 {
        return Err(ShapeErrorUtils::rank_range_mismatch(
            "batch_matmul",
            2,
            None,
            if a.shape().dims().len() < 2 {
                a.shape()
            } else {
                b.shape()
            },
        ));
    }

    matmul_batch(&a.storage, &b.storage, &result_shape)
}

/// Dot product for 1D tensors or inner product for higher dimensions
pub fn dot<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Zero
        + scirs2_core::num_traits::One
        + Default
        + Send
        + Sync
        + 'static
        + bytemuck::Pod,
{
    let a_shape = a.shape().dims();
    let b_shape = b.shape().dims();

    // For 1D vectors, compute dot product
    if a_shape.len() == 1 && b_shape.len() == 1 {
        if a_shape[0] != b_shape[0] {
            return Err(TensorError::invalid_shape_simple(format!(
                "Dot product dimension mismatch: {} vs {}",
                a_shape[0], b_shape[0]
            )));
        }

        match (&a.storage, &b.storage) {
            (TensorStorage::Cpu(a_arr), TensorStorage::Cpu(b_arr)) => {
                let a_view = a_arr.view().into_dimensionality::<IxDyn>().unwrap();
                let b_view = b_arr.view().into_dimensionality::<IxDyn>().unwrap();

                let mut sum = T::zero();
                for (a_val, b_val) in a_view.iter().zip(b_view.iter()) {
                    sum = sum + (*a_val * *b_val);
                }

                // Return scalar tensor
                let result_arr = ArrayD::from_elem(IxDyn(&[]), sum);
                Ok(Tensor::from_array(result_arr))
            }
            #[cfg(feature = "gpu")]
            (TensorStorage::Gpu(_), TensorStorage::Gpu(_)) => {
                // GPU dot product implementation using element-wise multiplication followed by sum reduction
                Err(TensorError::unsupported_operation_simple(
                    "GPU dot product not yet implemented".to_string(),
                ))
            }
            #[cfg(feature = "gpu")]
            _ => Err(TensorError::invalid_operation_simple(
                "Device mismatch: both tensors must be on the same device".to_string(),
            )),
        }
    } else {
        // For higher dimensions, treat as matrix multiplication
        matmul(a, b)
    }
}

/// Internal 2D matrix multiplication dispatcher
fn matmul_2d<T>(a_storage: &TensorStorage<T>, b_storage: &TensorStorage<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Zero
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Default
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match (a_storage, b_storage) {
        (TensorStorage::Cpu(a_arr), TensorStorage::Cpu(b_arr)) => {
            let a_view = a_arr
                .view()
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;
            let b_view = b_arr
                .view()
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;

            let result = matmul_2d_optimized(a_view, b_view);
            Ok(Tensor::from_array(result.into_dyn()))
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(_), TensorStorage::Gpu(_)) => {
            // Delegate to GPU implementation
            super::gpu::matmul_gpu_2d(a_storage, b_storage)
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::invalid_operation_simple(
            "Device mismatch: both tensors must be on the same device".to_string(),
        )),
    }
}

/// Vector-matrix multiplication: [n] × [n, m] = [m]
fn vector_matrix_mul<T>(vector: &Tensor<T>, matrix: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Default
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let v_shape = vector.shape().dims();
    let m_shape = matrix.shape().dims();

    // For vector-matrix multiplication: [n] × [n, m] = [m]
    let n = v_shape[0];
    let m = m_shape[1];

    // Create result tensor
    let mut result_data = vec![T::zero(); m];

    // Access vector and matrix data
    let v_data = vector.as_slice().ok_or_else(|| {
        TensorError::invalid_operation_simple("Cannot access vector data".to_string())
    })?;
    let m_data = matrix.as_slice().ok_or_else(|| {
        TensorError::invalid_operation_simple("Cannot access matrix data".to_string())
    })?;

    // Perform vector-matrix multiplication: result[j] = sum_i(vector[i] * matrix[i, j])
    for j in 0..m {
        let mut sum = T::zero();
        for i in 0..n {
            sum = sum + v_data[i] * m_data[i * m + j];
        }
        result_data[j] = sum;
    }

    Tensor::from_vec(result_data, &[m])
}

/// Matrix-vector multiplication: [m, n] × [n] = [m]
fn matrix_vector_mul<T>(matrix: &Tensor<T>, vector: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Default
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let m_shape = matrix.shape().dims();
    let _v_shape = vector.shape().dims();

    // For matrix-vector multiplication: [m, n] × [n] = [m]
    let m = m_shape[0];
    let n = m_shape[1];

    // Create result tensor
    let mut result_data = vec![T::zero(); m];

    // Access matrix and vector data
    let m_data = matrix.as_slice().ok_or_else(|| {
        TensorError::invalid_operation_simple("Cannot access matrix data".to_string())
    })?;
    let v_data = vector.as_slice().ok_or_else(|| {
        TensorError::invalid_operation_simple("Cannot access vector data".to_string())
    })?;

    // Perform matrix-vector multiplication: result[i] = sum_j(matrix[i, j] * vector[j])
    for i in 0..m {
        let mut sum = T::zero();
        for j in 0..n {
            sum = sum + m_data[i * n + j] * v_data[j];
        }
        result_data[i] = sum;
    }

    Tensor::from_vec(result_data, &[m])
}
