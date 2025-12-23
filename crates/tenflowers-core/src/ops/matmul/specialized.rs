//! Specialized Matrix Operations
//!
//! This module contains specialized matrix operations including
//! outer products, mixed precision operations, and advanced matrix functions.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::ndarray::{Array2, ArrayD, ArrayView2, IxDyn};
use scirs2_core::numeric::{Float, Zero};

/// Mixed precision matrix multiplication with automatic precision management
pub fn matmul_mixed_precision<T>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    precision_mode: MixedPrecisionMode,
) -> Result<Tensor<T>>
where
    T: Clone
        + Float
        + std::fmt::Debug
        + Default
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + bytemuck::Pod,
{
    match (&a.storage, &b.storage) {
        (TensorStorage::Cpu(a_arr), TensorStorage::Cpu(b_arr)) => {
            // For CPU, implement mixed precision using different algorithms
            // based on the precision requirements
            match precision_mode {
                MixedPrecisionMode::HighPrecision => {
                    // Use high-precision algorithm with careful accumulation
                    matmul_high_precision(a_arr.view(), b_arr.view())
                }
                MixedPrecisionMode::Balanced => {
                    // Use standard precision
                    super::core::matmul(a, b)
                }
                MixedPrecisionMode::Fast => {
                    // Use faster, lower precision algorithm
                    matmul_fast_precision(a_arr.view(), b_arr.view())
                }
            }
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(_), TensorStorage::Gpu(_)) => {
            // Delegate to GPU mixed precision implementation
            super::gpu::matmul_mixed_precision_gpu(a, b, precision_mode == MixedPrecisionMode::Fast)
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::invalid_operation_simple(
            "Device mismatch: both tensors must be on the same device".to_string(),
        )),
    }
}

/// Outer product of two tensors
pub fn outer<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Zero + std::ops::Mul<Output = T> + Default + Send + Sync + 'static,
{
    let a_shape = a.shape().dims();
    let b_shape = b.shape().dims();

    // Outer product is only defined for 1D tensors
    if a_shape.len() != 1 || b_shape.len() != 1 {
        return Err(TensorError::invalid_shape_simple(
            "Outer product requires 1D tensors".to_string(),
        ));
    }

    match (&a.storage, &b.storage) {
        (TensorStorage::Cpu(a_arr), TensorStorage::Cpu(b_arr)) => {
            let a_data: Vec<T> = a_arr.iter().cloned().collect();
            let b_data: Vec<T> = b_arr.iter().cloned().collect();

            let result_shape = vec![a_shape[0], b_shape[0]];
            let mut result_data = Vec::with_capacity(result_shape[0] * result_shape[1]);

            for a_val in &a_data {
                for b_val in &b_data {
                    result_data.push(a_val.clone() * b_val.clone());
                }
            }

            let result_arr = ArrayD::from_shape_vec(IxDyn(&result_shape), result_data)
                .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;

            Ok(Tensor::from_array(result_arr))
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(_), TensorStorage::Gpu(_)) => {
            Err(TensorError::unsupported_operation_simple(
                "GPU outer product not yet implemented".to_string(),
            ))
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::invalid_operation_simple(
            "Device mismatch: both tensors must be on the same device".to_string(),
        )),
    }
}

/// Outer product matrix multiplication for better cache performance
pub fn matmul_outer_product<T>(a: ArrayView2<T>, b: ArrayView2<T>) -> Array2<T>
where
    T: Clone + Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Default,
{
    let (m, k) = a.dim();
    let (_, n) = b.dim();

    let mut result = Array2::<T>::zeros((m, n));

    // Use outer product formulation: C = sum over k of (a[:, k] * b[k, :])
    for k_idx in 0..k {
        let a_col = a.column(k_idx);
        let b_row = b.row(k_idx);

        for i in 0..m {
            let a_val = a_col[i].clone();
            for j in 0..n {
                result[[i, j]] = result[[i, j]].clone() + (a_val.clone() * b_row[j].clone());
            }
        }
    }

    result
}

/// High-precision matrix multiplication with extended precision accumulation
fn matmul_high_precision<T>(
    a: scirs2_core::ndarray::ArrayView<T, IxDyn>,
    b: scirs2_core::ndarray::ArrayView<T, IxDyn>,
) -> Result<Tensor<T>>
where
    T: Clone + Float + Default + Send + Sync + 'static,
{
    // Convert to 2D views
    let a_2d = a
        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
        .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;
    let b_2d = b
        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
        .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;

    let (m, k) = a_2d.dim();
    let (_, n) = b_2d.dim();

    let mut result = Array2::zeros((m, n));

    // Use Kahan summation for better numerical stability
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            let mut compensation = T::zero();

            for k_idx in 0..k {
                let product = a_2d[[i, k_idx]] * b_2d[[k_idx, j]];
                let y = product - compensation;
                let t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            }
            result[[i, j]] = sum;
        }
    }

    Ok(Tensor::from_array(result.into_dyn()))
}

/// Fast, lower precision matrix multiplication
fn matmul_fast_precision<T>(
    a: scirs2_core::ndarray::ArrayView<T, IxDyn>,
    b: scirs2_core::ndarray::ArrayView<T, IxDyn>,
) -> Result<Tensor<T>>
where
    T: Clone + Float + Default + Send + Sync + 'static,
{
    // Use the standard optimized implementation for fast precision
    let a_2d = a
        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
        .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;
    let b_2d = b
        .into_dimensionality::<scirs2_core::ndarray::Ix2>()
        .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;

    let result = super::optimized::matmul_simple_optimized(a_2d, b_2d);
    Ok(Tensor::from_array(result.into_dyn()))
}

/// Mixed precision modes for matrix multiplication
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MixedPrecisionMode {
    /// High precision with extended accumulation
    HighPrecision,
    /// Balanced precision (standard)
    Balanced,
    /// Fast computation with potentially lower precision
    Fast,
}

#[cfg(test)]
#[allow(irrefutable_let_patterns)] // Pattern matching on TensorStorage is irrefutable when GPU feature is disabled
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_outer_product() {
        let a = Tensor::from_array(array![1.0, 2.0, 3.0].into_dyn());
        let b = Tensor::from_array(array![4.0, 5.0].into_dyn());

        let result = outer(&a, &b).unwrap();
        let expected_data = array![[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]];

        // Extract result data for comparison
        if let TensorStorage::Cpu(result_arr) = &result.storage {
            let result_2d = result_arr
                .view()
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .unwrap();
            assert_eq!(result_2d, expected_data);
        } else {
            panic!("Expected CPU tensor");
        }
    }

    #[test]
    fn test_outer_product_error() {
        let a = Tensor::from_array(array![[1.0, 2.0], [3.0, 4.0]].into_dyn());
        let b = Tensor::from_array(array![4.0, 5.0].into_dyn());

        let result = outer(&a, &b);
        assert!(result.is_err());
    }
}
