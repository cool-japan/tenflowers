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

    match (&a.storage, &b.storage) {
        (TensorStorage::Cpu(_), TensorStorage::Cpu(_)) => {
            matmul_batch(&a.storage, &b.storage, &result_shape)
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(_), TensorStorage::Gpu(_)) => {
            // The raw-buffer-level `matmul_batch_gpu` cannot correctly
            // implement batch matmul yet: it only receives `TensorStorage`/
            // output shape, not the full `&Tensor` with real per-batch shape
            // context. Read both operands back to the host here (where the
            // full `&Tensor<T>` is still available) and delegate to the
            // already-correct CPU implementation.
            let cpu_a = a.to_cpu()?;
            let cpu_b = b.to_cpu()?;
            matmul_batch(&cpu_a.storage, &cpu_b.storage, &result_shape)
        }
        #[cfg(feature = "gpu")]
        _ => matmul_batch(&a.storage, &b.storage, &result_shape),
    }
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
                let a_view = a_arr
                    .view()
                    .into_dimensionality::<IxDyn>()
                    .expect("tensor must be convertible to dynamic dimensionality");
                let b_view = b_arr
                    .view()
                    .into_dimensionality::<IxDyn>()
                    .expect("tensor must be convertible to dynamic dimensionality");

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
                // No native GPU dot-product kernel exists yet. Read both
                // operands back to the host (a real device->host transfer)
                // and delegate to the CPU implementation above, which is
                // known-correct.
                let cpu_a = a.to_cpu()?;
                let cpu_b = b.to_cpu()?;
                dot(&cpu_a, &cpu_b)
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

// GPU-resident correctness tests for the readback+delegate fixes in this
// file: `dot()`'s (Gpu,Gpu) arm and `batch_matmul()`'s GPU branch. A GPU
// adapter is not guaranteed to be present in every environment that builds
// with `--features gpu`; each test attempts the host->device transfer and
// skips its assertions (without failing the suite) if no adapter is
// available, mirroring the convention in `ops/einsum/gpu.rs`.
#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use super::*;
    use crate::Device;

    #[test]
    fn gpu_dot_matches_cpu_reference() {
        let a_cpu = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])
            .expect("test: from_vec should succeed");
        let b_cpu = Tensor::<f32>::from_vec(vec![4.0, 5.0, 6.0], &[3])
            .expect("test: from_vec should succeed");

        let (a_gpu, b_gpu) = match (a_cpu.to(Device::Gpu(0)), b_cpu.to(Device::Gpu(0))) {
            (Ok(a), Ok(b)) => (a, b),
            _ => return, // No GPU adapter available in this environment; skip.
        };

        let result = dot(&a_gpu, &b_gpu).expect("test: gpu dot should succeed with a real adapter");
        let data = result.to_vec().expect("test: to_vec should succeed");
        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(data, vec![32.0]);
    }

    #[test]
    fn gpu_batch_matmul_matches_cpu_reference() {
        // batch 0: [[1,2],[3,4]] @ [[5,6],[7,8]]    = [[19,22],[43,50]]
        // batch 1: [[1,0],[0,1]] @ [[9,10],[11,12]] = [[9,10],[11,12]]
        let a_cpu =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2])
                .expect("test: from_vec should succeed");
        let b_cpu =
            Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[2, 2, 2])
                .expect("test: from_vec should succeed");

        let (a_gpu, b_gpu) = match (a_cpu.to(Device::Gpu(0)), b_cpu.to(Device::Gpu(0))) {
            (Ok(a), Ok(b)) => (a, b),
            _ => return, // No GPU adapter available in this environment; skip.
        };

        let result = batch_matmul(&a_gpu, &b_gpu)
            .expect("test: gpu batch_matmul should succeed with a real adapter");
        assert_eq!(result.shape().dims(), &[2, 2, 2]);
        let data = result.to_vec().expect("test: to_vec should succeed");
        assert_eq!(data, vec![19.0, 22.0, 43.0, 50.0, 9.0, 10.0, 11.0, 12.0]);
    }
}
