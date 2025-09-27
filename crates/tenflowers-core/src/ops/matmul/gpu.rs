//! GPU-Accelerated Matrix Multiplication
//!
//! This module provides GPU implementations for matrix multiplication
//! using WGPU compute shaders for cross-platform GPU acceleration.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};

/// GPU-accelerated 2D matrix multiplication
#[cfg(feature = "gpu")]
pub fn matmul_gpu_2d<T>(a: &TensorStorage<T>, b: &TensorStorage<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    match (a, b) {
        (TensorStorage::Gpu(a_buffer), TensorStorage::Gpu(b_buffer)) => {
            use crate::gpu::{kernel_fusion::FusedOperation, GpuContext};
            use wgpu::util::DeviceExt;

            // Get global GPU context
            let context = GpuContext::global()?;

            // For now, use a basic implementation
            // TODO: Implement optimized GPU matrix multiplication kernel
            Err(TensorError::unsupported_operation_simple(
                "GPU matrix multiplication not yet fully implemented".to_string(),
            ))
        }
        _ => Err(TensorError::invalid_operation_simple(
            "GPU matmul requires both tensors to be on GPU".to_string(),
        )),
    }
}

/// GPU-accelerated batch matrix multiplication
#[cfg(feature = "gpu")]
pub fn matmul_batch_gpu<T>(
    a: &TensorStorage<T>,
    b: &TensorStorage<T>,
    result_shape: &[usize],
) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    match (a, b) {
        (TensorStorage::Gpu(_a_buffer), TensorStorage::Gpu(_b_buffer)) => {
            // TODO: Implement GPU batch matrix multiplication
            Err(TensorError::unsupported_operation_simple(
                "GPU batch matrix multiplication not yet implemented".to_string(),
            ))
        }
        _ => Err(TensorError::invalid_operation_simple(
            "GPU batch matmul requires both tensors to be on GPU".to_string(),
        )),
    }
}

/// GPU matrix multiplication with mixed precision support
#[cfg(feature = "gpu")]
pub fn matmul_mixed_precision_gpu<T>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    use_tf32: bool,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Send + Sync + 'static,
{
    // TODO: Implement mixed precision GPU matrix multiplication
    // This would use Tensor Cores when available for accelerated computation
    Err(TensorError::unsupported_operation_simple(
        "GPU mixed precision matmul not yet implemented".to_string(),
    ))
}

// Fallback implementations when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
pub fn matmul_gpu_2d<T>(_a: &TensorStorage<T>, _b: &TensorStorage<T>) -> Result<Tensor<T>>
where
    T: Clone,
{
    Err(TensorError::unsupported_operation_simple(
        "GPU support not compiled in".to_string(),
    ))
}

#[cfg(not(feature = "gpu"))]
pub fn matmul_batch_gpu<T>(
    _a: &TensorStorage<T>,
    _b: &TensorStorage<T>,
    _result_shape: &[usize],
) -> Result<Tensor<T>>
where
    T: Clone,
{
    Err(TensorError::unsupported_operation_simple(
        "GPU support not compiled in".to_string(),
    ))
}
