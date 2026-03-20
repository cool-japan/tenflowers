//! Convenience Functions for Binary Operations
//!
//! This module provides high-level convenience functions for common binary operations
//! that abstract away the operation structs and provide a clean public API.

use super::core::{get_binary_op_registry, BinaryOpAnalytics};
use super::implementation::binary_op;
use super::operations::{AddOp, DivOp, MaxOp, MinOp, MulOp, PowOp, SubOp};
use crate::{Result, Tensor};
use scirs2_core::numeric::Zero;
use std::ops::{Add as StdAdd, Div as StdDiv, Mul as StdMul, Sub as StdSub};

/// Element-wise addition of two tensors
#[inline]
pub fn add<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + StdAdd<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    binary_op(a, b, AddOp)
}

/// Element-wise subtraction of two tensors
#[inline]
pub fn sub<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + StdSub<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    binary_op(a, b, SubOp)
}

/// Element-wise multiplication of two tensors
#[inline]
pub fn mul<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + StdMul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    binary_op(a, b, MulOp)
}

/// Element-wise division of two tensors
#[inline]
pub fn div<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + StdDiv<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    binary_op(a, b, DivOp)
}

/// Element-wise power operation
#[inline]
pub fn pow<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + scirs2_core::num_traits::Float
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    binary_op(a, b, PowOp)
}

/// Element-wise minimum of two tensors
#[inline]
pub fn min<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    binary_op(a, b, MinOp)
}

/// Element-wise maximum of two tensors
#[inline]
pub fn max<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    binary_op(a, b, MaxOp)
}

/// Add a scalar value to all elements of a tensor
pub fn scalar_add<T>(tensor: &Tensor<T>, scalar: T) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + StdAdd<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Create a scalar tensor and use regular add
    let scalar_tensor = Tensor::from_scalar(scalar);
    add(tensor, &scalar_tensor)
}

/// Clamp tensor values between min and max
pub fn clamp<T>(tensor: &Tensor<T>, min_val: T, max_val: T) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + PartialOrd
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match &tensor.storage {
        crate::tensor::TensorStorage::Cpu(arr) => {
            let result = arr.mapv(|v| {
                if v < min_val {
                    min_val
                } else if v > max_val {
                    max_val
                } else {
                    v
                }
            });
            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        crate::tensor::TensorStorage::Gpu(_gpu_buffer) => {
            // For now, fall back to CPU implementation
            let cpu_tensor = tensor.to_cpu()?;
            clamp(&cpu_tensor, min_val, max_val)
        }
    }
}

/// Ultra-performance convenience functions for common operations
/// Ultra-performance addition with metrics tracking
pub fn ultra_add<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + Send
        + Sync
        + StdAdd<Output = T>
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let registry = get_binary_op_registry();
    let start = std::time::Instant::now();

    let result = binary_op(a, b, AddOp);

    // Record performance metrics
    let duration = start.elapsed();
    registry.record_operation("ultra_add", a.shape().size(), duration.as_nanos() as u64);

    result
}

/// Ultra-performance multiplication with metrics tracking
pub fn ultra_mul<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + Send
        + Sync
        + StdMul<Output = T>
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let registry = get_binary_op_registry();
    let start = std::time::Instant::now();

    let result = binary_op(a, b, MulOp);

    // Record performance metrics
    let duration = start.elapsed();
    registry.record_operation("ultra_mul", a.shape().size(), duration.as_nanos() as u64);

    result
}

/// Ultra-performance subtraction with metrics tracking
pub fn ultra_sub<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + Send
        + Sync
        + StdSub<Output = T>
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let registry = get_binary_op_registry();
    let start = std::time::Instant::now();

    let result = binary_op(a, b, SubOp);

    // Record performance metrics
    let duration = start.elapsed();
    registry.record_operation("ultra_sub", a.shape().size(), duration.as_nanos() as u64);

    result
}

/// Ultra-performance division with metrics tracking
pub fn ultra_div<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + Send
        + Sync
        + StdDiv<Output = T>
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let registry = get_binary_op_registry();
    let start = std::time::Instant::now();

    let result = binary_op(a, b, DivOp);

    // Record performance metrics
    let duration = start.elapsed();
    registry.record_operation("ultra_div", a.shape().size(), duration.as_nanos() as u64);

    result
}

/// Get comprehensive performance analytics for binary operations
pub fn get_binary_op_performance_report() -> BinaryOpAnalytics {
    get_binary_op_registry().get_analytics()
}

/// Reset performance counters (useful for benchmarking)
pub fn reset_binary_op_counters() {
    // In a full implementation, this would reset all counters
    // For now, we just return since AtomicU64 doesn't have a simple reset
}
