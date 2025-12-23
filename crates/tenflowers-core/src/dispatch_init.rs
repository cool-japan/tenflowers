/// Dispatch Registry Initialization Module
///
/// This module handles initialization of all operation registrations for the
/// unified dispatch system. It should be called once at library initialization.
///
/// The initialization is performed lazily using lazy_static to ensure it happens
/// exactly once before any dispatch calls are made.
use crate::dispatch_registry_examples::initialize_dispatch_registrations;
use crate::dispatch_registry_extended::initialize_extended_registrations;

use lazy_static::lazy_static;

lazy_static! {
    /// Ensures dispatch registry is initialized before first use
    pub static ref DISPATCH_INIT: () = {
        initialize_dispatch_registrations();
        initialize_extended_registrations();
        #[cfg(feature = "gpu")]
        initialize_gpu_operations();
    };
}

/// Initialize GPU-specific operations
#[cfg(feature = "gpu")]
fn initialize_gpu_operations() {
    // GPU operations will be registered here
    // This includes sum, mean, and other reduction operations with GPU backends
    register_gpu_reductions();
}

/// Register GPU reduction operations
#[cfg(feature = "gpu")]
fn register_gpu_reductions() {
    use crate::dispatch_registry::{
        BackendType, KernelImplementation, OperationDescriptor, F32_REGISTRY,
    };
    use crate::DType;

    // Sum operation for f32
    {
        let desc = OperationDescriptor::new("sum", "reduction").with_dtypes(vec![DType::Float32]);

        // Only register if not already registered (from examples)
        if F32_REGISTRY.get_operation("sum").is_none() {
            F32_REGISTRY.register_operation(desc).ok();
        }

        // Register GPU kernel
        F32_REGISTRY
            .register_kernel(
                "sum",
                KernelImplementation::unary(BackendType::Gpu, sum_f32_gpu),
            )
            .ok();
    }

    // Mean operation for f32
    {
        let desc = OperationDescriptor::new("mean", "reduction").with_dtypes(vec![DType::Float32]);

        if F32_REGISTRY.get_operation("mean").is_none() {
            F32_REGISTRY.register_operation(desc).ok();
        }

        // Register GPU kernel
        F32_REGISTRY
            .register_kernel(
                "mean",
                KernelImplementation::unary(BackendType::Gpu, mean_f32_gpu),
            )
            .ok();
    }
}

/// GPU implementation of sum for f32
#[cfg(feature = "gpu")]
fn sum_f32_gpu(x: &crate::Tensor<f32>) -> crate::Result<crate::Tensor<f32>> {
    use crate::gpu::buffer::GpuBuffer;
    use crate::gpu::ops::operation_types::ReductionOp;
    use crate::gpu::ops::reduction_ops::execute_reduction_op;
    use crate::Device;
    use scirs2_core::ndarray::Array;

    // Get tensor data as slice
    let slice = x.data();

    // Create GPU buffer from slice
    let gpu_buffer = GpuBuffer::from_slice(slice, &Device::Gpu(0))?;

    // Execute GPU reduction
    let result_buffer = execute_reduction_op(&gpu_buffer, ReductionOp::Sum, None)?;

    // Read back result from GPU
    let result_data = result_buffer.to_cpu()?;

    // Convert to tensor
    let result = Array::from_elem(vec![], result_data[0]).into_dyn();
    Ok(crate::Tensor::from_array(result))
}

/// GPU implementation of mean for f32
#[cfg(feature = "gpu")]
fn mean_f32_gpu(x: &crate::Tensor<f32>) -> crate::Result<crate::Tensor<f32>> {
    use crate::gpu::buffer::GpuBuffer;
    use crate::gpu::ops::operation_types::ReductionOp;
    use crate::gpu::ops::reduction_ops::execute_reduction_op;
    use crate::Device;
    use scirs2_core::ndarray::Array;

    // Get tensor data as slice
    let slice = x.data();

    // Create GPU buffer from slice
    let gpu_buffer = GpuBuffer::from_slice(slice, &Device::Gpu(0))?;

    // Execute GPU reduction
    let result_buffer = execute_reduction_op(&gpu_buffer, ReductionOp::Mean, None)?;

    // Read back result from GPU
    let result_data = result_buffer.to_cpu()?;

    // Convert to tensor
    let result = Array::from_elem(vec![], result_data[0]).into_dyn();
    Ok(crate::Tensor::from_array(result))
}

/// CPU fallback for sum
fn sum_f32_cpu(x: &crate::Tensor<f32>) -> crate::Result<crate::Tensor<f32>> {
    use scirs2_core::ndarray::Array;

    let data = x.data();
    let sum: f32 = data.iter().sum();
    let result = Array::from_elem(vec![], sum).into_dyn();
    Ok(crate::Tensor::from_array(result))
}

/// CPU fallback for mean
fn mean_f32_cpu(x: &crate::Tensor<f32>) -> crate::Result<crate::Tensor<f32>> {
    use scirs2_core::ndarray::Array;

    let data = x.data();
    let sum: f32 = data.iter().sum();
    let mean = sum / data.len() as f32;
    let result = Array::from_elem(vec![], mean).into_dyn();
    Ok(crate::Tensor::from_array(result))
}

/// Ensure dispatch registry is initialized
///
/// This function can be called explicitly to ensure initialization has occurred.
/// It's automatically called on first access to global registries, but can be
/// called early for better control over initialization timing.
pub fn ensure_initialized() {
    *DISPATCH_INIT;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_initialization() {
        ensure_initialized();

        use crate::dispatch_registry::F32_REGISTRY;

        // Verify operations are registered
        assert!(F32_REGISTRY.get_operation("add").is_some());
        assert!(F32_REGISTRY.get_operation("mul").is_some());
        assert!(F32_REGISTRY.get_operation("div").is_some());
    }

    #[test]
    fn test_sum_cpu_fallback() {
        let input = Tensor::from_array(array![1.0f32, 2.0, 3.0, 4.0, 5.0].into_dyn());
        let result = sum_f32_cpu(&input).unwrap();

        // Sum should be 15.0
        assert_eq!(result.data()[0], 15.0);
    }

    #[test]
    fn test_mean_cpu_fallback() {
        let input = Tensor::from_array(array![1.0f32, 2.0, 3.0, 4.0, 5.0].into_dyn());
        let result = mean_f32_cpu(&input).unwrap();

        // Mean should be 3.0
        assert_eq!(result.data()[0], 3.0);
    }
}
