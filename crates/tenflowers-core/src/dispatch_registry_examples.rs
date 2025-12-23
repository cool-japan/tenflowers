/// Example integrations showing how to use the dispatch registry
///
/// This module demonstrates best practices for registering operations with
/// the unified dispatch system and provides templates for common operation patterns.
use crate::dispatch_registry::{
    BackendType, BinaryKernelFn, DispatchRegistry, KernelImplementation, OperationDescriptor,
    UnaryKernelFn, F32_REGISTRY, F64_REGISTRY,
};
use crate::{DType, Device, Result, Tensor, TensorError};
use scirs2_core::ndarray::{Array, ArrayD, Zip};
use scirs2_core::random::Random;

/// Initialize all core operation registrations
///
/// This should be called at library initialization to populate the global registries
/// with all supported operations and their backend implementations.
pub fn initialize_dispatch_registrations() {
    // Register unary operations
    register_unary_operations();

    // Register binary operations
    register_binary_operations();

    // Register reduction operations
    register_reduction_operations();
}

/// Register unary operations with all available backends
fn register_unary_operations() {
    // Absolute value operation
    register_abs_op();

    // Negation operation
    register_neg_op();

    // Exponential operation
    register_exp_op();

    // Natural logarithm operation
    register_log_op();

    // Square root operation
    register_sqrt_op();

    // Trigonometric operations
    register_trig_ops();
}

/// Register binary operations with all available backends
fn register_binary_operations() {
    // Addition
    register_add_op();

    // Multiplication
    register_mul_op();

    // Division
    register_div_op();

    // Power
    register_pow_op();
}

/// Register reduction operations
fn register_reduction_operations() {
    // Sum, mean, min, max, etc. would go here
    // These require more complex signatures than simple unary/binary
}

/// Register absolute value operation
fn register_abs_op() {
    // F32 implementation
    {
        let desc = OperationDescriptor::new("abs", "unary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();

        // CPU kernel
        F32_REGISTRY
            .register_kernel(
                "abs",
                KernelImplementation::unary(BackendType::Cpu, abs_f32_cpu),
            )
            .ok();

        // SIMD kernel (if available)
        #[cfg(feature = "simd")]
        {
            F32_REGISTRY
                .register_kernel(
                    "abs",
                    KernelImplementation::unary(BackendType::SimdCpu, abs_f32_simd),
                )
                .ok();
        }

        // GPU kernel (if available)
        #[cfg(feature = "gpu")]
        {
            F32_REGISTRY
                .register_kernel(
                    "abs",
                    KernelImplementation::unary(BackendType::Gpu, abs_f32_gpu),
                )
                .ok();
        }
    }

    // F64 implementation
    {
        let desc = OperationDescriptor::new("abs", "unary")
            .with_dtypes(vec![DType::Float64])
            .with_broadcast();

        F64_REGISTRY.register_operation(desc).ok();

        F64_REGISTRY
            .register_kernel(
                "abs",
                KernelImplementation::unary(BackendType::Cpu, abs_f64_cpu),
            )
            .ok();
    }
}

/// Register negation operation
fn register_neg_op() {
    {
        let desc = OperationDescriptor::new("neg", "unary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();

        F32_REGISTRY
            .register_kernel(
                "neg",
                KernelImplementation::unary(BackendType::Cpu, neg_f32_cpu),
            )
            .ok();
    }
}

/// Register exponential operation
fn register_exp_op() {
    {
        let desc = OperationDescriptor::new("exp", "unary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();

        F32_REGISTRY
            .register_kernel(
                "exp",
                KernelImplementation::unary(BackendType::Cpu, exp_f32_cpu),
            )
            .ok();

        #[cfg(feature = "simd")]
        {
            F32_REGISTRY
                .register_kernel(
                    "exp",
                    KernelImplementation::unary(BackendType::SimdCpu, exp_f32_simd),
                )
                .ok();
        }
    }
}

/// Register natural logarithm operation
fn register_log_op() {
    {
        let desc = OperationDescriptor::new("log", "unary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();

        F32_REGISTRY
            .register_kernel(
                "log",
                KernelImplementation::unary(BackendType::Cpu, log_f32_cpu),
            )
            .ok();
    }
}

/// Register square root operation
fn register_sqrt_op() {
    {
        let desc = OperationDescriptor::new("sqrt", "unary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();

        F32_REGISTRY
            .register_kernel(
                "sqrt",
                KernelImplementation::unary(BackendType::Cpu, sqrt_f32_cpu),
            )
            .ok();
    }
}

/// Register trigonometric operations
fn register_trig_ops() {
    // Sin
    {
        let desc = OperationDescriptor::new("sin", "unary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();

        F32_REGISTRY
            .register_kernel(
                "sin",
                KernelImplementation::unary(BackendType::Cpu, sin_f32_cpu),
            )
            .ok();
    }

    // Cos
    {
        let desc = OperationDescriptor::new("cos", "unary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();

        F32_REGISTRY
            .register_kernel(
                "cos",
                KernelImplementation::unary(BackendType::Cpu, cos_f32_cpu),
            )
            .ok();
    }
}

/// Register addition operation
fn register_add_op() {
    {
        let desc = OperationDescriptor::new("add", "binary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();

        F32_REGISTRY
            .register_kernel(
                "add",
                KernelImplementation::binary(BackendType::Cpu, add_f32_cpu),
            )
            .ok();

        #[cfg(feature = "simd")]
        {
            F32_REGISTRY
                .register_kernel(
                    "add",
                    KernelImplementation::binary(BackendType::SimdCpu, add_f32_simd),
                )
                .ok();
        }

        #[cfg(feature = "gpu")]
        {
            F32_REGISTRY
                .register_kernel(
                    "add",
                    KernelImplementation::binary(BackendType::Gpu, add_f32_gpu),
                )
                .ok();
        }
    }
}

/// Register multiplication operation
fn register_mul_op() {
    {
        let desc = OperationDescriptor::new("mul", "binary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();

        F32_REGISTRY
            .register_kernel(
                "mul",
                KernelImplementation::binary(BackendType::Cpu, mul_f32_cpu),
            )
            .ok();
    }
}

/// Register division operation
fn register_div_op() {
    {
        let desc = OperationDescriptor::new("div", "binary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();

        F32_REGISTRY
            .register_kernel(
                "div",
                KernelImplementation::binary(BackendType::Cpu, div_f32_cpu),
            )
            .ok();
    }
}

/// Register power operation
fn register_pow_op() {
    {
        let desc = OperationDescriptor::new("pow", "binary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();

        F32_REGISTRY
            .register_kernel(
                "pow",
                KernelImplementation::binary(BackendType::Cpu, pow_f32_cpu),
            )
            .ok();
    }
}

// ============================================================================
// Kernel Implementations - CPU
// ============================================================================

fn abs_f32_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = x.data();
    let result: Vec<f32> = data.iter().map(|v| v.abs()).collect();
    let array = ArrayD::from_shape_vec(x.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape mismatch in abs: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn abs_f64_cpu(x: &Tensor<f64>) -> Result<Tensor<f64>> {
    let data = x.data();
    let result: Vec<f64> = data.iter().map(|v| v.abs()).collect();
    let array = ArrayD::from_shape_vec(x.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape mismatch in abs: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn neg_f32_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = x.data();
    let result: Vec<f32> = data.iter().map(|v| -v).collect();
    let array = ArrayD::from_shape_vec(x.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape mismatch in neg: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn exp_f32_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = x.data();
    let result: Vec<f32> = data.iter().map(|v| v.exp()).collect();
    let array = ArrayD::from_shape_vec(x.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape mismatch in exp: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn log_f32_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = x.data();
    let result: Vec<f32> = data.iter().map(|v| v.ln()).collect();
    let array = ArrayD::from_shape_vec(x.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape mismatch in log: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn sqrt_f32_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = x.data();
    let result: Vec<f32> = data.iter().map(|v| v.sqrt()).collect();
    let array = ArrayD::from_shape_vec(x.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape mismatch in sqrt: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn sin_f32_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = x.data();
    let result: Vec<f32> = data.iter().map(|v| v.sin()).collect();
    let array = ArrayD::from_shape_vec(x.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape mismatch in sin: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn cos_f32_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = x.data();
    let result: Vec<f32> = data.iter().map(|v| v.cos()).collect();
    let array = ArrayD::from_shape_vec(x.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape mismatch in cos: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn add_f32_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.shape() != b.shape() {
        return Err(TensorError::shape_mismatch(
            "add",
            &format!("{:?}", a.shape()),
            &format!("{:?}", b.shape()),
        ));
    }
    let a_data = a.data();
    let b_data = b.data();
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x + y)
        .collect();
    let array = ArrayD::from_shape_vec(a.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape mismatch in add: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn mul_f32_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.shape() != b.shape() {
        return Err(TensorError::shape_mismatch(
            "mul",
            &format!("{:?}", a.shape()),
            &format!("{:?}", b.shape()),
        ));
    }
    let a_data = a.data();
    let b_data = b.data();
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x * y)
        .collect();
    let array = ArrayD::from_shape_vec(a.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape mismatch in mul: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn div_f32_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.shape() != b.shape() {
        return Err(TensorError::shape_mismatch(
            "div",
            &format!("{:?}", a.shape()),
            &format!("{:?}", b.shape()),
        ));
    }
    let a_data = a.data();
    let b_data = b.data();
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x / y)
        .collect();
    let array = ArrayD::from_shape_vec(a.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape mismatch in div: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn pow_f32_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.shape() != b.shape() {
        return Err(TensorError::shape_mismatch(
            "pow",
            &format!("{:?}", a.shape()),
            &format!("{:?}", b.shape()),
        ));
    }
    let a_data = a.data();
    let b_data = b.data();
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x.powf(*y))
        .collect();
    let array = ArrayD::from_shape_vec(a.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape mismatch in pow: {}", e)))?;
    Ok(Tensor::from_array(array))
}

// ============================================================================
// Kernel Implementations - SIMD
// ============================================================================

#[cfg(feature = "simd")]
fn abs_f32_simd(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    // Use scirs2_core SIMD operations if available
    // For now, fallback to CPU implementation
    abs_f32_cpu(x)
}

#[cfg(feature = "simd")]
fn exp_f32_simd(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    // Use scirs2_core SIMD operations if available
    exp_f32_cpu(x)
}

#[cfg(feature = "simd")]
fn add_f32_simd(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    // Use scirs2_core SIMD operations if available
    add_f32_cpu(a, b)
}

// ============================================================================
// Kernel Implementations - GPU
// ============================================================================

#[cfg(feature = "gpu")]
fn abs_f32_gpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    // Use GPU kernels if available, otherwise fallback
    abs_f32_cpu(x) // TODO: Implement actual GPU kernel
}

#[cfg(feature = "gpu")]
fn add_f32_gpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    // Use GPU kernels if available, otherwise fallback
    add_f32_cpu(a, b) // TODO: Implement actual GPU kernel
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_registration_initialization() {
        initialize_dispatch_registrations();

        // Verify operations are registered
        assert!(F32_REGISTRY.get_operation("abs").is_some());
        assert!(F32_REGISTRY.get_operation("add").is_some());
        assert!(F32_REGISTRY.get_operation("mul").is_some());
    }

    #[test]
    fn test_dispatch_abs() {
        initialize_dispatch_registrations();

        let input = Tensor::from_array(array![-1.0f32, 2.0, -3.0, 4.0].into_dyn());
        let result = F32_REGISTRY.dispatch_unary("abs", &input).unwrap();

        assert_eq!(result.data(), &[1.0f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_dispatch_add() {
        initialize_dispatch_registrations();

        let a = Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn());
        let b = Tensor::from_array(array![4.0f32, 5.0, 6.0].into_dyn());
        let result = F32_REGISTRY.dispatch_binary("add", &a, &b).unwrap();

        assert_eq!(result.data(), &[5.0f32, 7.0, 9.0]);
    }

    #[test]
    fn test_backend_selection() {
        initialize_dispatch_registrations();

        let backends = F32_REGISTRY.available_backends("add");
        assert!(!backends.is_empty());
        assert!(backends.contains(&BackendType::Cpu));
    }
}
