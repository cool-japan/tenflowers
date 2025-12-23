/// Extended Operation Registrations for Dispatch Registry
///
/// This module extends the basic operations with additional mathematical,
/// comparison, and reduction operations to provide comprehensive coverage.
use crate::dispatch_registry::{
    BackendType, BinaryKernelFn, DispatchRegistry, KernelImplementation, OperationDescriptor,
    UnaryKernelFn, F32_REGISTRY, F64_REGISTRY,
};
use crate::{DType, Device, Result, Tensor, TensorError};
use scirs2_core::ndarray::{Array, ArrayD, Zip};

/// Initialize extended operation registrations
pub fn initialize_extended_registrations() {
    register_extended_unary_ops();
    register_extended_binary_ops();
    register_comparison_ops();
    register_extended_reduction_ops();
}

/// Register extended unary operations
fn register_extended_unary_ops() {
    // Tanh
    register_tanh();

    // Sigmoid
    register_sigmoid();

    // ReLU
    register_relu();

    // Sign
    register_sign();

    // Reciprocal
    register_reciprocal();

    // Square
    register_square();
}

/// Register extended binary operations
fn register_extended_binary_ops() {
    // Subtraction (if not already registered)
    register_sub();

    // Remainder/Modulo
    register_remainder();

    // Minimum element-wise
    register_minimum();

    // Maximum element-wise
    register_maximum();

    // Atan2
    register_atan2();
}

/// Register comparison operations
fn register_comparison_ops() {
    register_equal();
    register_not_equal();
    register_less_than();
    register_less_equal();
    register_greater_than();
    register_greater_equal();
}

/// Register extended reduction operations
fn register_extended_reduction_ops() {
    register_prod();
    register_all();
    register_any();
}

// ============================================================================
// Unary Operation Registrations
// ============================================================================

fn register_tanh() {
    {
        let desc = OperationDescriptor::new("tanh", "unary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "tanh",
                KernelImplementation::unary(BackendType::Cpu, tanh_f32_cpu),
            )
            .ok();
    }
}

fn register_sigmoid() {
    {
        let desc = OperationDescriptor::new("sigmoid", "unary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "sigmoid",
                KernelImplementation::unary(BackendType::Cpu, sigmoid_f32_cpu),
            )
            .ok();
    }
}

fn register_relu() {
    {
        let desc = OperationDescriptor::new("relu", "unary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "relu",
                KernelImplementation::unary(BackendType::Cpu, relu_f32_cpu),
            )
            .ok();
    }
}

fn register_sign() {
    {
        let desc = OperationDescriptor::new("sign", "unary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "sign",
                KernelImplementation::unary(BackendType::Cpu, sign_f32_cpu),
            )
            .ok();
    }
}

fn register_reciprocal() {
    {
        let desc = OperationDescriptor::new("reciprocal", "unary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "reciprocal",
                KernelImplementation::unary(BackendType::Cpu, reciprocal_f32_cpu),
            )
            .ok();
    }
}

fn register_square() {
    {
        let desc = OperationDescriptor::new("square", "unary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "square",
                KernelImplementation::unary(BackendType::Cpu, square_f32_cpu),
            )
            .ok();
    }
}

// ============================================================================
// Binary Operation Registrations
// ============================================================================

fn register_sub() {
    {
        let desc = OperationDescriptor::new("sub", "binary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        // Only register if not already registered
        if F32_REGISTRY.get_operation("sub").is_none() {
            F32_REGISTRY.register_operation(desc).ok();
        }

        F32_REGISTRY
            .register_kernel(
                "sub",
                KernelImplementation::binary(BackendType::Cpu, sub_f32_cpu),
            )
            .ok();
    }
}

fn register_remainder() {
    {
        let desc = OperationDescriptor::new("remainder", "binary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "remainder",
                KernelImplementation::binary(BackendType::Cpu, remainder_f32_cpu),
            )
            .ok();
    }
}

fn register_minimum() {
    {
        let desc = OperationDescriptor::new("minimum", "binary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "minimum",
                KernelImplementation::binary(BackendType::Cpu, minimum_f32_cpu),
            )
            .ok();
    }
}

fn register_maximum() {
    {
        let desc = OperationDescriptor::new("maximum", "binary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "maximum",
                KernelImplementation::binary(BackendType::Cpu, maximum_f32_cpu),
            )
            .ok();
    }
}

fn register_atan2() {
    {
        let desc = OperationDescriptor::new("atan2", "binary")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "atan2",
                KernelImplementation::binary(BackendType::Cpu, atan2_f32_cpu),
            )
            .ok();
    }
}

// ============================================================================
// Comparison Operation Registrations
// ============================================================================

fn register_equal() {
    {
        let desc = OperationDescriptor::new("equal", "comparison")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "equal",
                KernelImplementation::binary(BackendType::Cpu, equal_f32_cpu),
            )
            .ok();
    }
}

fn register_not_equal() {
    {
        let desc = OperationDescriptor::new("not_equal", "comparison")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "not_equal",
                KernelImplementation::binary(BackendType::Cpu, not_equal_f32_cpu),
            )
            .ok();
    }
}

fn register_less_than() {
    {
        let desc = OperationDescriptor::new("less", "comparison")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "less",
                KernelImplementation::binary(BackendType::Cpu, less_f32_cpu),
            )
            .ok();
    }
}

fn register_less_equal() {
    {
        let desc = OperationDescriptor::new("less_equal", "comparison")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "less_equal",
                KernelImplementation::binary(BackendType::Cpu, less_equal_f32_cpu),
            )
            .ok();
    }
}

fn register_greater_than() {
    {
        let desc = OperationDescriptor::new("greater", "comparison")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "greater",
                KernelImplementation::binary(BackendType::Cpu, greater_f32_cpu),
            )
            .ok();
    }
}

fn register_greater_equal() {
    {
        let desc = OperationDescriptor::new("greater_equal", "comparison")
            .with_dtypes(vec![DType::Float32])
            .with_broadcast();

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "greater_equal",
                KernelImplementation::binary(BackendType::Cpu, greater_equal_f32_cpu),
            )
            .ok();
    }
}

// ============================================================================
// Reduction Operation Registrations
// ============================================================================

fn register_prod() {
    {
        let desc = OperationDescriptor::new("prod", "reduction").with_dtypes(vec![DType::Float32]);

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "prod",
                KernelImplementation::unary(BackendType::Cpu, prod_f32_cpu),
            )
            .ok();
    }
}

fn register_all() {
    {
        let desc = OperationDescriptor::new("all", "reduction").with_dtypes(vec![DType::Float32]);

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "all",
                KernelImplementation::unary(BackendType::Cpu, all_f32_cpu),
            )
            .ok();
    }
}

fn register_any() {
    {
        let desc = OperationDescriptor::new("any", "reduction").with_dtypes(vec![DType::Float32]);

        F32_REGISTRY.register_operation(desc).ok();
        F32_REGISTRY
            .register_kernel(
                "any",
                KernelImplementation::unary(BackendType::Cpu, any_f32_cpu),
            )
            .ok();
    }
}

// ============================================================================
// CPU Kernel Implementations
// ============================================================================

// Unary operations
fn tanh_f32_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = x.data();
    let result: Vec<f32> = data.iter().map(|v| v.tanh()).collect();
    let array = ArrayD::from_shape_vec(x.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape error in tanh: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn sigmoid_f32_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = x.data();
    let result: Vec<f32> = data.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    let array = ArrayD::from_shape_vec(x.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape error in sigmoid: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn relu_f32_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = x.data();
    let result: Vec<f32> = data
        .iter()
        .map(|v| if *v > 0.0 { *v } else { 0.0 })
        .collect();
    let array = ArrayD::from_shape_vec(x.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape error in relu: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn sign_f32_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = x.data();
    let result: Vec<f32> = data
        .iter()
        .map(|v| {
            if *v > 0.0 {
                1.0
            } else if *v < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect();
    let array = ArrayD::from_shape_vec(x.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape error in sign: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn reciprocal_f32_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = x.data();
    let result: Vec<f32> = data.iter().map(|v| 1.0 / v).collect();
    let array = ArrayD::from_shape_vec(x.shape().dims(), result).map_err(|e| {
        TensorError::invalid_shape_simple(format!("Shape error in reciprocal: {}", e))
    })?;
    Ok(Tensor::from_array(array))
}

fn square_f32_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = x.data();
    let result: Vec<f32> = data.iter().map(|v| v * v).collect();
    let array = ArrayD::from_shape_vec(x.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape error in square: {}", e)))?;
    Ok(Tensor::from_array(array))
}

// Binary operations
fn sub_f32_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.shape() != b.shape() {
        return Err(TensorError::shape_mismatch(
            "sub",
            &format!("{:?}", a.shape()),
            &format!("{:?}", b.shape()),
        ));
    }
    let a_data = a.data();
    let b_data = b.data();
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x - y)
        .collect();
    let array = ArrayD::from_shape_vec(a.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape error in sub: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn remainder_f32_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.shape() != b.shape() {
        return Err(TensorError::shape_mismatch(
            "remainder",
            &format!("{:?}", a.shape()),
            &format!("{:?}", b.shape()),
        ));
    }
    let a_data = a.data();
    let b_data = b.data();
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x % y)
        .collect();
    let array = ArrayD::from_shape_vec(a.shape().dims(), result).map_err(|e| {
        TensorError::invalid_shape_simple(format!("Shape error in remainder: {}", e))
    })?;
    Ok(Tensor::from_array(array))
}

fn minimum_f32_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.shape() != b.shape() {
        return Err(TensorError::shape_mismatch(
            "minimum",
            &format!("{:?}", a.shape()),
            &format!("{:?}", b.shape()),
        ));
    }
    let a_data = a.data();
    let b_data = b.data();
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x.min(*y))
        .collect();
    let array = ArrayD::from_shape_vec(a.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape error in minimum: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn maximum_f32_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.shape() != b.shape() {
        return Err(TensorError::shape_mismatch(
            "maximum",
            &format!("{:?}", a.shape()),
            &format!("{:?}", b.shape()),
        ));
    }
    let a_data = a.data();
    let b_data = b.data();
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x.max(*y))
        .collect();
    let array = ArrayD::from_shape_vec(a.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape error in maximum: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn atan2_f32_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.shape() != b.shape() {
        return Err(TensorError::shape_mismatch(
            "atan2",
            &format!("{:?}", a.shape()),
            &format!("{:?}", b.shape()),
        ));
    }
    let a_data = a.data();
    let b_data = b.data();
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(y, x)| y.atan2(*x))
        .collect();
    let array = ArrayD::from_shape_vec(a.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape error in atan2: {}", e)))?;
    Ok(Tensor::from_array(array))
}

// Comparison operations (return f32 as 0.0 or 1.0 for compatibility)
fn equal_f32_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.shape() != b.shape() {
        return Err(TensorError::shape_mismatch(
            "equal",
            &format!("{:?}", a.shape()),
            &format!("{:?}", b.shape()),
        ));
    }
    let a_data = a.data();
    let b_data = b.data();
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| {
            if (*x - *y).abs() < f32::EPSILON {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let array = ArrayD::from_shape_vec(a.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape error in equal: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn not_equal_f32_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.shape() != b.shape() {
        return Err(TensorError::shape_mismatch(
            "not_equal",
            &format!("{:?}", a.shape()),
            &format!("{:?}", b.shape()),
        ));
    }
    let a_data = a.data();
    let b_data = b.data();
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| {
            if (*x - *y).abs() >= f32::EPSILON {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let array = ArrayD::from_shape_vec(a.shape().dims(), result).map_err(|e| {
        TensorError::invalid_shape_simple(format!("Shape error in not_equal: {}", e))
    })?;
    Ok(Tensor::from_array(array))
}

fn less_f32_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.shape() != b.shape() {
        return Err(TensorError::shape_mismatch(
            "less",
            &format!("{:?}", a.shape()),
            &format!("{:?}", b.shape()),
        ));
    }
    let a_data = a.data();
    let b_data = b.data();
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| if x < y { 1.0 } else { 0.0 })
        .collect();
    let array = ArrayD::from_shape_vec(a.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape error in less: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn less_equal_f32_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.shape() != b.shape() {
        return Err(TensorError::shape_mismatch(
            "less_equal",
            &format!("{:?}", a.shape()),
            &format!("{:?}", b.shape()),
        ));
    }
    let a_data = a.data();
    let b_data = b.data();
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| if x <= y { 1.0 } else { 0.0 })
        .collect();
    let array = ArrayD::from_shape_vec(a.shape().dims(), result).map_err(|e| {
        TensorError::invalid_shape_simple(format!("Shape error in less_equal: {}", e))
    })?;
    Ok(Tensor::from_array(array))
}

fn greater_f32_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.shape() != b.shape() {
        return Err(TensorError::shape_mismatch(
            "greater",
            &format!("{:?}", a.shape()),
            &format!("{:?}", b.shape()),
        ));
    }
    let a_data = a.data();
    let b_data = b.data();
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| if x > y { 1.0 } else { 0.0 })
        .collect();
    let array = ArrayD::from_shape_vec(a.shape().dims(), result)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Shape error in greater: {}", e)))?;
    Ok(Tensor::from_array(array))
}

fn greater_equal_f32_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.shape() != b.shape() {
        return Err(TensorError::shape_mismatch(
            "greater_equal",
            &format!("{:?}", a.shape()),
            &format!("{:?}", b.shape()),
        ));
    }
    let a_data = a.data();
    let b_data = b.data();
    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| if x >= y { 1.0 } else { 0.0 })
        .collect();
    let array = ArrayD::from_shape_vec(a.shape().dims(), result).map_err(|e| {
        TensorError::invalid_shape_simple(format!("Shape error in greater_equal: {}", e))
    })?;
    Ok(Tensor::from_array(array))
}

// Reduction operations
fn prod_f32_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = x.data();
    let product: f32 = data.iter().product();
    let result = Array::from_elem(vec![], product).into_dyn();
    Ok(Tensor::from_array(result))
}

fn all_f32_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = x.data();
    let all = data.iter().all(|&v| v != 0.0);
    let result = Array::from_elem(vec![], if all { 1.0 } else { 0.0 }).into_dyn();
    Ok(Tensor::from_array(result))
}

fn any_f32_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = x.data();
    let any = data.iter().any(|&v| v != 0.0);
    let result = Array::from_elem(vec![], if any { 1.0 } else { 0.0 }).into_dyn();
    Ok(Tensor::from_array(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_extended_initialization() {
        initialize_extended_registrations();

        // Verify unary operations
        assert!(F32_REGISTRY.get_operation("tanh").is_some());
        assert!(F32_REGISTRY.get_operation("sigmoid").is_some());
        assert!(F32_REGISTRY.get_operation("relu").is_some());

        // Verify binary operations
        assert!(F32_REGISTRY.get_operation("minimum").is_some());
        assert!(F32_REGISTRY.get_operation("maximum").is_some());

        // Verify comparison operations
        assert!(F32_REGISTRY.get_operation("equal").is_some());
        assert!(F32_REGISTRY.get_operation("less").is_some());
        assert!(F32_REGISTRY.get_operation("greater").is_some());

        // Verify reduction operations
        assert!(F32_REGISTRY.get_operation("prod").is_some());
        assert!(F32_REGISTRY.get_operation("all").is_some());
        assert!(F32_REGISTRY.get_operation("any").is_some());
    }

    #[test]
    fn test_tanh() {
        initialize_extended_registrations();

        let input = Tensor::from_array(array![0.0f32, 1.0, -1.0].into_dyn());
        let result = F32_REGISTRY.dispatch_unary("tanh", &input).unwrap();

        assert!((result.data()[0] - 0.0).abs() < 1e-6);
        assert!((result.data()[1] - 0.7616).abs() < 1e-3);
        assert!((result.data()[2] + 0.7616).abs() < 1e-3);
    }

    #[test]
    fn test_relu() {
        initialize_extended_registrations();

        let input = Tensor::from_array(array![-1.0f32, 0.0, 1.0, -5.0, 10.0].into_dyn());
        let result = F32_REGISTRY.dispatch_unary("relu", &input).unwrap();

        assert_eq!(result.data(), &[0.0f32, 0.0, 1.0, 0.0, 10.0]);
    }

    #[test]
    fn test_comparison_ops() {
        initialize_extended_registrations();

        let a = Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn());
        let b = Tensor::from_array(array![2.0f32, 2.0, 1.0].into_dyn());

        let less = F32_REGISTRY.dispatch_binary("less", &a, &b).unwrap();
        assert_eq!(less.data(), &[1.0f32, 0.0, 0.0]);

        let equal = F32_REGISTRY.dispatch_binary("equal", &a, &b).unwrap();
        assert_eq!(equal.data(), &[0.0f32, 1.0, 0.0]);

        let greater = F32_REGISTRY.dispatch_binary("greater", &a, &b).unwrap();
        assert_eq!(greater.data(), &[0.0f32, 0.0, 1.0]);
    }

    #[test]
    fn test_reduction_prod() {
        initialize_extended_registrations();

        let input = Tensor::from_array(array![2.0f32, 3.0, 4.0].into_dyn());
        let result = F32_REGISTRY.dispatch_unary("prod", &input).unwrap();

        assert_eq!(result.data()[0], 24.0);
    }
}
