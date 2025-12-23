use scirs2_core::ndarray::array;
/// Integration tests for dispatch registry system
///
/// Tests verify:
/// 1. Operation registration and discovery
/// 2. Kernel selection based on backend
/// 3. CPU/GPU consistency
/// 4. Performance characteristics
use tenflowers_core::{ensure_dispatch_initialized, BackendType, Tensor, F32_REGISTRY};

#[test]
fn test_dispatch_initialization() {
    ensure_dispatch_initialized();

    // Verify core operations are registered
    assert!(F32_REGISTRY.get_operation("add").is_some());
    assert!(F32_REGISTRY.get_operation("mul").is_some());
    assert!(F32_REGISTRY.get_operation("div").is_some());
    assert!(F32_REGISTRY.get_operation("abs").is_some());
    assert!(F32_REGISTRY.get_operation("neg").is_some());
}

#[test]
fn test_dispatch_add() {
    ensure_dispatch_initialized();

    let a = Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn());
    let b = Tensor::from_array(array![4.0f32, 5.0, 6.0].into_dyn());

    let result = F32_REGISTRY.dispatch_binary("add", &a, &b).unwrap();

    assert_eq!(result.data(), &[5.0f32, 7.0, 9.0]);
}

#[test]
fn test_dispatch_mul() {
    ensure_dispatch_initialized();

    let a = Tensor::from_array(array![2.0f32, 3.0, 4.0].into_dyn());
    let b = Tensor::from_array(array![5.0f32, 6.0, 7.0].into_dyn());

    let result = F32_REGISTRY.dispatch_binary("mul", &a, &b).unwrap();

    assert_eq!(result.data(), &[10.0f32, 18.0, 28.0]);
}

#[test]
fn test_dispatch_div() {
    ensure_dispatch_initialized();

    let a = Tensor::from_array(array![10.0f32, 20.0, 30.0].into_dyn());
    let b = Tensor::from_array(array![2.0f32, 4.0, 5.0].into_dyn());

    let result = F32_REGISTRY.dispatch_binary("div", &a, &b).unwrap();

    assert_eq!(result.data(), &[5.0f32, 5.0, 6.0]);
}

#[test]
fn test_dispatch_abs() {
    ensure_dispatch_initialized();

    let input = Tensor::from_array(array![-1.0f32, 2.0, -3.0, 4.0, -5.0].into_dyn());

    let result = F32_REGISTRY.dispatch_unary("abs", &input).unwrap();

    assert_eq!(result.data(), &[1.0f32, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_dispatch_neg() {
    ensure_dispatch_initialized();

    let input = Tensor::from_array(array![1.0f32, -2.0, 3.0, -4.0].into_dyn());

    let result = F32_REGISTRY.dispatch_unary("neg", &input).unwrap();

    assert_eq!(result.data(), &[-1.0f32, 2.0, -3.0, 4.0]);
}

#[test]
fn test_dispatch_exp() {
    ensure_dispatch_initialized();

    let input = Tensor::from_array(array![0.0f32, 1.0].into_dyn());

    let result = F32_REGISTRY.dispatch_unary("exp", &input).unwrap();

    // exp(0) = 1, exp(1) ≈ 2.718
    assert!((result.data()[0] - 1.0).abs() < 1e-6);
    assert!((result.data()[1] - std::f32::consts::E).abs() < 1e-6);
}

#[test]
fn test_dispatch_log() {
    ensure_dispatch_initialized();

    let input = Tensor::from_array(array![1.0f32, std::f32::consts::E].into_dyn());

    let result = F32_REGISTRY.dispatch_unary("log", &input).unwrap();

    // ln(1) = 0, ln(e) = 1
    assert!((result.data()[0] - 0.0).abs() < 1e-6);
    assert!((result.data()[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_dispatch_sqrt() {
    ensure_dispatch_initialized();

    let input = Tensor::from_array(array![1.0f32, 4.0, 9.0, 16.0].into_dyn());

    let result = F32_REGISTRY.dispatch_unary("sqrt", &input).unwrap();

    assert_eq!(result.data(), &[1.0f32, 2.0, 3.0, 4.0]);
}

#[test]
fn test_backend_availability() {
    ensure_dispatch_initialized();

    let backends = F32_REGISTRY.available_backends("add");

    // CPU backend should always be available
    assert!(backends.contains(&BackendType::Cpu));

    #[cfg(feature = "simd")]
    assert!(backends.contains(&BackendType::SimdCpu));

    #[cfg(feature = "gpu")]
    assert!(backends.contains(&BackendType::Gpu));
}

#[test]
fn test_operation_listing() {
    ensure_dispatch_initialized();

    let operations = F32_REGISTRY.list_operations();

    // Should have at least the core operations
    assert!(operations.contains(&"add".to_string()));
    assert!(operations.contains(&"mul".to_string()));
    assert!(operations.contains(&"div".to_string()));
    assert!(operations.contains(&"abs".to_string()));
}

#[test]
fn test_shape_mismatch_error() {
    ensure_dispatch_initialized();

    let a = Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn());
    let b = Tensor::from_array(array![1.0f32, 2.0].into_dyn());

    let result = F32_REGISTRY.dispatch_binary("add", &a, &b);

    // Should fail due to shape mismatch
    assert!(result.is_err());
}

#[test]
fn test_trig_functions() {
    ensure_dispatch_initialized();

    let input = Tensor::from_array(array![0.0f32, std::f32::consts::PI / 2.0].into_dyn());

    // Test sin
    let sin_result = F32_REGISTRY.dispatch_unary("sin", &input).unwrap();
    assert!((sin_result.data()[0] - 0.0).abs() < 1e-6);
    assert!((sin_result.data()[1] - 1.0).abs() < 1e-6);

    // Test cos
    let cos_result = F32_REGISTRY.dispatch_unary("cos", &input).unwrap();
    assert!((cos_result.data()[0] - 1.0).abs() < 1e-6);
    assert!((cos_result.data()[1] - 0.0).abs() < 1e-6);
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_reduction_sum() {
    ensure_dispatch_initialized();

    let input = Tensor::from_array(array![1.0f32, 2.0, 3.0, 4.0, 5.0].into_dyn());

    // Test sum operation (should use GPU kernel if available)
    if F32_REGISTRY.get_operation("sum").is_some() {
        let result = F32_REGISTRY.dispatch_unary("sum", &input).unwrap();
        assert_eq!(result.data()[0], 15.0);
    }
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_reduction_mean() {
    ensure_dispatch_initialized();

    let input = Tensor::from_array(array![1.0f32, 2.0, 3.0, 4.0, 5.0].into_dyn());

    // Test mean operation (should use GPU kernel if available)
    if F32_REGISTRY.get_operation("mean").is_some() {
        let result = F32_REGISTRY.dispatch_unary("mean", &input).unwrap();
        assert_eq!(result.data()[0], 3.0);
    }
}

#[test]
fn test_large_tensor_operations() {
    ensure_dispatch_initialized();

    // Create larger tensors to test performance
    let size = 1000;
    let a_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..size).map(|i| (i as f32) * 2.0).collect();

    let a = Tensor::from_array(scirs2_core::ndarray::Array1::from_vec(a_data).into_dyn());
    let b = Tensor::from_array(scirs2_core::ndarray::Array1::from_vec(b_data).into_dyn());

    let result = F32_REGISTRY.dispatch_binary("add", &a, &b).unwrap();

    // Check a few values
    assert_eq!(result.data()[0], 0.0);
    assert_eq!(result.data()[10], 30.0); // 10 + 20
    assert_eq!(result.data()[100], 300.0); // 100 + 200
}

#[test]
fn test_chained_operations() {
    ensure_dispatch_initialized();

    let a = Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn());
    let b = Tensor::from_array(array![2.0f32, 3.0, 4.0].into_dyn());
    let c = Tensor::from_array(array![1.0f32, 1.0, 1.0].into_dyn());

    // (a + b) * c
    let temp = F32_REGISTRY.dispatch_binary("add", &a, &b).unwrap();
    let result = F32_REGISTRY.dispatch_binary("mul", &temp, &c).unwrap();

    assert_eq!(result.data(), &[3.0f32, 5.0, 7.0]);
}
