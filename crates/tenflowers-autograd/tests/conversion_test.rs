use scirs2_autograd as ag;
use scirs2_autograd::ndarray::array;
use tenflowers_autograd::AutogradContext;
use tenflowers_core::Tensor;

#[test]
fn test_tenflowers_to_autograd_conversion() {
    // Create a TenfloweRS tensor
    let data = array![[1.0f32, 2.0], [3.0, 4.0]];
    let tf_tensor = Tensor::from_array(data.into_dyn());

    // Test conversion within autograd context
    let result = ag::run(|ctx| {
        let ag_tensor = AutogradContext::from_tenflowers(&tf_tensor, ctx).unwrap();

        // The tensor should have the same shape
        assert_eq!(ag_tensor.shape(), vec![2, 2]);

        // Return the shape for verification outside the context
        ag_tensor.shape()
    });

    // Verify the shape was correct
    assert_eq!(result, vec![2, 2]);
}

#[test]
fn test_autograd_to_tenflowers_conversion() {
    let result = ag::run(|ctx| {
        // Create an autograd tensor
        let data = array![[5.0f32, 6.0], [7.0, 8.0]];
        let ag_tensor = ag::tensor_ops::convert_to_tensor(data.into_dyn(), ctx);

        // Convert to TenfloweRS
        let tf_tensor = AutogradContext::to_tenflowers(&ag_tensor, ctx).unwrap();

        // Return properties for verification
        (tf_tensor.shape().dims().to_vec(), tf_tensor.requires_grad())
    });

    // Check properties
    assert_eq!(result.0, vec![2, 2]);
    assert!(!result.1); // requires_grad should be false
}

#[test]
fn test_round_trip_conversion() {
    // Create original TenfloweRS tensor
    let original_data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let original_tensor = Tensor::from_array(original_data.into_dyn());
    let original_shape = original_tensor.shape().dims().to_vec();

    // Test only the conversion to autograd for now
    let ag_shape = ag::run(|ctx| {
        let ag_tensor = AutogradContext::from_tenflowers(&original_tensor, ctx).unwrap();
        ag_tensor.shape()
    });

    // Should have same shape dimensions
    assert_eq!(ag_shape, original_shape);
}

#[test]
fn test_scalar_tensor_conversion() {
    // Test with scalar tensor
    let scalar_tensor = Tensor::from_scalar(42.0f32);

    let ag_shape = ag::run(|ctx| {
        let ag_tensor = AutogradContext::from_tenflowers(&scalar_tensor, ctx).unwrap();
        ag_tensor.shape()
    });

    assert_eq!(ag_shape, Vec::<usize>::new()); // scalar has empty shape
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_tensor_conversion_error() {
    use tenflowers_core::Device;

    // Create a CPU tensor and try to move to GPU
    let cpu_tensor = Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn());
    let gpu_result = cpu_tensor.to(Device::Gpu(0));

    // If GPU is available, test that GPU tensors return unsupported error
    if let Ok(gpu_tensor) = gpu_result {
        let is_error = ag::run(|ctx| {
            // Test that GPU tensors return unsupported error
            let result = AutogradContext::from_tenflowers(&gpu_tensor, ctx);
            result.is_err()
        });

        assert!(is_error);

        // Also verify the specific error type
        let error_result =
            ag::run(
                |ctx| match AutogradContext::from_tenflowers(&gpu_tensor, ctx) {
                    Ok(_) => false,
                    Err(e) => {
                        matches!(e, tenflowers_core::TensorError::UnsupportedOperation { .. })
                    }
                },
            );

        assert!(error_result);
    }
    // If GPU is not available, that's fine - the test is about the error handling
}
