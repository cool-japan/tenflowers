use scirs2_core::ndarray::{Array1, Array2, Array3};
use tenflowers_core::{ops::conv1d, Device, Tensor};

#[test]
fn test_conv1d_cpu_basic() {
    // Create test input: [batch=1, channels=2, length=5]
    let input_data = vec![
        // Channel 0
        1.0f32, 2.0, 3.0, 4.0, 5.0, // Channel 1
        2.0, 3.0, 4.0, 5.0, 6.0,
    ];
    let input = Tensor::from_vec(input_data, &[1, 2, 5]).unwrap();

    // Create kernel: [out_channels=1, in_channels=2, kernel_length=3]
    let kernel_data = vec![
        // Output channel 0, Input channel 0
        1.0f32, 0.0, -1.0, // Output channel 0, Input channel 1
        0.5, 0.5, 0.5,
    ];
    let kernel = Tensor::from_vec(kernel_data, &[1, 2, 3]).unwrap();

    // Create bias: [out_channels=1]
    let bias = Tensor::from_vec(vec![0.1f32], &[1]).unwrap();

    // Perform convolution with stride=1, padding="valid"
    let result = conv1d(&input, &kernel, Some(&bias), 1, "valid").unwrap();

    // Check output shape: [batch=1, out_channels=1, out_length=3]
    assert_eq!(result.shape().dims(), &[1, 1, 3]);

    // Verify values
    let result_data = result.to_vec().unwrap();

    // Manual calculation for verification:
    // Position 0: (1*1 + 2*0 + 3*(-1)) + (2*0.5 + 3*0.5 + 4*0.5) + 0.1 = -2 + 4.5 + 0.1 = 2.6
    // Position 1: (2*1 + 3*0 + 4*(-1)) + (3*0.5 + 4*0.5 + 5*0.5) + 0.1 = -2 + 6.0 + 0.1 = 4.1
    // Position 2: (3*1 + 4*0 + 5*(-1)) + (4*0.5 + 5*0.5 + 6*0.5) + 0.1 = -2 + 7.5 + 0.1 = 5.6

    let expected = vec![2.6f32, 4.1, 5.6];
    for (i, (&actual, &expected)) in result_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5f32,
            "Mismatch at position {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_conv1d_cpu_same_padding() {
    // Create test input: [batch=1, channels=1, length=5]
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let input = Tensor::from_vec(input_data, &[1, 1, 5]).unwrap();

    // Create simple kernel: [out_channels=1, in_channels=1, kernel_length=3]
    let kernel_data = vec![1.0f32, 0.0, -1.0];
    let kernel = Tensor::from_vec(kernel_data, &[1, 1, 3]).unwrap();

    // Perform convolution with stride=1, padding="same"
    let result = conv1d(&input, &kernel, None, 1, "same").unwrap();

    // Check output shape: [batch=1, out_channels=1, out_length=5] (same as input)
    assert_eq!(result.shape().dims(), &[1, 1, 5]);

    // Verify the convolution preserves input length with "same" padding
    let result_data = result.to_vec().unwrap();
    assert_eq!(result_data.len(), 5);
}

#[test]
fn test_conv1d_cpu_multiple_channels() {
    // Test with multiple input and output channels
    let input_data = vec![
        // Batch 0, Channel 0
        1.0f32, 2.0, 3.0, // Batch 0, Channel 1
        4.0, 5.0, 6.0,
    ];
    let input = Tensor::from_vec(input_data, &[1, 2, 3]).unwrap();

    // Create kernel: [out_channels=2, in_channels=2, kernel_length=2]
    let kernel_data = vec![
        // Output channel 0
        1.0f32, 1.0, // Input channel 0
        1.0, 1.0, // Input channel 1
        // Output channel 1
        0.5, 0.5, // Input channel 0
        -0.5, -0.5, // Input channel 1
    ];
    let kernel = Tensor::from_vec(kernel_data, &[2, 2, 2]).unwrap();

    // Perform convolution
    let result = conv1d(&input, &kernel, None, 1, "valid").unwrap();

    // Check output shape: [batch=1, out_channels=2, out_length=2]
    assert_eq!(result.shape().dims(), &[1, 2, 2]);

    let result_data = result.to_vec().unwrap();
    assert_eq!(result_data.len(), 4); // 1 * 2 * 2
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "GPU shader compilation issues - WGSL reserved keyword 'var'"]
fn test_conv1d_gpu_basic() {
    // This test requires GPU feature and proper GPU context
    // Create test input: [batch=1, channels=1, length=4]
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_cpu = Tensor::from_vec(input_data, &[1, 1, 4]).unwrap();

    // Try to convert to GPU (will skip if no GPU available)
    if let Ok(input_gpu) = input_cpu.to_device(Device::Gpu(0)) {
        // Create kernel: [out_channels=1, in_channels=1, kernel_length=2]
        let kernel_data = vec![1.0f32, -1.0];
        let kernel_cpu = Tensor::from_vec(kernel_data, &[1, 1, 2]).unwrap();
        let kernel_gpu = kernel_cpu.to_device(Device::Gpu(0)).unwrap();

        // Perform GPU convolution
        let result_gpu = conv1d(&input_gpu, &kernel_gpu, None, 1, "valid").unwrap();

        // Convert back to CPU for verification
        let result_cpu = result_gpu.to_device(Device::Cpu).unwrap();

        // Check output shape: [batch=1, out_channels=1, out_length=3]
        assert_eq!(result_cpu.shape().dims(), &[1, 1, 3]);

        // Verify correctness
        let result_data = result_cpu.to_vec().unwrap();
        let expected = vec![-1.0, -1.0, -1.0]; // [1-2, 2-3, 3-4]

        for (i, (&actual, &expected)) in result_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5f32,
                "GPU Conv1D mismatch at position {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }
}

#[test]
fn test_conv1d_stride() {
    // Test convolution with stride > 1
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Tensor::from_vec(input_data, &[1, 1, 6]).unwrap();

    let kernel_data = vec![1.0f32, 1.0];
    let kernel = Tensor::from_vec(kernel_data, &[1, 1, 2]).unwrap();

    // Convolution with stride=2
    let result = conv1d(&input, &kernel, None, 2, "valid").unwrap();

    // Check output shape: length should be (6-2)/2 + 1 = 3
    assert_eq!(result.shape().dims(), &[1, 1, 3]);

    // Expected: [1+2, 3+4, 5+6] = [3, 7, 11]
    let result_data = result.to_vec().unwrap();
    let expected = vec![3.0f32, 7.0, 11.0];

    for (i, (&actual, &expected)) in result_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5f32,
            "Stride test mismatch at position {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}
