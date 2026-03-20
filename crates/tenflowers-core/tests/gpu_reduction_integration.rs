/// GPU Reduction Integration Tests
///
/// These tests verify that GPU reduction kernels are properly connected to WGPU runtime
/// and that CPU fallback works correctly.

#[cfg(feature = "gpu")]
mod gpu_tests {
    use tenflowers_core::{Device, Tensor};

    /// Helper to check if GPU is available
    fn gpu_available() -> bool {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
        })
        .is_ok()
    }

    #[test]
    fn test_gpu_sum_reduction_1d() {
        if !gpu_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        // Create a 1D tensor on CPU
        let data: Vec<f32> = (1..=100).map(|x| x as f32).collect();
        let cpu_tensor = Tensor::<f32>::from_data(data, &[100]).expect("create tensor");

        // Move to GPU
        let gpu_tensor = cpu_tensor.to_device(Device::Gpu(0)).expect("move to GPU");

        // Perform GPU reduction
        let result = gpu_tensor
            .sum(Some(&[0]), false)
            .expect("GPU sum reduction");

        // Move result back to CPU to access data
        let result_cpu = result.to_cpu().expect("move to CPU");
        let result_data = result_cpu.data();
        let expected_sum: f32 = (1..=100).sum::<i32>() as f32; // 5050
        assert_eq!(result_data.len(), 1);
        assert!(
            (result_data[0] - expected_sum).abs() < 1e-3,
            "Expected {}, got {}",
            expected_sum,
            result_data[0]
        );
    }

    #[test]
    fn test_gpu_mean_reduction_2d() {
        if !gpu_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        // Create a 2D tensor on CPU: [[1, 2, 3], [4, 5, 6]]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cpu_tensor = Tensor::<f32>::from_data(data, &[2, 3]).expect("create tensor");

        // Move to GPU
        let gpu_tensor = cpu_tensor.to_device(Device::Gpu(0)).expect("move to GPU");

        // Perform GPU mean reduction along axis 1 (columns)
        let result = gpu_tensor
            .mean(Some(&[1]), false)
            .expect("GPU mean reduction");

        // Move result back to CPU to access data
        let result_cpu = result.to_cpu().expect("move to CPU");
        let result_data = result_cpu.data();
        assert_eq!(result_data.len(), 2);
        assert!(
            (result_data[0] - 2.0).abs() < 1e-3,
            "Expected 2.0, got {}",
            result_data[0]
        );
        assert!(
            (result_data[1] - 5.0).abs() < 1e-3,
            "Expected 5.0, got {}",
            result_data[1]
        );
    }

    #[test]
    fn test_gpu_max_reduction_2d() {
        if !gpu_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        // Create a 2D tensor: [[1, 5, 3], [4, 2, 6]]
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let cpu_tensor = Tensor::<f32>::from_data(data, &[2, 3]).expect("create tensor");

        // Move to GPU
        let gpu_tensor = cpu_tensor.to_device(Device::Gpu(0)).expect("move to GPU");

        // Perform GPU max reduction along axis 1
        let result = gpu_tensor
            .max(Some(&[1]), false)
            .expect("GPU max reduction");

        // Move result back to CPU to access data
        let result_cpu = result.to_cpu().expect("move to CPU");
        let result_data = result_cpu.data();
        assert_eq!(result_data.len(), 2);
        assert!(
            (result_data[0] - 5.0).abs() < 1e-3,
            "Expected 5.0, got {}",
            result_data[0]
        );
        assert!(
            (result_data[1] - 6.0).abs() < 1e-3,
            "Expected 6.0, got {}",
            result_data[1]
        );
    }

    #[test]
    fn test_gpu_min_reduction_2d() {
        if !gpu_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        // Create a 2D tensor: [[1, 5, 3], [4, 2, 6]]
        let data: Vec<f32> = vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
        let cpu_tensor = Tensor::<f32>::from_data(data, &[2, 3]).expect("create tensor");

        // Move to GPU
        let gpu_tensor = cpu_tensor.to_device(Device::Gpu(0)).expect("move to GPU");

        // Perform GPU min reduction along axis 1
        let result = gpu_tensor
            .min(Some(&[1]), false)
            .expect("GPU min reduction");

        // Move result back to CPU to access data
        let result_cpu = result.to_cpu().expect("move to CPU");
        let result_data = result_cpu.data();
        assert_eq!(result_data.len(), 2);
        assert!(
            (result_data[0] - 1.0).abs() < 1e-3,
            "Expected 1.0, got {}",
            result_data[0]
        );
        assert!(
            (result_data[1] - 2.0).abs() < 1e-3,
            "Expected 2.0, got {}",
            result_data[1]
        );
    }

    #[test]
    fn test_gpu_reduction_keepdims() {
        if !gpu_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        // Create a 2D tensor: [[1, 2], [3, 4]]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let cpu_tensor = Tensor::<f32>::from_data(data, &[2, 2]).expect("create tensor");

        // Move to GPU
        let gpu_tensor = cpu_tensor.to_device(Device::Gpu(0)).expect("move to GPU");

        // Perform GPU sum reduction with keepdims=true
        let result = gpu_tensor
            .sum(Some(&[1]), true)
            .expect("GPU sum reduction with keepdims");

        // Verify shape: should be [2, 1]
        assert_eq!(result.shape().dims(), &[2, 1]);

        // Move result back to CPU to access data
        let result_cpu = result.to_cpu().expect("move to CPU");
        let result_data = result_cpu.data();
        assert_eq!(result_data.len(), 2);
        assert!(
            (result_data[0] - 3.0).abs() < 1e-3,
            "Expected 3.0, got {}",
            result_data[0]
        );
        assert!(
            (result_data[1] - 7.0).abs() < 1e-3,
            "Expected 7.0, got {}",
            result_data[1]
        );
    }

    #[test]
    fn test_cpu_fallback_for_cpu_tensor() {
        // Create a CPU tensor
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cpu_tensor = Tensor::<f32>::from_data(data, &[2, 3]).expect("create tensor");

        // Perform reduction (should use CPU implementation)
        let result = cpu_tensor
            .sum(Some(&[1]), false)
            .expect("CPU sum reduction");

        // Verify result
        let result_data = result.data();
        assert_eq!(result_data.len(), 2);
        assert!(
            (result_data[0] - 6.0).abs() < 1e-3,
            "Expected 6.0, got {}",
            result_data[0]
        );
        assert!(
            (result_data[1] - 15.0).abs() < 1e-3,
            "Expected 15.0, got {}",
            result_data[1]
        );
    }

    #[test]
    fn test_gpu_vs_cpu_consistency() {
        if !gpu_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        // Create a larger tensor for better testing
        let data: Vec<f32> = (1..=1000).map(|x| (x as f32) * 0.1).collect();
        let cpu_tensor = Tensor::<f32>::from_data(data, &[10, 100]).expect("create tensor");

        // Perform CPU reduction
        let cpu_result = cpu_tensor
            .sum(Some(&[1]), false)
            .expect("CPU sum reduction");
        let cpu_data = cpu_result.data();

        // Move to GPU and perform GPU reduction
        let gpu_tensor = cpu_tensor.to_device(Device::Gpu(0)).expect("move to GPU");
        let gpu_result = gpu_tensor
            .sum(Some(&[1]), false)
            .expect("GPU sum reduction");
        let gpu_result_cpu = gpu_result.to_cpu().expect("move to CPU");
        let gpu_data = gpu_result_cpu.data();

        // Compare results
        assert_eq!(
            cpu_data.len(),
            gpu_data.len(),
            "Result lengths should match"
        );
        for (i, (cpu_val, gpu_val)) in cpu_data.iter().zip(gpu_data.iter()).enumerate() {
            assert!(
                (cpu_val - gpu_val).abs() < 1e-2,
                "Mismatch at index {}: CPU={}, GPU={}",
                i,
                cpu_val,
                gpu_val
            );
        }
    }

    #[test]
    fn test_gpu_large_reduction() {
        if !gpu_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        // Create a large tensor to test GPU performance
        let size = 10000;
        let data: Vec<f32> = (0..size).map(|x| (x as f32) * 0.001).collect();
        let cpu_tensor = Tensor::<f32>::from_data(data, &[size]).expect("create tensor");

        // Move to GPU
        let gpu_tensor = cpu_tensor.to_device(Device::Gpu(0)).expect("move to GPU");

        // Perform GPU reduction
        let result = gpu_tensor
            .sum(Some(&[0]), false)
            .expect("GPU sum on large tensor");

        // Move result back to CPU to access data
        let result_cpu = result.to_cpu().expect("move to CPU");
        let result_data = result_cpu.data();
        assert_eq!(result_data.len(), 1);

        let expected_sum: f32 = (0..size).map(|x| (x as f32) * 0.001).sum();
        assert!(
            (result_data[0] - expected_sum).abs() < 1.0,
            "Expected ~{}, got {}",
            expected_sum,
            result_data[0]
        );
    }
}

#[cfg(not(feature = "gpu"))]
#[test]
fn test_gpu_feature_disabled() {
    // When GPU feature is disabled, this test passes
    // to indicate that the build works without GPU support
}
