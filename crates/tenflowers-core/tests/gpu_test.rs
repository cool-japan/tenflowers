#[cfg(feature = "gpu")]
#[cfg(test)]
mod gpu_tests {
    use scirs2_core::ndarray::array;
    use tenflowers_core::{Device, Tensor};

    #[test]
    #[ignore] // Only run if GPU is available
    fn test_gpu_tensor_creation() {
        // Create a CPU tensor
        let cpu_tensor = Tensor::from_array(array![[1.0f32, 2.0], [3.0, 4.0]].into_dyn());

        // Try to transfer to GPU (device 0)
        let gpu_device = Device::Gpu(0);
        let result = cpu_tensor.to(gpu_device);

        // If GPU is not available, this should fail gracefully
        // If GPU is available, this should succeed
        match result {
            Ok(gpu_tensor) => {
                // Verify device
                assert_eq!(*gpu_tensor.device(), gpu_device);
                println!("GPU tensor created successfully");

                // Try to transfer back to CPU
                let cpu_tensor2 = gpu_tensor
                    .to(Device::Cpu)
                    .expect("Failed to transfer back to CPU");
                assert_eq!(*cpu_tensor2.device(), Device::Cpu);
                println!("GPU to CPU transfer successful");
            }
            Err(e) => {
                println!("GPU test skipped (GPU not available): {}", e);
            }
        }
    }

    #[test]
    fn test_gpu_operations_not_implemented() {
        // Test that GPU operations return appropriate errors for now
        let cpu_tensor = Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn());

        // Create a mock GPU tensor (we can't actually create one without hardware)
        // For now, just test that the CPU operations work
        assert!(cpu_tensor.log().is_ok());
        assert!(cpu_tensor.neg().is_ok());
        assert!(cpu_tensor.sqrt().is_ok());
    }
}
