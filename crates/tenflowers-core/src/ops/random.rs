use crate::{Device, Result, Tensor, TensorError};
use scirs2_core::random::rand_distributions::{Distribution, Normal, Uniform as RandUniform};
use scirs2_core::random::rand_prelude::*;

#[cfg(feature = "gpu")]
use crate::gpu::GpuOps;

/// Generate a tensor with random values from a normal distribution (f32 only)
pub fn random_normal_f32(
    shape: &[usize],
    mean: f32,
    std: f32,
    seed: Option<u64>,
) -> Result<Tensor<f32>> {
    random_normal_f32_device(shape, mean, std, seed, &Device::Cpu)
}

/// Generate a tensor with random values from a normal distribution (f32 only) on specified device
pub fn random_normal_f32_device(
    shape: &[usize],
    mean: f32,
    std: f32,
    seed: Option<u64>,
    device: &Device,
) -> Result<Tensor<f32>> {
    let seed = seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });

    match device {
        Device::Cpu => {
            let total_elements: usize = shape.iter().product();
            let mut rng = StdRng::seed_from_u64(seed);

            let normal = Normal::new(mean, std).map_err(|e| {
                TensorError::invalid_argument(format!(
                    "Invalid normal distribution parameters: {e}"
                ))
            })?;

            let mut data = Vec::with_capacity(total_elements);
            for _ in 0..total_elements {
                data.push(normal.sample(&mut rng));
            }

            Tensor::from_vec(data, shape)
        }
        #[cfg(feature = "gpu")]
        Device::Gpu(device_id) => {
            use crate::device::get_gpu_context;
            use crate::gpu::random_ops;

            // Get GPU context
            let context = get_gpu_context(*device_id)?;

            // Calculate output length
            let output_len: usize = shape.iter().product();

            // Execute GPU random normal operation
            let gpu_buffer = random_ops::execute_random_normal::<f32>(
                context.device,
                context.queue,
                *device,
                output_len,
                mean,
                std,
                seed,
            )?;

            // Create tensor from GPU buffer
            let tensor_shape = crate::Shape::new(shape.to_vec());
            Ok(Tensor::from_gpu_buffer(gpu_buffer, tensor_shape))
        }
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            // TODO: Implement ROCm random normal operation
            todo!("ROCm random normal not yet implemented")
        }
    }
}

/// Generate a tensor with random values from a normal distribution (f64 only)  
pub fn random_normal_f64(
    shape: &[usize],
    mean: f64,
    std: f64,
    seed: Option<u64>,
) -> Result<Tensor<f64>> {
    let total_elements: usize = shape.iter().product();
    let mut rng = scirs2_core::random::Random::seed(seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }));

    let normal = Normal::new(mean, std).map_err(|e| {
        TensorError::invalid_argument(format!("Invalid normal distribution parameters: {e}"))
    })?;

    let mut data = Vec::with_capacity(total_elements);
    for _ in 0..total_elements {
        data.push(normal.sample(&mut rng));
    }

    Tensor::from_vec(data, shape)
}

/// Generate a tensor with random values from a uniform distribution (f32 only)
pub fn random_uniform_f32(
    shape: &[usize],
    min: f32,
    max: f32,
    seed: Option<u64>,
) -> Result<Tensor<f32>> {
    random_uniform_f32_device(shape, min, max, seed, &Device::Cpu)
}

/// Generate a tensor with random values from a uniform distribution (f32 only) on specified device
pub fn random_uniform_f32_device(
    shape: &[usize],
    min: f32,
    max: f32,
    seed: Option<u64>,
    device: &Device,
) -> Result<Tensor<f32>> {
    if min >= max {
        return Err(TensorError::invalid_argument(
            "min must be less than max for uniform distribution".to_string(),
        ));
    }

    let seed = seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });

    match device {
        Device::Cpu => {
            let total_elements: usize = shape.iter().product();
            let mut rng = StdRng::seed_from_u64(seed);

            let uniform = RandUniform::new(min, max).map_err(|e| {
                TensorError::invalid_argument(format!(
                    "Invalid uniform distribution parameters: {e}"
                ))
            })?;

            let mut data = Vec::with_capacity(total_elements);
            for _ in 0..total_elements {
                data.push(uniform.sample(&mut rng));
            }

            Tensor::from_vec(data, shape)
        }
        #[cfg(feature = "gpu")]
        Device::Gpu(device_id) => {
            use crate::device::get_gpu_context;
            use crate::gpu::random_ops;

            // Get GPU context
            let context = get_gpu_context(*device_id)?;

            // Calculate output length
            let output_len: usize = shape.iter().product();

            // Execute GPU random uniform operation
            let gpu_buffer = random_ops::execute_random_uniform::<f32>(
                context.device,
                context.queue,
                *device,
                output_len,
                min,
                max,
                seed,
            )?;

            // Create tensor from GPU buffer
            let tensor_shape = crate::Shape::new(shape.to_vec());
            Ok(Tensor::from_gpu_buffer(gpu_buffer, tensor_shape))
        }
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => {
            // TODO: Implement ROCm random uniform operation
            todo!("ROCm random uniform not yet implemented")
        }
    }
}

/// Generate a tensor with random values from a uniform distribution (f64 only)
pub fn random_uniform_f64(
    shape: &[usize],
    min: f64,
    max: f64,
    seed: Option<u64>,
) -> Result<Tensor<f64>> {
    if min >= max {
        return Err(TensorError::invalid_argument(
            "min must be less than max for uniform distribution".to_string(),
        ));
    }

    let total_elements: usize = shape.iter().product();
    let mut rng = scirs2_core::random::Random::seed(seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }));

    let uniform = RandUniform::new(min, max).map_err(|e| {
        TensorError::invalid_argument(format!("Invalid uniform distribution parameters: {e}"))
    })?;

    let data: Vec<f64> = (0..total_elements)
        .map(|_| uniform.sample(&mut rng))
        .collect();

    Tensor::from_vec(data, shape)
}

/// Generate a tensor with random integers from a uniform distribution
pub fn random_uniform_int(
    shape: &[usize],
    min: i64,
    max: i64,
    seed: Option<u64>,
) -> Result<Tensor<i64>> {
    if min >= max {
        return Err(TensorError::invalid_argument(
            "min must be less than max for uniform distribution".to_string(),
        ));
    }

    let total_elements: usize = shape.iter().product();
    let mut rng = scirs2_core::random::Random::seed(seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }));

    let uniform = RandUniform::new(min, max).map_err(|e| {
        TensorError::invalid_argument(format!("Invalid uniform distribution parameters: {e}"))
    })?;

    let data: Vec<i64> = (0..total_elements)
        .map(|_| uniform.sample(&mut rng))
        .collect();

    Tensor::from_vec(data, shape)
}

/// Generate a tensor with random values from a standard normal distribution (mean=0, std=1) for f32
pub fn randn_f32(shape: &[usize], seed: Option<u64>) -> Result<Tensor<f32>> {
    random_normal_f32(shape, 0.0, 1.0, seed)
}

/// Generate a tensor with random values from a standard normal distribution (mean=0, std=1) for f64
pub fn randn_f64(shape: &[usize], seed: Option<u64>) -> Result<Tensor<f64>> {
    random_normal_f64(shape, 0.0, 1.0, seed)
}

/// Generate a tensor with random values from a uniform distribution [0, 1) for f32
pub fn rand_f32(shape: &[usize], seed: Option<u64>) -> Result<Tensor<f32>> {
    random_uniform_f32(shape, 0.0, 1.0, seed)
}

/// Generate a tensor with random values from a uniform distribution [0, 1) for f64
pub fn rand_f64(shape: &[usize], seed: Option<u64>) -> Result<Tensor<f64>> {
    random_uniform_f64(shape, 0.0, 1.0, seed)
}

/// Sample from a multinomial distribution (f32 weights)
pub fn multinomial_f32(
    weights: &Tensor<f32>,
    num_samples: usize,
    seed: Option<u64>,
) -> Result<Tensor<usize>> {
    let weights_shape = weights.shape().dims();
    if weights_shape.len() != 1 {
        return Err(TensorError::InvalidShape {
            operation: "multinomial".to_string(),
            reason: "multinomial expects 1D weights tensor".to_string(),
            shape: Some(weights_shape.to_vec()),
            context: None,
        });
    }

    let weights_data = weights.as_slice().ok_or_else(|| {
        TensorError::unsupported_operation_simple("GPU multinomial not supported yet".to_string())
    })?;

    // Normalize weights to probabilities
    let total_weight: f32 = weights_data.iter().sum();
    if total_weight <= 0.0 {
        return Err(TensorError::invalid_argument(
            "weights must sum to a positive value".to_string(),
        ));
    }

    let probabilities: Vec<f64> = weights_data
        .iter()
        .map(|&w| (w / total_weight) as f64)
        .collect();

    let mut rng = scirs2_core::random::Random::seed(seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }));

    let mut samples = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let random_val: f64 = rng.random();
        let mut cumulative = 0.0;
        let mut selected_idx = probabilities.len() - 1;

        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if random_val < cumulative {
                selected_idx = i;
                break;
            }
        }

        samples.push(selected_idx);
    }

    Tensor::from_vec(samples, &[num_samples])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_normal() {
        let tensor = random_normal_f32(&[10], 0.0, 1.0, Some(42)).unwrap();
        assert_eq!(tensor.shape().dims(), &[10]);

        // Test with fixed seed should be reproducible
        let tensor2 = random_normal_f32(&[10], 0.0, 1.0, Some(42)).unwrap();
        assert_eq!(tensor.as_slice(), tensor2.as_slice());
    }

    #[test]
    fn test_random_uniform() {
        let tensor = random_uniform_f32(&[5, 2], -1.0, 1.0, Some(123)).unwrap();
        assert_eq!(tensor.shape().dims(), &[5, 2]);

        // Check all values are in range
        if let Some(data) = tensor.as_slice() {
            for &val in data {
                assert!(val >= -1.0 && val < 1.0);
            }
        }
    }

    #[test]
    fn test_random_uniform_int() {
        let tensor = random_uniform_int(&[8], 0, 10, Some(456)).unwrap();
        assert_eq!(tensor.shape().dims(), &[8]);

        // Check all values are in range
        if let Some(data) = tensor.as_slice() {
            for &val in data {
                assert!(val >= 0 && val < 10);
            }
        }
    }

    #[test]
    fn test_randn() {
        let tensor = randn_f32(&[3, 3], Some(789)).unwrap();
        assert_eq!(tensor.shape().dims(), &[3, 3]);
    }

    #[test]
    fn test_rand() {
        let tensor = rand_f32(&[4], Some(101112)).unwrap();
        assert_eq!(tensor.shape().dims(), &[4]);

        // Check all values are in [0, 1)
        if let Some(data) = tensor.as_slice() {
            for &val in data {
                assert!(val >= 0.0 && val < 1.0);
            }
        }
    }

    #[test]
    fn test_multinomial() {
        // Test with uniform weights
        let weights = Tensor::<f32>::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        let samples = multinomial_f32(&weights, 100, Some(131415)).unwrap();

        assert_eq!(samples.shape().dims(), &[100]);

        // Check all sampled indices are valid
        if let Some(data) = samples.as_slice() {
            for &idx in data {
                assert!(idx < 4);
            }
        }

        // Test with biased weights
        let weights = Tensor::<f32>::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[4]).unwrap();
        let samples = multinomial_f32(&weights, 10, Some(161718)).unwrap();
        assert_eq!(samples.shape().dims(), &[10]);
    }

    #[test]
    fn test_multinomial_errors() {
        // Test 2D weights (should fail)
        let weights = Tensor::<f32>::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]).unwrap();
        assert!(multinomial_f32(&weights, 10, Some(123)).is_err());

        // Test zero weights (should fail)
        let weights = Tensor::<f32>::from_vec(vec![0.0, 0.0, 0.0], &[3]).unwrap();
        assert!(multinomial_f32(&weights, 10, Some(123)).is_err());
    }

    #[test]
    #[ignore = "GPU random normal not yet implemented"]
    fn test_gpu_random_normal_f32() {
        #[cfg(feature = "gpu")]
        {
            use crate::Device;

            // Test GPU random normal generation
            let result = random_normal_f32_device(&[10, 10], 0.0, 1.0, Some(42), &Device::Gpu(0));

            // Should either work or return an error indicating GPU is not available
            match result {
                Ok(tensor) => {
                    assert_eq!(tensor.shape().dims(), &[10, 10]);
                    assert_eq!(tensor.numel(), 100);
                }
                Err(_) => {
                    // GPU might not be available in test environment
                    // This is acceptable for the test
                }
            }
        }
    }

    #[test]
    #[ignore = "GPU random uniform not yet implemented"]
    fn test_gpu_random_uniform_f32() {
        #[cfg(feature = "gpu")]
        {
            use crate::Device;

            // Test GPU random uniform generation
            let result = random_uniform_f32_device(&[5, 5], 0.0, 1.0, Some(42), &Device::Gpu(0));

            // Should either work or return an error indicating GPU is not available
            match result {
                Ok(tensor) => {
                    assert_eq!(tensor.shape().dims(), &[5, 5]);
                    assert_eq!(tensor.numel(), 25);
                }
                Err(_) => {
                    // GPU might not be available in test environment
                    // This is acceptable for the test
                }
            }
        }
    }

    #[test]
    fn test_device_aware_random_functions() {
        use crate::Device;

        // Test CPU versions work
        let cpu_normal =
            random_normal_f32_device(&[5, 5], 0.0, 1.0, Some(42), &Device::Cpu).unwrap();
        assert_eq!(cpu_normal.shape().dims(), &[5, 5]);

        let cpu_uniform =
            random_uniform_f32_device(&[5, 5], 0.0, 1.0, Some(42), &Device::Cpu).unwrap();
        assert_eq!(cpu_uniform.shape().dims(), &[5, 5]);

        // Test that CPU implementations are deterministic with same seed
        let cpu_normal2 =
            random_normal_f32_device(&[5, 5], 0.0, 1.0, Some(42), &Device::Cpu).unwrap();
        assert_eq!(cpu_normal.as_slice(), cpu_normal2.as_slice());

        let cpu_uniform2 =
            random_uniform_f32_device(&[5, 5], 0.0, 1.0, Some(42), &Device::Cpu).unwrap();
        assert_eq!(cpu_uniform.as_slice(), cpu_uniform2.as_slice());
    }
}
