//! Tensor Creation and Construction
//!
//! This module contains all tensor constructor methods, including creation
//! from various data sources, initialization with specific patterns,
//! and range generation functions.

use super::core::{Tensor, TensorStorage};
use crate::{Device, Result, Shape, TensorError};
use scirs2_core::ndarray::{ArrayD, IxDyn};

// Impl block for constructors that need Default
impl<T: Clone + Default> Tensor<T> {
    /// Create a tensor filled with zeros
    pub fn zeros(shape: &[usize]) -> Self
    where
        T: scirs2_core::num_traits::Zero,
    {
        let array = ArrayD::zeros(IxDyn(shape));
        Self {
            storage: TensorStorage::Cpu(array),
            shape: Shape::from_slice(shape),
            device: Device::Cpu,
            requires_grad: false,
            grad: None,
        }
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize]) -> Self
    where
        T: scirs2_core::num_traits::One,
    {
        let array = ArrayD::ones(IxDyn(shape));
        Self {
            storage: TensorStorage::Cpu(array),
            shape: Shape::from_slice(shape),
            device: Device::Cpu,
            requires_grad: false,
            grad: None,
        }
    }

    /// Create a tensor from raw data vector with specified shape
    pub fn from_data(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        let total_elements: usize = shape.iter().product();
        if data.len() != total_elements {
            return Err(TensorError::invalid_shape_simple(format!(
                "Data length {} does not match shape {:?} (expected {} elements)",
                data.len(),
                shape,
                total_elements
            )));
        }

        let array = ArrayD::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;

        Ok(Self {
            storage: TensorStorage::Cpu(array),
            shape: Shape::from_slice(shape),
            device: Device::Cpu,
            requires_grad: false,
            grad: None,
        })
    }

    /// Create a tensor from an existing ndarray
    pub fn from_array(array: ArrayD<T>) -> Self {
        let shape = Shape::from_slice(array.shape());
        Self {
            storage: TensorStorage::Cpu(array),
            shape,
            device: Device::Cpu,
            requires_grad: false,
            grad: None,
        }
    }

    /// Create a tensor filled with random values from the standard normal
    /// distribution N(0, 1).
    ///
    /// Samples are drawn from a true Gaussian distribution (mean = 0,
    /// variance = 1) using the `scirs2_core` random facilities, so the result
    /// is suitable for random weight initialisation. A non-deterministic seed
    /// is derived from the system clock; use [`Tensor::randn_with_seed`] for a
    /// reproducible sequence.
    pub fn randn(shape: &[usize]) -> Result<Self>
    where
        T: Clone + Default + From<f32>,
    {
        let seed = {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0)
        };
        Self::randn_with_seed(shape, seed)
    }

    /// Create a tensor filled with values drawn from the standard normal
    /// distribution N(0, 1) using a fixed `seed` for reproducibility.
    pub fn randn_with_seed(shape: &[usize], seed: u64) -> Result<Self>
    where
        T: Clone + Default + From<f32>,
    {
        use scirs2_core::random::rand_distributions::{Distribution, Normal};
        use scirs2_core::random::rand_prelude::{SeedableRng, StdRng};

        let total_elements: usize = shape.iter().product();

        // Standard normal: mean 0, standard deviation 1.
        let normal = Normal::new(0.0_f32, 1.0_f32).map_err(|e| {
            TensorError::invalid_argument(format!(
                "Failed to construct standard normal distribution: {e}"
            ))
        })?;
        let mut rng = StdRng::seed_from_u64(seed);

        let mut data = Vec::with_capacity(total_elements);
        for _ in 0..total_elements {
            data.push(T::from(normal.sample(&mut rng)));
        }

        Self::from_data(data, shape)
    }

    /// Create a tensor from storage and device
    pub fn from_storage(storage: TensorStorage<T>, device: Device) -> Self {
        let shape = match &storage {
            TensorStorage::Cpu(array) => Shape::from_slice(array.shape()),
            #[cfg(feature = "gpu")]
            TensorStorage::Gpu(_buffer) => {
                // GPU buffers don't store shape information directly
                // This should use from_gpu_buffer instead which takes shape as parameter
                panic!("from_storage not supported for GPU buffers - use from_gpu_buffer instead")
            }
        };
        Self {
            storage,
            shape,
            device,
            requires_grad: false,
            grad: None,
        }
    }

    /// Create a tensor from a GPU buffer
    #[cfg(feature = "gpu")]
    pub fn from_gpu_buffer(buffer: crate::gpu::buffer::GpuBuffer<T>, shape: Shape) -> Self {
        // Derive the device from the buffer's own device info rather than
        // hardcoding device 0 - a buffer may live on any GPU device id.
        // `GpuBuffer::device_enum()` is a plain metadata accessor available
        // for any T (see gpu/buffer.rs), so no extra bound is needed here.
        let device = buffer.device_enum();
        Self {
            storage: TensorStorage::Gpu(buffer),
            shape,
            device,
            requires_grad: false,
            grad: None,
        }
    }

    /// Create a scalar tensor from a single value
    pub fn from_scalar(value: T) -> Self {
        Self::from_array(ArrayD::from_elem(IxDyn(&[]), value))
    }

    /// Create a tensor from a vector of data with specified shape
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        let total_size: usize = shape.iter().product();
        if data.len() != total_size {
            return Err(TensorError::invalid_shape_simple(format!(
                "Data length {} doesn't match shape {:?} (size {})",
                data.len(),
                shape,
                total_size
            )));
        }

        let array = ArrayD::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;

        Ok(Self::from_array(array))
    }

    /// Create a tensor filled with a specific value
    pub fn full(shape: &[usize], value: T) -> Self
    where
        T: Clone,
    {
        let array = ArrayD::from_elem(IxDyn(shape), value);
        Self {
            storage: TensorStorage::Cpu(array),
            shape: Shape::from_slice(shape),
            device: Device::Cpu,
            requires_grad: false,
            grad: None,
        }
    }

    /// Create an identity matrix tensor
    pub fn eye(n: usize) -> Self
    where
        T: scirs2_core::num_traits::Zero + scirs2_core::num_traits::One + Clone,
    {
        let mut array = ArrayD::zeros(IxDyn(&[n, n]));
        for i in 0..n {
            array[[i, i]] = T::one();
        }
        Self {
            storage: TensorStorage::Cpu(array),
            shape: Shape::from_slice(&[n, n]),
            device: Device::Cpu,
            requires_grad: false,
            grad: None,
        }
    }

    /// Create a tensor with evenly spaced values in a given interval
    pub fn arange(start: T, end: T, step: T) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float
            + scirs2_core::num_traits::ToPrimitive
            + scirs2_core::num_traits::FromPrimitive,
    {
        let start_f = start.to_f64().ok_or_else(|| {
            crate::TensorError::invalid_argument("Invalid start value".to_string())
        })?;
        let end_f = end
            .to_f64()
            .ok_or_else(|| crate::TensorError::invalid_argument("Invalid end value".to_string()))?;
        let step_f = step.to_f64().ok_or_else(|| {
            crate::TensorError::invalid_argument("Invalid step value".to_string())
        })?;

        if step_f == 0.0 {
            return Err(crate::TensorError::invalid_argument(
                "Step cannot be zero".to_string(),
            ));
        }

        let size = ((end_f - start_f) / step_f).ceil() as usize;
        let mut data = Vec::with_capacity(size);

        let mut current = start_f;
        while (step_f > 0.0 && current < end_f) || (step_f < 0.0 && current > end_f) {
            if let Some(val) = T::from_f64(current) {
                data.push(val);
            } else {
                return Err(crate::TensorError::invalid_argument(
                    "Failed to convert value".to_string(),
                ));
            }
            current += step_f;
        }

        let size = data.len();
        Self::from_vec(data, &[size])
    }

    /// Create a tensor with linearly spaced values between start and end
    pub fn linspace(start: T, end: T, steps: usize) -> Result<Self>
    where
        T: scirs2_core::num_traits::Float
            + scirs2_core::num_traits::ToPrimitive
            + scirs2_core::num_traits::FromPrimitive,
    {
        if steps == 0 {
            return Err(crate::TensorError::invalid_argument(
                "Steps must be greater than 0".to_string(),
            ));
        }

        let start_f = start.to_f64().ok_or_else(|| {
            crate::TensorError::invalid_argument("Invalid start value".to_string())
        })?;
        let end_f = end
            .to_f64()
            .ok_or_else(|| crate::TensorError::invalid_argument("Invalid end value".to_string()))?;

        if steps == 1 {
            let val = T::from_f64(start_f).ok_or_else(|| {
                crate::TensorError::invalid_argument("Failed to convert value".to_string())
            })?;
            return Self::from_vec(vec![val], &[1]);
        }

        let step = (end_f - start_f) / (steps - 1) as f64;
        let mut data = Vec::with_capacity(steps);

        for i in 0..steps {
            let val = start_f + step * i as f64;
            let tensor_val = T::from_f64(val).ok_or_else(|| {
                crate::TensorError::invalid_argument("Failed to convert value".to_string())
            })?;
            data.push(tensor_val);
        }

        Self::from_vec(data, &[steps])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `randn` must produce samples from the true standard normal N(0, 1):
    /// sample mean ~ 0 and sample variance ~ 1. The OLD broken implementation
    /// returned a shifted uniform with mean ~ -0.5 and variance ~ 1/3, so we
    /// also assert we are NOT in that regime.
    #[test]
    fn test_randn_is_standard_normal() {
        let n = 100_000;
        let tensor =
            Tensor::<f32>::randn_with_seed(&[n], 0xC0FFEE).expect("randn_with_seed should succeed");
        let data = tensor.as_slice().expect("randn tensor must be contiguous");
        assert_eq!(data.len(), n);

        let sum: f64 = data.iter().map(|&x| x as f64).sum();
        let mean = sum / n as f64;
        let var: f64 = data.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n as f64;

        // True N(0, 1): mean in [-0.05, 0.05], variance in [0.9, 1.1].
        assert!(
            (-0.05..=0.05).contains(&mean),
            "randn sample mean {mean} not in [-0.05, 0.05] (expected ~0 for N(0,1))"
        );
        assert!(
            (0.9..=1.1).contains(&var),
            "randn sample variance {var} not in [0.9, 1.1] (expected ~1 for N(0,1))"
        );

        // Guard against regression to the OLD shifted-uniform bug (mean ~ -0.5).
        assert!(
            (mean - (-0.5)).abs() > 0.1,
            "randn sample mean {mean} matches the OLD broken distribution (~ -0.5)"
        );
    }

    /// A fixed seed must reproduce the exact same draw.
    #[test]
    fn test_randn_with_seed_is_deterministic() {
        let a = Tensor::<f32>::randn_with_seed(&[64], 7).expect("randn_with_seed should succeed");
        let b = Tensor::<f32>::randn_with_seed(&[64], 7).expect("randn_with_seed should succeed");
        assert_eq!(a.as_slice(), b.as_slice());
    }
}
