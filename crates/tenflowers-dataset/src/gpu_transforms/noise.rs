//! GPU-accelerated Gaussian noise transform
//!
//! This module provides GPU-accelerated Gaussian noise addition using WGPU compute shaders
//! for data augmentation and noise injection.

use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::Transform;

#[cfg(feature = "gpu")]
use super::context::GpuContext;

#[cfg(feature = "gpu")]
use scirs2_core::RngExt;

/// GPU-accelerated Gaussian noise transform
#[cfg(feature = "gpu")]
pub struct GpuGaussianNoise {
    mean: f32,
    std_dev: f32,
    context: Arc<GpuContext>,
}

#[cfg(feature = "gpu")]
impl GpuGaussianNoise {
    /// Create a new GPU Gaussian noise transform
    pub fn new(mean: f32, std_dev: f32, context: Arc<GpuContext>) -> Result<Self> {
        Ok(Self {
            mean,
            std_dev,
            context,
        })
    }

    /// Apply Gaussian noise to image tensor.
    ///
    /// Samples from N(mean, std_dev) using the Box-Muller transform and adds the
    /// noise element-wise.  A native WGPU compute-shader path is deferred; this
    /// CPU path produces mathematically correct output.
    pub async fn add_noise_tensor(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let data = input.as_slice().ok_or_else(|| {
            TensorError::invalid_argument("GpuGaussianNoise: cannot access tensor data".to_string())
        })?;

        let mut rng = scirs2_core::random::rng();
        let mut noisy: Vec<f32> = Vec::with_capacity(data.len());

        // Box-Muller: consume pairs (u1, u2) → (z0, z1)
        let mut iter = data.iter().peekable();
        while let Some(&v0) = iter.next() {
            let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE); // avoid ln(0)
            let u2: f32 = rng.random::<f32>();
            let mag = (-2.0_f32 * u1.ln()).sqrt();
            let z0 = mag * (2.0 * std::f32::consts::PI * u2).cos();
            let z1 = mag * (2.0 * std::f32::consts::PI * u2).sin();

            noisy.push(v0 + self.mean + self.std_dev * z0);
            if let Some(&v1) = iter.next() {
                noisy.push(v1 + self.mean + self.std_dev * z1);
            }
        }

        Tensor::from_vec(noisy, input.shape().dims())
    }
}

#[cfg(feature = "gpu")]
impl Transform<f32> for GpuGaussianNoise {
    fn apply(&self, sample: (Tensor<f32>, Tensor<f32>)) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let (image_tensor, label_tensor) = sample;
        let noisy_tensor = pollster::block_on(self.add_noise_tensor(&image_tensor))?;
        Ok((noisy_tensor, label_tensor))
    }
}

/// CPU fallback noise transform when GPU is not available
#[cfg(not(feature = "gpu"))]
pub struct GpuGaussianNoise;

#[cfg(not(feature = "gpu"))]
impl GpuGaussianNoise {
    /// Create a new noise transform (fallback to CPU)
    pub fn new(_mean: f32, _std_dev: f32, _context: ()) -> Result<Self> {
        Err(TensorError::unsupported_operation_simple(
            "GPU transforms require 'gpu' feature to be enabled".to_string(),
        ))
    }
}
