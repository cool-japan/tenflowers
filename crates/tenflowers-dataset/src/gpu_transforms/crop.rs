//! GPU-accelerated random crop transform
//!
//! This module provides GPU-accelerated random cropping using WGPU compute shaders
//! for data augmentation and image preprocessing.

use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::Transform;

#[cfg(feature = "gpu")]
use super::context::GpuContext;

#[cfg(feature = "gpu")]
use scirs2_core::random::RngExt;

/// GPU-accelerated random crop transform
#[cfg(feature = "gpu")]
pub struct GpuRandomCrop {
    output_width: u32,
    output_height: u32,
    context: Arc<GpuContext>,
}

#[cfg(feature = "gpu")]
impl GpuRandomCrop {
    /// Create a new GPU random crop transform
    pub fn new(output_width: u32, output_height: u32, context: Arc<GpuContext>) -> Result<Self> {
        Ok(Self {
            output_width,
            output_height,
            context,
        })
    }

    /// Apply random crop to image tensor using GPU.
    ///
    /// This is a CPU-based implementation that performs the actual random crop logic.
    /// A true GPU (WGPU compute shader) path requires staging buffers and async
    /// readback which is deferred to a future kernel; this path produces correct output.
    pub async fn crop_tensor(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = input.shape().dims();

        if shape.len() < 2 {
            return Err(TensorError::invalid_argument(
                "GpuRandomCrop: input tensor must have at least 2 dimensions (H×W or C×H×W)"
                    .to_string(),
            ));
        }

        let (channels, in_height, in_width) = if shape.len() == 3 {
            (shape[0], shape[1], shape[2])
        } else {
            (1, shape[0], shape[1])
        };

        let out_h = self.output_height as usize;
        let out_w = self.output_width as usize;

        if out_h > in_height || out_w > in_width {
            return Err(TensorError::invalid_argument(format!(
                "GpuRandomCrop: requested output {}×{} is larger than input {}×{}",
                out_h, out_w, in_height, in_width
            )));
        }

        let data = input.as_slice().ok_or_else(|| {
            TensorError::invalid_argument("GpuRandomCrop: cannot access tensor data".to_string())
        })?;

        let max_y = in_height - out_h;
        let max_x = in_width - out_w;

        let mut rng = scirs2_core::random::rng();
        let origin_y = if max_y > 0 {
            rng.random_range(0..=max_y)
        } else {
            0
        };
        let origin_x = if max_x > 0 {
            rng.random_range(0..=max_x)
        } else {
            0
        };

        let mut out_data = Vec::with_capacity(channels * out_h * out_w);
        for c in 0..channels {
            for y in 0..out_h {
                for x in 0..out_w {
                    let src_y = origin_y + y;
                    let src_x = origin_x + x;
                    let idx = c * in_height * in_width + src_y * in_width + src_x;
                    out_data.push(if idx < data.len() { data[idx] } else { 0.0f32 });
                }
            }
        }

        let out_shape: &[usize] = if shape.len() == 3 {
            &[channels, out_h, out_w]
        } else {
            &[out_h, out_w]
        };

        Tensor::from_vec(out_data, out_shape)
    }
}

#[cfg(feature = "gpu")]
impl Transform<f32> for GpuRandomCrop {
    fn apply(&self, sample: (Tensor<f32>, Tensor<f32>)) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let (image_tensor, label_tensor) = sample;
        let cropped_tensor = pollster::block_on(self.crop_tensor(&image_tensor))?;
        Ok((cropped_tensor, label_tensor))
    }
}

/// CPU fallback crop transform when GPU is not available
#[cfg(not(feature = "gpu"))]
pub struct GpuRandomCrop;

#[cfg(not(feature = "gpu"))]
impl GpuRandomCrop {
    /// Create a new crop transform (fallback to CPU)
    pub fn new(_output_width: u32, _output_height: u32, _context: ()) -> Result<Self> {
        Err(TensorError::unsupported_operation_simple(
            "GPU transforms require 'gpu' feature to be enabled".to_string(),
        ))
    }
}
