//! GPU-accelerated image rotation transform
//!
//! This module provides GPU-accelerated image rotation using WGPU compute shaders
//! for data augmentation and geometric transformations.

use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::Transform;

#[cfg(feature = "gpu")]
use super::context::GpuContext;

#[cfg(feature = "gpu")]
use scirs2_core::random::RngExt;

/// GPU-accelerated rotation transform
#[cfg(feature = "gpu")]
pub struct GpuRotation {
    angle_range: (f32, f32),
    context: Arc<GpuContext>,
}

#[cfg(feature = "gpu")]
impl GpuRotation {
    /// Create a new GPU rotation transform
    pub fn new(angle_range: (f32, f32), context: Arc<GpuContext>) -> Result<Self> {
        Ok(Self {
            angle_range,
            context,
        })
    }

    /// Apply random rotation to a C×H×W image tensor with bilinear sampling.
    ///
    /// Samples a random angle uniformly from `angle_range` (degrees), converts to
    /// radians, builds the inverse rotation matrix about the image centre, and maps
    /// each output pixel back to a source coordinate with bilinear interpolation.
    /// Source coordinates outside the image boundary yield 0.0 (zero-fill).
    ///
    /// A native WGPU compute-shader path is deferred; this CPU path is
    /// mathematically correct and used as the fallback.
    pub async fn rotate_tensor(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let shape = input.shape().dims();

        if shape.len() != 3 {
            return Err(TensorError::invalid_argument(
                "GpuRotation: expected 3D tensor (C×H×W)".to_string(),
            ));
        }

        let (channels, height, width) = (shape[0], shape[1], shape[2]);

        let data = input.as_slice().ok_or_else(|| {
            TensorError::invalid_argument("GpuRotation: cannot access tensor data".to_string())
        })?;

        // Sample angle uniformly from [min, max] (degrees → radians).
        let (angle_min, angle_max) = self.angle_range;
        let mut rng = scirs2_core::random::rng();
        let angle_deg = if (angle_max - angle_min).abs() < f32::EPSILON {
            angle_min
        } else {
            rng.random_range(angle_min..=angle_max)
        };
        let angle_rad = angle_deg.to_radians();

        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        // Image centre (fractional).
        let cx = (width as f32 - 1.0) * 0.5;
        let cy = (height as f32 - 1.0) * 0.5;

        // Bilinear sample helper (closure over `data`, `height`, `width`, `channels`).
        let sample_bilinear = |c: usize, sy: f32, sx: f32| -> f32 {
            if sx < 0.0 || sy < 0.0 || sx > (width as f32 - 1.0) || sy > (height as f32 - 1.0) {
                return 0.0;
            }
            let x0 = sx.floor() as usize;
            let y0 = sy.floor() as usize;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;
            let idx = |r: usize, col: usize| c * height * width + r * width + col;
            let p00 = data[idx(y0, x0)];
            let p01 = data[idx(y0, x1)];
            let p10 = data[idx(y1, x0)];
            let p11 = data[idx(y1, x1)];
            p00 * (1.0 - fx) * (1.0 - fy)
                + p01 * fx * (1.0 - fy)
                + p10 * (1.0 - fx) * fy
                + p11 * fx * fy
        };

        let mut out = vec![0.0f32; channels * height * width];
        for c in 0..channels {
            for oy in 0..height {
                for ox in 0..width {
                    // Translate to centre, apply inverse rotation, translate back.
                    let dx = ox as f32 - cx;
                    let dy = oy as f32 - cy;
                    let sx = cos_a * dx + sin_a * dy + cx;
                    let sy = -sin_a * dx + cos_a * dy + cy;
                    out[c * height * width + oy * width + ox] = sample_bilinear(c, sy, sx);
                }
            }
        }

        Tensor::from_vec(out, &[channels, height, width])
    }
}

#[cfg(feature = "gpu")]
impl Transform<f32> for GpuRotation {
    fn apply(&self, sample: (Tensor<f32>, Tensor<f32>)) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let (image_tensor, label_tensor) = sample;
        let rotated_tensor = pollster::block_on(self.rotate_tensor(&image_tensor))?;
        Ok((rotated_tensor, label_tensor))
    }
}

/// CPU fallback rotation transform when GPU is not available
#[cfg(not(feature = "gpu"))]
pub struct GpuRotation;

#[cfg(not(feature = "gpu"))]
impl GpuRotation {
    /// Create a new rotation transform (fallback to CPU)
    pub fn new(_angle_range: (f32, f32), _context: ()) -> Result<Self> {
        Err(TensorError::unsupported_operation_simple(
            "GPU transforms require 'gpu' feature to be enabled".to_string(),
        ))
    }
}
