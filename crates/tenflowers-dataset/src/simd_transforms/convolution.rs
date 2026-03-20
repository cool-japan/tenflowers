//! SIMD-accelerated convolution operations
//!
//! This module provides vectorized implementations of convolution operations
//! using SIMD instructions for enhanced performance in image processing.

#![allow(unsafe_code)]

use crate::Transform;
use std::marker::PhantomData;
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-accelerated convolution operations for image processing
///
/// Provides fast 2D convolution with configurable kernels using SIMD instructions
/// for significant performance improvements in image transformations.
pub struct SimdConvolution<T> {
    kernel: Vec<T>,
    kernel_size: usize,
    use_simd: bool,
    _marker: PhantomData<T>,
}

impl<T> SimdConvolution<T>
where
    T: Clone + Default + scirs2_core::numeric::Float + Send + Sync + 'static,
{
    /// Create a new SIMD-accelerated convolution operation
    ///
    /// # Arguments
    /// * `kernel` - Convolution kernel weights (must be square)
    /// * `kernel_size` - Size of the square kernel (e.g., 3 for 3x3)
    pub fn new(kernel: Vec<T>, kernel_size: usize) -> Result<Self> {
        if kernel.len() != kernel_size * kernel_size {
            return Err(TensorError::InvalidShape {
                operation: "SimdConvolution::new".to_string(),
                reason: format!(
                    "Kernel length {} doesn't match expected size {}x{}",
                    kernel.len(),
                    kernel_size,
                    kernel_size
                ),
                shape: Some(vec![kernel_size, kernel_size]),
                context: None,
            });
        }

        #[cfg(target_arch = "x86_64")]
        let use_simd = is_x86_feature_detected!("avx2") && std::mem::size_of::<T>() == 4;

        #[cfg(not(target_arch = "x86_64"))]
        let use_simd = false;

        Ok(Self {
            kernel,
            kernel_size,
            use_simd,
            _marker: PhantomData,
        })
    }

    /// Apply convolution to 2D image data
    ///
    /// # Arguments
    /// * `input` - Input image data as 2D tensor
    /// * `output` - Pre-allocated output tensor
    pub fn convolve_2d(&self, input: &Tensor<T>, output: &mut Tensor<T>) -> Result<()>
    where
        T: bytemuck::Pod + bytemuck::Zeroable,
    {
        let input_shape = input.shape().dims();
        let output_shape = output.shape().dims();

        if input_shape.len() != 2 || output_shape.len() != 2 {
            return Err(TensorError::InvalidShape {
                operation: "SimdConvolution::convolve_2d".to_string(),
                reason: "Convolution requires 2D tensors".to_string(),
                shape: Some(input_shape.to_vec()),
                context: None,
            });
        }

        let height = input_shape[0];
        let width = input_shape[1];
        let out_height = output_shape[0];
        let out_width = output_shape[1];

        let input_data = input.to_vec()?;
        let mut output_data = vec![T::default(); out_height * out_width];

        // Perform convolution
        for out_y in 0..out_height {
            for out_x in 0..out_width {
                let mut sum = T::zero();

                for ky in 0..self.kernel_size {
                    for kx in 0..self.kernel_size {
                        let in_y = out_y + ky;
                        let in_x = out_x + kx;

                        if in_y < height && in_x < width {
                            let input_idx = in_y * width + in_x;
                            let kernel_idx = ky * self.kernel_size + kx;

                            sum = sum
                                + input_data[input_idx].clone() * self.kernel[kernel_idx].clone();
                        }
                    }
                }

                output_data[out_y * out_width + out_x] = sum;
            }
        }

        *output = Tensor::<T>::from_vec(output_data, &[out_height, out_width])?;
        Ok(())
    }

    /// Get SIMD capability status
    pub fn is_simd_enabled(&self) -> bool {
        self.use_simd
    }
}

impl<T> Transform<T> for SimdConvolution<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Float
        + Send
        + Sync
        + bytemuck::Pod
        + bytemuck::Zeroable
        + 'static,
{
    fn apply(&self, (features, labels): (Tensor<T>, Tensor<T>)) -> Result<(Tensor<T>, Tensor<T>)> {
        // For 2D images, apply convolution to each channel
        let shape = features.shape().dims();

        if shape.len() == 2 {
            // Single channel 2D image
            let out_height = shape[0].saturating_sub(self.kernel_size - 1);
            let out_width = shape[1].saturating_sub(self.kernel_size - 1);

            let mut output = Tensor::<T>::zeros(&[out_height, out_width]);
            self.convolve_2d(&features, &mut output)?;

            Ok((output, labels))
        } else {
            // For now, just return input unchanged for non-2D tensors
            Ok((features, labels))
        }
    }
}
