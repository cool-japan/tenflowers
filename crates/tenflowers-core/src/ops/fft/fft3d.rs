//! 3D Fast Fourier Transform operations
//!
//! This module provides 3D FFT implementations including forward FFT and inverse FFT
//! operations with both CPU and GPU acceleration support.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use rustfft::{num_complex::Complex, FftPlanner};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive, Signed, Zero};
use std::fmt::Debug;

// GPU FFT kernels are not yet implemented, using CPU fallbacks

/// 3D FFT along the last three axes
pub fn fft3<T>(input: &Tensor<T>) -> Result<Tensor<Complex<T>>>
where
    T: Float
        + Send
        + Sync
        + 'static
        + FromPrimitive
        + Signed
        + Debug
        + Default
        + bytemuck::Pod
        + bytemuck::Zeroable,
    Complex<T>: Default,
{
    match &input.storage {
        TensorStorage::Cpu(arr) => {
            let shape = arr.shape();
            let ndim = shape.len();

            if ndim < 3 {
                return Err(TensorError::InvalidShape {
                    operation: "fft3".to_string(),
                    reason: "FFT3 requires at least 3D input".to_string(),
                    shape: Some(shape.to_vec()),
                    context: None,
                });
            }

            let depth = shape[ndim - 3];
            let height = shape[ndim - 2];
            let width = shape[ndim - 1];

            let mut planners = (FftPlanner::new(), FftPlanner::new(), FftPlanner::new());
            let fft_width = planners.0.plan_fft_forward(width);
            let fft_height = planners.1.plan_fft_forward(height);
            let fft_depth = planners.2.plan_fft_forward(depth);

            // Calculate the number of 3D volumes to process
            let total_elements: usize = shape.iter().product();
            let elements_per_volume = depth * height * width;
            let num_volumes = total_elements / elements_per_volume;

            // Convert input to complex and prepare output
            let mut output_data = vec![Complex::zero(); total_elements];

            if let Some(input_slice) = arr.as_slice() {
                for volume_idx in 0..num_volumes {
                    let volume_start = volume_idx * elements_per_volume;

                    // Create a temporary buffer for this 3D volume
                    let mut volume_data: Vec<Complex<T>> = input_slice
                        [volume_start..volume_start + elements_per_volume]
                        .iter()
                        .map(|&x| Complex::new(x, T::zero()))
                        .collect();

                    // Apply FFT along width (last dimension)
                    for d in 0..depth {
                        for h in 0..height {
                            let row_start = (d * height + h) * width;
                            let row_end = row_start + width;
                            let mut row_buffer = volume_data[row_start..row_end].to_vec();
                            fft_width.process(&mut row_buffer);
                            volume_data[row_start..row_end].copy_from_slice(&row_buffer);
                        }
                    }

                    // Apply FFT along height (second-to-last dimension)
                    for d in 0..depth {
                        for w in 0..width {
                            let mut col_buffer = Vec::with_capacity(height);
                            for h in 0..height {
                                col_buffer.push(volume_data[(d * height + h) * width + w]);
                            }
                            fft_height.process(&mut col_buffer);
                            for (h, &val) in col_buffer.iter().enumerate() {
                                volume_data[(d * height + h) * width + w] = val;
                            }
                        }
                    }

                    // Apply FFT along depth (third-to-last dimension)
                    for h in 0..height {
                        for w in 0..width {
                            let mut depth_buffer = Vec::with_capacity(depth);
                            for d in 0..depth {
                                depth_buffer.push(volume_data[(d * height + h) * width + w]);
                            }
                            fft_depth.process(&mut depth_buffer);
                            for (d, &val) in depth_buffer.iter().enumerate() {
                                volume_data[(d * height + h) * width + w] = val;
                            }
                        }
                    }

                    // Copy result back to output
                    output_data[volume_start..volume_start + elements_per_volume]
                        .copy_from_slice(&volume_data);
                }

                // Create output tensor
                let output_array =
                    ArrayD::from_shape_vec(IxDyn(shape), output_data).map_err(|e| {
                        TensorError::InvalidShape {
                            operation: "fft".to_string(),
                            reason: e.to_string(),
                            shape: None,
                            context: None,
                        }
                    })?;

                Ok(Tensor::from_array(output_array))
            } else {
                Err(TensorError::unsupported_operation_simple(
                    "Cannot get slice from input array".to_string(),
                ))
            }
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_gpu_buffer) => {
            // GPU FFT3 not yet implemented, fallback to CPU
            let cpu_tensor = input.to_cpu()?;
            fft3(&cpu_tensor)
        }
    }
}

/// 3D inverse FFT along the last three axes
pub fn ifft3<T>(input: &Tensor<Complex<T>>) -> Result<Tensor<Complex<T>>>
where
    T: Float
        + Send
        + Sync
        + 'static
        + FromPrimitive
        + Signed
        + Debug
        + Default
        + bytemuck::Pod
        + bytemuck::Zeroable,
    Complex<T>: Default,
{
    match &input.storage {
        TensorStorage::Cpu(arr) => {
            let shape = arr.shape();
            let ndim = shape.len();

            if ndim < 3 {
                return Err(TensorError::InvalidShape {
                    operation: "ifft3".to_string(),
                    reason: "IFFT3 requires at least 3D input".to_string(),
                    shape: Some(shape.to_vec()),
                    context: None,
                });
            }

            let depth = shape[ndim - 3];
            let height = shape[ndim - 2];
            let width = shape[ndim - 1];

            let mut planners = (FftPlanner::new(), FftPlanner::new(), FftPlanner::new());
            let ifft_width = planners.0.plan_fft_inverse(width);
            let ifft_height = planners.1.plan_fft_inverse(height);
            let ifft_depth = planners.2.plan_fft_inverse(depth);

            // Calculate the number of 3D volumes to process
            let total_elements: usize = shape.iter().product();
            let elements_per_volume = depth * height * width;
            let num_volumes = total_elements / elements_per_volume;

            // Prepare output
            let mut output_data = vec![Complex::zero(); total_elements];

            if let Some(input_slice) = arr.as_slice() {
                for volume_idx in 0..num_volumes {
                    let volume_start = volume_idx * elements_per_volume;

                    // Create a temporary buffer for this 3D volume
                    let mut volume_data =
                        input_slice[volume_start..volume_start + elements_per_volume].to_vec();

                    // Apply IFFT along width (last dimension)
                    for d in 0..depth {
                        for h in 0..height {
                            let row_start = (d * height + h) * width;
                            let row_end = row_start + width;
                            let mut row_buffer = volume_data[row_start..row_end].to_vec();
                            ifft_width.process(&mut row_buffer);

                            // Normalize by width
                            let width_t = T::from(width).unwrap();
                            for val in &mut row_buffer {
                                *val = *val / width_t;
                            }

                            volume_data[row_start..row_end].copy_from_slice(&row_buffer);
                        }
                    }

                    // Apply IFFT along height (second-to-last dimension)
                    for d in 0..depth {
                        for w in 0..width {
                            let mut col_buffer = Vec::with_capacity(height);
                            for h in 0..height {
                                col_buffer.push(volume_data[(d * height + h) * width + w]);
                            }
                            ifft_height.process(&mut col_buffer);

                            // Normalize by height
                            let height_t = T::from(height).unwrap();
                            for val in &mut col_buffer {
                                *val = *val / height_t;
                            }

                            for (h, &val) in col_buffer.iter().enumerate() {
                                volume_data[(d * height + h) * width + w] = val;
                            }
                        }
                    }

                    // Apply IFFT along depth (third-to-last dimension)
                    for h in 0..height {
                        for w in 0..width {
                            let mut depth_buffer = Vec::with_capacity(depth);
                            for d in 0..depth {
                                depth_buffer.push(volume_data[(d * height + h) * width + w]);
                            }
                            ifft_depth.process(&mut depth_buffer);

                            // Normalize by depth
                            let depth_t = T::from(depth).unwrap();
                            for val in &mut depth_buffer {
                                *val = *val / depth_t;
                            }

                            for (d, &val) in depth_buffer.iter().enumerate() {
                                volume_data[(d * height + h) * width + w] = val;
                            }
                        }
                    }

                    // Copy result back to output
                    output_data[volume_start..volume_start + elements_per_volume]
                        .copy_from_slice(&volume_data);
                }

                // Create output tensor
                let output_array =
                    ArrayD::from_shape_vec(IxDyn(shape), output_data).map_err(|e| {
                        TensorError::InvalidShape {
                            operation: "fft".to_string(),
                            reason: e.to_string(),
                            shape: None,
                            context: None,
                        }
                    })?;

                Ok(Tensor::from_array(output_array))
            } else {
                Err(TensorError::unsupported_operation_simple(
                    "Cannot get slice from input array".to_string(),
                ))
            }
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_gpu_buffer) => {
            // GPU IFFT3 not yet implemented
            Err(TensorError::unsupported_operation_simple(
                "GPU IFFT3 not yet implemented".to_string(),
            ))
        }
    }
}
