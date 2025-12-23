//! 2D Fast Fourier Transform operations
//!
//! This module provides 2D FFT implementations including forward FFT and inverse FFT
//! operations with both CPU and GPU acceleration support.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use rustfft::{num_complex::Complex, FftPlanner};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive, Signed, Zero};
use std::fmt::Debug;

use super::fft1d::fft;

// GPU FFT kernels are not yet implemented, using CPU fallbacks

/// 2D FFT along the last two axes
pub fn fft2<T>(input: &Tensor<T>) -> Result<Tensor<Complex<T>>>
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

            if ndim < 2 {
                return Err(TensorError::InvalidShape {
                    operation: "fft2".to_string(),
                    reason: "FFT2 requires at least 2D input".to_string(),
                    shape: Some(shape.to_vec()),
                    context: None,
                });
            }

            let height = shape[ndim - 2];
            let width = shape[ndim - 1];

            // First, apply FFT along the last axis (width)
            let _fft_last = fft(input)?;

            // Now we need to apply FFT along the second-to-last axis (height)
            // This requires transposing the last two dimensions, applying FFT, and transposing back

            // For now, implement a simpler version that processes each row and column
            let mut planners = (FftPlanner::new(), FftPlanner::new());
            let fft_width = planners.0.plan_fft_forward(width);
            let fft_height = planners.1.plan_fft_forward(height);

            // Calculate the number of 2D slices to process
            let total_elements: usize = shape.iter().product();
            let elements_per_slice = height * width;
            let num_slices = total_elements / elements_per_slice;

            // Convert input to complex and prepare output
            let mut output_data = vec![Complex::zero(); total_elements];

            if let Some(input_slice) = arr.as_slice() {
                for slice_idx in 0..num_slices {
                    let slice_start = slice_idx * elements_per_slice;

                    // Create a temporary buffer for this 2D slice
                    let mut slice_data: Vec<Complex<T>> = input_slice
                        [slice_start..slice_start + elements_per_slice]
                        .iter()
                        .map(|&x| Complex::new(x, T::zero()))
                        .collect();

                    // Apply FFT along rows (width dimension)
                    for row in 0..height {
                        let row_start = row * width;
                        let row_end = row_start + width;
                        let mut row_buffer = slice_data[row_start..row_end].to_vec();
                        fft_width.process(&mut row_buffer);
                        slice_data[row_start..row_end].copy_from_slice(&row_buffer);
                    }

                    // Apply FFT along columns (height dimension)
                    for col in 0..width {
                        let mut col_buffer = Vec::with_capacity(height);
                        for row in 0..height {
                            col_buffer.push(slice_data[row * width + col]);
                        }
                        fft_height.process(&mut col_buffer);
                        for (row, &val) in col_buffer.iter().enumerate() {
                            slice_data[row * width + col] = val;
                        }
                    }

                    // Copy result back to output
                    output_data[slice_start..slice_start + elements_per_slice]
                        .copy_from_slice(&slice_data);
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
            // GPU FFT2 not yet implemented, fallback to CPU
            let cpu_tensor = input.to_cpu()?;
            fft2(&cpu_tensor)
        }
    }
}

/// 2D inverse FFT along the last two axes
pub fn ifft2<T>(input: &Tensor<Complex<T>>) -> Result<Tensor<Complex<T>>>
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

            if ndim < 2 {
                return Err(TensorError::InvalidShape {
                    operation: "ifft2".to_string(),
                    reason: "IFFT2 requires at least 2D input".to_string(),
                    shape: Some(shape.to_vec()),
                    context: None,
                });
            }

            let height = shape[ndim - 2];
            let width = shape[ndim - 1];

            let mut planners = (FftPlanner::new(), FftPlanner::new());
            let ifft_width = planners.0.plan_fft_inverse(width);
            let ifft_height = planners.1.plan_fft_inverse(height);

            // Calculate the number of 2D slices to process
            let total_elements: usize = shape.iter().product();
            let elements_per_slice = height * width;
            let num_slices = total_elements / elements_per_slice;

            // Prepare output
            let mut output_data = vec![Complex::zero(); total_elements];

            if let Some(input_slice) = arr.as_slice() {
                for slice_idx in 0..num_slices {
                    let slice_start = slice_idx * elements_per_slice;

                    // Create a temporary buffer for this 2D slice
                    let mut slice_data =
                        input_slice[slice_start..slice_start + elements_per_slice].to_vec();

                    // Apply IFFT along rows (width dimension)
                    for row in 0..height {
                        let row_start = row * width;
                        let row_end = row_start + width;
                        let mut row_buffer = slice_data[row_start..row_end].to_vec();
                        ifft_width.process(&mut row_buffer);

                        // Normalize by width
                        let width_t = T::from(width).unwrap();
                        for val in &mut row_buffer {
                            *val = *val / width_t;
                        }

                        slice_data[row_start..row_end].copy_from_slice(&row_buffer);
                    }

                    // Apply IFFT along columns (height dimension)
                    for col in 0..width {
                        let mut col_buffer = Vec::with_capacity(height);
                        for row in 0..height {
                            col_buffer.push(slice_data[row * width + col]);
                        }
                        ifft_height.process(&mut col_buffer);

                        // Normalize by height
                        let height_t = T::from(height).unwrap();
                        for val in &mut col_buffer {
                            *val = *val / height_t;
                        }

                        for (row, &val) in col_buffer.iter().enumerate() {
                            slice_data[row * width + col] = val;
                        }
                    }

                    // Copy result back to output
                    output_data[slice_start..slice_start + elements_per_slice]
                        .copy_from_slice(&slice_data);
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
            // GPU IFFT2 not yet implemented
            Err(TensorError::unsupported_operation_simple(
                "GPU IFFT2 not yet implemented".to_string(),
            ))
        }
    }
}
