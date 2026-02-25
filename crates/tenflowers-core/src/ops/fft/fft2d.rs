//! 2D Fast Fourier Transform operations
//!
//! This module provides 2D FFT implementations including forward FFT and inverse FFT
//! operations with both CPU and GPU acceleration support.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use num_complex::Complex;
use oxifft::{Direction, Flags, Plan};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive, Signed, Zero};
use std::fmt::Debug;

use super::fft1d::fft;

// GPU FFT kernels are not yet implemented, using CPU fallbacks

/// Convert num_complex slice to oxifft Complex slice
/// Both types have identical #[repr(C)] memory layout, making this conversion safe
#[inline]
fn to_oxifft_complex<T: oxifft::Float>(data: &[Complex<T>]) -> &[oxifft::kernel::Complex<T>] {
    // Safety: Both num_complex::Complex and oxifft::Complex have #[repr(C)] layout
    // with identical memory representation (re: T, im: T)
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const oxifft::kernel::Complex<T>,
            data.len(),
        )
    }
}

/// Convert num_complex mutable slice to oxifft Complex mutable slice
/// Both types have identical #[repr(C)] memory layout, making this conversion safe
#[inline]
fn to_oxifft_complex_mut<T: oxifft::Float>(
    data: &mut [Complex<T>],
) -> &mut [oxifft::kernel::Complex<T>] {
    // Safety: Both num_complex::Complex and oxifft::Complex have #[repr(C)] layout
    // with identical memory representation (re: T, im: T)
    unsafe {
        std::slice::from_raw_parts_mut(
            data.as_mut_ptr() as *mut oxifft::kernel::Complex<T>,
            data.len(),
        )
    }
}

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
        + bytemuck::Zeroable
        + oxifft::Float,
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
            let fft_width =
                Plan::dft_1d(width, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
                    TensorError::InvalidShape {
                        operation: "fft2".to_string(),
                        reason: "Failed to create width FFT plan".to_string(),
                        shape: Some(shape.to_vec()),
                        context: None,
                    }
                })?;
            let fft_height =
                Plan::dft_1d(height, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
                    TensorError::InvalidShape {
                        operation: "fft2".to_string(),
                        reason: "Failed to create height FFT plan".to_string(),
                        shape: Some(shape.to_vec()),
                        context: None,
                    }
                })?;

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
                        let mut row_input = slice_data[row_start..row_end].to_vec();
                        let mut row_output = vec![Complex::zero(); width];
                        fft_width.execute(
                            to_oxifft_complex(&row_input),
                            to_oxifft_complex_mut(&mut row_output),
                        );
                        slice_data[row_start..row_end].copy_from_slice(&row_output);
                    }

                    // Apply FFT along columns (height dimension)
                    for col in 0..width {
                        let mut col_input = Vec::with_capacity(height);
                        for row in 0..height {
                            col_input.push(slice_data[row * width + col]);
                        }
                        let mut col_output = vec![Complex::zero(); height];
                        fft_height.execute(
                            to_oxifft_complex(&col_input),
                            to_oxifft_complex_mut(&mut col_output),
                        );
                        for (row, &val) in col_output.iter().enumerate() {
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
        + bytemuck::Zeroable
        + oxifft::Float,
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

            let ifft_width =
                Plan::dft_1d(width, Direction::Backward, Flags::ESTIMATE).ok_or_else(|| {
                    TensorError::InvalidShape {
                        operation: "ifft2".to_string(),
                        reason: "Failed to create width IFFT plan".to_string(),
                        shape: Some(shape.to_vec()),
                        context: None,
                    }
                })?;
            let ifft_height = Plan::dft_1d(height, Direction::Backward, Flags::ESTIMATE)
                .ok_or_else(|| TensorError::InvalidShape {
                    operation: "ifft2".to_string(),
                    reason: "Failed to create height IFFT plan".to_string(),
                    shape: Some(shape.to_vec()),
                    context: None,
                })?;

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
                        let mut row_input = slice_data[row_start..row_end].to_vec();
                        let mut row_output = vec![Complex::zero(); width];
                        ifft_width.execute(
                            to_oxifft_complex(&row_input),
                            to_oxifft_complex_mut(&mut row_output),
                        );

                        // Normalize by width
                        let width_t = T::from(width).expect("width should convert to float type");
                        for val in &mut row_output {
                            *val /= width_t;
                        }

                        slice_data[row_start..row_end].copy_from_slice(&row_output);
                    }

                    // Apply IFFT along columns (height dimension)
                    for col in 0..width {
                        let mut col_input = Vec::with_capacity(height);
                        for row in 0..height {
                            col_input.push(slice_data[row * width + col]);
                        }
                        let mut col_output = vec![Complex::zero(); height];
                        ifft_height.execute(
                            to_oxifft_complex(&col_input),
                            to_oxifft_complex_mut(&mut col_output),
                        );

                        // Normalize by height
                        let height_t =
                            T::from(height).expect("height should convert to float type");
                        for val in &mut col_output {
                            *val /= height_t;
                        }

                        for (row, &val) in col_output.iter().enumerate() {
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
