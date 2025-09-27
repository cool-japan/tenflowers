//! 1D Fast Fourier Transform operations
//!
//! This module provides 1D FFT implementations including forward FFT, inverse FFT,
//! and real FFT operations with both CPU and GPU acceleration support.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use num_traits::{Float, FromPrimitive, Signed, Zero};
use rustfft::{num_complex::Complex, FftPlanner};
use scirs2_autograd::ndarray::{ArrayD, IxDyn};
use std::fmt::Debug;

// GPU FFT kernels are not yet implemented, using CPU fallbacks

/// Compute 1D FFT along the last axis
pub fn fft<T>(input: &Tensor<T>) -> Result<Tensor<Complex<T>>>
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

            if ndim == 0 {
                return Err(TensorError::InvalidShape {
                    operation: "fft".to_string(),
                    reason: "FFT requires at least 1D input".to_string(),
                    shape: Some(shape.to_vec()),
                    context: None,
                });
            }

            let n = shape[ndim - 1];
            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(n);

            // Calculate the number of FFTs to perform
            let total_elements: usize = shape.iter().product();
            let num_ffts = total_elements / n;

            // Prepare output
            let mut output_data = vec![Complex::zero(); total_elements];

            // Convert input data
            if let Some(input_slice) = arr.as_slice() {
                // Process each 1D slice along the last axis
                for i in 0..num_ffts {
                    let start_idx = i * n;
                    let end_idx = (i + 1) * n;

                    // Convert real input to complex
                    let mut buffer: Vec<Complex<T>> = input_slice[start_idx..end_idx]
                        .iter()
                        .map(|&x| Complex::new(x, T::zero()))
                        .collect();

                    // Perform FFT
                    fft.process(&mut buffer);

                    // Copy result to output
                    output_data[start_idx..end_idx].copy_from_slice(&buffer);
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
            // GPU FFT not yet implemented, fallback to CPU
            let cpu_tensor = input.to_cpu()?;
            fft(&cpu_tensor)
        }
    }
}

/// Compute 1D inverse FFT along the last axis
pub fn ifft<T>(input: &Tensor<Complex<T>>) -> Result<Tensor<Complex<T>>>
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

            if ndim == 0 {
                return Err(TensorError::InvalidShape {
                    operation: "ifft".to_string(),
                    reason: "IFFT requires at least 1D input".to_string(),
                    shape: Some(shape.to_vec()),
                    context: None,
                });
            }

            let n = shape[ndim - 1];
            let mut planner = FftPlanner::new();
            let ifft = planner.plan_fft_inverse(n);

            // Calculate the number of IFFTs to perform
            let total_elements: usize = shape.iter().product();
            let num_iffts = total_elements / n;

            // Prepare output
            let mut output_data = vec![Complex::zero(); total_elements];

            // Convert input data
            if let Some(input_slice) = arr.as_slice() {
                // Process each 1D slice along the last axis
                for i in 0..num_iffts {
                    let start_idx = i * n;
                    let end_idx = (i + 1) * n;

                    // Copy input to buffer
                    let mut buffer: Vec<Complex<T>> = input_slice[start_idx..end_idx].to_vec();

                    // Perform IFFT
                    ifft.process(&mut buffer);

                    // Normalize by 1/N
                    let n_t = T::from(n).unwrap();
                    for val in &mut buffer {
                        *val = *val / n_t;
                    }

                    // Copy result to output
                    output_data[start_idx..end_idx].copy_from_slice(&buffer);
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
            // GPU IFFT not yet implemented
            Err(TensorError::unsupported_operation_simple(
                "GPU IFFT not yet implemented".to_string(),
            ))
        }
    }
}

/// Compute real FFT (only positive frequencies) along the last axis
pub fn rfft<T>(input: &Tensor<T>) -> Result<Tensor<Complex<T>>>
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

            if ndim == 0 {
                return Err(TensorError::InvalidShape {
                    operation: "rfft".to_string(),
                    reason: "RFFT requires at least 1D input".to_string(),
                    shape: Some(shape.to_vec()),
                    context: None,
                });
            }

            let n = shape[ndim - 1];
            let output_len = n / 2 + 1; // Only positive frequencies for real input

            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(n);

            // Calculate output shape
            let mut output_shape = shape.to_vec();
            output_shape[ndim - 1] = output_len;

            // Calculate the number of FFTs to perform
            let input_total: usize = shape.iter().product();
            let output_total: usize = output_shape.iter().product();
            let num_ffts = input_total / n;

            // Prepare output
            let mut output_data = vec![Complex::zero(); output_total];

            // Convert input data
            if let Some(input_slice) = arr.as_slice() {
                // Process each 1D slice along the last axis
                for i in 0..num_ffts {
                    let input_start = i * n;
                    let input_end = (i + 1) * n;
                    let output_start = i * output_len;

                    // Convert real input to complex
                    let mut buffer: Vec<Complex<T>> = input_slice[input_start..input_end]
                        .iter()
                        .map(|&x| Complex::new(x, T::zero()))
                        .collect();

                    // Perform FFT
                    fft.process(&mut buffer);

                    // Copy only positive frequencies to output
                    output_data[output_start..output_start + output_len]
                        .copy_from_slice(&buffer[..output_len]);
                }

                // Create output tensor
                let output_array = ArrayD::from_shape_vec(IxDyn(&output_shape), output_data)
                    .map_err(|e| TensorError::InvalidShape {
                        operation: "fft".to_string(),
                        reason: e.to_string(),
                        shape: None,
                        context: None,
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
            // GPU RFFT not yet implemented, fallback to CPU
            let cpu_tensor = input.to_cpu()?;
            rfft(&cpu_tensor)
        }
    }
}
