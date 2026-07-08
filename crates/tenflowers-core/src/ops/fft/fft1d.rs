//! 1D Fast Fourier Transform operations
//!
//! This module provides 1D FFT implementations including forward FFT, inverse FFT,
//! and real FFT operations with both CPU and GPU acceleration support.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use num_complex::Complex;
use oxifft::{Direction, Flags, Plan};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive, Signed, Zero};
use std::fmt::Debug;

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
        + bytemuck::Zeroable
        + oxifft::Float,
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
            let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
                TensorError::InvalidShape {
                    operation: "fft".to_string(),
                    reason: "Failed to create FFT plan".to_string(),
                    shape: Some(shape.to_vec()),
                    context: None,
                }
            })?;

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
                    let mut input_buffer: Vec<Complex<T>> = input_slice[start_idx..end_idx]
                        .iter()
                        .map(|&x| Complex::new(x, T::zero()))
                        .collect();

                    // Prepare output buffer
                    let mut output_buffer = vec![Complex::zero(); n];

                    // Perform FFT - convert to oxifft types
                    plan.execute(
                        to_oxifft_complex(&input_buffer),
                        to_oxifft_complex_mut(&mut output_buffer),
                    );

                    // Copy result to output
                    output_data[start_idx..end_idx].copy_from_slice(&output_buffer);
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
        + bytemuck::Zeroable
        + oxifft::Float,
    Complex<T>: Default + bytemuck::Pod + bytemuck::Zeroable,
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
            let plan = Plan::dft_1d(n, Direction::Backward, Flags::ESTIMATE).ok_or_else(|| {
                TensorError::InvalidShape {
                    operation: "ifft".to_string(),
                    reason: "Failed to create IFFT plan".to_string(),
                    shape: Some(shape.to_vec()),
                    context: None,
                }
            })?;

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
                    let mut input_buffer: Vec<Complex<T>> =
                        input_slice[start_idx..end_idx].to_vec();

                    // Prepare output buffer
                    let mut output_buffer = vec![Complex::zero(); n];

                    // Perform IFFT - convert to oxifft types
                    plan.execute(
                        to_oxifft_complex(&input_buffer),
                        to_oxifft_complex_mut(&mut output_buffer),
                    );

                    // Normalize by 1/N
                    let n_t = T::from(n).expect("n must be convertible to float type");
                    for val in &mut output_buffer {
                        *val /= n_t;
                    }

                    // Copy result to output
                    output_data[start_idx..end_idx].copy_from_slice(&output_buffer);
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
            let cpu_tensor = input.to_cpu()?;
            ifft(&cpu_tensor)
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
        + bytemuck::Zeroable
        + oxifft::Float,
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

            let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
                TensorError::InvalidShape {
                    operation: "rfft".to_string(),
                    reason: "Failed to create RFFT plan".to_string(),
                    shape: Some(shape.to_vec()),
                    context: None,
                }
            })?;

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
                    let mut input_buffer: Vec<Complex<T>> = input_slice[input_start..input_end]
                        .iter()
                        .map(|&x| Complex::new(x, T::zero()))
                        .collect();

                    // Prepare full output buffer
                    let mut full_output = vec![Complex::zero(); n];

                    // Perform FFT - convert to oxifft types
                    plan.execute(
                        to_oxifft_complex(&input_buffer),
                        to_oxifft_complex_mut(&mut full_output),
                    );

                    // Copy only positive frequencies to output
                    output_data[output_start..output_start + output_len]
                        .copy_from_slice(&full_output[..output_len]);
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

// End-to-end test for the GPU-resident code path of `ifft`. The underlying GPU
// IFFT kernel is not implemented, so `ifft`'s GPU arm reads the operand back to
// the host (a real device->host transfer) and delegates to the CPU `ifft`
// implementation, which is known-correct. This builds a real GPU-resident
// tensor and calls the actual `ifft` function, verifying the readback +
// CPU-delegate path produces numerically correct results.
//
// A GPU adapter is not guaranteed to be present in every environment that
// builds with `--features gpu` (e.g. a headless CI runner). `Tensor::to(Device::Gpu(0))`
// surfaces adapter/device creation failures as an honest `Err` rather than
// panicking, so this test attempts the transfer and skips its assertions -
// without failing the suite - if no adapter is available. This mirrors the
// convention used by `ops::einsum::gpu::gpu_delegate_tests`.
#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use super::*;
    use crate::Device;

    #[test]
    fn gpu_ifft_matches_cpu_reference() {
        let input_cpu = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])
            .expect("test: from_vec should succeed");
        let freq_cpu = fft(&input_cpu).expect("test: CPU fft should succeed");

        let freq_gpu = match freq_cpu.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return, // No GPU adapter available in this environment; skip.
        };

        let expected = ifft(&freq_cpu).expect("test: CPU ifft should succeed");
        let actual = ifft(&freq_gpu).expect("test: GPU ifft should succeed with a real adapter");

        assert_eq!(actual.shape().dims(), expected.shape().dims());
        let actual_data = actual.to_vec().expect("test: to_vec should succeed");
        let expected_data = expected.to_vec().expect("test: to_vec should succeed");
        for (a, e) in actual_data.iter().zip(expected_data.iter()) {
            assert!((a.re - e.re).abs() < 1e-5, "re mismatch: {a:?} vs {e:?}");
            assert!((a.im - e.im).abs() < 1e-5, "im mismatch: {a:?} vs {e:?}");
        }
    }
}
