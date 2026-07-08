//! 3D Fast Fourier Transform operations
//!
//! This module provides 3D FFT implementations including forward FFT and inverse FFT
//! operations with both CPU and GPU acceleration support.

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
        + bytemuck::Zeroable
        + oxifft::Float,
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

            let fft_width =
                Plan::dft_1d(width, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
                    TensorError::InvalidShape {
                        operation: "fft3".to_string(),
                        reason: "Failed to create width FFT plan".to_string(),
                        shape: Some(shape.to_vec()),
                        context: None,
                    }
                })?;
            let fft_height =
                Plan::dft_1d(height, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
                    TensorError::InvalidShape {
                        operation: "fft3".to_string(),
                        reason: "Failed to create height FFT plan".to_string(),
                        shape: Some(shape.to_vec()),
                        context: None,
                    }
                })?;
            let fft_depth =
                Plan::dft_1d(depth, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
                    TensorError::InvalidShape {
                        operation: "fft3".to_string(),
                        reason: "Failed to create depth FFT plan".to_string(),
                        shape: Some(shape.to_vec()),
                        context: None,
                    }
                })?;

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
                            let mut row_input = volume_data[row_start..row_end].to_vec();
                            let mut row_output = vec![Complex::zero(); width];
                            fft_width.execute(
                                to_oxifft_complex(&row_input),
                                to_oxifft_complex_mut(&mut row_output),
                            );
                            volume_data[row_start..row_end].copy_from_slice(&row_output);
                        }
                    }

                    // Apply FFT along height (second-to-last dimension)
                    for d in 0..depth {
                        for w in 0..width {
                            let mut col_input = Vec::with_capacity(height);
                            for h in 0..height {
                                col_input.push(volume_data[(d * height + h) * width + w]);
                            }
                            let mut col_output = vec![Complex::zero(); height];
                            fft_height.execute(
                                to_oxifft_complex(&col_input),
                                to_oxifft_complex_mut(&mut col_output),
                            );
                            for (h, &val) in col_output.iter().enumerate() {
                                volume_data[(d * height + h) * width + w] = val;
                            }
                        }
                    }

                    // Apply FFT along depth (third-to-last dimension)
                    for h in 0..height {
                        for w in 0..width {
                            let mut depth_input = Vec::with_capacity(depth);
                            for d in 0..depth {
                                depth_input.push(volume_data[(d * height + h) * width + w]);
                            }
                            let mut depth_output = vec![Complex::zero(); depth];
                            fft_depth.execute(
                                to_oxifft_complex(&depth_input),
                                to_oxifft_complex_mut(&mut depth_output),
                            );
                            for (d, &val) in depth_output.iter().enumerate() {
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
        + bytemuck::Zeroable
        + oxifft::Float,
    Complex<T>: Default + bytemuck::Pod + bytemuck::Zeroable,
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

            let ifft_width =
                Plan::dft_1d(width, Direction::Backward, Flags::ESTIMATE).ok_or_else(|| {
                    TensorError::InvalidShape {
                        operation: "ifft3".to_string(),
                        reason: "Failed to create width IFFT plan".to_string(),
                        shape: Some(shape.to_vec()),
                        context: None,
                    }
                })?;
            let ifft_height = Plan::dft_1d(height, Direction::Backward, Flags::ESTIMATE)
                .ok_or_else(|| TensorError::InvalidShape {
                    operation: "ifft3".to_string(),
                    reason: "Failed to create height IFFT plan".to_string(),
                    shape: Some(shape.to_vec()),
                    context: None,
                })?;
            let ifft_depth =
                Plan::dft_1d(depth, Direction::Backward, Flags::ESTIMATE).ok_or_else(|| {
                    TensorError::InvalidShape {
                        operation: "ifft3".to_string(),
                        reason: "Failed to create depth IFFT plan".to_string(),
                        shape: Some(shape.to_vec()),
                        context: None,
                    }
                })?;

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
                            let mut row_input = volume_data[row_start..row_end].to_vec();
                            let mut row_output = vec![Complex::zero(); width];
                            ifft_width.execute(
                                to_oxifft_complex(&row_input),
                                to_oxifft_complex_mut(&mut row_output),
                            );

                            // Normalize by width
                            let width_t =
                                T::from(width).expect("width should convert to float type");
                            for val in &mut row_output {
                                *val /= width_t;
                            }

                            volume_data[row_start..row_end].copy_from_slice(&row_output);
                        }
                    }

                    // Apply IFFT along height (second-to-last dimension)
                    for d in 0..depth {
                        for w in 0..width {
                            let mut col_input = Vec::with_capacity(height);
                            for h in 0..height {
                                col_input.push(volume_data[(d * height + h) * width + w]);
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

                            for (h, &val) in col_output.iter().enumerate() {
                                volume_data[(d * height + h) * width + w] = val;
                            }
                        }
                    }

                    // Apply IFFT along depth (third-to-last dimension)
                    for h in 0..height {
                        for w in 0..width {
                            let mut depth_input = Vec::with_capacity(depth);
                            for d in 0..depth {
                                depth_input.push(volume_data[(d * height + h) * width + w]);
                            }
                            let mut depth_output = vec![Complex::zero(); depth];
                            ifft_depth.execute(
                                to_oxifft_complex(&depth_input),
                                to_oxifft_complex_mut(&mut depth_output),
                            );

                            // Normalize by depth
                            let depth_t =
                                T::from(depth).expect("depth should convert to float type");
                            for val in &mut depth_output {
                                *val /= depth_t;
                            }

                            for (d, &val) in depth_output.iter().enumerate() {
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
            let cpu_tensor = input.to_cpu()?;
            ifft3(&cpu_tensor)
        }
    }
}

// End-to-end test for the GPU-resident code path of `ifft3`. The underlying GPU
// IFFT3 kernel is not implemented, so `ifft3`'s GPU arm reads the operand back
// to the host (a real device->host transfer) and delegates to the CPU `ifft3`
// implementation, which is known-correct. This builds a real GPU-resident
// tensor and calls the actual `ifft3` function, verifying the readback +
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
    fn gpu_ifft3_matches_cpu_reference() {
        let input_cpu =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2])
                .expect("test: from_vec should succeed");
        let freq_cpu = fft3(&input_cpu).expect("test: CPU fft3 should succeed");

        let freq_gpu = match freq_cpu.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return, // No GPU adapter available in this environment; skip.
        };

        let expected = ifft3(&freq_cpu).expect("test: CPU ifft3 should succeed");
        let actual = ifft3(&freq_gpu).expect("test: GPU ifft3 should succeed with a real adapter");

        assert_eq!(actual.shape().dims(), expected.shape().dims());
        let actual_data = actual.to_vec().expect("test: to_vec should succeed");
        let expected_data = expected.to_vec().expect("test: to_vec should succeed");
        for (a, e) in actual_data.iter().zip(expected_data.iter()) {
            assert!((a.re - e.re).abs() < 1e-5, "re mismatch: {a:?} vs {e:?}");
            assert!((a.im - e.im).abs() < 1e-5, "im mismatch: {a:?} vs {e:?}");
        }
    }
}
