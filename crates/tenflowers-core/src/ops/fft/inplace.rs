//! In-place FFT operations
//!
//! This module provides in-place FFT implementations that modify the input tensor
//! directly. Each function calls the real oxifft-backed computation and writes the
//! result back into `*input`, so callers always see a transformed tensor.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use num_complex::Complex;
use oxifft::{Direction, Flags, Plan};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive, Signed, Zero};
use std::fmt::Debug;

use super::fft1d::ifft;
use super::fft2d::ifft2;
use super::fft3d::ifft3;

// ---------------------------------------------------------------------------
// Memory-layout helpers (identical repr(C) trick used in fft1d/fft2d/fft3d)
// ---------------------------------------------------------------------------

#[inline]
fn to_oxifft_complex<T: oxifft::Float>(data: &[Complex<T>]) -> &[oxifft::kernel::Complex<T>] {
    // Safety: Both num_complex::Complex and oxifft::Complex have #[repr(C)] layout
    // with identical memory representation (re: T, im: T).
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const oxifft::kernel::Complex<T>,
            data.len(),
        )
    }
}

#[inline]
fn to_oxifft_complex_mut<T: oxifft::Float>(
    data: &mut [Complex<T>],
) -> &mut [oxifft::kernel::Complex<T>] {
    // Safety: same repr(C) guarantee as above.
    unsafe {
        std::slice::from_raw_parts_mut(
            data.as_mut_ptr() as *mut oxifft::kernel::Complex<T>,
            data.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// 1D in-place FFT
// ---------------------------------------------------------------------------

/// In-place 1D forward FFT along the last axis (complex → complex).
///
/// Applies a complex-to-complex DFT in the forward direction (`e^{-2πi·k·n/N}`).
/// The input tensor is replaced with its frequency-domain representation.
pub fn fft_inplace<T>(input: &mut Tensor<Complex<T>>) -> Result<()>
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
                    operation: "fft_inplace".to_string(),
                    reason: "FFT requires at least 1D input".to_string(),
                    shape: Some(shape.to_vec()),
                    context: None,
                });
            }

            let n = shape[ndim - 1];
            let total_elements: usize = shape.iter().product();
            let num_ffts = total_elements / n;

            let plan = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
                TensorError::InvalidShape {
                    operation: "fft_inplace".to_string(),
                    reason: "Failed to create FFT plan".to_string(),
                    shape: Some(shape.to_vec()),
                    context: None,
                }
            })?;

            let input_slice = arr.as_slice().ok_or_else(|| {
                TensorError::unsupported_operation_simple(
                    "fft_inplace: input array is not contiguous".to_string(),
                )
            })?;

            let mut output_data = vec![Complex::zero(); total_elements];

            for i in 0..num_ffts {
                let start = i * n;
                let end = start + n;
                let row_in = input_slice[start..end].to_vec();
                let mut row_out = vec![Complex::zero(); n];
                plan.execute(
                    to_oxifft_complex(&row_in),
                    to_oxifft_complex_mut(&mut row_out),
                );
                output_data[start..end].copy_from_slice(&row_out);
            }

            let owned_shape = shape.to_vec();
            let output_array =
                ArrayD::from_shape_vec(IxDyn(&owned_shape), output_data).map_err(|e| {
                    TensorError::InvalidShape {
                        operation: "fft_inplace".to_string(),
                        reason: e.to_string(),
                        shape: None,
                        context: None,
                    }
                })?;

            *input = Tensor::from_array(output_array);
            Ok(())
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => {
            let mut cpu_complex = input.to_cpu()?;
            fft_inplace(&mut cpu_complex)?;
            *input = cpu_complex;
            Ok(())
        }
    }
}

/// In-place 1D inverse FFT along the last axis (complex → complex).
///
/// Applies the normalised inverse DFT (`1/N · Σ X[k] e^{+2πi·k·n/N}`).
/// The input tensor is replaced with its time-domain reconstruction.
pub fn ifft_inplace<T>(input: &mut Tensor<Complex<T>>) -> Result<()>
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
    let result = ifft(input as &Tensor<Complex<T>>)?;
    *input = result;
    Ok(())
}

// ---------------------------------------------------------------------------
// 2D in-place FFT
// ---------------------------------------------------------------------------

/// In-place 2D forward FFT along the last two axes (complex → complex).
///
/// Applies a separable 2D complex-to-complex DFT: first across rows (width),
/// then across columns (height).
pub fn fft2_inplace<T>(input: &mut Tensor<Complex<T>>) -> Result<()>
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
                    operation: "fft2_inplace".to_string(),
                    reason: "FFT2 requires at least 2D input".to_string(),
                    shape: Some(shape.to_vec()),
                    context: None,
                });
            }

            let height = shape[ndim - 2];
            let width = shape[ndim - 1];
            let total_elements: usize = shape.iter().product();
            let elements_per_slice = height * width;
            let num_slices = total_elements / elements_per_slice;

            let fft_width =
                Plan::dft_1d(width, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
                    TensorError::InvalidShape {
                        operation: "fft2_inplace".to_string(),
                        reason: "Failed to create width FFT plan".to_string(),
                        shape: Some(shape.to_vec()),
                        context: None,
                    }
                })?;
            let fft_height =
                Plan::dft_1d(height, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
                    TensorError::InvalidShape {
                        operation: "fft2_inplace".to_string(),
                        reason: "Failed to create height FFT plan".to_string(),
                        shape: Some(shape.to_vec()),
                        context: None,
                    }
                })?;

            let input_slice = arr.as_slice().ok_or_else(|| {
                TensorError::unsupported_operation_simple(
                    "fft2_inplace: input array is not contiguous".to_string(),
                )
            })?;

            let mut output_data = vec![Complex::zero(); total_elements];

            for slice_idx in 0..num_slices {
                let slice_start = slice_idx * elements_per_slice;
                let mut slice_buf: Vec<Complex<T>> =
                    input_slice[slice_start..slice_start + elements_per_slice].to_vec();

                // Forward FFT along rows (width dimension)
                for row in 0..height {
                    let row_start = row * width;
                    let row_end = row_start + width;
                    let row_in = slice_buf[row_start..row_end].to_vec();
                    let mut row_out = vec![Complex::zero(); width];
                    fft_width.execute(
                        to_oxifft_complex(&row_in),
                        to_oxifft_complex_mut(&mut row_out),
                    );
                    slice_buf[row_start..row_end].copy_from_slice(&row_out);
                }

                // Forward FFT along columns (height dimension)
                for col in 0..width {
                    let col_in: Vec<Complex<T>> = (0..height)
                        .map(|row| slice_buf[row * width + col])
                        .collect();
                    let mut col_out = vec![Complex::zero(); height];
                    fft_height.execute(
                        to_oxifft_complex(&col_in),
                        to_oxifft_complex_mut(&mut col_out),
                    );
                    for (row, &val) in col_out.iter().enumerate() {
                        slice_buf[row * width + col] = val;
                    }
                }

                output_data[slice_start..slice_start + elements_per_slice]
                    .copy_from_slice(&slice_buf);
            }

            let owned_shape = shape.to_vec();
            let output_array =
                ArrayD::from_shape_vec(IxDyn(&owned_shape), output_data).map_err(|e| {
                    TensorError::InvalidShape {
                        operation: "fft2_inplace".to_string(),
                        reason: e.to_string(),
                        shape: None,
                        context: None,
                    }
                })?;

            *input = Tensor::from_array(output_array);
            Ok(())
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => {
            let mut cpu_complex = input.to_cpu()?;
            fft2_inplace(&mut cpu_complex)?;
            *input = cpu_complex;
            Ok(())
        }
    }
}

/// In-place 2D inverse FFT along the last two axes (complex → complex).
///
/// Applies the separable normalised inverse 2D DFT.
pub fn ifft2_inplace<T>(input: &mut Tensor<Complex<T>>) -> Result<()>
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
    let result = ifft2(input as &Tensor<Complex<T>>)?;
    *input = result;
    Ok(())
}

// ---------------------------------------------------------------------------
// 3D in-place FFT
// ---------------------------------------------------------------------------

/// In-place 3D forward FFT along the last three axes (complex → complex).
///
/// Applies a separable 3D complex-to-complex DFT: first along width, then
/// height, then depth.
pub fn fft3_inplace<T>(input: &mut Tensor<Complex<T>>) -> Result<()>
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
                    operation: "fft3_inplace".to_string(),
                    reason: "FFT3 requires at least 3D input".to_string(),
                    shape: Some(shape.to_vec()),
                    context: None,
                });
            }

            let depth = shape[ndim - 3];
            let height = shape[ndim - 2];
            let width = shape[ndim - 1];
            let total_elements: usize = shape.iter().product();
            let elements_per_volume = depth * height * width;
            let num_volumes = total_elements / elements_per_volume;

            let fft_width =
                Plan::dft_1d(width, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
                    TensorError::InvalidShape {
                        operation: "fft3_inplace".to_string(),
                        reason: "Failed to create width FFT plan".to_string(),
                        shape: Some(shape.to_vec()),
                        context: None,
                    }
                })?;
            let fft_height =
                Plan::dft_1d(height, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
                    TensorError::InvalidShape {
                        operation: "fft3_inplace".to_string(),
                        reason: "Failed to create height FFT plan".to_string(),
                        shape: Some(shape.to_vec()),
                        context: None,
                    }
                })?;
            let fft_depth =
                Plan::dft_1d(depth, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
                    TensorError::InvalidShape {
                        operation: "fft3_inplace".to_string(),
                        reason: "Failed to create depth FFT plan".to_string(),
                        shape: Some(shape.to_vec()),
                        context: None,
                    }
                })?;

            let input_slice = arr.as_slice().ok_or_else(|| {
                TensorError::unsupported_operation_simple(
                    "fft3_inplace: input array is not contiguous".to_string(),
                )
            })?;

            let mut output_data = vec![Complex::zero(); total_elements];

            for volume_idx in 0..num_volumes {
                let volume_start = volume_idx * elements_per_volume;
                let mut vol: Vec<Complex<T>> =
                    input_slice[volume_start..volume_start + elements_per_volume].to_vec();

                // Forward FFT along width (last dimension)
                for d in 0..depth {
                    for h in 0..height {
                        let row_start = (d * height + h) * width;
                        let row_end = row_start + width;
                        let row_in = vol[row_start..row_end].to_vec();
                        let mut row_out = vec![Complex::zero(); width];
                        fft_width.execute(
                            to_oxifft_complex(&row_in),
                            to_oxifft_complex_mut(&mut row_out),
                        );
                        vol[row_start..row_end].copy_from_slice(&row_out);
                    }
                }

                // Forward FFT along height (second-to-last dimension)
                for d in 0..depth {
                    for w in 0..width {
                        let col_in: Vec<Complex<T>> = (0..height)
                            .map(|h| vol[(d * height + h) * width + w])
                            .collect();
                        let mut col_out = vec![Complex::zero(); height];
                        fft_height.execute(
                            to_oxifft_complex(&col_in),
                            to_oxifft_complex_mut(&mut col_out),
                        );
                        for (h, &val) in col_out.iter().enumerate() {
                            vol[(d * height + h) * width + w] = val;
                        }
                    }
                }

                // Forward FFT along depth (third-to-last dimension)
                for h in 0..height {
                    for w in 0..width {
                        let depth_in: Vec<Complex<T>> = (0..depth)
                            .map(|d| vol[(d * height + h) * width + w])
                            .collect();
                        let mut depth_out = vec![Complex::zero(); depth];
                        fft_depth.execute(
                            to_oxifft_complex(&depth_in),
                            to_oxifft_complex_mut(&mut depth_out),
                        );
                        for (d, &val) in depth_out.iter().enumerate() {
                            vol[(d * height + h) * width + w] = val;
                        }
                    }
                }

                output_data[volume_start..volume_start + elements_per_volume].copy_from_slice(&vol);
            }

            let owned_shape = shape.to_vec();
            let output_array =
                ArrayD::from_shape_vec(IxDyn(&owned_shape), output_data).map_err(|e| {
                    TensorError::InvalidShape {
                        operation: "fft3_inplace".to_string(),
                        reason: e.to_string(),
                        shape: None,
                        context: None,
                    }
                })?;

            *input = Tensor::from_array(output_array);
            Ok(())
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => {
            let mut cpu_complex = input.to_cpu()?;
            fft3_inplace(&mut cpu_complex)?;
            *input = cpu_complex;
            Ok(())
        }
    }
}

/// In-place 3D inverse FFT along the last three axes (complex → complex).
///
/// Applies the separable normalised inverse 3D DFT.
pub fn ifft3_inplace<T>(input: &mut Tensor<Complex<T>>) -> Result<()>
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
    let result = ifft3(input as &Tensor<Complex<T>>)?;
    *input = result;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{ArrayD, IxDyn};

    fn make_complex_tensor(data: Vec<Complex<f32>>, shape: &[usize]) -> Tensor<Complex<f32>> {
        let arr = ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape must match data len");
        Tensor::from_array(arr)
    }

    /// Round-trip: fft_inplace followed by ifft_inplace must recover the
    /// original signal to within floating-point tolerance.
    #[test]
    fn test_fft_ifft_1d_roundtrip() {
        let signal: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let n = signal.len();
        let data: Vec<Complex<f32>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();

        let mut tensor = make_complex_tensor(data.clone(), &[n]);

        fft_inplace(&mut tensor).expect("fft_inplace must not fail");
        // Verify tensor has actually changed (not a no-op)
        let after_fft = tensor.data().to_vec();
        let unchanged = after_fft
            .iter()
            .zip(data.iter())
            .all(|(a, b)| (a.re - b.re).abs() < 1e-6 && (a.im - b.im).abs() < 1e-6);
        assert!(!unchanged, "fft_inplace must change the tensor data");

        ifft_inplace(&mut tensor).expect("ifft_inplace must not fail");

        let result = tensor.data().to_vec();
        for (i, (got, &orig)) in result.iter().zip(signal.iter()).enumerate() {
            assert!(
                (got.re - orig).abs() < 1e-4,
                "element[{i}]: re = {:.6}, expected {:.6}",
                got.re,
                orig,
            );
            assert!(
                got.im.abs() < 1e-4,
                "element[{i}]: im = {:.6}, expected ~0",
                got.im,
            );
        }
    }

    /// 2D round-trip: fft2_inplace then ifft2_inplace.
    #[test]
    fn test_fft2_ifft2_2d_roundtrip() {
        // 2×4 complex tensor
        let signal: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = [2usize, 4];
        let data: Vec<Complex<f32>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();

        let mut tensor = make_complex_tensor(data, &shape);

        fft2_inplace(&mut tensor).expect("fft2_inplace must not fail");
        ifft2_inplace(&mut tensor).expect("ifft2_inplace must not fail");

        let result = tensor.data().to_vec();
        for (i, (got, &orig)) in result.iter().zip(signal.iter()).enumerate() {
            assert!(
                (got.re - orig).abs() < 1e-4,
                "element[{i}]: re = {:.6}, expected {:.6}",
                got.re,
                orig,
            );
            assert!(
                got.im.abs() < 1e-4,
                "element[{i}]: im = {:.6}, expected ~0",
                got.im,
            );
        }
    }

    /// 3D round-trip: fft3_inplace then ifft3_inplace.
    #[test]
    fn test_fft3_ifft3_3d_roundtrip() {
        // 2×2×4 complex tensor
        let signal: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let shape = [2usize, 2, 4];
        let data: Vec<Complex<f32>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();

        let mut tensor = make_complex_tensor(data, &shape);

        fft3_inplace(&mut tensor).expect("fft3_inplace must not fail");
        ifft3_inplace(&mut tensor).expect("ifft3_inplace must not fail");

        let result = tensor.data().to_vec();
        for (i, (got, &orig)) in result.iter().zip(signal.iter()).enumerate() {
            assert!(
                (got.re - orig).abs() < 1e-4,
                "element[{i}]: re = {:.6}, expected {:.6}",
                got.re,
                orig,
            );
            assert!(
                got.im.abs() < 1e-4,
                "element[{i}]: im = {:.6}, expected ~0",
                got.im,
            );
        }
    }
}
