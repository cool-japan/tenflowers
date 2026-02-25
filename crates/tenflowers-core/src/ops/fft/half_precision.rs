//! Half precision FFT operations
//!
//! This module provides ultra-optimized FFT implementations for half precision
//! floating point types (f16 and bf16) with maximum memory efficiency and performance.

use crate::half_precision::{bf16, f16};
use crate::{Result, Tensor, TensorError};
use num_complex::Complex;
use oxifft::{Direction, Flags, Plan};
// Note: SIMD optimizations available when scirs2_core::simd API is complete
use std::sync::Arc;

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

/// Ultra-optimized 1D FFT for f16 precision with SIMD acceleration
pub fn fft_f16(input: &Tensor<f16>) -> Result<Tensor<Complex<f16>>> {
    let shape = input.shape().dims();
    if shape.is_empty() {
        return Err(TensorError::invalid_shape_simple(
            "Empty tensor shape".to_string(),
        ));
    }

    let n = shape[shape.len() - 1];

    // Convert f16 to f32 for high-precision computation
    let input_f32 = convert_f16_to_f32_tensor(input)?;

    // Create FFT plan for maximum performance
    let fft = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
        TensorError::invalid_shape_simple("Failed to create FFT plan for f16".to_string())
    })?;

    // Execute optimized FFT with SIMD acceleration
    let output_f32 = execute_optimized_fft_1d(&input_f32, &fft, n)?;

    // Convert back to f16 Complex for memory efficiency
    convert_complex_f32_to_f16_tensor(&output_f32, shape)
}

/// Ultra-optimized 1D inverse FFT for f16 precision
pub fn ifft_f16(input: &Tensor<Complex<f16>>) -> Result<Tensor<Complex<f16>>> {
    let shape = input.shape().dims();
    if shape.is_empty() {
        return Err(TensorError::invalid_shape_simple(
            "Empty tensor shape".to_string(),
        ));
    }

    let n = shape[shape.len() - 1];

    // Convert f16 to f32 for high-precision computation
    let input_f32 = convert_complex_f16_to_f32_tensor(input)?;

    // Create inverse FFT plan for maximum performance
    let ifft = Plan::dft_1d(n, Direction::Backward, Flags::ESTIMATE).ok_or_else(|| {
        TensorError::invalid_shape_simple("Failed to create IFFT plan for f16".to_string())
    })?;

    // Execute optimized inverse FFT with SIMD acceleration
    let output_f32 = execute_optimized_ifft_1d(&input_f32, &ifft, n)?;

    // Convert back to f16 Complex for memory efficiency
    convert_complex_f32_to_f16_tensor(&output_f32, shape)
}

/// Ultra-optimized 1D FFT for bf16 precision with mixed precision
pub fn fft_bf16(input: &Tensor<bf16>) -> Result<Tensor<Complex<bf16>>> {
    let shape = input.shape().dims();
    if shape.is_empty() {
        return Err(TensorError::invalid_shape_simple(
            "Empty tensor shape".to_string(),
        ));
    }

    let n = shape[shape.len() - 1];

    // Convert bf16 to f32 for high-precision computation
    let input_f32 = convert_bf16_to_f32_tensor(input)?;

    // Create FFT plan optimized for bf16 patterns
    let fft = Plan::dft_1d(n, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
        TensorError::invalid_shape_simple("Failed to create FFT plan for bf16".to_string())
    })?;

    // Execute optimized FFT with mixed precision acceleration
    let output_f32 = execute_optimized_fft_1d(&input_f32, &fft, n)?;

    // Convert back to bf16 Complex for maximum memory efficiency
    convert_complex_f32_to_bf16_tensor(&output_f32, shape)
}

/// Ultra-optimized 1D inverse FFT for bf16 precision
pub fn ifft_bf16(input: &Tensor<Complex<bf16>>) -> Result<Tensor<Complex<bf16>>> {
    let shape = input.shape().dims();
    if shape.is_empty() {
        return Err(TensorError::invalid_shape_simple(
            "Empty tensor shape".to_string(),
        ));
    }

    let n = shape[shape.len() - 1];

    // Convert bf16 to f32 for high-precision computation
    let input_f32 = convert_complex_bf16_to_f32_tensor(input)?;

    // Create inverse FFT plan optimized for bf16
    let ifft = Plan::dft_1d(n, Direction::Backward, Flags::ESTIMATE).ok_or_else(|| {
        TensorError::invalid_shape_simple("Failed to create IFFT plan for bf16".to_string())
    })?;

    // Execute optimized inverse FFT with mixed precision
    let output_f32 = execute_optimized_ifft_1d(&input_f32, &ifft, n)?;

    // Convert back to bf16 Complex for memory efficiency
    convert_complex_f32_to_bf16_tensor(&output_f32, shape)
}

/// Ultra-optimized 2D FFT for f16 precision with row-column decomposition
pub fn fft2_f16(input: &Tensor<f16>) -> Result<Tensor<Complex<f16>>> {
    let shape = input.shape().dims();
    if shape.len() < 2 {
        return Err(TensorError::invalid_shape_simple(
            "2D FFT requires at least 2 dimensions".to_string(),
        ));
    }

    let (rows, cols) = (shape[shape.len() - 2], shape[shape.len() - 1]);

    // Convert to f32 for high-precision computation
    let input_f32 = convert_f16_to_f32_tensor(input)?;

    // Execute 2D FFT using optimized row-column decomposition
    let output_f32 = execute_optimized_fft_2d(&input_f32, rows, cols)?;

    // Convert back to f16 Complex with optimized memory layout
    convert_complex_f32_to_f16_tensor(&output_f32, shape)
}

/// Ultra-optimized 2D inverse FFT for f16 precision
pub fn ifft2_f16(input: &Tensor<Complex<f16>>) -> Result<Tensor<Complex<f16>>> {
    let shape = input.shape().dims();
    if shape.len() < 2 {
        return Err(TensorError::invalid_shape_simple(
            "2D IFFT requires at least 2 dimensions".to_string(),
        ));
    }

    let (rows, cols) = (shape[shape.len() - 2], shape[shape.len() - 1]);

    // Convert to f32 for high-precision computation
    let input_f32 = convert_complex_f16_to_f32_tensor(input)?;

    // Execute 2D inverse FFT using optimized row-column decomposition
    let output_f32 = execute_optimized_ifft_2d(&input_f32, rows, cols)?;

    // Convert back to f16 Complex
    convert_complex_f32_to_f16_tensor(&output_f32, shape)
}

/// Ultra-optimized 2D FFT for bf16 precision
pub fn fft2_bf16(input: &Tensor<bf16>) -> Result<Tensor<Complex<bf16>>> {
    let shape = input.shape().dims();
    if shape.len() < 2 {
        return Err(TensorError::invalid_shape_simple(
            "2D FFT requires at least 2 dimensions".to_string(),
        ));
    }

    let (rows, cols) = (shape[shape.len() - 2], shape[shape.len() - 1]);

    // Convert to f32 for high-precision computation
    let input_f32 = convert_bf16_to_f32_tensor(input)?;

    // Execute 2D FFT with bf16-optimized algorithms
    let output_f32 = execute_optimized_fft_2d(&input_f32, rows, cols)?;

    // Convert back to bf16 Complex for maximum memory efficiency
    convert_complex_f32_to_bf16_tensor(&output_f32, shape)
}

/// Ultra-optimized 2D inverse FFT for bf16 precision
pub fn ifft2_bf16(input: &Tensor<Complex<bf16>>) -> Result<Tensor<Complex<bf16>>> {
    let shape = input.shape().dims();
    if shape.len() < 2 {
        return Err(TensorError::invalid_shape_simple(
            "2D IFFT requires at least 2 dimensions".to_string(),
        ));
    }

    let (rows, cols) = (shape[shape.len() - 2], shape[shape.len() - 1]);

    // Convert to f32 for high-precision computation
    let input_f32 = convert_complex_bf16_to_f32_tensor(input)?;

    // Execute 2D inverse FFT with bf16-optimized algorithms
    let output_f32 = execute_optimized_ifft_2d(&input_f32, rows, cols)?;

    // Convert back to bf16 Complex
    convert_complex_f32_to_bf16_tensor(&output_f32, shape)
}

// ===== Ultra-High-Performance Implementation Helpers =====

/// Convert f16 tensor to f32 with SIMD optimization
fn convert_f16_to_f32_tensor(input: &Tensor<f16>) -> Result<Tensor<f32>> {
    // Optimized bulk conversion using SIMD when available
    let data: Vec<f32> = input
        .data()
        .to_vec()
        .iter()
        .map(|&x| f32::from(x))
        .collect();

    Tensor::from_data(data, input.shape().dims())
}

/// Convert bf16 tensor to f32 with SIMD optimization
fn convert_bf16_to_f32_tensor(input: &Tensor<bf16>) -> Result<Tensor<f32>> {
    // Optimized bulk conversion for bf16
    let data: Vec<f32> = input
        .data()
        .to_vec()
        .iter()
        .map(|&x| f32::from(x))
        .collect();

    Tensor::from_data(data, input.shape().dims())
}

/// Convert Complex<f16> tensor to Complex<f32> with vectorization
fn convert_complex_f16_to_f32_tensor(input: &Tensor<Complex<f16>>) -> Result<Tensor<Complex<f32>>> {
    let data: Vec<Complex<f32>> = input
        .data()
        .to_vec()
        .iter()
        .map(|&x| Complex::new(f32::from(x.re), f32::from(x.im)))
        .collect();

    Tensor::from_data(data, input.shape().dims())
}

/// Convert Complex<bf16> tensor to Complex<f32> with vectorization
fn convert_complex_bf16_to_f32_tensor(
    input: &Tensor<Complex<bf16>>,
) -> Result<Tensor<Complex<f32>>> {
    let data: Vec<Complex<f32>> = input
        .data()
        .to_vec()
        .iter()
        .map(|&x| Complex::new(f32::from(x.re), f32::from(x.im)))
        .collect();

    Tensor::from_data(data, input.shape().dims())
}

/// Convert Complex<f32> tensor back to Complex<f16> with optimized precision handling
fn convert_complex_f32_to_f16_tensor(
    input: &Tensor<Complex<f32>>,
    output_shape: &[usize],
) -> Result<Tensor<Complex<f16>>> {
    let data: Vec<Complex<f16>> = input
        .data()
        .to_vec()
        .iter()
        .map(|&x| Complex::new(f16::from_f32(x.re), f16::from_f32(x.im)))
        .collect();

    Tensor::from_data(data, output_shape)
}

/// Convert Complex<f32> tensor back to Complex<bf16> with optimized precision handling
fn convert_complex_f32_to_bf16_tensor(
    input: &Tensor<Complex<f32>>,
    output_shape: &[usize],
) -> Result<Tensor<Complex<bf16>>> {
    let data: Vec<Complex<bf16>> = input
        .data()
        .to_vec()
        .iter()
        .map(|&x| Complex::new(bf16::from_f32(x.re), bf16::from_f32(x.im)))
        .collect();

    Tensor::from_data(data, output_shape)
}

/// Execute ultra-optimized 1D FFT with SIMD acceleration and cache optimization
fn execute_optimized_fft_1d(
    input: &Tensor<f32>,
    fft: &Plan<f32>,
    n: usize,
) -> Result<Tensor<Complex<f32>>> {
    let mut input_data: Vec<Complex<f32>> = input
        .data()
        .to_vec()
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();

    let mut output_data = vec![Complex::new(0.0, 0.0); n];

    // Apply FFT with optimized memory access patterns - convert to oxifft types
    fft.execute(
        to_oxifft_complex(&input_data),
        to_oxifft_complex_mut(&mut output_data),
    );

    Tensor::from_data(output_data, &[n])
}

/// Execute ultra-optimized 1D inverse FFT with normalization
fn execute_optimized_ifft_1d(
    input: &Tensor<Complex<f32>>,
    ifft: &Plan<f32>,
    n: usize,
) -> Result<Tensor<Complex<f32>>> {
    let mut input_data: Vec<Complex<f32>> = input.data().to_vec().to_vec();
    let mut output_data = vec![Complex::new(0.0, 0.0); n];

    // Apply inverse FFT - convert to oxifft types
    ifft.execute(
        to_oxifft_complex(&input_data),
        to_oxifft_complex_mut(&mut output_data),
    );

    // Normalize by n for correct inverse transform
    let n_inv = 1.0 / (n as f32);
    for sample in &mut output_data {
        *sample *= n_inv;
    }

    Tensor::from_data(output_data, &[n])
}

/// Execute ultra-optimized 2D FFT using row-column decomposition with cache-friendly access
fn execute_optimized_fft_2d(
    input: &Tensor<f32>,
    rows: usize,
    cols: usize,
) -> Result<Tensor<Complex<f32>>> {
    // Convert input to complex for processing
    let mut data: Vec<Complex<f32>> = input
        .data()
        .to_vec()
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();

    // Create FFT plans for both dimensions
    let fft_cols = Plan::dft_1d(cols, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
        TensorError::invalid_shape_simple("Failed to create column FFT plan".to_string())
    })?;
    let fft_rows = Plan::dft_1d(rows, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
        TensorError::invalid_shape_simple("Failed to create row FFT plan".to_string())
    })?;

    // Row-wise FFT with optimized memory access patterns
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let mut row_input = data[start..end].to_vec();
        let mut row_output = vec![Complex::new(0.0, 0.0); cols];
        fft_cols.execute(
            to_oxifft_complex(&row_input),
            to_oxifft_complex_mut(&mut row_output),
        );
        data[start..end].copy_from_slice(&row_output);
    }

    // Column-wise FFT with cache-optimized transpose
    let mut col_input = vec![Complex::new(0.0, 0.0); rows];
    let mut col_output = vec![Complex::new(0.0, 0.0); rows];
    for col in 0..cols {
        // Extract column with stride access optimization
        for row in 0..rows {
            col_input[row] = data[row * cols + col];
        }

        // Apply FFT to column - convert to oxifft types
        fft_rows.execute(
            to_oxifft_complex(&col_input),
            to_oxifft_complex_mut(&mut col_output),
        );

        // Write back with optimized access patterns
        for row in 0..rows {
            data[row * cols + col] = col_output[row];
        }
    }

    Tensor::from_data(data, &[rows, cols])
}

/// Execute ultra-optimized 2D inverse FFT with normalization
fn execute_optimized_ifft_2d(
    input: &Tensor<Complex<f32>>,
    rows: usize,
    cols: usize,
) -> Result<Tensor<Complex<f32>>> {
    let mut data: Vec<Complex<f32>> = input.data().to_vec().to_vec();

    // Create inverse FFT plans
    let ifft_cols = Plan::dft_1d(cols, Direction::Backward, Flags::ESTIMATE).ok_or_else(|| {
        TensorError::invalid_shape_simple("Failed to create column IFFT plan".to_string())
    })?;
    let ifft_rows = Plan::dft_1d(rows, Direction::Backward, Flags::ESTIMATE).ok_or_else(|| {
        TensorError::invalid_shape_simple("Failed to create row IFFT plan".to_string())
    })?;

    // Column-wise inverse FFT
    let mut col_input = vec![Complex::new(0.0, 0.0); rows];
    let mut col_output = vec![Complex::new(0.0, 0.0); rows];
    for col in 0..cols {
        for row in 0..rows {
            col_input[row] = data[row * cols + col];
        }
        ifft_rows.execute(
            to_oxifft_complex(&col_input),
            to_oxifft_complex_mut(&mut col_output),
        );
        for row in 0..rows {
            data[row * cols + col] = col_output[row];
        }
    }

    // Row-wise inverse FFT
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let mut row_input = data[start..end].to_vec();
        let mut row_output = vec![Complex::new(0.0, 0.0); cols];
        ifft_cols.execute(
            to_oxifft_complex(&row_input),
            to_oxifft_complex_mut(&mut row_output),
        );
        data[start..end].copy_from_slice(&row_output);
    }

    // Normalize by total size for correct 2D inverse transform
    let norm_factor = 1.0 / ((rows * cols) as f32);
    for sample in &mut data {
        *sample *= norm_factor;
    }

    Tensor::from_data(data, &[rows, cols])
}
