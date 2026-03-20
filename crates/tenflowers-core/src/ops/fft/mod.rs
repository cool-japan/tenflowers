//! Fast Fourier Transform (FFT) operations
//!
//! This module provides comprehensive FFT implementations including 1D, 2D, and 3D
//! transforms with both CPU and GPU acceleration support.

#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

// Re-export specialized modules
pub mod fft1d;
pub mod fft2d;
pub mod fft3d;
pub mod gpu_kernels;
pub mod half_precision;
pub mod inplace;
pub mod utils;

// Re-export commonly used functions
pub use fft1d::{fft, ifft, rfft};
pub use fft2d::{fft2, ifft2};
pub use fft3d::{fft3, ifft3};
pub use half_precision::{
    fft2_bf16, fft2_f16, fft_bf16, fft_f16, ifft2_bf16, ifft2_f16, ifft_bf16, ifft_f16,
};
pub use inplace::{
    fft2_inplace, fft3_inplace, fft_inplace, ifft2_inplace, ifft3_inplace, ifft_inplace,
};
pub use utils::generate_twiddle_factors;
