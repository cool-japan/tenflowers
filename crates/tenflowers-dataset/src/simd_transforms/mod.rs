//! SIMD-accelerated transforms for high-performance data processing
//!
//! This module provides vectorized implementations of common data transformations
//! using SIMD instructions for significant performance improvements on supported hardware.
//!
//! ## Sub-modules
//!
//! - `normalization`: SIMD-accelerated normalization transforms
//! - `element_wise`: SIMD-accelerated element-wise operations
//! - `statistics`: SIMD-accelerated statistical computations
//! - `image_processing`: SIMD-accelerated image processing operations
//! - `convolution`: SIMD-accelerated convolution operations
//! - `matrix_ops`: SIMD-accelerated matrix operations
//! - `benchmarks`: Performance benchmarking utilities

pub mod benchmarks;
pub mod convolution;
pub mod element_wise;
pub mod image_processing;
pub mod matrix_ops;
pub mod normalization;
pub mod statistics;

// Re-export commonly used types for convenience
pub use benchmarks::{BenchmarkResult, SimdBenchmark};
pub use convolution::SimdConvolution;
pub use element_wise::{SimdElementWise, SimdOperation};
pub use image_processing::{SimdColorConvert, SimdHistogram, SimdHistogramTransform};
pub use matrix_ops::SimdMatrixOps;
pub use normalization::{SimdNormalize, SimdNormalizeScalarOnly};
pub use statistics::SimdStats;
