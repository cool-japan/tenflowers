//! Ultra-High-Performance SIMD Optimizations powered by SciRS2-Core
//!
//! This module provides state-of-the-art Single Instruction, Multiple Data (SIMD)
//! optimizations for core tensor operations, leveraging SciRS2-Core's advanced
//! vectorization capabilities and platform-specific acceleration.
//!
//! ## Modular Architecture
//!
//! - [`basic_ops`] - Element-wise operations (add, multiply, FMA)
//! - [`activation_functions`] - Neural network activation functions (ReLU, Sigmoid, etc.)
//! - [`matrix_ops`] - Matrix operations (matmul, dot product, transpose)
//! - [`math_functions`] - Mathematical functions (exp, sqrt, log, trig)
//! - [`reduction_ops`] - Reduction operations (sum, min/max, normalization)
//! - [`capabilities`] - Platform capabilities detection and utilities
//! - [`benchmarks`] - Performance benchmarking utilities
//!
//! ## Legacy Modules (Advanced Features)
//!
//! - [`advanced_kernels`] - Specialized kernel implementations
//! - [`cache_friendly_ops`] - Cache-optimized tensor operations
//! - [`ultra_simd_engine`] - Ultra-high performance SIMD engine

use crate::Result;

// Import SciRS2-Core parallel processing capabilities
#[cfg(feature = "simd")]
use scirs2_core::parallel_ops::{par_chunks, par_join, par_scope};

// New modular SIMD architecture
pub mod activation_functions;
pub mod basic_ops;
pub mod benchmarks;
pub mod capabilities;
pub mod math_functions;
pub mod matrix_ops;
pub mod reduction_ops;

// Legacy advanced modules
pub mod advanced_kernels;
pub mod cache_friendly_ops;
pub mod ultra_simd_engine;

// Re-export new modular API
pub use activation_functions::ActivationFunctions;
pub use basic_ops::BasicOps;
pub use benchmarks::{BenchmarkResult, Benchmarks};
pub use capabilities::{Capabilities, PerformanceHints, SimdCapabilities};
pub use math_functions::MathFunctions;
pub use matrix_ops::MatrixOps;
pub use reduction_ops::ReductionOps;

// Re-export legacy API for backward compatibility
pub use ultra_simd_engine::{
    global_simd_engine, ConvolutionParams, CpuFeatures, ElementWiseOp, ReductionOp,
    SimdEngineConfig, UltraSimdEngine,
};

pub use cache_friendly_ops::{CacheFriendlyMatMul, CacheOptimizedTensorOps, MemoryAccessPattern};

pub use advanced_kernels::{AdvancedKernelRegistry, KernelOptimizationStrategy, SpecializedKernel};

/// SIMD-optimized tensor operations using stable Rust features
///
/// This is the main entry point for SIMD operations, providing a unified
/// interface to all optimized functions while maintaining backward compatibility.
pub struct SimdOptimizer;

impl SimdOptimizer {
    // Re-export key operations for backward compatibility with the monolithic interface

    /// Fast inline element-wise addition without bounds checking (for hot paths)
    #[inline(always)]
    pub fn add_f32_unchecked(a: &[f32], b: &[f32], result: &mut [f32]) {
        BasicOps::add_f32_unchecked(a, b, result)
    }

    /// Element-wise addition with SciRS2-Core SIMD auto-vectorization (safe version)
    pub fn add_f32_optimized(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        BasicOps::add_f32_optimized(a, b, result)
    }

    /// Auto-selection element-wise addition (chooses optimal implementation based on size)
    pub fn add_f32_auto(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        BasicOps::add_f32_auto(a, b, result)
    }

    /// Fast inline element-wise multiplication without bounds checking (for hot paths)
    #[inline(always)]
    pub fn mul_f32_unchecked(a: &[f32], b: &[f32], result: &mut [f32]) {
        BasicOps::mul_f32_unchecked(a, b, result)
    }

    /// Element-wise multiplication with optimization hints for f32 tensors (safe version)
    pub fn mul_f32_optimized(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        BasicOps::mul_f32_optimized(a, b, result)
    }

    /// Element-wise subtraction with optimization hints for f32 tensors (safe version)
    pub fn sub_f32_optimized(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        BasicOps::sub_f32_optimized(a, b, result)
    }

    /// Optimized ReLU activation function
    pub fn relu_f32_optimized(input: &[f32], output: &mut [f32]) -> Result<()> {
        ActivationFunctions::relu_f32_optimized(input, output)
    }

    /// Optimized sigmoid activation with fast approximation
    pub fn sigmoid_f32_optimized(input: &[f32], output: &mut [f32]) -> Result<()> {
        ActivationFunctions::sigmoid_f32_optimized(input, output)
    }

    /// Advanced cache-friendly matrix multiplication using blocking
    pub fn matmul_f32_blocked(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        block_size: usize,
    ) -> Result<()> {
        MatrixOps::matmul_f32_blocked(a, b, c, m, n, k, block_size)
    }

    /// Optimized dot product for f32 vectors using Kahan summation
    pub fn dot_product_f32_optimized(a: &[f32], b: &[f32]) -> Result<f32> {
        MatrixOps::dot_product_f32_optimized(a, b)
    }

    /// Ultra-fast vectorized horizontal sum reduction
    #[inline(always)]
    pub fn sum_f32_unchecked(input: &[f32]) -> f32 {
        ReductionOps::sum_f32_unchecked(input)
    }

    /// Vectorized maximum element finding with SIMD optimization
    #[inline(always)]
    pub fn max_f32_unchecked(input: &[f32]) -> f32 {
        ReductionOps::max_f32_unchecked(input)
    }

    /// Platform-specific capabilities detection (stable approach)
    pub fn detect_capabilities() -> SimdCapabilities {
        Capabilities::detect_capabilities()
    }

    /// Advanced vectorized reduction operations with numerical stability
    pub fn reduce_sum_f32_optimized(input: &[f32]) -> Result<f32> {
        ReductionOps::reduce_sum_f32_optimized(input)
    }

    /// Vectorized min/max operations with efficient branching
    pub fn reduce_min_max_f32_optimized(input: &[f32]) -> Result<(f32, f32)> {
        ReductionOps::reduce_min_max_f32_optimized(input)
    }

    /// Advanced normalization with streaming computation for memory efficiency
    pub fn normalize_f32_optimized(input: &[f32], output: &mut [f32], eps: f32) -> Result<()> {
        ReductionOps::normalize_f32_optimized(input, output, eps)
    }
}
