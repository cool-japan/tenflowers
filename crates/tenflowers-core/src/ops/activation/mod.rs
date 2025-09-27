//! Ultra-performance activation functions with comprehensive optimizations
//!
//! This module provides a complete suite of activation functions optimized for:
//! - SIMD acceleration (AVX2, NEON)
//! - Parallel processing with Rayon
//! - GPU acceleration (when available)
//! - Memory-efficient implementations
//! - Comprehensive performance analytics
//!
//! ## Supported Activation Functions
//!
//! ### Core Activations
//! - `relu`, `relu_f32`, `relu_f64` - Rectified Linear Unit
//! - `sigmoid`, `sigmoid_f32`, `sigmoid_f64` - Sigmoid activation
//! - `tanh`, `tanh_f32`, `tanh_f64` - Hyperbolic tangent
//!
//! ### Modern Activations
//! - `gelu`, `gelu_f32` - Gaussian Error Linear Unit
//! - `swish` - Swish/SiLU activation
//! - `mish` - Mish activation
//!
//! ### Parameterized Activations
//! - `elu` - Exponential Linear Unit
//! - `leaky_relu` - Leaky ReLU
//! - `prelu` - Parametric ReLU
//!
//! ### Utility Functions
//! - `softmax` - Softmax normalization
//! - `log_softmax` - Log-softmax
//! - `hard_swish` - Hard Swish
//! - `relu6` - ReLU bounded at 6
//!
//! ## Performance Features
//!
//! ### Automatic Strategy Selection
//! The module automatically selects the optimal implementation based on:
//! - Input size (sequential, SIMD, parallel, or GPU)
//! - Data type (f32 gets SIMD acceleration)
//! - Function complexity (transcendental vs non-transcendental)
//!
//! ### SIMD Acceleration
//! - AVX2 support for x86_64 architectures
//! - NEON support for aarch64 architectures
//! - Automatic fallback to scalar implementations
//!
//! ### GPU Support
//! GPU-accelerated implementations are available when compiled with the `gpu` feature.
//!
//! ## Usage Examples
//!
//! ```rust
//! use tenflowers_core::ops::activation::*;
//! use tenflowers_core::Tensor;
//!
//! // Basic usage
//! let x = Tensor::<f32>::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();
//! let output = relu(&x).unwrap();
//!
//! // Optimized f32 functions
//! let sigmoid_out = sigmoid_f32(&x).unwrap();
//! let gelu_out = gelu_f32(&x).unwrap();
//!
//! // Parameterized activations
//! let elu_out = elu(&x, 1.0).unwrap();
//! let leaky_relu_out = leaky_relu(&x, 0.01).unwrap();
//!
//! // Softmax with axis specification
//! let softmax_out = softmax(&x, None).unwrap(); // Last axis
//! let softmax_axis0 = softmax(&x, Some(0)).unwrap(); // Specific axis
//! ```
//!
//! ## Performance Analytics
//!
//! ```rust
//! use tenflowers_core::ops::activation::*;
//!
//! // Get performance report
//! let analytics = get_activation_performance_report();
//! println!("SIMD accelerations: {}", analytics.simd_accelerations);
//! println!("GPU executions: {}", analytics.gpu_executions);
//! println!("Average throughput: {} MEPS", analytics.avg_throughput_meps);
//! ```

// Core infrastructure
pub mod core;
pub mod strategy;

// Implementation modules
pub mod functions;
pub mod implementations;
pub mod simd;

#[cfg(feature = "gpu")]
pub mod gpu;

// Tests
#[cfg(test)]
mod tests;

// Re-export public API
pub use core::{
    get_activation_performance_report, get_activation_registry, reset_activation_counters,
    ActivationAnalytics, ActivationFunction, ActivationRegistry, ActivationStrategy,
};

pub use strategy::{
    select_activation_strategy, ActivationConfig, APPROX_THRESHOLD, GPU_THRESHOLD,
    PARALLEL_THRESHOLD, SIMD_THRESHOLD,
};

pub use functions::{
    // Parameterized activations
    elu,
    // Modern activations
    gelu,
    gelu_f32,
    hard_swish,
    hardswish,
    leaky_relu,
    log_softmax,
    mish,

    prelu,

    // Core activations
    relu,
    relu6,
    relu_f32,
    relu_f64,
    sigmoid,
    sigmoid_f32,
    sigmoid_f64,
    // Utility functions
    softmax,
    swish,
    tanh,
    tanh_f32,
    tanh_f64,
};

#[cfg(feature = "gpu")]
pub use gpu::{
    batch_gpu_activations, dispatch_gpu_activation, execute_gpu_activation, softmax_gpu_f32,
    GpuActivationOp, GpuActivationStream,
};

// SIMD functions (for advanced users)
pub use simd::{
    fast_sigmoid_approx, simd_gelu_f32, simd_relu_f32, simd_sigmoid_f32, simd_tanh_f32,
};

// Implementation utilities (for advanced users)
pub use implementations::{
    gelu_parallel_f32, gelu_sequential_f32, sigmoid_vectorized, tanh_vectorized,
    ultra_relu_vectorized, ultra_sigmoid_vectorized,
};

/// Module-level documentation and feature summary
pub fn activation_module_info() -> &'static str {
    "TenfloweRS Activation Module v0.1.0-alpha.1

Features:
- 15+ activation functions with ultra-performance optimizations
- Automatic SIMD acceleration (AVX2/NEON)
- Parallel processing with adaptive chunking
- GPU acceleration (when available)
- Comprehensive performance analytics
- Memory-efficient implementations
- Extensive test coverage

Optimizations:
- Strategy selection based on input size
- SIMD vectorization for f32 operations
- Parallel processing for large arrays
- Fast approximations for transcendental functions
- GPU dispatch for massive workloads
- Zero-copy operations where possible

Supported Architectures:
- x86_64 with AVX2 SIMD
- aarch64 with NEON SIMD
- GPU acceleration via WGPU
- Fallback scalar implementations
"
}

/// Get current module version
pub fn version() -> &'static str {
    "0.1.0-alpha.1"
}

/// Get supported activation function count
pub fn supported_activation_count() -> usize {
    15
}

/// Check if SIMD acceleration is available
pub fn simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(target_arch = "aarch64")]
    {
        true // NEON is always available on aarch64
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// Check if GPU acceleration is available
pub fn gpu_available() -> bool {
    cfg!(feature = "gpu")
}

#[cfg(test)]
mod module_tests {
    use super::*;

    #[test]
    fn test_module_info() {
        let info = activation_module_info();
        assert!(info.contains("TenfloweRS"));
        assert!(info.contains("activation"));
        assert!(info.contains("SIMD"));
    }

    #[test]
    fn test_version() {
        let ver = version();
        assert!(ver.contains("0.1.0"));
    }

    #[test]
    fn test_activation_count() {
        let count = supported_activation_count();
        assert_eq!(count, 15);
    }

    #[test]
    fn test_simd_detection() {
        // Just verify the function works
        let _simd_avail = simd_available();
        // Can't assert specific values as it depends on the test environment
    }

    #[test]
    fn test_gpu_detection() {
        let gpu_avail = gpu_available();
        // Should match whether gpu feature is enabled
        assert_eq!(gpu_avail, cfg!(feature = "gpu"));
    }
}
