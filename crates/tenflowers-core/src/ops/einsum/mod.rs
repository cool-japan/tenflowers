//! Einstein Summation Operations - Modular Architecture
//!
//! This module has been refactored into a modular architecture for better maintainability
//! and organization. The functionality has been split into specialized modules by feature area:
//!
//! ## Module Organization
//!
//! - **core**: Main einsum function, parsing, and basic operations
//! - **blas**: BLAS-optimized implementations for CPU operations
//! - **gpu**: GPU-accelerated implementations using compute shaders
//! - **patterns**: Common pattern optimizations and fast paths
//! - **cache**: Cache-friendly implementations with optimal memory access
//! - **utils**: Utility functions for path optimization and tensor manipulation
//!
//! All functionality maintains 100% backward compatibility through strategic re-exports.

pub mod blas;
pub mod cache;
pub mod core;
pub mod gpu;
pub mod patterns;
pub mod utils;

// Re-export the main einsum function for backward compatibility
pub use core::einsum;

// Re-export utility functions that might be used externally
pub use cache::{cache_optimized_contraction, execute_contraction_path};
pub use core::{extract_diagonal, infer_output_subscript, parse_einsum_equation};
pub use utils::{batch_transpose, cache_friendly_trace, compute_outer_product};

// Re-export BLAS functions when available
#[cfg(any(
    all(feature = "blas-openblas", feature = "std"),
    all(feature = "blas-mkl", feature = "std"),
    all(feature = "blas-accelerate", feature = "std")
))]
pub use blas::try_blas_optimized_patterns;

// Re-export GPU functions when available
#[cfg(feature = "gpu")]
pub use gpu::{
    gpu_einsum_batched_matmul, gpu_einsum_diagonal, gpu_einsum_matmul, gpu_einsum_outer_product,
    gpu_einsum_trace, gpu_einsum_transpose, gpu_einsum_vector_dot,
};

// Re-export pattern optimization functions
pub use patterns::{try_optimize_common_patterns, try_optimize_gpu_patterns};

// Note: Internal utility functions and cache implementations are used internally
// and don't need re-export as they are implementation details
