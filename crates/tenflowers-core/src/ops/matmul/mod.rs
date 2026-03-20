//! Matrix Multiplication Operations
//!
//! This module provides comprehensive matrix multiplication capabilities including
//! standard matmul, batch processing, GPU acceleration, and optimized implementations.
//!
//! ## Module Structure
//!
//! - [`core`] - Core matrix multiplication implementations and public API
//! - [`batch`] - Batch matrix multiplication operations
//! - [`gpu`] - GPU-accelerated matrix multiplication
//! - [`optimized`] - CPU-optimized implementations (BLAS, blocked, cache-friendly)
//! - [`shapes`] - Shape computation and broadcasting utilities
//! - [`specialized`] - Specialized operations (outer product, mixed precision)

// Core modules
pub mod batch;
pub mod core;
pub mod gpu;
pub mod optimized;
pub mod shapes;
pub mod specialized;

// Re-export main API functions for backward compatibility
pub use core::{batch_matmul, dot, matmul};
pub use specialized::{matmul_mixed_precision, outer};
