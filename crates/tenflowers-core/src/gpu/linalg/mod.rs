//! GPU Linear Algebra Operations
//!
//! This module implements GPU-accelerated linear algebra operations using WGPU compute shaders.
//! The implementations follow cuSOLVER patterns adapted for WGPU, providing efficient GPU
//! implementations of matrix decompositions, eigenvalue computations, and linear solvers.
//!
//! # Features
//! - LU decomposition with partial pivoting
//! - Singular Value Decomposition (SVD)
//! - QR decomposition
//! - Eigenvalue/eigenvector computation
//! - Linear system solving
//! - Matrix inversion
//!
//! # Architecture
//! The operations are designed to integrate seamlessly with the existing tenflowers-core
//! GPU infrastructure, using the same buffer management, device abstraction, and
//! error handling patterns.
//!
//! # Modules
//! - [`context`] - GPU context, metadata, and configuration management
//! - [`basic_ops`] - Basic operations like transpose and matrix multiplication
//! - [`decompositions`] - Matrix decompositions (LU, SVD, QR)
//! - [`advanced_ops`] - Advanced operations (eigenvalues, solve, inverse, determinant)

// Re-export specialized modules
pub mod advanced_ops;
pub mod basic_ops;
pub mod context;
pub mod decompositions;

// Re-export core types and functions for backward compatibility
pub use context::{AdaptiveGemmConfig, GpuLinalgContext, LinalgMetadata};

pub use basic_ops::{adaptive_matmul_linalg, matmul_linalg, transpose};

pub use decompositions::{lu_decomposition, qr_decomposition, svd};

// TODO: Commented out until advanced_ops functions are implemented
// pub use advanced_ops::{determinant, eigenvalues, inverse, solve};
