//! GPU Operations Module
//!
//! This module provides comprehensive GPU operation implementations using WGPU compute shaders.
//! It includes optimized kernels for various tensor operations including basic arithmetic,
//! reductions, activations, comparisons, einsum operations, tensor manipulation, indexing,
//! and pooling operations.
//!
//! # Architecture
//!
//! The GPU operations are organized into logical groups:
//! - **Operation Types**: Core operation type definitions and enums
//! - **Basic Operations**: Fundamental unary, binary, and scalar operations
//! - **Reduction Operations**: Sum, mean, max, min, product operations
//! - **Activation Operations**: ReLU, sigmoid, tanh, GELU, and other activations
//! - **Comparison Operations**: Equality, inequality, greater than, less than operations
//! - **Einsum Operations**: Complex tensor contractions and linear algebra operations
//! - **Manipulation Operations**: Reshape, transpose, slice, pad, tile operations
//! - **Indexing Operations**: Gather, scatter, where, one-hot operations
//! - **Pooling Operations**: Max pooling, average pooling, and fractional pooling
//!
//! # Performance Features
//!
//! All operations support:
//! - WGPU compute shader optimization
//! - Automatic workgroup size selection
//! - Memory coalescing for optimal GPU memory access
//! - Type-safe operation dispatch with bytemuck integration
//! - Asynchronous execution with proper synchronization

use super::*;
use crate::Result;

// Re-export types from parent module and submodules
pub use super::binary_ops::BinaryOp;
pub use super::unary_ops::UnaryOp;
pub use super::BinaryScalarOp;

// Core operation modules
pub mod activation_ops;
pub mod basic_ops;
pub mod comparison_ops;
pub mod einsum_ops;
pub mod indexing_ops;
pub mod manipulation_ops;
pub mod operation_types;
pub mod pooling_ops;
pub mod reduction_ops;

// Re-export all operation types for backward compatibility
pub use operation_types::*;

// Re-export all operation functions for backward compatibility
pub use activation_ops::*;
pub use basic_ops::*;
pub use comparison_ops::*;
pub use einsum_ops::*;
pub use indexing_ops::*;
pub use manipulation_ops::*;
pub use pooling_ops::*;
pub use reduction_ops::*;
