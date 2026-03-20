//! Tensor Manipulation Operations - Modular Architecture
//!
//! This module has been refactored into a modular architecture for better maintainability
//! and organization. The functionality has been split into specialized modules by operation type:
//!
//! ## Module Organization
//!
//! - **common**: Shared helper functions and GPU dispatch utilities
//! - **shape**: Shape manipulation operations (reshape, expand_dims, squeeze, etc.)
//! - **transpose**: Transposition and permutation operations (transpose, flip, roll)
//! - **indexing**: Slicing and indexing operations (slice, gather, scatter, etc.)
//! - **concatenation**: Concatenation and stacking operations (concat, stack, split, etc.) ✓ extracted
//! - **utilities**: Utility operations (identity, cast, pad, one_hot) ✓ extracted
//!
//! All operations maintain 100% backward compatibility through strategic re-exports.

// Import the modularized tensor manipulation functionality
pub mod common;
pub mod concatenation;
pub mod indexing; // Indexing operations extracted from manipulation.rs.bak
pub mod shape;
pub mod transpose;
pub mod utilities; // Utility operations extracted from manipulation.rs.bak // Concatenation operations extracted from manipulation.rs.bak

// Re-export all operations for backward compatibility

// Shape operations
pub use shape::{broadcast_to, expand_as, expand_dims, flatten, reshape, squeeze, unsqueeze};

// Transposition operations
pub use transpose::{flip, roll, transpose, transpose_axes};

// Indexing operations
pub use indexing::{gather, scatter, select, slice, slice_with_stride, where_op};

// Concatenation operations
pub use concatenation::{concat, repeat, split, stack, tile};

// Utility operations
pub use utilities::{cast, identity, one_hot, pad};

// Re-export common utilities for internal use by other modules
pub use common::{broadcast_indices, calculate_strides, coords_to_flat, flat_to_coords};
