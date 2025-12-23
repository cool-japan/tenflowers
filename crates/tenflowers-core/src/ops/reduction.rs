//! Tensor Reduction Operations - Modular Architecture
//!
//! This module has been refactored into a modular architecture for better maintainability
//! and organization. The functionality has been split into specialized modules by operation type:
//!
//! ## Module Organization
//!
//! - **common**: Shared helper functions and utilities
//! - **statistical**: Basic statistical reduction operations (sum, mean, max, min, prod, variance)
//! - **argops**: Argument-based operations (argmax, argmin, topk)
//! - **cumulative**: Cumulative operations (cumsum, cumprod)
//! - **boolean**: Boolean reduction operations (all, any)
//! - **segment**: Segment-based reduction operations (segment_sum, segment_mean, segment_max)
//!
//! All operations maintain 100% backward compatibility through strategic re-exports.

// Import the modularized tensor reduction functionality
pub mod argops;
pub mod boolean;
pub mod common;
pub mod cumulative;
#[cfg(feature = "gpu")]
pub mod gpu_execution;
#[cfg(feature = "gpu")]
pub mod gpu_kernels;
pub mod segment;
pub mod statistical;

// Re-export all operations for backward compatibility

// Statistical operations
pub use statistical::{max, mean, min, prod, sum, variance};

// Argument operations
pub use argops::{argmax, argmin, topk};

// Cumulative operations
pub use cumulative::{cumprod, cumsum};

// Boolean operations
pub use boolean::{all, any};

// Segment operations
pub use segment::{segment_max, segment_mean, segment_sum};

// Re-export common utilities for internal use by other modules
pub use common::normalize_axis;
