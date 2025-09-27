//! Gradient operation modules
//!
//! This module contains the refactored gradient operations, organized by functionality
//! to maintain the CLAUDE.md policy of keeping modules under 2000 lines.

pub mod activation_ops;
pub mod binary_ops;
pub mod convolution_ops;
pub mod fft_ops;
pub mod linalg_ops;
pub mod normalization_ops;
pub mod reduction_ops;
pub mod shape_ops;
pub mod utils;

// Re-export commonly used functions for backward compatibility
pub use activation_ops::*;
pub use binary_ops::*;
pub use convolution_ops::*;
pub use fft_ops::*;
pub use linalg_ops::*;
pub use normalization_ops::*;
pub use reduction_ops::*;
pub use shape_ops::*;
pub use utils::*;
