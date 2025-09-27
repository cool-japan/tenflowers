//! Convolution operations for neural networks
//!
//! This module provides comprehensive convolution implementations including:
//! - 1D, 2D, and 3D standard convolutions
//! - Depthwise separable convolutions
//! - Transpose convolutions (deconvolutions)
//! - Layout optimization utilities

// Re-export specialized modules
pub mod conv1d;
pub mod conv2d;
pub mod conv3d;
pub mod depthwise;
pub mod layout;
pub mod transpose;

// Re-export commonly used functions for backward compatibility
pub use conv1d::conv1d;
pub use conv2d::{conv2d, conv2d_auto_layout, conv2d_with_layout};
pub use conv3d::conv3d;
pub use depthwise::depthwise_conv2d;
pub use layout::conv_layout_benchmark;
pub use transpose::conv_transpose2d;
