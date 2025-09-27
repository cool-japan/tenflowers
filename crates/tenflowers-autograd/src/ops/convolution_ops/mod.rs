//! Convolution Operations Module
//!
//! This module contains gradient operations for convolution-related functions including:
//! - 2D and 3D convolutions
//! - Transposed convolutions
//! - Pooling operations (max pooling, average pooling)
//! - Specialized convolutions (depthwise, grouped)

mod conv2d;
mod conv3d;
mod conv_transpose;
mod pooling;
mod specialized;
mod types;
mod utils;

// Re-export type aliases
pub use types::*;

// Re-export Conv2D operations
pub use conv2d::conv2d_backward;

// Re-export Conv3D operations
pub use conv3d::conv3d_backward;

// Re-export ConvTranspose operations
pub use conv_transpose::conv_transpose2d_backward;

// Re-export pooling operations
pub use pooling::{avg_pool2d_backward, max_pool2d_backward};

// Re-export specialized convolutions
pub use specialized::{depthwise_conv2d_backward, grouped_conv2d_backward};
