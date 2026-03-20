//! Pooling operations for neural networks
//!
//! This module provides comprehensive pooling implementations including:
//! - Basic max and average pooling in 2D and 3D
//! - Global pooling operations
//! - Adaptive pooling with variable output sizes
//! - Fractional pooling for flexible downsampling
//! - ROI pooling for object detection models

#[cfg(feature = "gpu")]
use crate::Shape;

// Re-export specialized modules
pub mod adaptive_pooling;
pub mod basic_pooling;
pub mod fractional_pooling;
pub mod global_pooling;
pub mod roi_pooling;

// Re-export commonly used functions
pub use adaptive_pooling::{adaptive_avg_pool2d, adaptive_max_pool2d};
pub use basic_pooling::{avg_pool2d, avg_pool3d, max_pool2d, max_pool3d};
pub use fractional_pooling::{fractional_avg_pool2d, fractional_max_pool2d};
pub use global_pooling::{
    global_avg_pool2d, global_avg_pool3d, global_max_pool2d, global_max_pool3d,
};
pub use roi_pooling::{roi_align2d, roi_pool2d};
