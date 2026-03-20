//! Convolution Layers - Modular Architecture
//!
//! This module has been refactored into a modular architecture for better maintainability
//! and organization. The functionality has been split into specialized modules by convolution type:
//!
//! ## Module Organization
//!
//! - **conv1d**: 1D convolution layers for sequence processing
//! - **conv2d**: 2D convolution layers for image and spatial data processing
//! - **conv3d**: 3D convolution layers for volumetric data processing
//! - **depthwise**: Depthwise separable convolution layers for efficient mobile architectures
//! - **separable**: Separable convolution layers combining depthwise and pointwise convolutions
//! - **transpose**: Transposed convolution layers for upsampling and deconvolution
//!
//! All layers maintain 100% backward compatibility through strategic re-exports.

// Import the modularized convolution layer functionality
pub mod conv1d;
pub mod conv2d;
pub mod conv3d;
pub mod depthwise;
pub mod separable;
pub mod transpose;

// Re-export all convolution layers for backward compatibility

// Standard convolution layers
pub use conv1d::Conv1D;
pub use conv2d::Conv2D;
pub use conv3d::Conv3D;

// Specialized convolution layers
pub use depthwise::DepthwiseConv2D;
pub use separable::SeparableConv2D;
pub use transpose::{AntiCheckerboardMode, ConvTranspose2D};
