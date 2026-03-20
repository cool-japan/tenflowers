//! Computer Vision Models Module
//!
//! This module contains state-of-the-art computer vision models including
//! convolutional neural networks and vision transformers.

// ResNet models and related types
pub mod resnet;

// EfficientNet models and configuration
pub mod efficientnet;

// Vision Transformer (ViT) and related components
pub mod vision_transformer;

// Re-export all vision models for backward compatibility
pub use efficientnet::*;
pub use resnet::*;
pub use vision_transformer::*;
