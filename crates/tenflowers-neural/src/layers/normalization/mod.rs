//! Normalization Layers Module
//!
//! This module provides various normalization techniques for neural networks.
//! Each normalization method is implemented in its own specialized module
//! for better organization and maintainability.
//!
//! # Available Normalization Layers
//!
//! - **BatchNorm**: Batch Normalization with training/inference modes and running statistics
//! - **LayerNorm**: Layer Normalization for transformer and sequence models
//! - **InstanceNorm**: Instance Normalization for style transfer tasks
//! - **GroupNorm**: Group Normalization as middle ground between LayerNorm and InstanceNorm
//! - **RMSNorm**: RMS Normalization used in modern language models like LLaMA
//! - **SpectralNorm**: Spectral Normalization for GAN training stability
//! - **WeightNorm**: Weight Normalization to decouple magnitude and direction
//! - **LocalResponseNorm**: Local Response Normalization from older CNN architectures
//!
//! # Usage
//!
//! ```rust,ignore
//! use tenflowers_neural::layers::normalization::{BatchNorm, LayerNorm, RMSNorm};
//!
//! // Batch normalization for CNN layers
//! let batch_norm = BatchNorm::<f32>::new(64)?;
//!
//! // Layer normalization for transformers
//! let layer_norm = LayerNorm::<f32>::new(&[768]);
//!
//! // RMS normalization for modern language models
//! let rms_norm = RMSNorm::<f32>::new(&[768]).with_epsilon(1e-6);
//! ```

// Import all specialized normalization modules
pub mod batch_norm;
pub mod group_norm;
pub mod instance_norm;
pub mod layer_norm;
pub mod local_response_norm;
pub mod rms_norm;
pub mod spectral_norm;
pub mod weight_norm;

// Re-export all public types for backward compatibility
// This ensures existing code continues to work without modification

// Batch Normalization
pub use batch_norm::BatchNorm;

// Layer Normalization
pub use layer_norm::LayerNorm;

// Instance Normalization
pub use instance_norm::InstanceNorm;

// Group Normalization
pub use group_norm::GroupNorm;

// RMS Normalization
pub use rms_norm::RMSNorm;

// Spectral Normalization
pub use spectral_norm::SpectralNorm;

// Weight Normalization
pub use weight_norm::WeightNorm;

// Local Response Normalization
pub use local_response_norm::{LocalResponseNorm, LocalResponseNormMode};
