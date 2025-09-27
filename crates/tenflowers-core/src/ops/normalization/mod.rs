//! Normalization Operations
//!
//! This module provides comprehensive normalization operations for deep learning,
//! including batch normalization, layer normalization, group normalization,
//! and synchronized batch normalization. All operations support both CPU and GPU
//! execution with optimized implementations for different tensor layouts.
//!
//! # Features
//! - Batch Normalization (BatchNorm) for CNN layers
//! - Layer Normalization (LayerNorm) for transformer architectures
//! - Group Normalization (GroupNorm) for stable training with small batch sizes
//! - Synchronized Batch Normalization for distributed training
//! - GPU-optimized implementations with memory coalescing
//! - Training and inference modes with running statistics
//!
//! # Architecture
//! Each normalization type is implemented in its own module with both CPU and GPU
//! variants, providing optimal performance across different hardware configurations.
//!
//! # Modules
//! - [`batch_norm`] - Standard batch normalization operations
//! - [`layer_norm`] - Layer normalization for sequence models
//! - [`group_norm`] - Group normalization for stable training
//! - [`sync_batch_norm`] - Synchronized batch normalization for distributed training

// Re-export specialized modules
pub mod batch_norm;
pub mod group_norm;
pub mod layer_norm;
pub mod sync_batch_norm;

#[cfg(test)]
pub mod tests;

// Re-export core functions for backward compatibility
pub use batch_norm::batch_norm;
pub use group_norm::group_norm;
pub use layer_norm::layer_norm;
pub use sync_batch_norm::sync_batch_norm;
