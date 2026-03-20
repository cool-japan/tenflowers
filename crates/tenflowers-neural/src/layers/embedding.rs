//! Embedding Layers - Modular Architecture
//!
//! This module has been refactored into a modular architecture for better maintainability
//! and organization. The functionality has been split into specialized modules by embedding type:
//!
//! ## Module Organization
//!
//! - **basic**: Basic embedding layers with regularization and dropout support
//! - **positional**: Positional encoding layers (sinusoidal, learned, rotary/RoPE)
//! - **sparse**: Sparse embedding layers with gradient optimization for large vocabularies
//!
//! All layers maintain 100% backward compatibility through strategic re-exports.

// Import the modularized embedding layer functionality
pub mod basic;
pub mod positional;
pub mod sparse;

// Re-export all embedding layers and utilities for backward compatibility

// Basic embedding functionality
pub use basic::{Embedding, EmbeddingRegularization};

// Positional encoding layers
pub use positional::{
    LearnedPositionalEncoding, RotaryPositionalEmbedding, SinusoidalPositionalEncoding,
};

// Sparse embedding functionality
pub use sparse::{SparseEmbedding, SparseEmbeddingGrad, SparseGradientStats};
