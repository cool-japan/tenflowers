//! Pretrained Models Module - Modular Architecture for Deep Learning Models
//!
//! This module provides a comprehensive collection of pretrained deep learning models
//! organized by domain and architecture type for better maintainability and clarity.
//!
//! ## Module Organization
//!
//! - **common**: Shared building blocks and activation functions used across models
//! - **vision**: Computer vision models (ResNet, EfficientNet, Vision Transformer)
//! - **nlp**: Natural language processing models (BERT, GPT)

// Common building blocks and utilities
pub mod common;

// Vision models
pub mod vision;

// Natural language processing models
pub mod nlp;

// Re-export common building blocks for backward compatibility
pub use common::*;

// Re-export all vision models for backward compatibility
pub use vision::*;

// Re-export all NLP models for backward compatibility
pub use nlp::*;
