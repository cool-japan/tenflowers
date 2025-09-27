//! Multimodal dataset support for modern LLM training
//!
//! This module provides comprehensive support for multimodal datasets that combine
//! text, images, audio, and other modalities commonly used in modern LLM training.
//!
//! ## Module Organization
//!
//! - **types**: Core types and enums (Modality, FusionStrategy)
//! - **sample**: MultimodalSample implementation
//! - **config**: Configuration structures and defaults
//! - **dataset**: Main MultimodalDataset implementation
//! - **builder**: Dataset builder pattern implementation
//! - **transforms**: Transformed dataset implementations

pub mod builder;
pub mod config;
pub mod dataset;
pub mod sample;
pub mod transforms;
pub mod types;

// Re-export all public types for convenience
pub use builder::{BuilderStatistics, MultimodalDatasetBuilder};
pub use config::MultimodalConfig;
pub use dataset::{MultimodalDataset, MultimodalDatasetSummary};
pub use sample::MultimodalSample;
pub use transforms::{
    ComposedTransform, ConditionalTransform, Identity, ModalitySpecificTransform,
    MultimodalTransform, MultimodalTransformedDataset, ProbabilisticTransform,
};
pub use types::{FusionStrategy, Modality};
