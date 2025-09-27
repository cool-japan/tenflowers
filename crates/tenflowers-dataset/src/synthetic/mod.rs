//! Synthetic Dataset Generation - Modular Architecture
//!
//! This module has been refactored into a modular architecture for better maintainability
//! and organization. The functionality has been split into specialized modules by feature area:
//!
//! ## Module Organization
//!
//! - **core**: Configuration types and base dataset structures
//! - **classic**: Classical ML dataset generation (moons, circles, blobs, etc.)
//! - **timeseries**: Time series data generation with various patterns
//! - **image**: Image pattern generation for computer vision tasks
//! - **text**: Text corpus generation for NLP tasks
//! - **modern_ml**: Modern ML paradigms (few-shot, meta-learning, contrastive learning)
//! - **tests**: Comprehensive test suite for all generation functions
//!
//! All functionality maintains 100% backward compatibility through strategic re-exports.

pub mod classic;
pub mod core;
pub mod image;
pub mod modern_ml;
pub mod tests;
pub mod text;
pub mod timeseries;

// Re-export core types for backward compatibility
pub use core::{DatasetGenerator, SyntheticConfig, SyntheticDataset};

// Re-export time series types
pub use timeseries::TimeSeriesPattern;

// Re-export image generation types
pub use image::{
    GeometricShape, GradientDirection, ImagePatternConfig, ImagePatternGenerator, ImagePatternType,
    NoiseDistribution, StripeOrientation,
};

// Re-export text generation types
pub use text::{SyntheticTextCorpus, TextCorpusConfig, TextSynthesisTask};

// Re-export modern ML types
pub use modern_ml::{
    ContrastiveLearningDataset, Episode, FewShotDataset, MetaLearningDataset, ModernMLConfig,
    SelfSupervisedDataset, TaskDataset,
};

// Re-export all generation functions through DatasetGenerator
// These are automatically available through the core module re-export

// Note: Test functions are kept internal to the tests module
// and don't need re-export as they are for verification purposes only
