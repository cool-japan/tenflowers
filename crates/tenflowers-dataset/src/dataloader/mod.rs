//! DataLoader Module
//!
//! This module provides comprehensive data loading utilities for machine learning workflows.
//! It has been refactored into specialized submodules for better organization and maintainability.
//!
//! # Available Components
//!
//! - **Batch Results**: Types for handling both individual samples and collated tensor batches
//! - **Collation**: Various strategies for combining samples into batches with padding support
//! - **Samplers**: Different sampling strategies including sequential, random, distributed, stratified, and importance-based
//! - **Core DataLoader**: Multi-threaded data loader with prefetching and NUMA-aware scheduling
//! - **Builder**: Fluent API for configuring and creating DataLoader instances
//!
//! # Usage
//!
//! ```rust,ignore
//! use tenflowers_dataset::dataloader::{DataLoaderBuilder, SequentialSampler};
//!
//! // Create a DataLoader with custom configuration
//! let dataloader = DataLoaderBuilder::new(dataset)
//!     .batch_size(32)
//!     .num_workers(4)
//!     .prefetch_factor(2)
//!     .drop_last(true)
//!     .build(SequentialSampler::new());
//!
//! // Iterate over batches
//! for batch in dataloader.iter() {
//!     let batch = batch.unwrap();
//!     // Process batch...
//! }
//! ```

// Import all specialized dataloader modules
pub mod batch_result;
pub mod builder;
pub mod collate;
pub mod core;
pub mod samplers;

// Re-export all public types for backward compatibility
// This ensures existing code continues to work without modification

// Batch Result Types
pub use batch_result::BatchResult;

// Collation Functions
pub use collate::{BucketCollate, CollateFn, DefaultCollate, PaddingCollate, PaddingStrategy};

// Sampling Strategies
pub use samplers::{
    DistributedSampler, ImportanceSampler, RandomSampler, Sampler, SequentialSampler,
    StratifiedSampler,
};

// Core DataLoader
pub use core::{DataLoader, DataLoaderConfig, DataLoaderIterator};

// Builder Pattern
pub use builder::DataLoaderBuilder;
