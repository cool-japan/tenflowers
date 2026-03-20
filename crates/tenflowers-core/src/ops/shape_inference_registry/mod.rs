//! Shape Inference Registry for TenfloweRS
//!
//! This module provides a centralized registry for shape inference rules across
//! all tensor operations, ensuring consistent shape validation and error reporting.
//!
//! ## Design Goals
//! - **Centralization**: Single source of truth for shape inference logic
//! - **Standardization**: Consistent error messages using ShapeErrorTaxonomy
//! - **Discoverability**: Easy to find shape inference rules for any operation
//! - **Type Safety**: Compile-time guarantees where possible
//!
//! ## Usage
//!
//! ```rust,ignore
//! use tenflowers_core::ops::shape_inference_registry::{ShapeInferenceRegistry, get_registry};
//!
//! // Get the global registry
//! let registry = get_registry();
//!
//! // Infer output shape for an operation
//! let output_shape = registry.infer("add", &[input1.shape(), input2.shape()], &metadata)?;
//!
//! // Validate inputs for an operation
//! registry.validate("matmul", &[a.shape(), b.shape()], &metadata)?;
//! ```
//!
//! Submodules:
//! - `types`: `OperationMetadata`, `MetadataValue`, `OperationCategory`, `ShapeInferenceFn`
//! - `registry`: `ShapeInferenceRegistry` struct + built-in registration
//! - `infer_fns`: all shape inference function implementations
//! - `global`: global singleton registry access (`get_registry`, `initialize_registry`)

pub mod global;
pub mod infer_fns;
pub mod registry;
pub mod types;

mod tests;

pub use global::{get_registry, initialize_registry};
pub use registry::ShapeInferenceRegistry;
pub use types::{MetadataValue, OperationCategory, OperationMetadata, ShapeInferenceFn};
