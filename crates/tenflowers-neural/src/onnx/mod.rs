//! ONNX Export/Import Support - Modular Architecture
//!
//! This module has been refactored into a modular architecture for better maintainability
//! and organization. The functionality has been split into specialized modules by feature area:
//!
//! ## Module Organization
//!
//! - **types**: Basic types, error definitions, and format enums
//! - **proto**: Protobuf schema definitions (when onnx feature is enabled)
//! - **data**: Core ONNX data structures (graphs, nodes, tensors, etc.)
//! - **model**: ONNX model structure and file I/O operations
//! - **traits**: Export and import trait definitions
//! - **sequential**: Sequential model ONNX implementation
//! - **serialization**: Serde implementations for JSON support
//!
//! All functionality maintains 100% backward compatibility through strategic re-exports.

// Import the modularized ONNX functionality
pub mod data;
pub mod model;
pub mod proto;
pub mod sequential;
pub mod serialization;
pub mod traits;
pub mod types;

// Re-export all types and functionality for backward compatibility

// Basic types and errors
pub use types::{OnnxDataType, OnnxError, OnnxFormat};

// Protobuf definitions (when onnx feature is enabled)
#[cfg(feature = "onnx")]
pub use proto::proto as onnx_proto;

// Core data structures
pub use data::{OnnxAttribute, OnnxGraph, OnnxNode, OnnxTensor, OnnxValueInfo};

// Model structure
pub use model::OnnxModel;

// Export/Import traits
pub use traits::{OnnxExport, OnnxImport};

// Sequential model implementations are automatically available through traits

// Note: Serialization implementations are used internally and don't need re-export
