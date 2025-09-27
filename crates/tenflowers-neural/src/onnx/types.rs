//! Basic Types, Errors, and Format Definitions
//!
//! This module provides the foundational types for ONNX support including
//! error types, format options, and data type definitions used throughout
//! the ONNX export/import functionality.

/// ONNX export errors
#[derive(Debug, thiserror::Error)]
pub enum OnnxError {
    #[error("Model conversion error: {0}")]
    ConversionError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[cfg(feature = "onnx")]
    #[error("Protobuf error: {0}")]
    ProtobufError(#[from] prost::DecodeError),
    #[cfg(feature = "onnx")]
    #[error("Protobuf encode error: {0}")]
    ProtobufEncodeError(#[from] prost::EncodeError),
}

/// ONNX format type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxFormat {
    /// Official ONNX protobuf format (requires onnx feature)
    Protobuf,
    /// Simplified JSON format for compatibility
    Json,
}

impl Default for OnnxFormat {
    fn default() -> Self {
        #[cfg(feature = "onnx")]
        {
            Self::Protobuf
        }
        #[cfg(not(feature = "onnx"))]
        {
            Self::Json
        }
    }
}

/// ONNX data types
#[derive(Debug, Clone, Copy)]
pub enum OnnxDataType {
    Float32 = 1,
    Float64 = 11,
    Int32 = 6,
    Int64 = 7,
}
