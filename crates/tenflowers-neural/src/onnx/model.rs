//! ONNX Model Structure and Operations
//!
//! This module contains the main OnnxModel structure and its operations
//! for serialization, file I/O, and conversion between different formats.

use super::data::OnnxGraph;
use super::types::{OnnxError, OnnxFormat};
use std::fs::File;
use std::io::{Read, Write};
use tenflowers_core::{Result, TensorError};

#[cfg(feature = "onnx")]
use super::proto;

/// ONNX model representation
#[derive(Debug, Clone)]
pub struct OnnxModel {
    pub ir_version: i64,
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: i64,
    pub doc_string: String,
    pub graph: OnnxGraph,
}

impl OnnxModel {
    /// Create a new ONNX model
    pub fn new(graph: OnnxGraph) -> Self {
        Self {
            ir_version: 7,
            producer_name: "TenfloweRS".to_string(),
            producer_version: "0.1.0".to_string(),
            domain: "ai.tenflowers".to_string(),
            model_version: 1,
            doc_string: "Model exported from TenfloweRS".to_string(),
            graph,
        }
    }

    /// Convert to JSON format
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            TensorError::serialization_error_simple(format!("JSON serialization failed: {}", e))
        })
    }

    /// Create from JSON format
    pub fn from_json(json_str: &str) -> Result<Self> {
        serde_json::from_str(json_str).map_err(|e| {
            TensorError::serialization_error_simple(format!("JSON deserialization failed: {}", e))
        })
    }

    /// Load model from file with format auto-detection
    pub fn from_file(path: &str) -> Result<Self> {
        // Try to detect format from file extension or content
        if path.ends_with(".json") {
            let content = std::fs::read_to_string(path).map_err(|e| {
                TensorError::serialization_error_simple(format!("Failed to read file: {}", e))
            })?;
            Self::from_json(&content)
        } else {
            // Assume protobuf format
            #[cfg(feature = "onnx")]
            {
                let content = std::fs::read(path).map_err(|e| {
                    TensorError::serialization_error_simple(format!("Failed to read file: {}", e))
                })?;
                Self::from_protobuf(&content)
            }
            #[cfg(not(feature = "onnx"))]
            {
                // Fallback to JSON
                let content = std::fs::read_to_string(path).map_err(|e| {
                    TensorError::serialization_error_simple(format!("Failed to read file: {}", e))
                })?;
                Self::from_json(&content)
            }
        }
    }

    /// Convert to protobuf format
    #[cfg(feature = "onnx")]
    pub fn to_protobuf(&self) -> Result<proto::ModelProto> {
        let graph_proto = self.graph.to_protobuf()?;

        Ok(proto::ModelProto {
            ir_version: Some(self.ir_version),
            opset_import: vec![proto::OperatorSetIdProto {
                domain: Some("".to_string()),
                version: Some(13),
            }],
            producer_name: Some(self.producer_name.clone()),
            producer_version: Some(self.producer_version.clone()),
            domain: Some(self.domain.clone()),
            model_version: Some(self.model_version),
            doc_string: Some(self.doc_string.clone()),
            graph: Some(graph_proto),
        })
    }

    /// Create from protobuf format
    #[cfg(feature = "onnx")]
    pub fn from_protobuf(data: &[u8]) -> Result<Self> {
        use prost::Message;

        let proto_model = proto::ModelProto::decode(data).map_err(|e| {
            TensorError::serialization_error_simple(format!("Protobuf decode failed: {}", e))
        })?;

        let graph = if let Some(graph_proto) = proto_model.graph {
            OnnxGraph::from_protobuf(&graph_proto)?
        } else {
            return Err(TensorError::serialization_error_simple(
                "Missing graph in model".to_string(),
            ));
        };

        Ok(OnnxModel {
            ir_version: proto_model.ir_version.unwrap_or(7),
            producer_name: proto_model
                .producer_name
                .unwrap_or_else(|| "Unknown".to_string()),
            producer_version: proto_model
                .producer_version
                .unwrap_or_else(|| "Unknown".to_string()),
            domain: proto_model.domain.unwrap_or_else(|| "".to_string()),
            model_version: proto_model.model_version.unwrap_or(1),
            doc_string: proto_model.doc_string.unwrap_or_else(|| "".to_string()),
            graph,
        })
    }

    /// Save model to file in specified format
    pub fn save_to_file(&self, path: &str, format: OnnxFormat) -> Result<()> {
        match format {
            OnnxFormat::Json => {
                let json_str = self.to_json()?;
                let mut file = File::create(path).map_err(|e| {
                    TensorError::serialization_error_simple(format!("Failed to create file: {}", e))
                })?;
                file.write_all(json_str.as_bytes()).map_err(|e| {
                    TensorError::serialization_error_simple(format!("Failed to write file: {}", e))
                })?;
            }
            OnnxFormat::Protobuf => {
                #[cfg(feature = "onnx")]
                {
                    use prost::Message;
                    let proto_model = self.to_protobuf()?;
                    let buffer = proto_model.encode_to_vec();
                    let mut file = File::create(path).map_err(|e| {
                        TensorError::serialization_error_simple(format!(
                            "Failed to create file: {}",
                            e
                        ))
                    })?;
                    file.write_all(&buffer).map_err(|e| {
                        TensorError::serialization_error_simple(format!(
                            "Failed to write file: {}",
                            e
                        ))
                    })?;
                }
                #[cfg(not(feature = "onnx"))]
                {
                    return Err(TensorError::serialization_error_simple(
                        "Protobuf format requires 'onnx' feature to be enabled".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }
}
