//! ONNX import/export functionality for model interoperability
//!
//! This module provides comprehensive ONNX (Open Neural Network Exchange) support,
//! enabling seamless model conversion between TenfloweRS and other ML frameworks
//! including PyTorch, TensorFlow, Keras, scikit-learn, and more.
//!
//! - `types`: the in-memory `Onnx*` data model (always available).
//! - `proto` / `convert`: the hand-rolled protobuf schema and its
//!   bidirectional conversion to/from `types::Onnx*`; only compiled when the
//!   `onnx` feature is enabled (`ModelProto::decode`/`.encode_to_vec()` are
//!   actually invoked there).
//! - `lowering`: mapping parsed `OnnxNode`s to TenfloweRS's own executable
//!   op representation (`OnnxOpMapping`/`TenfloweRSOperation`); always
//!   available, since it operates on in-memory structs without touching
//!   protobuf at all.

pub mod types;

#[cfg(feature = "onnx")]
pub mod convert;
pub mod lowering;
#[cfg(feature = "onnx")]
pub mod proto;

// Re-export all public types for convenience
pub use lowering::{lower_graph, StandardOpMapping};
pub use types::*;

use crate::{Result, TensorError};
use std::path::Path;

/// ONNX model importer
pub struct OnnxImporter {
    /// Import configuration
    config: OnnxConfig,
}

/// ONNX model exporter
pub struct OnnxExporter {
    /// Export configuration
    config: OnnxConfig,
}

impl OnnxImporter {
    /// Create a new ONNX importer with default configuration
    pub fn new() -> Self {
        Self {
            config: OnnxConfig::default(),
        }
    }

    /// Create a new ONNX importer with custom configuration
    pub fn with_config(config: OnnxConfig) -> Self {
        Self { config }
    }

    /// Import ONNX model from file path
    #[cfg(feature = "onnx")]
    pub fn import_from_file<P: AsRef<Path>>(&self, path: P) -> Result<OnnxModel> {
        let bytes = std::fs::read(path.as_ref()).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "Failed to read ONNX file {:?}: {e}",
                path.as_ref()
            ))
        })?;
        self.import_from_bytes(&bytes)
    }

    /// Import ONNX model from file path
    #[cfg(not(feature = "onnx"))]
    pub fn import_from_file<P: AsRef<Path>>(&self, path: P) -> Result<OnnxModel> {
        let _ = self.config.opset_version;
        Err(TensorError::not_implemented_simple(format!(
            "ONNX protobuf import requires the 'onnx' feature; cannot load model from {:?}. \
             Rebuild with `--features onnx` to enable.",
            path.as_ref()
        )))
    }

    /// Import ONNX model from bytes
    #[cfg(feature = "onnx")]
    pub fn import_from_bytes(&self, bytes: &[u8]) -> Result<OnnxModel> {
        // Reserved for future opset-compatibility validation against
        // `self.config.opset_version`; the model's own `opset_imports` are
        // decoded and preserved as-is for now.
        let _ = self.config.opset_version;
        OnnxModel::from_protobuf(bytes)
    }

    /// Import ONNX model from bytes
    #[cfg(not(feature = "onnx"))]
    pub fn import_from_bytes(&self, _bytes: &[u8]) -> Result<OnnxModel> {
        let _ = self.config.opset_version;
        Err(TensorError::not_implemented_simple(
            "ONNX protobuf import requires the 'onnx' feature; rebuild with `--features onnx` \
             to enable."
                .to_string(),
        ))
    }
}

impl OnnxExporter {
    /// Create a new ONNX exporter with default configuration
    pub fn new() -> Self {
        Self {
            config: OnnxConfig::default(),
        }
    }

    /// Create a new ONNX exporter with custom configuration
    pub fn with_config(config: OnnxConfig) -> Self {
        Self { config }
    }

    /// Export TenfloweRS model to ONNX format
    #[cfg(feature = "onnx")]
    pub fn export_to_file<P: AsRef<Path>>(&self, model: &OnnxModel, path: P) -> Result<()> {
        let bytes = self.export_to_bytes(model)?;
        std::fs::write(path.as_ref(), bytes).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "Failed to write ONNX file {:?}: {e}",
                path.as_ref()
            ))
        })
    }

    /// Export TenfloweRS model to ONNX format
    #[cfg(not(feature = "onnx"))]
    pub fn export_to_file<P: AsRef<Path>>(&self, model: &OnnxModel, path: P) -> Result<()> {
        let _ = (model, &self.config);
        Err(TensorError::not_implemented_simple(format!(
            "ONNX protobuf export requires the 'onnx' feature; cannot write model to {:?}. \
             Rebuild with `--features onnx` to enable.",
            path.as_ref()
        )))
    }

    /// Export TenfloweRS model to ONNX bytes.
    ///
    /// Honors `self.config.opset_version` for real: if `model.opset_imports`
    /// is empty, a single `OperatorSetIdProto` for the default ONNX domain
    /// is synthesized from the exporter's configured opset version. If the
    /// model already specifies its own opset imports, those are used
    /// as-is and the configured opset version is not applied (explicit
    /// model data takes priority over the exporter's default).
    ///
    /// `OnnxFormatOptions::{use_external_data, external_data_threshold,
    /// compress_large_tensors}` are out of scope for this slice: tensor
    /// data is always emitted inline via `TensorProto.raw_data`, with no
    /// external-data-file splitting and no compression.
    #[cfg(feature = "onnx")]
    pub fn export_to_bytes(&self, model: &OnnxModel) -> Result<Vec<u8>> {
        use prost::Message;

        let mut proto_model = model.to_protobuf()?;
        if proto_model.opset_import.is_empty() {
            proto_model.opset_import.push(proto::OperatorSetIdProto {
                domain: Some(String::new()),
                version: Some(self.config.opset_version),
            });
        }
        Ok(proto_model.encode_to_vec())
    }

    /// Export TenfloweRS model to ONNX bytes
    #[cfg(not(feature = "onnx"))]
    pub fn export_to_bytes(&self, model: &OnnxModel) -> Result<Vec<u8>> {
        let _ = (model, &self.config);
        Err(TensorError::not_implemented_simple(
            "ONNX protobuf export requires the 'onnx' feature; rebuild with `--features onnx` \
             to enable."
                .to_string(),
        ))
    }
}

impl Default for OnnxImporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for OnnxExporter {
    fn default() -> Self {
        Self::new()
    }
}
