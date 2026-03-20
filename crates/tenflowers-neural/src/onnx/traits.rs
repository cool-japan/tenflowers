//! ONNX Export and Import Traits
//!
//! This module defines the core traits for exporting models to ONNX format
//! and importing models from ONNX format, providing a standard interface
//! for ONNX interoperability.

use super::model::OnnxModel;
use super::types::OnnxFormat;
use std::fs::File;
use std::io::{Read, Write};
use tenflowers_core::{Result, TensorError};

/// Trait for exporting models to ONNX format
pub trait OnnxExport<T> {
    /// Export model to ONNX format
    fn to_onnx(&self, input_shape: &[usize]) -> Result<OnnxModel>;

    /// Save model to ONNX file with specified format
    fn save_onnx_format(
        &self,
        path: &str,
        input_shape: &[usize],
        format: OnnxFormat,
    ) -> Result<()> {
        let onnx_model = self.to_onnx(input_shape)?;
        onnx_model.save_to_file(path, format)
    }

    /// Save model to ONNX file (uses default format)
    fn save_onnx(&self, path: &str, input_shape: &[usize]) -> Result<()> {
        self.save_onnx_format(path, input_shape, OnnxFormat::default())
    }
}

/// Trait for importing models from ONNX format
pub trait OnnxImport<T> {
    /// Import model from ONNX format
    fn from_onnx(onnx_model: &OnnxModel) -> Result<Self>
    where
        Self: Sized;

    /// Load model from ONNX file with auto-detection of format
    fn load_onnx(path: &str) -> Result<Self>
    where
        Self: Sized,
    {
        let onnx_model = OnnxModel::from_file(path)?;
        Self::from_onnx(&onnx_model)
    }

    /// Load model from ONNX file with specified format
    fn load_onnx_format(path: &str, format: OnnxFormat) -> Result<Self>
    where
        Self: Sized,
    {
        let onnx_model = match format {
            OnnxFormat::Json => {
                let content = std::fs::read_to_string(path).map_err(|e| {
                    TensorError::serialization_error_simple(format!("Failed to read file: {}", e))
                })?;
                OnnxModel::from_json(&content)?
            }
            OnnxFormat::Protobuf => {
                #[cfg(feature = "onnx")]
                {
                    let content = std::fs::read(path).map_err(|e| {
                        TensorError::serialization_error_simple(format!(
                            "Failed to read file: {}",
                            e
                        ))
                    })?;
                    OnnxModel::from_protobuf(&content)?
                }
                #[cfg(not(feature = "onnx"))]
                {
                    return Err(TensorError::serialization_error_simple(
                        "Protobuf format requires 'onnx' feature to be enabled".to_string(),
                    ));
                }
            }
        };
        Self::from_onnx(&onnx_model)
    }
}
