//! Unified format reader abstraction
//!
//! This module provides a common interface for reading data from various formats
//! (JSON, CSV, Parquet, HDF5, etc.) with automatic format detection, schema
//! validation, and unified error handling.

use crate::error_taxonomy::{helpers as error_helpers, DatasetErrorContext};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tenflowers_core::{DType, Result, Tensor, TensorError};

/// Metadata about a data format
#[derive(Debug, Clone)]
pub struct FormatMetadata {
    /// Format name (e.g., "json", "csv", "parquet")
    pub format_name: String,
    /// Format version (if applicable)
    pub version: Option<String>,
    /// Number of samples/records
    pub num_samples: usize,
    /// Field/column information
    pub fields: Vec<FieldInfo>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Whether format supports random access
    pub supports_random_access: bool,
    /// Whether format supports streaming
    pub supports_streaming: bool,
}

/// Information about a field/column in the dataset
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldInfo {
    /// Field name
    pub name: String,
    /// Data type
    pub dtype: DataType,
    /// Shape information (for tensor fields)
    pub shape: Option<Vec<usize>>,
    /// Whether field is nullable
    pub nullable: bool,
    /// Description or documentation
    pub description: Option<String>,
}

/// Unified data type representation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataType {
    /// Boolean
    Bool,
    /// Signed integers
    Int8,
    Int16,
    Int32,
    Int64,
    /// Unsigned integers
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    /// Floating point
    Float32,
    Float64,
    /// String/text
    String,
    /// Binary/bytes
    Binary,
    /// Nested structure
    Struct(Vec<FieldInfo>),
    /// List/array
    List(Box<DataType>),
    /// Tensor with specific dtype
    Tensor(DType),
}

impl DataType {
    /// Convert to TenfloweRS DType if possible
    pub fn to_tensor_dtype(&self) -> Option<DType> {
        match self {
            DataType::Float32 => Some(DType::Float32),
            DataType::Float64 => Some(DType::Float64),
            DataType::Int32 => Some(DType::Int32),
            DataType::Int64 => Some(DType::Int64),
            DataType::Tensor(dtype) => Some(*dtype),
            _ => None,
        }
    }
}

/// Sample read from a format reader
#[derive(Debug, Clone)]
pub struct FormatSample {
    /// Feature tensor
    pub features: Tensor<f32>,
    /// Label tensor
    pub labels: Tensor<f32>,
    /// Original index in the source
    pub source_index: usize,
    /// Additional metadata for this sample
    pub metadata: HashMap<String, String>,
}

/// Trait for unified format reading
pub trait FormatReader: Send + Sync {
    /// Get format metadata
    fn metadata(&self) -> Result<FormatMetadata>;

    /// Get a sample by index
    fn get_sample(&self, index: usize) -> Result<FormatSample>;

    /// Get multiple samples efficiently
    fn get_samples(&self, indices: &[usize]) -> Result<Vec<FormatSample>> {
        indices.iter().map(|&i| self.get_sample(i)).collect()
    }

    /// Iterate through all samples
    fn iter(&self) -> Box<dyn Iterator<Item = Result<FormatSample>> + '_>;

    /// Validate schema against expected format
    fn validate_schema(&self, expected: &[FieldInfo]) -> Result<()> {
        let metadata = self.metadata()?;

        if metadata.fields.len() != expected.len() {
            return Err(error_helpers::schema_mismatch(
                "validate_schema",
                format!("{} fields", expected.len()),
                format!("{} fields", metadata.fields.len()),
            ));
        }

        for (actual, expected) in metadata.fields.iter().zip(expected.iter()) {
            if actual.name != expected.name {
                return Err(error_helpers::schema_mismatch(
                    "validate_schema",
                    format!("field name '{}'", expected.name),
                    format!("field name '{}'", actual.name),
                ));
            }

            if actual.dtype != expected.dtype {
                return Err(error_helpers::schema_mismatch(
                    "validate_schema",
                    format!("field '{}' type {:?}", expected.name, expected.dtype),
                    format!("field '{}' type {:?}", actual.name, actual.dtype),
                ));
            }
        }

        Ok(())
    }

    /// Check if format supports random access
    fn supports_random_access(&self) -> bool {
        self.metadata()
            .map(|m| m.supports_random_access)
            .unwrap_or(false)
    }

    /// Check if format supports streaming
    fn supports_streaming(&self) -> bool {
        self.metadata()
            .map(|m| m.supports_streaming)
            .unwrap_or(true)
    }

    /// Get total number of samples
    fn len(&self) -> usize {
        self.metadata().map(|m| m.num_samples).unwrap_or(0)
    }

    /// Check if reader is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Format detection result
#[derive(Debug, Clone)]
pub struct FormatDetection {
    /// Detected format name
    pub format_name: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Detection method used
    pub method: DetectionMethod,
}

/// Method used for format detection
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectionMethod {
    /// File extension
    Extension,
    /// Magic bytes/header
    MagicBytes,
    /// Content analysis
    ContentAnalysis,
    /// Explicit specification
    Explicit,
}

/// Format registry for auto-detection and creation
pub struct FormatRegistry {
    /// Registered format factories
    factories: HashMap<String, Box<dyn FormatFactory>>,
}

/// Factory for creating format readers
pub trait FormatFactory: Send + Sync {
    /// Format name
    fn format_name(&self) -> &str;

    /// File extensions supported
    fn extensions(&self) -> Vec<&str>;

    /// Detect if this factory can read the given file
    fn can_read(&self, path: &Path) -> Result<FormatDetection>;

    /// Create a reader for the given path
    fn create_reader(&self, path: &Path) -> Result<Box<dyn FormatReader>>;
}

impl FormatRegistry {
    /// Create a new format registry
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Register a format factory
    pub fn register(&mut self, factory: Box<dyn FormatFactory>) {
        self.factories
            .insert(factory.format_name().to_string(), factory);
    }

    /// Auto-detect format and create reader
    pub fn auto_detect(&self, path: &Path) -> Result<Box<dyn FormatReader>> {
        let mut detections = Vec::new();

        // Try all factories
        for factory in self.factories.values() {
            if let Ok(detection) = factory.can_read(path) {
                detections.push((detection, factory));
            }
        }

        if detections.is_empty() {
            return Err(error_helpers::data_corruption(
                "auto_detect",
                "No compatible format found",
                Some(path.to_path_buf()),
            ));
        }

        // Sort by confidence
        detections.sort_by(|a, b| b.0.confidence.partial_cmp(&a.0.confidence).unwrap());

        // Use highest confidence factory
        let (detection, factory) = &detections[0];

        if detection.confidence < 0.5 {
            return Err(error_helpers::data_corruption(
                "auto_detect",
                format!("Low confidence detection: {:.2}", detection.confidence),
                Some(path.to_path_buf()),
            ));
        }

        factory.create_reader(path)
    }

    /// Create reader for specific format
    pub fn create_reader(&self, format: &str, path: &Path) -> Result<Box<dyn FormatReader>> {
        match self.factories.get(format) {
            Some(factory) => factory.create_reader(path),
            None => Err(error_helpers::invalid_configuration(
                "create_reader",
                "format",
                format!("Unknown format: {}", format),
            )),
        }
    }

    /// Get supported formats
    pub fn supported_formats(&self) -> Vec<String> {
        self.factories.keys().cloned().collect()
    }

    /// Get factory for format
    pub fn get_factory(&self, format: &str) -> Option<&Box<dyn FormatFactory>> {
        self.factories.get(format)
    }
}

impl Default for FormatRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating format readers with options
pub struct FormatReaderBuilder {
    /// Path to data file
    path: PathBuf,
    /// Explicit format (if known)
    format: Option<String>,
    /// Expected schema (for validation)
    expected_schema: Option<Vec<FieldInfo>>,
    /// Additional options
    options: HashMap<String, String>,
}

impl FormatReaderBuilder {
    /// Create a new builder for the given path
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            format: None,
            expected_schema: None,
            options: HashMap::new(),
        }
    }

    /// Specify the format explicitly
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Set expected schema for validation
    pub fn with_schema(mut self, schema: Vec<FieldInfo>) -> Self {
        self.expected_schema = Some(schema);
        self
    }

    /// Add an option
    pub fn with_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.insert(key.into(), value.into());
        self
    }

    /// Build the reader using a registry
    pub fn build(self, registry: &FormatRegistry) -> Result<Box<dyn FormatReader>> {
        let reader = if let Some(format) = &self.format {
            registry.create_reader(format, &self.path)?
        } else {
            registry.auto_detect(&self.path)?
        };

        // Validate schema if provided
        if let Some(schema) = self.expected_schema {
            reader.validate_schema(&schema)?;
        }

        Ok(reader)
    }
}

/// Helper function to detect format from file extension
pub fn detect_format_from_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
}

/// Helper function to read magic bytes for format detection
pub fn read_magic_bytes(path: &Path, num_bytes: usize) -> Result<Vec<u8>> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path)
        .map_err(|e| error_helpers::file_not_found("read_magic_bytes", path.to_path_buf()))?;

    let mut buffer = vec![0u8; num_bytes];
    file.read_exact(&mut buffer).map_err(|e| {
        error_helpers::data_corruption(
            "read_magic_bytes",
            format!("Failed to read magic bytes: {}", e),
            Some(path.to_path_buf()),
        )
    })?;

    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_conversion() {
        assert_eq!(DataType::Float32.to_tensor_dtype(), Some(DType::Float32));
        assert_eq!(DataType::Float64.to_tensor_dtype(), Some(DType::Float64));
        assert_eq!(DataType::Int32.to_tensor_dtype(), Some(DType::Int32));
        assert_eq!(DataType::String.to_tensor_dtype(), None);
    }

    #[test]
    fn test_format_metadata_creation() {
        let metadata = FormatMetadata {
            format_name: "test_format".to_string(),
            version: Some("1.0".to_string()),
            num_samples: 100,
            fields: vec![FieldInfo {
                name: "feature".to_string(),
                dtype: DataType::Float32,
                shape: Some(vec![10]),
                nullable: false,
                description: None,
            }],
            metadata: HashMap::new(),
            supports_random_access: true,
            supports_streaming: true,
        };

        assert_eq!(metadata.format_name, "test_format");
        assert_eq!(metadata.num_samples, 100);
        assert_eq!(metadata.fields.len(), 1);
    }

    #[test]
    fn test_format_registry() {
        let registry = FormatRegistry::new();
        assert!(registry.supported_formats().is_empty());
    }

    #[test]
    fn test_format_detection_from_extension() {
        assert_eq!(
            detect_format_from_extension(Path::new("data.json")),
            Some("json".to_string())
        );
        assert_eq!(
            detect_format_from_extension(Path::new("data.csv")),
            Some("csv".to_string())
        );
        assert_eq!(
            detect_format_from_extension(Path::new("data.CSV")),
            Some("csv".to_string())
        );
        assert_eq!(detect_format_from_extension(Path::new("data")), None);
    }

    #[test]
    fn test_reader_builder() {
        let builder = FormatReaderBuilder::new("test.json")
            .with_format("json")
            .with_option("encoding", "utf-8");

        assert_eq!(builder.format, Some("json".to_string()));
        assert_eq!(builder.options.get("encoding"), Some(&"utf-8".to_string()));
    }

    #[test]
    fn test_field_info_creation() {
        let field = FieldInfo {
            name: "test_field".to_string(),
            dtype: DataType::Float32,
            shape: Some(vec![3, 224, 224]),
            nullable: false,
            description: Some("Test field".to_string()),
        };

        assert_eq!(field.name, "test_field");
        assert_eq!(field.dtype, DataType::Float32);
        assert_eq!(field.shape, Some(vec![3, 224, 224]));
        assert!(!field.nullable);
    }

    #[test]
    fn test_data_type_equality() {
        assert_eq!(DataType::Float32, DataType::Float32);
        assert_ne!(DataType::Float32, DataType::Float64);
        assert_eq!(
            DataType::List(Box::new(DataType::Int32)),
            DataType::List(Box::new(DataType::Int32))
        );
    }

    #[test]
    fn test_detection_method() {
        let detection = FormatDetection {
            format_name: "json".to_string(),
            confidence: 0.95,
            method: DetectionMethod::Extension,
        };

        assert_eq!(detection.format_name, "json");
        assert_eq!(detection.confidence, 0.95);
        assert_eq!(detection.method, DetectionMethod::Extension);
    }

    #[test]
    fn test_format_sample_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "test".to_string());

        let sample = FormatSample {
            features: Tensor::<f32>::zeros(&[10]),
            labels: Tensor::<f32>::zeros(&[1]),
            source_index: 42,
            metadata,
        };

        assert_eq!(sample.source_index, 42);
        assert_eq!(sample.metadata.get("source"), Some(&"test".to_string()));
    }
}
