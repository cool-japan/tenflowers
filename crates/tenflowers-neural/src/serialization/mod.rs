//! Advanced model serialization with version compatibility and migration support
//!
//! This module provides comprehensive serialization capabilities for TenfloweRS models,
//! including version compatibility, schema validation, and data migration.

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tenflowers_core::{Result, Tensor, TensorError};

pub mod checkpoint;
pub mod compression;
pub mod migration;
pub mod onnx;
pub mod schema;
pub mod versioned;
pub mod weight_loader;

pub use checkpoint::*;
pub use compression::*;
pub use migration::*;
pub use onnx::*;
pub use schema::*;
pub use versioned::*;
pub use weight_loader::*;

/// Comprehensive model metadata
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model type identifier
    pub model_type: String,
    /// Serialization format version
    pub version: SemanticVersion,
    /// Framework version that created this model
    pub framework_version: String,
    /// Creation timestamp
    pub created_at: String,
    /// Model architecture hash for validation
    pub architecture_hash: String,
    /// Number of parameters
    pub parameter_count: usize,
    /// Model size in bytes
    pub model_size: usize,
    /// Training information
    pub training_info: TrainingInfo,
    /// Hardware requirements
    pub hardware_requirements: HardwareRequirements,
    /// Custom metadata
    pub custom: HashMap<String, String>,
}

/// Training-related metadata
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct TrainingInfo {
    /// Number of epochs trained
    pub epochs: Option<u32>,
    /// Training loss
    pub final_loss: Option<f32>,
    /// Validation accuracy
    pub validation_accuracy: Option<f32>,
    /// Optimizer used
    pub optimizer: Option<String>,
    /// Learning rate
    pub learning_rate: Option<f32>,
    /// Training dataset information
    pub dataset_info: Option<String>,
}

/// Hardware requirements and optimizations
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct HardwareRequirements {
    /// Minimum memory required (in bytes)
    pub min_memory: u64,
    /// Recommended memory (in bytes)
    pub recommended_memory: u64,
    /// GPU requirements
    pub gpu_required: bool,
    /// CPU architecture optimizations
    pub cpu_features: Vec<String>,
    /// Target device type
    pub target_device: String,
}

/// Parameter information for validation
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    /// Parameter name/identifier
    pub name: String,
    /// Parameter shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: String,
    /// Device placement
    pub device: String,
    /// Whether parameter requires gradients
    pub requires_grad: bool,
    /// Parameter checksum for validation
    pub checksum: u64,
}

/// Complete model state with advanced metadata
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct AdvancedModelState {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Parameter information
    pub parameters_info: Vec<ParameterInfo>,
    /// Serialized parameter data
    pub parameters_data: Vec<u8>,
    /// Compression information
    pub compression_info: Option<CompressionInfo>,
    /// Schema validation hash
    pub schema_hash: String,
}

/// Serialization configuration
#[derive(Debug, Clone)]
pub struct SerializationConfig {
    /// Enable compression
    pub compression: bool,
    /// Compression algorithm
    pub compression_algorithm: CompressionAlgorithm,
    /// Compression level (0-9)
    pub compression_level: u8,
    /// Include training metadata
    pub include_training_info: bool,
    /// Validate checksums
    pub validate_checksums: bool,
    /// Custom metadata to include
    pub custom_metadata: HashMap<String, String>,
}

impl Default for SerializationConfig {
    fn default() -> Self {
        Self {
            compression: true,
            compression_algorithm: CompressionAlgorithm::Zstd,
            compression_level: 3,
            include_training_info: true,
            validate_checksums: true,
            custom_metadata: HashMap::new(),
        }
    }
}

/// Advanced serialization trait
pub trait AdvancedSerialization<T> {
    /// Save model with advanced serialization
    fn save_advanced<P: AsRef<Path>>(&self, path: P, config: &SerializationConfig) -> Result<()>;

    /// Load model with version compatibility checking
    fn load_advanced<P: AsRef<Path>>(&mut self, path: P) -> Result<LoadResult>;

    /// Get model metadata without loading parameters
    fn get_metadata<P: AsRef<Path>>(path: P) -> Result<ModelMetadata>;

    /// Validate model file integrity
    fn validate_model_file<P: AsRef<Path>>(path: P) -> Result<ValidationResult>;

    /// Create a model checkpoint
    fn create_checkpoint<P: AsRef<Path>>(
        &self,
        path: P,
        checkpoint_info: CheckpointInfo,
    ) -> Result<()>;

    /// Load from checkpoint
    fn load_checkpoint<P: AsRef<Path>>(&mut self, path: P) -> Result<CheckpointLoadResult>;
}

/// Result of loading a model
#[derive(Debug)]
pub struct LoadResult {
    /// Whether migration was performed
    pub migration_performed: bool,
    /// Original version if migration occurred
    pub original_version: Option<SemanticVersion>,
    /// Warnings encountered during loading
    pub warnings: Vec<String>,
    /// Loaded metadata
    pub metadata: ModelMetadata,
}

/// Result of model validation
#[derive(Debug)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Model metadata
    pub metadata: ModelMetadata,
}

/// Model file format utilities
pub struct ModelFileFormat;

impl ModelFileFormat {
    /// Detect model file format version
    pub fn detect_version<P: AsRef<Path>>(path: P) -> Result<SemanticVersion> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            TensorError::serialization_error_simple(format!("Failed to read file: {}", e))
        })?;

        // Try to parse as advanced format first
        if let Ok(state) = serde_json::from_str::<AdvancedModelState>(&content) {
            return Ok(state.metadata.version);
        }

        // Try legacy format
        if let Ok(legacy_state) = serde_json::from_str::<crate::model::ModelState>(&content) {
            // Legacy format is version 0.0.1
            return Ok(SemanticVersion::new(0, 0, 1));
        }

        Err(TensorError::serialization_error_simple(
            "Unknown model format".to_string(),
        ))
    }

    /// Check if file is a valid model file
    pub fn is_valid_model_file<P: AsRef<Path>>(path: P) -> bool {
        Self::detect_version(path).is_ok()
    }

    /// Get file format information
    pub fn get_format_info<P: AsRef<Path>>(path: P) -> Result<FormatInfo> {
        let version = Self::detect_version(&path)?;
        let file_size = std::fs::metadata(&path)
            .map_err(|e| {
                TensorError::serialization_error_simple(format!("Failed to get file size: {}", e))
            })?
            .len();

        Ok(FormatInfo {
            version: version.clone(),
            file_size,
            format_type: if version.major == 0 && version.minor == 0 {
                FormatType::Legacy
            } else {
                FormatType::Advanced
            },
        })
    }
}

/// File format information
#[derive(Debug)]
pub struct FormatInfo {
    pub version: SemanticVersion,
    pub file_size: u64,
    pub format_type: FormatType,
}

/// Model file format type
#[derive(Debug, Clone, PartialEq)]
pub enum FormatType {
    Legacy,
    Advanced,
}

/// Utility functions for model serialization
pub mod utils {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    /// Calculate hash for architecture validation
    pub fn calculate_architecture_hash_f32(model: &dyn crate::model::Model<f32>) -> String {
        let mut hasher = DefaultHasher::new();

        // Hash parameter shapes and types
        for param in model.parameters() {
            param.shape().dims().hash(&mut hasher);
            // Hash device type
            format!("{:?}", param.device()).hash(&mut hasher);
        }

        format!("{:x}", hasher.finish())
    }

    /// Calculate hash for architecture validation (f64 version)
    pub fn calculate_architecture_hash_f64(model: &dyn crate::model::Model<f64>) -> String {
        let mut hasher = DefaultHasher::new();

        // Hash parameter shapes and types
        for param in model.parameters() {
            param.shape().dims().hash(&mut hasher);
            // Hash device type
            format!("{:?}", param.device()).hash(&mut hasher);
        }

        format!("{:x}", hasher.finish())
    }

    /// Calculate parameter checksum
    pub fn calculate_parameter_checksum<T>(param: &Tensor<T>) -> u64
    where
        T: Clone + std::hash::Hash,
    {
        let mut hasher = DefaultHasher::new();

        // Hash shape
        param.shape().dims().hash(&mut hasher);

        // Hash data if available
        if let Some(data) = param.as_slice() {
            // For efficiency, hash a sample of the data
            let sample_size = std::cmp::min(data.len(), 1000);
            for i in 0..sample_size {
                if let Some(val) = data.get(i) {
                    // Hash the value representation
                    val.hash(&mut hasher);
                }
            }
        }

        hasher.finish()
    }

    /// Generate framework version string
    pub fn get_framework_version() -> String {
        format!("TenfloweRS-{}", env!("CARGO_PKG_VERSION"))
    }

    /// Get current timestamp
    #[cfg(feature = "serialize")]
    pub fn get_timestamp() -> String {
        chrono::Utc::now().to_rfc3339()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_compatibility() {
        let v1 = SemanticVersion::new(1, 0, 0);
        let v2 = SemanticVersion::new(1, 1, 0);
        let v3 = SemanticVersion::new(2, 0, 0);

        assert!(v1.is_compatible_with(&v2));
        assert!(v2.is_compatible_with(&v1));
        assert!(!v1.is_compatible_with(&v3));
        assert!(!v3.is_compatible_with(&v1));

        assert!(v2.needs_migration_from(&v1));
        assert!(!v1.needs_migration_from(&v2));
    }

    #[test]
    fn test_version_display() {
        let version = SemanticVersion::new(1, 2, 3);
        assert_eq!(format!("{}", version), "1.2.3");
    }

    #[test]
    fn test_serialization_config_default() {
        let config = SerializationConfig::default();
        assert!(config.compression);
        assert!(config.include_training_info);
        assert!(config.validate_checksums);
        assert_eq!(config.compression_level, 3);
    }
}
