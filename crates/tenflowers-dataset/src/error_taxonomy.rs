//! Error taxonomy and standardized error handling for dataset operations
//!
//! This module provides a comprehensive error handling system that maps dataset-specific
//! errors to the TenfloweRS core error taxonomy. It ensures consistent error messages,
//! provides helpful context, and suggests recovery strategies where appropriate.

use std::path::PathBuf;
use tenflowers_core::{Result, TensorError};

/// Dataset-specific error categories aligned with core error taxonomy
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetErrorCategory {
    /// File or data source not found
    NotFound,
    /// Permission denied or access error
    PermissionDenied,
    /// Invalid data format or corrupted data
    DataCorruption,
    /// Schema or structure mismatch
    SchemaMismatch,
    /// Index out of bounds
    IndexOutOfBounds,
    /// Insufficient memory or resources
    ResourceExhaustion,
    /// Network or remote access error
    NetworkError,
    /// Cache operation failure
    CacheFailure,
    /// Transform or augmentation error
    TransformError,
    /// Serialization or deserialization error
    SerializationError,
    /// Configuration error
    ConfigurationError,
    /// Timeout during operation
    Timeout,
    /// Other unclassified error
    Other,
}

/// Extended error context for dataset operations
#[derive(Debug, Clone)]
pub struct DatasetErrorContext {
    /// Dataset name or identifier
    pub dataset_name: Option<String>,
    /// File path involved in the error
    pub file_path: Option<PathBuf>,
    /// Index or indices involved
    pub indices: Option<Vec<usize>>,
    /// Data format (JSON, CSV, Parquet, etc.)
    pub format: Option<String>,
    /// Expected schema or structure
    pub expected_schema: Option<String>,
    /// Actual schema or structure found
    pub actual_schema: Option<String>,
    /// Number of samples processed before error
    pub samples_processed: Option<usize>,
    /// Total samples expected
    pub total_samples: Option<usize>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl DatasetErrorContext {
    /// Create a new dataset error context
    pub fn new() -> Self {
        Self {
            dataset_name: None,
            file_path: None,
            indices: None,
            format: None,
            expected_schema: None,
            actual_schema: None,
            samples_processed: None,
            total_samples: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set dataset name
    pub fn with_dataset_name(mut self, name: impl Into<String>) -> Self {
        self.dataset_name = Some(name.into());
        self
    }

    /// Set file path
    pub fn with_file_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.file_path = Some(path.into());
        self
    }

    /// Set indices involved in error
    pub fn with_indices(mut self, indices: Vec<usize>) -> Self {
        self.indices = Some(indices);
        self
    }

    /// Set data format
    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Set expected and actual schema
    pub fn with_schema_mismatch(
        mut self,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        self.expected_schema = Some(expected.into());
        self.actual_schema = Some(actual.into());
        self
    }

    /// Set sample processing progress
    pub fn with_progress(mut self, processed: usize, total: usize) -> Self {
        self.samples_processed = Some(processed);
        self.total_samples = Some(total);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Convert to core ErrorContext format (best effort)
    pub fn to_core_context(&self) -> tenflowers_core::error::ErrorContext {
        let mut ctx = tenflowers_core::error::ErrorContext::new();

        if let Some(ref name) = self.dataset_name {
            ctx = ctx.with_metadata("dataset_name".to_string(), name.clone());
        }

        if let Some(ref path) = self.file_path {
            ctx = ctx.with_metadata("file_path".to_string(), path.display().to_string());
        }

        if let Some(ref format) = self.format {
            ctx = ctx.with_metadata("format".to_string(), format.clone());
        }

        for (k, v) in &self.metadata {
            ctx = ctx.with_metadata(k.clone(), v.clone());
        }

        ctx
    }
}

impl Default for DatasetErrorContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Dataset error builder for creating well-structured errors
pub struct DatasetErrorBuilder {
    category: DatasetErrorCategory,
    operation: String,
    message: String,
    context: DatasetErrorContext,
    suggestions: Vec<String>,
}

impl DatasetErrorBuilder {
    /// Create a new error builder
    pub fn new(category: DatasetErrorCategory, operation: impl Into<String>) -> Self {
        Self {
            category,
            operation: operation.into(),
            message: String::new(),
            context: DatasetErrorContext::new(),
            suggestions: Vec::new(),
        }
    }

    /// Set error message
    pub fn message(mut self, msg: impl Into<String>) -> Self {
        self.message = msg.into();
        self
    }

    /// Set context
    pub fn context(mut self, context: DatasetErrorContext) -> Self {
        self.context = context;
        self
    }

    /// Add a suggestion for recovery
    pub fn suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    /// Build the TensorError
    pub fn build(self) -> TensorError {
        match self.category {
            DatasetErrorCategory::NotFound => {
                let details = if let Some(path) = &self.context.file_path {
                    format!("{}: {}", self.message, path.display())
                } else {
                    self.message.clone()
                };

                let path_str = self
                    .context
                    .file_path
                    .as_ref()
                    .map(|p| p.display().to_string());

                TensorError::IoError {
                    operation: self.operation,
                    details,
                    path: path_str,
                    context: Some(self.context.to_core_context()),
                }
            }
            DatasetErrorCategory::PermissionDenied => {
                let path_str = self
                    .context
                    .file_path
                    .as_ref()
                    .map(|p| p.display().to_string());

                TensorError::IoError {
                    operation: self.operation,
                    details: format!("Permission denied: {}", self.message),
                    path: path_str,
                    context: Some(self.context.to_core_context()),
                }
            }
            DatasetErrorCategory::DataCorruption => TensorError::InvalidArgument {
                operation: self.operation,
                reason: format!("Data corruption: {}", self.message),
                context: Some(self.context.to_core_context()),
            },
            DatasetErrorCategory::SchemaMismatch => {
                let reason = if let (Some(expected), Some(actual)) =
                    (&self.context.expected_schema, &self.context.actual_schema)
                {
                    format!(
                        "Schema mismatch: {}. Expected: {}, got: {}",
                        self.message, expected, actual
                    )
                } else {
                    format!("Schema mismatch: {}", self.message)
                };

                TensorError::InvalidShape {
                    operation: self.operation,
                    reason,
                    shape: None,
                    context: Some(self.context.to_core_context()),
                }
            }
            DatasetErrorCategory::IndexOutOfBounds => {
                let reason = if let Some(indices) = &self.context.indices {
                    format!(
                        "Index out of bounds: {} (indices: {:?})",
                        self.message, indices
                    )
                } else {
                    format!("Index out of bounds: {}", self.message)
                };

                TensorError::InvalidArgument {
                    operation: self.operation,
                    reason,
                    context: Some(self.context.to_core_context()),
                }
            }
            DatasetErrorCategory::ResourceExhaustion => TensorError::ResourceExhausted {
                operation: self.operation,
                resource: self.message.clone(),
                current_usage: None,
                limit: None,
                context: Some(self.context.to_core_context()),
            },
            DatasetErrorCategory::NetworkError => {
                let path_str = self
                    .context
                    .file_path
                    .as_ref()
                    .map(|p| p.display().to_string());

                TensorError::IoError {
                    operation: self.operation,
                    details: format!("Network error: {}", self.message),
                    path: path_str,
                    context: Some(self.context.to_core_context()),
                }
            }
            DatasetErrorCategory::CacheFailure => TensorError::CacheError {
                operation: self.operation,
                details: self.message.clone(),
                recoverable: true,
                context: Some(self.context.to_core_context()),
            },
            DatasetErrorCategory::TransformError => TensorError::InvalidOperation {
                operation: self.operation,
                reason: format!("Transform error: {}", self.message),
                context: Some(self.context.to_core_context()),
            },
            DatasetErrorCategory::SerializationError => TensorError::SerializationError {
                operation: self.operation,
                details: self.message.clone(),
                context: Some(self.context.to_core_context()),
            },
            DatasetErrorCategory::ConfigurationError => TensorError::InvalidArgument {
                operation: self.operation,
                reason: format!("Configuration error: {}", self.message),
                context: Some(self.context.to_core_context()),
            },
            DatasetErrorCategory::Timeout => {
                TensorError::Timeout {
                    operation: self.operation,
                    duration_ms: 0, // Would need to be passed in context
                    context: Some(self.context.to_core_context()),
                }
            }
            DatasetErrorCategory::Other => TensorError::Other {
                operation: self.operation,
                details: self.message.clone(),
                context: Some(self.context.to_core_context()),
            },
        }
    }
}

/// Helper functions for creating common dataset errors
pub mod helpers {
    use super::*;

    /// Create a file not found error
    pub fn file_not_found(operation: impl Into<String>, path: impl Into<PathBuf>) -> TensorError {
        DatasetErrorBuilder::new(DatasetErrorCategory::NotFound, operation)
            .message("File not found")
            .context(DatasetErrorContext::new().with_file_path(path))
            .suggestion("Check that the file path is correct and the file exists")
            .build()
    }

    /// Create a data corruption error
    pub fn data_corruption(
        operation: impl Into<String>,
        details: impl Into<String>,
        path: Option<PathBuf>,
    ) -> TensorError {
        let mut builder = DatasetErrorBuilder::new(DatasetErrorCategory::DataCorruption, operation)
            .message(details);

        if let Some(p) = path {
            builder = builder.context(DatasetErrorContext::new().with_file_path(p));
        }

        builder
            .suggestion("Verify data integrity and try redownloading if from remote source")
            .build()
    }

    /// Create a schema mismatch error
    pub fn schema_mismatch(
        operation: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> TensorError {
        DatasetErrorBuilder::new(DatasetErrorCategory::SchemaMismatch, operation)
            .message("Schema does not match expected format")
            .context(DatasetErrorContext::new().with_schema_mismatch(expected, actual))
            .suggestion("Check data format specification")
            .build()
    }

    /// Create an index out of bounds error
    pub fn index_out_of_bounds(
        operation: impl Into<String>,
        index: usize,
        dataset_len: usize,
    ) -> TensorError {
        DatasetErrorBuilder::new(DatasetErrorCategory::IndexOutOfBounds, operation)
            .message(format!(
                "Index {} out of bounds for dataset of length {}",
                index, dataset_len
            ))
            .context(
                DatasetErrorContext::new()
                    .with_indices(vec![index])
                    .with_progress(0, dataset_len),
            )
            .build()
    }

    /// Create a memory exhaustion error
    pub fn memory_exhausted(operation: impl Into<String>, requested: usize) -> TensorError {
        DatasetErrorBuilder::new(DatasetErrorCategory::ResourceExhaustion, operation)
            .message(format!(
                "Insufficient memory: requested {} bytes",
                requested
            ))
            .suggestion("Try reducing batch size or enabling memory-mapped loading")
            .build()
    }

    /// Create a cache error
    pub fn cache_error(operation: impl Into<String>, details: impl Into<String>) -> TensorError {
        DatasetErrorBuilder::new(DatasetErrorCategory::CacheFailure, operation)
            .message(details)
            .suggestion("Cache operation failed but dataset can continue without cache")
            .build()
    }

    /// Create a transform error
    pub fn transform_error(
        operation: impl Into<String>,
        transform_name: impl Into<String>,
        details: impl Into<String>,
    ) -> TensorError {
        let transform_name = transform_name.into();
        DatasetErrorBuilder::new(DatasetErrorCategory::TransformError, operation)
            .message(format!(
                "Transform '{}' failed: {}",
                transform_name,
                details.into()
            ))
            .context(DatasetErrorContext::new().with_metadata("transform", transform_name))
            .build()
    }

    /// Create a network error
    pub fn network_error(
        operation: impl Into<String>,
        url: impl Into<String>,
        details: impl Into<String>,
    ) -> TensorError {
        let url = url.into();
        DatasetErrorBuilder::new(DatasetErrorCategory::NetworkError, operation)
            .message(format!(
                "Network error accessing {}: {}",
                url,
                details.into()
            ))
            .context(DatasetErrorContext::new().with_metadata("url", url))
            .suggestion("Check network connectivity and URL accessibility")
            .build()
    }

    /// Create an empty dataset error
    pub fn empty_dataset(operation: impl Into<String>) -> TensorError {
        DatasetErrorBuilder::new(DatasetErrorCategory::ConfigurationError, operation)
            .message("Dataset is empty")
            .suggestion("Ensure dataset contains at least one sample")
            .build()
    }

    /// Create a configuration error
    pub fn invalid_configuration(
        operation: impl Into<String>,
        parameter: impl Into<String>,
        reason: impl Into<String>,
    ) -> TensorError {
        let param = parameter.into();
        DatasetErrorBuilder::new(DatasetErrorCategory::ConfigurationError, operation)
            .message(format!(
                "Invalid configuration parameter '{}': {}",
                param,
                reason.into()
            ))
            .context(DatasetErrorContext::new().with_metadata("parameter", param))
            .build()
    }
}

/// Error classification helpers
pub mod classification {
    use super::*;

    /// Check if an error is recoverable
    pub fn is_recoverable(error: &TensorError) -> bool {
        matches!(
            error,
            TensorError::CacheError {
                recoverable: true,
                ..
            } | TensorError::ComputeError {
                retry_possible: true,
                ..
            } | TensorError::AllocationError { .. }
                | TensorError::Timeout { .. }
        )
    }

    /// Check if an error is transient (might succeed on retry)
    pub fn is_transient(error: &TensorError) -> bool {
        matches!(
            error,
            TensorError::Timeout { .. }
                | TensorError::ResourceExhausted { .. }
                | TensorError::IoError { .. }
        )
    }

    /// Check if an error is a data quality issue
    pub fn is_data_quality_error(error: &TensorError) -> bool {
        matches!(
            error,
            TensorError::InvalidArgument { .. }
                | TensorError::InvalidShape { .. }
                | TensorError::NumericalError { .. }
        )
    }

    /// Get user-friendly error message with suggestions
    pub fn user_friendly_message(error: &TensorError) -> String {
        let base_msg = format!("{}", error);

        let suggestions = match error {
            TensorError::IoError { .. } => {
                vec![
                    "Check file path and permissions",
                    "Verify the file exists and is accessible",
                ]
            }
            TensorError::InvalidShape { .. } => {
                vec![
                    "Verify data dimensions match expected format",
                    "Check preprocessing and transformation pipeline",
                ]
            }
            TensorError::ResourceExhausted { .. } => {
                vec![
                    "Reduce batch size",
                    "Enable memory-mapped loading",
                    "Close unused datasets and free memory",
                ]
            }
            TensorError::CacheError { .. } => {
                vec![
                    "Cache operation failed - continuing without cache",
                    "Check available disk space",
                ]
            }
            _ => vec![],
        };

        if suggestions.is_empty() {
            base_msg
        } else {
            format!(
                "{}\n\nSuggestions:\n{}",
                base_msg,
                suggestions
                    .iter()
                    .map(|s| format!("  - {}", s))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_not_found_error() {
        let error = helpers::file_not_found("load_dataset", "/path/to/data.csv");
        assert!(matches!(error, TensorError::IoError { .. }));
        assert_eq!(error.operation(), "load_dataset");
    }

    #[test]
    fn test_schema_mismatch_error() {
        let error = helpers::schema_mismatch("parse_json", "Array[Object]", "Object");
        assert!(matches!(error, TensorError::InvalidShape { .. }));
        assert_eq!(error.operation(), "parse_json");
    }

    #[test]
    fn test_index_out_of_bounds_error() {
        let error = helpers::index_out_of_bounds("get_sample", 100, 50);
        assert!(matches!(error, TensorError::InvalidArgument { .. }));
        let msg = format!("{}", error);
        assert!(msg.contains("100"));
        assert!(msg.contains("50"));
    }

    #[test]
    fn test_cache_error() {
        let error = helpers::cache_error("cache_put", "Disk full");
        assert!(matches!(error, TensorError::CacheError { .. }));
        assert!(classification::is_recoverable(&error));
    }

    #[test]
    fn test_transform_error() {
        let error = helpers::transform_error("apply_transform", "Normalize", "Invalid mean value");
        assert!(matches!(error, TensorError::InvalidOperation { .. }));
    }

    #[test]
    fn test_error_context_builder() {
        let context = DatasetErrorContext::new()
            .with_dataset_name("mnist")
            .with_file_path("/data/mnist.npz")
            .with_format("npz")
            .with_progress(100, 1000);

        assert_eq!(context.dataset_name, Some("mnist".to_string()));
        assert_eq!(context.format, Some("npz".to_string()));
        assert_eq!(context.samples_processed, Some(100));
        assert_eq!(context.total_samples, Some(1000));
    }

    #[test]
    fn test_error_classification() {
        let cache_err = helpers::cache_error("test", "failed");
        assert!(classification::is_recoverable(&cache_err));

        let timeout_err = TensorError::Timeout {
            operation: "test".to_string(),
            duration_ms: 5000,
            context: None,
        };
        assert!(classification::is_transient(&timeout_err));
    }

    #[test]
    fn test_user_friendly_message() {
        let error = helpers::memory_exhausted("allocate_buffer", 1024 * 1024 * 1024);
        let msg = classification::user_friendly_message(&error);
        assert!(msg.contains("Suggestions"));
        assert!(msg.contains("Reduce batch size"));
    }

    #[test]
    fn test_empty_dataset_error() {
        let error = helpers::empty_dataset("iterate");
        assert!(matches!(error, TensorError::InvalidArgument { .. }));
        let msg = format!("{}", error);
        assert!(msg.contains("empty"));
    }

    #[test]
    fn test_invalid_configuration_error() {
        let error =
            helpers::invalid_configuration("create_dataloader", "batch_size", "must be positive");
        assert!(matches!(error, TensorError::InvalidArgument { .. }));
        let msg = format!("{}", error);
        assert!(msg.contains("batch_size"));
    }
}
