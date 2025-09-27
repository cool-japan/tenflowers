//! Configuration validation utilities
//!
//! This module provides comprehensive validation for configuration values
//! with descriptive error messages and suggestions for fixing issues.

use super::{
    AsyncIoConfig, AudioConfig, CacheConfig, DataLoaderConfig, DatasetConfig, FormatConfig,
    GlobalConfig, GpuConfig, Hdf5FormatConfig, ImageFormatConfig, LoggingConfig, MonitoringConfig,
    ParquetFormatConfig, PerformanceConfig, TextFormatConfig, TransformConfig,
};
use crate::{Result, TensorError};
use std::collections::HashMap;

/// Result type for validation operations
pub type ValidationResult<T = ()> = std::result::Result<T, ValidationError>;

/// Validation error with detailed information
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Field path that failed validation
    pub field: String,
    /// Error message
    pub message: String,
    /// Current value that failed validation
    pub current_value: Option<String>,
    /// Suggested valid values or ranges
    pub suggestions: Vec<String>,
}

impl ValidationError {
    /// Create a new validation error
    pub fn new(field: &str, message: &str) -> Self {
        Self {
            field: field.to_string(),
            message: message.to_string(),
            current_value: None,
            suggestions: Vec::new(),
        }
    }

    /// Set the current value that failed validation
    pub fn with_current_value(mut self, value: &str) -> Self {
        self.current_value = Some(value.to_string());
        self
    }

    /// Add suggestions for valid values
    pub fn with_suggestions(mut self, suggestions: Vec<&str>) -> Self {
        self.suggestions = suggestions.into_iter().map(|s| s.to_string()).collect();
        self
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Validation error in '{}': {}", self.field, self.message)?;

        if let Some(ref value) = self.current_value {
            write!(f, " (current value: {})", value)?;
        }

        if !self.suggestions.is_empty() {
            write!(f, " - Suggestions: {}", self.suggestions.join(", "))?;
        }

        Ok(())
    }
}

impl std::error::Error for ValidationError {}

impl From<ValidationError> for TensorError {
    fn from(err: ValidationError) -> Self {
        TensorError::invalid_argument(err.to_string())
    }
}

/// Configuration validation trait
pub trait ConfigValidation {
    /// Validate the configuration
    fn validate(&self) -> Result<()>;

    /// Get validation warnings (non-fatal issues)
    fn get_warnings(&self) -> Vec<String> {
        Vec::new()
    }
}

impl ConfigValidation for GlobalConfig {
    fn validate(&self) -> Result<()> {
        let mut errors = Vec::new();

        // Validate dataset configuration
        if let Err(e) = self.dataset.validate() {
            errors.push(e);
        }

        // Validate dataloader configuration
        if let Err(e) = self.dataloader.validate() {
            errors.push(e);
        }

        // Validate transforms configuration
        if let Err(e) = self.transforms.validate() {
            errors.push(e);
        }

        // Validate performance configuration
        if let Err(e) = self.performance.validate() {
            errors.push(e);
        }

        // Validate cache configuration
        if let Err(e) = self.cache.validate() {
            errors.push(e);
        }

        // Validate GPU configuration
        if let Err(e) = self.gpu.validate() {
            errors.push(e);
        }

        // Validate audio configuration
        if let Err(e) = self.audio.validate() {
            errors.push(e);
        }

        // Validate format configurations
        if let Err(e) = self.formats.validate() {
            errors.push(e);
        }

        // Validate logging configuration
        if let Err(e) = self.logging.validate() {
            errors.push(e);
        }

        // Cross-configuration validation
        self.validate_cross_config_constraints(&mut errors);

        if !errors.is_empty() {
            let error_messages: Vec<String> = errors.into_iter().map(|e| e.to_string()).collect();
            return Err(TensorError::invalid_argument(format!(
                "Configuration validation failed:\n{}",
                error_messages.join("\n")
            )));
        }

        Ok(())
    }

    fn get_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        // Performance warnings
        if self.dataloader.num_workers > num_cpus::get() * 2 {
            warnings.push(format!(
                "dataloader.num_workers ({}) is much higher than CPU count ({}). This may cause performance degradation.",
                self.dataloader.num_workers,
                num_cpus::get()
            ));
        }

        if self.performance.memory_pool_size > 8192 {
            warnings.push("performance.memory_pool_size is very large (>8GB). Make sure you have sufficient RAM.".to_string());
        }

        // GPU warnings
        if self.gpu.enabled && self.gpu.memory_pool_mb > 4096 {
            warnings.push(
                "gpu.memory_pool_mb is very large (>4GB). Make sure your GPU has sufficient VRAM."
                    .to_string(),
            );
        }

        // Cache warnings
        if self.cache.enabled && self.cache.size_mb > self.performance.memory_pool_size {
            warnings.push("cache.size_mb is larger than performance.memory_pool_size. This may cause memory pressure.".to_string());
        }

        warnings
    }
}

impl GlobalConfig {
    fn validate_cross_config_constraints(&self, errors: &mut Vec<TensorError>) {
        // Validate that cache size doesn't exceed memory pool
        if self.cache.enabled && self.cache.size_mb > self.performance.memory_pool_size {
            errors.push(
                ValidationError::new(
                    "cache.size_mb",
                    "Cache size cannot be larger than memory pool size",
                )
                .with_current_value(&self.cache.size_mb.to_string())
                .with_suggestions(vec![
                    &format!("Set to {} or less", self.performance.memory_pool_size),
                    "Increase performance.memory_pool_size",
                ])
                .into(),
            );
        }

        // Validate GPU settings consistency
        if self.gpu.enabled && self.transforms.enable_gpu && self.gpu.device_id.is_none() {
            errors.push(
                ValidationError::new(
                    "gpu.device_id",
                    "GPU device ID should be specified when GPU acceleration is enabled",
                )
                .with_suggestions(vec!["Set gpu.device_id to a valid GPU index"])
                .into(),
            );
        }

        // Validate async I/O settings
        if self.performance.async_io.enabled && self.performance.async_io.io_threads == 0 {
            errors.push(
                ValidationError::new(
                    "performance.async_io.io_threads",
                    "Async I/O threads must be greater than 0 when async I/O is enabled",
                )
                .with_current_value("0")
                .with_suggestions(vec!["Set to 1 or more", "Disable async I/O"])
                .into(),
            );
        }
    }
}

impl ConfigValidation for DatasetConfig {
    fn validate(&self) -> Result<()> {
        let mut errors = Vec::new();

        if self.batch_size == 0 {
            errors.push(
                ValidationError::new("dataset.batch_size", "Batch size must be greater than 0")
                    .with_current_value("0")
                    .with_suggestions(vec!["Set to 1 or more"]),
            );
        }

        if self.batch_size > 10000 {
            errors.push(
                ValidationError::new(
                    "dataset.batch_size",
                    "Batch size is very large and may cause memory issues",
                )
                .with_current_value(&self.batch_size.to_string())
                .with_suggestions(vec!["Consider reducing to 1000 or less"]),
            );
        }

        if !errors.is_empty() {
            return Err(errors.into_iter().next().unwrap().into());
        }

        Ok(())
    }
}

impl ConfigValidation for DataLoaderConfig {
    fn validate(&self) -> Result<()> {
        let mut errors = Vec::new();

        if self.num_workers == 0 {
            errors.push(
                ValidationError::new(
                    "dataloader.num_workers",
                    "Number of workers must be greater than 0",
                )
                .with_current_value("0")
                .with_suggestions(vec!["Set to 1 or more"]),
            );
        }

        if self.prefetch_factor == 0 {
            errors.push(
                ValidationError::new(
                    "dataloader.prefetch_factor",
                    "Prefetch factor must be greater than 0",
                )
                .with_current_value("0")
                .with_suggestions(vec!["Set to 1 or more"]),
            );
        }

        if self.prefetch_factor > 100 {
            errors.push(
                ValidationError::new(
                    "dataloader.prefetch_factor",
                    "Prefetch factor is very large and may cause memory issues",
                )
                .with_current_value(&self.prefetch_factor.to_string())
                .with_suggestions(vec!["Consider reducing to 10 or less"]),
            );
        }

        if !errors.is_empty() {
            return Err(errors.into_iter().next().unwrap().into());
        }

        Ok(())
    }
}

impl ConfigValidation for TransformConfig {
    fn validate(&self) -> Result<()> {
        let valid_resize_strategies = ["nearest", "bilinear", "bicubic", "lanczos"];

        if !valid_resize_strategies.contains(&self.default_resize_strategy.as_str()) {
            return Err(ValidationError::new(
                "transforms.default_resize_strategy",
                "Invalid resize strategy",
            )
            .with_current_value(&self.default_resize_strategy)
            .with_suggestions(valid_resize_strategies.to_vec())
            .into());
        }

        if self.augmentation_probability < 0.0 || self.augmentation_probability > 1.0 {
            return Err(ValidationError::new(
                "transforms.augmentation_probability",
                "Augmentation probability must be between 0.0 and 1.0",
            )
            .with_current_value(&self.augmentation_probability.to_string())
            .with_suggestions(vec!["Set to a value between 0.0 and 1.0"])
            .into());
        }

        Ok(())
    }
}

impl ConfigValidation for PerformanceConfig {
    fn validate(&self) -> Result<()> {
        if self.num_threads == 0 {
            return Err(ValidationError::new(
                "performance.num_threads",
                "Number of threads must be greater than 0",
            )
            .with_current_value("0")
            .with_suggestions(vec!["Set to 1 or more"])
            .into());
        }

        if self.memory_pool_size == 0 {
            return Err(ValidationError::new(
                "performance.memory_pool_size",
                "Memory pool size must be greater than 0",
            )
            .with_current_value("0")
            .with_suggestions(vec!["Set to 64 MB or more"])
            .into());
        }

        self.async_io.validate()?;
        self.monitoring.validate()?;

        Ok(())
    }
}

impl ConfigValidation for AsyncIoConfig {
    fn validate(&self) -> Result<()> {
        if self.enabled && self.io_threads == 0 {
            return Err(ValidationError::new(
                "performance.async_io.io_threads",
                "I/O threads must be greater than 0 when async I/O is enabled",
            )
            .with_current_value("0")
            .with_suggestions(vec!["Set to 1 or more", "Disable async I/O"])
            .into());
        }

        if self.buffer_size == 0 {
            return Err(ValidationError::new(
                "performance.async_io.buffer_size",
                "Buffer size must be greater than 0",
            )
            .with_current_value("0")
            .with_suggestions(vec!["Set to 4096 or more"])
            .into());
        }

        if self.queue_depth == 0 {
            return Err(ValidationError::new(
                "performance.async_io.queue_depth",
                "Queue depth must be greater than 0",
            )
            .with_current_value("0")
            .with_suggestions(vec!["Set to 1 or more"])
            .into());
        }

        Ok(())
    }
}

impl ConfigValidation for MonitoringConfig {
    fn validate(&self) -> Result<()> {
        if self.enabled && self.interval == 0 {
            return Err(ValidationError::new(
                "performance.monitoring.interval",
                "Monitoring interval must be greater than 0 when monitoring is enabled",
            )
            .with_current_value("0")
            .with_suggestions(vec!["Set to 1 or more seconds", "Disable monitoring"])
            .into());
        }

        Ok(())
    }
}

impl ConfigValidation for CacheConfig {
    fn validate(&self) -> Result<()> {
        if self.enabled && self.size_mb == 0 {
            return Err(ValidationError::new(
                "cache.size_mb",
                "Cache size must be greater than 0 when caching is enabled",
            )
            .with_current_value("0")
            .with_suggestions(vec!["Set to 64 MB or more", "Disable caching"])
            .into());
        }

        let valid_policies = ["lru", "lfu", "fifo", "random"];
        if !valid_policies.contains(&self.eviction_policy.as_str()) {
            return Err(ValidationError::new(
                "cache.eviction_policy",
                "Invalid cache eviction policy",
            )
            .with_current_value(&self.eviction_policy)
            .with_suggestions(valid_policies.to_vec())
            .into());
        }

        Ok(())
    }
}

impl ConfigValidation for GpuConfig {
    fn validate(&self) -> Result<()> {
        if self.enabled && self.memory_pool_mb == 0 {
            return Err(ValidationError::new(
                "gpu.memory_pool_mb",
                "GPU memory pool size must be greater than 0 when GPU is enabled",
            )
            .with_current_value("0")
            .with_suggestions(vec!["Set to 256 MB or more", "Disable GPU"])
            .into());
        }

        let valid_precisions = ["fp16", "fp32", "fp64", "bf16"];
        if !valid_precisions.contains(&self.precision.as_str()) {
            return Err(
                ValidationError::new("gpu.precision", "Invalid GPU precision setting")
                    .with_current_value(&self.precision)
                    .with_suggestions(valid_precisions.to_vec())
                    .into(),
            );
        }

        Ok(())
    }
}

impl ConfigValidation for AudioConfig {
    fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 {
            return Err(ValidationError::new(
                "audio.sample_rate",
                "Sample rate must be greater than 0",
            )
            .with_current_value("0")
            .with_suggestions(vec!["Set to 44100, 48000, or other valid sample rate"])
            .into());
        }

        if self.channels == 0 {
            return Err(ValidationError::new(
                "audio.channels",
                "Number of channels must be greater than 0",
            )
            .with_current_value("0")
            .with_suggestions(vec!["Set to 1 (mono) or 2 (stereo)"])
            .into());
        }

        if self.buffer_size == 0 {
            return Err(ValidationError::new(
                "audio.buffer_size",
                "Buffer size must be greater than 0",
            )
            .with_current_value("0")
            .with_suggestions(vec!["Set to 1024 or other power of 2"])
            .into());
        }

        let valid_formats = ["wav", "mp3", "flac", "ogg", "aac"];
        if !valid_formats.contains(&self.preferred_format.as_str()) {
            return Err(
                ValidationError::new("audio.preferred_format", "Invalid audio format")
                    .with_current_value(&self.preferred_format)
                    .with_suggestions(valid_formats.to_vec())
                    .into(),
            );
        }

        Ok(())
    }
}

impl ConfigValidation for FormatConfig {
    fn validate(&self) -> Result<()> {
        self.image.validate()?;
        self.text.validate()?;
        self.parquet.validate()?;
        self.hdf5.validate()?;
        Ok(())
    }
}

impl ConfigValidation for ImageFormatConfig {
    fn validate(&self) -> Result<()> {
        if self.default_size.0 == 0 || self.default_size.1 == 0 {
            return Err(ValidationError::new(
                "formats.image.default_size",
                "Image size dimensions must be greater than 0",
            )
            .with_current_value(&format!("{:?}", self.default_size))
            .with_suggestions(vec!["Set to (224, 224) or other valid dimensions"])
            .into());
        }

        Ok(())
    }
}

impl ConfigValidation for TextFormatConfig {
    fn validate(&self) -> Result<()> {
        let valid_encodings = ["utf-8", "utf-16", "latin-1", "ascii"];
        if !valid_encodings.contains(&self.encoding.as_str()) {
            return Err(
                ValidationError::new("formats.text.encoding", "Invalid text encoding")
                    .with_current_value(&self.encoding)
                    .with_suggestions(valid_encodings.to_vec())
                    .into(),
            );
        }

        Ok(())
    }
}

impl ConfigValidation for ParquetFormatConfig {
    fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(ValidationError::new(
                "formats.parquet.batch_size",
                "Parquet batch size must be greater than 0",
            )
            .with_current_value("0")
            .with_suggestions(vec!["Set to 1024 or more"])
            .into());
        }

        Ok(())
    }
}

impl ConfigValidation for Hdf5FormatConfig {
    fn validate(&self) -> Result<()> {
        if self.chunk_cache_size == 0 {
            return Err(ValidationError::new(
                "formats.hdf5.chunk_cache_size",
                "HDF5 chunk cache size must be greater than 0",
            )
            .with_current_value("0")
            .with_suggestions(vec!["Set to 1048576 (1MB) or more"])
            .into());
        }

        if let Some(level) = self.compression_level {
            if level > 9 {
                return Err(ValidationError::new(
                    "formats.hdf5.compression_level",
                    "HDF5 compression level must be between 0 and 9",
                )
                .with_current_value(&level.to_string())
                .with_suggestions(vec!["Set to a value between 0 and 9"])
                .into());
            }
        }

        Ok(())
    }
}

impl ConfigValidation for LoggingConfig {
    fn validate(&self) -> Result<()> {
        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.level.as_str()) {
            return Err(ValidationError::new("logging.level", "Invalid log level")
                .with_current_value(&self.level)
                .with_suggestions(valid_levels.to_vec())
                .into());
        }

        let valid_formats = ["json", "text", "compact"];
        if !valid_formats.contains(&self.format.as_str()) {
            return Err(ValidationError::new("logging.format", "Invalid log format")
                .with_current_value(&self.format)
                .with_suggestions(valid_formats.to_vec())
                .into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_error_creation() {
        let error = ValidationError::new("test.field", "Test error message")
            .with_current_value("invalid_value")
            .with_suggestions(vec!["suggestion1", "suggestion2"]);

        assert_eq!(error.field, "test.field");
        assert_eq!(error.message, "Test error message");
        assert_eq!(error.current_value, Some("invalid_value".to_string()));
        assert_eq!(error.suggestions, vec!["suggestion1", "suggestion2"]);
    }

    #[test]
    fn test_valid_global_config() {
        let config = GlobalConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_batch_size() {
        let mut config = GlobalConfig::default();
        config.dataset.batch_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_resize_strategy() {
        let mut config = GlobalConfig::default();
        config.transforms.default_resize_strategy = "invalid_strategy".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_cache_policy() {
        let mut config = GlobalConfig::default();
        config.cache.eviction_policy = "invalid_policy".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cross_config_validation() {
        let mut config = GlobalConfig::default();
        config.cache.size_mb = 2048;
        config.performance.memory_pool_size = 1024;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_warnings_generation() {
        let mut config = GlobalConfig::default();
        config.dataloader.num_workers = num_cpus::get() * 4;

        let warnings = config.get_warnings();
        assert!(!warnings.is_empty());
        assert!(warnings[0].contains("num_workers"));
    }

    #[test]
    fn test_audio_config_validation() {
        let mut config = AudioConfig::default();
        config.sample_rate = 0;
        assert!(config.validate().is_err());

        config.sample_rate = 44100;
        config.channels = 0;
        assert!(config.validate().is_err());

        config.channels = 2;
        config.preferred_format = "invalid".to_string();
        assert!(config.validate().is_err());

        config.preferred_format = "wav".to_string();
        assert!(config.validate().is_ok());
    }
}
