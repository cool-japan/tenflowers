//! Configuration management system for TenfloweRS Dataset
//!
//! This module provides a comprehensive configuration management system supporting
//! YAML and TOML formats for complex dataset setups. It includes validation,
//! environment variable overrides, and hot-reload capabilities.
//!
//! ## Features
//!
//! - **Multiple Formats**: Support for YAML and TOML configuration files
//! - **Environment Variables**: Override any configuration value using environment variables
//! - **Validation**: Comprehensive validation with descriptive error messages
//! - **Hot Reload**: Watch configuration files for changes and reload automatically
//! - **Defaults**: Sensible defaults for all configuration options
//! - **Hierarchical**: Support for nested configurations with inheritance
//!
//! ## Sub-modules
//!
//! - `core`: Core configuration types and traits
//! - `loader`: Configuration file loading and parsing
//! - `watcher`: Hot-reload functionality for configuration files
//! - `env`: Environment variable override handling
//! - `validation`: Configuration validation utilities

pub mod core;
pub mod env;
pub mod loader;
pub mod validation;
pub mod watcher;

// Re-export commonly used types
pub use core::{
    AsyncIoConfig, AudioConfig, CacheConfig, DataLoaderConfig, DatasetConfig, FormatConfig,
    GlobalConfig, GpuConfig, Hdf5FormatConfig, ImageFormatConfig, LoggingConfig, MonitoringConfig,
    ParquetFormatConfig, PerformanceConfig, TextFormatConfig, TransformConfig,
};
pub use env::EnvironmentOverride;
pub use loader::{ConfigLoader, ConfigSource};
pub use validation::{ConfigValidation, ValidationError, ValidationResult};
pub use watcher::{ConfigWatcher, WatchEvent};

use crate::Result;
use std::path::Path;

/// Configuration manager that orchestrates loading, validation, and watching
#[derive(Debug)]
pub struct ConfigManager {
    config: GlobalConfig,
    loader: ConfigLoader,
    watcher: Option<ConfigWatcher>,
    env_override: EnvironmentOverride,
}

impl ConfigManager {
    /// Create a new configuration manager with default settings
    pub fn new() -> Self {
        Self {
            config: GlobalConfig::default(),
            loader: ConfigLoader::new(),
            watcher: None,
            env_override: EnvironmentOverride::new(),
        }
    }

    /// Load configuration from a file path
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<&GlobalConfig> {
        let config = self.loader.load_from_file(path)?;
        self.config = self.env_override.apply_overrides(config)?;
        self.config.validate()?;
        Ok(&self.config)
    }

    /// Load configuration from a string
    pub fn load_from_string(
        &mut self,
        content: &str,
        format: ConfigFormat,
    ) -> Result<&GlobalConfig> {
        let config = self.loader.load_from_string(content, format)?;
        self.config = self.env_override.apply_overrides(config)?;
        self.config.validate()?;
        Ok(&self.config)
    }

    /// Get the current configuration
    pub fn config(&self) -> &GlobalConfig {
        &self.config
    }

    /// Enable hot-reload for a configuration file
    pub fn watch_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let watcher = ConfigWatcher::new(path)?;
        self.watcher = Some(watcher);
        Ok(())
    }

    /// Check if configuration file has changed and reload if necessary
    pub fn check_reload(&mut self) -> Result<bool> {
        if let Some(ref mut watcher) = self.watcher {
            if let Some(event) = watcher.check_changes()? {
                match event {
                    WatchEvent::Modified(path) => {
                        self.load_from_file(&path)?;
                        return Ok(true);
                    }
                    WatchEvent::Error(err) => {
                        eprintln!("Configuration watch error: {}", err);
                    }
                }
            }
        }
        Ok(false)
    }

    /// Merge additional configuration
    pub fn merge_config(&mut self, other: GlobalConfig) -> Result<()> {
        self.config.merge(other)?;
        self.config.validate()?;
        Ok(())
    }

    /// Set environment variable prefix for overrides
    pub fn set_env_prefix(&mut self, prefix: &str) {
        self.env_override.set_prefix(prefix);
    }
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Supported configuration file formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFormat {
    /// YAML format (.yaml, .yml)
    Yaml,
    /// TOML format (.toml)
    Toml,
    /// JSON format (.json)
    Json,
}

impl ConfigFormat {
    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "yaml" | "yml" => Some(Self::Yaml),
            "toml" => Some(Self::Toml),
            "json" => Some(Self::Json),
            _ => None,
        }
    }

    /// Get file extensions for this format
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Self::Yaml => &["yaml", "yml"],
            Self::Toml => &["toml"],
            Self::Json => &["json"],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_manager_creation() {
        let manager = ConfigManager::new();
        assert_eq!(manager.config().dataset.batch_size, 32); // Default value
    }

    #[test]
    fn test_config_format_detection() {
        assert_eq!(
            ConfigFormat::from_extension("yaml"),
            Some(ConfigFormat::Yaml)
        );
        assert_eq!(
            ConfigFormat::from_extension("yml"),
            Some(ConfigFormat::Yaml)
        );
        assert_eq!(
            ConfigFormat::from_extension("toml"),
            Some(ConfigFormat::Toml)
        );
        assert_eq!(
            ConfigFormat::from_extension("json"),
            Some(ConfigFormat::Json)
        );
        assert_eq!(ConfigFormat::from_extension("txt"), None);
    }

    #[test]
    fn test_config_loading_from_string() {
        let mut manager = ConfigManager::new();

        let yaml_config = r#"
dataset:
  batch_size: 64
  shuffle: true
  pin_memory: false
  drop_last: false
dataloader:
  num_workers: 8
  prefetch_factor: 2
  distributed: false
  work_stealing: true
transforms:
  enable_simd: true
  enable_gpu: false
performance:
  num_threads: 4
  enable_mmap: false
cache:
  enabled: false
  size_mb: 64
gpu:
  enabled: false
audio:
  sample_rate: 44100
  channels: 2
formats:
  image:
    default_size: [224, 224]
    supported_formats: ["jpg", "png"]
    lazy_loading: false
logging:
  level: "info"
"#;

        let config = manager
            .load_from_string(yaml_config, ConfigFormat::Yaml)
            .unwrap();
        assert_eq!(config.dataset.batch_size, 64);
        assert_eq!(config.dataloader.num_workers, 8);
    }

    #[test]
    fn test_config_file_loading() {
        let mut file = NamedTempFile::with_suffix(".toml").unwrap();
        writeln!(
            file,
            r#"
[dataset]
batch_size = 128
shuffle = false
pin_memory = true
drop_last = false

[dataloader]
num_workers = 16
prefetch_factor = 4
distributed = false
work_stealing = true

[transforms]
enable_simd = false
enable_gpu = false

[performance]
num_threads = 8
enable_mmap = true

[cache]
enabled = true
size_mb = 128

[gpu]
enabled = false

[audio]
sample_rate = 44100
channels = 2

[formats.image]
default_size = [224, 224]
supported_formats = ["jpg", "png"]
lazy_loading = false

[logging]
level = "info"
"#
        )
        .unwrap();

        let mut manager = ConfigManager::new();
        let config = manager.load_from_file(file.path()).unwrap();
        assert_eq!(config.dataset.batch_size, 128);
        assert_eq!(config.dataloader.num_workers, 16);
    }

    #[test]
    fn test_config_merging() {
        let mut manager = ConfigManager::new();

        let base_config = r#"
dataset:
  batch_size: 32
  shuffle: true
  pin_memory: false
  drop_last: false
dataloader:
  num_workers: 4
  prefetch_factor: 2
  distributed: false
  work_stealing: true
transforms:
  enable_simd: true
  enable_gpu: false
performance:
  num_threads: 2
  enable_mmap: false
cache:
  enabled: false
  size_mb: 32
gpu:
  enabled: false
audio:
  sample_rate: 44100
  channels: 2
formats:
  image:
    default_size: [224, 224]
    supported_formats: ["jpg", "png"]
    lazy_loading: false
logging:
  level: "info"
"#;

        manager
            .load_from_string(base_config, ConfigFormat::Yaml)
            .unwrap();

        let override_config = GlobalConfig {
            dataset: DatasetConfig {
                batch_size: 64,
                ..Default::default()
            },
            ..Default::default()
        };

        manager.merge_config(override_config).unwrap();
        assert_eq!(manager.config().dataset.batch_size, 64);
        assert_eq!(manager.config().dataloader.num_workers, 4); // Should keep original
    }
}
