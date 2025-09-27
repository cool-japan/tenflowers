//! Configuration file loading and parsing
//!
//! This module provides functionality to load configurations from various
//! file formats including YAML, TOML, and JSON.

use super::{ConfigFormat, GlobalConfig};
use crate::{Result, TensorError};
use std::fs;
use std::path::Path;

/// Configuration source information
#[derive(Debug, Clone)]
pub enum ConfigSource {
    /// Configuration loaded from a file
    File(std::path::PathBuf),
    /// Configuration loaded from a string
    String { format: ConfigFormat },
    /// Configuration created programmatically
    Programmatic,
}

/// Configuration loader for various formats
#[derive(Debug)]
pub struct ConfigLoader {
    /// History of loaded configurations
    load_history: Vec<ConfigSource>,
}

impl ConfigLoader {
    /// Create a new configuration loader
    pub fn new() -> Self {
        Self {
            load_history: Vec::new(),
        }
    }

    /// Load configuration from a file
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<GlobalConfig> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(|e| {
            TensorError::invalid_argument(format!(
                "Failed to read configuration file '{}': {}",
                path.display(),
                e
            ))
        })?;

        let format = self.detect_format(path)?;
        let config = self.parse_content(&content, format)?;

        self.load_history
            .push(ConfigSource::File(path.to_path_buf()));
        Ok(config)
    }

    /// Load configuration from a string
    pub fn load_from_string(
        &mut self,
        content: &str,
        format: ConfigFormat,
    ) -> Result<GlobalConfig> {
        let config = self.parse_content(content, format)?;
        self.load_history.push(ConfigSource::String { format });
        Ok(config)
    }

    /// Get the load history
    pub fn load_history(&self) -> &[ConfigSource] {
        &self.load_history
    }

    /// Clear the load history
    pub fn clear_history(&mut self) {
        self.load_history.clear();
    }

    /// Detect configuration format from file extension
    fn detect_format<P: AsRef<Path>>(&self, path: P) -> Result<ConfigFormat> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| {
                TensorError::invalid_argument(format!(
                    "Could not determine file extension for '{}'",
                    path.display()
                ))
            })?;

        ConfigFormat::from_extension(extension).ok_or_else(|| {
            TensorError::invalid_argument(format!(
                "Unsupported configuration file format: '{}'",
                extension
            ))
        })
    }

    /// Parse configuration content based on format
    fn parse_content(&self, content: &str, format: ConfigFormat) -> Result<GlobalConfig> {
        match format {
            ConfigFormat::Yaml => self.parse_yaml(content),
            ConfigFormat::Toml => self.parse_toml(content),
            ConfigFormat::Json => self.parse_json(content),
        }
    }

    /// Parse YAML configuration
    #[cfg(feature = "serialize")]
    fn parse_yaml(&self, content: &str) -> Result<GlobalConfig> {
        serde_yaml::from_str(content).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to parse YAML configuration: {}", e))
        })
    }

    /// Parse YAML configuration (fallback when serialize feature is disabled)
    #[cfg(not(feature = "serialize"))]
    fn parse_yaml(&self, _content: &str) -> Result<GlobalConfig> {
        Err(TensorError::invalid_argument(
            "YAML parsing requires 'serialize' feature to be enabled".to_string(),
        ))
    }

    /// Parse TOML configuration
    fn parse_toml(&self, content: &str) -> Result<GlobalConfig> {
        toml::from_str(content).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to parse TOML configuration: {}", e))
        })
    }

    /// Parse JSON configuration
    #[cfg(feature = "serialize")]
    fn parse_json(&self, content: &str) -> Result<GlobalConfig> {
        serde_json::from_str(content).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to parse JSON configuration: {}", e))
        })
    }

    /// Parse JSON configuration (fallback when serialize feature is disabled)
    #[cfg(not(feature = "serialize"))]
    fn parse_json(&self, _content: &str) -> Result<GlobalConfig> {
        Err(TensorError::invalid_argument(
            "JSON parsing requires 'serialize' feature to be enabled".to_string(),
        ))
    }

    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, config: &GlobalConfig, path: P) -> Result<()> {
        let path = path.as_ref();
        let format = self.detect_format(path)?;
        let content = self.serialize_config(config, format)?;

        fs::write(path, content).map_err(|e| {
            TensorError::invalid_argument(format!(
                "Failed to write configuration file '{}': {}",
                path.display(),
                e
            ))
        })
    }

    /// Serialize configuration to string
    fn serialize_config(&self, config: &GlobalConfig, format: ConfigFormat) -> Result<String> {
        match format {
            ConfigFormat::Yaml => self.serialize_yaml(config),
            ConfigFormat::Toml => self.serialize_toml(config),
            ConfigFormat::Json => self.serialize_json(config),
        }
    }

    /// Serialize to YAML
    #[cfg(feature = "serialize")]
    fn serialize_yaml(&self, config: &GlobalConfig) -> Result<String> {
        serde_yaml::to_string(config).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to serialize YAML configuration: {}", e))
        })
    }

    /// Serialize to YAML (fallback when serialize feature is disabled)
    #[cfg(not(feature = "serialize"))]
    fn serialize_yaml(&self, _config: &GlobalConfig) -> Result<String> {
        Err(TensorError::invalid_argument(
            "YAML serialization requires 'serialize' feature to be enabled".to_string(),
        ))
    }

    /// Serialize to TOML
    fn serialize_toml(&self, config: &GlobalConfig) -> Result<String> {
        toml::to_string_pretty(config).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to serialize TOML configuration: {}", e))
        })
    }

    /// Serialize to JSON
    #[cfg(feature = "serialize")]
    fn serialize_json(&self, config: &GlobalConfig) -> Result<String> {
        serde_json::to_string_pretty(config).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to serialize JSON configuration: {}", e))
        })
    }

    /// Serialize to JSON (fallback when serialize feature is disabled)
    #[cfg(not(feature = "serialize"))]
    fn serialize_json(&self, _config: &GlobalConfig) -> Result<String> {
        Err(TensorError::invalid_argument(
            "JSON serialization requires 'serialize' feature to be enabled".to_string(),
        ))
    }
}

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_format_detection() {
        let loader = ConfigLoader::new();

        // Test YAML
        let yaml_path = std::path::Path::new("config.yaml");
        assert!(matches!(
            loader.detect_format(yaml_path).unwrap(),
            ConfigFormat::Yaml
        ));

        let yml_path = std::path::Path::new("config.yml");
        assert!(matches!(
            loader.detect_format(yml_path).unwrap(),
            ConfigFormat::Yaml
        ));

        // Test TOML
        let toml_path = std::path::Path::new("config.toml");
        assert!(matches!(
            loader.detect_format(toml_path).unwrap(),
            ConfigFormat::Toml
        ));

        // Test JSON
        let json_path = std::path::Path::new("config.json");
        assert!(matches!(
            loader.detect_format(json_path).unwrap(),
            ConfigFormat::Json
        ));

        // Test unsupported
        let txt_path = std::path::Path::new("config.txt");
        assert!(loader.detect_format(txt_path).is_err());
    }

    #[test]
    fn test_toml_parsing() {
        let mut loader = ConfigLoader::new();

        let toml_content = r#"
[dataset]
batch_size = 64
shuffle = true
pin_memory = false
drop_last = false

[dataloader]
num_workers = 4
prefetch_factor = 2
distributed = false
work_stealing = true

[transforms]
enable_simd = true
enable_gpu = false
default_resize_strategy = "bilinear"
augmentation_probability = 0.5

[performance]
num_threads = 8
enable_mmap = false
memory_pool_size = 512
enable_zero_copy = false

[cache]
enabled = true
size_mb = 256
eviction_policy = "lru"
persistent = false
predictive_prefetch = false

[gpu]
enabled = false
memory_pool_mb = 1024
enable_pinned_memory = true
precision = "fp32"

[audio]
sample_rate = 44100
channels = 2
buffer_size = 1024
enable_augmentation = false
preferred_format = "wav"

[logging]
level = "info"
format = "text"
file_logging = false
collect_metrics = false

[formats.image]
default_size = [224, 224]
supported_formats = ["jpg", "png"]
lazy_loading = false

[formats.text]
encoding = "utf-8"
cache_tokenization = false

[formats.parquet]
batch_size = 1024
predicate_pushdown = true
column_pruning = true

[formats.hdf5]
chunk_cache_size = 1048576
parallel_io = false
"#;

        let config = loader
            .load_from_string(toml_content, ConfigFormat::Toml)
            .unwrap();
        assert_eq!(config.dataset.batch_size, 64);
        assert_eq!(config.dataset.shuffle, true);
        assert_eq!(config.performance.num_threads, 8);
        assert_eq!(config.performance.enable_mmap, false);
        assert_eq!(config.cache.enabled, true);
        assert_eq!(config.cache.size_mb, 256);
    }

    #[test]
    #[cfg(feature = "serialize")]
    fn test_yaml_parsing() {
        let mut loader = ConfigLoader::new();

        let yaml_content = r#"
dataset:
  batch_size: 128
  shuffle: false
  seed: 42
  pin_memory: true
  drop_last: false

dataloader:
  num_workers: 4
  prefetch_factor: 2
  distributed: false
  work_stealing: true

performance:
  num_threads: 16
  enable_mmap: true
  memory_pool_size: 2048
  enable_zero_copy: true

transforms:
  enable_simd: true
  enable_gpu: false
  default_resize_strategy: "bilinear"
  augmentation_probability: 0.5

cache:
  enabled: false
  size_mb: 128
  eviction_policy: "lru"
  persistent: false
  predictive_prefetch: false

gpu:
  enabled: false
  memory_pool_mb: 512
  enable_pinned_memory: true
  precision: "fp32"

audio:
  sample_rate: 48000
  channels: 1
  buffer_size: 2048
  enable_augmentation: true
  preferred_format: "flac"

logging:
  level: "debug"
  format: "json"
  file_logging: true
  collect_metrics: true

formats:
  image:
    default_size: [256, 256]
    supported_formats: ["jpg", "png", "webp"]
    lazy_loading: true
  text:
    encoding: "utf-16"
    cache_tokenization: true
  parquet:
    batch_size: 2048
    predicate_pushdown: false
    column_pruning: false
  hdf5:
    chunk_cache_size: 2097152
    parallel_io: true
"#;

        let config = loader
            .load_from_string(yaml_content, ConfigFormat::Yaml)
            .unwrap();
        assert_eq!(config.dataset.batch_size, 128);
        assert_eq!(config.dataset.shuffle, false);
        assert_eq!(config.dataset.seed, Some(42));
        assert_eq!(config.performance.num_threads, 16);
        assert_eq!(config.performance.memory_pool_size, 2048);
        assert_eq!(config.transforms.enable_simd, true);
        assert_eq!(config.transforms.enable_gpu, false);
    }

    #[test]
    #[cfg(feature = "serialize")]
    fn test_json_parsing() {
        let mut loader = ConfigLoader::new();

        let json_content = r#"
{
  "dataset": {
    "batch_size": 32,
    "shuffle": true,
    "pin_memory": false,
    "drop_last": false
  },
  "dataloader": {
    "num_workers": 4,
    "prefetch_factor": 3,
    "distributed": false,
    "work_stealing": true
  },
  "transforms": {
    "enable_simd": true,
    "enable_gpu": false,
    "default_resize_strategy": "bilinear",
    "augmentation_probability": 0.5
  },
  "performance": {
    "num_threads": 8,
    "enable_mmap": true,
    "memory_pool_size": 1024,
    "enable_zero_copy": false
  },
  "cache": {
    "enabled": false,
    "size_mb": 64,
    "eviction_policy": "lfu",
    "persistent": false,
    "predictive_prefetch": false
  },
  "gpu": {
    "enabled": true,
    "device_id": 0,
    "memory_pool_mb": 1024,
    "enable_pinned_memory": true,
    "precision": "fp32"
  },
  "audio": {
    "sample_rate": 22050,
    "channels": 1,
    "buffer_size": 512,
    "enable_augmentation": false,
    "preferred_format": "wav"
  },
  "logging": {
    "level": "warn",
    "format": "compact",
    "file_logging": false,
    "collect_metrics": false
  },
  "formats": {
    "image": {
      "default_size": [512, 512],
      "supported_formats": ["jpg", "png", "tiff"],
      "lazy_loading": true
    },
    "text": {
      "encoding": "latin-1",
      "cache_tokenization": false
    },
    "parquet": {
      "batch_size": 512,
      "predicate_pushdown": true,
      "column_pruning": false
    },
    "hdf5": {
      "chunk_cache_size": 524288,
      "parallel_io": false
    }
  }
}
"#;

        let config = loader
            .load_from_string(json_content, ConfigFormat::Json)
            .unwrap();
        assert_eq!(config.dataset.batch_size, 32);
        assert_eq!(config.dataset.shuffle, true);
        assert_eq!(config.dataloader.num_workers, 4);
        assert_eq!(config.dataloader.prefetch_factor, 3);
        assert_eq!(config.gpu.enabled, true);
        assert_eq!(config.gpu.device_id, Some(0));
    }

    #[test]
    fn test_file_loading() {
        let mut file = NamedTempFile::with_suffix(".toml").unwrap();
        writeln!(
            file,
            r#"
[dataset]
batch_size = 256
shuffle = true
pin_memory = false
drop_last = false

[dataloader]
num_workers = 2
prefetch_factor = 1
distributed = false
work_stealing = false

[transforms]
enable_simd = false
enable_gpu = false
default_resize_strategy = "nearest"
augmentation_probability = 0.0

[performance]
num_threads = 4
enable_mmap = true
memory_pool_size = 256
enable_zero_copy = true

[cache]
enabled = false
size_mb = 0
eviction_policy = "lru"
persistent = false
predictive_prefetch = false

[gpu]
enabled = false
memory_pool_mb = 512
enable_pinned_memory = false
precision = "fp32"

[audio]
sample_rate = 16000
channels = 1
buffer_size = 256
enable_augmentation = false
preferred_format = "wav"

[logging]
level = "error"
format = "text"
file_logging = false
collect_metrics = false

[formats.image]
default_size = [128, 128]
supported_formats = ["png", "bmp"]
lazy_loading = false

[formats.text]
encoding = "ascii"
cache_tokenization = true

[formats.parquet]
batch_size = 256
predicate_pushdown = false
column_pruning = true

[formats.hdf5]
chunk_cache_size = 131072
parallel_io = false
"#
        )
        .unwrap();

        let mut loader = ConfigLoader::new();
        let config = loader.load_from_file(file.path()).unwrap();
        assert_eq!(config.dataset.batch_size, 256);
        assert_eq!(config.cache.enabled, false);

        // Check load history
        let history = loader.load_history();
        assert_eq!(history.len(), 1);
        match &history[0] {
            ConfigSource::File(path) => assert_eq!(path, file.path()),
            _ => panic!("Expected File source"),
        }
    }

    #[test]
    fn test_config_serialization() {
        let loader = ConfigLoader::new();
        let config = GlobalConfig::default();

        // Test TOML serialization
        let toml_content = loader.serialize_toml(&config).unwrap();
        assert!(toml_content.contains("[dataset]"));
        assert!(toml_content.contains("batch_size = 32"));

        // Parse it back
        let mut loader2 = ConfigLoader::new();
        let parsed_config = loader2
            .load_from_string(&toml_content, ConfigFormat::Toml)
            .unwrap();
        assert_eq!(parsed_config.dataset.batch_size, config.dataset.batch_size);
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let mut config = GlobalConfig::default();
        config.dataset.batch_size = 128;
        config.performance.num_threads = 16;
        config.cache.size_mb = 1024;

        let temp_file = NamedTempFile::with_suffix(".toml").unwrap();
        let loader = ConfigLoader::new();

        // Save configuration
        loader.save_to_file(&config, temp_file.path()).unwrap();

        // Load it back
        let mut loader2 = ConfigLoader::new();
        let loaded_config = loader2.load_from_file(temp_file.path()).unwrap();

        assert_eq!(loaded_config.dataset.batch_size, 128);
        assert_eq!(loaded_config.performance.num_threads, 16);
        assert_eq!(loaded_config.cache.size_mb, 1024);
    }
}
