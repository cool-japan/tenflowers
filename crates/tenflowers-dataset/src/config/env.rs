//! Environment variable override handling
//!
//! This module provides functionality to override configuration values
//! using environment variables with a configurable prefix.

use super::{
    AudioConfig, CacheConfig, DataLoaderConfig, DatasetConfig, FormatConfig, GlobalConfig,
    GpuConfig, LoggingConfig, PerformanceConfig, TransformConfig,
};
use crate::{Result, TensorError};
use std::collections::HashMap;
use std::env;
use std::str::FromStr;

/// Environment variable override handler
#[derive(Debug)]
pub struct EnvironmentOverride {
    /// Prefix for environment variables
    prefix: String,
    /// Cache of environment variables
    env_cache: HashMap<String, String>,
    /// Whether to cache environment variables
    cache_enabled: bool,
}

impl EnvironmentOverride {
    /// Create a new environment override handler with default prefix "TENFLOWERS_"
    pub fn new() -> Self {
        Self {
            prefix: "TENFLOWERS_".to_string(),
            env_cache: HashMap::new(),
            cache_enabled: true,
        }
    }

    /// Create a new environment override handler with custom prefix
    pub fn with_prefix(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
            env_cache: HashMap::new(),
            cache_enabled: true,
        }
    }

    /// Set the environment variable prefix
    pub fn set_prefix(&mut self, prefix: &str) {
        self.prefix = prefix.to_string();
        self.clear_cache();
    }

    /// Get the current prefix
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Enable or disable environment variable caching
    pub fn set_cache_enabled(&mut self, enabled: bool) {
        self.cache_enabled = enabled;
        if !enabled {
            self.clear_cache();
        }
    }

    /// Clear the environment variable cache
    pub fn clear_cache(&mut self) {
        self.env_cache.clear();
    }

    /// Apply environment variable overrides to a configuration
    pub fn apply_overrides(&mut self, mut config: GlobalConfig) -> Result<GlobalConfig> {
        // Dataset overrides
        self.apply_dataset_overrides(&mut config.dataset)?;

        // DataLoader overrides
        self.apply_dataloader_overrides(&mut config.dataloader)?;

        // Transform overrides
        self.apply_transform_overrides(&mut config.transforms)?;

        // Performance overrides
        self.apply_performance_overrides(&mut config.performance)?;

        // Cache overrides
        self.apply_cache_overrides(&mut config.cache)?;

        // GPU overrides
        self.apply_gpu_overrides(&mut config.gpu)?;

        // Audio overrides
        self.apply_audio_overrides(&mut config.audio)?;

        // Format overrides
        self.apply_format_overrides(&mut config.formats)?;

        // Logging overrides
        self.apply_logging_overrides(&mut config.logging)?;

        Ok(config)
    }

    /// Get an environment variable with the configured prefix
    fn get_env_var(&mut self, key: &str) -> Option<String> {
        let full_key = format!("{}{}", self.prefix, key);

        if self.cache_enabled {
            if let Some(value) = self.env_cache.get(&full_key) {
                return Some(value.clone());
            }
        }

        if let Ok(value) = env::var(&full_key) {
            if self.cache_enabled {
                self.env_cache.insert(full_key, value.clone());
            }
            Some(value)
        } else {
            None
        }
    }

    /// Parse an environment variable value to a specific type
    fn parse_env_var<T>(&mut self, key: &str) -> Result<Option<T>>
    where
        T: FromStr,
        T::Err: std::fmt::Display,
    {
        if let Some(value) = self.get_env_var(key) {
            match value.parse::<T>() {
                Ok(parsed) => Ok(Some(parsed)),
                Err(e) => Err(TensorError::invalid_argument(format!(
                    "Failed to parse environment variable {}{}={}: {}",
                    self.prefix, key, value, e
                ))),
            }
        } else {
            Ok(None)
        }
    }

    /// Parse a boolean environment variable
    fn parse_bool_env_var(&mut self, key: &str) -> Result<Option<bool>> {
        if let Some(value) = self.get_env_var(key) {
            let lower_value = value.to_lowercase();
            match lower_value.as_str() {
                "true" | "1" | "yes" | "on" | "enabled" => Ok(Some(true)),
                "false" | "0" | "no" | "off" | "disabled" => Ok(Some(false)),
                _ => Err(TensorError::invalid_argument(format!(
                    "Invalid boolean value for environment variable {}{}: '{}'. Use true/false, 1/0, yes/no, on/off, or enabled/disabled",
                    self.prefix, key, value
                ))),
            }
        } else {
            Ok(None)
        }
    }

    /// Apply dataset configuration overrides
    fn apply_dataset_overrides(&mut self, config: &mut DatasetConfig) -> Result<()> {
        if let Some(batch_size) = self.parse_env_var::<usize>("DATASET_BATCH_SIZE")? {
            config.batch_size = batch_size;
        }

        if let Some(shuffle) = self.parse_bool_env_var("DATASET_SHUFFLE")? {
            config.shuffle = shuffle;
        }

        if let Some(seed) = self.parse_env_var::<u64>("DATASET_SEED")? {
            config.seed = Some(seed);
        }

        if let Some(data_root) = self.get_env_var("DATASET_DATA_ROOT") {
            config.data_root = Some(data_root.into());
        }

        if let Some(max_size) = self.parse_env_var::<usize>("DATASET_MAX_SIZE")? {
            config.max_size = Some(max_size);
        }

        if let Some(pin_memory) = self.parse_bool_env_var("DATASET_PIN_MEMORY")? {
            config.pin_memory = pin_memory;
        }

        if let Some(drop_last) = self.parse_bool_env_var("DATASET_DROP_LAST")? {
            config.drop_last = drop_last;
        }

        Ok(())
    }

    /// Apply dataloader configuration overrides
    fn apply_dataloader_overrides(&mut self, config: &mut DataLoaderConfig) -> Result<()> {
        if let Some(num_workers) = self.parse_env_var::<usize>("DATALOADER_NUM_WORKERS")? {
            config.num_workers = num_workers;
        }

        if let Some(prefetch_factor) = self.parse_env_var::<usize>("DATALOADER_PREFETCH_FACTOR")? {
            config.prefetch_factor = prefetch_factor;
        }

        if let Some(distributed) = self.parse_bool_env_var("DATALOADER_DISTRIBUTED")? {
            config.distributed = distributed;
        }

        if let Some(timeout) = self.parse_env_var::<u64>("DATALOADER_TIMEOUT")? {
            config.timeout = Some(timeout);
        }

        if let Some(work_stealing) = self.parse_bool_env_var("DATALOADER_WORK_STEALING")? {
            config.work_stealing = work_stealing;
        }

        Ok(())
    }

    /// Apply transform configuration overrides
    fn apply_transform_overrides(&mut self, config: &mut TransformConfig) -> Result<()> {
        if let Some(enable_simd) = self.parse_bool_env_var("TRANSFORMS_ENABLE_SIMD")? {
            config.enable_simd = enable_simd;
        }

        if let Some(enable_gpu) = self.parse_bool_env_var("TRANSFORMS_ENABLE_GPU")? {
            config.enable_gpu = enable_gpu;
        }

        if let Some(strategy) = self.get_env_var("TRANSFORMS_DEFAULT_RESIZE_STRATEGY") {
            config.default_resize_strategy = strategy;
        }

        if let Some(prob) = self.parse_env_var::<f32>("TRANSFORMS_AUGMENTATION_PROBABILITY")? {
            config.augmentation_probability = prob;
        }

        Ok(())
    }

    /// Apply performance configuration overrides
    fn apply_performance_overrides(&mut self, config: &mut PerformanceConfig) -> Result<()> {
        if let Some(num_threads) = self.parse_env_var::<usize>("PERFORMANCE_NUM_THREADS")? {
            config.num_threads = num_threads;
        }

        if let Some(enable_mmap) = self.parse_bool_env_var("PERFORMANCE_ENABLE_MMAP")? {
            config.enable_mmap = enable_mmap;
        }

        if let Some(memory_pool_size) =
            self.parse_env_var::<usize>("PERFORMANCE_MEMORY_POOL_SIZE")?
        {
            config.memory_pool_size = memory_pool_size;
        }

        if let Some(enable_zero_copy) = self.parse_bool_env_var("PERFORMANCE_ENABLE_ZERO_COPY")? {
            config.enable_zero_copy = enable_zero_copy;
        }

        // Async I/O overrides
        if let Some(enabled) = self.parse_bool_env_var("PERFORMANCE_ASYNC_IO_ENABLED")? {
            config.async_io.enabled = enabled;
        }

        if let Some(io_threads) = self.parse_env_var::<usize>("PERFORMANCE_ASYNC_IO_THREADS")? {
            config.async_io.io_threads = io_threads;
        }

        if let Some(buffer_size) =
            self.parse_env_var::<usize>("PERFORMANCE_ASYNC_IO_BUFFER_SIZE")?
        {
            config.async_io.buffer_size = buffer_size;
        }

        if let Some(queue_depth) =
            self.parse_env_var::<usize>("PERFORMANCE_ASYNC_IO_QUEUE_DEPTH")?
        {
            config.async_io.queue_depth = queue_depth;
        }

        // Monitoring overrides
        if let Some(enabled) = self.parse_bool_env_var("PERFORMANCE_MONITORING_ENABLED")? {
            config.monitoring.enabled = enabled;
        }

        if let Some(interval) = self.parse_env_var::<u64>("PERFORMANCE_MONITORING_INTERVAL")? {
            config.monitoring.interval = interval;
        }

        Ok(())
    }

    /// Apply cache configuration overrides
    fn apply_cache_overrides(&mut self, config: &mut CacheConfig) -> Result<()> {
        if let Some(enabled) = self.parse_bool_env_var("CACHE_ENABLED")? {
            config.enabled = enabled;
        }

        if let Some(size_mb) = self.parse_env_var::<usize>("CACHE_SIZE_MB")? {
            config.size_mb = size_mb;
        }

        if let Some(policy) = self.get_env_var("CACHE_EVICTION_POLICY") {
            config.eviction_policy = policy;
        }

        if let Some(persistent) = self.parse_bool_env_var("CACHE_PERSISTENT")? {
            config.persistent = persistent;
        }

        if let Some(cache_dir) = self.get_env_var("CACHE_DIR") {
            config.cache_dir = Some(cache_dir.into());
        }

        if let Some(predictive_prefetch) = self.parse_bool_env_var("CACHE_PREDICTIVE_PREFETCH")? {
            config.predictive_prefetch = predictive_prefetch;
        }

        Ok(())
    }

    /// Apply GPU configuration overrides
    fn apply_gpu_overrides(&mut self, config: &mut GpuConfig) -> Result<()> {
        if let Some(enabled) = self.parse_bool_env_var("GPU_ENABLED")? {
            config.enabled = enabled;
        }

        if let Some(device_id) = self.parse_env_var::<usize>("GPU_DEVICE_ID")? {
            config.device_id = Some(device_id);
        }

        if let Some(memory_pool_mb) = self.parse_env_var::<usize>("GPU_MEMORY_POOL_MB")? {
            config.memory_pool_mb = memory_pool_mb;
        }

        if let Some(enable_pinned_memory) = self.parse_bool_env_var("GPU_ENABLE_PINNED_MEMORY")? {
            config.enable_pinned_memory = enable_pinned_memory;
        }

        if let Some(precision) = self.get_env_var("GPU_PRECISION") {
            config.precision = precision;
        }

        Ok(())
    }

    /// Apply audio configuration overrides
    fn apply_audio_overrides(&mut self, config: &mut AudioConfig) -> Result<()> {
        if let Some(sample_rate) = self.parse_env_var::<u32>("AUDIO_SAMPLE_RATE")? {
            config.sample_rate = sample_rate;
        }

        if let Some(channels) = self.parse_env_var::<u16>("AUDIO_CHANNELS")? {
            config.channels = channels;
        }

        if let Some(buffer_size) = self.parse_env_var::<usize>("AUDIO_BUFFER_SIZE")? {
            config.buffer_size = buffer_size;
        }

        if let Some(enable_augmentation) = self.parse_bool_env_var("AUDIO_ENABLE_AUGMENTATION")? {
            config.enable_augmentation = enable_augmentation;
        }

        if let Some(preferred_format) = self.get_env_var("AUDIO_PREFERRED_FORMAT") {
            config.preferred_format = preferred_format;
        }

        Ok(())
    }

    /// Apply format configuration overrides
    fn apply_format_overrides(&mut self, config: &mut FormatConfig) -> Result<()> {
        // Image format overrides
        if let Some(width) = self.parse_env_var::<u32>("FORMAT_IMAGE_DEFAULT_WIDTH")? {
            config.image.default_size.0 = width;
        }

        if let Some(height) = self.parse_env_var::<u32>("FORMAT_IMAGE_DEFAULT_HEIGHT")? {
            config.image.default_size.1 = height;
        }

        if let Some(lazy_loading) = self.parse_bool_env_var("FORMAT_IMAGE_LAZY_LOADING")? {
            config.image.lazy_loading = lazy_loading;
        }

        // Text format overrides
        if let Some(encoding) = self.get_env_var("FORMAT_TEXT_ENCODING") {
            config.text.encoding = encoding;
        }

        if let Some(max_length) = self.parse_env_var::<usize>("FORMAT_TEXT_MAX_LINE_LENGTH")? {
            config.text.max_line_length = Some(max_length);
        }

        if let Some(cache_tokenization) =
            self.parse_bool_env_var("FORMAT_TEXT_CACHE_TOKENIZATION")?
        {
            config.text.cache_tokenization = cache_tokenization;
        }

        // Parquet format overrides
        if let Some(batch_size) = self.parse_env_var::<usize>("FORMAT_PARQUET_BATCH_SIZE")? {
            config.parquet.batch_size = batch_size;
        }

        if let Some(predicate_pushdown) =
            self.parse_bool_env_var("FORMAT_PARQUET_PREDICATE_PUSHDOWN")?
        {
            config.parquet.predicate_pushdown = predicate_pushdown;
        }

        if let Some(column_pruning) = self.parse_bool_env_var("FORMAT_PARQUET_COLUMN_PRUNING")? {
            config.parquet.column_pruning = column_pruning;
        }

        // HDF5 format overrides
        if let Some(chunk_cache_size) =
            self.parse_env_var::<usize>("FORMAT_HDF5_CHUNK_CACHE_SIZE")?
        {
            config.hdf5.chunk_cache_size = chunk_cache_size;
        }

        if let Some(parallel_io) = self.parse_bool_env_var("FORMAT_HDF5_PARALLEL_IO")? {
            config.hdf5.parallel_io = parallel_io;
        }

        if let Some(compression_level) =
            self.parse_env_var::<u32>("FORMAT_HDF5_COMPRESSION_LEVEL")?
        {
            config.hdf5.compression_level = Some(compression_level);
        }

        Ok(())
    }

    /// Apply logging configuration overrides
    fn apply_logging_overrides(&mut self, config: &mut LoggingConfig) -> Result<()> {
        if let Some(level) = self.get_env_var("LOGGING_LEVEL") {
            config.level = level;
        }

        if let Some(format) = self.get_env_var("LOGGING_FORMAT") {
            config.format = format;
        }

        if let Some(file_logging) = self.parse_bool_env_var("LOGGING_FILE_LOGGING")? {
            config.file_logging = file_logging;
        }

        if let Some(log_file) = self.get_env_var("LOGGING_LOG_FILE") {
            config.log_file = Some(log_file.into());
        }

        if let Some(collect_metrics) = self.parse_bool_env_var("LOGGING_COLLECT_METRICS")? {
            config.collect_metrics = collect_metrics;
        }

        Ok(())
    }

    /// List all supported environment variable names
    pub fn list_supported_variables(&self) -> Vec<String> {
        let vars = vec![
            // Dataset
            "DATASET_BATCH_SIZE",
            "DATASET_SHUFFLE",
            "DATASET_SEED",
            "DATASET_DATA_ROOT",
            "DATASET_MAX_SIZE",
            "DATASET_PIN_MEMORY",
            "DATASET_DROP_LAST",
            // DataLoader
            "DATALOADER_NUM_WORKERS",
            "DATALOADER_PREFETCH_FACTOR",
            "DATALOADER_DISTRIBUTED",
            "DATALOADER_TIMEOUT",
            "DATALOADER_WORK_STEALING",
            // Transforms
            "TRANSFORMS_ENABLE_SIMD",
            "TRANSFORMS_ENABLE_GPU",
            "TRANSFORMS_DEFAULT_RESIZE_STRATEGY",
            "TRANSFORMS_AUGMENTATION_PROBABILITY",
            // Performance
            "PERFORMANCE_NUM_THREADS",
            "PERFORMANCE_ENABLE_MMAP",
            "PERFORMANCE_MEMORY_POOL_SIZE",
            "PERFORMANCE_ENABLE_ZERO_COPY",
            "PERFORMANCE_ASYNC_IO_ENABLED",
            "PERFORMANCE_ASYNC_IO_THREADS",
            "PERFORMANCE_ASYNC_IO_BUFFER_SIZE",
            "PERFORMANCE_ASYNC_IO_QUEUE_DEPTH",
            "PERFORMANCE_MONITORING_ENABLED",
            "PERFORMANCE_MONITORING_INTERVAL",
            // Cache
            "CACHE_ENABLED",
            "CACHE_SIZE_MB",
            "CACHE_EVICTION_POLICY",
            "CACHE_PERSISTENT",
            "CACHE_DIR",
            "CACHE_PREDICTIVE_PREFETCH",
            // GPU
            "GPU_ENABLED",
            "GPU_DEVICE_ID",
            "GPU_MEMORY_POOL_MB",
            "GPU_ENABLE_PINNED_MEMORY",
            "GPU_PRECISION",
            // Audio
            "AUDIO_SAMPLE_RATE",
            "AUDIO_CHANNELS",
            "AUDIO_BUFFER_SIZE",
            "AUDIO_ENABLE_AUGMENTATION",
            "AUDIO_PREFERRED_FORMAT",
            // Formats
            "FORMAT_IMAGE_DEFAULT_WIDTH",
            "FORMAT_IMAGE_DEFAULT_HEIGHT",
            "FORMAT_IMAGE_LAZY_LOADING",
            "FORMAT_TEXT_ENCODING",
            "FORMAT_TEXT_MAX_LINE_LENGTH",
            "FORMAT_TEXT_CACHE_TOKENIZATION",
            "FORMAT_PARQUET_BATCH_SIZE",
            "FORMAT_PARQUET_PREDICATE_PUSHDOWN",
            "FORMAT_PARQUET_COLUMN_PRUNING",
            "FORMAT_HDF5_CHUNK_CACHE_SIZE",
            "FORMAT_HDF5_PARALLEL_IO",
            "FORMAT_HDF5_COMPRESSION_LEVEL",
            // Logging
            "LOGGING_LEVEL",
            "LOGGING_FORMAT",
            "LOGGING_FILE_LOGGING",
            "LOGGING_LOG_FILE",
            "LOGGING_COLLECT_METRICS",
        ];

        vars.into_iter()
            .map(|var| format!("{}{}", self.prefix, var))
            .collect()
    }
}

impl Default for EnvironmentOverride {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_environment_override_creation() {
        let env_override = EnvironmentOverride::new();
        assert_eq!(env_override.prefix(), "TENFLOWERS_");

        let custom_override = EnvironmentOverride::with_prefix("CUSTOM_");
        assert_eq!(custom_override.prefix(), "CUSTOM_");
    }

    #[test]
    fn test_prefix_setting() {
        let mut env_override = EnvironmentOverride::new();
        env_override.set_prefix("TEST_");
        assert_eq!(env_override.prefix(), "TEST_");
    }

    #[test]
    fn test_dataset_overrides() {
        // Clean up any potentially interfering environment variables first
        env::remove_var("TEST_DATASET_BATCH_SIZE");

        env::set_var("TEST_DATASET_BATCH_SIZE", "128");
        env::set_var("TEST_DATASET_SHUFFLE", "false");
        env::set_var("TEST_DATASET_SEED", "42");

        let mut env_override = EnvironmentOverride::with_prefix("TEST_");
        let config = GlobalConfig::default();
        let updated_config = env_override.apply_overrides(config).unwrap();

        assert_eq!(updated_config.dataset.batch_size, 128);
        assert_eq!(updated_config.dataset.shuffle, false);
        assert_eq!(updated_config.dataset.seed, Some(42));

        // Clean up
        env::remove_var("TEST_DATASET_BATCH_SIZE");
        env::remove_var("TEST_DATASET_SHUFFLE");
        env::remove_var("TEST_DATASET_SEED");
    }

    #[test]
    fn test_boolean_parsing() {
        let mut env_override = EnvironmentOverride::with_prefix("TEST_");

        // Test various boolean representations
        env::set_var("TEST_BOOL_TRUE", "true");
        assert_eq!(
            env_override.parse_bool_env_var("BOOL_TRUE").unwrap(),
            Some(true)
        );

        env::set_var("TEST_BOOL_FALSE", "false");
        assert_eq!(
            env_override.parse_bool_env_var("BOOL_FALSE").unwrap(),
            Some(false)
        );

        env::set_var("TEST_BOOL_1", "1");
        assert_eq!(
            env_override.parse_bool_env_var("BOOL_1").unwrap(),
            Some(true)
        );

        env::set_var("TEST_BOOL_0", "0");
        assert_eq!(
            env_override.parse_bool_env_var("BOOL_0").unwrap(),
            Some(false)
        );

        env::set_var("TEST_BOOL_YES", "yes");
        assert_eq!(
            env_override.parse_bool_env_var("BOOL_YES").unwrap(),
            Some(true)
        );

        env::set_var("TEST_BOOL_NO", "no");
        assert_eq!(
            env_override.parse_bool_env_var("BOOL_NO").unwrap(),
            Some(false)
        );

        env::set_var("TEST_BOOL_INVALID", "invalid");
        assert!(env_override.parse_bool_env_var("BOOL_INVALID").is_err());

        // Clean up
        env::remove_var("TEST_BOOL_TRUE");
        env::remove_var("TEST_BOOL_FALSE");
        env::remove_var("TEST_BOOL_1");
        env::remove_var("TEST_BOOL_0");
        env::remove_var("TEST_BOOL_YES");
        env::remove_var("TEST_BOOL_NO");
        env::remove_var("TEST_BOOL_INVALID");
    }

    #[test]
    fn test_performance_overrides() {
        env::set_var("TEST_PERFORMANCE_NUM_THREADS", "16");
        env::set_var("TEST_PERFORMANCE_ENABLE_MMAP", "false");
        env::set_var("TEST_PERFORMANCE_MEMORY_POOL_SIZE", "2048");

        let mut env_override = EnvironmentOverride::with_prefix("TEST_");
        let config = GlobalConfig::default();
        let updated_config = env_override.apply_overrides(config).unwrap();

        assert_eq!(updated_config.performance.num_threads, 16);
        assert_eq!(updated_config.performance.enable_mmap, false);
        assert_eq!(updated_config.performance.memory_pool_size, 2048);

        // Clean up
        env::remove_var("TEST_PERFORMANCE_NUM_THREADS");
        env::remove_var("TEST_PERFORMANCE_ENABLE_MMAP");
        env::remove_var("TEST_PERFORMANCE_MEMORY_POOL_SIZE");
    }

    #[test]
    fn test_invalid_numeric_override() {
        env::set_var("INVALID_TEST_DATASET_BATCH_SIZE", "invalid_number");

        let mut env_override = EnvironmentOverride::with_prefix("INVALID_TEST_");
        let config = GlobalConfig::default();
        let result = env_override.apply_overrides(config);

        assert!(result.is_err());

        // Clean up
        env::remove_var("INVALID_TEST_DATASET_BATCH_SIZE");
    }

    #[test]
    fn test_supported_variables_list() {
        let env_override = EnvironmentOverride::with_prefix("TEST_");
        let variables = env_override.list_supported_variables();

        assert!(!variables.is_empty());
        assert!(variables.iter().all(|var| var.starts_with("TEST_")));
        assert!(variables.contains(&"TEST_DATASET_BATCH_SIZE".to_string()));
        assert!(variables.contains(&"TEST_GPU_ENABLED".to_string()));
    }

    #[test]
    fn test_cache_functionality() {
        let mut env_override = EnvironmentOverride::with_prefix("TEST_");

        // Enable caching
        env_override.set_cache_enabled(true);

        env::set_var("TEST_CACHE_TEST", "test_value");

        // First access should cache the value
        let value1 = env_override.get_env_var("CACHE_TEST");
        assert_eq!(value1, Some("test_value".to_string()));

        // Change the environment variable
        env::set_var("TEST_CACHE_TEST", "new_value");

        // Should still return cached value
        let value2 = env_override.get_env_var("CACHE_TEST");
        assert_eq!(value2, Some("test_value".to_string()));

        // Clear cache and try again
        env_override.clear_cache();
        let value3 = env_override.get_env_var("CACHE_TEST");
        assert_eq!(value3, Some("new_value".to_string()));

        // Disable caching
        env_override.set_cache_enabled(false);
        env::set_var("TEST_CACHE_TEST", "final_value");
        let value4 = env_override.get_env_var("CACHE_TEST");
        assert_eq!(value4, Some("final_value".to_string()));

        // Clean up
        env::remove_var("TEST_CACHE_TEST");
    }
}
