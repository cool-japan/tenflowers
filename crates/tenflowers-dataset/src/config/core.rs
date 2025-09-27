//! Core configuration types and structures
//!
//! This module defines the main configuration structures used throughout
//! the TenfloweRS dataset library. All configurations support serialization
//! and deserialization for YAML, TOML, and JSON formats.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Global configuration containing all subsystem configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConfig {
    /// Dataset-related configuration
    pub dataset: DatasetConfig,
    /// Data loader configuration
    pub dataloader: DataLoaderConfig,
    /// Transform pipeline configuration
    pub transforms: TransformConfig,
    /// Performance and optimization settings
    pub performance: PerformanceConfig,
    /// Caching configuration
    pub cache: CacheConfig,
    /// GPU acceleration settings
    pub gpu: GpuConfig,
    /// Audio processing configuration
    pub audio: AudioConfig,
    /// Format-specific configurations
    pub formats: FormatConfig,
    /// Logging and monitoring configuration
    pub logging: LoggingConfig,
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            dataset: DatasetConfig::default(),
            dataloader: DataLoaderConfig::default(),
            transforms: TransformConfig::default(),
            performance: PerformanceConfig::default(),
            cache: CacheConfig::default(),
            gpu: GpuConfig::default(),
            audio: AudioConfig::default(),
            formats: FormatConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl GlobalConfig {
    /// Merge another configuration into this one
    pub fn merge(&mut self, other: GlobalConfig) -> crate::Result<()> {
        self.dataset.merge(other.dataset);
        self.dataloader.merge(other.dataloader);
        self.transforms.merge(other.transforms);
        self.performance.merge(other.performance);
        self.cache.merge(other.cache);
        self.gpu.merge(other.gpu);
        self.audio.merge(other.audio);
        self.formats.merge(other.formats);
        self.logging.merge(other.logging);
        Ok(())
    }
}

/// Dataset configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DatasetConfig {
    /// Default batch size for data loading
    pub batch_size: usize,
    /// Whether to shuffle data by default
    pub shuffle: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Default data root directory
    pub data_root: Option<PathBuf>,
    /// Maximum dataset size (for limiting large datasets)
    pub max_size: Option<usize>,
    /// Whether to pin memory for GPU transfers
    pub pin_memory: bool,
    /// Drop last incomplete batch
    pub drop_last: bool,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: true,
            seed: None,
            data_root: None,
            max_size: None,
            pin_memory: false,
            drop_last: false,
        }
    }
}

impl DatasetConfig {
    fn merge(&mut self, other: DatasetConfig) {
        if other.batch_size != Self::default().batch_size {
            self.batch_size = other.batch_size;
        }
        if other.shuffle != Self::default().shuffle {
            self.shuffle = other.shuffle;
        }
        if other.seed.is_some() {
            self.seed = other.seed;
        }
        if other.data_root.is_some() {
            self.data_root = other.data_root;
        }
        if other.max_size.is_some() {
            self.max_size = other.max_size;
        }
        if other.pin_memory != Self::default().pin_memory {
            self.pin_memory = other.pin_memory;
        }
        if other.drop_last != Self::default().drop_last {
            self.drop_last = other.drop_last;
        }
    }
}

/// Data loader configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DataLoaderConfig {
    /// Number of worker threads
    pub num_workers: usize,
    /// Prefetch factor for async loading
    pub prefetch_factor: usize,
    /// Enable distributed loading
    pub distributed: bool,
    /// NUMA node preferences
    pub numa_nodes: Option<Vec<usize>>,
    /// Timeout for data loading operations (seconds)
    pub timeout: Option<u64>,
    /// Enable work stealing between workers
    pub work_stealing: bool,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get(),
            prefetch_factor: 2,
            distributed: false,
            numa_nodes: None,
            timeout: Some(300), // 5 minutes
            work_stealing: true,
        }
    }
}

impl DataLoaderConfig {
    fn merge(&mut self, other: DataLoaderConfig) {
        if other.num_workers != Self::default().num_workers {
            self.num_workers = other.num_workers;
        }
        if other.prefetch_factor != Self::default().prefetch_factor {
            self.prefetch_factor = other.prefetch_factor;
        }
        if other.distributed != Self::default().distributed {
            self.distributed = other.distributed;
        }
        if other.numa_nodes.is_some() {
            self.numa_nodes = other.numa_nodes;
        }
        if other.timeout.is_some() {
            self.timeout = other.timeout;
        }
        if other.work_stealing != Self::default().work_stealing {
            self.work_stealing = other.work_stealing;
        }
    }
}

/// Transform pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TransformConfig {
    /// Enable CPU SIMD acceleration
    pub enable_simd: bool,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// Default image resize strategy
    pub default_resize_strategy: String,
    /// Augmentation probability
    pub augmentation_probability: f32,
    /// Random transform parameters
    pub random_params: HashMap<String, f32>,
    /// Transform pipeline stages
    pub pipeline_stages: Vec<String>,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            enable_gpu: false,
            default_resize_strategy: "bilinear".to_string(),
            augmentation_probability: 0.5,
            random_params: HashMap::new(),
            pipeline_stages: vec!["normalize".to_string(), "resize".to_string()],
        }
    }
}

impl TransformConfig {
    fn merge(&mut self, other: TransformConfig) {
        if other.enable_simd != Self::default().enable_simd {
            self.enable_simd = other.enable_simd;
        }
        if other.enable_gpu != Self::default().enable_gpu {
            self.enable_gpu = other.enable_gpu;
        }
        if other.default_resize_strategy != Self::default().default_resize_strategy {
            self.default_resize_strategy = other.default_resize_strategy;
        }
        if other.augmentation_probability != Self::default().augmentation_probability {
            self.augmentation_probability = other.augmentation_probability;
        }
        self.random_params.extend(other.random_params);
        if !other.pipeline_stages.is_empty() {
            self.pipeline_stages = other.pipeline_stages;
        }
    }
}

/// Performance and optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PerformanceConfig {
    /// Number of threads for parallel operations
    pub num_threads: usize,
    /// Enable memory mapping for large files
    pub enable_mmap: bool,
    /// Memory pool size (MB)
    pub memory_pool_size: usize,
    /// Enable zero-copy operations
    pub enable_zero_copy: bool,
    /// Async I/O configuration
    pub async_io: AsyncIoConfig,
    /// Performance monitoring settings
    pub monitoring: MonitoringConfig,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            enable_mmap: true,
            memory_pool_size: 1024, // 1GB
            enable_zero_copy: true,
            async_io: AsyncIoConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

impl PerformanceConfig {
    fn merge(&mut self, other: PerformanceConfig) {
        if other.num_threads != Self::default().num_threads {
            self.num_threads = other.num_threads;
        }
        if other.enable_mmap != Self::default().enable_mmap {
            self.enable_mmap = other.enable_mmap;
        }
        if other.memory_pool_size != Self::default().memory_pool_size {
            self.memory_pool_size = other.memory_pool_size;
        }
        if other.enable_zero_copy != Self::default().enable_zero_copy {
            self.enable_zero_copy = other.enable_zero_copy;
        }
        self.async_io.merge(other.async_io);
        self.monitoring.merge(other.monitoring);
    }
}

/// Async I/O configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AsyncIoConfig {
    /// Enable async I/O operations
    pub enabled: bool,
    /// Number of async I/O threads
    pub io_threads: usize,
    /// Buffer size for async operations
    pub buffer_size: usize,
    /// Queue depth for async operations
    pub queue_depth: usize,
}

impl Default for AsyncIoConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            io_threads: 4,
            buffer_size: 64 * 1024, // 64KB
            queue_depth: 32,
        }
    }
}

impl AsyncIoConfig {
    fn merge(&mut self, other: AsyncIoConfig) {
        if other.enabled != Self::default().enabled {
            self.enabled = other.enabled;
        }
        if other.io_threads != Self::default().io_threads {
            self.io_threads = other.io_threads;
        }
        if other.buffer_size != Self::default().buffer_size {
            self.buffer_size = other.buffer_size;
        }
        if other.queue_depth != Self::default().queue_depth {
            self.queue_depth = other.queue_depth;
        }
    }
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MonitoringConfig {
    /// Enable performance monitoring
    pub enabled: bool,
    /// Monitoring interval (seconds)
    pub interval: u64,
    /// Enable memory usage tracking
    pub track_memory: bool,
    /// Enable throughput tracking
    pub track_throughput: bool,
    /// Enable latency tracking
    pub track_latency: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval: 30,
            track_memory: true,
            track_throughput: true,
            track_latency: true,
        }
    }
}

impl MonitoringConfig {
    fn merge(&mut self, other: MonitoringConfig) {
        if other.enabled != Self::default().enabled {
            self.enabled = other.enabled;
        }
        if other.interval != Self::default().interval {
            self.interval = other.interval;
        }
        if other.track_memory != Self::default().track_memory {
            self.track_memory = other.track_memory;
        }
        if other.track_throughput != Self::default().track_throughput {
            self.track_throughput = other.track_throughput;
        }
        if other.track_latency != Self::default().track_latency {
            self.track_latency = other.track_latency;
        }
    }
}

/// Caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheConfig {
    /// Enable caching
    pub enabled: bool,
    /// Cache size in MB
    pub size_mb: usize,
    /// Cache eviction policy
    pub eviction_policy: String,
    /// Enable persistent cache
    pub persistent: bool,
    /// Cache directory
    pub cache_dir: Option<PathBuf>,
    /// Enable predictive prefetching
    pub predictive_prefetch: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            size_mb: 512,
            eviction_policy: "lru".to_string(),
            persistent: false,
            cache_dir: None,
            predictive_prefetch: false,
        }
    }
}

impl CacheConfig {
    fn merge(&mut self, other: CacheConfig) {
        if other.enabled != Self::default().enabled {
            self.enabled = other.enabled;
        }
        if other.size_mb != Self::default().size_mb {
            self.size_mb = other.size_mb;
        }
        if other.eviction_policy != Self::default().eviction_policy {
            self.eviction_policy = other.eviction_policy;
        }
        if other.persistent != Self::default().persistent {
            self.persistent = other.persistent;
        }
        if other.cache_dir.is_some() {
            self.cache_dir = other.cache_dir;
        }
        if other.predictive_prefetch != Self::default().predictive_prefetch {
            self.predictive_prefetch = other.predictive_prefetch;
        }
    }
}

/// GPU configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GpuConfig {
    /// Enable GPU acceleration
    pub enabled: bool,
    /// Preferred GPU device ID
    pub device_id: Option<usize>,
    /// GPU memory pool size (MB)
    pub memory_pool_mb: usize,
    /// Enable GPU-CPU memory pinning
    pub enable_pinned_memory: bool,
    /// GPU computation precision
    pub precision: String,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            device_id: None,
            memory_pool_mb: 1024,
            enable_pinned_memory: true,
            precision: "fp32".to_string(),
        }
    }
}

impl GpuConfig {
    fn merge(&mut self, other: GpuConfig) {
        if other.enabled != Self::default().enabled {
            self.enabled = other.enabled;
        }
        if other.device_id.is_some() {
            self.device_id = other.device_id;
        }
        if other.memory_pool_mb != Self::default().memory_pool_mb {
            self.memory_pool_mb = other.memory_pool_mb;
        }
        if other.enable_pinned_memory != Self::default().enable_pinned_memory {
            self.enable_pinned_memory = other.enable_pinned_memory;
        }
        if other.precision != Self::default().precision {
            self.precision = other.precision;
        }
    }
}

/// Audio processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    /// Default sample rate
    pub sample_rate: u32,
    /// Default number of channels
    pub channels: u16,
    /// Audio buffer size
    pub buffer_size: usize,
    /// Enable audio augmentation
    pub enable_augmentation: bool,
    /// Audio format preferences
    pub preferred_format: String,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 2,
            buffer_size: 1024,
            enable_augmentation: true,
            preferred_format: "wav".to_string(),
        }
    }
}

impl AudioConfig {
    fn merge(&mut self, other: AudioConfig) {
        if other.sample_rate != Self::default().sample_rate {
            self.sample_rate = other.sample_rate;
        }
        if other.channels != Self::default().channels {
            self.channels = other.channels;
        }
        if other.buffer_size != Self::default().buffer_size {
            self.buffer_size = other.buffer_size;
        }
        if other.enable_augmentation != Self::default().enable_augmentation {
            self.enable_augmentation = other.enable_augmentation;
        }
        if other.preferred_format != Self::default().preferred_format {
            self.preferred_format = other.preferred_format;
        }
    }
}

/// Format-specific configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FormatConfig {
    /// Image format configuration
    pub image: ImageFormatConfig,
    /// Text format configuration
    pub text: TextFormatConfig,
    /// Parquet format configuration
    pub parquet: ParquetFormatConfig,
    /// HDF5 format configuration
    pub hdf5: Hdf5FormatConfig,
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self {
            image: ImageFormatConfig::default(),
            text: TextFormatConfig::default(),
            parquet: ParquetFormatConfig::default(),
            hdf5: Hdf5FormatConfig::default(),
        }
    }
}

impl FormatConfig {
    fn merge(&mut self, other: FormatConfig) {
        self.image.merge(other.image);
        self.text.merge(other.text);
        self.parquet.merge(other.parquet);
        self.hdf5.merge(other.hdf5);
    }
}

/// Image format configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageFormatConfig {
    /// Default image size for resizing
    pub default_size: (u32, u32),
    /// Supported image formats
    pub supported_formats: Vec<String>,
    /// Enable lazy loading
    pub lazy_loading: bool,
}

impl Default for ImageFormatConfig {
    fn default() -> Self {
        Self {
            default_size: (224, 224),
            supported_formats: vec!["jpg".to_string(), "png".to_string(), "webp".to_string()],
            lazy_loading: true,
        }
    }
}

impl ImageFormatConfig {
    fn merge(&mut self, other: ImageFormatConfig) {
        if other.default_size != Self::default().default_size {
            self.default_size = other.default_size;
        }
        if !other.supported_formats.is_empty() {
            self.supported_formats = other.supported_formats;
        }
        if other.lazy_loading != Self::default().lazy_loading {
            self.lazy_loading = other.lazy_loading;
        }
    }
}

/// Text format configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextFormatConfig {
    /// Default encoding
    pub encoding: String,
    /// Maximum line length
    pub max_line_length: Option<usize>,
    /// Enable tokenization caching
    pub cache_tokenization: bool,
}

impl Default for TextFormatConfig {
    fn default() -> Self {
        Self {
            encoding: "utf-8".to_string(),
            max_line_length: Some(1024),
            cache_tokenization: true,
        }
    }
}

impl TextFormatConfig {
    fn merge(&mut self, other: TextFormatConfig) {
        if other.encoding != Self::default().encoding {
            self.encoding = other.encoding;
        }
        if other.max_line_length.is_some() {
            self.max_line_length = other.max_line_length;
        }
        if other.cache_tokenization != Self::default().cache_tokenization {
            self.cache_tokenization = other.cache_tokenization;
        }
    }
}

/// Parquet format configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParquetFormatConfig {
    /// Batch size for reading
    pub batch_size: usize,
    /// Enable predicate pushdown
    pub predicate_pushdown: bool,
    /// Enable column pruning
    pub column_pruning: bool,
}

impl Default for ParquetFormatConfig {
    fn default() -> Self {
        Self {
            batch_size: 1024,
            predicate_pushdown: true,
            column_pruning: true,
        }
    }
}

impl ParquetFormatConfig {
    fn merge(&mut self, other: ParquetFormatConfig) {
        if other.batch_size != Self::default().batch_size {
            self.batch_size = other.batch_size;
        }
        if other.predicate_pushdown != Self::default().predicate_pushdown {
            self.predicate_pushdown = other.predicate_pushdown;
        }
        if other.column_pruning != Self::default().column_pruning {
            self.column_pruning = other.column_pruning;
        }
    }
}

/// HDF5 format configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hdf5FormatConfig {
    /// Chunk cache size
    pub chunk_cache_size: usize,
    /// Enable parallel I/O
    pub parallel_io: bool,
    /// Compression level
    pub compression_level: Option<u32>,
}

impl Default for Hdf5FormatConfig {
    fn default() -> Self {
        Self {
            chunk_cache_size: 1024 * 1024, // 1MB
            parallel_io: true,
            compression_level: Some(6),
        }
    }
}

impl Hdf5FormatConfig {
    fn merge(&mut self, other: Hdf5FormatConfig) {
        if other.chunk_cache_size != Self::default().chunk_cache_size {
            self.chunk_cache_size = other.chunk_cache_size;
        }
        if other.parallel_io != Self::default().parallel_io {
            self.parallel_io = other.parallel_io;
        }
        if other.compression_level.is_some() {
            self.compression_level = other.compression_level;
        }
    }
}

/// Logging and monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    /// Log level
    pub level: String,
    /// Log format
    pub format: String,
    /// Enable file logging
    pub file_logging: bool,
    /// Log file path
    pub log_file: Option<PathBuf>,
    /// Enable metrics collection
    pub collect_metrics: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "json".to_string(),
            file_logging: false,
            log_file: None,
            collect_metrics: false,
        }
    }
}

impl LoggingConfig {
    fn merge(&mut self, other: LoggingConfig) {
        if other.level != Self::default().level {
            self.level = other.level;
        }
        if other.format != Self::default().format {
            self.format = other.format;
        }
        if other.file_logging != Self::default().file_logging {
            self.file_logging = other.file_logging;
        }
        if other.log_file.is_some() {
            self.log_file = other.log_file;
        }
        if other.collect_metrics != Self::default().collect_metrics {
            self.collect_metrics = other.collect_metrics;
        }
    }
}
