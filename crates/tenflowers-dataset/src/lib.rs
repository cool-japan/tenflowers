//! # TenfloweRS Dataset
//!
//! Efficient data loading, preprocessing, and augmentation for machine learning in TenfloweRS.
//! This crate provides high-performance data pipelines with support for various formats, transformations,
//! and distributed loading strategies.
//!
//! ## Features
//!
//! - **Multiple Data Formats**: CSV, image folders, HDF5, Arrow/Parquet, JSON, and custom formats
//! - **Efficient Data Loading**: Multi-threaded prefetching, NUMA-aware scheduling, zero-copy operations
//! - **Rich Transformations**: SIMD-accelerated transforms, GPU preprocessing, composition
//! - **Advanced Sampling**: Stratified, importance, and distributed sampling strategies
//! - **Data Quality**: Built-in quality analysis, outlier detection, and drift monitoring
//! - **Production Features**: Checkpointing, versioning, reproducibility, and debugging tools
//! - **Streaming Support**: Large dataset streaming with predictive prefetching
//!
//! ## Quick Start
//!
//! ### Loading Data from CSV
//!
//! ```rust,ignore
//! use tenflowers_dataset::{CsvDataset, CsvDatasetBuilder};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load CSV data
//! let dataset = CsvDatasetBuilder::new("data.csv")
//!     .has_header(true)
//!     .delimiter(b',')
//!     .build()?;
//!
//! println!("Dataset has {} samples", dataset.len());
//! # Ok(())
//! # }
//! ```
//!
//! ### Image Folder Dataset
//!
//! ```rust,ignore
//! use tenflowers_dataset::{ImageFolderDataset, ImageFolderDatasetBuilder};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load images from directory structure:
//! // train/
//! //   cat/
//! //     img1.jpg
//! //     img2.jpg
//! //   dog/
//! //     img3.jpg
//! let dataset = ImageFolderDatasetBuilder::new("train/")
//!     .image_size((224, 224))
//!     .build()?;
//!
//! println!("Found {} images in {} classes", dataset.len(), dataset.num_classes());
//! # Ok(())
//! # }
//! ```
//!
//! ### Data Loader with Batching
//!
//! ```rust,ignore
//! use tenflowers_dataset::{DataLoader, DataLoaderBuilder};
//! use tenflowers_dataset::{CsvDataset, RandomSampler};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let dataset = CsvDataset::new("data.csv")?;
//! // Create a data loader
//! let loader = DataLoaderBuilder::new(dataset)
//!     .batch_size(32)
//!     .shuffle(true)
//!     .num_workers(4)
//!     .prefetch(2)
//!     .build()?;
//!
//! // Iterate through batches
//! for batch in loader.iter() {
//!     let (features, labels) = batch?;
//!     // Training step...
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Data Transformations
//!
//! ```rust,ignore
//! use tenflowers_dataset::transforms::{Compose, Normalize, RandomCrop, ToTensor};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Compose multiple transformations
//! let transform = Compose::new(vec![
//!     Box::new(RandomCrop::new((224, 224))),
//!     Box::new(ToTensor),
//!     Box::new(Normalize::new(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225])),
//! ]);
//! # Ok(())
//! # }
//! ```
//!
//! ## Advanced Features
//!
//! ### Distributed Data Loading
//!
//! ```rust,ignore
//! use tenflowers_dataset::{DataLoaderBuilder, DistributedSampler};
//! use tenflowers_dataset::CsvDataset;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let dataset = CsvDataset::new("data.csv")?;
//! // Split dataset across multiple workers
//! let sampler = DistributedSampler::new(dataset.len(), 4, 0); // 4 workers, rank 0
//!
//! let loader = DataLoaderBuilder::new(dataset)
//!     .sampler(Box::new(sampler))
//!     .batch_size(32)
//!     .build()?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Data Quality Analysis
//!
//! ```rust,ignore
//! use tenflowers_dataset::{DataQualityAnalyzer, QualityAnalysisConfig};
//! use tenflowers_dataset::CsvDataset;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let dataset = CsvDataset::new("data.csv")?;
//! // Analyze data quality
//! let analyzer = DataQualityAnalyzer::new(QualityAnalysisConfig::default());
//! let report = analyzer.analyze(&dataset)?;
//!
//! println!("Data quality score: {:.2}", report.overall_score());
//! println!("Issues found: {}", report.num_issues());
//! # Ok(())
//! # }
//! ```
//!
//! ### Caching and Prefetching
//!
//! ```rust,ignore
//! use tenflowers_dataset::{EnhancedDataLoaderBuilder, CsvDataset};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let dataset = CsvDataset::new("data.csv")?;
//! // Use enhanced data loader with smart caching
//! let loader = EnhancedDataLoaderBuilder::new(dataset)
//!     .batch_size(64)
//!     .num_workers(8)
//!     .enable_caching(true)
//!     .cache_size_mb(512)
//!     .adaptive_prefetch(true)
//!     .build()?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Custom Dataset
//!
//! ```rust,no_run
//! use tenflowers_core::{Tensor, Result};
//! use std::marker::PhantomData;
//!
//! struct MyDataset<T> {
//!     data: Vec<Vec<T>>,
//!     _phantom: PhantomData<T>,
//! }
//!
//! impl<T: Clone> MyDataset<T> {
//!     fn len(&self) -> usize {
//!         self.data.len()
//!     }
//!
//!     fn get(&self, index: usize) -> Option<&Vec<T>> {
//!         self.data.get(index)
//!     }
//! }
//! ```
//!
//! ## Architecture Overview
//!
//! The crate is organized into the following modules:
//!
//! - [`formats`]: Data format readers (CSV, image, HDF5, Arrow, Parquet)
//! - [`dataloader`]: Multi-threaded data loading with batching and sampling
//! - [`transforms`]: Data transformation and augmentation operations
//! - [`cache`]: Caching strategies for frequently accessed data
//! - [`distributed_loading`]: Distributed and sharded data loading
//! - [`data_quality`]: Data quality analysis and validation
//! - [`statistics`]: Dataset statistics computation
//! - [`visualization`]: Dataset visualization utilities
//! - [`reproducibility`]: Reproducibility and versioning support
//! - [`debug_tools`]: Profiling and debugging utilities
//!
//! ## Performance Optimization
//!
//! ### SIMD Transformations
//!
//! Many transformations use SIMD instructions for maximum performance:
//!
//! ```rust,ignore
//! use tenflowers_dataset::simd_transforms::{SimdNormalize, SimdResize};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // SIMD-accelerated normalization
//! let normalize = SimdNormalize::new(vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5]);
//! # Ok(())
//! # }
//! ```
//!
//! ### GPU Preprocessing
//!
//! ```rust,ignore
//! use tenflowers_dataset::gpu_transforms::{GpuResize, GpuNormalize};
//! use tenflowers_core::Device;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # #[cfg(feature = "gpu")]
//! # {
//! // Run transformations on GPU
//! let device = Device::gpu(0)?;
//! let resize = GpuResize::new((224, 224), &device)?;
//! # }
//! # Ok(())
//! # }
//! ```
//!
//! ### Zero-Copy Operations
//!
//! ```rust,ignore
//! use tenflowers_dataset::zero_copy::{ZeroCopyLoader, MmapDataset};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Memory-mapped dataset for large files
//! let dataset = MmapDataset::new("large_dataset.bin")?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Integration with TenfloweRS Ecosystem
//!
//! This crate integrates seamlessly with:
//! - `tenflowers-core`: Tensor operations and device management
//! - `tenflowers-neural`: Neural network training pipelines
//! - `tenflowers-autograd`: Gradient-based transformations
//! - `scirs2-core`: Scientific computing utilities
//!
//! ## Supported Data Formats
//!
//! - **CSV**: Comma-separated values with customizable delimiters
//! - **Images**: JPEG, PNG, BMP, TIFF via image folder structure
//! - **HDF5**: Hierarchical data format for scientific data
//! - **Arrow/Parquet**: Columnar data formats for analytics
//! - **JSON**: Structured JSON data
//! - **Custom**: Extensible format registry for custom formats
//!
//! ## Debugging and Profiling
//!
//! ```rust,ignore
//! use tenflowers_dataset::{DatasetDebugger, PipelineProfiler};
//! use tenflowers_dataset::CsvDataset;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let dataset = CsvDataset::new("data.csv")?;
//! // Profile data loading pipeline
//! let profiler = PipelineProfiler::new();
//! profiler.start();
//!
//! // ... load data ...
//!
//! let report = profiler.generate_report();
//! println!("Bottlenecks: {:?}", report.bottlenecks());
//! # Ok(())
//! # }
//! ```

#![warn(unsafe_code)]
#![allow(unexpected_cfgs)]
#![allow(clippy::result_large_err)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::let_and_return)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::erasing_op)]
#![allow(clippy::identity_op)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::multiple_bound_locations)]
#![allow(clippy::iter_cloned_collect)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::type_complexity)]
#![allow(clippy::borrowed_box)]
#![allow(clippy::derivable_impls)]

// Re-export core types needed by sub-modules via `crate::Result` / `crate::TensorError`
pub use tenflowers_core::{Result, TensorError};

pub mod active_learning;
pub mod adaptive_prefetch;
pub mod advanced_benchmarks;
pub mod advanced_sampling;
pub mod attention_optimized;
pub mod benchmarks;
pub mod cache;
pub mod config;
pub mod data_quality;
pub mod dataloader;
pub mod dataset_core;
pub mod debug_tools;
pub mod distributed_loading;
pub mod distributed_sharding;
pub mod distributed_streaming;
pub mod enhanced_dataloader;
pub mod error_taxonomy;
pub mod federated;
pub mod formats;
pub mod gpu_transforms;
pub mod memory_pool;
pub mod multimodal;
pub mod numa_scheduler;
pub mod online_learning;
pub mod predictive_prefetch;
#[cfg(feature = "download")]
pub mod real_datasets;
pub mod reproducibility;
pub mod schema_inference;
pub mod simd_transforms;
pub mod smart_cache;
pub mod statistics;
pub mod stream_prefetch_optimizer;
pub mod streaming_optimized;
pub mod synthetic;
pub mod throughput_benchmark;
pub mod transforms;
pub mod validation;
pub mod versioning;
pub mod visualization;
pub mod work_stealing;
pub mod zero_copy;

pub use data_quality::{
    DataQualityAnalyzer, DataQualityExt, DataQualityIssue, DataQualityMetrics,
    DriftDetectionConfig, DriftDetectionResult, DriftType, IssueCategory, IssueSeverity,
    OutlierDetectionMethod, QualityAnalysisConfig, StatisticalTest,
};
pub use dataloader::{
    BatchResult, BucketCollate, CollateFn, DataLoader, DataLoaderBuilder, DataLoaderConfig,
    DefaultCollate, DistributedSampler, ImportanceSampler, PaddingCollate, PaddingStrategy,
    RandomSampler, Sampler, SequentialSampler, StratifiedSampler,
};
pub use debug_tools::{
    Bottleneck, BottleneckCategory, ConsistencyReport, DatasetDebugger, EventType,
    PipelineProfiler, ProfileEvent, ProfileReport, ProfilerConfig, SampleInfo as DebugSampleInfo,
    Severity, StageStatistics, StageTimer,
};
pub use enhanced_dataloader::{
    EnhancedDataLoader, EnhancedDataLoaderBuilder, LoaderStats, WorkerStats,
};
pub use error_taxonomy::{
    classification, helpers as error_helpers, DatasetErrorBuilder, DatasetErrorCategory,
    DatasetErrorContext,
};
pub use formats::common::{MissingValueStrategy, NamingPattern};
pub use formats::csv::{ChunkedCsvDataset, CsvChunk, CsvDataset, CsvDatasetBuilder};
pub use formats::image::{
    image_folder_dataset_with_transform, ImageFolderConfig, ImageFolderDataset,
    ImageFolderDatasetBuilder,
};
pub use formats::registry::{
    global as format_registry, register_format_factory, FormatInfo, GlobalFormatRegistry,
};
pub use transforms::{
    AddNoise, BackgroundNoise, DatasetExt, GaussianNoise, GlobalNormalize, MinMaxScale, NoiseType,
    Normalize, PerChannelNormalize, RealTimeAudioAugmentation, RobustScaler, Transform,
    TransformedDataset,
};
// Additional format modules:
#[cfg(feature = "serialize")]
pub use formats::json::{
    JsonConfig, JsonDataset, JsonDatasetBuilder, JsonDatasetInfo, JsonLDataset,
};
// #[cfg(feature = "mmap")]
// pub use formats::mmap::{MmapDataset, MmapMemoryInfo};
pub use formats::text::{
    LabelStrategy, TextConfig, TextDataset, TextDatasetBuilder, TextDatasetInfo,
    TokenizationStrategy, TokenizedDataset, Vocabulary,
};
// #[cfg(feature = "parquet")]
// pub use formats::streaming::StreamingCheckpoint;
pub use active_learning::{
    ActiveLearningDataset, ActiveLearningSampler, DiversityStrategy, LabeledSubset,
    UncertaintyStrategy, UnlabeledSubset,
};
pub use adaptive_prefetch::{
    AdaptationStrategy, AdaptivePrefetchPolicy, AdaptivePrefetchTuner,
    PrefetchMetrics as AdaptivePrefetchMetrics, TuningDecision,
};
pub use advanced_benchmarks::{
    AdvancedBenchmarkSuite, BenchmarkConfig, BenchmarkResult, CpuStats, GpuStats, MemoryStats,
    MemoryTracker as BenchmarkMemoryTracker, SystemInfo, ThroughputStats, TimingStats,
};
pub use advanced_sampling::{
    AdvancedImportanceSampler, BalancingStrategy, ClassBalancedSampler, CurriculumScheduler,
    CurriculumStrategy, HardNegativeMiner, MiningStrategy,
};
pub use attention_optimized::{
    AttentionOptimizedConfig, AttentionOptimizedDataset, AttentionOptimizedDatasetBuilder,
    AttentionPattern, AttentionSequence, SequenceMetadata as AttentionSequenceMetadata,
};
pub use benchmarks::{BenchmarkDatasets, CifarDataset, DatasetInfo, IrisDataset, MnistDataset};
pub use cache::{
    AggregatedStats, AlertSeverity, AlertThresholds, AlertType, CacheEvent, CacheEventType,
    CacheExt, CacheStats, CacheTelemetryCollector, CacheTelemetryMetrics, CachedDataset,
    EnhancedTelemetryCollector, LruCache, MetricsSnapshot, PerformanceAlert, PerformanceBaselines,
    TelemetryConfig, ThreadSafeLruCache, WarmingStrategy,
};
#[cfg(feature = "serialize")]
pub use cache::{PersistentCache, PersistentlyCachedDataset, TensorPersistentCache};
pub use distributed_loading::{
    create_distributed_dataloader, CollectiveOpType, CommunicationManager,
    DistributedLoadingConfig, DistributedLoadingStats, DistributedMessage,
    EnhancedDistributedSampler, NodeInfo,
};
pub use distributed_sharding::{
    DatasetShardingExt, ShardConfig, ShardStatistics, ShardStrategy, ShardableDataset,
    ShardedDataset,
};
pub use distributed_streaming::{
    CheckpointState, PartitionStrategy, StreamCoordinator, StreamingConfig, StreamingShardIterator,
    StreamingShardLoader, StreamingStats, WorkerHealth, WorkerMetrics, WorkerStatus,
};
pub use federated::{
    AggregationStrategy, ClientConfig, ClientId, ClientIndexedDataset, ClientStats,
    DataDistribution, FederatedAggregator, FederatedClientDataset, FederatedDatasetExt,
    FederatedFeatureStats, FederatedPartitioner, NoiseMechanism, PartitioningStrategy,
    PrivacyConfig, PrivacyManager, PrivateStats, QualityMetrics,
};
#[cfg(feature = "parquet")]
pub use formats::arrow::{
    ArrowArrayExt, ArrowConfig, ArrowDataset, ArrowDatasetBuilder, ArrowFormatFactory,
    ArrowFormatReader, ArrowTensorView,
};
#[cfg(feature = "audio")]
pub use formats::audio::{
    AudioConfig, AudioDataset, AudioDatasetBuilder, AudioDatasetInfo, AudioInfo,
    AudioLabelStrategy, FeatureType as AudioFeatureType,
};
#[cfg(feature = "hdf5")]
pub use formats::hdf5::{HDF5Config, HDF5Dataset, HDF5DatasetBuilder, HDF5DatasetInfo};
#[cfg(feature = "parquet")]
pub use formats::parquet::{
    ParquetConfig, ParquetDataset, ParquetDatasetBuilder, ParquetDatasetInfo,
};
#[cfg(feature = "tfrecord")]
pub use formats::tfrecord::{
    Feature, FeatureInfo, FeatureType, TFRecord, TFRecordConfig, TFRecordDataset,
    TFRecordDatasetBuilder, TFRecordDatasetInfo,
};
#[cfg(feature = "webdataset")]
pub use formats::webdataset::{
    StreamingWebDataset, WebDataset, WebDatasetBuilder, WebDatasetConfig, WebDatasetSample,
};
pub use formats::zarr::{
    ZarrArrayInfo, ZarrCompressionType, ZarrConfig, ZarrDataset, ZarrDatasetBuilder, ZarrDatasetExt,
};

#[cfg(feature = "cloud")]
pub use formats::zarr::CloudBackend;
pub use gpu_transforms::{
    GpuColorJitter, GpuContext, GpuGaussianBlur, GpuGaussianNoise, GpuRandomCrop,
    GpuRandomHorizontalFlip, GpuResize, GpuRotation,
};
pub use memory_pool::{GlobalMemoryPool, MemoryPool, MemoryPoolExt, PoolStats, PooledMemory};
pub use multimodal::{
    FusionStrategy, Modality, MultimodalConfig, MultimodalDataset, MultimodalDatasetBuilder,
    MultimodalSample, MultimodalTransform, MultimodalTransformedDataset,
};
pub use numa_scheduler::{
    NumaAssignmentStats, NumaAssignmentStrategy, NumaConfig, NumaNode, NumaScheduler, NumaTopology,
    NumaWorkerAssignment,
};
pub use online_learning::{
    ADWINDetector, DriftDetectionMethod, DriftDetector, ErrorRateDetector, KSDetector,
    OnlineLearningConfig, OnlineLearningDataset, OnlineStats, PageHinkleyDetector,
};
pub use predictive_prefetch::{
    AccessPattern, AccessStats, PredictivePrefetchDataset, PredictivePrefetcher, PrefetchConfig,
};
#[cfg(feature = "download")]
pub use real_datasets::{
    AgNewsConfig, Cifar10Config, ImageNetConfig, ImdbConfig, MnistConfig, RealAgNewsBuilder,
    RealAgNewsDataset, RealCifar10Builder, RealCifar10Dataset, RealImageNetBuilder,
    RealImageNetDataset, RealImdbBuilder, RealImdbDataset, RealMnistBuilder, RealMnistDataset,
};
pub use reproducibility::{
    DatasetConfig, DeterministicDataset, DeterministicOps, DeterministicOrdering, EnvironmentInfo,
    ExperimentConfig, ExperimentTracker, OperationRecord, OrderingStrategy, ReproducibilityExt,
    SamplingConfig, SeedInfo, SeedManager, TransformConfig,
};
pub use schema_inference::{
    FieldStatistics, InferenceConfig, InferredDataType, InferredField, InferredSchema,
    SchemaInferenceEngine,
};
pub use simd_transforms::{
    BenchmarkResult as SimdBenchmarkResult, SimdBenchmark, SimdColorConvert, SimdConvolution,
    SimdElementWise, SimdHistogram, SimdHistogramTransform, SimdMatrixOps, SimdNormalize,
    SimdOperation, SimdStats,
};
pub use smart_cache::{
    AccessPatternPredictor, CacheConfig, CacheLevel, EvictionPolicy, PredictiveSmartCache,
    SmartCache, SmartCachedDataset,
};
pub use statistics::{
    AdvancedStatistics, AdvancedStatisticsExt, CorrelationAnalyzer, DatasetStatisticsComputer,
    DatasetStatisticsExt, DatasetStats, Histogram, MultivariateStatistics, PCAResult,
    StatisticsConfig,
};
pub use stream_prefetch_optimizer::{
    AccessEvent, AccessPatternAnalyzer, AccessType, PatternPrediction, PatternSignature,
    PrefetchMetrics, PrefetchOptimizerConfig, StreamPrefetchOptimizer,
};
pub use streaming_optimized::{
    AdaptiveBuffer, CompressionType, StreamingOptimizedConfig, StreamingOptimizedDataset,
    StreamingOptimizedDatasetBuilder, StreamingOptimizedIterator,
    StreamingStats as OptimizedStreamingStats,
};
pub use synthetic::{
    ContrastiveLearningDataset, DatasetGenerator, Episode, FewShotDataset, GeometricShape,
    GradientDirection, ImagePatternConfig, ImagePatternGenerator, ImagePatternType,
    MetaLearningDataset, ModernMLConfig, NoiseDistribution, SelfSupervisedDataset,
    StripeOrientation, SyntheticConfig, SyntheticDataset, SyntheticTextCorpus, TaskDataset,
    TextCorpusConfig, TextSynthesisTask, TimeSeriesPattern,
};
pub use throughput_benchmark::{
    MemoryStats as ThroughputMemoryStats, ThreadStats as ThroughputThreadStats,
    ThroughputBenchmarkConfig, ThroughputBenchmarkHarness, ThroughputBenchmarkResult,
};
pub use validation::{
    DataValidator, DatasetValidationExt, RangeConstraint, SchemaInfo, ValidationConfig,
    ValidationResult,
};
pub use versioning::{
    DatasetLineage, DatasetSizeInfo, DatasetVersionManager, LineageTree, TransformationRecord,
    VersionId, VersionMetadata, VersionedDataset,
};
pub use visualization::{
    ClassDistribution, DatasetVisualizationExt, DatasetVisualizer, DistributionInfo,
    FeatureHistogram, FeatureStats, SampleInfo, SamplePreview,
};
pub use work_stealing::WorkStealingQueue;
pub use zero_copy::{MemoryMappedDataset, TensorView, ZeroCopyDataset};

#[cfg(feature = "mmap")]
pub use zero_copy::{MemoryMappedFileDataset, MemoryMappedFileStats};

// Core dataset traits and types extracted to dataset_core module.
pub use dataset_core::{
    BatchedDataset, ConcatDataset, Dataset, DatasetSplit, DatasetSplitter, DatasetUtilsExt,
    FilteredDataset, MergeStrategy, MergedDataset, SubsetDataset, TensorDataset,
};
