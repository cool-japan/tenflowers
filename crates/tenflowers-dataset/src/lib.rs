#![warn(unsafe_code)]
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

pub mod active_learning;
pub mod advanced_benchmarks;
pub mod attention_optimized;
pub mod benchmarks;
pub mod cache;
pub mod config;
pub mod dataloader;
pub mod distributed_loading;
pub mod enhanced_dataloader;
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
pub mod simd_transforms;
pub mod smart_cache;
pub mod statistics;
pub mod stream_prefetch_optimizer;
pub mod streaming_optimized;
pub mod synthetic;
pub mod transforms;
pub mod validation;
pub mod versioning;
pub mod visualization;
pub mod work_stealing;
pub mod zero_copy;

use scirs2_core::random::rng;
use std::marker::PhantomData;
use std::sync::Arc;
use tenflowers_core::ops::slice;
use tenflowers_core::{Result, Tensor, TensorError};

pub use dataloader::{
    BatchResult, BucketCollate, CollateFn, DataLoader, DataLoaderBuilder, DataLoaderConfig,
    DefaultCollate, DistributedSampler, ImportanceSampler, PaddingCollate, PaddingStrategy,
    RandomSampler, Sampler, SequentialSampler, StratifiedSampler,
};
pub use enhanced_dataloader::{
    EnhancedDataLoader, EnhancedDataLoaderBuilder, LoaderStats, WorkerStats,
};
pub use formats::common::{MissingValueStrategy, NamingPattern};
pub use formats::csv::{ChunkedCsvDataset, CsvChunk, CsvDataset, CsvDatasetBuilder};
pub use formats::image::{
    image_folder_dataset_with_transform, ImageFolderConfig, ImageFolderDataset,
    ImageFolderDatasetBuilder,
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
#[cfg(feature = "parquet")]
pub use formats::parquet::{
    ParquetConfig, ParquetDataset, ParquetDatasetBuilder, ParquetDatasetInfo,
};
// #[cfg(feature = "parquet")]
// pub use formats::arrow::{ArrowDataset, ArrowConfig, ArrowDatasetBuilder};
pub use active_learning::{
    ActiveLearningDataset, ActiveLearningSampler, DiversityStrategy, LabeledSubset,
    UncertaintyStrategy, UnlabeledSubset,
};
pub use advanced_benchmarks::{
    AdvancedBenchmarkSuite, BenchmarkConfig, BenchmarkResult, CpuStats, GpuStats, MemoryStats,
    MemoryTracker as BenchmarkMemoryTracker, SystemInfo, ThroughputStats, TimingStats,
};
pub use attention_optimized::{
    AttentionOptimizedConfig, AttentionOptimizedDataset, AttentionOptimizedDatasetBuilder,
    AttentionPattern, AttentionSequence, SequenceMetadata as AttentionSequenceMetadata,
};
pub use benchmarks::{BenchmarkDatasets, CifarDataset, DatasetInfo, IrisDataset, MnistDataset};
pub use cache::{
    CacheExt, CacheStats, CachedDataset, LruCache, ThreadSafeLruCache, WarmingStrategy,
};
#[cfg(feature = "serialize")]
pub use cache::{PersistentCache, PersistentlyCachedDataset, TensorPersistentCache};
pub use distributed_loading::{
    create_distributed_dataloader, CollectiveOpType, CommunicationManager,
    DistributedLoadingConfig, DistributedLoadingStats, DistributedMessage,
    EnhancedDistributedSampler, NodeInfo,
};
pub use federated::{
    AggregationStrategy, ClientConfig, ClientId, ClientIndexedDataset, ClientStats,
    DataDistribution, FederatedAggregator, FederatedClientDataset, FederatedDatasetExt,
    FederatedFeatureStats, FederatedPartitioner, NoiseMechanism, PartitioningStrategy,
    PrivacyConfig, PrivacyManager, PrivateStats, QualityMetrics,
};
#[cfg(feature = "audio")]
pub use formats::audio::{
    AudioConfig, AudioDataset, AudioDatasetBuilder, AudioDatasetInfo, AudioInfo,
    AudioLabelStrategy, FeatureType as AudioFeatureType,
};
#[cfg(feature = "hdf5")]
pub use formats::hdf5::{HDF5Config, HDF5Dataset, HDF5DatasetBuilder, HDF5DatasetInfo};
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

// Dataset utilities are defined in this module
// MergedDataset and MergeStrategy are defined directly in this file

pub trait Dataset<T> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)>;
    fn batch(self, batch_size: usize) -> BatchedDataset<T, Self>
    where
        Self: Sized,
    {
        BatchedDataset {
            dataset: self,
            batch_size,
            current_index: 0,
            _phantom: PhantomData,
        }
    }
}

/// Implement Dataset for `Arc<D>` to allow shared ownership
impl<T, D: Dataset<T>> Dataset<T> for Arc<D> {
    fn len(&self) -> usize {
        (**self).len()
    }

    fn is_empty(&self) -> bool {
        (**self).is_empty()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        (**self).get(index)
    }
}

/// Extension trait providing utility methods for datasets
pub trait DatasetUtilsExt<T>: Dataset<T> {
    /// Get multiple samples by their indices
    fn get_multiple(&self, indices: &[usize]) -> Result<Vec<(Tensor<T>, Tensor<T>)>> {
        let mut samples = Vec::with_capacity(indices.len());
        for &index in indices {
            samples.push(self.get(index)?);
        }
        Ok(samples)
    }

    /// Get a range of samples from start (inclusive) to end (exclusive)
    fn get_range(&self, start: usize, end: usize) -> Result<Vec<(Tensor<T>, Tensor<T>)>> {
        if start >= end {
            return Ok(Vec::new());
        }
        if end > self.len() {
            return Err(TensorError::invalid_argument(format!(
                "End index {} out of bounds for dataset of length {}",
                end,
                self.len()
            )));
        }

        let mut samples = Vec::with_capacity(end - start);
        for i in start..end {
            samples.push(self.get(i)?);
        }
        Ok(samples)
    }

    /// Get a random sample from the dataset
    fn get_random(&self) -> Result<(Tensor<T>, Tensor<T>)> {
        if self.is_empty() {
            return Err(TensorError::invalid_argument(
                "Cannot get random sample from empty dataset".to_string(),
            ));
        }
        let mut rng = rng();
        let index = rng.gen_range(0..self.len());
        self.get(index)
    }

    /// Get multiple random samples from the dataset (with replacement)
    fn get_random_samples(&self, count: usize) -> Result<Vec<(Tensor<T>, Tensor<T>)>> {
        if self.is_empty() {
            return Err(TensorError::invalid_argument(
                "Cannot get random samples from empty dataset".to_string(),
            ));
        }

        let mut rng = rng();
        let mut samples = Vec::with_capacity(count);
        for _ in 0..count {
            let index = rng.gen_range(0..self.len());
            samples.push(self.get(index)?);
        }
        Ok(samples)
    }
}

/// Implement DatasetUtilsExt for all types that implement Dataset
impl<T, D: Dataset<T>> DatasetUtilsExt<T> for D {}

#[derive(Clone)]
pub struct TensorDataset<T> {
    features: Tensor<T>,
    #[allow(dead_code)]
    labels: Tensor<T>,
}

impl<T: Clone + Default + num_traits::Zero + Send + Sync + 'static> TensorDataset<T> {
    pub fn new(features: Tensor<T>, labels: Tensor<T>) -> Self {
        Self { features, labels }
    }
}

impl<T: Clone + Default + num_traits::Zero + Send + Sync + 'static> Dataset<T>
    for TensorDataset<T>
{
    fn len(&self) -> usize {
        self.features.shape().dims()[0]
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index,
                self.len()
            )));
        }

        // Create slice ranges for the specific index
        let mut feature_ranges = Vec::new();
        let mut label_ranges = Vec::new();

        // For the first dimension (batch dimension), slice to get single index
        feature_ranges.push(index..index + 1);
        label_ranges.push(index..index + 1);

        // For remaining dimensions, take all elements
        for i in 1..self.features.shape().rank() {
            feature_ranges.push(0..self.features.shape().dims()[i]);
        }
        for i in 1..self.labels.shape().rank() {
            label_ranges.push(0..self.labels.shape().dims()[i]);
        }

        // Slice the tensors
        let feature_slice = slice(&self.features, &feature_ranges)?;
        let label_slice = slice(&self.labels, &label_ranges)?;

        // Squeeze the first dimension (remove batch dimension of size 1)
        let feature_squeezed = squeeze_first_dim(&feature_slice)?;
        let label_squeezed = squeeze_first_dim(&label_slice)?;

        Ok((feature_squeezed, label_squeezed))
    }
}

/// Helper function to squeeze the first dimension of a tensor
fn squeeze_first_dim<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + num_traits::Zero + Send + Sync + 'static,
{
    let shape = tensor.shape();
    if shape.rank() == 0 {
        return Ok(tensor.clone());
    }

    if shape.dims()[0] != 1 {
        return Err(TensorError::invalid_argument(format!(
            "Cannot squeeze dimension of size {}",
            shape.dims()[0]
        )));
    }

    let new_shape: Vec<usize> = shape.dims()[1..].to_vec();
    tenflowers_core::ops::reshape(tensor, &new_shape)
}

pub struct BatchedDataset<T, D: Dataset<T>> {
    dataset: D,
    batch_size: usize,
    current_index: usize,
    _phantom: PhantomData<T>,
}

impl<T, D: Dataset<T>> Iterator for BatchedDataset<T, D> {
    type Item = Vec<(Tensor<T>, Tensor<T>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.dataset.len() {
            return None;
        }

        let mut batch = Vec::new();
        let end_index = (self.current_index + self.batch_size).min(self.dataset.len());

        for i in self.current_index..end_index {
            match self.dataset.get(i) {
                Ok(sample) => batch.push(sample),
                Err(_) => break, // Stop on error, return partial batch if any
            }
        }

        self.current_index = end_index;

        if batch.is_empty() {
            None
        } else {
            Some(batch)
        }
    }
}

/// Dataset concatenation - combines multiple datasets into one
pub struct ConcatDataset<T, D: Dataset<T>> {
    datasets: Vec<D>,
    cumulative_lengths: Vec<usize>,
    total_length: usize,
    _phantom: PhantomData<T>,
}

impl<T, D: Dataset<T>> ConcatDataset<T, D> {
    pub fn new(datasets: Vec<D>) -> Self {
        let mut cumulative_lengths = Vec::with_capacity(datasets.len());
        let mut total_length = 0;

        for dataset in &datasets {
            total_length += dataset.len();
            cumulative_lengths.push(total_length);
        }

        Self {
            datasets,
            cumulative_lengths,
            total_length,
            _phantom: PhantomData,
        }
    }

    /// Find which dataset and local index for a global index
    fn find_dataset_and_index(&self, global_index: usize) -> Result<(usize, usize)> {
        for (dataset_idx, &cumulative_len) in self.cumulative_lengths.iter().enumerate() {
            if global_index < cumulative_len {
                let local_index = if dataset_idx == 0 {
                    global_index
                } else {
                    global_index - self.cumulative_lengths[dataset_idx - 1]
                };
                return Ok((dataset_idx, local_index));
            }
        }
        Err(TensorError::invalid_argument(format!(
            "Index {} out of bounds for dataset of total length {}",
            global_index, self.total_length
        )))
    }
}

impl<T, D: Dataset<T>> Dataset<T> for ConcatDataset<T, D> {
    fn len(&self) -> usize {
        self.total_length
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.total_length {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index, self.total_length
            )));
        }

        let (dataset_idx, local_index) = self.find_dataset_and_index(index)?;
        self.datasets[dataset_idx].get(local_index)
    }
}

/// Dataset filtering - creates a view of a dataset with only indices that match a predicate
pub struct FilteredDataset<T, D: Dataset<T>, F: Fn(&(Tensor<T>, Tensor<T>)) -> bool> {
    dataset: D,
    valid_indices: Vec<usize>,
    _phantom: PhantomData<(T, F)>,
}

impl<T, D: Dataset<T>, F: Fn(&(Tensor<T>, Tensor<T>)) -> bool> FilteredDataset<T, D, F> {
    pub fn new(dataset: D, predicate: F) -> Result<Self> {
        let mut valid_indices = Vec::new();

        for i in 0..dataset.len() {
            match dataset.get(i) {
                Ok(sample) => {
                    if predicate(&sample) {
                        valid_indices.push(i);
                    }
                }
                Err(_) => continue, // Skip invalid samples
            }
        }

        Ok(Self {
            dataset,
            valid_indices,
            _phantom: PhantomData,
        })
    }
}

impl<T, D: Dataset<T>, F: Fn(&(Tensor<T>, Tensor<T>)) -> bool> Dataset<T>
    for FilteredDataset<T, D, F>
{
    fn len(&self) -> usize {
        self.valid_indices.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.valid_indices.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for filtered dataset of length {}",
                index,
                self.valid_indices.len()
            )));
        }

        let actual_index = self.valid_indices[index];
        self.dataset.get(actual_index)
    }
}

/// Dataset splitting - splits a dataset into train/validation/test sets
pub struct DatasetSplit<T, D: Dataset<T>> {
    pub train: SubsetDataset<T, Arc<D>>,
    pub validation: Option<SubsetDataset<T, Arc<D>>>,
    pub test: Option<SubsetDataset<T, Arc<D>>>,
}

/// Subset dataset - creates a view of a dataset with only specified indices
pub struct SubsetDataset<T, D: Dataset<T>> {
    dataset: D,
    indices: Vec<usize>,
    _phantom: PhantomData<T>,
}

impl<T, D: Dataset<T>> SubsetDataset<T, D> {
    pub fn new(dataset: D, indices: Vec<usize>) -> Self {
        Self {
            dataset,
            indices,
            _phantom: PhantomData,
        }
    }
}

impl<T, D: Dataset<T>> Dataset<T> for SubsetDataset<T, D> {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.indices.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for subset dataset of length {}",
                index,
                self.indices.len()
            )));
        }

        let actual_index = self.indices[index];
        self.dataset.get(actual_index)
    }
}

/// Dataset merging - combines multiple datasets with different modalities
pub struct MergedDataset<T, D1: Dataset<T>, D2: Dataset<T>> {
    dataset1: D1,
    dataset2: D2,
    merge_strategy: MergeStrategy,
    _phantom: PhantomData<T>,
}

/// Strategy for merging datasets
#[derive(Debug, Clone)]
pub enum MergeStrategy {
    /// Concatenate features horizontally
    FeatureConcatenation,
    /// Average features element-wise
    FeatureAverage,
    /// Use features from first dataset, labels from second
    FeatureFromFirst,
    /// Use features from second dataset, labels from first
    FeatureFromSecond,
    /// Custom merge function
    Custom,
}

impl<T, D1: Dataset<T>, D2: Dataset<T>> MergedDataset<T, D1, D2> {
    /// Create a new merged dataset with feature concatenation
    pub fn new_concatenated(dataset1: D1, dataset2: D2) -> Result<Self> {
        if dataset1.len() != dataset2.len() {
            return Err(TensorError::invalid_argument(format!(
                "Dataset lengths must match: {} vs {}",
                dataset1.len(),
                dataset2.len()
            )));
        }

        Ok(Self {
            dataset1,
            dataset2,
            merge_strategy: MergeStrategy::FeatureConcatenation,
            _phantom: PhantomData,
        })
    }

    /// Create a new merged dataset with feature averaging
    pub fn new_averaged(dataset1: D1, dataset2: D2) -> Result<Self> {
        if dataset1.len() != dataset2.len() {
            return Err(TensorError::invalid_argument(format!(
                "Dataset lengths must match: {} vs {}",
                dataset1.len(),
                dataset2.len()
            )));
        }

        Ok(Self {
            dataset1,
            dataset2,
            merge_strategy: MergeStrategy::FeatureAverage,
            _phantom: PhantomData,
        })
    }

    /// Create a new merged dataset using features from first dataset and labels from second
    pub fn new_features_from_first(dataset1: D1, dataset2: D2) -> Result<Self> {
        if dataset1.len() != dataset2.len() {
            return Err(TensorError::invalid_argument(format!(
                "Dataset lengths must match: {} vs {}",
                dataset1.len(),
                dataset2.len()
            )));
        }

        Ok(Self {
            dataset1,
            dataset2,
            merge_strategy: MergeStrategy::FeatureFromFirst,
            _phantom: PhantomData,
        })
    }

    /// Create a new merged dataset using features from second dataset and labels from first
    pub fn new_features_from_second(dataset1: D1, dataset2: D2) -> Result<Self> {
        if dataset1.len() != dataset2.len() {
            return Err(TensorError::invalid_argument(format!(
                "Dataset lengths must match: {} vs {}",
                dataset1.len(),
                dataset2.len()
            )));
        }

        Ok(Self {
            dataset1,
            dataset2,
            merge_strategy: MergeStrategy::FeatureFromSecond,
            _phantom: PhantomData,
        })
    }

    /// Merge two tensors based on the merge strategy
    fn merge_tensors(&self, tensor1: &Tensor<T>, tensor2: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: Clone + Default + num_traits::Zero + num_traits::Float + Send + Sync + 'static,
    {
        match self.merge_strategy {
            MergeStrategy::FeatureConcatenation => {
                // Concatenate tensors along the feature dimension
                let data1 = tensor1.as_slice().ok_or_else(|| {
                    TensorError::invalid_argument(
                        "Cannot access tensor data (GPU tensor not supported)".to_string(),
                    )
                })?;
                let data2 = tensor2.as_slice().ok_or_else(|| {
                    TensorError::invalid_argument(
                        "Cannot access tensor data (GPU tensor not supported)".to_string(),
                    )
                })?;
                let mut merged_data = Vec::new();
                merged_data.extend_from_slice(data1);
                merged_data.extend_from_slice(data2);

                let new_shape = vec![data1.len() + data2.len()];
                Tensor::from_vec(merged_data, &new_shape)
            }
            MergeStrategy::FeatureAverage => {
                // Average tensors element-wise
                let data1 = tensor1.as_slice().ok_or_else(|| {
                    TensorError::invalid_argument(
                        "Cannot access tensor data (GPU tensor not supported)".to_string(),
                    )
                })?;
                let data2 = tensor2.as_slice().ok_or_else(|| {
                    TensorError::invalid_argument(
                        "Cannot access tensor data (GPU tensor not supported)".to_string(),
                    )
                })?;

                if data1.len() != data2.len() {
                    return Err(TensorError::invalid_argument(
                        "Cannot average tensors of different sizes".to_string(),
                    ));
                }

                let mut averaged_data = Vec::new();
                for (v1, v2) in data1.iter().zip(data2.iter()) {
                    let avg = (*v1 + *v2) / T::from(2.0).unwrap();
                    averaged_data.push(avg);
                }

                Tensor::from_vec(averaged_data, tensor1.shape().dims())
            }
            MergeStrategy::FeatureFromFirst => Ok(tensor1.clone()),
            MergeStrategy::FeatureFromSecond => Ok(tensor2.clone()),
            MergeStrategy::Custom => {
                // For custom merge, just return first tensor for now
                // This could be extended to accept custom merge functions
                Ok(tensor1.clone())
            }
        }
    }
}

impl<T, D1: Dataset<T>, D2: Dataset<T>> Dataset<T> for MergedDataset<T, D1, D2>
where
    T: Clone + Default + num_traits::Zero + num_traits::Float + Send + Sync + 'static,
{
    fn len(&self) -> usize {
        self.dataset1.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.dataset1.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for merged dataset of length {}",
                index,
                self.dataset1.len()
            )));
        }

        let (features1, labels1) = self.dataset1.get(index)?;
        let (features2, labels2) = self.dataset2.get(index)?;

        let merged_features = self.merge_tensors(&features1, &features2)?;

        // For labels, use the strategy to determine which label to use
        let merged_labels = match self.merge_strategy {
            MergeStrategy::FeatureFromFirst => labels1,
            MergeStrategy::FeatureFromSecond => labels2,
            _ => labels1, // Default to first dataset's labels
        };

        Ok((merged_features, merged_labels))
    }
}

/// Dataset splitting utilities
pub struct DatasetSplitter;

impl DatasetSplitter {
    /// Split dataset into train/validation/test sets with given ratios
    pub fn split<T, D: Dataset<T>>(
        dataset: D,
        train_ratio: f64,
        val_ratio: Option<f64>,
        test_ratio: Option<f64>,
        shuffle: bool,
    ) -> Result<DatasetSplit<T, D>> {
        let total_len = dataset.len();
        if total_len == 0 {
            return Err(TensorError::invalid_argument(
                "Cannot split empty dataset".to_string(),
            ));
        }

        // Validate ratios
        let val_ratio = val_ratio.unwrap_or(0.0);
        let test_ratio = test_ratio.unwrap_or(0.0);

        if train_ratio + val_ratio + test_ratio > 1.0 {
            return Err(TensorError::invalid_argument(
                "Sum of ratios cannot exceed 1.0".to_string(),
            ));
        }

        // Create indices
        let mut indices: Vec<usize> = (0..total_len).collect();

        // Shuffle if requested
        if shuffle {
            use scirs2_core::random::rand_prelude::*;
            let mut rng = rng();
            indices.shuffle(&mut rng);
        }

        // Calculate split points
        let train_end = (total_len as f64 * train_ratio) as usize;
        let val_end = train_end + (total_len as f64 * val_ratio) as usize;
        let test_end = val_end + (total_len as f64 * test_ratio) as usize;

        // Create subset datasets using Arc for sharing
        let dataset_arc = Arc::new(dataset);
        let train_indices = indices[0..train_end].to_vec();
        let train = SubsetDataset::new(dataset_arc.clone(), train_indices);

        let validation = if val_ratio > 0.0 {
            let val_indices = indices[train_end..val_end].to_vec();
            Some(SubsetDataset::new(dataset_arc.clone(), val_indices))
        } else {
            None
        };

        let test = if test_ratio > 0.0 {
            let test_indices = indices[val_end..test_end].to_vec();
            Some(SubsetDataset::new(dataset_arc.clone(), test_indices))
        } else {
            None
        };

        Ok(DatasetSplit {
            train,
            validation,
            test,
        })
    }

    /// Split dataset into k-folds for cross-validation
    #[allow(clippy::type_complexity)]
    pub fn k_fold<T, D: Dataset<T>>(
        dataset: D,
        k: usize,
        shuffle: bool,
    ) -> Result<Vec<(SubsetDataset<T, Arc<D>>, SubsetDataset<T, Arc<D>>)>> {
        if k <= 1 {
            return Err(TensorError::invalid_argument(
                "K must be greater than 1".to_string(),
            ));
        }

        let total_len = dataset.len();
        if total_len == 0 {
            return Err(TensorError::invalid_argument(
                "Cannot split empty dataset".to_string(),
            ));
        }

        let mut indices: Vec<usize> = (0..total_len).collect();

        if shuffle {
            use scirs2_core::random::rand_prelude::*;
            let mut rng = rng();
            indices.shuffle(&mut rng);
        }

        let fold_size = total_len / k;
        let mut folds = Vec::new();
        let dataset_arc = Arc::new(dataset);

        for i in 0..k {
            let start = i * fold_size;
            let end = if i == k - 1 {
                total_len
            } else {
                (i + 1) * fold_size
            };

            let val_indices = indices[start..end].to_vec();
            let train_indices: Vec<usize> = indices[0..start]
                .iter()
                .chain(indices[end..].iter())
                .cloned()
                .collect();

            let train_dataset = SubsetDataset::new(dataset_arc.clone(), train_indices);
            let val_dataset = SubsetDataset::new(dataset_arc.clone(), val_indices);

            folds.push((train_dataset, val_dataset));
        }

        Ok(folds)
    }

    /// Stratified split - maintains class distribution across splits
    pub fn stratified_split<T, D: Dataset<T>>(
        dataset: D,
        train_ratio: f64,
        val_ratio: Option<f64>,
        extract_class: fn(&(Tensor<T>, Tensor<T>)) -> usize,
    ) -> Result<(Vec<usize>, Vec<usize>)> {
        let total_len = dataset.len();
        if total_len == 0 {
            return Err(TensorError::invalid_argument(
                "Cannot split empty dataset".to_string(),
            ));
        }

        // Group indices by class
        let mut class_indices: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();

        for i in 0..total_len {
            if let Ok(sample) = dataset.get(i) {
                let class = extract_class(&sample);
                class_indices.entry(class).or_default().push(i);
            }
        }

        let mut train_indices = Vec::new();
        let mut val_indices = Vec::new();

        // Split each class proportionally
        for (_, mut indices) in class_indices {
            // Shuffle class indices
            use scirs2_core::random::rand_prelude::*;
            let mut rng = rng();
            indices.shuffle(&mut rng);

            let class_len = indices.len();
            let train_end = (class_len as f64 * train_ratio) as usize;

            train_indices.extend(indices[0..train_end].iter());

            if let Some(val_ratio) = val_ratio {
                let val_end = train_end + (class_len as f64 * val_ratio) as usize;
                val_indices.extend(indices[train_end..val_end].iter());
            }
        }

        Ok((train_indices, val_indices))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    #[test]
    fn test_tensor_dataset_creation() {
        let features =
            Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let labels = Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();

        let dataset = TensorDataset::new(features, labels);
        assert_eq!(dataset.len(), 3);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_tensor_dataset_get() {
        let features = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2], // 3 samples, 2 features each
        )
        .unwrap();
        let labels = Tensor::<f32>::from_vec(
            vec![10.0, 20.0, 30.0],
            &[3], // 3 labels
        )
        .unwrap();

        let dataset = TensorDataset::new(features, labels);

        // Test getting first sample
        let (feat, label) = dataset.get(0).unwrap();
        assert_eq!(feat.shape().dims(), &[2]); // Should be squeezed from [1, 2] to [2]
        assert_eq!(label.shape().dims(), &[] as &[usize]); // Should be squeezed from [1] to scalar

        // Test getting second sample
        let (feat2, label2) = dataset.get(1).unwrap();
        assert_eq!(feat2.shape().dims(), &[2]);
        assert_eq!(label2.shape().dims(), &[] as &[usize]);

        // Test out of bounds
        assert!(dataset.get(3).is_err());
    }

    #[test]
    fn test_batched_dataset() {
        let features = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[4, 2], // 4 samples, 2 features each
        )
        .unwrap();
        let labels = Tensor::<f32>::from_vec(
            vec![10.0, 20.0, 30.0, 40.0],
            &[4], // 4 labels
        )
        .unwrap();

        let dataset = TensorDataset::new(features, labels);
        let mut batched = dataset.batch(2);

        // First batch should have 2 samples
        let batch1 = batched.next().unwrap();
        assert_eq!(batch1.len(), 2);

        // Second batch should have 2 samples
        let batch2 = batched.next().unwrap();
        assert_eq!(batch2.len(), 2);

        // No more batches
        assert!(batched.next().is_none());
    }

    #[test]
    fn test_batched_dataset_partial_batch() {
        let features = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2], // 3 samples, 2 features each
        )
        .unwrap();
        let labels = Tensor::<f32>::from_vec(
            vec![10.0, 20.0, 30.0],
            &[3], // 3 labels
        )
        .unwrap();

        let dataset = TensorDataset::new(features, labels);
        let mut batched = dataset.batch(2);

        // First batch should have 2 samples
        let batch1 = batched.next().unwrap();
        assert_eq!(batch1.len(), 2);

        // Second batch should have 1 sample (partial)
        let batch2 = batched.next().unwrap();
        assert_eq!(batch2.len(), 1);

        // No more batches
        assert!(batched.next().is_none());
    }

    #[test]
    fn test_merged_dataset_concatenation() {
        // Create two datasets
        let features1 = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            &[2, 2], // 2 samples, 2 features each
        )
        .unwrap();
        let labels1 = Tensor::<f32>::from_vec(vec![10.0, 20.0], &[2]).unwrap();
        let dataset1 = TensorDataset::new(features1, labels1);

        let features2 = Tensor::<f32>::from_vec(
            vec![5.0, 6.0, 7.0, 8.0],
            &[2, 2], // 2 samples, 2 features each
        )
        .unwrap();
        let labels2 = Tensor::<f32>::from_vec(vec![30.0, 40.0], &[2]).unwrap();
        let dataset2 = TensorDataset::new(features2, labels2);

        // Create merged dataset
        let merged = MergedDataset::new_concatenated(dataset1, dataset2).unwrap();

        assert_eq!(merged.len(), 2);

        // Test getting first sample
        let (features, labels) = merged.get(0).unwrap();
        assert_eq!(features.shape().dims(), &[4]); // 2 + 2 features concatenated
        assert_eq!(labels.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn test_merged_dataset_averaging() {
        // Create two datasets with same feature size
        let features1 = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            &[2, 2], // 2 samples, 2 features each
        )
        .unwrap();
        let labels1 = Tensor::<f32>::from_vec(vec![10.0, 20.0], &[2]).unwrap();
        let dataset1 = TensorDataset::new(features1, labels1);

        let features2 = Tensor::<f32>::from_vec(
            vec![5.0, 6.0, 7.0, 8.0],
            &[2, 2], // 2 samples, 2 features each
        )
        .unwrap();
        let labels2 = Tensor::<f32>::from_vec(vec![30.0, 40.0], &[2]).unwrap();
        let dataset2 = TensorDataset::new(features2, labels2);

        // Create merged dataset with averaging
        let merged = MergedDataset::new_averaged(dataset1, dataset2).unwrap();

        assert_eq!(merged.len(), 2);

        // Test getting first sample - should be averaged
        let (features, _) = merged.get(0).unwrap();
        assert_eq!(features.shape().dims(), &[2]); // Same feature size
                                                   // First sample should be (1+5)/2=3, (2+6)/2=4
        let data = features.as_slice().unwrap();
        assert!((data[0] - 3.0).abs() < 1e-6);
        assert!((data[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_merged_dataset_mismatched_lengths() {
        // Create two datasets with different lengths
        let features1 = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            &[2, 2], // 2 samples
        )
        .unwrap();
        let labels1 = Tensor::<f32>::from_vec(vec![10.0, 20.0], &[2]).unwrap();
        let dataset1 = TensorDataset::new(features1, labels1);

        let features2 = Tensor::<f32>::from_vec(
            vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[3, 2], // 3 samples
        )
        .unwrap();
        let labels2 = Tensor::<f32>::from_vec(vec![30.0, 40.0, 50.0], &[3]).unwrap();
        let dataset2 = TensorDataset::new(features2, labels2);

        // Should fail with mismatched lengths
        assert!(MergedDataset::new_concatenated(dataset1, dataset2).is_err());
    }
}
