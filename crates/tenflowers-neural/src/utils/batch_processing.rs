//! Batch Processing Utilities
//!
//! This module provides efficient batch processing utilities for neural network training,
//! including batching strategies, collation functions, and data sampling methods.

use std::collections::HashMap;
use tenflowers_core::{Device, Result, Tensor, TensorError};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Batch sampling strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum SamplingStrategy {
    /// Sequential sampling (in order)
    Sequential,
    /// Random sampling (with replacement)
    Random,
    /// Shuffle once at the beginning
    Shuffle,
    /// Stratified sampling (balanced classes)
    Stratified,
    /// Weighted sampling based on importance
    Weighted,
}

/// Padding strategy for variable-length sequences
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum PaddingStrategy {
    /// Pad to the longest sequence in the batch
    LongestInBatch,
    /// Pad to a fixed maximum length
    FixedLength,
    /// Pad to the nearest multiple of a value
    NearestMultiple,
    /// No padding (all sequences must be same length)
    NoPadding,
}

/// Collation strategy for combining samples into batches
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum CollationStrategy {
    /// Stack tensors along batch dimension
    Stack,
    /// Concatenate tensors
    Concatenate,
    /// Pad and stack (for variable-length sequences)
    PadAndStack,
    /// Custom collation (user-defined)
    Custom,
}

/// Batch configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BatchConfig {
    /// Batch size
    pub batch_size: usize,
    /// Whether to drop the last incomplete batch
    pub drop_last: bool,
    /// Sampling strategy
    pub sampling_strategy: SamplingStrategy,
    /// Padding strategy (for sequences)
    pub padding_strategy: PaddingStrategy,
    /// Collation strategy
    pub collation_strategy: CollationStrategy,
    /// Maximum sequence length (for padding)
    pub max_sequence_length: Option<usize>,
    /// Padding value
    pub padding_value: f32,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            drop_last: false,
            sampling_strategy: SamplingStrategy::Sequential,
            padding_strategy: PaddingStrategy::LongestInBatch,
            collation_strategy: CollationStrategy::Stack,
            max_sequence_length: None,
            padding_value: 0.0,
            seed: None,
        }
    }
}

impl BatchConfig {
    /// Create new batch configuration
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            ..Default::default()
        }
    }

    /// Set whether to drop last incomplete batch
    pub fn with_drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Set sampling strategy
    pub fn with_sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.sampling_strategy = strategy;
        self
    }

    /// Set padding strategy
    pub fn with_padding_strategy(mut self, strategy: PaddingStrategy) -> Self {
        self.padding_strategy = strategy;
        self
    }

    /// Set collation strategy
    pub fn with_collation_strategy(mut self, strategy: CollationStrategy) -> Self {
        self.collation_strategy = strategy;
        self
    }

    /// Set maximum sequence length
    pub fn with_max_sequence_length(mut self, max_len: usize) -> Self {
        self.max_sequence_length = Some(max_len);
        self
    }

    /// Set padding value
    pub fn with_padding_value(mut self, value: f32) -> Self {
        self.padding_value = value;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Batch sampler for generating batch indices
pub struct BatchSampler {
    dataset_size: usize,
    config: BatchConfig,
    current_index: usize,
    indices: Vec<usize>,
}

impl BatchSampler {
    /// Create new batch sampler
    pub fn new(dataset_size: usize, config: BatchConfig) -> Self {
        let indices = match config.sampling_strategy {
            SamplingStrategy::Sequential => (0..dataset_size).collect(),
            SamplingStrategy::Shuffle => {
                // TODO: Implement proper shuffling with seed
                (0..dataset_size).collect()
            }
            _ => (0..dataset_size).collect(),
        };

        Self {
            dataset_size,
            config,
            current_index: 0,
            indices,
        }
    }

    /// Get next batch of indices
    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        if self.current_index >= self.dataset_size {
            return None;
        }

        let end_index = (self.current_index + self.config.batch_size).min(self.dataset_size);
        let batch_indices: Vec<usize> = self.indices[self.current_index..end_index].to_vec();

        self.current_index = end_index;

        // Check if we should drop the last incomplete batch
        if self.config.drop_last && batch_indices.len() < self.config.batch_size {
            None
        } else {
            Some(batch_indices)
        }
    }

    /// Reset the sampler to the beginning
    pub fn reset(&mut self) {
        self.current_index = 0;

        // Reshuffle if needed
        if self.config.sampling_strategy == SamplingStrategy::Shuffle {
            // TODO: Implement proper shuffling with seed
        }
    }

    /// Get total number of batches
    pub fn num_batches(&self) -> usize {
        let total = (self.dataset_size + self.config.batch_size - 1) / self.config.batch_size;
        if self.config.drop_last && self.dataset_size % self.config.batch_size != 0 {
            total - 1
        } else {
            total
        }
    }

    /// Get current batch index
    pub fn current_batch_index(&self) -> usize {
        self.current_index / self.config.batch_size
    }
}

/// Collation function for combining samples into batches
pub struct Collator {
    config: BatchConfig,
}

impl Collator {
    /// Create new collator
    pub fn new(config: BatchConfig) -> Self {
        Self { config }
    }

    /// Collate samples into a batch
    pub fn collate<T>(&self, samples: &[Tensor<T>]) -> Result<Tensor<T>>
    where
        T: Clone + Default,
    {
        if samples.is_empty() {
            return Err(TensorError::invalid_shape_simple(
                "Cannot collate empty batch".to_string(),
            ));
        }

        match self.config.collation_strategy {
            CollationStrategy::Stack => self.stack_samples(samples),
            CollationStrategy::PadAndStack => self.pad_and_stack_samples(samples),
            _ => {
                // Placeholder for other strategies
                self.stack_samples(samples)
            }
        }
    }

    /// Stack samples along batch dimension
    fn stack_samples<T>(&self, samples: &[Tensor<T>]) -> Result<Tensor<T>>
    where
        T: Clone + Default,
    {
        // Placeholder implementation
        // Real implementation would properly stack tensors
        Ok(samples[0].clone())
    }

    /// Pad and stack samples (for variable-length sequences)
    fn pad_and_stack_samples<T>(&self, samples: &[Tensor<T>]) -> Result<Tensor<T>>
    where
        T: Clone + Default,
    {
        // Placeholder implementation
        // Real implementation would:
        // 1. Find max length based on padding strategy
        // 2. Pad each sample to max length
        // 3. Stack padded samples
        Ok(samples[0].clone())
    }

    /// Get padding length based on strategy
    fn get_padding_length(&self, sample_lengths: &[usize]) -> usize {
        match self.config.padding_strategy {
            PaddingStrategy::LongestInBatch => *sample_lengths.iter().max().unwrap_or(&0),
            PaddingStrategy::FixedLength => self.config.max_sequence_length.unwrap_or(512),
            PaddingStrategy::NearestMultiple => {
                let max_len = *sample_lengths.iter().max().unwrap_or(&0);
                let multiple = self.config.max_sequence_length.unwrap_or(8);
                ((max_len + multiple - 1) / multiple) * multiple
            }
            PaddingStrategy::NoPadding => sample_lengths[0],
        }
    }
}

/// Batch statistics for monitoring
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct BatchStatistics {
    /// Total number of batches processed
    pub total_batches: usize,
    /// Total number of samples processed
    pub total_samples: usize,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Min batch size seen
    pub min_batch_size: usize,
    /// Max batch size seen
    pub max_batch_size: usize,
    /// Average padding ratio (for sequences)
    pub avg_padding_ratio: f64,
}

impl BatchStatistics {
    /// Create new batch statistics
    pub fn new() -> Self {
        Self {
            total_batches: 0,
            total_samples: 0,
            avg_batch_size: 0.0,
            min_batch_size: usize::MAX,
            max_batch_size: 0,
            avg_padding_ratio: 0.0,
        }
    }

    /// Record a batch
    pub fn record_batch(&mut self, batch_size: usize, padding_ratio: f64) {
        self.total_batches += 1;
        self.total_samples += batch_size;
        self.min_batch_size = self.min_batch_size.min(batch_size);
        self.max_batch_size = self.max_batch_size.max(batch_size);

        // Update running averages
        let n = self.total_batches as f64;
        self.avg_batch_size = (self.avg_batch_size * (n - 1.0) + batch_size as f64) / n;
        self.avg_padding_ratio = (self.avg_padding_ratio * (n - 1.0) + padding_ratio) / n;
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Get efficiency (ratio of real data to padded data)
    pub fn efficiency(&self) -> f64 {
        1.0 - self.avg_padding_ratio
    }
}

impl Default for BatchStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Utilities for batch processing
pub mod batch_utils {
    use super::*;

    /// Calculate optimal batch size for given memory constraints
    pub fn calculate_optimal_batch_size(
        sample_memory_bytes: usize,
        available_memory_bytes: usize,
        safety_factor: f64,
    ) -> usize {
        let usable_memory = (available_memory_bytes as f64 * safety_factor) as usize;
        (usable_memory / sample_memory_bytes).max(1)
    }

    /// Calculate number of batches for a dataset
    pub fn calculate_num_batches(dataset_size: usize, batch_size: usize, drop_last: bool) -> usize {
        let total = (dataset_size + batch_size - 1) / batch_size;
        if drop_last && dataset_size % batch_size != 0 {
            total - 1
        } else {
            total
        }
    }

    /// Calculate padding overhead
    pub fn calculate_padding_overhead(original_lengths: &[usize], padded_length: usize) -> f64 {
        let original_total: usize = original_lengths.iter().sum();
        let padded_total = original_lengths.len() * padded_length;

        if padded_total == 0 {
            0.0
        } else {
            1.0 - (original_total as f64 / padded_total as f64)
        }
    }

    /// Find optimal padding length to minimize overhead
    pub fn find_optimal_padding_length(lengths: &[usize], multiple: usize) -> usize {
        let max_len = lengths.iter().max().copied().unwrap_or(0);
        ((max_len + multiple - 1) / multiple) * multiple
    }

    /// Group samples by similar lengths for efficient batching
    pub fn group_by_length(lengths: Vec<usize>, num_groups: usize) -> Vec<Vec<usize>> {
        if lengths.is_empty() || num_groups == 0 {
            return vec![];
        }

        let mut indexed_lengths: Vec<_> = lengths.into_iter().enumerate().collect();
        indexed_lengths.sort_by_key(|(_, len)| *len);

        let group_size = (indexed_lengths.len() + num_groups - 1) / num_groups;
        let mut groups = vec![Vec::new(); num_groups];

        for (group_idx, chunk) in indexed_lengths.chunks(group_size).enumerate() {
            groups[group_idx] = chunk.iter().map(|(idx, _)| *idx).collect();
        }

        groups.into_iter().filter(|g| !g.is_empty()).collect()
    }

    /// Calculate memory efficiency of batching strategy
    pub fn calculate_memory_efficiency(
        batch_size: usize,
        avg_sequence_length: usize,
        max_sequence_length: usize,
    ) -> f64 {
        let used = batch_size * avg_sequence_length;
        let allocated = batch_size * max_sequence_length;

        if allocated == 0 {
            0.0
        } else {
            used as f64 / allocated as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_strategy_variants() {
        let strategies = [
            SamplingStrategy::Sequential,
            SamplingStrategy::Random,
            SamplingStrategy::Shuffle,
            SamplingStrategy::Stratified,
            SamplingStrategy::Weighted,
        ];

        assert_eq!(strategies.len(), 5);
    }

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.batch_size, 32);
        assert!(!config.drop_last);
        assert_eq!(config.sampling_strategy, SamplingStrategy::Sequential);
    }

    #[test]
    fn test_batch_config_builder() {
        let config = BatchConfig::new(64)
            .with_drop_last(true)
            .with_padding_value(1.0)
            .with_max_sequence_length(128)
            .with_seed(42);

        assert_eq!(config.batch_size, 64);
        assert!(config.drop_last);
        assert_eq!(config.padding_value, 1.0);
        assert_eq!(config.max_sequence_length, Some(128));
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_batch_sampler_creation() {
        let config = BatchConfig::new(10);
        let sampler = BatchSampler::new(100, config);

        assert_eq!(sampler.dataset_size, 100);
        assert_eq!(sampler.num_batches(), 10);
    }

    #[test]
    fn test_batch_sampler_next_batch() {
        let config = BatchConfig::new(10);
        let mut sampler = BatchSampler::new(25, config);

        let batch1 = sampler.next_batch();
        assert!(batch1.is_some());
        assert_eq!(batch1.unwrap().len(), 10);

        let batch2 = sampler.next_batch();
        assert!(batch2.is_some());
        assert_eq!(batch2.unwrap().len(), 10);

        let batch3 = sampler.next_batch();
        assert!(batch3.is_some());
        assert_eq!(batch3.unwrap().len(), 5); // Last incomplete batch
    }

    #[test]
    fn test_batch_sampler_drop_last() {
        let config = BatchConfig::new(10).with_drop_last(true);
        let mut sampler = BatchSampler::new(25, config);

        sampler.next_batch();
        sampler.next_batch();
        let batch3 = sampler.next_batch();

        assert!(batch3.is_none()); // Last batch should be dropped
    }

    #[test]
    fn test_batch_sampler_num_batches() {
        let config = BatchConfig::new(10);
        let sampler = BatchSampler::new(25, config);
        assert_eq!(sampler.num_batches(), 3);

        let config_drop = BatchConfig::new(10).with_drop_last(true);
        let sampler_drop = BatchSampler::new(25, config_drop);
        assert_eq!(sampler_drop.num_batches(), 2);
    }

    #[test]
    fn test_batch_sampler_reset() {
        let config = BatchConfig::new(10);
        let mut sampler = BatchSampler::new(25, config);

        sampler.next_batch();
        sampler.next_batch();
        assert_eq!(sampler.current_batch_index(), 2);

        sampler.reset();
        assert_eq!(sampler.current_batch_index(), 0);
    }

    #[test]
    fn test_collator_creation() {
        let config = BatchConfig::new(32);
        let collator = Collator::new(config);

        // Just verify it was created successfully
        assert_eq!(collator.config.batch_size, 32);
    }

    #[test]
    fn test_batch_statistics_creation() {
        let stats = BatchStatistics::new();
        assert_eq!(stats.total_batches, 0);
        assert_eq!(stats.total_samples, 0);
        assert_eq!(stats.avg_batch_size, 0.0);
    }

    #[test]
    fn test_batch_statistics_record() {
        let mut stats = BatchStatistics::new();

        stats.record_batch(32, 0.1);
        assert_eq!(stats.total_batches, 1);
        assert_eq!(stats.total_samples, 32);
        assert_eq!(stats.avg_batch_size, 32.0);

        stats.record_batch(30, 0.15);
        assert_eq!(stats.total_batches, 2);
        assert_eq!(stats.total_samples, 62);
        assert_eq!(stats.avg_batch_size, 31.0);
    }

    #[test]
    fn test_batch_statistics_min_max() {
        let mut stats = BatchStatistics::new();

        stats.record_batch(32, 0.1);
        stats.record_batch(20, 0.1);
        stats.record_batch(40, 0.1);

        assert_eq!(stats.min_batch_size, 20);
        assert_eq!(stats.max_batch_size, 40);
    }

    #[test]
    fn test_batch_statistics_efficiency() {
        let mut stats = BatchStatistics::new();
        stats.record_batch(32, 0.2); // 20% padding

        let efficiency = stats.efficiency();
        assert!((efficiency - 0.8).abs() < 0.01); // 80% efficiency
    }

    #[test]
    fn test_utils_calculate_optimal_batch_size() {
        let batch_size = batch_utils::calculate_optimal_batch_size(
            1024 * 1024,      // 1 MB per sample
            1024 * 1024 * 64, // 64 MB available
            0.8,              // 80% safety factor
        );

        assert!(batch_size > 0);
        assert!(batch_size <= 64);
    }

    #[test]
    fn test_utils_calculate_num_batches() {
        assert_eq!(batch_utils::calculate_num_batches(100, 32, false), 4);
        assert_eq!(batch_utils::calculate_num_batches(100, 32, true), 3);
        assert_eq!(batch_utils::calculate_num_batches(96, 32, false), 3);
        assert_eq!(batch_utils::calculate_num_batches(96, 32, true), 3);
    }

    #[test]
    fn test_utils_calculate_padding_overhead() {
        let lengths = vec![10, 15, 12, 8];
        let overhead = batch_utils::calculate_padding_overhead(&lengths, 20);

        // Total original: 45, padded total: 80
        // Overhead: 1 - (45/80) = 0.4375
        assert!((overhead - 0.4375).abs() < 0.01);
    }

    #[test]
    fn test_utils_find_optimal_padding_length() {
        let lengths = vec![10, 15, 18, 22];
        let optimal = batch_utils::find_optimal_padding_length(&lengths, 8);

        assert_eq!(optimal, 24); // Next multiple of 8 after 22
    }

    #[test]
    fn test_utils_group_by_length() {
        let lengths = vec![10, 25, 15, 30, 20, 12, 28];
        let groups = batch_utils::group_by_length(lengths, 3);

        assert_eq!(groups.len(), 3);
        // Each group should have similar-length sequences
        for group in &groups {
            assert!(!group.is_empty());
        }
    }

    #[test]
    fn test_utils_calculate_memory_efficiency() {
        let efficiency = batch_utils::calculate_memory_efficiency(
            32,  // batch size
            100, // avg sequence length
            128, // max sequence length
        );

        // Efficiency: (32 * 100) / (32 * 128) = 3200 / 4096 ≈ 0.78
        assert!((efficiency - 0.78125).abs() < 0.01);
    }

    #[test]
    fn test_padding_strategy_variants() {
        let strategies = [
            PaddingStrategy::LongestInBatch,
            PaddingStrategy::FixedLength,
            PaddingStrategy::NearestMultiple,
            PaddingStrategy::NoPadding,
        ];

        assert_eq!(strategies.len(), 4);
    }

    #[test]
    fn test_collation_strategy_variants() {
        let strategies = [
            CollationStrategy::Stack,
            CollationStrategy::Concatenate,
            CollationStrategy::PadAndStack,
            CollationStrategy::Custom,
        ];

        assert_eq!(strategies.len(), 4);
    }
}
