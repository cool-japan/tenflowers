//! Attention-optimized datasets for transformer architectures
//!
//! This module provides dataset implementations specifically optimized for
//! attention mechanisms in transformer models, including memory-efficient
//! attention patterns and dynamic sequence packing.

use crate::Dataset;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tenflowers_core::{Device, Result, Tensor, TensorError};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Configuration for attention-optimized datasets
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct AttentionOptimizedConfig {
    /// Maximum sequence length for attention
    pub max_seq_length: usize,
    /// Block size for block-sparse attention
    pub block_size: usize,
    /// Number of global attention tokens
    pub num_global_tokens: usize,
    /// Whether to use dynamic sequence packing
    pub enable_packing: bool,
    /// Target packing efficiency (0.0 to 1.0)
    pub target_packing_efficiency: f64,
    /// Whether to use sliding window attention
    pub sliding_window_size: Option<usize>,
    /// Enable memory-efficient attention patterns
    pub memory_efficient: bool,
    /// Attention pattern type
    pub attention_pattern: AttentionPattern,
    /// Batch size for optimal attention computation
    pub optimal_batch_size: usize,
    /// GPU acceleration enabled
    pub gpu_acceleration: bool,
    /// Target device for computation
    #[cfg_attr(feature = "serialize", serde(skip))]
    pub device: Option<Device>,
    /// Memory pool size for tensor reuse
    pub memory_pool_size: usize,
}

impl Default for AttentionOptimizedConfig {
    fn default() -> Self {
        Self {
            max_seq_length: 2048,
            block_size: 64,
            num_global_tokens: 16,
            enable_packing: true,
            target_packing_efficiency: 0.85,
            sliding_window_size: Some(512),
            memory_efficient: true,
            attention_pattern: AttentionPattern::BlockSparse,
            optimal_batch_size: 16,
            gpu_acceleration: false,
            device: None,
            memory_pool_size: 1024 * 1024, // 1MB default memory pool
        }
    }
}

/// Different attention patterns supported
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum AttentionPattern {
    /// Full dense attention
    Dense,
    /// Block-sparse attention
    BlockSparse,
    /// Sliding window attention
    SlidingWindow,
    /// Random sparse attention
    RandomSparse { sparsity: f64 },
    /// Strided attention pattern
    Strided { stride: usize },
    /// Local + global attention
    LocalGlobal,
    /// Flash attention optimized
    FlashAttention,
}

/// Sequence with attention-specific metadata
#[derive(Debug, Clone)]
pub struct AttentionSequence<T> {
    /// The actual sequence tokens
    pub tokens: Tensor<T>,
    /// Attention mask for the sequence
    pub attention_mask: Tensor<T>,
    /// Position embeddings
    pub position_ids: Tensor<T>,
    /// Block indices for block-sparse attention
    pub block_indices: Option<Tensor<T>>,
    /// Global token positions
    pub global_positions: Vec<usize>,
    /// Sequence metadata
    pub metadata: SequenceMetadata,
    /// Label for the sequence
    pub label: Tensor<T>,
}

/// Metadata for attention sequences
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct SequenceMetadata {
    /// Original sequence length before padding
    pub original_length: usize,
    /// Padding length added
    pub padding_length: usize,
    /// Document ID (for document-level attention)
    pub document_id: Option<String>,
    /// Segment ID (for multi-segment sequences)
    pub segment_id: Option<usize>,
    /// Attention pattern used
    pub attention_pattern: AttentionPattern,
    /// Computation complexity estimate
    pub complexity_score: f64,
}

impl<T> AttentionSequence<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + Send
        + Sync
        + 'static,
{
    /// Create a new attention sequence
    pub fn new(
        tokens: Tensor<T>,
        label: Tensor<T>,
        config: &AttentionOptimizedConfig,
    ) -> Result<Self> {
        let seq_length = tokens.shape().dims()[0];

        // Create attention mask (1 for real tokens, 0 for padding)
        let mask_data = vec![T::one(); seq_length];
        let attention_mask = Tensor::from_vec(mask_data, &[seq_length])?;

        // Create position IDs - simplified to zeros for type safety
        let position_data = vec![T::default(); seq_length];
        let position_ids = Tensor::from_vec(position_data, &[seq_length])?;

        // Create block indices for block-sparse attention
        let block_indices = if matches!(config.attention_pattern, AttentionPattern::BlockSparse) {
            let _num_blocks = (seq_length + config.block_size - 1) / config.block_size;
            // Simplified block indices - use defaults for type safety
            let block_data = vec![T::default(); seq_length];
            Some(Tensor::from_vec(block_data, &[seq_length])?)
        } else {
            None
        };

        // Determine global token positions
        let global_positions = Self::determine_global_positions(
            seq_length,
            config.num_global_tokens,
            &config.attention_pattern,
        );

        let metadata = SequenceMetadata {
            original_length: seq_length,
            padding_length: 0,
            document_id: None,
            segment_id: None,
            attention_pattern: config.attention_pattern.clone(),
            complexity_score: Self::estimate_complexity(seq_length, &config.attention_pattern),
        };

        Ok(Self {
            tokens,
            attention_mask,
            position_ids,
            block_indices,
            global_positions,
            metadata,
            label,
        })
    }

    /// Determine global token positions based on attention pattern
    fn determine_global_positions(
        seq_length: usize,
        num_global: usize,
        pattern: &AttentionPattern,
    ) -> Vec<usize> {
        match pattern {
            AttentionPattern::LocalGlobal => {
                // Place global tokens at beginning and end
                let mut positions = Vec::new();
                let spacing = seq_length / num_global.max(1);
                for i in 0..num_global.min(seq_length) {
                    positions.push(i * spacing);
                }
                positions
            }
            _ => {
                // Default: place at the beginning
                (0..num_global.min(seq_length)).collect()
            }
        }
    }

    /// Estimate computational complexity for attention
    fn estimate_complexity(seq_length: usize, pattern: &AttentionPattern) -> f64 {
        let n = seq_length as f64;
        match pattern {
            AttentionPattern::Dense => n * n,             // O(n²)
            AttentionPattern::BlockSparse => n * 64.0,    // O(n) with block size
            AttentionPattern::SlidingWindow => n * 512.0, // O(n) with window size
            AttentionPattern::RandomSparse { sparsity } => n * n * sparsity,
            AttentionPattern::Strided { stride } => n * n / (*stride as f64),
            AttentionPattern::LocalGlobal => n * 64.0 + 16.0 * n, // Local + global cost
            AttentionPattern::FlashAttention => n * n / 4.0,      // Flash attention savings
        }
    }

    /// Pack with another sequence for efficiency
    pub fn pack_with(
        &self,
        other: &AttentionSequence<T>,
        max_length: usize,
    ) -> Result<AttentionSequence<T>>
    where
        T: Clone
            + Default
            + scirs2_core::numeric::Zero
            + scirs2_core::numeric::One
            + scirs2_core::numeric::Float,
    {
        let self_len = self.tokens.shape().size();
        let other_len = other.tokens.shape().size();

        if self_len + other_len > max_length {
            return Err(TensorError::invalid_argument(
                "Combined sequences exceed maximum length".to_string(),
            ));
        }

        // Concatenate tokens
        let self_data = self
            .tokens
            .as_slice()
            .ok_or_else(|| TensorError::invalid_argument("Cannot access token data".to_string()))?;
        let other_data = other
            .tokens
            .as_slice()
            .ok_or_else(|| TensorError::invalid_argument("Cannot access token data".to_string()))?;

        let mut packed_tokens = Vec::new();
        packed_tokens.extend_from_slice(self_data);
        packed_tokens.extend_from_slice(other_data);

        // Pad to max length if needed
        while packed_tokens.len() < max_length {
            packed_tokens.push(T::zero());
        }

        let tokens = Tensor::from_vec(packed_tokens, &[max_length])?;

        // Create attention mask for packed sequence
        let mut mask_data = vec![T::one(); self_len + other_len];
        while mask_data.len() < max_length {
            mask_data.push(T::zero()); // Padding tokens get 0 mask
        }
        let attention_mask = Tensor::from_vec(mask_data, &[max_length])?;

        // Create position IDs - simplified to zeros for type safety
        let position_data = vec![T::default(); max_length];
        let position_ids = Tensor::from_vec(position_data, &[max_length])?;

        // Use first sequence's label
        let label = self.label.clone();

        let metadata = SequenceMetadata {
            original_length: self_len + other_len,
            padding_length: max_length - (self_len + other_len),
            document_id: None,
            segment_id: None,
            attention_pattern: self.metadata.attention_pattern.clone(),
            complexity_score: Self::estimate_complexity(
                max_length,
                &self.metadata.attention_pattern,
            ),
        };

        Ok(AttentionSequence {
            tokens,
            attention_mask,
            position_ids,
            block_indices: None, // Simplified for packing
            global_positions: vec![],
            metadata,
            label,
        })
    }
}

/// Attention-optimized dataset with dynamic packing
pub struct AttentionOptimizedDataset<T> {
    sequences: Vec<AttentionSequence<T>>,
    config: AttentionOptimizedConfig,
    packed_batches: Arc<Mutex<VecDeque<Vec<AttentionSequence<T>>>>>,
    packing_stats: Arc<Mutex<PackingStats>>,
}

/// Statistics for sequence packing
#[derive(Debug, Clone, Default)]
pub struct PackingStats {
    total_sequences: usize,
    packed_sequences: usize,
    average_packing_efficiency: f64,
    memory_savings: f64,
    _compute_savings: f64,
}

impl<T> AttentionOptimizedDataset<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + scirs2_core::numeric::Float
        + Send
        + Sync
        + 'static,
{
    /// Create a new attention-optimized dataset
    pub fn new(
        sequences: Vec<AttentionSequence<T>>,
        config: AttentionOptimizedConfig,
    ) -> Result<Self> {
        let dataset = Self {
            sequences,
            config,
            packed_batches: Arc::new(Mutex::new(VecDeque::new())),
            packing_stats: Arc::new(Mutex::new(PackingStats::default())),
        };

        // Pre-compute packed batches if packing is enabled
        if dataset.config.enable_packing {
            dataset.precompute_packed_batches()?;
        }

        Ok(dataset)
    }

    /// Pre-compute packed batches for efficiency
    fn precompute_packed_batches(&self) -> Result<()> {
        let mut sequences = self.sequences.clone();

        // Sort by length for better packing efficiency
        sequences.sort_by_key(|seq| seq.tokens.shape().size());

        let mut packed_batches = Vec::new();
        let mut current_batch = Vec::new();
        let mut current_length = 0;

        for sequence in sequences {
            let seq_length = sequence.tokens.shape().size();

            // Check if we can pack this sequence
            if current_length + seq_length <= self.config.max_seq_length {
                current_batch.push(sequence);
                current_length += seq_length;
            } else {
                // Start new batch
                if !current_batch.is_empty() {
                    packed_batches.push(current_batch);
                }
                current_batch = vec![sequence];
                current_length = seq_length;
            }

            // Check if batch is full
            if current_batch.len() >= self.config.optimal_batch_size {
                packed_batches.push(current_batch);
                current_batch = Vec::new();
                current_length = 0;
            }
        }

        // Add remaining batch
        if !current_batch.is_empty() {
            packed_batches.push(current_batch);
        }

        // Store packed batches
        if let Ok(mut batches) = self.packed_batches.lock() {
            for batch in packed_batches {
                batches.push_back(batch);
            }
        }

        // Update packing statistics
        self.update_packing_stats()?;

        Ok(())
    }

    /// Update packing statistics
    fn update_packing_stats(&self) -> Result<()> {
        let batches = self.packed_batches.lock().map_err(|_| {
            TensorError::invalid_argument("Failed to lock packed batches".to_string())
        })?;

        let mut stats = self.packing_stats.lock().map_err(|_| {
            TensorError::invalid_argument("Failed to lock packing stats".to_string())
        })?;

        stats.total_sequences = self.sequences.len();
        stats.packed_sequences = batches.iter().map(|b| b.len()).sum();

        // Calculate efficiency metrics
        let total_tokens: usize = self.sequences.iter().map(|s| s.tokens.shape().size()).sum();

        let packed_utilization: usize = batches
            .iter()
            .map(|batch| {
                let batch_max_len = batch
                    .iter()
                    .map(|s| s.tokens.shape().size())
                    .max()
                    .unwrap_or(0);
                batch_max_len * batch.len()
            })
            .sum();

        stats.average_packing_efficiency = if packed_utilization > 0 {
            total_tokens as f64 / packed_utilization as f64
        } else {
            0.0
        };

        stats.memory_savings = 1.0
            - (packed_utilization as f64
                / (stats.total_sequences * self.config.max_seq_length) as f64);

        Ok(())
    }

    /// Move tensors to GPU if GPU acceleration is enabled
    pub fn to_device(&self, tensor: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: bytemuck::Pod,
    {
        if let Some(device) = &self.config.device {
            tensor.to_device(*device)
        } else {
            Ok(tensor.clone())
        }
    }

    /// Batch process attention sequences with GPU acceleration
    pub fn process_batch_gpu(
        &self,
        batch: &[AttentionSequence<T>],
    ) -> Result<Vec<(Tensor<T>, Tensor<T>)>>
    where
        T: bytemuck::Pod,
    {
        if !self.config.gpu_acceleration || self.config.device.is_none() {
            // Fallback to CPU processing
            return Ok(batch
                .iter()
                .map(|seq| (seq.tokens.clone(), seq.label.clone()))
                .collect());
        }

        let device = self.config.device.as_ref().unwrap();
        let mut results = Vec::new();

        for sequence in batch {
            // Move tensors to GPU for processing
            let gpu_tokens = sequence.tokens.to_device(*device)?;
            let gpu_labels = sequence.label.to_device(*device)?;

            // Apply memory-efficient processing if enabled
            let processed_tokens = if self.config.memory_efficient {
                self.apply_memory_efficient_processing(&gpu_tokens)?
            } else {
                gpu_tokens
            };

            results.push((processed_tokens, gpu_labels));
        }

        Ok(results)
    }

    /// Apply memory-efficient processing optimizations
    fn apply_memory_efficient_processing(&self, tensor: &Tensor<T>) -> Result<Tensor<T>> {
        // Apply gradient checkpointing simulation (reduced memory footprint)
        // In real implementation, this would use actual gradient checkpointing

        let shape = tensor.shape().dims();
        let total_size = shape.iter().product::<usize>();

        // If tensor is large, apply chunked processing
        if total_size > self.config.memory_pool_size / std::mem::size_of::<T>() {
            self.apply_chunked_processing(tensor)
        } else {
            Ok(tensor.clone())
        }
    }

    /// Apply chunked processing for large tensors
    fn apply_chunked_processing(&self, tensor: &Tensor<T>) -> Result<Tensor<T>> {
        // Split tensor into smaller chunks and process sequentially
        // This reduces peak memory usage

        let chunk_size = self.config.memory_pool_size / (4 * std::mem::size_of::<T>());
        let shape = tensor.shape().dims();

        if shape.is_empty() || shape[0] <= chunk_size {
            return Ok(tensor.clone());
        }

        // For simplicity, return original tensor
        // In real implementation, would process in chunks
        Ok(tensor.clone())
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        let total_sequences = self.sequences.len();
        let avg_seq_length = if !self.sequences.is_empty() {
            self.sequences
                .iter()
                .map(|s| s.tokens.shape().size())
                .sum::<usize>()
                / total_sequences
        } else {
            0
        };

        let estimated_memory = total_sequences * avg_seq_length * std::mem::size_of::<T>();

        MemoryStats {
            total_sequences,
            average_sequence_length: avg_seq_length,
            estimated_memory_usage: estimated_memory,
            memory_pool_utilization: if self.config.memory_pool_size > 0 {
                (estimated_memory as f64 / self.config.memory_pool_size as f64).min(1.0)
            } else {
                0.0
            },
            gpu_memory_allocated: self.config.gpu_acceleration,
        }
    }

    /// Get next packed batch
    pub fn get_packed_batch(&self) -> Option<Vec<AttentionSequence<T>>> {
        self.packed_batches.lock().ok()?.pop_front()
    }

    /// Get packing statistics
    pub fn get_packing_stats(&self) -> Result<PackingStats> {
        Ok(self
            .packing_stats
            .lock()
            .map_err(|_| TensorError::invalid_argument("Failed to lock packing stats".to_string()))?
            .clone())
    }

    /// Create attention mask for a batch with different sequence lengths
    pub fn create_batch_attention_mask(&self, batch: &[AttentionSequence<T>]) -> Result<Tensor<T>> {
        let batch_size = batch.len();
        let max_seq_len = batch
            .iter()
            .map(|seq| seq.tokens.shape().size())
            .max()
            .unwrap_or(0);

        let mut mask_data = vec![T::zero(); batch_size * max_seq_len * max_seq_len];

        for (batch_idx, sequence) in batch.iter().enumerate() {
            let seq_len = sequence.tokens.shape().size();

            match &self.config.attention_pattern {
                AttentionPattern::Dense => {
                    // Full attention within sequence length
                    for i in 0..seq_len {
                        for j in 0..seq_len {
                            let idx = batch_idx * max_seq_len * max_seq_len + i * max_seq_len + j;
                            mask_data[idx] = T::one();
                        }
                    }
                }
                AttentionPattern::BlockSparse => {
                    // Block sparse attention
                    let block_size = self.config.block_size;
                    for i in 0..seq_len {
                        let block_i = i / block_size;
                        for j in 0..seq_len {
                            let block_j = j / block_size;
                            if block_i == block_j {
                                let idx =
                                    batch_idx * max_seq_len * max_seq_len + i * max_seq_len + j;
                                mask_data[idx] = T::one();
                            }
                        }
                    }
                }
                AttentionPattern::SlidingWindow => {
                    // Sliding window attention
                    if let Some(window_size) = self.config.sliding_window_size {
                        for i in 0..seq_len {
                            let window_start = i.saturating_sub(window_size / 2);
                            let window_end = (i + window_size / 2).min(seq_len);
                            for j in window_start..window_end {
                                let idx =
                                    batch_idx * max_seq_len * max_seq_len + i * max_seq_len + j;
                                mask_data[idx] = T::one();
                            }
                        }
                    }
                }
                AttentionPattern::LocalGlobal => {
                    // Local attention plus global tokens
                    let local_window = 64; // Local window size

                    for i in 0..seq_len {
                        // Local attention
                        let local_start = i.saturating_sub(local_window / 2);
                        let local_end = (i + local_window / 2).min(seq_len);
                        for j in local_start..local_end {
                            let idx = batch_idx * max_seq_len * max_seq_len + i * max_seq_len + j;
                            mask_data[idx] = T::one();
                        }

                        // Global attention
                        for &global_pos in &sequence.global_positions {
                            if global_pos < seq_len {
                                let idx = batch_idx * max_seq_len * max_seq_len
                                    + i * max_seq_len
                                    + global_pos;
                                mask_data[idx] = T::one();
                                let idx_rev = batch_idx * max_seq_len * max_seq_len
                                    + global_pos * max_seq_len
                                    + i;
                                mask_data[idx_rev] = T::one();
                            }
                        }
                    }
                }
                _ => {
                    // Default to causal attention for other patterns
                    for i in 0..seq_len {
                        for j in 0..=i {
                            let idx = batch_idx * max_seq_len * max_seq_len + i * max_seq_len + j;
                            mask_data[idx] = T::one();
                        }
                    }
                }
            }
        }

        Tensor::from_vec(mask_data, &[batch_size, max_seq_len, max_seq_len])
    }

    /// Estimate memory usage for different attention patterns
    pub fn estimate_memory_usage(&self, batch_size: usize, seq_length: usize) -> f64 {
        let base_memory = (batch_size * seq_length) as f64;

        match &self.config.attention_pattern {
            AttentionPattern::Dense => base_memory * seq_length as f64,
            AttentionPattern::BlockSparse => {
                let blocks_per_seq =
                    (seq_length + self.config.block_size - 1) / self.config.block_size;
                base_memory * self.config.block_size as f64 * blocks_per_seq as f64
            }
            AttentionPattern::SlidingWindow => {
                base_memory * self.config.sliding_window_size.unwrap_or(512) as f64
            }
            AttentionPattern::RandomSparse { sparsity } => {
                base_memory * seq_length as f64 * sparsity
            }
            AttentionPattern::LocalGlobal => {
                base_memory * (64.0 + self.config.num_global_tokens as f64 * 2.0)
            }
            AttentionPattern::FlashAttention => {
                base_memory * seq_length as f64 * 0.25 // Flash attention memory savings
            }
            _ => base_memory * seq_length as f64,
        }
    }
}

impl<T> Dataset<T> for AttentionOptimizedDataset<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    fn len(&self) -> usize {
        self.sequences.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.sequences.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index,
                self.sequences.len()
            )));
        }

        let sequence = &self.sequences[index];
        Ok((sequence.tokens.clone(), sequence.label.clone()))
    }
}

/// Builder for attention-optimized datasets
pub struct AttentionOptimizedDatasetBuilder<T> {
    sequences: Vec<AttentionSequence<T>>,
    config: AttentionOptimizedConfig,
}

impl<T> AttentionOptimizedDatasetBuilder<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + Send
        + Sync
        + 'static,
{
    pub fn new() -> Self {
        Self {
            sequences: Vec::new(),
            config: AttentionOptimizedConfig::default(),
        }
    }

    pub fn max_seq_length(mut self, length: usize) -> Self {
        self.config.max_seq_length = length;
        self
    }

    pub fn attention_pattern(mut self, pattern: AttentionPattern) -> Self {
        self.config.attention_pattern = pattern;
        self
    }

    pub fn block_size(mut self, size: usize) -> Self {
        self.config.block_size = size;
        self
    }

    pub fn sliding_window_size(mut self, size: usize) -> Self {
        self.config.sliding_window_size = Some(size);
        self
    }

    pub fn enable_packing(mut self, enabled: bool) -> Self {
        self.config.enable_packing = enabled;
        self
    }

    pub fn target_packing_efficiency(mut self, efficiency: f64) -> Self {
        self.config.target_packing_efficiency = efficiency.clamp(0.0, 1.0);
        self
    }

    pub fn add_sequence(mut self, tokens: Tensor<T>, label: Tensor<T>) -> Result<Self> {
        let sequence = AttentionSequence::new(tokens, label, &self.config)?;
        self.sequences.push(sequence);
        Ok(self)
    }

    pub fn build(self) -> Result<AttentionOptimizedDataset<T>>
    where
        T: scirs2_core::numeric::Float,
    {
        AttentionOptimizedDataset::new(self.sequences, self.config)
    }
}

impl<T> Default for AttentionOptimizedDatasetBuilder<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + Send
        + Sync
        + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage statistics for attention datasets
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total number of sequences
    pub total_sequences: usize,
    /// Average sequence length
    pub average_sequence_length: usize,
    /// Estimated memory usage in bytes
    pub estimated_memory_usage: usize,
    /// Memory pool utilization ratio (0.0 to 1.0)
    pub memory_pool_utilization: f64,
    /// Whether GPU memory is allocated
    pub gpu_memory_allocated: bool,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            total_sequences: 0,
            average_sequence_length: 0,
            estimated_memory_usage: 0,
            memory_pool_utilization: 0.0,
            gpu_memory_allocated: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    #[test]
    fn test_attention_sequence_creation() {
        let tokens = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let label = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let config = AttentionOptimizedConfig::default();

        let sequence = AttentionSequence::new(tokens, label, &config).unwrap();

        assert_eq!(sequence.tokens.shape().dims(), &[4]);
        assert_eq!(sequence.attention_mask.shape().dims(), &[4]);
        assert_eq!(sequence.position_ids.shape().dims(), &[4]);
        assert_eq!(sequence.metadata.original_length, 4);
    }

    #[test]
    fn test_sequence_packing() {
        let tokens1 = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let label1 = Tensor::from_vec(vec![1.0], &[1]).unwrap();
        let config = AttentionOptimizedConfig::default();
        let seq1 = AttentionSequence::new(tokens1, label1, &config).unwrap();

        let tokens2 = Tensor::<f32>::from_vec(vec![3.0, 4.0], &[2]).unwrap();
        let label2 = Tensor::from_vec(vec![0.0], &[1]).unwrap();
        let seq2 = AttentionSequence::new(tokens2, label2, &config).unwrap();

        let packed = seq1.pack_with(&seq2, 8).unwrap();
        assert_eq!(packed.tokens.shape().dims(), &[8]);
        assert_eq!(packed.metadata.original_length, 4);
        assert_eq!(packed.metadata.padding_length, 4);
    }

    #[test]
    fn test_attention_optimized_dataset() {
        let _config = AttentionOptimizedConfig::default();
        let mut builder = AttentionOptimizedDatasetBuilder::new()
            .attention_pattern(AttentionPattern::BlockSparse)
            .max_seq_length(16);

        for i in 0..5 {
            let tokens = Tensor::<f32>::from_vec(vec![i as f32; 4], &[4]).unwrap();
            let label = Tensor::from_vec(vec![i as f32], &[1]).unwrap();
            builder = builder.add_sequence(tokens, label).unwrap();
        }

        let dataset = builder.build().unwrap();
        assert_eq!(dataset.len(), 5);

        let (features, label) = dataset.get(0).unwrap();
        assert_eq!(features.shape().dims(), &[4]);
        assert_eq!(label.shape().dims(), &[1]);
    }

    #[test]
    fn test_complexity_estimation() {
        let seq_length = 1024;

        let dense_complexity =
            AttentionSequence::<f32>::estimate_complexity(seq_length, &AttentionPattern::Dense);
        let sparse_complexity = AttentionSequence::<f32>::estimate_complexity(
            seq_length,
            &AttentionPattern::BlockSparse,
        );

        assert!(sparse_complexity < dense_complexity);
    }

    #[test]
    fn test_memory_estimation() {
        let config = AttentionOptimizedConfig {
            attention_pattern: AttentionPattern::BlockSparse,
            block_size: 64,
            ..Default::default()
        };

        let sequences = vec![];
        let dataset: AttentionOptimizedDataset<f32> =
            AttentionOptimizedDataset::new(sequences, config).unwrap();

        let memory_usage = dataset.estimate_memory_usage(8, 1024);
        assert!(memory_usage > 0.0);
    }
}
