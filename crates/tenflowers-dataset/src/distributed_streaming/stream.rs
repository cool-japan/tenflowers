//! Streaming logic, channels, and the StreamingShardLoader / StreamingShardIterator

use crate::Dataset;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, RwLock};
use tenflowers_core::{Result, Tensor, TensorError};

use super::coordinator::StreamCoordinator;
use super::types::{CheckpointState, PartitionStrategy, StreamingConfig, StreamingStats};

/// Streaming shard loader with deterministic partitioning
pub struct StreamingShardLoader<T, D: Dataset<T>> {
    /// Underlying dataset
    pub(super) dataset: Arc<D>,
    /// Streaming configuration
    pub(super) config: StreamingConfig,
    /// Assigned indices for this worker
    pub(super) assigned_indices: Vec<usize>,
    /// Current position in the stream
    pub(super) current_position: Arc<Mutex<usize>>,
    /// Prefetch buffer
    pub(super) prefetch_buffer: Arc<Mutex<VecDeque<(Tensor<T>, Tensor<T>)>>>,
    /// Checkpoint state
    pub(super) checkpoint_state: Arc<RwLock<CheckpointState>>,
    /// Statistics collector
    pub(super) stats: Arc<RwLock<StreamingStats>>,
    /// Worker coordinator
    pub(super) coordinator: Option<Arc<StreamCoordinator>>,
    pub(super) _phantom: PhantomData<T>,
}

impl<T, D: Dataset<T>> StreamingShardLoader<T, D>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    /// Create a new streaming shard loader
    pub fn new(dataset: D, config: StreamingConfig) -> Result<Self> {
        config.validate()?;

        let dataset = Arc::new(dataset);
        let assigned_indices = Self::compute_assigned_indices(&dataset, &config)?;

        let checkpoint_state = CheckpointState {
            epoch: 0,
            position: 0,
            shuffle_seed: config.shuffle_seed,
            rank: config.rank,
            timestamp: Self::current_timestamp(),
            processed_indices: std::collections::HashSet::new(),
        };

        Ok(Self {
            dataset,
            config,
            assigned_indices,
            current_position: Arc::new(Mutex::new(0)),
            prefetch_buffer: Arc::new(Mutex::new(VecDeque::new())),
            checkpoint_state: Arc::new(RwLock::new(checkpoint_state)),
            stats: Arc::new(RwLock::new(StreamingStats::default())),
            coordinator: None,
            _phantom: PhantomData,
        })
    }

    /// Create with coordinator for multi-worker coordination
    pub fn with_coordinator(mut self, coordinator: Arc<StreamCoordinator>) -> Self {
        self.coordinator = Some(coordinator);
        self
    }

    /// Compute indices assigned to this worker based on partition strategy
    fn compute_assigned_indices(dataset: &D, config: &StreamingConfig) -> Result<Vec<usize>> {
        let total_size = dataset.len();
        if total_size == 0 {
            return Ok(Vec::new());
        }

        let mut all_indices: Vec<usize> = (0..total_size).collect();

        if let Some(seed) = config.shuffle_seed {
            Self::deterministic_shuffle(&mut all_indices, seed);
        }

        let assigned = match &config.partition_strategy {
            PartitionStrategy::RoundRobin => {
                Self::partition_round_robin(&all_indices, config.world_size, config.rank)
            }

            PartitionStrategy::Contiguous => {
                Self::partition_contiguous(&all_indices, config.world_size, config.rank)
            }

            PartitionStrategy::HashBased {
                num_partitions,
                hash_seed,
            } => Self::partition_hash_based(
                &all_indices,
                config.world_size,
                config.rank,
                *num_partitions,
                *hash_seed,
            ),

            PartitionStrategy::RangeBased { ranges } => {
                Self::partition_range_based(&all_indices, config.rank, ranges)
            }

            PartitionStrategy::Stratified { .. } => {
                // For stratified, we need label information
                // This is a simplified version - full implementation would require label access
                Self::partition_round_robin(&all_indices, config.world_size, config.rank)
            }

            PartitionStrategy::Adaptive { base_strategy, .. } => match **base_strategy {
                PartitionStrategy::RoundRobin => {
                    Self::partition_round_robin(&all_indices, config.world_size, config.rank)
                }
                PartitionStrategy::Contiguous => {
                    Self::partition_contiguous(&all_indices, config.world_size, config.rank)
                }
                _ => Self::partition_round_robin(&all_indices, config.world_size, config.rank),
            },

            PartitionStrategy::Custom { .. } => {
                Self::partition_round_robin(&all_indices, config.world_size, config.rank)
            }
        };

        Ok(assigned)
    }

    /// Round-robin partitioning
    fn partition_round_robin(indices: &[usize], world_size: usize, rank: usize) -> Vec<usize> {
        indices
            .iter()
            .enumerate()
            .filter(|(i, _)| i % world_size == rank)
            .map(|(_, &idx)| idx)
            .collect()
    }

    /// Contiguous block partitioning
    fn partition_contiguous(indices: &[usize], world_size: usize, rank: usize) -> Vec<usize> {
        let total_size = indices.len();
        let base_size = total_size / world_size;
        let extra = total_size % world_size;

        let start = if rank < extra {
            rank * (base_size + 1)
        } else {
            rank * base_size + extra
        };

        let size = if rank < extra {
            base_size + 1
        } else {
            base_size
        };

        indices[start..start + size].to_vec()
    }

    /// Hash-based partitioning for deterministic distribution
    fn partition_hash_based(
        indices: &[usize],
        world_size: usize,
        rank: usize,
        num_partitions: usize,
        hash_seed: u64,
    ) -> Vec<usize> {
        let effective_partitions = num_partitions.max(world_size);

        indices
            .iter()
            .filter(|&&idx| {
                let hash = Self::compute_hash(idx, hash_seed);
                let partition = hash % effective_partitions;
                partition % world_size == rank
            })
            .copied()
            .collect()
    }

    /// Range-based partitioning
    fn partition_range_based(
        indices: &[usize],
        rank: usize,
        ranges: &[(usize, usize)],
    ) -> Vec<usize> {
        if rank >= ranges.len() {
            return Vec::new();
        }

        let (start, end) = ranges[rank];
        indices
            .iter()
            .filter(|&&idx| idx >= start && idx < end)
            .copied()
            .collect()
    }

    /// Deterministic hash function for partitioning
    fn compute_hash(value: usize, seed: u64) -> usize {
        let mut hash = seed.wrapping_add(value as u64);
        hash = hash.wrapping_mul(0x9e3779b97f4a7c15);
        hash ^= hash >> 30;
        hash = hash.wrapping_mul(0xbf58476d1ce4e5b9);
        hash ^= hash >> 27;
        hash = hash.wrapping_mul(0x94d049bb133111eb);
        hash ^= hash >> 31;
        hash as usize
    }

    /// Deterministic shuffle using Fisher-Yates with LCG
    fn deterministic_shuffle(indices: &mut [usize], seed: u64) {
        let mut rng_state = seed;

        for i in (1..indices.len()).rev() {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (rng_state as usize) % (i + 1);
            indices.swap(i, j);
        }
    }

    /// Get the next sample from the stream
    pub fn next(&self) -> Result<Option<(Tensor<T>, Tensor<T>)>> {
        {
            let mut buffer = self
                .prefetch_buffer
                .lock()
                .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;
            if let Some(sample) = buffer.pop_front() {
                self.update_stats_hit();
                return Ok(Some(sample));
            }
        }

        self.update_stats_miss();

        let mut position = self
            .current_position
            .lock()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;

        if *position >= self.assigned_indices.len() {
            return Ok(None);
        }

        let index = self.assigned_indices[*position];
        *position += 1;

        let start_time = std::time::Instant::now();
        let sample = self.dataset.get(index)?;
        let load_time = start_time.elapsed().as_micros() as u64;

        self.update_stats_loaded(load_time);

        if let Some(interval) = self.config.checkpoint_interval {
            if *position % interval == 0 {
                self.create_checkpoint(*position)?;
            }
        }

        Ok(Some(sample))
    }

    /// Prefetch samples into buffer
    pub fn prefetch(&self, count: usize) -> Result<()> {
        let mut buffer = self
            .prefetch_buffer
            .lock()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;

        let position = *self
            .current_position
            .lock()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;

        let available = self.assigned_indices.len().saturating_sub(position);
        let to_prefetch = count.min(available);

        for i in 0..to_prefetch {
            let index = self.assigned_indices[position + i];
            let sample = self.dataset.get(index)?;
            buffer.push_back(sample);
        }

        Ok(())
    }

    /// Create a checkpoint of current state
    fn create_checkpoint(&self, position: usize) -> Result<()> {
        let mut state = self
            .checkpoint_state
            .write()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;

        state.position = position;
        state.timestamp = Self::current_timestamp();

        let mut stats = self
            .stats
            .write()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;
        stats.num_checkpoints += 1;

        Ok(())
    }

    /// Restore from checkpoint
    pub fn restore_from_checkpoint(&self, checkpoint: CheckpointState) -> Result<()> {
        let mut state = self
            .checkpoint_state
            .write()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;
        *state = checkpoint.clone();

        let mut position = self
            .current_position
            .lock()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;
        *position = checkpoint.position;

        Ok(())
    }

    /// Get current checkpoint state
    pub fn get_checkpoint(&self) -> Result<CheckpointState> {
        let position = *self
            .current_position
            .lock()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;

        let mut state = self
            .checkpoint_state
            .write()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;

        state.position = position;
        state.timestamp = Self::current_timestamp();

        Ok(state.clone())
    }

    /// Get streaming statistics
    pub fn get_stats(&self) -> Result<StreamingStats> {
        let stats = self
            .stats
            .read()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;
        Ok(stats.clone())
    }

    /// Reset the stream to beginning
    pub fn reset(&self) -> Result<()> {
        let mut position = self
            .current_position
            .lock()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;
        *position = 0;

        let mut buffer = self
            .prefetch_buffer
            .lock()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;
        buffer.clear();

        Ok(())
    }

    /// Get total number of samples assigned to this worker
    pub fn len(&self) -> usize {
        self.assigned_indices.len()
    }

    /// Check if stream is empty
    pub fn is_empty(&self) -> bool {
        self.assigned_indices.is_empty()
    }

    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    fn update_stats_hit(&self) {
        if let Ok(mut stats) = self.stats.write() {
            stats.prefetch_hits += 1;
        }
    }

    fn update_stats_miss(&self) {
        if let Ok(mut stats) = self.stats.write() {
            stats.prefetch_misses += 1;
        }
    }

    fn update_stats_loaded(&self, load_time_us: u64) {
        if let Ok(mut stats) = self.stats.write() {
            stats.samples_loaded += 1;
            stats.local_samples += 1;

            let n = stats.samples_loaded;
            stats.avg_load_time_us = ((stats.avg_load_time_us * (n - 1)) + load_time_us) / n;
        }
    }
}

/// Iterator adapter for streaming shard loader
pub struct StreamingShardIterator<T, D: Dataset<T>>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    loader: Arc<StreamingShardLoader<T, D>>,
}

impl<T, D: Dataset<T>> StreamingShardIterator<T, D>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    pub fn new(loader: Arc<StreamingShardLoader<T, D>>) -> Self {
        Self { loader }
    }
}

impl<T, D: Dataset<T>> Iterator for StreamingShardIterator<T, D>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    type Item = Result<(Tensor<T>, Tensor<T>)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.loader.next() {
            Ok(Some(sample)) => Some(Ok(sample)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}
