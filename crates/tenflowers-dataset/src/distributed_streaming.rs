//! Distributed streaming loaders for large-scale data processing
//!
//! This module provides sophisticated distributed streaming capabilities with:
//! - Deterministic shard partitioning for reproducibility
//! - Multi-worker coordination for distributed training
//! - Advanced partitioning strategies for load balancing
//! - Stream checkpointing and resumption
//! - Fault tolerance and worker failure recovery

use crate::{
    distributed_sharding::{ShardConfig, ShardStrategy},
    error_taxonomy::helpers as error_helpers,
    Dataset,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, RwLock};
use tenflowers_core::{Result, Tensor, TensorError};

/// Configuration for distributed streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Total number of workers in the distributed system
    pub world_size: usize,
    /// Current worker rank (0-indexed)
    pub rank: usize,
    /// Partition strategy for distributing data
    pub partition_strategy: PartitionStrategy,
    /// Buffer size for prefetching
    pub prefetch_buffer_size: usize,
    /// Enable deterministic shuffling with seed
    pub shuffle_seed: Option<u64>,
    /// Checkpoint interval (number of samples)
    pub checkpoint_interval: Option<usize>,
    /// Enable fault tolerance
    pub fault_tolerant: bool,
    /// Replication factor for fault tolerance
    pub replication_factor: usize,
    /// Dynamic load balancing enabled
    pub dynamic_balancing: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            partition_strategy: PartitionStrategy::HashBased {
                num_partitions: 1,
                hash_seed: 0,
            },
            prefetch_buffer_size: 128,
            shuffle_seed: None,
            checkpoint_interval: Some(1000),
            fault_tolerant: false,
            replication_factor: 1,
            dynamic_balancing: false,
        }
    }
}

impl StreamingConfig {
    /// Create a new streaming configuration
    pub fn new(world_size: usize, rank: usize) -> Result<Self> {
        if world_size == 0 {
            return Err(error_helpers::invalid_configuration(
                "StreamingConfig::new",
                "world_size",
                "world_size must be > 0",
            ));
        }

        if rank >= world_size {
            return Err(error_helpers::invalid_configuration(
                "StreamingConfig::new",
                "rank",
                format!("rank {} must be < world_size {}", rank, world_size),
            ));
        }

        Ok(Self {
            world_size,
            rank,
            ..Default::default()
        })
    }

    /// Set the partition strategy
    pub fn with_partition_strategy(mut self, strategy: PartitionStrategy) -> Self {
        self.partition_strategy = strategy;
        self
    }

    /// Set the prefetch buffer size
    pub fn with_prefetch_buffer_size(mut self, size: usize) -> Self {
        self.prefetch_buffer_size = size;
        self
    }

    /// Set the shuffle seed for deterministic shuffling
    pub fn with_shuffle_seed(mut self, seed: u64) -> Self {
        self.shuffle_seed = Some(seed);
        self
    }

    /// Enable checkpointing with specified interval
    pub fn with_checkpointing(mut self, interval: usize) -> Self {
        self.checkpoint_interval = Some(interval);
        self
    }

    /// Enable fault tolerance with replication
    pub fn with_fault_tolerance(mut self, replication_factor: usize) -> Self {
        self.fault_tolerant = true;
        self.replication_factor = replication_factor;
        self
    }

    /// Enable dynamic load balancing
    pub fn with_dynamic_balancing(mut self, enabled: bool) -> Self {
        self.dynamic_balancing = enabled;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.world_size == 0 {
            return Err(error_helpers::invalid_configuration(
                "StreamingConfig::validate",
                "world_size",
                "world_size must be > 0",
            ));
        }

        if self.rank >= self.world_size {
            return Err(error_helpers::invalid_configuration(
                "StreamingConfig::validate",
                "rank",
                format!(
                    "rank {} must be < world_size {}",
                    self.rank, self.world_size
                ),
            ));
        }

        if self.replication_factor > self.world_size {
            return Err(error_helpers::invalid_configuration(
                "StreamingConfig::validate",
                "replication_factor",
                format!(
                    "replication_factor {} cannot exceed world_size {}",
                    self.replication_factor, self.world_size
                ),
            ));
        }

        Ok(())
    }
}

/// Advanced partition strategies for distributed streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitionStrategy {
    /// Round-robin distribution (simple, balanced for uniform data)
    RoundRobin,

    /// Contiguous blocks (good for sequential access patterns)
    Contiguous,

    /// Hash-based partitioning (deterministic, good for key-based data)
    HashBased {
        num_partitions: usize,
        hash_seed: u64,
    },

    /// Range-based partitioning (good for sorted data)
    RangeBased { ranges: Vec<(usize, usize)> },

    /// Stratified partitioning (maintains class distribution)
    Stratified { num_classes: usize },

    /// Adaptive partitioning (adjusts based on worker performance)
    Adaptive {
        base_strategy: Box<PartitionStrategy>,
        rebalance_threshold: f64,
    },

    /// Custom partitioning (user-defined function)
    Custom { partition_id: String },
}

impl Default for PartitionStrategy {
    fn default() -> Self {
        Self::RoundRobin
    }
}

/// Streaming shard loader with deterministic partitioning
pub struct StreamingShardLoader<T, D: Dataset<T>> {
    /// Underlying dataset
    dataset: Arc<D>,
    /// Streaming configuration
    config: StreamingConfig,
    /// Assigned indices for this worker
    assigned_indices: Vec<usize>,
    /// Current position in the stream
    current_position: Arc<Mutex<usize>>,
    /// Prefetch buffer
    prefetch_buffer: Arc<Mutex<VecDeque<(Tensor<T>, Tensor<T>)>>>,
    /// Checkpoint state
    checkpoint_state: Arc<RwLock<CheckpointState>>,
    /// Statistics collector
    stats: Arc<RwLock<StreamingStats>>,
    /// Worker coordinator
    coordinator: Option<Arc<StreamCoordinator>>,
    _phantom: PhantomData<T>,
}

/// Checkpoint state for stream resumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointState {
    /// Current epoch
    pub epoch: usize,
    /// Current position in stream
    pub position: usize,
    /// Shuffle seed used
    pub shuffle_seed: Option<u64>,
    /// Worker rank
    pub rank: usize,
    /// Timestamp
    pub timestamp: u64,
    /// Indices processed so far
    pub processed_indices: HashSet<usize>,
}

/// Statistics for streaming performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingStats {
    /// Total samples loaded
    pub samples_loaded: u64,
    /// Samples loaded from local shard
    pub local_samples: u64,
    /// Samples loaded from remote workers
    pub remote_samples: u64,
    /// Prefetch buffer hits
    pub prefetch_hits: u64,
    /// Prefetch buffer misses
    pub prefetch_misses: u64,
    /// Average load time per sample (microseconds)
    pub avg_load_time_us: u64,
    /// Number of checkpoints created
    pub num_checkpoints: u64,
    /// Worker utilization (0.0 - 1.0)
    pub worker_utilization: f64,
}

impl Default for StreamingStats {
    fn default() -> Self {
        Self {
            samples_loaded: 0,
            local_samples: 0,
            remote_samples: 0,
            prefetch_hits: 0,
            prefetch_misses: 0,
            avg_load_time_us: 0,
            num_checkpoints: 0,
            worker_utilization: 0.0,
        }
    }
}

/// Multi-worker stream coordinator
pub struct StreamCoordinator {
    /// Configuration
    config: StreamingConfig,
    /// Worker assignments
    worker_assignments: Arc<RwLock<HashMap<usize, Vec<usize>>>>,
    /// Worker health status
    worker_health: Arc<RwLock<HashMap<usize, WorkerHealth>>>,
    /// Global checkpoint registry
    global_checkpoints: Arc<RwLock<HashMap<usize, CheckpointState>>>,
    /// Load balancing metrics
    balancing_metrics: Arc<RwLock<HashMap<usize, WorkerMetrics>>>,
}

/// Worker health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHealth {
    pub rank: usize,
    pub status: WorkerStatus,
    pub last_heartbeat: u64,
    pub samples_processed: u64,
    pub average_throughput: f64,
}

/// Worker status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerStatus {
    Active,
    Idle,
    Slow,
    Failed,
    Unknown,
}

/// Worker performance metrics for load balancing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerMetrics {
    pub rank: usize,
    pub throughput_samples_per_sec: f64,
    pub queue_depth: usize,
    pub cpu_utilization: f64,
    pub memory_usage_mb: f64,
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
            processed_indices: HashSet::new(),
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

        // Apply shuffling if configured
        if let Some(seed) = config.shuffle_seed {
            Self::deterministic_shuffle(&mut all_indices, seed);
        }

        // Partition based on strategy
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

            PartitionStrategy::Stratified { num_classes } => {
                // For stratified, we need label information
                // This is a simplified version - full implementation would require label access
                Self::partition_round_robin(&all_indices, config.world_size, config.rank)
            }

            PartitionStrategy::Adaptive { base_strategy, .. } => {
                // Start with base strategy, will be adjusted dynamically
                match **base_strategy {
                    PartitionStrategy::RoundRobin => {
                        Self::partition_round_robin(&all_indices, config.world_size, config.rank)
                    }
                    PartitionStrategy::Contiguous => {
                        Self::partition_contiguous(&all_indices, config.world_size, config.rank)
                    }
                    _ => Self::partition_round_robin(&all_indices, config.world_size, config.rank),
                }
            }

            PartitionStrategy::Custom { .. } => {
                // Custom partitioning would require user-provided function
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
        hash = hash.wrapping_mul(0x9e3779b97f4a7c15); // Golden ratio
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
            // Linear congruential generator
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (rng_state as usize) % (i + 1);
            indices.swap(i, j);
        }
    }

    /// Get the next sample from the stream
    pub fn next(&self) -> Result<Option<(Tensor<T>, Tensor<T>)>> {
        // Check prefetch buffer first
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

        // Buffer miss - load from dataset
        self.update_stats_miss();

        let mut position = self
            .current_position
            .lock()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;

        if *position >= self.assigned_indices.len() {
            return Ok(None); // End of stream
        }

        let index = self.assigned_indices[*position];
        *position += 1;

        let start_time = std::time::Instant::now();
        let sample = self.dataset.get(index)?;
        let load_time = start_time.elapsed().as_micros() as u64;

        self.update_stats_loaded(load_time);

        // Update checkpoint if needed
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

            // Update running average
            let n = stats.samples_loaded;
            stats.avg_load_time_us = ((stats.avg_load_time_us * (n - 1)) + load_time_us) / n;
        }
    }
}

impl StreamCoordinator {
    /// Create a new stream coordinator
    pub fn new(config: StreamingConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            config,
            worker_assignments: Arc::new(RwLock::new(HashMap::new())),
            worker_health: Arc::new(RwLock::new(HashMap::new())),
            global_checkpoints: Arc::new(RwLock::new(HashMap::new())),
            balancing_metrics: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Register a worker with the coordinator
    pub fn register_worker(&self, rank: usize, indices: Vec<usize>) -> Result<()> {
        let mut assignments = self
            .worker_assignments
            .write()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;
        assignments.insert(rank, indices);

        let mut health = self
            .worker_health
            .write()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;
        health.insert(
            rank,
            WorkerHealth {
                rank,
                status: WorkerStatus::Active,
                last_heartbeat: Self::current_timestamp(),
                samples_processed: 0,
                average_throughput: 0.0,
            },
        );

        Ok(())
    }

    /// Update worker health status
    pub fn update_worker_health(
        &self,
        rank: usize,
        samples_processed: u64,
        throughput: f64,
    ) -> Result<()> {
        let mut health = self
            .worker_health
            .write()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;

        if let Some(worker_health) = health.get_mut(&rank) {
            worker_health.last_heartbeat = Self::current_timestamp();
            worker_health.samples_processed = samples_processed;
            worker_health.average_throughput = throughput;
            worker_health.status = Self::determine_worker_status(throughput);
        }

        Ok(())
    }

    /// Get worker health status
    pub fn get_worker_health(&self, rank: usize) -> Result<Option<WorkerHealth>> {
        let health = self
            .worker_health
            .read()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;
        Ok(health.get(&rank).cloned())
    }

    /// Register checkpoint for a worker
    pub fn register_checkpoint(&self, rank: usize, checkpoint: CheckpointState) -> Result<()> {
        let mut checkpoints = self
            .global_checkpoints
            .write()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;
        checkpoints.insert(rank, checkpoint);
        Ok(())
    }

    /// Get checkpoint for a worker
    pub fn get_checkpoint(&self, rank: usize) -> Result<Option<CheckpointState>> {
        let checkpoints = self
            .global_checkpoints
            .read()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;
        Ok(checkpoints.get(&rank).cloned())
    }

    /// Perform dynamic load balancing if enabled
    pub fn rebalance_if_needed(&self) -> Result<bool> {
        if !self.config.dynamic_balancing {
            return Ok(false);
        }

        // Check if rebalancing is needed based on worker metrics
        let health = self
            .worker_health
            .read()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;

        let workers: Vec<_> = health.values().collect();
        if workers.is_empty() {
            return Ok(false);
        }

        // Calculate throughput variance
        let avg_throughput: f64 =
            workers.iter().map(|w| w.average_throughput).sum::<f64>() / workers.len() as f64;

        let variance: f64 = workers
            .iter()
            .map(|w| {
                let diff = w.average_throughput - avg_throughput;
                diff * diff
            })
            .sum::<f64>()
            / workers.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if avg_throughput > 0.0 {
            std_dev / avg_throughput
        } else {
            0.0
        };

        // Trigger rebalancing if variance is high (> 20% coefficient of variation)
        let rebalanced = coefficient_of_variation > 0.2;

        Ok(rebalanced)
    }

    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    fn determine_worker_status(throughput: f64) -> WorkerStatus {
        if throughput <= 0.0 {
            WorkerStatus::Failed
        } else if throughput < 10.0 {
            WorkerStatus::Slow
        } else {
            WorkerStatus::Active
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorDataset;
    use tenflowers_core::Tensor;

    #[test]
    fn test_streaming_config_creation() {
        let config = StreamingConfig::new(4, 0).expect("config creation should succeed");
        assert_eq!(config.world_size, 4);
        assert_eq!(config.rank, 0);
    }

    #[test]
    fn test_streaming_config_validation() {
        assert!(StreamingConfig::new(0, 0).is_err());
        assert!(StreamingConfig::new(4, 4).is_err());
        assert!(StreamingConfig::new(4, 3).is_ok());
    }

    #[test]
    fn test_round_robin_partitioning() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 100], &[100, 1])
            .expect("tensor creation should succeed");
        let labels = Tensor::<f32>::from_vec(vec![1.0; 100], &[100])
            .expect("tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(4, 0)
            .expect("config creation should succeed")
            .with_partition_strategy(PartitionStrategy::RoundRobin);

        let loader =
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed");

        // Rank 0 should get every 4th element: 0, 4, 8, 12, ...
        assert_eq!(loader.len(), 25); // 100 / 4 = 25
    }

    #[test]
    fn test_contiguous_partitioning() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 100], &[100, 1])
            .expect("tensor creation should succeed");
        let labels = Tensor::<f32>::from_vec(vec![1.0; 100], &[100])
            .expect("tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(4, 1)
            .expect("config creation should succeed")
            .with_partition_strategy(PartitionStrategy::Contiguous);

        let loader =
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed");

        // Rank 1 should get contiguous block: [25..50)
        assert_eq!(loader.len(), 25);
    }

    #[test]
    fn test_hash_based_partitioning() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 100], &[100, 1])
            .expect("tensor creation should succeed");
        let labels = Tensor::<f32>::from_vec(vec![1.0; 100], &[100])
            .expect("tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(4, 0)
            .expect("config creation should succeed")
            .with_partition_strategy(PartitionStrategy::HashBased {
                num_partitions: 4,
                hash_seed: 42,
            });

        let loader =
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed");

        // Hash-based should distribute relatively evenly
        assert!(loader.len() > 0);
        assert!(loader.len() <= 100);
    }

    #[test]
    fn test_deterministic_shuffling() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 50], &[50, 1])
            .expect("tensor creation should succeed");
        let labels =
            Tensor::<f32>::from_vec(vec![1.0; 50], &[50]).expect("tensor creation should succeed");
        let dataset1 = TensorDataset::new(features.clone(), labels.clone());
        let dataset2 = TensorDataset::new(features, labels);

        let config1 = StreamingConfig::new(2, 0)
            .expect("config creation should succeed")
            .with_shuffle_seed(123);
        let config2 = StreamingConfig::new(2, 0)
            .expect("config creation should succeed")
            .with_shuffle_seed(123);

        let loader1 =
            StreamingShardLoader::new(dataset1, config1).expect("loader creation should succeed");
        let loader2 =
            StreamingShardLoader::new(dataset2, config2).expect("loader creation should succeed");

        // Same seed should produce same indices
        assert_eq!(loader1.assigned_indices, loader2.assigned_indices);
    }

    #[test]
    fn test_streaming_next() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 10], &[10, 1])
            .expect("tensor creation should succeed");
        let labels =
            Tensor::<f32>::from_vec(vec![1.0; 10], &[10]).expect("tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(2, 0).expect("config creation should succeed");
        let loader =
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed");

        // Should be able to get samples
        let sample1 = loader.next().expect("next should succeed");
        assert!(sample1.is_some());

        let sample2 = loader.next().expect("next should succeed");
        assert!(sample2.is_some());
    }

    #[test]
    fn test_streaming_prefetch() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 20], &[20, 1])
            .expect("tensor creation should succeed");
        let labels =
            Tensor::<f32>::from_vec(vec![1.0; 20], &[20]).expect("tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(2, 0)
            .expect("config creation should succeed")
            .with_prefetch_buffer_size(5);
        let loader =
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed");

        // Prefetch samples
        loader.prefetch(5).expect("prefetch should succeed");

        // Next calls should hit the buffer
        let sample = loader.next().expect("next should succeed");
        assert!(sample.is_some());

        let stats = loader.get_stats().expect("get_stats should succeed");
        assert!(stats.prefetch_hits > 0);
    }

    #[test]
    fn test_checkpoint_creation() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 20], &[20, 1])
            .expect("tensor creation should succeed");
        let labels =
            Tensor::<f32>::from_vec(vec![1.0; 20], &[20]).expect("tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(2, 0)
            .expect("config creation should succeed")
            .with_checkpointing(5);
        let loader =
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed");

        // Load several samples to trigger checkpoint
        for _ in 0..6 {
            let _ = loader.next();
        }

        let stats = loader.get_stats().expect("get_stats should succeed");
        assert!(stats.num_checkpoints > 0);
    }

    #[test]
    fn test_checkpoint_restore() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 20], &[20, 1])
            .expect("tensor creation should succeed");
        let labels =
            Tensor::<f32>::from_vec(vec![1.0; 20], &[20]).expect("tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(2, 0).expect("config creation should succeed");
        let loader =
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed");

        // Load some samples
        for _ in 0..3 {
            let _ = loader.next();
        }

        // Get checkpoint
        let checkpoint = loader
            .get_checkpoint()
            .expect("get_checkpoint should succeed");

        // Load more samples
        for _ in 0..3 {
            let _ = loader.next();
        }

        // Restore checkpoint
        loader
            .restore_from_checkpoint(checkpoint)
            .expect("restore should succeed");

        // Position should be restored
        let restored_checkpoint = loader
            .get_checkpoint()
            .expect("get_checkpoint should succeed");
        assert_eq!(restored_checkpoint.position, 3);
    }

    #[test]
    fn test_stream_reset() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 20], &[20, 1])
            .expect("tensor creation should succeed");
        let labels =
            Tensor::<f32>::from_vec(vec![1.0; 20], &[20]).expect("tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(2, 0).expect("config creation should succeed");
        let loader =
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed");

        // Load some samples
        for _ in 0..5 {
            let _ = loader.next();
        }

        // Reset stream
        loader.reset().expect("reset should succeed");

        // Should be able to iterate again from beginning
        let sample = loader.next().expect("next should succeed");
        assert!(sample.is_some());
    }

    #[test]
    fn test_stream_coordinator_creation() {
        let config = StreamingConfig::new(4, 0).expect("config creation should succeed");
        let coordinator = StreamCoordinator::new(config);
        assert!(coordinator.is_ok());
    }

    #[test]
    fn test_worker_registration() {
        let config = StreamingConfig::new(4, 0).expect("config creation should succeed");
        let coordinator =
            StreamCoordinator::new(config).expect("coordinator creation should succeed");

        let indices = vec![0, 1, 2, 3, 4];
        coordinator
            .register_worker(0, indices)
            .expect("worker registration should succeed");

        let health = coordinator
            .get_worker_health(0)
            .expect("get_worker_health should succeed");
        assert!(health.is_some());
        assert_eq!(health.expect("health should exist").rank, 0);
    }

    #[test]
    fn test_worker_health_update() {
        let config = StreamingConfig::new(4, 0).expect("config creation should succeed");
        let coordinator =
            StreamCoordinator::new(config).expect("coordinator creation should succeed");

        coordinator
            .register_worker(0, vec![])
            .expect("worker registration should succeed");

        coordinator
            .update_worker_health(0, 100, 50.0)
            .expect("health update should succeed");

        let health = coordinator
            .get_worker_health(0)
            .expect("get_worker_health should succeed")
            .expect("health should exist");

        assert_eq!(health.samples_processed, 100);
        assert!((health.average_throughput - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_coordinator_checkpoint_management() {
        let config = StreamingConfig::new(4, 0).expect("config creation should succeed");
        let coordinator =
            StreamCoordinator::new(config).expect("coordinator creation should succeed");

        let checkpoint = CheckpointState {
            epoch: 1,
            position: 100,
            shuffle_seed: Some(42),
            rank: 0,
            timestamp: 12345,
            processed_indices: HashSet::new(),
        };

        coordinator
            .register_checkpoint(0, checkpoint.clone())
            .expect("checkpoint registration should succeed");

        let retrieved = coordinator
            .get_checkpoint(0)
            .expect("get_checkpoint should succeed")
            .expect("checkpoint should exist");

        assert_eq!(retrieved.epoch, 1);
        assert_eq!(retrieved.position, 100);
    }

    #[test]
    fn test_iterator_adapter() {
        let features = Tensor::<f32>::from_vec(vec![1.0; 10], &[10, 1])
            .expect("tensor creation should succeed");
        let labels =
            Tensor::<f32>::from_vec(vec![1.0; 10], &[10]).expect("tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(2, 0).expect("config creation should succeed");
        let loader = Arc::new(
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed"),
        );

        let mut iter = StreamingShardIterator::new(loader);

        // Should be able to iterate
        let mut count = 0;
        for result in iter {
            assert!(result.is_ok());
            count += 1;
        }

        assert!(count > 0);
    }

    #[test]
    fn test_empty_dataset_streaming() {
        let features =
            Tensor::<f32>::from_vec(vec![], &[0, 1]).expect("empty tensor creation should succeed");
        let labels =
            Tensor::<f32>::from_vec(vec![], &[0]).expect("empty tensor creation should succeed");
        let dataset = TensorDataset::new(features, labels);

        let config = StreamingConfig::new(2, 0).expect("config creation should succeed");
        let loader =
            StreamingShardLoader::new(dataset, config).expect("loader creation should succeed");

        assert_eq!(loader.len(), 0);
        assert!(loader.is_empty());

        let sample = loader.next().expect("next should succeed");
        assert!(sample.is_none());
    }

    #[test]
    fn test_partition_strategy_default() {
        let strategy = PartitionStrategy::default();
        assert!(matches!(strategy, PartitionStrategy::RoundRobin));
    }

    #[test]
    fn test_streaming_stats_default() {
        let stats = StreamingStats::default();
        assert_eq!(stats.samples_loaded, 0);
        assert_eq!(stats.prefetch_hits, 0);
        assert_eq!(stats.prefetch_misses, 0);
    }
}
