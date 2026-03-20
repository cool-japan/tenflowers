//! Types, configs, and errors for distributed streaming

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
