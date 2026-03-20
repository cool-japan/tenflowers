//! Distributed streaming loaders for large-scale data processing
//!
//! This module provides sophisticated distributed streaming capabilities with:
//! - Deterministic shard partitioning for reproducibility
//! - Multi-worker coordination for distributed training
//! - Advanced partitioning strategies for load balancing
//! - Stream checkpointing and resumption
//! - Fault tolerance and worker failure recovery

pub mod coordinator;
pub mod load_balancer;
pub mod stream;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export all public types for backward compatibility
pub use types::{
    CheckpointState, PartitionStrategy, StreamingConfig, StreamingStats, WorkerHealth,
    WorkerMetrics, WorkerStatus,
};

pub use stream::{StreamingShardIterator, StreamingShardLoader};

pub use coordinator::StreamCoordinator;

pub use load_balancer::{
    compute_load_imbalance, compute_redistribution_weights, find_overloaded_workers,
    find_underloaded_workers, LoadBalancingPolicy,
};
