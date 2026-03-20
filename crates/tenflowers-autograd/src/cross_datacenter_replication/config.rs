//! Configuration Management for Cross-Datacenter Replication
//!
//! This module provides comprehensive configuration structures for controlling
//! replication behavior, compression settings, fault tolerance policies,
//! and bandwidth optimization strategies.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for cross-datacenter replication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationConfig {
    /// Synchronization frequency
    pub sync_interval: Duration,
    /// Maximum allowed drift between datacenters
    pub max_drift_steps: usize,
    /// Compression settings
    pub compression: CompressionConfig,
    /// Fault tolerance settings
    pub fault_tolerance: FaultToleranceConfig,
    /// Bandwidth optimization settings
    pub bandwidth_optimization: BandwidthOptimizationConfig,
}

/// Compression configuration for bandwidth efficiency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub algorithm: CompressionAlgorithm,
    pub compression_ratio: f64,
    pub quality_threshold: f64,
    pub adaptive_compression: bool,
}

/// Available compression algorithms for gradient and parameter data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// Gradient quantization
    Quantization { bits: u8 },
    /// Sparse gradient compression
    Sparsification { threshold: f64 },
    /// TopK sparsification
    TopK { k: usize },
    /// Error feedback compression
    ErrorFeedback { compression_ratio: f64 },
    /// Adaptive compression based on network conditions
    Adaptive,
}

/// Fault tolerance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    /// Number of backup datacenters per primary
    pub backup_replicas: usize,
    /// Timeout for cross-datacenter operations
    pub operation_timeout: Duration,
    /// Retry policy for failed operations
    pub retry_policy: RetryPolicy,
    /// Consensus requirements for parameter updates
    pub consensus_threshold: f64, // 0.5 for majority, 1.0 for unanimous
}

/// Retry policy configuration for handling failures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_retries: usize,
    pub base_delay: Duration,
    pub backoff_multiplier: f64,
    pub jitter: bool,
}

/// Bandwidth optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthOptimizationConfig {
    /// Use delta compression for parameter updates
    pub delta_compression: bool,
    /// Batch multiple updates together
    pub batching_enabled: bool,
    pub batch_size: usize,
    /// Prioritize critical parameters
    pub parameter_prioritization: bool,
    /// Adaptive bandwidth allocation
    pub adaptive_bandwidth: bool,
}

/// Adaptive compression configuration
#[derive(Debug, Clone)]
pub struct AdaptiveCompressionConfig {
    pub target_bandwidth_utilization: f64,
    pub quality_vs_speed_tradeoff: f64,
    pub adaptation_window: Duration,
}

impl Default for AdaptiveCompressionConfig {
    fn default() -> Self {
        Self {
            target_bandwidth_utilization: 0.8,
            quality_vs_speed_tradeoff: 0.5,
            adaptation_window: Duration::from_secs(60),
        }
    }
}
