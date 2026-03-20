//! Replication Operations and Payloads
//!
//! This module defines the various types of operations that can be performed
//! across datacenters, including parameter synchronization, gradient updates,
//! and status monitoring operations.

use super::config::CompressionAlgorithm;
use crate::TrackedTensor;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Replication operation for cross-datacenter sync
#[derive(Debug, Clone)]
pub struct ReplicationOperation {
    pub operation_id: String,
    pub operation_type: OperationType,
    pub source_datacenter: String,
    pub target_datacenters: Vec<String>,
    pub payload: ReplicationPayload,
    pub priority: Priority,
    pub deadline: Option<Instant>,
    pub retry_count: usize,
}

/// Types of replication operations
#[derive(Debug, Clone)]
pub enum OperationType {
    ParameterSync,
    GradientSync,
    ModelSync,
    StateSync,
    HeartBeat,
    TopologyUpdate,
}

/// Payload data for replication operations
#[derive(Debug, Clone)]
pub enum ReplicationPayload {
    Parameters {
        tensors: Vec<TrackedTensor<f32>>,
        metadata: ParameterMetadata,
    },
    Gradients {
        gradients: Vec<TrackedTensor<f32>>,
        step_number: usize,
    },
    ModelState {
        model_data: Vec<u8>,
        version: String,
    },
    Heartbeat {
        timestamp: Instant,
        status: DatacenterStatus,
    },
}

/// Metadata associated with parameter tensors
#[derive(Debug, Clone)]
pub struct ParameterMetadata {
    pub model_version: String,
    pub step_number: usize,
    pub checksum: String,
    pub compression_info: CompressionInfo,
}

/// Information about compression applied to data
#[derive(Debug, Clone)]
pub struct CompressionInfo {
    pub algorithm: CompressionAlgorithm,
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_time: Duration,
}

/// Priority levels for replication operations
#[derive(Debug, Clone)]
pub enum Priority {
    Critical, // Model updates, critical gradients
    High,     // Important parameter syncs
    Medium,   // Regular gradient updates
    Low,      // Heartbeats, status updates
}

/// Status information for a datacenter
#[derive(Debug, Clone)]
pub struct DatacenterStatus {
    pub compute_utilization: f64,
    pub memory_utilization: f64,
    pub network_utilization: f64,
    pub active_training_jobs: usize,
    pub health_score: f64,
}
