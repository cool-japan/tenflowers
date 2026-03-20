//! Data types for parameter server implementation.

use crate::tape::TensorId;
use std::time::Instant;

/// Parameter entry in the server
#[derive(Debug)]
pub struct ParameterEntry {
    /// Current parameter value
    pub parameter: Box<dyn std::any::Any + Send + Sync>,
    /// Version number for staleness detection
    pub version: u64,
    /// Last update timestamp
    pub last_updated: Instant,
    /// Number of pending updates
    pub pending_updates: usize,
    /// Assigned worker for this parameter
    #[allow(dead_code)]
    pub assigned_worker: Option<usize>,
}

/// Gradient update from a worker
#[derive(Debug)]
pub struct GradientUpdate {
    /// Tensor ID for the parameter
    pub tensor_id: TensorId,
    /// Gradient data
    #[allow(dead_code)]
    pub gradient: Box<dyn std::any::Any + Send + Sync>,
    /// Worker ID
    pub worker_id: usize,
    /// Timestamp when gradient was computed
    #[allow(dead_code)]
    pub timestamp: Instant,
    /// Version number when gradient was computed
    pub parameter_version: u64,
}

/// Worker status for health monitoring and load balancing
#[derive(Debug, Clone)]
pub struct WorkerStatus {
    /// Worker ID
    #[allow(dead_code)]
    pub worker_id: usize,
    /// Whether worker is alive
    pub is_alive: bool,
    /// Last heartbeat timestamp
    pub last_heartbeat: Instant,
    /// Current computational load (0.0 = idle, 1.0 = fully loaded)
    pub computational_load: f64,
    /// Number of assigned parameters
    pub assigned_parameters: usize,
    /// Number of pending gradient updates
    pub pending_gradients: usize,
    /// Worker capacity (relative processing power)
    pub capacity: f64,
    /// Communication latency to this worker
    pub latency_ms: f64,
}

/// Parameter server statistics
#[derive(Debug, Clone)]
pub struct ParameterServerStats {
    /// Total number of parameters managed
    pub total_parameters: usize,
    /// Total gradient updates processed
    pub total_updates: u64,
    /// Number of stale updates discarded
    pub stale_updates: u64,
    /// Number of worker failures detected
    pub worker_failures: u64,
    /// Average update latency in milliseconds
    pub avg_update_latency_ms: f64,
    /// Current server load (0.0 = idle, 1.0 = fully loaded)
    pub server_load: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
}
