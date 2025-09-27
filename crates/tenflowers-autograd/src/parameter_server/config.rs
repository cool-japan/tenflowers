//! Configuration types for parameter server.

/// Configuration for parameter server
#[derive(Debug, Clone)]
pub struct ParameterServerConfig {
    /// Number of worker nodes
    pub num_workers: usize,
    /// Staleness tolerance for asynchronous updates
    pub staleness_threshold: usize,
    /// Heartbeat timeout for fault detection (in milliseconds)
    pub heartbeat_timeout_ms: u64,
    /// Maximum gradient queue size per worker
    pub max_queue_size: usize,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Fault tolerance mode
    pub fault_tolerance: FaultToleranceMode,
    /// Number of backup parameter servers
    pub num_backups: usize,
}

/// Load balancing strategies
#[derive(Debug, Clone, PartialEq)]
pub enum LoadBalancingStrategy {
    /// Round-robin assignment of parameters to workers
    RoundRobin,
    /// Assign parameters based on worker computational capacity
    CapacityBased,
    /// Assign parameters based on current worker load
    LoadBased,
    /// Dynamic load balancing with migration
    Dynamic,
}

/// Fault tolerance modes
#[derive(Debug, Clone, PartialEq)]
pub enum FaultToleranceMode {
    /// No fault tolerance
    None,
    /// Checkpoint-based recovery
    Checkpoint,
    /// Replication-based fault tolerance
    Replication,
    /// Hybrid checkpoint + replication
    Hybrid,
}

impl Default for ParameterServerConfig {
    fn default() -> Self {
        Self {
            num_workers: 1,
            staleness_threshold: 10,
            heartbeat_timeout_ms: 5000,
            max_queue_size: 1000,
            load_balancing: LoadBalancingStrategy::LoadBased,
            fault_tolerance: FaultToleranceMode::Checkpoint,
            num_backups: 1,
        }
    }
}
