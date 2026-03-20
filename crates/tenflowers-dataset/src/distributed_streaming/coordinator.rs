//! Coordinator and worker management for distributed streaming

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tenflowers_core::{Result, TensorError};

use super::types::{CheckpointState, StreamingConfig, WorkerHealth, WorkerMetrics, WorkerStatus};

/// Multi-worker stream coordinator
pub struct StreamCoordinator {
    /// Configuration
    pub(super) config: StreamingConfig,
    /// Worker assignments
    pub(super) worker_assignments: Arc<RwLock<HashMap<usize, Vec<usize>>>>,
    /// Worker health status
    pub(super) worker_health: Arc<RwLock<HashMap<usize, WorkerHealth>>>,
    /// Global checkpoint registry
    pub(super) global_checkpoints: Arc<RwLock<HashMap<usize, CheckpointState>>>,
    /// Load balancing metrics
    pub(super) balancing_metrics: Arc<RwLock<HashMap<usize, WorkerMetrics>>>,
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

        let health = self
            .worker_health
            .read()
            .map_err(|e| TensorError::invalid_operation_simple(format!("Lock error: {}", e)))?;

        let workers: Vec<_> = health.values().collect();
        if workers.is_empty() {
            return Ok(false);
        }

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
