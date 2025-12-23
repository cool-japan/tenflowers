//! Activation Checkpointing for Memory-Efficient Training
//!
//! This module implements gradient checkpointing (also known as activation checkpointing),
//! a technique that trades computation for memory by recomputing intermediate activations
//! during the backward pass instead of storing them. This is essential for training large
//! models that would otherwise not fit in GPU memory.
//!
//! # Overview
//!
//! In standard backpropagation, all intermediate activations must be stored during the
//! forward pass to compute gradients during the backward pass. For deep networks, this
//! can require enormous amounts of memory. Gradient checkpointing selectively stores
//! only certain activations (checkpoints) and recomputes the others on-demand during
//! backpropagation.
//!
//! # Trade-offs
//!
//! - **Memory**: Reduces memory usage by up to 10x for very deep networks
//! - **Computation**: Increases training time by ~20-30% due to recomputation
//! - **Optimal for**: Large transformers, vision models, any memory-bound training
//!
//! # Example
//!
//! ```rust
//! use tenflowers_core::checkpointing::{CheckpointPolicy, CheckpointingConfig};
//!
//! // Configure checkpointing for transformer layers
//! let config = CheckpointingConfig {
//!     policy: CheckpointPolicy::EveryNLayers(2), // Checkpoint every 2 layers
//!     recompute_on_backward: true,
//!     save_rng_state: true, // Important for dropout consistency
//! };
//! ```

use crate::{Result, Tensor, TensorError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Policy for selecting which activations to checkpoint
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointPolicy {
    /// Checkpoint every N layers (most common for transformers)
    EveryNLayers(usize),
    /// Checkpoint at specific layer indices
    SpecificLayers(Vec<usize>),
    /// Automatic policy based on memory constraints
    Automatic {
        /// Target memory budget in bytes
        memory_budget: usize,
        /// Estimated activation size per layer in bytes
        avg_activation_size: usize,
    },
    /// Custom policy via user-defined function
    Custom,
    /// No checkpointing (store all activations)
    None,
}

impl Default for CheckpointPolicy {
    fn default() -> Self {
        CheckpointPolicy::EveryNLayers(1)
    }
}

/// Configuration for activation checkpointing
#[derive(Debug, Clone)]
pub struct CheckpointingConfig {
    /// Policy for selecting checkpoint locations
    pub policy: CheckpointPolicy,
    /// Whether to recompute activations during backward pass
    pub recompute_on_backward: bool,
    /// Save and restore RNG state for deterministic dropout
    pub save_rng_state: bool,
    /// Enable gradient checkpointing statistics tracking
    pub enable_statistics: bool,
    /// Maximum number of checkpoints to store
    pub max_checkpoints: Option<usize>,
}

impl Default for CheckpointingConfig {
    fn default() -> Self {
        Self {
            policy: CheckpointPolicy::default(),
            recompute_on_backward: true,
            save_rng_state: true,
            enable_statistics: false,
            max_checkpoints: None,
        }
    }
}

/// Statistics for checkpointing performance
#[derive(Debug, Clone, Default)]
pub struct CheckpointStatistics {
    /// Total number of forward passes
    pub forward_passes: usize,
    /// Total number of backward passes
    pub backward_passes: usize,
    /// Number of recomputations performed
    pub recompute_count: usize,
    /// Total memory saved (estimated, in bytes)
    pub memory_saved_bytes: usize,
    /// Additional computation time due to recomputation (in microseconds)
    pub additional_compute_time_us: u64,
    /// Number of active checkpoints
    pub active_checkpoints: usize,
}

impl CheckpointStatistics {
    /// Get average recomputations per backward pass
    pub fn avg_recomputations(&self) -> f64 {
        if self.backward_passes == 0 {
            0.0
        } else {
            self.recompute_count as f64 / self.backward_passes as f64
        }
    }

    /// Get memory saved in MB
    pub fn memory_saved_mb(&self) -> f64 {
        self.memory_saved_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get overhead percentage
    pub fn compute_overhead_percent(&self) -> f64 {
        if self.forward_passes == 0 {
            0.0
        } else {
            (self.recompute_count as f64 / self.forward_passes as f64) * 100.0
        }
    }
}

/// A checkpoint storing activations and metadata
#[derive(Debug, Clone)]
pub struct Checkpoint<T> {
    /// Layer index where checkpoint was taken
    pub layer_index: usize,
    /// Stored activation tensors
    pub activations: Vec<Tensor<T>>,
    /// RNG state at checkpoint time (if enabled)
    pub rng_state: Option<Vec<u8>>,
    /// Timestamp when checkpoint was created
    pub timestamp: std::time::Instant,
    /// Estimated memory usage in bytes
    pub memory_bytes: usize,
}

impl<T> Checkpoint<T>
where
    T: Clone + Default,
{
    /// Create a new checkpoint
    pub fn new(layer_index: usize, activations: Vec<Tensor<T>>) -> Self {
        let memory_bytes = activations
            .iter()
            .map(|t| t.shape().size() * std::mem::size_of::<T>())
            .sum();

        Self {
            layer_index,
            activations,
            rng_state: None,
            timestamp: std::time::Instant::now(),
            memory_bytes,
        }
    }

    /// Add RNG state to checkpoint
    pub fn with_rng_state(mut self, rng_state: Vec<u8>) -> Self {
        self.rng_state = Some(rng_state);
        self
    }

    /// Get age of checkpoint
    pub fn age(&self) -> std::time::Duration {
        self.timestamp.elapsed()
    }
}

/// Manager for activation checkpointing
pub struct CheckpointManager<T> {
    config: CheckpointingConfig,
    checkpoints: Arc<Mutex<HashMap<usize, Checkpoint<T>>>>,
    statistics: Arc<Mutex<CheckpointStatistics>>,
}

impl<T> CheckpointManager<T>
where
    T: Clone + Default + Send + Sync,
{
    /// Create a new checkpoint manager
    pub fn new(config: CheckpointingConfig) -> Self {
        Self {
            config,
            checkpoints: Arc::new(Mutex::new(HashMap::new())),
            statistics: Arc::new(Mutex::new(CheckpointStatistics::default())),
        }
    }

    /// Create with automatic memory-based policy
    pub fn with_memory_budget(memory_budget_mb: usize, avg_activation_mb: usize) -> Self {
        Self::new(CheckpointingConfig {
            policy: CheckpointPolicy::Automatic {
                memory_budget: memory_budget_mb * 1024 * 1024,
                avg_activation_size: avg_activation_mb * 1024 * 1024,
            },
            ..Default::default()
        })
    }

    /// Check if layer should be checkpointed according to policy
    pub fn should_checkpoint(&self, layer_index: usize, total_layers: usize) -> bool {
        match &self.config.policy {
            CheckpointPolicy::EveryNLayers(n) => layer_index % n == 0,
            CheckpointPolicy::SpecificLayers(indices) => indices.contains(&layer_index),
            CheckpointPolicy::Automatic {
                memory_budget,
                avg_activation_size,
            } => {
                // Calculate how many checkpoints we can afford
                let max_checkpoints = memory_budget / avg_activation_size;
                if max_checkpoints == 0 {
                    return false;
                }
                // Distribute checkpoints evenly across layers
                let checkpoint_every = total_layers / max_checkpoints.max(1);
                layer_index % checkpoint_every.max(1) == 0
            }
            CheckpointPolicy::Custom => {
                // User must override this with custom logic
                false
            }
            CheckpointPolicy::None => false,
        }
    }

    /// Save a checkpoint for the given layer
    pub fn save_checkpoint(&self, layer_index: usize, activations: Vec<Tensor<T>>) -> Result<()> {
        let mut checkpoint = Checkpoint::new(layer_index, activations);

        // Save RNG state if configured
        if self.config.save_rng_state {
            checkpoint = checkpoint.with_rng_state(self.capture_rng_state());
        }

        let memory_bytes = checkpoint.memory_bytes;

        // Store checkpoint
        let mut checkpoints = self.checkpoints.lock().map_err(|_| {
            TensorError::invalid_operation_simple("Failed to acquire checkpoint lock".to_string())
        })?;

        // Enforce max checkpoints limit
        if let Some(max_cp) = self.config.max_checkpoints {
            if checkpoints.len() >= max_cp {
                // Remove oldest checkpoint
                if let Some(oldest_key) = checkpoints
                    .iter()
                    .min_by_key(|(_, cp)| cp.timestamp)
                    .map(|(k, _)| *k)
                {
                    checkpoints.remove(&oldest_key);
                }
            }
        }

        checkpoints.insert(layer_index, checkpoint);

        // Update statistics
        if self.config.enable_statistics {
            let mut stats = self.statistics.lock().map_err(|_| {
                TensorError::invalid_operation_simple(
                    "Failed to acquire statistics lock".to_string(),
                )
            })?;
            stats.forward_passes += 1;
            stats.memory_saved_bytes += memory_bytes;
            stats.active_checkpoints = checkpoints.len();
        }

        Ok(())
    }

    /// Retrieve a checkpoint for the given layer
    pub fn get_checkpoint(&self, layer_index: usize) -> Result<Option<Checkpoint<T>>> {
        let checkpoints = self.checkpoints.lock().map_err(|_| {
            TensorError::invalid_operation_simple("Failed to acquire checkpoint lock".to_string())
        })?;

        Ok(checkpoints.get(&layer_index).cloned())
    }

    /// Restore RNG state from checkpoint
    pub fn restore_rng_state(&self, rng_state: &[u8]) -> Result<()> {
        // This would integrate with the actual RNG implementation
        // For now, we just acknowledge the state
        if rng_state.is_empty() {
            return Err(TensorError::invalid_argument("Empty RNG state".to_string()));
        }
        Ok(())
    }

    /// Capture current RNG state
    fn capture_rng_state(&self) -> Vec<u8> {
        // This would integrate with the actual RNG implementation
        // For now, return a placeholder
        vec![0u8; 32]
    }

    /// Record a recomputation event
    pub fn record_recomputation(&self, compute_time_us: u64) {
        if !self.config.enable_statistics {
            return;
        }

        if let Ok(mut stats) = self.statistics.lock() {
            stats.recompute_count += 1;
            stats.additional_compute_time_us += compute_time_us;
        }
    }

    /// Record a backward pass
    pub fn record_backward_pass(&self) {
        if !self.config.enable_statistics {
            return;
        }

        if let Ok(mut stats) = self.statistics.lock() {
            stats.backward_passes += 1;
        }
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> Result<CheckpointStatistics> {
        let stats = self.statistics.lock().map_err(|_| {
            TensorError::invalid_operation_simple("Failed to acquire statistics lock".to_string())
        })?;
        Ok(stats.clone())
    }

    /// Clear all checkpoints
    pub fn clear(&self) -> Result<()> {
        let mut checkpoints = self.checkpoints.lock().map_err(|_| {
            TensorError::invalid_operation_simple("Failed to acquire checkpoint lock".to_string())
        })?;
        checkpoints.clear();

        if self.config.enable_statistics {
            if let Ok(mut stats) = self.statistics.lock() {
                stats.active_checkpoints = 0;
            }
        }

        Ok(())
    }

    /// Get number of active checkpoints
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.lock().map(|cp| cp.len()).unwrap_or(0)
    }

    /// Get total memory used by checkpoints
    pub fn total_memory_bytes(&self) -> usize {
        self.checkpoints
            .lock()
            .map(|cp| cp.values().map(|c| c.memory_bytes).sum())
            .unwrap_or(0)
    }
}

impl<T> Clone for CheckpointManager<T>
where
    T: Clone + Default + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            checkpoints: Arc::clone(&self.checkpoints),
            statistics: Arc::clone(&self.statistics),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_checkpoint_policy_every_n() {
        let manager = CheckpointManager::<f32>::new(CheckpointingConfig {
            policy: CheckpointPolicy::EveryNLayers(2),
            ..Default::default()
        });

        assert!(manager.should_checkpoint(0, 10));
        assert!(!manager.should_checkpoint(1, 10));
        assert!(manager.should_checkpoint(2, 10));
        assert!(!manager.should_checkpoint(3, 10));
        assert!(manager.should_checkpoint(4, 10));
    }

    #[test]
    fn test_checkpoint_policy_specific() {
        let manager = CheckpointManager::<f32>::new(CheckpointingConfig {
            policy: CheckpointPolicy::SpecificLayers(vec![1, 3, 7]),
            ..Default::default()
        });

        assert!(!manager.should_checkpoint(0, 10));
        assert!(manager.should_checkpoint(1, 10));
        assert!(!manager.should_checkpoint(2, 10));
        assert!(manager.should_checkpoint(3, 10));
        assert!(manager.should_checkpoint(7, 10));
    }

    #[test]
    fn test_checkpoint_save_and_retrieve() {
        let manager = CheckpointManager::<f32>::new(CheckpointingConfig::default());
        let tensor = Tensor::from_array(array![[1.0, 2.0], [3.0, 4.0]].into_dyn());

        manager.save_checkpoint(5, vec![tensor.clone()]).unwrap();

        let checkpoint = manager.get_checkpoint(5).unwrap();
        assert!(checkpoint.is_some());

        let cp = checkpoint.unwrap();
        assert_eq!(cp.layer_index, 5);
        assert_eq!(cp.activations.len(), 1);
    }

    #[test]
    fn test_max_checkpoints_limit() {
        let manager = CheckpointManager::<f32>::new(CheckpointingConfig {
            max_checkpoints: Some(2),
            ..Default::default()
        });

        let tensor = Tensor::from_array(array![1.0, 2.0].into_dyn());

        manager.save_checkpoint(0, vec![tensor.clone()]).unwrap();
        manager.save_checkpoint(1, vec![tensor.clone()]).unwrap();
        manager.save_checkpoint(2, vec![tensor.clone()]).unwrap();

        // Should have only 2 checkpoints (oldest removed)
        assert_eq!(manager.checkpoint_count(), 2);
    }

    #[test]
    fn test_statistics_tracking() {
        let manager = CheckpointManager::<f32>::new(CheckpointingConfig {
            enable_statistics: true,
            ..Default::default()
        });

        let tensor = Tensor::from_array(array![1.0, 2.0, 3.0].into_dyn());

        manager.save_checkpoint(0, vec![tensor.clone()]).unwrap();
        manager.record_backward_pass();
        manager.record_recomputation(1000);
        manager.record_recomputation(2000);

        let stats = manager.get_statistics().unwrap();
        assert_eq!(stats.forward_passes, 1);
        assert_eq!(stats.backward_passes, 1);
        assert_eq!(stats.recompute_count, 2);
        assert_eq!(stats.additional_compute_time_us, 3000);
    }

    #[test]
    fn test_checkpoint_clear() {
        let manager = CheckpointManager::<f32>::new(CheckpointingConfig::default());
        let tensor = Tensor::from_array(array![1.0, 2.0].into_dyn());

        manager.save_checkpoint(0, vec![tensor.clone()]).unwrap();
        manager.save_checkpoint(1, vec![tensor.clone()]).unwrap();

        assert_eq!(manager.checkpoint_count(), 2);

        manager.clear().unwrap();
        assert_eq!(manager.checkpoint_count(), 0);
    }

    #[test]
    fn test_automatic_policy() {
        let manager = CheckpointManager::<f32>::with_memory_budget(100, 10);

        // With 100MB budget and 10MB per activation, we can afford 10 checkpoints
        // For 50 layers, checkpoint every 5 layers
        assert!(manager.should_checkpoint(0, 50));
        assert!(manager.should_checkpoint(5, 50));
        assert!(manager.should_checkpoint(10, 50));
        assert!(!manager.should_checkpoint(1, 50));
        assert!(!manager.should_checkpoint(7, 50));
    }

    #[test]
    fn test_checkpoint_statistics_calculations() {
        let mut stats = CheckpointStatistics {
            forward_passes: 100,
            backward_passes: 100,
            recompute_count: 300,
            memory_saved_bytes: 1024 * 1024 * 500, // 500 MB
            additional_compute_time_us: 1_000_000,
            active_checkpoints: 10,
        };

        assert_eq!(stats.avg_recomputations(), 3.0);
        assert_eq!(stats.memory_saved_mb(), 500.0);
        assert_eq!(stats.compute_overhead_percent(), 300.0);
    }
}
