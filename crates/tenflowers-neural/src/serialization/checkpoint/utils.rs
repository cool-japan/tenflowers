//! Checkpoint utilities.

use super::types::CheckpointInfo;
use std::path::Path;
use tenflowers_core::{Result, TensorError};

/// Checkpoint utilities
pub mod utils {
    use super::*;
    use std::collections::HashMap;

    /// Create a checkpoint info
    pub fn create_checkpoint_info(
        epoch: u32,
        step: u64,
        loss: f32,
        learning_rate: f32,
    ) -> CheckpointInfo {
        CheckpointInfo {
            checkpoint_id: format!("checkpoint_{}_{}", epoch, step),
            epoch,
            step,
            loss,
            validation_metrics: HashMap::new(),
            timestamp: super::super::super::utils::get_timestamp(),
            has_optimizer_state: true,
            learning_rate,
            metadata: HashMap::new(),
        }
    }

    /// Add validation metric to checkpoint info
    pub fn add_validation_metric(
        checkpoint_info: &mut CheckpointInfo,
        metric_name: &str,
        value: f32,
    ) {
        checkpoint_info
            .validation_metrics
            .insert(metric_name.to_string(), value);
    }

    /// Get checkpoint directory size
    pub fn get_checkpoint_directory_size(checkpoint_dir: &Path) -> Result<u64> {
        if !checkpoint_dir.exists() {
            return Ok(0);
        }

        let mut total_size = 0;
        let entries = std::fs::read_dir(checkpoint_dir).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "Failed to read checkpoint directory: {}",
                e
            ))
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                TensorError::serialization_error_simple(format!(
                    "Failed to read directory entry: {}",
                    e
                ))
            })?;
            let metadata = entry.metadata().map_err(|e| {
                TensorError::serialization_error_simple(format!(
                    "Failed to get file metadata: {}",
                    e
                ))
            })?;

            if metadata.is_file() {
                total_size += metadata.len();
            }
        }

        Ok(total_size)
    }

    /// Find checkpoint by ID
    pub fn find_checkpoint_by_id<'a>(
        checkpoints: &'a [CheckpointInfo],
        checkpoint_id: &str,
    ) -> Option<&'a CheckpointInfo> {
        checkpoints
            .iter()
            .find(|c| c.checkpoint_id == checkpoint_id)
    }

    /// Find best checkpoint by loss
    pub fn find_best_checkpoint_by_loss(checkpoints: &[CheckpointInfo]) -> Option<&CheckpointInfo> {
        checkpoints.iter().min_by(|a, b| {
            a.loss
                .partial_cmp(&b.loss)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Find latest checkpoint by timestamp
    pub fn find_latest_checkpoint(checkpoints: &[CheckpointInfo]) -> Option<&CheckpointInfo> {
        checkpoints
            .iter()
            .max_by(|a, b| a.timestamp.cmp(&b.timestamp))
    }
}
