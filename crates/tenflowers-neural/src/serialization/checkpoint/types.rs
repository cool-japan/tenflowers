//! Checkpoint types: CheckpointInfo, CheckpointLoadResult, CheckpointConfig.

use super::super::{CompressionAlgorithm, ModelMetadata};
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Checkpoint information and metadata
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    /// Checkpoint identifier
    pub checkpoint_id: String,
    /// Training epoch when checkpoint was created
    pub epoch: u32,
    /// Training step when checkpoint was created
    pub step: u64,
    /// Training loss at checkpoint
    pub loss: f32,
    /// Validation metrics at checkpoint
    pub validation_metrics: HashMap<String, f32>,
    /// Timestamp when checkpoint was created
    pub timestamp: String,
    /// Optimizer state included
    pub has_optimizer_state: bool,
    /// Learning rate at checkpoint
    pub learning_rate: f32,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Checkpoint load result
#[derive(Debug)]
pub struct CheckpointLoadResult {
    /// Checkpoint information
    pub checkpoint_info: CheckpointInfo,
    /// Model metadata
    pub model_metadata: ModelMetadata,
    /// Whether optimizer state was loaded
    pub optimizer_state_loaded: bool,
    /// Warnings during loading
    pub warnings: Vec<String>,
}

/// Checkpoint configuration
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Directory to save checkpoints
    pub checkpoint_dir: PathBuf,
    /// Maximum number of checkpoints to keep
    pub max_checkpoints: usize,
    /// Save frequency (in epochs)
    pub save_frequency: u32,
    /// Enable compression
    pub compression: bool,
    /// Compression algorithm
    pub compression_algorithm: CompressionAlgorithm,
    /// Save optimizer state
    pub save_optimizer_state: bool,
    /// Checkpoint name pattern
    pub name_pattern: String,
    /// Enable automatic cleanup
    pub auto_cleanup: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("checkpoints"),
            max_checkpoints: 5,
            save_frequency: 1,
            compression: true,
            compression_algorithm: CompressionAlgorithm::Zstd,
            save_optimizer_state: true,
            name_pattern: "checkpoint_epoch_{epoch}_step_{step}".to_string(),
            auto_cleanup: true,
        }
    }
}
