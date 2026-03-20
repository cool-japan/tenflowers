//! Checkpoint management for model training.
//!
//! This module provides comprehensive checkpoint management for training,
//! including automatic checkpointing, recovery, and checkpoint optimization.
//!
//! Submodules:
//! - `types`: CheckpointInfo, CheckpointLoadResult, CheckpointConfig
//! - `manager`: CheckpointManager implementation
//! - `utils`: Checkpoint utility functions

pub mod manager;
pub mod types;
pub mod utils;

mod tests;

pub use manager::CheckpointManager;
pub use types::{CheckpointConfig, CheckpointInfo, CheckpointLoadResult};
pub use utils::utils::{
    add_validation_metric, create_checkpoint_info, find_best_checkpoint_by_loss,
    find_checkpoint_by_id, find_latest_checkpoint, get_checkpoint_directory_size,
};
