//! Model pruning techniques for mobile deployment.
//!
//! This module provides various pruning methods to reduce model size and computational
//! complexity by removing redundant or less important parameters and connections.
//!
//! Submodules:
//! - `types`: Enums, config, stats, mask, and PrunedLayer
//! - `engine`: ModelPruner implementation
//! - `api`: High-level public API functions

pub mod api;
pub mod engine;
pub mod types;

mod tests;

pub use api::{
    conservative_pruning_config, edge_pruning_config, mobile_pruning_config, prune_model,
};
pub use engine::ModelPruner;
pub use types::{
    PrunedLayer, PruningConfig, PruningMask, PruningScope, PruningStats, PruningStrategy,
};
