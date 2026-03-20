//! High-level public API functions for model pruning.

use super::engine::ModelPruner;
use super::types::{PruningConfig, PruningScope, PruningStats, PruningStrategy};
use crate::model::Sequential;
use tenflowers_core::TensorError;

/// High-level API for model pruning.
pub fn prune_model<T>(
    model: &Sequential<T>,
    config: Option<PruningConfig>,
) -> Result<(Sequential<T>, PruningStats), TensorError>
where
    T: Clone
        + Default
        + Send
        + Sync
        + scirs2_core::num_traits::Zero
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let pruner = ModelPruner::with_config(config.unwrap_or_default());
    pruner.prune_sequential(model)
}

/// Create a pruning configuration optimized for mobile devices.
pub fn mobile_pruning_config() -> PruningConfig {
    PruningConfig {
        strategy: PruningStrategy::Magnitude,
        scope: PruningScope::Global,
        target_sparsity: 0.6, // 60% sparsity for mobile
        skip_layers: vec![
            "output".to_string(),
            "softmax".to_string(),
            "sigmoid".to_string(),
        ],
        gradual_pruning: false, // Post-training pruning for mobile
        pruning_steps: 1,
        accuracy_threshold: Some(0.03), // 3% tolerance for mobile
        fine_tune: true,
    }
}

/// Create a pruning configuration optimized for edge devices.
pub fn edge_pruning_config() -> PruningConfig {
    PruningConfig {
        strategy: PruningStrategy::Structured,
        scope: PruningScope::ChannelWise,
        target_sparsity: 0.8, // 80% sparsity for edge (aggressive)
        skip_layers: vec!["output".to_string()], // Minimize skipped layers
        gradual_pruning: true, // More sophisticated pruning for edge
        pruning_steps: 5,
        accuracy_threshold: Some(0.05), // 5% tolerance for edge
        fine_tune: true,
    }
}

/// Create a conservative pruning configuration.
pub fn conservative_pruning_config() -> PruningConfig {
    PruningConfig {
        strategy: PruningStrategy::Magnitude,
        scope: PruningScope::LayerWise,
        target_sparsity: 0.3, // 30% sparsity (conservative)
        skip_layers: vec![
            "output".to_string(),
            "softmax".to_string(),
            "sigmoid".to_string(),
            "tanh".to_string(),
        ],
        gradual_pruning: true,
        pruning_steps: 10,              // Very gradual
        accuracy_threshold: Some(0.01), // 1% tolerance (strict)
        fine_tune: true,
    }
}
