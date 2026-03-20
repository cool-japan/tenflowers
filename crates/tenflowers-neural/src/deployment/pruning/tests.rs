//! Tests for pruning module.

#[cfg(test)]
mod tests {
    use super::super::api::{
        conservative_pruning_config, edge_pruning_config, mobile_pruning_config, prune_model,
    };
    use super::super::engine::ModelPruner;
    use super::super::types::{
        PrunedLayer, PruningConfig, PruningMask, PruningScope, PruningStats, PruningStrategy,
    };
    use crate::layers::Dense;
    use crate::model::{Model, Sequential};
    use tenflowers_core::Tensor;

    #[test]
    fn test_pruning_config_default() {
        let config = PruningConfig::default();
        assert_eq!(config.strategy, PruningStrategy::Magnitude);
        assert_eq!(config.scope, PruningScope::Global);
        assert_eq!(config.target_sparsity, 0.5);
        assert!(config.fine_tune);
    }

    #[test]
    fn test_pruning_stats() {
        let mut stats = PruningStats::new();
        stats.original_params = 1000;
        stats.remaining_params = 600;
        stats.pruned_params = 400;

        assert_eq!(stats.param_reduction_ratio(), 0.4);
        assert_eq!(stats.original_params, 1000);
        assert_eq!(stats.remaining_params, 600);
    }

    #[test]
    fn test_pruning_mask() {
        let mask_tensor = Tensor::ones(&[5, 5]);
        let mask = PruningMask::new("layer1".to_string(), mask_tensor, 0.5);

        assert_eq!(mask.layer_name, "layer1");
        assert_eq!(mask.sparsity, 0.5);

        let input = Tensor::ones(&[5, 5]);
        let result = mask.apply(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pruned_layer_creation() {
        let mask_tensor = Tensor::ones(&[3, 3]);
        let mask = PruningMask::new("layer1".to_string(), mask_tensor, 0.3);

        let layer = PrunedLayer::<f32>::new("dense1".to_string(), mask, vec![], vec![10], vec![20]);

        assert_eq!(layer.layer_name(), "dense1");
        assert_eq!(layer.sparsity(), 0.3);
    }

    #[test]
    fn test_model_pruner() {
        let pruner = ModelPruner::new();
        assert_eq!(pruner.config.strategy, PruningStrategy::Magnitude);

        let custom_config = PruningConfig {
            strategy: PruningStrategy::Structured,
            target_sparsity: 0.7,
            ..Default::default()
        };
        let custom_pruner = ModelPruner::with_config(custom_config);
        assert_eq!(custom_pruner.config.strategy, PruningStrategy::Structured);
        assert_eq!(custom_pruner.config.target_sparsity, 0.7);
    }

    #[test]
    fn test_sequential_pruning() {
        let model = Sequential::new(vec![
            Box::new(Dense::<f32>::new(10, 20, true)),
            Box::new(Dense::<f32>::new(20, 1, true)),
        ]);

        let result = prune_model(&model, None);
        assert!(result.is_ok());

        let (_pruned_model, stats) = result.expect("test: result should be valid");
        assert!(stats.layers_pruned > 0);
        assert!(stats.achieved_sparsity > 0.0);
        assert!(stats.param_reduction_ratio() > 0.0);
        assert!(stats.inference_speedup >= 1.0);
    }

    #[test]
    fn test_mobile_pruning_config() {
        let config = mobile_pruning_config();
        assert_eq!(config.strategy, PruningStrategy::Magnitude);
        assert_eq!(config.target_sparsity, 0.6);
        assert!(!config.gradual_pruning);
        assert_eq!(config.accuracy_threshold, Some(0.03));
    }

    #[test]
    fn test_edge_pruning_config() {
        let config = edge_pruning_config();
        assert_eq!(config.strategy, PruningStrategy::Structured);
        assert_eq!(config.scope, PruningScope::ChannelWise);
        assert_eq!(config.target_sparsity, 0.8);
        assert!(config.gradual_pruning);
        assert_eq!(config.pruning_steps, 5);
    }

    #[test]
    fn test_conservative_pruning_config() {
        let config = conservative_pruning_config();
        assert_eq!(config.target_sparsity, 0.3);
        assert_eq!(config.accuracy_threshold, Some(0.01));
        assert_eq!(config.pruning_steps, 10);
    }

    #[test]
    fn test_mask_generation() {
        let model = Sequential::new(vec![
            Box::new(Dense::<f32>::new(5, 10, true)),
            Box::new(Dense::<f32>::new(10, 1, true)),
        ]);

        let pruner = ModelPruner::new();
        let result = pruner.generate_masks(&model);
        assert!(result.is_ok());

        let masks = result.expect("test: result should be valid");
        assert!(!masks.is_empty());
        assert_eq!(masks.len(), model.parameters().len());
    }

    #[test]
    #[cfg(feature = "serialize")]
    fn test_pruning_serialization() {
        let strategy = PruningStrategy::Structured;
        let serialized = serde_json::to_string(&strategy).expect("test: operation should succeed");
        let deserialized: PruningStrategy =
            serde_json::from_str(&serialized).expect("test: operation should succeed");
        assert_eq!(strategy, deserialized);
    }
}
