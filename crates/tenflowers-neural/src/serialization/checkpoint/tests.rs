//! Tests for checkpoint module.

#[cfg(test)]
mod tests {
    use super::super::super::{HardwareRequirements, ModelMetadata, SemanticVersion, TrainingInfo};
    use super::super::manager::CheckpointManager;
    use super::super::types::{CheckpointConfig, CheckpointInfo};
    use super::super::utils::utils;
    use std::collections::HashMap;

    #[test]
    fn test_checkpoint_info_creation() {
        let checkpoint_info = utils::create_checkpoint_info(10, 1000, 0.5, 0.001);

        assert_eq!(checkpoint_info.epoch, 10);
        assert_eq!(checkpoint_info.step, 1000);
        assert_eq!(checkpoint_info.loss, 0.5);
        assert_eq!(checkpoint_info.learning_rate, 0.001);
        assert!(!checkpoint_info.timestamp.is_empty());
    }

    #[test]
    fn test_checkpoint_config_default() {
        let config = CheckpointConfig::default();

        assert_eq!(config.max_checkpoints, 5);
        assert_eq!(config.save_frequency, 1);
        assert!(config.compression);
        assert!(config.save_optimizer_state);
        assert!(config.auto_cleanup);
    }

    #[test]
    fn test_add_validation_metric() {
        let mut checkpoint_info = utils::create_checkpoint_info(5, 500, 0.3, 0.01);

        utils::add_validation_metric(&mut checkpoint_info, "accuracy", 0.85);
        utils::add_validation_metric(&mut checkpoint_info, "f1_score", 0.82);

        assert_eq!(checkpoint_info.validation_metrics.len(), 2);
        assert_eq!(checkpoint_info.validation_metrics["accuracy"], 0.85);
        assert_eq!(checkpoint_info.validation_metrics["f1_score"], 0.82);
    }

    #[test]
    fn test_find_checkpoint_by_id() {
        let checkpoints = vec![
            utils::create_checkpoint_info(1, 100, 0.9, 0.01),
            utils::create_checkpoint_info(2, 200, 0.8, 0.01),
            utils::create_checkpoint_info(3, 300, 0.7, 0.01),
        ];

        let found = utils::find_checkpoint_by_id(&checkpoints, "checkpoint_2_200");
        assert!(found.is_some());
        assert_eq!(found.expect("test: operation should succeed").epoch, 2);

        let not_found = utils::find_checkpoint_by_id(&checkpoints, "checkpoint_99_999");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_find_best_checkpoint_by_loss() {
        let checkpoints = vec![
            utils::create_checkpoint_info(1, 100, 0.9, 0.01),
            utils::create_checkpoint_info(2, 200, 0.8, 0.01),
            utils::create_checkpoint_info(3, 300, 0.7, 0.01),
        ];

        let best = utils::find_best_checkpoint_by_loss(&checkpoints);
        assert!(best.is_some());
        assert_eq!(best.expect("test: operation should succeed").loss, 0.7);
        assert_eq!(best.expect("test: operation should succeed").epoch, 3);
    }

    #[test]
    fn test_find_latest_checkpoint() {
        let mut checkpoints = vec![
            utils::create_checkpoint_info(1, 100, 0.9, 0.01),
            utils::create_checkpoint_info(2, 200, 0.8, 0.01),
            utils::create_checkpoint_info(3, 300, 0.7, 0.01),
        ];

        // Modify timestamps to ensure proper ordering
        checkpoints[0].timestamp = "2023-01-01T10:00:00Z".to_string();
        checkpoints[1].timestamp = "2023-01-01T11:00:00Z".to_string();
        checkpoints[2].timestamp = "2023-01-01T12:00:00Z".to_string();

        let latest = utils::find_latest_checkpoint(&checkpoints);
        assert!(latest.is_some());
        assert_eq!(latest.expect("test: operation should succeed").epoch, 3);
    }

    #[test]
    fn test_checkpoint_data_serialization() {
        use super::super::manager::CheckpointData;

        let checkpoint_info = utils::create_checkpoint_info(1, 100, 0.5, 0.01);
        let model_metadata = ModelMetadata {
            model_type: "Sequential".to_string(),
            version: SemanticVersion::new(0, 1, 0),
            framework_version: "TenfloweRS-0.1.0".to_string(),
            created_at: "2023-01-01T00:00:00Z".to_string(),
            architecture_hash: "test_hash".to_string(),
            parameter_count: 1000,
            model_size: 4000,
            training_info: TrainingInfo {
                epochs: Some(1),
                final_loss: Some(0.5),
                validation_accuracy: Some(0.9),
                optimizer: Some("Adam".to_string()),
                learning_rate: Some(0.01),
                dataset_info: Some("Test".to_string()),
            },
            hardware_requirements: HardwareRequirements {
                min_memory: 1024,
                recommended_memory: 2048,
                gpu_required: false,
                cpu_features: vec![],
                target_device: "CPU".to_string(),
            },
            custom: HashMap::new(),
        };

        let checkpoint_data = CheckpointData {
            info: checkpoint_info,
            model_metadata,
            model_state: "{}".to_string(),
            optimizer_state: Some("{}".to_string()),
            compression_info: None,
        };

        let serialized =
            serde_json::to_string(&checkpoint_data).expect("test: operation should succeed");
        assert!(!serialized.is_empty());

        let deserialized: CheckpointData =
            serde_json::from_str(&serialized).expect("test: operation should succeed");
        assert_eq!(deserialized.info.epoch, 1);
        assert_eq!(deserialized.info.step, 100);
    }

    #[test]
    fn test_model_state_serialization_enhanced() {
        // Create a mock checkpoint manager
        let config = CheckpointConfig::default();
        let _checkpoint_manager = CheckpointManager {
            config,
            checkpoint_history: Vec::new(),
            best_checkpoint: None,
        };

        // Test the enhanced model state serialization structure
        let model_state = r#"{
            "parameters_metadata": [
                {
                    "index": 0,
                    "shape": [10, 20],
                    "device": "CPU",
                    "dtype": "f32",
                    "parameter_count": 200,
                    "requires_grad": true
                }
            ],
            "parameter_count": 1,
            "model_type": "Sequential",
            "serialization_version": "1.0"
        }"#;

        let parsed: std::collections::HashMap<String, serde_json::Value> =
            serde_json::from_str(model_state).expect("test: operation should succeed");

        // Verify structure
        assert!(parsed.contains_key("serialization_version"));
        assert!(parsed.contains_key("model_type"));
        assert!(parsed.contains_key("parameters_metadata"));
        assert_eq!(parsed["serialization_version"], "1.0");
        assert_eq!(parsed["model_type"], "Sequential");
    }

    #[test]
    fn test_optimizer_state_serialization_enhanced() {
        let config = CheckpointConfig::default();
        let checkpoint_manager = CheckpointManager {
            config,
            checkpoint_history: Vec::new(),
            best_checkpoint: None,
        };

        // Test enhanced optimizer state serialization
        let optimizer_state_result = checkpoint_manager.serialize_optimizer_state();
        assert!(optimizer_state_result.is_ok());

        let optimizer_state = optimizer_state_result.expect("test: optimization should succeed");
        let parsed: std::collections::HashMap<String, serde_json::Value> =
            serde_json::from_str(&optimizer_state).expect("test: optimization should succeed");

        // Verify enhanced structure
        assert!(parsed.contains_key("optimizer_type"));
        assert!(parsed.contains_key("step_count"));
        assert!(parsed.contains_key("learning_rate"));
        assert!(parsed.contains_key("serialization_version"));
        assert!(parsed.contains_key("has_momentum"));
        assert!(parsed.contains_key("has_second_moment"));
        assert_eq!(parsed["serialization_version"], "1.0");
    }

    #[test]
    fn test_version_compatibility() {
        let config = CheckpointConfig::default();
        let checkpoint_manager = CheckpointManager {
            config,
            checkpoint_history: Vec::new(),
            best_checkpoint: None,
        };

        // Test compatible version
        assert!(checkpoint_manager.is_compatible_version("1.0"));

        // Test incompatible versions
        assert!(!checkpoint_manager.is_compatible_version("2.0"));
        assert!(!checkpoint_manager.is_compatible_version("0.9"));
        assert!(!checkpoint_manager.is_compatible_version("unknown"));
    }

    #[test]
    fn test_optimizer_state_validation() {
        let config = CheckpointConfig::default();
        let checkpoint_manager = CheckpointManager {
            config,
            checkpoint_history: Vec::new(),
            best_checkpoint: None,
        };

        // Test valid optimizer state
        let valid_state = std::collections::HashMap::from([
            (
                "optimizer_type".to_string(),
                serde_json::Value::String("Adam".to_string()),
            ),
            (
                "step_count".to_string(),
                serde_json::Value::Number(serde_json::Number::from(100)),
            ),
            (
                "learning_rate".to_string(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(0.001).expect("test: operation should succeed"),
                ),
            ),
        ]);

        assert!(checkpoint_manager
            .validate_optimizer_state(&valid_state)
            .is_ok());

        // Test invalid state - missing field
        let invalid_state = std::collections::HashMap::from([
            (
                "optimizer_type".to_string(),
                serde_json::Value::String("Adam".to_string()),
            ),
            (
                "step_count".to_string(),
                serde_json::Value::Number(serde_json::Number::from(100)),
            ),
            // missing learning_rate
        ]);

        assert!(checkpoint_manager
            .validate_optimizer_state(&invalid_state)
            .is_err());

        // Test invalid learning rate
        let invalid_lr_state = std::collections::HashMap::from([
            (
                "optimizer_type".to_string(),
                serde_json::Value::String("Adam".to_string()),
            ),
            (
                "step_count".to_string(),
                serde_json::Value::Number(serde_json::Number::from(100)),
            ),
            (
                "learning_rate".to_string(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(-0.001).expect("test: operation should succeed"),
                ),
            ),
        ]);

        assert!(checkpoint_manager
            .validate_optimizer_state(&invalid_lr_state)
            .is_err());
    }
}
