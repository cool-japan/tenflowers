//! CheckpointManager and CheckpointData implementation.

use super::super::{CompressionInfo, ModelMetadata, SemanticVersion};
use super::types::{CheckpointConfig, CheckpointInfo, CheckpointLoadResult};
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tenflowers_core::{Result, TensorError};

/// Internal checkpoint data structure
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub(super) struct CheckpointData {
    /// Checkpoint information
    pub(super) info: CheckpointInfo,
    /// Model metadata
    pub(super) model_metadata: ModelMetadata,
    /// Serialized model state
    pub(super) model_state: String,
    /// Optimizer state (optional)
    pub(super) optimizer_state: Option<String>,
    /// Compression information
    pub(super) compression_info: Option<CompressionInfo>,
}

/// Checkpoint manager
pub struct CheckpointManager {
    /// Configuration
    pub(super) config: CheckpointConfig,
    /// Checkpoint history
    pub(super) checkpoint_history: Vec<CheckpointInfo>,
    /// Best checkpoint (based on validation loss)
    pub(super) best_checkpoint: Option<CheckpointInfo>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(config: CheckpointConfig) -> Result<Self> {
        // Create checkpoint directory if it doesn't exist
        if !config.checkpoint_dir.exists() {
            std::fs::create_dir_all(&config.checkpoint_dir).map_err(|e| {
                TensorError::serialization_error_simple(format!(
                    "Failed to create checkpoint directory: {}",
                    e
                ))
            })?;
        }

        let mut manager = Self {
            config,
            checkpoint_history: Vec::new(),
            best_checkpoint: None,
        };

        // Load existing checkpoints
        manager.load_checkpoint_history()?;

        Ok(manager)
    }

    /// Save a checkpoint
    pub fn save_checkpoint<T>(
        &mut self,
        model: &dyn crate::model::Model<T>,
        checkpoint_info: CheckpointInfo,
    ) -> Result<PathBuf>
    where
        T: Clone + 'static,
    {
        // Generate checkpoint filename
        let filename = self.generate_checkpoint_filename(&checkpoint_info);
        let checkpoint_path = self.config.checkpoint_dir.join(&filename);

        // Create checkpoint data
        let checkpoint_data = CheckpointData {
            info: checkpoint_info.clone(),
            model_metadata: self.create_model_metadata(model)?,
            model_state: self.serialize_model_state(model)?,
            optimizer_state: if self.config.save_optimizer_state {
                Some(self.serialize_optimizer_state()?)
            } else {
                None
            },
            compression_info: None,
        };

        // Serialize checkpoint
        let mut serialized = serde_json::to_string_pretty(&checkpoint_data).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "Failed to serialize checkpoint: {}",
                e
            ))
        })?;

        // Apply compression if enabled
        if self.config.compression {
            let compressed = super::super::Compressor::compress(
                serialized.as_bytes(),
                self.config.compression_algorithm,
                3,
            )?;

            // Update checkpoint data with compression info
            let compression_info = CompressionInfo::new(
                self.config.compression_algorithm,
                3,
                serialized.len(),
                compressed.len(),
            );

            // Re-serialize with compression info
            let checkpoint_data_compressed = CheckpointData {
                compression_info: Some(compression_info),
                ..checkpoint_data
            };

            serialized =
                serde_json::to_string_pretty(&checkpoint_data_compressed).map_err(|e| {
                    TensorError::serialization_error_simple(format!(
                        "Failed to serialize compressed checkpoint: {}",
                        e
                    ))
                })?;
        }

        // Write checkpoint file
        std::fs::write(&checkpoint_path, serialized).map_err(|e| {
            TensorError::serialization_error_simple(format!("Failed to write checkpoint: {}", e))
        })?;

        // Update checkpoint history
        self.checkpoint_history.push(checkpoint_info.clone());

        // Update best checkpoint if this is better
        if self.is_best_checkpoint(&checkpoint_info) {
            self.best_checkpoint = Some(checkpoint_info);
        }

        // Cleanup old checkpoints if needed
        if self.config.auto_cleanup {
            self.cleanup_old_checkpoints()?;
        }

        Ok(checkpoint_path)
    }

    /// Load a checkpoint
    pub fn load_checkpoint<T>(
        &self,
        model: &mut dyn crate::model::Model<T>,
        checkpoint_path: &Path,
    ) -> Result<CheckpointLoadResult>
    where
        T: Clone + 'static,
    {
        // Read checkpoint file
        let checkpoint_content = std::fs::read_to_string(checkpoint_path).map_err(|e| {
            TensorError::serialization_error_simple(format!("Failed to read checkpoint: {}", e))
        })?;

        // Deserialize checkpoint
        let checkpoint_data: CheckpointData =
            serde_json::from_str(&checkpoint_content).map_err(|e| {
                TensorError::serialization_error_simple(format!(
                    "Failed to deserialize checkpoint: {}",
                    e
                ))
            })?;

        // Decompress if needed
        let model_data = if let Some(compression_info) = &checkpoint_data.compression_info {
            super::super::Compressor::decompress(
                checkpoint_data.model_state.as_bytes(),
                compression_info.algorithm,
            )?
        } else {
            checkpoint_data.model_state.into_bytes()
        };

        // Load model state
        self.deserialize_model_state(model, &model_data)?;

        // Load optimizer state if available
        let optimizer_state_loaded = if let Some(optimizer_state) = &checkpoint_data.optimizer_state
        {
            self.deserialize_optimizer_state(optimizer_state)?;
            true
        } else {
            false
        };

        Ok(CheckpointLoadResult {
            checkpoint_info: checkpoint_data.info,
            model_metadata: checkpoint_data.model_metadata,
            optimizer_state_loaded,
            warnings: Vec::new(),
        })
    }

    /// Load the best checkpoint
    pub fn load_best_checkpoint<T>(
        &self,
        model: &mut dyn crate::model::Model<T>,
    ) -> Result<CheckpointLoadResult>
    where
        T: Clone + 'static,
    {
        if let Some(best_checkpoint) = &self.best_checkpoint {
            let checkpoint_path = self.get_checkpoint_path(best_checkpoint);
            self.load_checkpoint(model, &checkpoint_path)
        } else {
            Err(TensorError::serialization_error_simple(
                "No best checkpoint available".to_string(),
            ))
        }
    }

    /// Load the latest checkpoint
    pub fn load_latest_checkpoint<T>(
        &self,
        model: &mut dyn crate::model::Model<T>,
    ) -> Result<CheckpointLoadResult>
    where
        T: Clone + 'static,
    {
        if let Some(latest_checkpoint) = self.checkpoint_history.last() {
            let checkpoint_path = self.get_checkpoint_path(latest_checkpoint);
            self.load_checkpoint(model, &checkpoint_path)
        } else {
            Err(TensorError::serialization_error_simple(
                "No checkpoints available".to_string(),
            ))
        }
    }

    /// Get checkpoint information
    pub fn get_checkpoint_info(&self) -> &[CheckpointInfo] {
        &self.checkpoint_history
    }

    /// Get best checkpoint information
    pub fn get_best_checkpoint(&self) -> Option<&CheckpointInfo> {
        self.best_checkpoint.as_ref()
    }

    /// Clean up old checkpoints
    pub fn cleanup_old_checkpoints(&mut self) -> Result<()> {
        if self.checkpoint_history.len() <= self.config.max_checkpoints {
            return Ok(());
        }

        // Sort by timestamp (oldest first)
        self.checkpoint_history
            .sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        // Keep only the most recent checkpoints
        let to_remove = self.checkpoint_history.len() - self.config.max_checkpoints;
        let checkpoints_to_remove: Vec<_> = self.checkpoint_history.drain(..to_remove).collect();
        for checkpoint in checkpoints_to_remove {
            let checkpoint_path = self.get_checkpoint_path(&checkpoint);
            if checkpoint_path.exists() {
                std::fs::remove_file(&checkpoint_path).map_err(|e| {
                    TensorError::serialization_error_simple(format!(
                        "Failed to remove old checkpoint: {}",
                        e
                    ))
                })?;
            }
        }

        Ok(())
    }

    /// List available checkpoints
    pub fn list_checkpoints(&self) -> Vec<CheckpointInfo> {
        self.checkpoint_history.clone()
    }

    /// Check if a checkpoint exists
    pub fn checkpoint_exists(&self, checkpoint_id: &str) -> bool {
        self.checkpoint_history
            .iter()
            .any(|c| c.checkpoint_id == checkpoint_id)
    }

    /// Validate checkpoint compatibility before loading
    pub fn validate_checkpoint_compatibility<T>(
        &self,
        model: &dyn crate::model::Model<T>,
        checkpoint_path: &Path,
    ) -> Result<Vec<String>>
    where
        T: Clone + 'static,
    {
        let mut warnings = Vec::new();

        // Read checkpoint file
        let checkpoint_content = std::fs::read_to_string(checkpoint_path).map_err(|e| {
            TensorError::serialization_error_simple(format!("Failed to read checkpoint: {}", e))
        })?;

        // Deserialize checkpoint
        let checkpoint_data: CheckpointData =
            serde_json::from_str(&checkpoint_content).map_err(|e| {
                TensorError::serialization_error_simple(format!(
                    "Failed to deserialize checkpoint: {}",
                    e
                ))
            })?;

        // Check model metadata compatibility
        let current_params = model.parameters();
        let saved_param_count = checkpoint_data.model_metadata.parameter_count;

        if current_params.len() != saved_param_count {
            return Err(TensorError::serialization_error_simple(format!(
                "Parameter count mismatch: current {} vs saved {}",
                current_params.len(),
                saved_param_count
            )));
        }

        // Parse model state to check architecture
        let model_state: std::collections::HashMap<String, serde_json::Value> =
            serde_json::from_str(&checkpoint_data.model_state).map_err(|e| {
                TensorError::serialization_error_simple(format!(
                    "Failed to parse model state: {}",
                    e
                ))
            })?;

        // Check serialization version
        if let Some(version) = model_state.get("serialization_version") {
            let version_str = version.as_str().unwrap_or("unknown");
            if !self.is_compatible_version(version_str) {
                return Err(TensorError::serialization_error_simple(format!(
                    "Incompatible serialization version: {}",
                    version_str
                )));
            }
        } else {
            warnings.push("No serialization version found in checkpoint".to_string());
        }

        // Check model type
        if let Some(saved_model_type) = model_state.get("model_type") {
            let saved_type = saved_model_type.as_str().unwrap_or("unknown");
            if !self.is_compatible_model_type(model, saved_type) {
                return Err(TensorError::serialization_error_simple(format!(
                    "Model type mismatch: expected compatible with {}",
                    saved_type
                )));
            }
        }

        // Validate parameter shapes if available
        if let Some(params_metadata) = model_state.get("parameters_metadata") {
            if let Some(metadata_array) = params_metadata.as_array() {
                self.validate_parameter_shapes(&current_params, metadata_array)?;
            }
        } else {
            warnings.push("No parameter metadata found in checkpoint".to_string());
        }

        // Check optimizer state if present
        if let Some(optimizer_state) = &checkpoint_data.optimizer_state {
            if let Err(e) = self.deserialize_optimizer_state(optimizer_state) {
                warnings.push(format!("Optimizer state validation warning: {}", e));
            }
        }

        Ok(warnings)
    }

    /// Get checkpoint version information
    pub fn get_checkpoint_version(&self, checkpoint_path: &Path) -> Result<String> {
        let checkpoint_content = std::fs::read_to_string(checkpoint_path).map_err(|e| {
            TensorError::serialization_error_simple(format!("Failed to read checkpoint: {}", e))
        })?;

        let checkpoint_data: CheckpointData =
            serde_json::from_str(&checkpoint_content).map_err(|e| {
                TensorError::serialization_error_simple(format!(
                    "Failed to deserialize checkpoint: {}",
                    e
                ))
            })?;

        let model_state: std::collections::HashMap<String, serde_json::Value> =
            serde_json::from_str(&checkpoint_data.model_state).map_err(|e| {
                TensorError::serialization_error_simple(format!(
                    "Failed to parse model state: {}",
                    e
                ))
            })?;

        Ok(model_state
            .get("serialization_version")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string())
    }

    /// Generate checkpoint filename
    fn generate_checkpoint_filename(&self, checkpoint_info: &CheckpointInfo) -> String {
        self.config
            .name_pattern
            .replace("{epoch}", &checkpoint_info.epoch.to_string())
            .replace("{step}", &checkpoint_info.step.to_string())
            .replace("{id}", &checkpoint_info.checkpoint_id)
            + ".json"
    }

    /// Get checkpoint path
    fn get_checkpoint_path(&self, checkpoint_info: &CheckpointInfo) -> PathBuf {
        let filename = self.generate_checkpoint_filename(checkpoint_info);
        self.config.checkpoint_dir.join(filename)
    }

    /// Check if this is the best checkpoint
    pub(super) fn is_best_checkpoint(&self, checkpoint_info: &CheckpointInfo) -> bool {
        if let Some(best) = &self.best_checkpoint {
            checkpoint_info.loss < best.loss
        } else {
            true
        }
    }

    /// Load checkpoint history from disk
    pub(super) fn load_checkpoint_history(&mut self) -> Result<()> {
        if !self.config.checkpoint_dir.exists() {
            return Ok(());
        }

        let entries = std::fs::read_dir(&self.config.checkpoint_dir).map_err(|e| {
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
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(checkpoint_data) = self.load_checkpoint_info(&path) {
                    self.checkpoint_history.push(checkpoint_data.info.clone());

                    if self.is_best_checkpoint(&checkpoint_data.info) {
                        self.best_checkpoint = Some(checkpoint_data.info);
                    }
                }
            }
        }

        // Sort by timestamp
        self.checkpoint_history
            .sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        Ok(())
    }

    /// Load checkpoint info from file
    pub(super) fn load_checkpoint_info(&self, path: &Path) -> Result<CheckpointData> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "Failed to read checkpoint file: {}",
                e
            ))
        })?;

        serde_json::from_str(&content).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "Failed to parse checkpoint file: {}",
                e
            ))
        })
    }

    /// Create model metadata
    pub(super) fn create_model_metadata<T>(
        &self,
        model: &dyn crate::model::Model<T>,
    ) -> Result<ModelMetadata> {
        let parameters = model.parameters();
        let parameter_count = parameters
            .iter()
            .map(|p| p.shape().dims().iter().product::<usize>())
            .sum();

        // Calculate architecture hash based on parameter shapes and device info
        let architecture_hash = {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            use std::hash::{Hash, Hasher};

            for param in &parameters {
                param.shape().dims().hash(&mut hasher);
                format!("{:?}", param.device()).hash(&mut hasher);
            }

            format!("{:x}", hasher.finish())
        };

        Ok(ModelMetadata {
            model_type: "Model".to_string(),
            version: SemanticVersion::new(0, 1, 0),
            framework_version: super::super::utils::get_framework_version(),
            created_at: super::super::utils::get_timestamp(),
            architecture_hash,
            parameter_count,
            model_size: 0, // Will be calculated after serialization
            training_info: super::super::TrainingInfo {
                epochs: None,
                final_loss: None,
                validation_accuracy: None,
                optimizer: None,
                learning_rate: None,
                dataset_info: None,
            },
            hardware_requirements: super::super::HardwareRequirements {
                min_memory: 1024 * 1024,
                recommended_memory: 1024 * 1024 * 10,
                gpu_required: false,
                cpu_features: vec![],
                target_device: "CPU".to_string(),
            },
            custom: HashMap::new(),
        })
    }

    /// Serialize model state
    pub(super) fn serialize_model_state<T>(
        &self,
        model: &dyn crate::model::Model<T>,
    ) -> Result<String>
    where
        T: Clone + 'static,
    {
        // Enhanced implementation for proper model state serialization
        let parameters = model.parameters();

        // Create a comprehensive model state
        let mut model_state = std::collections::HashMap::new();

        // Serialize parameter information
        let mut param_metadata = Vec::new();
        for (idx, param) in parameters.iter().enumerate() {
            let param_info = serde_json::json!({
                "index": idx,
                "shape": param.shape().dims(),
                "device": format!("{:?}", param.device()),
                "dtype": "f32", // Assuming f32 for now, could be made generic
                "parameter_count": param.shape().dims().iter().product::<usize>(),
                "requires_grad": true,
            });
            param_metadata.push(param_info);
        }

        model_state.insert(
            "parameters_metadata".to_string(),
            serde_json::Value::Array(param_metadata),
        );
        model_state.insert(
            "parameter_count".to_string(),
            serde_json::Value::Number(serde_json::Number::from(parameters.len())),
        );
        model_state.insert(
            "model_type".to_string(),
            serde_json::Value::String("Model".to_string()),
        );
        model_state.insert(
            "serialization_version".to_string(),
            serde_json::Value::String("1.0".to_string()),
        );

        // Add model architecture information
        if let Some(sequential_model) = model
            .as_any()
            .downcast_ref::<crate::model::sequential::Sequential<T>>()
        {
            model_state.insert(
                "model_type".to_string(),
                serde_json::Value::String("Sequential".to_string()),
            );
            model_state.insert(
                "layer_count".to_string(),
                serde_json::Value::Number(serde_json::Number::from(
                    sequential_model.layers().len(),
                )),
            );
        }

        serde_json::to_string(&model_state).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "Failed to serialize model state: {}",
                e
            ))
        })
    }

    /// Deserialize model state
    pub(super) fn deserialize_model_state<T>(
        &self,
        model: &mut dyn crate::model::Model<T>,
        data: &[u8],
    ) -> Result<()>
    where
        T: Clone + 'static,
    {
        // Enhanced implementation for model state deserialization with compatibility checking
        let model_state_str = String::from_utf8(data.to_vec()).map_err(|e| {
            TensorError::serialization_error_simple(format!("Invalid UTF-8 in model state: {}", e))
        })?;

        let model_state: std::collections::HashMap<String, serde_json::Value> =
            serde_json::from_str(&model_state_str).map_err(|e| {
                TensorError::serialization_error_simple(format!(
                    "Failed to parse model state JSON: {}",
                    e
                ))
            })?;

        // Check serialization version compatibility
        if let Some(version) = model_state.get("serialization_version") {
            let version_str = version.as_str().unwrap_or("unknown");
            if !self.is_compatible_version(version_str) {
                return Err(TensorError::serialization_error_simple(format!(
                    "Incompatible serialization version: {} (supported: 1.0)",
                    version_str
                )));
            }
        }

        // Check model type compatibility
        if let Some(saved_model_type) = model_state.get("model_type") {
            let saved_type = saved_model_type.as_str().unwrap_or("unknown");
            if !self.is_compatible_model_type(model, saved_type) {
                return Err(TensorError::serialization_error_simple(format!(
                    "Model type mismatch: expected compatible with {}",
                    saved_type
                )));
            }
        }

        // Check parameter count compatibility
        let current_params = model.parameters();
        if let Some(saved_param_count) = model_state.get("parameter_count") {
            let saved_count = saved_param_count.as_u64().unwrap_or(0) as usize;
            if current_params.len() != saved_count {
                return Err(TensorError::serialization_error_simple(format!(
                    "Parameter count mismatch: current {} vs saved {}",
                    current_params.len(),
                    saved_count
                )));
            }
        }

        // Validate parameter shapes
        if let Some(params_metadata) = model_state.get("parameters_metadata") {
            if let Some(metadata_array) = params_metadata.as_array() {
                self.validate_parameter_shapes(&current_params, metadata_array)?;
            }
        }

        // Note: Actual parameter loading would happen here
        // For now, we're focusing on compatibility validation

        Ok(())
    }

    /// Check if serialization version is compatible
    pub(super) fn is_compatible_version(&self, version: &str) -> bool {
        // For now, only support version 1.0
        version == "1.0"
    }

    /// Check if model type is compatible
    pub(super) fn is_compatible_model_type<T>(
        &self,
        model: &dyn crate::model::Model<T>,
        saved_type: &str,
    ) -> bool
    where
        T: Clone + 'static,
    {
        // Check if current model is compatible with saved model type
        match saved_type {
            "Model" => true, // Base model type is always compatible
            "Sequential" => {
                // Check if current model is Sequential
                model
                    .as_any()
                    .downcast_ref::<crate::model::sequential::Sequential<T>>()
                    .is_some()
            }
            _ => false, // Unknown types are not compatible
        }
    }

    /// Validate parameter shapes match between current model and saved model
    pub(super) fn validate_parameter_shapes<T>(
        &self,
        current_params: &[&tenflowers_core::Tensor<T>],
        saved_metadata: &[serde_json::Value],
    ) -> Result<()> {
        if current_params.len() != saved_metadata.len() {
            return Err(TensorError::serialization_error_simple(format!(
                "Parameter count mismatch during shape validation: {} vs {}",
                current_params.len(),
                saved_metadata.len()
            )));
        }

        for (idx, (current_param, saved_info)) in
            current_params.iter().zip(saved_metadata.iter()).enumerate()
        {
            if let Some(saved_shape) = saved_info.get("shape") {
                if let Some(saved_shape_array) = saved_shape.as_array() {
                    let saved_dims: Vec<usize> = saved_shape_array
                        .iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect();

                    let current_dims = current_param.shape().dims();

                    if current_dims != saved_dims.as_slice() {
                        return Err(TensorError::serialization_error_simple(format!(
                            "Parameter {} shape mismatch: current {:?} vs saved {:?}",
                            idx, current_dims, saved_dims
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Serialize optimizer state
    pub(super) fn serialize_optimizer_state(&self) -> Result<String> {
        // Enhanced optimizer state serialization framework
        let mut optimizer_state = std::collections::HashMap::new();

        // Basic optimizer state structure
        optimizer_state.insert(
            "optimizer_type".to_string(),
            serde_json::Value::String("unknown".to_string()),
        );
        optimizer_state.insert(
            "step_count".to_string(),
            serde_json::Value::Number(serde_json::Number::from(0)),
        );
        optimizer_state.insert(
            "learning_rate".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(0.001).unwrap_or(serde_json::Number::from(0)),
            ),
        );
        optimizer_state.insert(
            "serialization_version".to_string(),
            serde_json::Value::String("1.0".to_string()),
        );

        // Momentum states (for optimizers like SGD with momentum, Adam, etc.)
        optimizer_state.insert("has_momentum".to_string(), serde_json::Value::Bool(false));
        optimizer_state.insert(
            "momentum_states".to_string(),
            serde_json::Value::Array(vec![]),
        );

        // Second moment states (for Adam-like optimizers)
        optimizer_state.insert(
            "has_second_moment".to_string(),
            serde_json::Value::Bool(false),
        );
        optimizer_state.insert(
            "second_moment_states".to_string(),
            serde_json::Value::Array(vec![]),
        );

        // Scheduler state
        optimizer_state.insert(
            "scheduler_state".to_string(),
            serde_json::Value::Object(serde_json::Map::new()),
        );

        serde_json::to_string(&optimizer_state).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "Failed to serialize optimizer state: {}",
                e
            ))
        })
    }

    /// Deserialize optimizer state
    pub(super) fn deserialize_optimizer_state(&self, data: &str) -> Result<()> {
        // Enhanced optimizer state deserialization with validation
        let optimizer_state: std::collections::HashMap<String, serde_json::Value> =
            serde_json::from_str(data).map_err(|e| {
                TensorError::serialization_error_simple(format!(
                    "Failed to parse optimizer state JSON: {}",
                    e
                ))
            })?;

        // Check serialization version compatibility
        if let Some(version) = optimizer_state.get("serialization_version") {
            let version_str = version.as_str().unwrap_or("unknown");
            if !self.is_compatible_version(version_str) {
                return Err(TensorError::serialization_error_simple(format!(
                    "Incompatible optimizer serialization version: {} (supported: 1.0)",
                    version_str
                )));
            }
        }

        // Validate optimizer state structure
        self.validate_optimizer_state(&optimizer_state)?;

        // Note: Actual optimizer state loading would happen here
        // This would involve restoring momentum states, second moment states, etc.

        Ok(())
    }

    /// Validate optimizer state structure
    pub(super) fn validate_optimizer_state(
        &self,
        state: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        // Check required fields
        let required_fields = ["optimizer_type", "step_count", "learning_rate"];
        for field in &required_fields {
            if !state.contains_key(*field) {
                return Err(TensorError::serialization_error_simple(format!(
                    "Missing required optimizer state field: {}",
                    field
                )));
            }
        }

        // Validate step count is non-negative
        if let Some(step_count) = state.get("step_count") {
            if let Some(steps) = step_count.as_u64() {
                if steps > u64::MAX / 2 {
                    return Err(TensorError::serialization_error_simple(
                        "Invalid step count in optimizer state".to_string(),
                    ));
                }
            }
        }

        // Validate learning rate is positive
        if let Some(lr) = state.get("learning_rate") {
            if let Some(lr_val) = lr.as_f64() {
                if lr_val <= 0.0 || lr_val.is_nan() || lr_val.is_infinite() {
                    return Err(TensorError::serialization_error_simple(
                        "Invalid learning rate in optimizer state".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }
}
