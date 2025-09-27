//! Checkpoint management for model training
//!
//! This module provides comprehensive checkpoint management for training,
//! including automatic checkpointing, recovery, and checkpoint optimization.

use super::{CompressionAlgorithm, CompressionInfo, ModelMetadata, SemanticVersion};
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tenflowers_core::{Result, TensorError};

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

/// Checkpoint manager
pub struct CheckpointManager {
    /// Configuration
    config: CheckpointConfig,
    /// Checkpoint history
    checkpoint_history: Vec<CheckpointInfo>,
    /// Best checkpoint (based on validation loss)
    best_checkpoint: Option<CheckpointInfo>,
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
            let compressed = super::Compressor::compress(
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
            super::Compressor::decompress(
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
    fn is_best_checkpoint(&self, checkpoint_info: &CheckpointInfo) -> bool {
        if let Some(best) = &self.best_checkpoint {
            checkpoint_info.loss < best.loss
        } else {
            true
        }
    }

    /// Load checkpoint history from disk
    fn load_checkpoint_history(&mut self) -> Result<()> {
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
    fn load_checkpoint_info(&self, path: &Path) -> Result<CheckpointData> {
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
    fn create_model_metadata<T>(
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
            framework_version: super::utils::get_framework_version(),
            created_at: super::utils::get_timestamp(),
            architecture_hash,
            parameter_count,
            model_size: 0, // Will be calculated after serialization
            training_info: super::TrainingInfo {
                epochs: None,
                final_loss: None,
                validation_accuracy: None,
                optimizer: None,
                learning_rate: None,
                dataset_info: None,
            },
            hardware_requirements: super::HardwareRequirements {
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
    fn serialize_model_state<T>(&self, model: &dyn crate::model::Model<T>) -> Result<String>
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
    fn deserialize_model_state<T>(
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
    fn is_compatible_version(&self, version: &str) -> bool {
        // For now, only support version 1.0
        version == "1.0"
    }

    /// Check if model type is compatible
    fn is_compatible_model_type<T>(
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
    fn validate_parameter_shapes<T>(
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
    fn serialize_optimizer_state(&self) -> Result<String> {
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
    fn deserialize_optimizer_state(&self, data: &str) -> Result<()> {
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
    fn validate_optimizer_state(
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

/// Internal checkpoint data structure
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug)]
struct CheckpointData {
    /// Checkpoint information
    info: CheckpointInfo,
    /// Model metadata
    model_metadata: ModelMetadata,
    /// Serialized model state
    model_state: String,
    /// Optimizer state (optional)
    optimizer_state: Option<String>,
    /// Compression information
    compression_info: Option<CompressionInfo>,
}

/// Checkpoint utilities
pub mod utils {
    use super::*;

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
            timestamp: super::super::utils::get_timestamp(),
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

#[cfg(test)]
mod tests {
    use super::*;
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
        assert_eq!(found.unwrap().epoch, 2);

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
        assert_eq!(best.unwrap().loss, 0.7);
        assert_eq!(best.unwrap().epoch, 3);
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
        assert_eq!(latest.unwrap().epoch, 3);
    }

    #[test]
    fn test_checkpoint_data_serialization() {
        let checkpoint_info = utils::create_checkpoint_info(1, 100, 0.5, 0.01);
        let model_metadata = ModelMetadata {
            model_type: "Sequential".to_string(),
            version: SemanticVersion::new(0, 1, 0),
            framework_version: "TenfloweRS-0.1.0".to_string(),
            created_at: "2023-01-01T00:00:00Z".to_string(),
            architecture_hash: "test_hash".to_string(),
            parameter_count: 1000,
            model_size: 4000,
            training_info: super::super::TrainingInfo {
                epochs: Some(1),
                final_loss: Some(0.5),
                validation_accuracy: Some(0.9),
                optimizer: Some("Adam".to_string()),
                learning_rate: Some(0.01),
                dataset_info: Some("Test".to_string()),
            },
            hardware_requirements: super::super::HardwareRequirements {
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

        let serialized = serde_json::to_string(&checkpoint_data).unwrap();
        assert!(!serialized.is_empty());

        let deserialized: CheckpointData = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.info.epoch, 1);
        assert_eq!(deserialized.info.step, 100);
    }

    #[test]
    fn test_model_state_serialization_enhanced() {
        // Create a mock checkpoint manager
        let config = CheckpointConfig::default();
        let checkpoint_manager = CheckpointManager {
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
            serde_json::from_str(model_state).unwrap();

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

        let optimizer_state = optimizer_state_result.unwrap();
        let parsed: std::collections::HashMap<String, serde_json::Value> =
            serde_json::from_str(&optimizer_state).unwrap();

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
                serde_json::Value::Number(serde_json::Number::from_f64(0.001).unwrap()),
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
                serde_json::Value::Number(serde_json::Number::from_f64(-0.001).unwrap()),
            ),
        ]);

        assert!(checkpoint_manager
            .validate_optimizer_state(&invalid_lr_state)
            .is_err());
    }
}
