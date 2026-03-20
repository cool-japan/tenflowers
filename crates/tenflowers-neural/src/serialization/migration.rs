//! Model migration utilities for handling version compatibility
//!
//! This module provides functionality to migrate models between different serialization
//! format versions, ensuring backward compatibility.

use super::{
    AdvancedModelState, HardwareRequirements, ModelMetadata, SemanticVersion, TrainingInfo,
};
use crate::model::ModelState;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

/// Migration result information
#[derive(Debug, Clone)]
pub struct MigrationResult {
    /// Original version before migration
    pub original_version: SemanticVersion,
    /// Target version after migration
    pub target_version: SemanticVersion,
    /// Steps performed during migration
    pub migration_steps: Vec<MigrationStep>,
    /// Warnings generated during migration
    pub warnings: Vec<String>,
    /// Whether migration was successful
    pub success: bool,
}

/// Individual migration step
#[derive(Debug, Clone)]
pub struct MigrationStep {
    /// Step description
    pub description: String,
    /// From version
    pub from_version: SemanticVersion,
    /// To version
    pub to_version: SemanticVersion,
    /// Whether step was successful
    pub success: bool,
}

/// Migration strategy for different version transitions
#[derive(Debug, Clone)]
pub enum MigrationStrategy {
    /// Direct migration (no intermediate steps)
    Direct,
    /// Sequential migration through intermediate versions
    Sequential,
    /// Best-effort migration with potential data loss
    BestEffort,
}

/// Model migration engine
pub struct ModelMigrator {
    /// Available migration paths
    migration_paths: HashMap<(SemanticVersion, SemanticVersion), MigrationPath>,
}

/// Migration path definition
#[derive(Debug, Clone)]
pub struct MigrationPath {
    /// Source version
    pub from: SemanticVersion,
    /// Target version
    pub to: SemanticVersion,
    /// Migration strategy
    pub strategy: MigrationStrategy,
    /// Migration function
    pub migrator: fn(&str) -> Result<String>,
}

impl ModelMigrator {
    /// Create a new model migrator with default migration paths
    pub fn new() -> Self {
        let mut migrator = Self {
            migration_paths: HashMap::new(),
        };

        // Register default migration paths
        migrator.register_default_migrations();
        migrator
    }

    /// Register default migration paths
    fn register_default_migrations(&mut self) {
        // Migration from legacy format (0.0.1) to advanced format (0.1.0)
        self.register_migration(
            SemanticVersion::new(0, 0, 1),
            SemanticVersion::new(0, 1, 0),
            MigrationStrategy::Direct,
            Self::migrate_legacy_to_advanced,
        );

        // Future migrations can be added here
        // Example: 0.1.0 -> 0.2.0
        self.register_migration(
            SemanticVersion::new(0, 1, 0),
            SemanticVersion::new(0, 2, 0),
            MigrationStrategy::Sequential,
            Self::migrate_v01_to_v02,
        );
    }

    /// Register a migration path
    pub fn register_migration(
        &mut self,
        from: SemanticVersion,
        to: SemanticVersion,
        strategy: MigrationStrategy,
        migrator: fn(&str) -> Result<String>,
    ) {
        let path = MigrationPath {
            from: from.clone(),
            to: to.clone(),
            strategy,
            migrator,
        };

        self.migration_paths.insert((from, to), path);
    }

    /// Migrate model data from one version to another
    pub fn migrate(
        &self,
        model_data: &str,
        from_version: &SemanticVersion,
        to_version: &SemanticVersion,
    ) -> Result<MigrationResult> {
        let mut result = MigrationResult {
            original_version: from_version.clone(),
            target_version: to_version.clone(),
            migration_steps: Vec::new(),
            warnings: Vec::new(),
            success: false,
        };

        // Check if migration is needed
        if from_version == to_version {
            result.success = true;
            return Ok(result);
        }

        // Check if direct migration path exists
        if let Some(path) = self
            .migration_paths
            .get(&(from_version.clone(), to_version.clone()))
        {
            return self.perform_direct_migration(model_data, path, &mut result);
        }

        // Try to find sequential migration path
        if let Some(sequential_path) = self.find_sequential_path(from_version, to_version) {
            return self.perform_sequential_migration(model_data, &sequential_path, &mut result);
        }

        Err(TensorError::serialization_error_simple(format!(
            "No migration path found from {} to {}",
            from_version, to_version
        )))
    }

    /// Perform direct migration
    fn perform_direct_migration(
        &self,
        model_data: &str,
        path: &MigrationPath,
        result: &mut MigrationResult,
    ) -> Result<MigrationResult> {
        let migrated_data = (path.migrator)(model_data)?;

        let step = MigrationStep {
            description: format!("Direct migration from {} to {}", path.from, path.to),
            from_version: path.from.clone(),
            to_version: path.to.clone(),
            success: true,
        };

        result.migration_steps.push(step);
        result.success = true;

        Ok(result.clone())
    }

    /// Find sequential migration path
    fn find_sequential_path(
        &self,
        from: &SemanticVersion,
        to: &SemanticVersion,
    ) -> Option<Vec<MigrationPath>> {
        // Simple implementation - can be enhanced with graph algorithms
        let mut current = from.clone();
        let mut path = Vec::new();
        let mut visited = std::collections::HashSet::new();

        while current != *to && !visited.contains(&current) {
            visited.insert(current.clone());

            // Find next step
            let next_step = self
                .migration_paths
                .iter()
                .find(|((from_ver, _), _)| from_ver == &current)
                .map(|(_, path)| path.clone());

            if let Some(step) = next_step {
                current = step.to.clone();
                path.push(step);
            } else {
                return None;
            }
        }

        if current == *to {
            Some(path)
        } else {
            None
        }
    }

    /// Perform sequential migration
    fn perform_sequential_migration(
        &self,
        model_data: &str,
        path: &[MigrationPath],
        result: &mut MigrationResult,
    ) -> Result<MigrationResult> {
        let mut current_data = model_data.to_string();

        for step in path {
            match (step.migrator)(&current_data) {
                Ok(migrated_data) => {
                    current_data = migrated_data;

                    let migration_step = MigrationStep {
                        description: format!(
                            "Sequential migration from {} to {}",
                            step.from, step.to
                        ),
                        from_version: step.from.clone(),
                        to_version: step.to.clone(),
                        success: true,
                    };

                    result.migration_steps.push(migration_step);
                }
                Err(e) => {
                    let migration_step = MigrationStep {
                        description: format!(
                            "Failed migration from {} to {}: {}",
                            step.from, step.to, e
                        ),
                        from_version: step.from.clone(),
                        to_version: step.to.clone(),
                        success: false,
                    };

                    result.migration_steps.push(migration_step);
                    return Err(e);
                }
            }
        }

        result.success = true;
        Ok(result.clone())
    }

    /// Migrate from legacy format (0.0.1) to advanced format (0.1.0)
    fn migrate_legacy_to_advanced(legacy_data: &str) -> Result<String> {
        // Parse legacy format
        let legacy_state: ModelState = serde_json::from_str(legacy_data).map_err(|e| {
            TensorError::serialization_error_simple(format!("Failed to parse legacy format: {}", e))
        })?;

        // Create advanced metadata
        let metadata = ModelMetadata {
            model_type: legacy_state
                .metadata
                .get("model_type")
                .cloned()
                .unwrap_or_else(|| "Unknown".to_string()),
            version: SemanticVersion::new(0, 1, 0),
            framework_version: super::utils::get_framework_version(),
            created_at: super::utils::get_timestamp(),
            architecture_hash: "legacy_migrated".to_string(),
            parameter_count: legacy_state.parameters.iter().map(|p| p.len()).sum(),
            model_size: legacy_data.len(),
            training_info: TrainingInfo {
                epochs: None,
                final_loss: None,
                validation_accuracy: None,
                optimizer: None,
                learning_rate: None,
                dataset_info: None,
            },
            hardware_requirements: HardwareRequirements {
                min_memory: 1024 * 1024,              // 1MB default
                recommended_memory: 1024 * 1024 * 10, // 10MB default
                gpu_required: false,
                cpu_features: vec![],
                target_device: "CPU".to_string(),
            },
            custom: legacy_state.metadata.clone(),
        };

        // Convert parameters to advanced format
        let mut parameters_info = Vec::new();
        for (i, (param_data, shape)) in legacy_state
            .parameters
            .iter()
            .zip(legacy_state.shapes.iter())
            .enumerate()
        {
            let param_info = crate::serialization::ParameterInfo {
                name: format!("param_{}", i),
                shape: shape.clone(),
                dtype: "f32".to_string(),
                device: "CPU".to_string(),
                requires_grad: true,
                checksum: utils::calculate_parameter_checksum_data(param_data),
            };
            parameters_info.push(param_info);
        }

        // Serialize parameters
        let parameters_data = serde_json::to_vec(&legacy_state.parameters).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "Failed to serialize parameters: {}",
                e
            ))
        })?;

        // Create advanced state
        let advanced_state = AdvancedModelState {
            metadata,
            parameters_info,
            parameters_data,
            compression_info: None,
            schema_hash: "legacy_migration".to_string(),
        };

        // Serialize to JSON
        serde_json::to_string_pretty(&advanced_state).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "Failed to serialize advanced format: {}",
                e
            ))
        })
    }

    /// Migrate from v0.1.0 to v0.2.0 (placeholder for future versions)
    fn migrate_v01_to_v02(v01_data: &str) -> Result<String> {
        // This is a placeholder for future version migrations
        // For now, just return the data unchanged
        Ok(v01_data.to_string())
    }

    /// Check if migration is possible between two versions
    pub fn can_migrate(&self, from: &SemanticVersion, to: &SemanticVersion) -> bool {
        // Check direct migration
        if self
            .migration_paths
            .contains_key(&(from.clone(), to.clone()))
        {
            return true;
        }

        // Check sequential migration
        self.find_sequential_path(from, to).is_some()
    }

    /// Get all supported migration paths
    pub fn get_supported_paths(&self) -> Vec<&MigrationPath> {
        self.migration_paths.values().collect()
    }
}

impl Default for ModelMigrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for migration
pub mod utils {
    use super::*;

    /// Calculate checksum for parameter data
    pub fn calculate_parameter_checksum_data(data: &[f32]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash the data
        for &val in data {
            val.to_bits().hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Validate migration compatibility
    pub fn validate_migration_compatibility(
        from: &SemanticVersion,
        to: &SemanticVersion,
    ) -> Result<()> {
        // Major version changes require explicit migration
        if from.major != to.major {
            return Err(TensorError::serialization_error_simple(format!(
                "Major version change from {} to {} requires explicit migration",
                from.major, to.major
            )));
        }

        // Cannot migrate to older versions
        if from > to {
            return Err(TensorError::serialization_error_simple(format!(
                "Cannot migrate from newer version {} to older version {}",
                from, to
            )));
        }

        Ok(())
    }

    /// Get migration recommendations
    pub fn get_migration_recommendations(
        from: &SemanticVersion,
        to: &SemanticVersion,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Version difference recommendations
        if from.major != to.major {
            recommendations
                .push("Major version change detected. Review breaking changes.".to_string());
        }

        if from.minor != to.minor {
            recommendations
                .push("Minor version change detected. New features may be available.".to_string());
        }

        if from.patch != to.patch {
            recommendations
                .push("Patch version change detected. Bug fixes may be included.".to_string());
        }

        // Backup recommendation
        recommendations.push("Consider creating a backup before migration.".to_string());

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration_result_creation() {
        let result = MigrationResult {
            original_version: SemanticVersion::new(0, 0, 1),
            target_version: SemanticVersion::new(0, 1, 0),
            migration_steps: Vec::new(),
            warnings: Vec::new(),
            success: true,
        };

        assert!(result.success);
        assert_eq!(result.original_version, SemanticVersion::new(0, 0, 1));
        assert_eq!(result.target_version, SemanticVersion::new(0, 1, 0));
    }

    #[test]
    fn test_migrator_creation() {
        let migrator = ModelMigrator::new();

        // Check that default migrations are registered
        assert!(migrator.can_migrate(
            &SemanticVersion::new(0, 0, 1),
            &SemanticVersion::new(0, 1, 0)
        ));
    }

    #[test]
    fn test_migration_step() {
        let step = MigrationStep {
            description: "Test migration".to_string(),
            from_version: SemanticVersion::new(0, 0, 1),
            to_version: SemanticVersion::new(0, 1, 0),
            success: true,
        };

        assert!(step.success);
        assert_eq!(step.description, "Test migration");
    }

    #[test]
    fn test_migration_compatibility_validation() {
        // Valid migration (same major version)
        assert!(utils::validate_migration_compatibility(
            &SemanticVersion::new(1, 0, 0),
            &SemanticVersion::new(1, 1, 0)
        )
        .is_ok());

        // Invalid migration (different major version)
        assert!(utils::validate_migration_compatibility(
            &SemanticVersion::new(1, 0, 0),
            &SemanticVersion::new(2, 0, 0)
        )
        .is_err());

        // Invalid migration (newer to older)
        assert!(utils::validate_migration_compatibility(
            &SemanticVersion::new(1, 1, 0),
            &SemanticVersion::new(1, 0, 0)
        )
        .is_err());
    }

    #[test]
    fn test_migration_recommendations() {
        let recommendations = utils::get_migration_recommendations(
            &SemanticVersion::new(0, 0, 1),
            &SemanticVersion::new(0, 1, 0),
        );

        assert!(!recommendations.is_empty());
        assert!(recommendations
            .iter()
            .any(|r| r.contains("Minor version change")));
        assert!(recommendations.iter().any(|r| r.contains("backup")));
    }
}
