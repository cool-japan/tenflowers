//! Dataset versioning with snapshots and lineage tracking
//!
//! This module provides functionality for versioning datasets, creating snapshots,
//! and tracking data lineage for reproducibility and data governance.

use crate::{Dataset, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tenflowers_core::{Tensor, TensorError};

/// Unique identifier for dataset versions
pub type VersionId = String;

/// Metadata for a dataset version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionMetadata {
    /// Unique version identifier
    pub version_id: VersionId,
    /// Parent version (if any)
    pub parent_version: Option<VersionId>,
    /// Timestamp when version was created
    pub timestamp: u64,
    /// Human-readable description
    pub description: String,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Custom metadata
    pub custom_metadata: HashMap<String, String>,
    /// Checksum for data integrity
    pub checksum: String,
    /// Dataset size information
    pub size_info: DatasetSizeInfo,
}

/// Information about dataset size and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSizeInfo {
    /// Number of samples
    pub sample_count: usize,
    /// Feature dimensions
    pub feature_shape: Vec<usize>,
    /// Label dimensions
    pub label_shape: Vec<usize>,
    /// Approximate size in bytes
    pub size_bytes: u64,
}

/// Lineage tracking information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetLineage {
    /// Version metadata
    pub version: VersionMetadata,
    /// Transformations applied to create this version
    pub transformations: Vec<TransformationRecord>,
    /// Source versions this was derived from
    pub source_versions: Vec<VersionId>,
    /// Children versions derived from this
    pub child_versions: Vec<VersionId>,
}

/// Record of a transformation applied to the dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationRecord {
    /// Type of transformation
    pub transform_type: String,
    /// Parameters used
    pub parameters: HashMap<String, String>,
    /// Timestamp when transformation was applied
    pub timestamp: u64,
    /// Description of the transformation
    pub description: String,
}

/// Dataset version manager for creating and managing snapshots
#[derive(Debug)]
pub struct DatasetVersionManager {
    /// Base directory for storing versions
    base_path: PathBuf,
    /// In-memory lineage graph
    lineage_graph: HashMap<VersionId, DatasetLineage>,
    /// Current active version
    current_version: Option<VersionId>,
}

impl DatasetVersionManager {
    /// Create a new version manager
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();

        // Create base directory if it doesn't exist
        if !base_path.exists() {
            std::fs::create_dir_all(&base_path).map_err(|e| {
                TensorError::invalid_argument(format!("Failed to create version directory: {e}"))
            })?;
        }

        let mut manager = Self {
            base_path,
            lineage_graph: HashMap::new(),
            current_version: None,
        };

        // Load existing lineage graph
        manager.load_lineage_graph()?;

        Ok(manager)
    }

    /// Create a snapshot of a dataset
    pub fn create_snapshot<T>(
        &mut self,
        dataset: &dyn Dataset<T>,
        description: String,
        tags: Vec<String>,
        parent_version: Option<VersionId>,
    ) -> Result<VersionId>
    where
        T: Clone + Default + serde::Serialize + serde::de::DeserializeOwned + Send + Sync + 'static,
    {
        let version_id = self.generate_version_id();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Calculate dataset statistics
        let size_info = self.calculate_size_info(dataset)?;

        // Calculate checksum
        let checksum = self.calculate_checksum(dataset)?;

        // Create version metadata
        let metadata = VersionMetadata {
            version_id: version_id.clone(),
            parent_version: parent_version.clone(),
            timestamp,
            description,
            tags,
            custom_metadata: HashMap::new(),
            checksum,
            size_info,
        };

        // Create snapshot directory
        let version_dir = self.base_path.join(&version_id);
        std::fs::create_dir_all(&version_dir).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to create version directory: {e}"))
        })?;

        // Serialize dataset samples
        self.save_dataset_samples(dataset, &version_dir)?;

        // Save metadata
        self.save_metadata(&metadata, &version_dir)?;

        // Update lineage graph
        let lineage = DatasetLineage {
            version: metadata,
            transformations: Vec::new(),
            source_versions: if let Some(parent) = &parent_version {
                vec![parent.clone()]
            } else {
                Vec::new()
            },
            child_versions: Vec::new(),
        };

        self.lineage_graph.insert(version_id.clone(), lineage);

        // Update parent's children list
        if let Some(parent) = &parent_version {
            if let Some(parent_lineage) = self.lineage_graph.get_mut(parent) {
                parent_lineage.child_versions.push(version_id.clone());
            }
        }

        self.current_version = Some(version_id.clone());
        self.save_lineage_graph()?;

        Ok(version_id)
    }

    /// Load a dataset snapshot by version ID
    pub fn load_snapshot<T>(&self, version_id: &str) -> Result<VersionedDataset<T>>
    where
        T: Clone + Default + serde::de::DeserializeOwned + Send + Sync + 'static,
    {
        let version_dir = self.base_path.join(version_id);

        if !version_dir.exists() {
            return Err(TensorError::invalid_argument(format!(
                "Version {version_id} not found"
            )));
        }

        // Load metadata
        let metadata = self.load_metadata(&version_dir)?;

        // Load dataset samples
        let samples = self.load_dataset_samples(&version_dir)?;

        Ok(VersionedDataset { metadata, samples })
    }

    /// Get lineage information for a version
    pub fn get_lineage(&self, version_id: &str) -> Option<&DatasetLineage> {
        self.lineage_graph.get(version_id)
    }

    /// Add transformation record to a version
    pub fn add_transformation(
        &mut self,
        version_id: &str,
        transform_type: String,
        parameters: HashMap<String, String>,
        description: String,
    ) -> Result<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let transformation = TransformationRecord {
            transform_type,
            parameters,
            timestamp,
            description,
        };

        if let Some(lineage) = self.lineage_graph.get_mut(version_id) {
            lineage.transformations.push(transformation);
            self.save_lineage_graph()?;
        } else {
            return Err(TensorError::invalid_argument(format!(
                "Version {version_id} not found"
            )));
        }

        Ok(())
    }

    /// List all versions
    pub fn list_versions(&self) -> Vec<&VersionMetadata> {
        self.lineage_graph
            .values()
            .map(|lineage| &lineage.version)
            .collect()
    }

    /// Get versions by tag
    pub fn get_versions_by_tag(&self, tag: &str) -> Vec<&VersionMetadata> {
        self.lineage_graph
            .values()
            .filter(|lineage| lineage.version.tags.contains(&tag.to_string()))
            .map(|lineage| &lineage.version)
            .collect()
    }

    /// Get the lineage tree starting from a version
    pub fn get_lineage_tree(&self, version_id: &str) -> Option<LineageTree> {
        self.lineage_graph
            .get(version_id)
            .map(|lineage| self.build_lineage_tree(&lineage.version))
    }

    fn build_lineage_tree(&self, version: &VersionMetadata) -> LineageTree {
        let children = version.version_id.clone();
        let child_trees = if let Some(lineage) = self.lineage_graph.get(&children) {
            lineage
                .child_versions
                .iter()
                .filter_map(|child_id| {
                    self.lineage_graph
                        .get(child_id)
                        .map(|child_lineage| self.build_lineage_tree(&child_lineage.version))
                })
                .collect()
        } else {
            Vec::new()
        };

        LineageTree {
            version: version.clone(),
            children: child_trees,
        }
    }

    fn generate_version_id(&self) -> VersionId {
        format!("v_{}", uuid::Uuid::new_v4().to_string().replace('-', ""))
    }

    fn calculate_size_info<T>(&self, dataset: &dyn Dataset<T>) -> Result<DatasetSizeInfo>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        let sample_count = dataset.len();

        if sample_count == 0 {
            return Ok(DatasetSizeInfo {
                sample_count: 0,
                feature_shape: vec![0],
                label_shape: vec![0],
                size_bytes: 0,
            });
        }

        // Get first sample to determine shapes
        let (features, labels) = dataset.get(0)?;
        let feature_shape = features.shape().dims().to_vec();
        let label_shape = labels.shape().dims().to_vec();

        // Estimate size (rough calculation)
        let feature_size = feature_shape.iter().product::<usize>();
        let label_size = label_shape.iter().product::<usize>();
        let estimated_bytes_per_sample = (feature_size + label_size) * std::mem::size_of::<f32>();
        let size_bytes = (sample_count * estimated_bytes_per_sample) as u64;

        Ok(DatasetSizeInfo {
            sample_count,
            feature_shape,
            label_shape,
            size_bytes,
        })
    }

    fn calculate_checksum<T>(&self, dataset: &dyn Dataset<T>) -> Result<String>
    where
        T: Clone + Default + Send + Sync + 'static,
    {
        // Simple checksum based on dataset length and tensor shapes
        let len = dataset.len();
        if len == 0 {
            return Ok("empty_dataset".to_string());
        }

        let (first_features, first_labels) = dataset.get(0)?;
        let mut checksum_value = 0u64;

        // Include dataset length in checksum
        checksum_value = checksum_value.wrapping_mul(31).wrapping_add(len as u64);

        // Include feature shape
        for &dim in first_features.shape().dims() {
            checksum_value = checksum_value.wrapping_mul(31).wrapping_add(dim as u64);
        }

        // Include label shape
        for &dim in first_labels.shape().dims() {
            checksum_value = checksum_value.wrapping_mul(31).wrapping_add(dim as u64);
        }

        // Use hash of string representation for additional randomness
        let features_hash = format!("{:?}", first_features.shape().dims()).len() as u64;
        let labels_hash = format!("{:?}", first_labels.shape().dims()).len() as u64;

        checksum_value = checksum_value.wrapping_mul(31).wrapping_add(features_hash);
        checksum_value = checksum_value.wrapping_mul(31).wrapping_add(labels_hash);

        Ok(format!("{checksum_value:016x}"))
    }

    fn save_dataset_samples<T>(&self, dataset: &dyn Dataset<T>, version_dir: &Path) -> Result<()>
    where
        T: Clone + Default + serde::Serialize + Send + Sync + 'static,
    {
        let samples_file = version_dir.join("samples.json");
        let mut samples = Vec::new();

        for i in 0..dataset.len() {
            let (features, labels) = dataset.get(i)?;

            // Convert tensors to serializable format
            let features_data = if let Some(slice) = features.as_slice() {
                slice.to_vec()
            } else {
                vec![features.get(&[]).unwrap_or(T::default())]
            };

            let labels_data = if let Some(slice) = labels.as_slice() {
                slice.to_vec()
            } else {
                vec![labels.get(&[]).unwrap_or(T::default())]
            };

            samples.push(serde_json::json!({
                "features": features_data,
                "labels": labels_data,
                "feature_shape": features.shape().dims(),
                "label_shape": labels.shape().dims(),
            }));
        }

        let json_data = serde_json::to_string_pretty(&samples).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to serialize samples: {e}"))
        })?;

        std::fs::write(samples_file, json_data).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to write samples file: {e}"))
        })?;

        Ok(())
    }

    fn load_dataset_samples<T>(&self, version_dir: &Path) -> Result<Vec<(Tensor<T>, Tensor<T>)>>
    where
        T: Clone + Default + serde::de::DeserializeOwned + Send + Sync + 'static,
    {
        let samples_file = version_dir.join("samples.json");
        let json_data = std::fs::read_to_string(samples_file).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to read samples file: {e}"))
        })?;

        let json_samples: Vec<serde_json::Value> =
            serde_json::from_str(&json_data).map_err(|e| {
                TensorError::invalid_argument(format!("Failed to parse samples JSON: {e}"))
            })?;

        let mut samples = Vec::new();
        for sample in json_samples {
            let features_data: Vec<T> = serde_json::from_value(sample["features"].clone())
                .map_err(|e| {
                    TensorError::invalid_argument(format!("Failed to parse features: {e}"))
                })?;

            let labels_data: Vec<T> =
                serde_json::from_value(sample["labels"].clone()).map_err(|e| {
                    TensorError::invalid_argument(format!("Failed to parse labels: {e}"))
                })?;

            let feature_shape: Vec<usize> = serde_json::from_value(sample["feature_shape"].clone())
                .map_err(|e| {
                    TensorError::invalid_argument(format!("Failed to parse feature shape: {e}"))
                })?;

            let label_shape: Vec<usize> = serde_json::from_value(sample["label_shape"].clone())
                .map_err(|e| {
                    TensorError::invalid_argument(format!("Failed to parse label shape: {e}"))
                })?;

            let features_tensor = if feature_shape.is_empty() || feature_shape == vec![0] {
                Tensor::from_scalar(features_data.into_iter().next().unwrap_or_default())
            } else {
                Tensor::from_vec(features_data, &feature_shape)?
            };

            let labels_tensor = if label_shape.is_empty() || label_shape == vec![0] {
                Tensor::from_scalar(labels_data.into_iter().next().unwrap_or_default())
            } else {
                Tensor::from_vec(labels_data, &label_shape)?
            };

            samples.push((features_tensor, labels_tensor));
        }

        Ok(samples)
    }

    fn save_metadata(&self, metadata: &VersionMetadata, version_dir: &Path) -> Result<()> {
        let metadata_file = version_dir.join("metadata.json");
        let json_data = serde_json::to_string_pretty(metadata).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to serialize metadata: {e}"))
        })?;

        std::fs::write(metadata_file, json_data).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to write metadata file: {e}"))
        })?;

        Ok(())
    }

    fn load_metadata(&self, version_dir: &Path) -> Result<VersionMetadata> {
        let metadata_file = version_dir.join("metadata.json");
        let json_data = std::fs::read_to_string(metadata_file).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to read metadata file: {e}"))
        })?;

        serde_json::from_str(&json_data).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to parse metadata JSON: {e}"))
        })
    }

    fn save_lineage_graph(&self) -> Result<()> {
        let lineage_file = self.base_path.join("lineage.json");
        let json_data = serde_json::to_string_pretty(&self.lineage_graph).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to serialize lineage graph: {e}"))
        })?;

        std::fs::write(lineage_file, json_data).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to write lineage file: {e}"))
        })?;

        Ok(())
    }

    fn load_lineage_graph(&mut self) -> Result<()> {
        let lineage_file = self.base_path.join("lineage.json");

        if !lineage_file.exists() {
            return Ok(()); // No existing lineage graph
        }

        let json_data = std::fs::read_to_string(lineage_file).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to read lineage file: {e}"))
        })?;

        self.lineage_graph = serde_json::from_str(&json_data).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to parse lineage JSON: {e}"))
        })?;

        Ok(())
    }
}

/// Tree structure representing dataset lineage
#[derive(Debug, Clone)]
pub struct LineageTree {
    pub version: VersionMetadata,
    pub children: Vec<LineageTree>,
}

/// Versioned dataset that can be loaded from snapshots
#[derive(Debug)]
pub struct VersionedDataset<T> {
    metadata: VersionMetadata,
    samples: Vec<(Tensor<T>, Tensor<T>)>,
}

impl<T> VersionedDataset<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    /// Get the version metadata
    pub fn metadata(&self) -> &VersionMetadata {
        &self.metadata
    }

    /// Get the version ID
    pub fn version_id(&self) -> &str {
        &self.metadata.version_id
    }
}

impl<T> Dataset<T> for VersionedDataset<T>
where
    T: Clone + Default + Send + Sync + 'static,
{
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.samples.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index,
                self.samples.len()
            )));
        }

        Ok(self.samples[index].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorDataset;
    use tempfile::TempDir;

    #[test]
    fn test_version_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = DatasetVersionManager::new(temp_dir.path()).unwrap();

        assert!(temp_dir.path().exists());
        assert_eq!(manager.list_versions().len(), 0);
    }

    #[test]
    fn test_create_and_load_snapshot() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = DatasetVersionManager::new(temp_dir.path()).unwrap();

        // Create a simple test dataset
        let features_data = vec![1.0, 2.0, 3.0, 4.0];
        let labels_data = vec![0.0, 1.0];
        let features = Tensor::from_vec(features_data, &[2, 2]).unwrap();
        let labels = Tensor::from_vec(labels_data, &[2]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        // Create snapshot
        let version_id = manager
            .create_snapshot(
                &dataset,
                "Test snapshot".to_string(),
                vec!["test".to_string()],
                None,
            )
            .unwrap();

        assert!(!version_id.is_empty());
        assert_eq!(manager.list_versions().len(), 1);

        // Load snapshot
        let loaded_dataset = manager.load_snapshot::<f32>(&version_id).unwrap();
        assert_eq!(loaded_dataset.len(), 2);
        assert_eq!(loaded_dataset.version_id(), &version_id);

        // Test loaded data
        let (features, labels) = loaded_dataset.get(0).unwrap();
        let features_slice = features.as_slice().unwrap();
        assert_eq!(features_slice, &[1.0, 2.0]);
        assert_eq!(labels.get(&[]).unwrap(), 0.0);
    }

    #[test]
    fn test_lineage_tracking() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = DatasetVersionManager::new(temp_dir.path()).unwrap();

        // Create initial dataset
        let features_data1 = vec![1.0, 2.0];
        let labels_data1 = vec![0.0];
        let features1 = Tensor::from_vec(features_data1, &[1, 2]).unwrap();
        let labels1 = Tensor::from_vec(labels_data1, &[1]).unwrap();
        let dataset1 = TensorDataset::new(features1, labels1);

        let version1 = manager
            .create_snapshot(
                &dataset1,
                "Initial version".to_string(),
                vec!["v1".to_string()],
                None,
            )
            .unwrap();

        // Create derived dataset
        let features_data2 = vec![2.0, 4.0];
        let labels_data2 = vec![1.0];
        let features2 = Tensor::from_vec(features_data2, &[1, 2]).unwrap();
        let labels2 = Tensor::from_vec(labels_data2, &[1]).unwrap();
        let dataset2 = TensorDataset::new(features2, labels2);

        let version2 = manager
            .create_snapshot(
                &dataset2,
                "Scaled version".to_string(),
                vec!["v2".to_string()],
                Some(version1.clone()),
            )
            .unwrap();

        // Add transformation record
        let mut params = HashMap::new();
        params.insert("scale_factor".to_string(), "2.0".to_string());

        manager
            .add_transformation(
                &version2,
                "scale".to_string(),
                params,
                "Scale features by 2".to_string(),
            )
            .unwrap();

        // Test lineage
        let lineage = manager.get_lineage(&version2).unwrap();
        assert_eq!(lineage.source_versions, vec![version1.clone()]);
        assert_eq!(lineage.transformations.len(), 1);
        assert_eq!(lineage.transformations[0].transform_type, "scale");

        // Test lineage tree
        let tree = manager.get_lineage_tree(&version1).unwrap();
        assert_eq!(tree.version.version_id, version1);
        assert_eq!(tree.children.len(), 1);
        assert_eq!(tree.children[0].version.version_id, version2);
    }

    #[test]
    fn test_version_filtering() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = DatasetVersionManager::new(temp_dir.path()).unwrap();

        // Create multiple versions with different tags
        let features_data = vec![1.0];
        let labels_data = vec![0.0];
        let features = Tensor::from_vec(features_data, &[1, 1]).unwrap();
        let labels = Tensor::from_vec(labels_data, &[1]).unwrap();
        let dataset = TensorDataset::new(features, labels);

        let _version1 = manager
            .create_snapshot(
                &dataset,
                "Version 1".to_string(),
                vec!["production".to_string()],
                None,
            )
            .unwrap();

        let _version2 = manager
            .create_snapshot(
                &dataset,
                "Version 2".to_string(),
                vec!["development".to_string()],
                None,
            )
            .unwrap();

        let _version3 = manager
            .create_snapshot(
                &dataset,
                "Version 3".to_string(),
                vec!["production".to_string(), "validated".to_string()],
                None,
            )
            .unwrap();

        // Test filtering by tag
        let prod_versions = manager.get_versions_by_tag("production");
        assert_eq!(prod_versions.len(), 2);

        let dev_versions = manager.get_versions_by_tag("development");
        assert_eq!(dev_versions.len(), 1);

        let validated_versions = manager.get_versions_by_tag("validated");
        assert_eq!(validated_versions.len(), 1);
    }
}
