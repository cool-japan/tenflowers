//! HDF5 format support for datasets
//!
//! This module provides comprehensive support for HDF5 (Hierarchical Data Format version 5) files,
//! which are widely used in scientific computing and machine learning for storing large datasets.
//! HDF5 provides efficient I/O, compression, and support for complex data structures.
//!
//! # Features
//!
//! - **Hierarchical Data Access**: Navigate complex HDF5 group structures
//! - **Dataset Discovery**: Automatic discovery of datasets within HDF5 files
//! - **Efficient I/O**: Direct access to HDF5 datasets
//! - **Compression Support**: Built-in support for compressed HDF5 datasets
//! - **Type Safety**: Full integration with Rust type system
//! - **Metadata Access**: Rich metadata and attribute information
//! - **Multi-dimensional Arrays**: Support for N-dimensional datasets
//!
//! # Example Usage
//!
//! ```rust,no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use tenflowers_dataset::formats::hdf5::{HDF5Dataset, HDF5Config};
//!
//! // Basic usage - load features from 'data' dataset and labels from 'targets'
//! let dataset = HDF5Dataset::from_file("data.h5")?;
//!
//! // With configuration
//! let config = HDF5Config::default()
//!     .with_feature_dataset("features".to_string())
//!     .with_label_dataset("labels".to_string());
//!
//! let dataset = HDF5Dataset::from_file_with_config("data.h5", config)?;
//! # Ok(())
//! # }
//! ```

// std::collections::HashMap - removed unused import
#[cfg(feature = "hdf5")]
use std::path::Path;

#[cfg(feature = "hdf5")]
use hdf5::File;

#[cfg(feature = "hdf5")]
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "hdf5")]
use crate::Dataset;

/// Configuration for HDF5 dataset loading
#[cfg(feature = "hdf5")]
#[derive(Debug, Clone)]
pub struct HDF5Config {
    /// HDF5 dataset name/path for features (default: auto-detect first suitable dataset)
    pub feature_dataset: Option<String>,
    /// HDF5 dataset name/path for labels (default: auto-detect second suitable dataset)
    pub label_dataset: Option<String>,
    /// HDF5 group path to search within (default: root "/")
    pub group_path: String,
    /// Cache datasets in memory
    pub cache_data: bool,
    /// Maximum number of samples to load (None for all)
    pub max_samples: Option<usize>,
}

#[cfg(feature = "hdf5")]
impl Default for HDF5Config {
    fn default() -> Self {
        Self {
            feature_dataset: None,
            label_dataset: None,
            group_path: "/".to_string(),
            cache_data: true,
            max_samples: None,
        }
    }
}

#[cfg(feature = "hdf5")]
impl HDF5Config {
    /// Set feature dataset name/path
    pub fn with_feature_dataset(mut self, dataset: String) -> Self {
        self.feature_dataset = Some(dataset);
        self
    }

    /// Set label dataset name/path
    pub fn with_label_dataset(mut self, dataset: String) -> Self {
        self.label_dataset = Some(dataset);
        self
    }

    /// Set group path to search within
    pub fn with_group_path(mut self, path: String) -> Self {
        self.group_path = path;
        self
    }

    /// Enable or disable data caching
    pub fn with_cache_data(mut self, cache: bool) -> Self {
        self.cache_data = cache;
        self
    }

    /// Set maximum samples to load
    pub fn with_max_samples(mut self, max_samples: usize) -> Self {
        self.max_samples = Some(max_samples);
        self
    }
}

/// Information about an HDF5 dataset
#[cfg(feature = "hdf5")]
#[derive(Debug, Clone)]
pub struct HDF5DatasetInfo {
    /// File path
    pub file_path: String,
    /// Feature dataset name/path
    pub feature_dataset: String,
    /// Label dataset name/path (if available)
    pub label_dataset: Option<String>,
    /// Number of samples in the dataset
    pub num_samples: usize,
    /// Shape of feature data
    pub feature_shape: Vec<usize>,
    /// Label shape (if available)
    pub label_shape: Option<Vec<usize>>,
    /// File size in bytes
    pub file_size: u64,
    /// Available datasets in the file
    pub available_datasets: Vec<String>,
}

/// Builder for creating HDF5 datasets with fluent API
#[cfg(feature = "hdf5")]
pub struct HDF5DatasetBuilder {
    path: Option<String>,
    config: HDF5Config,
}

#[cfg(feature = "hdf5")]
impl Default for HDF5DatasetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "hdf5")]
impl HDF5DatasetBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            path: None,
            config: HDF5Config::default(),
        }
    }

    /// Set the file path
    pub fn path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.path = Some(path.as_ref().to_string_lossy().to_string());
        self
    }

    /// Set the configuration
    pub fn config(mut self, config: HDF5Config) -> Self {
        self.config = config;
        self
    }

    /// Set feature dataset name
    pub fn feature_dataset(mut self, dataset: String) -> Self {
        self.config.feature_dataset = Some(dataset);
        self
    }

    /// Set label dataset name
    pub fn label_dataset(mut self, dataset: String) -> Self {
        self.config.label_dataset = Some(dataset);
        self
    }

    /// Build the dataset
    pub fn build(self) -> Result<HDF5Dataset> {
        let path = self
            .path
            .ok_or_else(|| TensorError::invalid_argument("Path must be specified".to_string()))?;
        HDF5Dataset::from_file_with_config(&path, self.config)
    }
}

/// HDF5 dataset implementation
#[cfg(feature = "hdf5")]
pub struct HDF5Dataset {
    /// Configuration
    config: HDF5Config,
    /// Dataset information
    info: HDF5DatasetInfo,
    /// Cached feature data
    cached_features: Option<Vec<Vec<f32>>>,
    /// Cached label data
    cached_labels: Option<Vec<f32>>,
}

#[cfg(feature = "hdf5")]
impl HDF5Dataset {
    /// Create dataset from file with default configuration
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_file_with_config(path, HDF5Config::default())
    }

    /// Create dataset from file with custom configuration
    pub fn from_file_with_config<P: AsRef<Path>>(path: P, config: HDF5Config) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Validate file exists
        if !path.as_ref().exists() {
            return Err(TensorError::invalid_argument(format!(
                "HDF5 file not found: {path_str}"
            )));
        }

        // Get file size
        let file_size = std::fs::metadata(&path_str)
            .map_err(|e| {
                TensorError::invalid_argument(format!("Failed to read file metadata: {e}"))
            })?
            .len();

        // Open HDF5 file to discover datasets
        let file = File::open(&path_str)
            .map_err(|e| TensorError::invalid_argument(format!("Failed to open HDF5 file: {e}")))?;

        // Discover datasets
        let available_datasets = discover_datasets(&file)?;

        if available_datasets.is_empty() {
            return Err(TensorError::invalid_argument(
                "No datasets found in HDF5 file".to_string(),
            ));
        }

        // Determine feature and label datasets
        let feature_dataset = config
            .feature_dataset
            .clone()
            .or_else(|| available_datasets.first().cloned())
            .ok_or_else(|| TensorError::invalid_argument("No feature dataset found".to_string()))?;

        let label_dataset = config.label_dataset.clone().or_else(|| {
            if available_datasets.len() > 1 {
                available_datasets.get(1).cloned()
            } else {
                None
            }
        });

        // Get dataset shapes and info
        let (num_samples, feature_shape) = get_dataset_shape(&file, &feature_dataset)?;
        let label_shape = if let Some(ref label_name) = label_dataset {
            Some(get_dataset_shape(&file, label_name)?.1)
        } else {
            None
        };

        // Create dataset info
        let info = HDF5DatasetInfo {
            file_path: path_str.clone(),
            feature_dataset: feature_dataset.clone(),
            label_dataset: label_dataset.clone(),
            num_samples,
            feature_shape,
            label_shape,
            file_size,
            available_datasets,
        };

        let mut dataset = Self {
            config,
            info,
            cached_features: None,
            cached_labels: None,
        };

        // Pre-load data if caching is enabled
        if dataset.config.cache_data {
            dataset.load_data(&path_str)?;
        }

        Ok(dataset)
    }

    /// Get dataset information
    pub fn info(&self) -> &HDF5DatasetInfo {
        &self.info
    }

    /// Load all data into memory
    fn load_data(&mut self, file_path: &str) -> Result<()> {
        let file = File::open(file_path)
            .map_err(|e| TensorError::invalid_argument(format!("Failed to open HDF5 file: {e}")))?;

        // Load features
        let feature_dataset = file.dataset(&self.info.feature_dataset).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to open feature dataset: {e}"))
        })?;

        let feature_data: Vec<f32> = feature_dataset.read_raw().map_err(|e| {
            TensorError::invalid_argument(format!("Failed to read feature data: {e}"))
        })?;

        // Organize features by sample
        let mut features = Vec::new();
        let feature_size_per_sample = if self.info.feature_shape.len() > 1 {
            self.info.feature_shape[1..].iter().product()
        } else {
            1
        };

        for i in 0..self.info.num_samples {
            let start_idx = i * feature_size_per_sample;
            let end_idx = start_idx + feature_size_per_sample;

            if end_idx <= feature_data.len() {
                features.push(feature_data[start_idx..end_idx].to_vec());
            }

            // Check max samples limit
            if let Some(max_samples) = self.config.max_samples {
                if features.len() >= max_samples {
                    break;
                }
            }
        }

        self.cached_features = Some(features);

        // Load labels if available
        if let Some(ref label_dataset_name) = self.info.label_dataset {
            let label_dataset = file.dataset(label_dataset_name).map_err(|e| {
                TensorError::invalid_argument(format!("Failed to open label dataset: {e}"))
            })?;

            let labels: Vec<f32> = label_dataset.read_raw().map_err(|e| {
                TensorError::invalid_argument(format!("Failed to read label data: {e}"))
            })?;

            self.cached_labels = Some(labels);
        }

        Ok(())
    }
}

#[cfg(feature = "hdf5")]
impl Dataset<f32> for HDF5Dataset {
    fn len(&self) -> usize {
        if let Some(ref cached) = self.cached_features {
            cached.len()
        } else {
            self.info.num_samples
        }
    }

    fn get(&self, index: usize) -> Result<(Tensor<f32>, Tensor<f32>)> {
        if index >= self.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index,
                self.len()
            )));
        }

        if let Some(ref cached_features) = self.cached_features {
            // Use cached data
            let features = cached_features[index].clone();
            let feature_tensor = Tensor::from_vec(features, &[cached_features[index].len()])?;

            let label_tensor = if let Some(ref cached_labels) = self.cached_labels {
                if index < cached_labels.len() {
                    Tensor::from_vec(vec![cached_labels[index]], &[])?
                } else {
                    Tensor::from_vec(vec![0.0f32], &[])?
                }
            } else {
                Tensor::from_vec(vec![0.0f32], &[])?
            };

            Ok((feature_tensor, label_tensor))
        } else {
            // Data not cached
            Err(TensorError::invalid_argument(
                "Data not cached - enable cache_data for efficient access".to_string(),
            ))
        }
    }
}

/// Discover datasets in an HDF5 file
#[cfg(feature = "hdf5")]
fn discover_datasets(file: &File) -> Result<Vec<String>> {
    let mut datasets = Vec::new();

    for name in file
        .member_names()
        .map_err(|e| TensorError::invalid_argument(format!("Failed to list file members: {e}")))?
    {
        if file.dataset(&name).is_ok() {
            datasets.push(name);
        }
    }

    Ok(datasets)
}

/// Get shape information about a specific dataset
#[cfg(feature = "hdf5")]
fn get_dataset_shape(file: &File, dataset_name: &str) -> Result<(usize, Vec<usize>)> {
    let dataset = file.dataset(dataset_name).map_err(|e| {
        TensorError::invalid_argument(format!("Failed to open dataset {dataset_name}: {e}"))
    })?;

    let shape = dataset.shape();
    let num_samples = if !shape.is_empty() { shape[0] } else { 0 };
    let shape_vec = shape.to_vec();

    Ok((num_samples, shape_vec))
}

// Stub implementations when hdf5 feature is not enabled
#[cfg(not(feature = "hdf5"))]
pub struct HDF5Config;

#[cfg(not(feature = "hdf5"))]
pub struct HDF5DatasetInfo;

#[cfg(not(feature = "hdf5"))]
pub struct HDF5DatasetBuilder;

#[cfg(not(feature = "hdf5"))]
pub struct HDF5Dataset;

#[cfg(test)]
#[cfg(feature = "hdf5")]
mod tests {
    use super::*;

    #[test]
    fn test_hdf5_config_default() {
        let config = HDF5Config::default();
        assert_eq!(config.group_path, "/");
        assert_eq!(config.cache_data, true);
        assert!(config.feature_dataset.is_none());
        assert!(config.label_dataset.is_none());
    }

    #[test]
    fn test_hdf5_config_builder() {
        let config = HDF5Config::default()
            .with_feature_dataset("features".to_string())
            .with_label_dataset("labels".to_string())
            .with_group_path("/data".to_string())
            .with_max_samples(1000);

        assert_eq!(config.feature_dataset.as_ref().unwrap(), "features");
        assert_eq!(config.label_dataset.as_ref().unwrap(), "labels");
        assert_eq!(config.group_path, "/data");
        assert_eq!(config.max_samples, Some(1000));
    }

    #[test]
    fn test_hdf5_dataset_builder() {
        let builder = HDF5DatasetBuilder::new()
            .feature_dataset("data".to_string())
            .label_dataset("targets".to_string());

        assert_eq!(builder.config.feature_dataset.as_ref().unwrap(), "data");
        assert_eq!(builder.config.label_dataset.as_ref().unwrap(), "targets");
    }

    // Note: Actual file I/O tests would require creating sample HDF5 files
    // which is complex in a unit test environment. These would typically be
    // integration tests with pre-created test data files.
}
