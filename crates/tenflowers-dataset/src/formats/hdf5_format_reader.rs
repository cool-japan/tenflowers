//! HDF5 format factory and reader implementation
//!
//! This module implements the FormatFactory and FormatReader traits for HDF5 files,
//! enabling automatic format detection and unified data loading for scientific datasets.

#[cfg(feature = "hdf5")]
use crate::error_taxonomy::helpers as error_helpers;
#[cfg(feature = "hdf5")]
use crate::formats::unified_reader::{
    read_magic_bytes, DataType, DetectionMethod, FieldInfo, FormatDetection, FormatFactory,
    FormatMetadata, FormatReader, FormatSample,
};
#[cfg(feature = "hdf5")]
use std::collections::HashMap;
#[cfg(feature = "hdf5")]
use std::path::{Path, PathBuf};
#[cfg(feature = "hdf5")]
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "hdf5")]
use hdf5::{Dataset as HDF5Dataset, File};

/// HDF5 format factory for automatic detection and reader creation
#[cfg(feature = "hdf5")]
pub struct HDF5FormatFactory;

#[cfg(feature = "hdf5")]
impl FormatFactory for HDF5FormatFactory {
    fn format_name(&self) -> &str {
        "HDF5"
    }

    fn extensions(&self) -> Vec<&str> {
        vec!["h5", "hdf5", "he5"]
    }

    fn can_read(&self, path: &Path) -> Result<FormatDetection> {
        // Check file extension
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|s| s.to_lowercase());

        let mut confidence = 0.0;
        let mut method = DetectionMethod::Extension;

        match extension.as_deref() {
            Some("h5") | Some("hdf5") | Some("he5") => {
                confidence = 0.95;
                method = DetectionMethod::Extension;
            }
            _ => {
                // Try magic bytes detection (HDF5 signature)
                if let Ok(is_hdf5) = Self::check_hdf5_magic(path) {
                    if is_hdf5 {
                        confidence = 0.99;
                        method = DetectionMethod::MagicBytes;
                    }
                }
            }
        }

        Ok(FormatDetection {
            format_name: self.format_name().to_string(),
            confidence,
            method,
        })
    }

    fn create_reader(&self, path: &Path) -> Result<Box<dyn FormatReader>> {
        Ok(Box::new(HDF5FormatReader::new(path)?))
    }
}

#[cfg(feature = "hdf5")]
impl HDF5FormatFactory {
    /// Check HDF5 magic bytes
    fn check_hdf5_magic(path: &Path) -> Result<bool> {
        if let Ok(bytes) = read_magic_bytes(path, 8) {
            // HDF5 files start with signature: \x89HDF\r\n\x1a\n
            Ok(bytes.len() >= 4 && bytes[1..4] == *b"HDF")
        } else {
            Ok(false)
        }
    }
}

/// HDF5 format reader implementation
#[cfg(feature = "hdf5")]
pub struct HDF5FormatReader {
    path: PathBuf,
    metadata: FormatMetadata,
    feature_dataset_name: String,
    label_dataset_name: Option<String>,
    cached_features: Vec<Vec<f32>>,
    cached_labels: Vec<f32>,
}

#[cfg(feature = "hdf5")]
impl HDF5FormatReader {
    /// Create a new HDF5 format reader
    pub fn new(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            error_helpers::data_corruption(
                "HDF5FormatReader::new",
                format!("Failed to open HDF5 file: {}", e),
                Some(path.to_path_buf()),
            )
        })?;

        // Discover datasets
        let dataset_names = Self::discover_datasets(&file)?;

        if dataset_names.is_empty() {
            return Err(error_helpers::data_corruption(
                "HDF5FormatReader::new",
                "No datasets found in HDF5 file",
                Some(path.to_path_buf()),
            ));
        }

        // Find feature and label datasets
        let (feature_name, label_name) = Self::identify_feature_label_datasets(&dataset_names);

        // Load data
        let (cached_features, cached_labels) =
            Self::load_hdf5_data(&file, &feature_name, label_name.as_deref())?;

        let num_samples = cached_features.len();

        // Create field info
        let mut fields = vec![FieldInfo {
            name: feature_name.clone(),
            dtype: DataType::Float32,
            shape: Some(vec![cached_features.first().map(|f| f.len()).unwrap_or(0)]),
            nullable: false,
            description: Some("Feature data".to_string()),
        }];

        if let Some(name) = &label_name {
            fields.push(FieldInfo {
                name: name.clone(),
                dtype: DataType::Float32,
                shape: Some(vec![1]),
                nullable: false,
                description: Some("Label data".to_string()),
            });
        }

        let metadata = FormatMetadata {
            format_name: "HDF5".to_string(),
            version: None,
            num_samples,
            fields,
            metadata: HashMap::new(),
            supports_random_access: true,
            supports_streaming: false, // HDF5 requires full file access
        };

        Ok(Self {
            path: path.to_path_buf(),
            metadata,
            feature_dataset_name: feature_name,
            label_dataset_name: label_name,
            cached_features,
            cached_labels,
        })
    }

    /// Discover all datasets in HDF5 file
    fn discover_datasets(file: &File) -> Result<Vec<String>> {
        let mut dataset_names = Vec::new();

        // Simple discovery: check common dataset names
        let common_names = vec![
            "data",
            "features",
            "X",
            "x",
            "train_data",
            "test_data",
            "labels",
            "targets",
            "y",
            "Y",
        ];

        for name in common_names {
            if file.dataset(name).is_ok() {
                dataset_names.push(name.to_string());
            }
        }

        Ok(dataset_names)
    }

    /// Identify which datasets are features and labels
    fn identify_feature_label_datasets(names: &[String]) -> (String, Option<String>) {
        let feature_candidates = ["features", "data", "X", "x", "train_data"];
        let label_candidates = ["labels", "targets", "y", "Y"];

        let feature_name = names
            .iter()
            .find(|name| feature_candidates.iter().any(|c| name.contains(c)))
            .cloned()
            .or_else(|| names.first().cloned())
            .unwrap_or_else(|| "data".to_string());

        let label_name = names
            .iter()
            .find(|name| label_candidates.iter().any(|c| name.contains(c)))
            .cloned();

        (feature_name, label_name)
    }

    /// Load HDF5 data into memory
    fn load_hdf5_data(
        file: &File,
        feature_name: &str,
        label_name: Option<&str>,
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
        let feature_ds = file.dataset(feature_name).map_err(|e| {
            TensorError::io_error_simple(format!(
                "Failed to open feature dataset '{}': {}",
                feature_name, e
            ))
        })?;

        let feature_data: Vec<Vec<f32>> = match feature_ds.read_2d::<f32>() {
            Ok(arr) => {
                // Convert ndarray to Vec<Vec<f32>>
                arr.outer_iter().map(|row| row.to_vec()).collect()
            }
            Err(_) => {
                // Try 1D read as fallback
                let data_1d = feature_ds.read_1d::<f32>().map_err(|e| {
                    TensorError::io_error_simple(format!("Failed to read feature data: {}", e))
                })?;
                let data_vec: Vec<f32> = data_1d.to_vec();
                data_vec.into_iter().map(|v| vec![v]).collect()
            }
        };

        let label_data = if let Some(label_name) = label_name {
            if let Ok(label_ds) = file.dataset(label_name) {
                label_ds
                    .read_1d::<f32>()
                    .map(|arr| arr.to_vec())
                    .unwrap_or_else(|_| vec![0.0; feature_data.len()])
            } else {
                vec![0.0; feature_data.len()]
            }
        } else {
            vec![0.0; feature_data.len()]
        };

        Ok((feature_data, label_data))
    }
}

#[cfg(feature = "hdf5")]
impl FormatReader for HDF5FormatReader {
    fn metadata(&self) -> Result<FormatMetadata> {
        Ok(self.metadata.clone())
    }

    fn get_sample(&self, index: usize) -> Result<FormatSample> {
        if index >= self.cached_features.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index,
                self.cached_features.len()
            )));
        }

        let features = Tensor::from_vec(
            self.cached_features[index].clone(),
            &[self.cached_features[index].len()],
        )?;

        let labels = Tensor::from_vec(vec![self.cached_labels[index]], &[1])?;

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "HDF5".to_string());
        metadata.insert("index".to_string(), index.to_string());
        metadata.insert(
            "feature_dataset".to_string(),
            self.feature_dataset_name.clone(),
        );
        if let Some(ref label_name) = self.label_dataset_name {
            metadata.insert("label_dataset".to_string(), label_name.clone());
        }

        Ok(FormatSample {
            features,
            labels,
            source_index: index,
            metadata,
        })
    }

    fn iter(&self) -> Box<dyn Iterator<Item = Result<FormatSample>> + '_> {
        Box::new((0..self.cached_features.len()).map(move |i| self.get_sample(i)))
    }

    fn len(&self) -> usize {
        self.cached_features.len()
    }
}

#[cfg(test)]
#[cfg(feature = "hdf5")]
mod tests {
    use super::*;

    #[test]
    fn test_hdf5_format_detection() {
        let factory = HDF5FormatFactory;

        let h5_path = Path::new("data.h5");
        let detection = factory
            .can_read(h5_path)
            .expect("test: format detection should succeed");
        assert!(detection.confidence >= 0.9);
        assert_eq!(detection.format_name, "HDF5");
    }

    #[test]
    fn test_dataset_identification() {
        let names = vec![
            "features".to_string(),
            "labels".to_string(),
            "metadata".to_string(),
        ];

        let (feature_name, label_name) = HDF5FormatReader::identify_feature_label_datasets(&names);

        assert_eq!(feature_name, "features");
        assert_eq!(label_name, Some("labels".to_string()));
    }
}
