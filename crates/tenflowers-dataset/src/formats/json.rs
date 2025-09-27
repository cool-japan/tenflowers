//! JSON dataset format support
//!
//! This module provides support for loading datasets from JSON and JSON Lines (JSONL) files.
//! JSON datasets are useful for structured data, while JSONL is particularly common for
//! streaming large datasets and NLP tasks.

use crate::Dataset;
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Configuration for JSON dataset loading
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct JsonConfig {
    /// Path to the JSON file
    pub file_path: PathBuf,
    /// Field name containing the features (input data)
    pub feature_field: String,
    /// Field name containing the labels (target data)
    pub label_field: String,
    /// Whether to parse arrays as tensors automatically
    pub auto_parse_arrays: bool,
    /// Maximum number of samples to load (None for all)
    pub max_samples: Option<usize>,
    /// Whether to normalize numeric features to [0, 1]
    pub normalize_features: bool,
    /// Feature dimension (for reshaping flattened arrays)
    pub feature_shape: Option<Vec<usize>>,
    /// Label dimension (for reshaping flattened arrays)
    pub label_shape: Option<Vec<usize>>,
}

impl Default for JsonConfig {
    fn default() -> Self {
        Self {
            file_path: PathBuf::new(),
            feature_field: "features".to_string(),
            label_field: "label".to_string(),
            auto_parse_arrays: true,
            max_samples: None,
            normalize_features: false,
            feature_shape: None,
            label_shape: None,
        }
    }
}

/// JSON dataset for loading structured data from JSON files
#[derive(Debug, Clone)]
pub struct JsonDataset<T> {
    samples: Vec<(Tensor<T>, Tensor<T>)>,
    config: JsonConfig,
}

impl<T> JsonDataset<T>
where
    T: Clone + Default + num_traits::Zero + num_traits::Float + Send + Sync + 'static,
    T: serde::de::DeserializeOwned + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    /// Create a new JSON dataset from configuration
    pub fn from_config(config: JsonConfig) -> Result<Self> {
        let samples = Self::load_json_samples(&config)?;
        Ok(Self { samples, config })
    }

    /// Create a new JSON dataset from file path
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        feature_field: &str,
        label_field: &str,
    ) -> Result<Self> {
        let config = JsonConfig {
            file_path: path.as_ref().to_path_buf(),
            feature_field: feature_field.to_string(),
            label_field: label_field.to_string(),
            ..Default::default()
        };
        Self::from_config(config)
    }

    /// Load samples from JSON file
    fn load_json_samples(config: &JsonConfig) -> Result<Vec<(Tensor<T>, Tensor<T>)>> {
        let file = File::open(&config.file_path)
            .map_err(|e| TensorError::invalid_argument(format!("Cannot open JSON file: {e}")))?;

        let reader = BufReader::new(file);
        let json_value: serde_json::Value = serde_json::from_reader(reader)
            .map_err(|e| TensorError::invalid_argument(format!("Cannot parse JSON: {e}")))?;

        let mut samples = Vec::new();

        // Handle array of objects or single object
        let objects = match &json_value {
            serde_json::Value::Array(arr) => arr,
            serde_json::Value::Object(_) => {
                return Err(TensorError::invalid_argument(
                    "Single JSON object not supported, expected array of objects".to_string(),
                ));
            }
            _ => {
                return Err(TensorError::invalid_argument(
                    "Invalid JSON format, expected array of objects".to_string(),
                ))
            }
        };

        let max_samples = config.max_samples.unwrap_or(objects.len());

        for (idx, obj) in objects.iter().enumerate() {
            if idx >= max_samples {
                break;
            }

            if let serde_json::Value::Object(map) = obj {
                let features =
                    Self::extract_features(map, &config.feature_field, &config.feature_shape)?;
                let labels = Self::extract_labels(map, &config.label_field, &config.label_shape)?;

                samples.push((features, labels));
            } else {
                return Err(TensorError::invalid_argument(
                    "Expected JSON object in array".to_string(),
                ));
            }
        }

        if samples.is_empty() {
            return Err(TensorError::invalid_argument(
                "No valid samples found in JSON file".to_string(),
            ));
        }

        Ok(samples)
    }

    /// Extract features from JSON object
    fn extract_features(
        obj: &serde_json::Map<String, serde_json::Value>,
        field_name: &str,
        target_shape: &Option<Vec<usize>>,
    ) -> Result<Tensor<T>> {
        let feature_value = obj.get(field_name).ok_or_else(|| {
            TensorError::invalid_argument(format!("Feature field '{field_name}' not found"))
        })?;

        Self::json_value_to_tensor(feature_value, target_shape)
    }

    /// Extract labels from JSON object  
    fn extract_labels(
        obj: &serde_json::Map<String, serde_json::Value>,
        field_name: &str,
        target_shape: &Option<Vec<usize>>,
    ) -> Result<Tensor<T>> {
        let label_value = obj.get(field_name).ok_or_else(|| {
            TensorError::invalid_argument(format!("Label field '{field_name}' not found"))
        })?;

        Self::json_value_to_tensor(label_value, target_shape)
    }

    /// Convert JSON value to tensor
    fn json_value_to_tensor(
        value: &serde_json::Value,
        target_shape: &Option<Vec<usize>>,
    ) -> Result<Tensor<T>> {
        match value {
            serde_json::Value::Number(n) => {
                let val = n.as_f64().ok_or_else(|| {
                    TensorError::invalid_argument("Cannot convert number to f64".to_string())
                })?;
                let tensor_val = T::from(val as f32).unwrap();
                Ok(Tensor::from_scalar(tensor_val))
            }
            serde_json::Value::Array(arr) => {
                let mut data = Vec::new();
                Self::flatten_json_array(arr, &mut data)?;

                let shape = if let Some(target_shape) = target_shape {
                    target_shape.clone()
                } else {
                    vec![data.len()]
                };

                Tensor::from_vec(data, &shape)
            }
            serde_json::Value::String(s) => {
                // Try to parse string as number
                let val = s.parse::<f64>().map_err(|_| {
                    TensorError::invalid_argument(format!("Cannot parse string '{s}' as number"))
                })?;
                let tensor_val = T::from(val as f32).unwrap();
                Ok(Tensor::from_scalar(tensor_val))
            }
            _ => Err(TensorError::invalid_argument(
                "Unsupported JSON value type for tensor conversion".to_string(),
            )),
        }
    }

    /// Recursively flatten JSON array to Vec<T>
    fn flatten_json_array(arr: &[serde_json::Value], data: &mut Vec<T>) -> Result<()> {
        for value in arr {
            match value {
                serde_json::Value::Number(n) => {
                    let val = n.as_f64().ok_or_else(|| {
                        TensorError::invalid_argument("Cannot convert number to f64".to_string())
                    })?;
                    data.push(T::from(val as f32).unwrap());
                }
                serde_json::Value::Array(nested_arr) => {
                    Self::flatten_json_array(nested_arr, data)?;
                }
                serde_json::Value::String(s) => {
                    let val = s.parse::<f64>().map_err(|_| {
                        TensorError::invalid_argument(format!(
                            "Cannot parse string '{s}' as number"
                        ))
                    })?;
                    data.push(T::from(val as f32).unwrap());
                }
                _ => {
                    return Err(TensorError::invalid_argument(
                        "Unsupported JSON value type in array".to_string(),
                    ))
                }
            }
        }
        Ok(())
    }

    /// Get the configuration used for this dataset
    pub fn config(&self) -> &JsonConfig {
        &self.config
    }

    /// Get statistics about the loaded dataset
    pub fn info(&self) -> JsonDatasetInfo {
        let sample_count = self.samples.len();
        let feature_shape = if !self.samples.is_empty() {
            Some(self.samples[0].0.shape().dims().to_vec())
        } else {
            None
        };
        let label_shape = if !self.samples.is_empty() {
            Some(self.samples[0].1.shape().dims().to_vec())
        } else {
            None
        };

        JsonDatasetInfo {
            sample_count,
            feature_shape,
            label_shape,
            file_path: self.config.file_path.clone(),
        }
    }
}

impl<T> Dataset<T> for JsonDataset<T>
where
    T: Clone + Default + num_traits::Zero + Send + Sync + 'static,
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

/// JSON Lines (JSONL) dataset for streaming large datasets
#[derive(Debug)]
pub struct JsonLDataset<T> {
    file_path: PathBuf,
    config: JsonConfig,
    #[allow(clippy::type_complexity)]
    cached_samples: RefCell<Option<Vec<(Tensor<T>, Tensor<T>)>>>,
    total_lines: Option<usize>,
}

impl<T> JsonLDataset<T>
where
    T: Clone + Default + num_traits::Zero + num_traits::Float + Send + Sync + 'static,
    T: serde::de::DeserializeOwned + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    /// Create a new JSONL dataset from configuration
    pub fn from_config(config: JsonConfig) -> Result<Self> {
        let file_path = config.file_path.clone();

        // Count total lines for length estimation
        let total_lines = Self::count_lines(&file_path)?;

        Ok(Self {
            file_path,
            config,
            cached_samples: RefCell::new(None),
            total_lines: Some(total_lines),
        })
    }

    /// Create a new JSONL dataset from file path
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        feature_field: &str,
        label_field: &str,
    ) -> Result<Self> {
        let config = JsonConfig {
            file_path: path.as_ref().to_path_buf(),
            feature_field: feature_field.to_string(),
            label_field: label_field.to_string(),
            ..Default::default()
        };
        Self::from_config(config)
    }

    /// Count lines in JSONL file
    fn count_lines(path: &Path) -> Result<usize> {
        let file = File::open(path)
            .map_err(|e| TensorError::invalid_argument(format!("Cannot open JSONL file: {e}")))?;

        let reader = BufReader::new(file);
        let count = reader.lines().count();
        Ok(count)
    }

    /// Load all samples into memory (lazy loading)
    fn ensure_loaded(&self) -> Result<()> {
        if self.cached_samples.borrow().is_some() {
            return Ok(());
        }

        let file = File::open(&self.file_path)
            .map_err(|e| TensorError::invalid_argument(format!("Cannot open JSONL file: {e}")))?;

        let reader = BufReader::new(file);
        let mut samples = Vec::new();

        let max_samples = self.config.max_samples.unwrap_or(usize::MAX);

        for (idx, line) in reader.lines().enumerate() {
            if idx >= max_samples {
                break;
            }

            let line =
                line.map_err(|e| TensorError::invalid_argument(format!("Cannot read line: {e}")))?;

            if line.trim().is_empty() {
                continue;
            }

            let json_obj: serde_json::Value = serde_json::from_str(&line).map_err(|e| {
                TensorError::invalid_argument(format!("Cannot parse JSON line {}: {}", idx + 1, e))
            })?;

            if let serde_json::Value::Object(map) = json_obj {
                let features = JsonDataset::extract_features(
                    &map,
                    &self.config.feature_field,
                    &self.config.feature_shape,
                )?;
                let labels = JsonDataset::extract_labels(
                    &map,
                    &self.config.label_field,
                    &self.config.label_shape,
                )?;

                samples.push((features, labels));
            } else {
                return Err(TensorError::invalid_argument(format!(
                    "Expected JSON object on line {}",
                    idx + 1
                )));
            }
        }

        if samples.is_empty() {
            return Err(TensorError::invalid_argument(
                "No valid samples found in JSONL file".to_string(),
            ));
        }

        *self.cached_samples.borrow_mut() = Some(samples);
        Ok(())
    }

    /// Get dataset information
    pub fn info(&self) -> Result<JsonDatasetInfo> {
        self.ensure_loaded()?;

        let samples_ref = self.cached_samples.borrow();
        let samples = samples_ref.as_ref().unwrap();
        let sample_count = samples.len();
        let feature_shape = if !samples.is_empty() {
            Some(samples[0].0.shape().dims().to_vec())
        } else {
            None
        };
        let label_shape = if !samples.is_empty() {
            Some(samples[0].1.shape().dims().to_vec())
        } else {
            None
        };

        Ok(JsonDatasetInfo {
            sample_count,
            feature_shape,
            label_shape,
            file_path: self.file_path.clone(),
        })
    }
}

impl<T> Dataset<T> for JsonLDataset<T>
where
    T: Clone + Default + num_traits::Zero + num_traits::Float + Send + Sync + 'static,
    T: serde::de::DeserializeOwned + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    fn len(&self) -> usize {
        self.total_lines.unwrap_or(0)
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        self.ensure_loaded()?;

        let samples_ref = self.cached_samples.borrow();
        let samples = samples_ref.as_ref().unwrap();
        if index >= samples.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of length {}",
                index,
                samples.len()
            )));
        }
        Ok(samples[index].clone())
    }
}

/// Information about a JSON dataset
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct JsonDatasetInfo {
    pub sample_count: usize,
    pub feature_shape: Option<Vec<usize>>,
    pub label_shape: Option<Vec<usize>>,
    pub file_path: PathBuf,
}

/// Builder for JSON datasets
#[derive(Debug, Default)]
pub struct JsonDatasetBuilder {
    config: JsonConfig,
}

impl JsonDatasetBuilder {
    /// Create a new JSON dataset builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the file path
    pub fn file_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.config.file_path = path.as_ref().to_path_buf();
        self
    }

    /// Set the feature field name
    pub fn feature_field<S: Into<String>>(mut self, field: S) -> Self {
        self.config.feature_field = field.into();
        self
    }

    /// Set the label field name
    pub fn label_field<S: Into<String>>(mut self, field: S) -> Self {
        self.config.label_field = field.into();
        self
    }

    /// Set maximum number of samples to load
    pub fn max_samples(mut self, max: usize) -> Self {
        self.config.max_samples = Some(max);
        self
    }

    /// Enable feature normalization
    pub fn normalize_features(mut self, normalize: bool) -> Self {
        self.config.normalize_features = normalize;
        self
    }

    /// Set target feature shape
    pub fn feature_shape(mut self, shape: Vec<usize>) -> Self {
        self.config.feature_shape = Some(shape);
        self
    }

    /// Set target label shape
    pub fn label_shape(mut self, shape: Vec<usize>) -> Self {
        self.config.label_shape = Some(shape);
        self
    }

    /// Build a JSON dataset
    pub fn build_json<T>(self) -> Result<JsonDataset<T>>
    where
        T: Clone + Default + num_traits::Zero + num_traits::Float + Send + Sync + 'static,
        T: serde::de::DeserializeOwned + std::str::FromStr,
        <T as std::str::FromStr>::Err: std::fmt::Debug,
    {
        JsonDataset::from_config(self.config)
    }

    /// Build a JSON Lines dataset
    pub fn build_jsonl<T>(self) -> Result<JsonLDataset<T>>
    where
        T: Clone + Default + num_traits::Zero + num_traits::Float + Send + Sync + 'static,
        T: serde::de::DeserializeOwned + std::str::FromStr,
        <T as std::str::FromStr>::Err: std::fmt::Debug,
    {
        JsonLDataset::from_config(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_json_dataset_builder() {
        let builder = JsonDatasetBuilder::new()
            .feature_field("input")
            .label_field("output")
            .max_samples(100)
            .normalize_features(true);

        // Test that builder configuration is set correctly
        assert_eq!(builder.config.feature_field, "input");
        assert_eq!(builder.config.label_field, "output");
        assert_eq!(builder.config.max_samples, Some(100));
        assert!(builder.config.normalize_features);
    }

    #[test]
    fn test_json_config_default() {
        let config = JsonConfig::default();
        assert_eq!(config.feature_field, "features");
        assert_eq!(config.label_field, "label");
        assert!(config.auto_parse_arrays);
        assert_eq!(config.max_samples, None);
        assert!(!config.normalize_features);
    }

    #[test]
    fn test_json_dataset_from_file() {
        // Create a temporary JSON file
        let mut temp_file = NamedTempFile::new().unwrap();
        let json_content = r#"[
            {"features": [1.0, 2.0], "label": 0},
            {"features": [3.0, 4.0], "label": 1}
        ]"#;
        temp_file.write_all(json_content.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let dataset = JsonDataset::<f32>::from_file(temp_file.path(), "features", "label").unwrap();

        assert_eq!(dataset.len(), 2);

        let (features, label) = dataset.get(0).unwrap();
        assert_eq!(features.shape().dims(), &[2]);
        assert_eq!(label.shape().dims(), &[] as &[usize]); // scalar
    }

    #[test]
    fn test_jsonl_dataset_from_file() {
        // Create a temporary JSONL file
        let mut temp_file = NamedTempFile::new().unwrap();
        let jsonl_content = r#"{"features": [1.0, 2.0], "label": 0}
{"features": [3.0, 4.0], "label": 1}
{"features": [5.0, 6.0], "label": 0}"#;
        temp_file.write_all(jsonl_content.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let dataset =
            JsonLDataset::<f32>::from_file(temp_file.path(), "features", "label").unwrap();

        assert_eq!(dataset.len(), 3);

        let (features, label) = dataset.get(0).unwrap();
        assert_eq!(features.shape().dims(), &[2]);
        assert_eq!(label.shape().dims(), &[] as &[usize]); // scalar
    }

    #[test]
    fn test_json_dataset_info() {
        // Create a temporary JSON file
        let mut temp_file = NamedTempFile::new().unwrap();
        let json_content = r#"[
            {"features": [1.0, 2.0, 3.0], "label": 0},
            {"features": [4.0, 5.0, 6.0], "label": 1}
        ]"#;
        temp_file.write_all(json_content.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let dataset = JsonDataset::<f32>::from_file(temp_file.path(), "features", "label").unwrap();

        let info = dataset.info();
        assert_eq!(info.sample_count, 2);
        assert_eq!(info.feature_shape, Some(vec![3]));
        assert_eq!(info.label_shape, Some(vec![]));
    }

    #[test]
    fn test_json_dataset_nested_arrays() {
        // Create a temporary JSON file with nested arrays
        let mut temp_file = NamedTempFile::new().unwrap();
        let json_content = r#"[
            {"features": [[1.0, 2.0], [3.0, 4.0]], "label": 0}
        ]"#;
        temp_file.write_all(json_content.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let dataset = JsonDataset::<f32>::from_file(temp_file.path(), "features", "label").unwrap();

        assert_eq!(dataset.len(), 1);

        let (features, _) = dataset.get(0).unwrap();
        assert_eq!(features.shape().dims(), &[4]); // Flattened
    }

    #[test]
    fn test_invalid_json_file() {
        // Create a temporary file with invalid JSON
        let mut temp_file = NamedTempFile::new().unwrap();
        let invalid_json = r#"{"invalid": json}"#;
        temp_file.write_all(invalid_json.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let result = JsonDataset::<f32>::from_file(temp_file.path(), "features", "label");

        assert!(result.is_err());
    }

    #[test]
    fn test_missing_feature_field() {
        // Create a temporary JSON file without expected feature field
        let mut temp_file = NamedTempFile::new().unwrap();
        let json_content = r#"[
            {"other_field": [1.0, 2.0], "label": 0}
        ]"#;
        temp_file.write_all(json_content.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let result = JsonDataset::<f32>::from_file(temp_file.path(), "features", "label");

        assert!(result.is_err());
    }
}
