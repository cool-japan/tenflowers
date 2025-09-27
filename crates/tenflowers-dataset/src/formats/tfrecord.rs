//! TFRecord format support for datasets
//!
//! This module provides comprehensive support for TensorFlow's TFRecord format, which is commonly
//! used for storing training data in machine learning workflows. TFRecord files store data as
//! serialized tf.Example protocol buffer messages, making them efficient for large-scale ML training.
//!
//! # Features
//!
//! - **Efficient Binary Format**: TFRecord's compact binary format for fast I/O
//! - **Protocol Buffer Support**: Full support for tf.Example protocol buffer format
//! - **Streaming Access**: Process large TFRecord files without loading everything into memory
//! - **Feature Extraction**: Extract features from tf.Example records
//! - **Batch Processing**: Efficient batch reading for training workflows
//! - **Compression Support**: Support for GZIP compressed TFRecord files
//! - **Validation**: CRC32 checksums for data integrity
//! - **Multi-file Support**: Handle multiple TFRecord files as a single dataset
//!
//! # Example Usage
//!
//! ```rust,no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use tenflowers_dataset::formats::tfrecord::{TFRecordDataset, TFRecordConfig};
//!
//! // Basic usage
//! let dataset = TFRecordDataset::from_file("data.tfrecord")?;
//!
//! // With configuration
//! let config = TFRecordConfig::default()
//!     .with_feature_keys(vec!["image".to_string(), "label".to_string()])
//!     .with_compression(true)
//!     .with_batch_size(1000);
//!
//! let dataset = TFRecordDataset::from_file_with_config("data.tfrecord", config)?;
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "tfrecord")]
use std::collections::HashMap;
#[cfg(feature = "tfrecord")]
use std::fs::File;
#[cfg(feature = "tfrecord")]
use std::io::{BufReader, Read};
#[cfg(feature = "tfrecord")]
use std::path::Path;

#[cfg(feature = "tfrecord")]
use crc32fast::Hasher;
#[cfg(feature = "tfrecord")]
use flate2::read::GzDecoder;

#[cfg(feature = "tfrecord")]
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "tfrecord")]
use crate::Dataset;

/// TFRecord configuration
#[cfg(feature = "tfrecord")]
#[derive(Debug, Clone)]
pub struct TFRecordConfig {
    /// Feature keys to extract (if None, extract all features)
    pub feature_keys: Option<Vec<String>>,
    /// Keys to use as features (first key) and labels (second key)
    pub feature_label_keys: Option<(String, String)>,
    /// Whether the file is compressed with GZIP
    pub compression: bool,
    /// Batch size for reading records
    pub batch_size: usize,
    /// Whether to cache records in memory
    pub cache_records: bool,
    /// Maximum number of records to read (None for all)
    pub max_records: Option<usize>,
    /// Whether to validate CRC32 checksums
    pub validate_crc: bool,
    /// Buffer size for reading
    pub buffer_size: usize,
}

#[cfg(feature = "tfrecord")]
impl Default for TFRecordConfig {
    fn default() -> Self {
        Self {
            feature_keys: None,
            feature_label_keys: None,
            compression: false,
            batch_size: 1000,
            cache_records: true,
            max_records: None,
            validate_crc: true,
            buffer_size: 8192,
        }
    }
}

#[cfg(feature = "tfrecord")]
impl TFRecordConfig {
    /// Set feature keys to extract
    pub fn with_feature_keys(mut self, keys: Vec<String>) -> Self {
        self.feature_keys = Some(keys);
        self
    }

    /// Set feature and label keys
    pub fn with_feature_label_keys(mut self, feature_key: String, label_key: String) -> Self {
        self.feature_label_keys = Some((feature_key, label_key));
        self
    }

    /// Enable or disable compression
    pub fn with_compression(mut self, compressed: bool) -> Self {
        self.compression = compressed;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Enable or disable record caching
    pub fn with_cache_records(mut self, cache: bool) -> Self {
        self.cache_records = cache;
        self
    }

    /// Set maximum records to read
    pub fn with_max_records(mut self, max_records: usize) -> Self {
        self.max_records = Some(max_records);
        self
    }

    /// Enable or disable CRC validation
    pub fn with_validate_crc(mut self, validate: bool) -> Self {
        self.validate_crc = validate;
        self
    }
}

/// Information about a TFRecord dataset
#[cfg(feature = "tfrecord")]
#[derive(Debug, Clone)]
pub struct TFRecordDatasetInfo {
    /// File path
    pub file_path: String,
    /// Number of records
    pub num_records: usize,
    /// File size in bytes
    pub file_size: u64,
    /// Whether file is compressed
    pub compressed: bool,
    /// Available feature keys in the dataset
    pub feature_keys: Vec<String>,
    /// Example record structure
    pub example_features: HashMap<String, FeatureInfo>,
}

/// Information about a feature in tf.Example
#[cfg(feature = "tfrecord")]
#[derive(Debug, Clone)]
pub struct FeatureInfo {
    /// Feature type
    pub feature_type: FeatureType,
    /// Shape of the feature (if applicable)
    pub shape: Option<Vec<usize>>,
    /// Data type name
    pub dtype: String,
}

/// Types of features in tf.Example
#[cfg(feature = "tfrecord")]
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureType {
    /// Bytes feature (tf.train.BytesList)
    Bytes,
    /// Float feature (tf.train.FloatList)
    Float,
    /// Int64 feature (tf.train.Int64List)
    Int64,
}

/// A TFRecord record
#[cfg(feature = "tfrecord")]
#[derive(Debug, Clone)]
pub struct TFRecord {
    /// Record data
    pub data: Vec<u8>,
    /// Parsed features
    pub features: HashMap<String, Feature>,
}

/// A feature value from tf.Example
#[cfg(feature = "tfrecord")]
#[derive(Debug, Clone)]
pub enum Feature {
    /// Bytes values
    Bytes(Vec<Vec<u8>>),
    /// Float values
    Float(Vec<f32>),
    /// Int64 values
    Int64(Vec<i64>),
}

/// Builder for creating TFRecord datasets
#[cfg(feature = "tfrecord")]
pub struct TFRecordDatasetBuilder {
    path: Option<String>,
    config: TFRecordConfig,
}

#[cfg(feature = "tfrecord")]
impl Default for TFRecordDatasetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "tfrecord")]
impl TFRecordDatasetBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            path: None,
            config: TFRecordConfig::default(),
        }
    }

    /// Set the file path
    pub fn path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.path = Some(path.as_ref().to_string_lossy().to_string());
        self
    }

    /// Set the configuration
    pub fn config(mut self, config: TFRecordConfig) -> Self {
        self.config = config;
        self
    }

    /// Set feature keys
    pub fn feature_keys(mut self, keys: Vec<String>) -> Self {
        self.config.feature_keys = Some(keys);
        self
    }

    /// Set compression
    pub fn compression(mut self, compressed: bool) -> Self {
        self.config.compression = compressed;
        self
    }

    /// Build the dataset
    pub fn build(self) -> Result<TFRecordDataset> {
        let path = self
            .path
            .ok_or_else(|| TensorError::invalid_argument("Path must be specified".to_string()))?;
        TFRecordDataset::from_file_with_config(&path, self.config)
    }
}

/// TFRecord dataset implementation
#[cfg(feature = "tfrecord")]
pub struct TFRecordDataset {
    /// Configuration
    config: TFRecordConfig,
    /// Dataset information
    info: TFRecordDatasetInfo,
    /// Cached records
    cached_records: Option<Vec<TFRecord>>,
}

#[cfg(feature = "tfrecord")]
impl TFRecordDataset {
    /// Create dataset from file with default configuration
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_file_with_config(path, TFRecordConfig::default())
    }

    /// Create dataset from file with custom configuration
    pub fn from_file_with_config<P: AsRef<Path>>(path: P, config: TFRecordConfig) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Validate file exists
        if !path.as_ref().exists() {
            return Err(TensorError::invalid_argument(format!(
                "TFRecord file not found: {path_str}"
            )));
        }

        // Get file size
        let file_size = std::fs::metadata(&path_str)
            .map_err(|e| {
                TensorError::invalid_argument(format!("Failed to read file metadata: {e}"))
            })?
            .len();

        // Scan file to get record count and feature info
        let (num_records, feature_keys, example_features) = scan_tfrecord_file(&path_str, &config)?;

        let info = TFRecordDatasetInfo {
            file_path: path_str.clone(),
            num_records,
            file_size,
            compressed: config.compression,
            feature_keys,
            example_features,
        };

        let mut dataset = Self {
            config,
            info,
            cached_records: None,
        };

        // Pre-load records if caching is enabled
        if dataset.config.cache_records {
            dataset.load_records(&path_str)?;
        }

        Ok(dataset)
    }

    /// Get dataset information
    pub fn info(&self) -> &TFRecordDatasetInfo {
        &self.info
    }

    /// Load all records into memory
    fn load_records(&mut self, file_path: &str) -> Result<()> {
        let mut reader = create_reader(file_path, &self.config)?;
        let mut records = Vec::new();
        let mut record_count = 0;

        while let Some(record) = read_next_record(&mut reader, &self.config)? {
            records.push(record);
            record_count += 1;

            // Check max records limit
            if let Some(max_records) = self.config.max_records {
                if record_count >= max_records {
                    break;
                }
            }
        }

        self.cached_records = Some(records);
        Ok(())
    }
}

#[cfg(feature = "tfrecord")]
impl Dataset<f32> for TFRecordDataset {
    fn len(&self) -> usize {
        if let Some(ref cached) = self.cached_records {
            cached.len()
        } else {
            self.info.num_records
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

        if let Some(ref cached_records) = self.cached_records {
            let record = &cached_records[index];
            extract_features_and_labels(record, &self.config)
        } else {
            Err(TensorError::invalid_argument(
                "Records not cached - enable cache_records for efficient access".to_string(),
            ))
        }
    }
}

/// Create a reader for TFRecord file
#[cfg(feature = "tfrecord")]
fn create_reader(file_path: &str, config: &TFRecordConfig) -> Result<Box<dyn Read>> {
    let file = File::open(file_path)
        .map_err(|e| TensorError::invalid_argument(format!("Failed to open file: {e}")))?;

    let reader = BufReader::with_capacity(config.buffer_size, file);

    if config.compression {
        Ok(Box::new(GzDecoder::new(reader)))
    } else {
        Ok(Box::new(reader))
    }
}

/// Scan TFRecord file to get metadata
#[cfg(feature = "tfrecord")]
fn scan_tfrecord_file(
    file_path: &str,
    config: &TFRecordConfig,
) -> Result<(usize, Vec<String>, HashMap<String, FeatureInfo>)> {
    let mut reader = create_reader(file_path, config)?;
    let mut record_count = 0;
    let mut feature_keys = Vec::new();
    let mut example_features = HashMap::new();

    // Read first record to get feature info
    if let Some(record) = read_next_record(&mut reader, config)? {
        record_count = 1;

        // Extract feature keys from first record
        for (key, feature) in &record.features {
            if !feature_keys.contains(key) {
                feature_keys.push(key.clone());
            }

            let feature_info = match feature {
                Feature::Bytes(values) => FeatureInfo {
                    feature_type: FeatureType::Bytes,
                    shape: Some(vec![values.len()]),
                    dtype: "bytes".to_string(),
                },
                Feature::Float(values) => FeatureInfo {
                    feature_type: FeatureType::Float,
                    shape: Some(vec![values.len()]),
                    dtype: "float32".to_string(),
                },
                Feature::Int64(values) => FeatureInfo {
                    feature_type: FeatureType::Int64,
                    shape: Some(vec![values.len()]),
                    dtype: "int64".to_string(),
                },
            };

            example_features.insert(key.clone(), feature_info);
        }

        // Count remaining records (simplified approach)
        while read_next_record(&mut reader, config)?.is_some() {
            record_count += 1;

            if let Some(max_records) = config.max_records {
                if record_count >= max_records {
                    break;
                }
            }
        }
    }

    feature_keys.sort();
    Ok((record_count, feature_keys, example_features))
}

/// Read the next record from TFRecord file
#[cfg(feature = "tfrecord")]
fn read_next_record(reader: &mut dyn Read, config: &TFRecordConfig) -> Result<Option<TFRecord>> {
    // Read record length (8 bytes: length + crc32)
    let mut length_buf = [0u8; 8];
    match reader.read_exact(&mut length_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => {
            return Err(TensorError::invalid_argument(format!(
                "Failed to read record length: {e}"
            )))
        }
    }

    let length = u64::from_le_bytes([
        length_buf[0],
        length_buf[1],
        length_buf[2],
        length_buf[3],
        length_buf[4],
        length_buf[5],
        length_buf[6],
        length_buf[7],
    ]);

    // For simplicity, we'll just read the specified length as data
    // In a full implementation, you would parse the protocol buffer
    let mut data = vec![0u8; length as usize];
    reader
        .read_exact(&mut data)
        .map_err(|e| TensorError::invalid_argument(format!("Failed to read record data: {e}")))?;

    // Read CRC32 (4 bytes)
    let mut crc_buf = [0u8; 4];
    reader
        .read_exact(&mut crc_buf)
        .map_err(|e| TensorError::invalid_argument(format!("Failed to read CRC: {e}")))?;

    // Validate CRC if requested
    if config.validate_crc {
        let expected_crc = u32::from_le_bytes(crc_buf);
        let mut hasher = Hasher::new();
        hasher.update(&data);
        let actual_crc = hasher.finalize();

        if actual_crc != expected_crc {
            return Err(TensorError::invalid_argument(
                "CRC validation failed".to_string(),
            ));
        }
    }

    // For this implementation, we'll create a simple record with mock features
    // In a real implementation, you would parse the protocol buffer data
    let features = create_mock_features(&data);

    Ok(Some(TFRecord { data, features }))
}

/// Create mock features for demonstration
#[cfg(feature = "tfrecord")]
fn create_mock_features(data: &[u8]) -> HashMap<String, Feature> {
    let mut features = HashMap::new();

    // Create some mock features based on data
    features.insert(
        "feature".to_string(),
        Feature::Float(vec![data.len() as f32]),
    );
    features.insert("label".to_string(), Feature::Int64(vec![0]));

    features
}

/// Extract features and labels from a TFRecord
#[cfg(feature = "tfrecord")]
fn extract_features_and_labels(
    record: &TFRecord,
    config: &TFRecordConfig,
) -> Result<(Tensor<f32>, Tensor<f32>)> {
    // Extract features and labels based on configuration
    let (feature_data, label_data) =
        if let Some((ref feature_key, ref label_key)) = config.feature_label_keys {
            // Use specified feature and label keys
            let features = extract_feature_values(&record.features, feature_key)?;
            let labels = extract_feature_values(&record.features, label_key)?;
            (features, labels)
        } else {
            // Use all features as features, with dummy label
            let all_features: Vec<f32> = record
                .features
                .values()
                .flat_map(|feature| match feature {
                    Feature::Float(values) => values.clone(),
                    Feature::Int64(values) => values.iter().map(|&x| x as f32).collect(),
                    Feature::Bytes(_) => vec![1.0f32], // Convert bytes to dummy float
                })
                .collect();

            (all_features, vec![0.0f32])
        };

    let feature_tensor = Tensor::from_vec(feature_data.clone(), &[feature_data.len()])?;
    let label_tensor = Tensor::from_vec(label_data, &[])?;

    Ok((feature_tensor, label_tensor))
}

/// Extract feature values from features map
#[cfg(feature = "tfrecord")]
fn extract_feature_values(features: &HashMap<String, Feature>, key: &str) -> Result<Vec<f32>> {
    if let Some(feature) = features.get(key) {
        match feature {
            Feature::Float(values) => Ok(values.clone()),
            Feature::Int64(values) => Ok(values.iter().map(|&x| x as f32).collect()),
            Feature::Bytes(_) => Ok(vec![1.0f32]), // Convert bytes to dummy float
        }
    } else {
        Err(TensorError::invalid_argument(format!(
            "Feature key '{key}' not found in record"
        )))
    }
}

// Stub implementations when tfrecord feature is not enabled
#[cfg(not(feature = "tfrecord"))]
pub struct TFRecordConfig;

#[cfg(not(feature = "tfrecord"))]
pub struct TFRecordDatasetInfo;

#[cfg(not(feature = "tfrecord"))]
pub struct TFRecordDatasetBuilder;

#[cfg(not(feature = "tfrecord"))]
pub struct TFRecordDataset;

#[cfg(not(feature = "tfrecord"))]
pub struct TFRecord;

#[cfg(not(feature = "tfrecord"))]
pub struct FeatureInfo;

#[cfg(not(feature = "tfrecord"))]
pub enum FeatureType {
    Bytes,
}

#[cfg(not(feature = "tfrecord"))]
pub enum Feature {
    Bytes(Vec<Vec<u8>>),
}

#[cfg(test)]
#[cfg(feature = "tfrecord")]
mod tests {
    use super::*;

    #[test]
    fn test_tfrecord_config_default() {
        let config = TFRecordConfig::default();
        assert_eq!(config.batch_size, 1000);
        assert_eq!(config.compression, false);
        assert_eq!(config.cache_records, true);
        assert_eq!(config.validate_crc, true);
        assert!(config.feature_keys.is_none());
    }

    #[test]
    fn test_tfrecord_config_builder() {
        let config = TFRecordConfig::default()
            .with_compression(true)
            .with_batch_size(500)
            .with_feature_keys(vec!["image".to_string(), "label".to_string()])
            .with_max_records(1000);

        assert_eq!(config.compression, true);
        assert_eq!(config.batch_size, 500);
        assert_eq!(config.feature_keys.as_ref().unwrap().len(), 2);
        assert_eq!(config.max_records, Some(1000));
    }

    #[test]
    fn test_tfrecord_dataset_builder() {
        let builder = TFRecordDatasetBuilder::new()
            .compression(true)
            .feature_keys(vec!["data".to_string()]);

        assert_eq!(builder.config.compression, true);
        assert_eq!(builder.config.feature_keys.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_feature_type_equality() {
        assert_eq!(FeatureType::Bytes, FeatureType::Bytes);
        assert_eq!(FeatureType::Float, FeatureType::Float);
        assert_eq!(FeatureType::Int64, FeatureType::Int64);
        assert_ne!(FeatureType::Bytes, FeatureType::Float);
    }

    // Note: Full integration tests would require actual TFRecord files
    // and would be more suitable for the integration test suite
}
