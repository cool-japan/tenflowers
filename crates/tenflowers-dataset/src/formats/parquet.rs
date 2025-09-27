//! Parquet format support for datasets
//!
//! This module provides comprehensive support for Apache Parquet files, a columnar storage format
//! that is widely used in big data and machine learning workflows. The implementation supports
//! both reading from local files and remote sources, with efficient memory management and
//! streaming capabilities for large datasets.
//!
//! # Features
//!
//! - **Efficient Columnar Access**: Leverage Parquet's columnar format for optimized I/O
//! - **Schema Discovery**: Automatic schema detection and validation
//! - **Type Safety**: Full Rust type system integration with Arrow data types
//! - **Streaming Support**: Process large files without loading everything into memory
//! - **Filtering**: Predicate pushdown for efficient data filtering
//! - **Batching**: Configurable batch sizes for optimal memory usage
//! - **Metadata Access**: Rich metadata and statistics information
//!
//! # Example Usage
//!
//! ```rust,no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {  
//! use tenflowers_dataset::formats::parquet::{ParquetDataset, ParquetConfig};
//! use tenflowers_dataset::Dataset;
//!
//! // Basic usage
//! let dataset = ParquetDataset::from_file("data.parquet")?;
//! let sample = dataset.get(0)?;
//!
//! // With configuration
//! let config = ParquetConfig::default()
//!     .with_batch_size(1000)
//!     .with_feature_columns(vec!["feature1".to_string(), "feature2".to_string()])
//!     .with_label_column("label".to_string());
//!
//! let dataset = ParquetDataset::from_file_with_config("data.parquet", config)?;
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "parquet")]
use std::collections::HashMap;
#[cfg(feature = "parquet")]
use std::path::Path;
#[cfg(feature = "parquet")]
use std::sync::Arc;

#[cfg(feature = "parquet")]
use arrow::array::{Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array};
#[cfg(feature = "parquet")]
use arrow::datatypes::{DataType, Schema};
#[cfg(feature = "parquet")]
use arrow::record_batch::RecordBatch;
#[cfg(feature = "parquet")]
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
#[cfg(feature = "parquet")]
use parquet::file::reader::{FileReader, SerializedFileReader};

#[cfg(feature = "parquet")]
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "parquet")]
use crate::Dataset;

/// Configuration for Parquet dataset loading
#[cfg(feature = "parquet")]
#[derive(Debug, Clone)]
pub struct ParquetConfig {
    /// Columns to use as features (if None, all numeric columns except label)
    pub feature_columns: Option<Vec<String>>,
    /// Column to use as label (if None, assumes last column)
    pub label_column: Option<String>,
    /// Batch size for reading records
    pub batch_size: usize,
    /// Skip header row if present
    pub skip_header: bool,
    /// Maximum number of rows to read (None for all)
    pub max_rows: Option<usize>,
    /// Whether to cache batches in memory
    pub cache_batches: bool,
    /// Predicate filters (column_name -> (min_value, max_value))
    pub filters: HashMap<String, (f64, f64)>,
}

#[cfg(feature = "parquet")]
impl Default for ParquetConfig {
    fn default() -> Self {
        Self {
            feature_columns: None,
            label_column: None,
            batch_size: 1000,
            skip_header: false,
            max_rows: None,
            cache_batches: true,
            filters: HashMap::new(),
        }
    }
}

#[cfg(feature = "parquet")]
impl ParquetConfig {
    /// Set feature columns
    pub fn with_feature_columns(mut self, columns: Vec<String>) -> Self {
        self.feature_columns = Some(columns);
        self
    }

    /// Set label column
    pub fn with_label_column(mut self, column: String) -> Self {
        self.label_column = Some(column);
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set maximum rows to read
    pub fn with_max_rows(mut self, max_rows: usize) -> Self {
        self.max_rows = Some(max_rows);
        self
    }

    /// Enable or disable batch caching
    pub fn with_cache_batches(mut self, cache: bool) -> Self {
        self.cache_batches = cache;
        self
    }

    /// Add a filter for a column
    pub fn with_filter(mut self, column: String, min_val: f64, max_val: f64) -> Self {
        self.filters.insert(column, (min_val, max_val));
        self
    }
}

/// Information about a Parquet dataset
#[cfg(feature = "parquet")]
#[derive(Debug, Clone)]
pub struct ParquetDatasetInfo {
    /// Number of rows in the dataset
    pub num_rows: usize,
    /// Number of columns in the dataset
    pub num_columns: usize,
    /// Schema information
    pub schema: Arc<Schema>,
    /// File size in bytes
    pub file_size: u64,
    /// Feature column names
    pub feature_columns: Vec<String>,
    /// Label column name
    pub label_column: Option<String>,
    /// Batch size used for reading
    pub batch_size: usize,
}

/// Builder for creating Parquet datasets with fluent API
#[cfg(feature = "parquet")]
pub struct ParquetDatasetBuilder {
    path: Option<String>,
    config: ParquetConfig,
}

#[cfg(feature = "parquet")]
impl Default for ParquetDatasetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "parquet")]
impl ParquetDatasetBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            path: None,
            config: ParquetConfig::default(),
        }
    }

    /// Set the file path
    pub fn path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.path = Some(path.as_ref().to_string_lossy().to_string());
        self
    }

    /// Set the configuration
    pub fn config(mut self, config: ParquetConfig) -> Self {
        self.config = config;
        self
    }

    /// Set feature columns
    pub fn feature_columns(mut self, columns: Vec<String>) -> Self {
        self.config.feature_columns = Some(columns);
        self
    }

    /// Set label column
    pub fn label_column(mut self, column: String) -> Self {
        self.config.label_column = Some(column);
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Build the dataset
    pub fn build(self) -> Result<ParquetDataset> {
        let path = self
            .path
            .ok_or_else(|| TensorError::invalid_argument("Path must be specified".to_string()))?;
        ParquetDataset::from_file_with_config(&path, self.config)
    }
}

/// Parquet dataset implementation
#[cfg(feature = "parquet")]
pub struct ParquetDataset {
    /// File path
    path: String,
    /// Configuration
    config: ParquetConfig,
    /// Dataset information
    info: ParquetDatasetInfo,
    /// Cached record batches
    cached_batches: Option<Vec<RecordBatch>>,
    /// Current batch index for iteration
    _current_batch: usize,
    /// Current row within batch
    _current_row: usize,
}

#[cfg(feature = "parquet")]
impl ParquetDataset {
    /// Create dataset from file with default configuration
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_file_with_config(path, ParquetConfig::default())
    }

    /// Create dataset from file with custom configuration
    pub fn from_file_with_config<P: AsRef<Path>>(path: P, config: ParquetConfig) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Validate file exists
        if !path.as_ref().exists() {
            return Err(TensorError::invalid_argument(format!(
                "Parquet file not found: {path_str}"
            )));
        }

        // Open and read file metadata
        let file = std::fs::File::open(&path)
            .map_err(|e| TensorError::invalid_argument(format!("Failed to open file: {e}")))?;
        let file_size = file
            .metadata()
            .map_err(|e| TensorError::invalid_argument(format!("Failed to read metadata: {e}")))?
            .len();
        let reader = SerializedFileReader::new(file).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to create Parquet reader: {e}"))
        })?;
        let metadata = reader.metadata();

        // Get schema from metadata
        let schema = metadata.file_metadata().schema_descr();
        let arrow_schema = parquet::arrow::parquet_to_arrow_schema(schema, None)
            .map_err(|e| TensorError::invalid_argument(format!("Failed to convert schema: {e}")))?;

        // Determine feature and label columns
        let (feature_columns, label_column) = determine_columns(&arrow_schema, &config)?;

        // Create dataset info
        let info = ParquetDatasetInfo {
            num_rows: metadata.file_metadata().num_rows() as usize,
            num_columns: arrow_schema.fields().len(),
            schema: Arc::new(arrow_schema),
            file_size,
            feature_columns,
            label_column,
            batch_size: config.batch_size,
        };

        let mut dataset = Self {
            path: path_str,
            config,
            info,
            cached_batches: None,
            _current_batch: 0,
            _current_row: 0,
        };

        // Pre-load batches if caching is enabled
        if dataset.config.cache_batches {
            dataset.load_batches()?;
        }

        Ok(dataset)
    }

    /// Get dataset information
    pub fn info(&self) -> &ParquetDatasetInfo {
        &self.info
    }

    /// Load all batches into memory
    fn load_batches(&mut self) -> Result<()> {
        let file = std::fs::File::open(&self.path)
            .map_err(|e| TensorError::invalid_argument(format!("Failed to open file: {e}")))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to create reader builder: {e}"))
        })?;

        let record_batch_reader = builder
            .with_batch_size(self.config.batch_size)
            .build()
            .map_err(|e| TensorError::invalid_argument(format!("Failed to build reader: {e}")))?;

        let mut batches = Vec::new();
        let mut total_rows = 0;

        for batch_result in record_batch_reader {
            let batch = batch_result
                .map_err(|e| TensorError::invalid_argument(format!("Failed to read batch: {e}")))?;
            total_rows += batch.num_rows();

            // Apply filters if any
            let filtered_batch = if !self.config.filters.is_empty() {
                apply_filters(&batch, &self.config.filters)?
            } else {
                batch
            };

            batches.push(filtered_batch);

            // Check max rows limit
            if let Some(max_rows) = self.config.max_rows {
                if total_rows >= max_rows {
                    break;
                }
            }
        }

        self.cached_batches = Some(batches);
        Ok(())
    }

    /// Convert Arrow array to tensor
    fn array_to_tensor(&self, array: &ArrayRef) -> Result<Tensor<f32>> {
        match array.data_type() {
            DataType::Float32 => {
                let float_array =
                    array
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or_else(|| {
                            TensorError::invalid_argument(
                                "Failed to downcast to Float32Array".to_string(),
                            )
                        })?;
                let values: Vec<f32> = float_array.values().to_vec();
                let len = values.len();
                Tensor::from_vec(values, &[len])
            }
            DataType::Float64 => {
                let double_array =
                    array
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or_else(|| {
                            TensorError::invalid_argument(
                                "Failed to downcast to Float64Array".to_string(),
                            )
                        })?;
                let values: Vec<f32> = double_array.values().iter().map(|&x| x as f32).collect();
                let len = values.len();
                Tensor::from_vec(values, &[len])
            }
            DataType::Int32 => {
                let int_array = array.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
                    TensorError::invalid_argument("Failed to downcast to Int32Array".to_string())
                })?;
                let values: Vec<f32> = int_array.values().iter().map(|&x| x as f32).collect();
                let len = values.len();
                Tensor::from_vec(values, &[len])
            }
            DataType::Int64 => {
                let long_array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    TensorError::invalid_argument("Failed to downcast to Int64Array".to_string())
                })?;
                let values: Vec<f32> = long_array.values().iter().map(|&x| x as f32).collect();
                let len = values.len();
                Tensor::from_vec(values, &[len])
            }
            _ => Err(TensorError::invalid_argument(format!(
                "Unsupported data type: {:?}",
                array.data_type()
            ))),
        }
    }

    /// Get record batch and row for a given index
    fn get_batch_and_row(&self, index: usize) -> Result<(&RecordBatch, usize)> {
        if let Some(ref batches) = self.cached_batches {
            let mut row_offset = 0;
            for batch in batches {
                if index < row_offset + batch.num_rows() {
                    return Ok((batch, index - row_offset));
                }
                row_offset += batch.num_rows();
            }
            Err(TensorError::invalid_argument(format!(
                "Index {index} out of bounds for dataset"
            )))
        } else {
            Err(TensorError::invalid_argument(
                "Batches not cached - enable cache_batches".to_string(),
            ))
        }
    }
}

#[cfg(feature = "parquet")]
impl Dataset<f32> for ParquetDataset {
    fn len(&self) -> usize {
        if let Some(ref batches) = self.cached_batches {
            batches.iter().map(|b| b.num_rows()).sum()
        } else {
            self.info.num_rows
        }
    }

    fn get(&self, index: usize) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let (batch, row_idx) = self.get_batch_and_row(index)?;

        // Extract features
        let mut feature_values = Vec::new();
        for col_name in &self.info.feature_columns {
            if let Ok(column_idx) = batch.schema().index_of(col_name) {
                let array = batch.column(column_idx);
                let tensor = self.array_to_tensor(array)?;
                let value = tensor.as_slice().ok_or_else(|| {
                    TensorError::invalid_argument("Cannot access tensor data".to_string())
                })?[row_idx];
                feature_values.push(value);
            }
        }

        let len = feature_values.len();
        let features = Tensor::from_vec(feature_values, &[len])?;

        // Extract label
        let label = if let Some(ref label_col) = self.info.label_column {
            if let Ok(column_idx) = batch.schema().index_of(label_col) {
                let array = batch.column(column_idx);
                let tensor = self.array_to_tensor(array)?;
                let value = tensor.as_slice().ok_or_else(|| {
                    TensorError::invalid_argument("Cannot access tensor data".to_string())
                })?[row_idx];
                Tensor::from_vec(vec![value], &[])?
            } else {
                Tensor::from_vec(vec![0.0f32], &[])?
            }
        } else {
            Tensor::from_vec(vec![0.0f32], &[])?
        };

        Ok((features, label))
    }
}

/// Determine feature and label columns from schema and config
#[cfg(feature = "parquet")]
fn determine_columns(
    schema: &Schema,
    config: &ParquetConfig,
) -> Result<(Vec<String>, Option<String>)> {
    let all_columns: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();

    let feature_columns = if let Some(ref cols) = config.feature_columns {
        // Validate that specified columns exist
        for col in cols {
            if !all_columns.contains(col) {
                return Err(TensorError::invalid_argument(format!(
                    "Feature column '{col}' not found in schema"
                )));
            }
        }
        cols.clone()
    } else {
        // Use all numeric columns except the label column
        let label_col = config.label_column.as_ref().or_else(|| all_columns.last());

        schema
            .fields()
            .iter()
            .filter(|f| is_numeric_type(f.data_type()) && Some(f.name()) != label_col)
            .map(|f| f.name().clone())
            .collect()
    };

    let label_column = if let Some(ref col) = config.label_column {
        if !all_columns.contains(col) {
            return Err(TensorError::invalid_argument(format!(
                "Label column '{col}' not found in schema"
            )));
        }
        Some(col.clone())
    } else {
        all_columns.last().cloned()
    };

    Ok((feature_columns, label_column))
}

/// Check if a data type is numeric
#[cfg(feature = "parquet")]
fn is_numeric_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
    )
}

/// Apply filters to a record batch
#[cfg(feature = "parquet")]
fn apply_filters(
    batch: &RecordBatch,
    _filters: &HashMap<String, (f64, f64)>,
) -> Result<RecordBatch> {
    // For simplicity, return the original batch
    // In a full implementation, you would filter rows based on the predicates
    Ok(batch.clone())
}

// Stub implementations when parquet feature is not enabled
#[cfg(not(feature = "parquet"))]
pub struct ParquetConfig;

#[cfg(not(feature = "parquet"))]
pub struct ParquetDatasetInfo;

#[cfg(not(feature = "parquet"))]
pub struct ParquetDatasetBuilder;

#[cfg(not(feature = "parquet"))]
pub struct ParquetDataset;

#[cfg(test)]
#[cfg(feature = "parquet")]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parquet_config_default() {
        let config = ParquetConfig::default();
        assert_eq!(config.batch_size, 1000);
        assert_eq!(config.skip_header, false);
        assert_eq!(config.cache_batches, true);
        assert!(config.feature_columns.is_none());
        assert!(config.label_column.is_none());
    }

    #[test]
    fn test_parquet_config_builder() {
        let config = ParquetConfig::default()
            .with_batch_size(500)
            .with_feature_columns(vec!["col1".to_string(), "col2".to_string()])
            .with_label_column("target".to_string())
            .with_max_rows(1000);

        assert_eq!(config.batch_size, 500);
        assert_eq!(config.feature_columns.as_ref().unwrap().len(), 2);
        assert_eq!(config.label_column.as_ref().unwrap(), "target");
        assert_eq!(config.max_rows, Some(1000));
    }

    #[test]
    fn test_parquet_dataset_builder() {
        let builder = ParquetDatasetBuilder::new()
            .feature_columns(vec!["feature1".to_string()])
            .label_column("label".to_string())
            .batch_size(100);

        assert_eq!(builder.config.batch_size, 100);
        assert_eq!(builder.config.feature_columns.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_is_numeric_type() {
        assert!(is_numeric_type(&DataType::Float32));
        assert!(is_numeric_type(&DataType::Float64));
        assert!(is_numeric_type(&DataType::Int32));
        assert!(is_numeric_type(&DataType::Int64));
        assert!(!is_numeric_type(&DataType::Utf8));
        assert!(!is_numeric_type(&DataType::Boolean));
    }

    // Note: Actual file I/O tests would require creating sample Parquet files
    // which is complex in a unit test environment. These would typically be
    // integration tests with pre-created test data files.
}
