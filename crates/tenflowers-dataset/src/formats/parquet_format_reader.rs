//! Parquet format factory and reader implementation
//!
//! This module implements the FormatFactory and FormatReader traits for Parquet files,
//! enabling automatic format detection and unified data loading with zero-copy optimization.

#[cfg(feature = "parquet")]
use crate::error_taxonomy::helpers as error_helpers;
#[cfg(feature = "parquet")]
use crate::formats::unified_reader::{
    read_magic_bytes, DataType, DetectionMethod, FieldInfo, FormatDetection, FormatFactory,
    FormatMetadata, FormatReader, FormatSample,
};
#[cfg(feature = "parquet")]
use std::collections::HashMap;
#[cfg(feature = "parquet")]
use std::path::Path;
#[cfg(feature = "parquet")]
use std::sync::Arc;
#[cfg(feature = "parquet")]
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "parquet")]
use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array};
#[cfg(feature = "parquet")]
use arrow::datatypes::{DataType as ArrowDataType, Schema};
#[cfg(feature = "parquet")]
use arrow::record_batch::RecordBatch;
#[cfg(feature = "parquet")]
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// Parquet format factory for automatic detection and reader creation
#[cfg(feature = "parquet")]
pub struct ParquetFormatFactory;

#[cfg(feature = "parquet")]
impl FormatFactory for ParquetFormatFactory {
    fn format_name(&self) -> &str {
        "Parquet"
    }

    fn extensions(&self) -> Vec<&str> {
        vec!["parquet", "pq"]
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
            Some("parquet") | Some("pq") => {
                confidence = 0.95;
                method = DetectionMethod::Extension;
            }
            _ => {
                // Try magic bytes detection (Parquet magic number is "PAR1")
                if let Ok(is_parquet) = Self::check_parquet_magic(path) {
                    if is_parquet {
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
        Ok(Box::new(ParquetFormatReader::new(path)?))
    }
}

#[cfg(feature = "parquet")]
impl ParquetFormatFactory {
    /// Check Parquet magic bytes
    fn check_parquet_magic(path: &Path) -> Result<bool> {
        if let Ok(bytes) = read_magic_bytes(path, 4) {
            // Parquet files start with "PAR1" magic number
            Ok(bytes.len() >= 4 && &bytes[0..4] == b"PAR1")
        } else {
            Ok(false)
        }
    }
}

/// Parquet format reader implementation
#[cfg(feature = "parquet")]
pub struct ParquetFormatReader {
    batches: Vec<RecordBatch>,
    metadata: FormatMetadata,
    schema: Arc<Schema>,
    feature_columns: Vec<String>,
    label_column: String,
}

#[cfg(feature = "parquet")]
impl ParquetFormatReader {
    /// Create a new Parquet format reader
    pub fn new(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|_| error_helpers::file_not_found("ParquetFormatReader::new", path))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
            TensorError::io_error_simple(format!("Failed to open Parquet file: {}", e))
        })?;

        // Clone schema before moving builder
        let schema = builder.schema().clone();

        // Determine feature and label columns
        let column_names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();

        let label_column = column_names
            .iter()
            .find(|name| {
                name.to_lowercase().contains("label")
                    || name.to_lowercase().contains("target")
                    || name.to_lowercase() == "y"
            })
            .cloned()
            .unwrap_or_else(|| {
                column_names
                    .last()
                    .cloned()
                    .unwrap_or_else(|| "label".to_string())
            });

        let feature_columns: Vec<String> = column_names
            .iter()
            .filter(|name| name != &&label_column)
            .cloned()
            .collect();

        // Read all batches
        let reader = builder.build().map_err(|e| {
            TensorError::io_error_simple(format!("Failed to build Parquet reader: {}", e))
        })?;

        let mut batches = Vec::new();
        let mut total_rows = 0;

        for batch_result in reader {
            let batch = batch_result.map_err(|e| {
                TensorError::io_error_simple(format!("Failed to read batch: {}", e))
            })?;
            total_rows += batch.num_rows();
            batches.push(batch);
        }

        // Infer fields from schema
        let fields = Self::schema_to_fields(&schema);

        let metadata = FormatMetadata {
            format_name: "Parquet".to_string(),
            version: None,
            num_samples: total_rows,
            fields,
            metadata: HashMap::new(),
            supports_random_access: true,
            supports_streaming: true,
        };

        Ok(Self {
            batches,
            metadata,
            schema: schema.clone(),
            feature_columns,
            label_column,
        })
    }

    /// Convert Arrow schema to field info
    fn schema_to_fields(schema: &Schema) -> Vec<FieldInfo> {
        schema
            .fields()
            .iter()
            .map(|field| {
                let dtype = Self::arrow_type_to_dtype(field.data_type());
                FieldInfo {
                    name: field.name().clone(),
                    dtype,
                    shape: None,
                    nullable: field.is_nullable(),
                    description: None,
                }
            })
            .collect()
    }

    /// Convert Arrow data type to unified data type
    fn arrow_type_to_dtype(arrow_type: &ArrowDataType) -> DataType {
        match arrow_type {
            ArrowDataType::Boolean => DataType::Bool,
            ArrowDataType::Int8 => DataType::Int8,
            ArrowDataType::Int16 => DataType::Int16,
            ArrowDataType::Int32 => DataType::Int32,
            ArrowDataType::Int64 => DataType::Int64,
            ArrowDataType::UInt8 => DataType::UInt8,
            ArrowDataType::UInt16 => DataType::UInt16,
            ArrowDataType::UInt32 => DataType::UInt32,
            ArrowDataType::UInt64 => DataType::UInt64,
            ArrowDataType::Float32 => DataType::Float32,
            ArrowDataType::Float64 => DataType::Float64,
            ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 => DataType::String,
            ArrowDataType::Binary | ArrowDataType::LargeBinary => DataType::Binary,
            ArrowDataType::List(field) | ArrowDataType::LargeList(field) => {
                DataType::List(Box::new(Self::arrow_type_to_dtype(field.data_type())))
            }
            _ => DataType::Binary, // Fallback for unsupported types
        }
    }

    /// Find row index across batches
    fn find_batch_and_row(&self, global_index: usize) -> Result<(usize, usize)> {
        let mut cumulative = 0;
        for (batch_idx, batch) in self.batches.iter().enumerate() {
            let batch_size = batch.num_rows();
            if global_index < cumulative + batch_size {
                return Ok((batch_idx, global_index - cumulative));
            }
            cumulative += batch_size;
        }

        Err(TensorError::invalid_argument(format!(
            "Index {} out of bounds",
            global_index
        )))
    }

    /// Extract feature tensor from batch row
    fn extract_features(&self, batch: &RecordBatch, row_index: usize) -> Result<Tensor<f32>> {
        let mut feature_data = Vec::new();

        for col_name in &self.feature_columns {
            let column = batch.column_by_name(col_name).ok_or_else(|| {
                TensorError::invalid_argument(format!("Column '{}' not found", col_name))
            })?;

            let value = Self::extract_scalar_value(column.as_ref(), row_index)?;
            feature_data.push(value);
        }

        let len = feature_data.len();
        Tensor::from_vec(feature_data, &[len])
    }

    /// Extract label tensor from batch row
    fn extract_label(&self, batch: &RecordBatch, row_index: usize) -> Result<Tensor<f32>> {
        let column = batch.column_by_name(&self.label_column).ok_or_else(|| {
            TensorError::invalid_argument(format!("Label column '{}' not found", self.label_column))
        })?;

        let value = Self::extract_scalar_value(column.as_ref(), row_index)?;
        Tensor::from_vec(vec![value], &[1])
    }

    /// Extract scalar value from array at given index
    fn extract_scalar_value(array: &dyn Array, index: usize) -> Result<f32> {
        // Try different array types
        if let Some(arr) = array.as_any().downcast_ref::<Float32Array>() {
            if arr.is_null(index) {
                return Ok(0.0);
            }
            return Ok(arr.value(index));
        }

        if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
            if arr.is_null(index) {
                return Ok(0.0);
            }
            return Ok(arr.value(index) as f32);
        }

        if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
            if arr.is_null(index) {
                return Ok(0.0);
            }
            return Ok(arr.value(index) as f32);
        }

        if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
            if arr.is_null(index) {
                return Ok(0.0);
            }
            return Ok(arr.value(index) as f32);
        }

        Err(TensorError::invalid_argument(
            "Unsupported array type for tensor conversion".to_string(),
        ))
    }
}

#[cfg(feature = "parquet")]
impl FormatReader for ParquetFormatReader {
    fn metadata(&self) -> Result<FormatMetadata> {
        Ok(self.metadata.clone())
    }

    fn get_sample(&self, index: usize) -> Result<FormatSample> {
        let (batch_idx, row_idx) = self.find_batch_and_row(index)?;
        let batch = &self.batches[batch_idx];

        let features = self.extract_features(batch, row_idx)?;
        let labels = self.extract_label(batch, row_idx)?;

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "Parquet".to_string());
        metadata.insert("batch_index".to_string(), batch_idx.to_string());
        metadata.insert("row_index".to_string(), row_idx.to_string());

        Ok(FormatSample {
            features,
            labels,
            source_index: index,
            metadata,
        })
    }

    fn iter(&self) -> Box<dyn Iterator<Item = Result<FormatSample>> + '_> {
        let total_rows = self.batches.iter().map(|b| b.num_rows()).sum();
        Box::new((0..total_rows).map(move |i| self.get_sample(i)))
    }

    fn len(&self) -> usize {
        self.batches.iter().map(|b| b.num_rows()).sum()
    }
}

#[cfg(test)]
#[cfg(feature = "parquet")]
mod tests {
    use super::*;

    #[test]
    fn test_parquet_format_detection() {
        let factory = ParquetFormatFactory;

        let parquet_path = Path::new("data.parquet");
        let detection = factory
            .can_read(parquet_path)
            .expect("test: format detection should succeed");
        assert!(detection.confidence >= 0.9);
        assert_eq!(detection.format_name, "Parquet");
    }

    #[test]
    fn test_arrow_type_conversion() {
        let dt = ParquetFormatReader::arrow_type_to_dtype(&ArrowDataType::Float32);
        assert_eq!(dt, DataType::Float32);

        let dt = ParquetFormatReader::arrow_type_to_dtype(&ArrowDataType::Int64);
        assert_eq!(dt, DataType::Int64);

        let dt = ParquetFormatReader::arrow_type_to_dtype(&ArrowDataType::Utf8);
        assert_eq!(dt, DataType::String);
    }
}
