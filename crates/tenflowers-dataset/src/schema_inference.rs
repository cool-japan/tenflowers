//! Unified schema inference and auto-detection across data formats
//!
//! This module provides automatic schema detection, type inference, and
//! validation across multiple data formats. It can infer tensor shapes,
//! data types, and structure from raw data files.

use std::collections::HashMap;
use std::path::Path;
use tenflowers_core::{Result, TensorError};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Inferred data type for a field
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum InferredDataType {
    /// Boolean type
    Bool,
    /// 8-bit signed integer
    Int8,
    /// 16-bit signed integer
    Int16,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 8-bit unsigned integer
    UInt8,
    /// 16-bit unsigned integer
    UInt16,
    /// 32-bit unsigned integer
    UInt32,
    /// 64-bit unsigned integer
    UInt64,
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// UTF-8 string
    String,
    /// Binary blob
    Binary,
    /// Categorical (discrete values)
    Categorical { num_categories: usize },
    /// Timestamp
    Timestamp,
    /// Complex type (nested structure)
    Complex,
    /// Unknown/Mixed types
    Unknown,
}

impl InferredDataType {
    /// Check if this type is numeric
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            InferredDataType::Int8
                | InferredDataType::Int16
                | InferredDataType::Int32
                | InferredDataType::Int64
                | InferredDataType::UInt8
                | InferredDataType::UInt16
                | InferredDataType::UInt32
                | InferredDataType::UInt64
                | InferredDataType::Float32
                | InferredDataType::Float64
        )
    }

    /// Check if this type is integer
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            InferredDataType::Int8
                | InferredDataType::Int16
                | InferredDataType::Int32
                | InferredDataType::Int64
                | InferredDataType::UInt8
                | InferredDataType::UInt16
                | InferredDataType::UInt32
                | InferredDataType::UInt64
        )
    }

    /// Check if this type is floating point
    pub fn is_float(&self) -> bool {
        matches!(self, InferredDataType::Float32 | InferredDataType::Float64)
    }

    /// Get the size in bytes (if fixed size)
    pub fn size_bytes(&self) -> Option<usize> {
        match self {
            InferredDataType::Bool | InferredDataType::Int8 | InferredDataType::UInt8 => Some(1),
            InferredDataType::Int16 | InferredDataType::UInt16 => Some(2),
            InferredDataType::Int32 | InferredDataType::UInt32 | InferredDataType::Float32 => {
                Some(4)
            }
            InferredDataType::Int64 | InferredDataType::UInt64 | InferredDataType::Float64 => {
                Some(8)
            }
            _ => None, // Variable size
        }
    }
}

/// Inferred field in a schema
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct InferredField {
    /// Field name
    pub name: String,
    /// Inferred data type
    pub dtype: InferredDataType,
    /// Shape (None if scalar, Some for arrays)
    pub shape: Option<Vec<usize>>,
    /// Whether the field can be null/missing
    pub nullable: bool,
    /// Number of samples examined
    pub sample_count: usize,
    /// Percentage of null values found
    pub null_percentage: f64,
    /// For categorical data: unique values seen
    pub unique_values: Option<Vec<String>>,
    /// Statistical properties (for numeric types)
    pub statistics: Option<FieldStatistics>,
}

/// Statistical properties of a numeric field
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct FieldStatistics {
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Number of zeros
    pub zero_count: usize,
    /// Number of negative values
    pub negative_count: usize,
}

/// Inferred schema for a dataset
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct InferredSchema {
    /// Fields in the schema
    pub fields: Vec<InferredField>,
    /// Total number of samples examined
    pub total_samples: usize,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
    /// Format detected
    pub format: Option<String>,
    /// Warnings encountered during inference
    pub warnings: Vec<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl InferredSchema {
    /// Create a new empty schema
    pub fn new() -> Self {
        Self {
            fields: Vec::new(),
            total_samples: 0,
            confidence: 0.0,
            format: None,
            warnings: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Get field by name
    pub fn get_field(&self, name: &str) -> Option<&InferredField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Get all numeric fields
    pub fn numeric_fields(&self) -> Vec<&InferredField> {
        self.fields
            .iter()
            .filter(|f| f.dtype.is_numeric())
            .collect()
    }

    /// Get all categorical fields
    pub fn categorical_fields(&self) -> Vec<&InferredField> {
        self.fields
            .iter()
            .filter(|f| matches!(f.dtype, InferredDataType::Categorical { .. }))
            .collect()
    }

    /// Add a warning
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Check if schema has high confidence
    pub fn has_high_confidence(&self) -> bool {
        self.confidence >= 0.8
    }
}

impl Default for InferredSchema {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for schema inference
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct InferenceConfig {
    /// Maximum number of samples to examine
    pub max_samples: usize,
    /// Minimum samples required for confident inference
    pub min_samples: usize,
    /// Threshold for categorical vs string (max unique values)
    pub categorical_threshold: usize,
    /// Whether to compute statistics for numeric fields
    pub compute_statistics: bool,
    /// Whether to infer nullability
    pub infer_nullability: bool,
    /// Whether to infer shapes for array fields
    pub infer_shapes: bool,
    /// Maximum number of unique values to track for categorical
    pub max_unique_values: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_samples: 1000,
            min_samples: 10,
            categorical_threshold: 50,
            compute_statistics: true,
            infer_nullability: true,
            infer_shapes: true,
            max_unique_values: 1000,
        }
    }
}

/// Schema inference engine
pub struct SchemaInferenceEngine {
    config: InferenceConfig,
}

impl SchemaInferenceEngine {
    /// Create a new inference engine with default configuration
    pub fn new() -> Self {
        Self {
            config: InferenceConfig::default(),
        }
    }

    /// Create a new inference engine with custom configuration
    pub fn with_config(config: InferenceConfig) -> Self {
        Self { config }
    }

    /// Infer schema from a file path
    pub fn infer_from_file(&self, path: &Path) -> Result<InferredSchema> {
        // Detect format from extension
        let extension = path.extension().and_then(|e| e.to_str()).ok_or_else(|| {
            TensorError::unsupported_operation_simple("No file extension found".to_string())
        })?;

        match extension.to_lowercase().as_str() {
            "csv" => self.infer_from_csv(path),
            "json" | "jsonl" => self.infer_from_json(path),
            #[cfg(feature = "parquet")]
            "parquet" => self.infer_from_parquet(path),
            #[cfg(feature = "hdf5")]
            "h5" | "hdf5" => self.infer_from_hdf5(path),
            _ => Err(TensorError::unsupported_operation_simple(format!(
                "Unsupported file format: {}",
                extension
            ))),
        }
    }

    /// Infer schema from CSV file
    #[cfg(feature = "csv_format")]
    fn infer_from_csv(&self, path: &Path) -> Result<InferredSchema> {
        use std::collections::HashMap;
        use std::fs::File;

        let mut schema = InferredSchema::new();
        schema.format = Some("CSV".to_string());

        let file = File::open(path).map_err(|e| {
            TensorError::unsupported_operation_simple(format!("Failed to open CSV file: {}", e))
        })?;

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);

        // Get headers and clone them to avoid borrow issues
        let headers: Vec<String> = reader
            .headers()
            .map_err(|e| {
                TensorError::unsupported_operation_simple(format!(
                    "Failed to read CSV headers: {}",
                    e
                ))
            })?
            .iter()
            .map(|s| s.to_string())
            .collect();

        let num_columns = headers.len();
        let mut column_samples: Vec<Vec<String>> = vec![Vec::new(); num_columns];
        let mut null_counts: Vec<usize> = vec![0; num_columns];
        let mut total_rows = 0usize;

        // Sample up to 1000 rows for inference
        for result in reader.records().take(1000) {
            let record = result.map_err(|e| {
                TensorError::unsupported_operation_simple(format!(
                    "Failed to read CSV record: {}",
                    e
                ))
            })?;

            total_rows += 1;

            for (i, field) in record.iter().enumerate() {
                if i < num_columns {
                    if field.trim().is_empty() {
                        null_counts[i] += 1;
                    } else {
                        column_samples[i].push(field.to_string());
                    }
                }
            }
        }

        // Infer type for each column
        for (i, header) in headers.iter().enumerate() {
            let samples = &column_samples[i];
            let dtype = Self::infer_csv_column_type(samples);
            let null_percentage = if total_rows > 0 {
                (null_counts[i] as f64 / total_rows as f64) * 100.0
            } else {
                0.0
            };

            schema.fields.push(InferredField {
                name: header.to_string(),
                dtype,
                shape: None,
                nullable: null_counts[i] > 0,
                sample_count: total_rows,
                null_percentage,
                unique_values: None,
                statistics: None,
            });
        }

        schema.total_samples = total_rows;
        schema.confidence = 0.9;

        Ok(schema)
    }

    #[cfg(not(feature = "csv_format"))]
    fn infer_from_csv(&self, _path: &Path) -> Result<InferredSchema> {
        let mut schema = InferredSchema::new();
        schema.format = Some("CSV".to_string());
        schema.add_warning(
            "CSV schema inference requires 'csv_format' feature - use CsvDataset directly"
                .to_string(),
        );
        schema.confidence = 0.5;
        Ok(schema)
    }

    /// Infer the data type of a CSV column from sample values
    fn infer_csv_column_type(samples: &[String]) -> InferredDataType {
        if samples.is_empty() {
            return InferredDataType::Unknown;
        }

        let mut is_int = true;
        let mut is_float = true;
        let mut is_bool = true;

        for sample in samples.iter().take(100) {
            let trimmed = sample.trim();

            // Check if boolean
            if is_bool {
                let lower = trimmed.to_lowercase();
                if lower != "true" && lower != "false" && lower != "0" && lower != "1" {
                    is_bool = false;
                }
            }

            // Check if integer
            if is_int && trimmed.parse::<i64>().is_err() {
                is_int = false;
            }

            // Check if float
            if is_float && trimmed.parse::<f64>().is_err() {
                is_float = false;
            }

            // Early exit if all numeric types ruled out
            if !is_bool && !is_int && !is_float {
                break;
            }
        }

        if is_bool {
            InferredDataType::Bool
        } else if is_int {
            InferredDataType::Int64
        } else if is_float {
            InferredDataType::Float64
        } else {
            InferredDataType::String
        }
    }

    /// Infer schema from JSON file
    fn infer_from_json(&self, path: &Path) -> Result<InferredSchema> {
        let mut schema = InferredSchema::new();
        schema.format = Some("JSON".to_string());

        // For now, add a placeholder implementation
        schema.add_warning("JSON schema inference not fully implemented".to_string());
        schema.confidence = 0.5;

        Ok(schema)
    }

    /// Infer schema from Parquet file
    #[cfg(feature = "parquet")]
    fn infer_from_parquet(&self, path: &Path) -> Result<InferredSchema> {
        use arrow::array::ArrayRef;
        use arrow::datatypes::{DataType, Schema};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let file = std::fs::File::open(path).map_err(|e| {
            TensorError::unsupported_operation_simple(format!("Failed to open Parquet file: {}", e))
        })?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
            TensorError::unsupported_operation_simple(format!(
                "Failed to create Parquet reader: {}",
                e
            ))
        })?;

        let arrow_schema = builder.schema().clone();
        let _reader = builder.build().map_err(|e| {
            TensorError::unsupported_operation_simple(format!(
                "Failed to build Parquet reader: {}",
                e
            ))
        })?;

        let mut schema = InferredSchema::new();
        schema.format = Some("Parquet".to_string());

        // Convert Arrow schema to inferred schema
        for field in arrow_schema.fields() {
            let dtype = Self::arrow_type_to_inferred(field.data_type());
            let inferred_field = InferredField {
                name: field.name().clone(),
                dtype,
                shape: None,
                nullable: field.is_nullable(),
                sample_count: 0,
                null_percentage: 0.0,
                unique_values: None,
                statistics: None,
            };
            schema.fields.push(inferred_field);
        }

        schema.total_samples = 0; // Could read metadata for row count
        schema.confidence = 0.95; // High confidence from explicit schema

        Ok(schema)
    }

    #[cfg(feature = "parquet")]
    fn arrow_type_to_inferred(arrow_type: &arrow::datatypes::DataType) -> InferredDataType {
        use arrow::datatypes::DataType;

        match arrow_type {
            DataType::Boolean => InferredDataType::Bool,
            DataType::Int8 => InferredDataType::Int8,
            DataType::Int16 => InferredDataType::Int16,
            DataType::Int32 => InferredDataType::Int32,
            DataType::Int64 => InferredDataType::Int64,
            DataType::UInt8 => InferredDataType::UInt8,
            DataType::UInt16 => InferredDataType::UInt16,
            DataType::UInt32 => InferredDataType::UInt32,
            DataType::UInt64 => InferredDataType::UInt64,
            DataType::Float32 => InferredDataType::Float32,
            DataType::Float64 => InferredDataType::Float64,
            DataType::Utf8 | DataType::LargeUtf8 => InferredDataType::String,
            DataType::Binary | DataType::LargeBinary => InferredDataType::Binary,
            DataType::Timestamp(_, _) => InferredDataType::Timestamp,
            _ => InferredDataType::Complex,
        }
    }

    /// Infer schema from HDF5 file
    #[cfg(feature = "hdf5")]
    fn infer_from_hdf5(&self, _path: &Path) -> Result<InferredSchema> {
        let mut schema = InferredSchema::new();
        schema.format = Some("HDF5".to_string());
        schema.add_warning("HDF5 schema inference not fully implemented".to_string());
        schema.confidence = 0.5;
        Ok(schema)
    }

    /// Infer field type from string samples
    fn infer_field_type(
        &self,
        samples: &[String],
    ) -> (
        InferredDataType,
        Option<Vec<String>>,
        Option<FieldStatistics>,
    ) {
        if samples.is_empty() {
            return (InferredDataType::Unknown, None, None);
        }

        // Try to parse as different types
        let mut all_bool = true;
        let mut all_int = true;
        let mut all_float = true;
        let mut numeric_values = Vec::new();

        for sample in samples {
            let trimmed = sample.trim();

            // Check boolean
            if all_bool {
                all_bool = matches!(
                    trimmed.to_lowercase().as_str(),
                    "true" | "false" | "t" | "f" | "yes" | "no" | "y" | "n" | "1" | "0"
                );
            }

            // Check integer
            if all_int {
                if let Ok(val) = trimmed.parse::<i64>() {
                    numeric_values.push(val as f64);
                } else {
                    all_int = false;
                }
            }

            // Check float
            if all_float && !all_int {
                if let Ok(val) = trimmed.parse::<f64>() {
                    if !all_int {
                        numeric_values.push(val);
                    }
                } else {
                    all_float = false;
                }
            }
        }

        // Determine type
        let dtype = if all_bool {
            InferredDataType::Bool
        } else if all_int {
            // Choose appropriate int size based on values
            if let Some(&max_val) = numeric_values.iter().max_by(|a, b| a.total_cmp(b)) {
                if let Some(&min_val) = numeric_values.iter().min_by(|a, b| a.total_cmp(b)) {
                    if min_val >= 0.0 && max_val <= u8::MAX as f64 {
                        InferredDataType::UInt8
                    } else if min_val >= i8::MIN as f64 && max_val <= i8::MAX as f64 {
                        InferredDataType::Int8
                    } else if min_val >= 0.0 && max_val <= u16::MAX as f64 {
                        InferredDataType::UInt16
                    } else if min_val >= i16::MIN as f64 && max_val <= i16::MAX as f64 {
                        InferredDataType::Int16
                    } else if min_val >= 0.0 && max_val <= u32::MAX as f64 {
                        InferredDataType::UInt32
                    } else if min_val >= i32::MIN as f64 && max_val <= i32::MAX as f64 {
                        InferredDataType::Int32
                    } else {
                        InferredDataType::Int64
                    }
                } else {
                    InferredDataType::Int32
                }
            } else {
                InferredDataType::Int32
            }
        } else if all_float {
            InferredDataType::Float64
        } else {
            // Check for categorical vs string
            let unique_count = samples
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len();
            if unique_count <= self.config.categorical_threshold {
                InferredDataType::Categorical {
                    num_categories: unique_count,
                }
            } else {
                InferredDataType::String
            }
        };

        // Collect unique values for categorical
        let unique_values = if matches!(dtype, InferredDataType::Categorical { .. }) {
            let mut uniques: Vec<String> = samples
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .take(self.config.max_unique_values)
                .cloned()
                .collect();
            uniques.sort();
            Some(uniques)
        } else {
            None
        };

        // Compute statistics for numeric types
        let statistics =
            if self.config.compute_statistics && dtype.is_numeric() && !numeric_values.is_empty() {
                let n = numeric_values.len() as f64;
                let mean = numeric_values.iter().sum::<f64>() / n;
                let variance = numeric_values
                    .iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>()
                    / n;
                let std = variance.sqrt();
                let min = numeric_values
                    .iter()
                    .min_by(|a, b| a.total_cmp(b))
                    .copied()
                    .unwrap_or(0.0);
                let max = numeric_values
                    .iter()
                    .max_by(|a, b| a.total_cmp(b))
                    .copied()
                    .unwrap_or(0.0);
                let zero_count = numeric_values.iter().filter(|&&x| x == 0.0).count();
                let negative_count = numeric_values.iter().filter(|&&x| x < 0.0).count();

                Some(FieldStatistics {
                    min,
                    max,
                    mean,
                    std,
                    zero_count,
                    negative_count,
                })
            } else {
                None
            };

        (dtype, unique_values, statistics)
    }

    /// Generate a report from inferred schema
    pub fn generate_report(&self, schema: &InferredSchema) -> String {
        let mut report = String::new();

        report.push_str("=== Schema Inference Report ===\n\n");

        if let Some(ref format) = schema.format {
            report.push_str(&format!("Format: {}\n", format));
        }
        report.push_str(&format!("Total Samples: {}\n", schema.total_samples));
        report.push_str(&format!("Confidence: {:.2}%\n", schema.confidence * 100.0));
        report.push_str(&format!("Number of Fields: {}\n\n", schema.fields.len()));

        if !schema.warnings.is_empty() {
            report.push_str("Warnings:\n");
            for warning in &schema.warnings {
                report.push_str(&format!("  - {}\n", warning));
            }
            report.push('\n');
        }

        report.push_str("Fields:\n");
        for field in &schema.fields {
            report.push_str(&format!("  - {}\n", field.name));
            report.push_str(&format!("    Type: {:?}\n", field.dtype));
            report.push_str(&format!("    Nullable: {}\n", field.nullable));
            if field.nullable {
                report.push_str(&format!(
                    "    Null Percentage: {:.2}%\n",
                    field.null_percentage * 100.0
                ));
            }

            if let Some(ref stats) = field.statistics {
                report.push_str("    Statistics:\n");
                report.push_str(&format!("      Min: {:.4}\n", stats.min));
                report.push_str(&format!("      Max: {:.4}\n", stats.max));
                report.push_str(&format!("      Mean: {:.4}\n", stats.mean));
                report.push_str(&format!("      Std: {:.4}\n", stats.std));
            }

            if let Some(ref uniques) = field.unique_values {
                report.push_str(&format!("    Unique Values ({}):\n", uniques.len()));
                let display_count = uniques.len().min(10);
                for (i, val) in uniques.iter().take(display_count).enumerate() {
                    report.push_str(&format!("      {}: {}\n", i + 1, val));
                }
                if uniques.len() > display_count {
                    report.push_str(&format!(
                        "      ... and {} more\n",
                        uniques.len() - display_count
                    ));
                }
            }
        }

        report
    }
}

impl Default for SchemaInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inferred_data_type() {
        assert!(InferredDataType::Float32.is_numeric());
        assert!(InferredDataType::Int32.is_integer());
        assert!(InferredDataType::Float64.is_float());
        assert!(!InferredDataType::String.is_numeric());

        assert_eq!(InferredDataType::Int32.size_bytes(), Some(4));
        assert_eq!(InferredDataType::Float64.size_bytes(), Some(8));
        assert_eq!(InferredDataType::String.size_bytes(), None);
    }

    #[test]
    fn test_inferred_schema_creation() {
        let schema = InferredSchema::new();
        assert_eq!(schema.fields.len(), 0);
        assert_eq!(schema.total_samples, 0);
        assert_eq!(schema.confidence, 0.0);
        assert!(schema.format.is_none());
    }

    #[test]
    fn test_schema_field_operations() {
        let mut schema = InferredSchema::new();

        schema.fields.push(InferredField {
            name: "age".to_string(),
            dtype: InferredDataType::Int32,
            shape: None,
            nullable: false,
            sample_count: 100,
            null_percentage: 0.0,
            unique_values: None,
            statistics: None,
        });

        schema.fields.push(InferredField {
            name: "score".to_string(),
            dtype: InferredDataType::Float32,
            shape: None,
            nullable: false,
            sample_count: 100,
            null_percentage: 0.0,
            unique_values: None,
            statistics: None,
        });

        assert!(schema.get_field("age").is_some());
        assert!(schema.get_field("unknown").is_none());
        assert_eq!(schema.numeric_fields().len(), 2);
    }

    #[test]
    fn test_inference_config() {
        let config = InferenceConfig::default();
        assert_eq!(config.max_samples, 1000);
        assert_eq!(config.min_samples, 10);
        assert_eq!(config.categorical_threshold, 50);
        assert!(config.compute_statistics);
    }

    #[test]
    fn test_schema_inference_engine_creation() {
        let engine = SchemaInferenceEngine::new();
        assert_eq!(engine.config.max_samples, 1000);

        let custom_config = InferenceConfig {
            max_samples: 500,
            ..Default::default()
        };
        let engine2 = SchemaInferenceEngine::with_config(custom_config);
        assert_eq!(engine2.config.max_samples, 500);
    }

    #[test]
    fn test_field_type_inference_integers() {
        let engine = SchemaInferenceEngine::new();
        let samples = vec!["1".to_string(), "2".to_string(), "3".to_string()];

        let (dtype, _, stats) = engine.infer_field_type(&samples);
        assert!(dtype.is_integer());
        assert!(stats.is_some());

        if let Some(s) = stats {
            assert_eq!(s.min, 1.0);
            assert_eq!(s.max, 3.0);
            assert_eq!(s.mean, 2.0);
        }
    }

    #[test]
    fn test_field_type_inference_floats() {
        let engine = SchemaInferenceEngine::new();
        let samples = vec!["1.5".to_string(), "2.5".to_string(), "3.5".to_string()];

        let (dtype, _, stats) = engine.infer_field_type(&samples);
        assert_eq!(dtype, InferredDataType::Float64);
        assert!(stats.is_some());
    }

    #[test]
    fn test_field_type_inference_categorical() {
        let engine = SchemaInferenceEngine::new();
        let samples = vec![
            "red".to_string(),
            "green".to_string(),
            "blue".to_string(),
            "red".to_string(),
        ];

        let (dtype, unique_vals, _) = engine.infer_field_type(&samples);
        assert!(matches!(dtype, InferredDataType::Categorical { .. }));
        assert!(unique_vals.is_some());

        if let Some(vals) = unique_vals {
            assert_eq!(vals.len(), 3);
        }
    }

    #[test]
    fn test_field_type_inference_boolean() {
        let engine = SchemaInferenceEngine::new();
        let samples = vec!["true".to_string(), "false".to_string(), "true".to_string()];

        let (dtype, _, _) = engine.infer_field_type(&samples);
        assert_eq!(dtype, InferredDataType::Bool);
    }

    #[test]
    fn test_generate_report() {
        let engine = SchemaInferenceEngine::new();
        let mut schema = InferredSchema::new();
        schema.format = Some("CSV".to_string());
        schema.total_samples = 100;
        schema.confidence = 0.9;

        schema.fields.push(InferredField {
            name: "age".to_string(),
            dtype: InferredDataType::Int32,
            shape: None,
            nullable: false,
            sample_count: 100,
            null_percentage: 0.0,
            unique_values: None,
            statistics: Some(FieldStatistics {
                min: 18.0,
                max: 65.0,
                mean: 35.5,
                std: 12.3,
                zero_count: 0,
                negative_count: 0,
            }),
        });

        let report = engine.generate_report(&schema);
        assert!(report.contains("Schema Inference Report"));
        assert!(report.contains("Format: CSV"));
        assert!(report.contains("Confidence: 90"));
        assert!(report.contains("age"));
    }
}
