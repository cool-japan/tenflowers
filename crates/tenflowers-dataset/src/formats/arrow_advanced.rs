//! Advanced Apache Arrow features for zero-copy and performance optimization
//!
//! This module provides advanced Arrow features including:
//! - Predicate pushdown for efficient filtering
//! - Streaming large datasets without loading all data into memory
//! - Advanced zero-copy operations with buffer reuse
//! - Arrow Flight integration for distributed data access
//! - Query optimization using Arrow statistics

use crate::error_taxonomy::helpers as error_helpers;
use std::path::Path;
use std::sync::Arc;
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "parquet")]
use arrow::array::*;
#[cfg(feature = "parquet")]
use arrow::compute;
#[cfg(feature = "parquet")]
use arrow::compute::kernels::cmp::{eq, gt, lt};
#[cfg(feature = "parquet")]
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
#[cfg(feature = "parquet")]
use arrow::record_batch::RecordBatch;
#[cfg(feature = "parquet")]
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
#[cfg(feature = "parquet")]
use parquet::file::metadata::RowGroupMetaData;
#[cfg(feature = "parquet")]
use parquet::file::statistics::Statistics;

/// Predicate for filtering Arrow data
#[derive(Debug, Clone)]
pub enum ArrowPredicate {
    /// Column equals value
    Equals(String, ArrowValue),
    /// Column not equals value
    NotEquals(String, ArrowValue),
    /// Column greater than value
    GreaterThan(String, ArrowValue),
    /// Column less than value
    LessThan(String, ArrowValue),
    /// Column is in a list of values
    In(String, Vec<ArrowValue>),
    /// Column is null
    IsNull(String),
    /// Column is not null
    IsNotNull(String),
    /// Logical AND of predicates
    And(Vec<ArrowPredicate>),
    /// Logical OR of predicates
    Or(Vec<ArrowPredicate>),
    /// Logical NOT of predicate
    Not(Box<ArrowPredicate>),
}

/// Value types for predicates
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ArrowValue {
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    String(String),
    Bool(bool),
}

impl ArrowPredicate {
    /// Create an equals predicate
    pub fn eq(column: impl Into<String>, value: ArrowValue) -> Self {
        Self::Equals(column.into(), value)
    }

    /// Create a not equals predicate
    pub fn ne(column: impl Into<String>, value: ArrowValue) -> Self {
        Self::NotEquals(column.into(), value)
    }

    /// Create a greater than predicate
    pub fn gt(column: impl Into<String>, value: ArrowValue) -> Self {
        Self::GreaterThan(column.into(), value)
    }

    /// Create a less than predicate
    pub fn lt(column: impl Into<String>, value: ArrowValue) -> Self {
        Self::LessThan(column.into(), value)
    }

    /// Create an IN predicate
    pub fn is_in(column: impl Into<String>, values: Vec<ArrowValue>) -> Self {
        Self::In(column.into(), values)
    }

    /// Create an IS NULL predicate
    pub fn is_null(column: impl Into<String>) -> Self {
        Self::IsNull(column.into())
    }

    /// Create an IS NOT NULL predicate
    pub fn is_not_null(column: impl Into<String>) -> Self {
        Self::IsNotNull(column.into())
    }

    /// Combine predicates with AND
    pub fn and(predicates: Vec<ArrowPredicate>) -> Self {
        Self::And(predicates)
    }

    /// Combine predicates with OR
    pub fn or(predicates: Vec<ArrowPredicate>) -> Self {
        Self::Or(predicates)
    }

    /// Negate predicate
    pub fn not(predicate: ArrowPredicate) -> Self {
        Self::Not(Box::new(predicate))
    }
}

/// Statistics extracted from Arrow/Parquet metadata
#[cfg(feature = "parquet")]
#[derive(Debug, Clone)]
pub struct ArrowStatistics {
    /// Column name
    pub column_name: String,
    /// Minimum value
    pub min: Option<ArrowValue>,
    /// Maximum value
    pub max: Option<ArrowValue>,
    /// Number of null values
    pub null_count: usize,
    /// Number of distinct values (if available)
    pub distinct_count: Option<usize>,
    /// Total number of values
    pub row_count: usize,
}

#[cfg(feature = "parquet")]
impl ArrowStatistics {
    /// Check if a predicate can be eliminated using these statistics
    pub fn can_skip_with_predicate(&self, predicate: &ArrowPredicate) -> bool {
        match predicate {
            ArrowPredicate::Equals(col, val) if col == &self.column_name => {
                // If value is outside min/max range, can skip
                if let (Some(min), Some(max)) = (&self.min, &self.max) {
                    val < min || val > max
                } else {
                    false
                }
            }
            ArrowPredicate::GreaterThan(col, val) if col == &self.column_name => {
                // If max <= value, can skip
                if let Some(max) = &self.max {
                    max <= val
                } else {
                    false
                }
            }
            ArrowPredicate::LessThan(col, val) if col == &self.column_name => {
                // If min >= value, can skip
                if let Some(min) = &self.min {
                    min >= val
                } else {
                    false
                }
            }
            ArrowPredicate::IsNull(col) if col == &self.column_name => {
                // If no nulls, can skip
                self.null_count == 0
            }
            ArrowPredicate::IsNotNull(col) if col == &self.column_name => {
                // If all nulls, can skip
                self.null_count == self.row_count
            }
            _ => false,
        }
    }
}

/// Configuration for streaming Arrow data
#[derive(Debug, Clone)]
pub struct StreamingArrowConfig {
    /// Batch size for streaming
    pub batch_size: usize,
    /// Predicates for filtering (pushdown)
    pub predicates: Vec<ArrowPredicate>,
    /// Columns to project (None = all columns)
    pub projection: Option<Vec<String>>,
    /// Maximum memory budget in bytes
    pub memory_limit: Option<usize>,
    /// Enable statistics-based pruning
    pub enable_statistics_pruning: bool,
}

impl Default for StreamingArrowConfig {
    fn default() -> Self {
        Self {
            batch_size: 8192,
            predicates: Vec::new(),
            projection: None,
            memory_limit: None,
            enable_statistics_pruning: true,
        }
    }
}

/// Streaming Arrow reader for large datasets
#[cfg(feature = "parquet")]
pub struct StreamingArrowReader {
    path: std::path::PathBuf,
    config: StreamingArrowConfig,
    current_batch: usize,
    total_batches: usize,
    schema: Arc<Schema>,
}

/// Shared downcast + compute-kernel dispatch for column-vs-scalar comparisons.
///
/// Centralizes the `ArrowValue` match that used to be duplicated (with
/// drifting coverage) across `evaluate_equals`, `evaluate_greater_than`, and
/// `evaluate_less_than`: `evaluate_equals` covered Int32/Int64/Float64,
/// while `evaluate_greater_than`/`evaluate_less_than` covered only
/// Int32/Float64. This single function now covers all six `ArrowValue`
/// variants (Int32/Int64/Float32/Float64/String/Bool) for every caller, so
/// the coverage cannot drift apart again.
///
/// # Scalar broadcasting correctness
///
/// The comparison value is wrapped in [`arrow::array::Scalar::new`] before
/// being handed to `kernel`. This is required for correctness: arrow's
/// `cmp` kernels (`eq`/`gt`/`lt`) only broadcast one side against a longer
/// array when that side is explicitly marked scalar via `Scalar`. Comparing
/// a plain, non-scalar length-1 array against an N-row column is *not*
/// broadcasting: `arrow_ord::cmp::compare_op` explicitly rejects two
/// non-scalar operands of different lengths with
/// `ArrowError::InvalidArgumentError("Cannot compare arrays of different
/// lengths, got N vs 1")`. In other words, without this `Scalar` wrapper,
/// every call to `evaluate_equals`/`evaluate_greater_than`/
/// `evaluate_less_than` on a batch with more than one row would fail
/// outright (verified empirically against arrow 59.0.0) -- this predicate
/// subsystem was unusable for realistically-sized batches until this fix.
#[cfg(feature = "parquet")]
fn evaluate_comparison(
    batch: &RecordBatch,
    column: &str,
    value: &ArrowValue,
    kernel: impl Fn(
        &dyn Datum,
        &dyn Datum,
    ) -> std::result::Result<BooleanArray, arrow::error::ArrowError>,
    op_name: &str,
) -> Result<BooleanArray> {
    let col = batch
        .column_by_name(column)
        .ok_or_else(|| error_helpers::schema_mismatch(op_name, column, "column not found"))?;

    // A local macro parameterized over the concrete Arrow array type. `col`
    // is passed explicitly (rather than captured as a free identifier) so
    // this doesn't depend on macro-hygiene name resolution at all.
    macro_rules! compare_column_as {
        ($col:expr, $array_ty:ty, $scalar_value:expr) => {{
            let typed_col = $col.as_any().downcast_ref::<$array_ty>().ok_or_else(|| {
                TensorError::unsupported_operation_simple(format!(
                    "{}: column '{}' has Arrow type {:?}, incompatible with predicate value {:?}",
                    op_name,
                    column,
                    $col.data_type(),
                    value
                ))
            })?;
            let scalar = Scalar::new(<$array_ty>::from(vec![$scalar_value]));
            kernel(typed_col, &scalar).map_err(|e| {
                TensorError::unsupported_operation_simple(format!(
                    "{}: comparison kernel failed on column '{}': {}",
                    op_name, column, e
                ))
            })
        }};
    }

    match value {
        ArrowValue::Int32(v) => compare_column_as!(col, Int32Array, *v),
        ArrowValue::Int64(v) => compare_column_as!(col, Int64Array, *v),
        ArrowValue::Float32(v) => compare_column_as!(col, Float32Array, *v),
        ArrowValue::Float64(v) => compare_column_as!(col, Float64Array, *v),
        ArrowValue::String(v) => compare_column_as!(col, StringArray, v.clone()),
        ArrowValue::Bool(v) => compare_column_as!(col, BooleanArray, *v),
    }
}

#[cfg(feature = "parquet")]
impl StreamingArrowReader {
    /// Create a new streaming reader
    pub fn new(path: impl AsRef<Path>, config: StreamingArrowConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        if !path.exists() {
            return Err(error_helpers::file_not_found(
                "StreamingArrowReader::new",
                &path,
            ));
        }

        let file = std::fs::File::open(&path)
            .map_err(|_| error_helpers::file_not_found("StreamingArrowReader::new", &path))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| TensorError::io_error_simple(format!("Failed to open Parquet: {}", e)))?;

        let metadata = builder.metadata();
        let schema = builder.schema();
        let total_batches = metadata.num_row_groups();

        Ok(Self {
            path,
            config,
            current_batch: 0,
            total_batches,
            schema: schema.clone(),
        })
    }

    /// Get the schema
    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    /// Get total number of batches
    pub fn total_batches(&self) -> usize {
        self.total_batches
    }

    /// Read next batch
    pub fn read_next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.current_batch >= self.total_batches {
            return Ok(None);
        }

        let file = std::fs::File::open(&self.path).map_err(|_| {
            error_helpers::file_not_found("StreamingArrowReader::read_next_batch", &self.path)
        })?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| TensorError::io_error_simple(format!("Failed to open Parquet: {}", e)))?;

        // Apply projection if specified
        let mut builder = builder.with_batch_size(self.config.batch_size);

        // Projection support - simplified for compatibility
        // Advanced projection can be added once Arrow API is stabilized
        // if let Some(ref projection) = self.config.projection {
        //     ...projection logic...
        // }

        let mut reader = builder
            .build()
            .map_err(|e| TensorError::io_error_simple(format!("Failed to build reader: {}", e)))?;

        // Skip to current batch
        for _ in 0..self.current_batch {
            if reader.next().is_none() {
                return Ok(None);
            }
        }

        // Read the batch
        let batch = match reader.next() {
            Some(Ok(batch)) => batch,
            Some(Err(e)) => {
                return Err(TensorError::io_error_simple(format!(
                    "Failed to read batch: {}",
                    e
                )));
            }
            None => return Ok(None),
        };

        self.current_batch += 1;

        // Apply predicates if any
        let batch = if !self.config.predicates.is_empty() {
            self.apply_predicates(batch)?
        } else {
            batch
        };

        Ok(Some(batch))
    }

    /// Reset reader to beginning
    pub fn reset(&mut self) {
        self.current_batch = 0;
    }

    /// Apply predicates to filter a batch
    fn apply_predicates(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let mut mask: Option<BooleanArray> = None;

        for predicate in &self.config.predicates {
            let pred_mask = self.evaluate_predicate(&batch, predicate)?;

            mask = match mask {
                None => Some(pred_mask),
                Some(existing) => {
                    // Combine with AND
                    Some(compute::and(&existing, &pred_mask).map_err(|e| {
                        TensorError::unsupported_operation_simple(format!(
                            "Failed to combine predicates: {}",
                            e
                        ))
                    })?)
                }
            };
        }

        // Filter the batch using the mask
        if let Some(mask) = mask {
            compute::filter_record_batch(&batch, &mask).map_err(|e| {
                TensorError::unsupported_operation_simple(format!("Failed to filter batch: {}", e))
            })
        } else {
            Ok(batch)
        }
    }

    /// Evaluate a single predicate on a batch
    fn evaluate_predicate(
        &self,
        batch: &RecordBatch,
        predicate: &ArrowPredicate,
    ) -> Result<BooleanArray> {
        match predicate {
            ArrowPredicate::Equals(column, value) => self.evaluate_equals(batch, column, value),
            ArrowPredicate::NotEquals(column, value) => {
                let equals = self.evaluate_equals(batch, column, value)?;
                Ok(compute::not(&equals).map_err(|e| {
                    TensorError::unsupported_operation_simple(format!("Failed to negate: {}", e))
                })?)
            }
            ArrowPredicate::GreaterThan(column, value) => {
                self.evaluate_greater_than(batch, column, value)
            }
            ArrowPredicate::LessThan(column, value) => {
                self.evaluate_less_than(batch, column, value)
            }
            ArrowPredicate::In(column, values) => {
                if values.is_empty() {
                    // An empty IN-list can never match any row: this is
                    // "match nothing", not an error condition.
                    let col = batch.column_by_name(column).ok_or_else(|| {
                        error_helpers::schema_mismatch(
                            "evaluate_predicate",
                            column,
                            "column not found",
                        )
                    })?;
                    return Ok(BooleanArray::from(vec![false; col.len()]));
                }

                let mut result: Option<BooleanArray> = None;
                for value in values {
                    let mask = self.evaluate_equals(batch, column, value)?;
                    result = match result {
                        None => Some(mask),
                        Some(existing) => Some(compute::or(&existing, &mask).map_err(|e| {
                            TensorError::unsupported_operation_simple(format!(
                                "Failed to OR IN-predicate masks: {}",
                                e
                            ))
                        })?),
                    };
                }
                result.ok_or_else(|| {
                    TensorError::unsupported_operation_simple("Empty IN predicate".to_string())
                })
            }
            ArrowPredicate::IsNull(column) => {
                let col = batch.column_by_name(column).ok_or_else(|| {
                    error_helpers::schema_mismatch("evaluate_predicate", column, "column not found")
                })?;
                Ok(compute::is_null(col.as_ref()).map_err(|e| {
                    TensorError::unsupported_operation_simple(format!(
                        "Failed to check null: {}",
                        e
                    ))
                })?)
            }
            ArrowPredicate::IsNotNull(column) => {
                let col = batch.column_by_name(column).ok_or_else(|| {
                    error_helpers::schema_mismatch("evaluate_predicate", column, "column not found")
                })?;
                Ok(compute::is_not_null(col.as_ref()).map_err(|e| {
                    TensorError::unsupported_operation_simple(format!(
                        "Failed to check not null: {}",
                        e
                    ))
                })?)
            }
            ArrowPredicate::And(predicates) => {
                let mut result = None;
                for pred in predicates {
                    let mask = self.evaluate_predicate(batch, pred)?;
                    result = match result {
                        None => Some(mask),
                        Some(existing) => Some(compute::and(&existing, &mask).map_err(|e| {
                            TensorError::unsupported_operation_simple(format!(
                                "Failed to AND predicates: {}",
                                e
                            ))
                        })?),
                    };
                }
                result.ok_or_else(|| {
                    TensorError::unsupported_operation_simple("Empty AND predicate".to_string())
                })
            }
            ArrowPredicate::Or(predicates) => {
                let mut result = None;
                for pred in predicates {
                    let mask = self.evaluate_predicate(batch, pred)?;
                    result = match result {
                        None => Some(mask),
                        Some(existing) => Some(compute::or(&existing, &mask).map_err(|e| {
                            TensorError::unsupported_operation_simple(format!(
                                "Failed to OR predicates: {}",
                                e
                            ))
                        })?),
                    };
                }
                result.ok_or_else(|| {
                    TensorError::unsupported_operation_simple("Empty OR predicate".to_string())
                })
            }
            ArrowPredicate::Not(pred) => {
                let mask = self.evaluate_predicate(batch, pred)?;
                Ok(compute::not(&mask).map_err(|e| {
                    TensorError::unsupported_operation_simple(format!("Failed to NOT: {}", e))
                })?)
            }
        }
    }

    fn evaluate_equals(
        &self,
        batch: &RecordBatch,
        column: &str,
        value: &ArrowValue,
    ) -> Result<BooleanArray> {
        evaluate_comparison(batch, column, value, eq, "evaluate_equals")
    }

    fn evaluate_greater_than(
        &self,
        batch: &RecordBatch,
        column: &str,
        value: &ArrowValue,
    ) -> Result<BooleanArray> {
        evaluate_comparison(batch, column, value, gt, "evaluate_greater_than")
    }

    fn evaluate_less_than(
        &self,
        batch: &RecordBatch,
        column: &str,
        value: &ArrowValue,
    ) -> Result<BooleanArray> {
        evaluate_comparison(batch, column, value, lt, "evaluate_less_than")
    }
}

/// Zero-copy buffer wrapper for Arrow data
#[cfg(feature = "parquet")]
pub struct ArrowBuffer<T> {
    data: Arc<dyn Array>,
    offset: usize,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "parquet")]
impl<T> ArrowBuffer<T> {
    /// Create a new buffer from an Arrow array
    pub fn from_array(array: Arc<dyn Array>) -> Self {
        let len = array.len();
        Self {
            data: array,
            offset: 0,
            len,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Slice the buffer
    pub fn slice(&self, offset: usize, length: usize) -> Result<Self> {
        if offset + length > self.len {
            return Err(TensorError::invalid_argument(
                "Slice out of bounds".to_string(),
            ));
        }

        Ok(Self {
            data: self.data.clone(),
            offset: self.offset + offset,
            len: length,
            _phantom: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
#[cfg(feature = "parquet")]
mod tests {
    use super::*;

    #[test]
    fn test_arrow_predicate_creation() {
        let pred = ArrowPredicate::eq("age", ArrowValue::Int32(25));
        match pred {
            ArrowPredicate::Equals(col, val) => {
                assert_eq!(col, "age");
                assert_eq!(val, ArrowValue::Int32(25));
            }
            _ => panic!("Wrong predicate type"),
        }
    }

    #[test]
    fn test_arrow_predicate_and() {
        let pred1 = ArrowPredicate::gt("age", ArrowValue::Int32(18));
        let pred2 = ArrowPredicate::lt("age", ArrowValue::Int32(65));
        let combined = ArrowPredicate::and(vec![pred1, pred2]);

        match combined {
            ArrowPredicate::And(preds) => {
                assert_eq!(preds.len(), 2);
            }
            _ => panic!("Wrong predicate type"),
        }
    }

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingArrowConfig::default();
        assert_eq!(config.batch_size, 8192);
        assert!(config.predicates.is_empty());
        assert!(config.projection.is_none());
        assert!(config.enable_statistics_pruning);
    }

    #[test]
    fn test_arrow_value_equality() {
        assert_eq!(ArrowValue::Int32(42), ArrowValue::Int32(42));
        assert_ne!(ArrowValue::Int32(42), ArrowValue::Int32(43));
        assert_eq!(ArrowValue::Float64(2.5), ArrowValue::Float64(2.5));
    }

    // -----------------------------------------------------------------
    // evaluate_comparison / evaluate_equals / evaluate_greater_than /
    // evaluate_less_than / In-predicate coverage
    //
    // These use a genuinely multi-row (5-row) RecordBatch on purpose: a
    // 1-row batch would not have caught the Scalar::new broadcast bug (see
    // `evaluate_comparison`'s doc comment). Without the `Scalar::new`
    // wrapper, arrow's cmp kernels reject two non-scalar operands of
    // different lengths outright, so every one of these tests would fail
    // at its `.expect(...)` call (not silently return a wrong-length
    // result) against a 1-row-only regression check.
    // -----------------------------------------------------------------

    /// Build a 5-row RecordBatch with one column per `ArrowValue` variant.
    fn build_predicate_test_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("int32_col", ArrowDataType::Int32, false),
            Field::new("int64_col", ArrowDataType::Int64, false),
            Field::new("float32_col", ArrowDataType::Float32, false),
            Field::new("float64_col", ArrowDataType::Float64, false),
            Field::new("string_col", ArrowDataType::Utf8, false),
            Field::new("bool_col", ArrowDataType::Boolean, false),
        ]));

        let int32_col: ArrayRef = Arc::new(Int32Array::from(vec![10, 20, 20, 30, 40]));
        let int64_col: ArrayRef = Arc::new(Int64Array::from(vec![100i64, 200, 300, 200, 500]));
        let float32_col: ArrayRef = Arc::new(Float32Array::from(vec![1.5f32, 2.5, 3.5, 2.5, 5.5]));
        let float64_col: ArrayRef = Arc::new(Float64Array::from(vec![1.5f64, 2.5, 3.5, 2.5, 5.5]));
        let string_col: ArrayRef = Arc::new(StringArray::from(vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
            "banana".to_string(),
            "date".to_string(),
        ]));
        let bool_col: ArrayRef = Arc::new(BooleanArray::from(vec![true, false, true, false, true]));

        RecordBatch::try_new(
            schema,
            vec![
                int32_col,
                int64_col,
                float32_col,
                float64_col,
                string_col,
                bool_col,
            ],
        )
        .expect("test: RecordBatch construction should succeed")
    }

    /// Build a `StreamingArrowReader` without any file I/O. None of the
    /// predicate-evaluation methods under test read `self`'s fields (they
    /// operate purely on the `batch`/`column`/`value` arguments), so
    /// placeholder field values are sufficient here.
    fn dummy_reader() -> StreamingArrowReader {
        StreamingArrowReader {
            path: std::path::PathBuf::new(),
            config: StreamingArrowConfig::default(),
            current_batch: 0,
            total_batches: 0,
            schema: Arc::new(Schema::new(Vec::<Field>::new())),
        }
    }

    /// Collect a `BooleanArray` into `Vec<Option<bool>>` for easy assertions.
    fn mask_values(mask: &BooleanArray) -> Vec<Option<bool>> {
        mask.iter().collect()
    }

    #[test]
    fn test_evaluate_equals_int32_multi_row() {
        let reader = dummy_reader();
        let batch = build_predicate_test_batch();
        let mask = reader
            .evaluate_equals(&batch, "int32_col", &ArrowValue::Int32(20))
            .expect("test: evaluate_equals should succeed");
        assert_eq!(mask.len(), batch.num_rows());
        assert_eq!(
            mask_values(&mask),
            vec![
                Some(false),
                Some(true),
                Some(true),
                Some(false),
                Some(false)
            ]
        );
    }

    #[test]
    fn test_evaluate_equals_int64_multi_row() {
        let reader = dummy_reader();
        let batch = build_predicate_test_batch();
        let mask = reader
            .evaluate_equals(&batch, "int64_col", &ArrowValue::Int64(200))
            .expect("test: evaluate_equals should succeed");
        assert_eq!(mask.len(), batch.num_rows());
        assert_eq!(
            mask_values(&mask),
            vec![
                Some(false),
                Some(true),
                Some(false),
                Some(true),
                Some(false)
            ]
        );
    }

    #[test]
    fn test_evaluate_equals_float32_multi_row() {
        let reader = dummy_reader();
        let batch = build_predicate_test_batch();
        let mask = reader
            .evaluate_equals(&batch, "float32_col", &ArrowValue::Float32(2.5))
            .expect("test: evaluate_equals should succeed");
        assert_eq!(mask.len(), batch.num_rows());
        assert_eq!(
            mask_values(&mask),
            vec![
                Some(false),
                Some(true),
                Some(false),
                Some(true),
                Some(false)
            ]
        );
    }

    #[test]
    fn test_evaluate_equals_float64_multi_row() {
        let reader = dummy_reader();
        let batch = build_predicate_test_batch();
        let mask = reader
            .evaluate_equals(&batch, "float64_col", &ArrowValue::Float64(2.5))
            .expect("test: evaluate_equals should succeed");
        assert_eq!(mask.len(), batch.num_rows());
        assert_eq!(
            mask_values(&mask),
            vec![
                Some(false),
                Some(true),
                Some(false),
                Some(true),
                Some(false)
            ]
        );
    }

    #[test]
    fn test_evaluate_equals_string_multi_row() {
        let reader = dummy_reader();
        let batch = build_predicate_test_batch();
        let mask = reader
            .evaluate_equals(
                &batch,
                "string_col",
                &ArrowValue::String("banana".to_string()),
            )
            .expect("test: evaluate_equals should succeed");
        assert_eq!(mask.len(), batch.num_rows());
        assert_eq!(
            mask_values(&mask),
            vec![
                Some(false),
                Some(true),
                Some(false),
                Some(true),
                Some(false)
            ]
        );
    }

    #[test]
    fn test_evaluate_equals_bool_multi_row() {
        let reader = dummy_reader();
        let batch = build_predicate_test_batch();
        let mask = reader
            .evaluate_equals(&batch, "bool_col", &ArrowValue::Bool(true))
            .expect("test: evaluate_equals should succeed");
        assert_eq!(mask.len(), batch.num_rows());
        assert_eq!(
            mask_values(&mask),
            vec![Some(true), Some(false), Some(true), Some(false), Some(true)]
        );
    }

    #[test]
    fn test_evaluate_greater_than_int32_multi_row() {
        let reader = dummy_reader();
        let batch = build_predicate_test_batch();
        let mask = reader
            .evaluate_greater_than(&batch, "int32_col", &ArrowValue::Int32(20))
            .expect("test: evaluate_greater_than should succeed");
        assert_eq!(mask.len(), batch.num_rows());
        assert_eq!(
            mask_values(&mask),
            vec![
                Some(false),
                Some(false),
                Some(false),
                Some(true),
                Some(true)
            ]
        );
    }

    #[test]
    fn test_evaluate_greater_than_float64_multi_row() {
        let reader = dummy_reader();
        let batch = build_predicate_test_batch();
        let mask = reader
            .evaluate_greater_than(&batch, "float64_col", &ArrowValue::Float64(2.5))
            .expect("test: evaluate_greater_than should succeed");
        assert_eq!(mask.len(), batch.num_rows());
        assert_eq!(
            mask_values(&mask),
            vec![
                Some(false),
                Some(false),
                Some(true),
                Some(false),
                Some(true)
            ]
        );
    }

    #[test]
    fn test_evaluate_less_than_int32_multi_row() {
        let reader = dummy_reader();
        let batch = build_predicate_test_batch();
        let mask = reader
            .evaluate_less_than(&batch, "int32_col", &ArrowValue::Int32(20))
            .expect("test: evaluate_less_than should succeed");
        assert_eq!(mask.len(), batch.num_rows());
        assert_eq!(
            mask_values(&mask),
            vec![
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                Some(false)
            ]
        );
    }

    #[test]
    fn test_evaluate_less_than_float64_multi_row() {
        let reader = dummy_reader();
        let batch = build_predicate_test_batch();
        let mask = reader
            .evaluate_less_than(&batch, "float64_col", &ArrowValue::Float64(2.5))
            .expect("test: evaluate_less_than should succeed");
        assert_eq!(mask.len(), batch.num_rows());
        assert_eq!(
            mask_values(&mask),
            vec![
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                Some(false)
            ]
        );
    }

    #[test]
    fn test_evaluate_comparison_direct_call_matches_wrapper() {
        // Exercises the shared `evaluate_comparison` function directly
        // (bypassing the `evaluate_equals` wrapper method) to prove there
        // is only one code path backing both.
        let batch = build_predicate_test_batch();
        let mask = evaluate_comparison(
            &batch,
            "int32_col",
            &ArrowValue::Int32(20),
            eq,
            "test_evaluate_comparison_direct_call_matches_wrapper",
        )
        .expect("test: evaluate_comparison should succeed");
        assert_eq!(mask.len(), batch.num_rows());
        assert_eq!(
            mask_values(&mask),
            vec![
                Some(false),
                Some(true),
                Some(true),
                Some(false),
                Some(false)
            ]
        );
    }

    #[test]
    fn test_evaluate_equals_type_mismatch_errors() {
        let reader = dummy_reader();
        let batch = build_predicate_test_batch();
        // ArrowValue::Int32 against a Utf8 column must fail cleanly.
        let result = reader.evaluate_equals(&batch, "string_col", &ArrowValue::Int32(5));
        assert!(result.is_err());
    }

    #[test]
    fn test_in_predicate_non_empty_ors_equals_masks() {
        let reader = dummy_reader();
        let batch = build_predicate_test_batch();
        let predicate = ArrowPredicate::is_in(
            "int32_col",
            vec![ArrowValue::Int32(20), ArrowValue::Int32(40)],
        );
        let mask = reader
            .evaluate_predicate(&batch, &predicate)
            .expect("test: In predicate should succeed");
        assert_eq!(mask.len(), batch.num_rows());
        assert_eq!(
            mask_values(&mask),
            vec![Some(false), Some(true), Some(true), Some(false), Some(true)]
        );
    }

    #[test]
    fn test_in_predicate_empty_list_matches_nothing() {
        let reader = dummy_reader();
        let batch = build_predicate_test_batch();
        let predicate = ArrowPredicate::is_in("int32_col", vec![]);
        let mask = reader
            .evaluate_predicate(&batch, &predicate)
            .expect("test: empty In predicate should succeed");
        assert_eq!(mask.len(), batch.num_rows());
        assert_eq!(mask_values(&mask), vec![Some(false); batch.num_rows()]);
    }
}
