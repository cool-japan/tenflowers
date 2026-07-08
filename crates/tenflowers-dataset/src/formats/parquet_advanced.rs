//! Advanced Parquet features: column projection, row-group iteration,
//! predicate pushdown, and schema inspection.
//!
//! All APIs are gated behind the `parquet` feature flag. When the feature is
//! absent the public types still exist but every method returns an error.
//!
//! # Key types
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`read_columns`] | Project-read specific columns as `Vec<(String, Vec<f64>)>` |
//! | [`ParquetRowGroupReader`] | Iterates row groups one at a time |
//! | [`read_filtered`] | Row-filtering with [`FilterPredicate`] |
//! | [`inspect_schema`] | Returns [`ParquetSchemaInfo`] |
//!
//! # Example
//!
//! ```rust,no_run
//! # #[cfg(feature = "parquet")]
//! # fn run() -> tenflowers_core::Result<()> {
//! use tenflowers_dataset::formats::parquet_advanced::{
//!     read_columns, FilterPredicate, read_filtered, inspect_schema, ParquetRowGroupReader,
//! };
//!
//! // Column projection
//! let cols = read_columns("data.parquet", &["age", "salary"])?;
//! for (name, values) in &cols {
//!     println!("{}: {} rows", name, values.len());
//! }
//!
//! // Row-group iterator
//! let reader = ParquetRowGroupReader::open("data.parquet")?;
//! for group in reader {
//!     let rg = group?;
//!     println!("{} rows in group", rg.num_rows);
//! }
//!
//! // Predicate pushdown
//! let rows = read_filtered("data.parquet", "salary", FilterPredicate::Gt(50_000.0))?;
//!
//! // Schema inspection
//! let schema = inspect_schema("data.parquet")?;
//! for col in &schema.columns {
//!     println!("{}: {:?}", col.name, col.dtype);
//! }
//! # Ok(())
//! # }
//! ```

// `HashMap` is a std type used by the always-available stub types
// (e.g. `ParquetSchemaInfo`), so it must not be feature-gated.
use std::collections::HashMap;
#[cfg(feature = "parquet")]
use std::path::Path;
#[cfg(feature = "parquet")]
use std::sync::Arc;

#[cfg(feature = "parquet")]
use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array};
#[cfg(feature = "parquet")]
use arrow::datatypes::{DataType as ArrowDataType, Schema};
#[cfg(feature = "parquet")]
use arrow::record_batch::RecordBatch;
#[cfg(feature = "parquet")]
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
#[cfg(feature = "parquet")]
use parquet::arrow::ProjectionMask;

#[cfg(feature = "parquet")]
use tenflowers_core::{Result, TensorError};

// ---------------------------------------------------------------------------
// FilterPredicate
// ---------------------------------------------------------------------------

/// A simple filter predicate for column-based row filtering.
///
/// Applied against the numeric (f64-cast) value of a column cell.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterPredicate {
    /// Keep rows where `value == target` (epsilon-exact)
    Eq(f64),
    /// Keep rows where `value < threshold`
    Lt(f64),
    /// Keep rows where `value > threshold`
    Gt(f64),
    /// Keep rows where `low <= value <= high`
    Between(f64, f64),
}

impl FilterPredicate {
    /// Test whether `value` passes this predicate.
    pub fn test(&self, value: f64) -> bool {
        match self {
            FilterPredicate::Eq(target) => (value - target).abs() < 1e-9,
            FilterPredicate::Lt(threshold) => value < *threshold,
            FilterPredicate::Gt(threshold) => value > *threshold,
            FilterPredicate::Between(low, high) => value >= *low && value <= *high,
        }
    }
}

// ---------------------------------------------------------------------------
// Column dtype summary
// ---------------------------------------------------------------------------

/// High-level dtype category for a Parquet column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnDtype {
    /// 32-bit float
    Float32,
    /// 64-bit float
    Float64,
    /// 32-bit integer
    Int32,
    /// 64-bit integer
    Int64,
    /// UTF-8 string
    Utf8,
    /// Boolean
    Boolean,
    /// Binary / bytes
    Binary,
    /// Other / unsupported
    Other(String),
}

// ---------------------------------------------------------------------------
// Schema info
// ---------------------------------------------------------------------------

/// Metadata about a single Parquet column.
#[derive(Debug, Clone)]
pub struct ColumnInfo {
    /// Column name
    pub name: String,
    /// High-level dtype category
    pub dtype: ColumnDtype,
    /// Arrow dtype string (verbose)
    pub arrow_dtype: String,
    /// Whether the column is nullable
    pub nullable: bool,
}

/// Schema summary returned by [`inspect_schema`].
#[derive(Debug, Clone)]
pub struct ParquetSchemaInfo {
    /// Per-column metadata, ordered by schema position
    pub columns: Vec<ColumnInfo>,
    /// Total number of rows across all row groups
    pub total_rows: usize,
    /// Number of row groups
    pub num_row_groups: usize,
    /// File size in bytes
    pub file_size: u64,
    /// Key/value metadata from the file footer
    pub key_value_metadata: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Row-group batch wrapper
// ---------------------------------------------------------------------------

/// A single row group materialized as a list of named float columns.
///
/// Non-numeric columns are skipped; numeric columns are cast to `f64`.
#[derive(Debug, Clone)]
pub struct RowGroupBatch {
    /// Row group index (0-based)
    pub index: usize,
    /// Number of rows in this group
    pub num_rows: usize,
    /// Column data: `(column_name, values)`
    pub columns: Vec<(String, Vec<f64>)>,
}

// ---------------------------------------------------------------------------
// Row-group iterator
// ---------------------------------------------------------------------------

/// Iterates over Parquet row groups one at a time without loading the whole
/// file into memory.
pub struct ParquetRowGroupReader {
    #[cfg(feature = "parquet")]
    path: std::path::PathBuf,
    #[cfg(feature = "parquet")]
    num_row_groups: usize,
    #[cfg(feature = "parquet")]
    current_group: usize,
    #[cfg(not(feature = "parquet"))]
    _phantom: (),
}

#[cfg(feature = "parquet")]
impl ParquetRowGroupReader {
    /// Open a Parquet file and prepare for row-group iteration.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();

        // Quick metadata peek to get group count
        let file = std::fs::File::open(&path_buf)
            .map_err(|e| TensorError::invalid_argument(format!("Cannot open file: {e}")))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| TensorError::invalid_argument(format!("Not a valid Parquet file: {e}")))?;
        let num_row_groups = builder.metadata().num_row_groups();

        Ok(Self {
            path: path_buf,
            num_row_groups,
            current_group: 0,
        })
    }

    /// Total number of row groups.
    pub fn num_row_groups(&self) -> usize {
        self.num_row_groups
    }

    /// Read a specific row group by index.
    fn read_group(&self, group_idx: usize) -> Result<RowGroupBatch> {
        let file = std::fs::File::open(&self.path)
            .map_err(|e| TensorError::invalid_argument(format!("Cannot open file: {e}")))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| TensorError::invalid_argument(format!("Cannot create reader: {e}")))?;

        let row_group_meta = builder.metadata().row_group(group_idx);
        let num_rows = row_group_meta.num_rows() as usize;

        // Select only this row group
        let reader = builder
            .with_row_groups(vec![group_idx])
            .build()
            .map_err(|e| TensorError::invalid_argument(format!("Cannot build reader: {e}")))?;

        let mut col_map: HashMap<String, Vec<f64>> = HashMap::new();
        let mut schema_ref: Option<Arc<Schema>> = None;

        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| TensorError::invalid_argument(format!("Batch read error: {e}")))?;
            if schema_ref.is_none() {
                schema_ref = Some(batch.schema());
            }
            collect_numeric_columns_into(&batch, &mut col_map)?;
        }

        // Preserve schema column order
        let columns = if let Some(schema) = schema_ref {
            schema
                .fields()
                .iter()
                .filter_map(|f| {
                    col_map
                        .remove(f.name())
                        .map(|vals| (f.name().clone(), vals))
                })
                .collect()
        } else {
            col_map.into_iter().collect()
        };

        Ok(RowGroupBatch {
            index: group_idx,
            num_rows,
            columns,
        })
    }
}

#[cfg(feature = "parquet")]
impl Iterator for ParquetRowGroupReader {
    type Item = Result<RowGroupBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_group >= self.num_row_groups {
            return None;
        }
        let idx = self.current_group;
        self.current_group += 1;
        Some(self.read_group(idx))
    }
}

#[cfg(not(feature = "parquet"))]
impl ParquetRowGroupReader {
    /// Stub constructor.
    pub fn open<P: AsRef<std::path::Path>>(_path: P) -> tenflowers_core::Result<Self> {
        Err(tenflowers_core::TensorError::invalid_argument(
            "parquet feature not enabled".to_string(),
        ))
    }

    /// Stub.
    pub fn num_row_groups(&self) -> usize {
        0
    }
}

#[cfg(not(feature = "parquet"))]
impl Iterator for ParquetRowGroupReader {
    type Item = tenflowers_core::Result<RowGroupBatch>;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

// ---------------------------------------------------------------------------
// Column projection
// ---------------------------------------------------------------------------

/// Read only the specified columns from a Parquet file.
///
/// Returns `Vec<(column_name, Vec<f64>)>` in the order given by `columns`.
/// Non-numeric source columns that appear in the projection list are returned
/// with an empty value vector and a warning-level error is silently skipped.
#[cfg(feature = "parquet")]
pub fn read_columns<P: AsRef<Path>>(path: P, columns: &[&str]) -> Result<Vec<(String, Vec<f64>)>> {
    if columns.is_empty() {
        return Ok(Vec::new());
    }
    let file = std::fs::File::open(path.as_ref())
        .map_err(|e| TensorError::invalid_argument(format!("Cannot open file: {e}")))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| TensorError::invalid_argument(format!("Not a valid Parquet file: {e}")))?;

    // Validate all requested columns exist in the schema
    let arrow_schema = builder.schema().clone();
    for col in columns {
        if arrow_schema.index_of(col).is_err() {
            return Err(TensorError::invalid_argument(format!(
                "Column '{col}' not found in schema"
            )));
        }
    }

    // Build projection mask using root-column indices from the parquet schema
    let parquet_schema = builder.parquet_schema();
    let projection_indices: Vec<usize> = columns
        .iter()
        .filter_map(|col| arrow_schema.index_of(col).ok())
        .collect();
    let mask = ProjectionMask::roots(parquet_schema, projection_indices);

    let reader = builder
        .with_projection(mask)
        .build()
        .map_err(|e| TensorError::invalid_argument(format!("Cannot build reader: {e}")))?;

    let mut col_map: HashMap<String, Vec<f64>> = HashMap::new();

    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| TensorError::invalid_argument(format!("Batch read error: {e}")))?;
        collect_numeric_columns_into(&batch, &mut col_map)?;
    }

    // Preserve requested column order
    let result: Vec<(String, Vec<f64>)> = columns
        .iter()
        .map(|&col| {
            let values = col_map.remove(col).unwrap_or_default();
            (col.to_string(), values)
        })
        .collect();

    Ok(result)
}

/// Stub — parquet feature disabled.
#[cfg(not(feature = "parquet"))]
pub fn read_columns<P: AsRef<std::path::Path>>(
    _path: P,
    _columns: &[&str],
) -> tenflowers_core::Result<Vec<(String, Vec<f64>)>> {
    Err(tenflowers_core::TensorError::invalid_argument(
        "parquet feature not enabled".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Predicate pushdown / row filtering
// ---------------------------------------------------------------------------

/// Read rows from `path` where `column` satisfies `predicate`.
///
/// Returns all matching rows as `Vec<HashMap<String, f64>>` where each map
/// contains the numeric columns of that row (non-numeric columns are omitted).
#[cfg(feature = "parquet")]
pub fn read_filtered<P: AsRef<Path>>(
    path: P,
    column: &str,
    predicate: FilterPredicate,
) -> Result<Vec<HashMap<String, f64>>> {
    let file = std::fs::File::open(path.as_ref())
        .map_err(|e| TensorError::invalid_argument(format!("Cannot open file: {e}")))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| TensorError::invalid_argument(format!("Not a valid Parquet file: {e}")))?;

    // Validate filter column
    if builder.schema().index_of(column).is_err() {
        return Err(TensorError::invalid_argument(format!(
            "Filter column '{column}' not found in schema"
        )));
    }

    let reader = builder
        .build()
        .map_err(|e| TensorError::invalid_argument(format!("Cannot build reader: {e}")))?;

    let mut result: Vec<HashMap<String, f64>> = Vec::new();

    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| TensorError::invalid_argument(format!("Batch read error: {e}")))?;

        // Get the filter column index
        let filter_col_idx = batch
            .schema()
            .index_of(column)
            .map_err(|e| TensorError::invalid_argument(format!("Column lookup error: {e}")))?;

        let num_rows = batch.num_rows();

        // Extract all numeric columns as f64 vectors
        let mut col_data: HashMap<String, Vec<f64>> = HashMap::new();
        let batch_schema = batch.schema();
        for i in 0..batch.num_columns() {
            let field_name = batch_schema.field(i).name().clone();
            if let Ok(vals) = array_to_f64_vec(batch.column(i).as_ref()) {
                col_data.insert(field_name, vals);
            }
        }

        // Determine filter column values
        let filter_values = col_data
            .get(column)
            .ok_or_else(|| {
                TensorError::invalid_argument(format!("Filter column '{column}' is not numeric"))
            })?
            .clone();

        // Emit matching rows
        for row_idx in 0..num_rows {
            let filter_val = filter_values.get(row_idx).copied().unwrap_or(f64::NAN);
            if predicate.test(filter_val) {
                let _ = filter_col_idx; // suppress unused warning
                let row: HashMap<String, f64> = col_data
                    .iter()
                    .filter_map(|(name, vals)| {
                        vals.get(row_idx).copied().map(|v| (name.clone(), v))
                    })
                    .collect();
                result.push(row);
            }
        }
    }

    Ok(result)
}

/// Stub — parquet feature disabled.
#[cfg(not(feature = "parquet"))]
pub fn read_filtered<P: AsRef<std::path::Path>>(
    _path: P,
    _column: &str,
    _predicate: FilterPredicate,
) -> tenflowers_core::Result<Vec<std::collections::HashMap<String, f64>>> {
    Err(tenflowers_core::TensorError::invalid_argument(
        "parquet feature not enabled".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Schema inspection
// ---------------------------------------------------------------------------

/// Inspect the schema and file metadata of a Parquet file.
#[cfg(feature = "parquet")]
pub fn inspect_schema<P: AsRef<Path>>(path: P) -> Result<ParquetSchemaInfo> {
    let file = std::fs::File::open(path.as_ref())
        .map_err(|e| TensorError::invalid_argument(format!("Cannot open file: {e}")))?;
    let file_size = std::fs::metadata(path.as_ref())
        .map_err(|e| TensorError::invalid_argument(format!("Cannot read metadata: {e}")))?
        .len();

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| TensorError::invalid_argument(format!("Not a valid Parquet file: {e}")))?;

    let parquet_meta = builder.metadata().clone();
    let arrow_schema = builder.schema().clone();

    let num_row_groups = parquet_meta.num_row_groups();
    let total_rows: usize = (0..num_row_groups)
        .map(|i| parquet_meta.row_group(i).num_rows() as usize)
        .sum();

    // Key/value metadata from file footer
    let key_value_metadata: HashMap<String, String> = parquet_meta
        .file_metadata()
        .key_value_metadata()
        .map(|kvs| {
            kvs.iter()
                .map(|kv| (kv.key.clone(), kv.value.clone().unwrap_or_default()))
                .collect()
        })
        .unwrap_or_default();

    // Column info
    let columns: Vec<ColumnInfo> = arrow_schema
        .fields()
        .iter()
        .map(|field| {
            let dtype = arrow_dtype_to_column_dtype(field.data_type());
            ColumnInfo {
                name: field.name().clone(),
                dtype,
                arrow_dtype: format!("{:?}", field.data_type()),
                nullable: field.is_nullable(),
            }
        })
        .collect();

    Ok(ParquetSchemaInfo {
        columns,
        total_rows,
        num_row_groups,
        file_size,
        key_value_metadata,
    })
}

/// Stub — parquet feature disabled.
#[cfg(not(feature = "parquet"))]
pub fn inspect_schema<P: AsRef<std::path::Path>>(
    _path: P,
) -> tenflowers_core::Result<ParquetSchemaInfo> {
    Err(tenflowers_core::TensorError::invalid_argument(
        "parquet feature not enabled".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert Arrow dtype → ColumnDtype summary.
#[cfg(feature = "parquet")]
fn arrow_dtype_to_column_dtype(dt: &ArrowDataType) -> ColumnDtype {
    match dt {
        ArrowDataType::Float32 => ColumnDtype::Float32,
        ArrowDataType::Float64 => ColumnDtype::Float64,
        ArrowDataType::Int32 => ColumnDtype::Int32,
        ArrowDataType::Int64 => ColumnDtype::Int64,
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 => ColumnDtype::Utf8,
        ArrowDataType::Boolean => ColumnDtype::Boolean,
        ArrowDataType::Binary | ArrowDataType::LargeBinary => ColumnDtype::Binary,
        other => ColumnDtype::Other(format!("{other:?}")),
    }
}

/// Convert a single Arrow array to a `Vec<f64>` for numeric types.
#[cfg(feature = "parquet")]
fn array_to_f64_vec(array: &dyn Array) -> Result<Vec<f64>> {
    match array.data_type() {
        ArrowDataType::Float32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    TensorError::invalid_argument("Float32Array downcast failed".to_string())
                })?;
            Ok(arr.values().iter().map(|&v| v as f64).collect())
        }
        ArrowDataType::Float64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    TensorError::invalid_argument("Float64Array downcast failed".to_string())
                })?;
            Ok(arr.values().to_vec())
        }
        ArrowDataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
                TensorError::invalid_argument("Int32Array downcast failed".to_string())
            })?;
            Ok(arr.values().iter().map(|&v| v as f64).collect())
        }
        ArrowDataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                TensorError::invalid_argument("Int64Array downcast failed".to_string())
            })?;
            Ok(arr.values().iter().map(|&v| v as f64).collect())
        }
        _ => Err(TensorError::invalid_argument(
            "unsupported numeric type".to_string(),
        )),
    }
}

/// Accumulate numeric columns from a `RecordBatch` into a running `HashMap`.
#[cfg(feature = "parquet")]
fn collect_numeric_columns_into(
    batch: &RecordBatch,
    col_map: &mut HashMap<String, Vec<f64>>,
) -> Result<()> {
    let schema = batch.schema();
    for i in 0..batch.num_columns() {
        let field_name = schema.field(i).name().clone();
        if let Ok(vals) = array_to_f64_vec(batch.column(i).as_ref()) {
            col_map.entry(field_name).or_default().extend(vals);
        }
        // Non-numeric columns are silently skipped
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Predicate logic tests (no feature gate required) ---

    #[test]
    fn test_filter_predicate_eq() {
        let p = FilterPredicate::Eq(std::f64::consts::PI);
        assert!(p.test(std::f64::consts::PI));
        assert!(!p.test(3.15));
        // Within epsilon
        assert!(p.test(std::f64::consts::PI + 1e-10));
    }

    #[test]
    fn test_filter_predicate_lt() {
        let p = FilterPredicate::Lt(10.0);
        assert!(p.test(9.99));
        assert!(!p.test(10.0));
        assert!(!p.test(100.0));
    }

    #[test]
    fn test_filter_predicate_gt() {
        let p = FilterPredicate::Gt(5.0);
        assert!(p.test(5.1));
        assert!(!p.test(5.0));
        assert!(!p.test(0.0));
    }

    #[test]
    fn test_filter_predicate_between() {
        let p = FilterPredicate::Between(1.0, 5.0);
        assert!(p.test(1.0));
        assert!(p.test(3.0));
        assert!(p.test(5.0));
        assert!(!p.test(0.999));
        assert!(!p.test(5.001));
    }

    #[test]
    fn test_column_dtype_debug() {
        let d = ColumnDtype::Float64;
        assert_eq!(format!("{d:?}"), "Float64");
    }

    // --- Stub tests when parquet feature is NOT enabled ---
    #[cfg(not(feature = "parquet"))]
    mod stub_tests {
        use super::*;
        use std::collections::HashMap;

        #[test]
        fn test_read_columns_stub() {
            let err = read_columns("data.parquet", &["col1"]).expect_err("should be error");
            assert!(err.to_string().contains("parquet feature not enabled"));
        }

        #[test]
        fn test_row_group_reader_stub() {
            // The stub reader type is not `Debug` (its feature-enabled variant
            // wraps an arrow reader), so match on the result instead of using
            // `expect_err`, which would require `Ok: Debug`.
            match ParquetRowGroupReader::open("data.parquet") {
                Ok(_) => panic!("stub should return an error"),
                Err(err) => {
                    assert!(err.to_string().contains("parquet feature not enabled"))
                }
            }
        }

        #[test]
        fn test_read_filtered_stub() {
            let err = read_filtered("data.parquet", "col", FilterPredicate::Gt(0.0))
                .expect_err("should be error");
            assert!(err.to_string().contains("parquet feature not enabled"));
        }

        #[test]
        fn test_inspect_schema_stub() {
            let err = inspect_schema("data.parquet").expect_err("should be error");
            assert!(err.to_string().contains("parquet feature not enabled"));
        }
    }

    // --- Feature-enabled tests ---
    #[cfg(feature = "parquet")]
    mod feature_tests {
        use super::*;
        use arrow::array::{Float64Array, Int32Array};
        use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::io::Write;
        use std::sync::Arc;

        /// Write a small Parquet file to a temp path and return the path.
        fn write_test_parquet() -> (tempfile::NamedTempFile, std::path::PathBuf) {
            let tmp = tempfile::NamedTempFile::new().expect("tmp file");
            let path = tmp.path().to_path_buf();

            let schema = Arc::new(Schema::new(vec![
                Field::new("x", ArrowDataType::Float64, false),
                Field::new("y", ArrowDataType::Float64, false),
                Field::new("label", ArrowDataType::Int32, false),
            ]));

            let x_arr = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            let y_arr = Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
            let l_arr = Int32Array::from(vec![0, 1, 0, 1, 0]);

            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(x_arr), Arc::new(y_arr), Arc::new(l_arr)],
            )
            .expect("record batch");

            let file = std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .expect("open for write");
            let mut writer = ArrowWriter::try_new(file, schema, None).expect("writer");
            writer.write(&batch).expect("write batch");
            writer.close().expect("close");

            (tmp, path)
        }

        #[test]
        fn test_inspect_schema_returns_columns() {
            let (_tmp, path) = write_test_parquet();
            let info = inspect_schema(&path).expect("inspect_schema");
            assert_eq!(info.columns.len(), 3);
            assert_eq!(info.columns[0].name, "x");
            assert_eq!(info.columns[0].dtype, ColumnDtype::Float64);
            assert_eq!(info.total_rows, 5);
            assert_eq!(info.num_row_groups, 1);
        }

        #[test]
        fn test_read_columns_projection() {
            let (_tmp, path) = write_test_parquet();
            let cols = read_columns(&path, &["x", "label"]).expect("read_columns");
            assert_eq!(cols.len(), 2);
            assert_eq!(cols[0].0, "x");
            assert_eq!(cols[0].1.len(), 5);
            assert_eq!(cols[1].0, "label");
        }

        #[test]
        fn test_read_columns_missing_column() {
            let (_tmp, path) = write_test_parquet();
            let err =
                read_columns(&path, &["nonexistent"]).expect_err("should fail for missing column");
            assert!(err.to_string().contains("not found in schema"));
        }

        #[test]
        fn test_read_filtered_gt() {
            let (_tmp, path) = write_test_parquet();
            let rows = read_filtered(&path, "x", FilterPredicate::Gt(3.0)).expect("read_filtered");
            // rows where x > 3 → x=4.0, x=5.0 → 2 rows
            assert_eq!(rows.len(), 2);
        }

        #[test]
        fn test_read_filtered_between() {
            let (_tmp, path) = write_test_parquet();
            let rows = read_filtered(&path, "y", FilterPredicate::Between(20.0, 40.0))
                .expect("read_filtered");
            // y=20,30,40 → 3 rows
            assert_eq!(rows.len(), 3);
        }

        #[test]
        fn test_row_group_reader_iterates() {
            let (_tmp, path) = write_test_parquet();
            let reader = ParquetRowGroupReader::open(&path).expect("open");
            assert_eq!(reader.num_row_groups(), 1);
            let groups: Vec<_> = reader.collect();
            assert_eq!(groups.len(), 1);
            let rg = groups.into_iter().next().expect("one group").expect("ok");
            assert_eq!(rg.index, 0);
            assert_eq!(rg.num_rows, 5);
        }

        #[test]
        fn test_arrow_dtype_to_column_dtype() {
            assert_eq!(
                arrow_dtype_to_column_dtype(&ArrowDataType::Float32),
                ColumnDtype::Float32
            );
            assert_eq!(
                arrow_dtype_to_column_dtype(&ArrowDataType::Utf8),
                ColumnDtype::Utf8
            );
            assert_eq!(
                arrow_dtype_to_column_dtype(&ArrowDataType::Boolean),
                ColumnDtype::Boolean
            );
        }

        #[test]
        fn test_filter_predicate_eq_on_real_data() {
            let (_tmp, path) = write_test_parquet();
            // label == 1 → rows 1 and 3 (0-indexed) → 2 rows
            let rows =
                read_filtered(&path, "label", FilterPredicate::Eq(1.0)).expect("read_filtered");
            assert_eq!(rows.len(), 2);
        }
    }
}
