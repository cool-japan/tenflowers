//! File format support for datasets
//!
//! This module provides loading capabilities for common data formats including
//! CSV files, image directories, WebDataset, Zarr arrays, JSON/JSONL, text files, and more.
//!
//! The module is organized by format type:
//! - `audio`: Audio format support for machine learning with audio files
//! - `csv`: CSV and delimited text file support
//! - `image`: Image directory and folder structure support
//! - `webdataset`: WebDataset format for streaming large datasets
//! - `zarr`: Zarr multidimensional array format for scientific datasets
//! - `json`: JSON and JSON Lines format support for structured data
//! - `text`: Text dataset format support for NLP tasks
//! - `parquet`: Apache Parquet columnar format support for big data workflows
//! - `hdf5`: HDF5 hierarchical format support for scientific datasets
//! - `tfrecord`: TensorFlow TFRecord format support for ML training data
//! - `common`: Shared types and utilities used across formats
//! - `unified_reader`: Unified format reader abstraction layer
//! - `cross_format`: Cross-format operations and utilities
//! - `schema_validator`: Enhanced schema validation system

#[cfg(feature = "parquet")]
pub mod arrow;
#[cfg(feature = "parquet")]
pub mod arrow_advanced;
#[cfg(feature = "audio")]
pub mod audio;
#[cfg(feature = "compression")]
pub mod blosc;
pub mod common;
pub mod cross_format;
pub mod csv;
#[cfg(feature = "csv_format")]
pub mod csv_format_reader;
#[cfg(feature = "hdf5")]
pub mod hdf5;
pub mod hdf5_advanced;
#[cfg(feature = "hdf5")]
pub mod hdf5_format_reader;
pub mod image;
#[cfg(feature = "serialize")]
pub mod json;
#[cfg(feature = "serialize")]
pub mod json_format_reader;
#[cfg(feature = "parquet")]
pub mod parquet;
pub mod parquet_advanced;
#[cfg(feature = "parquet")]
pub mod parquet_format_reader;
pub mod registry;
pub mod schema_validator;
pub mod text;
#[cfg(feature = "tfrecord")]
pub mod tfrecord;
pub mod tfrecord_advanced;
pub mod unified_reader;
#[cfg(feature = "webdataset")]
pub mod webdataset;
pub mod zarr;

// Re-export public types with disambiguation for FeatureType conflicts
#[cfg(feature = "parquet")]
pub use arrow::{
    ArrowArrayExt, ArrowConfig, ArrowDataset, ArrowDatasetBuilder, ArrowFormatFactory,
    ArrowFormatReader, ArrowTensorView,
};
#[cfg(feature = "parquet")]
pub use arrow_advanced::{
    ArrowBuffer, ArrowPredicate, ArrowStatistics, ArrowValue, StreamingArrowConfig,
    StreamingArrowReader,
};
#[cfg(feature = "audio")]
pub use audio::{AudioConfig, AudioDataset, AudioLabelStrategy, FeatureType as AudioFeatureType};
pub use common::*;
pub use csv::*;
#[cfg(feature = "hdf5")]
pub use hdf5::*;
pub use image::*;
#[cfg(feature = "serialize")]
pub use json::*;
#[cfg(feature = "parquet")]
pub use parquet::*;
pub use registry::{global, register_format_factory, FormatInfo, GlobalFormatRegistry};
pub use text::*;
#[cfg(feature = "tfrecord")]
pub use tfrecord::{Feature, FeatureType as TFRecordFeatureType, TFRecordConfig, TFRecordDataset};
pub use unified_reader::{
    detect_format_from_extension, read_magic_bytes, DataType as UnifiedDataType, DetectionMethod,
    FieldInfo, FormatDetection, FormatFactory, FormatMetadata, FormatReader, FormatReaderBuilder,
    FormatRegistry, FormatSample,
};
#[cfg(feature = "webdataset")]
pub use webdataset::*;
pub use zarr::*;

// Re-export cross-format utilities
pub use cross_format::{
    CrossFormatConcatenation, CrossFormatIterator, FormatConverter, SchemaCompatibility,
    UnifiedBatchReader,
};

// Re-export schema validation
pub use schema_validator::{
    FieldDiff, SchemaValidator, ValidationConfig, ValidationError, ValidationErrorCategory,
    ValidationPolicy, ValidationReport as SchemaValidationReport, ValidationResult,
    ValidationWarning,
};

// Re-export format readers
#[cfg(feature = "csv_format")]
pub use csv_format_reader::{CsvFormatFactory, CsvFormatReader};
#[cfg(feature = "hdf5")]
pub use hdf5_format_reader::{HDF5FormatFactory, HDF5FormatReader};
#[cfg(feature = "serialize")]
pub use json_format_reader::{JsonFormatFactory, JsonFormatReader};
#[cfg(feature = "parquet")]
pub use parquet_format_reader::{ParquetFormatFactory, ParquetFormatReader};

// Re-export advanced HDF5 types (stubs always available)
pub use hdf5_advanced::{
    CompressionKind, DatasetInfo, Hdf5AttributeReader, Hdf5AttributeValue, Hdf5ChunkedReader,
    Hdf5SliceReader, Hdf5TreeWalker, TreeNode,
};

// Re-export advanced Parquet types (stubs always available)
pub use parquet_advanced::{
    inspect_schema, read_columns, read_filtered, ColumnDtype, ColumnInfo, FilterPredicate,
    ParquetRowGroupReader, ParquetSchemaInfo, RowGroupBatch,
};

// Re-export advanced TFRecord types (stubs always available)
#[cfg(feature = "tfrecord")]
pub use tfrecord_advanced::parse_feature_proto;
// `SequenceExample`/`SequenceStep` embed the `Feature` type from the base
// `tfrecord` module, so they only exist when that feature is enabled.
pub use tfrecord_advanced::{
    masked_crc32, verify_masked_crc32, RawTfRecord, TfRecordRawReader, TfRecordSequenceReader,
};
#[cfg(feature = "tfrecord")]
pub use tfrecord_advanced::{SequenceExample, SequenceStep};
