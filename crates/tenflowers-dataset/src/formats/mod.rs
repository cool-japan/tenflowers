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

#[cfg(feature = "audio")]
pub mod audio;
pub mod common;
pub mod csv;
#[cfg(feature = "hdf5")]
pub mod hdf5;
pub mod image;
#[cfg(feature = "serialize")]
pub mod json;
#[cfg(feature = "parquet")]
pub mod parquet;
pub mod text;
#[cfg(feature = "tfrecord")]
pub mod tfrecord;
#[cfg(feature = "webdataset")]
pub mod webdataset;
pub mod zarr;

// Re-export public types with disambiguation for FeatureType conflicts
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
pub use text::*;
#[cfg(feature = "tfrecord")]
pub use tfrecord::{Feature, FeatureType as TFRecordFeatureType, TFRecordConfig, TFRecordDataset};
#[cfg(feature = "webdataset")]
pub use webdataset::*;
pub use zarr::*;
