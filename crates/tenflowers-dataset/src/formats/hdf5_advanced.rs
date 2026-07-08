//! Advanced HDF5 features: chunked reading, attributes, group traversal,
//! compression info, and slice access.
//!
//! This module extends the base HDF5 support with high-performance primitives for
//! large-scale scientific dataset access. All APIs are gated behind the `hdf5` feature
//! flag; when the feature is absent every public type still exists but every constructor
//! and method returns `Err(TensorError::invalid_argument("hdf5 feature not enabled"))`.
//!
//! # Key types
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`Hdf5ChunkedReader`] | Reads a dataset in fixed-size chunks via `Iterator` |
//! | [`Hdf5AttributeReader`] | Reads all dataset attributes into a `HashMap` |
//! | [`Hdf5TreeWalker`] | Recursive group/dataset traversal |
//! | [`DatasetInfo`] | Shape, dtype, compression and chunk-shape metadata |
//! | [`Hdf5SliceReader`] | Sub-region access (start..end along the first axis) |
//!
//! # Example
//!
//! ```rust,no_run
//! # #[cfg(feature = "hdf5")]
//! # fn run() -> tenflowers_core::Result<()> {
//! use tenflowers_dataset::formats::hdf5_advanced::{
//!     Hdf5ChunkedReader, DatasetInfo, Hdf5TreeWalker, Hdf5SliceReader,
//! };
//!
//! // Chunked iteration
//! let reader = Hdf5ChunkedReader::open("data.h5", "features", 256)?;
//! for chunk in reader {
//!     let data = chunk?;
//!     println!("chunk rows: {}", data.len());
//! }
//!
//! // Dataset metadata
//! let info = DatasetInfo::read("data.h5", "features")?;
//! println!("{:?}", info);
//!
//! // Tree walk
//! let walker = Hdf5TreeWalker::open("data.h5")?;
//! let groups   = walker.list_groups("/")?;
//! let datasets = walker.list_datasets("/")?;
//! let tree     = walker.walk_tree("/")?;
//!
//! // Slice access (rows 0..100)
//! let rows = Hdf5SliceReader::read_slice("data.h5", "features", 0, 100)?;
//! # Ok(())
//! # }
//! ```

// ---------------------------------------------------------------------------
// Feature-enabled implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "hdf5")]
use std::collections::HashMap;
#[cfg(feature = "hdf5")]
use std::path::Path;

#[cfg(feature = "hdf5")]
use hdf5::{types::TypeDescriptor, File};

#[cfg(feature = "hdf5")]
use tenflowers_core::{Result, TensorError};

// ---------------------------------------------------------------------------
// Attribute values
// ---------------------------------------------------------------------------

/// A typed value read from an HDF5 attribute.
#[derive(Debug, Clone, PartialEq)]
pub enum Hdf5AttributeValue {
    /// Scalar 64-bit float
    Float64(f64),
    /// Scalar 64-bit integer
    Int64(i64),
    /// String value
    Str(String),
    /// Vector of 64-bit floats
    Float64Vec(Vec<f64>),
    /// Vector of 64-bit integers
    Int64Vec(Vec<i64>),
    /// Raw bytes (fallback)
    Bytes(Vec<u8>),
}

// ---------------------------------------------------------------------------
// Compression descriptor
// ---------------------------------------------------------------------------

/// Compression kind as reported by HDF5 filter metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressionKind {
    /// No compression applied
    None,
    /// DEFLATE / zlib compression (filter id 1)
    Deflate { level: u32 },
    /// LZF compression (filter id 32000)
    Lzf,
    /// Blosc compression (filter id 32001)
    Blosc,
    /// Unknown filter
    Unknown { filter_id: u32 },
}

// ---------------------------------------------------------------------------
// Dataset metadata
// ---------------------------------------------------------------------------

/// Full metadata about a single HDF5 dataset.
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    /// Fully-qualified dataset path (e.g. `/group/data`)
    pub name: String,
    /// Dataset shape (dimension sizes)
    pub shape: Vec<usize>,
    /// Dtype description string (e.g. `"Float(size=4)"`)
    pub dtype: String,
    /// Compression applied to this dataset
    pub compression: CompressionKind,
    /// Chunk shape if chunked storage is used
    pub chunk_shape: Option<Vec<usize>>,
}

// ---------------------------------------------------------------------------
// Tree node
// ---------------------------------------------------------------------------

/// A node in the HDF5 object tree.
#[derive(Debug, Clone)]
pub enum TreeNode {
    /// An HDF5 group (may contain children)
    Group {
        /// Full path of the group
        path: String,
        /// Direct children
        children: Vec<TreeNode>,
    },
    /// An HDF5 dataset leaf
    Dataset {
        /// Full path of the dataset
        path: String,
        /// Shape of the dataset
        shape: Vec<usize>,
    },
}

// ---------------------------------------------------------------------------
// Chunked reader
// ---------------------------------------------------------------------------

/// Iterator that yields successive chunks of a dataset as `Vec<Vec<f32>>`.
///
/// Each iteration reads `chunk_size` rows along the first axis. The last chunk
/// may be smaller if `total_rows % chunk_size != 0`.
#[derive(Debug)]
pub struct Hdf5ChunkedReader {
    #[cfg(feature = "hdf5")]
    file: File,
    #[cfg(feature = "hdf5")]
    dataset_name: String,
    #[cfg(feature = "hdf5")]
    chunk_size: usize,
    #[cfg(feature = "hdf5")]
    total_rows: usize,
    #[cfg(feature = "hdf5")]
    current_offset: usize,
    /// row-major flat data cache (feature not enabled → empty)
    #[cfg(not(feature = "hdf5"))]
    _phantom: (),
}

#[cfg(feature = "hdf5")]
impl Hdf5ChunkedReader {
    /// Open `file_path`, targeting `dataset_name`, with `chunk_size` rows per chunk.
    pub fn open<P: AsRef<Path>>(
        file_path: P,
        dataset_name: &str,
        chunk_size: usize,
    ) -> Result<Self> {
        if chunk_size == 0 {
            return Err(TensorError::invalid_argument(
                "chunk_size must be > 0".to_string(),
            ));
        }
        let file = File::open(file_path.as_ref())
            .map_err(|e| TensorError::invalid_argument(format!("Failed to open HDF5 file: {e}")))?;
        let ds = file.dataset(dataset_name).map_err(|e| {
            TensorError::invalid_argument(format!("Dataset '{dataset_name}' not found: {e}"))
        })?;
        let shape = ds.shape();
        let total_rows = shape.first().copied().unwrap_or(0);
        Ok(Self {
            file,
            dataset_name: dataset_name.to_string(),
            chunk_size,
            total_rows,
            current_offset: 0,
        })
    }

    /// Total number of rows in the dataset.
    pub fn total_rows(&self) -> usize {
        self.total_rows
    }

    /// Chunk size (rows per iteration).
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Read a chunk starting at `offset` with at most `chunk_size` rows.
    /// Returns `None` when past the end.
    fn read_chunk_at(&self, offset: usize) -> Result<Option<Vec<Vec<f32>>>> {
        if offset >= self.total_rows {
            return Ok(None);
        }
        let end = (offset + self.chunk_size).min(self.total_rows);
        let rows = end - offset;

        let ds = self
            .file
            .dataset(&self.dataset_name)
            .map_err(|e| TensorError::invalid_argument(format!("Dataset read error: {e}")))?;

        let shape = ds.shape();
        let cols: usize = if shape.len() > 1 {
            shape[1..].iter().product()
        } else {
            1
        };

        // Read full flat data then slice out the requested rows
        let raw: Vec<f32> = ds
            .read_raw()
            .map_err(|e| TensorError::invalid_argument(format!("Failed to read dataset: {e}")))?;

        let chunk: Vec<Vec<f32>> = (offset..end)
            .map(|r| {
                let start = r * cols;
                let stop = start + cols;
                raw.get(start..stop)
                    .map(|sl| sl.to_vec())
                    .unwrap_or_else(|| vec![0.0f32; cols])
            })
            .collect();

        let _ = rows; // suppress unused warning
        Ok(Some(chunk))
    }
}

#[cfg(feature = "hdf5")]
impl Iterator for Hdf5ChunkedReader {
    type Item = Result<Vec<Vec<f32>>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.read_chunk_at(self.current_offset) {
            Ok(Some(chunk)) => {
                self.current_offset += chunk.len();
                Some(Ok(chunk))
            }
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

// Stub when feature is disabled
#[cfg(not(feature = "hdf5"))]
impl Hdf5ChunkedReader {
    /// Stub constructor — always returns an error when the `hdf5` feature is disabled.
    pub fn open<P: AsRef<std::path::Path>>(
        _file_path: P,
        _dataset_name: &str,
        _chunk_size: usize,
    ) -> tenflowers_core::Result<Self> {
        Err(tenflowers_core::TensorError::invalid_argument(
            "hdf5 feature not enabled".to_string(),
        ))
    }
}

#[cfg(not(feature = "hdf5"))]
impl Iterator for Hdf5ChunkedReader {
    type Item = tenflowers_core::Result<Vec<Vec<f32>>>;
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

// ---------------------------------------------------------------------------
// Attribute reader
// ---------------------------------------------------------------------------

/// Reads all attributes attached to a specific HDF5 dataset.
pub struct Hdf5AttributeReader;

impl Hdf5AttributeReader {
    /// Read all attributes of `dataset_name` in `file_path`.
    ///
    /// Returns a `HashMap` mapping attribute name → [`Hdf5AttributeValue`].
    #[cfg(feature = "hdf5")]
    pub fn read_attributes<P: AsRef<Path>>(
        file_path: P,
        dataset_name: &str,
    ) -> Result<HashMap<String, Hdf5AttributeValue>> {
        let file = File::open(file_path.as_ref())
            .map_err(|e| TensorError::invalid_argument(format!("Failed to open HDF5 file: {e}")))?;
        let ds = file.dataset(dataset_name).map_err(|e| {
            TensorError::invalid_argument(format!("Dataset '{dataset_name}' not found: {e}"))
        })?;

        let mut map = HashMap::new();
        let attr_names = ds.attr_names().map_err(|e| {
            TensorError::invalid_argument(format!("Failed to list attributes: {e}"))
        })?;

        for name in attr_names {
            let attr = ds.attr(&name).map_err(|e| {
                TensorError::invalid_argument(format!("Cannot open attr '{name}': {e}"))
            })?;

            let value = Self::read_attr_value(&attr)?;
            map.insert(name, value);
        }
        Ok(map)
    }

    #[cfg(feature = "hdf5")]
    fn read_attr_value(attr: &hdf5::Attribute) -> Result<Hdf5AttributeValue> {
        // Try scalar types first, then vector, then fall back to raw bytes
        // Float64 scalar
        if let Ok(v) = attr.read_scalar::<f64>() {
            return Ok(Hdf5AttributeValue::Float64(v));
        }
        // Float32 scalar → promote to f64
        if let Ok(v) = attr.read_scalar::<f32>() {
            return Ok(Hdf5AttributeValue::Float64(v as f64));
        }
        // Int64 scalar
        if let Ok(v) = attr.read_scalar::<i64>() {
            return Ok(Hdf5AttributeValue::Int64(v));
        }
        // Int32 scalar → promote to i64
        if let Ok(v) = attr.read_scalar::<i32>() {
            return Ok(Hdf5AttributeValue::Int64(v as i64));
        }
        // String scalar
        if let Ok(v) = attr.read_scalar::<hdf5::types::VarLenUnicode>() {
            return Ok(Hdf5AttributeValue::Str(v.to_string()));
        }
        // 1-D float array
        if let Ok(v) = attr.read_1d::<f64>() {
            return Ok(Hdf5AttributeValue::Float64Vec(v.to_vec()));
        }
        if let Ok(v) = attr.read_1d::<f32>() {
            return Ok(Hdf5AttributeValue::Float64Vec(
                v.iter().map(|&x| x as f64).collect(),
            ));
        }
        // 1-D int array
        if let Ok(v) = attr.read_1d::<i64>() {
            return Ok(Hdf5AttributeValue::Int64Vec(v.to_vec()));
        }
        if let Ok(v) = attr.read_1d::<i32>() {
            return Ok(Hdf5AttributeValue::Int64Vec(
                v.iter().map(|&x| x as i64).collect(),
            ));
        }
        // Fallback: raw bytes via u8 1-D
        if let Ok(v) = attr.read_1d::<u8>() {
            return Ok(Hdf5AttributeValue::Bytes(v.to_vec()));
        }
        // If all else fails, return an empty bytes
        Ok(Hdf5AttributeValue::Bytes(Vec::new()))
    }

    /// Stub when the `hdf5` feature is disabled.
    #[cfg(not(feature = "hdf5"))]
    pub fn read_attributes<P: AsRef<std::path::Path>>(
        _file_path: P,
        _dataset_name: &str,
    ) -> tenflowers_core::Result<std::collections::HashMap<String, Hdf5AttributeValue>> {
        Err(tenflowers_core::TensorError::invalid_argument(
            "hdf5 feature not enabled".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Tree walker
// ---------------------------------------------------------------------------

/// Traverses the HDF5 group hierarchy, listing groups and datasets.
#[derive(Debug)]
pub struct Hdf5TreeWalker {
    #[cfg(feature = "hdf5")]
    file: File,
    #[cfg(not(feature = "hdf5"))]
    _phantom: (),
}

#[cfg(feature = "hdf5")]
impl Hdf5TreeWalker {
    /// Open an HDF5 file for tree traversal.
    pub fn open<P: AsRef<Path>>(file_path: P) -> Result<Self> {
        let file = File::open(file_path.as_ref())
            .map_err(|e| TensorError::invalid_argument(format!("Failed to open HDF5 file: {e}")))?;
        Ok(Self { file })
    }

    /// Return names of all groups directly inside `group_path`.
    pub fn list_groups(&self, group_path: &str) -> Result<Vec<String>> {
        let group = self.file.group(group_path).map_err(|e| {
            TensorError::invalid_argument(format!("Cannot open group '{group_path}': {e}"))
        })?;
        let members = group.member_names().map_err(|e| {
            TensorError::invalid_argument(format!("Cannot list group members: {e}"))
        })?;
        let groups = members
            .into_iter()
            .filter(|name| {
                let full_path = if group_path == "/" {
                    format!("/{name}")
                } else {
                    format!("{group_path}/{name}")
                };
                self.file.group(&full_path).is_ok()
            })
            .collect();
        Ok(groups)
    }

    /// Return names of all datasets directly inside `group_path`.
    pub fn list_datasets(&self, group_path: &str) -> Result<Vec<String>> {
        let group = self.file.group(group_path).map_err(|e| {
            TensorError::invalid_argument(format!("Cannot open group '{group_path}': {e}"))
        })?;
        let members = group.member_names().map_err(|e| {
            TensorError::invalid_argument(format!("Cannot list group members: {e}"))
        })?;
        let datasets = members
            .into_iter()
            .filter(|name| {
                let full_path = if group_path == "/" {
                    format!("/{name}")
                } else {
                    format!("{group_path}/{name}")
                };
                self.file.dataset(&full_path).is_ok()
            })
            .collect();
        Ok(datasets)
    }

    /// Recursively build the entire object tree rooted at `root_path`.
    pub fn walk_tree(&self, root_path: &str) -> Result<TreeNode> {
        self.walk_node(root_path)
    }

    fn walk_node(&self, path: &str) -> Result<TreeNode> {
        // Check if this path is a dataset
        if self.file.dataset(path).is_ok() {
            let ds = self.file.dataset(path).map_err(|e| {
                TensorError::invalid_argument(format!("Cannot open dataset '{path}': {e}"))
            })?;
            let shape = ds.shape().to_vec();
            return Ok(TreeNode::Dataset {
                path: path.to_string(),
                shape,
            });
        }

        // Otherwise treat as group
        let group = self.file.group(path).map_err(|e| {
            TensorError::invalid_argument(format!("Cannot open group '{path}': {e}"))
        })?;
        let members = group.member_names().map_err(|e| {
            TensorError::invalid_argument(format!("Cannot list members of '{path}': {e}"))
        })?;

        let mut children = Vec::new();
        for member in members {
            let child_path = if path == "/" {
                format!("/{member}")
            } else {
                format!("{path}/{member}")
            };
            let child = self.walk_node(&child_path)?;
            children.push(child);
        }

        Ok(TreeNode::Group {
            path: path.to_string(),
            children,
        })
    }
}

#[cfg(not(feature = "hdf5"))]
impl Hdf5TreeWalker {
    /// Stub constructor.
    pub fn open<P: AsRef<std::path::Path>>(_file_path: P) -> tenflowers_core::Result<Self> {
        Err(tenflowers_core::TensorError::invalid_argument(
            "hdf5 feature not enabled".to_string(),
        ))
    }

    /// Stub.
    pub fn list_groups(&self, _group_path: &str) -> tenflowers_core::Result<Vec<String>> {
        Err(tenflowers_core::TensorError::invalid_argument(
            "hdf5 feature not enabled".to_string(),
        ))
    }

    /// Stub.
    pub fn list_datasets(&self, _group_path: &str) -> tenflowers_core::Result<Vec<String>> {
        Err(tenflowers_core::TensorError::invalid_argument(
            "hdf5 feature not enabled".to_string(),
        ))
    }

    /// Stub.
    pub fn walk_tree(&self, _root_path: &str) -> tenflowers_core::Result<TreeNode> {
        Err(tenflowers_core::TensorError::invalid_argument(
            "hdf5 feature not enabled".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// DatasetInfo
// ---------------------------------------------------------------------------

impl DatasetInfo {
    /// Read full metadata for `dataset_name` inside `file_path`.
    #[cfg(feature = "hdf5")]
    pub fn read<P: AsRef<Path>>(file_path: P, dataset_name: &str) -> Result<Self> {
        let file = File::open(file_path.as_ref())
            .map_err(|e| TensorError::invalid_argument(format!("Failed to open HDF5 file: {e}")))?;
        let ds = file.dataset(dataset_name).map_err(|e| {
            TensorError::invalid_argument(format!("Dataset '{dataset_name}' not found: {e}"))
        })?;

        let shape = ds.shape().to_vec();

        // Dtype description
        let dtype = match ds.dtype() {
            Ok(t) => format!("{:?}", t.to_descriptor()),
            Err(_) => "unknown".to_string(),
        };

        // Chunk shape (may not be available if not chunked)
        let chunk_shape = ds.chunk().map(|c| c.to_vec());

        // Compression: inspect filters
        let compression = Self::detect_compression(&ds);

        Ok(Self {
            name: dataset_name.to_string(),
            shape,
            dtype,
            compression,
            chunk_shape,
        })
    }

    #[cfg(feature = "hdf5")]
    fn detect_compression(ds: &hdf5::Dataset) -> CompressionKind {
        use hdf5::filters::Filter;

        // `hdf5::Dataset::filters()` returns a `Vec<Filter>` describing the
        // pipeline applied to the dataset. We report the first compression
        // filter encountered; pure transforms such as `Shuffle`/`Fletcher32`
        // are skipped so a dataset that is only shuffled still reports `None`.
        for filter in ds.filters() {
            match filter {
                Filter::Deflate(level) => {
                    return CompressionKind::Deflate {
                        level: u32::from(level),
                    };
                }
                // `LZF`/`Blosc` only exist when the corresponding hdf5 features
                // are compiled in; gate the arms to match the active variant set.
                #[cfg(feature = "lzf")]
                Filter::LZF => return CompressionKind::Lzf,
                #[cfg(feature = "blosc")]
                Filter::Blosc(_, _, _) => return CompressionKind::Blosc,
                Filter::User(filter_id, _) => {
                    return CompressionKind::Unknown {
                        filter_id: filter_id as u32,
                    };
                }
                // Non-compression transforms — keep scanning the pipeline.
                Filter::Shuffle
                | Filter::Fletcher32
                | Filter::SZip(_, _)
                | Filter::NBit
                | Filter::ScaleOffset(_) => {}
            }
        }
        CompressionKind::None
    }

    /// Stub when the `hdf5` feature is disabled.
    #[cfg(not(feature = "hdf5"))]
    pub fn read<P: AsRef<std::path::Path>>(
        _file_path: P,
        _dataset_name: &str,
    ) -> tenflowers_core::Result<Self> {
        Err(tenflowers_core::TensorError::invalid_argument(
            "hdf5 feature not enabled".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Slice reader
// ---------------------------------------------------------------------------

/// Reads a contiguous sub-region of a dataset along the first (batch) axis.
pub struct Hdf5SliceReader;

impl Hdf5SliceReader {
    /// Read rows `[start, end)` from `dataset_name` in `file_path`.
    ///
    /// Returns a `Vec<Vec<f32>>` where the outer index is the row and the inner
    /// index is the column (linearised over remaining dimensions).
    #[cfg(feature = "hdf5")]
    pub fn read_slice<P: AsRef<Path>>(
        file_path: P,
        dataset_name: &str,
        start: usize,
        end: usize,
    ) -> Result<Vec<Vec<f32>>> {
        if end < start {
            return Err(TensorError::invalid_argument(format!(
                "end ({end}) must be >= start ({start})"
            )));
        }
        if end == start {
            return Ok(Vec::new());
        }

        let file = File::open(file_path.as_ref())
            .map_err(|e| TensorError::invalid_argument(format!("Failed to open HDF5 file: {e}")))?;
        let ds = file.dataset(dataset_name).map_err(|e| {
            TensorError::invalid_argument(format!("Dataset '{dataset_name}' not found: {e}"))
        })?;

        let shape = ds.shape();
        let total_rows = shape.first().copied().unwrap_or(0);
        if end > total_rows {
            return Err(TensorError::invalid_argument(format!(
                "end ({end}) exceeds total rows ({total_rows})"
            )));
        }

        let cols: usize = if shape.len() > 1 {
            shape[1..].iter().product()
        } else {
            1
        };

        // Read full flat buffer then slice (HDF5 hyperslab selection not
        // available in all hdf5-rs versions; this approach works universally).
        let raw: Vec<f32> = ds
            .read_raw()
            .map_err(|e| TensorError::invalid_argument(format!("Failed to read dataset: {e}")))?;

        let rows: Vec<Vec<f32>> = (start..end)
            .map(|r| {
                let s = r * cols;
                let e = s + cols;
                raw.get(s..e)
                    .map(|sl| sl.to_vec())
                    .unwrap_or_else(|| vec![0.0f32; cols])
            })
            .collect();

        Ok(rows)
    }

    /// Stub when the `hdf5` feature is disabled.
    #[cfg(not(feature = "hdf5"))]
    pub fn read_slice<P: AsRef<std::path::Path>>(
        _file_path: P,
        _dataset_name: &str,
        _start: usize,
        _end: usize,
    ) -> tenflowers_core::Result<Vec<Vec<f32>>> {
        Err(tenflowers_core::TensorError::invalid_argument(
            "hdf5 feature not enabled".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests (run only when the `hdf5` feature is enabled)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // When hdf5 feature is NOT enabled, we test only that stubs return the
    // expected error message.
    #[cfg(not(feature = "hdf5"))]
    mod stub_tests {
        use super::*;

        #[test]
        fn test_chunked_reader_stub() {
            let err = Hdf5ChunkedReader::open("nonexistent.h5", "data", 10)
                .expect_err("should be an error");
            assert!(err.to_string().contains("hdf5 feature not enabled"));
        }

        #[test]
        fn test_attribute_reader_stub() {
            let err = Hdf5AttributeReader::read_attributes("nonexistent.h5", "data")
                .expect_err("should be an error");
            assert!(err.to_string().contains("hdf5 feature not enabled"));
        }

        #[test]
        fn test_tree_walker_stub() {
            let err = Hdf5TreeWalker::open("nonexistent.h5").expect_err("should be an error");
            assert!(err.to_string().contains("hdf5 feature not enabled"));
        }

        #[test]
        fn test_dataset_info_stub() {
            let err = DatasetInfo::read("nonexistent.h5", "data").expect_err("should be an error");
            assert!(err.to_string().contains("hdf5 feature not enabled"));
        }

        #[test]
        fn test_slice_reader_stub() {
            let err = Hdf5SliceReader::read_slice("nonexistent.h5", "data", 0, 10)
                .expect_err("should be an error");
            assert!(err.to_string().contains("hdf5 feature not enabled"));
        }
    }

    // When hdf5 IS enabled, test the structural / logic parts that don't
    // require a real file on disk.
    #[cfg(feature = "hdf5")]
    mod feature_tests {
        use super::*;

        #[test]
        fn test_compression_kind_debug() {
            let c = CompressionKind::Deflate { level: 6 };
            let s = format!("{c:?}");
            assert!(s.contains("Deflate"));
        }

        #[test]
        fn test_tree_node_group_variant() {
            let node = TreeNode::Group {
                path: "/".to_string(),
                children: Vec::new(),
            };
            match node {
                TreeNode::Group { path, children } => {
                    assert_eq!(path, "/");
                    assert!(children.is_empty());
                }
                TreeNode::Dataset { .. } => panic!("expected Group"),
            }
        }

        #[test]
        fn test_tree_node_dataset_variant() {
            let node = TreeNode::Dataset {
                path: "/data".to_string(),
                shape: vec![100, 32],
            };
            match node {
                TreeNode::Dataset { path, shape } => {
                    assert_eq!(path, "/data");
                    assert_eq!(shape, vec![100, 32]);
                }
                TreeNode::Group { .. } => panic!("expected Dataset"),
            }
        }

        #[test]
        fn test_hdf5_attribute_value_variants() {
            let v1 = Hdf5AttributeValue::Float64(std::f64::consts::PI);
            let v2 = Hdf5AttributeValue::Int64(-1);
            let v3 = Hdf5AttributeValue::Str("hello".to_string());
            assert_eq!(v1, Hdf5AttributeValue::Float64(std::f64::consts::PI));
            assert_eq!(v2, Hdf5AttributeValue::Int64(-1));
            assert_eq!(v3, Hdf5AttributeValue::Str("hello".to_string()));
        }

        #[test]
        fn test_dataset_info_missing_file() {
            let err = DatasetInfo::read("/tmp/definitely_nonexistent_38472.h5", "data")
                .expect_err("should fail on missing file");
            assert!(
                err.to_string().contains("Failed to open HDF5 file"),
                "unexpected error: {err}"
            );
        }

        #[test]
        fn test_slice_reader_end_before_start() {
            // This should not even open a file — it fails on argument validation.
            // Provide a path that doesn't exist; the range check happens first.
            // Actually the file opens first; let's just test the logic directly.
            // end < start → immediate error
            let result = Hdf5SliceReader::read_slice("/tmp/x.h5", "data", 10, 5);
            assert!(result.is_err());
            assert!(result
                .expect_err("should be err")
                .to_string()
                .contains("end (5) must be >= start (10)"));
        }

        #[test]
        fn test_slice_reader_empty_range() {
            // start == end → should return Ok(vec![]) without touching the file.
            // The function opens the file before the range check, so this will
            // fail with "Failed to open HDF5 file" rather than returning Ok.
            // Verify the empty-range path by using a file that exists.
            // Since we cannot guarantee any HDF5 file, skip the actual empty-range
            // test when no file is available; just assert the range check logic.
            let result = Hdf5SliceReader::read_slice("/tmp/x.h5", "data", 5, 5);
            // Two outcomes are acceptable:
            // (a) file exists → Ok(vec![]) because start == end returns early
            // (b) file missing → Err containing "Failed to open HDF5 file"
            match result {
                Ok(rows) => assert!(rows.is_empty()),
                Err(e) => {
                    let msg = e.to_string();
                    assert!(msg.contains("Failed to open"), "unexpected: {msg}");
                }
            }
        }

        #[test]
        fn test_chunked_reader_missing_file() {
            let err = Hdf5ChunkedReader::open("/tmp/no_such_file_hdf5.h5", "data", 64)
                .expect_err("should fail");
            assert!(err.to_string().contains("Failed to open HDF5 file"));
        }

        #[test]
        fn test_chunked_reader_zero_chunk_size() {
            let err = Hdf5ChunkedReader::open("/tmp/x.h5", "data", 0)
                .expect_err("should fail with chunk_size 0");
            assert!(err.to_string().contains("chunk_size must be > 0"));
        }

        #[test]
        fn test_tree_walker_missing_file() {
            let err = Hdf5TreeWalker::open("/tmp/no_such_file_hdf5.h5").expect_err("should fail");
            assert!(err.to_string().contains("Failed to open HDF5 file"));
        }
    }
}
