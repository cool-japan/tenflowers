//! Zarr format support for large-scale multidimensional array datasets
//!
//! This module provides support for the Zarr format, which is particularly useful for:
//! - Large multidimensional scientific datasets  
//! - Cloud-native storage with efficient chunking
//! - Parallel I/O operations
//! - Medical imaging, satellite imagery, and other scientific ML applications

use crate::Dataset;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tenflowers_core::{Result, Tensor, TensorError};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

// Cloud-native and compression support
#[cfg(feature = "compression")]
use oxiarc_archive::{GzipReader, Lz4Reader, ZstdReader};
#[cfg(feature = "cloud")]
/// Cloud storage backend types
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum CloudBackend {
    /// Local filesystem (default)
    Local,
    /// Amazon S3 compatible storage
    #[cfg(feature = "cloud")]
    S3 {
        bucket: String,
        region: String,
        endpoint: Option<String>,
    },
    /// Google Cloud Storage
    #[cfg(feature = "cloud")]
    Gcs { bucket: String },
    /// Azure Blob Storage
    #[cfg(feature = "cloud")]
    Azure { account: String, container: String },
}

#[cfg(feature = "cloud")]
impl Default for CloudBackend {
    fn default() -> Self {
        Self::Local
    }
}

/// Compression algorithms supported for Zarr format
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum ZarrCompressionType {
    None,
    #[cfg(feature = "compression")]
    Gzip,
    #[cfg(feature = "compression")]
    Blosc,
    #[cfg(feature = "compression")]
    Lz4,
    #[cfg(feature = "compression")]
    Zstd,
}

impl Default for ZarrCompressionType {
    fn default() -> Self {
        Self::None
    }
}

/// Configuration for Zarr dataset loading
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ZarrConfig {
    /// Path to the Zarr array directory
    pub array_path: PathBuf,
    /// Optional path to labels array
    pub labels_path: Option<PathBuf>,
    /// Chunk cache size in bytes
    pub chunk_cache_size: usize,
    /// Whether to load data lazily
    pub lazy_loading: bool,
    /// Maximum number of chunks to load in parallel
    pub max_parallel_chunks: usize,
    /// Whether to use memory mapping for chunks
    pub use_memory_mapping: bool,
    /// Dimension order for tensor conversion (e.g., "NCHW" for images)
    pub dimension_order: Option<String>,
    /// Cloud storage backend configuration
    #[cfg(feature = "cloud")]
    pub cloud_backend: CloudBackend,
    /// Compression type for chunk storage
    pub compression: ZarrCompressionType,
    /// Enable async I/O operations
    pub async_io: bool,
    /// Connection timeout for cloud backends (in seconds)
    pub connection_timeout: u64,
    /// Retry attempts for failed operations
    pub retry_attempts: usize,
}

impl Default for ZarrConfig {
    fn default() -> Self {
        Self {
            array_path: PathBuf::new(),
            labels_path: None,
            chunk_cache_size: 100_000_000, // 100MB
            lazy_loading: true,
            max_parallel_chunks: 4,
            use_memory_mapping: true,
            dimension_order: None,
            #[cfg(feature = "cloud")]
            cloud_backend: CloudBackend::default(),
            compression: ZarrCompressionType::default(),
            async_io: false,
            connection_timeout: 30,
            retry_attempts: 3,
        }
    }
}

/// Zarr array metadata
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ZarrArrayInfo {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub chunks: Vec<usize>,
    pub compressor: Option<String>,
    pub fill_value: Option<f64>,
    pub order: String, // 'C' or 'F'
    pub zarr_format: u32,
}

/// Builder for Zarr datasets
#[derive(Debug, Clone)]
pub struct ZarrDatasetBuilder<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + std::str::FromStr
        + Send
        + Sync
        + 'static,
{
    config: ZarrConfig,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> ZarrDatasetBuilder<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + std::str::FromStr
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::cast::NumCast,
{
    /// Create a new Zarr dataset builder
    pub fn new() -> Self {
        Self {
            config: ZarrConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the path to the Zarr array
    pub fn array_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.config.array_path = path.as_ref().to_path_buf();
        self
    }

    /// Set the path to the labels array
    pub fn labels_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.config.labels_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set chunk cache size in bytes
    pub fn chunk_cache_size(mut self, size: usize) -> Self {
        self.config.chunk_cache_size = size;
        self
    }

    /// Enable or disable lazy loading
    pub fn lazy_loading(mut self, enabled: bool) -> Self {
        self.config.lazy_loading = enabled;
        self
    }

    /// Set maximum parallel chunks
    pub fn max_parallel_chunks(mut self, count: usize) -> Self {
        self.config.max_parallel_chunks = count;
        self
    }

    /// Enable or disable memory mapping
    pub fn use_memory_mapping(mut self, enabled: bool) -> Self {
        self.config.use_memory_mapping = enabled;
        self
    }

    /// Set dimension order for tensor conversion
    pub fn dimension_order<S: AsRef<str>>(mut self, order: S) -> Self {
        self.config.dimension_order = Some(order.as_ref().to_string());
        self
    }

    /// Set cloud backend configuration
    #[cfg(feature = "cloud")]
    pub fn cloud_backend(mut self, backend: CloudBackend) -> Self {
        self.config.cloud_backend = backend;
        self
    }

    /// Set compression type
    pub fn compression(mut self, compression: ZarrCompressionType) -> Self {
        self.config.compression = compression;
        self
    }

    /// Enable async I/O operations
    pub fn async_io(mut self, enabled: bool) -> Self {
        self.config.async_io = enabled;
        self
    }

    /// Set connection timeout for cloud backends
    pub fn connection_timeout(mut self, timeout: u64) -> Self {
        self.config.connection_timeout = timeout;
        self
    }

    /// Set retry attempts for failed operations
    pub fn retry_attempts(mut self, attempts: usize) -> Self {
        self.config.retry_attempts = attempts;
        self
    }

    /// Build the Zarr dataset
    pub fn build(self) -> Result<ZarrDataset<T>> {
        ZarrDataset::from_config(self.config)
    }
}

impl<T> Default for ZarrDatasetBuilder<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + std::str::FromStr
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::cast::NumCast,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Zarr dataset implementation
#[derive(Debug, Clone)]
pub struct ZarrDataset<T>
where
    T: Clone + Default + scirs2_core::numeric::Zero + Send + Sync + 'static,
{
    config: ZarrConfig,
    array_info: ZarrArrayInfo,
    labels_info: Option<ZarrArrayInfo>,
    chunk_cache: Arc<std::sync::Mutex<HashMap<Vec<usize>, Vec<T>>>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> ZarrDataset<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + std::str::FromStr
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::cast::NumCast,
{
    /// Create a new Zarr dataset from configuration
    pub fn from_config(config: ZarrConfig) -> Result<Self> {
        // Read array metadata
        let array_info = Self::read_array_metadata(&config.array_path)?;

        // Read labels metadata if provided
        let labels_info = if let Some(ref labels_path) = config.labels_path {
            Some(Self::read_array_metadata(labels_path)?)
        } else {
            None
        };

        let chunk_cache = Arc::new(std::sync::Mutex::new(HashMap::new()));

        Ok(Self {
            config,
            array_info,
            labels_info,
            chunk_cache,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Read Zarr array metadata from .zarray file
    fn read_array_metadata(array_path: &Path) -> Result<ZarrArrayInfo> {
        let zarray_path = array_path.join(".zarray");

        if !zarray_path.exists() {
            return Err(TensorError::invalid_argument(format!(
                "Zarr metadata file not found: {zarray_path:?}"
            )));
        }

        let metadata_content = fs::read_to_string(&zarray_path).map_err(|e| {
            TensorError::invalid_argument(format!("Failed to read Zarr metadata: {e}"))
        })?;

        // Parse JSON metadata (simplified - in a real implementation, use serde_json)
        Self::parse_zarr_metadata(&metadata_content)
    }

    /// Parse Zarr metadata JSON with enhanced parsing logic
    fn parse_zarr_metadata(content: &str) -> Result<ZarrArrayInfo> {
        // Enhanced JSON parsing logic - still simplified but more robust
        // In a production environment, you would use serde_json for proper JSON parsing

        // Extract key fields using improved string parsing
        let shape = if content.contains("\"shape\"") {
            // Try to extract shape array with basic parsing
            if let Some(start) = content.find("\"shape\"") {
                if let Some(array_start) = content[start..].find('[') {
                    if let Some(array_end) = content[start + array_start..].find(']') {
                        let array_content =
                            &content[start + array_start + 1..start + array_start + array_end];
                        let numbers: Vec<usize> = array_content
                            .split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();
                        if !numbers.is_empty() {
                            numbers
                        } else {
                            vec![1000, 224, 224, 3] // Default
                        }
                    } else {
                        vec![1000, 224, 224, 3]
                    }
                } else {
                    vec![1000, 224, 224, 3]
                }
            } else {
                vec![1000, 224, 224, 3]
            }
        } else {
            vec![1000, 224, 224, 3]
        };

        let dtype = if content.contains("\"dtype\"") {
            if let Some(start) = content.find("\"dtype\"") {
                if let Some(colon) = content[start..].find(':') {
                    if let Some(quote_start) = content[start + colon..].find('\"') {
                        if let Some(quote_end) =
                            content[start + colon + quote_start + 1..].find('\"')
                        {
                            content[start + colon + quote_start + 1
                                ..start + colon + quote_start + 1 + quote_end]
                                .to_string()
                        } else {
                            "<f4".to_string()
                        }
                    } else {
                        "<f4".to_string()
                    }
                } else {
                    "<f4".to_string()
                }
            } else {
                "<f4".to_string()
            }
        } else {
            "<f4".to_string()
        };

        let chunks = if content.contains("\"chunks\"") {
            // Similar extraction logic for chunks array
            if let Some(start) = content.find("\"chunks\"") {
                if let Some(array_start) = content[start..].find('[') {
                    if let Some(array_end) = content[start + array_start..].find(']') {
                        let array_content =
                            &content[start + array_start + 1..start + array_start + array_end];
                        let numbers: Vec<usize> = array_content
                            .split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();
                        if !numbers.is_empty() {
                            numbers
                        } else {
                            vec![100, 224, 224, 3] // Default
                        }
                    } else {
                        vec![100, 224, 224, 3]
                    }
                } else {
                    vec![100, 224, 224, 3]
                }
            } else {
                vec![100, 224, 224, 3]
            }
        } else {
            vec![100, 224, 224, 3]
        };

        let compressor = if content.contains("\"compressor\"") {
            if content.contains("\"blosc\"") {
                Some("blosc".to_string())
            } else if content.contains("\"gzip\"") {
                Some("gzip".to_string())
            } else if content.contains("\"lz4\"") {
                Some("lz4".to_string())
            } else if content.contains("\"zstd\"") {
                Some("zstd".to_string())
            } else {
                None
            }
        } else {
            None
        };

        let fill_value = if content.contains("\"fill_value\"") {
            // Try to extract numeric fill value
            if let Some(start) = content.find("\"fill_value\"") {
                if let Some(colon) = content[start..].find(':') {
                    let after_colon = &content[start + colon + 1..];
                    // Find the next comma or closing brace
                    let end_pos = after_colon
                        .find(',')
                        .or_else(|| after_colon.find('}'))
                        .unwrap_or(after_colon.len());
                    let value_str = after_colon[..end_pos].trim();
                    if value_str == "null" {
                        None
                    } else {
                        value_str.parse::<f64>().ok()
                    }
                } else {
                    Some(0.0)
                }
            } else {
                Some(0.0)
            }
        } else {
            None
        };

        let order = if content.contains("\"order\": \"F\"") {
            "F".to_string()
        } else {
            "C".to_string()
        };

        let zarr_format = if content.contains("\"zarr_format\": 3") {
            3
        } else {
            2
        };

        Ok(ZarrArrayInfo {
            shape,
            dtype,
            chunks,
            compressor,
            fill_value,
            order,
            zarr_format,
        })
    }

    /// Get array information
    pub fn array_info(&self) -> &ZarrArrayInfo {
        &self.array_info
    }

    /// Get labels array information
    pub fn labels_info(&self) -> Option<&ZarrArrayInfo> {
        self.labels_info.as_ref()
    }

    /// Load a specific chunk by coordinates
    pub fn load_chunk(&self, chunk_coords: &[usize]) -> Result<Vec<T>> {
        // Check cache first
        {
            let cache = self.chunk_cache.lock().map_err(|_| {
                TensorError::invalid_operation_simple("chunk cache lock poisoned".to_string())
            })?;
            if let Some(cached_data) = cache.get(chunk_coords) {
                return Ok(cached_data.clone());
            }
        }

        // Load chunk from disk
        let chunk_data = self.load_chunk_from_disk(chunk_coords)?;

        // Cache the loaded chunk
        {
            let mut cache = self.chunk_cache.lock().map_err(|_| {
                TensorError::invalid_operation_simple("chunk cache lock poisoned".to_string())
            })?;
            cache.insert(chunk_coords.to_vec(), chunk_data.clone());
        }

        Ok(chunk_data)
    }

    /// Load chunk from disk storage
    fn load_chunk_from_disk(&self, chunk_coords: &[usize]) -> Result<Vec<T>> {
        // Construct chunk file path
        let chunk_name = chunk_coords
            .iter()
            .map(|&coord| coord.to_string())
            .collect::<Vec<_>>()
            .join(".");

        let chunk_path = self.config.array_path.join(chunk_name);

        if !chunk_path.exists() {
            return Err(TensorError::invalid_argument(format!(
                "Chunk file not found: {chunk_path:?}"
            )));
        }

        // Read and decompress chunk data
        let chunk_bytes = fs::read(&chunk_path)
            .map_err(|e| TensorError::invalid_argument(format!("Failed to read chunk: {e}")))?;

        // Decompress if needed
        let decompressed_data = self.decompress_chunk_data(&chunk_bytes)?;

        // Convert bytes to typed data
        self.bytes_to_typed_data(&decompressed_data)
    }

    /// Convert raw bytes to typed data with proper type handling
    fn bytes_to_typed_data(&self, bytes: &[u8]) -> Result<Vec<T>> {
        // Handle different data types based on the dtype string
        match self.array_info.dtype.as_str() {
            "<f4" | ">f4" => self.parse_float32_data(bytes),
            "<f8" | ">f8" => self.parse_float64_data(bytes),
            "<i4" | ">i4" => self.parse_int32_data(bytes),
            "<i8" | ">i8" => self.parse_int64_data(bytes),
            "<u1" | ">u1" => self.parse_uint8_data(bytes),
            _ => {
                // Default to float32 for unknown types
                self.parse_float32_data(bytes)
            }
        }
    }

    /// Parse float32 data from bytes
    fn parse_float32_data(&self, bytes: &[u8]) -> Result<Vec<T>> {
        if bytes.len() % 4 != 0 {
            return Err(TensorError::invalid_argument(
                "Byte array length not divisible by 4 for float32 data".to_string(),
            ));
        }

        let num_elements = bytes.len() / 4;
        let mut data = Vec::with_capacity(num_elements);
        let is_little_endian = self.array_info.dtype.starts_with('<');

        for i in 0..num_elements {
            let byte_slice = &bytes[i * 4..(i + 1) * 4];
            let _value = if is_little_endian {
                f32::from_le_bytes([byte_slice[0], byte_slice[1], byte_slice[2], byte_slice[3]])
            } else {
                f32::from_be_bytes([byte_slice[0], byte_slice[1], byte_slice[2], byte_slice[3]])
            };

            // Convert to target type T using num_traits for safe conversion
            let converted =
                scirs2_core::num_traits::cast::NumCast::from(_value).unwrap_or_else(|| {
                    eprintln!(
                        "Warning: Failed to convert f32 {_value} to target type, using default"
                    );
                    T::default()
                });
            data.push(converted);
        }

        Ok(data)
    }

    /// Parse float64 data from bytes
    fn parse_float64_data(&self, bytes: &[u8]) -> Result<Vec<T>> {
        if bytes.len() % 8 != 0 {
            return Err(TensorError::invalid_argument(
                "Byte array length not divisible by 8 for float64 data".to_string(),
            ));
        }

        let num_elements = bytes.len() / 8;
        let mut data = Vec::with_capacity(num_elements);
        let is_little_endian = self.array_info.dtype.starts_with('<');

        for i in 0..num_elements {
            let byte_slice = &bytes[i * 8..(i + 1) * 8];
            let _value = if is_little_endian {
                f64::from_le_bytes([
                    byte_slice[0],
                    byte_slice[1],
                    byte_slice[2],
                    byte_slice[3],
                    byte_slice[4],
                    byte_slice[5],
                    byte_slice[6],
                    byte_slice[7],
                ])
            } else {
                f64::from_be_bytes([
                    byte_slice[0],
                    byte_slice[1],
                    byte_slice[2],
                    byte_slice[3],
                    byte_slice[4],
                    byte_slice[5],
                    byte_slice[6],
                    byte_slice[7],
                ])
            };

            // Convert to target type T using num_traits for safe conversion
            let converted =
                scirs2_core::num_traits::cast::NumCast::from(_value).unwrap_or_else(|| {
                    eprintln!(
                        "Warning: Failed to convert f64 {_value} to target type, using default"
                    );
                    T::default()
                });
            data.push(converted);
        }

        Ok(data)
    }

    /// Parse int32 data from bytes
    fn parse_int32_data(&self, bytes: &[u8]) -> Result<Vec<T>> {
        if bytes.len() % 4 != 0 {
            return Err(TensorError::invalid_argument(
                "Byte array length not divisible by 4 for int32 data".to_string(),
            ));
        }

        let num_elements = bytes.len() / 4;
        let mut data = Vec::with_capacity(num_elements);
        let is_little_endian = self.array_info.dtype.starts_with('<');

        for i in 0..num_elements {
            let byte_slice = &bytes[i * 4..(i + 1) * 4];
            let _value = if is_little_endian {
                i32::from_le_bytes([byte_slice[0], byte_slice[1], byte_slice[2], byte_slice[3]])
            } else {
                i32::from_be_bytes([byte_slice[0], byte_slice[1], byte_slice[2], byte_slice[3]])
            };

            // Convert to target type T using num_traits for safe conversion
            let converted =
                scirs2_core::num_traits::cast::NumCast::from(_value).unwrap_or_else(|| {
                    eprintln!(
                        "Warning: Failed to convert i32 {_value} to target type, using default"
                    );
                    T::default()
                });
            data.push(converted);
        }

        Ok(data)
    }

    /// Parse int64 data from bytes
    fn parse_int64_data(&self, bytes: &[u8]) -> Result<Vec<T>> {
        if bytes.len() % 8 != 0 {
            return Err(TensorError::invalid_argument(
                "Byte array length not divisible by 8 for int64 data".to_string(),
            ));
        }

        let num_elements = bytes.len() / 8;
        let mut data = Vec::with_capacity(num_elements);
        let is_little_endian = self.array_info.dtype.starts_with('<');

        for i in 0..num_elements {
            let byte_slice = &bytes[i * 8..(i + 1) * 8];
            let _value = if is_little_endian {
                i64::from_le_bytes([
                    byte_slice[0],
                    byte_slice[1],
                    byte_slice[2],
                    byte_slice[3],
                    byte_slice[4],
                    byte_slice[5],
                    byte_slice[6],
                    byte_slice[7],
                ])
            } else {
                i64::from_be_bytes([
                    byte_slice[0],
                    byte_slice[1],
                    byte_slice[2],
                    byte_slice[3],
                    byte_slice[4],
                    byte_slice[5],
                    byte_slice[6],
                    byte_slice[7],
                ])
            };

            // Convert to target type T using num_traits for safe conversion
            let converted =
                scirs2_core::num_traits::cast::NumCast::from(_value).unwrap_or_else(|| {
                    eprintln!(
                        "Warning: Failed to convert i64 {_value} to target type, using default"
                    );
                    T::default()
                });
            data.push(converted);
        }

        Ok(data)
    }

    /// Parse uint8 data from bytes
    fn parse_uint8_data(&self, bytes: &[u8]) -> Result<Vec<T>> {
        let mut data = Vec::with_capacity(bytes.len());

        for &_byte in bytes {
            // Convert to target type T using num_traits for safe conversion
            let converted =
                scirs2_core::num_traits::cast::NumCast::from(_byte).unwrap_or_else(|| {
                    eprintln!(
                        "Warning: Failed to convert u8 {_byte} to target type, using default"
                    );
                    T::default()
                });
            data.push(converted);
        }

        Ok(data)
    }

    /// Decompress chunk data based on the array's declared compressor.
    ///
    /// Supported codecs (including `blosc`, with all five of its inner
    /// codecs and both shuffle filters -- see `formats::blosc`) are
    /// decompressed for real. Codecs nobody has implemented yet return an
    /// honest error rather than handing back the still-compressed bytes
    /// pretending they were decoded, which would silently corrupt the data.
    fn decompress_chunk_data(&self, compressed_data: &[u8]) -> Result<Vec<u8>> {
        match &self.array_info.compressor {
            Some(compressor) => {
                match compressor.as_str() {
                    #[cfg(feature = "compression")]
                    "gzip" => {
                        let mut gzip_reader = GzipReader::new(std::io::Cursor::new(
                            compressed_data,
                        ))
                        .map_err(|e| {
                            TensorError::invalid_argument(format!(
                                "Gzip decompression init failed: {e}"
                            ))
                        })?;
                        let decompressed = gzip_reader.decompress().map_err(|e| {
                            TensorError::invalid_argument(format!("Gzip decompression failed: {e}"))
                        })?;
                        Ok(decompressed)
                    }
                    #[cfg(feature = "compression")]
                    "lz4" => {
                        let mut lz4_reader = Lz4Reader::new(std::io::Cursor::new(compressed_data))
                            .map_err(|e| {
                                TensorError::invalid_argument(format!(
                                    "LZ4 decompression init failed: {e}"
                                ))
                            })?;
                        let decompressed = lz4_reader.decompress().map_err(|e| {
                            TensorError::invalid_argument(format!("LZ4 decompression failed: {e}"))
                        })?;
                        Ok(decompressed)
                    }
                    #[cfg(feature = "compression")]
                    "zstd" => {
                        let mut zstd_reader = ZstdReader::new(std::io::Cursor::new(
                            compressed_data,
                        ))
                        .map_err(|e| {
                            TensorError::invalid_argument(format!(
                                "Zstd decompression init failed: {e}"
                            ))
                        })?;
                        let decompressed = zstd_reader.decompress().map_err(|e| {
                            TensorError::invalid_argument(format!("Zstd decompression failed: {e}"))
                        })?;
                        Ok(decompressed)
                    }
                    #[cfg(feature = "compression")]
                    "blosc" => super::blosc::decompress(compressed_data),
                    other => {
                        // Unsupported codec (or compression feature
                        // disabled). Never return still-compressed bytes as if
                        // they were decompressed: that would silently corrupt
                        // every downstream value.
                        Err(TensorError::not_implemented_simple(format!(
                            "Zarr chunk codec '{other}' is not supported; cannot decompress \
                             chunk data. Supported codecs: gzip, lz4, zstd, blosc (with the \
                             'compression' feature enabled)."
                        )))
                    }
                }
            }
            None => {
                // No compression declared: the chunk bytes are already raw.
                Ok(compressed_data.to_vec())
            }
        }
    }

    /// Calculate chunk coordinates for a given sample index
    fn sample_to_chunk_coords(&self, index: usize) -> Vec<usize> {
        let chunk_size = self.array_info.chunks[0];
        vec![index / chunk_size]
    }

    /// Get sample data by index
    pub fn get_sample_data(&self, index: usize) -> Result<(Vec<T>, Option<T>)> {
        // Calculate which chunk contains this sample
        let chunk_coords = self.sample_to_chunk_coords(index);

        // Load the chunk
        let chunk_data = self.load_chunk(&chunk_coords)?;

        // Extract the specific sample from the chunk
        let chunk_size = self.array_info.chunks[0];
        let sample_offset = index % chunk_size;
        let sample_size = self.array_info.shape[1..].iter().product::<usize>();

        let start_idx = sample_offset * sample_size;
        let end_idx = start_idx + sample_size;

        if end_idx > chunk_data.len() {
            return Err(TensorError::invalid_argument(format!(
                "Sample index {index} out of bounds"
            )));
        }

        let features = chunk_data[start_idx..end_idx].to_vec();

        // Load label if available
        let label = if self.labels_info.is_some() {
            // Load label from labels array (simplified)
            Some(T::default())
        } else {
            None
        };

        Ok((features, label))
    }
}

impl<T> Dataset<T> for ZarrDataset<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + std::str::FromStr
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::cast::NumCast,
{
    fn len(&self) -> usize {
        self.array_info.shape[0]
    }

    fn get(&self, index: usize) -> Result<(Tensor<T>, Tensor<T>)> {
        if index >= self.len() {
            return Err(TensorError::invalid_argument(format!(
                "Index {} out of bounds for dataset of size {}",
                index,
                self.len()
            )));
        }

        let (features, label) = self.get_sample_data(index)?;

        // Create feature tensor
        let feature_shape = self.array_info.shape[1..].to_vec();
        let feature_tensor = Tensor::from_vec(features, &feature_shape)?;

        // Create label tensor
        let label_tensor = if let Some(label_val) = label {
            Tensor::from_vec(vec![label_val], &[1])?
        } else {
            // Default label if no labels provided
            Tensor::from_vec(vec![T::default()], &[1])?
        };

        Ok((feature_tensor, label_tensor))
    }
}

/// Extension trait for easy Zarr dataset creation
pub trait ZarrDatasetExt<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + std::str::FromStr
        + Send
        + Sync
        + 'static,
{
    /// Create a Zarr dataset from a directory path
    fn from_zarr_path<P: AsRef<Path>>(path: P) -> Result<ZarrDataset<T>>;

    /// Create a Zarr dataset with labels
    fn from_zarr_with_labels<P: AsRef<Path>>(
        array_path: P,
        labels_path: P,
    ) -> Result<ZarrDataset<T>>;
}

impl<T> ZarrDatasetExt<T> for ZarrDataset<T>
where
    T: Clone
        + Default
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + std::str::FromStr
        + Send
        + Sync
        + 'static
        + scirs2_core::num_traits::cast::NumCast,
{
    fn from_zarr_path<P: AsRef<Path>>(path: P) -> Result<ZarrDataset<T>> {
        ZarrDatasetBuilder::new().array_path(path).build()
    }

    fn from_zarr_with_labels<P: AsRef<Path>>(
        array_path: P,
        labels_path: P,
    ) -> Result<ZarrDataset<T>> {
        ZarrDatasetBuilder::new()
            .array_path(array_path)
            .labels_path(labels_path)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zarr_config_creation() {
        let config = ZarrConfig::default();
        assert_eq!(config.chunk_cache_size, 100_000_000);
        assert!(config.lazy_loading);
        assert_eq!(config.max_parallel_chunks, 4);
        assert!(config.use_memory_mapping);
        #[cfg(feature = "cloud")]
        assert_eq!(config.cloud_backend, CloudBackend::Local);
        assert_eq!(config.compression, ZarrCompressionType::None);
        assert!(!config.async_io);
        assert_eq!(config.connection_timeout, 30);
        assert_eq!(config.retry_attempts, 3);
    }

    #[test]
    fn test_zarr_builder() {
        let builder = ZarrDatasetBuilder::<f32>::new()
            .array_path("/path/to/array")
            .chunk_cache_size(50_000_000)
            .lazy_loading(false)
            .max_parallel_chunks(8)
            .compression(ZarrCompressionType::None)
            .async_io(true)
            .connection_timeout(60)
            .retry_attempts(5);

        assert_eq!(builder.config.array_path, PathBuf::from("/path/to/array"));
        assert_eq!(builder.config.chunk_cache_size, 50_000_000);
        assert!(!builder.config.lazy_loading);
        assert_eq!(builder.config.max_parallel_chunks, 8);
        assert_eq!(builder.config.compression, ZarrCompressionType::None);
        assert!(builder.config.async_io);
        assert_eq!(builder.config.connection_timeout, 60);
        assert_eq!(builder.config.retry_attempts, 5);
    }

    #[test]
    fn test_chunk_coordinate_calculation() {
        let config = ZarrConfig {
            array_path: PathBuf::from("/test"),
            ..Default::default()
        };

        let array_info = ZarrArrayInfo {
            shape: vec![1000, 224, 224, 3],
            dtype: "<f4".to_string(),
            chunks: vec![100, 224, 224, 3],
            compressor: None,
            fill_value: None,
            order: "C".to_string(),
            zarr_format: 2,
        };

        let dataset = ZarrDataset::<f32> {
            config,
            array_info,
            labels_info: None,
            chunk_cache: Arc::new(std::sync::Mutex::new(HashMap::new())),
            _phantom: std::marker::PhantomData,
        };

        // Test chunk coordinate calculation
        assert_eq!(dataset.sample_to_chunk_coords(0), vec![0]);
        assert_eq!(dataset.sample_to_chunk_coords(50), vec![0]);
        assert_eq!(dataset.sample_to_chunk_coords(100), vec![1]);
        assert_eq!(dataset.sample_to_chunk_coords(250), vec![2]);
    }

    #[test]
    fn test_zarr_array_info() {
        let info = ZarrArrayInfo {
            shape: vec![1000, 224, 224, 3],
            dtype: "<f4".to_string(),
            chunks: vec![100, 224, 224, 3],
            compressor: Some("blosc".to_string()),
            fill_value: Some(0.0),
            order: "C".to_string(),
            zarr_format: 2,
        };

        assert_eq!(info.shape, vec![1000, 224, 224, 3]);
        assert_eq!(info.dtype, "<f4");
        assert_eq!(
            info.compressor
                .as_ref()
                .expect("test: value should be present"),
            "blosc"
        );
        assert_eq!(info.zarr_format, 2);
    }

    #[test]
    fn test_cloud_backend_configuration() {
        #[cfg(feature = "cloud")]
        {
            let s3_backend = CloudBackend::S3 {
                bucket: "my-bucket".to_string(),
                region: "us-west-2".to_string(),
                endpoint: None,
            };
            assert!(matches!(s3_backend, CloudBackend::S3 { .. }));

            let gcs_backend = CloudBackend::Gcs {
                bucket: "my-gcs-bucket".to_string(),
            };
            assert!(matches!(gcs_backend, CloudBackend::Gcs { .. }));

            let local_backend = CloudBackend::Local;
            assert_eq!(local_backend, CloudBackend::default());
        }
    }

    #[test]
    fn test_compression_types() {
        let none_compression = ZarrCompressionType::None;
        assert_eq!(none_compression, ZarrCompressionType::default());

        #[cfg(feature = "compression")]
        {
            let gzip_compression = ZarrCompressionType::Gzip;
            assert!(matches!(gzip_compression, ZarrCompressionType::Gzip));

            let blosc_compression = ZarrCompressionType::Blosc;
            assert!(matches!(blosc_compression, ZarrCompressionType::Blosc));
        }
    }

    #[test]
    fn test_enhanced_metadata_parsing() {
        let json_content = r#"{
            "chunks": [100, 224, 224, 3],
            "compressor": {"id": "blosc"},
            "dtype": "<f4",
            "fill_value": 0.0,
            "filters": null,
            "order": "C",
            "shape": [1000, 224, 224, 3],
            "zarr_format": 2
        }"#;

        let result = ZarrDataset::<f32>::parse_zarr_metadata(json_content);
        assert!(result.is_ok());

        let metadata = result.expect("test: operation should succeed");
        assert_eq!(metadata.shape, vec![1000, 224, 224, 3]);
        assert_eq!(metadata.dtype, "<f4");
        assert_eq!(metadata.chunks, vec![100, 224, 224, 3]);
        assert_eq!(metadata.order, "C");
        assert_eq!(metadata.zarr_format, 2);
    }

    /// Build a minimal `ZarrDataset<u8>` whose array declares `compressor`.
    fn dataset_with_compressor(compressor: Option<&str>) -> ZarrDataset<u8> {
        let array_info = ZarrArrayInfo {
            shape: vec![16],
            dtype: "|u1".to_string(),
            chunks: vec![16],
            compressor: compressor.map(|s| s.to_string()),
            fill_value: None,
            order: "C".to_string(),
            zarr_format: 2,
        };
        ZarrDataset::<u8> {
            config: ZarrConfig {
                array_path: PathBuf::from("/test"),
                ..Default::default()
            },
            array_info,
            labels_info: None,
            chunk_cache: Arc::new(std::sync::Mutex::new(HashMap::new())),
            _phantom: std::marker::PhantomData,
        }
    }

    #[test]
    fn test_decompress_chunk_no_compressor_returns_raw() {
        let dataset = dataset_with_compressor(None);
        let raw = b"raw chunk bytes!".to_vec();
        let out = dataset
            .decompress_chunk_data(&raw)
            .expect("test: no-compression path should succeed");
        assert_eq!(out, raw);
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_decompress_chunk_gzip_roundtrip() {
        let original: Vec<u8> = (0u16..512).map(|v| (v % 251) as u8).collect();
        let compressed = oxiarc_archive::gzip::compress(&original, 6)
            .expect("test: gzip compression should succeed");

        let dataset = dataset_with_compressor(Some("gzip"));
        let decompressed = dataset
            .decompress_chunk_data(&compressed)
            .expect("test: gzip decompression should succeed");
        assert_eq!(decompressed, original, "gzip must round-trip exactly");
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_decompress_chunk_zstd_roundtrip() {
        let original: Vec<u8> = (0u16..512).map(|v| (v % 251) as u8).collect();
        let compressed = oxiarc_archive::zstd::ZstdWriter::new()
            .compress(&original)
            .expect("test: zstd compression should succeed");

        let dataset = dataset_with_compressor(Some("zstd"));
        let decompressed = dataset
            .decompress_chunk_data(&compressed)
            .expect("test: zstd decompression should succeed");
        assert_eq!(decompressed, original, "zstd must round-trip exactly");
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_decompress_chunk_lz4_roundtrip() {
        let original: Vec<u8> = (0u16..512).map(|v| (v % 251) as u8).collect();
        let mut writer = oxiarc_archive::Lz4Writer::new(Vec::new());
        writer
            .write_compressed(&original)
            .expect("test: lz4 compression should succeed");
        let compressed = writer.into_inner();

        let dataset = dataset_with_compressor(Some("lz4"));
        let decompressed = dataset
            .decompress_chunk_data(&compressed)
            .expect("test: lz4 decompression should succeed");
        assert_eq!(decompressed, original, "lz4 must round-trip exactly");
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_decompress_chunk_blosc_roundtrip() {
        // Hand-assemble a real (single-block, single-split, zstd-inner-codec)
        // blosc chunk around real oxiarc-zstd-compressed bytes, then verify
        // `decompress_chunk_data` round-trips it via `formats::blosc`. The
        // blosc container format itself (header/bstarts/splits/shuffle) is
        // covered far more thoroughly by `formats::blosc`'s own tests,
        // including vectors produced by the real reference C-Blosc library;
        // this test only needs to prove the "blosc" compressor string is
        // wired up correctly here.
        let original: Vec<u8> = (0u16..512).map(|v| (v % 251) as u8).collect();
        let compressed_split = oxiarc_archive::zstd::ZstdWriter::new()
            .compress(&original)
            .expect("test: zstd compression should succeed");

        let mut chunk = Vec::with_capacity(16 + 4 + 4 + compressed_split.len());
        chunk.push(0); // version (unchecked by the decoder)
        chunk.push(1); // versionlz (unchecked by the decoder)
        chunk.push(4 << 5); // flags: format code 4 == zstd, no shuffle/memcpy
        chunk.push(1); // typesize
        chunk.extend_from_slice(&(original.len() as u32).to_le_bytes()); // nbytes
        chunk.extend_from_slice(&(original.len() as u32).to_le_bytes()); // blocksize (single block)
        let cbytes = (16 + 4 + 4 + compressed_split.len()) as u32;
        chunk.extend_from_slice(&cbytes.to_le_bytes()); // cbytes
        chunk.extend_from_slice(&20u32.to_le_bytes()); // bstarts[0]
        chunk.extend_from_slice(&(compressed_split.len() as u32).to_le_bytes()); // split length prefix
        chunk.extend_from_slice(&compressed_split);
        assert_eq!(chunk.len(), cbytes as usize);

        let dataset = dataset_with_compressor(Some("blosc"));
        let decompressed = dataset
            .decompress_chunk_data(&chunk)
            .expect("test: blosc decompression should succeed");
        assert_eq!(decompressed, original, "blosc must round-trip exactly");
    }

    #[test]
    fn test_decompress_chunk_unsupported_codec_is_honest_error() {
        // A codec nobody has implemented must error, never return the still
        // compressed bytes pretending they were decompressed.
        let dataset = dataset_with_compressor(Some("brotli"));
        let result = dataset.decompress_chunk_data(b"fake brotli payload");
        assert!(
            result.is_err(),
            "unsupported codec must return an error, not raw compressed bytes"
        );
    }
}
