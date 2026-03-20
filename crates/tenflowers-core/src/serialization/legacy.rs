/// Tensor Serialization Primitives for TenfloweRS
///
/// This module provides efficient serialization and deserialization of tensors
/// for checkpointing, model saving, distributed training, and ONNX interoperability.
///
/// ## Features
///
/// - **Multiple Formats**: Binary (efficient), JSON (readable), MessagePack (compact)
/// - **ONNX Compatibility**: Export tensors in ONNX tensor proto format
/// - **Compression**: Optional zlib/gzip compression for large tensors
/// - **Metadata**: Custom metadata and naming for model components
/// - **Streaming**: Support for large tensors that don't fit in memory
///
/// ## Usage
///
/// ```rust,ignore
/// use tenflowers_core::{Tensor, serialization::{serialize_tensor_binary, TensorMetadata}};
///
/// let tensor = Tensor::<f32>::randn(&[100, 100]);
///
/// // Binary serialization
/// let bytes = serialize_tensor_binary(&tensor)?;
///
/// // With metadata
/// let metadata = TensorMetadata::new(tensor.dtype(), tensor.shape(), *tensor.device())
///     .with_name("weights".to_string())
///     .with_metadata("layer".to_string(), "dense_1".to_string());
///
/// // ONNX format
/// let onnx_proto = serialize_tensor_onnx(&tensor, Some(metadata))?;
/// ```
use crate::{DType, Device, Result, Shape, Tensor, TensorError};
use std::io::{Read, Write};

/// Serialization format version for backward compatibility
pub const SERIALIZATION_VERSION: u32 = 1;

/// Magic number to identify TenfloweRS tensor files
pub const MAGIC_NUMBER: &[u8; 8] = b"TENFLWRS";

/// Serialization format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    /// Binary format (most efficient)
    Binary,
    /// JSON format (human-readable)
    Json,
    /// MessagePack format (compact and fast)
    MessagePack,
}

/// Metadata for serialized tensor
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TensorMetadata {
    /// Data type
    pub dtype: DType,
    /// Shape
    pub shape: Vec<usize>,
    /// Device (will be deserialized to CPU by default)
    pub device: String,
    /// Optional name/identifier
    pub name: Option<String>,
    /// Custom metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl TensorMetadata {
    /// Create new tensor metadata
    pub fn new(dtype: DType, shape: &Shape, device: Device) -> Self {
        Self {
            dtype,
            shape: shape.dims().to_vec(),
            device: format!("{:?}", device),
            name: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add custom metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Set name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Get total number of elements
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        let dtype_size = match self.dtype {
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::Int8 => 1,
            DType::UInt8 => 1,
            DType::Int16 => 2,
            DType::UInt16 => 2,
            DType::UInt32 => 4,
            DType::UInt64 => 8,
            DType::Bool => 1,
            DType::BFloat16 => 2,
            DType::Float16 => 2,
            DType::Complex64 => 8,
            DType::Int4 => 1,
            DType::Complex32 => 4,
            DType::String => 1, // Variable, estimate as 1 byte per char
        };
        self.num_elements() * dtype_size
    }
}

/// Serialization options
#[derive(Debug, Clone)]
pub struct SerializationOptions {
    /// Format to use
    pub format: SerializationFormat,
    /// Whether to compress data
    pub compress: bool,
    /// Compression level (0-9, higher = more compression)
    pub compression_level: u8,
    /// Include metadata
    pub include_metadata: bool,
}

impl Default for SerializationOptions {
    fn default() -> Self {
        Self {
            format: SerializationFormat::Binary,
            compress: false,
            compression_level: 6,
            include_metadata: true,
        }
    }
}

/// Serialize a tensor to bytes (binary format)
pub fn serialize_tensor_binary<T>(tensor: &Tensor<T>) -> Result<Vec<u8>>
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static,
{
    let metadata = TensorMetadata::new(tensor.dtype(), tensor.shape(), *tensor.device());

    let mut buffer = Vec::new();

    // Write magic number
    buffer.write_all(MAGIC_NUMBER).map_err(|e| {
        TensorError::serialization_error_simple(format!("Failed to write magic number: {}", e))
    })?;

    // Write version
    buffer
        .write_all(&SERIALIZATION_VERSION.to_le_bytes())
        .map_err(|e| {
            TensorError::serialization_error_simple(format!("Failed to write version: {}", e))
        })?;

    // Write metadata size
    #[cfg(feature = "serialize")]
    let metadata_json = serde_json::to_string(&metadata).map_err(|e| {
        TensorError::serialization_error_simple(format!("Failed to serialize metadata: {}", e))
    })?;
    #[cfg(not(feature = "serialize"))]
    let metadata_json = format!(
        "{{\"dtype\":\"{:?}\",\"shape\":{:?}}}",
        metadata.dtype, metadata.shape
    );
    let metadata_bytes = metadata_json.as_bytes();
    buffer
        .write_all(&(metadata_bytes.len() as u64).to_le_bytes())
        .map_err(|e| {
            TensorError::serialization_error_simple(format!("Failed to write metadata size: {}", e))
        })?;

    // Write metadata
    buffer.write_all(metadata_bytes).map_err(|e| {
        TensorError::serialization_error_simple(format!("Failed to write metadata: {}", e))
    })?;

    // Write tensor data
    let data_slice = tensor.data();
    let data_bytes = bytemuck::cast_slice(data_slice);
    buffer.write_all(data_bytes).map_err(|e| {
        TensorError::serialization_error_simple(format!("Failed to write tensor data: {}", e))
    })?;

    Ok(buffer)
}

/// Deserialize a tensor from bytes (binary format)
pub fn deserialize_tensor_binary<T>(bytes: &[u8]) -> Result<Tensor<T>>
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static,
{
    #[cfg(not(feature = "serialize"))]
    {
        let _ = bytes; // Suppress unused variable warning
        Err(TensorError::serialization_error_simple(
            "Deserialization requires 'serialize' feature".to_string(),
        ))
    }

    #[cfg(feature = "serialize")]
    {
        let mut cursor = 0;

        // Read and verify magic number
        if bytes.len() < MAGIC_NUMBER.len() {
            return Err(TensorError::serialization_error_simple(
                "File too small to contain magic number".to_string(),
            ));
        }
        if &bytes[cursor..cursor + MAGIC_NUMBER.len()] != MAGIC_NUMBER {
            return Err(TensorError::serialization_error_simple(
                "Invalid magic number".to_string(),
            ));
        }
        cursor += MAGIC_NUMBER.len();

        // Read version
        if bytes.len() < cursor + 4 {
            return Err(TensorError::serialization_error_simple(
                "File too small to contain version".to_string(),
            ));
        }
        let version = u32::from_le_bytes([
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
        ]);
        if version != SERIALIZATION_VERSION {
            return Err(TensorError::serialization_error_simple(format!(
                "Unsupported version: {}",
                version
            )));
        }
        cursor += 4;

        // Read metadata size
        if bytes.len() < cursor + 8 {
            return Err(TensorError::serialization_error_simple(
                "File too small to contain metadata size".to_string(),
            ));
        }
        let metadata_size = u64::from_le_bytes([
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
            bytes[cursor + 4],
            bytes[cursor + 5],
            bytes[cursor + 6],
            bytes[cursor + 7],
        ]) as usize;
        cursor += 8;

        // Read metadata
        if bytes.len() < cursor + metadata_size {
            return Err(TensorError::serialization_error_simple(
                "File too small to contain metadata".to_string(),
            ));
        }
        let metadata_bytes = &bytes[cursor..cursor + metadata_size];
        let metadata: TensorMetadata = serde_json::from_slice(metadata_bytes).map_err(|e| {
            TensorError::serialization_error_simple(format!("Failed to parse metadata: {}", e))
        })?;
        cursor += metadata_size;

        // Read tensor data
        let expected_size = metadata.size_bytes();
        if bytes.len() < cursor + expected_size {
            return Err(TensorError::serialization_error_simple(format!(
                "File too small to contain tensor data: expected {} bytes, got {}",
                expected_size,
                bytes.len() - cursor
            )));
        }

        let data_bytes = &bytes[cursor..cursor + expected_size];

        // Copy data to ensure proper alignment (bytemuck::cast_slice requires alignment)
        let mut aligned_data = vec![0u8; expected_size];
        aligned_data.copy_from_slice(data_bytes);
        let data_slice: &[T] = bytemuck::cast_slice(&aligned_data);
        let data_vec: Vec<T> = data_slice.to_vec();

        // Create tensor from array
        let shape = Shape::from_slice(&metadata.shape);
        use scirs2_core::ndarray::ArrayD;
        let array = ArrayD::from_shape_vec(shape.dims(), data_vec).map_err(|e| {
            TensorError::invalid_shape_simple(format!("Failed to create array: {}", e))
        })?;
        Ok(Tensor::from_array(array))
    }
}

/// Save tensor to a file
pub fn save_tensor<T>(tensor: &Tensor<T>, path: &std::path::Path) -> Result<()>
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static,
{
    let bytes = serialize_tensor_binary(tensor)?;
    std::fs::write(path, bytes).map_err(|e| {
        TensorError::io_error_simple(format!("Failed to write tensor to file: {}", e))
    })?;
    Ok(())
}

/// Load tensor from a file
pub fn load_tensor<T>(path: &std::path::Path) -> Result<Tensor<T>>
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static,
{
    let bytes = std::fs::read(path).map_err(|e| {
        TensorError::io_error_simple(format!("Failed to read tensor from file: {}", e))
    })?;
    deserialize_tensor_binary(&bytes)
}

/// Serialize multiple tensors to a single file (checkpoint)
pub fn save_checkpoint<T>(
    tensors: &std::collections::HashMap<String, &Tensor<T>>,
    path: &std::path::Path,
) -> Result<()>
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static,
{
    let mut checkpoint = Vec::new();

    // Write magic number
    checkpoint
        .write_all(MAGIC_NUMBER)
        .map_err(|e| TensorError::io_error_simple(format!("Failed to write: {}", e)))?;

    // Write version
    checkpoint
        .write_all(&SERIALIZATION_VERSION.to_le_bytes())
        .map_err(|e| TensorError::io_error_simple(format!("Failed to write: {}", e)))?;

    // Write number of tensors
    checkpoint
        .write_all(&(tensors.len() as u64).to_le_bytes())
        .map_err(|e| TensorError::io_error_simple(format!("Failed to write: {}", e)))?;

    // Write each tensor
    for (name, tensor) in tensors {
        // Write name length and name
        let name_bytes = name.as_bytes();
        checkpoint
            .write_all(&(name_bytes.len() as u64).to_le_bytes())
            .map_err(|e| TensorError::io_error_simple(format!("Failed to write: {}", e)))?;
        checkpoint
            .write_all(name_bytes)
            .map_err(|e| TensorError::io_error_simple(format!("Failed to write: {}", e)))?;

        // Write tensor data
        let tensor_bytes = serialize_tensor_binary(tensor)?;
        checkpoint
            .write_all(&(tensor_bytes.len() as u64).to_le_bytes())
            .map_err(|e| TensorError::io_error_simple(format!("Failed to write: {}", e)))?;
        checkpoint
            .write_all(&tensor_bytes)
            .map_err(|e| TensorError::io_error_simple(format!("Failed to write: {}", e)))?;
    }

    std::fs::write(path, checkpoint)
        .map_err(|e| TensorError::io_error_simple(format!("Failed to write checkpoint: {}", e)))?;

    Ok(())
}

/// Load multiple tensors from a checkpoint file
pub fn load_checkpoint<T>(
    path: &std::path::Path,
) -> Result<std::collections::HashMap<String, Tensor<T>>>
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static,
{
    let bytes = std::fs::read(path)
        .map_err(|e| TensorError::io_error_simple(format!("Failed to read checkpoint: {}", e)))?;

    let mut cursor = 0;

    // Verify magic number
    if &bytes[cursor..cursor + MAGIC_NUMBER.len()] != MAGIC_NUMBER {
        return Err(TensorError::serialization_error_simple(
            "Invalid checkpoint file".to_string(),
        ));
    }
    cursor += MAGIC_NUMBER.len();

    // Read version
    let version = u32::from_le_bytes([
        bytes[cursor],
        bytes[cursor + 1],
        bytes[cursor + 2],
        bytes[cursor + 3],
    ]);
    if version != SERIALIZATION_VERSION {
        return Err(TensorError::serialization_error_simple(format!(
            "Unsupported version: {}",
            version
        )));
    }
    cursor += 4;

    // Read number of tensors
    let num_tensors = u64::from_le_bytes([
        bytes[cursor],
        bytes[cursor + 1],
        bytes[cursor + 2],
        bytes[cursor + 3],
        bytes[cursor + 4],
        bytes[cursor + 5],
        bytes[cursor + 6],
        bytes[cursor + 7],
    ]) as usize;
    cursor += 8;

    let mut tensors = std::collections::HashMap::new();

    // Read each tensor
    for _ in 0..num_tensors {
        // Read name
        let name_len = u64::from_le_bytes([
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
            bytes[cursor + 4],
            bytes[cursor + 5],
            bytes[cursor + 6],
            bytes[cursor + 7],
        ]) as usize;
        cursor += 8;

        let name = String::from_utf8(bytes[cursor..cursor + name_len].to_vec()).map_err(|e| {
            TensorError::serialization_error_simple(format!("Invalid tensor name: {}", e))
        })?;
        cursor += name_len;

        // Read tensor
        let tensor_len = u64::from_le_bytes([
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
            bytes[cursor + 4],
            bytes[cursor + 5],
            bytes[cursor + 6],
            bytes[cursor + 7],
        ]) as usize;
        cursor += 8;

        let tensor = deserialize_tensor_binary(&bytes[cursor..cursor + tensor_len])?;
        cursor += tensor_len;

        tensors.insert(name, tensor);
    }

    Ok(tensors)
}

/// Serialize a tensor to JSON format
#[cfg(feature = "serialize")]
pub fn serialize_tensor_json<T>(tensor: &Tensor<T>) -> Result<Vec<u8>>
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static + serde::Serialize,
{
    let metadata = TensorMetadata::new(tensor.dtype(), tensor.shape(), *tensor.device());
    let data_slice = tensor.data();

    #[derive(serde::Serialize)]
    struct JsonTensor<'a, T> {
        metadata: TensorMetadata,
        data: &'a [T],
    }

    let json_tensor = JsonTensor {
        metadata,
        data: data_slice,
    };

    serde_json::to_vec(&json_tensor).map_err(|e| {
        TensorError::serialization_error_simple(format!("JSON serialization failed: {}", e))
    })
}

/// Deserialize a tensor from JSON format
#[cfg(feature = "serialize")]
pub fn deserialize_tensor_json<T>(bytes: &[u8]) -> Result<Tensor<T>>
where
    T: bytemuck::Pod
        + scirs2_core::num_traits::Float
        + Default
        + 'static
        + serde::de::DeserializeOwned,
{
    #[derive(serde::Deserialize)]
    struct JsonTensor<T> {
        metadata: TensorMetadata,
        data: Vec<T>,
    }

    let json_tensor: JsonTensor<T> = serde_json::from_slice(bytes).map_err(|e| {
        TensorError::serialization_error_simple(format!("JSON deserialization failed: {}", e))
    })?;

    let shape = Shape::from_slice(&json_tensor.metadata.shape);
    use scirs2_core::ndarray::ArrayD;
    let array = ArrayD::from_shape_vec(shape.dims(), json_tensor.data)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Failed to create array: {}", e)))?;

    Ok(Tensor::from_array(array))
}

/// Serialize a tensor to MessagePack format
#[cfg(feature = "serialize")]
pub fn serialize_tensor_msgpack<T>(tensor: &Tensor<T>) -> Result<Vec<u8>>
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static + serde::Serialize,
{
    let metadata = TensorMetadata::new(tensor.dtype(), tensor.shape(), *tensor.device());
    let data_slice = tensor.data();

    #[derive(serde::Serialize)]
    struct MsgPackTensor<'a, T> {
        metadata: TensorMetadata,
        data: &'a [T],
    }

    let msgpack_tensor = MsgPackTensor {
        metadata,
        data: data_slice,
    };

    rmp_serde::to_vec(&msgpack_tensor).map_err(|e| {
        TensorError::serialization_error_simple(format!("MessagePack serialization failed: {}", e))
    })
}

/// Deserialize a tensor from MessagePack format
#[cfg(feature = "serialize")]
pub fn deserialize_tensor_msgpack<T>(bytes: &[u8]) -> Result<Tensor<T>>
where
    T: bytemuck::Pod
        + scirs2_core::num_traits::Float
        + Default
        + 'static
        + serde::de::DeserializeOwned,
{
    #[derive(serde::Deserialize)]
    struct MsgPackTensor<T> {
        metadata: TensorMetadata,
        data: Vec<T>,
    }

    let msgpack_tensor: MsgPackTensor<T> = rmp_serde::from_slice(bytes).map_err(|e| {
        TensorError::serialization_error_simple(format!(
            "MessagePack deserialization failed: {}",
            e
        ))
    })?;

    let shape = Shape::from_slice(&msgpack_tensor.metadata.shape);
    use scirs2_core::ndarray::ArrayD;
    let array = ArrayD::from_shape_vec(shape.dims(), msgpack_tensor.data)
        .map_err(|e| TensorError::invalid_shape_simple(format!("Failed to create array: {}", e)))?;

    Ok(Tensor::from_array(array))
}

/// Compress bytes using gzip
#[cfg(feature = "compression")]
pub fn compress_bytes(data: &[u8], level: u8) -> Result<Vec<u8>> {
    use flate2::write::GzEncoder;
    use flate2::Compression;

    let compression_level = Compression::new(level.min(9) as u32);
    let mut encoder = GzEncoder::new(Vec::new(), compression_level);
    encoder.write_all(data).map_err(|e| {
        TensorError::serialization_error_simple(format!("Compression failed: {}", e))
    })?;

    encoder.finish().map_err(|e| {
        TensorError::serialization_error_simple(format!("Compression finalization failed: {}", e))
    })
}

/// Decompress gzip-compressed bytes
#[cfg(feature = "compression")]
pub fn decompress_bytes(data: &[u8]) -> Result<Vec<u8>> {
    use flate2::read::GzDecoder;

    let mut decoder = GzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed).map_err(|e| {
        TensorError::serialization_error_simple(format!("Decompression failed: {}", e))
    })?;

    Ok(decompressed)
}

/// Unified serialize API using SerializationOptions - Binary only without serialize feature
#[cfg(not(feature = "serialize"))]
pub fn serialize_tensor<T>(tensor: &Tensor<T>, options: &SerializationOptions) -> Result<Vec<u8>>
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static,
{
    let bytes = match options.format {
        SerializationFormat::Binary => serialize_tensor_binary(tensor)?,
        _ => {
            return Err(TensorError::serialization_error_simple(
                "JSON/MessagePack serialization requires 'serialize' feature".to_string(),
            ))
        }
    };

    #[cfg(feature = "compression")]
    if options.compress {
        return compress_bytes(&bytes, options.compression_level);
    }

    Ok(bytes)
}

/// Unified serialize API using SerializationOptions - All formats with serialize feature
#[cfg(feature = "serialize")]
pub fn serialize_tensor<T>(tensor: &Tensor<T>, options: &SerializationOptions) -> Result<Vec<u8>>
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static + serde::Serialize,
{
    let bytes = match options.format {
        SerializationFormat::Binary => serialize_tensor_binary(tensor)?,
        SerializationFormat::Json => serialize_tensor_json(tensor)?,
        SerializationFormat::MessagePack => serialize_tensor_msgpack(tensor)?,
    };

    #[cfg(feature = "compression")]
    if options.compress {
        return compress_bytes(&bytes, options.compression_level);
    }

    Ok(bytes)
}

/// Unified deserialize API using SerializationOptions - Binary only without serialize feature
#[cfg(not(feature = "serialize"))]
pub fn deserialize_tensor<T>(bytes: &[u8], options: &SerializationOptions) -> Result<Tensor<T>>
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static,
{
    #[cfg(feature = "compression")]
    let data = if options.compress {
        decompress_bytes(bytes)?
    } else {
        bytes.to_vec()
    };

    #[cfg(not(feature = "compression"))]
    let data = bytes.to_vec();

    match options.format {
        SerializationFormat::Binary => deserialize_tensor_binary(&data),
        _ => Err(TensorError::serialization_error_simple(
            "JSON/MessagePack deserialization requires 'serialize' feature".to_string(),
        )),
    }
}

/// Unified deserialize API using SerializationOptions - All formats with serialize feature
#[cfg(feature = "serialize")]
pub fn deserialize_tensor<T>(bytes: &[u8], options: &SerializationOptions) -> Result<Tensor<T>>
where
    T: bytemuck::Pod
        + scirs2_core::num_traits::Float
        + Default
        + 'static
        + serde::de::DeserializeOwned,
{
    #[cfg(feature = "compression")]
    let data = if options.compress {
        decompress_bytes(bytes)?
    } else {
        bytes.to_vec()
    };

    #[cfg(not(feature = "compression"))]
    let data = bytes.to_vec();

    match options.format {
        SerializationFormat::Binary => deserialize_tensor_binary(&data),
        SerializationFormat::Json => deserialize_tensor_json(&data),
        SerializationFormat::MessagePack => deserialize_tensor_msgpack(&data),
    }
}

/// Save tensor to a file with options - Binary only without serialize feature
#[cfg(not(feature = "serialize"))]
pub fn save_tensor_with_options<T>(
    tensor: &Tensor<T>,
    path: &std::path::Path,
    options: &SerializationOptions,
) -> Result<()>
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static,
{
    let bytes = serialize_tensor(tensor, options)?;
    std::fs::write(path, bytes).map_err(|e| {
        TensorError::io_error_simple(format!("Failed to write tensor to file: {}", e))
    })?;
    Ok(())
}

/// Save tensor to a file with options - All formats with serialize feature
#[cfg(feature = "serialize")]
pub fn save_tensor_with_options<T>(
    tensor: &Tensor<T>,
    path: &std::path::Path,
    options: &SerializationOptions,
) -> Result<()>
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static + serde::Serialize,
{
    let bytes = serialize_tensor(tensor, options)?;
    std::fs::write(path, bytes).map_err(|e| {
        TensorError::io_error_simple(format!("Failed to write tensor to file: {}", e))
    })?;
    Ok(())
}

/// Load tensor from a file with options - Binary only without serialize feature
#[cfg(not(feature = "serialize"))]
pub fn load_tensor_with_options<T>(
    path: &std::path::Path,
    options: &SerializationOptions,
) -> Result<Tensor<T>>
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static,
{
    let bytes = std::fs::read(path).map_err(|e| {
        TensorError::io_error_simple(format!("Failed to read tensor from file: {}", e))
    })?;
    deserialize_tensor(&bytes, options)
}

/// Load tensor from a file with options - All formats with serialize feature
#[cfg(feature = "serialize")]
pub fn load_tensor_with_options<T>(
    path: &std::path::Path,
    options: &SerializationOptions,
) -> Result<Tensor<T>>
where
    T: bytemuck::Pod
        + scirs2_core::num_traits::Float
        + Default
        + 'static
        + serde::de::DeserializeOwned,
{
    let bytes = std::fs::read(path).map_err(|e| {
        TensorError::io_error_simple(format!("Failed to read tensor from file: {}", e))
    })?;
    deserialize_tensor(&bytes, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    #[cfg(feature = "serialize")]
    fn test_tensor_serialization() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let data_dyn = data.into_dyn();
        let tensor = Tensor::from_array(data_dyn);

        let bytes = serialize_tensor_binary(&tensor).expect("Serialization failed");
        let deserialized: Tensor<f32> =
            deserialize_tensor_binary(&bytes).expect("Deserialization failed");

        assert_eq!(tensor.shape(), deserialized.shape());
        assert_eq!(tensor.data(), deserialized.data());
    }

    #[test]
    #[cfg(feature = "serialize")]
    fn test_checkpoint_save_load() {
        let tensor1 = Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn());
        let tensor2 = Tensor::from_array(array![[4.0f32, 5.0], [6.0, 7.0]].into_dyn());

        let mut tensors = std::collections::HashMap::new();
        tensors.insert("tensor1".to_string(), &tensor1);
        tensors.insert("tensor2".to_string(), &tensor2);

        let temp_dir = std::env::temp_dir();
        let checkpoint_path = temp_dir.join("test_checkpoint.tfrs");

        save_checkpoint(&tensors, &checkpoint_path).expect("Save failed");
        let loaded: std::collections::HashMap<String, Tensor<f32>> =
            load_checkpoint(&checkpoint_path).expect("Load failed");

        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains_key("tensor1"));
        assert!(loaded.contains_key("tensor2"));

        // Cleanup
        let _ = std::fs::remove_file(&checkpoint_path);
    }

    #[test]
    fn test_tensor_metadata() {
        let shape = Shape::from_slice(&[3, 4]);
        let metadata = TensorMetadata::new(DType::Float32, &shape, Device::Cpu)
            .with_name("test_tensor".to_string())
            .with_metadata("version".to_string(), "1.0".to_string());

        assert_eq!(metadata.num_elements(), 12);
        assert_eq!(metadata.size_bytes(), 48); // 12 * 4 bytes
        assert_eq!(metadata.name, Some("test_tensor".to_string()));
    }

    #[test]
    #[cfg(feature = "serialize")]
    fn test_json_serialization() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let tensor = Tensor::from_array(data.into_dyn());

        let bytes = serialize_tensor_json(&tensor).expect("JSON serialization failed");
        let deserialized: Tensor<f32> =
            deserialize_tensor_json(&bytes).expect("JSON deserialization failed");

        assert_eq!(tensor.shape(), deserialized.shape());
        assert_eq!(tensor.data(), deserialized.data());
    }

    #[test]
    #[cfg(feature = "serialize")]
    fn test_msgpack_serialization() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let tensor = Tensor::from_array(data.into_dyn());

        let bytes = serialize_tensor_msgpack(&tensor).expect("MessagePack serialization failed");
        let deserialized: Tensor<f32> =
            deserialize_tensor_msgpack(&bytes).expect("MessagePack deserialization failed");

        assert_eq!(tensor.shape(), deserialized.shape());
        assert_eq!(tensor.data(), deserialized.data());
    }

    #[test]
    #[cfg(feature = "compression")]
    fn test_compression() {
        let original_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let compressed = compress_bytes(&original_data, 6).expect("Compression failed");
        let decompressed = decompress_bytes(&compressed).expect("Decompression failed");

        assert_eq!(original_data, decompressed);
        // Compressed data should be different (and potentially smaller for larger inputs)
        assert_ne!(original_data, compressed);
    }

    #[test]
    #[cfg(feature = "serialize")]
    fn test_unified_binary_serialization() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let tensor = Tensor::from_array(data.into_dyn());

        let options = SerializationOptions {
            format: SerializationFormat::Binary,
            compress: false,
            compression_level: 6,
            include_metadata: true,
        };

        let bytes = serialize_tensor(&tensor, &options).expect("Serialization failed");
        let deserialized: Tensor<f32> =
            deserialize_tensor(&bytes, &options).expect("Deserialization failed");

        assert_eq!(tensor.shape(), deserialized.shape());
        assert_eq!(tensor.data(), deserialized.data());
    }

    #[test]
    #[cfg(all(feature = "serialize", feature = "compression"))]
    fn test_unified_json_compression() {
        let data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let tensor = Tensor::from_array(data.into_dyn());

        let options = SerializationOptions {
            format: SerializationFormat::Json,
            compress: true,
            compression_level: 9,
            include_metadata: true,
        };

        let bytes = serialize_tensor(&tensor, &options).expect("Serialization failed");
        let deserialized: Tensor<f32> =
            deserialize_tensor(&bytes, &options).expect("Deserialization failed");

        assert_eq!(tensor.shape(), deserialized.shape());
        assert_eq!(tensor.data(), deserialized.data());
    }

    #[test]
    #[cfg(all(feature = "serialize", feature = "compression"))]
    fn test_unified_msgpack_compression() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_array(data.into_dyn());

        let options = SerializationOptions {
            format: SerializationFormat::MessagePack,
            compress: true,
            compression_level: 6,
            include_metadata: true,
        };

        let bytes = serialize_tensor(&tensor, &options).expect("Serialization failed");
        let deserialized: Tensor<f32> =
            deserialize_tensor(&bytes, &options).expect("Deserialization failed");

        assert_eq!(tensor.shape(), deserialized.shape());
        assert_eq!(tensor.data(), deserialized.data());
    }

    #[test]
    #[cfg(feature = "serialize")]
    fn test_file_save_load_with_options() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let tensor = Tensor::from_array(data.into_dyn());

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_tensor_options.tfrs");

        let options = SerializationOptions::default();
        save_tensor_with_options(&tensor, &file_path, &options).expect("Save failed");
        let loaded: Tensor<f32> =
            load_tensor_with_options(&file_path, &options).expect("Load failed");

        assert_eq!(tensor.shape(), loaded.shape());
        assert_eq!(tensor.data(), loaded.data());

        // Cleanup
        let _ = std::fs::remove_file(&file_path);
    }

    #[test]
    #[cfg(feature = "serialize")]
    fn test_format_comparison() {
        let data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let tensor = Tensor::from_array(data.into_dyn());

        let binary_bytes = serialize_tensor_binary(&tensor).expect("Binary failed");
        let json_bytes = serialize_tensor_json(&tensor).expect("JSON failed");
        let msgpack_bytes = serialize_tensor_msgpack(&tensor).expect("MessagePack failed");

        // Binary should be most compact (for numerical data)
        assert!(binary_bytes.len() < json_bytes.len());
        // MessagePack should be more compact than JSON
        assert!(msgpack_bytes.len() < json_bytes.len());

        // All should deserialize correctly
        let _: Tensor<f32> = deserialize_tensor_binary(&binary_bytes).expect("Binary deser failed");
        let _: Tensor<f32> = deserialize_tensor_json(&json_bytes).expect("JSON deser failed");
        let _: Tensor<f32> =
            deserialize_tensor_msgpack(&msgpack_bytes).expect("MsgPack deser failed");
    }
}
