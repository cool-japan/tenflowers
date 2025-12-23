//! Binary tensor serialization format
//!
//! Efficient binary format for saving and loading tensors with metadata.
//! Format specification:
//!
//! Header (64 bytes):
//! - Magic number (4 bytes): "TFRS" (TenfloweRS)
//! - Version (4 bytes): Format version number
//! - Dtype (4 bytes): Data type identifier
//! - Device (4 bytes): Device type (0=CPU, 1=GPU, etc.)
//! - Rank (4 bytes): Number of dimensions
//! - Shape (32 bytes max): Dimension sizes (up to 8 dimensions x 4 bytes each)
//! - Data size (8 bytes): Size of data payload in bytes
//! - Checksum (4 bytes): CRC32 checksum of data
//!
//! Data payload:
//! - Raw tensor data in native byte order
//!
//! Optional metadata section:
//! - Metadata marker (4 bytes): "META"
//! - Metadata size (4 bytes): Size of JSON metadata
//! - JSON metadata: Custom metadata as JSON string

use std::collections::HashMap;
use std::io::{Read, Write};

use crate::{dtype_from_type, DType, Device, Result, Tensor, TensorError};

/// Binary format version
const FORMAT_VERSION: u32 = 1;

/// Magic number for format identification
const MAGIC_NUMBER: &[u8; 4] = b"TFRS";

/// Metadata section marker
const METADATA_MARKER: &[u8; 4] = b"META";

/// Maximum rank supported
const MAX_RANK: usize = 8;

/// Header size in bytes
const HEADER_SIZE: usize = 64;

/// Data type identifiers for serialization
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializedDType {
    F32 = 1,
    F64 = 2,
    I32 = 3,
    I64 = 4,
    U8 = 5,
    I8 = 6,
    U16 = 7,
    I16 = 8,
    U32 = 9,
    U64 = 10,
    Bool = 11,
    BF16 = 12,
    F16 = 13,
}

impl SerializedDType {
    /// Convert from DType
    pub fn from_dtype(dtype: DType) -> Result<Self> {
        match dtype {
            DType::Float32 => Ok(Self::F32),
            DType::Float64 => Ok(Self::F64),
            DType::Int32 => Ok(Self::I32),
            DType::Int64 => Ok(Self::I64),
            DType::UInt8 => Ok(Self::U8),
            DType::Int8 => Ok(Self::I8),
            DType::UInt16 => Ok(Self::U16),
            DType::Int16 => Ok(Self::I16),
            DType::UInt32 => Ok(Self::U32),
            DType::UInt64 => Ok(Self::U64),
            DType::Bool => Ok(Self::Bool),
            DType::BFloat16 => Ok(Self::BF16),
            DType::Float16 => Ok(Self::F16),
            _ => Err(TensorError::SerializationError {
                operation: "serialize".to_string(),
                details: format!("Unsupported dtype: {:?}", dtype),
                context: None,
            }),
        }
    }

    /// Convert to DType
    pub fn to_dtype(self) -> DType {
        match self {
            Self::F32 => DType::Float32,
            Self::F64 => DType::Float64,
            Self::I32 => DType::Int32,
            Self::I64 => DType::Int64,
            Self::U8 => DType::UInt8,
            Self::I8 => DType::Int8,
            Self::U16 => DType::UInt16,
            Self::I16 => DType::Int16,
            Self::U32 => DType::UInt32,
            Self::U64 => DType::UInt64,
            Self::Bool => DType::Bool,
            Self::BF16 => DType::BFloat16,
            Self::F16 => DType::Float16,
        }
    }
}

/// Device identifiers for serialization
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializedDevice {
    Cpu = 0,
    Gpu = 1,
    Cuda = 2,
    Rocm = 3,
    Metal = 4,
}

impl SerializedDevice {
    /// Convert from Device
    pub fn from_device(device: &Device) -> Self {
        match device {
            Device::Cpu => Self::Cpu,
            #[cfg(feature = "gpu")]
            Device::Gpu(_) => Self::Gpu,
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => Self::Rocm,
        }
    }

    /// Convert to Device
    pub fn to_device(self) -> Device {
        match self {
            Self::Cpu => Device::Cpu,
            // Default to CPU for compatibility when deserializing unknown devices
            Self::Gpu | Self::Cuda | Self::Rocm | Self::Metal => Device::Cpu,
        }
    }
}

/// Tensor metadata for serialization
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Custom metadata fields
    pub fields: HashMap<String, String>,
    /// Tensor name (optional)
    pub name: Option<String>,
    /// Creation timestamp
    pub created_at: Option<String>,
    /// Gradient flag
    pub requires_grad: bool,
}

impl TensorMetadata {
    /// Create empty metadata
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            name: None,
            created_at: None,
            requires_grad: false,
        }
    }

    /// Add a custom field
    pub fn add_field(&mut self, key: String, value: String) {
        self.fields.insert(key, value);
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| TensorError::SerializationError {
            operation: "serialize".to_string(),
            details: format!("Failed to serialize metadata: {}", e),
            context: None,
        })
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| TensorError::SerializationError {
            operation: "serialize".to_string(),
            details: format!("Failed to deserialize metadata: {}", e),
            context: None,
        })
    }
}

impl Default for TensorMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl serde::Serialize for TensorMetadata {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("TensorMetadata", 4)?;
        state.serialize_field("fields", &self.fields)?;
        state.serialize_field("name", &self.name)?;
        state.serialize_field("created_at", &self.created_at)?;
        state.serialize_field("requires_grad", &self.requires_grad)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for TensorMetadata {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct TensorMetadataHelper {
            fields: HashMap<String, String>,
            name: Option<String>,
            created_at: Option<String>,
            requires_grad: bool,
        }

        let helper = TensorMetadataHelper::deserialize(deserializer)?;
        Ok(TensorMetadata {
            fields: helper.fields,
            name: helper.name,
            created_at: helper.created_at,
            requires_grad: helper.requires_grad,
        })
    }
}

/// Binary tensor serializer
pub struct BinarySerializer;

impl BinarySerializer {
    /// Serialize a tensor to binary format
    pub fn serialize<T, W>(
        tensor: &Tensor<T>,
        writer: &mut W,
        metadata: Option<&TensorMetadata>,
    ) -> Result<()>
    where
        T: Clone + bytemuck::Pod + bytemuck::Zeroable + Send + Sync + 'static,
        W: Write,
    {
        // Write header
        Self::write_header(tensor, writer)?;

        // Write data
        Self::write_data(tensor, writer)?;

        // Write metadata if provided
        if let Some(meta) = metadata {
            Self::write_metadata(meta, writer)?;
        }

        Ok(())
    }

    /// Deserialize a tensor from binary format
    pub fn deserialize<T, R>(reader: &mut R) -> Result<(Tensor<T>, Option<TensorMetadata>)>
    where
        T: Clone + bytemuck::Pod + bytemuck::Zeroable + Send + Sync + Default + 'static,
        R: Read,
    {
        // Read and validate header
        let (shape, dtype, device, data_size) = Self::read_header(reader)?;

        // Validate dtype matches T
        let expected_dtype = dtype_from_type::<T>();
        if dtype != expected_dtype {
            return Err(TensorError::SerializationError {
                operation: "serialize".to_string(),
                details: format!(
                    "Type mismatch: expected {:?}, got {:?}",
                    expected_dtype, dtype
                ),
                context: None,
            });
        }

        // Read data
        let data = Self::read_data::<T, R>(reader, data_size)?;

        // Create tensor
        let tensor = Tensor::from_data(data, &shape)?;

        // Try to read metadata
        let metadata = Self::read_metadata(reader).ok();

        Ok((tensor, metadata))
    }

    /// Write header
    fn write_header<T, W>(tensor: &Tensor<T>, writer: &mut W) -> Result<()>
    where
        T: Clone + bytemuck::Pod + bytemuck::Zeroable + Send + Sync + 'static,
        W: Write,
    {
        let mut header = [0u8; HEADER_SIZE];
        let mut offset = 0;

        // Magic number
        header[offset..offset + 4].copy_from_slice(MAGIC_NUMBER);
        offset += 4;

        // Version
        header[offset..offset + 4].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        offset += 4;

        // Dtype
        let dtype = dtype_from_type::<T>();
        let serialized_dtype = SerializedDType::from_dtype(dtype)?;
        header[offset..offset + 4].copy_from_slice(&(serialized_dtype as u32).to_le_bytes());
        offset += 4;

        // Device
        let serialized_device = SerializedDevice::from_device(tensor.device());
        header[offset..offset + 4].copy_from_slice(&(serialized_device as u32).to_le_bytes());
        offset += 4;

        // Rank
        let shape = tensor.shape().dims();
        let rank = shape.len() as u32;
        if rank > MAX_RANK as u32 {
            return Err(TensorError::SerializationError {
                operation: "serialize".to_string(),
                details: format!("Rank {} exceeds maximum {}", rank, MAX_RANK),
                context: None,
            });
        }
        header[offset..offset + 4].copy_from_slice(&rank.to_le_bytes());
        offset += 4;

        // Shape (up to 8 dimensions)
        for (i, &dim) in shape.iter().enumerate() {
            if i >= MAX_RANK {
                break;
            }
            let dim_offset = offset + i * 4;
            header[dim_offset..dim_offset + 4].copy_from_slice(&(dim as u32).to_le_bytes());
        }
        offset += 32;

        // Data size
        let element_size = std::mem::size_of::<T>();
        let data_size = tensor.numel() * element_size;
        header[offset..offset + 8].copy_from_slice(&(data_size as u64).to_le_bytes());
        offset += 8;

        // Checksum (simple for now, could use CRC32)
        let checksum = Self::compute_checksum(tensor)?;
        header[offset..offset + 4].copy_from_slice(&checksum.to_le_bytes());

        writer
            .write_all(&header)
            .map_err(|e| TensorError::SerializationError {
                operation: "serialize".to_string(),
                details: format!("Failed to write header: {}", e),
                context: None,
            })
    }

    /// Read header
    fn read_header<R>(reader: &mut R) -> Result<(Vec<usize>, DType, Device, usize)>
    where
        R: Read,
    {
        let mut header = [0u8; HEADER_SIZE];
        reader
            .read_exact(&mut header)
            .map_err(|e| TensorError::SerializationError {
                operation: "serialize".to_string(),
                details: format!("Failed to read header: {}", e),
                context: None,
            })?;

        let mut offset = 0;

        // Validate magic number
        let magic = &header[offset..offset + 4];
        if magic != MAGIC_NUMBER {
            return Err(TensorError::SerializationError {
                operation: "serialize".to_string(),
                details: "Invalid magic number".to_string(),
                context: None,
            });
        }
        offset += 4;

        // Version
        let version = u32::from_le_bytes([
            header[offset],
            header[offset + 1],
            header[offset + 2],
            header[offset + 3],
        ]);
        if version != FORMAT_VERSION {
            return Err(TensorError::SerializationError {
                operation: "serialize".to_string(),
                details: format!("Unsupported version: {}", version),
                context: None,
            });
        }
        offset += 4;

        // Dtype
        let dtype_id = u32::from_le_bytes([
            header[offset],
            header[offset + 1],
            header[offset + 2],
            header[offset + 3],
        ]);
        let dtype = match dtype_id {
            1 => SerializedDType::F32,
            2 => SerializedDType::F64,
            3 => SerializedDType::I32,
            _ => {
                return Err(TensorError::SerializationError {
                    operation: "serialize".to_string(),
                    details: format!("Unknown dtype: {}", dtype_id),
                    context: None,
                })
            }
        }
        .to_dtype();
        offset += 4;

        // Device
        let device_id = u32::from_le_bytes([
            header[offset],
            header[offset + 1],
            header[offset + 2],
            header[offset + 3],
        ]);
        let device = match device_id {
            0 => SerializedDevice::Cpu,
            1 => SerializedDevice::Gpu,
            _ => SerializedDevice::Cpu,
        }
        .to_device();
        offset += 4;

        // Rank
        let rank = u32::from_le_bytes([
            header[offset],
            header[offset + 1],
            header[offset + 2],
            header[offset + 3],
        ]) as usize;
        if rank > MAX_RANK {
            return Err(TensorError::SerializationError {
                operation: "serialize".to_string(),
                details: format!("Invalid rank: {}", rank),
                context: None,
            });
        }
        offset += 4;

        // Shape
        let mut shape = Vec::with_capacity(rank);
        for i in 0..rank {
            let dim_offset = offset + i * 4;
            let dim = u32::from_le_bytes([
                header[dim_offset],
                header[dim_offset + 1],
                header[dim_offset + 2],
                header[dim_offset + 3],
            ]) as usize;
            shape.push(dim);
        }
        offset += 32;

        // Data size
        let data_size = u64::from_le_bytes([
            header[offset],
            header[offset + 1],
            header[offset + 2],
            header[offset + 3],
            header[offset + 4],
            header[offset + 5],
            header[offset + 6],
            header[offset + 7],
        ]) as usize;

        Ok((shape, dtype, device, data_size))
    }

    /// Write tensor data
    fn write_data<T, W>(tensor: &Tensor<T>, writer: &mut W) -> Result<()>
    where
        T: Clone + bytemuck::Pod + bytemuck::Zeroable + Send + Sync + 'static,
        W: Write,
    {
        let data = tensor
            .as_slice()
            .ok_or_else(|| TensorError::SerializationError {
                operation: "serialize".to_string(),
                details: "Cannot serialize GPU tensor directly".to_string(),
                context: None,
            })?;
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };

        writer
            .write_all(bytes)
            .map_err(|e| TensorError::SerializationError {
                operation: "serialize".to_string(),
                details: format!("Failed to write data: {}", e),
                context: None,
            })
    }

    /// Read tensor data
    fn read_data<T, R>(reader: &mut R, data_size: usize) -> Result<Vec<T>>
    where
        T: Clone + bytemuck::Pod + bytemuck::Zeroable + Send + Sync + Default + 'static,
        R: Read,
    {
        let element_size = std::mem::size_of::<T>();
        let num_elements = data_size / element_size;

        let mut data = vec![T::default(); num_elements];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                data.as_mut_ptr() as *mut u8,
                num_elements * element_size,
            )
        };

        reader
            .read_exact(bytes)
            .map_err(|e| TensorError::SerializationError {
                operation: "serialize".to_string(),
                details: format!("Failed to read data: {}", e),
                context: None,
            })?;

        Ok(data)
    }

    /// Write metadata
    fn write_metadata<W>(metadata: &TensorMetadata, writer: &mut W) -> Result<()>
    where
        W: Write,
    {
        let json = metadata.to_json()?;
        let json_bytes = json.as_bytes();

        // Write metadata marker
        writer
            .write_all(METADATA_MARKER)
            .map_err(|e| TensorError::SerializationError {
                operation: "serialize".to_string(),
                details: format!("Failed to write metadata marker: {}", e),
                context: None,
            })?;

        // Write metadata size
        let size = json_bytes.len() as u32;
        writer
            .write_all(&size.to_le_bytes())
            .map_err(|e| TensorError::SerializationError {
                operation: "serialize".to_string(),
                details: format!("Failed to write metadata size: {}", e),
                context: None,
            })?;

        // Write metadata
        writer
            .write_all(json_bytes)
            .map_err(|e| TensorError::SerializationError {
                operation: "serialize".to_string(),
                details: format!("Failed to write metadata: {}", e),
                context: None,
            })
    }

    /// Read metadata
    fn read_metadata<R>(reader: &mut R) -> Result<TensorMetadata>
    where
        R: Read,
    {
        // Read metadata marker
        let mut marker = [0u8; 4];
        reader
            .read_exact(&mut marker)
            .map_err(|_| TensorError::SerializationError {
                operation: "serialize".to_string(),
                details: "No metadata section found".to_string(),
                context: None,
            })?;

        if &marker != METADATA_MARKER {
            return Err(TensorError::SerializationError {
                operation: "serialize".to_string(),
                details: "Invalid metadata marker".to_string(),
                context: None,
            });
        }

        // Read metadata size
        let mut size_bytes = [0u8; 4];
        reader
            .read_exact(&mut size_bytes)
            .map_err(|e| TensorError::SerializationError {
                operation: "serialize".to_string(),
                details: format!("Failed to read metadata size: {}", e),
                context: None,
            })?;
        let size = u32::from_le_bytes(size_bytes) as usize;

        // Read metadata
        let mut json_bytes = vec![0u8; size];
        reader
            .read_exact(&mut json_bytes)
            .map_err(|e| TensorError::SerializationError {
                operation: "serialize".to_string(),
                details: format!("Failed to read metadata: {}", e),
                context: None,
            })?;

        let json = String::from_utf8(json_bytes).map_err(|e| TensorError::SerializationError {
            operation: "serialize".to_string(),
            details: format!("Invalid UTF-8 in metadata: {}", e),
            context: None,
        })?;

        TensorMetadata::from_json(&json)
    }

    /// Compute simple checksum
    fn compute_checksum<T>(_tensor: &Tensor<T>) -> Result<u32>
    where
        T: Clone + bytemuck::Pod + bytemuck::Zeroable + Send + Sync + 'static,
    {
        // Simplified checksum - in production would use CRC32
        // For now, just return 0 as placeholder
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_serialize_deserialize_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_data(data.clone(), &[2, 3]).unwrap();

        let mut buffer = Vec::new();
        BinarySerializer::serialize(&tensor, &mut buffer, None).unwrap();

        let mut cursor = Cursor::new(buffer);
        let (deserialized, _): (Tensor<f32>, _) =
            BinarySerializer::deserialize(&mut cursor).unwrap();

        assert_eq!(tensor.shape().dims(), deserialized.shape().dims());
        assert_eq!(tensor.as_slice().unwrap(), deserialized.as_slice().unwrap());
    }

    #[test]
    fn test_serialize_with_metadata() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(data.clone(), &[2, 2]).unwrap();

        let mut metadata = TensorMetadata::new();
        metadata.name = Some("test_tensor".to_string());
        metadata.requires_grad = true;
        metadata.add_field("version".to_string(), "1.0".to_string());

        let mut buffer = Vec::new();
        BinarySerializer::serialize(&tensor, &mut buffer, Some(&metadata)).unwrap();

        let mut cursor = Cursor::new(buffer);
        let (deserialized, meta): (Tensor<f32>, _) =
            BinarySerializer::deserialize(&mut cursor).unwrap();

        assert!(meta.is_some());
        let meta = meta.unwrap();
        assert_eq!(meta.name, Some("test_tensor".to_string()));
        assert_eq!(meta.requires_grad, true);
        assert_eq!(meta.fields.get("version"), Some(&"1.0".to_string()));

        assert_eq!(tensor.as_slice().unwrap(), deserialized.as_slice().unwrap());
    }

    #[test]
    fn test_different_shapes() {
        // 1D tensor
        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_data(data.clone(), &[3]).unwrap();
        let mut buffer = Vec::new();
        BinarySerializer::serialize(&tensor, &mut buffer, None).unwrap();
        let mut cursor = Cursor::new(buffer);
        let (deserialized, _): (Tensor<f32>, _) =
            BinarySerializer::deserialize(&mut cursor).unwrap();
        assert_eq!(tensor.shape().dims(), deserialized.shape().dims());

        // 3D tensor
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_data(data.clone(), &[2, 2, 2]).unwrap();
        let mut buffer = Vec::new();
        BinarySerializer::serialize(&tensor, &mut buffer, None).unwrap();
        let mut cursor = Cursor::new(buffer);
        let (deserialized, _): (Tensor<f32>, _) =
            BinarySerializer::deserialize(&mut cursor).unwrap();
        assert_eq!(tensor.shape().dims(), deserialized.shape().dims());
    }

    #[test]
    fn test_metadata_serialization() {
        let mut metadata = TensorMetadata::new();
        metadata.name = Some("test".to_string());
        metadata.requires_grad = true;
        metadata.add_field("key1".to_string(), "value1".to_string());
        metadata.add_field("key2".to_string(), "value2".to_string());

        let json = metadata.to_json().unwrap();
        let deserialized = TensorMetadata::from_json(&json).unwrap();

        assert_eq!(metadata.name, deserialized.name);
        assert_eq!(metadata.requires_grad, deserialized.requires_grad);
        assert_eq!(metadata.fields, deserialized.fields);
    }

    #[test]
    fn test_invalid_magic_number() {
        let mut buffer = vec![0u8; 64];
        buffer[0..4].copy_from_slice(b"XXXX"); // Wrong magic number

        let mut cursor = Cursor::new(buffer);
        let result: Result<(Tensor<f32>, _)> = BinarySerializer::deserialize(&mut cursor);
        assert!(result.is_err());
    }
}
