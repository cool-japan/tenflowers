/// ONNX-Compatible Tensor Serialization for TenfloweRS
///
/// This module provides ONNX tensor proto-compatible serialization
/// for interoperability with ONNX Runtime and other frameworks.
///
/// ONNX Tensor Proto Format:
/// - Data type enumeration compatible with ONNX standard
/// - Row-major (C-contiguous) data layout
/// - Shape and stride information
/// - Optional external data references
///
/// ## References
/// - [ONNX Tensor Proto Spec](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3)
/// - [ONNX Data Types](https://github.com/onnx/onnx/blob/main/docs/Operators.md#types)
use crate::{DType, Device, Result, Shape, Tensor, TensorError};
use std::collections::HashMap;

/// ONNX data type enumeration (matching onnx.proto3)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum OnnxDataType {
    Undefined = 0,
    Float = 1,       // float32
    UInt8 = 2,       // uint8
    Int8 = 3,        // int8
    UInt16 = 4,      // uint16
    Int16 = 5,       // int16
    Int32 = 6,       // int32
    Int64 = 7,       // int64
    String = 8,      // string
    Bool = 9,        // bool
    Float16 = 10,    // float16
    Double = 11,     // float64
    UInt32 = 12,     // uint32
    UInt64 = 13,     // uint64
    Complex64 = 14,  // complex64
    Complex128 = 15, // complex128
    BFloat16 = 16,   // bfloat16
}

impl OnnxDataType {
    /// Convert TenfloweRS DType to ONNX DataType
    pub fn from_dtype(dtype: DType) -> Result<Self> {
        match dtype {
            DType::Float32 => Ok(Self::Float),
            DType::Float64 => Ok(Self::Double),
            DType::Int8 => Ok(Self::Int8),
            DType::UInt8 => Ok(Self::UInt8),
            DType::Int16 => Ok(Self::Int16),
            DType::UInt16 => Ok(Self::UInt16),
            DType::Int32 => Ok(Self::Int32),
            DType::UInt32 => Ok(Self::UInt32),
            DType::Int64 => Ok(Self::Int64),
            DType::UInt64 => Ok(Self::UInt64),
            DType::Bool => Ok(Self::Bool),
            DType::Float16 => Ok(Self::Float16),
            DType::BFloat16 => Ok(Self::BFloat16),
            DType::Complex64 => Ok(Self::Complex64),
            DType::String => Ok(Self::String),
            _ => Err(TensorError::unsupported_operation_simple(format!(
                "ONNX does not support dtype: {:?}",
                dtype
            ))),
        }
    }

    /// Convert ONNX DataType to TenfloweRS DType
    pub fn to_dtype(&self) -> Result<DType> {
        match self {
            Self::Float => Ok(DType::Float32),
            Self::Double => Ok(DType::Float64),
            Self::Int8 => Ok(DType::Int8),
            Self::UInt8 => Ok(DType::UInt8),
            Self::Int16 => Ok(DType::Int16),
            Self::UInt16 => Ok(DType::UInt16),
            Self::Int32 => Ok(DType::Int32),
            Self::UInt32 => Ok(DType::UInt32),
            Self::Int64 => Ok(DType::Int64),
            Self::UInt64 => Ok(DType::UInt64),
            Self::Bool => Ok(DType::Bool),
            Self::Float16 => Ok(DType::Float16),
            Self::BFloat16 => Ok(DType::BFloat16),
            Self::Complex64 => Ok(DType::Complex64),
            Self::String => Ok(DType::String),
            Self::Undefined => Err(TensorError::invalid_argument(
                "Cannot convert undefined ONNX type to DType".to_string(),
            )),
            Self::Complex128 => Err(TensorError::unsupported_operation_simple(
                "Complex128 not yet supported in TenfloweRS".to_string(),
            )),
        }
    }

    /// Get element size in bytes
    pub fn element_size(&self) -> usize {
        match self {
            Self::Float => 4,
            Self::Double => 8,
            Self::Int8 => 1,
            Self::UInt8 => 1,
            Self::Int16 => 2,
            Self::UInt16 => 2,
            Self::Int32 => 4,
            Self::UInt32 => 4,
            Self::Int64 => 8,
            Self::UInt64 => 8,
            Self::Bool => 1,
            Self::Float16 => 2,
            Self::BFloat16 => 2,
            Self::Complex64 => 8,
            Self::Complex128 => 16,
            Self::String => 0, // Variable size
            Self::Undefined => 0,
        }
    }
}

/// ONNX Tensor Proto representation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct OnnxTensorProto {
    /// Tensor dimensions
    pub dims: Vec<i64>,
    /// Data type
    pub data_type: i32,
    /// Optional segment info for large tensors
    pub segment: Option<OnnxSegment>,
    /// Raw float data (for Float type)
    pub float_data: Vec<f32>,
    /// Raw int32 data (for Int32 type)
    pub int32_data: Vec<i32>,
    /// Raw string data (for String type)
    pub string_data: Vec<Vec<u8>>,
    /// Raw int64 data (for Int64 type)
    pub int64_data: Vec<i64>,
    /// Tensor name
    pub name: String,
    /// Optional documentation string
    pub doc_string: String,
    /// Raw data (for all types) - binary packed
    pub raw_data: Vec<u8>,
    /// External data info (for large tensors stored separately)
    pub external_data: Vec<OnnxExternalData>,
}

impl Default for OnnxTensorProto {
    fn default() -> Self {
        Self {
            dims: Vec::new(),
            data_type: OnnxDataType::Undefined as i32,
            segment: None,
            float_data: Vec::new(),
            int32_data: Vec::new(),
            string_data: Vec::new(),
            int64_data: Vec::new(),
            name: String::new(),
            doc_string: String::new(),
            raw_data: Vec::new(),
            external_data: Vec::new(),
        }
    }
}

/// ONNX Segment for large tensors
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct OnnxSegment {
    pub begin: i64,
    pub end: i64,
}

/// ONNX External Data reference
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct OnnxExternalData {
    /// Key for this external data entry
    pub key: String,
    /// Value (e.g., file path, offset, length)
    pub value: String,
}

impl OnnxTensorProto {
    /// Create a new ONNX tensor proto
    pub fn new() -> Self {
        Self::default()
    }

    /// Set tensor name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    /// Set documentation string
    pub fn with_doc(mut self, doc: String) -> Self {
        self.doc_string = doc;
        self
    }

    /// Get total number of elements
    pub fn num_elements(&self) -> i64 {
        if self.dims.is_empty() {
            0
        } else {
            self.dims.iter().product()
        }
    }
}

/// Serialize a tensor to ONNX TensorProto format
pub fn serialize_tensor_onnx<T>(tensor: &Tensor<T>, name: Option<String>) -> Result<OnnxTensorProto>
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static,
{
    let dtype = tensor.dtype();
    let onnx_dtype = OnnxDataType::from_dtype(dtype)?;

    let mut proto = OnnxTensorProto::new();

    // Set dimensions (convert usize to i64)
    proto.dims = tensor.shape().dims().iter().map(|&d| d as i64).collect();

    // Set data type
    proto.data_type = onnx_dtype as i32;

    // Set name
    if let Some(n) = name {
        proto.name = n;
    }

    // Pack raw data in little-endian byte order
    let data_slice = tensor.data();
    let data_bytes: &[u8] = bytemuck::cast_slice(data_slice);
    proto.raw_data = data_bytes.to_vec();

    Ok(proto)
}

/// Deserialize a tensor from ONNX TensorProto format
pub fn deserialize_tensor_onnx<T>(proto: &OnnxTensorProto) -> Result<Tensor<T>>
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static,
{
    // Convert ONNX data type
    let onnx_dtype = match proto.data_type {
        x if x == OnnxDataType::Float as i32 => OnnxDataType::Float,
        x if x == OnnxDataType::Double as i32 => OnnxDataType::Double,
        x if x == OnnxDataType::Int32 as i32 => OnnxDataType::Int32,
        x if x == OnnxDataType::Int64 as i32 => OnnxDataType::Int64,
        x if x == OnnxDataType::Float16 as i32 => OnnxDataType::Float16,
        x if x == OnnxDataType::BFloat16 as i32 => OnnxDataType::BFloat16,
        _ => {
            return Err(TensorError::unsupported_operation_simple(format!(
                "Unsupported ONNX data type: {}",
                proto.data_type
            )))
        }
    };

    let _dtype = onnx_dtype.to_dtype()?;

    // Extract shape
    if proto.dims.is_empty() {
        return Err(TensorError::invalid_shape_simple(
            "ONNX tensor has empty dimensions".to_string(),
        ));
    }

    let shape_vec: Vec<usize> = proto.dims.iter().map(|&d| d as usize).collect();
    let shape = Shape::from_slice(&shape_vec);

    // Extract data from raw_data
    if !proto.raw_data.is_empty() {
        // Use raw_data (most common case)
        let data_slice: &[T] = bytemuck::cast_slice(&proto.raw_data);
        let data_vec = data_slice.to_vec();

        use scirs2_core::ndarray::ArrayD;
        let array = ArrayD::from_shape_vec(shape.dims(), data_vec).map_err(|e| {
            TensorError::invalid_shape_simple(format!("Failed to create array from ONNX: {}", e))
        })?;

        return Ok(Tensor::from_array(array));
    }

    // Fallback to type-specific data fields
    // (This is less common but required by ONNX spec)
    if !proto.float_data.is_empty() {
        // Convert f32 to T if needed
        // For now, assume T is f32
        let data_slice: &[T] = bytemuck::cast_slice(&proto.float_data);
        let data_vec = data_slice.to_vec();

        use scirs2_core::ndarray::ArrayD;
        let array = ArrayD::from_shape_vec(shape.dims(), data_vec).map_err(|e| {
            TensorError::invalid_shape_simple(format!("Failed to create array from ONNX: {}", e))
        })?;

        return Ok(Tensor::from_array(array));
    }

    Err(TensorError::serialization_error_simple(
        "ONNX tensor has no data".to_string(),
    ))
}

/// Serialize an f32 tensor to ONNX format (specialized for f32)
pub fn serialize_f32_tensor_onnx(tensor: &Tensor<f32>, name: Option<String>) -> Result<Vec<u8>> {
    let proto = serialize_tensor_onnx(tensor, name)?;

    #[cfg(feature = "serialize")]
    {
        serde_json::to_vec(&proto).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "ONNX JSON serialization failed: {}",
                e
            ))
        })
    }

    #[cfg(not(feature = "serialize"))]
    {
        // Simple binary format without serde
        let mut bytes = Vec::new();

        // Write dimensions
        bytes.extend_from_slice(&(proto.dims.len() as u32).to_le_bytes());
        for dim in &proto.dims {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }

        // Write data type
        bytes.extend_from_slice(&proto.data_type.to_le_bytes());

        // Write raw data length and data
        bytes.extend_from_slice(&(proto.raw_data.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&proto.raw_data);

        Ok(bytes)
    }
}

/// Deserialize an f32 tensor from ONNX format
pub fn deserialize_f32_tensor_onnx(bytes: &[u8]) -> Result<Tensor<f32>> {
    #[cfg(feature = "serialize")]
    {
        let proto: OnnxTensorProto = serde_json::from_slice(bytes).map_err(|e| {
            TensorError::serialization_error_simple(format!(
                "ONNX JSON deserialization failed: {}",
                e
            ))
        })?;

        deserialize_tensor_onnx(&proto)
    }

    #[cfg(not(feature = "serialize"))]
    {
        // Simple binary format without serde
        let mut cursor = 0;

        // Read dimensions
        if bytes.len() < 4 {
            return Err(TensorError::serialization_error_simple(
                "ONNX data too small".to_string(),
            ));
        }

        let num_dims = u32::from_le_bytes([
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
        ]) as usize;
        cursor += 4;

        let mut dims = Vec::with_capacity(num_dims);
        for _ in 0..num_dims {
            if bytes.len() < cursor + 8 {
                return Err(TensorError::serialization_error_simple(
                    "ONNX data too small for dimensions".to_string(),
                ));
            }
            let dim = i64::from_le_bytes([
                bytes[cursor],
                bytes[cursor + 1],
                bytes[cursor + 2],
                bytes[cursor + 3],
                bytes[cursor + 4],
                bytes[cursor + 5],
                bytes[cursor + 6],
                bytes[cursor + 7],
            ]);
            dims.push(dim);
            cursor += 8;
        }

        // Read data type
        if bytes.len() < cursor + 4 {
            return Err(TensorError::serialization_error_simple(
                "ONNX data too small for data type".to_string(),
            ));
        }
        let _data_type = i32::from_le_bytes([
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
        ]);
        cursor += 4;

        // Read raw data
        if bytes.len() < cursor + 8 {
            return Err(TensorError::serialization_error_simple(
                "ONNX data too small for data length".to_string(),
            ));
        }
        let data_len = u64::from_le_bytes([
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

        if bytes.len() < cursor + data_len {
            return Err(TensorError::serialization_error_simple(
                "ONNX data too small for raw data".to_string(),
            ));
        }

        let raw_data = &bytes[cursor..cursor + data_len];
        let data_slice: &[f32] = bytemuck::cast_slice(raw_data);
        let data_vec = data_slice.to_vec();

        let shape_vec: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
        let shape = Shape::from_slice(&shape_vec);

        use scirs2_core::ndarray::ArrayD;
        let array = ArrayD::from_shape_vec(shape.dims(), data_vec).map_err(|e| {
            TensorError::invalid_shape_simple(format!("Failed to create array from ONNX: {}", e))
        })?;

        Ok(Tensor::from_array(array))
    }
}

/// Helper to get ONNX-compatible strides for a shape
pub fn onnx_strides(shape: &[usize]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    if shape.is_empty() {
        return strides;
    }

    // Row-major (C-contiguous) layout
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * (shape[i + 1] as i64);
    }

    strides
}

/// Check if tensor is ONNX-compatible (C-contiguous)
pub fn is_onnx_compatible<T>(tensor: &Tensor<T>) -> bool
where
    T: bytemuck::Pod + scirs2_core::num_traits::Float + Default + 'static,
{
    // Check if tensor is contiguous in memory (C-order)
    // For now, assume all TenfloweRS tensors are C-contiguous
    // TODO: Check actual memory layout when we support strides
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_onnx_dtype_conversion() {
        assert_eq!(
            OnnxDataType::from_dtype(DType::Float32).unwrap(),
            OnnxDataType::Float
        );
        assert_eq!(
            OnnxDataType::from_dtype(DType::Float64).unwrap(),
            OnnxDataType::Double
        );
        assert_eq!(
            OnnxDataType::from_dtype(DType::Int32).unwrap(),
            OnnxDataType::Int32
        );

        assert_eq!(OnnxDataType::Float.to_dtype().unwrap(), DType::Float32);
        assert_eq!(OnnxDataType::Double.to_dtype().unwrap(), DType::Float64);
    }

    #[test]
    fn test_onnx_element_size() {
        assert_eq!(OnnxDataType::Float.element_size(), 4);
        assert_eq!(OnnxDataType::Double.element_size(), 8);
        assert_eq!(OnnxDataType::Int32.element_size(), 4);
        assert_eq!(OnnxDataType::Int8.element_size(), 1);
    }

    #[test]
    fn test_onnx_tensor_proto_creation() {
        let proto = OnnxTensorProto::new()
            .with_name("test_tensor".to_string())
            .with_doc("Test documentation".to_string());

        assert_eq!(proto.name, "test_tensor");
        assert_eq!(proto.doc_string, "Test documentation");
    }

    #[test]
    fn test_serialize_onnx_f32() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let tensor = Tensor::from_array(data.into_dyn());

        let proto = serialize_tensor_onnx(&tensor, Some("weights".to_string())).unwrap();

        assert_eq!(proto.name, "weights");
        assert_eq!(proto.dims, vec![2, 2]);
        assert_eq!(proto.data_type, OnnxDataType::Float as i32);
        assert!(!proto.raw_data.is_empty());
        assert_eq!(proto.raw_data.len(), 4 * 4); // 4 floats * 4 bytes
    }

    #[test]
    fn test_deserialize_onnx_f32() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let tensor = Tensor::from_array(data.into_dyn());

        let proto = serialize_tensor_onnx(&tensor, None).unwrap();
        let deserialized = deserialize_tensor_onnx::<f32>(&proto).unwrap();

        assert_eq!(tensor.shape(), deserialized.shape());
        assert_eq!(tensor.data(), deserialized.data());
    }

    #[test]
    fn test_onnx_strides() {
        let shape = vec![2, 3, 4];
        let strides = onnx_strides(&shape);

        assert_eq!(strides, vec![12, 4, 1]);

        let shape2 = vec![5];
        let strides2 = onnx_strides(&shape2);
        assert_eq!(strides2, vec![1]);
    }

    #[test]
    fn test_serialize_deserialize_f32_onnx() {
        let data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let tensor = Tensor::from_array(data.into_dyn());

        let bytes = serialize_f32_tensor_onnx(&tensor, Some("test".to_string())).unwrap();
        let deserialized = deserialize_f32_tensor_onnx(&bytes).unwrap();

        assert_eq!(tensor.shape(), deserialized.shape());
        assert_eq!(tensor.data(), deserialized.data());
    }

    #[test]
    fn test_onnx_compatible_check() {
        let data = array![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_array(data.into_dyn());

        assert!(is_onnx_compatible(&tensor));
    }

    #[test]
    fn test_onnx_proto_num_elements() {
        let mut proto = OnnxTensorProto::new();
        proto.dims = vec![2, 3, 4];

        assert_eq!(proto.num_elements(), 24);

        proto.dims = vec![];
        assert_eq!(proto.num_elements(), 0);
    }
}
