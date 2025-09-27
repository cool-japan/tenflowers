/// Structured Arrays for TenfloweRS
///
/// This module provides support for structured arrays (record arrays) that can contain
/// heterogeneous data types. Each element in a structured array can have multiple named
/// fields of different types, similar to a struct or record.
///
/// Example:
/// ```rust
/// use tenflowers_core::structured_arrays::{StructuredArray, FieldDescriptor};
/// use tenflowers_core::DType;
///
/// // Define a structured array with fields: name (string), age (i32), score (f32)
/// let fields = vec![
///     FieldDescriptor::new("name", DType::String, Some(32)),
///     FieldDescriptor::new("age", DType::Int32, None),
///     FieldDescriptor::new("score", DType::Float32, None),
/// ];
///
/// let mut array = StructuredArray::new(fields, 100);
/// array.set_field_value(0, "name", "Alice".into());
/// array.set_field_value(0, "age", 25i32.into());
/// array.set_field_value(0, "score", 95.5f32.into());
/// ```
use crate::{DType, Result, Shape, TensorError};
use std::collections::HashMap;
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A field descriptor that defines the structure of a field in a structured array
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FieldDescriptor {
    /// The name of the field
    pub name: String,
    /// The data type of the field
    pub dtype: DType,
    /// Optional size for variable-length types (e.g., strings)
    pub size: Option<usize>,
    /// Byte offset within each record
    pub offset: usize,
}

impl FieldDescriptor {
    /// Create a new field descriptor
    pub fn new(name: impl Into<String>, dtype: DType, size: Option<usize>) -> Self {
        Self {
            name: name.into(),
            dtype,
            size,
            offset: 0, // Will be computed later
        }
    }

    /// Get the byte size of this field
    pub fn byte_size(&self) -> usize {
        match self.dtype {
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::UInt32 => 4,
            DType::UInt64 => 8,
            DType::Int16 => 2,
            DType::UInt16 => 2,
            DType::Int8 => 1,
            DType::UInt8 => 1,
            DType::Bool => 1,
            DType::String => self.size.unwrap_or(64), // Default string size
            _ => 8,                                   // Default for complex types
        }
    }
}

/// A value that can be stored in a structured array field
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FieldValue {
    Float32(f32),
    Float64(f64),
    Int32(i32),
    Int64(i64),
    UInt32(u32),
    UInt64(u64),
    Int16(i16),
    UInt16(u16),
    Int8(i8),
    UInt8(u8),
    Bool(bool),
    String(String),
    Bytes(Vec<u8>),
}

impl FieldValue {
    /// Get the DType of this value
    pub fn dtype(&self) -> DType {
        match self {
            FieldValue::Float32(_) => DType::Float32,
            FieldValue::Float64(_) => DType::Float64,
            FieldValue::Int32(_) => DType::Int32,
            FieldValue::Int64(_) => DType::Int64,
            FieldValue::UInt32(_) => DType::UInt32,
            FieldValue::UInt64(_) => DType::UInt64,
            FieldValue::Int16(_) => DType::Int16,
            FieldValue::UInt16(_) => DType::UInt16,
            FieldValue::Int8(_) => DType::Int8,
            FieldValue::UInt8(_) => DType::UInt8,
            FieldValue::Bool(_) => DType::Bool,
            FieldValue::String(_) => DType::String,
            FieldValue::Bytes(_) => DType::UInt8, // Byte array
        }
    }

    /// Convert value to bytes for storage
    pub fn to_bytes(&self, expected_size: usize) -> Vec<u8> {
        match self {
            FieldValue::Float32(v) => v.to_le_bytes().to_vec(),
            FieldValue::Float64(v) => v.to_le_bytes().to_vec(),
            FieldValue::Int32(v) => v.to_le_bytes().to_vec(),
            FieldValue::Int64(v) => v.to_le_bytes().to_vec(),
            FieldValue::UInt32(v) => v.to_le_bytes().to_vec(),
            FieldValue::UInt64(v) => v.to_le_bytes().to_vec(),
            FieldValue::Int16(v) => v.to_le_bytes().to_vec(),
            FieldValue::UInt16(v) => v.to_le_bytes().to_vec(),
            FieldValue::Int8(v) => vec![*v as u8],
            FieldValue::UInt8(v) => vec![*v],
            FieldValue::Bool(v) => vec![if *v { 1 } else { 0 }],
            FieldValue::String(s) => {
                let mut bytes = s.as_bytes().to_vec();
                bytes.resize(expected_size, 0); // Pad with zeros
                bytes
            }
            FieldValue::Bytes(b) => {
                let mut bytes = b.clone();
                bytes.resize(expected_size, 0); // Pad with zeros
                bytes
            }
        }
    }

    /// Create value from bytes
    pub fn from_bytes(bytes: &[u8], dtype: DType) -> Result<Self> {
        match dtype {
            DType::Float32 => {
                if bytes.len() >= 4 {
                    Ok(FieldValue::Float32(f32::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3],
                    ])))
                } else {
                    Err(TensorError::invalid_argument(
                        "Insufficient bytes for f32".to_string(),
                    ))
                }
            }
            DType::Float64 => {
                if bytes.len() >= 8 {
                    Ok(FieldValue::Float64(f64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7],
                    ])))
                } else {
                    Err(TensorError::invalid_argument(
                        "Insufficient bytes for f64".to_string(),
                    ))
                }
            }
            DType::Int32 => {
                if bytes.len() >= 4 {
                    Ok(FieldValue::Int32(i32::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3],
                    ])))
                } else {
                    Err(TensorError::invalid_argument(
                        "Insufficient bytes for i32".to_string(),
                    ))
                }
            }
            DType::Int64 => {
                if bytes.len() >= 8 {
                    Ok(FieldValue::Int64(i64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7],
                    ])))
                } else {
                    Err(TensorError::invalid_argument(
                        "Insufficient bytes for i64".to_string(),
                    ))
                }
            }
            DType::UInt32 => {
                if bytes.len() >= 4 {
                    Ok(FieldValue::UInt32(u32::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3],
                    ])))
                } else {
                    Err(TensorError::invalid_argument(
                        "Insufficient bytes for u32".to_string(),
                    ))
                }
            }
            DType::UInt64 => {
                if bytes.len() >= 8 {
                    Ok(FieldValue::UInt64(u64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7],
                    ])))
                } else {
                    Err(TensorError::invalid_argument(
                        "Insufficient bytes for u64".to_string(),
                    ))
                }
            }
            DType::Int16 => {
                if bytes.len() >= 2 {
                    Ok(FieldValue::Int16(i16::from_le_bytes([bytes[0], bytes[1]])))
                } else {
                    Err(TensorError::invalid_argument(
                        "Insufficient bytes for i16".to_string(),
                    ))
                }
            }
            DType::UInt16 => {
                if bytes.len() >= 2 {
                    Ok(FieldValue::UInt16(u16::from_le_bytes([bytes[0], bytes[1]])))
                } else {
                    Err(TensorError::invalid_argument(
                        "Insufficient bytes for u16".to_string(),
                    ))
                }
            }
            DType::Int8 => {
                if !bytes.is_empty() {
                    Ok(FieldValue::Int8(bytes[0] as i8))
                } else {
                    Err(TensorError::invalid_argument(
                        "Insufficient bytes for i8".to_string(),
                    ))
                }
            }
            DType::UInt8 => {
                if !bytes.is_empty() {
                    Ok(FieldValue::UInt8(bytes[0]))
                } else {
                    Err(TensorError::invalid_argument(
                        "Insufficient bytes for u8".to_string(),
                    ))
                }
            }
            DType::Bool => {
                if !bytes.is_empty() {
                    Ok(FieldValue::Bool(bytes[0] != 0))
                } else {
                    Err(TensorError::invalid_argument(
                        "Insufficient bytes for bool".to_string(),
                    ))
                }
            }
            DType::String => {
                let null_pos = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                let string_bytes = &bytes[..null_pos];
                let s = String::from_utf8_lossy(string_bytes).to_string();
                Ok(FieldValue::String(s))
            }
            _ => Err(TensorError::not_implemented_simple(
                "Unsupported dtype for structured arrays".to_string(),
            )),
        }
    }
}

impl From<f32> for FieldValue {
    fn from(v: f32) -> Self {
        FieldValue::Float32(v)
    }
}

impl From<f64> for FieldValue {
    fn from(v: f64) -> Self {
        FieldValue::Float64(v)
    }
}

impl From<i32> for FieldValue {
    fn from(v: i32) -> Self {
        FieldValue::Int32(v)
    }
}

impl From<i64> for FieldValue {
    fn from(v: i64) -> Self {
        FieldValue::Int64(v)
    }
}

impl From<u32> for FieldValue {
    fn from(v: u32) -> Self {
        FieldValue::UInt32(v)
    }
}

impl From<u64> for FieldValue {
    fn from(v: u64) -> Self {
        FieldValue::UInt64(v)
    }
}

impl From<i16> for FieldValue {
    fn from(v: i16) -> Self {
        FieldValue::Int16(v)
    }
}

impl From<u16> for FieldValue {
    fn from(v: u16) -> Self {
        FieldValue::UInt16(v)
    }
}

impl From<i8> for FieldValue {
    fn from(v: i8) -> Self {
        FieldValue::Int8(v)
    }
}

impl From<u8> for FieldValue {
    fn from(v: u8) -> Self {
        FieldValue::UInt8(v)
    }
}

impl From<bool> for FieldValue {
    fn from(v: bool) -> Self {
        FieldValue::Bool(v)
    }
}

impl From<String> for FieldValue {
    fn from(v: String) -> Self {
        FieldValue::String(v)
    }
}

impl From<&str> for FieldValue {
    fn from(v: &str) -> Self {
        FieldValue::String(v.to_string())
    }
}

impl From<Vec<u8>> for FieldValue {
    fn from(v: Vec<u8>) -> Self {
        FieldValue::Bytes(v)
    }
}

/// A structured array that can hold records with multiple named fields of different types
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StructuredArray {
    /// Field descriptors defining the structure
    fields: Vec<FieldDescriptor>,
    /// Mapping from field names to field indices
    field_map: HashMap<String, usize>,
    /// Total size of each record in bytes
    record_size: usize,
    /// Raw data storage (all records concatenated)
    data: Vec<u8>,
    /// Number of records
    len: usize,
    /// Shape of the array (for multi-dimensional structured arrays)
    shape: Shape,
}

impl StructuredArray {
    /// Create a new structured array with the given fields and capacity
    pub fn new(mut fields: Vec<FieldDescriptor>, len: usize) -> Self {
        // Compute offsets and total record size
        let mut offset = 0;
        for field in &mut fields {
            field.offset = offset;
            offset += field.byte_size();
        }
        let record_size = offset;

        // Create field name mapping
        let field_map: HashMap<String, usize> = fields
            .iter()
            .enumerate()
            .map(|(i, field)| (field.name.clone(), i))
            .collect();

        // Initialize data storage
        let data = vec![0u8; record_size * len];

        Self {
            fields,
            field_map,
            record_size,
            data,
            len,
            shape: Shape::from_slice(&[len]),
        }
    }

    /// Create a new multi-dimensional structured array
    pub fn with_shape(fields: Vec<FieldDescriptor>, shape: &[usize]) -> Self {
        let total_len = shape.iter().product::<usize>();
        let mut array = Self::new(fields, total_len);
        array.shape = Shape::from_slice(shape);
        array
    }

    /// Get the number of records
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the shape of the array
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the field descriptors
    pub fn fields(&self) -> &[FieldDescriptor] {
        &self.fields
    }

    /// Get a field descriptor by name
    pub fn field(&self, name: &str) -> Option<&FieldDescriptor> {
        self.field_map.get(name).map(|&i| &self.fields[i])
    }

    /// Get field names
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// Set a field value for a specific record
    pub fn set_field_value(
        &mut self,
        record_idx: usize,
        field_name: &str,
        value: FieldValue,
    ) -> Result<()> {
        if record_idx >= self.len {
            return Err(TensorError::invalid_argument(format!(
                "Record index {record_idx} out of bounds"
            )));
        }

        let field_idx = self
            .field_map
            .get(field_name)
            .ok_or_else(|| TensorError::invalid_argument(format!("Unknown field: {field_name}")))?;

        let field = &self.fields[*field_idx];

        // Validate type compatibility
        if value.dtype() != field.dtype && field.dtype != DType::String {
            return Err(TensorError::invalid_argument(format!(
                "Type mismatch: expected {:?}, got {:?}",
                field.dtype,
                value.dtype()
            )));
        }

        // Convert value to bytes and store
        let value_bytes = value.to_bytes(field.byte_size());
        let record_start = record_idx * self.record_size;
        let field_start = record_start + field.offset;
        let field_end = field_start + field.byte_size();

        self.data[field_start..field_end].copy_from_slice(&value_bytes);
        Ok(())
    }

    /// Get a field value for a specific record
    pub fn get_field_value(&self, record_idx: usize, field_name: &str) -> Result<FieldValue> {
        if record_idx >= self.len {
            return Err(TensorError::invalid_argument(format!(
                "Record index {record_idx} out of bounds"
            )));
        }

        let field_idx = self
            .field_map
            .get(field_name)
            .ok_or_else(|| TensorError::invalid_argument(format!("Unknown field: {field_name}")))?;

        let field = &self.fields[*field_idx];
        let record_start = record_idx * self.record_size;
        let field_start = record_start + field.offset;
        let field_end = field_start + field.byte_size();

        let field_bytes = &self.data[field_start..field_end];
        FieldValue::from_bytes(field_bytes, field.dtype)
    }

    /// Get all field values for a specific record as a map
    pub fn get_record(&self, record_idx: usize) -> Result<HashMap<String, FieldValue>> {
        if record_idx >= self.len {
            return Err(TensorError::invalid_argument(format!(
                "Record index {record_idx} out of bounds"
            )));
        }

        let mut record = HashMap::new();
        for field in &self.fields {
            let value = self.get_field_value(record_idx, &field.name)?;
            record.insert(field.name.clone(), value);
        }
        Ok(record)
    }

    /// Set all field values for a specific record
    pub fn set_record(
        &mut self,
        record_idx: usize,
        values: HashMap<String, FieldValue>,
    ) -> Result<()> {
        for (field_name, value) in values {
            self.set_field_value(record_idx, &field_name, value)?;
        }
        Ok(())
    }

    /// Extract a column (field) as a vector of values
    pub fn get_column(&self, field_name: &str) -> Result<Vec<FieldValue>> {
        let mut values = Vec::with_capacity(self.len);
        for i in 0..self.len {
            values.push(self.get_field_value(i, field_name)?);
        }
        Ok(values)
    }

    /// Get a slice of records
    pub fn slice(&self, start: usize, end: usize) -> Result<StructuredArray> {
        if start >= self.len || end > self.len || start >= end {
            return Err(TensorError::invalid_argument(
                "Invalid slice range".to_string(),
            ));
        }

        let slice_len = end - start;
        let mut sliced = StructuredArray::new(self.fields.clone(), slice_len);

        let start_byte = start * self.record_size;
        let end_byte = end * self.record_size;
        sliced
            .data
            .copy_from_slice(&self.data[start_byte..end_byte]);

        Ok(sliced)
    }

    /// Resize the array (add or remove records)
    pub fn resize(&mut self, new_len: usize) {
        if new_len != self.len {
            self.data.resize(new_len * self.record_size, 0);
            self.len = new_len;
            self.shape = Shape::from_slice(&[new_len]);
        }
    }

    /// Get raw data as bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Create from raw bytes (unsafe - no validation)
    pub fn from_bytes(fields: Vec<FieldDescriptor>, data: Vec<u8>, len: usize) -> Result<Self> {
        let mut field_map = HashMap::new();
        let mut offset = 0;

        let mut corrected_fields = fields;
        for (i, field) in corrected_fields.iter_mut().enumerate() {
            field.offset = offset;
            offset += field.byte_size();
            field_map.insert(field.name.clone(), i);
        }

        let record_size = offset;

        if data.len() != record_size * len {
            return Err(TensorError::invalid_argument(
                "Data size doesn't match expected record structure".to_string(),
            ));
        }

        Ok(Self {
            fields: corrected_fields,
            field_map,
            record_size,
            data,
            len,
            shape: Shape::from_slice(&[len]),
        })
    }
}

impl fmt::Display for StructuredArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "StructuredArray(len={}, fields=[{}])",
            self.len,
            self.fields
                .iter()
                .map(|f| format!("{}:{:?}", f.name, f.dtype))
                .collect::<Vec<_>>()
                .join(", ")
        )?;

        // Show first few records
        let show_count = std::cmp::min(5, self.len);
        for i in 0..show_count {
            if let Ok(record) = self.get_record(i) {
                write!(f, "  [{i}]: ")?;
                let field_strs: Vec<String> = self
                    .fields
                    .iter()
                    .map(|field| {
                        if let Some(value) = record.get(&field.name) {
                            format!("{}={:?}", field.name, value)
                        } else {
                            format!("{}=<missing>", field.name)
                        }
                    })
                    .collect();
                writeln!(f, "{{{}}}", field_strs.join(", "))?;
            }
        }

        if self.len > show_count {
            writeln!(f, "  ... ({} more records)", self.len - show_count)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_descriptor() {
        let field = FieldDescriptor::new("test", DType::Float32, None);
        assert_eq!(field.name, "test");
        assert_eq!(field.dtype, DType::Float32);
        assert_eq!(field.byte_size(), 4);
    }

    #[test]
    fn test_field_value_conversions() {
        let value = FieldValue::Float32(3.14);
        assert_eq!(value.dtype(), DType::Float32);

        let bytes = value.to_bytes(4);
        assert_eq!(bytes.len(), 4);

        let recovered = FieldValue::from_bytes(&bytes, DType::Float32).unwrap();
        if let FieldValue::Float32(v) = recovered {
            assert!((v - 3.14).abs() < 1e-6);
        } else {
            panic!("Wrong type recovered");
        }
    }

    #[test]
    fn test_structured_array_creation() {
        let fields = vec![
            FieldDescriptor::new("id", DType::Int32, None),
            FieldDescriptor::new("score", DType::Float32, None),
            FieldDescriptor::new("name", DType::String, Some(16)),
        ];

        let array = StructuredArray::new(fields, 10);
        assert_eq!(array.len(), 10);
        assert_eq!(array.fields().len(), 3);
        assert!(array.field("id").is_some());
        assert!(array.field("unknown").is_none());
    }

    #[test]
    fn test_field_operations() {
        let fields = vec![
            FieldDescriptor::new("id", DType::Int32, None),
            FieldDescriptor::new("score", DType::Float32, None),
            FieldDescriptor::new("name", DType::String, Some(16)),
        ];

        let mut array = StructuredArray::new(fields, 2);

        // Set values
        array.set_field_value(0, "id", 42i32.into()).unwrap();
        array.set_field_value(0, "score", 95.5f32.into()).unwrap();
        array.set_field_value(0, "name", "Alice".into()).unwrap();

        array.set_field_value(1, "id", 43i32.into()).unwrap();
        array.set_field_value(1, "score", 87.2f32.into()).unwrap();
        array.set_field_value(1, "name", "Bob".into()).unwrap();

        // Get values
        let id0 = array.get_field_value(0, "id").unwrap();
        if let FieldValue::Int32(v) = id0 {
            assert_eq!(v, 42);
        } else {
            panic!("Wrong type");
        }

        let name1 = array.get_field_value(1, "name").unwrap();
        if let FieldValue::String(s) = name1 {
            assert_eq!(s, "Bob");
        } else {
            panic!("Wrong type");
        }
    }

    #[test]
    fn test_record_operations() {
        let fields = vec![
            FieldDescriptor::new("x", DType::Float32, None),
            FieldDescriptor::new("y", DType::Float32, None),
        ];

        let mut array = StructuredArray::new(fields, 1);

        let mut record = HashMap::new();
        record.insert("x".to_string(), 1.0f32.into());
        record.insert("y".to_string(), 2.0f32.into());

        array.set_record(0, record).unwrap();

        let retrieved = array.get_record(0).unwrap();
        assert_eq!(retrieved.len(), 2);

        if let Some(FieldValue::Float32(x)) = retrieved.get("x") {
            assert_eq!(*x, 1.0);
        } else {
            panic!("Wrong value for x");
        }
    }

    #[test]
    fn test_column_extraction() {
        let fields = vec![FieldDescriptor::new("values", DType::Float32, None)];

        let mut array = StructuredArray::new(fields, 3);

        array.set_field_value(0, "values", 1.0f32.into()).unwrap();
        array.set_field_value(1, "values", 2.0f32.into()).unwrap();
        array.set_field_value(2, "values", 3.0f32.into()).unwrap();

        let column = array.get_column("values").unwrap();
        assert_eq!(column.len(), 3);

        if let FieldValue::Float32(v) = &column[1] {
            assert_eq!(*v, 2.0);
        } else {
            panic!("Wrong type");
        }
    }

    #[test]
    fn test_array_slice() {
        let fields = vec![FieldDescriptor::new("id", DType::Int32, None)];

        let mut array = StructuredArray::new(fields, 5);

        for i in 0..5 {
            array.set_field_value(i, "id", (i as i32).into()).unwrap();
        }

        let slice = array.slice(1, 4).unwrap();
        assert_eq!(slice.len(), 3);

        let id = slice.get_field_value(0, "id").unwrap();
        if let FieldValue::Int32(v) = id {
            assert_eq!(v, 1); // First element of slice should be original index 1
        } else {
            panic!("Wrong type");
        }
    }
}
