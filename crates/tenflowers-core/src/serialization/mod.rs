//! Tensor serialization formats
//!
//! This module provides efficient serialization and deserialization of tensors
//! in various formats including binary, JSON, and potentially ONNX in the future.

pub mod binary;
pub mod legacy;

// Re-export new binary serialization types
pub use binary::{BinarySerializer, SerializedDType, SerializedDevice};

// Re-export legacy types for backward compatibility
pub use legacy::{
    compress_bytes, decompress_bytes, deserialize_tensor_binary, deserialize_tensor_json,
    deserialize_tensor_msgpack, load_checkpoint, load_tensor, save_checkpoint, save_tensor,
    serialize_tensor_binary, serialize_tensor_json, serialize_tensor_msgpack, SerializationFormat,
    TensorMetadata as LegacyTensorMetadata, MAGIC_NUMBER, SERIALIZATION_VERSION,
};

// Use the new binary::TensorMetadata as the primary export
pub use binary::TensorMetadata;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Tensor};
    use std::io::Cursor;

    #[test]
    fn test_roundtrip_serialization() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let original = Tensor::from_data(data.clone(), &[3, 3]).unwrap();

        // Serialize
        let mut buffer = Vec::new();
        BinarySerializer::serialize(&original, &mut buffer, None).unwrap();

        // Deserialize
        let mut cursor = Cursor::new(buffer);
        let (restored, _): (Tensor<f32>, _) = BinarySerializer::deserialize(&mut cursor).unwrap();

        // Verify
        assert_eq!(original.shape().dims(), restored.shape().dims());
        assert_eq!(original.as_slice().unwrap(), restored.as_slice().unwrap());
    }

    #[test]
    fn test_metadata_preservation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let original = Tensor::from_data(data.clone(), &[2, 2]).unwrap();

        let mut metadata = TensorMetadata::new();
        metadata.name = Some("weight_matrix".to_string());
        metadata.requires_grad = true;
        metadata.add_field("layer".to_string(), "conv1".to_string());
        metadata.add_field("param_type".to_string(), "weight".to_string());

        // Serialize with metadata
        let mut buffer = Vec::new();
        BinarySerializer::serialize(&original, &mut buffer, Some(&metadata)).unwrap();

        // Deserialize
        let mut cursor = Cursor::new(buffer);
        let (restored, meta): (Tensor<f32>, _) =
            BinarySerializer::deserialize(&mut cursor).unwrap();

        // Verify tensor data
        assert_eq!(original.as_slice().unwrap(), restored.as_slice().unwrap());

        // Verify metadata
        assert!(meta.is_some());
        let meta = meta.unwrap();
        assert_eq!(meta.name, Some("weight_matrix".to_string()));
        assert!(meta.requires_grad);
        assert_eq!(meta.fields.get("layer"), Some(&"conv1".to_string()));
        assert_eq!(meta.fields.get("param_type"), Some(&"weight".to_string()));
    }

    #[test]
    fn test_large_tensor_serialization() {
        // Create a larger tensor to test performance
        let size = 1000;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let original = Tensor::from_data(data.clone(), &[size]).unwrap();

        let mut buffer = Vec::new();
        BinarySerializer::serialize(&original, &mut buffer, None).unwrap();

        // Verify buffer size is reasonable
        let expected_size = std::mem::size_of::<f32>() * size + 64; // data + header
        assert!(buffer.len() >= expected_size);

        let mut cursor = Cursor::new(buffer);
        let (restored, _): (Tensor<f32>, _) = BinarySerializer::deserialize(&mut cursor).unwrap();

        assert_eq!(original.as_slice().unwrap(), restored.as_slice().unwrap());
    }
}
