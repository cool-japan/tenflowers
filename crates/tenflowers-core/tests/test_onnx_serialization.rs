#![cfg(feature = "serialize")]

use scirs2_core::ndarray::{array, Array, Array1};
/// ONNX Serialization Roundtrip Tests
///
/// Tests verify that tensors can be serialized to ONNX TensorProto format
/// and deserialized back without loss of data or precision.
use tenflowers_core::Tensor;

// Available functions: serialize_tensor_onnx, deserialize_tensor_onnx

#[test]
fn test_onnx_roundtrip_f32_1d() {
    let original = Tensor::from_array(array![1.0f32, 2.0, 3.0, 4.0, 5.0].into_dyn());

    // Serialize to ONNX
    let serialized = tenflowers_core::serialization_onnx::serialize_tensor_onnx(&original, None)
        .expect("Failed to serialize to ONNX");

    // Deserialize back
    let deserialized: Tensor<f32> =
        tenflowers_core::serialization_onnx::deserialize_tensor_onnx(&serialized)
            .expect("Failed to deserialize from ONNX");

    // Compare
    assert_eq!(original.shape(), deserialized.shape());
    assert_eq!(original.data(), deserialized.data());
}

#[test]
fn test_onnx_roundtrip_f32_2d() {
    let data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let original = Tensor::from_array(data.into_dyn());

    let serialized = tenflowers_core::serialization_onnx::serialize_tensor_onnx(&original, None)
        .expect("Failed to serialize to ONNX");

    let deserialized: Tensor<f32> =
        tenflowers_core::serialization_onnx::deserialize_tensor_onnx(&serialized)
            .expect("Failed to deserialize from ONNX");

    assert_eq!(original.shape(), deserialized.shape());
    assert_eq!(original.data(), deserialized.data());
}

#[test]
fn test_onnx_roundtrip_f32_3d() {
    let data = array![[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    let original = Tensor::from_array(data.into_dyn());

    let serialized = tenflowers_core::serialization_onnx::serialize_tensor_onnx(&original, None)
        .expect("Failed to serialize to ONNX");

    let deserialized: Tensor<f32> =
        tenflowers_core::serialization_onnx::deserialize_tensor_onnx(&serialized)
            .expect("Failed to deserialize from ONNX");

    assert_eq!(original.shape(), deserialized.shape());
    assert_eq!(original.data(), deserialized.data());
}

#[test]
fn test_onnx_roundtrip_f64() {
    let original = Tensor::from_array(array![1.0f64, 2.0, 3.0, 4.0].into_dyn());

    let serialized = tenflowers_core::serialization_onnx::serialize_tensor_onnx(&original, None)
        .expect("Failed to serialize to ONNX");

    let deserialized: Tensor<f64> =
        tenflowers_core::serialization_onnx::deserialize_tensor_onnx(&serialized)
            .expect("Failed to deserialize from ONNX");

    assert_eq!(original.shape(), deserialized.shape());
    assert_eq!(original.data(), deserialized.data());
}

// NOTE: ONNX serialization currently only supports Float types (f32, f64)
// i32 support would require relaxing the scirs2_core::num_traits::Float bound

#[test]
#[ignore] // Edge case: scalar tensors not yet fully supported in ONNX serialization
fn test_onnx_roundtrip_scalar() {
    // Scalar tensor (0-dimensional)
    let original = Tensor::from_array(Array::from_elem(vec![], 42.0f32).into_dyn());

    let serialized = tenflowers_core::serialization_onnx::serialize_tensor_onnx(&original, None)
        .expect("Failed to serialize to ONNX");

    let deserialized: Tensor<f32> =
        tenflowers_core::serialization_onnx::deserialize_tensor_onnx(&serialized)
            .expect("Failed to deserialize from ONNX");

    assert_eq!(original.shape(), deserialized.shape());
    assert_eq!(original.data(), deserialized.data());
}

#[test]
fn test_onnx_roundtrip_large_tensor() {
    // Large tensor to test efficiency
    let size = 1000;
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let original = Tensor::from_array(Array1::from_vec(data).into_dyn());

    let serialized = tenflowers_core::serialization_onnx::serialize_tensor_onnx(&original, None)
        .expect("Failed to serialize to ONNX");

    let deserialized: Tensor<f32> =
        tenflowers_core::serialization_onnx::deserialize_tensor_onnx(&serialized)
            .expect("Failed to deserialize from ONNX");

    assert_eq!(original.shape(), deserialized.shape());
    assert_eq!(original.data(), deserialized.data());
}

#[test]
fn test_onnx_roundtrip_negative_values() {
    let original = Tensor::from_array(array![-1.0f32, -2.5, -3.14, -100.0].into_dyn());

    let serialized = tenflowers_core::serialization_onnx::serialize_tensor_onnx(&original, None)
        .expect("Failed to serialize to ONNX");

    let deserialized: Tensor<f32> =
        tenflowers_core::serialization_onnx::deserialize_tensor_onnx(&serialized)
            .expect("Failed to deserialize from ONNX");

    assert_eq!(original.shape(), deserialized.shape());
    assert_eq!(original.data(), deserialized.data());
}

#[test]
fn test_onnx_roundtrip_special_values() {
    let original = Tensor::from_array(
        array![0.0f32, 1.0, -1.0, std::f32::consts::PI, std::f32::consts::E].into_dyn(),
    );

    let serialized = tenflowers_core::serialization_onnx::serialize_tensor_onnx(&original, None)
        .expect("Failed to serialize to ONNX");

    let deserialized: Tensor<f32> =
        tenflowers_core::serialization_onnx::deserialize_tensor_onnx(&serialized)
            .expect("Failed to deserialize from ONNX");

    // Compare with tolerance for floating point
    assert_eq!(original.shape(), deserialized.shape());
    for (a, b) in original.data().iter().zip(deserialized.data().iter()) {
        assert!((*a - *b).abs() < 1e-6);
    }
}

#[test]
fn test_onnx_dtype_preservation() {
    // Test f32
    let f32_tensor = Tensor::from_array(array![1.0f32, 2.0].into_dyn());
    let f32_proto =
        tenflowers_core::serialization_onnx::serialize_tensor_onnx(&f32_tensor, None).unwrap();
    // ONNX data_type for FLOAT is 1
    assert_eq!(f32_proto.data_type, 1);

    // Test f64
    let f64_tensor = Tensor::from_array(array![1.0f64, 2.0].into_dyn());
    let f64_proto =
        tenflowers_core::serialization_onnx::serialize_tensor_onnx(&f64_tensor, None).unwrap();
    // ONNX data_type for DOUBLE is 11
    assert_eq!(f64_proto.data_type, 11);

    // NOTE: i32 support not yet available - requires relaxing Float trait bound
}

#[test]
fn test_onnx_shape_preservation() {
    let shapes = vec![
        vec![5],          // 1D
        vec![3, 4],       // 2D
        vec![2, 3, 4],    // 3D
        vec![2, 3, 4, 5], // 4D
    ];

    for shape in shapes {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let array = scirs2_core::ndarray::ArrayD::from_shape_vec(shape.clone(), data).unwrap();
        let original = Tensor::from_array(array);

        let serialized =
            tenflowers_core::serialization_onnx::serialize_tensor_onnx(&original, None).unwrap();
        let deserialized: Tensor<f32> =
            tenflowers_core::serialization_onnx::deserialize_tensor_onnx(&serialized).unwrap();

        assert_eq!(original.shape().dims(), deserialized.shape().dims());
        assert_eq!(original.data(), deserialized.data());
    }
}

#[test]
fn test_onnx_multiple_roundtrips() {
    // Test that multiple serialization/deserialization cycles don't degrade data
    let mut tensor = Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn());

    for _ in 0..5 {
        let proto =
            tenflowers_core::serialization_onnx::serialize_tensor_onnx(&tensor, None).unwrap();
        tensor = tenflowers_core::serialization_onnx::deserialize_tensor_onnx(&proto).unwrap();
    }

    // After 5 roundtrips, should still have original values
    assert_eq!(tensor.data(), &[1.0f32, 2.0, 3.0]);
}

#[test]
#[ignore] // Edge case: empty dimensions not yet fully supported in ONNX serialization
fn test_onnx_empty_dimensions() {
    // Edge case: tensor with an empty dimension
    let array = scirs2_core::ndarray::ArrayD::<f32>::zeros(vec![0]);
    let original = Tensor::from_array(array);

    let serialized =
        tenflowers_core::serialization_onnx::serialize_tensor_onnx(&original, None).unwrap();
    let deserialized: Tensor<f32> =
        tenflowers_core::serialization_onnx::deserialize_tensor_onnx(&serialized).unwrap();

    assert_eq!(original.shape(), deserialized.shape());
    assert_eq!(original.data().len(), 0);
    assert_eq!(deserialized.data().len(), 0);
}
