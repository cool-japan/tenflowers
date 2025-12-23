use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use tenflowers_core::ops::registry::{ArgDef, Kernel, OpDef, OpVersion, OP_REGISTRY};
use tenflowers_core::{DType, Device, Shape, Tensor};

/// Test that built-in operations are properly registered
#[test]
fn test_builtin_ops_registered() {
    // Test that core operations are registered
    assert!(OP_REGISTRY.get_op("Add").is_some());
    assert!(OP_REGISTRY.get_op("MatMul").is_some());
    assert!(OP_REGISTRY.get_op("ReLU").is_some());

    // Verify operation details
    let add_op = OP_REGISTRY.get_op("Add").unwrap();
    assert_eq!(add_op.name, "Add");
    assert_eq!(add_op.version, OpVersion::new(1, 0, 0));
    assert_eq!(add_op.inputs.len(), 2);
    assert_eq!(add_op.outputs.len(), 1);
    assert!(!add_op.deprecated);
}

/// Test that kernels are registered for basic operations
#[test]
fn test_builtin_kernels_registered() {
    // Test that kernels exist for common device/dtype combinations
    let devices = [Device::Cpu];
    let dtypes = [DType::Float32, DType::Float64, DType::Int32, DType::Int64];

    for &device in &devices {
        for &dtype in &dtypes {
            // Test Add kernel
            let add_kernel = OP_REGISTRY.get_kernel("Add", device, dtype);
            assert!(
                add_kernel.is_some(),
                "Add kernel missing for {:?}/{:?}",
                device,
                dtype
            );

            // Test kernel properties
            let kernel = add_kernel.unwrap();
            assert_eq!(kernel.device(), device);
            assert_eq!(kernel.dtype(), dtype);
        }
    }
}

/// Test kernel execution with actual computation
#[test]
fn test_kernel_execution_f32() {
    let kernel = OP_REGISTRY
        .get_kernel("Add", Device::Cpu, DType::Float32)
        .expect("Add kernel for f32 should be available");

    // Create test tensors
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();

    // Prepare inputs as Any references
    let inputs: Vec<&dyn Any> = vec![&a, &b];
    let attrs = HashMap::new();

    // Execute kernel
    let results = kernel.compute(&inputs, &attrs).unwrap();
    assert_eq!(results.len(), 1);

    // Extract result and verify
    let result = results[0]
        .downcast_ref::<Tensor<f32>>()
        .expect("Result should be f32 tensor");

    let expected = vec![5.0, 7.0, 9.0];
    let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage else {
        panic!("Expected CPU storage in test");
    };
    assert_eq!(arr.as_slice().unwrap(), &expected);
}

/// Test kernel execution with f64
#[test]
fn test_kernel_execution_f64() {
    let kernel = OP_REGISTRY
        .get_kernel("Add", Device::Cpu, DType::Float64)
        .expect("Add kernel for f64 should be available");

    let a = Tensor::<f64>::from_vec(vec![1.5, 2.5, 3.5], &[3]).unwrap();
    let b = Tensor::<f64>::from_vec(vec![0.5, 1.5, 2.5], &[3]).unwrap();

    let inputs: Vec<&dyn Any> = vec![&a, &b];
    let attrs = HashMap::new();

    let results = kernel.compute(&inputs, &attrs).unwrap();
    let result = results[0].downcast_ref::<Tensor<f64>>().unwrap();

    let expected = vec![2.0, 4.0, 6.0];
    let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage else {
        panic!("Expected CPU storage in test");
    };
    assert_eq!(arr.as_slice().unwrap(), &expected);
}

/// Test kernel execution with integer types
#[test]
fn test_kernel_execution_i32() {
    let kernel = OP_REGISTRY
        .get_kernel("Add", Device::Cpu, DType::Int32)
        .expect("Add kernel for i32 should be available");

    let a = Tensor::<i32>::from_vec(vec![10, 20, 30], &[3]).unwrap();
    let b = Tensor::<i32>::from_vec(vec![1, 2, 3], &[3]).unwrap();

    let inputs: Vec<&dyn Any> = vec![&a, &b];
    let attrs = HashMap::new();

    let results = kernel.compute(&inputs, &attrs).unwrap();
    let result = results[0].downcast_ref::<Tensor<i32>>().unwrap();

    let expected = vec![11, 22, 33];
    let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage else {
        panic!("Expected CPU storage in test");
    };
    assert_eq!(arr.as_slice().unwrap(), &expected);
}

/// Test kernel execution with extended dtype support
#[test]
fn test_kernel_execution_i8() {
    let kernel = OP_REGISTRY
        .get_kernel("Add", Device::Cpu, DType::Int8)
        .expect("Add kernel for i8 should be available");

    let a = Tensor::<i8>::from_vec(vec![10, 20, 30], &[3]).unwrap();
    let b = Tensor::<i8>::from_vec(vec![5, 10, 15], &[3]).unwrap();

    let inputs: Vec<&dyn Any> = vec![&a, &b];
    let attrs = HashMap::new();

    let results = kernel.compute(&inputs, &attrs).unwrap();
    let result = results[0].downcast_ref::<Tensor<i8>>().unwrap();

    let expected = vec![15i8, 30i8, 45i8];
    let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage else {
        panic!("Expected CPU storage in test");
    };
    assert_eq!(arr.as_slice().unwrap(), &expected);
}

/// Test kernel execution with u8 type
#[test]
fn test_kernel_execution_u8() {
    let kernel = OP_REGISTRY
        .get_kernel("Add", Device::Cpu, DType::UInt8)
        .expect("Add kernel for u8 should be available");

    let a = Tensor::<u8>::from_vec(vec![100, 150, 200], &[3]).unwrap();
    let b = Tensor::<u8>::from_vec(vec![10, 20, 30], &[3]).unwrap();

    let inputs: Vec<&dyn Any> = vec![&a, &b];
    let attrs = HashMap::new();

    let results = kernel.compute(&inputs, &attrs).unwrap();
    let result = results[0].downcast_ref::<Tensor<u8>>().unwrap();

    let expected = vec![110u8, 170u8, 230u8];
    let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage else {
        panic!("Expected CPU storage in test");
    };
    assert_eq!(arr.as_slice().unwrap(), &expected);
}

/// Test error handling for invalid inputs
#[test]
fn test_kernel_error_handling() {
    let kernel = OP_REGISTRY
        .get_kernel("Add", Device::Cpu, DType::Float32)
        .expect("Add kernel for f32 should be available");

    // Test with wrong number of inputs
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let inputs_wrong_count: Vec<&dyn Any> = vec![&a]; // Only 1 input instead of 2
    let attrs = HashMap::new();

    let result = kernel.compute(&inputs_wrong_count, &attrs);
    assert!(result.is_err());

    // Test with wrong type
    let b = Tensor::<i32>::from_vec(vec![1, 2], &[2]).unwrap();
    let inputs_wrong_type: Vec<&dyn Any> = vec![&a, &b]; // Mixed types

    let result = kernel.compute(&inputs_wrong_type, &attrs);
    assert!(result.is_err());
}

/// Test operation versioning
#[test]
fn test_operation_versioning() {
    // Test getting specific version
    let add_v1 = OP_REGISTRY.get_op_version("Add", &OpVersion::new(1, 0, 0));
    assert!(add_v1.is_some());

    // Test version compatibility
    let compatible = OP_REGISTRY.get_op_compatible("Add", &OpVersion::new(1, 0, 0));
    assert!(compatible.is_some());

    // Test latest version retrieval
    let latest = OP_REGISTRY.get_latest_version("Add");
    assert!(latest.is_some());
    assert_eq!(latest.unwrap(), OpVersion::new(1, 0, 0));
}

/// Test operation listing
#[test]
fn test_operation_listing() {
    let ops = OP_REGISTRY.list_ops();
    assert!(ops.contains(&"Add".to_string()));
    assert!(ops.contains(&"MatMul".to_string()));
    assert!(ops.contains(&"ReLU".to_string()));

    // Test version listing for specific operation
    let add_versions = OP_REGISTRY.list_op_versions("Add");
    assert!(!add_versions.is_empty());
    assert!(add_versions.contains(&OpVersion::new(1, 0, 0)));
}

/// Test kernel registration for multiple operations
#[test]
fn test_multiple_operations_registered() {
    // Only test operations that are actually registered in the built-in registry
    let operations = ["Add", "MatMul", "ReLU"];

    for &op_name in &operations {
        // Test that operation is registered
        assert!(
            OP_REGISTRY.get_op(op_name).is_some(),
            "Operation {} not registered",
            op_name
        );

        // Test that kernels exist for basic types
        for &dtype in &[DType::Float32, DType::Float64] {
            let kernel = OP_REGISTRY.get_kernel(op_name, Device::Cpu, dtype);
            assert!(
                kernel.is_some(),
                "Kernel for {} with {:?} not found",
                op_name,
                dtype
            );
        }
    }

    // Note: Kernels for Sub, Mul, Div are registered but can be accessed through the "Add" kernel
    // since the SimpleKernel implementation handles multiple operation types
}

/// Test shape inference function
#[test]
fn test_shape_inference() {
    let add_op = OP_REGISTRY.get_op("Add").unwrap();

    if let Some(shape_fn) = add_op.shape_fn {
        let shape_a = Shape::from_slice(&[2, 3]);
        let shape_b = Shape::from_slice(&[2, 3]);
        let inputs = [&shape_a, &shape_b];
        let attrs = HashMap::new();

        let output_shapes = shape_fn(&inputs, &attrs).unwrap();
        assert_eq!(output_shapes.len(), 1);
        assert_eq!(output_shapes[0].dims(), &[2, 3]);
    }
}

/// Test that kernels handle broadcasting correctly through the registry
#[test]
fn test_registry_broadcasting() {
    let kernel = OP_REGISTRY
        .get_kernel("Add", Device::Cpu, DType::Float32)
        .expect("Add kernel should be available");

    // Create tensors with different but broadcastable shapes
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![10.0, 20.0], &[1, 2]).unwrap();

    let inputs: Vec<&dyn Any> = vec![&a, &b];
    let attrs = HashMap::new();

    let results = kernel.compute(&inputs, &attrs).unwrap();
    let result = results[0].downcast_ref::<Tensor<f32>>().unwrap();

    // Should broadcast to [3, 2]
    assert_eq!(result.shape().dims(), &[3, 2]);

    // Expected: [[11, 21], [12, 22], [13, 23]]
    let expected = vec![11.0, 21.0, 12.0, 22.0, 13.0, 23.0];
    let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage else {
        panic!("Expected CPU storage in test");
    };
    assert_eq!(arr.as_slice().unwrap(), &expected);
}

/// Test performance with larger tensors through registry
#[test]
fn test_registry_performance() {
    let kernel = OP_REGISTRY
        .get_kernel("Add", Device::Cpu, DType::Float32)
        .expect("Add kernel should be available");

    // Create larger tensors
    let size = 10000;
    let data_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

    let a = Tensor::<f32>::from_vec(data_a, &[size]).unwrap();
    let b = Tensor::<f32>::from_vec(data_b, &[size]).unwrap();

    let inputs: Vec<&dyn Any> = vec![&a, &b];
    let attrs = HashMap::new();

    // Time the operation
    let start = std::time::Instant::now();
    let results = kernel.compute(&inputs, &attrs).unwrap();
    let duration = start.elapsed();

    println!(
        "Registry Add operation on {} elements took: {:?}",
        size, duration
    );

    // Verify correctness
    let result = results[0].downcast_ref::<Tensor<f32>>().unwrap();
    let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage else {
        panic!("Expected CPU storage in test");
    };
    let slice = arr.as_slice().unwrap();
    assert_eq!(slice[0], 1.0); // 0 + 1
    assert_eq!(slice[100], 201.0); // 100 + 101
    assert_eq!(slice[size - 1], (2 * size - 1) as f32); // (size-1) + size
}
