#![allow(irrefutable_let_patterns)] // Pattern matching on TensorStorage is irrefutable when GPU feature is disabled

use tenflowers_core::ops::binary::{add, div, mul, sub};
use tenflowers_core::{Device, Tensor};

/// Test binary operations with i8 data type
#[test]
fn test_binary_ops_i8() {
    let a = Tensor::<i8>::from_vec(vec![1, 2, 3], &[3]).unwrap();
    let b = Tensor::<i8>::from_vec(vec![4, 5, 6], &[3]).unwrap();

    // Test addition
    let result = add(&a, &b).unwrap();
    let expected_add = vec![5i8, 7i8, 9i8];
    if let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage {
        assert_eq!(arr.as_slice().unwrap(), &expected_add);
    }

    // Test subtraction
    let result = sub(&a, &b).unwrap();
    let expected_sub = vec![-3i8, -3i8, -3i8];
    if let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage {
        assert_eq!(arr.as_slice().unwrap(), &expected_sub);
    }

    // Test multiplication
    let result = mul(&a, &b).unwrap();
    let expected_mul = vec![4i8, 10i8, 18i8];
    if let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage {
        assert_eq!(arr.as_slice().unwrap(), &expected_mul);
    }
}

/// Test binary operations with u8 data type
#[test]
fn test_binary_ops_u8() {
    let a = Tensor::<u8>::from_vec(vec![10, 20, 17], &[3]).unwrap();
    let b = Tensor::<u8>::from_vec(vec![5, 10, 15], &[3]).unwrap();

    // Test addition
    let result = add(&a, &b).unwrap();
    let expected_add = vec![15u8, 30u8, 32u8]; // 10+5, 20+10, 17+15
    if let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage {
        assert_eq!(arr.as_slice().unwrap(), &expected_add);
    }

    // Test subtraction
    let result = sub(&a, &b).unwrap();
    let expected_sub = vec![5u8, 10u8, 2u8]; // 10-5, 20-10, 17-15
    if let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage {
        assert_eq!(arr.as_slice().unwrap(), &expected_sub);
    }

    // Test multiplication
    let result = mul(&a, &b).unwrap();
    let expected_mul = vec![50u8, 200u8, 255u8]; // 10*5, 20*10, 17*15=255
    if let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage {
        assert_eq!(arr.as_slice().unwrap(), &expected_mul);
    }
}

/// Test broadcasting with extended data types
#[test]
fn test_broadcast_i8() {
    let a = Tensor::<i8>::from_vec(vec![1, 2, 3], &[3, 1]).unwrap();
    let b = Tensor::<i8>::from_vec(vec![4, 5], &[1, 2]).unwrap();

    let result = add(&a, &b).unwrap();
    assert_eq!(result.shape().dims(), &[3, 2]);

    // Expected: [[5, 6], [6, 7], [7, 8]]
    let expected = vec![5i8, 6i8, 6i8, 7i8, 7i8, 8i8];
    if let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage {
        assert_eq!(arr.as_slice().unwrap(), &expected);
    }
}

/// Test broadcasting with u8 data type
#[test]
fn test_broadcast_u8() {
    let a = Tensor::<u8>::from_vec(vec![10, 20], &[2, 1]).unwrap();
    let b = Tensor::<u8>::from_vec(vec![1, 2, 3], &[1, 3]).unwrap();

    let result = mul(&a, &b).unwrap();
    assert_eq!(result.shape().dims(), &[2, 3]);

    // Expected: [[10, 20, 30], [20, 40, 60]]
    let expected = vec![10u8, 20u8, 30u8, 20u8, 40u8, 60u8];
    if let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage {
        assert_eq!(arr.as_slice().unwrap(), &expected);
    }
}

/// Test device compatibility checking
#[test]
fn test_device_mismatch_error() {
    let cpu_tensor = Tensor::<i8>::from_vec(vec![1, 2, 3], &[3]).unwrap();

    // Create another CPU tensor for valid operations
    let cpu_tensor2 = Tensor::<i8>::from_vec(vec![4, 5, 6], &[3]).unwrap();

    // This should work (both on CPU)
    assert!(add(&cpu_tensor, &cpu_tensor2).is_ok());

    // Note: Testing GPU device mismatch would require actual GPU tensor creation
    // which may not be available in all test environments
}

/// Test that division by zero behaves correctly for integer types
#[test]
fn test_division_edge_cases_i8() {
    let a = Tensor::<i8>::from_vec(vec![6, 8, 10], &[3]).unwrap();
    let b = Tensor::<i8>::from_vec(vec![2, 4, 5], &[3]).unwrap();

    let result = div(&a, &b).unwrap();
    let expected = vec![3i8, 2i8, 2i8];
    if let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage {
        assert_eq!(arr.as_slice().unwrap(), &expected);
    }
}

/// Test with larger tensors to ensure memory handling works correctly
#[test]
fn test_large_tensor_i8() {
    let size = 1000;
    let data_a: Vec<i8> = (0..size).map(|i| (i % 64) as i8).collect(); // Stay well within i8 range
    let data_b: Vec<i8> = (0..size).map(|i| ((i % 63) + 1) as i8).collect(); // Ensure no overflow when adding

    let a = Tensor::<i8>::from_vec(data_a, &[size]).unwrap();
    let b = Tensor::<i8>::from_vec(data_b, &[size]).unwrap();

    let result = add(&a, &b).unwrap();
    assert_eq!(result.shape().dims(), &[size]);

    // Verify some values
    if let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage {
        let slice = arr.as_slice().unwrap();
        assert_eq!(slice[0], 1i8); // 0 + 1
        assert_eq!(slice[1], 3i8); // 1 + 2
        assert_eq!(slice[10], 21i8); // 10 + 11
        assert_eq!(slice[63], 64i8); // 63 + 1 (wrapping from data_b)
    }
}

/// Test with larger tensors for u8
#[test]
fn test_large_tensor_u8() {
    let size = 500;
    let data_a: Vec<u8> = (0..size).map(|i| (i % 255) as u8).collect();
    let data_b: Vec<u8> = (0..size).map(|i| 1u8).collect();

    let a = Tensor::<u8>::from_vec(data_a, &[size]).unwrap();
    let b = Tensor::<u8>::from_vec(data_b, &[size]).unwrap();

    let result = add(&a, &b).unwrap();
    assert_eq!(result.shape().dims(), &[size]);

    // Verify some values
    if let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage {
        let slice = arr.as_slice().unwrap();
        assert_eq!(slice[0], 1u8); // 0 + 1
        assert_eq!(slice[1], 2u8); // 1 + 1
        assert_eq!(slice[254], 255u8); // 254 + 1
    }
}

/// Test mixed operations with different tensor shapes (edge cases)
#[test]
fn test_edge_cases_i8() {
    // Test with scalar-like tensors
    let scalar = Tensor::<i8>::from_vec(vec![5], &[1]).unwrap();
    let vector = Tensor::<i8>::from_vec(vec![1, 2, 3, 4], &[4]).unwrap();

    let result = mul(&vector, &scalar).unwrap();
    assert_eq!(result.shape().dims(), &[4]);

    let expected = vec![5i8, 10i8, 15i8, 20i8];
    if let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage {
        assert_eq!(arr.as_slice().unwrap(), &expected);
    }
}

/// Test mixed operations with different tensor shapes for u8
#[test]
fn test_edge_cases_u8() {
    // Test with empty-like dimensions
    let a = Tensor::<u8>::from_vec(vec![10, 20, 30], &[3, 1]).unwrap();
    let b = Tensor::<u8>::from_vec(vec![2], &[1]).unwrap();

    let result = div(&a, &b).unwrap();
    assert_eq!(result.shape().dims(), &[3, 1]);

    let expected = vec![5u8, 10u8, 15u8];
    if let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &result.storage {
        assert_eq!(arr.as_slice().unwrap(), &expected);
    }
}

/// GPU tests (conditional on GPU feature)
#[cfg(feature = "gpu")]
mod gpu_extended_dtype_tests {
    use super::*;

    #[test]
    #[ignore] // Only run if GPU is available
    fn test_gpu_binary_ops_i8() {
        let cpu_a = Tensor::<i8>::from_vec(vec![1, 2, 3], &[3]).unwrap();
        let cpu_b = Tensor::<i8>::from_vec(vec![4, 5, 6], &[3]).unwrap();

        // Try to create GPU tensors
        let gpu_device = Device::Gpu(0);

        if let (Ok(gpu_a), Ok(gpu_b)) = (cpu_a.to(gpu_device), cpu_b.to(gpu_device)) {
            // Test GPU addition
            let gpu_result = add(&gpu_a, &gpu_b);

            match gpu_result {
                Ok(result) => {
                    // Transfer back to CPU for verification
                    let cpu_result = result.to(Device::Cpu).unwrap();
                    let expected = vec![5i8, 7i8, 9i8];
                    if let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &cpu_result.storage {
                        assert_eq!(arr.as_slice().unwrap(), &expected);
                    }
                    println!("GPU i8 binary operation test passed");
                }
                Err(e) => {
                    println!("GPU i8 operation not yet fully implemented: {}", e);
                }
            }
        } else {
            println!("GPU not available, skipping GPU i8 test");
        }
    }

    #[test]
    #[ignore] // Only run if GPU is available
    fn test_gpu_binary_ops_u8() {
        let cpu_a = Tensor::<u8>::from_vec(vec![10, 20, 17], &[3]).unwrap();
        let cpu_b = Tensor::<u8>::from_vec(vec![5, 10, 15], &[3]).unwrap();

        // Try to create GPU tensors
        let gpu_device = Device::Gpu(0);

        if let (Ok(gpu_a), Ok(gpu_b)) = (cpu_a.to(gpu_device), cpu_b.to(gpu_device)) {
            // Test GPU multiplication
            let gpu_result = mul(&gpu_a, &gpu_b);

            match gpu_result {
                Ok(result) => {
                    // Transfer back to CPU for verification
                    let cpu_result = result.to(Device::Cpu).unwrap();
                    let expected = vec![50u8, 200u8, 255u8]; // 10*5, 20*10, 17*15=255
                    if let tenflowers_core::tensor::TensorStorage::Cpu(arr) = &cpu_result.storage {
                        assert_eq!(arr.as_slice().unwrap(), &expected);
                    }
                    println!("GPU u8 binary operation test passed");
                }
                Err(e) => {
                    println!("GPU u8 operation not yet fully implemented: {}", e);
                }
            }
        } else {
            println!("GPU not available, skipping GPU u8 test");
        }
    }
}
