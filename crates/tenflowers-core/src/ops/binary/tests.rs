//! Tests for Binary Operations
//!
//! Comprehensive test suite for binary operations including same-shape operations,
//! broadcasting, SIMD optimizations, and performance verification.

#[cfg(test)]
#[allow(irrefutable_let_patterns)] // Pattern matching on TensorStorage is irrefutable when GPU feature is disabled
mod tests {
    use super::super::convenience::*;
    use crate::tensor::TensorStorage;
    use crate::Tensor;

    #[test]
    fn test_add_same_shape() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])
            .expect("test: from_vec should succeed");
        let b = Tensor::<f32>::from_vec(vec![4.0, 5.0, 6.0], &[3])
            .expect("test: from_vec should succeed");

        let c = add(&a, &b).expect("test: add should succeed");
        let expected = vec![5.0, 7.0, 9.0];

        // Compare results
        if let TensorStorage::Cpu(arr) = &c.storage {
            assert_eq!(
                arr.as_slice().expect("tensor should be contiguous"),
                &expected
            );
        }
    }

    #[test]
    fn test_mul_broadcast() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3, 1])
            .expect("test: from_vec should succeed");
        let b = Tensor::<f32>::from_vec(vec![2.0, 3.0], &[1, 2])
            .expect("test: from_vec should succeed");

        let c = mul(&a, &b).expect("test: mul should succeed");
        assert_eq!(c.shape().dims(), &[3, 2]);

        // Expected: [[2, 3], [4, 6], [6, 9]]
        let expected = vec![2.0, 3.0, 4.0, 6.0, 6.0, 9.0];
        if let TensorStorage::Cpu(arr) = &c.storage {
            assert_eq!(
                arr.as_slice().expect("tensor should be contiguous"),
                &expected
            );
        }
    }

    #[test]
    fn test_pow() {
        let a = Tensor::<f32>::from_vec(vec![2.0, 3.0, 4.0], &[3])
            .expect("test: from_vec should succeed");
        let b = Tensor::<f32>::from_vec(vec![2.0, 2.0, 2.0], &[3])
            .expect("test: from_vec should succeed");

        let c = pow(&a, &b).expect("test: pow should succeed");
        let expected = vec![4.0, 9.0, 16.0];

        if let TensorStorage::Cpu(arr) = &c.storage {
            assert_eq!(
                arr.as_slice().expect("tensor should be contiguous"),
                &expected
            );
        }

        // Test broadcasting
        let scalar =
            Tensor::<f32>::from_vec(vec![3.0], &[1]).expect("test: from_vec should succeed");
        let d = pow(&a, &scalar).expect("test: pow should succeed");
        let expected = vec![8.0, 27.0, 64.0];

        if let TensorStorage::Cpu(arr) = &d.storage {
            assert_eq!(
                arr.as_slice().expect("tensor should be contiguous"),
                &expected
            );
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_add_optimization() {
        use crate::Device;

        // Test SIMD-optimized addition for f32 tensors
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8])
            .expect("test: from_vec should succeed");
        let b = Tensor::from_vec(vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[8])
            .expect("test: from_vec should succeed");

        let result = add(&a, &b).expect("test: add should succeed");
        let expected = vec![3.0f32, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0];

        if let TensorStorage::Cpu(arr) = &result.storage {
            assert_eq!(
                arr.as_slice().expect("tensor should be contiguous"),
                &expected
            );
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_mul_optimization() {
        use crate::Device;

        // Test SIMD-optimized multiplication for f32 tensors
        let a = Tensor::from_vec(vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[8])
            .expect("test: from_vec should succeed");
        let b = Tensor::from_vec(vec![1.5f32, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], &[8])
            .expect("test: from_vec should succeed");

        let result = mul(&a, &b).expect("test: mul should succeed");
        let expected = vec![3.0f32, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0];

        if let TensorStorage::Cpu(arr) = &result.storage {
            assert_eq!(
                arr.as_slice().expect("tensor should be contiguous"),
                &expected
            );
        }
    }

    #[test]
    #[cfg(feature = "autograd")]
    fn test_memory_profiling_integration() {
        // Clear profiler state for this test
        // TODO: Implement profiler integration
        // if let Ok(mut p) = get_profiler().lock() {
        //     p.operations.clear();
        //     p.allocations = 0;
        // }

        // Perform a binary operation
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4])
            .expect("test: from_vec should succeed");
        let b = Tensor::from_vec(vec![2.0f32, 3.0, 4.0, 5.0], &[4])
            .expect("test: from_vec should succeed");
        let _result = add(&a, &b).expect("test: add should succeed");

        // Check that simple profiling captured the operation
        // if let Ok(p) = get_profiler().lock() {
        //     // Should have recorded the binary_Add operation
        //     assert!(p.operations.contains_key("binary_Add"));
        //     // Should have recorded at least one allocation
        //     assert!(p.allocations > 0);
        // }
    }

    #[test]
    fn test_subtraction() {
        let a = Tensor::<f32>::from_vec(vec![5.0, 7.0, 9.0], &[3])
            .expect("test: from_vec should succeed");
        let b = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])
            .expect("test: from_vec should succeed");

        let c = sub(&a, &b).expect("test: sub should succeed");
        let expected = vec![4.0, 5.0, 6.0];

        if let TensorStorage::Cpu(arr) = &c.storage {
            assert_eq!(
                arr.as_slice().expect("tensor should be contiguous"),
                &expected
            );
        }
    }

    #[test]
    fn test_division() {
        let a = Tensor::<f32>::from_vec(vec![6.0, 8.0, 10.0], &[3])
            .expect("test: from_vec should succeed");
        let b = Tensor::<f32>::from_vec(vec![2.0, 4.0, 5.0], &[3])
            .expect("test: from_vec should succeed");

        let c = div(&a, &b).expect("test: div should succeed");
        let expected = vec![3.0, 2.0, 2.0];

        if let TensorStorage::Cpu(arr) = &c.storage {
            assert_eq!(
                arr.as_slice().expect("tensor should be contiguous"),
                &expected
            );
        }
    }

    #[test]
    fn test_min_max() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 5.0, 3.0], &[3])
            .expect("test: from_vec should succeed");
        let b = Tensor::<f32>::from_vec(vec![2.0, 3.0, 4.0], &[3])
            .expect("test: from_vec should succeed");

        let min_result = min(&a, &b).expect("test: min should succeed");
        let expected_min = vec![1.0, 3.0, 3.0];

        if let TensorStorage::Cpu(arr) = &min_result.storage {
            assert_eq!(
                arr.as_slice().expect("tensor should be contiguous"),
                &expected_min
            );
        }

        let max_result = max(&a, &b).expect("test: max should succeed");
        let expected_max = vec![2.0, 5.0, 4.0];

        if let TensorStorage::Cpu(arr) = &max_result.storage {
            assert_eq!(
                arr.as_slice().expect("tensor should be contiguous"),
                &expected_max
            );
        }
    }

    #[test]
    fn test_scalar_add() {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])
            .expect("test: from_vec should succeed");
        let result = scalar_add(&tensor, 5.0).expect("test: scalar_add should succeed");
        let expected = vec![6.0, 7.0, 8.0];

        if let TensorStorage::Cpu(arr) = &result.storage {
            assert_eq!(
                arr.as_slice().expect("tensor should be contiguous"),
                &expected
            );
        }
    }

    #[test]
    fn test_clamp() {
        let tensor = Tensor::<f32>::from_vec(vec![-1.0, 0.5, 2.0, 5.0], &[4])
            .expect("test: from_vec should succeed");
        let result = clamp(&tensor, 0.0, 3.0).expect("test: clamp should succeed");
        let expected = vec![0.0, 0.5, 2.0, 3.0];

        if let TensorStorage::Cpu(arr) = &result.storage {
            assert_eq!(
                arr.as_slice().expect("tensor should be contiguous"),
                &expected
            );
        }
    }

    #[test]
    fn test_ultra_performance_functions() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])
            .expect("test: from_vec should succeed");
        let b = Tensor::<f32>::from_vec(vec![4.0, 5.0, 6.0], &[3])
            .expect("test: from_vec should succeed");

        // Test ultra functions (they should behave the same but with metrics)
        let ultra_add_result = ultra_add(&a, &b).expect("test: ultra_add should succeed");
        let ultra_mul_result = ultra_mul(&a, &b).expect("test: ultra_mul should succeed");
        let ultra_sub_result = ultra_sub(&b, &a).expect("test: ultra_sub should succeed");
        let ultra_div_result = ultra_div(&b, &a).expect("test: ultra_div should succeed");

        // Expected results
        let expected_add = vec![5.0, 7.0, 9.0];
        let expected_mul = vec![4.0, 10.0, 18.0];
        let expected_sub = vec![3.0, 3.0, 3.0];
        let expected_div = vec![4.0, 2.5, 2.0];

        if let TensorStorage::Cpu(arr) = &ultra_add_result.storage {
            assert_eq!(
                arr.as_slice().expect("tensor should be contiguous"),
                &expected_add
            );
        }
        if let TensorStorage::Cpu(arr) = &ultra_mul_result.storage {
            assert_eq!(
                arr.as_slice().expect("tensor should be contiguous"),
                &expected_mul
            );
        }
        if let TensorStorage::Cpu(arr) = &ultra_sub_result.storage {
            assert_eq!(
                arr.as_slice().expect("tensor should be contiguous"),
                &expected_sub
            );
        }
        if let TensorStorage::Cpu(arr) = &ultra_div_result.storage {
            assert_eq!(
                arr.as_slice().expect("tensor should be contiguous"),
                &expected_div
            );
        }
    }
}
