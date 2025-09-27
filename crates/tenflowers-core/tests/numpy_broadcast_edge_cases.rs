//! Comprehensive tests for NumPy broadcasting edge cases
//!
//! This test suite ensures that TenfloweRS handles all the complex broadcasting
//! scenarios that NumPy supports, including edge cases and corner cases.

use tenflowers_core::ops::{apply_ufunc, basic, numpy_broadcast_arrays};
use tenflowers_core::{Result, Tensor};

#[test]
fn test_basic_broadcasting() {
    // Test basic broadcasting: (3,) + (3, 1) -> (3, 3)
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0], &[3, 1]).unwrap();

    let result = basic::add(&a, &b).unwrap();
    assert_eq!(result.shape().dims(), &[3, 3]);

    if let Some(data) = result.as_slice() {
        let expected = &[
            11.0, 12.0, 13.0, // 10 + [1, 2, 3]
            21.0, 22.0, 23.0, // 20 + [1, 2, 3]
            31.0, 32.0, 33.0, // 30 + [1, 2, 3]
        ];
        assert_eq!(data, expected);
    }
}

#[test]
fn test_scalar_broadcasting() {
    // Test scalar broadcasting: () + (2, 3) -> (2, 3)
    let scalar = Tensor::<f32>::from_vec(vec![5.0], &[1]).unwrap();
    let array = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

    let result = basic::add(&scalar, &array).unwrap();
    assert_eq!(result.shape().dims(), &[2, 3]);

    if let Some(data) = result.as_slice() {
        assert_eq!(data, &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    }
}

#[test]
fn test_prepending_ones() {
    // Test broadcasting where dimensions are prepended: (3,) + (2, 1, 3) -> (2, 1, 3)
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![10.0, 20.0], &[2, 1, 1]).unwrap();

    let broadcasted = numpy_broadcast_arrays(&[&a, &b]).unwrap();
    assert_eq!(broadcasted[0].shape().dims(), &[2, 1, 3]);
    assert_eq!(broadcasted[1].shape().dims(), &[2, 1, 3]);
}

#[test]
fn test_complex_broadcasting() {
    // Test complex broadcasting: (2, 1, 4) + (1, 3, 1) -> (2, 3, 4)
    let a =
        Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 1, 4]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0], &[1, 3, 1]).unwrap();

    let result = basic::add(&a, &b).unwrap();
    assert_eq!(result.shape().dims(), &[2, 3, 4]);

    // Verify some values
    if let Some(data) = result.as_slice() {
        // First slice: [1, 2, 3, 4] + 10 = [11, 12, 13, 14]
        assert_eq!(data[0], 11.0);
        assert_eq!(data[1], 12.0);
        assert_eq!(data[2], 13.0);
        assert_eq!(data[3], 14.0);

        // Second slice: [1, 2, 3, 4] + 20 = [21, 22, 23, 24]
        assert_eq!(data[4], 21.0);
        assert_eq!(data[5], 22.0);
    }
}

#[test]
fn test_multiple_array_broadcasting() {
    // Test broadcasting with multiple arrays
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();
    let c = Tensor::<f32>::from_vec(vec![100.0], &[1, 1]).unwrap();

    let broadcasted = numpy_broadcast_arrays(&[&a, &b, &c]).unwrap();

    // All should have shape (2, 3)
    for tensor in &broadcasted {
        assert_eq!(tensor.shape().dims(), &[2, 3]);
    }

    // Verify the values
    if let Some(a_data) = broadcasted[0].as_slice() {
        assert_eq!(a_data, &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    if let Some(b_data) = broadcasted[1].as_slice() {
        assert_eq!(b_data, &[10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
    }

    if let Some(c_data) = broadcasted[2].as_slice() {
        assert_eq!(c_data, &[100.0; 6]);
    }
}

#[test]
fn test_high_dimensional_broadcasting() {
    // Test high dimensional broadcasting: (1, 2, 1, 3) + (2, 1, 4, 1) -> (2, 2, 4, 3)
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 1, 3]).unwrap();
    let b = Tensor::<f32>::from_vec(
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        &[2, 1, 4, 1],
    )
    .unwrap();

    let result = basic::add(&a, &b).unwrap();
    assert_eq!(result.shape().dims(), &[2, 2, 4, 3]);
    assert_eq!(result.shape().size(), 48); // 2 * 2 * 4 * 3 = 48
}

#[test]
fn test_degenerate_dimensions() {
    // Test with degenerate dimensions (size 1)
    let a = Tensor::<f32>::from_vec(vec![5.0], &[1, 1, 1]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

    let result = basic::mul(&a, &b).unwrap();
    assert_eq!(result.shape().dims(), &[1, 1, 3]);

    if let Some(data) = result.as_slice() {
        assert_eq!(data, &[5.0, 10.0, 15.0]);
    }
}

#[test]
fn test_broadcasting_errors() {
    // Test incompatible shapes that should fail
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![10.0, 20.0], &[2]).unwrap();

    // This should fail because 3 and 2 are incompatible
    let result = basic::add(&a, &b);
    assert!(result.is_err());
}

#[test]
fn test_broadcasting_with_ufuncs() {
    // Test broadcasting using the ufunc system
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();

    let result = apply_ufunc("add", &[&a, &b]).unwrap();
    assert_eq!(result.shape().dims(), &[2, 3]);

    if let Some(data) = result.as_slice() {
        assert_eq!(data, &[11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
    }
}

#[test]
fn test_edge_case_empty_tensors() {
    // Test with empty tensors (zero dimensions)
    let empty = Tensor::<f32>::from_vec(vec![], &[0]).unwrap();
    let normal = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap();

    // Broadcasting with empty tensor should handle gracefully
    let broadcasted = numpy_broadcast_arrays(&[&empty, &normal]);
    // This might be implementation-dependent behavior
    // NumPy has specific rules for empty arrays
}

#[test]
fn test_broadcasting_preservation() {
    // Test that broadcasting preserves the original tensors
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![10.0], &[1]).unwrap();

    let original_a_data = a.as_slice().unwrap().to_vec();
    let original_b_data = b.as_slice().unwrap().to_vec();

    let _result = basic::add(&a, &b).unwrap();

    // Original tensors should be unchanged
    assert_eq!(a.as_slice().unwrap(), &original_a_data);
    assert_eq!(b.as_slice().unwrap(), &original_b_data);
}

#[test]
fn test_numpy_style_broadcasting_rules() {
    // Test NumPy's specific broadcasting rules

    // Rule 1: Arrays with same shape always broadcast
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    let result = basic::add(&a, &b).unwrap();
    assert_eq!(result.shape().dims(), &[2, 2]);

    // Rule 2: Arrays are aligned from the rightmost dimension
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0], &[1, 3]).unwrap();
    let result = basic::add(&a, &b).unwrap();
    assert_eq!(result.shape().dims(), &[1, 3]);

    // Rule 3: Dimensions of size 1 can be stretched
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0], &[1, 3]).unwrap();
    let result = basic::add(&a, &b).unwrap();
    assert_eq!(result.shape().dims(), &[2, 3]);

    // Rule 4: Missing dimensions are assumed to be 1
    let a = Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
    let result = basic::add(&a, &b).unwrap();
    assert_eq!(result.shape().dims(), &[2, 2]);
}

#[test]
fn test_advanced_broadcasting_patterns() {
    // Test some advanced broadcasting patterns from real-world usage

    // Matrix-vector operations
    let matrix = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let vector = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();
    let result = basic::add(&matrix, &vector).unwrap();
    assert_eq!(result.shape().dims(), &[2, 3]);

    // Batch operations
    let batch = Tensor::<f32>::from_vec(vec![1.0; 24], &[4, 2, 3]).unwrap();
    let bias = Tensor::<f32>::from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap();
    let result = basic::add(&batch, &bias).unwrap();
    assert_eq!(result.shape().dims(), &[4, 2, 3]);

    // Channel-wise operations (common in neural networks)
    let image = Tensor::<f32>::from_vec(vec![1.0; 48], &[2, 3, 4, 2]).unwrap(); // batch, channel, height, width
    let channel_bias = Tensor::<f32>::from_vec(vec![0.1, 0.2, 0.3], &[1, 3, 1, 1]).unwrap();
    let result = basic::add(&image, &channel_bias).unwrap();
    assert_eq!(result.shape().dims(), &[2, 3, 4, 2]);
}

#[test]
fn test_broadcasting_with_different_dtypes() {
    // This would test mixed-dtype broadcasting if supported
    // For now, we test with the same dtype but different shapes
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0], &[1, 3]).unwrap();

    let result = basic::mul(&a, &b).unwrap();
    assert_eq!(result.shape().dims(), &[2, 3]);
}

#[test]
fn test_mathematical_ufunc_broadcasting() {
    // Test mathematical ufuncs with broadcasting
    let x = Tensor::<f32>::from_vec(vec![0.0, 1.0, 2.0], &[3]).unwrap();
    let y = Tensor::<f32>::from_vec(vec![1.0], &[1]).unwrap();

    // Test power function with broadcasting
    let result = apply_ufunc("power", &[&x, &y]).unwrap();
    assert_eq!(result.shape().dims(), &[3]);

    // Test trigonometric functions
    let angles = Tensor::<f32>::from_vec(vec![0.0, std::f32::consts::PI / 2.0], &[2, 1]).unwrap();
    let ones = Tensor::<f32>::from_vec(vec![1.0, 1.0, 1.0], &[3]).unwrap();

    let sin_result = apply_ufunc("sin", &[&angles]).unwrap();
    assert_eq!(sin_result.shape().dims(), &[2, 1]);

    // Test with broadcasting in mathematical operations
    let cos_result = apply_ufunc("cos", &[&angles]).unwrap();
    let combined = basic::add(&sin_result, &ones).unwrap(); // Broadcasting sin result with ones
    assert_eq!(combined.shape().dims(), &[2, 3]);
}

#[test]
fn test_reduction_after_broadcasting() {
    // Test that reductions work correctly after broadcasting
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0], &[3]).unwrap();

    let broadcasted = basic::add(&a, &b).unwrap();
    assert_eq!(broadcasted.shape().dims(), &[2, 3]);

    // This would test reductions on the broadcasted result
    // let sum_result = reduction::sum(&broadcasted, Some(&[1]), false).unwrap();
    // assert_eq!(sum_result.shape().dims(), &[2]);
}
