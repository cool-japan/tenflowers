use tenflowers_core::{ops::*, Tensor};

#[test]
fn test_comparison_operations_comprehensive() {
    // Test all comparison operations with various shapes
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![2.0, 2.0, 2.0, 2.0], &[2, 2]).unwrap();

    // Test all comparison operations
    let eq_result = eq(&a, &b).unwrap();
    let ne_result = ne(&a, &b).unwrap();
    let lt_result = lt(&a, &b).unwrap();
    let le_result = le(&a, &b).unwrap();
    let gt_result = gt(&a, &b).unwrap();
    let ge_result = ge(&a, &b).unwrap();

    // Verify shapes
    assert_eq!(eq_result.shape().dims(), &[2, 2]);
    assert_eq!(ne_result.shape().dims(), &[2, 2]);
    assert_eq!(lt_result.shape().dims(), &[2, 2]);
    assert_eq!(le_result.shape().dims(), &[2, 2]);
    assert_eq!(gt_result.shape().dims(), &[2, 2]);
    assert_eq!(ge_result.shape().dims(), &[2, 2]);

    // Verify results
    assert_eq!(eq_result.as_slice().unwrap(), &[0u8, 1u8, 0u8, 0u8]);
    assert_eq!(ne_result.as_slice().unwrap(), &[1u8, 0u8, 1u8, 1u8]);
    assert_eq!(lt_result.as_slice().unwrap(), &[1u8, 0u8, 0u8, 0u8]);
    assert_eq!(le_result.as_slice().unwrap(), &[1u8, 1u8, 0u8, 0u8]);
    assert_eq!(gt_result.as_slice().unwrap(), &[0u8, 0u8, 1u8, 1u8]);
    assert_eq!(ge_result.as_slice().unwrap(), &[0u8, 1u8, 1u8, 1u8]);
}

#[test]
fn test_comparison_operations_broadcasting() {
    // Test broadcasting with comparison operations
    let a = Tensor::<f32>::from_vec(vec![1.0, 3.0], &[2, 1]).unwrap();
    let b = Tensor::<f32>::from_vec(vec![2.0, 1.0], &[1, 2]).unwrap();

    let lt_result = lt(&a, &b).unwrap();
    assert_eq!(lt_result.shape().dims(), &[2, 2]);

    // Expected: [[1.0 < 2.0, 1.0 < 1.0], [3.0 < 2.0, 3.0 < 1.0]]
    //          = [[true, false], [false, false]]
    //          = [[1, 0], [0, 0]]
    assert_eq!(lt_result.as_slice().unwrap(), &[1u8, 0u8, 0u8, 0u8]);
}

#[test]
fn test_logical_operations_comprehensive() {
    // Test all logical operations
    let a = Tensor::<u8>::from_vec(vec![1u8, 0u8, 1u8, 0u8], &[2, 2]).unwrap();
    let b = Tensor::<u8>::from_vec(vec![1u8, 1u8, 0u8, 0u8], &[2, 2]).unwrap();

    let and_result = logical_and(&a, &b).unwrap();
    let or_result = logical_or(&a, &b).unwrap();
    let xor_result = logical_xor(&a, &b).unwrap();
    let not_result = logical_not(&a).unwrap();

    // Verify shapes
    assert_eq!(and_result.shape().dims(), &[2, 2]);
    assert_eq!(or_result.shape().dims(), &[2, 2]);
    assert_eq!(xor_result.shape().dims(), &[2, 2]);
    assert_eq!(not_result.shape().dims(), &[2, 2]);

    // Verify results
    assert_eq!(and_result.as_slice().unwrap(), &[1u8, 0u8, 0u8, 0u8]);
    assert_eq!(or_result.as_slice().unwrap(), &[1u8, 1u8, 1u8, 0u8]);
    assert_eq!(xor_result.as_slice().unwrap(), &[0u8, 1u8, 1u8, 0u8]);
    assert_eq!(not_result.as_slice().unwrap(), &[0u8, 1u8, 0u8, 1u8]);
}

#[test]
fn test_logical_operations_broadcasting() {
    // Test broadcasting with logical operations
    let a = Tensor::<u8>::from_vec(vec![1u8, 0u8], &[2, 1]).unwrap();
    let b = Tensor::<u8>::from_vec(vec![1u8, 0u8], &[1, 2]).unwrap();

    let and_result = logical_and(&a, &b).unwrap();
    assert_eq!(and_result.shape().dims(), &[2, 2]);

    // Expected: [[1 AND 1, 1 AND 0], [0 AND 1, 0 AND 0]]
    //          = [[1, 0], [0, 0]]
    assert_eq!(and_result.as_slice().unwrap(), &[1u8, 0u8, 0u8, 0u8]);
}

#[test]
fn test_advanced_pooling_operations() {
    // Test global max pooling
    let input = Tensor::<f32>::from_vec(
        vec![1.0, 8.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.0, 9.0],
        &[1, 1, 3, 3],
    )
    .unwrap();

    let global_max = global_max_pool2d(&input).unwrap();
    assert_eq!(global_max.shape().dims(), &[1, 1, 1, 1]);
    assert_eq!(global_max.as_slice().unwrap()[0], 9.0);

    let global_avg = global_avg_pool2d(&input).unwrap();
    assert_eq!(global_avg.shape().dims(), &[1, 1, 1, 1]);
    let expected_avg = (1.0 + 8.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 2.0 + 9.0) / 9.0;
    assert!((global_avg.as_slice().unwrap()[0] - expected_avg).abs() < 1e-6);
}

#[test]
fn test_adaptive_pooling_different_sizes() {
    // Create a 6x6 input
    let mut data = Vec::new();
    for i in 0..36 {
        data.push(i as f32);
    }
    let input = Tensor::<f32>::from_vec(data, &[1, 1, 6, 6]).unwrap();

    // Test adaptive pooling to different output sizes
    let adaptive_1x1 = adaptive_avg_pool2d(&input, (1, 1)).unwrap();
    assert_eq!(adaptive_1x1.shape().dims(), &[1, 1, 1, 1]);

    let adaptive_2x2 = adaptive_avg_pool2d(&input, (2, 2)).unwrap();
    assert_eq!(adaptive_2x2.shape().dims(), &[1, 1, 2, 2]);

    let adaptive_3x3 = adaptive_avg_pool2d(&input, (3, 3)).unwrap();
    assert_eq!(adaptive_3x3.shape().dims(), &[1, 1, 3, 3]);
}

#[test]
fn test_multichannel_pooling() {
    // Test with multiple channels
    let input = Tensor::<f32>::from_vec(
        vec![
            // Channel 0
            1.0, 2.0, 3.0, 4.0, // Channel 1
            5.0, 6.0, 7.0, 8.0,
        ],
        &[1, 2, 2, 2], // NCHW: batch=1, channels=2, height=2, width=2
    )
    .unwrap();

    let global_max = global_max_pool2d(&input).unwrap();
    assert_eq!(global_max.shape().dims(), &[1, 2, 1, 1]);
    assert_eq!(global_max.as_slice().unwrap(), &[4.0, 8.0]);

    let global_avg = global_avg_pool2d(&input).unwrap();
    assert_eq!(global_avg.shape().dims(), &[1, 2, 1, 1]);
    assert_eq!(global_avg.as_slice().unwrap(), &[2.5, 6.5]);
}

#[test]
fn test_comparison_with_integers() {
    // Test comparison operations with integer types
    let a = Tensor::<i32>::from_vec(vec![1, 2, 3], &[3]).unwrap();
    let b = Tensor::<i32>::from_vec(vec![2, 2, 1], &[3]).unwrap();

    let lt_result = lt(&a, &b).unwrap();
    let eq_result = eq(&a, &b).unwrap();
    let gt_result = gt(&a, &b).unwrap();

    assert_eq!(lt_result.as_slice().unwrap(), &[1u8, 0u8, 0u8]);
    assert_eq!(eq_result.as_slice().unwrap(), &[0u8, 1u8, 0u8]);
    assert_eq!(gt_result.as_slice().unwrap(), &[0u8, 0u8, 1u8]);
}

#[test]
fn test_edge_cases() {
    // Test edge cases for pooling operations

    // Minimum input size (1x1)
    let tiny_input = Tensor::<f32>::from_vec(vec![42.0], &[1, 1, 1, 1]).unwrap();

    let global_max = global_max_pool2d(&tiny_input).unwrap();
    assert_eq!(global_max.as_slice().unwrap()[0], 42.0);

    let global_avg = global_avg_pool2d(&tiny_input).unwrap();
    assert_eq!(global_avg.as_slice().unwrap()[0], 42.0);

    let adaptive = adaptive_avg_pool2d(&tiny_input, (1, 1)).unwrap();
    assert_eq!(adaptive.as_slice().unwrap()[0], 42.0);
}

#[test]
fn test_batched_operations() {
    // Test operations with batched inputs
    let input = Tensor::<f32>::from_vec(
        vec![
            // Batch 0, Channel 0
            1.0, 2.0, 3.0, 4.0, // Batch 1, Channel 0
            5.0, 6.0, 7.0, 8.0,
        ],
        &[2, 1, 2, 2], // NCHW: batch=2, channels=1, height=2, width=2
    )
    .unwrap();

    let global_max = global_max_pool2d(&input).unwrap();
    assert_eq!(global_max.shape().dims(), &[2, 1, 1, 1]);
    assert_eq!(global_max.as_slice().unwrap(), &[4.0, 8.0]);
}
