//! Integration test: verify type aliases compile and resolve to Tensor<T>.

use tenflowers::type_aliases::*;
use tenflowers_core::Tensor;

/// Verify generic 1-D/2-D/3-D/4-D aliases are just `Tensor<T>`.
#[test]
fn generic_aliases_are_tensors() {
    let _t1: Tensor1D<f32> = Tensor::zeros(&[4]);
    let _t2: Tensor2D<f32> = Tensor::zeros(&[2, 3]);
    let _t3: Tensor3D<f32> = Tensor::zeros(&[2, 3, 4]);
    let _t4: Tensor4D<f32> = Tensor::zeros(&[1, 3, 8, 8]);
}

/// Verify f32-typed aliases resolve.
#[test]
fn f32_aliases_are_tensors() {
    let mat: Tensor2Df32 = Tensor::zeros(&[3, 4]);
    assert_eq!(mat.shape().dims(), &[3, 4]);

    let vec: Vector = Tensor::zeros(&[5]);
    assert_eq!(vec.shape().dims(), &[5]);

    let batch: BatchTensor = Tensor::zeros(&[8, 10]);
    assert_eq!(batch.shape().dims(), &[8, 10]);

    let t1: Tensor1Df32 = Tensor::zeros(&[7]);
    assert_eq!(t1.shape().dims(), &[7]);

    let t3: Tensor3Df32 = Tensor::zeros(&[2, 3, 4]);
    assert_eq!(t3.shape().dims(), &[2, 3, 4]);

    let t4: Tensor4Df32 = Tensor::zeros(&[1, 3, 224, 224]);
    assert_eq!(t4.shape().dims(), &[1, 3, 224, 224]);
}

/// Verify f64-typed aliases resolve.
#[test]
fn f64_aliases_are_tensors() {
    let t1: Tensor1Df64 = Tensor::zeros(&[6]);
    assert_eq!(t1.shape().dims(), &[6]);

    let t2: Tensor2Df64 = Tensor::zeros(&[4, 5]);
    assert_eq!(t2.shape().dims(), &[4, 5]);

    let t3: Tensor3Df64 = Tensor::zeros(&[2, 4, 6]);
    assert_eq!(t3.shape().dims(), &[2, 4, 6]);

    let t4: Tensor4Df64 = Tensor::zeros(&[2, 1, 8, 8]);
    assert_eq!(t4.shape().dims(), &[2, 1, 8, 8]);
}

/// Matrix alias is a Tensor<f32> (can be used interchangeably).
#[test]
fn matrix_alias_interchangeable_with_tensor_f32() {
    let m: Matrix = Tensor::<f32>::zeros(&[3, 3]);
    // Can round-trip: Matrix is just Tensor<f32>
    let _: Tensor<f32> = m;
}
