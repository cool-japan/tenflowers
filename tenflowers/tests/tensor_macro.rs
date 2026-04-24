//! Integration test: verify the `tensor!` macro produces tensors with correct shapes.

use tenflowers::tensor;

/// 1-D macro arm.
#[test]
fn tensor_macro_1d() {
    let t = tensor![1.0_f32, 2.0, 3.0];
    assert_eq!(t.shape().dims(), &[3]);
}

/// 1-D macro arm with integer values.
#[test]
fn tensor_macro_1d_int() {
    let t = tensor![0i32, 1, 2, 3, 4];
    assert_eq!(t.shape().dims(), &[5]);
}

/// 1-D macro arm: trailing comma is accepted.
#[test]
fn tensor_macro_1d_trailing_comma() {
    let t = tensor![1.0_f32, 2.0,];
    assert_eq!(t.shape().dims(), &[2]);
}

/// 2-D macro arm.
#[test]
fn tensor_macro_2d() {
    let t = tensor![[1.0_f32, 2.0], [3.0, 4.0]];
    assert_eq!(t.shape().dims(), &[2, 2]);
}

/// 2-D macro arm: non-square shape.
#[test]
fn tensor_macro_2d_rect() {
    let t = tensor![[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
    assert_eq!(t.shape().dims(), &[2, 3]);
}

/// Explicit dtype (1-D).
#[test]
fn tensor_macro_dtype_1d() {
    let t = tensor![dtype = f64; 1.0, 2.0, 3.0];
    assert_eq!(t.shape().dims(), &[3]);
}

/// Explicit dtype (2-D).
#[test]
fn tensor_macro_dtype_2d() {
    let t = tensor![dtype = f64; [1.0, 2.0], [3.0, 4.0]];
    assert_eq!(t.shape().dims(), &[2, 2]);
}

/// Empty tensor macro.
#[test]
fn tensor_macro_empty() {
    let t = tensor![];
    assert_eq!(t.shape().dims(), &[0]);
}
