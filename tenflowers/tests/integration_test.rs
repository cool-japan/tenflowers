//! Integration tests for the `tenflowers` meta crate.
//!
//! These tests exercise the public API surface exported through `tenflowers::prelude`
//! and the convenience macros, verifying that all re-exports resolve correctly and
//! that basic tensor / model construction works end-to-end.

use tenflowers::prelude::*;
use tenflowers::tensor;

// ---------------------------------------------------------------------------
// Basic tensor construction via prelude
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_zeros_and_ones() {
    let zeros = Tensor::<f32>::zeros(&[3, 4]);
    assert_eq!(zeros.shape().dims(), &[3, 4]);

    let ones = Tensor::<f32>::ones(&[2, 5]);
    assert_eq!(ones.shape().dims(), &[2, 5]);
}

#[test]
fn test_tensor_from_data() {
    let t = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("from_data must succeed for matching sizes");
    assert_eq!(t.shape().dims(), &[2, 2]);
}

// ---------------------------------------------------------------------------
// tensor![] macro
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_macro_basic() {
    let t = tensor![1.0f32, 2.0, 3.0];
    assert_eq!(t.shape().dims(), &[3]);
}

#[test]
fn test_tensor_macro_single_element() {
    let t = tensor![42.0f64];
    assert_eq!(t.shape().dims(), &[1]);
}

#[test]
fn test_tensor_macro_integers() {
    let t = tensor![0i32, 1, 2, 3, 4];
    assert_eq!(t.shape().dims(), &[5]);
}

#[test]
fn test_tensor_macro_trailing_comma() {
    // Trailing commas must be accepted by the macro.
    let t = tensor![10.0f32, 20.0, 30.0,];
    assert_eq!(t.shape().dims(), &[3]);
}

#[test]
fn test_tensor_macro_empty() {
    let t: tenflowers::core::Tensor<f32> = tensor![];
    assert_eq!(t.shape().dims(), &[0]);
}

// ---------------------------------------------------------------------------
// Dense layer
// ---------------------------------------------------------------------------

#[test]
fn test_dense_forward() {
    let layer = Dense::<f32>::new(4, 2, true);
    let input = Tensor::<f32>::zeros(&[1, 4]);
    let output = layer.forward(&input).expect("dense forward must succeed");
    assert_eq!(output.shape().dims(), &[1, 2]);
}

// ---------------------------------------------------------------------------
// Sequential model
// ---------------------------------------------------------------------------

#[test]
fn test_sequential_construction_and_forward() {
    let model = Sequential::<f32>::new(vec![])
        .add(Box::new(Dense::new(8, 4, true)))
        .add(Box::new(Dense::new(4, 2, true)));

    assert_eq!(model.len(), 2);

    let input = Tensor::<f32>::zeros(&[1, 8]);
    let output = model
        .forward(&input)
        .expect("sequential forward must succeed");
    assert_eq!(output.shape().dims(), &[1, 2]);
}

#[test]
fn test_sequential_with_activation() {
    let model = Sequential::<f32>::new(vec![])
        .add(Box::new(
            Dense::new(4, 4, true).with_activation("relu".to_string()),
        ))
        .add(Box::new(Dense::new(4, 1, true)));

    let input = Tensor::<f32>::zeros(&[3, 4]);
    let output = model.forward(&input).expect("forward must succeed");
    assert_eq!(output.shape().dims(), &[3, 1]);
}

// ---------------------------------------------------------------------------
// Adam optimizer — construction only (training loop is complex to set up)
// ---------------------------------------------------------------------------

#[test]
fn test_adam_construction() {
    let adam = Adam::<f32>::new(0.001);
    // Just verifying that construction succeeds without panicking.
    drop(adam);
}

#[test]
fn test_adam_default() {
    // Adam::default() should also be valid.
    let _adam: Adam<f32> = Default::default();
}

// ---------------------------------------------------------------------------
// Module aliases (nn / data / optim)
// ---------------------------------------------------------------------------

#[test]
fn test_nn_module_alias() {
    // Access Dense via the nn module alias.
    let _layer = tenflowers::nn::Dense::<f32>::new(2, 2, false);
}

#[test]
fn test_optim_module_alias() {
    let _opt = tenflowers::optim::Adam::<f32>::new(0.01);
}
