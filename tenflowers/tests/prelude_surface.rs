//! Integration test: verify the prelude surface compiles without extra imports.

use tenflowers::prelude::*;

/// Verify that core prelude types are resolvable and constructable under
/// default features — no extra `use` statements required.
#[test]
fn prelude_constructs_without_extra_imports() {
    // Dense layer
    let _d: Dense<f32> = Dense::new(4, 2, true);

    // Sequential model
    let _model = Sequential::<f32>::new(vec![]);

    // Adam optimizer
    let _opt = Adam::<f32>::new(1e-3);

    // SGD optimizer
    let _sgd = SGD::<f32>::new(0.01);

    // DType enum
    let _dt = DType::Float32;

    // Tensor construction
    let t = Tensor::<f32>::zeros(&[2, 3]);
    assert_eq!(t.shape().dims(), &[2, 3]);
}

/// Verify that type aliases (re-exported from prelude) resolve to Tensor<f32>.
#[test]
fn prelude_type_aliases_resolve() {
    let _m: Matrix = Tensor::<f32>::zeros(&[3, 4]);
    let _v: Vector = Tensor::<f32>::zeros(&[5]);
    let _b: BatchTensor = Tensor::<f32>::zeros(&[8, 10]);
    let _t2: Tensor2Df32 = Tensor::<f32>::zeros(&[2, 2]);
}

/// Verify Result and TensorError are in scope from prelude.
#[test]
fn prelude_result_type_in_scope() {
    #[allow(clippy::result_large_err)]
    fn returns_result() -> Result<i32> {
        Ok(42)
    }
    assert_eq!(returns_result().unwrap(), 42);

    // TensorError can be used as an error variant
    let _err: TensorError = TensorError::invalid_shape_simple("test".to_string());
}
