//! End-to-end gradient-correctness tests for `Operation::Slice`, `Concat`,
//! `Stack`, `Split`, and `Gather`.
//!
//! Every test in this file goes through the full tape record -> backward
//! path via `TrackedTensor` methods and `GradientTape::gradient(...)`, to
//! prove the WIRING works end-to-end (dispatch arm in
//! `process_operation_backward`, forward-record methods on `TrackedTensor`,
//! and the underlying `grad_ops::*_backward` math), not just the isolated
//! math functions (which have their own unit tests in
//! `crates/tenflowers-autograd/src/grad_ops/tensor_ops.rs`).

use tenflowers_autograd::grad_ops::SliceSpec;
use tenflowers_autograd::{GradientTape, TrackedTensor};
use tenflowers_core::Tensor;

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "index {i}: actual {a} != expected {e} (tol {tol})"
        );
    }
}

// ---------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------

#[test]
fn test_tape_slice_contiguous_range_gradient() {
    // x shape [6], slice [1..4]: grad_input must be [0, g0, g1, g2, 0, 0].
    let tape = GradientTape::new();
    let x = Tensor::<f32>::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[6])
        .expect("test: tensor creation should succeed");
    let x_tracked = tape.watch(x);

    let y = x_tracked
        .slice(&[SliceSpec::range(1, 4)])
        .expect("test: slice should succeed");
    assert_eq!(y.shape().dims(), &[3]);

    let gradients = tape
        .gradient(&[y], &[x_tracked])
        .expect("test: gradient computation should succeed");
    let grad_x = gradients[0]
        .as_ref()
        .expect("test: gradient for x should exist");
    assert_eq!(grad_x.shape().dims(), &[6]);
    let data = grad_x.to_vec().expect("test: to_vec should succeed");
    // Target is the sum of y (grad_output = ones), so grad lands as 1s at
    // the sliced positions and 0 elsewhere.
    assert_close(&data, &[0.0, 1.0, 1.0, 1.0, 0.0, 0.0], 1e-6);
}

#[test]
fn test_tape_slice_strided_step_2_gradient() {
    // x shape [6], slice picking indices [0, 2, 4] (step=2): grad must land
    // ONLY at those 3 positions.
    let tape = GradientTape::new();
    let x = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6])
        .expect("test: tensor creation should succeed");
    let x_tracked = tape.watch(x);

    let y = x_tracked
        .slice(&[SliceSpec::range_with_step(0, 6, 2)])
        .expect("test: strided slice should succeed");
    assert_eq!(y.shape().dims(), &[3]);

    let gradients = tape
        .gradient(&[y], &[x_tracked])
        .expect("test: gradient computation should succeed");
    let grad_x = gradients[0]
        .as_ref()
        .expect("test: gradient for x should exist");
    let data = grad_x.to_vec().expect("test: to_vec should succeed");
    assert_close(&data, &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 1e-6);
}

// ---------------------------------------------------------------------
// Concat
// ---------------------------------------------------------------------

#[test]
fn test_tape_concat_axis0_different_sizes_gradient() {
    // Three inputs of DIFFERENT sizes along axis 0: shapes [2,4], [3,4], [1,4]
    // concatenated on axis 0 -> [6,4]. Each gradient must have the exact
    // input shape and exact values (a slice of grad_output, which here is
    // all-ones since the target is sum(concat_result)).
    let tape = GradientTape::new();
    let a = Tensor::<f32>::from_vec((0..8).map(|i| i as f32).collect(), &[2, 4])
        .expect("test: tensor creation should succeed");
    let b = Tensor::<f32>::from_vec((0..12).map(|i| i as f32).collect(), &[3, 4])
        .expect("test: tensor creation should succeed");
    let c = Tensor::<f32>::from_vec((0..4).map(|i| i as f32).collect(), &[1, 4])
        .expect("test: tensor creation should succeed");

    let a_tracked = tape.watch(a);
    let b_tracked = tape.watch(b);
    let c_tracked = tape.watch(c);

    let y = TrackedTensor::concat(&[&a_tracked, &b_tracked, &c_tracked], 0)
        .expect("test: concat should succeed");
    assert_eq!(y.shape().dims(), &[6, 4]);

    let gradients = tape
        .gradient(&[y], &[a_tracked, b_tracked, c_tracked])
        .expect("test: gradient computation should succeed");
    assert_eq!(gradients.len(), 3);

    let grad_a = gradients[0].as_ref().expect("test: grad_a should exist");
    let grad_b = gradients[1].as_ref().expect("test: grad_b should exist");
    let grad_c = gradients[2].as_ref().expect("test: grad_c should exist");

    assert_eq!(grad_a.shape().dims(), &[2, 4]);
    assert_eq!(grad_b.shape().dims(), &[3, 4]);
    assert_eq!(grad_c.shape().dims(), &[1, 4]);

    // grad_output for sum() is all ones, so every gradient is all ones of
    // its own input's shape.
    assert_close(
        &grad_a.to_vec().expect("test: to_vec should succeed"),
        &[1.0; 8],
        1e-6,
    );
    assert_close(
        &grad_b.to_vec().expect("test: to_vec should succeed"),
        &[1.0; 12],
        1e-6,
    );
    assert_close(
        &grad_c.to_vec().expect("test: to_vec should succeed"),
        &[1.0; 4],
        1e-6,
    );
}

#[test]
fn test_tape_concat_nonzero_nonlast_axis_gradient() {
    // Two 3D tensors concatenated on axis=1 (non-zero, non-last axis).
    // a shape [2, 1, 3], b shape [2, 2, 3] -> concat axis=1 -> [2, 3, 3].
    let tape = GradientTape::new();
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3])
        .expect("test: tensor creation should succeed");
    let b = Tensor::<f32>::from_vec((0..12).map(|i| (i as f32) * 10.0).collect(), &[2, 2, 3])
        .expect("test: tensor creation should succeed");

    let a_tracked = tape.watch(a);
    let b_tracked = tape.watch(b);

    let y = TrackedTensor::concat(&[&a_tracked, &b_tracked], 1).expect("test: concat should succeed");
    assert_eq!(y.shape().dims(), &[2, 3, 3]);

    let gradients = tape
        .gradient(&[y], &[a_tracked, b_tracked])
        .expect("test: gradient computation should succeed");
    let grad_a = gradients[0].as_ref().expect("test: grad_a should exist");
    let grad_b = gradients[1].as_ref().expect("test: grad_b should exist");

    assert_eq!(grad_a.shape().dims(), &[2, 1, 3]);
    assert_eq!(grad_b.shape().dims(), &[2, 2, 3]);
    assert_close(
        &grad_a.to_vec().expect("test: to_vec should succeed"),
        &[1.0; 6],
        1e-6,
    );
    assert_close(
        &grad_b.to_vec().expect("test: to_vec should succeed"),
        &[1.0; 12],
        1e-6,
    );
}

// ---------------------------------------------------------------------
// Stack
// ---------------------------------------------------------------------

#[test]
fn test_tape_stack_axis0_gradient() {
    let tape = GradientTape::new();
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])
        .expect("test: tensor creation should succeed");
    let b = Tensor::<f32>::from_vec(vec![4.0, 5.0, 6.0], &[3])
        .expect("test: tensor creation should succeed");

    let a_tracked = tape.watch(a);
    let b_tracked = tape.watch(b);

    let y = TrackedTensor::stack(&[&a_tracked, &b_tracked], 0).expect("test: stack should succeed");
    assert_eq!(y.shape().dims(), &[2, 3]);

    let gradients = tape
        .gradient(&[y], &[a_tracked, b_tracked])
        .expect("test: gradient computation should succeed");
    let grad_a = gradients[0].as_ref().expect("test: grad_a should exist");
    let grad_b = gradients[1].as_ref().expect("test: grad_b should exist");

    assert_eq!(grad_a.shape().dims(), &[3]);
    assert_eq!(grad_b.shape().dims(), &[3]);
    assert_close(
        &grad_a.to_vec().expect("test: to_vec should succeed"),
        &[1.0, 1.0, 1.0],
        1e-6,
    );
    assert_close(
        &grad_b.to_vec().expect("test: to_vec should succeed"),
        &[1.0, 1.0, 1.0],
        1e-6,
    );
}

#[test]
fn test_tape_stack_negative_axis_gradient() {
    // axis=-1 on two [2] tensors -> stacked shape [2, 2] (new dim last).
    let tape = GradientTape::new();
    let a = Tensor::<f32>::from_vec(vec![1.0, 2.0], &[2])
        .expect("test: tensor creation should succeed");
    let b = Tensor::<f32>::from_vec(vec![3.0, 4.0], &[2])
        .expect("test: tensor creation should succeed");

    let a_tracked = tape.watch(a);
    let b_tracked = tape.watch(b);

    let y =
        TrackedTensor::stack(&[&a_tracked, &b_tracked], -1).expect("test: stack should succeed");
    // Stacking two [2] tensors on the new last axis -> [2, 2].
    assert_eq!(y.shape().dims(), &[2, 2]);

    let gradients = tape
        .gradient(&[y], &[a_tracked, b_tracked])
        .expect("test: gradient computation should succeed");
    let grad_a = gradients[0].as_ref().expect("test: grad_a should exist");
    let grad_b = gradients[1].as_ref().expect("test: grad_b should exist");

    assert_eq!(grad_a.shape().dims(), &[2]);
    assert_eq!(grad_b.shape().dims(), &[2]);
    assert_close(
        &grad_a.to_vec().expect("test: to_vec should succeed"),
        &[1.0, 1.0],
        1e-6,
    );
    assert_close(
        &grad_b.to_vec().expect("test: to_vec should succeed"),
        &[1.0, 1.0],
        1e-6,
    );
}

// ---------------------------------------------------------------------
// Split
// ---------------------------------------------------------------------

#[test]
fn test_tape_split_inverts_concat_gradient() {
    // x shape [4, 2], split into 2 pieces along axis 0 -> two [2, 2] pieces.
    // Verify each piece's gradient is exactly the corresponding slice of the
    // upstream gradient (here: distinct per-piece target multipliers, so we
    // can tell the pieces apart).
    let tape = GradientTape::new();
    let x = Tensor::<f32>::from_vec((0..8).map(|i| i as f32).collect(), &[4, 2])
        .expect("test: tensor creation should succeed");
    let x_tracked = tape.watch(x);

    let pieces = x_tracked
        .split(2, 0)
        .expect("test: split should succeed");
    assert_eq!(pieces.len(), 2);
    assert_eq!(pieces[0].shape().dims(), &[2, 2]);
    assert_eq!(pieces[1].shape().dims(), &[2, 2]);

    // Use only the first piece as the target: its gradient should be all
    // ones (since target = sum(piece0)), and the SECOND piece contributes
    // NOTHING to grad_x (rows 2..4 must be exactly zero), proving the split
    // wiring correctly isolates each output's contribution.
    let gradients = tape
        .gradient(&[pieces[0].clone()], &[x_tracked])
        .expect("test: gradient computation should succeed");
    let grad_x = gradients[0]
        .as_ref()
        .expect("test: gradient for x should exist");
    assert_eq!(grad_x.shape().dims(), &[4, 2]);
    let data = grad_x.to_vec().expect("test: to_vec should succeed");
    assert_close(&data, &[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], 1e-6);
}

#[test]
fn test_tape_split_axis1_three_pieces_gradient() {
    // x shape [2, 6], split into 3 pieces along axis 1 -> three [2, 2]
    // pieces. Sum ALL pieces back together (equivalent to identity on x) and
    // verify grad_x is all-ones everywhere (every element of x contributes
    // to exactly one piece, each weighted 1).
    let tape = GradientTape::new();
    let x = Tensor::<f32>::from_vec((0..12).map(|i| i as f32).collect(), &[2, 6])
        .expect("test: tensor creation should succeed");
    let x_tracked = tape.watch(x);

    let pieces = x_tracked.split(3, 1).expect("test: split should succeed");
    assert_eq!(pieces.len(), 3);
    for piece in &pieces {
        assert_eq!(piece.shape().dims(), &[2, 2]);
    }

    // Passing all three pieces as targets seeds EACH piece's own gradient to
    // ones independently; the backward pass then accumulates (adds) every
    // piece's contribution into the shared grad_x, exercising the multi-node
    // accumulation path end-to-end.
    let gradients = tape
        .gradient(&pieces, &[x_tracked])
        .expect("test: gradient computation should succeed");
    let grad_x = gradients[0]
        .as_ref()
        .expect("test: gradient for x should exist");
    assert_eq!(grad_x.shape().dims(), &[2, 6]);
    let data = grad_x.to_vec().expect("test: to_vec should succeed");
    assert_close(&data, &[1.0; 12], 1e-6);
}

// ---------------------------------------------------------------------
// Gather
// ---------------------------------------------------------------------

#[test]
fn test_tape_gather_repeated_index_accumulates_gradient() {
    // params shape [4, 3], indices = [0, 2, 0] along axis 0 (mandatory
    // same-index-gathered-twice accumulation case). Since grad_output for
    // sum(gathered) is all ones, grad_input[0] must equal 2.0 (gathered
    // TWICE: by indices[0] and indices[2]), grad_input[2] must equal 1.0
    // (gathered once, by indices[1]), and grad_input[1]/grad_input[3] (never
    // gathered) must be exactly zero.
    let tape = GradientTape::new();
    let params = Tensor::<f32>::from_vec((0..12).map(|i| i as f32).collect(), &[4, 3])
        .expect("test: tensor creation should succeed");
    let indices = Tensor::<i32>::from_vec(vec![0, 2, 0], &[3])
        .expect("test: index tensor creation should succeed");

    let params_tracked = tape.watch(params);
    let gathered = params_tracked
        .gather(&indices, 0)
        .expect("test: gather should succeed");
    assert_eq!(gathered.shape().dims(), &[3, 3]);

    let gradients = tape
        .gradient(&[gathered], &[params_tracked])
        .expect("test: gradient computation should succeed");
    let grad_params = gradients[0]
        .as_ref()
        .expect("test: gradient for params should exist");
    assert_eq!(grad_params.shape().dims(), &[4, 3]);
    let data = grad_params.to_vec().expect("test: to_vec should succeed");
    #[rustfmt::skip]
    let expected = vec![
        2.0, 2.0, 2.0, // row 0: gathered twice (indices[0] and indices[2])
        0.0, 0.0, 0.0, // row 1: never gathered
        1.0, 1.0, 1.0, // row 2: gathered once (indices[1])
        0.0, 0.0, 0.0, // row 3: never gathered
    ];
    assert_close(&data, &expected, 1e-6);
}

#[test]
fn test_tape_gather_scalar_index_squeezes_axis_gradient() {
    // Scalar (0-D) indices select a single slice along `axis` and drop that
    // dimension entirely (TensorFlow-style tf.gather semantics, NOT
    // PyTorch's same-rank torch.gather). params shape [3, 4], scalar
    // index=1 along axis 0 -> output shape [4].
    let tape = GradientTape::new();
    let params = Tensor::<f32>::from_vec((0..12).map(|i| i as f32).collect(), &[3, 4])
        .expect("test: tensor creation should succeed");
    let indices =
        Tensor::<i32>::from_vec(vec![1], &[]).expect("test: scalar index tensor should succeed");

    let params_tracked = tape.watch(params);
    let gathered = params_tracked
        .gather(&indices, 0)
        .expect("test: scalar gather should succeed");
    assert_eq!(gathered.shape().dims(), &[4]);

    let gradients = tape
        .gradient(&[gathered], &[params_tracked])
        .expect("test: gradient computation should succeed");
    let grad_params = gradients[0]
        .as_ref()
        .expect("test: gradient for params should exist");
    assert_eq!(grad_params.shape().dims(), &[3, 4]);
    let data = grad_params.to_vec().expect("test: to_vec should succeed");
    #[rustfmt::skip]
    let expected = vec![
        0.0, 0.0, 0.0, 0.0, // row 0: never gathered
        1.0, 1.0, 1.0, 1.0, // row 1: gathered (scalar index=1)
        0.0, 0.0, 0.0, 0.0, // row 2: never gathered
    ];
    assert_close(&data, &expected, 1e-6);
}

#[test]
fn test_tape_gather_out_of_bounds_index_is_error() {
    let tape = GradientTape::new();
    let params = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])
        .expect("test: tensor creation should succeed");
    let indices =
        Tensor::<i32>::from_vec(vec![7], &[1]).expect("test: index tensor creation should succeed");
    let params_tracked = tape.watch(params);
    let result = params_tracked.gather(&indices, 0);
    assert!(result.is_err());
}
