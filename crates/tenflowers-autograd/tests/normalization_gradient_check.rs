//! Finite-difference gradient checks for BatchNorm and LayerNorm.
//!
//! Regression coverage for a bug where `process_batchnorm_backward` and
//! `process_layernorm_backward` (in
//! `tape/gradient_computation/neural_ops.rs`) hardcoded the gamma/beta
//! gradients to zero, meaning trainable scale/shift parameters could never
//! learn. These tests exercise the real analytical backward kernels
//! (`ops::normalization_ops::batch_norm_backward` /
//! `layer_norm_backward`) and compare every returned gradient against an
//! independently computed numerical gradient.
//!
//! LayerNorm and rank-4 (NCHW) BatchNorm tests go through the full
//! `GradientTape`/`TrackedTensor` path end-to-end. Rank-2/rank-3 BatchNorm
//! tests call `batch_norm_backward` directly instead, because the "official"
//! forward kernel (`tenflowers_core::ops::batch_norm`, which
//! `TrackedTensor::batch_norm` calls internally) unconditionally rejects any
//! non-rank-4 input -- so the tape path is architecturally unreachable for
//! those ranks today, even though `batch_norm_backward` itself supports them
//! (see `run_batch_norm_backward_kernel_check` below for the full
//! rationale).
//!
//! Follows the finite-difference helper pattern established in
//! `numerical_gradient_check.rs` (the crate's internal
//! `tape::helpers::{compute_numerical_gradient, compare_gradients}` are
//! `pub(crate)`-only and not visible from `tests/`, so the helper is
//! reimplemented here).

use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;

/// Step size for central-difference numerical gradients. Chosen larger than
/// `numerical_gradient_check.rs`'s `1e-4` because the normalization ops here
/// combine several chained divisions/square-roots and are evaluated on
/// tensors with a handful of elements per reduction axis; at `eps=1e-4` the
/// resulting `f(x+h)-f(x-h)` central difference is dominated by f32 rounding
/// noise rather than signal (independently confirmed by hand: shrinking eps
/// below `1e-2` made the discrepancy against the analytical gradient grow,
/// not shrink -- the textbook signature of catastrophic cancellation, not a
/// formula bug).
const EPS: f32 = 1e-2;
/// Relative tolerance for analytical vs numerical gradient comparison.
const RTOL: f32 = 1e-2;
/// Absolute tolerance for analytical vs numerical gradient comparison.
const ATOL: f32 = 1e-3;

/// Compute the numerical (central-difference) gradient of a scalar-valued
/// function `f` with respect to one tensor in `inputs`, identified by
/// `input_idx`. All other tensors in `inputs` are held fixed.
fn numerical_gradient<F>(f: &F, inputs: &[Tensor<f32>], input_idx: usize, eps: f32) -> Tensor<f32>
where
    F: Fn(&[Tensor<f32>]) -> Tensor<f32>,
{
    let input = &inputs[input_idx];
    let input_data = input
        .as_slice()
        .expect("test: tensor should be contiguous")
        .to_vec();
    let mut grad_data = vec![0.0f32; input_data.len()];

    for i in 0..input_data.len() {
        let mut plus_data = input_data.clone();
        plus_data[i] += eps;
        let plus_tensor = Tensor::from_vec(plus_data, input.shape().dims())
            .expect("test: tensor construction should succeed");

        let mut minus_data = input_data.clone();
        minus_data[i] -= eps;
        let minus_tensor = Tensor::from_vec(minus_data, input.shape().dims())
            .expect("test: tensor construction should succeed");

        let mut inputs_plus = inputs.to_vec();
        let mut inputs_minus = inputs.to_vec();
        inputs_plus[input_idx] = plus_tensor;
        inputs_minus[input_idx] = minus_tensor;

        let f_plus = f(&inputs_plus);
        let f_minus = f(&inputs_minus);

        let val_plus = f_plus.as_slice().expect("test: output should be scalar")[0];
        let val_minus = f_minus.as_slice().expect("test: output should be scalar")[0];

        grad_data[i] = (val_plus - val_minus) / (2.0 * eps);
    }

    Tensor::from_vec(grad_data, input.shape().dims())
        .expect("test: gradient tensor construction should succeed")
}

/// Compare an analytical gradient tensor against a numerical one, element by
/// element, using combined relative/absolute tolerance. Panics with a
/// descriptive message on the first mismatch.
fn compare_gradients(analytical: &Tensor<f32>, numerical: &Tensor<f32>, label: &str) {
    let anal_data = analytical
        .as_slice()
        .expect("test: analytical gradient should be contiguous");
    let num_data = numerical
        .as_slice()
        .expect("test: numerical gradient should be contiguous");

    assert_eq!(
        anal_data.len(),
        num_data.len(),
        "{label}: gradient length mismatch (analytical={}, numerical={})",
        anal_data.len(),
        num_data.len()
    );

    for (i, (&a, &n)) in anal_data.iter().zip(num_data.iter()).enumerate() {
        let diff = (a - n).abs();
        let tolerance = ATOL + RTOL * n.abs();
        assert!(
            diff <= tolerance,
            "{label}: element {i} differs: analytical={a:?}, numerical={n:?}, diff={diff:?}, tolerance={tolerance:?}"
        );
    }
}

/// Assert that at least one element of a gradient tensor has a magnitude
/// exceeding a small threshold — i.e. the gradient is not degenerately all
/// zero. This is the exact regression signature of the bug being tested:
/// gamma/beta gradients were previously hardcoded to `Tensor::zeros(...)`.
fn assert_gradient_nonzero(grad: &Tensor<f32>, label: &str) {
    let data = grad
        .as_slice()
        .expect("test: gradient tensor should be contiguous");
    assert!(
        data.iter().any(|v| v.abs() > 1e-3),
        "{label}: expected a non-zero gradient, but all elements were ~0: {data:?}"
    );
}

/// Deterministic, bounded pseudo-random sequence in roughly `[-1, 1]`,
/// seeded by an arbitrary offset so different callers get different (but
/// reproducible) sequences. Uses `sin` of a linearly-growing phase rather
/// than a linearly-growing raw value, so the output stays bounded no matter
/// how many elements are requested -- unlike a naive `a + b*i` formula, which
/// drifts to large magnitudes for large `i` and makes finite-difference
/// gradient checks numerically noisy (large function values make
/// `f(x+h)-f(x-h)` dominated by f32 rounding error rather than signal).
fn bounded_pseudo_random(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 12.9898 + seed).sin() * 0.9)
        .collect()
}

/// Build a fixed, non-uniform, bounded-magnitude weight tensor with the
/// given shape, used to break the symmetry of a plain `sum()` reduction so
/// that tests are sensitive to bugs that only manifest under non-uniform
/// upstream gradients (a uniform `grad_output` of all-ones can accidentally
/// hide bugs that a real training loss would expose).
fn non_uniform_weights(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    // Offset from bounded_pseudo_random's default seed range used for inputs
    // so weights are a distinct (not-identical) sequence.
    let data: Vec<f32> = bounded_pseudo_random(n, 3.7)
        .into_iter()
        .map(|v| v + 1.2) // keep weights away from 0 to avoid a degenerate near-zero-weight element
        .collect();
    Tensor::from_vec(data, shape).expect("test: weight tensor construction should succeed")
}

// ---------------------------------------------------------------------
// BatchNorm
// ---------------------------------------------------------------------

/// Manual reference implementation of the BatchNorm forward pass, mirroring
/// exactly the formulas `ops::normalization_ops::batch_norm_backward` uses
/// internally (training mode: batch mean/var over all axes except channel
/// axis 1; eval mode: running statistics), for an arbitrary channel-second
/// rank (2, 3, or 4).
///
/// This exists because `tenflowers_core::ops::batch_norm` (the "official"
/// forward kernel wired to `TrackedTensor::batch_norm`) unconditionally
/// rejects any input that is not exactly rank 4 -- so it cannot be used as a
/// finite-difference reference for the rank-2/rank-3 branches of
/// `batch_norm_backward`, even though those branches are themselves
/// well-defined and exercised directly (bypassing the tape) below. Both
/// `batch_norm_backward`'s axis inference and this function's axis
/// inference must agree for the comparison to be meaningful, so this
/// mirrors that logic precisely rather than reimplementing it independently.
fn manual_batch_norm_forward(
    input: &Tensor<f32>,
    gamma: &Tensor<f32>,
    beta: &Tensor<f32>,
    running_mean: &Tensor<f32>,
    running_var: &Tensor<f32>,
    epsilon: f32,
    training: bool,
) -> Tensor<f32> {
    let input_shape = input.shape().dims().to_vec();
    let ndim = input_shape.len();
    let channels = input_shape[1];

    let axes: Vec<i32> = if ndim == 4 {
        vec![0, 2, 3]
    } else if ndim == 3 {
        vec![0, 2]
    } else {
        vec![0]
    };

    let mut reshape_shape = vec![1usize; ndim];
    reshape_shape[1] = channels;

    if training {
        let mean = input
            .mean(Some(&axes), true)
            .expect("test: mean should succeed");
        let centered = input.sub(&mean).expect("test: sub should succeed");
        let var = centered
            .mul(&centered)
            .expect("test: mul should succeed")
            .mean(Some(&axes), true)
            .expect("test: mean should succeed");
        let std = var
            .add(&Tensor::from_scalar(epsilon))
            .expect("test: add should succeed")
            .sqrt()
            .expect("test: sqrt should succeed");
        let normalized = centered.div(&std).expect("test: div should succeed");
        let gamma_r = gamma
            .reshape(&reshape_shape)
            .expect("test: reshape should succeed");
        let beta_r = beta
            .reshape(&reshape_shape)
            .expect("test: reshape should succeed");
        normalized
            .mul(&gamma_r)
            .expect("test: mul should succeed")
            .add(&beta_r)
            .expect("test: add should succeed")
    } else {
        let std = running_var
            .add(&Tensor::from_scalar(epsilon))
            .expect("test: add should succeed")
            .sqrt()
            .expect("test: sqrt should succeed");
        let mean_r = running_mean
            .reshape(&reshape_shape)
            .expect("test: reshape should succeed");
        let std_r = std
            .reshape(&reshape_shape)
            .expect("test: reshape should succeed");
        let centered = input.sub(&mean_r).expect("test: sub should succeed");
        let normalized = centered.div(&std_r).expect("test: div should succeed");
        let gamma_r = gamma
            .reshape(&reshape_shape)
            .expect("test: reshape should succeed");
        let beta_r = beta
            .reshape(&reshape_shape)
            .expect("test: reshape should succeed");
        normalized
            .mul(&gamma_r)
            .expect("test: mul should succeed")
            .add(&beta_r)
            .expect("test: add should succeed")
    }
}

/// Direct-kernel BatchNorm gradient check harness for ranks the "official"
/// `tenflowers_core::ops::batch_norm` forward kernel cannot accept (2 and
/// 3): calls `ops::normalization_ops::batch_norm_backward` directly with a
/// hand-supplied, non-uniform `grad_output` (bypassing `TrackedTensor`/the
/// tape entirely, since `TrackedTensor::batch_norm` would itself fail its
/// internal forward call for these ranks), and compares against a numerical
/// gradient computed via `manual_batch_norm_forward` above.
#[allow(clippy::too_many_arguments)]
fn run_batch_norm_backward_kernel_check(
    label: &str,
    input_shape: &[usize],
    channels: usize,
    input_data: Vec<f32>,
    gamma_data: Vec<f32>,
    beta_data: Vec<f32>,
    running_mean_data: Vec<f32>,
    running_var_data: Vec<f32>,
    epsilon: f32,
    training: bool,
) {
    let grad_output = non_uniform_weights(input_shape);

    let input = Tensor::from_vec(input_data.clone(), input_shape)
        .expect("test: input tensor construction should succeed");
    let gamma = Tensor::from_vec(gamma_data.clone(), &[channels])
        .expect("test: gamma tensor construction should succeed");
    let beta = Tensor::from_vec(beta_data.clone(), &[channels])
        .expect("test: beta tensor construction should succeed");
    let running_mean = Tensor::from_vec(running_mean_data.clone(), &[channels])
        .expect("test: running_mean tensor construction should succeed");
    let running_var = Tensor::from_vec(running_var_data.clone(), &[channels])
        .expect("test: running_var tensor construction should succeed");

    let (grad_input, grad_gamma, grad_beta) =
        tenflowers_autograd::ops::normalization_ops::batch_norm_backward(
            &grad_output,
            &input,
            &gamma,
            &beta,
            &running_mean,
            &running_var,
            training,
            epsilon,
        )
        .expect("test: batch_norm_backward should succeed");

    let raw_inputs = vec![input, gamma, beta, running_mean, running_var];
    let loss_fn = |t: &[Tensor<f32>]| -> Tensor<f32> {
        let out = manual_batch_norm_forward(&t[0], &t[1], &t[2], &t[3], &t[4], epsilon, training);
        let weighted = out.mul(&grad_output).expect("test: raw mul should succeed");
        weighted
            .sum(None, false)
            .expect("test: raw sum should succeed")
    };

    let num_grad_input = numerical_gradient(&loss_fn, &raw_inputs, 0, EPS);
    let num_grad_gamma = numerical_gradient(&loss_fn, &raw_inputs, 1, EPS);
    let num_grad_beta = numerical_gradient(&loss_fn, &raw_inputs, 2, EPS);

    compare_gradients(
        &grad_input,
        &num_grad_input,
        &format!("{label}: grad_input"),
    );
    compare_gradients(
        &grad_gamma,
        &num_grad_gamma,
        &format!("{label}: grad_gamma"),
    );
    compare_gradients(&grad_beta, &num_grad_beta, &format!("{label}: grad_beta"));

    assert_gradient_nonzero(&grad_gamma, &format!("{label}: grad_gamma"));
    assert_gradient_nonzero(&grad_beta, &format!("{label}: grad_beta"));

    println!(
        "[{label}] grad_gamma analytical={:?}",
        grad_gamma.as_slice().unwrap_or(&[])
    );
    println!(
        "[{label}] grad_gamma numerical ={:?}",
        num_grad_gamma.as_slice().unwrap_or(&[])
    );
    println!(
        "[{label}] grad_beta  analytical={:?}",
        grad_beta.as_slice().unwrap_or(&[])
    );
    println!(
        "[{label}] grad_beta  numerical ={:?}",
        num_grad_beta.as_slice().unwrap_or(&[])
    );
}

/// Shared harness for a BatchNorm gradient check: builds tracked tensors,
/// computes a weighted-sum scalar loss through `TrackedTensor::batch_norm`,
/// gets analytical gradients from the tape, and compares each against a
/// numerical gradient computed by re-running the raw
/// `tenflowers_core::ops::batch_norm` forward kernel under central finite
/// differences.
///
/// Only usable for rank-4 (NCHW) input, since `tenflowers_core::ops::batch_norm`
/// unconditionally rejects any other rank. Rank 2/3 coverage is provided by
/// `run_batch_norm_backward_kernel_check` above instead.
#[allow(clippy::too_many_arguments)]
fn run_batch_norm_gradient_check(
    label: &str,
    input_shape: &[usize],
    channels: usize,
    input_data: Vec<f32>,
    gamma_data: Vec<f32>,
    beta_data: Vec<f32>,
    running_mean_data: Vec<f32>,
    running_var_data: Vec<f32>,
    epsilon: f32,
    training: bool,
) {
    let weights = non_uniform_weights(input_shape);

    // --- Analytical gradients via GradientTape ---
    let tape = GradientTape::new();

    let mut input_tensor = Tensor::from_vec(input_data.clone(), input_shape)
        .expect("test: input tensor construction should succeed");
    let mut gamma_tensor = Tensor::from_vec(gamma_data.clone(), &[channels])
        .expect("test: gamma tensor construction should succeed");
    let mut beta_tensor = Tensor::from_vec(beta_data.clone(), &[channels])
        .expect("test: beta tensor construction should succeed");
    let running_mean_tensor = Tensor::from_vec(running_mean_data.clone(), &[channels])
        .expect("test: running_mean tensor construction should succeed");
    let running_var_tensor = Tensor::from_vec(running_var_data.clone(), &[channels])
        .expect("test: running_var tensor construction should succeed");

    input_tensor.set_requires_grad(true);
    gamma_tensor.set_requires_grad(true);
    beta_tensor.set_requires_grad(true);

    let input_t = tape.watch(input_tensor);
    let gamma_t = tape.watch(gamma_tensor);
    let beta_t = tape.watch(beta_tensor);
    let running_mean_t = tape.watch(running_mean_tensor);
    let running_var_t = tape.watch(running_var_tensor);
    let weights_t = tape.watch(weights.clone());

    let normalized = input_t
        .batch_norm(
            &gamma_t,
            &beta_t,
            &running_mean_t,
            &running_var_t,
            epsilon,
            training,
        )
        .expect("test: batch_norm forward should succeed");
    let weighted = normalized
        .mul(&weights_t)
        .expect("test: elementwise mul should succeed");
    let loss = weighted
        .sum(None, false)
        .expect("test: sum reduction should succeed");

    let grads = tape
        .gradient(&[loss], &[input_t, gamma_t, beta_t])
        .expect("test: gradient computation should succeed");

    let grad_input = grads[0]
        .clone()
        .expect("test: input should receive a gradient");
    let grad_gamma = grads[1]
        .clone()
        .expect("test: gamma should receive a gradient");
    let grad_beta = grads[2]
        .clone()
        .expect("test: beta should receive a gradient");

    // --- Numerical gradients via raw batch_norm forward kernel ---
    // inputs layout: [input, gamma, beta, running_mean, running_var]
    let raw_inputs = vec![
        Tensor::from_vec(input_data, input_shape)
            .expect("test: raw input tensor construction should succeed"),
        Tensor::from_vec(gamma_data, &[channels])
            .expect("test: raw gamma tensor construction should succeed"),
        Tensor::from_vec(beta_data, &[channels])
            .expect("test: raw beta tensor construction should succeed"),
        Tensor::from_vec(running_mean_data, &[channels])
            .expect("test: raw running_mean tensor construction should succeed"),
        Tensor::from_vec(running_var_data, &[channels])
            .expect("test: raw running_var tensor construction should succeed"),
    ];

    let loss_fn = |t: &[Tensor<f32>]| -> Tensor<f32> {
        let out =
            tenflowers_core::ops::batch_norm(&t[0], &t[1], &t[2], &t[3], &t[4], epsilon, training)
                .expect("test: raw batch_norm forward should succeed");
        let weighted = out.mul(&weights).expect("test: raw mul should succeed");
        weighted
            .sum(None, false)
            .expect("test: raw sum should succeed")
    };

    let num_grad_input = numerical_gradient(&loss_fn, &raw_inputs, 0, EPS);
    let num_grad_gamma = numerical_gradient(&loss_fn, &raw_inputs, 1, EPS);
    let num_grad_beta = numerical_gradient(&loss_fn, &raw_inputs, 2, EPS);

    compare_gradients(
        &grad_input,
        &num_grad_input,
        &format!("{label}: grad_input"),
    );
    compare_gradients(
        &grad_gamma,
        &num_grad_gamma,
        &format!("{label}: grad_gamma"),
    );
    compare_gradients(&grad_beta, &num_grad_beta, &format!("{label}: grad_beta"));

    // This is the exact regression the bug produced: gamma/beta gradients
    // were previously hardcoded to all-zero.
    assert_gradient_nonzero(&grad_gamma, &format!("{label}: grad_gamma"));
    assert_gradient_nonzero(&grad_beta, &format!("{label}: grad_beta"));

    println!(
        "[{label}] grad_gamma analytical={:?}",
        grad_gamma.as_slice().unwrap_or(&[])
    );
    println!(
        "[{label}] grad_gamma numerical ={:?}",
        num_grad_gamma.as_slice().unwrap_or(&[])
    );
    println!(
        "[{label}] grad_beta  analytical={:?}",
        grad_beta.as_slice().unwrap_or(&[])
    );
    println!(
        "[{label}] grad_beta  numerical ={:?}",
        num_grad_beta.as_slice().unwrap_or(&[])
    );
}

// NOTE on rank 2/3 BatchNorm tests below: `tenflowers_core::ops::batch_norm`
// (the forward kernel `TrackedTensor::batch_norm` calls internally)
// unconditionally rejects any input that is not rank 4 -- confirmed
// empirically (`InvalidShape { reason: "BatchNorm expects 4D input (NCHW
// format)" }`) even for rank 2, which is not obvious from the kernel's own
// per-rank backward-side axis handling. This means `TrackedTensor::batch_norm`
// can never succeed end-to-end for rank 2/3 today, so these tests instead
// call `ops::normalization_ops::batch_norm_backward` directly (still the
// real production kernel, just bypassing the tape/forward-kernel rank gate)
// via `run_batch_norm_backward_kernel_check`, verified against a manually
// implemented reference forward pass. Rank 4 (`test_batch_norm_*_rank4_*`
// below) exercises the full `TrackedTensor`/tape path end-to-end instead.

#[test]
fn test_batch_norm_training_mode_rank2_gradients() {
    // input [N=4, C=3]
    let input_data: Vec<f32> = vec![
        0.5, -1.2, 2.0, // sample 0
        1.3, 0.7, -0.4, // sample 1
        -0.8, 1.9, 0.2, // sample 2
        2.2, -0.3, -1.5, // sample 3
    ];
    let gamma_data = vec![1.5, 0.8, 1.2];
    let beta_data = vec![0.1, -0.2, 0.3];
    // running stats are unused in training mode but must still be provided
    // to build the op.
    let running_mean_data = vec![0.0, 0.0, 0.0];
    let running_var_data = vec![1.0, 1.0, 1.0];

    run_batch_norm_backward_kernel_check(
        "batch_norm_training_rank2",
        &[4, 3],
        3,
        input_data,
        gamma_data,
        beta_data,
        running_mean_data,
        running_var_data,
        1e-5,
        true,
    );
}

#[test]
fn test_batch_norm_eval_mode_rank2_gradients() {
    // input [N=4, C=3], eval mode with non-trivial running stats distinct
    // from training-mode batch stats (0/1) so the eval-mode formula is
    // actually exercised.
    let input_data: Vec<f32> = vec![
        0.5, -1.2, 2.0, 1.3, 0.7, -0.4, -0.8, 1.9, 0.2, 2.2, -0.3, -1.5,
    ];
    let gamma_data = vec![1.5, 0.8, 1.2];
    let beta_data = vec![0.1, -0.2, 0.3];
    let running_mean_data = vec![0.5, -0.2, 0.1];
    let running_var_data = vec![2.0, 0.5, 1.5];

    run_batch_norm_backward_kernel_check(
        "batch_norm_eval_rank2",
        &[4, 3],
        3,
        input_data,
        gamma_data,
        beta_data,
        running_mean_data,
        running_var_data,
        1e-5,
        false,
    );
}

#[test]
fn test_batch_norm_training_mode_rank3_ncl_gradients() {
    // input [N=2, C=3, L=4] (NCL convention: channel at index 1)
    let input_data: Vec<f32> = bounded_pseudo_random(24, 0.5);
    let gamma_data = vec![1.1, 0.9, 1.4];
    let beta_data = vec![-0.1, 0.2, 0.05];
    let running_mean_data = vec![0.0, 0.0, 0.0];
    let running_var_data = vec![1.0, 1.0, 1.0];

    run_batch_norm_backward_kernel_check(
        "batch_norm_training_rank3_ncl",
        &[2, 3, 4],
        3,
        input_data,
        gamma_data,
        beta_data,
        running_mean_data,
        running_var_data,
        1e-5,
        true,
    );
}

#[test]
fn test_batch_norm_eval_mode_rank3_ncl_gradients() {
    // input [N=2, C=3, L=4], eval mode with non-trivial running stats.
    let input_data: Vec<f32> = bounded_pseudo_random(24, 0.5);
    let gamma_data = vec![1.1, 0.9, 1.4];
    let beta_data = vec![-0.1, 0.2, 0.05];
    let running_mean_data = vec![0.4, -0.3, 0.2];
    let running_var_data = vec![1.8, 0.6, 2.2];

    run_batch_norm_backward_kernel_check(
        "batch_norm_eval_rank3_ncl",
        &[2, 3, 4],
        3,
        input_data,
        gamma_data,
        beta_data,
        running_mean_data,
        running_var_data,
        1e-5,
        false,
    );
}

#[test]
fn test_batch_norm_training_mode_rank4_nchw_gradients() {
    // input [N=2, C=3, H=4, W=4]
    let input_data: Vec<f32> = bounded_pseudo_random(96, 1.1);
    let gamma_data = vec![1.3, 0.7, 1.6];
    let beta_data = vec![0.05, -0.1, 0.2];
    let running_mean_data = vec![0.0, 0.0, 0.0];
    let running_var_data = vec![1.0, 1.0, 1.0];

    run_batch_norm_gradient_check(
        "batch_norm_training_rank4_nchw",
        &[2, 3, 4, 4],
        3,
        input_data,
        gamma_data,
        beta_data,
        running_mean_data,
        running_var_data,
        1e-5,
        true,
    );
}

// ---------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------

/// Shared harness for a LayerNorm gradient check, mirroring
/// `run_batch_norm_gradient_check` above.
fn run_layer_norm_gradient_check(
    label: &str,
    input_shape: &[usize],
    normalized_shape: Vec<usize>,
    input_data: Vec<f32>,
    gamma_data: Vec<f32>,
    beta_data: Vec<f32>,
    epsilon: f32,
) {
    let weights = non_uniform_weights(input_shape);
    let gamma_beta_shape: Vec<usize> = normalized_shape.clone();

    // --- Analytical gradients via GradientTape ---
    let tape = GradientTape::new();

    let mut input_tensor = Tensor::from_vec(input_data.clone(), input_shape)
        .expect("test: input tensor construction should succeed");
    let mut gamma_tensor = Tensor::from_vec(gamma_data.clone(), &gamma_beta_shape)
        .expect("test: gamma tensor construction should succeed");
    let mut beta_tensor = Tensor::from_vec(beta_data.clone(), &gamma_beta_shape)
        .expect("test: beta tensor construction should succeed");

    input_tensor.set_requires_grad(true);
    gamma_tensor.set_requires_grad(true);
    beta_tensor.set_requires_grad(true);

    let input_t = tape.watch(input_tensor);
    let gamma_t = tape.watch(gamma_tensor);
    let beta_t = tape.watch(beta_tensor);
    let weights_t = tape.watch(weights.clone());

    let normalized = input_t
        .layer_norm(&gamma_t, &beta_t, normalized_shape.clone(), epsilon)
        .expect("test: layer_norm forward should succeed");
    let weighted = normalized
        .mul(&weights_t)
        .expect("test: elementwise mul should succeed");
    let loss = weighted
        .sum(None, false)
        .expect("test: sum reduction should succeed");

    let grads = tape
        .gradient(&[loss], &[input_t, gamma_t, beta_t])
        .expect("test: gradient computation should succeed");

    let grad_input = grads[0]
        .clone()
        .expect("test: input should receive a gradient");
    let grad_gamma = grads[1]
        .clone()
        .expect("test: gamma should receive a gradient");
    let grad_beta = grads[2]
        .clone()
        .expect("test: beta should receive a gradient");

    // --- Numerical gradients via raw layer_norm forward kernel ---
    // inputs layout: [input, gamma, beta]
    let raw_inputs = vec![
        Tensor::from_vec(input_data, input_shape)
            .expect("test: raw input tensor construction should succeed"),
        Tensor::from_vec(gamma_data, &gamma_beta_shape)
            .expect("test: raw gamma tensor construction should succeed"),
        Tensor::from_vec(beta_data, &gamma_beta_shape)
            .expect("test: raw beta tensor construction should succeed"),
    ];

    let loss_fn = |t: &[Tensor<f32>]| -> Tensor<f32> {
        let out = tenflowers_core::ops::layer_norm(&t[0], &t[1], &t[2], &normalized_shape, epsilon)
            .expect("test: raw layer_norm forward should succeed");
        let weighted = out.mul(&weights).expect("test: raw mul should succeed");
        weighted
            .sum(None, false)
            .expect("test: raw sum should succeed")
    };

    let num_grad_input = numerical_gradient(&loss_fn, &raw_inputs, 0, EPS);
    let num_grad_gamma = numerical_gradient(&loss_fn, &raw_inputs, 1, EPS);
    let num_grad_beta = numerical_gradient(&loss_fn, &raw_inputs, 2, EPS);

    compare_gradients(
        &grad_input,
        &num_grad_input,
        &format!("{label}: grad_input"),
    );
    compare_gradients(
        &grad_gamma,
        &num_grad_gamma,
        &format!("{label}: grad_gamma"),
    );
    compare_gradients(&grad_beta, &num_grad_beta, &format!("{label}: grad_beta"));

    assert_gradient_nonzero(&grad_gamma, &format!("{label}: grad_gamma"));
    assert_gradient_nonzero(&grad_beta, &format!("{label}: grad_beta"));

    println!(
        "[{label}] grad_gamma analytical={:?}",
        grad_gamma.as_slice().unwrap_or(&[])
    );
    println!(
        "[{label}] grad_gamma numerical ={:?}",
        num_grad_gamma.as_slice().unwrap_or(&[])
    );
    println!(
        "[{label}] grad_beta  analytical={:?}",
        grad_beta.as_slice().unwrap_or(&[])
    );
    println!(
        "[{label}] grad_beta  numerical ={:?}",
        num_grad_beta.as_slice().unwrap_or(&[])
    );
}

#[test]
fn test_layer_norm_rank2_gradients() {
    // input [N=4, features=8], normalized_shape=[8]
    let input_data: Vec<f32> = bounded_pseudo_random(32, 2.3);
    let gamma_data: Vec<f32> = (0..8).map(|i| 0.8 + 0.1 * i as f32).collect();
    let beta_data: Vec<f32> = (0..8).map(|i| -0.2 + 0.05 * i as f32).collect();

    run_layer_norm_gradient_check(
        "layer_norm_rank2",
        &[4, 8],
        vec![8],
        input_data,
        gamma_data,
        beta_data,
        1e-5,
    );
}

#[test]
fn test_layer_norm_rank3_gradients() {
    // input [N=2, S=3, features=8], normalized_shape=[8] (normalizing only
    // the last dim -- the common transformer LayerNorm case).
    let input_data: Vec<f32> = bounded_pseudo_random(48, 4.1);
    let gamma_data: Vec<f32> = (0..8).map(|i| 1.0 + 0.07 * i as f32).collect();
    let beta_data: Vec<f32> = (0..8).map(|i| 0.1 - 0.03 * i as f32).collect();

    run_layer_norm_gradient_check(
        "layer_norm_rank3",
        &[2, 3, 8],
        vec![8],
        input_data,
        gamma_data,
        beta_data,
        1e-5,
    );
}
