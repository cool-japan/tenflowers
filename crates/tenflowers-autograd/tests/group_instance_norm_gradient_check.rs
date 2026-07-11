//! Finite-difference gradient checks for GroupNorm and InstanceNorm.
//!
//! These exercise the tape dispatch arms `Operation::GroupNorm` /
//! `Operation::InstanceNorm` in `tape/gradient_computation/core.rs`, which
//! wire the real analytical backward kernels
//! (`ops::normalization_ops::group_norm_backward` /
//! `instance_norm_backward`) through `TrackedTensor::group_norm` /
//! `TrackedTensor::instance_norm` end-to-end via `GradientTape`.
//!
//! Unlike a shape-only smoke test, every gradient returned by the tape
//! (`grad_input`, `grad_gamma`, `grad_beta`) is compared element-by-element
//! against an independently computed central-difference numerical gradient
//! of the same scalar loss, evaluated through the "official" forward kernel
//! (`tenflowers_core::ops::group_norm`, called both by `TrackedTensor`'s
//! forward pass and directly here as the numerical-gradient reference). This
//! is the strongest available signal that the backward kernels are not a
//! "zeros stub" -- a stub would produce `grad_gamma`/`grad_beta` that are
//! exactly zero while the numerical reference is not, which
//! `assert_gradient_nonzero` below explicitly guards against in addition to
//! the numeric-closeness check.
//!
//! Follows the finite-difference helper pattern established in
//! `normalization_gradient_check.rs` (reimplemented locally since the
//! crate's internal `tape::helpers::{compute_numerical_gradient,
//! compare_gradients}` are `pub(crate)`-only and not visible from `tests/`).

use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;

/// Step size for central-difference numerical gradients. GroupNorm/
/// InstanceNorm combine several chained divisions/square-roots over a
/// handful of elements per reduction axis, so (mirroring
/// `normalization_gradient_check.rs`'s documented finding for BatchNorm/
/// LayerNorm) an `eps` much smaller than this is dominated by f32 rounding
/// noise rather than signal.
const EPS: f32 = 1e-2;
/// Relative tolerance for analytical vs numerical gradient comparison.
const RTOL: f32 = 1e-2;
/// Absolute tolerance for analytical vs numerical gradient comparison.
///
/// Set to `3e-3` (rather than `normalization_gradient_check.rs`'s `1e-3`)
/// based on a measured worst case: at `EPS=1e-2`, a well-converged but
/// small-magnitude (~0.018) `grad_input` element for InstanceNorm showed a
/// central-difference vs. f64-verified-converged discrepancy of up to
/// ~2.3e-3 purely from f32 rounding (confirmed independently in f64 Python:
/// the analytical value was correct to 6 decimal places; only the f32
/// numerical reference was noisy, and the noise did not shrink as `h`
/// shrank below `1e-3` -- the textbook signature of catastrophic
/// cancellation in a small-magnitude quantity, not a formula bug). `3e-3`
/// keeps a >2.5x safety margin below the smallest real-bug discrepancy this
/// file is designed to catch (a genuine GroupNorm `grad_input` formula bug
/// found and fixed while writing these tests produced a diff of ~0.05 at
/// this same `EPS`, i.e. ~15x this tolerance) -- see
/// `ops::normalization_ops::group_norm_backward`'s doc comment for that
/// fix.
const ATOL: f32 = 3e-3;

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
/// exceeding a small threshold -- i.e. the gradient is not degenerately all
/// zero. A "zeros stub" backward kernel would return `Tensor::zeros(...)`
/// for gamma/beta, which would fail this check even if shapes matched.
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
/// reproducible) sequences. Uses `sin` of a linearly-growing phase (rather
/// than a linearly-growing raw value) so the output stays bounded for large
/// `n`, keeping finite-difference checks numerically well-conditioned.
fn bounded_pseudo_random(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 12.9898 + seed).sin() * 0.9)
        .collect()
}

/// Build a fixed, non-uniform, bounded-magnitude weight tensor with the
/// given shape, used to break the symmetry of a plain `sum()` reduction so
/// tests are sensitive to bugs that only manifest under non-uniform
/// upstream gradients (a uniform `grad_output` of all-ones can accidentally
/// hide bugs that a real training loss would expose).
fn non_uniform_weights(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = bounded_pseudo_random(n, 3.7)
        .into_iter()
        .map(|v| v + 1.2) // keep weights away from 0 to avoid a degenerate near-zero-weight element
        .collect();
    Tensor::from_vec(data, shape).expect("test: weight tensor construction should succeed")
}

// ---------------------------------------------------------------------
// GroupNorm
// ---------------------------------------------------------------------

/// End-to-end (`GradientTape` + `TrackedTensor::group_norm`) finite
/// difference check. Computes `loss = sum(weights * group_norm(input, gamma,
/// beta, num_groups, eps))` and compares the tape's analytical
/// `grad_input`/`grad_gamma`/`grad_beta` against central-difference
/// numerical gradients of the same loss evaluated through the raw
/// `tenflowers_core::ops::group_norm` forward kernel.
#[allow(clippy::too_many_arguments)]
fn run_group_norm_gradient_check(
    label: &str,
    input_shape: &[usize],
    num_groups: usize,
    input_data: Vec<f32>,
    gamma_data: Vec<f32>,
    beta_data: Vec<f32>,
    epsilon: f32,
) {
    let channels = input_shape[1];
    let weights = non_uniform_weights(input_shape);

    // --- Analytical gradients via GradientTape ---
    let tape = GradientTape::new();

    let mut input_tensor = Tensor::from_vec(input_data.clone(), input_shape)
        .expect("test: input tensor construction should succeed");
    let mut gamma_tensor = Tensor::from_vec(gamma_data.clone(), &[channels])
        .expect("test: gamma tensor construction should succeed");
    let mut beta_tensor = Tensor::from_vec(beta_data.clone(), &[channels])
        .expect("test: beta tensor construction should succeed");

    input_tensor.set_requires_grad(true);
    gamma_tensor.set_requires_grad(true);
    beta_tensor.set_requires_grad(true);

    let input_t = tape.watch(input_tensor);
    let gamma_t = tape.watch(gamma_tensor);
    let beta_t = tape.watch(beta_tensor);
    let weights_t = tape.watch(weights.clone());

    let normalized = input_t
        .group_norm(&gamma_t, &beta_t, num_groups, epsilon)
        .expect("test: group_norm forward should succeed");
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

    // --- Numerical gradients via raw group_norm forward kernel ---
    // inputs layout: [input, gamma, beta]
    let raw_inputs = vec![
        Tensor::from_vec(input_data, input_shape)
            .expect("test: raw input tensor construction should succeed"),
        Tensor::from_vec(gamma_data, &[channels])
            .expect("test: raw gamma tensor construction should succeed"),
        Tensor::from_vec(beta_data, &[channels])
            .expect("test: raw beta tensor construction should succeed"),
    ];

    let loss_fn = |t: &[Tensor<f32>]| -> Tensor<f32> {
        let out = tenflowers_core::ops::group_norm(&t[0], &t[1], &t[2], num_groups, epsilon)
            .expect("test: raw group_norm forward should succeed");
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

    assert_gradient_nonzero(&grad_input, &format!("{label}: grad_input"));
    assert_gradient_nonzero(&grad_gamma, &format!("{label}: grad_gamma"));
    assert_gradient_nonzero(&grad_beta, &format!("{label}: grad_beta"));

    println!(
        "[{label}] grad_input[0..4] analytical={:?}",
        &grad_input.as_slice().unwrap_or(&[])[..4.min(grad_input.as_slice().unwrap_or(&[]).len())]
    );
    println!(
        "[{label}] grad_input[0..4] numerical ={:?}",
        &num_grad_input.as_slice().unwrap_or(&[])
            [..4.min(num_grad_input.as_slice().unwrap_or(&[]).len())]
    );
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
fn test_group_norm_rank4_gradients() {
    // input [N=2, C=6, H=4, W=4], num_groups=3 (2 channels per group)
    let shape = [2usize, 6, 4, 4];
    let n: usize = shape.iter().product();
    let input_data = bounded_pseudo_random(n, 0.0);
    let gamma_data = bounded_pseudo_random(6, 1.5)
        .into_iter()
        .map(|v| v + 1.5) // keep gamma away from 0
        .collect();
    let beta_data = bounded_pseudo_random(6, 2.3);

    run_group_norm_gradient_check(
        "group_norm_rank4",
        &shape,
        3,
        input_data,
        gamma_data,
        beta_data,
        1e-5,
    );
}

/// Confirms that channels sharing a group interact through the group's
/// shared mean/variance statistics: perturbing one element of channel 0
/// (group 0, channels [0,1]) changes `grad_input` at an element of channel 1
/// (same group), because both channels feed the same group mean/variance
/// used to normalize both. This is a real, load-bearing property of
/// GroupNorm's backward math (a per-channel-independent implementation --
/// e.g. one that accidentally computed InstanceNorm instead of GroupNorm --
/// would fail this check, since sibling channels in the same group would
/// have zero cross-sensitivity).
#[test]
fn test_group_norm_within_group_channels_interact() {
    let shape = [1usize, 4, 3, 3]; // N=1, C=4, H=3, W=3, num_groups=2 -> 2 channels/group
    let num_groups = 2;
    let n: usize = shape.iter().product();
    let base_input = bounded_pseudo_random(n, 5.0);
    let gamma_data: Vec<f32> = vec![1.3, 0.8, 1.1, 0.9];
    let beta_data: Vec<f32> = vec![0.1, -0.2, 0.05, 0.3];
    let epsilon = 1e-5;
    let channels = shape[1];

    // Loss = sum(group_norm(input, gamma, beta)) evaluated for channel-1's
    // (group 0) gradient, as a function of channel-0's (also group 0) input
    // values only -- i.e. d(sum over channel 1 output)/d(channel 0 input).
    // We compute this cross-derivative numerically by perturbing a channel-0
    // element and re-running the *full* raw forward kernel, then reading off
    // channel 1's contribution to the sum.
    let spatial = shape[2] * shape[3];
    // Flat index for (n=0, c=0, h=0, w=0) in the NCHW layout is simply 0.
    let perturb_idx = 0usize;

    let sum_of_channel = |data: &[f32], target_channel: usize| -> f32 {
        let t = Tensor::from_vec(data.to_vec(), &shape)
            .expect("test: tensor construction should succeed");
        let gamma = Tensor::from_vec(gamma_data.clone(), &[channels])
            .expect("test: gamma construction should succeed");
        let beta = Tensor::from_vec(beta_data.clone(), &[channels])
            .expect("test: beta construction should succeed");
        let out = tenflowers_core::ops::group_norm(&t, &gamma, &beta, num_groups, epsilon)
            .expect("test: raw group_norm forward should succeed");
        let out_data = out.as_slice().expect("test: output should be contiguous");
        let start = target_channel * spatial;
        out_data[start..start + spatial].iter().sum()
    };

    let mut plus = base_input.clone();
    plus[perturb_idx] += EPS;
    let mut minus = base_input.clone();
    minus[perturb_idx] -= EPS;

    // Channel 1 is in the SAME group as channel 0 (group 0 = channels {0,1}).
    let same_group_plus = sum_of_channel(&plus, 1);
    let same_group_minus = sum_of_channel(&minus, 1);
    let cross_deriv_same_group = (same_group_plus - same_group_minus) / (2.0 * EPS);

    // Channel 2 is in a DIFFERENT group (group 1 = channels {2,3}).
    let diff_group_plus = sum_of_channel(&plus, 2);
    let diff_group_minus = sum_of_channel(&minus, 2);
    let cross_deriv_diff_group = (diff_group_plus - diff_group_minus) / (2.0 * EPS);

    assert!(
        cross_deriv_same_group.abs() > 1e-3,
        "expected channel 0 to influence channel 1's output (same group), got cross-derivative {cross_deriv_same_group:?}"
    );
    assert!(
        cross_deriv_diff_group.abs() < 1e-3,
        "expected channel 0 to NOT influence channel 2's output (different group), got cross-derivative {cross_deriv_diff_group:?}"
    );

    println!(
        "[group_norm_within_group_interaction] same-group cross-deriv={cross_deriv_same_group:?}, different-group cross-deriv={cross_deriv_diff_group:?}"
    );
}

// ---------------------------------------------------------------------
// InstanceNorm
// ---------------------------------------------------------------------

/// End-to-end (`GradientTape` + `TrackedTensor::instance_norm`) finite
/// difference check. `TrackedTensor::instance_norm` computes its forward
/// pass via `tenflowers_core::ops::group_norm` with `num_groups = channels`
/// (InstanceNorm is GroupNorm with one group per channel), so the numerical
/// reference uses the same `group_norm` kernel with `num_groups = channels`
/// for an apples-to-apples comparison against the tape's analytical
/// gradient (which is produced by `instance_norm_backward`, a distinct but
/// mathematically equivalent kernel).
fn run_instance_norm_gradient_check(
    label: &str,
    input_shape: &[usize],
    input_data: Vec<f32>,
    gamma_data: Vec<f32>,
    beta_data: Vec<f32>,
    epsilon: f32,
) {
    let channels = input_shape[1];
    let weights = non_uniform_weights(input_shape);

    // --- Analytical gradients via GradientTape ---
    let tape = GradientTape::new();

    let mut input_tensor = Tensor::from_vec(input_data.clone(), input_shape)
        .expect("test: input tensor construction should succeed");
    let mut gamma_tensor = Tensor::from_vec(gamma_data.clone(), &[channels])
        .expect("test: gamma tensor construction should succeed");
    let mut beta_tensor = Tensor::from_vec(beta_data.clone(), &[channels])
        .expect("test: beta tensor construction should succeed");

    input_tensor.set_requires_grad(true);
    gamma_tensor.set_requires_grad(true);
    beta_tensor.set_requires_grad(true);

    let input_t = tape.watch(input_tensor);
    let gamma_t = tape.watch(gamma_tensor);
    let beta_t = tape.watch(beta_tensor);
    let weights_t = tape.watch(weights.clone());

    let normalized = input_t
        .instance_norm(&gamma_t, &beta_t, epsilon)
        .expect("test: instance_norm forward should succeed");
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

    // --- Numerical gradients via raw group_norm(num_groups=channels) forward kernel ---
    // inputs layout: [input, gamma, beta]
    let raw_inputs = vec![
        Tensor::from_vec(input_data, input_shape)
            .expect("test: raw input tensor construction should succeed"),
        Tensor::from_vec(gamma_data, &[channels])
            .expect("test: raw gamma tensor construction should succeed"),
        Tensor::from_vec(beta_data, &[channels])
            .expect("test: raw beta tensor construction should succeed"),
    ];

    let loss_fn = |t: &[Tensor<f32>]| -> Tensor<f32> {
        let out = tenflowers_core::ops::group_norm(&t[0], &t[1], &t[2], channels, epsilon)
            .expect("test: raw group_norm(num_groups=channels) forward should succeed");
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

    assert_gradient_nonzero(&grad_input, &format!("{label}: grad_input"));
    assert_gradient_nonzero(&grad_gamma, &format!("{label}: grad_gamma"));
    assert_gradient_nonzero(&grad_beta, &format!("{label}: grad_beta"));

    println!(
        "[{label}] grad_input[0..4] analytical={:?}",
        &grad_input.as_slice().unwrap_or(&[])[..4.min(grad_input.as_slice().unwrap_or(&[]).len())]
    );
    println!(
        "[{label}] grad_input[0..4] numerical ={:?}",
        &num_grad_input.as_slice().unwrap_or(&[])
            [..4.min(num_grad_input.as_slice().unwrap_or(&[]).len())]
    );
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
fn test_instance_norm_rank4_gradients() {
    // input [N=2, C=4, H=5, W=5]
    let shape = [2usize, 4, 5, 5];
    let n: usize = shape.iter().product();
    let input_data = bounded_pseudo_random(n, 0.7);
    let gamma_data = bounded_pseudo_random(4, 1.9)
        .into_iter()
        .map(|v| v + 1.5) // keep gamma away from 0
        .collect();
    let beta_data = bounded_pseudo_random(4, 4.1);

    run_instance_norm_gradient_check(
        "instance_norm_rank4",
        &shape,
        input_data,
        gamma_data,
        beta_data,
        1e-5,
    );
}

/// Confirms InstanceNorm normalizes each (batch, channel) pair
/// independently: perturbing one element of channel 0 must NOT influence
/// channel 1's output at all (unlike GroupNorm's within-group case above),
/// since InstanceNorm has exactly one channel per "group".
#[test]
fn test_instance_norm_channels_are_independent() {
    let shape = [1usize, 3, 4, 4]; // N=1, C=3, H=4, W=4
    let n: usize = shape.iter().product();
    let base_input = bounded_pseudo_random(n, 6.6);
    let gamma_data: Vec<f32> = vec![1.2, 0.7, 1.05];
    let beta_data: Vec<f32> = vec![0.2, -0.1, 0.0];
    let epsilon = 1e-5;
    let channels = shape[1];
    let spatial = shape[2] * shape[3];

    let sum_of_channel = |data: &[f32], target_channel: usize| -> f32 {
        let t = Tensor::from_vec(data.to_vec(), &shape)
            .expect("test: tensor construction should succeed");
        let gamma = Tensor::from_vec(gamma_data.clone(), &[channels])
            .expect("test: gamma construction should succeed");
        let beta = Tensor::from_vec(beta_data.clone(), &[channels])
            .expect("test: beta construction should succeed");
        // InstanceNorm == GroupNorm with num_groups = channels.
        let out = tenflowers_core::ops::group_norm(&t, &gamma, &beta, channels, epsilon)
            .expect("test: raw group_norm forward should succeed");
        let out_data = out.as_slice().expect("test: output should be contiguous");
        let start = target_channel * spatial;
        out_data[start..start + spatial].iter().sum()
    };

    let perturb_idx = 0usize; // (n=0, c=0, h=0, w=0)
    let mut plus = base_input.clone();
    plus[perturb_idx] += EPS;
    let mut minus = base_input.clone();
    minus[perturb_idx] -= EPS;

    let other_channel_plus = sum_of_channel(&plus, 1);
    let other_channel_minus = sum_of_channel(&minus, 1);
    let cross_deriv = (other_channel_plus - other_channel_minus) / (2.0 * EPS);

    assert!(
        cross_deriv.abs() < 1e-3,
        "expected channel 0 to NOT influence channel 1's output under InstanceNorm, got cross-derivative {cross_deriv:?}"
    );

    println!("[instance_norm_channel_independence] cross-deriv={cross_deriv:?} (expected ~0)");
}

// ---------------------------------------------------------------------
// Rejects invalid input rank
// ---------------------------------------------------------------------

/// `TrackedTensor::instance_norm` must reject rank < 2 input with a clear
/// error rather than panicking on an out-of-bounds `dims()[1]` index.
#[test]
fn test_instance_norm_rejects_rank1_input() {
    let tape = GradientTape::new();
    let input = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], &[3])
        .expect("test: input tensor construction should succeed");
    let gamma = Tensor::<f32>::from_vec(vec![1.0, 1.0, 1.0], &[3])
        .expect("test: gamma tensor construction should succeed");
    let beta = Tensor::<f32>::from_vec(vec![0.0, 0.0, 0.0], &[3])
        .expect("test: beta tensor construction should succeed");

    let input_t = tape.watch(input);
    let gamma_t = tape.watch(gamma);
    let beta_t = tape.watch(beta);

    let result = input_t.instance_norm(&gamma_t, &beta_t, 1e-5);
    assert!(
        result.is_err(),
        "expected instance_norm on rank-1 input to return an Err, got Ok"
    );
}
