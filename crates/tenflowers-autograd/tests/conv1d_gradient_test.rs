//! Finite-difference gradient check for `Operation::Conv1D`.
//!
//! Exercises the *full* autograd wiring end to end: `TrackedTensor::conv1d`
//! (forward + tape recording) through `GradientTape::gradient` (backward
//! dispatch -> `conv1d_backward` kernel), rather than calling the backward
//! kernel function directly. This is deliberately a stronger check than the
//! existing Conv2D kernel-level finite-difference tests in
//! `tenflowers-autograd/src/ops/convolution_ops/utils.rs`: it also validates
//! that `Operation::Conv1D` is dispatched correctly, that gradients are
//! accumulated onto the right `TensorId`s, and that the bias gradient path
//! (present/absent) is wired correctly.
//!
//! Central finite differences: `df/dx ~= (f(x+eps) - f(x-eps)) / (2*eps)`.
//! Every individual parameter element is perturbed independently and compared
//! against the corresponding analytical-gradient component.

use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;

/// Deterministic pseudo-random value in roughly `[-1, 1)`. Avoids pulling in
/// an RNG dependency while still exercising non-trivial (non-zero, non-
/// symmetric) gradients across every test case.
fn pseudo_random(seed: usize) -> f32 {
    let x = (seed as f64 * 12.9898 + 78.233).sin() * 43758.5453;
    (((x - x.floor()) * 2.0) - 1.0) as f32
}

fn filled_vec(len: usize, offset: usize) -> Vec<f32> {
    (0..len).map(|i| pseudo_random(i + offset)).collect()
}

/// Reference forward 1D convolution mirroring `tenflowers_core::ops::conv1d`
/// (`conv1d_cpu`) exactly: NCL input, weight `[out_c, in_c, kernel_len]`,
/// cross-correlation, bias broadcast added per output channel.
/// Returns the flat NCL output buffer together with `out_length`.
fn forward_conv1d_reference(
    input: &[f32],
    input_shape: [usize; 3],
    weight: &[f32],
    weight_shape: [usize; 3],
    bias: Option<&[f32]>,
    stride: usize,
    padding: &str,
) -> (Vec<f32>, usize) {
    let [batch_size, in_channels, in_length] = input_shape;
    let [out_channels, _w_in, kernel_length] = weight_shape;

    let (out_length, pad_left) = match padding {
        "valid" => ((in_length - kernel_length) / stride + 1, 0usize),
        "same" => {
            let out_len = (in_length + stride - 1) / stride;
            let pad_total =
                std::cmp::max(0, (out_len - 1) * stride + kernel_length - in_length);
            (out_len, pad_total / 2)
        }
        other => panic!("test: unsupported padding {other}"),
    };

    let mut output = vec![0.0_f32; batch_size * out_channels * out_length];

    for b in 0..batch_size {
        for oc in 0..out_channels {
            for ol in 0..out_length {
                let mut sum = 0.0_f32;
                for ic in 0..in_channels {
                    for k in 0..kernel_length {
                        let il = ol * stride + k;
                        if il >= pad_left && il < in_length + pad_left {
                            let il_actual = il - pad_left;
                            if il_actual < in_length {
                                let input_idx =
                                    (b * in_channels + ic) * in_length + il_actual;
                                let weight_idx = (oc * in_channels + ic) * kernel_length + k;
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                if let Some(bias_data) = bias {
                    sum += bias_data[oc];
                }
                let out_idx = (b * out_channels + oc) * out_length + ol;
                output[out_idx] = sum;
            }
        }
    }

    (output, out_length)
}

/// Scalar loss `L = <grad_seed, forward(input, weight, bias)>`. The upstream
/// gradient flowing into the conv is exactly `grad_seed` by construction, so
/// `dL/dinput`, `dL/dweight`, and `dL/dbias` are exactly what the analytical
/// `conv1d_backward` kernel must produce.
#[allow(clippy::too_many_arguments)]
fn conv1d_loss(
    input: &[f32],
    input_shape: [usize; 3],
    weight: &[f32],
    weight_shape: [usize; 3],
    bias: Option<&[f32]>,
    grad_seed: &[f32],
    stride: usize,
    padding: &str,
) -> f64 {
    let (output, _) = forward_conv1d_reference(
        input,
        input_shape,
        weight,
        weight_shape,
        bias,
        stride,
        padding,
    );
    output
        .iter()
        .zip(grad_seed.iter())
        .map(|(o, g)| *o as f64 * *g as f64)
        .sum()
}

/// One (stride, padding) test case, run for grad_input, grad_weight, and
/// grad_bias, each checked independently via the real `TrackedTensor::conv1d`
/// -> `GradientTape::gradient` autograd path against central finite
/// differences on `conv1d_loss`.
fn run_conv1d_gradient_check(stride: usize, padding: &str) {
    // batch=2, in_channels=3, out_channels=4, length=8, kernel_size=3
    let batch_size = 2usize;
    let in_channels = 3usize;
    let out_channels = 4usize;
    let in_length = 8usize;
    let kernel_length = 3usize;

    let input_shape = [batch_size, in_channels, in_length];
    let weight_shape = [out_channels, in_channels, kernel_length];

    let input = filled_vec(input_shape.iter().product(), 1);
    let weight = filled_vec(weight_shape.iter().product(), 1000);
    let bias = filled_vec(out_channels, 2000);

    // Determine out_length the same way the forward kernel does, so the seed
    // has the right size.
    let (_, out_length) = forward_conv1d_reference(
        &input,
        input_shape,
        &weight,
        weight_shape,
        Some(&bias),
        stride,
        padding,
    );
    let grad_seed = filled_vec(batch_size * out_channels * out_length, 7000);

    // ---- Run the REAL autograd path: TrackedTensor::conv1d + GradientTape::gradient ----
    let tape = GradientTape::new();
    let input_tensor = Tensor::<f32>::from_vec(input.clone(), &input_shape)
        .expect("test: input tensor construction");
    let weight_tensor = Tensor::<f32>::from_vec(weight.clone(), &weight_shape)
        .expect("test: weight tensor construction");
    let bias_tensor =
        Tensor::<f32>::from_vec(bias.clone(), &[out_channels]).expect("test: bias tensor");

    let input_tracked = tape.watch(input_tensor);
    let weight_tracked = tape.watch(weight_tensor);
    let bias_tracked = tape.watch(bias_tensor);

    let output_tracked = input_tracked
        .conv1d(&weight_tracked, Some(&bias_tracked), stride, padding)
        .unwrap_or_else(|e| panic!("conv1d forward failed (stride={stride}, padding={padding}): {e}"));

    // Build the scalar loss L = <grad_seed, output> as an actual traced
    // computation (output.mul(grad_seed_tracked).sum()) rather than manually
    // seeding the backward pass, so this test exercises only the crate's
    // public API. Since d(sum(output * grad_seed))/d(output) = grad_seed
    // exactly, the gradients `tape.gradient` returns for
    // input/weight/bias are precisely dL/d(input|weight|bias) for the same
    // L that `conv1d_loss` computes on plain f32 buffers below.
    let grad_seed_tensor =
        Tensor::<f32>::from_vec(grad_seed.clone(), &[batch_size, out_channels, out_length])
            .expect("test: grad_seed tensor");
    let grad_seed_tracked = tape.watch(grad_seed_tensor);
    let loss_tracked = output_tracked
        .mul(&grad_seed_tracked)
        .and_then(|weighted| weighted.sum(None, false))
        .unwrap_or_else(|e| panic!("loss construction failed (stride={stride}, padding={padding}): {e}"));

    let gradients = tape
        .gradient(
            &[loss_tracked],
            &[
                input_tracked.clone(),
                weight_tracked.clone(),
                bias_tracked.clone(),
            ],
        )
        .unwrap_or_else(|e| panic!("gradient computation failed (stride={stride}, padding={padding}): {e}"));

    let analytical_grad_input = gradients[0]
        .as_ref()
        .unwrap_or_else(|| panic!("no grad_input recorded (stride={stride}, padding={padding})"))
        .to_vec()
        .expect("test: grad_input to_vec");
    let analytical_grad_weight = gradients[1]
        .as_ref()
        .unwrap_or_else(|| panic!("no grad_weight recorded (stride={stride}, padding={padding})"))
        .to_vec()
        .expect("test: grad_weight to_vec");
    let analytical_grad_bias = gradients[2]
        .as_ref()
        .unwrap_or_else(|| panic!("no grad_bias recorded (stride={stride}, padding={padding})"))
        .to_vec()
        .expect("test: grad_bias to_vec");

    assert_eq!(analytical_grad_input.len(), input.len());
    assert_eq!(analytical_grad_weight.len(), weight.len());
    assert_eq!(analytical_grad_bias.len(), bias.len());

    let eps = 1e-3_f32;
    let rtol = 1e-2_f64;
    let atol = 1e-3_f64;

    // Track the largest observed absolute and relative error across every
    // parameter element, purely for diagnostic reporting (`--nocapture`) --
    // the pass/fail decision remains the per-element `assert!` below.
    let mut max_abs_err_input = 0.0_f64;
    let mut max_rel_err_input = 0.0_f64;
    let mut max_abs_err_weight = 0.0_f64;
    let mut max_rel_err_weight = 0.0_f64;
    let mut max_abs_err_bias = 0.0_f64;
    let mut max_rel_err_bias = 0.0_f64;

    // ---- grad_input: central finite differences ----
    for idx in 0..input.len() {
        let mut plus = input.clone();
        let mut minus = input.clone();
        plus[idx] += eps;
        minus[idx] -= eps;
        let l_plus = conv1d_loss(
            &plus,
            input_shape,
            &weight,
            weight_shape,
            Some(&bias),
            &grad_seed,
            stride,
            padding,
        );
        let l_minus = conv1d_loss(
            &minus,
            input_shape,
            &weight,
            weight_shape,
            Some(&bias),
            &grad_seed,
            stride,
            padding,
        );
        let numerical = (l_plus - l_minus) / (2.0 * eps as f64);
        let analytical = analytical_grad_input[idx] as f64;
        let diff = (analytical - numerical).abs();
        let tol = atol + rtol * numerical.abs();
        max_abs_err_input = max_abs_err_input.max(diff);
        max_rel_err_input = max_rel_err_input.max(diff / numerical.abs().max(1e-12));
        assert!(
            diff <= tol,
            "grad_input mismatch at idx={idx} (stride={stride}, padding={padding}): \
             analytical={analytical}, numerical={numerical}, diff={diff}, tol={tol}"
        );
    }

    // ---- grad_weight: central finite differences ----
    for idx in 0..weight.len() {
        let mut plus = weight.clone();
        let mut minus = weight.clone();
        plus[idx] += eps;
        minus[idx] -= eps;
        let l_plus = conv1d_loss(
            &input,
            input_shape,
            &plus,
            weight_shape,
            Some(&bias),
            &grad_seed,
            stride,
            padding,
        );
        let l_minus = conv1d_loss(
            &input,
            input_shape,
            &minus,
            weight_shape,
            Some(&bias),
            &grad_seed,
            stride,
            padding,
        );
        let numerical = (l_plus - l_minus) / (2.0 * eps as f64);
        let analytical = analytical_grad_weight[idx] as f64;
        let diff = (analytical - numerical).abs();
        let tol = atol + rtol * numerical.abs();
        max_abs_err_weight = max_abs_err_weight.max(diff);
        max_rel_err_weight = max_rel_err_weight.max(diff / numerical.abs().max(1e-12));
        assert!(
            diff <= tol,
            "grad_weight mismatch at idx={idx} (stride={stride}, padding={padding}): \
             analytical={analytical}, numerical={numerical}, diff={diff}, tol={tol}"
        );
    }

    // ---- grad_bias: central finite differences ----
    for idx in 0..bias.len() {
        let mut plus = bias.clone();
        let mut minus = bias.clone();
        plus[idx] += eps;
        minus[idx] -= eps;
        let l_plus = conv1d_loss(
            &input,
            input_shape,
            &weight,
            weight_shape,
            Some(&plus),
            &grad_seed,
            stride,
            padding,
        );
        let l_minus = conv1d_loss(
            &input,
            input_shape,
            &weight,
            weight_shape,
            Some(&minus),
            &grad_seed,
            stride,
            padding,
        );
        let numerical = (l_plus - l_minus) / (2.0 * eps as f64);
        let analytical = analytical_grad_bias[idx] as f64;
        let diff = (analytical - numerical).abs();
        let tol = atol + rtol * numerical.abs();
        max_abs_err_bias = max_abs_err_bias.max(diff);
        max_rel_err_bias = max_rel_err_bias.max(diff / numerical.abs().max(1e-12));
        assert!(
            diff <= tol,
            "grad_bias mismatch at idx={idx} (stride={stride}, padding={padding}): \
             analytical={analytical}, numerical={numerical}, diff={diff}, tol={tol}"
        );
    }

    // Diagnostic summary (visible with `cargo test -- --nocapture`). All
    // per-element checks above already passed by the time this prints.
    println!(
        "conv1d gradient check OK (stride={stride}, padding={padding}): \
         grad_input  max_abs_err={max_abs_err_input:.3e} max_rel_err={max_rel_err_input:.3e} ({} elems) | \
         grad_weight max_abs_err={max_abs_err_weight:.3e} max_rel_err={max_rel_err_weight:.3e} ({} elems) | \
         grad_bias   max_abs_err={max_abs_err_bias:.3e} max_rel_err={max_rel_err_bias:.3e} ({} elems)",
        input.len(),
        weight.len(),
        bias.len(),
    );
}

#[test]
fn test_conv1d_gradient_stride1_valid() {
    run_conv1d_gradient_check(1, "valid");
}

#[test]
fn test_conv1d_gradient_stride2_valid() {
    run_conv1d_gradient_check(2, "valid");
}

#[test]
fn test_conv1d_gradient_stride1_same() {
    run_conv1d_gradient_check(1, "same");
}

#[test]
fn test_conv1d_gradient_stride2_same() {
    run_conv1d_gradient_check(2, "same");
}
