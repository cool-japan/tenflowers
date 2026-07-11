//! Finite-difference gradient check for `Operation::Conv3D`.
//!
//! Exercises the *full* autograd wiring end to end: `TrackedTensor::conv3d`
//! (forward + tape recording) through `GradientTape::gradient` (backward
//! dispatch -> `conv3d_backward` kernel), rather than calling the backward
//! kernel function directly. This validates that `Operation::Conv3D` is
//! dispatched correctly in `process_operation_backward`, that gradients are
//! accumulated onto the right `TensorId`s, and that the bias gradient path
//! is wired correctly -- mirroring the existing Conv1D end-to-end check in
//! `conv1d_gradient_test.rs`.
//!
//! Central finite differences: `df/dx ~= (f(x+eps) - f(x-eps)) / (2*eps)`.
//! Every individual parameter element is perturbed independently and compared
//! against the corresponding analytical-gradient component, for both
//! `input`, `weight`, and `bias` -- not just index 0, so a transposition or
//! padding-offset bug that only manifests away from the origin would be
//! caught.

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

/// Reference forward 3D convolution mirroring `tenflowers_core::ops::conv3d`
/// (`conv3d_cpu`) exactly: NCDHW input, weight
/// `[out_c, in_c, kernel_d, kernel_h, kernel_w]`, cross-correlation, bias
/// broadcast added per output channel. Returns the flat NCDHW output buffer
/// together with `(out_depth, out_height, out_width)`.
#[allow(clippy::too_many_arguments)]
fn forward_conv3d_reference(
    input: &[f32],
    input_shape: [usize; 5],
    weight: &[f32],
    weight_shape: [usize; 5],
    bias: Option<&[f32]>,
    stride: (usize, usize, usize),
    padding: &str,
) -> (Vec<f32>, (usize, usize, usize)) {
    let [batch_size, in_channels, in_depth, in_height, in_width] = input_shape;
    let [out_channels, _w_in, kernel_depth, kernel_height, kernel_width] = weight_shape;

    let (out_depth, out_height, out_width, pad_front, pad_top, pad_left) = match padding {
        "valid" => {
            let out_d = (in_depth - kernel_depth) / stride.0 + 1;
            let out_h = (in_height - kernel_height) / stride.1 + 1;
            let out_w = (in_width - kernel_width) / stride.2 + 1;
            (out_d, out_h, out_w, 0usize, 0usize, 0usize)
        }
        "same" => {
            let out_d = (in_depth + stride.0 - 1) / stride.0;
            let out_h = (in_height + stride.1 - 1) / stride.1;
            let out_w = (in_width + stride.2 - 1) / stride.2;
            let pad_d = ((out_d - 1) * stride.0 + kernel_depth).saturating_sub(in_depth);
            let pad_h = ((out_h - 1) * stride.1 + kernel_height).saturating_sub(in_height);
            let pad_w = ((out_w - 1) * stride.2 + kernel_width).saturating_sub(in_width);
            (out_d, out_h, out_w, pad_d / 2, pad_h / 2, pad_w / 2)
        }
        other => panic!("test: unsupported padding {other}"),
    };

    let mut output = vec![0.0_f32; batch_size * out_channels * out_depth * out_height * out_width];

    for b in 0..batch_size {
        for oc in 0..out_channels {
            for od in 0..out_depth {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = 0.0_f32;
                        for ic in 0..in_channels {
                            for kd in 0..kernel_depth {
                                let id = od * stride.0 + kd;
                                if id < pad_front || id >= in_depth + pad_front {
                                    continue;
                                }
                                let id_actual = id - pad_front;
                                for kh in 0..kernel_height {
                                    let ih = oh * stride.1 + kh;
                                    if ih < pad_top || ih >= in_height + pad_top {
                                        continue;
                                    }
                                    let ih_actual = ih - pad_top;
                                    for kw in 0..kernel_width {
                                        let iw = ow * stride.2 + kw;
                                        if iw < pad_left || iw >= in_width + pad_left {
                                            continue;
                                        }
                                        let iw_actual = iw - pad_left;

                                        let input_idx = (((b * in_channels + ic) * in_depth
                                            + id_actual)
                                            * in_height
                                            + ih_actual)
                                            * in_width
                                            + iw_actual;
                                        let weight_idx = ((((oc * in_channels + ic)
                                            * kernel_depth
                                            + kd)
                                            * kernel_height
                                            + kh)
                                            * kernel_width)
                                            + kw;
                                        sum += input[input_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }
                        if let Some(bias_data) = bias {
                            sum += bias_data[oc];
                        }
                        let out_idx = (((b * out_channels + oc) * out_depth + od) * out_height
                            + oh)
                            * out_width
                            + ow;
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }

    (output, (out_depth, out_height, out_width))
}

/// Scalar loss `L = <grad_seed, forward(input, weight, bias)>`. The upstream
/// gradient flowing into the conv is exactly `grad_seed` by construction, so
/// `dL/dinput`, `dL/dweight`, and `dL/dbias` are exactly what the analytical
/// `conv3d_backward` kernel must produce.
#[allow(clippy::too_many_arguments)]
fn conv3d_loss(
    input: &[f32],
    input_shape: [usize; 5],
    weight: &[f32],
    weight_shape: [usize; 5],
    bias: Option<&[f32]>,
    grad_seed: &[f32],
    stride: (usize, usize, usize),
    padding: &str,
) -> f64 {
    let (output, _) = forward_conv3d_reference(
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
/// grad_bias, each checked independently via the real `TrackedTensor::conv3d`
/// -> `GradientTape::gradient` autograd path against central finite
/// differences on `conv3d_loss`. Also asserts basic gradient/parameter shape
/// agreement.
fn run_conv3d_gradient_check(stride: (usize, usize, usize), padding: &str) {
    // batch=1, in_channels=2, out_channels=2, depth=height=width=3, kernel=2x2x2
    let batch_size = 1usize;
    let in_channels = 2usize;
    let out_channels = 2usize;
    let spatial = 3usize;
    let kernel = 2usize;

    let input_shape = [batch_size, in_channels, spatial, spatial, spatial];
    let weight_shape = [out_channels, in_channels, kernel, kernel, kernel];

    let input = filled_vec(input_shape.iter().product(), 1);
    let weight = filled_vec(weight_shape.iter().product(), 1000);
    let bias = filled_vec(out_channels, 2000);

    // Determine output spatial dims the same way the forward kernel does, so
    // the seed has the right size.
    let (_, (out_depth, out_height, out_width)) = forward_conv3d_reference(
        &input,
        input_shape,
        &weight,
        weight_shape,
        Some(&bias),
        stride,
        padding,
    );
    let grad_seed = filled_vec(
        batch_size * out_channels * out_depth * out_height * out_width,
        7000,
    );

    // ---- Run the REAL autograd path: TrackedTensor::conv3d + GradientTape::gradient ----
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
        .conv3d(&weight_tracked, Some(&bias_tracked), stride, padding)
        .unwrap_or_else(|e| {
            panic!("conv3d forward failed (stride={stride:?}, padding={padding}): {e}")
        });

    // Shape sanity: forward output must match the reference forward's
    // computed spatial dims.
    assert_eq!(
        output_tracked.tensor.shape().dims(),
        &[batch_size, out_channels, out_depth, out_height, out_width],
        "conv3d forward output shape mismatch (stride={stride:?}, padding={padding})"
    );

    // Build the scalar loss L = <grad_seed, output> as an actual traced
    // computation (output.mul(grad_seed_tracked).sum()) rather than manually
    // seeding the backward pass, so this test exercises only the crate's
    // public API. Since d(sum(output * grad_seed))/d(output) = grad_seed
    // exactly, the gradients `tape.gradient` returns for
    // input/weight/bias are precisely dL/d(input|weight|bias) for the same
    // L that `conv3d_loss` computes on plain f32 buffers below.
    let grad_seed_tensor = Tensor::<f32>::from_vec(
        grad_seed.clone(),
        &[batch_size, out_channels, out_depth, out_height, out_width],
    )
    .expect("test: grad_seed tensor");
    let grad_seed_tracked = tape.watch(grad_seed_tensor);
    let loss_tracked = output_tracked
        .mul(&grad_seed_tracked)
        .and_then(|weighted| weighted.sum(None, false))
        .unwrap_or_else(|e| {
            panic!("loss construction failed (stride={stride:?}, padding={padding}): {e}")
        });

    let gradients = tape
        .gradient(
            &[loss_tracked],
            &[
                input_tracked.clone(),
                weight_tracked.clone(),
                bias_tracked.clone(),
            ],
        )
        .unwrap_or_else(|e| {
            panic!("gradient computation failed (stride={stride:?}, padding={padding}): {e}")
        });

    let analytical_grad_input = gradients[0]
        .as_ref()
        .unwrap_or_else(|| panic!("no grad_input recorded (stride={stride:?}, padding={padding})"))
        .to_vec()
        .expect("test: grad_input to_vec");
    let analytical_grad_weight = gradients[1]
        .as_ref()
        .unwrap_or_else(|| {
            panic!("no grad_weight recorded (stride={stride:?}, padding={padding})")
        })
        .to_vec()
        .expect("test: grad_weight to_vec");
    let analytical_grad_bias = gradients[2]
        .as_ref()
        .unwrap_or_else(|| panic!("no grad_bias recorded (stride={stride:?}, padding={padding})"))
        .to_vec()
        .expect("test: grad_bias to_vec");

    // ---- Shape sanity checks ----
    assert_eq!(
        gradients[0].as_ref().expect("grad_input present").shape().dims(),
        &input_shape,
        "grad_input shape must match input shape (stride={stride:?}, padding={padding})"
    );
    assert_eq!(
        gradients[1]
            .as_ref()
            .expect("grad_weight present")
            .shape()
            .dims(),
        &weight_shape,
        "grad_weight shape must match weight shape (stride={stride:?}, padding={padding})"
    );
    assert_eq!(
        gradients[2].as_ref().expect("grad_bias present").shape().dims(),
        &[out_channels],
        "grad_bias shape must match bias shape (stride={stride:?}, padding={padding})"
    );
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

    // Record a few representative (non-index-0) positions for the report,
    // in addition to checking *every* element below (this input/weight are
    // small enough -- 54 and 32 elements respectively -- that exhaustive
    // per-element finite differencing is still cheap).
    let mut reported_input_positions: Vec<(usize, f64, f64)> = Vec::new();
    let mut reported_weight_positions: Vec<(usize, f64, f64)> = Vec::new();
    let mut reported_bias_positions: Vec<(usize, f64, f64)> = Vec::new();

    // ---- grad_input: central finite differences ----
    for idx in 0..input.len() {
        let mut plus = input.clone();
        let mut minus = input.clone();
        plus[idx] += eps;
        minus[idx] -= eps;
        let l_plus = conv3d_loss(
            &plus,
            input_shape,
            &weight,
            weight_shape,
            Some(&bias),
            &grad_seed,
            stride,
            padding,
        );
        let l_minus = conv3d_loss(
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
        if idx == 0 || idx == input.len() / 2 || idx == input.len() - 1 {
            reported_input_positions.push((idx, analytical, numerical));
        }
        assert!(
            diff <= tol,
            "grad_input mismatch at idx={idx} (stride={stride:?}, padding={padding}): \
             analytical={analytical}, numerical={numerical}, diff={diff}, tol={tol}"
        );
    }

    // ---- grad_weight: central finite differences ----
    for idx in 0..weight.len() {
        let mut plus = weight.clone();
        let mut minus = weight.clone();
        plus[idx] += eps;
        minus[idx] -= eps;
        let l_plus = conv3d_loss(
            &input,
            input_shape,
            &plus,
            weight_shape,
            Some(&bias),
            &grad_seed,
            stride,
            padding,
        );
        let l_minus = conv3d_loss(
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
        if idx == 0 || idx == weight.len() / 2 || idx == weight.len() - 1 {
            reported_weight_positions.push((idx, analytical, numerical));
        }
        assert!(
            diff <= tol,
            "grad_weight mismatch at idx={idx} (stride={stride:?}, padding={padding}): \
             analytical={analytical}, numerical={numerical}, diff={diff}, tol={tol}"
        );
    }

    // ---- grad_bias: central finite differences ----
    for idx in 0..bias.len() {
        let mut plus = bias.clone();
        let mut minus = bias.clone();
        plus[idx] += eps;
        minus[idx] -= eps;
        let l_plus = conv3d_loss(
            &input,
            input_shape,
            &weight,
            weight_shape,
            Some(&plus),
            &grad_seed,
            stride,
            padding,
        );
        let l_minus = conv3d_loss(
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
        reported_bias_positions.push((idx, analytical, numerical));
        assert!(
            diff <= tol,
            "grad_bias mismatch at idx={idx} (stride={stride:?}, padding={padding}): \
             analytical={analytical}, numerical={numerical}, diff={diff}, tol={tol}"
        );
    }

    // Diagnostic summary (visible with `cargo test -- --nocapture`). All
    // per-element checks above already passed by the time this prints.
    println!(
        "conv3d gradient check OK (stride={stride:?}, padding={padding}): \
         grad_input  max_abs_err={max_abs_err_input:.3e} max_rel_err={max_rel_err_input:.3e} ({} elems) positions={reported_input_positions:?} | \
         grad_weight max_abs_err={max_abs_err_weight:.3e} max_rel_err={max_rel_err_weight:.3e} ({} elems) positions={reported_weight_positions:?} | \
         grad_bias   max_abs_err={max_abs_err_bias:.3e} max_rel_err={max_rel_err_bias:.3e} ({} elems) positions={reported_bias_positions:?}",
        input.len(),
        weight.len(),
        bias.len(),
    );
}

#[test]
fn test_conv3d_gradient_stride1_valid() {
    run_conv3d_gradient_check((1, 1, 1), "valid");
}

#[test]
fn test_conv3d_gradient_stride2_valid() {
    run_conv3d_gradient_check((2, 2, 2), "valid");
}

#[test]
fn test_conv3d_gradient_stride1_same() {
    run_conv3d_gradient_check((1, 1, 1), "same");
}

#[test]
fn test_conv3d_gradient_mixed_stride_valid() {
    run_conv3d_gradient_check((1, 2, 1), "valid");
}
