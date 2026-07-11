//! Gradient tests for the 11 tape-aware activation/math operations that were
//! added to `TrackedTensor` without dedicated gradient regression coverage:
//!
//! Part 1: `gelu`, `swish`, `mish`, `leaky_relu`, `elu`.
//! Part 2: `log`, `abs`, `clamp`, `relu6`, `hard_swish`, `log_softmax`.
//!
//! Structural template follows
//! `simple_gradient_test.rs::test_softmax_gradient_rank2_is_axis_aware_not_flattened`:
//! `GradientTape::new()` -> `tape.watch(x)` -> apply the op -> multiply by a
//! non-uniform, watched-but-undifferentiated upstream weight tensor `w` ->
//! `.sum(None, false)` to get a scalar loss -> `tape.gradient(&[loss],
//! &[x_tracked])` -> compare the returned analytical gradient against a
//! central-difference finite-difference (or, for special-case boundary
//! points, hand-derived analytical) expected value.
//!
//! For central differences: each element `x_i` is perturbed by `+eps`/`-eps`
//! independently (holding every other element fixed), the SAME scalar loss
//! `sum(w * op(x))` is recomputed fresh both times using plain `Tensor` math
//! (no tape), and `(loss(x_i+eps) - loss(x_i-eps)) / (2*eps)` is taken as the
//! numerical gradient for that element.

use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;

/// Central-difference numerical gradient of `loss(x) = sum(w * op(x))` with
/// respect to every element of `x`, holding `w` fixed. `op` and the loss are
/// recomputed from scratch (plain `Tensor` math, no tape) for each
/// perturbation, matching exactly what the tape-based analytical path
/// computes so the two are a fair apples-to-apples comparison.
fn numerical_gradient<F>(x: &Tensor<f32>, w: &Tensor<f32>, eps: f32, op: F) -> Vec<f32>
where
    F: Fn(&Tensor<f32>) -> Tensor<f32>,
{
    let x_data = x.as_slice().expect("x should be contiguous");
    let dims = x.shape().dims().to_vec();
    let n = x_data.len();
    let mut grad = vec![0.0_f32; n];

    let loss_of = |xv: &Tensor<f32>| -> f32 {
        let y = op(xv);
        let weighted = y.mul(w).expect("mul should succeed in numerical_gradient");
        let loss = weighted
            .sum(None, false)
            .expect("sum should succeed in numerical_gradient");
        loss.get(&[]).expect("scalar loss should be readable")
    };

    for i in 0..n {
        let mut plus = x_data.to_vec();
        plus[i] += eps;
        let x_plus = Tensor::from_vec(plus, &dims).expect("x_plus should build");

        let mut minus = x_data.to_vec();
        minus[i] -= eps;
        let x_minus = Tensor::from_vec(minus, &dims).expect("x_minus should build");

        let loss_plus = loss_of(&x_plus);
        let loss_minus = loss_of(&x_minus);
        grad[i] = (loss_plus - loss_minus) / (2.0 * eps);
    }

    grad
}

/// Assert two same-length slices are element-wise close within `tol`,
/// printing a descriptive message (including the differing index and both
/// values) on the first failing element.
fn assert_close(analytical: &[f32], numerical: &[f32], tol: f32, label: &str) {
    assert_eq!(
        analytical.len(),
        numerical.len(),
        "{label}: analytical/numerical gradient length mismatch"
    );
    for (i, (&a, &n)) in analytical.iter().zip(numerical.iter()).enumerate() {
        assert!(
            (a - n).abs() < tol,
            "{label}: grad[{i}] mismatch -- analytical={a}, numerical={n}, diff={}",
            (a - n).abs()
        );
    }
}

/// Plain-`Tensor` replica of `TrackedTensor::log_softmax`'s forward formula
/// (`x - max_axis(x) - log(sum_axis(exp(x - max_axis(x))))`), since
/// `log_softmax` is only exposed as a `TrackedTensor` method (implemented
/// inline in `tracked_tensor.rs`), not as a plain `Tensor` method. Used for
/// the finite-difference probe half of the log_softmax test, so both halves
/// (analytical via the tape, numerical via this helper) compute the exact
/// same forward function.
fn plain_log_softmax(x: &Tensor<f32>, axis: i32) -> Tensor<f32> {
    let axis_slice = [axis];
    let max_x = x
        .max(Some(&axis_slice), true)
        .expect("max should succeed in plain_log_softmax");
    let shifted = x
        .sub(&max_x)
        .expect("sub should succeed in plain_log_softmax");
    let exp_shifted = shifted
        .exp()
        .expect("exp should succeed in plain_log_softmax");
    let sum_exp = exp_shifted
        .sum(Some(&axis_slice), true)
        .expect("sum should succeed in plain_log_softmax");
    let log_sum_exp = sum_exp
        .log()
        .expect("log should succeed in plain_log_softmax");
    shifted
        .sub(&log_sum_exp)
        .expect("final sub should succeed in plain_log_softmax")
}

// ---------------------------------------------------------------------
// Part 1: gelu, swish, mish, leaky_relu, elu
// ---------------------------------------------------------------------

#[test]
fn test_gelu_gradient_basic_tensor_matches_finite_difference() {
    let tape = GradientTape::new();

    let x_vals = vec![-2.0_f32, -0.5, 0.3, 1.0, 2.5];
    let w_vals = vec![0.4_f32, -1.1, 0.7, -0.3, 1.5];
    let x = Tensor::<f32>::from_vec(x_vals.clone(), &[5]).expect("x should build");
    let w = Tensor::<f32>::from_vec(w_vals.clone(), &[5]).expect("w should build");

    let x_tracked = tape.watch(x.clone());
    let w_tracked = tape.watch(w.clone());

    let y = x_tracked.gelu().expect("gelu forward should succeed");
    let weighted = y.mul(&w_tracked).expect("mul should succeed");
    let loss = weighted.sum(None, false).expect("sum should succeed");

    let grads = tape
        .gradient(&[loss], &[x_tracked])
        .expect("gradient computation should succeed");
    let grad_x = grads[0].as_ref().expect("gradient must exist for x");
    let analytical = grad_x.as_slice().expect("gradient should be contiguous");

    let eps = 1e-3_f32;
    let numerical = numerical_gradient(&x, &w, eps, |t| {
        t.gelu().expect("plain gelu should succeed for fd probe")
    });

    assert_close(analytical, &numerical, 5e-2, "gelu");
}

#[test]
fn test_swish_gradient_basic_tensor_matches_finite_difference() {
    let tape = GradientTape::new();

    let x_vals = vec![-3.0_f32, -1.0, 0.0, 1.0, 3.0];
    let w_vals = vec![0.6_f32, -0.9, 1.2, -1.4, 0.3];
    let x = Tensor::<f32>::from_vec(x_vals.clone(), &[5]).expect("x should build");
    let w = Tensor::<f32>::from_vec(w_vals.clone(), &[5]).expect("w should build");

    let x_tracked = tape.watch(x.clone());
    let w_tracked = tape.watch(w.clone());

    let y = x_tracked.swish().expect("swish forward should succeed");
    let weighted = y.mul(&w_tracked).expect("mul should succeed");
    let loss = weighted.sum(None, false).expect("sum should succeed");

    let grads = tape
        .gradient(&[loss], &[x_tracked])
        .expect("gradient computation should succeed");
    let grad_x = grads[0].as_ref().expect("gradient must exist for x");
    let analytical = grad_x.as_slice().expect("gradient should be contiguous");

    let eps = 1e-3_f32;
    let numerical = numerical_gradient(&x, &w, eps, |t| {
        t.swish().expect("plain swish should succeed for fd probe")
    });

    assert_close(analytical, &numerical, 5e-2, "swish");
}

#[test]
fn test_mish_gradient_basic_tensor_matches_finite_difference() {
    let tape = GradientTape::new();

    let x_vals = vec![-2.5_f32, -1.0, 0.2, 1.5, 3.0];
    let w_vals = vec![1.0_f32, -0.5, 0.8, -1.2, 0.4];
    let x = Tensor::<f32>::from_vec(x_vals.clone(), &[5]).expect("x should build");
    let w = Tensor::<f32>::from_vec(w_vals.clone(), &[5]).expect("w should build");

    let x_tracked = tape.watch(x.clone());
    let w_tracked = tape.watch(w.clone());

    let y = x_tracked.mish().expect("mish forward should succeed");
    let weighted = y.mul(&w_tracked).expect("mul should succeed");
    let loss = weighted.sum(None, false).expect("sum should succeed");

    let grads = tape
        .gradient(&[loss], &[x_tracked])
        .expect("gradient computation should succeed");
    let grad_x = grads[0].as_ref().expect("gradient must exist for x");
    let analytical = grad_x.as_slice().expect("gradient should be contiguous");

    let eps = 1e-3_f32;
    let numerical = numerical_gradient(&x, &w, eps, |t| {
        t.mish().expect("plain mish should succeed for fd probe")
    });

    assert_close(analytical, &numerical, 5e-2, "mish");
}

#[test]
fn test_leaky_relu_gradient_basic_tensor_matches_finite_difference() {
    let tape = GradientTape::new();

    // Non-default alpha (default is typically 0.01) to confirm it threads
    // through the tape operation rather than being silently hardcoded.
    let alpha = 0.2_f32;

    let x_vals = vec![-4.0_f32, -1.5, -0.1, 0.5, 2.0];
    let w_vals = vec![0.3_f32, -1.0, 0.9, -0.4, 1.1];
    let x = Tensor::<f32>::from_vec(x_vals.clone(), &[5]).expect("x should build");
    let w = Tensor::<f32>::from_vec(w_vals.clone(), &[5]).expect("w should build");

    let x_tracked = tape.watch(x.clone());
    let w_tracked = tape.watch(w.clone());

    let y = x_tracked
        .leaky_relu(alpha)
        .expect("leaky_relu forward should succeed");
    let weighted = y.mul(&w_tracked).expect("mul should succeed");
    let loss = weighted.sum(None, false).expect("sum should succeed");

    let grads = tape
        .gradient(&[loss], &[x_tracked])
        .expect("gradient computation should succeed");
    let grad_x = grads[0].as_ref().expect("gradient must exist for x");
    let analytical = grad_x.as_slice().expect("gradient should be contiguous");

    let eps = 1e-3_f32;
    let numerical = numerical_gradient(&x, &w, eps, |t| {
        t.leaky_relu(alpha)
            .expect("plain leaky_relu should succeed for fd probe")
    });

    assert_close(analytical, &numerical, 5e-2, "leaky_relu (alpha=0.2)");

    // Confirm alpha actually threads through: the negative-input elements'
    // analytical gradient should be close to alpha * w_i (since d/dx
    // leaky_relu(x) = alpha for x < 0), and NOT close to the default alpha
    // (0.01) times w_i, nor to 1.0 * w_i (as if alpha were ignored and it
    // behaved like plain ReLU).
    for (i, &xv) in x_vals.iter().enumerate() {
        if xv < 0.0 {
            let expected = alpha * w_vals[i];
            assert!(
                (analytical[i] - expected).abs() < 1e-4,
                "leaky_relu alpha=0.2 at negative x[{i}]={xv}: expected grad {expected} \
                 (alpha * w), got {}",
                analytical[i]
            );
            let would_be_default_alpha = 0.01 * w_vals[i];
            assert!(
                (analytical[i] - would_be_default_alpha).abs() > 1e-3,
                "leaky_relu alpha=0.2 at negative x[{i}]={xv}: gradient {} is suspiciously \
                 close to what default alpha=0.01 would produce ({would_be_default_alpha}); \
                 alpha may not be threading through",
                analytical[i]
            );
        }
    }
}

#[test]
fn test_elu_gradient_basic_tensor_matches_finite_difference() {
    let tape = GradientTape::new();

    // Non-default alpha (default is typically 1.0) to confirm it threads
    // through the tape operation.
    let alpha = 1.5_f32;

    let x_vals = vec![-3.0_f32, -1.0, -0.2, 0.5, 2.0];
    let w_vals = vec![0.7_f32, -0.6, 1.3, -0.9, 0.2];
    let x = Tensor::<f32>::from_vec(x_vals.clone(), &[5]).expect("x should build");
    let w = Tensor::<f32>::from_vec(w_vals.clone(), &[5]).expect("w should build");

    let x_tracked = tape.watch(x.clone());
    let w_tracked = tape.watch(w.clone());

    let y = x_tracked.elu(alpha).expect("elu forward should succeed");
    let weighted = y.mul(&w_tracked).expect("mul should succeed");
    let loss = weighted.sum(None, false).expect("sum should succeed");

    let grads = tape
        .gradient(&[loss], &[x_tracked])
        .expect("gradient computation should succeed");
    let grad_x = grads[0].as_ref().expect("gradient must exist for x");
    let analytical = grad_x.as_slice().expect("gradient should be contiguous");

    let eps = 1e-3_f32;
    let numerical = numerical_gradient(&x, &w, eps, |t| {
        t.elu(alpha).expect("plain elu should succeed for fd probe")
    });

    assert_close(analytical, &numerical, 5e-2, "elu (alpha=1.5)");

    // Confirm alpha threads through: for x < 0, d/dx elu(x) = alpha * exp(x),
    // so analytical[i] should equal alpha * exp(x_i) * w_i.
    for (i, &xv) in x_vals.iter().enumerate() {
        if xv < 0.0 {
            let expected = alpha * xv.exp() * w_vals[i];
            assert!(
                (analytical[i] - expected).abs() < 1e-3,
                "elu alpha=1.5 at negative x[{i}]={xv}: expected grad {expected} \
                 (alpha * exp(x) * w), got {}",
                analytical[i]
            );
        }
    }
}

// ---------------------------------------------------------------------
// Part 2: log, abs, clamp, relu6, hard_swish, log_softmax
// ---------------------------------------------------------------------

#[test]
fn test_log_gradient_basic_positive_tensor_matches_finite_difference() {
    let tape = GradientTape::new();

    // Log requires strictly positive inputs; avoid values near/at zero.
    let x_vals = vec![0.5_f32, 1.0, 2.0, 3.5];
    let w_vals = vec![0.8_f32, -1.2, 0.5, -0.3];
    let x = Tensor::<f32>::from_vec(x_vals.clone(), &[4]).expect("x should build");
    let w = Tensor::<f32>::from_vec(w_vals.clone(), &[4]).expect("w should build");

    let x_tracked = tape.watch(x.clone());
    let w_tracked = tape.watch(w.clone());

    let y = x_tracked.log().expect("log forward should succeed");
    let weighted = y.mul(&w_tracked).expect("mul should succeed");
    let loss = weighted.sum(None, false).expect("sum should succeed");

    let grads = tape
        .gradient(&[loss], &[x_tracked])
        .expect("gradient computation should succeed");
    let grad_x = grads[0].as_ref().expect("gradient must exist for x");
    let analytical = grad_x.as_slice().expect("gradient should be contiguous");

    let eps = 1e-3_f32;
    let numerical = numerical_gradient(&x, &w, eps, |t| {
        t.log().expect("plain log should succeed for fd probe")
    });

    assert_close(analytical, &numerical, 5e-2, "log");

    // Also confirm against the closed-form d/dx log(x) = 1/x, so
    // grad_x[i] == w[i] / x[i].
    for i in 0..x_vals.len() {
        let expected = w_vals[i] / x_vals[i];
        assert!(
            (analytical[i] - expected).abs() < 1e-3,
            "log closed-form at x[{i}]={}: expected w/x = {expected}, got {}",
            x_vals[i],
            analytical[i]
        );
    }
}

#[test]
fn test_abs_gradient_mixed_sign_tensor_matches_finite_difference() {
    let tape = GradientTape::new();

    // Genuine mix of positive and negative values (no zero, since |x| has no
    // unique derivative there and this test targets the well-defined sign
    // flip away from the kink).
    let x_vals = vec![-3.0_f32, -1.0, 2.0, 4.5];
    let w_vals = vec![0.6_f32, -0.9, 1.1, -0.4];
    let x = Tensor::<f32>::from_vec(x_vals.clone(), &[4]).expect("x should build");
    let w = Tensor::<f32>::from_vec(w_vals.clone(), &[4]).expect("w should build");

    let x_tracked = tape.watch(x.clone());
    let w_tracked = tape.watch(w.clone());

    let y = x_tracked.abs().expect("abs forward should succeed");
    let weighted = y.mul(&w_tracked).expect("mul should succeed");
    let loss = weighted.sum(None, false).expect("sum should succeed");

    let grads = tape
        .gradient(&[loss], &[x_tracked])
        .expect("gradient computation should succeed");
    let grad_x = grads[0].as_ref().expect("gradient must exist for x");
    let analytical = grad_x.as_slice().expect("gradient should be contiguous");

    let eps = 1e-3_f32;
    let numerical = numerical_gradient(&x, &w, eps, |t| {
        t.abs().expect("plain abs should succeed for fd probe")
    });

    assert_close(analytical, &numerical, 5e-2, "abs (mixed sign)");

    // Confirm sign(x) flips correctly per-element: grad_x[i] == sign(x_i) * w[i].
    for i in 0..x_vals.len() {
        let sign = if x_vals[i] > 0.0 { 1.0 } else { -1.0 };
        let expected = sign * w_vals[i];
        assert!(
            (analytical[i] - expected).abs() < 1e-3,
            "abs sign check at x[{i}]={}: expected sign(x)*w = {expected}, got {}",
            x_vals[i],
            analytical[i]
        );
    }
}

#[test]
fn test_clamp_gradient_basic_tensor_both_bounds_matches_finite_difference() {
    let tape = GradientTape::new();

    let min = -1.0_f32;
    let max = 2.0_f32;

    // Values inside and outside both bounds (not exactly at the boundary --
    // boundary behavior is covered by dedicated tests below).
    let x_vals = vec![-3.0_f32, -0.5, 0.5, 1.5, 4.0];
    let w_vals = vec![0.5_f32, -0.8, 1.0, -0.3, 0.7];
    let x = Tensor::<f32>::from_vec(x_vals.clone(), &[5]).expect("x should build");
    let w = Tensor::<f32>::from_vec(w_vals.clone(), &[5]).expect("w should build");

    let x_tracked = tape.watch(x.clone());
    let w_tracked = tape.watch(w.clone());

    let y = x_tracked
        .clamp(Some(min), Some(max))
        .expect("clamp forward should succeed");
    let weighted = y.mul(&w_tracked).expect("mul should succeed");
    let loss = weighted.sum(None, false).expect("sum should succeed");

    let grads = tape
        .gradient(&[loss], &[x_tracked])
        .expect("gradient computation should succeed");
    let grad_x = grads[0].as_ref().expect("gradient must exist for x");
    let analytical = grad_x.as_slice().expect("gradient should be contiguous");

    let eps = 1e-3_f32;
    let numerical = numerical_gradient(&x, &w, eps, |t| {
        t.clamp(min, max)
            .expect("plain clamp should succeed for fd probe")
    });

    assert_close(analytical, &numerical, 5e-2, "clamp both bounds");

    // Closed-form check: gradient is w[i] where min <= x[i] <= max, else 0.
    for i in 0..x_vals.len() {
        let in_range = x_vals[i] >= min && x_vals[i] <= max;
        let expected = if in_range { w_vals[i] } else { 0.0 };
        assert!(
            (analytical[i] - expected).abs() < 1e-3,
            "clamp closed-form at x[{i}]={}: expected {expected} (in_range={in_range}), got {}",
            x_vals[i],
            analytical[i]
        );
    }
}

/// Clamp's backward mask uses `ge`/`le` (inclusive) comparisons against the
/// bounds, so gradient DOES flow at the boundary itself. This test places
/// points exactly at `x == min` and `x == max` and confirms the analytical
/// gradient equals `w[i]` there (not 0), matching the inclusive convention
/// documented on `process_clamp_backward`.
#[test]
fn test_clamp_gradient_at_inclusive_boundary_does_not_crash_and_flows() {
    let tape = GradientTape::new();

    let min = -1.0_f32;
    let max = 2.0_f32;

    // x[0] == min exactly, x[1] == max exactly, x[2] is a clearly interior
    // point as a sanity control.
    let x_vals = vec![min, max, 0.5_f32];
    let w_vals = vec![0.9_f32, -1.3, 0.4];
    let x = Tensor::<f32>::from_vec(x_vals.clone(), &[3]).expect("x should build");
    let w = Tensor::<f32>::from_vec(w_vals.clone(), &[3]).expect("w should build");

    let x_tracked = tape.watch(x);
    let w_tracked = tape.watch(w);

    let y = x_tracked
        .clamp(Some(min), Some(max))
        .expect("clamp forward should succeed at boundary");
    let weighted = y.mul(&w_tracked).expect("mul should succeed");
    let loss = weighted.sum(None, false).expect("sum should succeed");

    let grads = tape
        .gradient(&[loss], &[x_tracked])
        .expect("gradient computation should succeed at boundary");
    let grad_x = grads[0].as_ref().expect("gradient must exist for x");
    let analytical = grad_x.as_slice().expect("gradient should be contiguous");

    // Inclusive convention: gradient flows (equals w[i]) exactly AT the
    // boundary, for both x == min and x == max.
    assert!(
        (analytical[0] - w_vals[0]).abs() < 1e-5,
        "clamp at x == min ({min}) should let gradient flow (inclusive ge convention): \
         expected {}, got {}",
        w_vals[0],
        analytical[0]
    );
    assert!(
        (analytical[1] - w_vals[1]).abs() < 1e-5,
        "clamp at x == max ({max}) should let gradient flow (inclusive le convention): \
         expected {}, got {}",
        w_vals[1],
        analytical[1]
    );
    assert!(
        (analytical[2] - w_vals[2]).abs() < 1e-5,
        "clamp at interior point should let gradient flow: expected {}, got {}",
        w_vals[2],
        analytical[2]
    );
}

/// `max: None` (only `min` set) -- the max side must be unconstrained, so
/// gradient flows for arbitrarily large `x` as long as `x >= min`.
#[test]
fn test_clamp_gradient_max_none_only_min_set() {
    let tape = GradientTape::new();

    let min = 0.0_f32;

    // Note: deliberately avoid placing any value exactly at `min` here --
    // `clamp` has a kink there, and central differences straddling a kink
    // don't converge to the inclusive-boundary analytical value (the
    // dedicated boundary test above,
    // `test_clamp_gradient_at_inclusive_boundary_does_not_crash_and_flows`,
    // covers that case directly instead of via finite differences). `0.3`
    // replaces what would otherwise be an accidental boundary coincidence
    // at `0.0`.
    let x_vals = vec![-2.0_f32, -0.1, 0.3, 5.0, 1000.0];
    let w_vals = vec![0.3_f32, -0.7, 1.1, -0.5, 0.2];
    let x = Tensor::<f32>::from_vec(x_vals.clone(), &[5]).expect("x should build");
    let w = Tensor::<f32>::from_vec(w_vals.clone(), &[5]).expect("w should build");

    let x_tracked = tape.watch(x.clone());
    let w_tracked = tape.watch(w.clone());

    let y = x_tracked
        .clamp(Some(min), None)
        .expect("clamp forward (max=None) should succeed");
    let weighted = y.mul(&w_tracked).expect("mul should succeed");
    let loss = weighted.sum(None, false).expect("sum should succeed");

    let grads = tape
        .gradient(&[loss], &[x_tracked])
        .expect("gradient computation should succeed (max=None)");
    let grad_x = grads[0].as_ref().expect("gradient must exist for x");
    let analytical = grad_x.as_slice().expect("gradient should be contiguous");

    let eps = 1e-3_f32;
    let numerical = numerical_gradient(&x, &w, eps, |t| {
        t.clamp(min, f32::INFINITY)
            .expect("plain clamp (max=inf) should succeed for fd probe")
    });
    assert_close(analytical, &numerical, 5e-2, "clamp max=None");

    for i in 0..x_vals.len() {
        // Unconstrained on the max side: in-range iff x[i] >= min.
        let in_range = x_vals[i] >= min;
        let expected = if in_range { w_vals[i] } else { 0.0 };
        assert!(
            (analytical[i] - expected).abs() < 1e-3,
            "clamp(min={min}, max=None) at x[{i}]={}: expected {expected} \
             (in_range={in_range}), got {}",
            x_vals[i],
            analytical[i]
        );
    }
}

/// `min: None` (only `max` set) -- the min side must be unconstrained, so
/// gradient flows for arbitrarily negative `x` as long as `x <= max`.
#[test]
fn test_clamp_gradient_min_none_only_max_set() {
    let tape = GradientTape::new();

    let max = 3.0_f32;

    // Note: deliberately avoid placing any value exactly at `max` here --
    // `clamp` has a kink there, and central differences straddling a kink
    // don't converge to the inclusive-boundary analytical value (the
    // dedicated boundary test above,
    // `test_clamp_gradient_at_inclusive_boundary_does_not_crash_and_flows`,
    // covers that case directly instead of via finite differences). `2.5`
    // replaces what would otherwise be an accidental boundary coincidence
    // at `3.0`.
    let x_vals = vec![-1000.0_f32, -5.0, 0.0, 2.5, 10.0];
    let w_vals = vec![0.4_f32, -0.6, 0.9, -1.2, 0.5];
    let x = Tensor::<f32>::from_vec(x_vals.clone(), &[5]).expect("x should build");
    let w = Tensor::<f32>::from_vec(w_vals.clone(), &[5]).expect("w should build");

    let x_tracked = tape.watch(x.clone());
    let w_tracked = tape.watch(w.clone());

    let y = x_tracked
        .clamp(None, Some(max))
        .expect("clamp forward (min=None) should succeed");
    let weighted = y.mul(&w_tracked).expect("mul should succeed");
    let loss = weighted.sum(None, false).expect("sum should succeed");

    let grads = tape
        .gradient(&[loss], &[x_tracked])
        .expect("gradient computation should succeed (min=None)");
    let grad_x = grads[0].as_ref().expect("gradient must exist for x");
    let analytical = grad_x.as_slice().expect("gradient should be contiguous");

    let eps = 1e-3_f32;
    let numerical = numerical_gradient(&x, &w, eps, |t| {
        t.clamp(f32::NEG_INFINITY, max)
            .expect("plain clamp (min=-inf) should succeed for fd probe")
    });
    assert_close(analytical, &numerical, 5e-2, "clamp min=None");

    for i in 0..x_vals.len() {
        // Unconstrained on the min side: in-range iff x[i] <= max.
        let in_range = x_vals[i] <= max;
        let expected = if in_range { w_vals[i] } else { 0.0 };
        assert!(
            (analytical[i] - expected).abs() < 1e-3,
            "clamp(min=None, max={max}) at x[{i}]={}: expected {expected} \
             (in_range={in_range}), got {}",
            x_vals[i],
            analytical[i]
        );
    }
}

#[test]
fn test_relu6_gradient_interior_points_matches_finite_difference() {
    let tape = GradientTape::new();

    // Interior (non-boundary) points spanning below 0, between 0 and 6, and
    // above 6.
    let x_vals = vec![-2.0_f32, 1.0, 3.0, 5.0, 8.0];
    let w_vals = vec![0.6_f32, -0.9, 1.2, -0.4, 0.3];
    let x = Tensor::<f32>::from_vec(x_vals.clone(), &[5]).expect("x should build");
    let w = Tensor::<f32>::from_vec(w_vals.clone(), &[5]).expect("w should build");

    let x_tracked = tape.watch(x.clone());
    let w_tracked = tape.watch(w.clone());

    let y = x_tracked.relu6().expect("relu6 forward should succeed");
    let weighted = y.mul(&w_tracked).expect("mul should succeed");
    let loss = weighted.sum(None, false).expect("sum should succeed");

    let grads = tape
        .gradient(&[loss], &[x_tracked])
        .expect("gradient computation should succeed");
    let grad_x = grads[0].as_ref().expect("gradient must exist for x");
    let analytical = grad_x.as_slice().expect("gradient should be contiguous");

    let eps = 1e-3_f32;
    let numerical = numerical_gradient(&x, &w, eps, |t| {
        tenflowers_core::ops::activation::relu6(t).expect("plain relu6 should succeed for fd probe")
    });

    assert_close(analytical, &numerical, 5e-2, "relu6 (interior points)");

    // Closed-form check: gradient is w[i] where 0 < x[i] < 6 (strict), else 0.
    for i in 0..x_vals.len() {
        let in_range = x_vals[i] > 0.0 && x_vals[i] < 6.0;
        let expected = if in_range { w_vals[i] } else { 0.0 };
        assert!(
            (analytical[i] - expected).abs() < 1e-3,
            "relu6 closed-form at x[{i}]={}: expected {expected} (in_range={in_range}), got {}",
            x_vals[i],
            analytical[i]
        );
    }
}

/// Relu6's backward mask uses a STRICT/EXCLUSIVE boundary convention
/// (`0 < x < 6`, mirroring plain Relu's own strict `val > 0` convention),
/// unlike Relu6's *forward* clamp which is inclusive. This test places points
/// exactly AT `x = 0` and `x = 6` and asserts the analytical gradient is
/// exactly 0 there, per `process_relu6_backward`'s documented convention.
///
/// A genuine central-difference finite-difference probe is NOT meaningful
/// at these exact points: `relu6` has a kink (non-differentiable corner) at
/// both x=0 and x=6, so central differences straddle the kink and average
/// the left-derivative and right-derivative together, producing a numerical
/// value that is neither the strict-left nor strict-right one-sided
/// derivative -- there is no single well-defined "the" derivative there to
/// numerically approximate. Instead this test checks the ANALYTICALLY KNOWN
/// value (0, per the implementation's own documented strict convention)
/// directly.
#[test]
fn test_relu6_gradient_at_strict_exclusive_boundaries_is_exactly_zero() {
    let tape = GradientTape::new();

    // x[0] == 0 exactly, x[1] == 6 exactly.
    let x_vals = vec![0.0_f32, 6.0_f32];
    let w_vals = vec![1.0_f32, 1.0_f32];
    let x = Tensor::<f32>::from_vec(x_vals, &[2]).expect("x should build");
    let w = Tensor::<f32>::from_vec(w_vals, &[2]).expect("w should build");

    let x_tracked = tape.watch(x);
    let w_tracked = tape.watch(w);

    let y = x_tracked
        .relu6()
        .expect("relu6 forward should succeed at boundary");
    let weighted = y.mul(&w_tracked).expect("mul should succeed");
    let loss = weighted.sum(None, false).expect("sum should succeed");

    let grads = tape
        .gradient(&[loss], &[x_tracked])
        .expect("gradient computation should succeed at boundary");
    let grad_x = grads[0].as_ref().expect("gradient must exist for x");
    let analytical = grad_x.as_slice().expect("gradient should be contiguous");

    assert!(
        analytical[0].abs() < 1e-6,
        "relu6 gradient at x=0 (strict exclusive boundary) must be exactly 0, got {}",
        analytical[0]
    );
    assert!(
        analytical[1].abs() < 1e-6,
        "relu6 gradient at x=6 (strict exclusive boundary) must be exactly 0, got {}",
        analytical[1]
    );
}

#[test]
fn test_hard_swish_gradient_doc_comment_sanity_points() {
    // The implementation's own doc comments state the accumulated gradient
    // (with grad_output = 1) at three specific points:
    //   x = -5 -> local_grad = 0
    //   x =  0 -> local_grad = 0.5
    //   x =  5 -> local_grad = 1
    // Verified by hand from local_grad = (relu6(x+3) + x*relu6'(x+3)) / 6:
    //   x=-5: x+3=-2, relu6(-2)=0, relu6'(-2)=0 => (0 + (-5)*0)/6 = 0
    //   x=0:  x+3=3,  relu6(3)=3,  relu6'(3)=1  => (3 + 0*1)/6 = 0.5
    //   x=5:  x+3=8,  relu6(8)=6,  relu6'(8)=0  => (6 + 5*0)/6 = 1
    let tape = GradientTape::new();

    let x_vals = vec![-5.0_f32, 0.0, 5.0];
    // grad_output = 1 for every element (w = ones), so the accumulated
    // gradient equals local_grad directly, matching the doc comment values.
    let w_vals = vec![1.0_f32, 1.0, 1.0];
    let x = Tensor::<f32>::from_vec(x_vals, &[3]).expect("x should build");
    let w = Tensor::<f32>::from_vec(w_vals, &[3]).expect("w should build");

    let x_tracked = tape.watch(x);
    let w_tracked = tape.watch(w);

    let y = x_tracked
        .hard_swish()
        .expect("hard_swish forward should succeed");
    let weighted = y.mul(&w_tracked).expect("mul should succeed");
    let loss = weighted.sum(None, false).expect("sum should succeed");

    let grads = tape
        .gradient(&[loss], &[x_tracked])
        .expect("gradient computation should succeed");
    let grad_x = grads[0].as_ref().expect("gradient must exist for x");
    let analytical = grad_x.as_slice().expect("gradient should be contiguous");

    assert!(
        (analytical[0] - 0.0).abs() < 1e-5,
        "hard_swish local_grad at x=-5 should be 0 (doc comment sanity point), got {}",
        analytical[0]
    );
    assert!(
        (analytical[1] - 0.5).abs() < 1e-5,
        "hard_swish local_grad at x=0 should be 0.5 (doc comment sanity point), got {}",
        analytical[1]
    );
    assert!(
        (analytical[2] - 1.0).abs() < 1e-5,
        "hard_swish local_grad at x=5 should be 1 (doc comment sanity point), got {}",
        analytical[2]
    );
}

#[test]
fn test_hard_swish_gradient_multi_region_matches_finite_difference() {
    let tape = GradientTape::new();

    // Spans multiple regions: below -3 (zero-grad region), between -3 and 3
    // (interior), at 0, and above 3 (the region where relu6(x+3) saturates
    // at 6 but x itself is nonzero, so the second product-rule term still
    // matters near the transition).
    let x_vals = vec![-5.0_f32, -1.0, 0.0, 1.5, 5.0];
    let w_vals = vec![0.5_f32, -0.8, 1.0, -0.6, 0.4];
    let x = Tensor::<f32>::from_vec(x_vals.clone(), &[5]).expect("x should build");
    let w = Tensor::<f32>::from_vec(w_vals.clone(), &[5]).expect("w should build");

    let x_tracked = tape.watch(x.clone());
    let w_tracked = tape.watch(w.clone());

    let y = x_tracked
        .hard_swish()
        .expect("hard_swish forward should succeed");
    let weighted = y.mul(&w_tracked).expect("mul should succeed");
    let loss = weighted.sum(None, false).expect("sum should succeed");

    let grads = tape
        .gradient(&[loss], &[x_tracked])
        .expect("gradient computation should succeed");
    let grad_x = grads[0].as_ref().expect("gradient must exist for x");
    let analytical = grad_x.as_slice().expect("gradient should be contiguous");

    let eps = 1e-3_f32;
    let numerical = numerical_gradient(&x, &w, eps, |t| {
        t.hard_swish()
            .expect("plain hard_swish should succeed for fd probe")
    });

    assert_close(analytical, &numerical, 5e-2, "hard_swish (multi-region)");
}

/// LogSoftmax at rank 2 (`[batch, features]`, softmax over the last axis),
/// mirroring `test_softmax_gradient_rank2_is_axis_aware_not_flattened`'s
/// structure closely, to guard against a flattened/axis-blind bug (the same
/// class of bug that regression test documents for plain softmax).
///
/// IMPORTANT: log_softmax's gradient formula is
///   grad_x = grad_y - softmax(x) * sum_axis(grad_y)
/// which is genuinely DIFFERENT from plain softmax's own formula
///   grad_x = y * (grad_y - sum_axis(y * grad_y))
/// This test's expected values are derived from the LOG-softmax formula
/// specifically (cross-checked against finite differences below), not
/// against softmax's formula.
#[test]
fn test_log_softmax_gradient_rank2_is_axis_aware_not_flattened() {
    let tape = GradientTape::new();

    // x: [batch=2, features=3]
    let x_vals = vec![1.0_f32, 2.0, 3.0, 0.5, -1.0, 2.5];
    let x = Tensor::<f32>::from_vec(x_vals.clone(), &[2, 3]).expect("x should build");
    let x_tracked = tape.watch(x.clone());

    let y = x_tracked
        .log_softmax(Some(-1))
        .expect("log_softmax forward should succeed");

    // exp(log_softmax(x)) should sum to 1.0 per row (i.e. log_softmax's
    // implicit softmax is a valid probability distribution per row).
    let y_data = y.tensor.as_slice().expect("tensor should be contiguous");
    for row in y_data.chunks(3) {
        let row_sum: f32 = row.iter().map(|v| v.exp()).sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-4,
            "exp(log_softmax) row should sum to 1.0, got {row_sum}"
        );
    }

    // Non-uniform upstream weight w, watched as a plain (undifferentiated)
    // constant, so grad_output arriving at the log_softmax node equals w
    // exactly: loss = sum(w * log_softmax(x)) => d(loss)/d(log_softmax(x)) = w.
    let w_vals = vec![0.3_f32, -0.2, 1.0, -0.5, 0.7, 0.1];
    let w = Tensor::<f32>::from_vec(w_vals.clone(), &[2, 3]).expect("w should build");
    let w_tracked = tape.watch(w.clone());

    let weighted = y.mul(&w_tracked).expect("mul should succeed");
    let loss = weighted.sum(None, false).expect("sum should succeed");

    let grads = tape
        .gradient(&[loss], &[x_tracked])
        .expect("gradient computation should succeed");
    let grad_x = grads[0].as_ref().expect("gradient must exist for x");
    assert_eq!(grad_x.shape().dims(), &[2, 3]);
    let analytical = grad_x.as_slice().expect("gradient should be contiguous");

    // Finite-difference cross-check using the LOG-softmax formula's own
    // forward (not plain softmax's forward).
    let eps = 1e-3_f32;
    let numerical = numerical_gradient(&x, &w, eps, |t| plain_log_softmax(t, -1));
    assert_close(
        analytical,
        &numerical,
        5e-2,
        "log_softmax rank2 (finite difference)",
    );

    // Hand-derived closed form per row, via grad_x = grad_y - softmax(x) *
    // sum_axis(grad_y), i.e. grad_x = w - softmax(x) * sum_axis(w). This is
    // explicitly the LOG-softmax formula, NOT plain softmax's
    // y * (grad_y - sum_axis(y * grad_y)) formula -- the two differ whenever
    // grad_y is non-uniform, which it is here.
    //
    // softmax(x) per row (same values as
    // test_softmax_gradient_rank2_is_axis_aware_not_flattened's expected_y,
    // since it's the same x): row0 = [0.09003057, 0.24472847, 0.66524096],
    // row1 = [0.11611453, 0.02590865, 0.85797681].
    let softmax_row0 = [0.090_030_57_f32, 0.244_728_47, 0.665_240_96];
    let softmax_row1 = [0.116_114_53_f32, 0.025_908_65, 0.857_976_8];
    let w_row0_sum: f32 = w_vals[0..3].iter().sum(); // 0.3 - 0.2 + 1.0 = 1.1
    let w_row1_sum: f32 = w_vals[3..6].iter().sum(); // -0.5 + 0.7 + 0.1 = 0.3

    let mut expected = [0.0_f32; 6];
    for j in 0..3 {
        expected[j] = w_vals[j] - softmax_row0[j] * w_row0_sum;
        expected[3 + j] = w_vals[3 + j] - softmax_row1[j] * w_row1_sum;
    }

    for (i, (&got, &exp)) in analytical.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "log_softmax grad_x[{i}]: expected axis-aware log-softmax-formula value {exp}, \
             got {got}"
        );
    }

    // Negative check: confirm this is NOT plain softmax's formula. Plain
    // softmax backward would give y * (grad_y - sum_axis(y * grad_y)); at
    // this input the two formulas produce visibly different numbers (since
    // w is non-uniform), so the analytical result should NOT match softmax's
    // formula evaluated with the same softmax(x) and w.
    let mut softmax_formula_result = [0.0_f32; 6];
    for j in 0..3 {
        let dot0: f32 = (0..3).map(|k| softmax_row0[k] * w_vals[k]).sum();
        softmax_formula_result[j] = softmax_row0[j] * (w_vals[j] - dot0);
        let dot1: f32 = (0..3).map(|k| softmax_row1[k] * w_vals[3 + k]).sum();
        softmax_formula_result[3 + j] = softmax_row1[j] * (w_vals[3 + j] - dot1);
    }
    let max_diff_from_softmax_formula = analytical
        .iter()
        .zip(softmax_formula_result.iter())
        .map(|(&a, &s)| (a - s).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff_from_softmax_formula > 0.05,
        "log_softmax's analytical gradient is suspiciously close to plain softmax's \
         gradient formula (max diff {max_diff_from_softmax_formula}); the two formulas \
         are genuinely different for non-uniform upstream gradients and should diverge here"
    );
}
