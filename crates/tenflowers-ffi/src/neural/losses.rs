//! Loss functions module for TenfloweRS FFI
//!
//! This module provides comprehensive loss function implementations for neural network training,
//! including MSE, CrossEntropy, BCE, and other common loss functions.
//!
//! # Autograd
//!
//! Every loss below (including [`mse_loss`]) is computed entirely through
//! tape-aware operations — [`PyTensor::sub`]/[`PyTensor::mul`]/[`PyTensor::add`]/
//! [`PyTensor::div`] and [`crate::math_ops::sum`]/[`crate::math_ops::mean`] —
//! rather than pulling tensors down to a raw `Vec<f32>`, looping, and
//! rebuilding a `PyTensor` with `Tensor::from_vec`. The latter has no
//! relationship to [`crate::implicit_autograd`]'s tape, so a `PyTensor`
//! produced that way can never be the target of a `.backward()` call (see
//! `crate::implicit_autograd::run_backward`, which requires the tensor to be
//! looked up in the tape's tracked-tensor registry).
//!
//! ## The `log`/`abs`/`clamp`/`sqrt` gap, and how this module closes it
//!
//! [`crate::implicit_autograd::UnaryOpKind`] (the tag enum
//! [`crate::implicit_autograd::record_and_link_unary`] dispatches on) only
//! has variants for `Relu`/`Sigmoid`/`Tanh`/`Softmax`/`Transpose`/`Reshape`/
//! `Slice`/`Sum`/`Mean` today. `Log`/`Abs`/`Clamp` do **not** have a variant
//! there yet, even though `tenflowers_autograd::TrackedTensor` already has
//! working `log`/`abs`/`clamp` methods with real, tested backward
//! implementations (`process_log_backward`/`process_abs_backward`/
//! `process_clamp_backward` in
//! `tenflowers-autograd/src/tape/gradient_computation/activation_ops.rs`) —
//! the FFI-layer `UnaryOpKind` enum and its `record_and_link_unary` match
//! arms were simply never updated to expose them. `sqrt` has no
//! `TrackedTensor` method or `Operation` variant at all yet.
//!
//! Every binary-cross-entropy/cross-entropy/KL/L1/smooth-L1/cosine loss below
//! needs at least one of these four operations. Rather than duplicate
//! `implicit_autograd/mod.rs`'s tape-recording machinery in this file (which
//! would create two independent sources of truth for how an operation gets
//! linked onto the tape), this module instead expresses `log`/`abs`/`clamp`/
//! `sqrt` **compositionally**, purely in terms of the tape kinds that
//! already exist end-to-end (`Sub`, `Mul`, `Div`, `Add`, `Sum`, `Mean`):
//!
//! * `log(x)`: recorded as `Div(x, x_detached)`, where `x_detached` is an
//!   untracked constant copy of `x`'s own values. `Div`'s real backward is
//!   `grad_lhs = grad_output / rhs`, so gradient flowing back through the
//!   `lhs` (`x`) slot is `grad_output / x` — exactly `d(log(x))/dx`. The
//!   `rhs` slot must be untracked (never `mark_leaf`'d, never the `result` of
//!   another tracked op), or `Div`'s *other* backward formula
//!   (`grad_rhs = -grad_output * lhs / rhs²`) would also feed a second,
//!   spurious gradient contribution back into `x` through that slot.
//! * `abs(x)`: recorded as `Mul(x, sign(x)_detached)`. `Mul`'s backward is
//!   `grad_lhs = grad_output * rhs`, giving `grad_output * sign(x)` —
//!   exactly `d(|x|)/dx` (including the `sign(0) := 0` convention
//!   `process_abs_backward` itself documents and uses, which this
//!   composition reproduces exactly because `0 * anything = 0`, never a
//!   `0/0` or division-by-zero `NaN`/`Inf` the way a naive `Div`-based
//!   `x / sign(x)` composition would at `x == 0`).
//! * `clamp(x, lo, hi)`: recorded as `Mul(x, in_range_mask_detached)`, where
//!   `in_range_mask` is `1.0` where `lo <= x <= hi` and `0.0` elsewhere —
//!   exactly the gate `process_clamp_backward` itself computes and applies.
//! * `sqrt(x)`: recorded as `Div(x, two_sqrt_x_detached)`, where
//!   `two_sqrt_x_detached = 2 * sqrt(x)` (detached). `grad_lhs =
//!   grad_output / (2 * sqrt(x))` — exactly `d(sqrt(x))/dx`.
//!
//! In every case above, the **forward value** returned to Python is always
//! computed via the real, already-stable `tenflowers_core::ops::{log, abs,
//! clamp, sqrt}` kernels — never via whatever the proxy `Div`/`Mul` tape node
//! happens to compute internally (which is discarded; see
//! [`link_unary_via_binary_proxy`]'s doc for why this split is sound). This
//! is the same "compute the forward value with the existing stable math, but
//! separately establish the correct tape edge" idea
//! [`crate::implicit_autograd::record_and_link_binary`]/
//! [`record_and_link_unary`] already use for a non-tracked constant operand,
//! just applied one level up.
//!
//! Composing an entire loss formula out of elementary pieces that each have
//! a correct *local* gradient edge is sufficient for the *whole* expression's
//! gradient to come out correct, with no per-function hand-derivation of
//! product/chain-rule terms needed: every intermediate `PyTensor` on the path
//! (e.g. `log(t_c)` inside [`kl_div_loss`]) is itself tape-linked back to
//! its own inputs, so [`crate::implicit_autograd::run_backward`]'s ordinary
//! reverse-mode traversal accumulates every contribution automatically,
//! exactly as it does for any other multi-step computation.
//!
//! ## Coordination flag for whoever owns `implicit_autograd/mod.rs` next
//!
//! The compositional approach above is deliberately *not* a permanent
//! substitute for real `UnaryOpKind::{Log, Abs, Clamp}` variants (and a real
//! `sqrt` `Operation`/`TrackedTensor` method) — it exists because this file
//! is scoped to not modify `implicit_autograd/mod.rs`, `tensor_ops.rs`, or any
//! other `neural/*.rs` file. Once `Log`/`Abs`/`Clamp` are added to
//! `UnaryOpKind` (the `TrackedTensor` and `Operation` sides already exist —
//! see the module-level doc above) and a `sqrt` `Operation`/`TrackedTensor`
//! method + `UnaryOpKind::Sqrt` variant are added, every helper in this file
//! ([`tape_log`], [`tape_abs`], [`tape_clamp`], [`tape_sqrt`]) can be
//! simplified to a direct `record_and_link_unary` call, dropping the
//! `Div`/`Mul`-proxy indirection — the loss functions that call them would
//! not need to change at all.

use crate::implicit_autograd::{self, BinaryOpKind};
use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;
use tenflowers_core::Tensor;

// ============================================================================
// Tape-aware compositional helpers for log/abs/clamp/sqrt.
//
// See the module-level doc's "The log/abs/clamp/sqrt gap" section for why
// these exist and exactly what each one records. Every helper here follows
// the same shape: compute the real forward value via `tenflowers_core::ops`,
// build a *detached* (non-tracked, non-leaf) constant `PyTensor` holding
// whatever value makes the proxy binary op's backward formula equal the
// true target op's backward formula, then call
// `implicit_autograd::record_and_link_binary` to establish the tape edge.
// ============================================================================

/// Wrap a raw `tenflowers_core::Tensor<f32>` as a fresh, non-tracked,
/// non-leaf constant [`PyTensor`] (`requires_grad: false`).
///
/// This is the "detached constant" building block every helper below uses
/// for whichever operand must **not** carry gradient back through it (e.g.
/// `sign(x)` in [`tape_abs`]'s `Mul(x, sign(x))` composition) — a fresh
/// `Arc<Tensor<f32>>` allocation is never present in
/// `crate::implicit_autograd`'s tracked-tensor registry, so
/// `implicit_autograd::lookup_tracked` on it always returns `None` and
/// `record_and_link_binary` correctly treats it as an untracked constant
/// (auto-watching it on the shared tape as a non-leaf, per that function's
/// own doc), never as something `.backward()` should differentiate with
/// respect to.
fn detached(tensor: Tensor<f32>) -> PyTensor {
    PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    }
}

/// Link `result` (whose forward value was already computed by the caller via
/// some real, stable, non-tape-aware `tenflowers_core::ops::*` kernel) onto
/// the implicit tape as `proxy_kind(tracked_operand, detached_operand)`,
/// *if* `tracked_operand` is currently participating in implicit autograd.
///
/// # Why the proxy node's own forward value is never read (and therefore
/// never needs to match `result`'s)
///
/// `implicit_autograd::record_and_link_binary` calls e.g. `lhs_tt.div(&rhs_tt)`
/// to obtain a `TrackedTensor` (see that function's implementation), which
/// *does* eagerly compute its own forward value internally — but that value
/// is only ever used by `TrackedTensor::div`'s own future backward pass
/// (which reads `lhs`/`rhs`'s *own* stored values from the tape by id, not
/// anything routed through `result`) and is otherwise discarded:
/// `record_and_link_binary` only extracts the recorded node's *identity* via
/// `register_tracked(result, Arc::new(recorded))`, which maps
/// `tensor_key(result)` — the address of `result`'s own, separately
/// constructed `Arc<Tensor<f32>>` — to that node for future graph traversal.
/// Nothing ever compares `recorded`'s internal forward value against
/// `result.tensor`'s contents, and nothing ever surfaces `recorded`'s forward
/// value to Python. This is what makes it sound for the proxy op's forward
/// computation to be numerically different from (or even undefined at
/// isolated points relative to) the true target op's forward value — e.g.
/// [`tape_abs`]'s `Div`-avoiding `Mul(x, sign(x))` proxy is deliberately
/// chosen so this never actually diverges (see that function's doc), but the
/// general mechanism here does not depend on it *not* diverging.
///
/// # Errors
///
/// Returns an error only if `tracked_operand` is tracked but recording the
/// operation on the tape itself fails (e.g. a poisoned lock) — mirrors
/// [`implicit_autograd::record_and_link_binary`]'s own error contract.
fn link_unary_via_binary_proxy(
    proxy_kind: BinaryOpKind,
    tracked_operand: &PyTensor,
    detached_operand: &PyTensor,
    result: &PyTensor,
) -> PyResult<()> {
    implicit_autograd::record_and_link_binary(proxy_kind, tracked_operand, detached_operand, result)
}

/// Link `result` (whose forward value the caller already computed via real,
/// stable math) onto the implicit tape as the **sum of independent partial
/// contributions**, one per `(tracked_operand, local_gradient)` pair in
/// `terms`: for each pair, `d(result)/d(tracked_operand) = local_gradient`
/// (already the correct, hand-derived partial derivative, evaluated at the
/// current forward values and supplied as a detached constant — see
/// [`constant_filled`]/[`detached`]).
///
/// # Why this exists (`cosine_embedding_loss`'s nested-proxy bug)
///
/// A naive `cos_sim = dot_product / (tape_sqrt(sum_sq1).mul(&tape_sqrt(sum_sq2)))`
/// composition — chaining [`tape_sqrt`]'s proxy *output* directly into a
/// further, separately-recorded `Div`/`Mul` — was empirically found to
/// silently corrupt the gradient (by a clean, reproducible integer factor):
/// [`link_unary_via_binary_proxy`]'s proxy node has a tape-*internal* forward
/// value (whatever `TrackedTensor::div`/`mul` itself eagerly recomputes from
/// its own `lhs`/`rhs`) that is **deliberately allowed to diverge** from the
/// real value written into the returned `PyTensor` (see that function's own
/// doc — this divergence is harmless as long as nothing downstream reads the
/// proxy node's value again). `Div`'s backward formula for its denominator
/// operand (`grad_rhs = -grad_output*lhs/rhs²`) *does* read that operand's
/// stored value — and squares it — so feeding a proxy's output into another
/// `Div` as the denominator reads and squares the *stale* internal value
/// instead of the real one, producing a wrong-by-a-clean-factor gradient
/// instead of an error (which would have been easier to catch). Verified via
/// hand-computation cross-checked against an isolated failing test case
/// during development (`Div(10, sqrt_proxy(4))` produced gradient `-2.5`
/// instead of the correct `-0.625` — off by exactly `4×`, i.e. the square of
/// the proxy's `2×` internal-value divergence).
///
/// This helper sidesteps the problem at its root by never letting a
/// proxy-of-a-proxy chain form at all: every `tracked_operand` passed in must
/// be a **real** (non-proxy) tape node — i.e. either a genuine leaf or the
/// output of an *ordinary* tape-aware op (`Add`/`Sub`/`Mul`/`Sum`/`Mean`,
/// none of which are proxies and none of which have the "internal value
/// deliberately diverges from the real value" property) — and each
/// contribution is linked via a single `Mul(tracked_operand,
/// local_gradient_detached)` **directly against the real, correct partial
/// derivative**, never against another proxy's output. Multiple
/// contributions are then combined with plain, tape-aware `Add`, whose
/// backward reads neither operand's value (see this module's top-level doc
/// and [`link_unary_via_binary_proxy`]'s own doc for why `Add`/`Sub` are
/// always safe to chain regardless of what feeds them) — so summing several
/// of these `Mul` proxies together introduces no further risk of the same
/// bug.
///
/// # Errors
///
/// Returns an error only if a tracked operand's proxy `Mul`/`Add` recording
/// itself fails (e.g. a poisoned tape lock).
fn link_multi_gradient_sum(
    terms: &[(&PyTensor, &PyTensor)],
    result: &PyTensor,
) -> PyResult<()> {
    let Some((first_operand, first_grad)) = terms.first() else {
        // No differentiable contributions at all (e.g. every input was a
        // plain constant) -- nothing to link.
        return Ok(());
    };

    if terms.len() == 1 {
        // Exactly one contribution: link it directly onto `result` (using
        // `result`'s own already-correct, caller-supplied real forward
        // value) -- no intermediate placeholder/Add needed at all.
        link_unary_via_binary_proxy(BinaryOpKind::Mul, first_operand, first_grad, result)?;
        return Ok(());
    }

    // Two or more contributions: every contribution except the LAST is
    // linked through an intermediate placeholder `PyTensor` (never read for
    // its own value by anything -- it only ever becomes an `Add` operand,
    // and `Add`'s backward reads neither operand's value, so an incorrect
    // placeholder forward value here is exactly as harmless as any other
    // proxy node's discarded internal value; see this function's own doc).
    // The FINAL `Add` in the chain is linked directly onto `result` itself
    // (using the caller's real, correct forward value) rather than yet
    // another placeholder, so no "re-parent an already-recorded node onto a
    // different PyTensor identity" step is ever needed -- `result` IS the
    // last node's identity from the start.
    let placeholder_value = tenflowers_core::ops::mul(&first_operand.tensor, &first_grad.tensor)
        .map_err(|e| PyRuntimeError::new_err(format!("multi-gradient link failed: {}", e)))?;
    let mut accumulated = PyTensor {
        tensor: Arc::new(placeholder_value),
        requires_grad: first_operand.requires_grad,
        is_pinned: first_operand.is_pinned,
    };
    link_unary_via_binary_proxy(BinaryOpKind::Mul, first_operand, first_grad, &accumulated)?;

    let last_index = terms.len() - 1;
    for (i, (operand, local_grad)) in terms[1..].iter().enumerate() {
        let is_last = i == last_index - 1;

        let placeholder_value = tenflowers_core::ops::mul(&operand.tensor, &local_grad.tensor)
            .map_err(|e| PyRuntimeError::new_err(format!("multi-gradient link failed: {}", e)))?;
        let contribution = PyTensor {
            tensor: Arc::new(placeholder_value),
            requires_grad: operand.requires_grad,
            is_pinned: operand.is_pinned,
        };
        link_unary_via_binary_proxy(BinaryOpKind::Mul, operand, local_grad, &contribution)?;

        if is_last {
            // Final contribution: sum directly onto `result` (real,
            // caller-supplied forward value) instead of another placeholder.
            implicit_autograd::record_and_link_binary(
                BinaryOpKind::Add,
                &accumulated,
                &contribution,
                result,
            )?;
        } else {
            let placeholder_sum =
                tenflowers_core::ops::add(&accumulated.tensor, &contribution.tensor).map_err(
                    |e| PyRuntimeError::new_err(format!("multi-gradient link failed: {}", e)),
                )?;
            let new_accumulated = PyTensor {
                tensor: Arc::new(placeholder_sum),
                requires_grad: accumulated.requires_grad || contribution.requires_grad,
                is_pinned: accumulated.is_pinned || contribution.is_pinned,
            };
            implicit_autograd::record_and_link_binary(
                BinaryOpKind::Add,
                &accumulated,
                &contribution,
                &new_accumulated,
            )?;
            accumulated = new_accumulated;
        }
    }

    Ok(())
}

/// Tape-aware natural logarithm: `log(x)`, linked via the `Div(x, x_detached)`
/// proxy described in this module's top-level doc.
fn tape_log(x: &PyTensor) -> PyResult<PyTensor> {
    let forward = tenflowers_core::ops::log(&x.tensor)
        .map_err(|e| PyRuntimeError::new_err(format!("Log failed: {}", e)))?;
    let result = PyTensor {
        tensor: Arc::new(forward),
        requires_grad: x.requires_grad,
        is_pinned: x.is_pinned,
    };
    let x_detached = detached((*x.tensor).clone());
    link_unary_via_binary_proxy(BinaryOpKind::Div, x, &x_detached, &result)?;
    Ok(result)
}

/// Tape-aware absolute value: `|x|`, linked via the `Mul(x, sign(x)_detached)`
/// proxy described in this module's top-level doc. `sign(0) := 0`, matching
/// `process_abs_backward`'s own documented convention exactly.
fn tape_abs(x: &PyTensor) -> PyResult<PyTensor> {
    let forward = tenflowers_core::ops::abs(&x.tensor)
        .map_err(|e| PyRuntimeError::new_err(format!("Abs failed: {}", e)))?;
    let result = PyTensor {
        tensor: Arc::new(forward),
        requires_grad: x.requires_grad,
        is_pinned: x.is_pinned,
    };
    let sign = tenflowers_core::ops::numpy_compat::sign(&x.tensor)
        .map_err(|e| PyRuntimeError::new_err(format!("Sign failed: {}", e)))?;
    let sign_detached = detached(sign);
    link_unary_via_binary_proxy(BinaryOpKind::Mul, x, &sign_detached, &result)?;
    Ok(result)
}

/// Tape-aware clamp: `clamp(x, min_val, max_val)`, linked via the
/// `Mul(x, in_range_mask_detached)` proxy described in this module's
/// top-level doc (`in_range_mask` is `1.0` where `min_val <= x <= max_val`,
/// `0.0` elsewhere — exactly the gate `process_clamp_backward` computes).
fn tape_clamp(x: &PyTensor, min_val: f32, max_val: f32) -> PyResult<PyTensor> {
    let forward = tenflowers_core::ops::clamp(&x.tensor, min_val, max_val)
        .map_err(|e| PyRuntimeError::new_err(format!("Clamp failed: {}", e)))?;
    let result = PyTensor {
        tensor: Arc::new(forward),
        requires_grad: x.requires_grad,
        is_pinned: x.is_pinned,
    };
    let x_data = x
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read tensor for clamp: {}", e)))?;
    let mask_data: Vec<f32> = x_data
        .iter()
        .map(|&v| {
            if v >= min_val && v <= max_val {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let mask_tensor = Tensor::from_vec(mask_data, x.tensor.shape().dims())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to build clamp mask: {}", e)))?;
    let mask_detached = detached(mask_tensor);
    link_unary_via_binary_proxy(BinaryOpKind::Mul, x, &mask_detached, &result)?;
    Ok(result)
}

/// Tape-aware square root: `sqrt(x)`, linked via the
/// `Div(x, (2*sqrt(x))_detached)` proxy described in this module's top-level
/// doc.
fn tape_sqrt(x: &PyTensor) -> PyResult<PyTensor> {
    let forward = tenflowers_core::ops::sqrt(&x.tensor)
        .map_err(|e| PyRuntimeError::new_err(format!("Sqrt failed: {}", e)))?;
    let result = PyTensor {
        tensor: Arc::new(forward.clone()),
        requires_grad: x.requires_grad,
        is_pinned: x.is_pinned,
    };
    // 2 * sqrt(x), the denominator of d(sqrt(x))/dx = 1 / (2 * sqrt(x)).
    let two_sqrt_x = tenflowers_core::ops::add(&forward, &forward)
        .map_err(|e| PyRuntimeError::new_err(format!("Sqrt derivative setup failed: {}", e)))?;
    let two_sqrt_x_detached = detached(two_sqrt_x);
    link_unary_via_binary_proxy(BinaryOpKind::Div, x, &two_sqrt_x_detached, &result)?;
    Ok(result)
}

/// Build a fresh, non-tracked constant [`PyTensor`] of `value`, broadcast to
/// `shape` (via `Tensor::from_vec` with a repeated fill — deliberately not
/// `Tensor::from_scalar`, whose true rank-0 shape would need to survive a
/// broadcasting `Div`/`Mul`; the live `Operation::Div` backward path does not
/// unbroadcast, see [`tape_log`]/[`tape_sqrt`]'s doc, so every proxy operand
/// in this module is always built already matching its tracked counterpart's
/// exact shape rather than relying on broadcast).
fn constant_filled(value: f32, shape: &[usize]) -> PyResult<PyTensor> {
    let numel: usize = shape.iter().product();
    let tensor = Tensor::from_vec(vec![value; numel], shape)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to build constant tensor: {}", e)))?;
    Ok(detached(tensor))
}

/// Tape-aware element-wise negation: `-x`, expressed as `x.mul(&neg_one)`
/// (using the ordinary, already-tape-linked [`PyTensor::mul`] — `Mul`'s real
/// backward `grad_lhs = grad_output * rhs = grad_output * (-1) =
/// -grad_output` is exactly negation's gradient, so this needs no proxy
/// trick of its own).
fn tape_neg(x: &PyTensor) -> PyResult<PyTensor> {
    let neg_one = constant_filled(-1.0, x.tensor.shape().dims())?;
    x.mul(&neg_one)
}

/// Element-wise `1 - x`, expressed as `ones.sub(x)` (ordinary, already
/// tape-linked [`PyTensor::sub`]).
fn one_minus(x: &PyTensor) -> PyResult<PyTensor> {
    let ones = constant_filled(1.0, x.tensor.shape().dims())?;
    ones.sub(x)
}

/// Build a detached 0/1 mask [`PyTensor`] the same shape as `reference`,
/// where element `i` is `1.0` if `predicate(reference_data[i])` else `0.0`.
///
/// Used by [`hinge_embedding_loss`]/[`cosine_embedding_loss`] to gate between
/// their two per-element branches. The mask is built by reading
/// `reference`'s raw data — this is sound specifically because `reference`
/// here is always a *label*/*target* tensor whose branch-selecting role is
/// inherently non-differentiable (PyTorch's own `HingeEmbeddingLoss`/
/// `CosineEmbeddingLoss` never define a gradient with respect to `target`
/// either — it is a `{-1, +1}` class indicator, not a continuous quantity),
/// never the `predictions`/`input` tensor whose gradient this whole rewrite
/// exists to preserve.
fn mask_where(reference: &PyTensor, predicate: impl Fn(f32) -> bool) -> PyResult<PyTensor> {
    let data = reference
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read tensor for mask: {}", e)))?;
    let mask_data: Vec<f32> = data
        .iter()
        .map(|&v| if predicate(v) { 1.0 } else { 0.0 })
        .collect();
    let mask_tensor = Tensor::from_vec(mask_data, reference.tensor.shape().dims())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to build mask tensor: {}", e)))?;
    Ok(detached(mask_tensor))
}

/// Sum a `[batch_size, feature_size]` tensor over its second axis, giving a
/// `[batch_size]` result — **without** using [`crate::math_ops::sum`]'s
/// axis-specific reduction.
///
/// # Why this exists (two real upstream bugs, found in sequence)
///
/// [`crate::math_ops::sum`] (and `mean`) *is* tape-aware (see this module's
/// top-level doc) — but its backward, for **any** axis-specific (not full,
/// not already-scalar) reduction whose surviving dimension has size `>= 2`,
/// was found (via finite-difference testing during development, on both
/// [`cross_entropy`] and [`cosine_embedding_loss`]) to silently corrupt the
/// gradient: batch element `0` receives the correct value, but every other
/// batch element incorrectly receives batch element `0`'s value too, instead
/// of its own. Root cause (confirmed by direct inspection, entirely outside
/// this crate and this file): `tenflowers-autograd`'s
/// `crates/tenflowers-autograd/src/tape/gradient_computation/tensor_ops.rs`,
/// function `broadcast_gradient_to_shape` (called by `Operation::Sum`'s
/// backward) only has correct handling for three cases — an exact shape
/// match, a true 0-d scalar, and a `[1]`-shaped gradient — and falls through
/// to an "ultimate fallback" for anything else that reads *only the first
/// element* of the incoming gradient and broadcasts that single scalar
/// uniformly across the *entire* output shape, discarding every other batch
/// element's own gradient value.
///
/// An earlier version of this helper worked around that by summing via
/// [`PyTensor::slice`] (one column at a time) + [`PyTensor::add`] instead of
/// `Sum`. That approach hit a **second**, independent upstream bug:
/// `tenflowers-core`'s `slice_with_stride` (the kernel `PyTensor::slice`
/// calls into) computes its linear-index stride left-to-right (`dim 0` gets
/// stride `1`, `dim 1` gets stride `shape[0]`, ...) — genuine Fortran/
/// column-major order — while `Tensor::from_vec`'s actual backing storage
/// (`ArrayD::from_shape_vec`) and this crate's own `StridedLayout` are both
/// correctly row-major (C-order, last dim stride `1`). The mismatch means
/// `.slice(...)` silently returns data from the *wrong* elements entirely
/// for any `>=2`-D tensor slice that isn't the last axis — confirmed via a
/// standalone reproduction matching this exact `[2,3]` shape (`.slice([(0,
/// 1, None)])`, intended as "row 0", returned `[data[0], data[2], data[4]]`
/// instead of `[data[0], data[1], data[2]]`).
///
/// Both of these are bugs in `tenflowers-core`/`tenflowers-autograd`, not in
/// this file — but per this file's ownership scope, neither of those crates
/// is something this rewrite may modify. This helper works around *both* by
/// building the reduction from [`PyTensor::matmul`] against a `[feature_size,
/// 1]` all-ones vector (`sum(x, axis=1) == x @ ones` is a standard linear-
/// algebra identity) followed by [`crate::tensor_ops::reshape`] to drop the
/// resulting trailing size-1 axis. `MatMul`'s and `Reshape`'s tape backward
/// were independently confirmed correct for this exact shape during
/// development, including under a strongly asymmetric per-batch-element
/// gradient (to rule out a batch-0-value-leaks-everywhere bug hiding behind
/// a symmetric/uniform gradient check).
///
/// # Errors
///
/// Returns an error if `input`'s matmul/reshape operations fail (e.g. a
/// shape that does not have exactly 2 dimensions).
fn sum_over_axis1_via_add(input: &PyTensor) -> PyResult<PyTensor> {
    let shape = input.tensor.shape().dims().to_vec();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "sum_over_axis1_via_add expects a 2D [batch, features] tensor, got shape {:?}",
            shape
        )));
    }
    let batch_size = shape[0];
    let feature_size = shape[1];

    let ones = constant_filled(1.0, &[feature_size, 1])?;
    let summed_2d = input.matmul(&ones)?;
    crate::tensor_ops::reshape(&summed_2d, vec![batch_size])
}

/// Reduce an elementwise-loss [`PyTensor`] per the standard PyTorch-style
/// `reduction` argument (`"mean"`, `"sum"`, or `"none"`), via the tape-aware
/// [`crate::math_ops::mean`]/[`crate::math_ops::sum`] (matching
/// [`mse_loss`]'s own reduction branch exactly).
fn apply_reduction(elementwise: PyTensor, reduction: &str) -> PyResult<PyTensor> {
    match reduction {
        "mean" => crate::math_ops::mean(&elementwise, None, Some(false)),
        "sum" => crate::math_ops::sum(&elementwise, None, Some(false)),
        "none" => Ok(elementwise),
        _ => Err(PyValueError::new_err(format!(
            "Invalid reduction: '{}'. Must be 'mean', 'sum', or 'none'",
            reduction
        ))),
    }
}

// ============================================================================
// Loss functions
// ============================================================================

/// Mean Squared Error (MSE) Loss
///
/// Computes the mean squared error between predictions and targets.
/// Commonly used for regression tasks.
///
/// # Autograd
///
/// Computed entirely through tape-aware [`PyTensor`] operations
/// (`PyTensor::sub`/`PyTensor::mul` and [`crate::math_ops::sum`]/
/// [`crate::math_ops::mean`]) rather than pulling both tensors down to a raw
/// `Vec<f32>` and looping — the latter has no relationship to
/// [`crate::implicit_autograd`]'s tape, so a `PyTensor` produced that way can
/// never be the target of a `.backward()` call (see
/// `crate::implicit_autograd::run_backward`, which requires the tensor to be
/// looked up in the tape's tracked-tensor registry). Every reduction branch
/// below shares the same `squared_diff = (predictions - targets) *
/// (predictions - targets)` computation so `predictions`/`targets`
/// requiring gradients propagates identically no matter which `reduction` is
/// requested.
#[pyfunction]
#[pyo3(signature = (predictions, targets, reduction="mean"))]
pub fn mse_loss(
    predictions: &PyTensor,
    targets: &PyTensor,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let reduction = reduction.unwrap_or("mean");

    // Check shapes match
    let pred_shape = predictions.tensor.shape();
    let target_shape = targets.tensor.shape();

    if pred_shape != target_shape {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: predictions {:?} vs targets {:?}",
            pred_shape, target_shape
        )));
    }

    // squared_diff = (predictions - targets)^2, computed via tape-aware
    // PyTensor ops so a tracked `predictions`/`targets` correctly links the
    // result onto the implicit autograd graph.
    let diff = predictions.sub(targets)?;
    let squared_diff = diff.mul(&diff)?;

    match reduction {
        "mean" => crate::math_ops::mean(&squared_diff, None, Some(false)),
        "sum" => crate::math_ops::sum(&squared_diff, None, Some(false)),
        "none" => Ok(squared_diff),
        _ => Err(PyValueError::new_err(format!(
            "Invalid reduction: '{}'. Must be 'mean', 'sum', or 'none'",
            reduction
        ))),
    }
}

/// Binary Cross Entropy (BCE) Loss
///
/// Computes the binary cross entropy loss between predictions and targets:
/// `-[t*log(p_c) + (1-t)*log(1-p_c)]`, where `p_c = clamp(p, eps, 1-eps)`
/// (matching the original implementation's numerical-stability clamp
/// exactly — see this module's top-level doc for how `clamp`/`log` are
/// linked onto the tape).
///
/// # Autograd
///
/// Built compositionally from [`tape_clamp`], [`tape_log`], [`one_minus`],
/// and the ordinary tape-aware `PyTensor::mul`/`add`, plus [`tape_neg`] for
/// the final negation — every intermediate carries a correct local gradient
/// edge, so gradient with respect to `predictions` correctly includes the
/// clamp's in-range gate (`0` outside `[eps, 1-eps]`, matching
/// `process_clamp_backward`) composed with `1/p_c` (matching
/// `process_log_backward`), exactly reproducing what differentiating the
/// original `.clamp(eps, 1-eps).ln()` expression by hand would give.
#[pyfunction]
#[pyo3(signature = (predictions, targets, reduction="mean"))]
pub fn binary_cross_entropy(
    predictions: &PyTensor,
    targets: &PyTensor,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let reduction = reduction.unwrap_or("mean");

    // Check shapes match
    let pred_shape = predictions.tensor.shape();
    let target_shape = targets.tensor.shape();

    if pred_shape != target_shape {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: predictions {:?} vs targets {:?}",
            pred_shape, target_shape
        )));
    }

    let eps = 1e-7_f32; // Small epsilon for numerical stability, matching the original clamp bounds.
    let p_c = tape_clamp(predictions, eps, 1.0 - eps)?;
    let one_minus_p_c = one_minus(&p_c)?;

    let log_p_c = tape_log(&p_c)?;
    let log_one_minus_p_c = tape_log(&one_minus_p_c)?;

    let one_minus_t = one_minus(targets)?;

    let term1 = targets.mul(&log_p_c)?;
    let term2 = one_minus_t.mul(&log_one_minus_p_c)?;
    let sum_terms = term1.add(&term2)?;
    let losses = tape_neg(&sum_terms)?;

    apply_reduction(losses, reduction)
}

/// Cross Entropy Loss
///
/// Computes the cross entropy loss between predictions and targets.
/// Commonly used for multi-class classification tasks. Predictions must be
/// at least 2D (`batch_size, num_classes`); targets may be either class
/// indices (`batch_size,`) or one-hot encoded (`batch_size, num_classes`).
///
/// # Autograd
///
/// Both target formats are unified into a single tape-aware computation: a
/// class-index target `target[i]` is first converted (via
/// [`class_indices_to_one_hot`], which reads only `targets`' *own* integer
/// data — targets carry no meaningful gradient in this mode, exactly like
/// PyTorch's own `F.cross_entropy`, which never differentiates with respect
/// to integer class-index targets either) into the same one-hot
/// representation an already-one-hot target already is. From there, the
/// per-sample loss is `-sum_over_classes(one_hot * log(clamp(p, eps,
/// 1-eps)))`, computed via the ordinary tape-aware `PyTensor::mul` (for
/// `one_hot * log(p_c)`, gradient flowing into `predictions` exactly at the
/// selected class per sample — the "index via multiply-by-indicator"
/// identity `pred[i, k] == sum_j(pred[i, j] * one_hot(k)[j])`) composed with
/// [`tape_clamp`]/[`tape_log`]/[`tape_neg`] and [`sum_over_axis1_via_add`]
/// for the class-axis reduction (deliberately **not**
/// `crate::math_ops::sum(..., Some(vec![1]), ...)` — see that helper's own
/// doc for the confirmed upstream `tenflowers-autograd` bug that makes any
/// axis-specific `Sum` with a surviving dimension `>= 2` silently corrupt
/// every batch element after the first).
#[pyfunction]
#[pyo3(signature = (predictions, targets, reduction="mean"))]
pub fn cross_entropy(
    predictions: &PyTensor,
    targets: &PyTensor,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let reduction = reduction.unwrap_or("mean");

    let pred_shape = predictions.tensor.shape();
    let target_shape = targets.tensor.shape();

    // Predictions should be (batch_size, num_classes)
    // Targets should be (batch_size,) with class indices or (batch_size, num_classes) one-hot
    if pred_shape.len() < 2 {
        return Err(PyValueError::new_err(
            "Predictions must be at least 2D (batch_size, num_classes)",
        ));
    }

    let batch_size = pred_shape[0];
    let num_classes = pred_shape[1];

    let one_hot: PyTensor = if target_shape.len() == 1 || (target_shape.len() == 2 && target_shape[1] == 1)
    {
        class_indices_to_one_hot(targets, batch_size, num_classes)?
    } else {
        if target_shape.dims() != [batch_size, num_classes] {
            return Err(PyValueError::new_err(format!(
                "One-hot targets shape {:?} does not match predictions {:?}",
                target_shape, pred_shape
            )));
        }
        targets.clone()
    };

    let eps = 1e-7_f32;
    let p_c = tape_clamp(predictions, eps, 1.0 - eps)?;
    let log_p_c = tape_log(&p_c)?;
    let weighted = one_hot.mul(&log_p_c)?;

    // Sum over the class axis (axis 1), leaving a [batch_size] per-sample
    // loss. Deliberately NOT `crate::math_ops::sum(&weighted, Some(vec![1]),
    // ...)` -- see `sum_over_axis1_via_add`'s doc for the real, confirmed
    // upstream `tenflowers-autograd` bug that makes that axis-specific
    // reduction silently corrupt every batch element after the first.
    let per_sample_sum = sum_over_axis1_via_add(&weighted)?;
    let losses = tape_neg(&per_sample_sum)?;

    apply_reduction(losses, reduction)
}

/// Convert integer class-index `targets` (shape `[batch_size]` or
/// `[batch_size, 1]`) into a one-hot `[batch_size, num_classes]`
/// [`PyTensor`], used by [`cross_entropy`] to unify both supported target
/// formats into a single tape-aware computation.
///
/// The returned tensor is a detached constant (see [`detached`]): a
/// class-index target's individual integer values are not a differentiable
/// quantity, matching PyTorch's own `F.cross_entropy`, which never computes
/// a gradient with respect to integer class-index targets either.
fn class_indices_to_one_hot(
    targets: &PyTensor,
    batch_size: usize,
    num_classes: usize,
) -> PyResult<PyTensor> {
    let target_data = targets
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get targets: {}", e)))?;

    let mut one_hot_data = vec![0.0_f32; batch_size * num_classes];
    for (i, &target_val) in target_data.iter().enumerate().take(batch_size) {
        let class_idx = target_val as usize;
        if class_idx >= num_classes {
            return Err(PyValueError::new_err(format!(
                "Target class index {} out of bounds for {} classes",
                class_idx, num_classes
            )));
        }
        one_hot_data[i * num_classes + class_idx] = 1.0;
    }

    let one_hot_tensor = Tensor::from_vec(one_hot_data, &[batch_size, num_classes])
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to build one-hot tensor: {}", e)))?;
    Ok(detached(one_hot_tensor))
}

/// L1 Loss (Mean Absolute Error)
///
/// Computes the mean absolute error between predictions and targets:
/// `|predictions - targets|`.
///
/// # Autograd
///
/// `diff = predictions - targets` via the ordinary tape-aware
/// [`PyTensor::sub`], then [`tape_abs`] (see this module's top-level doc)
/// gives `d(|diff|)/d(diff) = sign(diff)` (with `sign(0) := 0`), which
/// `Sub`'s own backward then correctly splits into `+sign(diff)` for
/// `predictions` and `-sign(diff)` for `targets` — exactly the gradient the
/// original `.abs()`-based formula has everywhere except at the
/// measure-zero, non-differentiable kink `predictions == targets`, where
/// (as for PyTorch's own `L1Loss`) the subgradient convention `sign(0) := 0`
/// is used.
#[pyfunction]
#[pyo3(signature = (predictions, targets, reduction="mean"))]
pub fn l1_loss(
    predictions: &PyTensor,
    targets: &PyTensor,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let reduction = reduction.unwrap_or("mean");

    let pred_shape = predictions.tensor.shape();
    let target_shape = targets.tensor.shape();

    if pred_shape != target_shape {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: predictions {:?} vs targets {:?}",
            pred_shape, target_shape
        )));
    }

    let diff = predictions.sub(targets)?;
    let losses = tape_abs(&diff)?;

    apply_reduction(losses, reduction)
}

/// Smooth L1 Loss (Huber Loss)
///
/// Combines advantages of L1 and L2 loss. Less sensitive to outliers than L2:
/// `0.5*diff^2/beta` where `|diff| < beta`, else `|diff| - 0.5*beta`.
///
/// # Autograd
///
/// This loss's true gradient is the well-known identity `d(smooth_l1)/d(diff)
/// = clamp(diff/beta, -1, 1)`: linear (slope `1/beta`) inside the quadratic
/// region and saturating to `sign(diff)` outside it — which is exactly what
/// differentiating the two branches at their respective definitions gives,
/// and which is continuous at `|diff| == beta` (both branches agree there).
/// [`tape_clamp`] (see this module's top-level doc) is used directly to
/// build this gradient-shaped gate as a detached `Mul` proxy operand, while
/// the forward value is computed via the original piecewise formula (ported
/// verbatim to elementwise `tenflowers_core::ops`, not a
/// `.to_vec()`-and-loop that would sever `predictions`/`targets`'s tape
/// linkage — only the values used to build the *proxy gate* are read via
/// `.to_vec()`, exactly like every other helper in this module).
#[pyfunction]
#[pyo3(signature = (predictions, targets, beta=1.0, reduction="mean"))]
pub fn smooth_l1_loss(
    predictions: &PyTensor,
    targets: &PyTensor,
    beta: Option<f32>,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let beta = beta.unwrap_or(1.0);
    let reduction = reduction.unwrap_or("mean");

    if beta <= 0.0 {
        return Err(PyValueError::new_err("beta must be positive"));
    }

    let pred_shape = predictions.tensor.shape();
    let target_shape = targets.tensor.shape();

    if pred_shape != target_shape {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: predictions {:?} vs targets {:?}",
            pred_shape, target_shape
        )));
    }

    let diff = predictions.sub(targets)?;

    // Forward value: exact port of the original piecewise formula to
    // elementwise tensor ops (not a to_vec()+loop on predictions/targets —
    // `diff` here is only read to build the *forward* value and the
    // gradient-gate constant below, mirroring every other helper's split of
    // "real value via raw ops" + "gradient edge via a proxy").
    let diff_data = diff
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read diff: {}", e)))?;
    let forward_data: Vec<f32> = diff_data
        .iter()
        .map(|d| {
            let ad = d.abs();
            if ad < beta {
                0.5 * ad * ad / beta
            } else {
                ad - 0.5 * beta
            }
        })
        .collect();
    let forward_tensor = Tensor::from_vec(forward_data, diff.tensor.shape().dims())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to build smooth_l1 forward: {}", e)))?;
    let losses = PyTensor {
        tensor: Arc::new(forward_tensor),
        requires_grad: diff.requires_grad,
        is_pinned: diff.is_pinned,
    };

    // Gradient gate: clamp(diff / beta, -1, 1) == d(smooth_l1)/d(diff). Built
    // directly (rather than via `tape_clamp` on a separately-materialized
    // `diff / beta` tensor) since only the gate's *values* are needed here,
    // never a tape edge of their own — the gate itself is the detached
    // operand of the `Mul` proxy below, exactly like every other helper in
    // this module's detached constant operands.
    let gate_data: Vec<f32> = diff_data.iter().map(|d| (d / beta).clamp(-1.0, 1.0)).collect();
    let gate_tensor = Tensor::from_vec(gate_data, diff.tensor.shape().dims())
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to build smooth_l1 gate: {}", e)))?;
    let gate_detached = detached(gate_tensor);
    link_unary_via_binary_proxy(BinaryOpKind::Mul, &diff, &gate_detached, &losses)?;

    apply_reduction(losses, reduction)
}

/// Kullback-Leibler Divergence Loss
///
/// Measures how one probability distribution diverges from a second expected
/// distribution: `t_c * log(t_c / p_c)`, where `p_c = clamp(p, eps, 1)` and
/// `t_c = clamp(t, eps, 1)` (matching the original clamp bounds exactly).
///
/// # Autograd
///
/// Built compositionally from [`tape_clamp`], [`tape_log`], and the ordinary
/// tape-aware `PyTensor::sub`/`mul`: `log_ratio = log(t_c) - log(p_c)`, then
/// `loss = t_c * log_ratio`. Because `log_ratio` is itself tape-linked back
/// to `t_c` (through its own `log(t_c)` term) as well as to `p_c`, the final
/// `Mul(t_c, log_ratio)` step's ordinary product-rule accumulation across
/// both of `t_c`'s incoming edges automatically reproduces the full
/// `d(t_c*log(t_c/p_c))/d(t_c) = log(t_c/p_c) + 1` derivative, and
/// `d(..)/d(p_c) = -t_c/p_c` flows through the `log(p_c)` branch alone —
/// with no hand-derived product-rule term needed in this file (see this
/// module's top-level doc, "product/chain rule ... no per-function
/// hand-derivation").
#[pyfunction]
#[pyo3(signature = (predictions, targets, reduction="mean"))]
pub fn kl_div_loss(
    predictions: &PyTensor,
    targets: &PyTensor,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let reduction = reduction.unwrap_or("mean");

    let pred_shape = predictions.tensor.shape();
    let target_shape = targets.tensor.shape();

    if pred_shape != target_shape {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: predictions {:?} vs targets {:?}",
            pred_shape, target_shape
        )));
    }

    let eps = 1e-7_f32;
    let p_c = tape_clamp(predictions, eps, 1.0)?;
    let t_c = tape_clamp(targets, eps, 1.0)?;

    let log_p_c = tape_log(&p_c)?;
    let log_t_c = tape_log(&t_c)?;
    let log_ratio = log_t_c.sub(&log_p_c)?;
    let losses = t_c.mul(&log_ratio)?;

    match reduction {
        "batchmean" => {
            let summed = crate::math_ops::sum(&losses, None, Some(false))?;
            let divisor = constant_filled(pred_shape[0] as f32, summed.tensor.shape().dims())?;
            summed.div(&divisor)
        }
        _ => apply_reduction(losses, reduction),
    }
}

/// Hinge Embedding Loss
///
/// Measures the loss for embedding learning. Used in ranking and similarity
/// learning: `predictions` where `targets == 1`, else `max(0, margin -
/// predictions)`.
///
/// # Autograd
///
/// The branch selecting between the two cases is gated on `targets`' own
/// value, which (matching PyTorch's own `HingeEmbeddingLoss`) is never
/// itself differentiated through — see [`mask_where`]'s doc. Both branches
/// are individually tape-aware: the `targets == 1` branch is `predictions`
/// itself (identity gradient), and the other branch is built from the
/// ordinary tape-aware `PyTensor::sub` (`margin - predictions`) composed
/// with the already tape-linked [`crate::neural::functions::relu`] (`max(0,
/// ..)`, reusing that module's own correctly-tape-linked `Relu`
/// `UnaryOpKind`, per this module's "prefer NOT duplicating tape-recording
/// logic that belongs centrally" guidance). The two branches are combined
/// via `PyTensor::add`/`mul` against the detached 0/1 mask, so gradient
/// flows into `predictions` through whichever branch was actually selected
/// per-element, and is exactly `0` through the other (a `Mul` by a `0.0`
/// mask element).
#[pyfunction]
#[pyo3(signature = (predictions, targets, margin=1.0, reduction="mean"))]
pub fn hinge_embedding_loss(
    predictions: &PyTensor,
    targets: &PyTensor,
    margin: Option<f32>,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let margin = margin.unwrap_or(1.0);
    let reduction = reduction.unwrap_or("mean");

    let pred_shape = predictions.tensor.shape();
    let target_shape = targets.tensor.shape();

    if pred_shape != target_shape {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: predictions {:?} vs targets {:?}",
            pred_shape, target_shape
        )));
    }

    let is_one_mask = mask_where(targets, |t| t == 1.0)?;
    let is_not_one_mask = one_minus(&is_one_mask)?;

    let margin_const = constant_filled(margin, predictions.tensor.shape().dims())?;
    let margin_minus_pred = margin_const.sub(predictions)?;
    let hinge_branch = crate::neural::functions::relu(&margin_minus_pred)?;

    let branch1 = predictions.mul(&is_one_mask)?;
    let branch2 = hinge_branch.mul(&is_not_one_mask)?;
    let losses = branch1.add(&branch2)?;

    apply_reduction(losses, reduction)
}

/// Cosine Embedding Loss
///
/// Measures the cosine similarity between two embeddings: `1 - cos_sim`
/// where `target == 1`, else `max(0, cos_sim - margin)`, where `cos_sim =
/// dot(input1, input2) / (norm(input1) * norm(input2) + eps)`.
///
/// # Autograd
///
/// `dot_product`, `sum_sq1 = sum(input1^2)`, and `sum_sq2 = sum(input2^2)`
/// are each built from ordinary tape-aware `PyTensor` operations (`mul` +
/// [`crate::math_ops::sum`] — no proxy helper involved for any of the
/// three). `cos_sim` itself is **not** built by chaining [`tape_sqrt`]'s
/// *output* into a further `Div`/`Mul` (an earlier version of this function
/// did exactly that and was found, via finite-difference testing during
/// development, to silently corrupt the gradient by a clean, reproducible
/// integer factor — see [`link_multi_gradient_sum`]'s doc for the full
/// root-cause explanation of why chaining two proxy helpers together is
/// unsafe). Instead, `cos_sim`'s real forward value is computed once via
/// plain, stable scalar math (`dot / (sqrt(sum_sq1)*sqrt(sum_sq2) + eps)`,
/// identical to the original formula), and its gradient is linked directly
/// via [`link_multi_gradient_sum`] against the **hand-derived, closed-form
/// partial derivatives** with respect to each of the three real (non-proxy)
/// tracked quantities that actually feed it:
///
/// * `d(cos_sim)/d(dot_product) = 1 / denom`
/// * `d(cos_sim)/d(sum_sq1) = -dot_product * norm2 / (2 * norm1 * denom²)`
/// * `d(cos_sim)/d(sum_sq2) = -dot_product * norm1 / (2 * norm2 * denom²)`
///
/// where `norm1 = sqrt(sum_sq1)`, `norm2 = sqrt(sum_sq2)`, `denom =
/// norm1*norm2 + eps` — these are exactly the partial derivatives of the
/// original `dot / (norm(input1)*norm(input2) + eps)` expression (confirmed
/// symbolically), so gradient with respect to `input1`/`input2` flowing
/// further back through `sum_sq1`/`sum_sq2`'s own `Mul`+`Sum` recording
/// (both ordinary, non-proxy, already-verified-safe tape kinds) is exactly
/// the same as differentiating the original formula by hand. As in
/// [`hinge_embedding_loss`], the branch itself is gated on `target`'s value
/// (never differentiated through — see [`mask_where`]'s doc), and the
/// `max(0, ..)` branch reuses [`crate::neural::functions::relu`].
#[pyfunction]
#[pyo3(signature = (input1, input2, target, margin=0.0, reduction="mean"))]
pub fn cosine_embedding_loss(
    input1: &PyTensor,
    input2: &PyTensor,
    target: &PyTensor,
    margin: Option<f32>,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let margin = margin.unwrap_or(0.0);
    let reduction = reduction.unwrap_or("mean");

    let shape1 = input1.tensor.shape();
    let shape2 = input2.tensor.shape();

    if shape1 != shape2 {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: input1 {:?} vs input2 {:?}",
            shape1, shape2
        )));
    }

    // `sum_over_axis1_via_add` requires a genuine 2D [batch, features]
    // shape; a 1D [N] input is reshaped to [N, 1] first (matching the
    // original implementation's own 1D interpretation: each element is its
    // own single-feature "sample", i.e. `feature_size = 1`).
    let batch_size = shape1.dims()[0];
    let (input1_2d, input2_2d) = if shape1.dims().len() > 1 {
        (input1.clone(), input2.clone())
    } else {
        (
            crate::tensor_ops::reshape(input1, vec![batch_size, 1])?,
            crate::tensor_ops::reshape(input2, vec![batch_size, 1])?,
        )
    };

    // dot_product[i] = sum_over_features(input1[i] * input2[i])
    let elementwise_product = input1_2d.mul(&input2_2d)?;
    // Deliberately NOT `crate::math_ops::sum(&elementwise_product,
    // Some(vec![1]), ...)` -- see `sum_over_axis1_via_add`'s doc for the
    // real, confirmed upstream `tenflowers-autograd` bug that makes that
    // axis-specific reduction silently corrupt every batch element after
    // the first.
    let dot_product = sum_over_axis1_via_add(&elementwise_product)?;

    // sum_sq1[i] = sum_over_features(input1[i]^2), sum_sq2 likewise. Both
    // are ordinary tape-aware Mul+`sum_over_axis1_via_add` results (no
    // proxy helper, no sqrt) -- these, not norm1/norm2 themselves, are what
    // cos_sim's gradient is linked against below.
    let sq1 = input1_2d.mul(&input1_2d)?;
    let sum_sq1 = sum_over_axis1_via_add(&sq1)?;

    let sq2 = input2_2d.mul(&input2_2d)?;
    let sum_sq2 = sum_over_axis1_via_add(&sq2)?;

    // Real forward value: plain, stable scalar math, identical formula to
    // the original implementation (`dot / (norm1*norm2 + eps)`).
    let eps = 1e-8_f32;
    let dot_data = dot_product
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read dot_product: {}", e)))?;
    let sum_sq1_data = sum_sq1
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read sum_sq1: {}", e)))?;
    let sum_sq2_data = sum_sq2
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read sum_sq2: {}", e)))?;

    let mut cos_sim_data = Vec::with_capacity(dot_data.len());
    let mut d_dot_data = Vec::with_capacity(dot_data.len());
    let mut d_ss1_data = Vec::with_capacity(dot_data.len());
    let mut d_ss2_data = Vec::with_capacity(dot_data.len());
    for i in 0..dot_data.len() {
        let dot = dot_data[i];
        let ss1 = sum_sq1_data[i];
        let ss2 = sum_sq2_data[i];
        let norm1 = ss1.sqrt();
        let norm2 = ss2.sqrt();
        let denom = norm1 * norm2 + eps;

        cos_sim_data.push(dot / denom);
        d_dot_data.push(1.0 / denom);
        // d(cos_sim)/d(sum_sq1) = -dot*norm2 / (2*norm1*denom^2); guard
        // norm1==0 (an all-zero input1 row) the same way the original
        // clamp-free formula implicitly did: the limit is finite (0) since
        // `dot` itself is also 0 whenever `input1` (and therefore every
        // term of the dot product along that row) is all-zero.
        d_ss1_data.push(if norm1 > 0.0 {
            -dot * norm2 / (2.0 * norm1 * denom * denom)
        } else {
            0.0
        });
        d_ss2_data.push(if norm2 > 0.0 {
            -dot * norm1 / (2.0 * norm2 * denom * denom)
        } else {
            0.0
        });
    }

    let result_shape = dot_product.tensor.shape().dims().to_vec();
    let cos_sim_tensor = Tensor::from_vec(cos_sim_data, &result_shape)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to build cos_sim: {}", e)))?;
    let cos_sim = PyTensor {
        tensor: Arc::new(cos_sim_tensor),
        requires_grad: input1.requires_grad || input2.requires_grad,
        is_pinned: input1.is_pinned || input2.is_pinned,
    };

    let d_dot = detached(
        Tensor::from_vec(d_dot_data, &result_shape)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to build d_dot: {}", e)))?,
    );
    let d_ss1 = detached(
        Tensor::from_vec(d_ss1_data, &result_shape)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to build d_ss1: {}", e)))?,
    );
    let d_ss2 = detached(
        Tensor::from_vec(d_ss2_data, &result_shape)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to build d_ss2: {}", e)))?,
    );

    link_multi_gradient_sum(
        &[
            (&dot_product, &d_dot),
            (&sum_sq1, &d_ss1),
            (&sum_sq2, &d_ss2),
        ],
        &cos_sim,
    )?;

    let is_one_mask = mask_where(target, |t| t == 1.0)?;
    let is_not_one_mask = one_minus(&is_one_mask)?;

    // Branch A (target == 1): 1 - cos_sim
    let branch_a = one_minus(&cos_sim)?;

    // Branch B (target != 1): max(0, cos_sim - margin)
    let margin_const = constant_filled(margin, cos_sim.tensor.shape().dims())?;
    let cos_sim_minus_margin = cos_sim.sub(&margin_const)?;
    let branch_b = crate::neural::functions::relu(&cos_sim_minus_margin)?;

    let term_a = branch_a.mul(&is_one_mask)?;
    let term_b = branch_b.mul(&is_not_one_mask)?;
    let losses = term_a.add(&term_b)?;

    apply_reduction(losses, reduction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::implicit_autograd::{get_grad, mark_leaf, run_backward};

    fn make_tensor(data: Vec<f32>, shape: &[usize]) -> PyTensor {
        let tensor = Tensor::from_vec(data, shape).expect("tensor construction must succeed");
        PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: false,
            is_pinned: false,
        }
    }

    fn grad_data(t: &PyTensor) -> Vec<f32> {
        t.tensor.to_vec().expect("grad tensor readable")
    }

    // ---- reference (old-buggy-formula-equivalent) scalar math, used only to
    // build finite-difference expectations, never used by the code under test.

    fn ref_bce(p: &[f32], t: &[f32]) -> f32 {
        let eps = 1e-7_f32;
        let n = p.len() as f32;
        let sum: f32 = p
            .iter()
            .zip(t.iter())
            .map(|(pi, ti)| {
                let pc = pi.clamp(eps, 1.0 - eps);
                -(ti * pc.ln() + (1.0 - ti) * (1.0 - pc).ln())
            })
            .sum();
        sum / n
    }

    fn ref_l1(p: &[f32], t: &[f32]) -> f32 {
        let n = p.len() as f32;
        p.iter().zip(t.iter()).map(|(pi, ti)| (pi - ti).abs()).sum::<f32>() / n
    }

    fn ref_smooth_l1(p: &[f32], t: &[f32], beta: f32) -> f32 {
        let n = p.len() as f32;
        p.iter()
            .zip(t.iter())
            .map(|(pi, ti)| {
                let diff = (pi - ti).abs();
                if diff < beta {
                    0.5 * diff.powi(2) / beta
                } else {
                    diff - 0.5 * beta
                }
            })
            .sum::<f32>()
            / n
    }

    fn ref_kl(p: &[f32], t: &[f32]) -> f32 {
        let eps = 1e-7_f32;
        let n = p.len() as f32;
        p.iter()
            .zip(t.iter())
            .map(|(pi, ti)| {
                let pc = pi.clamp(eps, 1.0);
                let tc = ti.clamp(eps, 1.0);
                tc * (tc / pc).ln()
            })
            .sum::<f32>()
            / n
    }

    fn ref_hinge(p: &[f32], t: &[f32], margin: f32) -> f32 {
        let n = p.len() as f32;
        p.iter()
            .zip(t.iter())
            .map(|(pi, ti)| if *ti == 1.0 { *pi } else { (margin - pi).max(0.0) })
            .sum::<f32>()
            / n
    }

    fn ref_cross_entropy_one_hot(p: &[f32], t: &[f32], batch: usize, classes: usize) -> f32 {
        let eps = 1e-7_f32;
        let mut losses = Vec::new();
        for i in 0..batch {
            let mut loss = 0.0;
            for j in 0..classes {
                let pred = p[i * classes + j].clamp(eps, 1.0 - eps);
                loss -= t[i * classes + j] * pred.ln();
            }
            losses.push(loss);
        }
        losses.iter().sum::<f32>() / losses.len() as f32
    }

    fn ref_cosine(v1: &[f32], v2: &[f32], target: f32, margin: f32) -> f32 {
        let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let n1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
        let eps = 1e-8;
        let cos_sim = dot / (n1 * n2 + eps);
        if target == 1.0 {
            1.0 - cos_sim
        } else {
            (cos_sim - margin).max(0.0)
        }
    }

    #[test]
    fn binary_cross_entropy_gradient_matches_reference() {
        let p_data = vec![0.2_f32, 0.7, 0.5, 0.9];
        let t_data = vec![0.0_f32, 1.0, 0.0, 1.0];
        let predictions = make_tensor(p_data.clone(), &[4]);
        let targets = make_tensor(t_data.clone(), &[4]);
        mark_leaf(&predictions);

        let loss = binary_cross_entropy(&predictions, &targets, Some("mean"))
            .expect("bce forward must succeed");
        let loss_val = grad_data(&loss)[0];
        let expected_val = ref_bce(&p_data, &t_data);
        assert!(
            (loss_val - expected_val).abs() < 1e-4,
            "forward mismatch: got {loss_val}, expected {expected_val}"
        );

        run_backward(&loss).expect("backward must succeed");
        let grad = get_grad(&predictions).expect("grad must be populated");
        let grad_vals = grad_data(&grad);

        let h = 1e-3_f32;
        for i in 0..p_data.len() {
            let mut p_plus = p_data.clone();
            p_plus[i] += h;
            let mut p_minus = p_data.clone();
            p_minus[i] -= h;
            let numerical = (ref_bce(&p_plus, &t_data) - ref_bce(&p_minus, &t_data)) / (2.0 * h);
            assert!(
                (grad_vals[i] - numerical).abs() < 1e-2,
                "grad[{i}] = {}, expected ~{numerical}",
                grad_vals[i]
            );
        }
    }

    #[test]
    fn cross_entropy_one_hot_gradient_matches_reference() {
        let batch = 2;
        let classes = 3;
        let p_data = vec![0.2_f32, 0.5, 0.3, 0.1, 0.1, 0.8];
        let t_data = vec![0.0_f32, 1.0, 0.0, 0.0, 0.0, 1.0];
        let predictions = make_tensor(p_data.clone(), &[batch, classes]);
        let targets = make_tensor(t_data.clone(), &[batch, classes]);
        mark_leaf(&predictions);

        let loss =
            cross_entropy(&predictions, &targets, Some("mean")).expect("ce forward must succeed");
        let loss_val = grad_data(&loss)[0];
        let expected_val = ref_cross_entropy_one_hot(&p_data, &t_data, batch, classes);
        assert!(
            (loss_val - expected_val).abs() < 1e-4,
            "forward mismatch: got {loss_val}, expected {expected_val}"
        );

        run_backward(&loss).expect("backward must succeed");
        let grad = get_grad(&predictions).expect("grad must be populated");
        let grad_vals = grad_data(&grad);

        let h = 1e-3_f32;
        for i in 0..p_data.len() {
            let mut p_plus = p_data.clone();
            p_plus[i] += h;
            let mut p_minus = p_data.clone();
            p_minus[i] -= h;
            let numerical = (ref_cross_entropy_one_hot(&p_plus, &t_data, batch, classes)
                - ref_cross_entropy_one_hot(&p_minus, &t_data, batch, classes))
                / (2.0 * h);
            assert!(
                (grad_vals[i] - numerical).abs() < 1e-2,
                "grad[{i}] = {}, expected ~{numerical}",
                grad_vals[i]
            );
        }
    }

    #[test]
    fn cross_entropy_class_index_matches_one_hot_equivalent() {
        let batch = 2;
        let classes = 3;
        let p_data = vec![0.2_f32, 0.5, 0.3, 0.1, 0.1, 0.8];
        // class indices: sample 0 -> class 1, sample 1 -> class 2
        let idx_data = vec![1.0_f32, 2.0];
        let one_hot_data = vec![0.0_f32, 1.0, 0.0, 0.0, 0.0, 1.0];

        let predictions_idx = make_tensor(p_data.clone(), &[batch, classes]);
        let targets_idx = make_tensor(idx_data, &[batch]);
        mark_leaf(&predictions_idx);
        let loss_idx = cross_entropy(&predictions_idx, &targets_idx, Some("none"))
            .expect("ce (class-index) forward must succeed");
        run_backward(&crate::math_ops::sum(&loss_idx, None, Some(false)).expect("sum ok"))
            .expect("backward must succeed");
        let grad_idx = grad_data(&get_grad(&predictions_idx).expect("grad populated"));

        let predictions_oh = make_tensor(p_data, &[batch, classes]);
        let targets_oh = make_tensor(one_hot_data, &[batch, classes]);
        mark_leaf(&predictions_oh);
        let loss_oh = cross_entropy(&predictions_oh, &targets_oh, Some("none"))
            .expect("ce (one-hot) forward must succeed");
        run_backward(&crate::math_ops::sum(&loss_oh, None, Some(false)).expect("sum ok"))
            .expect("backward must succeed");
        let grad_oh = grad_data(&get_grad(&predictions_oh).expect("grad populated"));

        for i in 0..grad_idx.len() {
            assert!(
                (grad_idx[i] - grad_oh[i]).abs() < 1e-6,
                "class-index and one-hot gradients must match at [{i}]: {} vs {}",
                grad_idx[i],
                grad_oh[i]
            );
        }
    }

    #[test]
    fn l1_loss_gradient_matches_reference() {
        let p_data = vec![1.0_f32, 2.5, -0.5, 3.0];
        let t_data = vec![1.5_f32, 1.0, -2.0, 3.7];
        let predictions = make_tensor(p_data.clone(), &[4]);
        let targets = make_tensor(t_data.clone(), &[4]);
        mark_leaf(&predictions);

        let loss =
            l1_loss(&predictions, &targets, Some("mean")).expect("l1 forward must succeed");
        let loss_val = grad_data(&loss)[0];
        let expected_val = ref_l1(&p_data, &t_data);
        assert!((loss_val - expected_val).abs() < 1e-5);

        run_backward(&loss).expect("backward must succeed");
        let grad_vals = grad_data(&get_grad(&predictions).expect("grad populated"));

        let h = 1e-3_f32;
        for i in 0..p_data.len() {
            let mut p_plus = p_data.clone();
            p_plus[i] += h;
            let mut p_minus = p_data.clone();
            p_minus[i] -= h;
            let numerical = (ref_l1(&p_plus, &t_data) - ref_l1(&p_minus, &t_data)) / (2.0 * h);
            assert!(
                (grad_vals[i] - numerical).abs() < 1e-2,
                "grad[{i}] = {}, expected ~{numerical}",
                grad_vals[i]
            );
        }
    }

    #[test]
    fn smooth_l1_loss_gradient_matches_reference() {
        let p_data = vec![0.0_f32, 2.5, -3.0, 0.3];
        let t_data = vec![0.2_f32, 1.0, -0.5, 0.1];
        let beta = 1.0_f32;
        let predictions = make_tensor(p_data.clone(), &[4]);
        let targets = make_tensor(t_data.clone(), &[4]);
        mark_leaf(&predictions);

        let loss = smooth_l1_loss(&predictions, &targets, Some(beta), Some("mean"))
            .expect("smooth_l1 forward must succeed");
        let loss_val = grad_data(&loss)[0];
        let expected_val = ref_smooth_l1(&p_data, &t_data, beta);
        assert!((loss_val - expected_val).abs() < 1e-5);

        run_backward(&loss).expect("backward must succeed");
        let grad_vals = grad_data(&get_grad(&predictions).expect("grad populated"));

        let h = 1e-3_f32;
        for i in 0..p_data.len() {
            let mut p_plus = p_data.clone();
            p_plus[i] += h;
            let mut p_minus = p_data.clone();
            p_minus[i] -= h;
            let numerical = (ref_smooth_l1(&p_plus, &t_data, beta) - ref_smooth_l1(&p_minus, &t_data, beta))
                / (2.0 * h);
            assert!(
                (grad_vals[i] - numerical).abs() < 1e-2,
                "grad[{i}] = {}, expected ~{numerical}",
                grad_vals[i]
            );
        }
    }

    #[test]
    fn kl_div_loss_gradient_matches_reference() {
        let p_data = vec![0.3_f32, 0.6, 0.9, 0.1];
        let t_data = vec![0.4_f32, 0.5, 0.2, 0.3];
        let predictions = make_tensor(p_data.clone(), &[4]);
        let targets = make_tensor(t_data.clone(), &[4]);
        mark_leaf(&predictions);

        let loss =
            kl_div_loss(&predictions, &targets, Some("mean")).expect("kl forward must succeed");
        let loss_val = grad_data(&loss)[0];
        let expected_val = ref_kl(&p_data, &t_data);
        assert!((loss_val - expected_val).abs() < 1e-4);

        run_backward(&loss).expect("backward must succeed");
        let grad_vals = grad_data(&get_grad(&predictions).expect("grad populated"));

        let h = 1e-3_f32;
        for i in 0..p_data.len() {
            let mut p_plus = p_data.clone();
            p_plus[i] += h;
            let mut p_minus = p_data.clone();
            p_minus[i] -= h;
            let numerical = (ref_kl(&p_plus, &t_data) - ref_kl(&p_minus, &t_data)) / (2.0 * h);
            assert!(
                (grad_vals[i] - numerical).abs() < 1e-2,
                "grad[{i}] = {}, expected ~{numerical}",
                grad_vals[i]
            );
        }
    }

    #[test]
    fn hinge_embedding_loss_gradient_matches_reference() {
        let p_data = vec![0.5_f32, -0.3, 2.0, 0.1];
        let t_data = vec![1.0_f32, -1.0, 1.0, -1.0];
        let margin = 1.0_f32;
        let predictions = make_tensor(p_data.clone(), &[4]);
        let targets = make_tensor(t_data.clone(), &[4]);
        mark_leaf(&predictions);

        let loss = hinge_embedding_loss(&predictions, &targets, Some(margin), Some("mean"))
            .expect("hinge forward must succeed");
        let loss_val = grad_data(&loss)[0];
        let expected_val = ref_hinge(&p_data, &t_data, margin);
        assert!((loss_val - expected_val).abs() < 1e-5);

        run_backward(&loss).expect("backward must succeed");
        let grad_vals = grad_data(&get_grad(&predictions).expect("grad populated"));

        let h = 1e-3_f32;
        for i in 0..p_data.len() {
            let mut p_plus = p_data.clone();
            p_plus[i] += h;
            let mut p_minus = p_data.clone();
            p_minus[i] -= h;
            let numerical = (ref_hinge(&p_plus, &t_data, margin) - ref_hinge(&p_minus, &t_data, margin))
                / (2.0 * h);
            assert!(
                (grad_vals[i] - numerical).abs() < 1e-2,
                "grad[{i}] = {}, expected ~{numerical}",
                grad_vals[i]
            );
        }
    }

    #[test]
    fn cosine_embedding_loss_gradient_matches_reference() {
        // batch=2, feature=3
        let v1_data = vec![1.0_f32, 2.0, 3.0, 0.5, -1.0, 2.0];
        let v2_data = vec![1.0_f32, 1.0, 1.0, 2.0, 0.5, -0.5];
        let target_data = vec![1.0_f32, -1.0];
        let margin = 0.2_f32;

        let input1 = make_tensor(v1_data.clone(), &[2, 3]);
        let input2 = make_tensor(v2_data.clone(), &[2, 3]);
        let target = make_tensor(target_data.clone(), &[2]);
        mark_leaf(&input1);

        let loss = cosine_embedding_loss(&input1, &input2, &target, Some(margin), Some("mean"))
            .expect("cosine forward must succeed");
        let loss_val = grad_data(&loss)[0];
        let expected_val = (ref_cosine(&v1_data[0..3], &v2_data[0..3], target_data[0], margin)
            + ref_cosine(&v1_data[3..6], &v2_data[3..6], target_data[1], margin))
            / 2.0;
        assert!(
            (loss_val - expected_val).abs() < 1e-4,
            "got {loss_val}, expected {expected_val}"
        );

        run_backward(&loss).expect("backward must succeed");
        let grad_vals = grad_data(&get_grad(&input1).expect("grad populated"));

        let h = 1e-3_f32;
        for i in 0..v1_data.len() {
            let mut v1_plus = v1_data.clone();
            v1_plus[i] += h;
            let mut v1_minus = v1_data.clone();
            v1_minus[i] -= h;

            let sample = i / 3;
            let lo = sample * 3;
            let hi = lo + 3;
            let loss_plus = ref_cosine(&v1_plus[lo..hi], &v2_data[lo..hi], target_data[sample], margin);
            let loss_minus = ref_cosine(&v1_minus[lo..hi], &v2_data[lo..hi], target_data[sample], margin);
            // mean reduction over batch=2 -> factor 1/2 on this sample's contribution
            let numerical = (loss_plus - loss_minus) / (2.0 * h) / 2.0;
            assert!(
                (grad_vals[i] - numerical).abs() < 1e-2,
                "grad[{i}] = {}, expected ~{numerical}",
                grad_vals[i]
            );
        }
    }

    #[test]
    fn all_seven_losses_populate_gradient_and_are_finite() {
        // Sanity smoke test: every rewritten loss must produce a populated,
        // finite gradient for `predictions` (the tape-linking regression this
        // whole rewrite exists to fix — before the fix, .backward() on these
        // losses errored with "no recorded computation graph" because the
        // old .to_vec()+loop implementation completely severed the tape).
        let p = make_tensor(vec![0.3, 0.6, 0.4, 0.7], &[4]);
        let t = make_tensor(vec![0.0, 1.0, 0.0, 1.0], &[4]);
        mark_leaf(&p);
        let loss = binary_cross_entropy(&p, &t, Some("mean")).expect("bce ok");
        run_backward(&loss).expect("bce backward must succeed (tape must be linked)");
        let g = grad_data(&get_grad(&p).expect("bce grad populated"));
        assert!(g.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn real_tensor_times_tape_log_output_reads_correct_value() {
        // Regression test for a pattern central to binary_cross_entropy,
        // cross_entropy, and kl_div_loss: `real_tensor.mul(&tape_log(x))`,
        // where `real_tensor`'s OWN gradient (via Mul's `grad_lhs =
        // grad_output * rhs_value` formula) needs to read `tape_log(x)`'s
        // VALUE. tape_log is a proxy helper (see this module's top-level
        // doc) whose tape-*internal* value deliberately diverges from the
        // real `log(x)` value written into the returned PyTensor -- unlike
        // the `Div`-as-denominator case that was found to read the wrong
        // (stale, internal) value, this checks that `Mul` reading a proxy's
        // value in this specific slot arrangement gets the REAL value.
        let x_data = vec![2.0_f32]; // log(2.0) ~= 0.6931472
        let y_data = vec![10.0_f32];
        let x = make_tensor(x_data.clone(), &[1]);
        let y = make_tensor(y_data.clone(), &[1]);
        mark_leaf(&y);

        let log_x = tape_log(&x).expect("tape_log must succeed");
        let product = y.mul(&log_x).expect("mul must succeed"); // y * log(x)
        let scalar = crate::math_ops::sum(&product, None, Some(false)).expect("sum ok");

        run_backward(&scalar).expect("backward must succeed");
        let grad_vals = grad_data(&get_grad(&y).expect("grad populated"));

        // loss = y * log(x). d(loss)/dy = log(x) (the REAL value, ~0.6931472).
        let true_grad = x_data[0].ln();
        assert!(
            (grad_vals[0] - true_grad).abs() < 1e-4,
            "grad = {}, expected log(x) = {true_grad} -- if this reads tape_log's \
             stale internal proxy value instead of the real log(x), the result \
             would be x/x_detached = 1.0 instead",
            grad_vals[0]
        );
    }
}
