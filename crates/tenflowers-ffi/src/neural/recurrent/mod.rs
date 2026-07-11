//! Recurrent layers module for TenfloweRS FFI
//!
//! This module provides recurrent layer implementations including LSTM, GRU, and
//! vanilla RNN for sequence modeling, time series prediction, and NLP tasks:
//!
//! | Sub-module | Contents |
//! |---|---|
//! | [`lstm`] | [`lstm::PyLSTM`] (multi-layer, optionally bidirectional) and [`lstm::PyLSTMCell`] (single time-step) |
//! | [`gru`]  | [`gru::PyGRU`] (multi-layer, optionally bidirectional) and [`gru::PyGRUCell`] (single time-step) |
//! | [`rnn`]  | [`rnn::PyRNN`] (multi-layer Elman RNN, tanh/relu, optionally bidirectional) |
//!
//! # Autograd
//!
//! Every learnable weight/bias in this module is held as a
//! [`Py<`super::layers::PyParameter`>`] — the same stable-identity,
//! mutable-in-place parameter cell [`super::layers::PyDense`] and every layer
//! in [`super::normalization`] use (see [`super::layers::PyParameter`]'s own
//! doc for why a plain `Arc<Tensor<f32>>` cannot support this, and
//! [`super::normalization`]'s module doc for the `impl Clone`-via-`clone_ref`
//! pattern every struct here follows verbatim). The multi-layer structs
//! ([`lstm::PyLSTM`], [`gru::PyGRU`], [`rnn::PyRNN`]) hold one
//! `Py<PyParameter>` **per layer, per weight kind** (`weight_ih`/`weight_hh`/
//! optionally `bias_ih`/`bias_hh`), doubled into a second parallel set when
//! `bidirectional` is set — see each struct's own field docs for the exact
//! list, which mirrors the per-layer `Vec<Tensor<T>>` storage
//! `tenflowers_neural::layers::rnn::{LSTM, GRU, RNN}` itself uses internally
//! (confirmed by reading those types directly: `weight_ih: Vec<Tensor<T>>`,
//! one entry per layer, plus `*_reverse: Option<Vec<Tensor<T>>>` for the
//! bidirectional reverse direction).
//!
//! Unlike the pre-autograd version of this module, **no struct here holds a
//! backing `tenflowers_neural::layers::rnn::{LSTM,GRU,RNN}` "inner" layer**.
//! That field existed only so `forward()` could delegate to
//! `Layer::forward`/`forward_with_hidden`, whose raw `Tensor<f32>` arithmetic
//! has no relationship to the implicit tape at all — exactly the gap this
//! module closes. Every `forward()` below computes the *entire* recurrence
//! (every layer, every time step, both directions when bidirectional) purely
//! from tape-aware [`super::tensor_ops::PyTensor`] operations
//! (`matmul`/`add`/`sub`/`mul`/`slice`, plus the tape-aware
//! [`super::functions::sigmoid`]/[`super::functions::tanh`]/[`super::functions::relu`]
//! free functions) and the tape-aware [`stack_on_tape`]/[`concat_on_tape`]
//! helpers below — each of these self-records onto
//! [`crate::implicit_autograd`]'s implicit tape internally, so nothing in
//! this module ever calls `record_and_link_*` directly for elementwise/matmul
//! math. The one exception is the [`stack_on_tape`]/[`concat_on_tape`]
//! helpers themselves, which *do* call
//! [`crate::implicit_autograd::record_and_link_variadic`] directly, because
//! stacking/concatenating a `Vec<PyTensor>` collected one time step at a time
//! has no single-call `PyTensor` method to delegate to (see those helpers'
//! own docs).
//!
//! # Gate conventions (unchanged from the pre-autograd version)
//!
//! * LSTM: fused gate dimension is `4 * hidden_size`, order `[input, forget,
//!   cell, output]`.
//! * GRU: fused gate dimension is `3 * hidden_size`, order `[reset, update,
//!   new]` (PyTorch convention); the update step is
//!   `h_new = (1 - z) * n + z * h`.
//! * RNN: no gate fusion — a single `hidden_size`-wide pre-activation per
//!   time step, passed through `tanh` or `relu` depending on
//!   [`rnn::PyRNN`]'s `nonlinearity` field.
//!
//! # Zero- vs. randn-initialisation (preserved exactly, not "fixed")
//!
//! [`lstm::PyLSTM`]/[`gru::PyGRU`]/[`rnn::PyRNN`]'s multi-layer weights are
//! **zero-initialised** (`Tensor::zeros`), matching the pre-autograd
//! version's `new()`/`reset_parameters()` bodies exactly — this looks
//! surprising for a trainable weight (zero gradients out of a matmul branch
//! in general), but is deliberately preserved rather than "fixed" into a
//! proper randomised init, since that would be an unrequested behaviour
//! change out of scope for this migration. [`lstm::PyLSTMCell`]/
//! [`gru::PyGRUCell`] use a proper `1/sqrt(hidden_size)`-scaled
//! `Tensor::randn` init, also preserved exactly.

mod gru;
mod lstm;
mod rnn;
#[cfg(test)]
mod tests;

pub use gru::{PyGRU, PyGRUCell};
pub use lstm::{PyLSTM, PyLSTMCell};
pub use rnn::PyRNN;

use super::layers::PyParameter;
use crate::tensor_ops::PyTensor;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;
use tenflowers_core::Tensor;
use tenflowers_neural::layers::rnn::RnnNonlinearity;

/// One layer's freshly-constructed `(weight_ih, weight_hh, bias_ih,
/// bias_hh)` parameter set — the return type every `new_layer_params`
/// helper in [`lstm`]/[`gru`]/[`rnn`] shares (only the fused gate dimension
/// inside each `Py<PyParameter>`'s tensor shape differs between them: `4 *
/// hidden_size` for LSTM, `3 * hidden_size` for GRU, plain `hidden_size` for
/// RNN — the parameter *count* and *nesting shape* are identical). Named
/// here, once, rather than repeating the same four-tuple type (which
/// clippy's `type_complexity` lint flags as "very complex" past a certain
/// nesting depth) in three separate files.
pub(super) type LayerParamSet = (
    Py<PyParameter>,
    Py<PyParameter>,
    Option<Py<PyParameter>>,
    Option<Py<PyParameter>>,
);

/// Extract a single time-step slice `[batch, feature]` from a 3-D sequence
/// [`PyTensor`], via the tape-aware [`PyTensor::slice`] (so gradients flow
/// back into the correct time step of the original sequence).
///
/// Mirrors the pre-autograd free function of the same name, but operates on
/// [`PyTensor`] (tape-aware) rather than a raw `Tensor<f32>`.
pub(super) fn sequence_timestep(
    input: &PyTensor,
    t: usize,
    batch: usize,
    feature: usize,
    batch_first: bool,
) -> PyResult<PyTensor> {
    let sliced = if batch_first {
        input.slice(vec![
            (Some(0), Some(batch as isize), None),
            (Some(t as isize), Some((t + 1) as isize), None),
            (Some(0), Some(feature as isize), None),
        ])?
    } else {
        input.slice(vec![
            (Some(t as isize), Some((t + 1) as isize), None),
            (Some(0), Some(batch as isize), None),
            (Some(0), Some(feature as isize), None),
        ])?
    };

    // The slice above keeps a length-1 axis at the sliced time position
    // (either axis 0 or axis 1); reshape it away via the tape-aware
    // `crate::tensor_ops::reshape` free function (`PyTensor::reshape` itself
    // is a private `#[pymethods]` method, not directly callable from Rust —
    // this free function is its public wrapper, exposed as the `tf.reshape`
    // Python builtin) so the result is exactly `[batch, feature]`, matching
    // what every gate matmul below expects, while keeping the reshape itself
    // recorded on the tape.
    crate::tensor_ops::reshape(&sliced, vec![batch, feature])
}

/// Slice gate `g` (0-indexed, out of `num_gates` total) out of a fused
/// `[batch, num_gates * hidden_size]` gate tensor, via the tape-aware
/// [`PyTensor::slice`].
///
/// E.g. for LSTM (`num_gates = 4`, order `[input, forget, cell, output]`),
/// `gate_slice(gates, 1, 5, 4)` extracts the forget gate's `[batch, 5]` slice.
pub(super) fn gate_slice(
    fused: &PyTensor,
    g: usize,
    hidden_size: usize,
    _num_gates: usize,
) -> PyResult<PyTensor> {
    fused.slice(vec![
        (None, None, None),
        (
            Some((g * hidden_size) as isize),
            Some(((g + 1) * hidden_size) as isize),
            None,
        ),
    ])
}

/// Snapshot `param`'s current value and (re-)register it as this forward
/// pass's tape leaf, keyed by its own stable `id()`. Idempotent across
/// repeated calls within a forward/backward cycle (see
/// [`crate::implicit_autograd::mark_leaf_param`]'s own doc). Shared by every
/// struct in this module — see e.g. [`super::layers::PyDense::forward`] for
/// the same pattern spelled out inline (this helper simply factors that
/// three-line dance into one call, since every struct here needs it many
/// times per `forward()` — once per layer per weight kind).
pub(super) fn snapshot_leaf(py: Python<'_>, param: &Py<PyParameter>) -> PyResult<PyTensor> {
    let borrowed = param.borrow(py);
    let snapshot = borrowed.to_tensor()?;
    crate::implicit_autograd::mark_leaf_param(&snapshot, borrowed.id());
    Ok(snapshot)
}

/// Index into a per-layer reverse-direction parameter list
/// (`Option<Vec<Py<PyParameter>>>`, as every bidirectional-capable struct in
/// this module stores its `*_reverse` fields), returning a `PyResult` rather
/// than panicking.
///
/// Every call site in `gru.rs`/`rnn.rs` only ever reaches this helper after
/// already checking `self.bidirectional` (for `weight_ih_reverse`/
/// `weight_hh_reverse`) or `self.bidirectional && self.bias` (for
/// `bias_ih_reverse`/`bias_hh_reverse`) is true — under either of those
/// conditions `new()`/`reset_parameters()` guarantee the `Option` is `Some`
/// with exactly `num_layers` entries, so this can never actually observe
/// `None` or an out-of-range `layer_idx` in practice. It is still written as
/// a fallible `PyResult` lookup (an `.ok_or_else`/`.get` chain) rather than
/// `.unwrap()`/`.expect()`/direct indexing, so a violation of that invariant
/// — e.g. from a future refactor — surfaces as a catchable `PyRuntimeError`
/// instead of a hard panic that could poison the whole embedding Python
/// process, matching this crate's no-`unwrap()`/`expect()`-in-production
/// policy.
pub(super) fn reverse_layer_param<'a>(
    reverse: &'a Option<Vec<Py<PyParameter>>>,
    layer_idx: usize,
    what: &str,
) -> PyResult<&'a Py<PyParameter>> {
    reverse
        .as_ref()
        .and_then(|v| v.get(layer_idx))
        .ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "internal error: missing bidirectional reverse-direction {what} for layer \
                 {layer_idx}"
            ))
        })
}

/// Reshape `PyTensor` `input`'s slice of the `[num_layers (* directions),
/// batch, hidden_size]` initial-state tensor down to a plain `[batch,
/// hidden_size]` for layer `layer_idx`, direction `direction_idx` (0 for
/// forward, 1 for reverse) out of `num_directions` total directions. Used to
/// thread an explicit `h_0`/`c_0` into a specific layer/direction's starting
/// hidden/cell state.
pub(super) fn slice_initial_state(
    state: &PyTensor,
    layer_idx: usize,
    direction_idx: usize,
    num_directions: usize,
    batch: usize,
    hidden_size: usize,
) -> PyResult<PyTensor> {
    let row = layer_idx * num_directions + direction_idx;
    let sliced = state.slice(vec![
        (Some(row as isize), Some((row + 1) as isize), None),
        (Some(0), Some(batch as isize), None),
        (Some(0), Some(hidden_size as isize), None),
    ])?;
    crate::tensor_ops::reshape(&sliced, vec![batch, hidden_size])
}

/// Build an untracked `[batch, hidden_size]` tensor of zeros, used as the
/// default initial hidden/cell state when no explicit `h_0`/`c_0` is
/// provided. `requires_grad: false` since an all-zero initial state should
/// never itself receive a gradient (nothing downstream should ever call
/// `.set_requires_grad(true)` on it, matching the analogous constant-`ones`
/// tensor built inline in `gru.rs`'s `h_new = (1 - z) * n + z * h` step).
pub(super) fn zeros_state(batch: usize, hidden_size: usize) -> PyTensor {
    PyTensor {
        tensor: Arc::new(Tensor::zeros(&[batch, hidden_size])),
        requires_grad: false,
        is_pinned: false,
    }
}

/// Stack `inputs` (typically one `[batch, hidden_size]` `PyTensor` per time
/// step) along a new axis `axis`, recording the stack on the implicit tape
/// via [`crate::implicit_autograd::record_and_link_variadic`] so gradients
/// flow back to each individual time step's contribution.
///
/// This is the "key unblock" the migration plan calls out: there is no
/// single-call, self-recording `PyTensor::stack` method (unlike
/// `matmul`/`add`/`slice`/etc, which each record themselves internally), so
/// this helper does the two-step "compute raw, then record" dance the
/// `record_and_link_*` family is designed for, directly on a plain `&[&Tensor<f32>]`
/// slice built from each input's own `.tensor` field — deliberately NOT going
/// through the `#[pyfunction] stack` in `crate::math_ops` (that overload
/// exists only for Python callers building a literal Python list).
///
/// # Errors
///
/// Propagates any error from the underlying `tenflowers_core::ops::stack`
/// call (e.g. mismatched shapes) or from tape recording.
pub(super) fn stack_on_tape(inputs: &[PyTensor], axis: usize) -> PyResult<PyTensor> {
    let raw_refs: Vec<&Tensor<f32>> = inputs.iter().map(|t| t.tensor.as_ref()).collect();
    let stacked_raw = tenflowers_core::ops::stack(&raw_refs, axis)
        .map_err(|e| PyRuntimeError::new_err(format!("recurrent stack failed: {e}")))?;
    let stacked = PyTensor {
        tensor: Arc::new(stacked_raw),
        requires_grad: true,
        is_pinned: false,
    };
    let py_refs: Vec<&PyTensor> = inputs.iter().collect();
    crate::implicit_autograd::record_and_link_variadic(
        crate::implicit_autograd::VariadicOpKind::Stack { axis: axis as i32 },
        &py_refs,
        &stacked,
    )?;
    Ok(stacked)
}

/// Concatenate `inputs` along an existing axis `axis`, recording the concat
/// on the implicit tape. Counterpart of [`stack_on_tape`]; used to combine a
/// bidirectional layer's forward- and reverse-direction output sequences
/// along the feature axis (each already `[seq, batch, hidden_size]` /
/// `[batch, seq, hidden_size]`, doubling the trailing feature dimension to
/// `2 * hidden_size`) and to combine two directions' final hidden states
/// along the `num_directions` axis.
///
/// # Errors
///
/// Propagates any error from the underlying `tenflowers_core::ops::concat`
/// call (e.g. mismatched shapes on non-`axis` dimensions) or from tape
/// recording.
pub(super) fn concat_on_tape(inputs: &[PyTensor], axis: usize) -> PyResult<PyTensor> {
    let raw_refs: Vec<&Tensor<f32>> = inputs.iter().map(|t| t.tensor.as_ref()).collect();
    let concatenated_raw = tenflowers_core::ops::concat(&raw_refs, axis)
        .map_err(|e| PyRuntimeError::new_err(format!("recurrent concat failed: {e}")))?;
    let concatenated = PyTensor {
        tensor: Arc::new(concatenated_raw),
        requires_grad: true,
        is_pinned: false,
    };
    let py_refs: Vec<&PyTensor> = inputs.iter().collect();
    crate::implicit_autograd::record_and_link_variadic(
        crate::implicit_autograd::VariadicOpKind::Concat { axis: axis as i32 },
        &py_refs,
        &concatenated,
    )?;
    Ok(concatenated)
}

/// Convert a validated `nonlinearity` string into the neural-layer enum.
///
/// Callers validate the string up front (only `"tanh"` or `"relu"` reach
/// here), so any value other than `"relu"` maps to [`RnnNonlinearity::Tanh`].
pub(super) fn parse_nonlinearity(s: &str) -> RnnNonlinearity {
    if s == "relu" {
        RnnNonlinearity::Relu
    } else {
        RnnNonlinearity::Tanh
    }
}

/// Apply the tape-aware activation matching `nonlinearity` to `input`.
pub(super) fn apply_nonlinearity(
    input: &PyTensor,
    nonlinearity: RnnNonlinearity,
) -> PyResult<PyTensor> {
    match nonlinearity {
        RnnNonlinearity::Relu => super::functions::relu(input),
        RnnNonlinearity::Tanh => super::functions::tanh(input),
    }
}

/// Deterministic, non-constant test data so the recurrences cannot collapse
/// to a trivially zero output. Shared by every gradient/forward test in
/// [`tests`].
#[cfg(test)]
pub(super) fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * 0.1 - 1.0).collect()
}

#[cfg(test)]
pub(super) fn make_tensor(data: Vec<f32>, shape: &[usize]) -> PyTensor {
    let tensor = Tensor::from_vec(data, shape).expect("tensor construction");
    PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    }
}
