//! GRU (multi-layer, optionally bidirectional) and GRUCell (single time
//! step), reimplemented on top of the implicit autograd tape.
//!
//! See the [`super`] module doc for the shared autograd design this follows.

use super::{
    concat_on_tape, gate_slice, reverse_layer_param, sequence_timestep, slice_initial_state,
    snapshot_leaf, stack_on_tape, zeros_state,
};
use crate::neural::layers::PyParameter;
use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;
use tenflowers_core::Tensor;

const NUM_GATES: usize = 3;

/// Single GRU cell step (PyTorch gate convention: reset, update, new),
/// computed entirely from tape-aware [`PyTensor`] operations.
///
/// `weight_ih` is `[input_size, 3 * hidden_size]`, `weight_hh` is
/// `[hidden_size, 3 * hidden_size]`.
#[allow(clippy::too_many_arguments)]
fn gru_cell_step(
    x_t: &PyTensor,
    h: &PyTensor,
    weight_ih: &PyTensor,
    weight_hh: &PyTensor,
    bias_ih: Option<&PyTensor>,
    bias_hh: Option<&PyTensor>,
    hidden_size: usize,
) -> PyResult<PyTensor> {
    let gi = x_t.matmul(weight_ih)?;
    let gi = match bias_ih {
        Some(b) => gi.add(b)?,
        None => gi,
    };
    let gh = h.matmul(weight_hh)?;
    let gh = match bias_hh {
        Some(b) => gh.add(b)?,
        None => gh,
    };

    let i_r = gate_slice(&gi, 0, hidden_size, NUM_GATES)?;
    let i_z = gate_slice(&gi, 1, hidden_size, NUM_GATES)?;
    let i_n = gate_slice(&gi, 2, hidden_size, NUM_GATES)?;
    let h_r = gate_slice(&gh, 0, hidden_size, NUM_GATES)?;
    let h_z = gate_slice(&gh, 1, hidden_size, NUM_GATES)?;
    let h_n = gate_slice(&gh, 2, hidden_size, NUM_GATES)?;

    let r = crate::neural::functions::sigmoid(&i_r.add(&h_r)?)?;
    let z = crate::neural::functions::sigmoid(&i_z.add(&h_z)?)?;
    let n = crate::neural::functions::tanh(&i_n.add(&r.mul(&h_n)?)?)?;

    // h_new = (1 - z) * n + z * h
    //
    // There is deliberately no `BinaryOpKind::ScalarMul`/scalar-op variant
    // available on the implicit tape (see `implicit_autograd::UnaryOpKind`'s
    // own doc on why `ScalarMul` is excluded: no `TrackedTensor` backward
    // method exists for it yet), so `(1 - z)` cannot be built via a raw
    // `Tensor::multiply_scalar`/`.add_scalar` call on `z`'s tracked value —
    // doing so would silently bypass the tape entirely, exactly the class of
    // bug this migration eliminates. Instead, build a full untracked
    // `[batch, hidden_size]` `ones` constant and go through the tape-aware
    // `PyTensor::sub`, which self-records via `record_and_link_binary`
    // (whose `tracked_or_watch_as_constant` helper automatically treats the
    // untracked `ones` operand as a tape constant the first time it
    // participates in an op — nothing else needs to be done to make this
    // safe, since nothing ever requests a gradient on `ones` itself).
    let batch = z.tensor.shape().dims()[0];
    let ones = PyTensor {
        tensor: Arc::new(Tensor::ones(&[batch, hidden_size])),
        requires_grad: false,
        is_pinned: false,
    };
    let one_minus_z = ones.sub(&z)?;
    one_minus_z.mul(&n)?.add(&z.mul(h)?)
}

/// Test-only re-export of the private [`gru_cell_step`] free function, so
/// `tests.rs` can independently recompute a single GRU cell step from
/// snapshotted parameter values and compare it against `PyGRU::forward`'s
/// own output for a `seq_len = 1` input (see
/// `tests::gru_forward_matches_single_cell_step`). Not `pub` outside
/// `#[cfg(test)]`: this is a private implementation detail everywhere else.
#[cfg(test)]
pub(super) fn gru_cell_step_for_tests(
    x_t: &PyTensor,
    h: &PyTensor,
    weight_ih: &PyTensor,
    weight_hh: &PyTensor,
    bias_ih: Option<&PyTensor>,
    bias_hh: Option<&PyTensor>,
    hidden_size: usize,
) -> PyResult<PyTensor> {
    gru_cell_step(x_t, h, weight_ih, weight_hh, bias_ih, bias_hh, hidden_size)
}

/// Gated Recurrent Unit (GRU) Layer
///
/// Applies a multi-layer gated recurrent unit (GRU) RNN to an input
/// sequence. GRUs are similar to LSTMs but with fewer parameters.
///
/// # Parameters (autograd)
///
/// One [`Py<PyParameter>`] per layer per weight kind, matching
/// `tenflowers_neural::layers::rnn::GRU`'s own per-layer `Vec<Tensor<T>>`
/// storage exactly (see the [`super`] module doc):
///
/// * `weight_ih[layer]`: `[layer_input_size, 3 * hidden_size]`
/// * `weight_hh[layer]`: `[hidden_size, 3 * hidden_size]`
/// * `bias_ih[layer]`, `bias_hh[layer]` (only when `bias == true`): `[3 * hidden_size]`
///
/// When `bidirectional == true`, an entire second parallel set holds the
/// reverse direction's parameters, same per-layer shapes.
#[pyclass(name = "GRU")]
#[derive(Debug)]
pub struct PyGRU {
    /// Number of expected features in the input
    pub input_size: usize,
    /// Number of features in the hidden state
    pub hidden_size: usize,
    /// Number of recurrent layers
    pub num_layers: usize,
    /// If True, use bias weights
    pub bias: bool,
    /// If True, use batch_first format (batch, seq, feature)
    pub batch_first: bool,
    /// Dropout probability for outputs of each GRU layer except last
    pub dropout: f32,
    /// If True, becomes a bidirectional GRU
    pub bidirectional: bool,
    /// Per-layer input-to-hidden weights, forward direction.
    weight_ih: Vec<Py<PyParameter>>,
    /// Per-layer hidden-to-hidden weights, forward direction.
    weight_hh: Vec<Py<PyParameter>>,
    /// Per-layer input-to-hidden biases, forward direction (empty when `!bias`).
    bias_ih: Vec<Py<PyParameter>>,
    /// Per-layer hidden-to-hidden biases, forward direction (empty when `!bias`).
    bias_hh: Vec<Py<PyParameter>>,
    /// Per-layer input-to-hidden weights, reverse direction (`None` when
    /// `!bidirectional`).
    weight_ih_reverse: Option<Vec<Py<PyParameter>>>,
    /// Per-layer hidden-to-hidden weights, reverse direction.
    weight_hh_reverse: Option<Vec<Py<PyParameter>>>,
    /// Per-layer input-to-hidden biases, reverse direction.
    bias_ih_reverse: Option<Vec<Py<PyParameter>>>,
    /// Per-layer hidden-to-hidden biases, reverse direction.
    bias_hh_reverse: Option<Vec<Py<PyParameter>>>,
}

impl Clone for PyGRU {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            input_size: self.input_size,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            bias: self.bias,
            batch_first: self.batch_first,
            dropout: self.dropout,
            bidirectional: self.bidirectional,
            weight_ih: self.weight_ih.iter().map(|p| p.clone_ref(py)).collect(),
            weight_hh: self.weight_hh.iter().map(|p| p.clone_ref(py)).collect(),
            bias_ih: self.bias_ih.iter().map(|p| p.clone_ref(py)).collect(),
            bias_hh: self.bias_hh.iter().map(|p| p.clone_ref(py)).collect(),
            weight_ih_reverse: self
                .weight_ih_reverse
                .as_ref()
                .map(|v| v.iter().map(|p| p.clone_ref(py)).collect()),
            weight_hh_reverse: self
                .weight_hh_reverse
                .as_ref()
                .map(|v| v.iter().map(|p| p.clone_ref(py)).collect()),
            bias_ih_reverse: self
                .bias_ih_reverse
                .as_ref()
                .map(|v| v.iter().map(|p| p.clone_ref(py)).collect()),
            bias_hh_reverse: self
                .bias_hh_reverse
                .as_ref()
                .map(|v| v.iter().map(|p| p.clone_ref(py)).collect()),
        })
    }
}

/// Build one layer's freshly zero-initialised weight/bias `Py<PyParameter>`
/// set (mirrors `lstm.rs`'s `new_layer_params`, but with `3 * hidden_size`
/// fused gates instead of `4 *`).
fn new_layer_params(
    py: Python<'_>,
    layer_input_size: usize,
    hidden_size: usize,
    bias: bool,
) -> PyResult<super::LayerParamSet> {
    let weight_ih = Py::new(
        py,
        PyParameter::new(
            PyTensor {
                tensor: Arc::new(Tensor::zeros(&[layer_input_size, 3 * hidden_size])),
                requires_grad: true,
                is_pinned: false,
            },
            Some(true),
        ),
    )?;
    let weight_hh = Py::new(
        py,
        PyParameter::new(
            PyTensor {
                tensor: Arc::new(Tensor::zeros(&[hidden_size, 3 * hidden_size])),
                requires_grad: true,
                is_pinned: false,
            },
            Some(true),
        ),
    )?;
    let (bias_ih, bias_hh) = if bias {
        let b_ih = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(Tensor::zeros(&[3 * hidden_size])),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;
        let b_hh = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(Tensor::zeros(&[3 * hidden_size])),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;
        (Some(b_ih), Some(b_hh))
    } else {
        (None, None)
    };
    Ok((weight_ih, weight_hh, bias_ih, bias_hh))
}

/// Build a layer's `(weight_ih, weight_hh, bias_ih, bias_hh)` snapshot,
/// marking each parameter as this forward pass's tape leaf. Shared by both
/// the forward and (when `bidirectional`) reverse direction call sites in
/// `PyGRU::forward` below.
struct LayerSnapshot {
    w_ih: PyTensor,
    w_hh: PyTensor,
    b_ih: Option<PyTensor>,
    b_hh: Option<PyTensor>,
}

fn snapshot_layer(
    py: Python<'_>,
    w_ih: &Py<PyParameter>,
    w_hh: &Py<PyParameter>,
    b_ih: Option<&Py<PyParameter>>,
    b_hh: Option<&Py<PyParameter>>,
) -> PyResult<LayerSnapshot> {
    Ok(LayerSnapshot {
        w_ih: snapshot_leaf(py, w_ih)?,
        w_hh: snapshot_leaf(py, w_hh)?,
        b_ih: b_ih.map(|p| snapshot_leaf(py, p)).transpose()?,
        b_hh: b_hh.map(|p| snapshot_leaf(py, p)).transpose()?,
    })
}

#[pymethods]
impl PyGRU {
    /// Create a new GRU layer
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers=None, bias=None, batch_first=None, dropout=None, bidirectional=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        input_size: usize,
        hidden_size: usize,
        num_layers: Option<usize>,
        bias: Option<bool>,
        batch_first: Option<bool>,
        dropout: Option<f32>,
        bidirectional: Option<bool>,
    ) -> PyResult<Self> {
        let num_layers = num_layers.unwrap_or(1);
        let bias = bias.unwrap_or(true);
        let batch_first = batch_first.unwrap_or(false);
        let dropout = dropout.unwrap_or(0.0);
        let bidirectional = bidirectional.unwrap_or(false);

        if input_size == 0 {
            return Err(PyValueError::new_err("input_size must be positive"));
        }
        if hidden_size == 0 {
            return Err(PyValueError::new_err("hidden_size must be positive"));
        }
        if num_layers == 0 {
            return Err(PyValueError::new_err("num_layers must be positive"));
        }
        if !(0.0..=1.0).contains(&dropout) {
            return Err(PyValueError::new_err("dropout must be between 0 and 1"));
        }

        let directions = if bidirectional { 2 } else { 1 };

        let mut weight_ih = Vec::with_capacity(num_layers);
        let mut weight_hh = Vec::with_capacity(num_layers);
        let mut bias_ih = Vec::new();
        let mut bias_hh = Vec::new();
        let mut weight_ih_reverse = bidirectional.then(|| Vec::with_capacity(num_layers));
        let mut weight_hh_reverse = bidirectional.then(|| Vec::with_capacity(num_layers));
        let mut bias_ih_reverse = bidirectional.then(Vec::new);
        let mut bias_hh_reverse = bidirectional.then(Vec::new);

        for layer in 0..num_layers {
            let layer_input_size = if layer == 0 {
                input_size
            } else {
                hidden_size * directions
            };

            let (w_ih, w_hh, b_ih, b_hh) =
                new_layer_params(py, layer_input_size, hidden_size, bias)?;
            weight_ih.push(w_ih);
            weight_hh.push(w_hh);
            if let (Some(b_ih), Some(b_hh)) = (b_ih, b_hh) {
                bias_ih.push(b_ih);
                bias_hh.push(b_hh);
            }

            if bidirectional {
                let (w_ih_r, w_hh_r, b_ih_r, b_hh_r) =
                    new_layer_params(py, layer_input_size, hidden_size, bias)?;
                if let Some(v) = weight_ih_reverse.as_mut() {
                    v.push(w_ih_r);
                }
                if let Some(v) = weight_hh_reverse.as_mut() {
                    v.push(w_hh_r);
                }
                if let (Some(b_ih_r), Some(b_hh_r)) = (b_ih_r, b_hh_r) {
                    if let Some(v) = bias_ih_reverse.as_mut() {
                        v.push(b_ih_r);
                    }
                    if let Some(v) = bias_hh_reverse.as_mut() {
                        v.push(b_hh_r);
                    }
                }
            }
        }

        Ok(PyGRU {
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            weight_ih_reverse,
            weight_hh_reverse,
            bias_ih_reverse,
            bias_hh_reverse,
        })
    }

    /// Forward pass through the GRU layer
    #[pyo3(signature = (input, hidden=None))]
    pub fn forward(
        &self,
        py: Python<'_>,
        input: &PyTensor,
        hidden: Option<PyTensor>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let input_shape = input.tensor.shape();

        if input_shape.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "Expected 3D input (seq_len, batch, input_size), got {}D",
                input_shape.len()
            )));
        }

        let input_dim = input_shape[2];
        if input_dim != self.input_size {
            return Err(PyValueError::new_err(format!(
                "Expected input_size={}, got {}",
                self.input_size, input_dim
            )));
        }

        if self.bidirectional && hidden.is_some() {
            // Bidirectional GRU: threading an explicit initial hidden state is not
            // supported here (matching LSTM's analogous unidirectional-only limit).
            return Err(PyRuntimeError::new_err(
                "GRU: providing an explicit initial hidden state is currently \
                 supported only for unidirectional GRUs",
            ));
        }

        let dims = input_shape.dims().to_vec();
        let (seq_len, batch) = if self.batch_first {
            (dims[1], dims[0])
        } else {
            (dims[0], dims[1])
        };
        let num_directions = if self.bidirectional { 2 } else { 1 };

        let init = hidden.as_ref();

        let mut layer_input = input.clone();
        let mut h_finals: Vec<PyTensor> = Vec::with_capacity(self.num_layers * num_directions);

        for l in 0..self.num_layers {
            let feature = if l == 0 {
                self.input_size
            } else {
                self.hidden_size * num_directions
            };

            let fwd_snap = snapshot_layer(
                py,
                &self.weight_ih[l],
                &self.weight_hh[l],
                self.bias.then(|| &self.bias_ih[l]),
                self.bias.then(|| &self.bias_hh[l]),
            )?;

            let mut h_fwd = match init {
                Some(h0) => slice_initial_state(h0, l, 0, num_directions, batch, self.hidden_size)?,
                None => zeros_state(batch, self.hidden_size),
            };
            let mut fwd_outputs = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let x_t = sequence_timestep(&layer_input, t, batch, feature, self.batch_first)?;
                h_fwd = gru_cell_step(
                    &x_t,
                    &h_fwd,
                    &fwd_snap.w_ih,
                    &fwd_snap.w_hh,
                    fwd_snap.b_ih.as_ref(),
                    fwd_snap.b_hh.as_ref(),
                    self.hidden_size,
                )?;
                fwd_outputs.push(h_fwd.clone());
            }

            if !self.bidirectional {
                let time_axis = if self.batch_first { 1 } else { 0 };
                layer_input = stack_on_tape(&fwd_outputs, time_axis)?;
                h_finals.push(h_fwd);
                continue;
            }

            // Reverse direction: same layer, iterating time steps backward.
            // `reverse_layer_param` is a fallible (`PyResult`) lookup rather
            // than `.unwrap()`/`.expect()` — see that helper's own doc for
            // why this branch is only ever reached when the invariant it
            // relies on (bidirectional=true implies these `Option`s are
            // `Some`) already holds.
            let w_ih_rev =
                reverse_layer_param(&self.weight_ih_reverse, l, "weight_ih")?.clone_ref(py);
            let w_hh_rev =
                reverse_layer_param(&self.weight_hh_reverse, l, "weight_hh")?.clone_ref(py);
            let b_ih_rev = self
                .bias
                .then(|| reverse_layer_param(&self.bias_ih_reverse, l, "bias_ih"))
                .transpose()?
                .map(|p| p.clone_ref(py));
            let b_hh_rev = self
                .bias
                .then(|| reverse_layer_param(&self.bias_hh_reverse, l, "bias_hh"))
                .transpose()?
                .map(|p| p.clone_ref(py));
            let rev_snap = snapshot_layer(
                py,
                &w_ih_rev,
                &w_hh_rev,
                b_ih_rev.as_ref(),
                b_hh_rev.as_ref(),
            )?;

            let mut h_rev = match init {
                // Bidirectional forward() rejects an explicit `hidden` above,
                // so `init` is always `None` here; this arm is unreachable in
                // practice but kept for structural symmetry with `h_fwd`
                // above rather than a `.expect()`/`unreachable!()`.
                Some(h0) => slice_initial_state(h0, l, 1, num_directions, batch, self.hidden_size)?,
                None => zeros_state(batch, self.hidden_size),
            };
            // Collected in reverse time order (t = seq_len-1 down to 0) since
            // that is the order the recurrence naturally produces each
            // output; reversed back into forward time order below via
            // `.rev()` rather than pre-allocating an
            // `Vec<Option<PyTensor>>` and indexing into it (which would need
            // an infallible-in-practice `.expect()`/`.unwrap()` to unwrap
            // each `Option` afterward — avoided entirely by this ordering
            // instead, per this crate's no-`unwrap()`/`expect()`-in-
            // production policy).
            let mut rev_outputs_reverse_order = Vec::with_capacity(seq_len);
            for t in (0..seq_len).rev() {
                let x_t = sequence_timestep(&layer_input, t, batch, feature, self.batch_first)?;
                h_rev = gru_cell_step(
                    &x_t,
                    &h_rev,
                    &rev_snap.w_ih,
                    &rev_snap.w_hh,
                    rev_snap.b_ih.as_ref(),
                    rev_snap.b_hh.as_ref(),
                    self.hidden_size,
                )?;
                rev_outputs_reverse_order.push(h_rev.clone());
            }
            let rev_outputs: Vec<PyTensor> = rev_outputs_reverse_order.into_iter().rev().collect();

            let time_axis = if self.batch_first { 1 } else { 0 };
            let fwd_seq = stack_on_tape(&fwd_outputs, time_axis)?;
            let rev_seq = stack_on_tape(&rev_outputs, time_axis)?;
            // Both `fwd_seq`/`rev_seq` are `[seq, batch, hidden]` (or
            // `[batch, seq, hidden]` if batch_first) — the feature axis is
            // the last one either way.
            layer_input = concat_on_tape(&[fwd_seq, rev_seq], 2)?;
            h_finals.push(h_fwd);
            h_finals.push(h_rev);
        }

        let h_n = stack_on_tape(&h_finals, 0)?;
        Ok((layer_input, h_n))
    }

    /// Reset layer parameters
    pub fn reset_parameters(&mut self, py: Python<'_>) -> PyResult<()> {
        self.weight_ih.clear();
        self.weight_hh.clear();
        self.bias_ih.clear();
        self.bias_hh.clear();
        if let Some(v) = self.weight_ih_reverse.as_mut() {
            v.clear();
        }
        if let Some(v) = self.weight_hh_reverse.as_mut() {
            v.clear();
        }
        if let Some(v) = self.bias_ih_reverse.as_mut() {
            v.clear();
        }
        if let Some(v) = self.bias_hh_reverse.as_mut() {
            v.clear();
        }

        let directions = if self.bidirectional { 2 } else { 1 };

        for layer in 0..self.num_layers {
            let layer_input_size = if layer == 0 {
                self.input_size
            } else {
                self.hidden_size * directions
            };

            let (w_ih, w_hh, b_ih, b_hh) =
                new_layer_params(py, layer_input_size, self.hidden_size, self.bias)?;
            self.weight_ih.push(w_ih);
            self.weight_hh.push(w_hh);
            if let (Some(b_ih), Some(b_hh)) = (b_ih, b_hh) {
                self.bias_ih.push(b_ih);
                self.bias_hh.push(b_hh);
            }

            if self.bidirectional {
                let (w_ih_r, w_hh_r, b_ih_r, b_hh_r) =
                    new_layer_params(py, layer_input_size, self.hidden_size, self.bias)?;
                if let Some(v) = self.weight_ih_reverse.as_mut() {
                    v.push(w_ih_r);
                }
                if let Some(v) = self.weight_hh_reverse.as_mut() {
                    v.push(w_hh_r);
                }
                if let (Some(b_ih_r), Some(b_hh_r)) = (b_ih_r, b_hh_r) {
                    if let Some(v) = self.bias_ih_reverse.as_mut() {
                        v.push(b_ih_r);
                    }
                    if let Some(v) = self.bias_hh_reverse.as_mut() {
                        v.push(b_hh_r);
                    }
                }
            }
        }

        Ok(())
    }

    /// Get layer parameters, in stable order: for each layer `0..num_layers`,
    /// `weight_ih`, `weight_hh`, then (if `bias`) `bias_ih`, `bias_hh`; then,
    /// if `bidirectional`, the same sequence again for the reverse direction.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyParameter>> {
        let mut params = Vec::new();
        for l in 0..self.num_layers {
            params.push(self.weight_ih[l].clone_ref(py));
            params.push(self.weight_hh[l].clone_ref(py));
            if self.bias {
                params.push(self.bias_ih[l].clone_ref(py));
                params.push(self.bias_hh[l].clone_ref(py));
            }
        }
        if self.bidirectional {
            for l in 0..self.num_layers {
                if let Some(v) = &self.weight_ih_reverse {
                    params.push(v[l].clone_ref(py));
                }
                if let Some(v) = &self.weight_hh_reverse {
                    params.push(v[l].clone_ref(py));
                }
                if self.bias {
                    if let Some(v) = &self.bias_ih_reverse {
                        params.push(v[l].clone_ref(py));
                    }
                    if let Some(v) = &self.bias_hh_reverse {
                        params.push(v[l].clone_ref(py));
                    }
                }
            }
        }
        params
    }

    fn __repr__(&self) -> String {
        format!(
            "GRU(input_size={}, hidden_size={}, num_layers={}, bias={}, batch_first={}, dropout={}, bidirectional={})",
            self.input_size, self.hidden_size, self.num_layers, self.bias,
            self.batch_first, self.dropout, self.bidirectional
        )
    }
}

/// GRU Cell
///
/// A single GRU cell (one time step).
///
/// # Parameters (autograd)
///
/// * `weight_ih_param`: `[input_size, 3 * hidden_size]`
/// * `weight_hh_param`: `[hidden_size, 3 * hidden_size]`
///
/// Initialised via `1/sqrt(hidden_size)`-scaled `Tensor::randn`, matching the
/// pre-autograd version exactly (see the [`super`] module doc). As with
/// [`super::lstm::PyLSTMCell`], `bias` is accepted for signature
/// compatibility but `forward()` does not apply bias parameters, preserving
/// the pre-existing forward behaviour exactly (`gru_cell_step` was always
/// called with `None, None` for the bias arguments).
#[pyclass(name = "GRUCell")]
#[derive(Debug)]
pub struct PyGRUCell {
    /// Number of expected features in the input
    pub input_size: usize,
    /// Number of features in the hidden state
    pub hidden_size: usize,
    /// If True, use bias weights (currently unused by `forward()` — see the
    /// struct doc)
    pub bias: bool,
    /// Input-hidden weight, stored pre-transposed as `[input_size, 3 *
    /// hidden_size]` — see `PyLSTMCell::weight_ih_param`'s doc for why this
    /// differs from the pre-autograd version's `[3*hidden, in]` storage.
    weight_ih_param: Py<PyParameter>,
    /// Hidden-hidden weight, stored pre-transposed as `[hidden_size, 3 *
    /// hidden_size]`.
    weight_hh_param: Py<PyParameter>,
}

impl Clone for PyGRUCell {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            input_size: self.input_size,
            hidden_size: self.hidden_size,
            bias: self.bias,
            weight_ih_param: self.weight_ih_param.clone_ref(py),
            weight_hh_param: self.weight_hh_param.clone_ref(py),
        })
    }
}

#[pymethods]
impl PyGRUCell {
    /// Create a new GRU cell
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias=None))]
    pub fn new(
        py: Python<'_>,
        input_size: usize,
        hidden_size: usize,
        bias: Option<bool>,
    ) -> PyResult<Self> {
        let bias = bias.unwrap_or(true);

        if input_size == 0 {
            return Err(PyValueError::new_err("input_size must be positive"));
        }
        if hidden_size == 0 {
            return Err(PyValueError::new_err("hidden_size must be positive"));
        }

        let scale = 1.0_f32 / (hidden_size as f32).sqrt();
        let weight_ih = Tensor::randn(&[input_size, 3 * hidden_size])
            .and_then(|t| t.multiply_scalar(scale))
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to init GRUCell weight_ih: {}", e))
            })?;
        let weight_hh = Tensor::randn(&[hidden_size, 3 * hidden_size])
            .and_then(|t| t.multiply_scalar(scale))
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to init GRUCell weight_hh: {}", e))
            })?;

        let weight_ih_param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(weight_ih),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;
        let weight_hh_param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(weight_hh),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;

        Ok(PyGRUCell {
            input_size,
            hidden_size,
            bias,
            weight_ih_param,
            weight_hh_param,
        })
    }

    /// Forward pass through the GRU cell
    #[pyo3(signature = (input, hidden=None))]
    pub fn forward(
        &self,
        py: Python<'_>,
        input: &PyTensor,
        hidden: Option<PyTensor>,
    ) -> PyResult<PyTensor> {
        let input_shape = input.tensor.shape();

        if input_shape.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "Expected 2D input (batch, input_size), got {}D",
                input_shape.len()
            )));
        }

        if input_shape[1] != self.input_size {
            return Err(PyValueError::new_err(format!(
                "Expected input_size={}, got {}",
                self.input_size, input_shape[1]
            )));
        }

        let batch_size = input_shape[0];

        let weight_ih = snapshot_leaf(py, &self.weight_ih_param)?;
        let weight_hh = snapshot_leaf(py, &self.weight_hh_param)?;

        let h_0 = hidden.unwrap_or_else(|| zeros_state(batch_size, self.hidden_size));

        gru_cell_step(
            input,
            &h_0,
            &weight_ih,
            &weight_hh,
            None,
            None,
            self.hidden_size,
        )
    }

    /// Get layer parameters: `[weight_ih, weight_hh]`.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyParameter>> {
        vec![
            self.weight_ih_param.clone_ref(py),
            self.weight_hh_param.clone_ref(py),
        ]
    }

    fn __repr__(&self) -> String {
        format!(
            "GRUCell(input_size={}, hidden_size={}, bias={})",
            self.input_size, self.hidden_size, self.bias
        )
    }
}
