//! Vanilla (Elman) RNN, reimplemented on top of the implicit autograd tape.
//!
//! See the [`super`] module doc for the shared autograd design this follows.

use super::{
    apply_nonlinearity, concat_on_tape, parse_nonlinearity, reverse_layer_param,
    sequence_timestep, slice_initial_state, snapshot_leaf, stack_on_tape, zeros_state,
};
use crate::neural::layers::PyParameter;
use crate::tensor_ops::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;
use tenflowers_core::Tensor;
use tenflowers_neural::layers::rnn::RnnNonlinearity;

/// Single vanilla-RNN cell step, computed entirely from tape-aware
/// [`PyTensor`] operations.
///
/// `weight_ih` is `[input_size, hidden_size]`, `weight_hh` is `[hidden_size,
/// hidden_size]` — no gate fusion, unlike LSTM/GRU.
#[allow(clippy::too_many_arguments)]
fn rnn_cell_step(
    x_t: &PyTensor,
    h: &PyTensor,
    weight_ih: &PyTensor,
    weight_hh: &PyTensor,
    bias_ih: Option<&PyTensor>,
    bias_hh: Option<&PyTensor>,
    nonlinearity: RnnNonlinearity,
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
    let pre_activation = gi.add(&gh)?;
    apply_nonlinearity(&pre_activation, nonlinearity)
}

/// Vanilla RNN Layer
///
/// Applies a multi-layer Elman RNN with tanh or ReLU non-linearity to an
/// input sequence.
///
/// # Parameters (autograd)
///
/// One [`Py<PyParameter>`] per layer per weight kind, matching
/// `tenflowers_neural::layers::rnn::RNN`'s own per-layer `Vec<Tensor<T>>`
/// storage exactly (see the [`super`] module doc):
///
/// * `weight_ih[layer]`: `[layer_input_size, hidden_size]`
/// * `weight_hh[layer]`: `[hidden_size, hidden_size]`
/// * `bias_ih[layer]`, `bias_hh[layer]` (only when `bias == true`): `[hidden_size]`
///
/// When `bidirectional == true`, an entire second parallel set holds the
/// reverse direction's parameters, same per-layer shapes.
#[pyclass(name = "RNN")]
#[derive(Debug)]
pub struct PyRNN {
    /// Number of expected features in the input
    pub input_size: usize,
    /// Number of features in the hidden state
    pub hidden_size: usize,
    /// Number of recurrent layers
    pub num_layers: usize,
    /// Non-linearity to use ('tanh' or 'relu')
    pub nonlinearity: String,
    /// If True, use bias weights
    pub bias: bool,
    /// If True, use batch_first format (batch, seq, feature)
    pub batch_first: bool,
    /// Dropout probability for outputs of each RNN layer except last
    pub dropout: f32,
    /// If True, becomes a bidirectional RNN
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

impl Clone for PyRNN {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            input_size: self.input_size,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            nonlinearity: self.nonlinearity.clone(),
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
/// set (mirrors `lstm.rs`/`gru.rs`'s `new_layer_params`, but with plain
/// `hidden_size`-wide, unfused gates).
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
                tensor: Arc::new(Tensor::zeros(&[layer_input_size, hidden_size])),
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
                tensor: Arc::new(Tensor::zeros(&[hidden_size, hidden_size])),
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
                    tensor: Arc::new(Tensor::zeros(&[hidden_size])),
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
                    tensor: Arc::new(Tensor::zeros(&[hidden_size])),
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

#[pymethods]
impl PyRNN {
    /// Create a new RNN layer
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers=None, nonlinearity=None, bias=None, batch_first=None, dropout=None, bidirectional=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        input_size: usize,
        hidden_size: usize,
        num_layers: Option<usize>,
        nonlinearity: Option<String>,
        bias: Option<bool>,
        batch_first: Option<bool>,
        dropout: Option<f32>,
        bidirectional: Option<bool>,
    ) -> PyResult<Self> {
        let num_layers = num_layers.unwrap_or(1);
        let nonlinearity = nonlinearity.unwrap_or_else(|| "tanh".to_string());
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
        if nonlinearity != "tanh" && nonlinearity != "relu" {
            return Err(PyValueError::new_err(
                "nonlinearity must be 'tanh' or 'relu'",
            ));
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

        Ok(PyRNN {
            input_size,
            hidden_size,
            num_layers,
            nonlinearity,
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

    /// Forward pass through the RNN layer
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

        let dims = input_shape.dims().to_vec();
        let (seq_len, batch) = if self.batch_first {
            (dims[1], dims[0])
        } else {
            (dims[0], dims[1])
        };
        let num_directions = if self.bidirectional { 2 } else { 1 };
        let nonlinearity = parse_nonlinearity(&self.nonlinearity);

        let init = hidden.as_ref();

        let mut layer_input = input.clone();
        let mut h_finals: Vec<PyTensor> = Vec::with_capacity(self.num_layers * num_directions);

        for l in 0..self.num_layers {
            let feature = if l == 0 {
                self.input_size
            } else {
                self.hidden_size * num_directions
            };

            let w_ih_fwd = snapshot_leaf(py, &self.weight_ih[l])?;
            let w_hh_fwd = snapshot_leaf(py, &self.weight_hh[l])?;
            let b_ih_fwd = self
                .bias
                .then(|| snapshot_leaf(py, &self.bias_ih[l]))
                .transpose()?;
            let b_hh_fwd = self
                .bias
                .then(|| snapshot_leaf(py, &self.bias_hh[l]))
                .transpose()?;

            let mut h_fwd = match init {
                Some(h0) => {
                    slice_initial_state(h0, l, 0, num_directions, batch, self.hidden_size)?
                }
                None => zeros_state(batch, self.hidden_size),
            };
            let mut fwd_outputs = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let x_t = sequence_timestep(&layer_input, t, batch, feature, self.batch_first)?;
                h_fwd = rnn_cell_step(
                    &x_t,
                    &h_fwd,
                    &w_ih_fwd,
                    &w_hh_fwd,
                    b_ih_fwd.as_ref(),
                    b_hh_fwd.as_ref(),
                    nonlinearity,
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
                snapshot_leaf(py, reverse_layer_param(&self.weight_ih_reverse, l, "weight_ih")?)?;
            let w_hh_rev =
                snapshot_leaf(py, reverse_layer_param(&self.weight_hh_reverse, l, "weight_hh")?)?;
            let b_ih_rev = self
                .bias
                .then(|| {
                    snapshot_leaf(py, reverse_layer_param(&self.bias_ih_reverse, l, "bias_ih")?)
                })
                .transpose()?;
            let b_hh_rev = self
                .bias
                .then(|| {
                    snapshot_leaf(py, reverse_layer_param(&self.bias_hh_reverse, l, "bias_hh")?)
                })
                .transpose()?;

            let mut h_rev = match init {
                // Explicit `hidden` with bidirectional=true is accepted (no
                // guard rejects it, unlike LSTM/GRU); when provided, both
                // directions read from the same `[num_directions, batch,
                // hidden]` state via `slice_initial_state`'s
                // `direction_idx=1` row.
                Some(h0) => {
                    slice_initial_state(h0, l, 1, num_directions, batch, self.hidden_size)?
                }
                None => zeros_state(batch, self.hidden_size),
            };
            // Collected in reverse time order, then reversed back into
            // forward time order below (see the analogous comment in
            // `gru.rs::forward` for why this avoids an
            // infallible-in-practice `Vec<Option<PyTensor>>` unwrap).
            let mut rev_outputs_reverse_order = Vec::with_capacity(seq_len);
            for t in (0..seq_len).rev() {
                let x_t = sequence_timestep(&layer_input, t, batch, feature, self.batch_first)?;
                h_rev = rnn_cell_step(
                    &x_t,
                    &h_rev,
                    &w_ih_rev,
                    &w_hh_rev,
                    b_ih_rev.as_ref(),
                    b_hh_rev.as_ref(),
                    nonlinearity,
                )?;
                rev_outputs_reverse_order.push(h_rev.clone());
            }
            let rev_outputs: Vec<PyTensor> = rev_outputs_reverse_order.into_iter().rev().collect();

            let time_axis = if self.batch_first { 1 } else { 0 };
            let fwd_seq = stack_on_tape(&fwd_outputs, time_axis)?;
            let rev_seq = stack_on_tape(&rev_outputs, time_axis)?;
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
            "RNN(input_size={}, hidden_size={}, num_layers={}, nonlinearity='{}', bias={}, batch_first={}, dropout={}, bidirectional={})",
            self.input_size, self.hidden_size, self.num_layers, self.nonlinearity,
            self.bias, self.batch_first, self.dropout, self.bidirectional
        )
    }
}
