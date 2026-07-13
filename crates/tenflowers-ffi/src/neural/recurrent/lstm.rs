//! LSTM (multi-layer, optionally bidirectional) and LSTMCell (single time
//! step), reimplemented on top of the implicit autograd tape.
//!
//! See the [`super`] module doc for the shared autograd design this follows.

use super::{
    gate_slice, sequence_timestep, slice_initial_state, snapshot_leaf, stack_on_tape, zeros_state,
};
use crate::neural::layers::PyParameter;
use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;
use tenflowers_core::Tensor;

const NUM_GATES: usize = 4;

/// Single LSTM cell step, computed entirely from tape-aware [`PyTensor`]
/// operations.
///
/// `weight_ih` is `[input_size, 4 * hidden_size]`, `weight_hh` is
/// `[hidden_size, 4 * hidden_size]`, and optional bias vectors have length
/// `4 * hidden_size`. Gate order is `[input, forget, cell, output]`.
#[allow(clippy::too_many_arguments)]
fn lstm_cell_step(
    x_t: &PyTensor,
    h: &PyTensor,
    c: &PyTensor,
    weight_ih: &PyTensor,
    weight_hh: &PyTensor,
    bias_ih: Option<&PyTensor>,
    bias_hh: Option<&PyTensor>,
    hidden_size: usize,
) -> PyResult<(PyTensor, PyTensor)> {
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
    let gates = gi.add(&gh)?;

    let i_gate =
        crate::neural::functions::sigmoid(&gate_slice(&gates, 0, hidden_size, NUM_GATES)?)?;
    let f_gate =
        crate::neural::functions::sigmoid(&gate_slice(&gates, 1, hidden_size, NUM_GATES)?)?;
    let g_gate = crate::neural::functions::tanh(&gate_slice(&gates, 2, hidden_size, NUM_GATES)?)?;
    let o_gate =
        crate::neural::functions::sigmoid(&gate_slice(&gates, 3, hidden_size, NUM_GATES)?)?;

    let new_c = f_gate.mul(c)?.add(&i_gate.mul(&g_gate)?)?;
    let new_h = o_gate.mul(&crate::neural::functions::tanh(&new_c)?)?;
    Ok((new_h, new_c))
}

/// Long Short-Term Memory (LSTM) Layer
///
/// Applies a multi-layer long short-term memory (LSTM) RNN to an input
/// sequence. LSTMs are excellent at capturing long-term dependencies in
/// sequences.
///
/// # Parameters (autograd)
///
/// One [`Py<PyParameter>`] per layer per weight kind, matching
/// `tenflowers_neural::layers::rnn::LSTM`'s own per-layer `Vec<Tensor<T>>`
/// storage exactly (see the [`super`] module doc):
///
/// * `weight_ih[layer]`: `[layer_input_size, 4 * hidden_size]`
/// * `weight_hh[layer]`: `[hidden_size, 4 * hidden_size]`
/// * `bias_ih[layer]`, `bias_hh[layer]` (only when `bias == true`): `[4 * hidden_size]`
///
/// When `bidirectional == true`, an entire second parallel set
/// (`weight_ih_reverse`/`weight_hh_reverse`/`bias_ih_reverse`/`bias_hh_reverse`)
/// holds the reverse direction's parameters, same per-layer shapes.
#[pyclass(name = "LSTM")]
#[derive(Debug)]
pub struct PyLSTM {
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
    /// Dropout probability for outputs of each LSTM layer except last
    pub dropout: f32,
    /// If True, becomes a bidirectional LSTM
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

impl Clone for PyLSTM {
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
/// set (see the [`super`] module doc on why zero-init is preserved exactly
/// as the pre-autograd version had it, rather than "fixed").
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
                tensor: Arc::new(Tensor::zeros(&[layer_input_size, 4 * hidden_size])),
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
                tensor: Arc::new(Tensor::zeros(&[hidden_size, 4 * hidden_size])),
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
                    tensor: Arc::new(Tensor::zeros(&[4 * hidden_size])),
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
                    tensor: Arc::new(Tensor::zeros(&[4 * hidden_size])),
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
impl PyLSTM {
    /// Create a new LSTM layer
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of expected features in input x
    /// * `hidden_size` - Number of features in hidden state h
    /// * `num_layers` - Number of recurrent layers (default: 1)
    /// * `bias` - If False, layer doesn't use bias weights (default: True)
    /// * `batch_first` - If True, input/output shape is (batch, seq, feature) (default: False)
    /// * `dropout` - Dropout probability (default: 0.0)
    /// * `bidirectional` - If True, becomes bidirectional LSTM (default: False)
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

        Ok(PyLSTM {
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

    /// Forward pass through the LSTM layer
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first
    /// * `hidden` - Optional initial hidden state (h_0, c_0)
    ///
    /// # Returns
    ///
    /// Tuple of (output, (h_n, c_n))
    #[pyo3(signature = (input, hidden=None))]
    pub fn forward(
        &self,
        py: Python<'_>,
        input: &PyTensor,
        hidden: Option<(PyTensor, PyTensor)>,
    ) -> PyResult<(PyTensor, (PyTensor, PyTensor))> {
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

        if self.bidirectional {
            return Err(PyRuntimeError::new_err(
                "LSTM: (h_n, c_n) state extraction is currently supported only for \
                 unidirectional LSTMs",
            ));
        }

        let dims = input_shape.dims().to_vec();
        let (seq_len, batch) = if self.batch_first {
            (dims[1], dims[0])
        } else {
            (dims[0], dims[1])
        };

        // Snapshot every parameter this forward pass needs exactly once,
        // marking each as a tape leaf keyed by its own stable `id()` (see the
        // `super` module doc / `PyDense::forward`'s doc for why this must go
        // through `mark_leaf_param` rather than `mark_leaf`).
        let mut w_ih_snap = Vec::with_capacity(self.num_layers);
        let mut w_hh_snap = Vec::with_capacity(self.num_layers);
        let mut b_ih_snap: Vec<Option<PyTensor>> = Vec::with_capacity(self.num_layers);
        let mut b_hh_snap: Vec<Option<PyTensor>> = Vec::with_capacity(self.num_layers);
        for l in 0..self.num_layers {
            w_ih_snap.push(snapshot_leaf(py, &self.weight_ih[l])?);
            w_hh_snap.push(snapshot_leaf(py, &self.weight_hh[l])?);
            if self.bias {
                b_ih_snap.push(Some(snapshot_leaf(py, &self.bias_ih[l])?));
                b_hh_snap.push(Some(snapshot_leaf(py, &self.bias_hh[l])?));
            } else {
                b_ih_snap.push(None);
                b_hh_snap.push(None);
            }
        }

        let init = hidden.as_ref();

        let mut layer_input = input.clone();
        // One entry per layer (NOT overwritten each iteration — `h_n`/`c_n`
        // must report every layer's final state, shaped `[num_layers,
        // batch, hidden_size]`, matching PyTorch's LSTM convention).
        let mut h_finals: Vec<PyTensor> = Vec::with_capacity(self.num_layers);
        let mut c_finals: Vec<PyTensor> = Vec::with_capacity(self.num_layers);

        for l in 0..self.num_layers {
            let feature = if l == 0 {
                self.input_size
            } else {
                self.hidden_size
            };

            let mut h = match init {
                Some((h0, _)) => slice_initial_state(h0, l, 0, 1, batch, self.hidden_size)?,
                None => zeros_state(batch, self.hidden_size),
            };
            let mut c = match init {
                Some((_, c0)) => slice_initial_state(c0, l, 0, 1, batch, self.hidden_size)?,
                None => zeros_state(batch, self.hidden_size),
            };

            let mut step_outputs = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let x_t = sequence_timestep(&layer_input, t, batch, feature, self.batch_first)?;
                let (new_h, new_c) = lstm_cell_step(
                    &x_t,
                    &h,
                    &c,
                    &w_ih_snap[l],
                    &w_hh_snap[l],
                    b_ih_snap[l].as_ref(),
                    b_hh_snap[l].as_ref(),
                    self.hidden_size,
                )?;
                h = new_h;
                c = new_c;
                step_outputs.push(h.clone());
            }

            let time_axis = if self.batch_first { 1 } else { 0 };
            layer_input = stack_on_tape(&step_outputs, time_axis)?;
            h_finals.push(h);
            c_finals.push(c);
        }

        // `num_layers >= 1` is enforced in `new()`, so the loop above always
        // runs at least once and `stack_on_tape` therefore always receives a
        // non-empty slice (it errors on empty input — see its own doc);
        // stacking all `num_layers` entries (not just the last one) is what
        // makes `h_n`/`c_n` correctly report every layer's final state.
        let h_n = stack_on_tape(&h_finals, 0)?;
        let c_n = stack_on_tape(&c_finals, 0)?;

        Ok((layer_input, (h_n, c_n)))
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
    ///
    /// Shares identity with the exact same `Py<PyParameter>` objects
    /// `forward()` marks as leaves (via [`Py::clone_ref`], never
    /// [`PyParameter::clone_param`] — see [`PyDense::parameters`]'s doc for
    /// why that distinction matters), so `.grad()`/`.set_data()` on a
    /// returned handle correctly affects (and is affected by) this layer.
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
            "LSTM(input_size={}, hidden_size={}, num_layers={}, bias={}, batch_first={}, dropout={}, bidirectional={})",
            self.input_size, self.hidden_size, self.num_layers, self.bias,
            self.batch_first, self.dropout, self.bidirectional
        )
    }
}

/// LSTM Cell
///
/// A single LSTM cell (one time step).
///
/// # Parameters (autograd)
///
/// * `weight_ih_param`: `[input_size, 4 * hidden_size]`
/// * `weight_hh_param`: `[hidden_size, 4 * hidden_size]`
///
/// Initialised via `1/sqrt(hidden_size)`-scaled `Tensor::randn`, matching the
/// pre-autograd version exactly (see the [`super`] module doc).
///
/// Unlike the pre-autograd version, `bias` is currently accepted for
/// signature compatibility but the cell does not (and previously did not —
/// `lstm_cell_step` was always called with `None, None` for the bias
/// arguments in `PyLSTMCell::forward`) allocate or apply bias parameters;
/// this preserves the exact pre-existing forward behaviour (bias-free math)
/// while only changing *how* `weight_ih`/`weight_hh` participate in
/// autograd.
#[pyclass(name = "LSTMCell")]
#[derive(Debug)]
pub struct PyLSTMCell {
    /// Number of expected features in the input
    pub input_size: usize,
    /// Number of features in the hidden state
    pub hidden_size: usize,
    /// If True, use bias weights (currently unused by `forward()` — see the
    /// struct doc)
    pub bias: bool,
    /// Input-hidden weight, stored pre-transposed as `[input_size, 4 *
    /// hidden_size]` (the orientation `lstm_cell_step`'s matmul expects) —
    /// unlike the pre-autograd version's `[4*hidden, in]` storage, which
    /// required a `transpose()` call inside every `forward()`. Storing the
    /// already-transposed shape means `forward()` needs no extra tape op
    /// (and no extra backward differentiation through a transpose) beyond
    /// the matmul itself.
    weight_ih_param: Py<PyParameter>,
    /// Hidden-hidden weight, stored pre-transposed as `[hidden_size, 4 *
    /// hidden_size]` (see `weight_ih_param`'s doc for why).
    weight_hh_param: Py<PyParameter>,
}

impl Clone for PyLSTMCell {
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
impl PyLSTMCell {
    /// Create a new LSTM cell
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
        // Stored directly in the `[in, 4*hidden]`/`[hidden, 4*hidden]`
        // orientation `lstm_cell_step` expects (see `weight_ih_param`'s
        // field doc) — the pre-autograd version stored `[4*hidden, in]` and
        // transposed inside `forward()` every call; initialising directly in
        // the needed orientation is numerically identical (same
        // distribution, same scale) since `Tensor::randn`'s entries are iid.
        let weight_ih = Tensor::randn(&[input_size, 4 * hidden_size])
            .and_then(|t| t.multiply_scalar(scale))
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to init LSTMCell weight_ih: {}", e))
            })?;
        let weight_hh = Tensor::randn(&[hidden_size, 4 * hidden_size])
            .and_then(|t| t.multiply_scalar(scale))
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to init LSTMCell weight_hh: {}", e))
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

        Ok(PyLSTMCell {
            input_size,
            hidden_size,
            bias,
            weight_ih_param,
            weight_hh_param,
        })
    }

    /// Forward pass through the LSTM cell
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (batch, input_size)
    /// * `hidden` - Optional tuple (h_0, c_0) of shape (batch, hidden_size)
    ///
    /// # Returns
    ///
    /// Tuple (h_1, c_1) of shape (batch, hidden_size)
    #[pyo3(signature = (input, hidden=None))]
    pub fn forward(
        &self,
        py: Python<'_>,
        input: &PyTensor,
        hidden: Option<(PyTensor, PyTensor)>,
    ) -> PyResult<(PyTensor, PyTensor)> {
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

        let (h_0, c_0) = match hidden {
            Some((h, c)) => (h, c),
            None => (
                zeros_state(batch_size, self.hidden_size),
                zeros_state(batch_size, self.hidden_size),
            ),
        };

        lstm_cell_step(
            input,
            &h_0,
            &c_0,
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
            "LSTMCell(input_size={}, hidden_size={}, bias={})",
            self.input_size, self.hidden_size, self.bias
        )
    }
}
