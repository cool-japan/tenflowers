//! Recurrent layers module for TenfloweRS FFI
//!
//! This module provides recurrent layer implementations including LSTM, GRU, and vanilla RNN
//! for sequence modeling, time series prediction, and NLP tasks.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;
use tenflowers_core::{Result as CoreResult, Tensor};
use tenflowers_neural::layers::rnn::{RnnNonlinearity, GRU, LSTM, RNN};
use tenflowers_neural::layers::Layer;

/// Extract a single time-step slice `[batch, feature]` from a 3-D sequence tensor.
fn sequence_timestep(
    input: &Tensor<f32>,
    t: usize,
    batch: usize,
    feature: usize,
    batch_first: bool,
) -> CoreResult<Tensor<f32>> {
    if batch_first {
        input
            .slice(&[0..batch, t..t + 1, 0..feature])?
            .squeeze(Some(&[1]))
    } else {
        input
            .slice(&[t..t + 1, 0..batch, 0..feature])?
            .squeeze(Some(&[0]))
    }
}

/// Single LSTM cell step using the `tenflowers_neural` gate convention.
///
/// `weight_ih` is `[input_size, 4 * hidden_size]`, `weight_hh` is
/// `[hidden_size, 4 * hidden_size]`, and optional bias vectors have length
/// `4 * hidden_size`. The gate order is `[input, forget, cell, output]`.
#[allow(clippy::too_many_arguments)]
fn lstm_cell_step(
    x_t: &Tensor<f32>,
    h: &Tensor<f32>,
    c: &Tensor<f32>,
    weight_ih: &Tensor<f32>,
    weight_hh: &Tensor<f32>,
    bias_ih: Option<&Tensor<f32>>,
    bias_hh: Option<&Tensor<f32>>,
    hidden_size: usize,
) -> CoreResult<(Tensor<f32>, Tensor<f32>)> {
    let gi = x_t.matmul(weight_ih)?;
    let gh = h.matmul(weight_hh)?;
    let gi = if let Some(b) = bias_ih {
        gi.add(b)?
    } else {
        gi
    };
    let gh = if let Some(b) = bias_hh {
        gh.add(b)?
    } else {
        gh
    };
    let gates = gi.add(&gh)?;

    let batch = gates.shape().dims()[0];
    let i_gate = gates.slice(&[0..batch, 0..hidden_size])?.sigmoid()?;
    let f_gate = gates
        .slice(&[0..batch, hidden_size..2 * hidden_size])?
        .sigmoid()?;
    let g_gate = gates
        .slice(&[0..batch, 2 * hidden_size..3 * hidden_size])?
        .tanh()?;
    let o_gate = gates
        .slice(&[0..batch, 3 * hidden_size..4 * hidden_size])?
        .sigmoid()?;

    let new_c = f_gate.mul(c)?.add(&i_gate.mul(&g_gate)?)?;
    let new_h = o_gate.mul(&new_c.tanh()?)?;
    Ok((new_h, new_c))
}

/// Single GRU cell step (PyTorch gate convention: reset, update, new).
///
/// `weight_ih` is `[input_size, 3 * hidden_size]`, `weight_hh` is
/// `[hidden_size, 3 * hidden_size]`.
#[allow(clippy::too_many_arguments)]
fn gru_cell_step(
    x_t: &Tensor<f32>,
    h: &Tensor<f32>,
    weight_ih: &Tensor<f32>,
    weight_hh: &Tensor<f32>,
    bias_ih: Option<&Tensor<f32>>,
    bias_hh: Option<&Tensor<f32>>,
    hidden_size: usize,
) -> CoreResult<Tensor<f32>> {
    let gi = x_t.matmul(weight_ih)?;
    let gh = h.matmul(weight_hh)?;
    let gi = if let Some(b) = bias_ih {
        gi.add(b)?
    } else {
        gi
    };
    let gh = if let Some(b) = bias_hh {
        gh.add(b)?
    } else {
        gh
    };

    let batch = gi.shape().dims()[0];
    let i_r = gi.slice(&[0..batch, 0..hidden_size])?;
    let i_z = gi.slice(&[0..batch, hidden_size..2 * hidden_size])?;
    let i_n = gi.slice(&[0..batch, 2 * hidden_size..3 * hidden_size])?;
    let h_r = gh.slice(&[0..batch, 0..hidden_size])?;
    let h_z = gh.slice(&[0..batch, hidden_size..2 * hidden_size])?;
    let h_n = gh.slice(&[0..batch, 2 * hidden_size..3 * hidden_size])?;

    let r = i_r.add(&h_r)?.sigmoid()?;
    let z = i_z.add(&h_z)?.sigmoid()?;
    let n = i_n.add(&r.mul(&h_n)?)?.tanh()?;

    // h_new = (1 - z) * n + z * h
    let one_minus_z = z.multiply_scalar(-1.0)?.add(&Tensor::from_scalar(1.0))?;
    one_minus_z.mul(&n)?.add(&z.mul(h)?)
}

/// Full unidirectional (multi-layer) LSTM forward computed from neural weights.
///
/// Returns `(output, h_n, c_n)` where `output` is the last layer's stacked
/// hidden states and `h_n`/`c_n` are `[num_layers, batch, hidden_size]`.
/// `init` optionally provides `(h_0, c_0)`, each `[num_layers, batch, hidden]`.
fn lstm_unidirectional_forward(
    input: &Tensor<f32>,
    params: &[&Tensor<f32>],
    num_layers: usize,
    hidden_size: usize,
    bias: bool,
    batch_first: bool,
    init: Option<(&Tensor<f32>, &Tensor<f32>)>,
) -> CoreResult<(Tensor<f32>, Tensor<f32>, Tensor<f32>)> {
    let dims = input.shape().dims().to_vec();
    let (seq_len, batch) = if batch_first {
        (dims[1], dims[0])
    } else {
        (dims[0], dims[1])
    };

    let mut layer_input = input.clone();
    let mut h_finals: Vec<Tensor<f32>> = Vec::with_capacity(num_layers);
    let mut c_finals: Vec<Tensor<f32>> = Vec::with_capacity(num_layers);

    for l in 0..num_layers {
        let w_ih = params[l];
        let w_hh = params[num_layers + l];
        let (b_ih, b_hh) = if bias {
            (
                Some(params[2 * num_layers + l]),
                Some(params[3 * num_layers + l]),
            )
        } else {
            (None, None)
        };

        let feature = layer_input.shape().dims()[2];

        let mut h = match init {
            Some((h0, _)) => h0
                .slice(&[l..l + 1, 0..batch, 0..hidden_size])?
                .squeeze(Some(&[0]))?,
            None => Tensor::zeros(&[batch, hidden_size]),
        };
        let mut c = match init {
            Some((_, c0)) => c0
                .slice(&[l..l + 1, 0..batch, 0..hidden_size])?
                .squeeze(Some(&[0]))?,
            None => Tensor::zeros(&[batch, hidden_size]),
        };

        let mut step_outputs: Vec<Tensor<f32>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let x_t = sequence_timestep(&layer_input, t, batch, feature, batch_first)?;
            let (new_h, new_c) = lstm_cell_step(&x_t, &h, &c, w_ih, w_hh, b_ih, b_hh, hidden_size)?;
            h = new_h;
            c = new_c;
            step_outputs.push(h.clone());
        }

        let refs: Vec<&Tensor<f32>> = step_outputs.iter().collect();
        let time_axis = if batch_first { 1 } else { 0 };
        layer_input = tenflowers_core::ops::stack(&refs, time_axis)?;
        h_finals.push(h);
        c_finals.push(c);
    }

    let h_refs: Vec<&Tensor<f32>> = h_finals.iter().collect();
    let c_refs: Vec<&Tensor<f32>> = c_finals.iter().collect();
    let h_n = tenflowers_core::ops::stack(&h_refs, 0)?;
    let c_n = tenflowers_core::ops::stack(&c_refs, 0)?;
    Ok((layer_input, h_n, c_n))
}

/// Full unidirectional (multi-layer) GRU forward computed from neural weights.
///
/// Returns `(output, h_n)` where `output` is the last layer's stacked hidden
/// states and `h_n` is `[num_layers, batch, hidden_size]`. `init` optionally
/// provides `h_0`, shaped `[num_layers, batch, hidden]`.
///
/// Structurally mirrors [`lstm_unidirectional_forward`] but carries a single
/// hidden state (no cell state) and is built on [`gru_cell_step`]. It relies on
/// the same parameter ordering the LSTM path assumes: `weight_ih[all layers]`,
/// `weight_hh[all layers]`, then (if `bias`) `bias_ih[all layers]` and
/// `bias_hh[all layers]`.
fn gru_unidirectional_forward(
    input: &Tensor<f32>,
    params: &[&Tensor<f32>],
    num_layers: usize,
    hidden_size: usize,
    bias: bool,
    batch_first: bool,
    init: Option<&Tensor<f32>>,
) -> CoreResult<(Tensor<f32>, Tensor<f32>)> {
    let dims = input.shape().dims().to_vec();
    let (seq_len, batch) = if batch_first {
        (dims[1], dims[0])
    } else {
        (dims[0], dims[1])
    };

    let mut layer_input = input.clone();
    let mut h_finals: Vec<Tensor<f32>> = Vec::with_capacity(num_layers);

    for l in 0..num_layers {
        let w_ih = params[l];
        let w_hh = params[num_layers + l];
        let (b_ih, b_hh) = if bias {
            (
                Some(params[2 * num_layers + l]),
                Some(params[3 * num_layers + l]),
            )
        } else {
            (None, None)
        };

        let feature = layer_input.shape().dims()[2];

        let mut h = match init {
            Some(h0) => h0
                .slice(&[l..l + 1, 0..batch, 0..hidden_size])?
                .squeeze(Some(&[0]))?,
            None => Tensor::zeros(&[batch, hidden_size]),
        };

        let mut step_outputs: Vec<Tensor<f32>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let x_t = sequence_timestep(&layer_input, t, batch, feature, batch_first)?;
            let new_h = gru_cell_step(&x_t, &h, w_ih, w_hh, b_ih, b_hh, hidden_size)?;
            h = new_h;
            step_outputs.push(h.clone());
        }

        let refs: Vec<&Tensor<f32>> = step_outputs.iter().collect();
        let time_axis = if batch_first { 1 } else { 0 };
        layer_input = tenflowers_core::ops::stack(&refs, time_axis)?;
        h_finals.push(h);
    }

    let h_refs: Vec<&Tensor<f32>> = h_finals.iter().collect();
    let h_n = tenflowers_core::ops::stack(&h_refs, 0)?;
    Ok((layer_input, h_n))
}

/// Convert a validated `nonlinearity` string into the neural-layer enum.
///
/// Callers validate the string up front (only `"tanh"` or `"relu"` reach here),
/// so any value other than `"relu"` maps to [`RnnNonlinearity::Tanh`].
fn parse_nonlinearity(s: &str) -> RnnNonlinearity {
    if s == "relu" {
        RnnNonlinearity::Relu
    } else {
        RnnNonlinearity::Tanh
    }
}

/// Derive the last-layer final hidden state `[num_directions, batch, hidden]`
/// from a recurrent layer's output sequence.
fn final_hidden_state(
    output: &Tensor<f32>,
    batch_first: bool,
    hidden_size: usize,
    bidirectional: bool,
) -> CoreResult<Tensor<f32>> {
    let dims = output.shape().dims().to_vec();
    let (seq_len, batch) = if batch_first {
        (dims[1], dims[0])
    } else {
        (dims[0], dims[1])
    };

    // Forward direction: last time-step, first `hidden_size` features.
    let fwd = if batch_first {
        let slice = output.slice(&[0..batch, seq_len - 1..seq_len, 0..hidden_size])?;
        tenflowers_core::ops::manipulation::transpose_axes(&slice, Some(&[1, 0, 2]))?
    } else {
        output.slice(&[seq_len - 1..seq_len, 0..batch, 0..hidden_size])?
    };

    if !bidirectional {
        return Ok(fwd);
    }

    // Backward direction: first time-step, second half of the features.
    let bwd = if batch_first {
        let slice = output.slice(&[0..batch, 0..1, hidden_size..2 * hidden_size])?;
        tenflowers_core::ops::manipulation::transpose_axes(&slice, Some(&[1, 0, 2]))?
    } else {
        output.slice(&[0..1, 0..batch, hidden_size..2 * hidden_size])?
    };

    tenflowers_core::ops::concat(&[&fwd, &bwd], 0)
}

/// Long Short-Term Memory (LSTM) Layer
///
/// Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.
/// LSTMs are excellent at capturing long-term dependencies in sequences.
#[pyclass(name = "LSTM")]
#[derive(Debug, Clone)]
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
    /// Weight matrices for each layer
    pub weights: Vec<Option<Tensor<f32>>>,
    /// Real neural LSTM implementation backing this layer
    inner: LSTM<f32>,
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
    pub fn new(
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

        // Initialize weights for each layer
        // LSTM has 4 gates (input, forget, cell, output) per direction
        let directions = if bidirectional { 2 } else { 1 };
        let mut weights = Vec::new();

        for layer in 0..num_layers {
            let layer_input_size = if layer == 0 {
                input_size
            } else {
                hidden_size * directions
            };

            // Weight matrix shape: (4 * hidden_size, layer_input_size + hidden_size)
            // 4 gates: input, forget, cell, output
            let weight_ih_shape = vec![4 * hidden_size, layer_input_size];
            let weight_hh_shape = vec![4 * hidden_size, hidden_size];

            weights.push(Some(Tensor::zeros(&weight_ih_shape)));
            weights.push(Some(Tensor::zeros(&weight_hh_shape)));

            if bidirectional {
                weights.push(Some(Tensor::zeros(&weight_ih_shape)));
                weights.push(Some(Tensor::zeros(&weight_hh_shape)));
            }
        }

        let inner = LSTM::<f32>::new(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize LSTM: {}", e)))?;

        Ok(PyLSTM {
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            weights,
            inner,
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

        // Optional initial (h_0, c_0) states.
        let init = hidden
            .as_ref()
            .map(|(h0, c0)| (h0.tensor.as_ref(), c0.tensor.as_ref()));

        // Real weights are sourced from the backing neural LSTM layer.
        let params = self.inner.parameters();
        let expected = if self.bias {
            4 * self.num_layers
        } else {
            2 * self.num_layers
        };
        if params.len() < expected {
            return Err(PyRuntimeError::new_err("LSTM: weights not initialized"));
        }

        let (output, h_n, c_n) = lstm_unidirectional_forward(
            &input.tensor,
            &params,
            self.num_layers,
            self.hidden_size,
            self.bias,
            self.batch_first,
            init,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("LSTM forward failed: {}", e)))?;

        Ok((
            PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            },
            (
                PyTensor {
                    tensor: Arc::new(h_n),
                    requires_grad: input.requires_grad,
                    is_pinned: input.is_pinned,
                },
                PyTensor {
                    tensor: Arc::new(c_n),
                    requires_grad: input.requires_grad,
                    is_pinned: input.is_pinned,
                },
            ),
        ))
    }

    /// Reset layer parameters
    pub fn reset_parameters(&mut self) -> PyResult<()> {
        // Reinitialize all weights
        self.weights.clear();

        let directions = if self.bidirectional { 2 } else { 1 };

        for layer in 0..self.num_layers {
            let layer_input_size = if layer == 0 {
                self.input_size
            } else {
                self.hidden_size * directions
            };

            let weight_ih_shape = vec![4 * self.hidden_size, layer_input_size];
            let weight_hh_shape = vec![4 * self.hidden_size, self.hidden_size];

            self.weights.push(Some(Tensor::zeros(&weight_ih_shape)));
            self.weights.push(Some(Tensor::zeros(&weight_hh_shape)));

            if self.bidirectional {
                self.weights.push(Some(Tensor::zeros(&weight_ih_shape)));
                self.weights.push(Some(Tensor::zeros(&weight_hh_shape)));
            }
        }

        self.inner = LSTM::<f32>::new(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.bias,
            self.batch_first,
            self.dropout,
            self.bidirectional,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to reset LSTM: {}", e)))?;

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "LSTM(input_size={}, hidden_size={}, num_layers={}, bias={}, batch_first={}, dropout={}, bidirectional={})",
            self.input_size, self.hidden_size, self.num_layers, self.bias,
            self.batch_first, self.dropout, self.bidirectional
        )
    }
}

/// Gated Recurrent Unit (GRU) Layer
///
/// Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
/// GRUs are similar to LSTMs but with fewer parameters.
#[pyclass(name = "GRU")]
#[derive(Debug, Clone)]
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
    /// Weight matrices for each layer
    pub weights: Vec<Option<Tensor<f32>>>,
    /// Real neural GRU implementation backing this layer
    inner: GRU<f32>,
}

#[pymethods]
impl PyGRU {
    /// Create a new GRU layer
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers=None, bias=None, batch_first=None, dropout=None, bidirectional=None))]
    pub fn new(
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

        // Initialize weights for each layer
        // GRU has 3 gates (reset, update, new) per direction
        let directions = if bidirectional { 2 } else { 1 };
        let mut weights = Vec::new();

        for layer in 0..num_layers {
            let layer_input_size = if layer == 0 {
                input_size
            } else {
                hidden_size * directions
            };

            // Weight matrix shape: (3 * hidden_size, layer_input_size + hidden_size)
            let weight_ih_shape = vec![3 * hidden_size, layer_input_size];
            let weight_hh_shape = vec![3 * hidden_size, hidden_size];

            weights.push(Some(Tensor::zeros(&weight_ih_shape)));
            weights.push(Some(Tensor::zeros(&weight_hh_shape)));

            if bidirectional {
                weights.push(Some(Tensor::zeros(&weight_ih_shape)));
                weights.push(Some(Tensor::zeros(&weight_hh_shape)));
            }
        }

        let inner = GRU::<f32>::new(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize GRU: {}", e)))?;

        Ok(PyGRU {
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            weights,
            inner,
        })
    }

    /// Forward pass through the GRU layer
    #[pyo3(signature = (input, hidden=None))]
    pub fn forward(
        &self,
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

        if self.bidirectional {
            // Bidirectional GRU: threading an explicit initial hidden state is not
            // supported here (matching LSTM's analogous unidirectional-only limit).
            if hidden.is_some() {
                return Err(PyRuntimeError::new_err(
                    "GRU: providing an explicit initial hidden state is currently \
                     supported only for unidirectional GRUs",
                ));
            }

            // Real output sequence from the backing neural GRU layer, which
            // internally handles the bidirectional forward pass.
            let output = Layer::forward(&self.inner, &input.tensor)
                .map_err(|e| PyRuntimeError::new_err(format!("GRU forward failed: {}", e)))?;

            // h_n is the final hidden state derived from the real output sequence.
            let h_n = final_hidden_state(
                &output,
                self.batch_first,
                self.hidden_size,
                self.bidirectional,
            )
            .map_err(|e| {
                PyRuntimeError::new_err(format!("GRU hidden-state extraction failed: {}", e))
            })?;

            return Ok((
                PyTensor {
                    tensor: Arc::new(output),
                    requires_grad: input.requires_grad,
                    is_pinned: input.is_pinned,
                },
                PyTensor {
                    tensor: Arc::new(h_n),
                    requires_grad: input.requires_grad,
                    is_pinned: input.is_pinned,
                },
            ));
        }

        // Unidirectional GRU: thread the optional initial hidden state through the
        // real neural weights, mirroring PyLSTM::forward.
        let init = hidden.as_ref().map(|h| h.tensor.as_ref());

        let params = self.inner.parameters();
        let expected = if self.bias {
            4 * self.num_layers
        } else {
            2 * self.num_layers
        };
        if params.len() < expected {
            return Err(PyRuntimeError::new_err("GRU: weights not initialized"));
        }

        let (output, h_n) = gru_unidirectional_forward(
            &input.tensor,
            &params,
            self.num_layers,
            self.hidden_size,
            self.bias,
            self.batch_first,
            init,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("GRU forward failed: {}", e)))?;

        Ok((
            PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            },
            PyTensor {
                tensor: Arc::new(h_n),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            },
        ))
    }

    /// Reset layer parameters
    pub fn reset_parameters(&mut self) -> PyResult<()> {
        self.weights.clear();

        let directions = if self.bidirectional { 2 } else { 1 };

        for layer in 0..self.num_layers {
            let layer_input_size = if layer == 0 {
                self.input_size
            } else {
                self.hidden_size * directions
            };

            let weight_ih_shape = vec![3 * self.hidden_size, layer_input_size];
            let weight_hh_shape = vec![3 * self.hidden_size, self.hidden_size];

            self.weights.push(Some(Tensor::zeros(&weight_ih_shape)));
            self.weights.push(Some(Tensor::zeros(&weight_hh_shape)));

            if self.bidirectional {
                self.weights.push(Some(Tensor::zeros(&weight_ih_shape)));
                self.weights.push(Some(Tensor::zeros(&weight_hh_shape)));
            }
        }

        self.inner = GRU::<f32>::new(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.bias,
            self.batch_first,
            self.dropout,
            self.bidirectional,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to reset GRU: {}", e)))?;

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "GRU(input_size={}, hidden_size={}, num_layers={}, bias={}, batch_first={}, dropout={}, bidirectional={})",
            self.input_size, self.hidden_size, self.num_layers, self.bias,
            self.batch_first, self.dropout, self.bidirectional
        )
    }
}

/// Vanilla RNN Layer
///
/// Applies a multi-layer Elman RNN with tanh or ReLU non-linearity to an input sequence.
#[pyclass(name = "RNN")]
#[derive(Debug, Clone)]
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
    /// Weight matrices for each layer
    pub weights: Vec<Option<Tensor<f32>>>,
    /// Real neural RNN implementation backing this layer
    inner: RNN<f32>,
}

#[pymethods]
impl PyRNN {
    /// Create a new RNN layer
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers=None, nonlinearity=None, bias=None, batch_first=None, dropout=None, bidirectional=None))]
    pub fn new(
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

        // Initialize weights for each layer
        let directions = if bidirectional { 2 } else { 1 };
        let mut weights = Vec::new();

        for layer in 0..num_layers {
            let layer_input_size = if layer == 0 {
                input_size
            } else {
                hidden_size * directions
            };

            let weight_ih_shape = vec![hidden_size, layer_input_size];
            let weight_hh_shape = vec![hidden_size, hidden_size];

            weights.push(Some(Tensor::zeros(&weight_ih_shape)));
            weights.push(Some(Tensor::zeros(&weight_hh_shape)));

            if bidirectional {
                weights.push(Some(Tensor::zeros(&weight_ih_shape)));
                weights.push(Some(Tensor::zeros(&weight_hh_shape)));
            }
        }

        let inner = RNN::<f32>::new_with_nonlinearity(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            parse_nonlinearity(&nonlinearity),
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize RNN: {}", e)))?;

        Ok(PyRNN {
            input_size,
            hidden_size,
            num_layers,
            nonlinearity,
            bias,
            batch_first,
            dropout,
            bidirectional,
            weights,
            inner,
        })
    }

    /// Forward pass through the RNN layer
    #[pyo3(signature = (input, hidden=None))]
    pub fn forward(
        &self,
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

        // Real output and final hidden state from the backing neural RNN layer.
        let init = hidden.as_ref().map(|h| h.tensor.as_ref());
        let (output, h_n) = self
            .inner
            .forward_with_hidden(&input.tensor, init)
            .map_err(|e| PyRuntimeError::new_err(format!("RNN forward failed: {}", e)))?;

        Ok((
            PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            },
            PyTensor {
                tensor: Arc::new(h_n),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            },
        ))
    }

    /// Reset layer parameters
    pub fn reset_parameters(&mut self) -> PyResult<()> {
        self.weights.clear();

        let directions = if self.bidirectional { 2 } else { 1 };

        for layer in 0..self.num_layers {
            let layer_input_size = if layer == 0 {
                self.input_size
            } else {
                self.hidden_size * directions
            };

            let weight_ih_shape = vec![self.hidden_size, layer_input_size];
            let weight_hh_shape = vec![self.hidden_size, self.hidden_size];

            self.weights.push(Some(Tensor::zeros(&weight_ih_shape)));
            self.weights.push(Some(Tensor::zeros(&weight_hh_shape)));

            if self.bidirectional {
                self.weights.push(Some(Tensor::zeros(&weight_ih_shape)));
                self.weights.push(Some(Tensor::zeros(&weight_hh_shape)));
            }
        }

        self.inner = RNN::<f32>::new_with_nonlinearity(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.bias,
            self.batch_first,
            self.dropout,
            self.bidirectional,
            parse_nonlinearity(&self.nonlinearity),
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to reset RNN: {}", e)))?;

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "RNN(input_size={}, hidden_size={}, num_layers={}, nonlinearity='{}', bias={}, batch_first={}, dropout={}, bidirectional={})",
            self.input_size, self.hidden_size, self.num_layers, self.nonlinearity,
            self.bias, self.batch_first, self.dropout, self.bidirectional
        )
    }
}

/// LSTM Cell
///
/// A single LSTM cell (one time step).
#[pyclass(name = "LSTMCell")]
#[derive(Debug, Clone)]
pub struct PyLSTMCell {
    /// Number of expected features in the input
    pub input_size: usize,
    /// Number of features in the hidden state
    pub hidden_size: usize,
    /// If True, use bias weights
    pub bias: bool,
    /// Input-hidden weight
    pub weight_ih: Option<Tensor<f32>>,
    /// Hidden-hidden weight
    pub weight_hh: Option<Tensor<f32>>,
}

#[pymethods]
impl PyLSTMCell {
    /// Create a new LSTM cell
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias=None))]
    pub fn new(input_size: usize, hidden_size: usize, bias: Option<bool>) -> PyResult<Self> {
        let bias = bias.unwrap_or(true);

        if input_size == 0 {
            return Err(PyValueError::new_err("input_size must be positive"));
        }
        if hidden_size == 0 {
            return Err(PyValueError::new_err("hidden_size must be positive"));
        }

        let scale = 1.0_f32 / (hidden_size as f32).sqrt();
        let weight_ih = Tensor::randn(&[4 * hidden_size, input_size])
            .and_then(|t| t.multiply_scalar(scale))
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to init LSTMCell weight_ih: {}", e))
            })?;
        let weight_hh = Tensor::randn(&[4 * hidden_size, hidden_size])
            .and_then(|t| t.multiply_scalar(scale))
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to init LSTMCell weight_hh: {}", e))
            })?;

        Ok(PyLSTMCell {
            input_size,
            hidden_size,
            bias,
            weight_ih: Some(weight_ih),
            weight_hh: Some(weight_hh),
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

        let weight_ih = self
            .weight_ih
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("LSTMCell: weight_ih not initialized"))?;
        let weight_hh = self
            .weight_hh
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("LSTMCell: weight_hh not initialized"))?;

        // Stored weights are [4*hidden, in]; transpose to the [in, 4*hidden]
        // orientation expected by the gate computation.
        let weight_ih_t = weight_ih
            .transpose()
            .map_err(|e| PyRuntimeError::new_err(format!("LSTMCell transpose failed: {}", e)))?;
        let weight_hh_t = weight_hh
            .transpose()
            .map_err(|e| PyRuntimeError::new_err(format!("LSTMCell transpose failed: {}", e)))?;

        let (h_0, c_0) = match &hidden {
            Some((h, c)) => (h.tensor.as_ref().clone(), c.tensor.as_ref().clone()),
            None => (
                Tensor::zeros(&[batch_size, self.hidden_size]),
                Tensor::zeros(&[batch_size, self.hidden_size]),
            ),
        };

        let (h_1, c_1) = lstm_cell_step(
            &input.tensor,
            &h_0,
            &c_0,
            &weight_ih_t,
            &weight_hh_t,
            None,
            None,
            self.hidden_size,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("LSTMCell forward failed: {}", e)))?;

        Ok((
            PyTensor {
                tensor: Arc::new(h_1),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            },
            PyTensor {
                tensor: Arc::new(c_1),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            },
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "LSTMCell(input_size={}, hidden_size={}, bias={})",
            self.input_size, self.hidden_size, self.bias
        )
    }
}

/// GRU Cell
///
/// A single GRU cell (one time step).
#[pyclass(name = "GRUCell")]
#[derive(Debug, Clone)]
pub struct PyGRUCell {
    /// Number of expected features in the input
    pub input_size: usize,
    /// Number of features in the hidden state
    pub hidden_size: usize,
    /// If True, use bias weights
    pub bias: bool,
    /// Input-hidden weight
    pub weight_ih: Option<Tensor<f32>>,
    /// Hidden-hidden weight
    pub weight_hh: Option<Tensor<f32>>,
}

#[pymethods]
impl PyGRUCell {
    /// Create a new GRU cell
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias=None))]
    pub fn new(input_size: usize, hidden_size: usize, bias: Option<bool>) -> PyResult<Self> {
        let bias = bias.unwrap_or(true);

        if input_size == 0 {
            return Err(PyValueError::new_err("input_size must be positive"));
        }
        if hidden_size == 0 {
            return Err(PyValueError::new_err("hidden_size must be positive"));
        }

        let scale = 1.0_f32 / (hidden_size as f32).sqrt();
        let weight_ih = Tensor::randn(&[3 * hidden_size, input_size])
            .and_then(|t| t.multiply_scalar(scale))
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to init GRUCell weight_ih: {}", e))
            })?;
        let weight_hh = Tensor::randn(&[3 * hidden_size, hidden_size])
            .and_then(|t| t.multiply_scalar(scale))
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to init GRUCell weight_hh: {}", e))
            })?;

        Ok(PyGRUCell {
            input_size,
            hidden_size,
            bias,
            weight_ih: Some(weight_ih),
            weight_hh: Some(weight_hh),
        })
    }

    /// Forward pass through the GRU cell
    #[pyo3(signature = (input, hidden=None))]
    pub fn forward(&self, input: &PyTensor, hidden: Option<PyTensor>) -> PyResult<PyTensor> {
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

        let weight_ih = self
            .weight_ih
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("GRUCell: weight_ih not initialized"))?;
        let weight_hh = self
            .weight_hh
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("GRUCell: weight_hh not initialized"))?;

        // Stored weights are [3*hidden, in]; transpose to the [in, 3*hidden]
        // orientation expected by the gate computation.
        let weight_ih_t = weight_ih
            .transpose()
            .map_err(|e| PyRuntimeError::new_err(format!("GRUCell transpose failed: {}", e)))?;
        let weight_hh_t = weight_hh
            .transpose()
            .map_err(|e| PyRuntimeError::new_err(format!("GRUCell transpose failed: {}", e)))?;

        let h_0 = match &hidden {
            Some(h) => h.tensor.as_ref().clone(),
            None => Tensor::zeros(&[batch_size, self.hidden_size]),
        };

        let h_1 = gru_cell_step(
            &input.tensor,
            &h_0,
            &weight_ih_t,
            &weight_hh_t,
            None,
            None,
            self.hidden_size,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("GRUCell forward failed: {}", e)))?;

        Ok(PyTensor {
            tensor: Arc::new(h_1),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "GRUCell(input_size={}, hidden_size={}, bias={})",
            self.input_size, self.hidden_size, self.bias
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(data: Vec<f32>, shape: &[usize]) -> PyTensor {
        let tensor = Tensor::from_vec(data, shape).expect("tensor construction");
        PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: false,
            is_pinned: false,
        }
    }

    /// Deterministic, non-constant test data so the recurrences cannot
    /// collapse to a trivially zero output.
    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * 0.1 - 1.0).collect()
    }

    #[test]
    fn lstm_forward_is_real() {
        let lstm = PyLSTM::new(4, 5, None, None, None, None, None).expect("lstm");
        let input = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);
        let (output, (h_n, c_n)) = lstm.forward(&input, None).expect("forward");

        assert_eq!(output.tensor.shape().dims().to_vec(), vec![3, 2, 5]);
        assert_eq!(h_n.tensor.shape().dims().to_vec(), vec![1, 2, 5]);
        assert_eq!(c_n.tensor.shape().dims().to_vec(), vec![1, 2, 5]);

        let out_vec = output.tensor.to_vec().expect("vec");
        assert!(
            out_vec.iter().any(|&v| v != 0.0),
            "LSTM output must not be all zeros"
        );
    }

    #[test]
    fn gru_forward_is_real() {
        let gru = PyGRU::new(4, 5, None, None, None, None, None).expect("gru");
        let input = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);
        let (output, h_n) = gru.forward(&input, None).expect("forward");

        assert_eq!(output.tensor.shape().dims().to_vec(), vec![3, 2, 5]);
        assert_eq!(h_n.tensor.shape().dims().to_vec(), vec![1, 2, 5]);

        let out_vec = output.tensor.to_vec().expect("vec");
        assert!(
            out_vec.iter().any(|&v| v != 0.0),
            "GRU output must not be all zeros"
        );
    }

    #[test]
    fn rnn_forward_is_real() {
        let rnn = PyRNN::new(4, 5, None, None, None, None, None, None).expect("rnn");
        let input = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);
        let (output, h_n) = rnn.forward(&input, None).expect("forward");

        assert_eq!(output.tensor.shape().dims().to_vec(), vec![3, 2, 5]);
        assert_eq!(h_n.tensor.shape().dims().to_vec(), vec![1, 2, 5]);

        let out_vec = output.tensor.to_vec().expect("vec");
        assert!(
            out_vec.iter().any(|&v| v != 0.0),
            "RNN output must not be all zeros"
        );
    }

    #[test]
    fn lstm_cell_forward_is_real() {
        let cell = PyLSTMCell::new(4, 5, None).expect("cell");
        let input = make_tensor(ramp(2 * 4), &[2, 4]);
        let (h1, c1) = cell.forward(&input, None).expect("forward");

        assert_eq!(h1.tensor.shape().dims().to_vec(), vec![2, 5]);
        assert_eq!(c1.tensor.shape().dims().to_vec(), vec![2, 5]);

        let h_vec = h1.tensor.to_vec().expect("vec");
        assert!(
            h_vec.iter().any(|&v| v != 0.0),
            "LSTMCell output must not be all zeros"
        );
    }

    #[test]
    fn gru_cell_forward_is_real() {
        let cell = PyGRUCell::new(4, 5, None).expect("cell");
        let input = make_tensor(ramp(2 * 4), &[2, 4]);
        let h1 = cell.forward(&input, None).expect("forward");

        assert_eq!(h1.tensor.shape().dims().to_vec(), vec![2, 5]);

        let h_vec = h1.tensor.to_vec().expect("vec");
        assert!(
            h_vec.iter().any(|&v| v != 0.0),
            "GRUCell output must not be all zeros"
        );
    }

    // ---- Part 1: GRU explicit initial hidden state ----

    #[test]
    fn gru_forward_threads_initial_hidden_state() {
        let gru = PyGRU::new(4, 5, None, None, None, None, None).expect("gru");
        let input = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);

        let (out_none, _) = gru.forward(&input, None).expect("forward none");

        // A nonzero initial hidden state [num_layers = 1, batch = 2, hidden = 5].
        let h0 = make_tensor(ramp(2 * 5), &[1, 2, 5]);
        let (out_h0, _) = gru.forward(&input, Some(h0)).expect("forward h0");

        let a = out_none.tensor.to_vec().expect("vec");
        let b = out_h0.tensor.to_vec().expect("vec");
        assert_eq!(a.len(), b.len());
        assert!(
            a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
            "an explicit nonzero h_0 must change the GRU output"
        );
    }

    #[test]
    fn gru_forward_matches_single_cell_step() {
        let gru = PyGRU::new(4, 5, None, None, None, None, None).expect("gru");
        // seq_len = 1 so the entire forward reduces to one gru_cell_step call.
        let input = make_tensor(ramp(2 * 4), &[1, 2, 4]);
        let h0 = make_tensor(ramp(2 * 5), &[1, 2, 5]);

        let (output, h_n) = gru.forward(&input, Some(h0.clone())).expect("forward");
        assert_eq!(output.tensor.shape().dims().to_vec(), vec![1, 2, 5]);
        assert_eq!(h_n.tensor.shape().dims().to_vec(), vec![1, 2, 5]);

        // Recompute the expected single step directly from the real neural weights.
        let params = gru.inner.parameters();
        let num_layers = gru.num_layers; // 1
        let w_ih = params[0];
        let w_hh = params[num_layers];
        let b_ih = Some(params[2 * num_layers]);
        let b_hh = Some(params[3 * num_layers]);

        let x_t = sequence_timestep(input.tensor.as_ref(), 0, 2, 4, false).expect("x_t");
        let h_prev = h0
            .tensor
            .slice(&[0..1, 0..2, 0..5])
            .expect("slice")
            .squeeze(Some(&[0]))
            .expect("squeeze");

        let expected =
            gru_cell_step(&x_t, &h_prev, w_ih, w_hh, b_ih, b_hh, 5).expect("gru_cell_step");
        let exp = expected.to_vec().expect("vec");
        let got = output.tensor.to_vec().expect("vec");
        assert_eq!(exp.len(), got.len());
        for (e, g) in exp.iter().zip(got.iter()) {
            assert!(
                (e - g).abs() < 1e-5,
                "GRU forward must match a single gru_cell_step (expected {e}, got {g})"
            );
        }
    }

    #[test]
    fn gru_bidirectional_forward_none_still_works() {
        // Regression: unchanged bidirectional path (hidden = None) still succeeds.
        let gru = PyGRU::new(4, 5, None, None, None, None, Some(true)).expect("gru");
        let input = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);
        let (output, h_n) = gru.forward(&input, None).expect("bidirectional forward");
        // Bidirectional doubles the output feature dim.
        assert_eq!(output.tensor.shape().dims().to_vec(), vec![3, 2, 10]);
        assert_eq!(h_n.tensor.shape().dims().to_vec(), vec![2, 2, 5]);
        let out = output.tensor.to_vec().expect("vec");
        assert!(
            out.iter().any(|&v| v != 0.0),
            "output must not be all zeros"
        );
    }

    #[test]
    fn gru_bidirectional_forward_with_hidden_errors() {
        // Regression: the bidirectional guard is preserved (only reworded).
        let gru = PyGRU::new(4, 5, None, None, None, None, Some(true)).expect("gru");
        let input = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);
        let h0 = make_tensor(ramp(2 * 2 * 5), &[2, 2, 5]);
        let result = gru.forward(&input, Some(h0));
        assert!(
            result.is_err(),
            "bidirectional GRU with explicit hidden must still error"
        );
    }

    // ---- Part 2: RNN nonlinearity ----

    #[test]
    fn rnn_relu_forward_succeeds() {
        // Previously this constructed fine but failed on the first forward() call.
        let rnn =
            PyRNN::new(4, 5, None, Some("relu".to_string()), None, None, None, None).expect("rnn");
        let input = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);
        let (output, h_n) = rnn
            .forward(&input, None)
            .expect("relu forward must succeed");
        assert_eq!(output.tensor.shape().dims().to_vec(), vec![3, 2, 5]);
        assert_eq!(h_n.tensor.shape().dims().to_vec(), vec![1, 2, 5]);
        let out = output.tensor.to_vec().expect("vec");
        assert!(
            out.iter().all(|&v| v >= 0.0),
            "relu RNN output must be non-negative"
        );
    }

    #[test]
    fn rnn_relu_zeroes_negative_preactivations() {
        // input_size = 1, hidden_size = 2, single layer, single timestep, relu.
        let mut rnn = PyRNN::new(
            1,
            2,
            None,
            Some("relu".to_string()),
            None,
            Some(true),
            None,
            None,
        )
        .expect("rnn");

        {
            // Order for 1 layer with bias: [w_ih, w_hh, b_ih, b_hh].
            let mut params = rnn.inner.parameters_mut();
            *params[0] = Tensor::from_vec(vec![1.0, 1.0], &[1, 2]).expect("w_ih");
            *params[1] = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]).expect("w_hh");
            *params[2] = Tensor::from_vec(vec![-2.0, 0.5], &[2]).expect("b_ih");
            *params[3] = Tensor::from_vec(vec![0.0, 0.0], &[2]).expect("b_hh");
        }

        // batch_first input [batch = 1, seq = 1, feat = 1]; h_0 defaults to zeros.
        let input = make_tensor(vec![1.0], &[1, 1, 1]);
        let (output, _h_n) = rnn.forward(&input, None).expect("relu forward");

        let out = output.tensor.to_vec().expect("vec");
        // Pre-activations are [-1.0, 1.5]; relu zeroes the negative entry exactly,
        // whereas tanh(-1.0) ~= -0.76 would be strictly negative.
        assert_eq!(out.len(), 2);
        assert!(
            (out[0] - 0.0).abs() < 1e-6,
            "negative pre-activation must relu to 0, got {}",
            out[0]
        );
        assert!(
            (out[1] - 1.5).abs() < 1e-5,
            "positive pre-activation must pass through relu, got {}",
            out[1]
        );
    }

    #[test]
    fn rnn_bidirectional_relu_is_nonnegative() {
        // Non-negativity across the whole bidirectional output confirms BOTH the
        // forward and the reverse call sites use relu (tanh could be negative).
        let rnn = PyRNN::new(
            4,
            5,
            None,
            Some("relu".to_string()),
            None,
            None,
            None,
            Some(true),
        )
        .expect("rnn");
        let input = make_tensor(ramp(3 * 2 * 4), &[3, 2, 4]);
        let (output, h_n) = rnn.forward(&input, None).expect("bi relu forward");
        // Bidirectional doubles the output feature dim.
        assert_eq!(output.tensor.shape().dims().to_vec(), vec![3, 2, 10]);
        assert_eq!(h_n.tensor.shape().dims().to_vec(), vec![2, 2, 5]);
        let out = output.tensor.to_vec().expect("vec");
        assert!(
            out.iter().all(|&v| v >= 0.0),
            "bidirectional relu output must be non-negative everywhere"
        );
    }
}
