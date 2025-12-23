//! Recurrent layers module for TenfloweRS FFI
//!
//! This module provides recurrent layer implementations including LSTM, GRU, and vanilla RNN
//! for sequence modeling, time series prediction, and NLP tasks.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tenflowers_core::Tensor;

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

        Ok(PyLSTM {
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            weights,
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

        let (seq_len, batch_size, input_dim) = if self.batch_first {
            (input_shape[1], input_shape[0], input_shape[2])
        } else {
            (input_shape[0], input_shape[1], input_shape[2])
        };

        if input_dim != self.input_size {
            return Err(PyValueError::new_err(format!(
                "Expected input_size={}, got {}",
                self.input_size, input_dim
            )));
        }

        let directions = if self.bidirectional { 2 } else { 1 };

        // Output shape
        let output_shape = if self.batch_first {
            vec![batch_size, seq_len, self.hidden_size * directions]
        } else {
            vec![seq_len, batch_size, self.hidden_size * directions]
        };

        // Hidden and cell state shapes
        let hidden_shape = vec![self.num_layers * directions, batch_size, self.hidden_size];

        // For now, return placeholder tensors
        let output = Tensor::zeros(&output_shape);
        let h_n = Tensor::zeros(&hidden_shape);
        let c_n = Tensor::zeros(&hidden_shape);

        Ok((
            PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad,
                is_pinned: false,
            },
            (
                PyTensor {
                    tensor: Arc::new(h_n),
                    requires_grad: input.requires_grad,
                    is_pinned: false,
                },
                PyTensor {
                    tensor: Arc::new(c_n),
                    requires_grad: input.requires_grad,
                    is_pinned: false,
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

        Ok(PyGRU {
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            weights,
        })
    }

    /// Forward pass through the GRU layer
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

        let (seq_len, batch_size, input_dim) = if self.batch_first {
            (input_shape[1], input_shape[0], input_shape[2])
        } else {
            (input_shape[0], input_shape[1], input_shape[2])
        };

        if input_dim != self.input_size {
            return Err(PyValueError::new_err(format!(
                "Expected input_size={}, got {}",
                self.input_size, input_dim
            )));
        }

        let directions = if self.bidirectional { 2 } else { 1 };

        // Output shape
        let output_shape = if self.batch_first {
            vec![batch_size, seq_len, self.hidden_size * directions]
        } else {
            vec![seq_len, batch_size, self.hidden_size * directions]
        };

        // Hidden state shape
        let hidden_shape = vec![self.num_layers * directions, batch_size, self.hidden_size];

        // For now, return placeholder tensors
        let output = Tensor::zeros(&output_shape);
        let h_n = Tensor::zeros(&hidden_shape);

        Ok((
            PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad,
                is_pinned: false,
            },
            PyTensor {
                tensor: Arc::new(h_n),
                requires_grad: input.requires_grad,
                is_pinned: false,
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
        })
    }

    /// Forward pass through the RNN layer
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

        let (seq_len, batch_size, input_dim) = if self.batch_first {
            (input_shape[1], input_shape[0], input_shape[2])
        } else {
            (input_shape[0], input_shape[1], input_shape[2])
        };

        if input_dim != self.input_size {
            return Err(PyValueError::new_err(format!(
                "Expected input_size={}, got {}",
                self.input_size, input_dim
            )));
        }

        let directions = if self.bidirectional { 2 } else { 1 };

        // Output shape
        let output_shape = if self.batch_first {
            vec![batch_size, seq_len, self.hidden_size * directions]
        } else {
            vec![seq_len, batch_size, self.hidden_size * directions]
        };

        // Hidden state shape
        let hidden_shape = vec![self.num_layers * directions, batch_size, self.hidden_size];

        // For now, return placeholder tensors
        let output = Tensor::zeros(&output_shape);
        let h_n = Tensor::zeros(&hidden_shape);

        Ok((
            PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad,
                is_pinned: false,
            },
            PyTensor {
                tensor: Arc::new(h_n),
                requires_grad: input.requires_grad,
                is_pinned: false,
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

        let weight_ih = Tensor::zeros(&[4 * hidden_size, input_size]);
        let weight_hh = Tensor::zeros(&[4 * hidden_size, hidden_size]);

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

        let batch_size = input_shape[0];

        // For now, return placeholder tensors
        let h_1 = Tensor::zeros(&[batch_size, self.hidden_size]);
        let c_1 = Tensor::zeros(&[batch_size, self.hidden_size]);

        Ok((
            PyTensor {
                tensor: Arc::new(h_1),
                requires_grad: input.requires_grad,
                is_pinned: false,
            },
            PyTensor {
                tensor: Arc::new(c_1),
                requires_grad: input.requires_grad,
                is_pinned: false,
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

        let weight_ih = Tensor::zeros(&[3 * hidden_size, input_size]);
        let weight_hh = Tensor::zeros(&[3 * hidden_size, hidden_size]);

        Ok(PyGRUCell {
            input_size,
            hidden_size,
            bias,
            weight_ih: Some(weight_ih),
            weight_hh: Some(weight_hh),
        })
    }

    /// Forward pass through the GRU cell
    pub fn forward(&self, input: &PyTensor, hidden: Option<PyTensor>) -> PyResult<PyTensor> {
        let input_shape = input.tensor.shape();

        if input_shape.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "Expected 2D input (batch, input_size), got {}D",
                input_shape.len()
            )));
        }

        let batch_size = input_shape[0];

        // For now, return placeholder tensor
        let h_1 = Tensor::zeros(&[batch_size, self.hidden_size]);

        Ok(PyTensor {
            tensor: Arc::new(h_1),
            requires_grad: input.requires_grad,
            is_pinned: false,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "GRUCell(input_size={}, hidden_size={}, bias={})",
            self.input_size, self.hidden_size, self.bias
        )
    }
}
