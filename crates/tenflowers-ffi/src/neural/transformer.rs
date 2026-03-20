//! Transformer building blocks module for TenfloweRS FFI
//!
//! This module provides transformer architecture components including encoder/decoder
//! layers and positional encodings for sequence-to-sequence models.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::f32::consts::PI;
use std::sync::Arc;
use tenflowers_core::Tensor;

/// Transformer Encoder Layer
///
/// A single layer of the transformer encoder with multi-head attention and feedforward network.
#[pyclass(name = "TransformerEncoderLayer")]
#[derive(Debug, Clone)]
pub struct PyTransformerEncoderLayer {
    /// Dimension of the model
    pub d_model: usize,
    /// Number of attention heads
    pub nhead: usize,
    /// Dimension of feedforward network
    pub dim_feedforward: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Activation function ('relu' or 'gelu')
    pub activation: String,
    /// Whether to use batch_first format
    pub batch_first: bool,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
}

#[pymethods]
impl PyTransformerEncoderLayer {
    /// Create a new transformer encoder layer
    ///
    /// # Arguments
    ///
    /// * `d_model` - Dimension of the model
    /// * `nhead` - Number of attention heads
    /// * `dim_feedforward` - Dimension of feedforward network (default: 2048)
    /// * `dropout` - Dropout probability (default: 0.1)
    /// * `activation` - Activation function 'relu' or 'gelu' (default: 'relu')
    /// * `batch_first` - If True, input is (batch, seq, feature) (default: False)
    /// * `layer_norm_eps` - Layer normalization epsilon (default: 1e-5)
    #[new]
    #[pyo3(signature = (d_model, nhead, dim_feedforward=None, dropout=None, activation=None, batch_first=None, layer_norm_eps=None))]
    pub fn new(
        d_model: usize,
        nhead: usize,
        dim_feedforward: Option<usize>,
        dropout: Option<f32>,
        activation: Option<String>,
        batch_first: Option<bool>,
        layer_norm_eps: Option<f32>,
    ) -> PyResult<Self> {
        let dim_feedforward = dim_feedforward.unwrap_or(2048);
        let dropout = dropout.unwrap_or(0.1);
        let activation = activation.unwrap_or_else(|| "relu".to_string());
        let batch_first = batch_first.unwrap_or(false);
        let layer_norm_eps = layer_norm_eps.unwrap_or(1e-5);

        if d_model == 0 {
            return Err(PyValueError::new_err("d_model must be positive"));
        }
        if nhead == 0 {
            return Err(PyValueError::new_err("nhead must be positive"));
        }
        if d_model % nhead != 0 {
            return Err(PyValueError::new_err(format!(
                "d_model {} must be divisible by nhead {}",
                d_model, nhead
            )));
        }
        if dim_feedforward == 0 {
            return Err(PyValueError::new_err("dim_feedforward must be positive"));
        }
        if !(0.0..=1.0).contains(&dropout) {
            return Err(PyValueError::new_err("dropout must be between 0 and 1"));
        }
        if activation != "relu" && activation != "gelu" {
            return Err(PyValueError::new_err("activation must be 'relu' or 'gelu'"));
        }

        Ok(PyTransformerEncoderLayer {
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            batch_first,
            layer_norm_eps,
        })
    }

    /// Forward pass through the encoder layer
    ///
    /// # Arguments
    ///
    /// * `src` - Source sequence tensor
    /// * `src_mask` - Optional mask for source sequence
    /// * `src_key_padding_mask` - Optional padding mask
    ///
    /// # Returns
    ///
    /// Output tensor with same shape as input
    pub fn forward(
        &self,
        src: &PyTensor,
        src_mask: Option<&PyTensor>,
        src_key_padding_mask: Option<&PyTensor>,
    ) -> PyResult<PyTensor> {
        let src_shape = src.tensor.shape();

        if src_shape.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "Expected 3D input, got {}D",
                src_shape.len()
            )));
        }

        let (_seq_len, _batch_size, feature_dim) = if self.batch_first {
            (src_shape[1], src_shape[0], src_shape[2])
        } else {
            (src_shape[0], src_shape[1], src_shape[2])
        };

        if feature_dim != self.d_model {
            return Err(PyValueError::new_err(format!(
                "Expected feature dimension {}, got {}",
                self.d_model, feature_dim
            )));
        }

        // For now, return placeholder output with same shape
        let output_shape: Vec<usize> = src_shape.iter().copied().collect();
        let output = Tensor::zeros(&output_shape);

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: src.requires_grad,
            is_pinned: false,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "TransformerEncoderLayer(d_model={}, nhead={}, dim_feedforward={}, dropout={}, activation='{}')",
            self.d_model, self.nhead, self.dim_feedforward, self.dropout, self.activation
        )
    }
}

/// Transformer Decoder Layer
///
/// A single layer of the transformer decoder with self-attention, cross-attention, and feedforward.
#[pyclass(name = "TransformerDecoderLayer")]
#[derive(Debug, Clone)]
pub struct PyTransformerDecoderLayer {
    /// Dimension of the model
    pub d_model: usize,
    /// Number of attention heads
    pub nhead: usize,
    /// Dimension of feedforward network
    pub dim_feedforward: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Activation function
    pub activation: String,
    /// Whether to use batch_first format
    pub batch_first: bool,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
}

#[pymethods]
impl PyTransformerDecoderLayer {
    /// Create a new transformer decoder layer
    #[new]
    #[pyo3(signature = (d_model, nhead, dim_feedforward=None, dropout=None, activation=None, batch_first=None, layer_norm_eps=None))]
    pub fn new(
        d_model: usize,
        nhead: usize,
        dim_feedforward: Option<usize>,
        dropout: Option<f32>,
        activation: Option<String>,
        batch_first: Option<bool>,
        layer_norm_eps: Option<f32>,
    ) -> PyResult<Self> {
        let dim_feedforward = dim_feedforward.unwrap_or(2048);
        let dropout = dropout.unwrap_or(0.1);
        let activation = activation.unwrap_or_else(|| "relu".to_string());
        let batch_first = batch_first.unwrap_or(false);
        let layer_norm_eps = layer_norm_eps.unwrap_or(1e-5);

        if d_model == 0 {
            return Err(PyValueError::new_err("d_model must be positive"));
        }
        if nhead == 0 {
            return Err(PyValueError::new_err("nhead must be positive"));
        }
        if d_model % nhead != 0 {
            return Err(PyValueError::new_err(format!(
                "d_model {} must be divisible by nhead {}",
                d_model, nhead
            )));
        }
        if dim_feedforward == 0 {
            return Err(PyValueError::new_err("dim_feedforward must be positive"));
        }
        if !(0.0..=1.0).contains(&dropout) {
            return Err(PyValueError::new_err("dropout must be between 0 and 1"));
        }
        if activation != "relu" && activation != "gelu" {
            return Err(PyValueError::new_err("activation must be 'relu' or 'gelu'"));
        }

        Ok(PyTransformerDecoderLayer {
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            batch_first,
            layer_norm_eps,
        })
    }

    /// Forward pass through the decoder layer
    ///
    /// # Arguments
    ///
    /// * `tgt` - Target sequence tensor
    /// * `memory` - Encoder output tensor
    /// * `tgt_mask` - Optional mask for target sequence
    /// * `memory_mask` - Optional mask for encoder output
    /// * `tgt_key_padding_mask` - Optional target padding mask
    /// * `memory_key_padding_mask` - Optional memory padding mask
    ///
    /// # Returns
    ///
    /// Output tensor with same shape as target
    #[pyo3(signature = (tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None))]
    pub fn forward(
        &self,
        tgt: &PyTensor,
        memory: &PyTensor,
        tgt_mask: Option<&PyTensor>,
        memory_mask: Option<&PyTensor>,
        tgt_key_padding_mask: Option<&PyTensor>,
        memory_key_padding_mask: Option<&PyTensor>,
    ) -> PyResult<PyTensor> {
        let tgt_shape = tgt.tensor.shape();
        let memory_shape = memory.tensor.shape();

        if tgt_shape.len() != 3 || memory_shape.len() != 3 {
            return Err(PyValueError::new_err("Expected 3D inputs"));
        }

        let output_shape: Vec<usize> = tgt_shape.iter().copied().collect();
        let output = Tensor::zeros(&output_shape);

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: tgt.requires_grad || memory.requires_grad,
            is_pinned: false,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "TransformerDecoderLayer(d_model={}, nhead={}, dim_feedforward={}, dropout={})",
            self.d_model, self.nhead, self.dim_feedforward, self.dropout
        )
    }
}

/// Positional Encoding
///
/// Adds positional information to embeddings using sinusoidal functions.
#[pyclass(name = "PositionalEncoding")]
#[derive(Debug, Clone)]
pub struct PyPositionalEncoding {
    /// Dimension of the model
    pub d_model: usize,
    /// Maximum sequence length
    pub max_len: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Precomputed positional encodings
    pub pe: Vec<f32>,
}

#[pymethods]
impl PyPositionalEncoding {
    /// Create a new positional encoding
    ///
    /// # Arguments
    ///
    /// * `d_model` - Dimension of the model
    /// * `max_len` - Maximum sequence length (default: 5000)
    /// * `dropout` - Dropout probability (default: 0.1)
    #[new]
    #[pyo3(signature = (d_model, max_len=None, dropout=None))]
    pub fn new(d_model: usize, max_len: Option<usize>, dropout: Option<f32>) -> PyResult<Self> {
        let max_len = max_len.unwrap_or(5000);
        let dropout = dropout.unwrap_or(0.1);

        if d_model == 0 {
            return Err(PyValueError::new_err("d_model must be positive"));
        }
        if max_len == 0 {
            return Err(PyValueError::new_err("max_len must be positive"));
        }
        if !(0.0..=1.0).contains(&dropout) {
            return Err(PyValueError::new_err("dropout must be between 0 and 1"));
        }

        // Compute positional encodings
        let mut pe = vec![0.0; max_len * d_model];

        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f32 / 10000_f32.powf(2.0 * (i / 2) as f32 / d_model as f32);

                if i % 2 == 0 {
                    pe[pos * d_model + i] = angle.sin();
                } else {
                    pe[pos * d_model + i] = angle.cos();
                }
            }
        }

        Ok(PyPositionalEncoding {
            d_model,
            max_len,
            dropout,
            pe,
        })
    }

    /// Apply positional encoding to input
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape (seq_len, batch, d_model) or (batch, seq_len, d_model)
    /// * `batch_first` - If True, input is (batch, seq_len, d_model)
    ///
    /// # Returns
    ///
    /// Tensor with positional encoding added
    #[pyo3(signature = (x, batch_first=None))]
    pub fn forward(&self, x: &PyTensor, batch_first: Option<bool>) -> PyResult<PyTensor> {
        let batch_first = batch_first.unwrap_or(false);
        let x_shape = x.tensor.shape();

        if x_shape.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "Expected 3D input, got {}D",
                x_shape.len()
            )));
        }

        let (seq_len, _batch, d_model) = if batch_first {
            (x_shape[1], x_shape[0], x_shape[2])
        } else {
            (x_shape[0], x_shape[1], x_shape[2])
        };

        if d_model != self.d_model {
            return Err(PyValueError::new_err(format!(
                "Expected d_model={}, got {}",
                self.d_model, d_model
            )));
        }

        if seq_len > self.max_len {
            return Err(PyValueError::new_err(format!(
                "Sequence length {} exceeds max_len {}",
                seq_len, self.max_len
            )));
        }

        // For now, return input unchanged (positional encoding would be added)
        let x_data = x
            .tensor
            .to_vec()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get tensor data: {}", e)))?;

        let output_shape: Vec<usize> = x_shape.iter().copied().collect();
        let output = Tensor::from_vec(x_data, &output_shape)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output: {}", e)))?;

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: x.requires_grad,
            is_pinned: false,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PositionalEncoding(d_model={}, max_len={}, dropout={})",
            self.d_model, self.max_len, self.dropout
        )
    }
}

/// Generate square subsequent mask for autoregressive decoding
///
/// Creates a mask to prevent attention to future positions.
///
/// # Arguments
///
/// * `size` - Size of the square mask
///
/// # Returns
///
/// Square mask tensor of shape (size, size)
#[pyfunction]
pub fn generate_square_subsequent_mask(size: usize) -> PyResult<PyTensor> {
    if size == 0 {
        return Err(PyValueError::new_err("size must be positive"));
    }

    // Create upper triangular matrix with -inf
    let mut mask = vec![0.0f32; size * size];

    for i in 0..size {
        for j in (i + 1)..size {
            mask[i * size + j] = f32::NEG_INFINITY;
        }
    }

    let tensor = Tensor::from_vec(mask, &[size, size])
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create mask: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    })
}

/// Create padding mask from sequence lengths
///
/// Creates a boolean mask indicating padding positions.
///
/// # Arguments
///
/// * `lengths` - Sequence lengths for each batch element
/// * `max_len` - Maximum sequence length
///
/// # Returns
///
/// Boolean mask tensor of shape (batch_size, max_len)
#[pyfunction]
pub fn create_padding_mask(lengths: Vec<usize>, max_len: usize) -> PyResult<PyTensor> {
    if max_len == 0 {
        return Err(PyValueError::new_err("max_len must be positive"));
    }

    let batch_size = lengths.len();
    let mut mask = vec![1.0f32; batch_size * max_len];

    for (i, &length) in lengths.iter().enumerate() {
        if length > max_len {
            return Err(PyValueError::new_err(format!(
                "Length {} exceeds max_len {}",
                length, max_len
            )));
        }

        // Set positions beyond length to 0 (padding)
        for j in length..max_len {
            mask[i * max_len + j] = 0.0;
        }
    }

    let tensor = Tensor::from_vec(mask, &[batch_size, max_len])
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create mask: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    })
}
