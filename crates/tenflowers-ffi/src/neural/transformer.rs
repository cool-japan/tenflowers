//! Transformer building blocks module for TenfloweRS FFI
//!
//! This module provides transformer architecture components including encoder/decoder
//! layers and positional encodings for sequence-to-sequence models.

use crate::neural::attention::PyMultiheadAttention;
use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;
use tenflowers_core::{Result as CoreResult, Tensor};

/// Swap the first two axes of a 3-D tensor when not batch-first, converting
/// between `[seq, batch, feature]` and `[batch, seq, feature]`.
///
/// The transform is its own inverse, so the same helper aligns into and out of
/// batch-first layout.
fn align_batch_first(t: &Tensor<f32>, batch_first: bool) -> CoreResult<Tensor<f32>> {
    if batch_first {
        Ok(t.clone())
    } else {
        tenflowers_core::ops::manipulation::transpose_axes(t, Some(&[1, 0, 2]))
    }
}

/// Initialise a tensor with scaled standard-normal values.
fn randn_scaled(shape: &[usize], scale: f32) -> CoreResult<Tensor<f32>> {
    Tensor::randn(shape)?.multiply_scalar(scale)
}

/// Apply a linear projection `input @ weight^T (+ bias)` where `weight` is
/// stored as `[out_features, in_features]`.
fn linear_proj(
    input: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
) -> CoreResult<Tensor<f32>> {
    let weight_t = weight.transpose()?;
    let projected = input.matmul(&weight_t)?;
    match bias {
        Some(b) => projected.add(b),
        None => Ok(projected),
    }
}

/// Position-wise feed-forward network: `linear2(activation(linear1(x)))`.
fn feed_forward(
    input: &Tensor<f32>,
    w1: &Tensor<f32>,
    b1: &Tensor<f32>,
    w2: &Tensor<f32>,
    b2: &Tensor<f32>,
    activation: &str,
) -> CoreResult<Tensor<f32>> {
    let hidden = linear_proj(input, w1, Some(b1))?;
    let activated = if activation == "gelu" {
        hidden.gelu()?
    } else {
        hidden.relu()?
    };
    linear_proj(&activated, w2, Some(b2))
}

/// Apply layer normalization over the final (feature) dimension.
fn apply_layer_norm(
    input: &Tensor<f32>,
    gamma: &Tensor<f32>,
    beta: &Tensor<f32>,
    d_model: usize,
    eps: f32,
) -> CoreResult<Tensor<f32>> {
    tenflowers_core::ops::normalization::layer_norm(input, gamma, beta, &[d_model], eps)
}

/// Build a `PyMultiheadAttention` configured for batch-first input.
fn build_attention(d_model: usize, nhead: usize) -> PyResult<PyMultiheadAttention> {
    PyMultiheadAttention::new(
        d_model,
        nhead,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(true),
    )
}

/// Wrap a tensor as a non-grad, non-pinned `PyTensor` for attention calls.
fn wrap(tensor: Tensor<f32>) -> PyTensor {
    PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    }
}

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
    /// Multi-head self-attention sublayer
    self_attn: PyMultiheadAttention,
    /// Feed-forward weight 1: `[dim_feedforward, d_model]`
    ff_w1: Tensor<f32>,
    /// Feed-forward bias 1: `[dim_feedforward]`
    ff_b1: Tensor<f32>,
    /// Feed-forward weight 2: `[d_model, dim_feedforward]`
    ff_w2: Tensor<f32>,
    /// Feed-forward bias 2: `[d_model]`
    ff_b2: Tensor<f32>,
    /// LayerNorm gain `[d_model]`
    ln_gamma: Tensor<f32>,
    /// LayerNorm bias `[d_model]`
    ln_beta: Tensor<f32>,
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

        let self_attn = build_attention(d_model, nhead)?;
        let init_err = |e: tenflowers_core::TensorError| {
            PyRuntimeError::new_err(format!(
                "Failed to initialize TransformerEncoderLayer: {}",
                e
            ))
        };
        let scale1 = 1.0_f32 / (d_model as f32).sqrt();
        let scale2 = 1.0_f32 / (dim_feedforward as f32).sqrt();
        let ff_w1 = randn_scaled(&[dim_feedforward, d_model], scale1).map_err(init_err)?;
        let ff_b1 = Tensor::zeros(&[dim_feedforward]);
        let ff_w2 = randn_scaled(&[d_model, dim_feedforward], scale2).map_err(init_err)?;
        let ff_b2 = Tensor::zeros(&[d_model]);
        let ln_gamma = Tensor::from_vec(vec![1.0f32; d_model], &[d_model]).map_err(init_err)?;
        let ln_beta = Tensor::zeros(&[d_model]);

        Ok(PyTransformerEncoderLayer {
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            batch_first,
            layer_norm_eps,
            self_attn,
            ff_w1,
            ff_b1,
            ff_w2,
            ff_b2,
            ln_gamma,
            ln_beta,
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
    #[pyo3(signature = (src, src_mask=None, src_key_padding_mask=None))]
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

        let to_err = |e: tenflowers_core::TensorError| {
            PyRuntimeError::new_err(format!("TransformerEncoderLayer forward failed: {}", e))
        };

        // Work in batch-first layout: [batch, seq, d_model].
        let x = align_batch_first(&src.tensor, self.batch_first).map_err(to_err)?;

        // Self-attention sublayer (post-norm): x = LayerNorm(x + SelfAttn(x)).
        // `src_mask` is the [seq, seq] additive attention mask; `src_key_padding_mask`
        // is the [batch, seq] key-padding mask (parameter order:
        // query, key, value, key_padding_mask, need_weights, attn_mask, average).
        let x_py = wrap(x.clone());
        let (attn_py, _) = self.self_attn.forward(
            &x_py,
            &x_py,
            &x_py,
            src_key_padding_mask,
            Some(false),
            src_mask,
            None,
        )?;
        let residual1 = x.add(attn_py.tensor.as_ref()).map_err(to_err)?;
        let normed1 = apply_layer_norm(
            &residual1,
            &self.ln_gamma,
            &self.ln_beta,
            self.d_model,
            self.layer_norm_eps,
        )
        .map_err(to_err)?;

        // Feed-forward sublayer (post-norm): x = LayerNorm(x + FFN(x)).
        let ff = feed_forward(
            &normed1,
            &self.ff_w1,
            &self.ff_b1,
            &self.ff_w2,
            &self.ff_b2,
            &self.activation,
        )
        .map_err(to_err)?;
        let residual2 = normed1.add(&ff).map_err(to_err)?;
        let normed2 = apply_layer_norm(
            &residual2,
            &self.ln_gamma,
            &self.ln_beta,
            self.d_model,
            self.layer_norm_eps,
        )
        .map_err(to_err)?;

        let output = align_batch_first(&normed2, self.batch_first).map_err(to_err)?;

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: src.requires_grad,
            is_pinned: src.is_pinned,
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
    /// Multi-head self-attention sublayer
    self_attn: PyMultiheadAttention,
    /// Multi-head cross-attention sublayer (attends to encoder memory)
    cross_attn: PyMultiheadAttention,
    /// Feed-forward weight 1: `[dim_feedforward, d_model]`
    ff_w1: Tensor<f32>,
    /// Feed-forward bias 1: `[dim_feedforward]`
    ff_b1: Tensor<f32>,
    /// Feed-forward weight 2: `[d_model, dim_feedforward]`
    ff_w2: Tensor<f32>,
    /// Feed-forward bias 2: `[d_model]`
    ff_b2: Tensor<f32>,
    /// LayerNorm gain `[d_model]`
    ln_gamma: Tensor<f32>,
    /// LayerNorm bias `[d_model]`
    ln_beta: Tensor<f32>,
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

        let self_attn = build_attention(d_model, nhead)?;
        let cross_attn = build_attention(d_model, nhead)?;
        let init_err = |e: tenflowers_core::TensorError| {
            PyRuntimeError::new_err(format!(
                "Failed to initialize TransformerDecoderLayer: {}",
                e
            ))
        };
        let scale1 = 1.0_f32 / (d_model as f32).sqrt();
        let scale2 = 1.0_f32 / (dim_feedforward as f32).sqrt();
        let ff_w1 = randn_scaled(&[dim_feedforward, d_model], scale1).map_err(init_err)?;
        let ff_b1 = Tensor::zeros(&[dim_feedforward]);
        let ff_w2 = randn_scaled(&[d_model, dim_feedforward], scale2).map_err(init_err)?;
        let ff_b2 = Tensor::zeros(&[d_model]);
        let ln_gamma = Tensor::from_vec(vec![1.0f32; d_model], &[d_model]).map_err(init_err)?;
        let ln_beta = Tensor::zeros(&[d_model]);

        Ok(PyTransformerDecoderLayer {
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            batch_first,
            layer_norm_eps,
            self_attn,
            cross_attn,
            ff_w1,
            ff_b1,
            ff_w2,
            ff_b2,
            ln_gamma,
            ln_beta,
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

        if tgt_shape[2] != self.d_model || memory_shape[2] != self.d_model {
            return Err(PyValueError::new_err(format!(
                "Expected feature dimension {}",
                self.d_model
            )));
        }

        let to_err = |e: tenflowers_core::TensorError| {
            PyRuntimeError::new_err(format!("TransformerDecoderLayer forward failed: {}", e))
        };

        // Work in batch-first layout.
        let tgt_bf = align_batch_first(&tgt.tensor, self.batch_first).map_err(to_err)?;
        let memory_bf = align_batch_first(&memory.tensor, self.batch_first).map_err(to_err)?;
        let memory_py = wrap(memory_bf);

        // Masked self-attention sublayer (post-norm). `tgt_mask` is the
        // [tgt, tgt] attention mask; `tgt_key_padding_mask` is [batch, tgt].
        let tgt_py = wrap(tgt_bf.clone());
        let (sa_py, _) = self.self_attn.forward(
            &tgt_py,
            &tgt_py,
            &tgt_py,
            tgt_key_padding_mask,
            Some(false),
            tgt_mask,
            None,
        )?;
        let residual1 = tgt_bf.add(sa_py.tensor.as_ref()).map_err(to_err)?;
        let normed1 = apply_layer_norm(
            &residual1,
            &self.ln_gamma,
            &self.ln_beta,
            self.d_model,
            self.layer_norm_eps,
        )
        .map_err(to_err)?;

        // Cross-attention sublayer: query = decoder state, key/value = memory.
        // `memory_mask` is the [tgt, src] attention mask; `memory_key_padding_mask`
        // is [batch, src] where src is the memory sequence length.
        let normed1_py = wrap(normed1.clone());
        let (ca_py, _) = self.cross_attn.forward(
            &normed1_py,
            &memory_py,
            &memory_py,
            memory_key_padding_mask,
            Some(false),
            memory_mask,
            None,
        )?;
        let residual2 = normed1.add(ca_py.tensor.as_ref()).map_err(to_err)?;
        let normed2 = apply_layer_norm(
            &residual2,
            &self.ln_gamma,
            &self.ln_beta,
            self.d_model,
            self.layer_norm_eps,
        )
        .map_err(to_err)?;

        // Feed-forward sublayer.
        let ff = feed_forward(
            &normed2,
            &self.ff_w1,
            &self.ff_b1,
            &self.ff_w2,
            &self.ff_b2,
            &self.activation,
        )
        .map_err(to_err)?;
        let residual3 = normed2.add(&ff).map_err(to_err)?;
        let normed3 = apply_layer_norm(
            &residual3,
            &self.ln_gamma,
            &self.ln_beta,
            self.d_model,
            self.layer_norm_eps,
        )
        .map_err(to_err)?;

        let output = align_batch_first(&normed3, self.batch_first).map_err(to_err)?;

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: tgt.requires_grad || memory.requires_grad,
            is_pinned: tgt.is_pinned,
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

        let (seq_len, batch, d_model) = if batch_first {
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

        // Add the precomputed sinusoidal positional encoding to the input.
        let x_data = x
            .tensor
            .to_vec()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get tensor data: {}", e)))?;

        let mut out_data = Vec::with_capacity(x_data.len());
        for (idx, &value) in x_data.iter().enumerate() {
            let feature = idx % d_model;
            let position_block = idx / d_model;
            let position = if batch_first {
                position_block % seq_len
            } else {
                position_block / batch
            };
            out_data.push(value + self.pe[position * d_model + feature]);
        }

        let output_shape: Vec<usize> = x_shape.iter().copied().collect();
        let output = Tensor::from_vec(out_data, &output_shape)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output: {}", e)))?;

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: x.requires_grad,
            is_pinned: x.is_pinned,
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

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * 0.05 - 0.5).collect()
    }

    #[test]
    fn positional_encoding_modifies_input() {
        let pe = PyPositionalEncoding::new(4, None, None).expect("pe construction");
        let input = make_tensor(vec![1.0; 3 * 2 * 4], &[3, 2, 4]);
        let out = pe.forward(&input, None).expect("forward");

        assert_eq!(out.tensor.shape().dims().to_vec(), vec![3, 2, 4]);
        let in_vec = input.tensor.to_vec().expect("in vec");
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert!(
            in_vec
                .iter()
                .zip(out_vec.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "positional encoding must change the input"
        );
    }

    #[test]
    fn transformer_encoder_forward_is_real() {
        let enc = PyTransformerEncoderLayer::new(8, 2, None, None, None, None, None).expect("enc");
        let src = make_tensor(ramp(3 * 2 * 8), &[3, 2, 8]);
        let out = enc.forward(&src, None, None).expect("forward");

        assert_eq!(out.tensor.shape().dims().to_vec(), vec![3, 2, 8]);
        let out_vec = out.tensor.to_vec().expect("vec");
        assert!(
            out_vec.iter().any(|&x| x != 0.0),
            "encoder output must not be all zeros"
        );
    }

    #[test]
    fn transformer_decoder_forward_is_real() {
        let dec = PyTransformerDecoderLayer::new(8, 2, None, None, None, None, None).expect("dec");
        let tgt = make_tensor(ramp(3 * 2 * 8), &[3, 2, 8]);
        let memory = make_tensor(ramp(4 * 2 * 8), &[4, 2, 8]);
        let out = dec
            .forward(&tgt, &memory, None, None, None, None)
            .expect("forward");

        assert_eq!(out.tensor.shape().dims().to_vec(), vec![3, 2, 8]);
        let out_vec = out.tensor.to_vec().expect("vec");
        assert!(
            out_vec.iter().any(|&x| x != 0.0),
            "decoder output must not be all zeros"
        );
    }

    #[test]
    fn transformer_encoder_forward_with_masks_succeeds() {
        // Providing real masks previously returned an error; it must now succeed.
        let enc = PyTransformerEncoderLayer::new(8, 2, None, None, None, None, None).expect("enc");
        // Default batch_first=false: src is [seq=3, batch=2, d_model=8].
        let src = make_tensor(ramp(3 * 2 * 8), &[3, 2, 8]);
        // Causal [seq, seq] = [3, 3] additive attention mask.
        let src_mask = generate_square_subsequent_mask(3).expect("square mask");
        // key padding [batch, seq] = [2, 3]; mask position 2 for batch 0 only.
        // (No row becomes fully masked, so softmax stays finite.)
        let src_kpm = make_tensor(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0], &[2, 3]);

        let out = enc
            .forward(&src, Some(&src_mask), Some(&src_kpm))
            .expect("masked encoder forward must now succeed");

        assert_eq!(out.tensor.shape().dims().to_vec(), vec![3, 2, 8]);
        let out_vec = out.tensor.to_vec().expect("vec");
        assert!(
            out_vec.iter().all(|&x| x.is_finite()),
            "masked encoder output must be finite (no NaN/inf)"
        );
        assert!(
            out_vec.iter().any(|&x| x != 0.0),
            "masked encoder output must not be all zeros"
        );
    }

    #[test]
    fn transformer_decoder_forward_with_masks_succeeds() {
        // All four decoder mask parameters provided together must succeed.
        let dec = PyTransformerDecoderLayer::new(8, 2, None, None, None, None, None).expect("dec");
        // Default batch_first=false: tgt [tgt=3, batch=2, 8], memory [src=4, batch=2, 8].
        let tgt = make_tensor(ramp(3 * 2 * 8), &[3, 2, 8]);
        let memory = make_tensor(ramp(4 * 2 * 8), &[4, 2, 8]);
        let tgt_mask = generate_square_subsequent_mask(3).expect("tgt mask"); // [3, 3]
        let memory_mask = make_tensor(vec![0.0; 3 * 4], &[3, 4]); // [tgt=3, src=4]
        let tgt_kpm = make_tensor(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0], &[2, 3]); // [batch, tgt]
                                                                                // [batch, src=4]; mask memory position 3 for batch 0 only.
        let memory_kpm = make_tensor(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], &[2, 4]);

        let out = dec
            .forward(
                &tgt,
                &memory,
                Some(&tgt_mask),
                Some(&memory_mask),
                Some(&tgt_kpm),
                Some(&memory_kpm),
            )
            .expect("masked decoder forward must now succeed");

        assert_eq!(out.tensor.shape().dims().to_vec(), vec![3, 2, 8]);
        let out_vec = out.tensor.to_vec().expect("vec");
        assert!(
            out_vec.iter().all(|&x| x.is_finite()),
            "masked decoder output must be finite (no NaN/inf)"
        );
        assert!(
            out_vec.iter().any(|&x| x != 0.0),
            "masked decoder output must not be all zeros"
        );
    }
}
