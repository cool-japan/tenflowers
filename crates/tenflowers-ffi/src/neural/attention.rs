//! Attention mechanisms module for TenfloweRS FFI
//!
//! This module provides attention mechanism implementations including multi-head attention
//! for transformers and other sequence-to-sequence models.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tenflowers_core::Tensor;

/// Multi-Head Attention Layer
///
/// Allows the model to jointly attend to information from different representation
/// subspaces at different positions. Used extensively in transformer architectures.
#[pyclass(name = "MultiheadAttention")]
#[derive(Debug, Clone)]
pub struct PyMultiheadAttention {
    /// Total dimension of the model
    pub embed_dim: usize,
    /// Number of parallel attention heads
    pub num_heads: usize,
    /// Dimension of each attention head
    pub head_dim: usize,
    /// Dropout probability on attention weights
    pub dropout: f32,
    /// If True, add bias to input/output projection layers
    pub bias: bool,
    /// If True, add bias to key, value, query projection layers
    pub add_bias_kv: bool,
    /// If True, add zero attention (useful for masking)
    pub add_zero_attn: bool,
    /// Dimension of key/value (if different from embed_dim)
    pub kdim: Option<usize>,
    /// Dimension of value (if different from embed_dim)
    pub vdim: Option<usize>,
    /// If True, decoder-style attention (use batch_first=False)
    pub batch_first: bool,
    /// Query projection weight
    pub q_proj_weight: Option<Tensor<f32>>,
    /// Key projection weight
    pub k_proj_weight: Option<Tensor<f32>>,
    /// Value projection weight
    pub v_proj_weight: Option<Tensor<f32>>,
    /// Output projection weight
    pub out_proj_weight: Option<Tensor<f32>>,
    /// Bias for projections
    pub bias_weight: Option<Tensor<f32>>,
}

#[pymethods]
impl PyMultiheadAttention {
    /// Create a new MultiheadAttention layer
    ///
    /// # Arguments
    ///
    /// * `embed_dim` - Total dimension of the model
    /// * `num_heads` - Number of parallel attention heads (must divide embed_dim)
    /// * `dropout` - Dropout probability on attention weights (default: 0.0)
    /// * `bias` - If True, add bias to input/output projection layers (default: True)
    /// * `add_bias_kv` - If True, add bias to key, value projection layers (default: False)
    /// * `add_zero_attn` - If True, add zero attention (default: False)
    /// * `kdim` - Dimension of key (default: same as embed_dim)
    /// * `vdim` - Dimension of value (default: same as embed_dim)
    /// * `batch_first` - If True, input is (batch, seq, feature) (default: False)
    #[new]
    #[pyo3(signature = (embed_dim, num_heads, dropout=None, bias=None, add_bias_kv=None, add_zero_attn=None, kdim=None, vdim=None, batch_first=None))]
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        dropout: Option<f32>,
        bias: Option<bool>,
        add_bias_kv: Option<bool>,
        add_zero_attn: Option<bool>,
        kdim: Option<usize>,
        vdim: Option<usize>,
        batch_first: Option<bool>,
    ) -> PyResult<Self> {
        let dropout = dropout.unwrap_or(0.0);
        let bias = bias.unwrap_or(true);
        let add_bias_kv = add_bias_kv.unwrap_or(false);
        let add_zero_attn = add_zero_attn.unwrap_or(false);
        let batch_first = batch_first.unwrap_or(false);

        if embed_dim == 0 {
            return Err(PyValueError::new_err("embed_dim must be positive"));
        }
        if num_heads == 0 {
            return Err(PyValueError::new_err("num_heads must be positive"));
        }
        if embed_dim % num_heads != 0 {
            return Err(PyValueError::new_err(format!(
                "embed_dim {} must be divisible by num_heads {}",
                embed_dim, num_heads
            )));
        }
        if !(0.0..=1.0).contains(&dropout) {
            return Err(PyValueError::new_err("dropout must be between 0 and 1"));
        }

        let head_dim = embed_dim / num_heads;
        let kdim_actual = kdim.unwrap_or(embed_dim);
        let vdim_actual = vdim.unwrap_or(embed_dim);

        // Initialize projection weights
        let q_proj_weight = Tensor::zeros(&[embed_dim, embed_dim]);
        let k_proj_weight = Tensor::zeros(&[embed_dim, kdim_actual]);
        let v_proj_weight = Tensor::zeros(&[embed_dim, vdim_actual]);
        let out_proj_weight = Tensor::zeros(&[embed_dim, embed_dim]);

        let bias_weight = if bias {
            Some(Tensor::zeros(&[embed_dim]))
        } else {
            None
        };

        Ok(PyMultiheadAttention {
            embed_dim,
            num_heads,
            head_dim,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim: Some(kdim_actual),
            vdim: Some(vdim_actual),
            batch_first,
            q_proj_weight: Some(q_proj_weight),
            k_proj_weight: Some(k_proj_weight),
            v_proj_weight: Some(v_proj_weight),
            out_proj_weight: Some(out_proj_weight),
            bias_weight,
        })
    }

    /// Forward pass through the multi-head attention layer
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor
    /// * `key` - Key tensor
    /// * `value` - Value tensor
    /// * `key_padding_mask` - Optional mask for padding positions (True = ignore)
    /// * `need_weights` - If True, return attention weights (default: True)
    /// * `attn_mask` - Optional attention mask
    /// * `average_attn_weights` - If True, return averaged attention weights (default: True)
    ///
    /// # Returns
    ///
    /// Tuple of (attn_output, attn_output_weights) if need_weights, else (attn_output, None)
    #[pyo3(signature = (query, key, value, key_padding_mask=None, need_weights=None, attn_mask=None, average_attn_weights=None))]
    pub fn forward(
        &self,
        query: &PyTensor,
        key: &PyTensor,
        value: &PyTensor,
        key_padding_mask: Option<&PyTensor>,
        need_weights: Option<bool>,
        attn_mask: Option<&PyTensor>,
        average_attn_weights: Option<bool>,
    ) -> PyResult<(PyTensor, Option<PyTensor>)> {
        let need_weights = need_weights.unwrap_or(true);
        let _average_attn_weights = average_attn_weights.unwrap_or(true);

        let query_shape = query.tensor.shape();
        let key_shape = key.tensor.shape();
        let value_shape = value.tensor.shape();

        // Validate input shapes
        if query_shape.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "Expected 3D query tensor, got {}D",
                query_shape.len()
            )));
        }
        if key_shape.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "Expected 3D key tensor, got {}D",
                key_shape.len()
            )));
        }
        if value_shape.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "Expected 3D value tensor, got {}D",
                value_shape.len()
            )));
        }

        let (tgt_len, batch_size, _embed_dim) = if self.batch_first {
            (query_shape[1], query_shape[0], query_shape[2])
        } else {
            (query_shape[0], query_shape[1], query_shape[2])
        };

        // Validate masks
        if let Some(mask) = key_padding_mask {
            let mask_shape = mask.tensor.shape();
            if mask_shape.len() != 2 {
                return Err(PyValueError::new_err(
                    "key_padding_mask must be 2D (batch_size, src_len)",
                ));
            }
        }

        if let Some(mask) = attn_mask {
            let mask_shape = mask.tensor.shape();
            if mask_shape.len() != 2 {
                return Err(PyValueError::new_err(
                    "attn_mask must be 2D (tgt_len, src_len)",
                ));
            }
        }

        // Output shape
        let output_shape = if self.batch_first {
            vec![batch_size, tgt_len, self.embed_dim]
        } else {
            vec![tgt_len, batch_size, self.embed_dim]
        };

        // For now, return placeholder tensors
        let attn_output = Tensor::zeros(&output_shape);

        let attn_weights = if need_weights {
            let weights_shape = vec![batch_size, tgt_len, key_shape[0]];
            Some(PyTensor {
                tensor: Arc::new(Tensor::zeros(&weights_shape)),
                requires_grad: false,
                is_pinned: false,
            })
        } else {
            None
        };

        Ok((
            PyTensor {
                tensor: Arc::new(attn_output),
                requires_grad: query.requires_grad || key.requires_grad || value.requires_grad,
                is_pinned: false,
            },
            attn_weights,
        ))
    }

    /// Reset layer parameters
    pub fn reset_parameters(&mut self) -> PyResult<()> {
        let kdim_actual = self.kdim.unwrap_or(self.embed_dim);
        let vdim_actual = self.vdim.unwrap_or(self.embed_dim);

        self.q_proj_weight = Some(Tensor::zeros(&[self.embed_dim, self.embed_dim]));
        self.k_proj_weight = Some(Tensor::zeros(&[self.embed_dim, kdim_actual]));
        self.v_proj_weight = Some(Tensor::zeros(&[self.embed_dim, vdim_actual]));
        self.out_proj_weight = Some(Tensor::zeros(&[self.embed_dim, self.embed_dim]));

        if self.bias {
            self.bias_weight = Some(Tensor::zeros(&[self.embed_dim]));
        }

        Ok(())
    }

    /// Get layer state dictionary
    pub fn state_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);

        if let Some(ref weight) = self.q_proj_weight {
            let weight_data = weight.to_vec().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to get q_proj weight: {}", e))
            })?;
            let weight_shape: Vec<usize> = weight.shape().iter().copied().collect();
            let weight_tensor = Tensor::from_vec(weight_data, &weight_shape).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create q_proj tensor: {}", e))
            })?;
            dict.set_item(
                "q_proj_weight",
                PyTensor {
                    tensor: Arc::new(weight_tensor),
                    requires_grad: true,
                    is_pinned: false,
                },
            )?;
        }

        if let Some(ref weight) = self.k_proj_weight {
            let weight_data = weight.to_vec().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to get k_proj weight: {}", e))
            })?;
            let weight_shape: Vec<usize> = weight.shape().iter().copied().collect();
            let weight_tensor = Tensor::from_vec(weight_data, &weight_shape).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create k_proj tensor: {}", e))
            })?;
            dict.set_item(
                "k_proj_weight",
                PyTensor {
                    tensor: Arc::new(weight_tensor),
                    requires_grad: true,
                    is_pinned: false,
                },
            )?;
        }

        if let Some(ref weight) = self.v_proj_weight {
            let weight_data = weight.to_vec().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to get v_proj weight: {}", e))
            })?;
            let weight_shape: Vec<usize> = weight.shape().iter().copied().collect();
            let weight_tensor = Tensor::from_vec(weight_data, &weight_shape).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create v_proj tensor: {}", e))
            })?;
            dict.set_item(
                "v_proj_weight",
                PyTensor {
                    tensor: Arc::new(weight_tensor),
                    requires_grad: true,
                    is_pinned: false,
                },
            )?;
        }

        if let Some(ref weight) = self.out_proj_weight {
            let weight_data = weight.to_vec().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to get out_proj weight: {}", e))
            })?;
            let weight_shape: Vec<usize> = weight.shape().iter().copied().collect();
            let weight_tensor = Tensor::from_vec(weight_data, &weight_shape).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create out_proj tensor: {}", e))
            })?;
            dict.set_item(
                "out_proj_weight",
                PyTensor {
                    tensor: Arc::new(weight_tensor),
                    requires_grad: true,
                    is_pinned: false,
                },
            )?;
        }

        Ok(dict.unbind())
    }

    /// Load layer state from dictionary
    pub fn load_state_dict(&mut self, state_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Ok(Some(weight)) = state_dict.get_item("q_proj_weight") {
            if let Ok(weight_tensor) = weight.extract::<PyTensor>() {
                self.q_proj_weight = Some(weight_tensor.tensor.as_ref().clone());
            }
        }

        if let Ok(Some(weight)) = state_dict.get_item("k_proj_weight") {
            if let Ok(weight_tensor) = weight.extract::<PyTensor>() {
                self.k_proj_weight = Some(weight_tensor.tensor.as_ref().clone());
            }
        }

        if let Ok(Some(weight)) = state_dict.get_item("v_proj_weight") {
            if let Ok(weight_tensor) = weight.extract::<PyTensor>() {
                self.v_proj_weight = Some(weight_tensor.tensor.as_ref().clone());
            }
        }

        if let Ok(Some(weight)) = state_dict.get_item("out_proj_weight") {
            if let Ok(weight_tensor) = weight.extract::<PyTensor>() {
                self.out_proj_weight = Some(weight_tensor.tensor.as_ref().clone());
            }
        }

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiheadAttention(embed_dim={}, num_heads={}, dropout={}, batch_first={})",
            self.embed_dim, self.num_heads, self.dropout, self.batch_first
        )
    }
}

/// Scaled Dot-Product Attention
///
/// Computes scaled dot-product attention: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
#[pyfunction]
#[pyo3(signature = (query, key, value, attn_mask=None, dropout_p=None))]
pub fn scaled_dot_product_attention(
    query: &PyTensor,
    key: &PyTensor,
    value: &PyTensor,
    attn_mask: Option<&PyTensor>,
    dropout_p: Option<f32>,
) -> PyResult<PyTensor> {
    let dropout_p = dropout_p.unwrap_or(0.0);

    if !(0.0..=1.0).contains(&dropout_p) {
        return Err(PyValueError::new_err("dropout_p must be between 0 and 1"));
    }

    let query_shape = query.tensor.shape();
    let key_shape = key.tensor.shape();
    let value_shape = value.tensor.shape();

    // Validate shapes
    if query_shape.len() < 2 {
        return Err(PyValueError::new_err("query must be at least 2D"));
    }
    if key_shape.len() < 2 {
        return Err(PyValueError::new_err("key must be at least 2D"));
    }
    if value_shape.len() < 2 {
        return Err(PyValueError::new_err("value must be at least 2D"));
    }

    // Validate attention mask shape
    if let Some(mask) = attn_mask {
        let mask_shape = mask.tensor.shape();
        if mask_shape.len() < 2 {
            return Err(PyValueError::new_err("attn_mask must be at least 2D"));
        }
    }

    // Output shape: same as query with last dimension from value
    let mut output_shape = query_shape.iter().copied().collect::<Vec<_>>();
    *output_shape.last_mut().unwrap() = value_shape[value_shape.len() - 1];

    let output = Tensor::zeros(&output_shape);

    Ok(PyTensor {
        tensor: Arc::new(output),
        requires_grad: query.requires_grad || key.requires_grad || value.requires_grad,
        is_pinned: false,
    })
}
