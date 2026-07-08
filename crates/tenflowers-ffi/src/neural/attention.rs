//! Attention mechanisms module for TenfloweRS FFI
//!
//! This module provides attention mechanism implementations including multi-head attention
//! for transformers and other sequence-to-sequence models.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tenflowers_core::{Result as CoreResult, Tensor};
use tenflowers_neural::layers::attention::scaled_dot_product_attention as neural_sdpa;

/// Initialise a tensor with scaled standard-normal values.
fn randn_scaled(shape: &[usize], scale: f32) -> CoreResult<Tensor<f32>> {
    Tensor::randn(shape)?.multiply_scalar(scale)
}

/// Swap the first two axes of a 3-D tensor when the data is not batch-first,
/// converting between `[seq, batch, embed]` and `[batch, seq, embed]`.
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

/// Apply a linear projection `input @ weight^T (+ bias)` where `weight` is
/// stored as `[out_features, in_features]`.
fn linear_projection(
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

        // Initialize projection weights with scaled standard-normal values so the
        // layer performs a real (non-degenerate) projection.
        let scale = 1.0_f32 / (embed_dim as f32).sqrt();
        let init_err = |e: tenflowers_core::TensorError| {
            PyRuntimeError::new_err(format!("init failed: {}", e))
        };
        let q_proj_weight = randn_scaled(&[embed_dim, embed_dim], scale).map_err(init_err)?;
        let k_proj_weight = randn_scaled(&[embed_dim, kdim_actual], scale).map_err(init_err)?;
        let v_proj_weight = randn_scaled(&[embed_dim, vdim_actual], scale).map_err(init_err)?;
        let out_proj_weight = randn_scaled(&[embed_dim, embed_dim], scale).map_err(init_err)?;

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

        let to_err = |e: tenflowers_core::TensorError| {
            PyRuntimeError::new_err(format!("MultiheadAttention forward failed: {}", e))
        };

        // Obtain projection weights (honest error if uninitialized).
        let wq = self.q_proj_weight.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("MultiheadAttention: q_proj_weight not initialized")
        })?;
        let wk = self.k_proj_weight.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("MultiheadAttention: k_proj_weight not initialized")
        })?;
        let wv = self.v_proj_weight.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("MultiheadAttention: v_proj_weight not initialized")
        })?;
        let wo = self.out_proj_weight.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("MultiheadAttention: out_proj_weight not initialized")
        })?;

        // Align Q/K/V to batch-first `[batch, seq, embed]`.
        let query_bf = align_batch_first(&query.tensor, self.batch_first).map_err(to_err)?;
        let key_bf = align_batch_first(&key.tensor, self.batch_first).map_err(to_err)?;
        let value_bf = align_batch_first(&value.tensor, self.batch_first).map_err(to_err)?;

        // Linear projections into the model dimension.
        let q_proj = linear_projection(&query_bf, wq, self.bias_weight.as_ref()).map_err(to_err)?;
        let k_proj = linear_projection(&key_bf, wk, self.bias_weight.as_ref()).map_err(to_err)?;
        let v_proj = linear_projection(&value_bf, wv, self.bias_weight.as_ref()).map_err(to_err)?;

        let q_dims = q_proj.shape().dims().to_vec();
        let k_dims = k_proj.shape().dims().to_vec();
        let (batch, tgt_len) = (q_dims[0], q_dims[1]);
        let src_len = k_dims[1];

        // Combine the optional additive attention mask and the key-padding mask
        // into a single [batch, tgt_len, src_len] additive bias. This folds
        // `key_padding_mask` into the real masking math instead of silently
        // ignoring it, and reproduces the exact unmasked path when both are None.
        let combined_mask = tenflowers_neural::layers::attention::combine_attention_masks(
            attn_mask.map(|m| m.tensor.as_ref()),
            key_padding_mask.map(|m| m.tensor.as_ref()),
            batch,
            tgt_len,
            src_len,
        )
        .map_err(to_err)?;

        // Per-head scaled dot-product attention.
        let mut head_outputs: Vec<Tensor<f32>> = Vec::with_capacity(self.num_heads);
        let mut head_weights: Vec<Tensor<f32>> = Vec::with_capacity(self.num_heads);
        for h in 0..self.num_heads {
            let start = h * self.head_dim;
            let end = start + self.head_dim;
            let q_head = q_proj
                .slice(&[0..batch, 0..tgt_len, start..end])
                .map_err(to_err)?;
            let k_head = k_proj
                .slice(&[0..batch, 0..src_len, start..end])
                .map_err(to_err)?;
            let v_head = v_proj
                .slice(&[0..batch, 0..src_len, start..end])
                .map_err(to_err)?;
            let (out_head, weights_head) = neural_sdpa(
                &q_head,
                &k_head,
                &v_head,
                combined_mask.as_ref(),
                0.0,
                false,
            )
            .map_err(to_err)?;
            head_outputs.push(out_head);
            head_weights.push(weights_head);
        }

        // Concatenate head outputs and apply the output projection.
        let head_refs: Vec<&Tensor<f32>> = head_outputs.iter().collect();
        let combined = tenflowers_core::ops::concat(&head_refs, 2).map_err(to_err)?;
        let projected =
            linear_projection(&combined, wo, self.bias_weight.as_ref()).map_err(to_err)?;
        let attn_output = align_batch_first(&projected, self.batch_first).map_err(to_err)?;

        // Average the real attention weights across heads
        // (PyTorch's `average_attn_weights=True`).
        let attn_weights = if need_weights {
            let mut acc = head_weights
                .first()
                .ok_or_else(|| PyRuntimeError::new_err("MultiheadAttention: no attention heads"))?
                .clone();
            for w in head_weights.iter().skip(1) {
                acc = acc.add(w).map_err(to_err)?;
            }
            let averaged = acc
                .multiply_scalar(1.0 / self.num_heads as f32)
                .map_err(to_err)?;
            Some(PyTensor {
                tensor: Arc::new(averaged),
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

        let scale = 1.0_f32 / (self.embed_dim as f32).sqrt();
        let init_err = |e: tenflowers_core::TensorError| {
            PyRuntimeError::new_err(format!("init failed: {}", e))
        };
        self.q_proj_weight =
            Some(randn_scaled(&[self.embed_dim, self.embed_dim], scale).map_err(init_err)?);
        self.k_proj_weight =
            Some(randn_scaled(&[self.embed_dim, kdim_actual], scale).map_err(init_err)?);
        self.v_proj_weight =
            Some(randn_scaled(&[self.embed_dim, vdim_actual], scale).map_err(init_err)?);
        self.out_proj_weight =
            Some(randn_scaled(&[self.embed_dim, self.embed_dim], scale).map_err(init_err)?);

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

    // The neural scaled dot-product attention operates on 3-D tensors
    // `[batch, seq, d_k]`; require that here and return an honest error otherwise.
    if query_shape.len() != 3 || key_shape.len() != 3 || value_shape.len() != 3 {
        return Err(PyValueError::new_err(
            "scaled_dot_product_attention requires 3D tensors [batch, seq, d_k]",
        ));
    }

    let (output, _weights) = neural_sdpa(
        &query.tensor,
        &key.tensor,
        &value.tensor,
        attn_mask.map(|m| m.tensor.as_ref()),
        dropout_p,
        false,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("scaled_dot_product_attention failed: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(output),
        requires_grad: query.requires_grad || key.requires_grad || value.requires_grad,
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
    fn multihead_attention_is_real_with_normalized_weights() {
        // embed_dim = 8, num_heads = 2, batch_first.
        let mha = PyMultiheadAttention::new(8, 2, None, None, None, None, None, None, Some(true))
            .expect("mha construction");
        let q = make_tensor(ramp(2 * 3 * 8), &[2, 3, 8]);
        let k = make_tensor(ramp(2 * 3 * 8), &[2, 3, 8]);
        let v = make_tensor(ramp(2 * 3 * 8), &[2, 3, 8]);

        let (output, weights) = mha
            .forward(&q, &k, &v, None, Some(true), None, None)
            .expect("forward");

        assert_eq!(output.tensor.shape().dims().to_vec(), vec![2, 3, 8]);
        let out_vec = output.tensor.to_vec().expect("vec");
        assert!(
            out_vec.iter().any(|&x| x != 0.0),
            "attention output must not be all zeros"
        );

        let weights = weights.expect("attention weights present");
        assert_eq!(weights.tensor.shape().dims().to_vec(), vec![2, 3, 3]);
        let w = weights.tensor.to_vec().expect("weights vec");
        // Weights shape [batch=2, tgt=3, src=3]: every row over `src` sums to ~1.
        let src = 3usize;
        for row in 0..(2 * 3) {
            let sum: f32 = (0..src).map(|j| w[row * src + j]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "attention weight row must sum to ~1, got {}",
                sum
            );
        }
    }

    #[test]
    fn scaled_dot_product_attention_is_real() {
        let q = make_tensor(ramp(8), &[1, 2, 4]);
        let k = make_tensor(ramp(8), &[1, 2, 4]);
        let v = make_tensor(vec![2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0], &[1, 2, 4]);

        let out = scaled_dot_product_attention(&q, &k, &v, None, None).expect("sdpa");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 2, 4]);
        let out_vec = out.tensor.to_vec().expect("vec");
        assert!(
            out_vec.iter().any(|&x| x != 0.0),
            "SDPA output must not be all zeros"
        );
    }

    /// 2x2 identity matrix used to make projections a no-op so that pre-softmax
    /// scores are exactly `Q·K^T / sqrt(d_k)` and can be reasoned about by hand.
    fn identity_2x2() -> Tensor<f32> {
        Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], &[2, 2]).expect("identity")
    }

    #[test]
    fn multihead_attention_key_padding_mask_zeros_padded_weight() {
        // embed_dim=2, num_heads=1, batch_first: single head sees the full embed.
        let mut mha =
            PyMultiheadAttention::new(2, 1, None, None, None, None, None, None, Some(true))
                .expect("mha construction");
        // Identity projections => scores are exactly Q·K^T / sqrt(2).
        mha.q_proj_weight = Some(identity_2x2());
        mha.k_proj_weight = Some(identity_2x2());
        mha.v_proj_weight = Some(identity_2x2());
        mha.out_proj_weight = Some(identity_2x2());
        mha.bias_weight = None;

        // batch=1, tgt_len=1, src_len=3, embed=2.
        // q=[1,0]; k0=[2,0], k1=[1,0], k2=[0,0] => raw scores [2,1,0].
        let q = make_tensor(vec![1.0, 0.0], &[1, 1, 2]);
        let k = make_tensor(vec![2.0, 0.0, 1.0, 0.0, 0.0, 0.0], &[1, 3, 2]);
        // v0=[1,0], v1=[0,1], v2=[5,5]: masking v2 (large) changes the output a lot.
        let v = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 5.0, 5.0], &[1, 3, 2]);

        // Unmasked: attention spreads across all three source positions.
        let (out_unmasked, w_unmasked) = mha
            .forward(&q, &k, &v, None, Some(true), None, None)
            .expect("unmasked forward");
        let wu = w_unmasked
            .expect("weights present")
            .tensor
            .to_vec()
            .expect("wu");
        assert_eq!(wu.len(), 3);
        assert!(
            wu[2] > 0.01,
            "without a mask, padded pos 2 must carry real weight, got {}",
            wu[2]
        );

        // Masked: key_padding_mask marks source position 2 as padding (nonzero).
        let kpm = make_tensor(vec![0.0, 0.0, 1.0], &[1, 3]);
        let (out_masked, w_masked) = mha
            .forward(&q, &k, &v, Some(&kpm), Some(true), None, None)
            .expect("masked forward");
        let wm = w_masked
            .expect("weights present")
            .tensor
            .to_vec()
            .expect("wm");

        // (b) padded position weight is ~0, remaining weights still sum to ~1.
        assert!(wm[2] < 1e-3, "padded weight must be ~0, got {}", wm[2]);
        assert!(
            (wm[0] + wm[1] - 1.0).abs() < 1e-3,
            "surviving weights must sum to ~1, got {}",
            wm[0] + wm[1]
        );
        assert!(
            wm[0] > wm[1],
            "pos 0 has the higher score so must retain more weight ({} vs {})",
            wm[0],
            wm[1]
        );

        // (a) the attention output changes vs the unmasked case.
        let ou = out_unmasked.tensor.to_vec().expect("ou");
        let om = out_masked.tensor.to_vec().expect("om");
        let changed = ou.iter().zip(om.iter()).any(|(a, b)| (a - b).abs() > 1e-3);
        assert!(changed, "masking a key must change the attention output");
    }

    #[test]
    fn multihead_attention_combines_attn_and_key_padding_masks() {
        let mut mha =
            PyMultiheadAttention::new(2, 1, None, None, None, None, None, None, Some(true))
                .expect("mha construction");
        mha.q_proj_weight = Some(identity_2x2());
        mha.k_proj_weight = Some(identity_2x2());
        mha.v_proj_weight = Some(identity_2x2());
        mha.out_proj_weight = Some(identity_2x2());
        mha.bias_weight = None;

        let q = make_tensor(vec![1.0, 0.0], &[1, 1, 2]);
        let k = make_tensor(vec![2.0, 0.0, 1.0, 0.0, 0.0, 0.0], &[1, 3, 2]);
        let v = make_tensor(vec![1.0, 0.0, 0.0, 1.0, 5.0, 5.0], &[1, 3, 2]);

        // attn_mask [tgt=1, src=3] blocks source position 0;
        // key_padding_mask [batch=1, src=3] blocks a DIFFERENT position (2).
        let attn_mask = make_tensor(vec![-1.0e9, 0.0, 0.0], &[1, 3]);
        let kpm = make_tensor(vec![0.0, 0.0, 1.0], &[1, 3]);

        let (_out, weights) = mha
            .forward(&q, &k, &v, Some(&kpm), Some(true), Some(&attn_mask), None)
            .expect("masked forward");
        let w = weights
            .expect("weights present")
            .tensor
            .to_vec()
            .expect("w");

        // Both masked positions must be ~0 simultaneously; only pos 1 survives.
        assert!(w[0] < 1e-3, "attn-masked pos 0 must be ~0, got {}", w[0]);
        assert!(
            w[2] < 1e-3,
            "key-padding-masked pos 2 must be ~0, got {}",
            w[2]
        );
        assert!(
            (w[1] - 1.0).abs() < 1e-3,
            "the single surviving pos 1 must keep ~all the mass, got {}",
            w[1]
        );
    }
}
