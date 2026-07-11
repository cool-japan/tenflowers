//! Transformer building blocks module for TenfloweRS FFI
//!
//! This module provides transformer architecture components including encoder/decoder
//! layers and positional encodings for sequence-to-sequence models.
//!
//! # Autograd
//!
//! [`PyTransformerEncoderLayer`]/[`PyTransformerDecoderLayer`] compose from
//! [`PyMultiheadAttention`] (already tape-aware — see that type's own module
//! doc for the full per-`(batch, head)` design this reuses unchanged) plus a
//! position-wise feedforward network (`ff_w1`/`ff_b1`/`ff_w2`/`ff_b2`, each a
//! [`Py<PyParameter>`] following the exact same stable-identity pattern) plus
//! LayerNorm.
//!
//! ## LayerNorm: reuse [`super::normalization::PyLayerNorm`], don't re-derive it
//!
//! Rather than hand-rolling `TernaryOpKind::LayerNorm` recording again in this
//! file, each layer holds one internal [`super::normalization::PyLayerNorm`]
//! value (not `Py<PyLayerNorm>` — `PyLayerNorm` already manages its own
//! `gamma_param`/`beta_param` identity internally via `Py<PyParameter>`, and
//! is itself `Clone`, so a plain field is sufficient) and calls its real,
//! already-tape-aware `forward()`/`parameters()` directly. This reuses a layer
//! already fully verified this session and avoids any chance of diverging
//! from the verified LayerNorm backward wiring.
//!
//! Per this codebase's existing, deliberate behaviour (predating this
//! autograd migration): the encoder layer holds exactly **one** internal
//! `PyLayerNorm`, reused at BOTH residual sites (self-attention and
//! feedforward); the decoder layer holds exactly **one** internal
//! `PyLayerNorm`, reused at ALL THREE residual sites (self-attention,
//! cross-attention, feedforward). This is unusual vs. standard Transformer
//! architecture (which normally uses 2 or 3 *independent* LayerNorm instances
//! with independent gamma/beta each) but is preserved exactly here, not
//! "fixed" to independent LayerNorms.
//!
//! `PyTransformerEncoderLayer::parameters()`/`PyTransformerDecoderLayer::parameters()`
//! return every trainable weight reachable from the composed layer:
//! `self_attn.parameters()` (+ `cross_attn.parameters()` for the decoder) +
//! `[ff_w1, ff_b1, ff_w2, ff_b2]` + `layer_norm.parameters()` (gamma, beta).
//!
//! [`PyPositionalEncoding`] has no trainable parameters (`pe` is a
//! precomputed, fixed sinusoidal buffer, never learned) — its `forward()` is
//! migrated to a real tape-aware [`PyTensor::add`] purely so gradient flows
//! *through* the add to whatever feeds `x`, even though `pe` itself carries no
//! gradient (addition's backward for a non-differentiable operand is simply
//! "receives no gradient, but the other side still does" — see
//! [`crate::implicit_autograd`]'s `BinaryOpKind::Add` wiring).

use super::attention::PyMultiheadAttention;
use super::layers::PyParameter;
use super::normalization::PyLayerNorm;
use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;
use tenflowers_core::{Result as CoreResult, Tensor};

/// Initialise a tensor with scaled standard-normal values.
fn randn_scaled(shape: &[usize], scale: f32) -> CoreResult<Tensor<f32>> {
    Tensor::randn(shape)?.multiply_scalar(scale)
}

/// Construct a fresh `Py<PyParameter>` from a raw `Tensor<f32>`, always
/// `requires_grad=true` — the shape every learnable weight/bias in this file
/// is constructed with.
fn new_param(py: Python<'_>, tensor: Tensor<f32>) -> PyResult<Py<PyParameter>> {
    Py::new(
        py,
        PyParameter::new(
            PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: true,
                is_pinned: false,
            },
            Some(true),
        ),
    )
}

/// Snapshot `param`'s current value and (re-)register it as this forward
/// pass's tape leaf. Idempotent per parameter `id()` (see
/// [`crate::implicit_autograd::mark_leaf_param`]'s own doc), so calling this
/// on every `forward()` call is safe and cheap after the first call within a
/// given forward/backward cycle.
fn snapshot_and_mark(py: Python<'_>, param: &Py<PyParameter>) -> PyResult<PyTensor> {
    let snapshot = param.borrow(py).to_tensor()?;
    let id = param.borrow(py).id();
    crate::implicit_autograd::mark_leaf_param(&snapshot, id);
    Ok(snapshot)
}

/// Build a non-leaf, non-differentiable `PyTensor` wrapping `tensor`.
fn constant(tensor: Tensor<f32>) -> PyTensor {
    PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    }
}

/// Apply a tape-aware linear projection `input @ weight^T (+ bias)`, where
/// `input` is `[dim0, dim1, in_features]` and `weight` is `[out_features,
/// in_features]`.
///
/// Every [`PyTensor::matmul`] call in this function operates on genuinely
/// 2-D operands (matmul on this tape is only gradient-correct for rank <= 2
/// operands — see [`super::attention`]'s module-level doc for the full
/// reasoning, which applies identically here): `input` is reshaped down to
/// `[dim0*dim1, in_features]` (a safe, tape-aware merge-adjacent-dims
/// reshape) before the matmul, and the result is reshaped back up to `[dim0,
/// dim1, out_features]` afterward.
fn linear_projection_tape(
    input: &PyTensor,
    weight: &PyTensor,
    bias: &PyTensor,
) -> PyResult<PyTensor> {
    let dims = input.tensor.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(PyRuntimeError::new_err(format!(
            "linear_projection_tape: expected a 3D input, got {}D",
            dims.len()
        )));
    }
    let (dim0, dim1, in_features) = (dims[0], dims[1], dims[2]);
    let out_features = weight.tensor.shape().dims()[0];

    let input_2d = crate::tensor_ops::reshape(input, vec![dim0 * dim1, in_features])?;
    let weight_t = weight.transpose(Some(vec![1, 0]))?;
    let projected_2d = input_2d.matmul(&weight_t)?.add(bias)?;

    crate::tensor_ops::reshape(&projected_2d, vec![dim0, dim1, out_features])
}

/// Position-wise feed-forward network: `linear2(activation(linear1(x)))`,
/// computed entirely through tape-aware operations.
fn feed_forward_tape(
    input: &PyTensor,
    w1: &PyTensor,
    b1: &PyTensor,
    w2: &PyTensor,
    b2: &PyTensor,
    activation: &str,
) -> PyResult<PyTensor> {
    let hidden = linear_projection_tape(input, w1, b1)?;
    let activated = if activation == "gelu" {
        super::functions::gelu(&hidden)?
    } else {
        super::functions::relu(&hidden)?
    };
    linear_projection_tape(&activated, w2, b2)
}

/// Build a `PyMultiheadAttention` sharing the SAME `batch_first` layout as
/// the outer `PyTransformerEncoderLayer`/`PyTransformerDecoderLayer` that
/// owns it.
///
/// This match is load-bearing, not a stylistic choice: `forward()` passes
/// `src`/`tgt`/`memory` straight through to `self_attn.forward`/
/// `cross_attn.forward` with no realignment step in between (unlike the
/// pre-migration code, which physically transposed to/from a hardcoded
/// batch-first layout via `align_batch_first` around every attention call).
/// If the inner attention's own `batch_first` ever diverged from the outer
/// layer's, it would silently misinterpret which axis is `batch` and which
/// is `seq` for any `PyTransformerEncoderLayer`/`PyTransformerDecoderLayer`
/// constructed with `batch_first=false`. Passing `batch_first` through here
/// (rather than hardcoding `Some(true)`) is what makes the realignment-free
/// design correct for both layouts, not just the batch-first one.
fn build_attention(
    py: Python<'_>,
    d_model: usize,
    nhead: usize,
    batch_first: bool,
) -> PyResult<PyMultiheadAttention> {
    PyMultiheadAttention::new(
        py,
        d_model,
        nhead,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(batch_first),
    )
}

/// Transformer Encoder Layer
///
/// A single layer of the transformer encoder with multi-head attention and feedforward network.
#[pyclass(name = "TransformerEncoderLayer")]
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
    ff_w1_param: Py<PyParameter>,
    /// Feed-forward bias 1: `[dim_feedforward]`
    ff_b1_param: Py<PyParameter>,
    /// Feed-forward weight 2: `[d_model, dim_feedforward]`
    ff_w2_param: Py<PyParameter>,
    /// Feed-forward bias 2: `[d_model]`
    ff_b2_param: Py<PyParameter>,
    /// Shared LayerNorm, reused at BOTH residual sites (self-attn and
    /// feedforward) — see the module-level "LayerNorm" doc for why this is
    /// exactly ONE instance, not one-per-call-site.
    layer_norm: PyLayerNorm,
}

impl Clone for PyTransformerEncoderLayer {
    /// Produce an **independent** layer: a fresh copy of every current
    /// parameter value, under **fresh** parameter identities distinct from
    /// `self`'s. See [`super::layers::PyDense`]'s `impl Clone` for the full
    /// rationale.
    ///
    /// # Why `layer_norm` cannot just be `self.layer_norm.clone()`
    ///
    /// [`PyLayerNorm`]'s own `Clone` impl uses [`Py::clone_ref`] (a shared,
    /// SAME-identity reference — appropriate for `PyLayerNorm`'s own
    /// call sites within `normalization.rs`, but wrong here: this impl's
    /// whole contract is that the clone's parameters are independent of
    /// `self`'s). Using it directly would silently give the clone's
    /// `layer_norm.gamma_param`/`beta_param` the exact same `id()` as
    /// `self`'s — caught by
    /// `encoder_clone_produces_independent_parameter_identities` during
    /// development. Instead, a fresh `PyLayerNorm` is constructed by hand
    /// from independently `clone_param()`-derived gamma/beta (accessing
    /// [`PyLayerNorm`]'s `pub` fields directly, since it has no
    /// `clone_independent`-style constructor of its own).
    fn clone(&self) -> Self {
        Python::attach(|py| {
            let ff_w1 = self.ff_w1_param.borrow(py).clone_param();
            let ff_b1 = self.ff_b1_param.borrow(py).clone_param();
            let ff_w2 = self.ff_w2_param.borrow(py).clone_param();
            let ff_b2 = self.ff_b2_param.borrow(py).clone_param();
            let ln_gamma = self.layer_norm.gamma_param.borrow(py).clone_param();
            let ln_beta = self.layer_norm.beta_param.borrow(py).clone_param();
            Self {
                d_model: self.d_model,
                nhead: self.nhead,
                dim_feedforward: self.dim_feedforward,
                dropout: self.dropout,
                activation: self.activation.clone(),
                batch_first: self.batch_first,
                layer_norm_eps: self.layer_norm_eps,
                self_attn: self.self_attn.clone(),
                ff_w1_param: Py::new(py, ff_w1)
                    .expect("PyParameter::clone_param()'s result must construct as a Py<PyParameter>"),
                ff_b1_param: Py::new(py, ff_b1)
                    .expect("PyParameter::clone_param()'s result must construct as a Py<PyParameter>"),
                ff_w2_param: Py::new(py, ff_w2)
                    .expect("PyParameter::clone_param()'s result must construct as a Py<PyParameter>"),
                ff_b2_param: Py::new(py, ff_b2)
                    .expect("PyParameter::clone_param()'s result must construct as a Py<PyParameter>"),
                layer_norm: PyLayerNorm {
                    normalized_shape: self.layer_norm.normalized_shape.clone(),
                    eps: self.layer_norm.eps,
                    gamma_param: Py::new(py, ln_gamma).expect(
                        "PyParameter::clone_param()'s result must construct as a Py<PyParameter>",
                    ),
                    beta_param: Py::new(py, ln_beta).expect(
                        "PyParameter::clone_param()'s result must construct as a Py<PyParameter>",
                    ),
                },
            }
        })
    }
}

impl std::fmt::Debug for PyTransformerEncoderLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyTransformerEncoderLayer")
            .field("d_model", &self.d_model)
            .field("nhead", &self.nhead)
            .field("dim_feedforward", &self.dim_feedforward)
            .field("dropout", &self.dropout)
            .field("activation", &self.activation)
            .field("batch_first", &self.batch_first)
            .finish()
    }
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
        py: Python<'_>,
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

        let self_attn = build_attention(py, d_model, nhead, batch_first)?;
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

        let ff_w1_param = new_param(py, ff_w1)?;
        let ff_b1_param = new_param(py, ff_b1)?;
        let ff_w2_param = new_param(py, ff_w2)?;
        let ff_b2_param = new_param(py, ff_b2)?;
        let layer_norm = PyLayerNorm::new(py, vec![d_model], Some(layer_norm_eps))?;

        Ok(PyTransformerEncoderLayer {
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            batch_first,
            layer_norm_eps,
            self_attn,
            ff_w1_param,
            ff_b1_param,
            ff_w2_param,
            ff_b2_param,
            layer_norm,
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
    ///
    /// # Tape-aware op sequence
    ///
    /// `self_attn.forward(...)` (see [`PyMultiheadAttention`]'s own op
    /// sequence) `-> add(residual) -> layer_norm.forward(...) ->
    /// [linear_projection_tape -> activation -> linear_projection_tape] (feed
    /// forward) -> add(residual) -> layer_norm.forward(...)`.
    #[pyo3(signature = (src, src_mask=None, src_key_padding_mask=None))]
    pub fn forward(
        &self,
        py: Python<'_>,
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

        // For a 3D [batch, seq, feature] or [seq, batch, feature] tensor,
        // the feature axis is always index 2 regardless of `batch_first`.
        let feature_dim = src_shape[2];

        if feature_dim != self.d_model {
            return Err(PyValueError::new_err(format!(
                "Expected feature dimension {}, got {}",
                self.d_model, feature_dim
            )));
        }

        // Self-attention sublayer (post-norm): x = LayerNorm(x + SelfAttn(x)).
        // `src_mask` is the [seq, seq] additive attention mask;
        // `src_key_padding_mask` is the [batch, seq] key-padding mask.
        let (attn_out, _) = self.self_attn.forward(
            py,
            src,
            src,
            src,
            src_key_padding_mask,
            Some(false),
            src_mask,
            None,
        )?;
        let residual1 = src.add(&attn_out)?;
        let normed1 = self.layer_norm.forward(py, &residual1)?;

        // Feed-forward sublayer (post-norm): x = LayerNorm(x + FFN(x)).
        let ff_w1 = snapshot_and_mark(py, &self.ff_w1_param)?;
        let ff_b1 = snapshot_and_mark(py, &self.ff_b1_param)?;
        let ff_w2 = snapshot_and_mark(py, &self.ff_w2_param)?;
        let ff_b2 = snapshot_and_mark(py, &self.ff_b2_param)?;
        let ff = feed_forward_tape(&normed1, &ff_w1, &ff_b1, &ff_w2, &ff_b2, &self.activation)?;
        let residual2 = normed1.add(&ff)?;
        let normed2 = self.layer_norm.forward(py, &residual2)?;

        Ok(normed2)
    }

    /// Get every trainable weight reachable from this composed layer:
    /// `self_attn.parameters()` + `[ff_w1, ff_b1, ff_w2, ff_b2]` +
    /// `layer_norm.parameters()` (gamma, beta).
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyParameter>> {
        let mut params = self.self_attn.parameters(py);
        params.push(self.ff_w1_param.clone_ref(py));
        params.push(self.ff_b1_param.clone_ref(py));
        params.push(self.ff_w2_param.clone_ref(py));
        params.push(self.ff_b2_param.clone_ref(py));
        params.extend(self.layer_norm.parameters(py));
        params
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
    ff_w1_param: Py<PyParameter>,
    /// Feed-forward bias 1: `[dim_feedforward]`
    ff_b1_param: Py<PyParameter>,
    /// Feed-forward weight 2: `[d_model, dim_feedforward]`
    ff_w2_param: Py<PyParameter>,
    /// Feed-forward bias 2: `[d_model]`
    ff_b2_param: Py<PyParameter>,
    /// Shared LayerNorm, reused at ALL THREE residual sites (self-attn,
    /// cross-attn, feedforward) — see the module-level "LayerNorm" doc for
    /// why this is exactly ONE instance, not one-per-call-site.
    layer_norm: PyLayerNorm,
}

impl Clone for PyTransformerDecoderLayer {
    /// See [`PyTransformerEncoderLayer`]'s `impl Clone` for the full
    /// rationale, including why `layer_norm` cannot just be
    /// `self.layer_norm.clone()`.
    fn clone(&self) -> Self {
        Python::attach(|py| {
            let ff_w1 = self.ff_w1_param.borrow(py).clone_param();
            let ff_b1 = self.ff_b1_param.borrow(py).clone_param();
            let ff_w2 = self.ff_w2_param.borrow(py).clone_param();
            let ff_b2 = self.ff_b2_param.borrow(py).clone_param();
            let ln_gamma = self.layer_norm.gamma_param.borrow(py).clone_param();
            let ln_beta = self.layer_norm.beta_param.borrow(py).clone_param();
            Self {
                d_model: self.d_model,
                nhead: self.nhead,
                dim_feedforward: self.dim_feedforward,
                dropout: self.dropout,
                activation: self.activation.clone(),
                batch_first: self.batch_first,
                layer_norm_eps: self.layer_norm_eps,
                self_attn: self.self_attn.clone(),
                cross_attn: self.cross_attn.clone(),
                ff_w1_param: Py::new(py, ff_w1)
                    .expect("PyParameter::clone_param()'s result must construct as a Py<PyParameter>"),
                ff_b1_param: Py::new(py, ff_b1)
                    .expect("PyParameter::clone_param()'s result must construct as a Py<PyParameter>"),
                ff_w2_param: Py::new(py, ff_w2)
                    .expect("PyParameter::clone_param()'s result must construct as a Py<PyParameter>"),
                ff_b2_param: Py::new(py, ff_b2)
                    .expect("PyParameter::clone_param()'s result must construct as a Py<PyParameter>"),
                layer_norm: PyLayerNorm {
                    normalized_shape: self.layer_norm.normalized_shape.clone(),
                    eps: self.layer_norm.eps,
                    gamma_param: Py::new(py, ln_gamma).expect(
                        "PyParameter::clone_param()'s result must construct as a Py<PyParameter>",
                    ),
                    beta_param: Py::new(py, ln_beta).expect(
                        "PyParameter::clone_param()'s result must construct as a Py<PyParameter>",
                    ),
                },
            }
        })
    }
}

impl std::fmt::Debug for PyTransformerDecoderLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyTransformerDecoderLayer")
            .field("d_model", &self.d_model)
            .field("nhead", &self.nhead)
            .field("dim_feedforward", &self.dim_feedforward)
            .field("dropout", &self.dropout)
            .field("activation", &self.activation)
            .field("batch_first", &self.batch_first)
            .finish()
    }
}

#[pymethods]
impl PyTransformerDecoderLayer {
    /// Create a new transformer decoder layer
    #[new]
    #[pyo3(signature = (d_model, nhead, dim_feedforward=None, dropout=None, activation=None, batch_first=None, layer_norm_eps=None))]
    pub fn new(
        py: Python<'_>,
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

        let self_attn = build_attention(py, d_model, nhead, batch_first)?;
        let cross_attn = build_attention(py, d_model, nhead, batch_first)?;
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

        let ff_w1_param = new_param(py, ff_w1)?;
        let ff_b1_param = new_param(py, ff_b1)?;
        let ff_w2_param = new_param(py, ff_w2)?;
        let ff_b2_param = new_param(py, ff_b2)?;
        let layer_norm = PyLayerNorm::new(py, vec![d_model], Some(layer_norm_eps))?;

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
            ff_w1_param,
            ff_b1_param,
            ff_w2_param,
            ff_b2_param,
            layer_norm,
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
    ///
    /// # Tape-aware op sequence
    ///
    /// `self_attn.forward(...) -> add(residual) -> layer_norm.forward(...) ->
    /// cross_attn.forward(...) -> add(residual) -> layer_norm.forward(...) ->
    /// [feed forward] -> add(residual) -> layer_norm.forward(...)`.
    #[pyo3(signature = (tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None))]
    pub fn forward(
        &self,
        py: Python<'_>,
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

        // Masked self-attention sublayer (post-norm). `tgt_mask` is the
        // [tgt, tgt] attention mask; `tgt_key_padding_mask` is [batch, tgt].
        let (sa_out, _) = self.self_attn.forward(
            py,
            tgt,
            tgt,
            tgt,
            tgt_key_padding_mask,
            Some(false),
            tgt_mask,
            None,
        )?;
        let residual1 = tgt.add(&sa_out)?;
        let normed1 = self.layer_norm.forward(py, &residual1)?;

        // Cross-attention sublayer: query = decoder state, key/value = memory.
        // `memory_mask` is the [tgt, src] attention mask;
        // `memory_key_padding_mask` is [batch, src] where src is the memory
        // sequence length.
        let (ca_out, _) = self.cross_attn.forward(
            py,
            &normed1,
            memory,
            memory,
            memory_key_padding_mask,
            Some(false),
            memory_mask,
            None,
        )?;
        let residual2 = normed1.add(&ca_out)?;
        let normed2 = self.layer_norm.forward(py, &residual2)?;

        // Feed-forward sublayer.
        let ff_w1 = snapshot_and_mark(py, &self.ff_w1_param)?;
        let ff_b1 = snapshot_and_mark(py, &self.ff_b1_param)?;
        let ff_w2 = snapshot_and_mark(py, &self.ff_w2_param)?;
        let ff_b2 = snapshot_and_mark(py, &self.ff_b2_param)?;
        let ff = feed_forward_tape(&normed2, &ff_w1, &ff_b1, &ff_w2, &ff_b2, &self.activation)?;
        let residual3 = normed2.add(&ff)?;
        let normed3 = self.layer_norm.forward(py, &residual3)?;

        Ok(normed3)
    }

    /// Get every trainable weight reachable from this composed layer:
    /// `self_attn.parameters()` + `cross_attn.parameters()` + `[ff_w1, ff_b1,
    /// ff_w2, ff_b2]` + `layer_norm.parameters()` (gamma, beta).
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyParameter>> {
        let mut params = self.self_attn.parameters(py);
        params.extend(self.cross_attn.parameters(py));
        params.push(self.ff_w1_param.clone_ref(py));
        params.push(self.ff_b1_param.clone_ref(py));
        params.push(self.ff_w2_param.clone_ref(py));
        params.push(self.ff_b2_param.clone_ref(py));
        params.extend(self.layer_norm.parameters(py));
        params
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
    /// Precomputed positional encodings (fixed, non-differentiable — see the
    /// module-level "Autograd" doc).
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
    ///
    /// # Tape-aware op sequence
    ///
    /// `add` (broadcast-add a non-leaf, `requires_grad=false` `pe` slice
    /// against `x` — `pe` itself carries no gradient, but `x`'s gradient
    /// still flows through this add unchanged).
    #[pyo3(signature = (x, batch_first=None))]
    pub fn forward(&self, x: &PyTensor, batch_first: Option<bool>) -> PyResult<PyTensor> {
        let batch_first = batch_first.unwrap_or(false);
        let x_shape = x.tensor.shape().dims().to_vec();

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

        // Materialise the full-batch `pe` tensor directly (rather than
        // relying on `.add()`'s broadcasting over a `[1, seq_len, d_model]`/
        // `[seq_len, 1, d_model]` slice, which would ALSO be correct but adds
        // an extra tape node for no benefit here since `pe` needs no tape
        // leaf marking either way): the same `[seq_len, d_model]` positional
        // block is repeated once per batch element, laid out to match `x`'s
        // own `batch_first` axis order exactly, then added via a single real
        // `PyTensor::add`.
        let pe_slice = &self.pe[..seq_len * d_model];
        let mut pe_data = Vec::with_capacity(seq_len * batch * d_model);
        if batch_first {
            // x is [batch, seq_len, d_model]: repeat the whole [seq_len,
            // d_model] block once per batch element.
            for _ in 0..batch {
                pe_data.extend_from_slice(pe_slice);
            }
        } else {
            // x is [seq_len, batch, d_model]: repeat each position's
            // [d_model] row once per batch element, position-major.
            for pos in 0..seq_len {
                let row = &pe_slice[pos * d_model..(pos + 1) * d_model];
                for _ in 0..batch {
                    pe_data.extend_from_slice(row);
                }
            }
        }
        let pe_tensor = Tensor::from_vec(pe_data, &x_shape)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to build pe tensor: {}", e)))?;

        x.add(&constant(pe_tensor))
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
mod tests;
