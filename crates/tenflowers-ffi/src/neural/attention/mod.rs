//! Attention mechanisms module for TenfloweRS FFI
//!
//! This module provides attention mechanism implementations including multi-head attention
//! for transformers and other sequence-to-sequence models.
//!
//! # Autograd
//!
//! [`PyMultiheadAttention`]'s four projection weights (`q_proj`/`k_proj`/`v_proj`/
//! `out_proj`) and its single shared bias (see the "One shared bias" note below)
//! are each held as a [`Py<PyParameter>`] — the same stable-identity,
//! mutable-in-place parameter cell [`super::layers::PyDense`] and
//! [`super::normalization::PyLayerNorm`] already use (see those types' own docs
//! for the full design this mirrors). `forward()` snapshots each parameter's
//! current value via [`super::layers::PyParameter::to_tensor`], marks the
//! snapshot as a tape leaf via [`crate::implicit_autograd::mark_leaf_param`]
//! (keyed by the parameter's own stable `id`, not the snapshot's own transient
//! identity), and computes the ENTIRE forward pass using only tape-aware
//! [`PyTensor`] operations — never `tenflowers_neural::layers::attention`'s own
//! raw, non-tape `scaled_dot_product_attention` free function, and never a
//! `to_vec()`/`from_vec()` round-trip through untracked data partway through
//! the computation (either would silently drop the tape edge at that point).
//! `.parameters()` returns `self.<field>.clone_ref(py)` for every parameter —
//! sharing identity with whatever `forward()` reads, so `.grad()` on the
//! returned handles is populated after `.backward()`, and an optimizer's
//! `.set_data()` through one of those handles is visible to this layer's own
//! next `forward()` call. **Never** [`super::layers::PyParameter::clone_param`]
//! here — that mints a fresh, independent id, breaking the optimizer round-trip
//! (see [`super::layers::PyDense::parameters`]'s doc for the full failure mode
//! this would cause).
//!
//! ## One shared bias across all four projections
//!
//! Unlike standard PyTorch (which gives Q/K/V/out each an independent bias),
//! this layer reuses a single `bias_param` for all four linear projections —
//! this is this codebase's existing, deliberate behaviour (predating this
//! autograd migration) and is preserved exactly here, not "fixed" to four
//! independent biases.
//!
//! ## Why the per-(batch, head) attention loop, and why it cannot be batched
//!
//! `process_matmul_backward` (in `tenflowers-autograd`) computes each operand's
//! gradient via `tenflowers_core::ops::transpose(&other_operand)` — the PLAIN
//! transpose free function, which reverses ALL axes of the tensor (confirmed by
//! reading `tenflowers_core::ops::manipulation::transpose_axes`: `axes: None`
//! defaults to `(0..rank).rev().collect()`). For a rank-2 operand this
//! coincidentally IS the correct transpose (only two axes exist, so "reverse
//! all" and "swap the last two" are the same permutation) — but for any rank-3+
//! operand, reversing ALL axes computes completely the wrong tensor for the
//! matmul gradient formula. There is no `BatchMatMul`/batched op kind anywhere
//! on this tape. Consequently, **every** [`PyTensor::matmul`] call in this
//! file's `forward()` operates on genuinely 2-D operands — the per-head
//! scaled-dot-product attention is computed via an explicit loop over
//! `(batch_idx, head_idx)` pairs, each iteration slicing out real 2-D
//! `[seq, head_dim]` matrices before ever calling `.matmul()`. This is
//! measurably slower than a single batched 3-D matmul per head would be; that
//! is an accepted, deliberate trade-off — a fast-but-wrong-gradient forward
//! path is a worse defect than a slower, provably-correct one.
//!
//! The same rank restriction applies to [`PyTensor::transpose`]: its `axes`
//! parameter is (independently, in `tenflowers-autograd`) confirmed to be
//! ignored by both the forward computation and the backward gradient of the
//! underlying `TrackedTensor::transpose`, which unconditionally reverses ALL
//! axes regardless of what `axes` requests. This is harmless for the rank-2
//! transposes this file performs (a full reversal of a 2-element axis list
//! *is* "swap the last two"), but means this file must never call
//! `.transpose(Some(vec![...]))` with anything other than a full-reversal
//! permutation. In particular, the classic "align batch-first vs
//! sequence-first" `[1, 0, 2]` axis swap on a full 3-D tensor is NOT a full
//! reversal for rank 3, so it is never performed via `.transpose()` here at
//! all: `forward()` instead threads `batch_first` through
//! [`slice_batch_head`]'s slicing math directly and reassembles the final
//! output via [`crate::implicit_autograd::VariadicOpKind::Stack`] at whichever
//! axis matches the caller's requested layout, rather than ever transposing a
//! 3-D tensor.

use super::layers::PyParameter;
use crate::implicit_autograd::{mark_leaf_param, VariadicOpKind};
use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
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
/// [`mark_leaf_param`]'s own doc), so calling this on every `forward()` call
/// is safe and cheap after the first call within a given forward/backward
/// cycle.
fn snapshot_and_mark(py: Python<'_>, param: &Py<PyParameter>) -> PyResult<PyTensor> {
    let snapshot = param.borrow(py).to_tensor()?;
    let id = param.borrow(py).id();
    mark_leaf_param(&snapshot, id);
    Ok(snapshot)
}

/// Build a non-leaf, non-differentiable `PyTensor` wrapping `tensor` — the
/// idiom used throughout this file for values (masks, scale constants) that
/// must be present on the tape as a constant operand but must never be
/// differentiated with respect to.
fn constant(tensor: Tensor<f32>) -> PyTensor {
    PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    }
}

/// Build an `n x n` identity matrix as a non-leaf, non-differentiable
/// `PyTensor` — the fixed operand [`normalize_layout_on_tape`] multiplies by
/// to re-materialise a tensor into standard memory layout without changing
/// its logical values or its gradient.
fn identity_matrix(n: usize) -> Tensor<f32> {
    let mut data = vec![0.0f32; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    // `Tensor::from_vec` with a matching data/shape length cannot fail here;
    // `n * n` elements for an `[n, n]` shape always matches by construction.
    // An honest fallback (identity's own definition, expressed once more via
    // `from_vec` with a trivial 1x1 shape) is used only in the astronomically
    // unlikely case this ever does fail, to avoid a production `.expect()`.
    Tensor::from_vec(data, &[n, n]).unwrap_or_else(|_| Tensor::from_scalar(1.0))
}

/// Re-materialise `tensor` into standard (row-major/C-contiguous) memory
/// layout by multiplying it by an identity matrix on the right — a real,
/// tape-recorded [`PyTensor::matmul`] call, not a raw data round-trip.
///
/// # Why this is necessary
///
/// `tenflowers_core::ops::concat` (via `scirs2_core::ndarray::concatenate`)
/// on a non-leading axis can produce a tensor whose underlying array is NOT
/// standard-layout (confirmed empirically via a standalone diagnostic test
/// during development: eager `concat(axis=1)` succeeds, but the immediately
/// following `stack` on `concat`'s own output then fails). Feeding such a
/// tensor into `tenflowers_core::ops::stack` (or `reshape`) fails: both call
/// `ndarray`'s `into_shape_with_order(..)`, which requires
/// `is_standard_layout()` for its default `Order::RowMajor` and returns
/// `ShapeError::IncompatibleLayout` otherwise (confirmed by reading
/// `ndarray::impl_methods::into_shape_with_order_impl` directly) — exactly
/// the `"Expand dims failed"`/`"Reshape failed": ShapeError/IncompatibleLayout`
/// error this function exists to prevent.
///
/// Critically, this affects not just this file's own EAGER `PyTensor`
/// values, but *independently* the implicit-autograd tape's own internal
/// recomputation: [`crate::implicit_autograd::record_and_link_variadic`]'s
/// `VariadicOpKind::Concat` case calls `TrackedTensor::concat`, which
/// internally calls the SAME `tenflowers_core::ops::concat` fresh — so the
/// TAPE's own tracked value for a `Concat{axis:1}` result carries the exact
/// same non-standard layout as the eager one, entirely independently of
/// whatever this file's own eager `PyTensor` happens to look like. A plain
/// `to_vec()`/`from_vec()` round-trip on only the eager value (which was
/// tried first during development) therefore does **not** fix this: the
/// very next `record_and_link_variadic(VariadicOpKind::Stack, ...)` call
/// still fails, because it operates on the tape's own independently
/// non-standard-layout tracked value, never on the eager one this file
/// normalised.
///
/// This is why the fix must itself be a **real, tape-recorded operation**
/// (not a raw round-trip) applied to the `PyTensor` handle that subsequent
/// tape ops will look up: [`PyTensor::matmul`] against a fixed identity
/// matrix was confirmed (via the same diagnostic test) to produce
/// standard-layout output on BOTH the eager path (`matmul`'s CPU kernel
/// always allocates a fresh, standard-layout result array, regardless of
/// its operands' layout) AND the tape's own independent
/// `TrackedTensor::matmul` recomputation — and, being mathematically the
/// identity transform, it changes neither `tensor`'s forward values nor its
/// backward gradient (`d(loss)/d(tensor)` is unaffected by post-multiplying
/// by a constant identity matrix).
fn normalize_layout_on_tape(tensor: &PyTensor) -> PyResult<PyTensor> {
    let last_dim = *tensor
        .tensor
        .shape()
        .dims()
        .last()
        .ok_or_else(|| PyRuntimeError::new_err("normalize_layout_on_tape: 0-D tensor"))?;
    let identity = constant(identity_matrix(last_dim));
    tensor.matmul(&identity)
}

/// Apply a tape-aware linear projection `input @ weight^T (+ bias)`, where
/// `input` is `[dim0, dim1, in_features]` and `weight` is
/// `[out_features, in_features]` (the PyTorch convention this codebase's
/// `randn_scaled`-initialised projection weights already use).
///
/// Every [`PyTensor::matmul`] call in this function operates on genuinely
/// 2-D operands (see the module-level "per-(batch, head)" doc for why this
/// restriction exists): `input` is reshaped down to `[dim0*dim1,
/// in_features]` (a safe, tape-aware merge-adjacent-dims reshape — always
/// layout-safe for a row-major tensor) before the matmul, and the
/// `[dim0*dim1, out_features]` result is reshaped back up to `[dim0, dim1,
/// out_features]` afterward.
fn linear_projection_tape(
    input: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
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
    let mut projected_2d = input_2d.matmul(&weight_t)?;
    if let Some(b) = bias {
        projected_2d = projected_2d.add(b)?;
    }

    crate::tensor_ops::reshape(&projected_2d, vec![dim0, dim1, out_features])
}

/// Multi-Head Attention Layer
///
/// Allows the model to jointly attend to information from different representation
/// subspaces at different positions. Used extensively in transformer architectures.
#[pyclass(name = "MultiheadAttention")]
#[derive(Debug)]
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
    /// Query projection weight `[embed_dim, embed_dim]`.
    q_proj_param: Py<PyParameter>,
    /// Key projection weight `[embed_dim, kdim]`.
    k_proj_param: Py<PyParameter>,
    /// Value projection weight `[embed_dim, vdim]`.
    v_proj_param: Py<PyParameter>,
    /// Output projection weight `[embed_dim, embed_dim]`.
    out_proj_param: Py<PyParameter>,
    /// Shared bias `[embed_dim]`, reused across all four projections above
    /// (see the module-level "One shared bias" doc). `None` when
    /// `bias=false`.
    bias_param: Option<Py<PyParameter>>,
}

impl Clone for PyMultiheadAttention {
    /// Produce an **independent** layer: a fresh copy of every current
    /// parameter value, under **fresh** parameter identities distinct from
    /// `self`'s. See [`super::layers::PyDense`]'s `impl Clone` for the full
    /// rationale — `parameters()` and `Clone` deliberately want opposite
    /// things, so [`PyParameter::clone_param`] (fresh id) is correct here
    /// even though it would be wrong for `parameters()`.
    fn clone(&self) -> Self {
        Python::attach(|py| {
            let q_proj_param = self.q_proj_param.borrow(py).clone_param();
            let k_proj_param = self.k_proj_param.borrow(py).clone_param();
            let v_proj_param = self.v_proj_param.borrow(py).clone_param();
            let out_proj_param = self.out_proj_param.borrow(py).clone_param();
            let bias_param = self.bias_param.as_ref().map(|b| b.borrow(py).clone_param());
            Self {
                embed_dim: self.embed_dim,
                num_heads: self.num_heads,
                head_dim: self.head_dim,
                dropout: self.dropout,
                bias: self.bias,
                add_bias_kv: self.add_bias_kv,
                add_zero_attn: self.add_zero_attn,
                kdim: self.kdim,
                vdim: self.vdim,
                batch_first: self.batch_first,
                q_proj_param: Py::new(py, q_proj_param).expect(
                    "PyParameter::clone_param()'s result must construct as a Py<PyParameter>",
                ),
                k_proj_param: Py::new(py, k_proj_param).expect(
                    "PyParameter::clone_param()'s result must construct as a Py<PyParameter>",
                ),
                v_proj_param: Py::new(py, v_proj_param).expect(
                    "PyParameter::clone_param()'s result must construct as a Py<PyParameter>",
                ),
                out_proj_param: Py::new(py, out_proj_param).expect(
                    "PyParameter::clone_param()'s result must construct as a Py<PyParameter>",
                ),
                bias_param: bias_param.map(|b| {
                    Py::new(py, b).expect(
                        "PyParameter::clone_param()'s result must construct as a Py<PyParameter>",
                    )
                }),
            }
        })
    }
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
        py: Python<'_>,
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

        let q_proj_param = new_param(py, q_proj_weight)?;
        let k_proj_param = new_param(py, k_proj_weight)?;
        let v_proj_param = new_param(py, v_proj_weight)?;
        let out_proj_param = new_param(py, out_proj_weight)?;
        let bias_param = if bias {
            Some(new_param(py, Tensor::zeros(&[embed_dim]))?)
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
            q_proj_param,
            k_proj_param,
            v_proj_param,
            out_proj_param,
            bias_param,
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
    ///
    /// # Tape-aware op sequence
    ///
    /// For each of Q/K/V: `reshape -> matmul -> add(bias) -> reshape`. Then,
    /// for every `(batch, head)` pair: `slice -> reshape -> matmul ->
    /// mul(scale) -> [add(mask)] -> softmax -> matmul`. Head outputs are
    /// combined per-batch via `concat` (axis=-1 over head_dim), and per-batch
    /// rows are combined via `stack` (at whichever axis matches
    /// `batch_first`). Finally: `reshape -> matmul -> add(bias) -> reshape`
    /// for the output projection.
    #[pyo3(signature = (query, key, value, key_padding_mask=None, need_weights=None, attn_mask=None, average_attn_weights=None))]
    pub fn forward(
        &self,
        py: Python<'_>,
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

        let query_shape = query.tensor.shape().dims().to_vec();
        let key_shape = key.tensor.shape().dims().to_vec();
        let value_shape = value.tensor.shape().dims().to_vec();

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

        // Snapshot and (re-)register every parameter as a tape leaf for this
        // forward pass. Idempotent across repeated calls (see
        // `snapshot_and_mark`'s doc).
        let wq = snapshot_and_mark(py, &self.q_proj_param)?;
        let wk = snapshot_and_mark(py, &self.k_proj_param)?;
        let wv = snapshot_and_mark(py, &self.v_proj_param)?;
        let wo = snapshot_and_mark(py, &self.out_proj_param)?;
        let bias_snapshot = self
            .bias_param
            .as_ref()
            .map(|p| snapshot_and_mark(py, p))
            .transpose()?;

        // `batch_first` determines which of the two leading axes is `batch`
        // vs `seq` — this is threaded through the slicing math below instead
        // of ever transposing the full 3D tensor (see the module-level doc
        // for why a rank-3, non-full-reversal transpose is unsafe on this
        // tape).
        let (batch, tgt_len) = if self.batch_first {
            (query_shape[0], query_shape[1])
        } else {
            (query_shape[1], query_shape[0])
        };
        let src_len = if self.batch_first {
            key_shape[1]
        } else {
            key_shape[0]
        };

        // Linear projections into the model dimension. `linear_projection_tape`
        // preserves whatever `[dim0, dim1, features]` layout `query`/`key`/
        // `value` already have (batch-first or not) — the projection itself
        // is applied uniformly along the last axis regardless of what the
        // first two axes represent.
        let q_proj = linear_projection_tape(query, &wq, bias_snapshot.as_ref())?;
        let k_proj = linear_projection_tape(key, &wk, bias_snapshot.as_ref())?;
        let v_proj = linear_projection_tape(value, &wv, bias_snapshot.as_ref())?;

        // Combine the optional additive attention mask and the key-padding
        // mask into a single [batch, tgt_len, src_len] additive bias. This
        // folds `key_padding_mask` into the real masking math instead of
        // silently ignoring it, and reproduces the exact unmasked path when
        // both are None. `combine_attention_masks` is a plain,
        // non-differentiable `Tensor<f32>` helper (masks are never
        // differentiated with respect to), so it is fine to call it directly
        // on the raw mask tensors here.
        let combined_mask = tenflowers_neural::layers::attention::combine_attention_masks(
            attn_mask.map(|m| m.tensor.as_ref()),
            key_padding_mask.map(|m| m.tensor.as_ref()),
            batch,
            tgt_len,
            src_len,
        )
        .map_err(to_err)?;

        let scale_const = constant(Tensor::from_scalar(1.0_f32 / (self.head_dim as f32).sqrt()));

        // Per-(batch, head) scaled dot-product attention. Every matmul here
        // operates on genuinely 2-D `[seq, head_dim]`/`[head_dim, src_len]`
        // operands (see the module-level doc for why this loop cannot be
        // collapsed into a single batched matmul on this tape).
        let mut batch_rows: Vec<PyTensor> = Vec::with_capacity(batch);
        let mut weight_rows: Vec<Tensor<f32>> = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut head_outputs: Vec<PyTensor> = Vec::with_capacity(self.num_heads);
            let mut head_weight_accum: Option<Tensor<f32>> = None;

            for h in 0..self.num_heads {
                let start = (h * self.head_dim) as isize;
                let end = ((h + 1) * self.head_dim) as isize;

                let q_bh = slice_batch_head(&q_proj, b, tgt_len, start, end, self.batch_first)?;
                let k_bh = slice_batch_head(&k_proj, b, src_len, start, end, self.batch_first)?;
                let v_bh = slice_batch_head(&v_proj, b, src_len, start, end, self.batch_first)?;

                // Re-materialise `k_bh` via `normalize_layout_on_tape` BEFORE
                // transposing it. This is a DIFFERENT bug from the one that
                // function's doc otherwise describes (that one is about a
                // `Concat`/`Stack` result's own forward layout), confirmed
                // via a standalone diagnostic test: `process_transpose_backward`
                // (`tenflowers-autograd`) computes its output via a genuine
                // `permuted_axes` VIEW (non-standard-layout, like every other
                // transpose in this codebase), and when that flows directly
                // into `process_reshape_backward` (reached here because
                // `k_bh` itself came from `slice_batch_head`'s internal
                // `Reshape`), `process_reshape_backward` calls
                // `tenflowers_core::ops::reshape(grad_output, ..)` with
                // *no* layout-safety step, hitting the identical
                // `ShapeError::IncompatibleLayout`. Inserting a `MatMul`
                // node (via `normalize_layout_on_tape`) between the
                // `Reshape` and the `Transpose` routes `Transpose`'s
                // backward output into `MatMul`'s backward instead — which
                // (per `process_matmul_backward`) always produces a fresh,
                // standard-layout gradient regardless of its input's
                // layout, exactly like the forward direction. Only `k_bh` is
                // ever transposed in this loop (`q_bh`/`v_bh` are not), so
                // only `k_bh` needs this.
                let k_bh = normalize_layout_on_tape(&k_bh)?;

                // scores_bh = Q_bh @ K_bh^T / sqrt(head_dim): [tgt_len, head_dim] @ [head_dim, src_len].
                let k_bh_t = k_bh.transpose(Some(vec![1, 0]))?;
                let raw_scores = q_bh.matmul(&k_bh_t)?;
                let scaled_scores = raw_scores.mul(&scale_const)?;

                let masked_scores = if let Some(ref mask) = combined_mask {
                    let mask_row = mask
                        .slice(&[b..b + 1, 0..tgt_len, 0..src_len])
                        .map_err(to_err)?;
                    let mask_row_2d = tenflowers_core::ops::reshape(&mask_row, &[tgt_len, src_len])
                        .map_err(to_err)?;
                    scaled_scores.add(&constant(mask_row_2d))?
                } else {
                    scaled_scores
                };

                let attn_bh = super::functions::softmax(&masked_scores, Some(-1))?;

                // out_bh = attn_bh @ V_bh: [tgt_len, src_len] @ [src_len, head_dim].
                let out_bh = attn_bh.matmul(&v_bh)?;
                head_outputs.push(out_bh);

                if need_weights {
                    let w = attn_bh.tensor.to_vec().map_err(to_err)?;
                    let w_tensor = Tensor::from_vec(w, &[tgt_len, src_len]).map_err(to_err)?;
                    head_weight_accum = Some(match head_weight_accum {
                        Some(acc) => acc.add(&w_tensor).map_err(to_err)?,
                        None => w_tensor,
                    });
                }
            }

            // Concatenate this batch row's head outputs along head_dim:
            // num_heads x [tgt_len, head_dim] -> [tgt_len, embed_dim].
            let head_refs: Vec<&Tensor<f32>> =
                head_outputs.iter().map(|t| t.tensor.as_ref()).collect();
            let combined_raw = tenflowers_core::ops::concat(&head_refs, 1).map_err(to_err)?;
            let combined_row = PyTensor {
                tensor: Arc::new(combined_raw),
                requires_grad: head_outputs.iter().any(|t| t.requires_grad),
                is_pinned: false,
            };
            let head_refs_for_tape: Vec<&PyTensor> = head_outputs.iter().collect();
            crate::implicit_autograd::record_and_link_variadic(
                VariadicOpKind::Concat { axis: 1 },
                &head_refs_for_tape,
                &combined_row,
            )?;
            // Re-materialise into standard memory layout (see
            // `normalize_layout_on_tape`'s doc): this row is collected into
            // `batch_rows` below and will be fed into `Stack`, which
            // requires a standard-layout input on BOTH the eager value and
            // the tape's own independently-recomputed tracked value —
            // `Concat{axis:1}`'s result satisfies neither without this step.
            let combined_row = normalize_layout_on_tape(&combined_row)?;
            batch_rows.push(combined_row);

            if need_weights {
                let denom = self.num_heads as f32;
                // `head_weight_accum` is `None` only when `self.num_heads == 0`,
                // which `new()` already rejects (`num_heads must be positive`);
                // an honest error here is still preferred to an `.expect()`
                // panic should that invariant ever be violated by a future
                // change to `new()`.
                let accum = head_weight_accum.ok_or_else(|| {
                    PyRuntimeError::new_err(
                        "MultiheadAttention: no attention heads (num_heads == 0)",
                    )
                })?;
                let averaged = accum.multiply_scalar(1.0 / denom).map_err(to_err)?;
                weight_rows.push(averaged);
            }
        }

        // Stack the per-batch [tgt_len, embed_dim] rows into
        // [batch, tgt_len, embed_dim] via `Stack{axis:0}` — proven, via a
        // standalone diagnostic test during development, to be the ONLY
        // axis choice whose result is safe for a subsequent reshape on both
        // the eager AND tape-tracked paths (`Stack{axis:1}`'s own CPU
        // implementation is `expand_dims`+`concat` on axis 1, hitting the
        // exact same non-standard-layout issue `Concat{axis:1}` does — see
        // `normalize_layout_on_tape`'s doc — except its OUTPUT cannot be
        // fixed by a subsequent `matmul`-with-identity the way a genuinely
        // 2-D `combined_row` can, since `PyTensor::matmul` is restricted to
        // rank <= 2 operands). This intermediate is always batch-first
        // regardless of `self.batch_first`.
        let stacked_raw = {
            let refs: Vec<&Tensor<f32>> = batch_rows.iter().map(|t| t.tensor.as_ref()).collect();
            tenflowers_core::ops::stack(&refs, 0).map_err(to_err)?
        };
        let batch_first_stacked = PyTensor {
            tensor: Arc::new(stacked_raw),
            requires_grad: batch_rows.iter().any(|t| t.requires_grad),
            is_pinned: false,
        };
        let batch_row_refs: Vec<&PyTensor> = batch_rows.iter().collect();
        crate::implicit_autograd::record_and_link_variadic(
            VariadicOpKind::Stack { axis: 0 },
            &batch_row_refs,
            &batch_first_stacked,
        )?;

        // If the caller wants sequence-first output, realign
        // [batch, tgt_len, embed_dim] -> [tgt_len, batch, embed_dim] via
        // `realign_batch_first_to_seq_first` (slice + reshape + `Concat{axis:0}`
        // only — never `Stack{axis:1}`/`transpose` on this 3-D tensor).
        let stacked = if self.batch_first {
            batch_first_stacked
        } else {
            realign_batch_first_to_seq_first(&batch_first_stacked, batch, tgt_len, self.embed_dim)?
        };

        // Output projection.
        let attn_output = linear_projection_tape(&stacked, &wo, bias_snapshot.as_ref())?;

        // Average the real attention weights across heads
        // (PyTorch's `average_attn_weights=True`). The returned weights are
        // NOT tape-tracked/differentiable — PyTorch's own
        // `nn.MultiheadAttention` does not backprop through the returned
        // weights tuple element either, only through `attn_output` — matching
        // pre-existing behaviour here. Because this value is never fed into
        // any subsequent reshape/tape op (it is wrapped and returned
        // as-is), `tenflowers_core::ops::stack` is safe here at EITHER axis
        // choice: the non-standard-layout hazard this file otherwise designs
        // around only bites a *subsequent* reshape/`Stack`/`Concat`
        // consumer, and `weight_rows`'s own elements are already
        // standard-layout (built via `Tensor::from_vec` + `.add()`, both
        // confirmed to always produce standard-layout output).
        let weights_stack_axis = if self.batch_first { 0 } else { 1 };
        let attn_weights = if need_weights {
            let refs: Vec<&Tensor<f32>> = weight_rows.iter().collect();
            let stacked_weights =
                tenflowers_core::ops::stack(&refs, weights_stack_axis).map_err(to_err)?;
            Some(PyTensor {
                tensor: Arc::new(stacked_weights),
                requires_grad: false,
                is_pinned: false,
            })
        } else {
            None
        };

        Ok((attn_output, attn_weights))
    }

    /// Reset layer parameters
    pub fn reset_parameters(&mut self, py: Python<'_>) -> PyResult<()> {
        let kdim_actual = self.kdim.unwrap_or(self.embed_dim);
        let vdim_actual = self.vdim.unwrap_or(self.embed_dim);

        let scale = 1.0_f32 / (self.embed_dim as f32).sqrt();
        let init_err = |e: tenflowers_core::TensorError| {
            PyRuntimeError::new_err(format!("init failed: {}", e))
        };
        self.q_proj_param = new_param(
            py,
            randn_scaled(&[self.embed_dim, self.embed_dim], scale).map_err(init_err)?,
        )?;
        self.k_proj_param = new_param(
            py,
            randn_scaled(&[self.embed_dim, kdim_actual], scale).map_err(init_err)?,
        )?;
        self.v_proj_param = new_param(
            py,
            randn_scaled(&[self.embed_dim, vdim_actual], scale).map_err(init_err)?,
        )?;
        self.out_proj_param = new_param(
            py,
            randn_scaled(&[self.embed_dim, self.embed_dim], scale).map_err(init_err)?,
        )?;

        if self.bias {
            self.bias_param = Some(new_param(py, Tensor::zeros(&[self.embed_dim]))?);
        }

        Ok(())
    }

    /// Get layer parameters: `[q_proj, k_proj, v_proj, out_proj]`, plus
    /// `bias` if `bias=true`. Shares identity with the exact same
    /// `PyParameter` objects `forward()` marks as leaves (see the
    /// module-level "Autograd" doc), so `.grad()` on the returned handles is
    /// populated after a `.backward()` call that passes through this layer's
    /// `forward()`.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyParameter>> {
        let mut params = vec![
            self.q_proj_param.clone_ref(py),
            self.k_proj_param.clone_ref(py),
            self.v_proj_param.clone_ref(py),
            self.out_proj_param.clone_ref(py),
        ];
        if let Some(ref bias_param) = self.bias_param {
            params.push(bias_param.clone_ref(py));
        }
        params
    }

    /// Get layer state dictionary
    pub fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);

        dict.set_item("q_proj_weight", self.q_proj_param.borrow(py).to_tensor()?)?;
        dict.set_item("k_proj_weight", self.k_proj_param.borrow(py).to_tensor()?)?;
        dict.set_item("v_proj_weight", self.v_proj_param.borrow(py).to_tensor()?)?;
        dict.set_item(
            "out_proj_weight",
            self.out_proj_param.borrow(py).to_tensor()?,
        )?;
        if let Some(ref bias_param) = self.bias_param {
            dict.set_item("bias", bias_param.borrow(py).to_tensor()?)?;
        }

        Ok(dict.unbind())
    }

    /// Load layer state from dictionary
    pub fn load_state_dict(
        &mut self,
        py: Python<'_>,
        state_dict: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        if let Ok(Some(weight)) = state_dict.get_item("q_proj_weight") {
            if let Ok(weight_tensor) = weight.extract::<PyTensor>() {
                self.q_proj_param
                    .borrow(py)
                    .set_data((*weight_tensor.tensor).clone())?;
            }
        }

        if let Ok(Some(weight)) = state_dict.get_item("k_proj_weight") {
            if let Ok(weight_tensor) = weight.extract::<PyTensor>() {
                self.k_proj_param
                    .borrow(py)
                    .set_data((*weight_tensor.tensor).clone())?;
            }
        }

        if let Ok(Some(weight)) = state_dict.get_item("v_proj_weight") {
            if let Ok(weight_tensor) = weight.extract::<PyTensor>() {
                self.v_proj_param
                    .borrow(py)
                    .set_data((*weight_tensor.tensor).clone())?;
            }
        }

        if let Ok(Some(weight)) = state_dict.get_item("out_proj_weight") {
            if let Ok(weight_tensor) = weight.extract::<PyTensor>() {
                self.out_proj_param
                    .borrow(py)
                    .set_data((*weight_tensor.tensor).clone())?;
            }
        }

        if let Ok(Some(weight)) = state_dict.get_item("bias") {
            if let (Ok(weight_tensor), Some(ref bias_param)) =
                (weight.extract::<PyTensor>(), &self.bias_param)
            {
                bias_param
                    .borrow(py)
                    .set_data((*weight_tensor.tensor).clone())?;
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

/// Slice out a single `(batch, head)` pair's genuinely 2-D `[seq_len,
/// head_dim]` matrix from a `[dim0, dim1, embed_dim]` projection tensor,
/// where `dim0`/`dim1` is `[batch, seq]` when `batch_first` or `[seq,
/// batch]` otherwise.
///
/// This is the slicing half of the module-level "per-(batch, head)" design:
/// it never transposes a 3-D tensor (see that doc for why that would be
/// unsafe on this tape) — instead it slices directly along whichever axis
/// order the input already has, then reshapes the resulting `[1, seq_len,
/// head_dim]` (batch-first) or `[seq_len, 1, head_dim]` (seq-first) slice
/// down to a real 2-D `[seq_len, head_dim]` matrix (a safe, tape-aware
/// drop-a-size-1-axis reshape).
fn slice_batch_head(
    projection: &PyTensor,
    batch_idx: usize,
    seq_len: usize,
    head_start: isize,
    head_end: isize,
    batch_first: bool,
) -> PyResult<PyTensor> {
    let b = batch_idx as isize;
    let ranges: Vec<(Option<isize>, Option<isize>, Option<isize>)> = if batch_first {
        vec![
            (Some(b), Some(b + 1), None),
            (None, None, None),
            (Some(head_start), Some(head_end), None),
        ]
    } else {
        vec![
            (None, None, None),
            (Some(b), Some(b + 1), None),
            (Some(head_start), Some(head_end), None),
        ]
    };
    let sliced = projection.slice(ranges)?;
    let head_dim = (head_end - head_start) as usize;
    crate::tensor_ops::reshape(&sliced, vec![seq_len, head_dim])
}

/// Realign a `[batch, tgt_len, embed_dim]` tensor to `[tgt_len, batch,
/// embed_dim]`, entirely through `slice` + `reshape` + `Concat{axis:0}` —
/// never `Stack{axis:1}` or a rank-3 `.transpose()`.
///
/// # Why not `Stack{axis:1}` or `.transpose(Some(vec![1,0,2]))`
///
/// Both were tried during development and confirmed unsafe on this tape via
/// standalone diagnostic tests:
///
/// * `.transpose(Some(vec![1,0,2]))` is not a full-axis-reversal permutation
///   for rank 3, and `TrackedTensor::transpose`'s backward is confirmed (by
///   direct source read) to ignore its `axes` parameter entirely, always
///   reversing ALL axes regardless of what is requested — silently
///   computing the wrong VALUE, not just the wrong gradient (see the
///   module-level doc's "per-(batch, head)" section).
/// * `Stack{axis:1}` was confirmed to internally perform `expand_dims`
///   (axis 1) then `concat` (axis 1) — the same non-leading-axis `concat`
///   that makes `Concat{axis:1}` (used for per-head combination above)
///   produce a non-standard-layout result — except `Stack{axis:1}`'s
///   3-D *output* cannot be repaired by [`normalize_layout_on_tape`]'s
///   `matmul`-with-identity trick, since [`PyTensor::matmul`] is restricted
///   to rank <= 2 operands (the same restriction driving this whole file's
///   per-`(batch, head)` design).
///
/// # The technique
///
/// For each target position `t` in `0..tgt_len`: slice out `input`'s `t`-th
/// row — `[batch, 1, embed_dim]` (a safe, always-standard-layout `slice`) —
/// then reshape it to `[1, batch, embed_dim]`. This reshape is safe (not
/// merely "safe by the general merge-adjacent-dims rule" but literally a
/// no-op on the underlying flat buffer): a size-1 axis contributes nothing
/// to a row-major flat layout regardless of *where* among the dimensions it
/// sits, so `[batch, 1, embed_dim]` and `[1, batch, embed_dim]` denote the
/// exact same `batch * embed_dim`-element flat array, just with a different
/// shape label — confirmed via a standalone diagnostic test asserting
/// bit-exact value equality between this reassembly and the original
/// `[batch, tgt_len, embed_dim]` tensor's own `[b, t, e]` indexing. Finally,
/// `Concat{axis:0}` (proven safe — see [`normalize_layout_on_tape`]'s doc)
/// combines all `tgt_len` many `[1, batch, embed_dim]` rows into
/// `[tgt_len, batch, embed_dim]`.
fn realign_batch_first_to_seq_first(
    input: &PyTensor,
    batch: usize,
    tgt_len: usize,
    embed_dim: usize,
) -> PyResult<PyTensor> {
    let mut seq_rows: Vec<PyTensor> = Vec::with_capacity(tgt_len);
    for t in 0..tgt_len {
        let t_isize = t as isize;
        let sliced = input.slice(vec![
            (None, None, None),
            (Some(t_isize), Some(t_isize + 1), None),
        ])?;
        let row = crate::tensor_ops::reshape(&sliced, vec![1, batch, embed_dim])?;
        seq_rows.push(row);
    }

    let refs: Vec<&Tensor<f32>> = seq_rows.iter().map(|t| t.tensor.as_ref()).collect();
    let concat_raw = tenflowers_core::ops::concat(&refs, 0).map_err(|e| {
        PyRuntimeError::new_err(format!("realign_batch_first_to_seq_first failed: {e}"))
    })?;
    let concat_py = PyTensor {
        tensor: Arc::new(concat_raw),
        requires_grad: seq_rows.iter().any(|t| t.requires_grad),
        is_pinned: false,
    };
    let seq_row_refs: Vec<&PyTensor> = seq_rows.iter().collect();
    crate::implicit_autograd::record_and_link_variadic(
        VariadicOpKind::Concat { axis: 0 },
        &seq_row_refs,
        &concat_py,
    )?;

    Ok(concat_py)
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
    // `dropout_p` is validated but never applied to the computation below,
    // exactly matching this function's pre-migration behaviour: the old
    // implementation called `tenflowers_neural::layers::attention::
    // scaled_dot_product_attention(..., dropout_p, /* training */ false)`
    // with `training` hardcoded to `false`, so that function's own internal
    // `if training && dropout_prob > 0.0 { .. }` dropout branch never fired
    // regardless of `dropout_p`'s value. This is a pre-existing behaviour,
    // not something introduced by this migration.

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

    // This free function operates on 3-D tensors `[batch, seq, d_k]`; require
    // that here and return an honest error otherwise.
    if query_shape.len() != 3 || key_shape.len() != 3 || value_shape.len() != 3 {
        return Err(PyValueError::new_err(
            "scaled_dot_product_attention requires 3D tensors [batch, seq, d_k]",
        ));
    }

    let batch = query_shape[0];
    let tgt_len = query_shape[1];
    let src_len = key_shape[1];
    let d_k = query_shape[2];

    // Validate attention mask shape. The pre-migration implementation
    // delegated masking to `apply_attention_mask`, a plain
    // `tenflowers_core::ops::add(scores, mask)` with no rank restriction of
    // its own — any shape broadcastable against `[batch, tgt_len, src_len]`
    // was accepted. This per-(batch) loop implementation supports the two
    // concrete shapes that matter in practice and are unambiguous to slice
    // per batch row: `[tgt_len, src_len]` (broadcast identically across every
    // batch) and `[batch, tgt_len, src_len]` (one row per batch, matching
    // `combine_attention_masks`'s own output shape). Any other rank
    // (including "at least 2D but not one of these two", e.g. a `[1,
    // tgt_len, src_len]` NumPy-broadcastable-but-not-identical shape) is
    // rejected with an honest error instead of being silently mis-sliced.
    if let Some(mask) = attn_mask {
        let mask_shape = mask.tensor.shape().dims().to_vec();
        let is_2d_broadcast = mask_shape == [tgt_len, src_len];
        let is_3d_per_batch = mask_shape == [batch, tgt_len, src_len];
        if !is_2d_broadcast && !is_3d_per_batch {
            return Err(PyValueError::new_err(format!(
                "attn_mask must have shape [{tgt_len}, {src_len}] (broadcast across the \
                 batch) or [{batch}, {tgt_len}, {src_len}] (one row per batch), got {mask_shape:?}"
            )));
        }
    }

    let to_err = |e: tenflowers_core::TensorError| {
        PyRuntimeError::new_err(format!("scaled_dot_product_attention failed: {}", e))
    };

    let scale_const = constant(Tensor::from_scalar(1.0_f32 / (d_k as f32).sqrt()));

    let mut batch_rows: Vec<PyTensor> = Vec::with_capacity(batch);
    for b in 0..batch {
        let q_b = slice_batch_head(query, b, tgt_len, 0, d_k as isize, true)?;
        let k_b = slice_batch_head(key, b, src_len, 0, d_k as isize, true)?;
        let v_b = slice_batch_head(value, b, src_len, 0, d_k as isize, true)?;

        // See `PyMultiheadAttention::forward`'s identical comment on its own
        // `k_bh` normalisation for the full explanation: `Transpose`'s
        // backward output (a non-standard-layout `permuted_axes` view) must
        // not flow directly into `slice_batch_head`'s internal `Reshape`
        // node's own backward, which has no layout-safety step.
        let k_b = normalize_layout_on_tape(&k_b)?;
        let k_b_t = k_b.transpose(Some(vec![1, 0]))?;
        let raw_scores = q_b.matmul(&k_b_t)?;
        let scaled_scores = raw_scores.mul(&scale_const)?;

        let masked_scores = if let Some(mask) = attn_mask {
            let mask_shape = mask.tensor.shape().dims().to_vec();
            let mask_2d = if mask_shape.len() == 2 {
                // [tgt_len, src_len]: already the exact 2D shape needed,
                // broadcasting identically across every batch row.
                mask.clone()
            } else {
                // [batch, tgt_len, src_len]: slice out this batch's own row.
                let mask_row = mask.slice(vec![(Some(b as isize), Some(b as isize + 1), None)])?;
                crate::tensor_ops::reshape(&mask_row, vec![tgt_len, src_len])?
            };
            scaled_scores.add(&mask_2d)?
        } else {
            scaled_scores
        };

        let attn_b = super::functions::softmax(&masked_scores, Some(-1))?;
        let out_b = attn_b.matmul(&v_b)?;
        batch_rows.push(out_b);
    }

    let stacked_raw = {
        let refs: Vec<&Tensor<f32>> = batch_rows.iter().map(|t| t.tensor.as_ref()).collect();
        tenflowers_core::ops::stack(&refs, 0).map_err(to_err)?
    };
    let stacked = PyTensor {
        tensor: Arc::new(stacked_raw),
        requires_grad: batch_rows.iter().any(|t| t.requires_grad),
        is_pinned: false,
    };
    let batch_row_refs: Vec<&PyTensor> = batch_rows.iter().collect();
    crate::implicit_autograd::record_and_link_variadic(
        VariadicOpKind::Stack { axis: 0 },
        &batch_row_refs,
        &stacked,
    )?;

    Ok(stacked)
}

#[cfg(test)]
mod tests;
