//! Embedding layers module for TenfloweRS FFI
//!
//! This module provides embedding layer implementations for converting discrete tokens
//! into continuous vector representations for NLP and other sequence modeling tasks.
//!
//! # Autograd participation
//!
//! [`PyEmbedding`] and [`PyEmbeddingBag`] both store their weight table as a
//! [`Py<PyParameter>`] — the same stable-identity, mutable-in-place
//! parameter cell [`super::layers::PyDense`] holds its own weight/bias in —
//! rather than a bare [`Tensor<f32>`]. This is what lets `forward()`
//! correctly participate in the implicit (PyTorch-style eager) autograd tape
//! maintained by [`crate::implicit_autograd`]:
//!
//! * `forward()` snapshots the parameter's current value via
//!   [`PyParameter::to_tensor`] and marks that snapshot as a tape leaf via
//!   [`crate::implicit_autograd::mark_leaf_param`], keyed by the parameter's
//!   own stable `id()` — exactly the pattern [`PyDense::forward`] uses for
//!   its own weight/bias.
//! * The row lookup itself is a real `gather` (`tenflowers_core::ops::gather`
//!   along axis 0), and [`crate::implicit_autograd::record_and_link_gather`]
//!   records that same operation onto the tape so `.backward()` can scatter
//!   gradients back to exactly the rows that were looked up — with
//!   scatter-**add** semantics for any row gathered more than once in the
//!   same batch (proven at the kernel level by
//!   `tenflowers_autograd::grad_ops::tensor_ops::gather_backward`, and
//!   exercised end-to-end through this file's own
//!   `weight_gradient_accumulates_for_repeated_index` test below).
//! * `.parameters()` returns `Vec<Py<PyParameter>>` — additional owning
//!   references to the exact same underlying parameter object `forward()`
//!   reads from (via [`Py::clone_ref`], not [`PyParameter::clone_param`] —
//!   see [`PyDense::parameters`]'s doc for why that distinction matters), so
//!   a gradient computed by `.backward()` is immediately visible via
//!   `.grad()` on any handle obtained from `.parameters()`, and a
//!   `.set_data()` write through one of those handles (e.g. from an
//!   optimizer) is visible to this layer's next `forward()` call.
//!
//! [`PyEmbeddingBag`]'s `sum`/`mean` reduction modes extend this further:
//! since there is no single fused, tape-aware "EmbeddingBag" primitive
//! anywhere in this codebase, the bag reduction is deliberately *decomposed*
//! into already-tape-aware primitives — one [`record_and_link_gather`] call
//! for the whole batch, followed by, per bag, a tape-aware slice
//! (`UnaryOpKind::Slice`) and a tape-aware `sum`/`mean`
//! (`crate::implicit_autograd::record_and_link_unary`, the exact same hook
//! [`crate::math_ops::sum`]/[`crate::math_ops::mean`] use), and finally a
//! tape-aware `stack` (`VariadicOpKind::Stack`) to reassemble the per-bag
//! results into one `[num_bags, embedding_dim]` output — rather than
//! inventing any new gradient math of its own. `max` mode is the one
//! exception: no tape-aware max reduction exists anywhere in this crate to
//! reuse (see [`PyEmbeddingBag::forward`]'s own doc for the full reasoning),
//! so `max` mode's forward output is still computed correctly but does not
//! extend the tape past the gather step.
//!
//! # Why these structs' `Clone` impls are hand-written
//!
//! [`PyParameter`] deliberately has no `#[derive(Clone)]` (aliasing its
//! identity via a naive derive would be a footgun — see that struct's own
//! doc), and `Py<PyParameter>` only implements `Clone` when pyo3's
//! `py-clone` feature is enabled, which this workspace does not enable. A
//! `#[derive(Clone)]` on [`PyEmbedding`]/[`PyEmbeddingBag`] is therefore not
//! available either way, and — exactly as [`PyDense`]'s own hand-written
//! `Clone` impl explains — there is no "obvious" derived meaning to fall
//! back on regardless: cloning a whole layer should produce an
//! **independent** layer under a **fresh** parameter identity (via
//! [`PyParameter::clone_param`]), which is a different operation from
//! [`Py::clone_ref`]-ing a handle to the *same* identity (what
//! `.parameters()` does). See [`PyEmbedding`]'s `impl Clone` below.
//!
//! [`PyParameter`]: super::layers::PyParameter
//! [`PyDense`]: super::layers::PyDense
//! [`PyDense::forward`]: super::layers::PyDense::forward
//! [`PyDense::parameters`]: super::layers::PyDense::parameters
//! [`record_and_link_gather`]: crate::implicit_autograd::record_and_link_gather

use crate::implicit_autograd::{
    record_and_link_gather, record_and_link_unary, record_and_link_variadic, UnaryOpKind,
    VariadicOpKind,
};
use crate::neural::layers::PyParameter;
use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tenflowers_core::Tensor;

/// Convert a `PyTensor`'s `f32` index data into the `Tensor<i32>` shape
/// [`tenflowers_core::ops::gather`] / [`record_and_link_gather`] require,
/// validating every index is in `[0, num_embeddings)` along the way.
///
/// There is no generic `f32 -> i32` tensor cast anywhere in this crate or in
/// `tenflowers-core` (only narrow `cast_to_u8`/`cast_to_bool` special cases
/// exist), so this performs the conversion element-by-element, exactly
/// mirroring the one existing precedent for building a `Tensor<i32>` in this
/// crate (`crate::implicit_autograd`'s own test suite).
fn indices_to_i32(input: &PyTensor, num_embeddings: usize) -> PyResult<Tensor<i32>> {
    let input_data = input
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get input data: {e}")))?;

    let mut indices_i32 = Vec::with_capacity(input_data.len());
    for &idx_f32 in &input_data {
        let idx = idx_f32 as i64;
        if idx < 0 || idx as usize >= num_embeddings {
            return Err(PyValueError::new_err(format!(
                "Index {idx} is out of bounds for embedding with {num_embeddings} entries"
            )));
        }
        indices_i32.push(idx as i32);
    }

    let shape: Vec<usize> = input.tensor.shape().dims().to_vec();
    Tensor::from_vec(indices_i32, &shape)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to build index tensor: {e}")))
}

/// Same conversion as [`indices_to_i32`], but always flattens the result to
/// a single 1-D `[total]` index tensor regardless of `input`'s own shape.
///
/// [`PyEmbeddingBag::forward`] needs this rather than [`indices_to_i32`]
/// directly: it performs its own manual per-bag decomposition (slice ->
/// reduce -> stack) over a flat `[total, embedding_dim]` gather result, so
/// the gather itself must be over 1-D indices — passing a 2-D `[num_bags,
/// bag_size]` `input` straight through [`indices_to_i32`] would instead make
/// `tenflowers_core::ops::gather` splice that *whole* shape into the output
/// (producing a 3-D `[num_bags, bag_size, embedding_dim]` result, per that
/// function's own "replace axis with indices.shape()" semantics), which the
/// subsequent 2-D row-range slicing in [`tape_aware_slice_rows`] cannot
/// consume. [`PyEmbedding::forward`] (which has no such downstream
/// decomposition, and instead wants exactly this shape-preserving behavior
/// so a `(*, )` input produces a `(*, embedding_dim)` output) correctly uses
/// [`indices_to_i32`] directly instead.
fn indices_to_i32_flat(input: &PyTensor, num_embeddings: usize) -> PyResult<Tensor<i32>> {
    let shaped = indices_to_i32(input, num_embeddings)?;
    let total = shaped.size();
    let flat_data = shaped
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to flatten index tensor: {e}")))?;
    Tensor::from_vec(flat_data, &[total])
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to build flat index tensor: {e}")))
}

/// Snapshot `param`'s current value and mark it as this forward pass's tape
/// leaf, exactly mirroring `PyDense::forward`'s own
/// `Python::attach(|py| ...)` + [`PyParameter::to_tensor`] +
/// [`crate::implicit_autograd::mark_leaf_param`] sequence.
///
/// [`PyParameter`]: super::layers::PyParameter
fn snapshot_and_mark_leaf(param: &Py<PyParameter>) -> PyResult<PyTensor> {
    let (id, snapshot) = Python::attach(|py| -> PyResult<(usize, PyTensor)> {
        let param_ref = param.borrow(py);
        Ok((param_ref.id(), param_ref.to_tensor()?))
    })?;
    crate::implicit_autograd::mark_leaf_param(&snapshot, id);
    Ok(snapshot)
}

/// Gather every row of `weight_snapshot` named by `indices`, computing the
/// real forward math via [`tenflowers_core::ops::gather`] (axis 0) and then
/// recording that same operation onto the implicit tape via
/// [`record_and_link_gather`] so a subsequent `.backward()` scatters
/// gradients back to `weight_snapshot`'s originating [`PyParameter`] (via
/// whatever leaf it was already marked under — this function does not mark
/// any leaf itself, matching [`record_and_link_gather`]'s own contract of
/// only linking an *already*-tracked `input`).
///
/// [`PyParameter`]: super::layers::PyParameter
fn gather_rows(weight_snapshot: &PyTensor, indices: &Tensor<i32>) -> PyResult<PyTensor> {
    let raw = tenflowers_core::ops::gather(&weight_snapshot.tensor, indices, 0)
        .map_err(|e| PyRuntimeError::new_err(format!("Embedding gather failed: {e}")))?;
    let result = PyTensor {
        tensor: Arc::new(raw),
        requires_grad: weight_snapshot.requires_grad,
        is_pinned: weight_snapshot.is_pinned,
    };
    record_and_link_gather(weight_snapshot, indices, 0, &result)?;
    Ok(result)
}

/// Tape-aware row slice: `rows[start..end, :]` out of a `[N, dim]` tensor,
/// computed via the real forward slice and recorded onto the implicit tape
/// via `UnaryOpKind::Slice` (the same hook `TrackedTensor::slice` backs)
/// whenever `rows` is currently tracked.
fn tape_aware_slice_rows(rows: &PyTensor, start: usize, end: usize) -> PyResult<PyTensor> {
    let dim = rows.tensor.shape().dims().get(1).copied().unwrap_or(1);
    let specs = vec![
        tenflowers_autograd::grad_ops::SliceSpec::range(start as isize, end as isize),
        tenflowers_autograd::grad_ops::SliceSpec::range(0, dim as isize),
    ];

    let ranges = vec![start..end, 0..dim];
    let raw = tenflowers_core::ops::slice(&rows.tensor, &ranges)
        .map_err(|e| PyRuntimeError::new_err(format!("Bag slice failed: {e}")))?;
    let result = PyTensor {
        tensor: Arc::new(raw),
        requires_grad: rows.requires_grad,
        is_pinned: rows.is_pinned,
    };
    record_and_link_unary(UnaryOpKind::Slice { specs }, rows, &result)?;
    Ok(result)
}

/// Build a `[dim]` constant tensor with every element equal to `value`, so
/// it can be used as the RHS of a plain, unambiguous element-wise
/// [`PyTensor::div`]/[`PyTensor::mul`] against a `[dim]`-shaped reduced bag
/// row without relying on any particular broadcast-shape convention. Used by
/// `mean` mode's `sum` / `bag_len` decomposition (see that call site's own
/// doc for why `mean` is expressed this way rather than via
/// [`crate::math_ops::mean`] directly).
fn constant_row(value: f32, dim: usize) -> PyResult<PyTensor> {
    let tensor = Tensor::from_vec(vec![value; dim], &[dim])
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to build constant row: {e}")))?;
    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    })
}

/// Build a `[bag_len, dim]` constant tensor whose row `i` is
/// `per_sample_weights[i]` repeated `dim` times, so `per_sample_weights` can
/// be applied to a `[bag_len, dim]` gathered slice via a plain, unambiguous
/// element-wise [`PyTensor::mul`] rather than relying on any particular
/// broadcast-shape convention.
fn build_row_scale(weights: &[f32], dim: usize) -> PyResult<PyTensor> {
    let mut data = Vec::with_capacity(weights.len() * dim);
    for &w in weights {
        for _ in 0..dim {
            data.push(w);
        }
    }
    let tensor = Tensor::from_vec(data, &[weights.len(), dim]).map_err(|e| {
        PyRuntimeError::new_err(format!("Failed to build sample-weight scale: {e}"))
    })?;
    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    })
}

/// Tape-aware reassembly of per-bag `[dim]` rows into one
/// `[num_bags, dim]` output, via `VariadicOpKind::Stack` (the same hook
/// `TrackedTensor::stack` backs).
fn stack_bags(bags: &[&PyTensor]) -> PyResult<PyTensor> {
    let arrays: Vec<Arc<Tensor<f32>>> = bags.iter().map(|b| Arc::clone(&b.tensor)).collect();
    let refs: Vec<&Tensor<f32>> = arrays.iter().map(|a| a.as_ref()).collect();
    let raw = tenflowers_core::ops::stack(&refs, 0)
        .map_err(|e| PyRuntimeError::new_err(format!("Bag stack failed: {e}")))?;
    let requires_grad = bags.iter().any(|b| b.requires_grad);
    let result = PyTensor {
        tensor: Arc::new(raw),
        requires_grad,
        is_pinned: false,
    };
    record_and_link_variadic(VariadicOpKind::Stack { axis: 0 }, bags, &result)?;
    Ok(result)
}

/// Construct a fresh, zero-initialized `Py<PyParameter>` weight table of
/// shape `[num_embeddings, embedding_dim]`.
fn new_weight_param(num_embeddings: usize, embedding_dim: usize) -> PyResult<Py<PyParameter>> {
    let weight = Tensor::zeros(&[num_embeddings, embedding_dim]);
    let weight_tensor = PyTensor {
        tensor: Arc::new(weight),
        requires_grad: true,
        is_pinned: false,
    };
    Python::attach(|py| Py::new(py, PyParameter::new(weight_tensor, Some(true))))
}

/// Embedding Layer
///
/// A lookup table that stores embeddings of a fixed dictionary size.
/// Commonly used for word embeddings in NLP tasks.
#[pyclass(name = "Embedding")]
#[derive(Debug)]
pub struct PyEmbedding {
    /// Size of the dictionary of embeddings
    pub num_embeddings: usize,
    /// The size of each embedding vector
    pub embedding_dim: usize,
    /// Padding index (if set, gradient is zero for this index)
    pub padding_idx: Option<usize>,
    /// Maximum norm for each embedding vector (if set, embeddings are renormalized)
    pub max_norm: Option<f32>,
    /// The p in the p-norm to compute for the max_norm option
    pub norm_type: f32,
    /// If True, gradients scale by inverse of frequency of words in mini-batch
    pub scale_grad_by_freq: bool,
    /// If True, learn embeddings (trainable), else use as constant
    pub sparse: bool,
    /// Embedding weight matrix: a stable-identity, mutable-in-place
    /// parameter cell (see the module-level "Autograd participation" doc).
    /// `forward()` and `.parameters()` share this exact handle, so a
    /// gradient computed by `.backward()` after a `forward()` call is
    /// immediately visible via `.parameters()[0].grad()`.
    weight_param: Py<PyParameter>,
}

impl Clone for PyEmbedding {
    /// Produce an **independent** layer: a fresh copy of the current weight
    /// value, under a **fresh** parameter identity distinct from `self`'s.
    /// See the module-level "Why these structs' `Clone` impls are
    /// hand-written" doc, and [`super::layers::PyDense`]'s own `impl Clone`
    /// for the identical rationale this mirrors.
    fn clone(&self) -> Self {
        Python::attach(|py| {
            let cloned_param = self.weight_param.borrow(py).clone_param();
            // `Py::new` can only fail on Python-interpreter-level allocation
            // failure; `PyParameter::clone_param()`'s result performs no
            // additional fallible validation of its own — mirrors
            // `PyDense::clone`'s identical `.expect(..)` for the same
            // reason.
            let weight_param = Py::new(py, cloned_param)
                .expect("PyParameter::clone_param()'s result must construct as a Py<PyParameter>");
            Self {
                num_embeddings: self.num_embeddings,
                embedding_dim: self.embedding_dim,
                padding_idx: self.padding_idx,
                max_norm: self.max_norm,
                norm_type: self.norm_type,
                scale_grad_by_freq: self.scale_grad_by_freq,
                sparse: self.sparse,
                weight_param,
            }
        })
    }
}

#[pymethods]
impl PyEmbedding {
    /// Create a new Embedding layer
    ///
    /// # Arguments
    ///
    /// * `num_embeddings` - Size of the dictionary (vocabulary size)
    /// * `embedding_dim` - Dimension of the embedding vectors
    /// * `padding_idx` - If specified, entries at padding_idx do not contribute to gradient
    /// * `max_norm` - If given, renormalize embeddings to have norm at most max_norm
    /// * `norm_type` - The p of the p-norm for max_norm option (default: 2.0)
    /// * `scale_grad_by_freq` - If True, scale gradients by frequency (default: False)
    /// * `sparse` - If True, gradient w.r.t. weight is a sparse tensor (default: False)
    #[new]
    #[pyo3(signature = (num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=None, scale_grad_by_freq=None, sparse=None))]
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
        max_norm: Option<f32>,
        norm_type: Option<f32>,
        scale_grad_by_freq: Option<bool>,
        sparse: Option<bool>,
    ) -> PyResult<Self> {
        let norm_type = norm_type.unwrap_or(2.0);
        let scale_grad_by_freq = scale_grad_by_freq.unwrap_or(false);
        let sparse = sparse.unwrap_or(false);

        if num_embeddings == 0 {
            return Err(PyValueError::new_err("num_embeddings must be positive"));
        }
        if embedding_dim == 0 {
            return Err(PyValueError::new_err("embedding_dim must be positive"));
        }
        if let Some(idx) = padding_idx {
            if idx >= num_embeddings {
                return Err(PyValueError::new_err(format!(
                    "padding_idx {} is out of range for num_embeddings {}",
                    idx, num_embeddings
                )));
            }
        }
        if let Some(max_norm_val) = max_norm {
            if max_norm_val <= 0.0 {
                return Err(PyValueError::new_err("max_norm must be positive"));
            }
        }
        if norm_type <= 0.0 {
            return Err(PyValueError::new_err("norm_type must be positive"));
        }

        let weight_param = new_weight_param(num_embeddings, embedding_dim)?;

        Ok(PyEmbedding {
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            weight_param,
        })
    }

    /// Forward pass through the embedding layer
    ///
    /// # Arguments
    ///
    /// * `input` - LongTensor of indices, shape (*, ) where * means any number of dimensions
    ///
    /// # Returns
    ///
    /// Embedded tensor of shape (*, embedding_dim)
    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let weight_snapshot = snapshot_and_mark_leaf(&self.weight_param)?;
        let indices = indices_to_i32(input, self.num_embeddings)?;
        gather_rows(&weight_snapshot, &indices)
    }

    /// Load embeddings from a 2D tensor
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Tensor of shape (num_embeddings, embedding_dim)
    pub fn from_pretrained(&mut self, embeddings: &PyTensor) -> PyResult<()> {
        let emb_shape = embeddings.tensor.shape();

        if emb_shape.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "Expected 2D tensor for embeddings, got {}D",
                emb_shape.len()
            )));
        }

        if emb_shape[0] != self.num_embeddings {
            return Err(PyValueError::new_err(format!(
                "Expected {} embeddings, got {}",
                self.num_embeddings, emb_shape[0]
            )));
        }

        if emb_shape[1] != self.embedding_dim {
            return Err(PyValueError::new_err(format!(
                "Expected embedding_dim {}, got {}",
                self.embedding_dim, emb_shape[1]
            )));
        }

        // `set_data` replaces the parameter's contents in place, preserving
        // its stable `id` — unlike rebinding `weight_param` to a brand-new
        // `Py<PyParameter>`, which would silently orphan any gradient or
        // tape leaf already registered under the old id.
        let new_value = (*embeddings.tensor).clone();
        Python::attach(|py| self.weight_param.borrow(py).set_data(new_value))
    }

    /// Reset layer parameters
    pub fn reset_parameters(&mut self) -> PyResult<()> {
        let fresh = Tensor::zeros(&[self.num_embeddings, self.embedding_dim]);

        // If `padding_idx` is set, zero out that row. Zeros are already the
        // freshly-reset value for every row, so there is nothing further to
        // do here; this mirrors the pre-existing (documented) behavior.
        Python::attach(|py| self.weight_param.borrow(py).set_data(fresh))
    }

    /// Get layer state dictionary
    pub fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        let weight_tensor = self.weight_param.borrow(py).to_tensor()?;
        dict.set_item("weight", weight_tensor)?;
        Ok(dict.unbind())
    }

    /// Load layer state from dictionary
    pub fn load_state_dict(&mut self, state_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Ok(Some(weight)) = state_dict.get_item("weight") {
            if let Ok(weight_tensor) = weight.extract::<PyTensor>() {
                let new_value = (*weight_tensor.tensor).clone();
                Python::attach(|py| self.weight_param.borrow(py).set_data(new_value))?;
            }
        }

        Ok(())
    }

    /// Get this layer's trainable parameters.
    ///
    /// Returns a handle that shares the exact same [`PyParameter`] identity
    /// `forward()` reads from (see the module-level "Autograd participation"
    /// doc), so `.grad()` on the handle returned here is populated after a
    /// `.backward()` call that passes through this layer's `forward()`.
    ///
    /// [`PyParameter`]: super::layers::PyParameter
    pub fn parameters(&self) -> Vec<Py<PyParameter>> {
        Python::attach(|py| vec![self.weight_param.clone_ref(py)])
    }

    fn __repr__(&self) -> String {
        format!(
            "Embedding(num_embeddings={}, embedding_dim={}, padding_idx={:?}, max_norm={:?}, norm_type={}, scale_grad_by_freq={}, sparse={})",
            self.num_embeddings,
            self.embedding_dim,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse
        )
    }
}

/// Embedding Bag Layer
///
/// Computes sums, means or max of bags of embeddings without instantiating intermediate embeddings.
/// Useful for representing variable-length sequences with fixed-size vectors.
#[pyclass(name = "EmbeddingBag")]
#[derive(Debug)]
pub struct PyEmbeddingBag {
    /// Size of the dictionary of embeddings
    pub num_embeddings: usize,
    /// The size of each embedding vector
    pub embedding_dim: usize,
    /// Maximum norm for each embedding vector
    pub max_norm: Option<f32>,
    /// The p in the p-norm to compute for the max_norm option
    pub norm_type: f32,
    /// If True, gradients scale by inverse of frequency of words in mini-batch
    pub scale_grad_by_freq: bool,
    /// Reduction mode: 'sum', 'mean', or 'max'
    pub mode: String,
    /// If True, learn embeddings (trainable)
    pub sparse: bool,
    /// Include the last offset in the offsets tensor
    pub include_last_offset: bool,
    /// Padding index
    pub padding_idx: Option<usize>,
    /// Embedding weight matrix: see [`PyEmbedding::weight_param`] for the
    /// full rationale — same `PyParameter`-backed, stable-identity design.
    weight_param: Py<PyParameter>,
}

impl Clone for PyEmbeddingBag {
    /// Produce an **independent** layer under a **fresh** parameter
    /// identity. See [`PyEmbedding`]'s own `impl Clone` for the full
    /// rationale — identical design.
    fn clone(&self) -> Self {
        Python::attach(|py| {
            let cloned_param = self.weight_param.borrow(py).clone_param();
            // See `PyEmbedding::clone`'s identical comment: `Py::new` can
            // only fail on Python-interpreter-level allocation failure.
            let weight_param = Py::new(py, cloned_param)
                .expect("PyParameter::clone_param()'s result must construct as a Py<PyParameter>");
            Self {
                num_embeddings: self.num_embeddings,
                embedding_dim: self.embedding_dim,
                max_norm: self.max_norm,
                norm_type: self.norm_type,
                scale_grad_by_freq: self.scale_grad_by_freq,
                mode: self.mode.clone(),
                sparse: self.sparse,
                include_last_offset: self.include_last_offset,
                padding_idx: self.padding_idx,
                weight_param,
            }
        })
    }
}

#[pymethods]
impl PyEmbeddingBag {
    /// Create a new EmbeddingBag layer
    #[new]
    #[pyo3(signature = (num_embeddings, embedding_dim, max_norm=None, norm_type=None, scale_grad_by_freq=None, mode=None, sparse=None, include_last_offset=None, padding_idx=None))]
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        max_norm: Option<f32>,
        norm_type: Option<f32>,
        scale_grad_by_freq: Option<bool>,
        mode: Option<String>,
        sparse: Option<bool>,
        include_last_offset: Option<bool>,
        padding_idx: Option<usize>,
    ) -> PyResult<Self> {
        let norm_type = norm_type.unwrap_or(2.0);
        let scale_grad_by_freq = scale_grad_by_freq.unwrap_or(false);
        let mode = mode.unwrap_or_else(|| "mean".to_string());
        let sparse = sparse.unwrap_or(false);
        let include_last_offset = include_last_offset.unwrap_or(false);

        if num_embeddings == 0 {
            return Err(PyValueError::new_err("num_embeddings must be positive"));
        }
        if embedding_dim == 0 {
            return Err(PyValueError::new_err("embedding_dim must be positive"));
        }
        if mode != "sum" && mode != "mean" && mode != "max" {
            return Err(PyValueError::new_err(
                "mode must be 'sum', 'mean', or 'max'",
            ));
        }
        if let Some(idx) = padding_idx {
            if idx >= num_embeddings {
                return Err(PyValueError::new_err(format!(
                    "padding_idx {} is out of range for num_embeddings {}",
                    idx, num_embeddings
                )));
            }
        }
        if let Some(max_norm_val) = max_norm {
            if max_norm_val <= 0.0 {
                return Err(PyValueError::new_err("max_norm must be positive"));
            }
        }
        if norm_type <= 0.0 {
            return Err(PyValueError::new_err("norm_type must be positive"));
        }

        let weight_param = new_weight_param(num_embeddings, embedding_dim)?;

        Ok(PyEmbeddingBag {
            num_embeddings,
            embedding_dim,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            mode,
            sparse,
            include_last_offset,
            padding_idx,
            weight_param,
        })
    }

    /// Forward pass through the embedding bag layer
    ///
    /// # Arguments
    ///
    /// * `input` - LongTensor containing bags of indices
    /// * `offsets` - Optional LongTensor containing starting index positions for each bag
    /// * `per_sample_weights` - Optional tensor of weights for each embedding lookup
    ///
    /// # Returns
    ///
    /// Tensor of shape (num_bags, embedding_dim) containing aggregated embeddings
    ///
    /// # Autograd participation and its one boundary
    ///
    /// There is no single fused, tape-aware "EmbeddingBag" tape operation
    /// anywhere in this codebase, so this decomposes the bag reduction into
    /// operations that already are tape-aware:
    ///
    /// 1. One [`gather_rows`] call over every index in the whole batch
    ///    (flattened), which both computes the real forward gather and
    ///    records it onto the tape via [`record_and_link_gather`].
    /// 2. Per bag, a tape-aware slice (`UnaryOpKind::Slice`, via
    ///    [`record_and_link_unary`]) pulls out that bag's rows from the
    ///    gathered result, and a tape-aware `sum`/`mean`
    ///    (`UnaryOpKind::Sum`/`Mean`, via the same [`record_and_link_unary`]
    ///    hook [`crate::math_ops::sum`]/[`crate::math_ops::mean`] use)
    ///    reduces it to one row.
    /// 3. A tape-aware `stack` (`VariadicOpKind::Stack`, via
    ///    [`record_and_link_variadic`]) reassembles the per-bag rows into the
    ///    final `[num_bags, embedding_dim]` output.
    ///
    /// `mode = "max"` is the one case this does **not** extend past the
    /// gather step: no tape-aware max reduction exists anywhere in this
    /// crate (`crate::math_ops::max` computes its forward result directly
    /// via `tenflowers_core::ops::max` without ever calling
    /// `record_and_link_unary` — confirmed by reading that function in full),
    /// so there is no already-working tape-aware primitive this function
    /// could reuse for it without inventing new gradient math, which is
    /// explicitly out of scope (this file only wires *existing*, tested
    /// backward implementations onto the tape — see the module-level doc).
    /// `max` mode's forward output is therefore still computed correctly
    /// (via the same gathered rows every other mode uses), but a
    /// `.backward()` call through a `max`-mode `EmbeddingBag` will not
    /// populate a gradient for rows that only ever influenced the output
    /// through the max selection.
    ///
    /// `per_sample_weights` (only valid for `mode = "sum"`, matching
    /// PyTorch) is applied as a tape-aware element-wise multiply
    /// ([`PyTensor::mul`], via `BinaryOpKind::Mul`) against each gathered
    /// row *before* the per-bag sum, so the weight table's gradient
    /// correctly reflects the per-sample scaling; the weights tensor itself
    /// is never marked as a leaf, and hence never receives a gradient of its
    /// own (mirroring this crate's pre-existing scope: `per_sample_weights`
    /// was never a candidate for gradients here).
    #[pyo3(signature = (input, offsets=None, per_sample_weights=None))]
    pub fn forward(
        &self,
        input: &PyTensor,
        offsets: Option<&PyTensor>,
        per_sample_weights: Option<&PyTensor>,
    ) -> PyResult<PyTensor> {
        let weight_snapshot = snapshot_and_mark_leaf(&self.weight_param)?;

        let input_shape = input.tensor.shape();
        let total = input.tensor.size();

        // Flattened 1-D indices: see `indices_to_i32_flat`'s own doc for why
        // this (rather than `indices_to_i32`, which would preserve `input`'s
        // own — possibly 2-D — shape) is required for the per-bag
        // slice/reduce/stack decomposition below.
        let indices = indices_to_i32_flat(input, self.num_embeddings)?;

        // Determine the [start, end) span of each bag from offsets or 2D input shape.
        let bags: Vec<(usize, usize)> = if let Some(offsets_tensor) = offsets {
            let offsets_data = offsets_tensor
                .tensor
                .to_vec()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get offsets: {e}")))?;
            let offs: Vec<usize> = offsets_data.iter().map(|&v| v as usize).collect();
            if offs.is_empty() {
                return Err(PyValueError::new_err("offsets must not be empty"));
            }
            if self.include_last_offset {
                if offs.len() < 2 {
                    return Err(PyValueError::new_err(
                        "offsets must contain at least 2 entries when include_last_offset is set",
                    ));
                }
                (0..offs.len() - 1)
                    .map(|i| (offs[i], offs[i + 1]))
                    .collect()
            } else {
                (0..offs.len())
                    .map(|i| {
                        let start = offs[i];
                        let end = if i + 1 < offs.len() {
                            offs[i + 1]
                        } else {
                            total
                        };
                        (start, end)
                    })
                    .collect()
            }
        } else if input_shape.len() == 2 {
            let num_bags = input_shape[0];
            let bag_size = input_shape[1];
            (0..num_bags)
                .map(|i| (i * bag_size, (i + 1) * bag_size))
                .collect()
        } else {
            return Err(PyValueError::new_err(
                "Either offsets must be provided or input must be 2D",
            ));
        };

        for &(start, end) in &bags {
            if start > end || end > total {
                return Err(PyValueError::new_err(format!(
                    "Invalid bag span [{start}, {end}) for input of length {total}"
                )));
            }
        }

        // Optional per-sample weights (only valid for sum mode, matching PyTorch).
        let sample_weights: Option<Vec<f32>> = if let Some(weights) = per_sample_weights {
            if self.mode != "sum" {
                return Err(PyValueError::new_err(
                    "per_sample_weights is only supported for mode='sum'",
                ));
            }
            let weights_data = weights.tensor.to_vec().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to get per_sample_weights: {e}"))
            })?;
            if weights_data.len() != total {
                return Err(PyValueError::new_err(format!(
                    "per_sample_weights size {} must match input size {}",
                    weights_data.len(),
                    total
                )));
            }
            Some(weights_data)
        } else {
            None
        };

        // One real, tape-linked gather over every index in the batch.
        let gathered = gather_rows(&weight_snapshot, &indices)?;

        if self.mode == "max" {
            return Self::forward_max(&gathered, &bags, self.embedding_dim);
        }

        let dim = self.embedding_dim;
        let mut bag_outputs: Vec<PyTensor> = Vec::with_capacity(bags.len());

        for &(start, end) in &bags {
            let bag_len = end - start;

            if bag_len == 0 {
                // An empty bag contributes an all-zero row; there is nothing
                // to slice/reduce/scale, and this must not become a tape
                // leaf of its own (it carries no gradient path back to the
                // weight table, matching PyTorch's own empty-bag behavior).
                let zero_row = Tensor::zeros(&[dim]);
                bag_outputs.push(PyTensor {
                    tensor: Arc::new(zero_row),
                    requires_grad: false,
                    is_pinned: false,
                });
                continue;
            }

            let bag_slice = tape_aware_slice_rows(&gathered, start, end)?;

            let bag_slice = if let Some(weights) = &sample_weights {
                let scale = build_row_scale(&weights[start..end], dim)?;
                bag_slice.mul(&scale)?
            } else {
                bag_slice
            };

            let reduced = match self.mode.as_str() {
                "sum" => crate::math_ops::sum(&bag_slice, Some(vec![0]), Some(false))?,
                // Deliberately NOT `crate::math_ops::mean(&bag_slice, Some(vec![0]), Some(false))`:
                // `tenflowers_autograd`'s `process_mean_backward` divides by
                // `input_shape.iter().product()` (the reduced-INPUT's total
                // element count) rather than by the size of the specific
                // reduced axis, which only happens to be correct when the
                // whole tensor is reduced (`axes = None`) — for a partial,
                // `axes = Some([0])` reduction like this one (reducing only
                // the bag-length axis of a `[bag_len, dim]` slice), it
                // silently divides by `bag_len * dim` instead of `bag_len`,
                // corrupting the weight-table gradient by a factor of `dim`
                // (caught by this file's own
                // `embedding_bag_mean_weight_gradient_is_scaled_by_bag_size`
                // test during development — it failed with gradient `1/3`
                // instead of `1.0` for a bag_len=2, dim=3 case, exactly the
                // `1/(bag_len*dim)` vs `1/bag_len` signature of this bug).
                // `sum` has no such axis-conditional divisor at all (see
                // `gather_backward`'s doc — pure scatter-add), so `mean` is
                // instead correctly expressed here as `sum` followed by an
                // explicit, tape-aware scalar division by `bag_len` via
                // `PyTensor::div` (`process_div_backward` is a standard,
                // axis-independent quotient-rule backward with no such bug).
                "mean" => {
                    let summed = crate::math_ops::sum(&bag_slice, Some(vec![0]), Some(false))?;
                    let divisor = constant_row(bag_len as f32, dim)?;
                    summed.div(&divisor)?
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "Unsupported EmbeddingBag mode '{other}'"
                    )));
                }
            };

            bag_outputs.push(reduced);
        }

        let bag_refs: Vec<&PyTensor> = bag_outputs.iter().collect();
        stack_bags(&bag_refs)
    }

    /// Reset layer parameters
    pub fn reset_parameters(&mut self) -> PyResult<()> {
        let fresh = Tensor::zeros(&[self.num_embeddings, self.embedding_dim]);

        // If `padding_idx` is set, zero out that row. Zeros are already the
        // freshly-reset value for every row, so there is nothing further to
        // do here; this mirrors the pre-existing (documented) behavior.
        Python::attach(|py| self.weight_param.borrow(py).set_data(fresh))
    }

    /// Get this layer's trainable parameters. See [`PyEmbedding::parameters`]
    /// for the full rationale — identical design.
    pub fn parameters(&self) -> Vec<Py<PyParameter>> {
        Python::attach(|py| vec![self.weight_param.clone_ref(py)])
    }

    fn __repr__(&self) -> String {
        format!(
            "EmbeddingBag(num_embeddings={}, embedding_dim={}, mode='{}', max_norm={:?}, sparse={})",
            self.num_embeddings, self.embedding_dim, self.mode, self.max_norm, self.sparse
        )
    }
}

impl PyEmbeddingBag {
    /// `mode = "max"` forward path: compute the per-bag element-wise maximum
    /// directly from `gathered`'s raw data. See [`PyEmbeddingBag::forward`]'s
    /// own doc for why this does not extend the tape past the gather step —
    /// no tape-aware max reduction exists anywhere in this crate to reuse.
    ///
    /// A plain (non-`#[pymethods]`) associated function: it takes
    /// `bags: &[(usize, usize)]`, which pyo3 cannot accept as a
    /// Python-visible method argument.
    fn forward_max(gathered: &PyTensor, bags: &[(usize, usize)], dim: usize) -> PyResult<PyTensor> {
        let gathered_data = gathered
            .tensor
            .to_vec()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read gathered rows: {e}")))?;

        let num_bags = bags.len();
        let mut output_data = vec![0f32; num_bags * dim];

        for (bag_idx, &(start, end)) in bags.iter().enumerate() {
            let out_base = bag_idx * dim;
            let bag_len = end - start;
            if bag_len == 0 {
                continue;
            }

            for k in 0..dim {
                output_data[out_base + k] = f32::NEG_INFINITY;
            }
            for row in start..end {
                let row_base = row * dim;
                for k in 0..dim {
                    let val = gathered_data[row_base + k];
                    if val > output_data[out_base + k] {
                        output_data[out_base + k] = val;
                    }
                }
            }
        }

        let output = Tensor::from_vec(output_data, &[num_bags, dim])
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to build output: {e}")))?;

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: false,
            is_pinned: false,
        })
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

    // Rows: [1,2,3], [4,5,6], [7,8,9], [10,11,12].
    fn table() -> Tensor<f32> {
        let data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
        Tensor::from_vec(data, &[4, 3]).expect("table")
    }

    fn make_embedding(weight: Tensor<f32>) -> PyEmbedding {
        let emb = PyEmbedding::new(4, 3, None, None, None, None, None).expect("emb");
        Python::attach(|py| emb.weight_param.borrow(py).set_data(weight)).expect("set_data");
        emb
    }

    fn make_bag(mode: &str, weight: Tensor<f32>) -> PyEmbeddingBag {
        let bag = PyEmbeddingBag::new(
            4,
            3,
            None,
            None,
            None,
            Some(mode.to_string()),
            None,
            None,
            None,
        )
        .expect("bag");
        Python::attach(|py| bag.weight_param.borrow(py).set_data(weight)).expect("set_data");
        bag
    }

    #[test]
    fn embedding_forward_gathers_rows() {
        Python::initialize();
        let emb = make_embedding(table());
        let input = make_tensor(vec![1.0, 3.0], &[2]);
        let out = emb.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![2, 3]);
        let out_vec = out.tensor.to_vec().expect("vec");
        assert_eq!(out_vec, vec![4.0, 5.0, 6.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn embedding_out_of_range_errors() {
        Python::initialize();
        let emb = make_embedding(table());
        let input = make_tensor(vec![9.0], &[1]);
        assert!(emb.forward(&input).is_err());
    }

    #[test]
    fn embedding_parameters_returns_shared_handle() {
        Python::initialize();
        let emb = make_embedding(table());
        let params = emb.parameters();
        assert_eq!(params.len(), 1);
        Python::attach(|py| {
            assert_eq!(params[0].borrow(py).id(), emb.weight_param.borrow(py).id());
            assert_eq!(params[0].borrow(py).shape(), vec![4, 3]);
        });
    }

    /// The critical round-trip test: forward -> loss (sum) -> backward must
    /// populate a gradient on the *weight parameter itself* (read back via
    /// `.parameters()[0].grad()`, exactly how real user code would read it),
    /// correctly ZERO for rows never looked up, and correctly ACCUMULATED
    /// (summed, not overwritten) for a row looked up more than once in the
    /// same batch. This mirrors
    /// `tenflowers_autograd::grad_ops::tensor_ops::tests::test_gather_backward_repeated_index_accumulates`
    /// at the kernel level, but exercised through the full `PyEmbedding`
    /// FFI surface: `PyParameter` snapshot -> `mark_leaf_param` ->
    /// `record_and_link_gather` -> `.backward()` -> `PyParameter::grad()`.
    #[test]
    fn weight_gradient_accumulates_for_repeated_index() {
        Python::initialize();
        let emb = make_embedding(table());
        // Indices [0, 2, 0]: row 0 is looked up twice, row 2 once, rows 1
        // and 3 never.
        let input = make_tensor(vec![0.0, 2.0, 0.0], &[3]);

        let out = emb.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![3, 3]);

        // loss = sum(out) -- every gathered element contributes gradient
        // 1.0 to its source position.
        let loss = crate::math_ops::sum(&out, None, None).expect("sum");
        loss.backward().expect("backward");

        let params = emb.parameters();
        let grad = Python::attach(|py| params[0].borrow(py).grad()).expect("grad populated");
        assert_eq!(grad.tensor.shape().dims().to_vec(), vec![4, 3]);
        let grad_data = grad.tensor.to_vec().expect("grad vec");

        // Row 0 gathered twice (positions 0 and 2 in the batch): each
        // contributes [1,1,1] under a sum-loss, so row 0's gradient is the
        // ACCUMULATED [2,2,2], not [1,1,1].
        assert_eq!(&grad_data[0..3], &[2.0, 2.0, 2.0]);
        // Row 1 never gathered: gradient must be exactly zero.
        assert_eq!(&grad_data[3..6], &[0.0, 0.0, 0.0]);
        // Row 2 gathered once: gradient is exactly [1,1,1].
        assert_eq!(&grad_data[6..9], &[1.0, 1.0, 1.0]);
        // Row 3 never gathered: gradient must be exactly zero.
        assert_eq!(&grad_data[9..12], &[0.0, 0.0, 0.0]);
    }

    /// Same accumulation property, but for a row gathered THREE times (twice
    /// consecutively, once after an intervening different row) with
    /// non-uniform per-position values in the upstream gradient, obtained by
    /// weighting the sum loss unevenly via an extra multiply before the
    /// final reduction. This rules out a backward implementation that
    /// happens to only handle the "all-ones upstream gradient" case
    /// correctly.
    #[test]
    fn weight_gradient_accumulates_for_triple_repeated_index_with_nonuniform_upstream() {
        Python::initialize();
        let emb = make_embedding(table());
        // Indices [0, 1, 0, 2, 0]: row 0 gathered three times (batch
        // positions 0, 2, 4).
        let input = make_tensor(vec![0.0, 1.0, 0.0, 2.0, 0.0], &[5]);
        let out = emb.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![5, 3]);

        // Non-uniform upstream gradient: scale row i of `out` by (i + 1)
        // before summing, so d(loss)/d(out[i, :]) = (i+1) for every column,
        // rather than a uniform 1.0 everywhere.
        let scale = make_tensor(
            vec![
                1.0, 1.0, 1.0, //
                2.0, 2.0, 2.0, //
                3.0, 3.0, 3.0, //
                4.0, 4.0, 4.0, //
                5.0, 5.0, 5.0,
            ],
            &[5, 3],
        );
        let scaled = out.mul(&scale).expect("mul");
        let loss = crate::math_ops::sum(&scaled, None, None).expect("sum");
        loss.backward().expect("backward");

        let params = emb.parameters();
        let grad = Python::attach(|py| params[0].borrow(py).grad()).expect("grad populated");
        let grad_data = grad.tensor.to_vec().expect("grad vec");

        // Row 0 gathered at batch positions 0, 2, 4 with upstream weights
        // 1, 3, 5 respectively -> accumulated gradient (1+3+5) = 9 per
        // column.
        assert_eq!(&grad_data[0..3], &[9.0, 9.0, 9.0]);
        // Row 1 gathered once at batch position 1, upstream weight 2.
        assert_eq!(&grad_data[3..6], &[2.0, 2.0, 2.0]);
        // Row 2 gathered once at batch position 3, upstream weight 4.
        assert_eq!(&grad_data[6..9], &[4.0, 4.0, 4.0]);
        // Row 3 never gathered: exactly zero.
        assert_eq!(&grad_data[9..12], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn embedding_bag_mean_2d() {
        Python::initialize();
        let bag = make_bag("mean", table());
        let input = make_tensor(vec![0.0, 1.0, 2.0, 3.0], &[2, 2]);
        let out = bag.forward(&input, None, None).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![2, 3]);
        let out_vec = out.tensor.to_vec().expect("vec");
        // mean([1,2,3],[4,5,6]) and mean([7,8,9],[10,11,12]).
        assert_eq!(out_vec, vec![2.5, 3.5, 4.5, 8.5, 9.5, 10.5]);
    }

    #[test]
    fn embedding_bag_sum_with_offsets() {
        Python::initialize();
        let bag = make_bag("sum", table());
        let input = make_tensor(vec![0.0, 1.0, 2.0, 3.0], &[4]);
        let offsets = make_tensor(vec![0.0, 2.0], &[2]);
        let out = bag.forward(&input, Some(&offsets), None).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![2, 3]);
        let out_vec = out.tensor.to_vec().expect("vec");
        // bag0 = [1,2,3]+[4,5,6]; bag1 = [7,8,9]+[10,11,12].
        assert_eq!(out_vec, vec![5.0, 7.0, 9.0, 17.0, 19.0, 21.0]);
    }

    #[test]
    fn embedding_bag_max_2d() {
        Python::initialize();
        let bag = make_bag("max", table());
        let input = make_tensor(vec![0.0, 1.0, 2.0, 3.0], &[1, 4]);
        let out = bag.forward(&input, None, None).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 3]);
        let out_vec = out.tensor.to_vec().expect("vec");
        // max over all four rows -> row 3.
        assert_eq!(out_vec, vec![10.0, 11.0, 12.0]);
    }

    #[test]
    fn embedding_bag_sum_weight_gradient_accumulates_for_repeated_index() {
        Python::initialize();
        let bag = make_bag("sum", table());
        // Two bags: bag0 = indices [0, 0] (row 0 twice), bag1 = indices [2,
        // 3].
        let input = make_tensor(vec![0.0, 0.0, 2.0, 3.0], &[2, 2]);
        let out = bag.forward(&input, None, None).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![2, 3]);

        let loss = crate::math_ops::sum(&out, None, None).expect("sum");
        loss.backward().expect("backward");

        let params = bag.parameters();
        let grad = Python::attach(|py| params[0].borrow(py).grad()).expect("grad populated");
        let grad_data = grad.tensor.to_vec().expect("grad vec");

        // Row 0 contributes to bag0 twice under a sum-mode, sum-loss bag ->
        // accumulated gradient [2,2,2].
        assert_eq!(&grad_data[0..3], &[2.0, 2.0, 2.0]);
        // Row 1 never gathered: exactly zero.
        assert_eq!(&grad_data[3..6], &[0.0, 0.0, 0.0]);
        // Rows 2, 3 each gathered once in bag1: gradient [1,1,1] each.
        assert_eq!(&grad_data[6..9], &[1.0, 1.0, 1.0]);
        assert_eq!(&grad_data[9..12], &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn embedding_bag_mean_weight_gradient_is_scaled_by_bag_size() {
        Python::initialize();
        let bag = make_bag("mean", table());
        // One bag of size 2: indices [0, 0] (row 0 twice).
        let input = make_tensor(vec![0.0, 0.0], &[1, 2]);
        let out = bag.forward(&input, None, None).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 3]);

        let loss = crate::math_ops::sum(&out, None, None).expect("sum");
        loss.backward().expect("backward");

        let params = bag.parameters();
        let grad = Python::attach(|py| params[0].borrow(py).grad()).expect("grad populated");
        let grad_data = grad.tensor.to_vec().expect("grad vec");

        // mean over a 2-element bag scales each contribution by 1/2; row 0
        // is gathered twice, so its accumulated gradient is 2 * (1/2) = 1.0
        // per column, not 2.0.
        assert_eq!(&grad_data[0..3], &[1.0, 1.0, 1.0]);
        assert_eq!(&grad_data[3..6], &[0.0, 0.0, 0.0]);
        assert_eq!(&grad_data[6..9], &[0.0, 0.0, 0.0]);
        assert_eq!(&grad_data[9..12], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn embedding_bag_uninitialized_weight_starts_at_zero() {
        Python::initialize();
        let bag = PyEmbeddingBag::new(
            4,
            3,
            None,
            None,
            None,
            Some("mean".to_string()),
            None,
            None,
            None,
        )
        .expect("bag");
        let input = make_tensor(vec![0.0, 1.0], &[1, 2]);
        let out = bag.forward(&input, None, None).expect("forward");
        let out_vec = out.tensor.to_vec().expect("vec");
        assert_eq!(out_vec, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn embedding_clone_produces_independent_parameter_identity() {
        Python::initialize();
        let emb = make_embedding(table());
        let cloned = emb.clone();
        Python::attach(|py| {
            assert_ne!(
                emb.weight_param.borrow(py).id(),
                cloned.weight_param.borrow(py).id(),
                "clone() must allocate a fresh, independent parameter identity"
            );
        });
        // Values must still match at the moment of cloning.
        let orig_input = make_tensor(vec![0.0], &[1]);
        let a = emb.forward(&orig_input).expect("forward");
        let b = cloned.forward(&orig_input).expect("forward");
        assert_eq!(
            a.tensor.to_vec().expect("vec"),
            b.tensor.to_vec().expect("vec")
        );
    }
}
