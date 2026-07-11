//! Normalization layers module for TenfloweRS FFI
//!
//! This module provides comprehensive normalization layer implementations including
//! BatchNorm, LayerNorm, GroupNorm, and InstanceNorm for neural network training.
//!
//! # Autograd
//!
//! Every layer's learnable affine parameters (`gamma`/`beta`) are held as a
//! [`Py<PyParameter>`] pair — the same stable-identity, mutable-in-place
//! parameter cell [`super::layers::PyDense`] does not yet use but
//! [`super::layers::PyParameter`] itself was designed for (see that struct's
//! doc). `Py<T>`'s `Clone` impl is gated behind pyo3's `py-clone` feature
//! (not enabled by this workspace), so each layer below implements `Clone`
//! manually via [`Py::clone_ref`] under a freshly attached GIL token — a
//! cheap refcount bump, not a deep copy — rather than deriving it.
//!
//! `forward()` snapshots each parameter's current value via
//! [`super::layers::PyParameter::to_tensor`], marks that snapshot as a tape
//! leaf via [`crate::implicit_autograd::mark_leaf_param`] (keyed by the
//! parameter's own stable `id`, not the snapshot's own transient `Arc`
//! identity), computes the real forward normalization, and then calls
//! [`crate::implicit_autograd::record_and_link_ternary`] with the matching
//! [`crate::implicit_autograd::TernaryOpKind`] so a later `.backward()` call
//! can differentiate through it. `.parameters()` returns
//! `vec![gamma_param.clone_ref(py), beta_param.clone_ref(py)]` — the exact
//! same underlying `PyParameter` objects `forward()` marks as leaves, so
//! `.grad()` on the returned handles is populated after `.backward()`.
//!
//! `running_mean`/`running_var` (BatchNorm1d only) are **not** trainable
//! parameters — they are forward-only statistics updated in place by an
//! exponential moving average during training, never by gradient descent —
//! so they stay plain `Tensor<f32>` fields, wrapped as transient, non-leaf
//! [`PyTensor`]s only for the duration of a single `record_and_link_ternary`
//! call (via that function's `running_stats` parameter). They are
//! deliberately excluded from `.parameters()`.
//!
//! ## Why some layers reshape to 4D before recording on the tape
//!
//! [`tenflowers_core::ops::batch_norm`] hard-requires exactly 4D (NCHW)
//! input, and (less obviously) so do [`tenflowers_autograd`]'s
//! `group_norm_backward`/`instance_norm_backward` kernels — even though
//! their *forward* counterparts (`tenflowers_core::ops::group_norm`, reused
//! for both GroupNorm and InstanceNorm) accept any rank >= 2. A 2D `(N, C)`
//! or 3D `(N, C, L)` input is therefore reshaped up to 4D
//! (`(N, C, 1, 1)`/`(N, C, L, 1)`) — preserving every element and each
//! channel's statistics exactly — *before* it is handed to
//! `record_and_link_ternary`, and the ternary op's result is reshaped back
//! down afterward. Both reshapes are themselves recorded onto the implicit
//! tape (via [`crate::implicit_autograd::record_and_link_unary`] with
//! [`crate::implicit_autograd::UnaryOpKind::Reshape`]), so gradients flow
//! through them correctly rather than silently detaching the graph.
//! LayerNorm's backward kernel is rank-agnostic (it derives its reduction
//! axes from `normalized_shape` directly), so `PyLayerNorm` needs no such
//! reshape.

use super::layers::PyParameter;
use crate::implicit_autograd::{
    mark_leaf_param, record_and_link_ternary, record_and_link_unary, TernaryOpKind, UnaryOpKind,
};
use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tenflowers_core::Tensor;

/// Reshape `tensor`'s tracked/leaf value up to at least `min_rank` dimensions
/// by appending trailing size-1 axes, recording the reshape on the implicit
/// tape so gradients flow back through it. A no-op (returns a shallow clone
/// of `tensor`, still correctly tape-linked to itself via identity) when
/// `tensor` is already at least `min_rank`-D.
///
/// Returns `(reshaped, original_dims)` so the caller can reshape the result
/// back down to `original_dims` afterward.
fn tape_reshape_up(tensor: &PyTensor, min_rank: usize) -> PyResult<(PyTensor, Vec<usize>)> {
    let original_dims: Vec<usize> = tensor.tensor.shape().dims().to_vec();
    if original_dims.len() >= min_rank {
        return Ok((tensor.clone(), original_dims));
    }

    let mut new_shape = original_dims.clone();
    new_shape.resize(min_rank, 1);

    let raw = tenflowers_core::ops::reshape(tensor.tensor.as_ref(), &new_shape)
        .map_err(|e| PyRuntimeError::new_err(format!("normalization reshape failed: {e}")))?;
    let reshaped = PyTensor {
        tensor: Arc::new(raw),
        requires_grad: tensor.requires_grad,
        is_pinned: tensor.is_pinned,
    };
    record_and_link_unary(
        UnaryOpKind::Reshape {
            shape: new_shape.clone(),
        },
        tensor,
        &reshaped,
    )?;

    Ok((reshaped, original_dims))
}

/// Reshape `tensor` back down to `target_dims`, recording the reshape on the
/// implicit tape. Counterpart of [`tape_reshape_up`]; a no-op passthrough
/// (still tape-linked to itself via identity) when `tensor` is already
/// shaped `target_dims`.
fn tape_reshape_down(tensor: &PyTensor, target_dims: &[usize]) -> PyResult<PyTensor> {
    if tensor.tensor.shape().dims() == target_dims {
        return Ok(tensor.clone());
    }

    let raw = tenflowers_core::ops::reshape(tensor.tensor.as_ref(), target_dims)
        .map_err(|e| PyRuntimeError::new_err(format!("normalization reshape failed: {e}")))?;
    let reshaped = PyTensor {
        tensor: Arc::new(raw),
        requires_grad: tensor.requires_grad,
        is_pinned: tensor.is_pinned,
    };
    record_and_link_unary(
        UnaryOpKind::Reshape {
            shape: target_dims.to_vec(),
        },
        tensor,
        &reshaped,
    )?;

    Ok(reshaped)
}

/// Batch Normalization layer
///
/// Normalizes the input across the batch dimension, commonly used in CNNs.
/// Maintains running statistics for inference and learnable affine parameters.
#[pyclass(name = "BatchNorm1d")]
#[derive(Debug)]
pub struct PyBatchNorm1d {
    /// Number of features (channels)
    pub num_features: usize,
    /// Small constant for numerical stability
    pub eps: f32,
    /// Momentum for running statistics
    pub momentum: f32,
    /// Whether to track running statistics
    pub track_running_stats: bool,
    /// Learnable scale parameter (gamma). Always present: normalization
    /// layers, unlike a conv's optional bias, always have an affine
    /// transform in this crate's design (see the module-level "Autograd"
    /// doc) — there is no `affine=False` forward path here.
    pub gamma_param: Py<PyParameter>,
    /// Learnable bias parameter (beta). Always present, mirroring `gamma_param`.
    pub beta_param: Py<PyParameter>,
    /// Running mean for inference. Not a trainable parameter (see the
    /// module-level doc), so a plain tensor rather than a `PyParameter`.
    pub running_mean: Tensor<f32>,
    /// Running variance for inference. Not a trainable parameter.
    pub running_var: Tensor<f32>,
    /// Number of batches tracked
    pub num_batches_tracked: usize,
    /// Training mode flag
    pub training: bool,
}

impl Clone for PyBatchNorm1d {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            num_features: self.num_features,
            eps: self.eps,
            momentum: self.momentum,
            track_running_stats: self.track_running_stats,
            gamma_param: self.gamma_param.clone_ref(py),
            beta_param: self.beta_param.clone_ref(py),
            running_mean: self.running_mean.clone(),
            running_var: self.running_var.clone(),
            num_batches_tracked: self.num_batches_tracked,
            training: self.training,
        })
    }
}

#[pymethods]
impl PyBatchNorm1d {
    /// Create a new BatchNorm1d layer
    ///
    /// # Arguments
    ///
    /// * `num_features` - Number of features (C from an expected input of size (N, C, L))
    /// * `eps` - Value added to denominator for numerical stability (default: 1e-5)
    /// * `momentum` - Value used for running_mean and running_var computation (default: 0.1)
    /// * `track_running_stats` - Whether to track running statistics (default: true)
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5, momentum=0.1, track_running_stats=true))]
    pub fn new(
        py: Python<'_>,
        num_features: usize,
        eps: Option<f32>,
        momentum: Option<f32>,
        track_running_stats: Option<bool>,
    ) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);
        let track_running_stats = track_running_stats.unwrap_or(true);

        if num_features == 0 {
            return Err(PyValueError::new_err("num_features must be positive"));
        }

        let gamma_param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(Tensor::ones(&[num_features])),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;
        let beta_param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(Tensor::zeros(&[num_features])),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;

        Ok(PyBatchNorm1d {
            num_features,
            eps,
            momentum,
            track_running_stats,
            gamma_param,
            beta_param,
            running_mean: Tensor::zeros(&[num_features]),
            running_var: Tensor::ones(&[num_features]),
            num_batches_tracked: 0,
            training: true,
        })
    }

    /// Forward pass through the batch normalization layer
    pub fn forward(&mut self, py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
        // Input shape: (N, C) or (N, C, L)
        let input_shape = input.tensor.shape();

        if input_shape.len() < 2 || input_shape.len() > 3 {
            return Err(PyValueError::new_err(format!(
                "Expected 2D or 3D input, got {}D",
                input_shape.len()
            )));
        }

        if input_shape[1] != self.num_features {
            return Err(PyValueError::new_err(format!(
                "Expected {} features, got {}",
                self.num_features, input_shape[1]
            )));
        }

        // Snapshot the current parameter values and (re-)register them as
        // tape leaves for this forward pass. Idempotent across repeated
        // calls (see `mark_leaf_param`'s doc) — cheap after the first call
        // within a given forward/backward cycle.
        let gamma_snapshot = self.gamma_param.borrow(py).to_tensor()?;
        let gamma_id = self.gamma_param.borrow(py).id();
        mark_leaf_param(&gamma_snapshot, gamma_id);

        let beta_snapshot = self.beta_param.borrow(py).to_tensor()?;
        let beta_id = self.beta_param.borrow(py).id();
        mark_leaf_param(&beta_snapshot, beta_id);

        // The core batch_norm op (and, critically, the autograd crate's
        // group_norm/instance_norm backward kernels used by the sibling
        // layers below) hard-require 4D input. Reshape (N, C) -> (N, C, 1, 1)
        // and (N, C, L) -> (N, C, L, 1) — both keep per-channel statistics
        // intact — recording the reshape on the tape so gradients flow back
        // through it.
        let (input_4d, orig_dims) = tape_reshape_up(input, 4)?;

        // running_mean/running_var are forward-only statistics: never
        // trainable parameters, so wrapped as transient, non-leaf PyTensors
        // purely to satisfy record_and_link_ternary's `running_stats`
        // argument (required for BatchNorm — see that function's doc).
        let running_mean_tensor = PyTensor {
            tensor: Arc::new(self.running_mean.clone()),
            requires_grad: false,
            is_pinned: false,
        };
        let running_var_tensor = PyTensor {
            tensor: Arc::new(self.running_var.clone()),
            requires_grad: false,
            is_pinned: false,
        };

        // Without tracked running stats, normalisation always uses batch
        // statistics (the training path), matching standard BatchNorm
        // semantics.
        let use_batch_stats = self.training || !self.track_running_stats;

        let normalized_raw = tenflowers_core::ops::batch_norm(
            input_4d.tensor.as_ref(),
            gamma_snapshot.tensor.as_ref(),
            beta_snapshot.tensor.as_ref(),
            running_mean_tensor.tensor.as_ref(),
            running_var_tensor.tensor.as_ref(),
            self.eps,
            use_batch_stats,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("BatchNorm1d forward failed: {e}")))?;
        let normalized_4d = PyTensor {
            tensor: Arc::new(normalized_raw),
            requires_grad: true,
            is_pinned: input.is_pinned,
        };

        record_and_link_ternary(
            TernaryOpKind::BatchNorm {
                epsilon: self.eps,
                training: use_batch_stats,
            },
            &input_4d,
            &gamma_snapshot,
            Some(&beta_snapshot),
            Some((&running_mean_tensor, &running_var_tensor)),
            &normalized_4d,
        )?;

        if use_batch_stats && self.track_running_stats {
            self.update_running_stats(&input_4d)?;
            self.num_batches_tracked += 1;
        }

        tape_reshape_down(&normalized_4d, &orig_dims)
    }

    /// Set the layer to training mode
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set the layer to evaluation mode
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Get layer parameters: `[gamma, beta]`. Shares identity with the
    /// exact same `PyParameter` objects `forward()` marks as leaves (see
    /// the module-level "Autograd" doc), so `.grad()` on the returned
    /// handles is populated after a `.backward()` call that passes through
    /// this layer's `forward()`.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyParameter>> {
        vec![self.gamma_param.clone_ref(py), self.beta_param.clone_ref(py)]
    }

    /// Reset running statistics
    pub fn reset_running_stats(&mut self) -> PyResult<()> {
        self.running_mean = Tensor::zeros(&[self.num_features]);
        self.running_var = Tensor::ones(&[self.num_features]);
        self.num_batches_tracked = 0;
        Ok(())
    }

    /// Reset parameters
    pub fn reset_parameters(&mut self, py: Python<'_>) -> PyResult<()> {
        self.reset_running_stats()?;
        self.gamma_param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(Tensor::ones(&[self.num_features])),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;
        self.beta_param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(Tensor::zeros(&[self.num_features])),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;
        Ok(())
    }

    /// Get layer state dict
    pub fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);

        let weight_data: Vec<f32> = self
            .gamma_param
            .borrow(py)
            .to_tensor()?
            .tensor
            .to_vec()
            .map_err(|e| PyValueError::new_err(format!("Failed to convert weight: {}", e)))?;
        dict.set_item("weight", weight_data)?;

        let bias_data: Vec<f32> = self
            .beta_param
            .borrow(py)
            .to_tensor()?
            .tensor
            .to_vec()
            .map_err(|e| PyValueError::new_err(format!("Failed to convert bias: {}", e)))?;
        dict.set_item("bias", bias_data)?;

        let mean_data: Vec<f32> = self
            .running_mean
            .to_vec()
            .map_err(|e| PyValueError::new_err(format!("Failed to convert running_mean: {}", e)))?;
        dict.set_item("running_mean", mean_data)?;

        let var_data: Vec<f32> = self
            .running_var
            .to_vec()
            .map_err(|e| PyValueError::new_err(format!("Failed to convert running_var: {}", e)))?;
        dict.set_item("running_var", var_data)?;

        dict.set_item("num_batches_tracked", self.num_batches_tracked)?;

        Ok(dict.into())
    }

    /// Load layer state dict
    pub fn load_state_dict(&mut self, py: Python<'_>, state_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(weight) = state_dict.get_item("weight")? {
            let weight_vec: Vec<f32> = weight.extract()?;
            let tensor = Tensor::from_vec(weight_vec, &[self.num_features])
                .map_err(|e| PyValueError::new_err(format!("Failed to load weight: {}", e)))?;
            self.gamma_param.borrow(py).set_data(tensor)?;
        }

        if let Some(bias) = state_dict.get_item("bias")? {
            let bias_vec: Vec<f32> = bias.extract()?;
            let tensor = Tensor::from_vec(bias_vec, &[self.num_features])
                .map_err(|e| PyValueError::new_err(format!("Failed to load bias: {}", e)))?;
            self.beta_param.borrow(py).set_data(tensor)?;
        }

        if let Some(running_mean) = state_dict.get_item("running_mean")? {
            let mean_vec: Vec<f32> = running_mean.extract()?;
            self.running_mean = Tensor::from_vec(mean_vec, &[self.num_features]).map_err(|e| {
                PyValueError::new_err(format!("Failed to load running_mean: {}", e))
            })?;
        }

        if let Some(running_var) = state_dict.get_item("running_var")? {
            let var_vec: Vec<f32> = running_var.extract()?;
            self.running_var = Tensor::from_vec(var_vec, &[self.num_features])
                .map_err(|e| PyValueError::new_err(format!("Failed to load running_var: {}", e)))?;
        }

        if let Some(num_batches) = state_dict.get_item("num_batches_tracked")? {
            self.num_batches_tracked = num_batches.extract()?;
        }

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchNorm1d(num_features={}, eps={}, momentum={}, track_running_stats={})",
            self.num_features, self.eps, self.momentum, self.track_running_stats
        )
    }
}

impl PyBatchNorm1d {
    /// Update `running_mean`/`running_var` in place via an exponential
    /// moving average of this forward pass's batch statistics, mirroring
    /// standard BatchNorm training semantics
    /// (`running = momentum * batch + (1 - momentum) * running`). Operates
    /// on the already-4D `input_4d` (see [`tape_reshape_up`]) purely to
    /// compute plain `Tensor<f32>` statistics — deliberately untracked by
    /// the implicit tape, since running stats are never differentiated
    /// through (see the module-level doc).
    fn update_running_stats(&mut self, input_4d: &PyTensor) -> PyResult<()> {
        let dims = input_4d.tensor.shape().dims().to_vec();
        let channels = dims[1];
        let reduce_axes: Vec<i32> = vec![0, 2, 3];

        let batch_mean = tenflowers_core::ops::mean(input_4d.tensor.as_ref(), Some(&reduce_axes), true)
            .map_err(|e| PyRuntimeError::new_err(format!("BatchNorm1d stats update failed: {e}")))?;
        let centered = tenflowers_core::ops::sub(input_4d.tensor.as_ref(), &batch_mean)
            .map_err(|e| PyRuntimeError::new_err(format!("BatchNorm1d stats update failed: {e}")))?;
        let squared = tenflowers_core::ops::mul(&centered, &centered)
            .map_err(|e| PyRuntimeError::new_err(format!("BatchNorm1d stats update failed: {e}")))?;
        let batch_var = tenflowers_core::ops::mean(&squared, Some(&reduce_axes), true)
            .map_err(|e| PyRuntimeError::new_err(format!("BatchNorm1d stats update failed: {e}")))?;

        let batch_mean_flat = tenflowers_core::ops::reshape(&batch_mean, &[channels])
            .map_err(|e| PyRuntimeError::new_err(format!("BatchNorm1d stats update failed: {e}")))?;
        let batch_var_flat = tenflowers_core::ops::reshape(&batch_var, &[channels])
            .map_err(|e| PyRuntimeError::new_err(format!("BatchNorm1d stats update failed: {e}")))?;

        let batch_size: usize = reduce_axes.iter().map(|&a| dims[a as usize]).product();
        // Unbiased variance estimate (N / (N - 1)) for the running-var update,
        // matching standard BatchNorm training semantics; falls back to the
        // biased estimate when there is exactly one element per channel (N-1
        // would be zero).
        let unbiased_var_flat = if batch_size > 1 {
            let scale = batch_size as f32 / (batch_size as f32 - 1.0);
            batch_var_flat
                .multiply_scalar(scale)
                .map_err(|e| PyRuntimeError::new_err(format!("BatchNorm1d stats update failed: {e}")))?
        } else {
            batch_var_flat
        };

        let momentum = self.momentum;
        let new_running_mean = self
            .running_mean
            .multiply_scalar(1.0 - momentum)
            .and_then(|scaled_old| {
                batch_mean_flat
                    .multiply_scalar(momentum)
                    .and_then(|scaled_new| tenflowers_core::ops::add(&scaled_old, &scaled_new))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("BatchNorm1d stats update failed: {e}")))?;
        let new_running_var = self
            .running_var
            .multiply_scalar(1.0 - momentum)
            .and_then(|scaled_old| {
                unbiased_var_flat
                    .multiply_scalar(momentum)
                    .and_then(|scaled_new| tenflowers_core::ops::add(&scaled_old, &scaled_new))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("BatchNorm1d stats update failed: {e}")))?;

        self.running_mean = new_running_mean;
        self.running_var = new_running_var;
        Ok(())
    }
}

/// Layer Normalization layer
///
/// Normalizes the input across the feature dimension, commonly used in Transformers.
/// Applies normalization over the last D dimensions where D is the length of normalized_shape.
#[pyclass(name = "LayerNorm")]
#[derive(Debug)]
pub struct PyLayerNorm {
    /// Shape of normalized features
    pub normalized_shape: Vec<usize>,
    /// Small constant for numerical stability
    pub eps: f32,
    /// Learnable scale parameter (gamma). Always present — see the
    /// module-level "Autograd" doc.
    pub gamma_param: Py<PyParameter>,
    /// Learnable bias parameter (beta). Always present.
    pub beta_param: Py<PyParameter>,
}

impl Clone for PyLayerNorm {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            normalized_shape: self.normalized_shape.clone(),
            eps: self.eps,
            gamma_param: self.gamma_param.clone_ref(py),
            beta_param: self.beta_param.clone_ref(py),
        })
    }
}

#[pymethods]
impl PyLayerNorm {
    /// Create a new LayerNorm layer
    ///
    /// # Arguments
    ///
    /// * `normalized_shape` - Input shape from an expected input of size
    /// * `eps` - Value added to denominator for numerical stability (default: 1e-5)
    #[new]
    #[pyo3(signature = (normalized_shape, eps=1e-5))]
    pub fn new(py: Python<'_>, normalized_shape: Vec<usize>, eps: Option<f32>) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-5);

        if normalized_shape.is_empty() {
            return Err(PyValueError::new_err("normalized_shape must not be empty"));
        }

        let gamma_param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(Tensor::ones(&normalized_shape)),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;
        let beta_param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(Tensor::zeros(&normalized_shape)),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;

        Ok(PyLayerNorm {
            normalized_shape,
            eps,
            gamma_param,
            beta_param,
        })
    }

    /// Forward pass through the layer normalization layer
    pub fn forward(&self, py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
        let input_shape = input.tensor.shape();

        // Verify that the last dimensions match normalized_shape
        if input_shape.len() < self.normalized_shape.len() {
            return Err(PyValueError::new_err(format!(
                "Input has {} dimensions, but normalized_shape has {} dimensions",
                input_shape.len(),
                self.normalized_shape.len()
            )));
        }

        let start_idx = input_shape.len() - self.normalized_shape.len();
        let shape_vec: Vec<usize> = input_shape.iter().copied().collect();
        if shape_vec[start_idx..] != self.normalized_shape[..] {
            return Err(PyValueError::new_err(format!(
                "Expected last dimensions to be {:?}, got {:?}",
                self.normalized_shape,
                &shape_vec[start_idx..]
            )));
        }

        let gamma_snapshot = self.gamma_param.borrow(py).to_tensor()?;
        let gamma_id = self.gamma_param.borrow(py).id();
        mark_leaf_param(&gamma_snapshot, gamma_id);

        let beta_snapshot = self.beta_param.borrow(py).to_tensor()?;
        let beta_id = self.beta_param.borrow(py).id();
        mark_leaf_param(&beta_snapshot, beta_id);

        let output_raw = tenflowers_core::ops::layer_norm(
            input.tensor.as_ref(),
            gamma_snapshot.tensor.as_ref(),
            beta_snapshot.tensor.as_ref(),
            &self.normalized_shape,
            self.eps,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("LayerNorm forward failed: {e}")))?;
        let output = PyTensor {
            tensor: Arc::new(output_raw),
            requires_grad: true,
            is_pinned: input.is_pinned,
        };

        record_and_link_ternary(
            TernaryOpKind::LayerNorm {
                epsilon: self.eps,
                normalized_shape: self.normalized_shape.clone(),
            },
            input,
            &gamma_snapshot,
            Some(&beta_snapshot),
            None,
            &output,
        )?;

        Ok(output)
    }

    /// Get layer parameters: `[gamma, beta]`.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyParameter>> {
        vec![self.gamma_param.clone_ref(py), self.beta_param.clone_ref(py)]
    }

    /// Reset parameters
    pub fn reset_parameters(&mut self, py: Python<'_>) -> PyResult<()> {
        self.gamma_param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(Tensor::ones(&self.normalized_shape)),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;
        self.beta_param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(Tensor::zeros(&self.normalized_shape)),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;
        Ok(())
    }

    /// Get layer state dict
    pub fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);

        let weight_data: Vec<f32> = self
            .gamma_param
            .borrow(py)
            .to_tensor()?
            .tensor
            .to_vec()
            .map_err(|e| PyValueError::new_err(format!("Failed to convert weight: {}", e)))?;
        dict.set_item("weight", weight_data)?;

        let bias_data: Vec<f32> = self
            .beta_param
            .borrow(py)
            .to_tensor()?
            .tensor
            .to_vec()
            .map_err(|e| PyValueError::new_err(format!("Failed to convert bias: {}", e)))?;
        dict.set_item("bias", bias_data)?;

        Ok(dict.into())
    }

    /// Load layer state dict
    pub fn load_state_dict(&mut self, py: Python<'_>, state_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(weight) = state_dict.get_item("weight")? {
            let weight_vec: Vec<f32> = weight.extract()?;
            let tensor = Tensor::from_vec(weight_vec, &self.normalized_shape)
                .map_err(|e| PyValueError::new_err(format!("Failed to load weight: {}", e)))?;
            self.gamma_param.borrow(py).set_data(tensor)?;
        }

        if let Some(bias) = state_dict.get_item("bias")? {
            let bias_vec: Vec<f32> = bias.extract()?;
            let tensor = Tensor::from_vec(bias_vec, &self.normalized_shape)
                .map_err(|e| PyValueError::new_err(format!("Failed to load bias: {}", e)))?;
            self.beta_param.borrow(py).set_data(tensor)?;
        }

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "LayerNorm(normalized_shape={:?}, eps={})",
            self.normalized_shape, self.eps
        )
    }
}

/// Group Normalization layer
///
/// Divides channels into groups and normalizes within each group.
/// Useful when batch size is small.
#[pyclass(name = "GroupNorm")]
#[derive(Debug)]
pub struct PyGroupNorm {
    /// Number of groups
    pub num_groups: usize,
    /// Number of channels
    pub num_channels: usize,
    /// Small constant for numerical stability
    pub eps: f32,
    /// Learnable scale parameter (gamma). Always present.
    pub gamma_param: Py<PyParameter>,
    /// Learnable bias parameter (beta). Always present.
    pub beta_param: Py<PyParameter>,
}

impl Clone for PyGroupNorm {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            num_groups: self.num_groups,
            num_channels: self.num_channels,
            eps: self.eps,
            gamma_param: self.gamma_param.clone_ref(py),
            beta_param: self.beta_param.clone_ref(py),
        })
    }
}

#[pymethods]
impl PyGroupNorm {
    /// Create a new GroupNorm layer
    ///
    /// # Arguments
    ///
    /// * `num_groups` - Number of groups to separate the channels into
    /// * `num_channels` - Number of channels expected in input
    /// * `eps` - Value added to denominator for numerical stability (default: 1e-5)
    #[new]
    #[pyo3(signature = (num_groups, num_channels, eps=1e-5))]
    pub fn new(
        py: Python<'_>,
        num_groups: usize,
        num_channels: usize,
        eps: Option<f32>,
    ) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-5);

        if num_groups == 0 {
            return Err(PyValueError::new_err("num_groups must be positive"));
        }

        if num_channels == 0 {
            return Err(PyValueError::new_err("num_channels must be positive"));
        }

        if num_channels % num_groups != 0 {
            return Err(PyValueError::new_err(format!(
                "num_channels ({}) must be divisible by num_groups ({})",
                num_channels, num_groups
            )));
        }

        let gamma_param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(Tensor::ones(&[num_channels])),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;
        let beta_param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(Tensor::zeros(&[num_channels])),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;

        Ok(PyGroupNorm {
            num_groups,
            num_channels,
            eps,
            gamma_param,
            beta_param,
        })
    }

    /// Forward pass through the group normalization layer
    pub fn forward(&self, py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
        let input_shape = input.tensor.shape();

        if input_shape.len() < 2 {
            return Err(PyValueError::new_err(format!(
                "Expected at least 2D input, got {}D",
                input_shape.len()
            )));
        }

        if input_shape[1] != self.num_channels {
            return Err(PyValueError::new_err(format!(
                "Expected {} channels, got {}",
                self.num_channels, input_shape[1]
            )));
        }

        let gamma_snapshot = self.gamma_param.borrow(py).to_tensor()?;
        let gamma_id = self.gamma_param.borrow(py).id();
        mark_leaf_param(&gamma_snapshot, gamma_id);

        let beta_snapshot = self.beta_param.borrow(py).to_tensor()?;
        let beta_id = self.beta_param.borrow(py).id();
        mark_leaf_param(&beta_snapshot, beta_id);

        // group_norm_backward hard-requires 4D input even though the forward
        // kernel accepts any rank >= 2 (see the module-level doc) — reshape
        // up before recording on the tape, and back down afterward.
        let (input_4d, orig_dims) = tape_reshape_up(input, 4)?;

        let output_raw = tenflowers_core::ops::group_norm(
            input_4d.tensor.as_ref(),
            gamma_snapshot.tensor.as_ref(),
            beta_snapshot.tensor.as_ref(),
            self.num_groups,
            self.eps,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("GroupNorm forward failed: {e}")))?;
        let output_4d = PyTensor {
            tensor: Arc::new(output_raw),
            requires_grad: true,
            is_pinned: input.is_pinned,
        };

        record_and_link_ternary(
            TernaryOpKind::GroupNorm {
                num_groups: self.num_groups,
                epsilon: self.eps,
            },
            &input_4d,
            &gamma_snapshot,
            Some(&beta_snapshot),
            None,
            &output_4d,
        )?;

        tape_reshape_down(&output_4d, &orig_dims)
    }

    /// Get layer parameters: `[gamma, beta]`.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyParameter>> {
        vec![self.gamma_param.clone_ref(py), self.beta_param.clone_ref(py)]
    }

    /// Reset parameters
    pub fn reset_parameters(&mut self, py: Python<'_>) -> PyResult<()> {
        self.gamma_param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(Tensor::ones(&[self.num_channels])),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;
        self.beta_param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(Tensor::zeros(&[self.num_channels])),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;
        Ok(())
    }

    /// Get layer state dict
    pub fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);

        let weight_data: Vec<f32> = self
            .gamma_param
            .borrow(py)
            .to_tensor()?
            .tensor
            .to_vec()
            .map_err(|e| PyValueError::new_err(format!("Failed to convert weight: {}", e)))?;
        dict.set_item("weight", weight_data)?;

        let bias_data: Vec<f32> = self
            .beta_param
            .borrow(py)
            .to_tensor()?
            .tensor
            .to_vec()
            .map_err(|e| PyValueError::new_err(format!("Failed to convert bias: {}", e)))?;
        dict.set_item("bias", bias_data)?;

        Ok(dict.into())
    }

    /// Load layer state dict
    pub fn load_state_dict(&mut self, py: Python<'_>, state_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(weight) = state_dict.get_item("weight")? {
            let weight_vec: Vec<f32> = weight.extract()?;
            let tensor = Tensor::from_vec(weight_vec, &[self.num_channels])
                .map_err(|e| PyValueError::new_err(format!("Failed to load weight: {}", e)))?;
            self.gamma_param.borrow(py).set_data(tensor)?;
        }

        if let Some(bias) = state_dict.get_item("bias")? {
            let bias_vec: Vec<f32> = bias.extract()?;
            let tensor = Tensor::from_vec(bias_vec, &[self.num_channels])
                .map_err(|e| PyValueError::new_err(format!("Failed to load bias: {}", e)))?;
            self.beta_param.borrow(py).set_data(tensor)?;
        }

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "GroupNorm(num_groups={}, num_channels={}, eps={})",
            self.num_groups, self.num_channels, self.eps
        )
    }
}

/// Instance Normalization layer
///
/// Applies normalization over each channel in each data sample independently.
/// Commonly used in style transfer and GANs.
#[pyclass(name = "InstanceNorm1d")]
#[derive(Debug)]
pub struct PyInstanceNorm1d {
    /// Number of features (channels)
    pub num_features: usize,
    /// Small constant for numerical stability
    pub eps: f32,
    /// Learnable scale parameter (gamma). Always present.
    pub gamma_param: Py<PyParameter>,
    /// Learnable bias parameter (beta). Always present.
    pub beta_param: Py<PyParameter>,
}

impl Clone for PyInstanceNorm1d {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            num_features: self.num_features,
            eps: self.eps,
            gamma_param: self.gamma_param.clone_ref(py),
            beta_param: self.beta_param.clone_ref(py),
        })
    }
}

#[pymethods]
impl PyInstanceNorm1d {
    /// Create a new InstanceNorm1d layer
    ///
    /// # Arguments
    ///
    /// * `num_features` - Number of features (channels) from an expected input of size (N, C, L)
    /// * `eps` - Value added to denominator for numerical stability (default: 1e-5)
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5))]
    pub fn new(py: Python<'_>, num_features: usize, eps: Option<f32>) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-5);

        if num_features == 0 {
            return Err(PyValueError::new_err("num_features must be positive"));
        }

        let gamma_param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(Tensor::ones(&[num_features])),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;
        let beta_param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(Tensor::zeros(&[num_features])),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;

        Ok(PyInstanceNorm1d {
            num_features,
            eps,
            gamma_param,
            beta_param,
        })
    }

    /// Forward pass through the instance normalization layer
    pub fn forward(&self, py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
        let input_shape = input.tensor.shape();

        if input_shape.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "Expected 3D input (N, C, L), got {}D",
                input_shape.len()
            )));
        }

        if input_shape[1] != self.num_features {
            return Err(PyValueError::new_err(format!(
                "Expected {} features, got {}",
                self.num_features, input_shape[1]
            )));
        }

        let gamma_snapshot = self.gamma_param.borrow(py).to_tensor()?;
        let gamma_id = self.gamma_param.borrow(py).id();
        mark_leaf_param(&gamma_snapshot, gamma_id);

        let beta_snapshot = self.beta_param.borrow(py).to_tensor()?;
        let beta_id = self.beta_param.borrow(py).id();
        mark_leaf_param(&beta_snapshot, beta_id);

        // instance_norm_backward hard-requires 4D input, exactly like
        // group_norm_backward (see the module-level doc): (N, C, L) ->
        // (N, C, L, 1) before recording on the tape, and back down after.
        let (input_4d, orig_dims) = tape_reshape_up(input, 4)?;

        // Instance norm is exactly group norm with one group per channel, so
        // the group_norm op is reused with num_groups == num_channels.
        let output_raw = tenflowers_core::ops::group_norm(
            input_4d.tensor.as_ref(),
            gamma_snapshot.tensor.as_ref(),
            beta_snapshot.tensor.as_ref(),
            self.num_features,
            self.eps,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("InstanceNorm1d forward failed: {e}")))?;
        let output_4d = PyTensor {
            tensor: Arc::new(output_raw),
            requires_grad: true,
            is_pinned: input.is_pinned,
        };

        record_and_link_ternary(
            TernaryOpKind::InstanceNorm { epsilon: self.eps },
            &input_4d,
            &gamma_snapshot,
            Some(&beta_snapshot),
            None,
            &output_4d,
        )?;

        tape_reshape_down(&output_4d, &orig_dims)
    }

    /// Get layer parameters: `[gamma, beta]`.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyParameter>> {
        vec![self.gamma_param.clone_ref(py), self.beta_param.clone_ref(py)]
    }

    /// Reset parameters
    pub fn reset_parameters(&mut self, py: Python<'_>) -> PyResult<()> {
        self.gamma_param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(Tensor::ones(&[self.num_features])),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;
        self.beta_param = Py::new(
            py,
            PyParameter::new(
                PyTensor {
                    tensor: Arc::new(Tensor::zeros(&[self.num_features])),
                    requires_grad: true,
                    is_pinned: false,
                },
                Some(true),
            ),
        )?;
        Ok(())
    }

    /// Get layer state dict
    pub fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);

        let weight_data: Vec<f32> = self
            .gamma_param
            .borrow(py)
            .to_tensor()?
            .tensor
            .to_vec()
            .map_err(|e| PyValueError::new_err(format!("Failed to convert weight: {}", e)))?;
        dict.set_item("weight", weight_data)?;

        let bias_data: Vec<f32> = self
            .beta_param
            .borrow(py)
            .to_tensor()?
            .tensor
            .to_vec()
            .map_err(|e| PyValueError::new_err(format!("Failed to convert bias: {}", e)))?;
        dict.set_item("bias", bias_data)?;

        Ok(dict.into())
    }

    /// Load layer state dict
    pub fn load_state_dict(&mut self, py: Python<'_>, state_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(weight) = state_dict.get_item("weight")? {
            let weight_vec: Vec<f32> = weight.extract()?;
            let tensor = Tensor::from_vec(weight_vec, &[self.num_features])
                .map_err(|e| PyValueError::new_err(format!("Failed to load weight: {}", e)))?;
            self.gamma_param.borrow(py).set_data(tensor)?;
        }

        if let Some(bias) = state_dict.get_item("bias")? {
            let bias_vec: Vec<f32> = bias.extract()?;
            let tensor = Tensor::from_vec(bias_vec, &[self.num_features])
                .map_err(|e| PyValueError::new_err(format!("Failed to load bias: {}", e)))?;
            self.beta_param.borrow(py).set_data(tensor)?;
        }

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "InstanceNorm1d(num_features={}, eps={})",
            self.num_features, self.eps
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::implicit_autograd::{
        record_and_link_unary as record_unary_for_test, run_backward, UnaryOpKind as UnaryKindForTest,
    };

    fn make_tensor(data: Vec<f32>, shape: &[usize]) -> PyTensor {
        let tensor = Tensor::from_vec(data, shape).expect("tensor construction");
        PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: false,
            is_pinned: false,
        }
    }

    fn approx_zero(values: &[f32], tol: f32) -> bool {
        let sum: f32 = values.iter().sum();
        sum.abs() < tol
    }

    /// Sum every element of `tensor` into a scalar `PyTensor`, recording the
    /// reduction on the implicit tape — the `loss = sum(output)` step every
    /// gradient test in this module needs before calling `run_backward`.
    fn tape_sum_to_scalar(tensor: &PyTensor) -> PyTensor {
        let raw = tenflowers_core::ops::sum(&tensor.tensor, None, false).expect("sum must succeed");
        let scalar = PyTensor {
            tensor: Arc::new(raw),
            requires_grad: true,
            is_pinned: false,
        };
        record_unary_for_test(
            UnaryKindForTest::Sum {
                axes: None,
                keepdims: false,
            },
            tensor,
            &scalar,
        )
        .expect("recording sum must succeed");
        scalar
    }

    /// Central-difference numerical gradient of `f` at each element of
    /// `tensor_data`, used to independently verify the analytic gradients
    /// `.backward()` populates. `f` must return a scalar loss.
    fn finite_difference_grad(tensor_data: &[f32], h: f32, f: impl Fn(&[f32]) -> f32) -> Vec<f32> {
        let mut grad = vec![0.0f32; tensor_data.len()];
        for i in 0..tensor_data.len() {
            let mut plus = tensor_data.to_vec();
            plus[i] += h;
            let mut minus = tensor_data.to_vec();
            minus[i] -= h;
            let f_plus = f(&plus);
            let f_minus = f(&minus);
            grad[i] = (f_plus - f_minus) / (2.0 * h);
        }
        grad
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "{label}[{i}]: actual={a}, expected={e}, diff={}",
                (a - e).abs()
            );
        }
    }

    #[test]
    fn batch_norm_normalizes_per_channel() {
        Python::initialize();
        Python::attach(|py| {
            let mut bn = PyBatchNorm1d::new(py, 3, None, None, None).expect("bn construction");
            // (N=4, C=3) with non-constant columns.
            let data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
            let input = make_tensor(data, &[4, 3]);
            let out = bn.forward(py, &input).expect("forward");
            assert_eq!(out.tensor.shape().dims().to_vec(), vec![4, 3]);
            let out_vec = out.tensor.to_vec().expect("out vec");
            assert!(out_vec.iter().any(|&v| v != 0.0), "must not be all zeros");
            assert!(out_vec.iter().all(|v| v.is_finite()));
            // Each channel (column) is mean-centred after training-mode BatchNorm.
            for c in 0..3 {
                let col: Vec<f32> = (0..4).map(|n| out_vec[n * 3 + c]).collect();
                assert!(approx_zero(&col, 1e-3), "channel {c} should have mean 0");
            }
        });
    }

    #[test]
    fn batch_norm_rejects_4d_input() {
        Python::initialize();
        Python::attach(|py| {
            let mut bn = PyBatchNorm1d::new(py, 2, None, None, None).expect("bn construction");
            let input = make_tensor(vec![0.0; 16], &[2, 2, 2, 2]);
            assert!(bn.forward(py, &input).is_err());
        });
    }

    #[test]
    fn layer_norm_centres_last_dim() {
        Python::initialize();
        Python::attach(|py| {
            let ln = PyLayerNorm::new(py, vec![3], None).expect("ln construction");
            let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
            let out = ln.forward(py, &input).expect("forward");
            assert_eq!(out.tensor.shape().dims().to_vec(), vec![2, 3]);
            let out_vec = out.tensor.to_vec().expect("out vec");
            assert!(out_vec.iter().any(|&v| v != 0.0), "must not be all zeros");
            assert!(out_vec.iter().all(|v| v.is_finite()));
            for row in 0..2 {
                let r: Vec<f32> = (0..3).map(|i| out_vec[row * 3 + i]).collect();
                assert!(approx_zero(&r, 1e-3), "row {row} should have mean 0");
            }
        });
    }

    #[test]
    fn group_norm_forward_real_output() {
        Python::initialize();
        Python::attach(|py| {
            let gn = PyGroupNorm::new(py, 2, 4, None).expect("gn construction");
            // (N=1, C=4, L=2)
            let data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
            let input = make_tensor(data, &[1, 4, 2]);
            let out = gn.forward(py, &input).expect("forward");
            assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 4, 2]);
            let out_vec = out.tensor.to_vec().expect("out vec");
            assert!(out_vec.iter().any(|&v| v != 0.0), "must not be all zeros");
            assert!(out_vec.iter().all(|v| v.is_finite()));
        });
    }

    #[test]
    fn instance_norm_centres_each_channel() {
        Python::initialize();
        Python::attach(|py| {
            let inorm = PyInstanceNorm1d::new(py, 2, None).expect("in construction");
            // (N=1, C=2, L=4)
            let data = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
            let input = make_tensor(data, &[1, 2, 4]);
            let out = inorm.forward(py, &input).expect("forward");
            assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 2, 4]);
            let out_vec = out.tensor.to_vec().expect("out vec");
            assert!(out_vec.iter().any(|&v| v != 0.0), "must not be all zeros");
            assert!(out_vec.iter().all(|v| v.is_finite()));
            // Each channel is normalised over its own L dimension -> mean 0.
            assert!(approx_zero(&out_vec[0..4], 1e-3));
            assert!(approx_zero(&out_vec[4..8], 1e-3));
        });
    }

    // -----------------------------------------------------------------
    // Gradient flow tests: forward -> loss(sum) -> backward -> assert
    // gamma AND beta gradients are populated, non-zero, and numerically
    // correct against a finite-difference oracle.
    // -----------------------------------------------------------------

    #[test]
    fn batch_norm_training_mode_gamma_beta_gradients_are_correct() {
        Python::initialize();
        Python::attach(|py| {
            let mut bn = PyBatchNorm1d::new(py, 2, None, None, None).expect("bn construction");
            bn.train();
            let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let input = make_tensor(input_data.clone(), &[4, 2]);

            let out = bn.forward(py, &input).expect("forward");

            // NOTE on loss function: `loss = sum(out)` (this test's original
            // loss) makes training-mode grad_gamma IDENTICALLY ZERO as a pure
            // algebraic identity, independent of any implementation detail:
            // `grad_output` is then uniformly 1 across the batch per channel,
            // so `grad_gamma_c = sum_n(grad_output[n,c] * normalized[n,c])
            // = sum_n(normalized[n,c])`, and `normalized[n,c] =
            // (x[n,c] - batch_mean_c)/std_c` where `batch_mean_c` is exactly
            // `mean_n(x[n,c])` — so `sum_n(x[n,c] - batch_mean_c)
            // = sum_n(x[n,c]) - N*batch_mean_c = 0` exactly, for every
            // channel. Confirmed empirically (not just algebraically) via a
            // standalone finite-difference probe against the real,
            // unmodified `tenflowers_core::ops::batch_norm` kernel: FD
            // grad_gamma under `sum(out)` came back as
            // `[-5.960464e-5, -5.960464e-5]` (pure fp noise at the 1e-3
            // finite-difference step size), matching the tape's exact
            // `[0.0, 0.0]` output — this is a real degeneracy of the loss
            // choice, not a backward-pass bug.
            //
            // To exercise a genuinely non-zero, well-conditioned gamma
            // gradient, weight each sample by a distinct per-sample scalar
            // before summing: `loss = sum(out * per_sample_weight)` with
            // weight = [0.5, 1.0, 1.5, 2.0] (one per batch row, broadcast
            // over both channels). This makes `grad_output[n, c] =
            // weight[n]`, non-uniform in `n`, so the cancellation above no
            // longer applies. Built as REAL tape-tracked ops (an elementwise
            // `Mul` recorded via `record_and_link_binary`, then the existing
            // `tape_sum_to_scalar` reduction) so gradients flow through to
            // gamma/beta exactly as `.backward()` would compute them for any
            // other loss — not a change to the oracle math alone.
            let weight = make_tensor(vec![0.5, 1.0, 1.5, 2.0], &[4, 1]);
            let weighted_raw = tenflowers_core::ops::mul(&out.tensor, &weight.tensor)
                .expect("weighted mul must succeed");
            let weighted = PyTensor {
                tensor: Arc::new(weighted_raw),
                requires_grad: true,
                is_pinned: false,
            };
            crate::implicit_autograd::record_and_link_binary(
                crate::implicit_autograd::BinaryOpKind::Mul,
                &out,
                &weight,
                &weighted,
            )
            .expect("recording weighted mul must succeed");

            let loss = tape_sum_to_scalar(&weighted);
            run_backward(&loss).expect("backward must succeed");

            let gamma_id = bn.gamma_param.borrow(py).id();
            let beta_id = bn.beta_param.borrow(py).id();
            let grad_gamma = crate::implicit_autograd::get_grad_by_id(gamma_id)
                .expect("gamma grad must be populated")
                .tensor
                .to_vec()
                .expect("grad readable");
            let grad_beta = crate::implicit_autograd::get_grad_by_id(beta_id)
                .expect("beta grad must be populated")
                .tensor
                .to_vec()
                .expect("grad readable");

            assert_eq!(grad_gamma.len(), 2);
            assert_eq!(grad_beta.len(), 2);
            assert!(
                grad_gamma.iter().any(|&g| g != 0.0),
                "gamma grad must be non-zero: {grad_gamma:?}"
            );
            assert!(
                grad_beta.iter().any(|&g| g != 0.0),
                "beta grad must be non-zero: {grad_beta:?}"
            );
            assert!(grad_gamma.iter().all(|g| g.is_finite()));
            assert!(grad_beta.iter().all(|g| g.is_finite()));

            // beta's gradient for `loss = sum((gamma*normalized+beta) *
            // weight)` is:
            //   d(loss)/d(beta_c) = sum_n(weight[n] * d(out[n,c])/d(beta_c))
            //                     = sum_n(weight[n] * 1) = sum(weight)
            //                     = 0.5 + 1.0 + 1.5 + 2.0 = 5.0
            // — independent of gamma/beta/normalized's own values (so, unlike
            // the sum-of-squares alternative, this stays non-degenerate even
            // at the layer's fresh beta=0 initialisation), for every channel.
            assert_close(&grad_beta, &[5.0, 5.0], 1e-3, "batch_norm train grad_beta");

            // gamma's gradient closed form:
            //   d(loss)/d(gamma_c) = sum_n(weight[n] * normalized[n,c])
            // With input columns c0=[1,3,5,7] (mean=4) and c1=[2,4,6,8]
            // (mean=5), both have batch variance
            // mean((x-mean)^2) = (9+1+1+9)/4 = 5, so std = sqrt(5+1e-5), and
            // both channels normalise to the SAME values (only shifted by a
            // constant per-channel mean, which cancels):
            // normalized = [-3,-1,1,3] / std ~= [-1.341641, -0.447214,
            // 0.447214, 1.341641] for both channels. So:
            //   grad_gamma_c = 0.5*(-1.341641) + 1.0*(-0.447214)
            //                + 1.5*(0.447214) + 2.0*(1.341641)
            //                ~= 2.236068
            // for both channels (matches the FD cross-check below and the
            // standalone empirical probe's `[2.2361279, 2.2361279]`).
            let std = (5.0f32 + 1e-5).sqrt();
            let normalized = [-3.0f32, -1.0, 1.0, 3.0].map(|v| v / std);
            let weights = [0.5f32, 1.0, 1.5, 2.0];
            let expected_gamma: f32 = normalized
                .iter()
                .zip(weights.iter())
                .map(|(&n, &w)| n * w)
                .sum();
            assert_close(
                &grad_gamma,
                &[expected_gamma, expected_gamma],
                1e-2,
                "batch_norm train grad_gamma vs closed-form oracle",
            );

            // gamma's gradient must ALSO match a finite-difference oracle
            // that re-runs the SAME real forward kernel this layer uses (and
            // the SAME weighted-sum loss), so this test cannot pass on a
            // fudged/oracle-mismatched formula shared by both the closed
            // form and the tape-based computation above.
            let gamma_vals = bn
                .gamma_param
                .borrow(py)
                .to_tensor()
                .expect("to_tensor")
                .tensor
                .to_vec()
                .expect("readable");
            let beta_vals = bn
                .beta_param
                .borrow(py)
                .to_tensor()
                .expect("to_tensor")
                .tensor
                .to_vec()
                .expect("readable");
            // batch_norm hard-requires exactly 4D (NCHW) input; the native
            // [4, 2] input must be reshaped up to [4, 2, 1, 1] before calling
            // the op directly here (mirroring `tape_reshape_up`'s own
            // reshape call, see that function above, but without tape
            // recording — this closure only computes a raw scalar loss for
            // finite-differencing).
            let input_arc_4d = Arc::new(
                tenflowers_core::ops::reshape(
                    &Tensor::<f32>::from_vec(input_data.clone(), &[4, 2]).expect("input"),
                    &[4, 2, 1, 1],
                )
                .expect("reshape to 4D"),
            );
            let weight_vals = [0.5f32, 1.0, 1.5, 2.0];
            let loss_fn = |gamma: &[f32], beta: &[f32]| -> f32 {
                let gamma_t = Tensor::from_vec(gamma.to_vec(), &[2]).expect("gamma");
                let beta_t = Tensor::from_vec(beta.to_vec(), &[2]).expect("beta");
                let running_mean = Tensor::<f32>::zeros(&[2]);
                let running_var = Tensor::<f32>::ones(&[2]);
                let out = tenflowers_core::ops::batch_norm(
                    &input_arc_4d, &gamma_t, &beta_t, &running_mean, &running_var, 1e-5, true,
                )
                .expect("batch_norm");
                let out_vec = out.to_vec().expect("readable");
                // out_vec is NCHW-flattened (N=4, C=2, H=1, W=1): index
                // n*2 + c for (n, c).
                (0..4)
                    .flat_map(|n| (0..2).map(move |c| (n, c)))
                    .map(|(n, c)| out_vec[n * 2 + c] * weight_vals[n])
                    .sum()
            };
            let fd_grad_gamma =
                finite_difference_grad(&gamma_vals, 1e-3, |g| loss_fn(g, &beta_vals));
            assert_close(
                &grad_gamma,
                &fd_grad_gamma,
                5e-2,
                "batch_norm train grad_gamma vs finite-difference",
            );
        });
    }

    #[test]
    fn batch_norm_eval_mode_gamma_beta_gradients_are_correct() {
        Python::initialize();
        Python::attach(|py| {
            let mut bn = PyBatchNorm1d::new(py, 2, None, None, None).expect("bn construction");
            // Give eval mode non-trivial running statistics to normalize
            // against, distinct from the freshly-initialised (0, 1) prior,
            // so this test cannot spuriously pass via a degenerate default.
            bn.running_mean = Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("running_mean");
            bn.running_var = Tensor::from_vec(vec![4.0, 9.0], &[2]).expect("running_var");
            bn.eval();

            let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let input = make_tensor(input_data.clone(), &[4, 2]);

            let out = bn.forward(py, &input).expect("forward");
            let loss = tape_sum_to_scalar(&out);
            run_backward(&loss).expect("backward must succeed");

            let gamma_id = bn.gamma_param.borrow(py).id();
            let beta_id = bn.beta_param.borrow(py).id();
            let grad_gamma = crate::implicit_autograd::get_grad_by_id(gamma_id)
                .expect("gamma grad must be populated")
                .tensor
                .to_vec()
                .expect("grad readable");
            let grad_beta = crate::implicit_autograd::get_grad_by_id(beta_id)
                .expect("beta grad must be populated")
                .tensor
                .to_vec()
                .expect("grad readable");

            assert_eq!(grad_gamma.len(), 2);
            assert_eq!(grad_beta.len(), 2);
            assert!(
                grad_gamma.iter().any(|&g| g != 0.0),
                "gamma grad must be non-zero: {grad_gamma:?}"
            );
            assert!(
                grad_beta.iter().any(|&g| g != 0.0),
                "beta grad must be non-zero: {grad_beta:?}"
            );
            assert!(grad_gamma.iter().all(|g| g.is_finite()));
            assert!(grad_beta.iter().all(|g| g.is_finite()));
            assert_close(&grad_beta, &[4.0, 4.0], 1e-3, "batch_norm eval grad_beta");

            // Eval-mode gamma gradient oracle:
            // d(sum(gamma*(x-running_mean)/std+beta))/d(gamma) =
            // sum((x - running_mean) / std) over the batch, per channel —
            // exactly the eval-mode formula documented on `batch_norm_backward`.
            let std0 = (4.0f32 + 1e-5).sqrt();
            let std1 = (9.0f32 + 1e-5).sqrt();
            let expected_gamma_c0: f32 = [1.0, 3.0, 5.0, 7.0].iter().map(|&x| (x - 1.0) / std0).sum();
            let expected_gamma_c1: f32 = [2.0, 4.0, 6.0, 8.0].iter().map(|&x| (x - 2.0) / std1).sum();
            assert_close(
                &grad_gamma,
                &[expected_gamma_c0, expected_gamma_c1],
                1e-2,
                "batch_norm eval grad_gamma vs closed-form oracle",
            );

            // Cross-check the closed-form oracle above against an
            // independent finite-difference oracle over the SAME real eval
            // forward kernel, so a coincidental agreement between two
            // hand-derived formulas cannot mask a shared mistake.
            let gamma_vals = bn
                .gamma_param
                .borrow(py)
                .to_tensor()
                .expect("to_tensor")
                .tensor
                .to_vec()
                .expect("readable");
            let beta_vals = bn
                .beta_param
                .borrow(py)
                .to_tensor()
                .expect("to_tensor")
                .tensor
                .to_vec()
                .expect("readable");
            // batch_norm hard-requires exactly 4D (NCHW) input; the native
            // [4, 2] input must be reshaped up to [4, 2, 1, 1] before calling
            // the op directly here (mirroring `tape_reshape_up`'s own
            // reshape call, see that function above, but without tape
            // recording — this closure only computes a raw scalar loss for
            // finite-differencing).
            let input_arc = Arc::new(
                tenflowers_core::ops::reshape(
                    &Tensor::<f32>::from_vec(input_data, &[4, 2]).expect("input"),
                    &[4, 2, 1, 1],
                )
                .expect("reshape to 4D"),
            );
            let running_mean = bn.running_mean.clone();
            let running_var = bn.running_var.clone();
            let loss_fn = |gamma: &[f32], beta: &[f32]| -> f32 {
                let gamma_t = Tensor::from_vec(gamma.to_vec(), &[2]).expect("gamma");
                let beta_t = Tensor::from_vec(beta.to_vec(), &[2]).expect("beta");
                let out = tenflowers_core::ops::batch_norm(
                    &input_arc, &gamma_t, &beta_t, &running_mean, &running_var, 1e-5, false,
                )
                .expect("batch_norm");
                out.to_vec().expect("readable").iter().sum()
            };
            let fd_grad_gamma =
                finite_difference_grad(&gamma_vals, 1e-3, |g| loss_fn(g, &beta_vals));
            assert_close(
                &grad_gamma,
                &fd_grad_gamma,
                5e-2,
                "batch_norm eval grad_gamma vs finite-difference",
            );
        });
    }

    #[test]
    fn layer_norm_gamma_beta_gradients_are_correct() {
        Python::initialize();
        Python::attach(|py| {
            let ln = PyLayerNorm::new(py, vec![3], None).expect("ln construction");
            let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 9.0];
            let input = make_tensor(input_data.clone(), &[2, 3]);

            let out = ln.forward(py, &input).expect("forward");
            let loss = tape_sum_to_scalar(&out);
            run_backward(&loss).expect("backward must succeed");

            let gamma_id = ln.gamma_param.borrow(py).id();
            let beta_id = ln.beta_param.borrow(py).id();
            let grad_gamma = crate::implicit_autograd::get_grad_by_id(gamma_id)
                .expect("gamma grad must be populated")
                .tensor
                .to_vec()
                .expect("grad readable");
            let grad_beta = crate::implicit_autograd::get_grad_by_id(beta_id)
                .expect("beta grad must be populated")
                .tensor
                .to_vec()
                .expect("grad readable");

            assert_eq!(grad_gamma.len(), 3);
            assert_eq!(grad_beta.len(), 3);
            assert!(grad_gamma.iter().any(|&g| g != 0.0), "gamma grad must be non-zero");
            assert!(grad_beta.iter().all(|v| v.is_finite()));
            assert!(grad_gamma.iter().all(|v| v.is_finite()));
            // d(sum(gamma*normalized+beta))/d(beta) = row count = 2 for every element.
            assert_close(&grad_beta, &[2.0, 2.0, 2.0], 1e-3, "layer_norm grad_beta");

            let gamma_vals = ln
                .gamma_param
                .borrow(py)
                .to_tensor()
                .expect("to_tensor")
                .tensor
                .to_vec()
                .expect("readable");
            let beta_vals = ln
                .beta_param
                .borrow(py)
                .to_tensor()
                .expect("to_tensor")
                .tensor
                .to_vec()
                .expect("readable");
            let input_arc = Arc::new(Tensor::<f32>::from_vec(input_data, &[2, 3]).expect("input"));
            let loss_fn = |gamma: &[f32], beta: &[f32]| -> f32 {
                let gamma_t = Tensor::from_vec(gamma.to_vec(), &[3]).expect("gamma");
                let beta_t = Tensor::from_vec(beta.to_vec(), &[3]).expect("beta");
                let out =
                    tenflowers_core::ops::layer_norm(&input_arc, &gamma_t, &beta_t, &[3], 1e-5)
                        .expect("layer_norm");
                out.to_vec().expect("readable").iter().sum()
            };
            let fd_grad_gamma =
                finite_difference_grad(&gamma_vals, 1e-3, |g| loss_fn(g, &beta_vals));
            assert_close(
                &grad_gamma,
                &fd_grad_gamma,
                5e-2,
                "layer_norm grad_gamma vs finite-difference",
            );
        });
    }

    #[test]
    fn group_norm_gamma_beta_gradients_are_correct() {
        Python::initialize();
        Python::attach(|py| {
            let gn = PyGroupNorm::new(py, 2, 4, None).expect("gn construction");
            // (N=1, C=4, L=2)
            let input_data: Vec<f32> = vec![1.0, 5.0, 2.0, 3.0, 8.0, 1.0, 4.0, 6.0];
            let input = make_tensor(input_data.clone(), &[1, 4, 2]);

            let out = gn.forward(py, &input).expect("forward");
            let loss = tape_sum_to_scalar(&out);
            run_backward(&loss).expect("backward must succeed");

            let gamma_id = gn.gamma_param.borrow(py).id();
            let beta_id = gn.beta_param.borrow(py).id();
            let grad_gamma = crate::implicit_autograd::get_grad_by_id(gamma_id)
                .expect("gamma grad must be populated")
                .tensor
                .to_vec()
                .expect("grad readable");
            let grad_beta = crate::implicit_autograd::get_grad_by_id(beta_id)
                .expect("beta grad must be populated")
                .tensor
                .to_vec()
                .expect("grad readable");

            assert_eq!(grad_gamma.len(), 4);
            assert_eq!(grad_beta.len(), 4);
            assert!(grad_gamma.iter().any(|&g| g != 0.0), "gamma grad must be non-zero");
            assert!(grad_beta.iter().all(|v| v.is_finite()));
            assert!(grad_gamma.iter().all(|v| v.is_finite()));
            // d(sum(...))/d(beta[c]) = number of positions beta[c] is added
            // at = N * L = 1 * 2 = 2, for every channel.
            assert_close(&grad_beta, &[2.0, 2.0, 2.0, 2.0], 1e-3, "group_norm grad_beta");

            let gamma_vals = gn
                .gamma_param
                .borrow(py)
                .to_tensor()
                .expect("to_tensor")
                .tensor
                .to_vec()
                .expect("readable");
            let beta_vals = gn
                .beta_param
                .borrow(py)
                .to_tensor()
                .expect("to_tensor")
                .tensor
                .to_vec()
                .expect("readable");
            // Finite-difference oracle re-runs the exact same reshape-to-4D
            // + group_norm forward path this layer's forward() uses.
            let input_arc_4d =
                Arc::new(Tensor::<f32>::from_vec(input_data, &[1, 4, 2, 1]).expect("input"));
            let loss_fn = |gamma: &[f32], beta: &[f32]| -> f32 {
                let gamma_t = Tensor::from_vec(gamma.to_vec(), &[4]).expect("gamma");
                let beta_t = Tensor::from_vec(beta.to_vec(), &[4]).expect("beta");
                let out =
                    tenflowers_core::ops::group_norm(&input_arc_4d, &gamma_t, &beta_t, 2, 1e-5)
                        .expect("group_norm");
                out.to_vec().expect("readable").iter().sum()
            };
            let fd_grad_gamma =
                finite_difference_grad(&gamma_vals, 1e-3, |g| loss_fn(g, &beta_vals));
            assert_close(
                &grad_gamma,
                &fd_grad_gamma,
                5e-2,
                "group_norm grad_gamma vs finite-difference",
            );
        });
    }

    #[test]
    fn instance_norm_gamma_beta_gradients_are_correct() {
        Python::initialize();
        Python::attach(|py| {
            let inorm = PyInstanceNorm1d::new(py, 2, None).expect("in construction");
            // (N=1, C=2, L=4)
            let input_data = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
            let input = make_tensor(input_data.clone(), &[1, 2, 4]);

            let out = inorm.forward(py, &input).expect("forward");
            let loss = tape_sum_to_scalar(&out);
            run_backward(&loss).expect("backward must succeed");

            let gamma_id = inorm.gamma_param.borrow(py).id();
            let beta_id = inorm.beta_param.borrow(py).id();
            let grad_gamma = crate::implicit_autograd::get_grad_by_id(gamma_id)
                .expect("gamma grad must be populated")
                .tensor
                .to_vec()
                .expect("grad readable");
            let grad_beta = crate::implicit_autograd::get_grad_by_id(beta_id)
                .expect("beta grad must be populated")
                .tensor
                .to_vec()
                .expect("grad readable");

            assert_eq!(grad_gamma.len(), 2);
            assert_eq!(grad_beta.len(), 2);
            assert!(grad_gamma.iter().any(|&g| g != 0.0), "gamma grad must be non-zero");
            assert!(grad_beta.iter().all(|v| v.is_finite()));
            assert!(grad_gamma.iter().all(|v| v.is_finite()));
            // d(sum(...))/d(beta[c]) = N * L = 1 * 4 = 4, for every channel.
            assert_close(&grad_beta, &[4.0, 4.0], 1e-3, "instance_norm grad_beta");

            let gamma_vals = inorm
                .gamma_param
                .borrow(py)
                .to_tensor()
                .expect("to_tensor")
                .tensor
                .to_vec()
                .expect("readable");
            let beta_vals = inorm
                .beta_param
                .borrow(py)
                .to_tensor()
                .expect("to_tensor")
                .tensor
                .to_vec()
                .expect("readable");
            let input_arc_4d =
                Arc::new(Tensor::<f32>::from_vec(input_data, &[1, 2, 4, 1]).expect("input"));
            let loss_fn = |gamma: &[f32], beta: &[f32]| -> f32 {
                let gamma_t = Tensor::from_vec(gamma.to_vec(), &[2]).expect("gamma");
                let beta_t = Tensor::from_vec(beta.to_vec(), &[2]).expect("beta");
                let out =
                    tenflowers_core::ops::group_norm(&input_arc_4d, &gamma_t, &beta_t, 2, 1e-5)
                        .expect("group_norm (instance_norm equivalent)");
                out.to_vec().expect("readable").iter().sum()
            };
            let fd_grad_gamma =
                finite_difference_grad(&gamma_vals, 1e-3, |g| loss_fn(g, &beta_vals));
            assert_close(
                &grad_gamma,
                &fd_grad_gamma,
                5e-2,
                "instance_norm grad_gamma vs finite-difference",
            );
        });
    }

    #[test]
    fn parameters_returns_gamma_and_beta_with_matching_identity() {
        Python::initialize();
        Python::attach(|py| {
            let bn = PyBatchNorm1d::new(py, 3, None, None, None).expect("bn construction");
            let params = bn.parameters(py);
            assert_eq!(params.len(), 2);
            assert_eq!(params[0].borrow(py).id(), bn.gamma_param.borrow(py).id());
            assert_eq!(params[1].borrow(py).id(), bn.beta_param.borrow(py).id());

            let ln = PyLayerNorm::new(py, vec![4], None).expect("ln construction");
            let ln_params = ln.parameters(py);
            assert_eq!(ln_params.len(), 2);
            assert_eq!(ln_params[0].borrow(py).id(), ln.gamma_param.borrow(py).id());
            assert_eq!(ln_params[1].borrow(py).id(), ln.beta_param.borrow(py).id());

            let gn = PyGroupNorm::new(py, 2, 4, None).expect("gn construction");
            let gn_params = gn.parameters(py);
            assert_eq!(gn_params.len(), 2);
            assert_eq!(gn_params[0].borrow(py).id(), gn.gamma_param.borrow(py).id());
            assert_eq!(gn_params[1].borrow(py).id(), gn.beta_param.borrow(py).id());

            let inorm = PyInstanceNorm1d::new(py, 3, None).expect("in construction");
            let in_params = inorm.parameters(py);
            assert_eq!(in_params.len(), 2);
            assert_eq!(in_params[0].borrow(py).id(), inorm.gamma_param.borrow(py).id());
            assert_eq!(in_params[1].borrow(py).id(), inorm.beta_param.borrow(py).id());
        });
    }
}
