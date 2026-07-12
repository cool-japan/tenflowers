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
        vec![
            self.gamma_param.clone_ref(py),
            self.beta_param.clone_ref(py),
        ]
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
    pub fn load_state_dict(
        &mut self,
        py: Python<'_>,
        state_dict: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
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

        let batch_mean =
            tenflowers_core::ops::mean(input_4d.tensor.as_ref(), Some(&reduce_axes), true)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("BatchNorm1d stats update failed: {e}"))
                })?;
        let centered =
            tenflowers_core::ops::sub(input_4d.tensor.as_ref(), &batch_mean).map_err(|e| {
                PyRuntimeError::new_err(format!("BatchNorm1d stats update failed: {e}"))
            })?;
        let squared = tenflowers_core::ops::mul(&centered, &centered).map_err(|e| {
            PyRuntimeError::new_err(format!("BatchNorm1d stats update failed: {e}"))
        })?;
        let batch_var =
            tenflowers_core::ops::mean(&squared, Some(&reduce_axes), true).map_err(|e| {
                PyRuntimeError::new_err(format!("BatchNorm1d stats update failed: {e}"))
            })?;

        let batch_mean_flat =
            tenflowers_core::ops::reshape(&batch_mean, &[channels]).map_err(|e| {
                PyRuntimeError::new_err(format!("BatchNorm1d stats update failed: {e}"))
            })?;
        let batch_var_flat =
            tenflowers_core::ops::reshape(&batch_var, &[channels]).map_err(|e| {
                PyRuntimeError::new_err(format!("BatchNorm1d stats update failed: {e}"))
            })?;

        let batch_size: usize = reduce_axes.iter().map(|&a| dims[a as usize]).product();
        // Unbiased variance estimate (N / (N - 1)) for the running-var update,
        // matching standard BatchNorm training semantics; falls back to the
        // biased estimate when there is exactly one element per channel (N-1
        // would be zero).
        let unbiased_var_flat = if batch_size > 1 {
            let scale = batch_size as f32 / (batch_size as f32 - 1.0);
            batch_var_flat.multiply_scalar(scale).map_err(|e| {
                PyRuntimeError::new_err(format!("BatchNorm1d stats update failed: {e}"))
            })?
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
            .map_err(|e| {
                PyRuntimeError::new_err(format!("BatchNorm1d stats update failed: {e}"))
            })?;
        let new_running_var = self
            .running_var
            .multiply_scalar(1.0 - momentum)
            .and_then(|scaled_old| {
                unbiased_var_flat
                    .multiply_scalar(momentum)
                    .and_then(|scaled_new| tenflowers_core::ops::add(&scaled_old, &scaled_new))
            })
            .map_err(|e| {
                PyRuntimeError::new_err(format!("BatchNorm1d stats update failed: {e}"))
            })?;

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
        vec![
            self.gamma_param.clone_ref(py),
            self.beta_param.clone_ref(py),
        ]
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
    pub fn load_state_dict(
        &mut self,
        py: Python<'_>,
        state_dict: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
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
        vec![
            self.gamma_param.clone_ref(py),
            self.beta_param.clone_ref(py),
        ]
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
    pub fn load_state_dict(
        &mut self,
        py: Python<'_>,
        state_dict: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
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
        vec![
            self.gamma_param.clone_ref(py),
            self.beta_param.clone_ref(py),
        ]
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
    pub fn load_state_dict(
        &mut self,
        py: Python<'_>,
        state_dict: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
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
mod tests;
