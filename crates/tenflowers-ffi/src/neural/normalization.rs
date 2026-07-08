//! Normalization layers module for TenfloweRS FFI
//!
//! This module provides comprehensive normalization layer implementations including
//! BatchNorm, LayerNorm, GroupNorm, and InstanceNorm for neural network training.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tenflowers_core::Tensor;

/// Batch Normalization layer
///
/// Normalizes the input across the batch dimension, commonly used in CNNs.
/// Maintains running statistics for inference and learnable affine parameters.
#[pyclass(name = "BatchNorm1d")]
#[derive(Debug, Clone)]
pub struct PyBatchNorm1d {
    /// Number of features (channels)
    pub num_features: usize,
    /// Small constant for numerical stability
    pub eps: f32,
    /// Momentum for running statistics
    pub momentum: f32,
    /// Whether to use affine transformation (learnable scale and bias)
    pub affine: bool,
    /// Whether to track running statistics
    pub track_running_stats: bool,
    /// Learnable scale parameter (gamma)
    pub weight: Option<Tensor<f32>>,
    /// Learnable bias parameter (beta)
    pub bias: Option<Tensor<f32>>,
    /// Running mean for inference
    pub running_mean: Option<Tensor<f32>>,
    /// Running variance for inference
    pub running_var: Option<Tensor<f32>>,
    /// Number of batches tracked
    pub num_batches_tracked: usize,
    /// Training mode flag
    pub training: bool,
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
    /// * `affine` - Whether to learn affine parameters (default: true)
    /// * `track_running_stats` - Whether to track running statistics (default: true)
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5, momentum=0.1, affine=true, track_running_stats=true))]
    pub fn new(
        num_features: usize,
        eps: Option<f32>,
        momentum: Option<f32>,
        affine: Option<bool>,
        track_running_stats: Option<bool>,
    ) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);
        let affine = affine.unwrap_or(true);
        let track_running_stats = track_running_stats.unwrap_or(true);

        if num_features == 0 {
            return Err(PyValueError::new_err("num_features must be positive"));
        }

        let weight = if affine {
            Some(Tensor::ones(&[num_features]))
        } else {
            None
        };

        let bias = if affine {
            Some(Tensor::zeros(&[num_features]))
        } else {
            None
        };

        let running_mean = if track_running_stats {
            Some(Tensor::zeros(&[num_features]))
        } else {
            None
        };

        let running_var = if track_running_stats {
            Some(Tensor::ones(&[num_features]))
        } else {
            None
        };

        Ok(PyBatchNorm1d {
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            weight,
            bias,
            running_mean,
            running_var,
            num_batches_tracked: 0,
            training: true,
        })
    }

    /// Forward pass through the batch normalization layer
    pub fn forward(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
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

        // The batch_norm op normalises 4D NCHW tensors. Reshape (N, C) -> (N, C, 1, 1)
        // and (N, C, L) -> (N, C, L, 1); both keep per-channel statistics intact.
        let orig_dims: Vec<usize> = input_shape.dims().to_vec();
        let mut shape_4d = orig_dims.clone();
        while shape_4d.len() < 4 {
            shape_4d.push(1);
        }

        let input_4d = tenflowers_core::ops::reshape(input.tensor.as_ref(), &shape_4d)
            .map_err(|e| PyRuntimeError::new_err(format!("BatchNorm1d reshape failed: {e}")))?;

        // gamma/beta default to the identity affine when not learned; running stats
        // default to the standard-normal prior when not tracked.
        let gamma = match &self.weight {
            Some(w) => w.clone(),
            None => Tensor::ones(&[self.num_features]),
        };
        let beta = match &self.bias {
            Some(b) => b.clone(),
            None => Tensor::zeros(&[self.num_features]),
        };
        let running_mean = match &self.running_mean {
            Some(m) => m.clone(),
            None => Tensor::zeros(&[self.num_features]),
        };
        let running_var = match &self.running_var {
            Some(v) => v.clone(),
            None => Tensor::ones(&[self.num_features]),
        };

        // Without tracked running stats, normalisation always uses batch statistics
        // (the training path), matching standard BatchNorm semantics.
        let use_batch_stats = self.training || !self.track_running_stats;

        let normalized = tenflowers_core::ops::batch_norm(
            &input_4d,
            &gamma,
            &beta,
            &running_mean,
            &running_var,
            self.eps,
            use_batch_stats,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("BatchNorm1d forward failed: {e}")))?;

        if use_batch_stats && self.track_running_stats {
            self.num_batches_tracked += 1;
        }

        let output = tenflowers_core::ops::reshape(&normalized, &orig_dims)
            .map_err(|e| PyRuntimeError::new_err(format!("BatchNorm1d reshape failed: {e}")))?;

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        })
    }

    /// Set the layer to training mode
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set the layer to evaluation mode
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Reset running statistics
    pub fn reset_running_stats(&mut self) -> PyResult<()> {
        if self.track_running_stats {
            self.running_mean = Some(Tensor::zeros(&[self.num_features]));
            self.running_var = Some(Tensor::ones(&[self.num_features]));
            self.num_batches_tracked = 0;
        }
        Ok(())
    }

    /// Reset parameters
    pub fn reset_parameters(&mut self) -> PyResult<()> {
        self.reset_running_stats()?;
        if self.affine {
            self.weight = Some(Tensor::ones(&[self.num_features]));
            self.bias = Some(Tensor::zeros(&[self.num_features]));
        }
        Ok(())
    }

    /// Get layer state dict
    pub fn state_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);

        if let Some(ref weight) = self.weight {
            let weight_data: Vec<f32> = weight
                .to_vec()
                .map_err(|e| PyValueError::new_err(format!("Failed to convert weight: {}", e)))?;
            dict.set_item("weight", weight_data)?;
        }

        if let Some(ref bias) = self.bias {
            let bias_data: Vec<f32> = bias
                .to_vec()
                .map_err(|e| PyValueError::new_err(format!("Failed to convert bias: {}", e)))?;
            dict.set_item("bias", bias_data)?;
        }

        if let Some(ref running_mean) = self.running_mean {
            let mean_data: Vec<f32> = running_mean.to_vec().map_err(|e| {
                PyValueError::new_err(format!("Failed to convert running_mean: {}", e))
            })?;
            dict.set_item("running_mean", mean_data)?;
        }

        if let Some(ref running_var) = self.running_var {
            let var_data: Vec<f32> = running_var.to_vec().map_err(|e| {
                PyValueError::new_err(format!("Failed to convert running_var: {}", e))
            })?;
            dict.set_item("running_var", var_data)?;
        }

        dict.set_item("num_batches_tracked", self.num_batches_tracked)?;

        Ok(dict.into())
    }

    /// Load layer state dict
    pub fn load_state_dict(&mut self, state_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(weight) = state_dict.get_item("weight")? {
            let weight_vec: Vec<f32> = weight.extract()?;
            self.weight = Some(
                Tensor::from_vec(weight_vec, &[self.num_features])
                    .map_err(|e| PyValueError::new_err(format!("Failed to load weight: {}", e)))?,
            );
        }

        if let Some(bias) = state_dict.get_item("bias")? {
            let bias_vec: Vec<f32> = bias.extract()?;
            self.bias = Some(
                Tensor::from_vec(bias_vec, &[self.num_features])
                    .map_err(|e| PyValueError::new_err(format!("Failed to load bias: {}", e)))?,
            );
        }

        if let Some(running_mean) = state_dict.get_item("running_mean")? {
            let mean_vec: Vec<f32> = running_mean.extract()?;
            self.running_mean = Some(Tensor::from_vec(mean_vec, &[self.num_features]).map_err(
                |e| PyValueError::new_err(format!("Failed to load running_mean: {}", e)),
            )?);
        }

        if let Some(running_var) = state_dict.get_item("running_var")? {
            let var_vec: Vec<f32> = running_var.extract()?;
            self.running_var = Some(Tensor::from_vec(var_vec, &[self.num_features]).map_err(
                |e| PyValueError::new_err(format!("Failed to load running_var: {}", e)),
            )?);
        }

        if let Some(num_batches) = state_dict.get_item("num_batches_tracked")? {
            self.num_batches_tracked = num_batches.extract()?;
        }

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchNorm1d(num_features={}, eps={}, momentum={}, affine={}, track_running_stats={})",
            self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats
        )
    }
}

/// Layer Normalization layer
///
/// Normalizes the input across the feature dimension, commonly used in Transformers.
/// Applies normalization over the last D dimensions where D is the length of normalized_shape.
#[pyclass(name = "LayerNorm")]
#[derive(Debug, Clone)]
pub struct PyLayerNorm {
    /// Shape of normalized features
    pub normalized_shape: Vec<usize>,
    /// Small constant for numerical stability
    pub eps: f32,
    /// Whether to use learnable affine parameters
    pub elementwise_affine: bool,
    /// Learnable scale parameter (gamma)
    pub weight: Option<Tensor<f32>>,
    /// Learnable bias parameter (beta)
    pub bias: Option<Tensor<f32>>,
}

#[pymethods]
impl PyLayerNorm {
    /// Create a new LayerNorm layer
    ///
    /// # Arguments
    ///
    /// * `normalized_shape` - Input shape from an expected input of size
    /// * `eps` - Value added to denominator for numerical stability (default: 1e-5)
    /// * `elementwise_affine` - Whether to learn affine parameters (default: true)
    #[new]
    #[pyo3(signature = (normalized_shape, eps=1e-5, elementwise_affine=true))]
    pub fn new(
        normalized_shape: Vec<usize>,
        eps: Option<f32>,
        elementwise_affine: Option<bool>,
    ) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-5);
        let elementwise_affine = elementwise_affine.unwrap_or(true);

        if normalized_shape.is_empty() {
            return Err(PyValueError::new_err("normalized_shape must not be empty"));
        }

        let weight = if elementwise_affine {
            Some(Tensor::ones(&normalized_shape))
        } else {
            None
        };

        let bias = if elementwise_affine {
            Some(Tensor::zeros(&normalized_shape))
        } else {
            None
        };

        Ok(PyLayerNorm {
            normalized_shape,
            eps,
            elementwise_affine,
            weight,
            bias,
        })
    }

    /// Forward pass through the layer normalization layer
    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
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

        // gamma/beta default to the identity affine when not learned.
        let gamma = match &self.weight {
            Some(w) => w.clone(),
            None => Tensor::ones(&self.normalized_shape),
        };
        let beta = match &self.bias {
            Some(b) => b.clone(),
            None => Tensor::zeros(&self.normalized_shape),
        };

        match tenflowers_core::ops::layer_norm(
            input.tensor.as_ref(),
            &gamma,
            &beta,
            &self.normalized_shape,
            self.eps,
        ) {
            Ok(output) => Ok(PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "LayerNorm forward failed: {e}"
            ))),
        }
    }

    /// Reset parameters
    pub fn reset_parameters(&mut self) -> PyResult<()> {
        if self.elementwise_affine {
            self.weight = Some(Tensor::ones(&self.normalized_shape));
            self.bias = Some(Tensor::zeros(&self.normalized_shape));
        }
        Ok(())
    }

    /// Get layer state dict
    pub fn state_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);

        if let Some(ref weight) = self.weight {
            let weight_data: Vec<f32> = weight
                .to_vec()
                .map_err(|e| PyValueError::new_err(format!("Failed to convert weight: {}", e)))?;
            dict.set_item("weight", weight_data)?;
        }

        if let Some(ref bias) = self.bias {
            let bias_data: Vec<f32> = bias
                .to_vec()
                .map_err(|e| PyValueError::new_err(format!("Failed to convert bias: {}", e)))?;
            dict.set_item("bias", bias_data)?;
        }

        Ok(dict.into())
    }

    /// Load layer state dict
    pub fn load_state_dict(&mut self, state_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(weight) = state_dict.get_item("weight")? {
            let weight_vec: Vec<f32> = weight.extract()?;
            self.weight = Some(
                Tensor::from_vec(weight_vec, &self.normalized_shape)
                    .map_err(|e| PyValueError::new_err(format!("Failed to load weight: {}", e)))?,
            );
        }

        if let Some(bias) = state_dict.get_item("bias")? {
            let bias_vec: Vec<f32> = bias.extract()?;
            self.bias = Some(
                Tensor::from_vec(bias_vec, &self.normalized_shape)
                    .map_err(|e| PyValueError::new_err(format!("Failed to load bias: {}", e)))?,
            );
        }

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "LayerNorm(normalized_shape={:?}, eps={}, elementwise_affine={})",
            self.normalized_shape, self.eps, self.elementwise_affine
        )
    }
}

/// Group Normalization layer
///
/// Divides channels into groups and normalizes within each group.
/// Useful when batch size is small.
#[pyclass(name = "GroupNorm")]
#[derive(Debug, Clone)]
pub struct PyGroupNorm {
    /// Number of groups
    pub num_groups: usize,
    /// Number of channels
    pub num_channels: usize,
    /// Small constant for numerical stability
    pub eps: f32,
    /// Whether to use learnable affine parameters
    pub affine: bool,
    /// Learnable scale parameter (gamma)
    pub weight: Option<Tensor<f32>>,
    /// Learnable bias parameter (beta)
    pub bias: Option<Tensor<f32>>,
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
    /// * `affine` - Whether to learn affine parameters (default: true)
    #[new]
    #[pyo3(signature = (num_groups, num_channels, eps=1e-5, affine=true))]
    pub fn new(
        num_groups: usize,
        num_channels: usize,
        eps: Option<f32>,
        affine: Option<bool>,
    ) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-5);
        let affine = affine.unwrap_or(true);

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

        let weight = if affine {
            Some(Tensor::ones(&[num_channels]))
        } else {
            None
        };

        let bias = if affine {
            Some(Tensor::zeros(&[num_channels]))
        } else {
            None
        };

        Ok(PyGroupNorm {
            num_groups,
            num_channels,
            eps,
            affine,
            weight,
            bias,
        })
    }

    /// Forward pass through the group normalization layer
    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
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

        // gamma/beta default to the identity affine when not learned.
        let gamma = match &self.weight {
            Some(w) => w.clone(),
            None => Tensor::ones(&[self.num_channels]),
        };
        let beta = match &self.bias {
            Some(b) => b.clone(),
            None => Tensor::zeros(&[self.num_channels]),
        };

        match tenflowers_core::ops::group_norm(
            input.tensor.as_ref(),
            &gamma,
            &beta,
            self.num_groups,
            self.eps,
        ) {
            Ok(output) => Ok(PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "GroupNorm forward failed: {e}"
            ))),
        }
    }

    /// Reset parameters
    pub fn reset_parameters(&mut self) -> PyResult<()> {
        if self.affine {
            self.weight = Some(Tensor::ones(&[self.num_channels]));
            self.bias = Some(Tensor::zeros(&[self.num_channels]));
        }
        Ok(())
    }

    /// Get layer state dict
    pub fn state_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);

        if let Some(ref weight) = self.weight {
            let weight_data: Vec<f32> = weight
                .to_vec()
                .map_err(|e| PyValueError::new_err(format!("Failed to convert weight: {}", e)))?;
            dict.set_item("weight", weight_data)?;
        }

        if let Some(ref bias) = self.bias {
            let bias_data: Vec<f32> = bias
                .to_vec()
                .map_err(|e| PyValueError::new_err(format!("Failed to convert bias: {}", e)))?;
            dict.set_item("bias", bias_data)?;
        }

        Ok(dict.into())
    }

    /// Load layer state dict
    pub fn load_state_dict(&mut self, state_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(weight) = state_dict.get_item("weight")? {
            let weight_vec: Vec<f32> = weight.extract()?;
            self.weight = Some(
                Tensor::from_vec(weight_vec, &[self.num_channels])
                    .map_err(|e| PyValueError::new_err(format!("Failed to load weight: {}", e)))?,
            );
        }

        if let Some(bias) = state_dict.get_item("bias")? {
            let bias_vec: Vec<f32> = bias.extract()?;
            self.bias = Some(
                Tensor::from_vec(bias_vec, &[self.num_channels])
                    .map_err(|e| PyValueError::new_err(format!("Failed to load bias: {}", e)))?,
            );
        }

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "GroupNorm(num_groups={}, num_channels={}, eps={}, affine={})",
            self.num_groups, self.num_channels, self.eps, self.affine
        )
    }
}

/// Instance Normalization layer
///
/// Applies normalization over each channel in each data sample independently.
/// Commonly used in style transfer and GANs.
#[pyclass(name = "InstanceNorm1d")]
#[derive(Debug, Clone)]
pub struct PyInstanceNorm1d {
    /// Number of features (channels)
    pub num_features: usize,
    /// Small constant for numerical stability
    pub eps: f32,
    /// Momentum for running statistics
    pub momentum: f32,
    /// Whether to use learnable affine parameters
    pub affine: bool,
    /// Whether to track running statistics
    pub track_running_stats: bool,
    /// Learnable scale parameter (gamma)
    pub weight: Option<Tensor<f32>>,
    /// Learnable bias parameter (beta)
    pub bias: Option<Tensor<f32>>,
    /// Running mean for inference
    pub running_mean: Option<Tensor<f32>>,
    /// Running variance for inference
    pub running_var: Option<Tensor<f32>>,
}

#[pymethods]
impl PyInstanceNorm1d {
    /// Create a new InstanceNorm1d layer
    ///
    /// # Arguments
    ///
    /// * `num_features` - Number of features (channels) from an expected input of size (N, C, L)
    /// * `eps` - Value added to denominator for numerical stability (default: 1e-5)
    /// * `momentum` - Value used for running_mean and running_var computation (default: 0.1)
    /// * `affine` - Whether to learn affine parameters (default: false)
    /// * `track_running_stats` - Whether to track running statistics (default: false)
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5, momentum=0.1, affine=false, track_running_stats=false))]
    pub fn new(
        num_features: usize,
        eps: Option<f32>,
        momentum: Option<f32>,
        affine: Option<bool>,
        track_running_stats: Option<bool>,
    ) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);
        let affine = affine.unwrap_or(false);
        let track_running_stats = track_running_stats.unwrap_or(false);

        if num_features == 0 {
            return Err(PyValueError::new_err("num_features must be positive"));
        }

        let weight = if affine {
            Some(Tensor::ones(&[num_features]))
        } else {
            None
        };

        let bias = if affine {
            Some(Tensor::zeros(&[num_features]))
        } else {
            None
        };

        let running_mean = if track_running_stats {
            Some(Tensor::zeros(&[num_features]))
        } else {
            None
        };

        let running_var = if track_running_stats {
            Some(Tensor::ones(&[num_features]))
        } else {
            None
        };

        Ok(PyInstanceNorm1d {
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            weight,
            bias,
            running_mean,
            running_var,
        })
    }

    /// Forward pass through the instance normalization layer
    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
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

        // Instance norm is exactly group norm with one group per channel, so the
        // group_norm op is reused with num_groups == num_channels.
        let gamma = match &self.weight {
            Some(w) => w.clone(),
            None => Tensor::ones(&[self.num_features]),
        };
        let beta = match &self.bias {
            Some(b) => b.clone(),
            None => Tensor::zeros(&[self.num_features]),
        };

        match tenflowers_core::ops::group_norm(
            input.tensor.as_ref(),
            &gamma,
            &beta,
            self.num_features,
            self.eps,
        ) {
            Ok(output) => Ok(PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "InstanceNorm1d forward failed: {e}"
            ))),
        }
    }

    /// Reset parameters
    pub fn reset_parameters(&mut self) -> PyResult<()> {
        if self.affine {
            self.weight = Some(Tensor::ones(&[self.num_features]));
            self.bias = Some(Tensor::zeros(&[self.num_features]));
        }
        Ok(())
    }

    /// Get layer state dict
    pub fn state_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);

        if let Some(ref weight) = self.weight {
            let weight_data: Vec<f32> = weight
                .to_vec()
                .map_err(|e| PyValueError::new_err(format!("Failed to convert weight: {}", e)))?;
            dict.set_item("weight", weight_data)?;
        }

        if let Some(ref bias) = self.bias {
            let bias_data: Vec<f32> = bias
                .to_vec()
                .map_err(|e| PyValueError::new_err(format!("Failed to convert bias: {}", e)))?;
            dict.set_item("bias", bias_data)?;
        }

        Ok(dict.into())
    }

    /// Load layer state dict
    pub fn load_state_dict(&mut self, state_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(weight) = state_dict.get_item("weight")? {
            let weight_vec: Vec<f32> = weight.extract()?;
            self.weight = Some(
                Tensor::from_vec(weight_vec, &[self.num_features])
                    .map_err(|e| PyValueError::new_err(format!("Failed to load weight: {}", e)))?,
            );
        }

        if let Some(bias) = state_dict.get_item("bias")? {
            let bias_vec: Vec<f32> = bias.extract()?;
            self.bias = Some(
                Tensor::from_vec(bias_vec, &[self.num_features])
                    .map_err(|e| PyValueError::new_err(format!("Failed to load bias: {}", e)))?,
            );
        }

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "InstanceNorm1d(num_features={}, eps={}, momentum={}, affine={}, track_running_stats={})",
            self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats
        )
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

    fn approx_zero(values: &[f32], tol: f32) -> bool {
        let sum: f32 = values.iter().sum();
        sum.abs() < tol
    }

    #[test]
    fn batch_norm_normalizes_per_channel() {
        let mut bn = PyBatchNorm1d::new(3, None, None, None, None).expect("bn construction");
        // (N=4, C=3) with non-constant columns.
        let data: Vec<f32> = (1..=12).map(|v| v as f32).collect();
        let input = make_tensor(data, &[4, 3]);
        let out = bn.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![4, 3]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert!(out_vec.iter().any(|&v| v != 0.0), "must not be all zeros");
        assert!(out_vec.iter().all(|v| v.is_finite()));
        // Each channel (column) is mean-centred after training-mode BatchNorm.
        for c in 0..3 {
            let col: Vec<f32> = (0..4).map(|n| out_vec[n * 3 + c]).collect();
            assert!(approx_zero(&col, 1e-3), "channel {c} should have mean 0");
        }
    }

    #[test]
    fn batch_norm_rejects_4d_input() {
        let mut bn = PyBatchNorm1d::new(2, None, None, None, None).expect("bn construction");
        let input = make_tensor(vec![0.0; 16], &[2, 2, 2, 2]);
        assert!(bn.forward(&input).is_err());
    }

    #[test]
    fn layer_norm_centres_last_dim() {
        let ln = PyLayerNorm::new(vec![3], None, None).expect("ln construction");
        let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let out = ln.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![2, 3]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert!(out_vec.iter().any(|&v| v != 0.0), "must not be all zeros");
        assert!(out_vec.iter().all(|v| v.is_finite()));
        for row in 0..2 {
            let r: Vec<f32> = (0..3).map(|i| out_vec[row * 3 + i]).collect();
            assert!(approx_zero(&r, 1e-3), "row {row} should have mean 0");
        }
    }

    #[test]
    fn group_norm_forward_real_output() {
        let gn = PyGroupNorm::new(2, 4, None, None).expect("gn construction");
        // (N=1, C=4, L=2)
        let data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let input = make_tensor(data, &[1, 4, 2]);
        let out = gn.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 4, 2]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert!(out_vec.iter().any(|&v| v != 0.0), "must not be all zeros");
        assert!(out_vec.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn instance_norm_centres_each_channel() {
        let inorm = PyInstanceNorm1d::new(2, None, None, None, None).expect("in construction");
        // (N=1, C=2, L=4)
        let data = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let input = make_tensor(data, &[1, 2, 4]);
        let out = inorm.forward(&input).expect("forward");
        assert_eq!(out.tensor.shape().dims().to_vec(), vec![1, 2, 4]);
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert!(out_vec.iter().any(|&v| v != 0.0), "must not be all zeros");
        assert!(out_vec.iter().all(|v| v.is_finite()));
        // Each channel is normalised over its own L dimension -> mean 0.
        assert!(approx_zero(&out_vec[0..4], 1e-3));
        assert!(approx_zero(&out_vec[4..8], 1e-3));
    }
}
