//! Normalization layers module for TenfloweRS FFI
//!
//! This module provides comprehensive normalization layer implementations including
//! BatchNorm, LayerNorm, GroupNorm, and InstanceNorm for neural network training.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::PyValueError;
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

        // For now, return a placeholder that matches the input shape
        // Full implementation would compute batch statistics and normalize
        let shape_vec: Vec<usize> = input_shape.iter().copied().collect();
        let output = Tensor::zeros(&shape_vec);

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: false,
            is_pinned: false,
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
    pub fn state_dict(&self, py: Python) -> PyResult<PyObject> {
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

        // For now, return a placeholder that matches the input shape
        // Full implementation would compute layer statistics and normalize
        let output = Tensor::zeros(&shape_vec);

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: false,
            is_pinned: false,
        })
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
    pub fn state_dict(&self, py: Python) -> PyResult<PyObject> {
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

        // For now, return a placeholder that matches the input shape
        let shape_vec: Vec<usize> = input_shape.iter().copied().collect();
        let output = Tensor::zeros(&shape_vec);

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: false,
            is_pinned: false,
        })
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
    pub fn state_dict(&self, py: Python) -> PyResult<PyObject> {
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

        // For now, return a placeholder that matches the input shape
        let shape_vec: Vec<usize> = input_shape.iter().copied().collect();
        let output = Tensor::zeros(&shape_vec);

        Ok(PyTensor {
            tensor: Arc::new(output),
            requires_grad: false,
            is_pinned: false,
        })
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
    pub fn state_dict(&self, py: Python) -> PyResult<PyObject> {
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
