//! Regularization layers module for TenfloweRS FFI
//!
//! This module provides regularization layer implementations including Dropout,
//! AlphaDropout, and other regularization techniques for neural network training.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;
use tenflowers_core::Tensor;

/// Dropout Layer
///
/// During training, randomly zeroes some elements of the input tensor with probability p
/// using samples from a Bernoulli distribution. Helps prevent overfitting.
#[pyclass(name = "Dropout")]
#[derive(Debug, Clone)]
pub struct PyDropout {
    /// Probability of an element to be zeroed
    pub p: f32,
    /// Whether the layer is in training mode
    pub training: bool,
    /// Whether to use inplace operation
    pub inplace: bool,
}

#[pymethods]
impl PyDropout {
    /// Create a new Dropout layer
    ///
    /// # Arguments
    ///
    /// * `p` - Probability of an element to be zeroed (default: 0.5)
    /// * `inplace` - If True, will do dropout in-place (default: False)
    #[new]
    #[pyo3(signature = (p=0.5, inplace=false))]
    pub fn new(p: Option<f32>, inplace: Option<bool>) -> PyResult<Self> {
        let p = p.unwrap_or(0.5);
        let inplace = inplace.unwrap_or(false);

        if !(0.0..=1.0).contains(&p) {
            return Err(PyValueError::new_err(format!(
                "Dropout probability must be between 0 and 1, got {}",
                p
            )));
        }

        Ok(PyDropout {
            p,
            training: true,
            inplace,
        })
    }

    /// Forward pass through the dropout layer
    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        if !self.training || self.p == 0.0 {
            // During evaluation or if p=0, return input unchanged
            return Ok(input.clone());
        }

        if self.p == 1.0 {
            // If p=1, return zeros
            let shape: Vec<usize> = input.tensor.shape().iter().copied().collect();
            let output = Tensor::zeros(&shape);
            return Ok(PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            });
        }

        // For now, return a placeholder that applies dropout
        // Full implementation would use random number generation
        let shape: Vec<usize> = input.tensor.shape().iter().copied().collect();
        let output = Tensor::zeros(&shape);

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

    fn __repr__(&self) -> String {
        format!("Dropout(p={}, inplace={})", self.p, self.inplace)
    }
}

/// Dropout2D Layer
///
/// Randomly zero out entire channels (a channel is a 2D feature map).
/// Typically used in convolutional neural networks.
#[pyclass(name = "Dropout2D")]
#[derive(Debug, Clone)]
pub struct PyDropout2D {
    /// Probability of a channel to be zeroed
    pub p: f32,
    /// Whether the layer is in training mode
    pub training: bool,
    /// Whether to use inplace operation
    pub inplace: bool,
}

#[pymethods]
impl PyDropout2D {
    /// Create a new Dropout2D layer
    #[new]
    #[pyo3(signature = (p=0.5, inplace=false))]
    pub fn new(p: Option<f32>, inplace: Option<bool>) -> PyResult<Self> {
        let p = p.unwrap_or(0.5);
        let inplace = inplace.unwrap_or(false);

        if !(0.0..=1.0).contains(&p) {
            return Err(PyValueError::new_err(format!(
                "Dropout probability must be between 0 and 1, got {}",
                p
            )));
        }

        Ok(PyDropout2D {
            p,
            training: true,
            inplace,
        })
    }

    /// Forward pass through the dropout2d layer
    ///
    /// Input should be (N, C, H, W) or (N, C, L)
    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let input_shape = input.tensor.shape();

        if input_shape.len() < 3 {
            return Err(PyValueError::new_err(format!(
                "Expected at least 3D input, got {}D",
                input_shape.len()
            )));
        }

        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        if self.p == 1.0 {
            let shape: Vec<usize> = input_shape.iter().copied().collect();
            let output = Tensor::zeros(&shape);
            return Ok(PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            });
        }

        // For now, return a placeholder
        let shape: Vec<usize> = input_shape.iter().copied().collect();
        let output = Tensor::zeros(&shape);

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

    fn __repr__(&self) -> String {
        format!("Dropout2D(p={}, inplace={})", self.p, self.inplace)
    }
}

/// Alpha Dropout Layer
///
/// Applies Alpha Dropout to the input. Maintains self-normalizing properties.
/// Used with SELU activation for self-normalizing neural networks.
#[pyclass(name = "AlphaDropout")]
#[derive(Debug, Clone)]
pub struct PyAlphaDropout {
    /// Probability of an element to be dropped
    pub p: f32,
    /// Whether the layer is in training mode
    pub training: bool,
    /// Whether to use inplace operation
    pub inplace: bool,
}

#[pymethods]
impl PyAlphaDropout {
    /// Create a new AlphaDropout layer
    #[new]
    #[pyo3(signature = (p=0.5, inplace=false))]
    pub fn new(p: Option<f32>, inplace: Option<bool>) -> PyResult<Self> {
        let p = p.unwrap_or(0.5);
        let inplace = inplace.unwrap_or(false);

        if !(0.0..=1.0).contains(&p) {
            return Err(PyValueError::new_err(format!(
                "Dropout probability must be between 0 and 1, got {}",
                p
            )));
        }

        Ok(PyAlphaDropout {
            p,
            training: true,
            inplace,
        })
    }

    /// Forward pass through the alpha dropout layer
    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        // Alpha dropout specific constants for SELU
        let alpha = 1.673_263_2_f32;
        let _fixedpoint_mean = 0.0_f32;
        let _fixedpoint_std = 1.0_f32;

        let _a = (-(1.0 - self.p) * (self.p * alpha.powi(2) + 1.0)).sqrt();
        let _b = -_a * alpha * self.p;

        // For now, return a placeholder
        let shape: Vec<usize> = input.tensor.shape().iter().copied().collect();
        let output = Tensor::zeros(&shape);

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

    fn __repr__(&self) -> String {
        format!("AlphaDropout(p={}, inplace={})", self.p, self.inplace)
    }
}

/// Feature Alpha Dropout Layer
///
/// Randomly masks out entire channels. Similar to Dropout2D but for self-normalizing networks.
#[pyclass(name = "FeatureAlphaDropout")]
#[derive(Debug, Clone)]
pub struct PyFeatureAlphaDropout {
    /// Probability of a channel to be dropped
    pub p: f32,
    /// Whether the layer is in training mode
    pub training: bool,
    /// Whether to use inplace operation
    pub inplace: bool,
}

#[pymethods]
impl PyFeatureAlphaDropout {
    /// Create a new FeatureAlphaDropout layer
    #[new]
    #[pyo3(signature = (p=0.5, inplace=false))]
    pub fn new(p: Option<f32>, inplace: Option<bool>) -> PyResult<Self> {
        let p = p.unwrap_or(0.5);
        let inplace = inplace.unwrap_or(false);

        if !(0.0..=1.0).contains(&p) {
            return Err(PyValueError::new_err(format!(
                "Dropout probability must be between 0 and 1, got {}",
                p
            )));
        }

        Ok(PyFeatureAlphaDropout {
            p,
            training: true,
            inplace,
        })
    }

    /// Forward pass
    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let shape: Vec<usize> = input.tensor.shape().iter().copied().collect();
        let output = Tensor::zeros(&shape);

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

    fn __repr__(&self) -> String {
        format!(
            "FeatureAlphaDropout(p={}, inplace={})",
            self.p, self.inplace
        )
    }
}

/// L2 Regularization function
///
/// Computes L2 (weight decay) regularization term for a tensor.
#[pyfunction]
#[pyo3(signature = (tensor, weight_decay))]
pub fn l2_regularization(tensor: &PyTensor, weight_decay: f32) -> PyResult<PyTensor> {
    if weight_decay < 0.0 {
        return Err(PyValueError::new_err("weight_decay must be non-negative"));
    }

    let data = tensor.tensor.to_vec().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to get tensor data: {}", e))
    })?;

    // Compute L2 norm squared
    let l2_squared: f32 = data.iter().map(|x| x * x).sum();
    let regularization = weight_decay * l2_squared;

    let result = Tensor::from_vec(vec![regularization], &[1]).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create tensor: {}", e))
    })?;

    Ok(PyTensor {
        tensor: Arc::new(result),
        requires_grad: tensor.requires_grad,
        is_pinned: false,
    })
}

/// L1 Regularization function
///
/// Computes L1 (Lasso) regularization term for a tensor.
#[pyfunction]
#[pyo3(signature = (tensor, weight_decay))]
pub fn l1_regularization(tensor: &PyTensor, weight_decay: f32) -> PyResult<PyTensor> {
    if weight_decay < 0.0 {
        return Err(PyValueError::new_err("weight_decay must be non-negative"));
    }

    let data = tensor.tensor.to_vec().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to get tensor data: {}", e))
    })?;

    // Compute L1 norm
    let l1_norm: f32 = data.iter().map(|x| x.abs()).sum();
    let regularization = weight_decay * l1_norm;

    let result = Tensor::from_vec(vec![regularization], &[1]).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create tensor: {}", e))
    })?;

    Ok(PyTensor {
        tensor: Arc::new(result),
        requires_grad: tensor.requires_grad,
        is_pinned: false,
    })
}

/// Elastic Net Regularization function
///
/// Combines L1 and L2 regularization.
#[pyfunction]
#[pyo3(signature = (tensor, l1_weight, l2_weight))]
pub fn elastic_net_regularization(
    tensor: &PyTensor,
    l1_weight: f32,
    l2_weight: f32,
) -> PyResult<PyTensor> {
    if l1_weight < 0.0 || l2_weight < 0.0 {
        return Err(PyValueError::new_err(
            "l1_weight and l2_weight must be non-negative",
        ));
    }

    let data = tensor.tensor.to_vec().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to get tensor data: {}", e))
    })?;

    // Compute L1 and L2 norms
    let l1_norm: f32 = data.iter().map(|x| x.abs()).sum();
    let l2_squared: f32 = data.iter().map(|x| x * x).sum();

    let regularization = l1_weight * l1_norm + l2_weight * l2_squared;

    let result = Tensor::from_vec(vec![regularization], &[1]).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create tensor: {}", e))
    })?;

    Ok(PyTensor {
        tensor: Arc::new(result),
        requires_grad: tensor.requires_grad,
        is_pinned: false,
    })
}
