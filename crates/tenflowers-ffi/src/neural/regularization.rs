//! Regularization layers module for TenfloweRS FFI
//!
//! This module provides regularization layer implementations including Dropout,
//! AlphaDropout, and other regularization techniques for neural network training.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use scirs2_core::random::thread_rng;
use std::sync::Arc;
use tenflowers_core::Tensor;
use tenflowers_neural::layers::dropout::{Dropout, SpatialDropout2D};
use tenflowers_neural::layers::Layer;

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

        // Training mode with 0 < p < 1: apply real inverted dropout via the
        // backing neural Dropout layer (random Bernoulli mask + 1/(1-p) scaling).
        let dropout = Dropout::<f32>::new(self.p);
        let output = Layer::forward(&dropout, &input.tensor)
            .map_err(|e| PyRuntimeError::new_err(format!("Dropout forward failed: {}", e)))?;

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

        // Training mode with 0 < p < 1: drop entire channels (feature maps).
        if input_shape.len() == 4 {
            // Canonical [N, C, H, W]: use the backing neural spatial dropout.
            let spatial_dropout = SpatialDropout2D::<f32>::new(self.p);
            let output = Layer::forward(&spatial_dropout, &input.tensor)
                .map_err(|e| PyRuntimeError::new_err(format!("Dropout2D forward failed: {}", e)))?;
            return Ok(PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            });
        }

        // General case (e.g. [N, C, L]): mask whole channels and rescale by 1/(1-p).
        let channels_total = input_shape[0] * input_shape[1];
        let spatial: usize = input_shape.iter().skip(2).product();
        let data = input
            .tensor
            .to_vec()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read tensor data: {}", e)))?;

        let scale = 1.0 / (1.0 - self.p);
        let mut rng = thread_rng();
        let mut channel_mask = Vec::with_capacity(channels_total);
        for _ in 0..channels_total {
            let random_val: f64 = rng.gen_range(0.0..1.0);
            channel_mask.push(if random_val < self.p as f64 {
                0.0f32
            } else {
                scale
            });
        }

        let mut out_data = Vec::with_capacity(data.len());
        for (idx, &value) in data.iter().enumerate() {
            out_data.push(value * channel_mask[idx / spatial]);
        }

        let shape: Vec<usize> = input_shape.iter().copied().collect();
        let output = Tensor::from_vec(out_data, &shape)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output: {}", e)))?;

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

        // Alpha Dropout preserves the self-normalizing (SELU) statistics: dropped
        // units are set to the negative SELU saturation value alpha' = -lambda*alpha,
        // then an affine transform x' = a*(mask*x + (1-mask)*alpha') + b restores the
        // mean and variance (with keep probability q = 1 - p).
        let lambda = 1.050_701_f32;
        let alpha = 1.673_263_2_f32;
        let alpha_prime = -lambda * alpha;

        if self.p >= 1.0 {
            // Keep probability is zero; the preserving affine transform sends
            // every element to zero in the limit.
            let shape: Vec<usize> = input.tensor.shape().iter().copied().collect();
            return Ok(PyTensor {
                tensor: Arc::new(Tensor::zeros(&shape)),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            });
        }

        let q = 1.0 - self.p;
        let a = (q + alpha_prime.powi(2) * self.p * q).powf(-0.5);
        let b = -a * self.p * alpha_prime;

        let data = input
            .tensor
            .to_vec()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read tensor data: {}", e)))?;

        let mut rng = thread_rng();
        let mut out_data = Vec::with_capacity(data.len());
        for &value in data.iter() {
            let random_val: f64 = rng.gen_range(0.0..1.0);
            let base = if random_val >= self.p as f64 {
                value
            } else {
                alpha_prime
            };
            out_data.push(a * base + b);
        }

        let shape: Vec<usize> = input.tensor.shape().iter().copied().collect();
        let output = Tensor::from_vec(out_data, &shape)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output: {}", e)))?;

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

        let input_shape = input.tensor.shape();
        if input_shape.len() < 2 {
            return Err(PyValueError::new_err(
                "FeatureAlphaDropout expects at least 2D input [N, C, ...]",
            ));
        }

        // Same SELU-preserving affine transform as AlphaDropout, but the
        // Bernoulli mask is applied to whole channels (feature maps).
        let lambda = 1.050_701_f32;
        let alpha = 1.673_263_2_f32;
        let alpha_prime = -lambda * alpha;

        if self.p >= 1.0 {
            let shape: Vec<usize> = input_shape.iter().copied().collect();
            return Ok(PyTensor {
                tensor: Arc::new(Tensor::zeros(&shape)),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            });
        }

        let q = 1.0 - self.p;
        let a = (q + alpha_prime.powi(2) * self.p * q).powf(-0.5);
        let b = -a * self.p * alpha_prime;

        let channels_total = input_shape[0] * input_shape[1];
        let spatial: usize = input_shape.iter().skip(2).product();

        let data = input
            .tensor
            .to_vec()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read tensor data: {}", e)))?;

        let mut rng = thread_rng();
        let mut channel_kept = Vec::with_capacity(channels_total);
        for _ in 0..channels_total {
            let random_val: f64 = rng.gen_range(0.0..1.0);
            channel_kept.push(random_val >= self.p as f64);
        }

        let mut out_data = Vec::with_capacity(data.len());
        for (idx, &value) in data.iter().enumerate() {
            let base = if channel_kept[idx / spatial] {
                value
            } else {
                alpha_prime
            };
            out_data.push(a * base + b);
        }

        let shape: Vec<usize> = input_shape.iter().copied().collect();
        let output = Tensor::from_vec(out_data, &shape)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output: {}", e)))?;

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

    #[test]
    fn dropout_eval_mode_is_identity() {
        let mut dropout = PyDropout::new(Some(0.5), Some(false)).expect("dropout");
        dropout.eval();
        let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let out = dropout.forward(&input).expect("forward");

        let in_vec = input.tensor.to_vec().expect("in vec");
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert_eq!(in_vec, out_vec, "dropout in eval mode must be identity");
    }

    #[test]
    fn dropout_train_zero_prob_is_identity() {
        // p == 0 in training mode must be a no-op.
        let dropout = PyDropout::new(Some(0.0), Some(false)).expect("dropout");
        assert!(dropout.training, "dropout defaults to training mode");
        let input = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let out = dropout.forward(&input).expect("forward");

        let in_vec = input.tensor.to_vec().expect("in vec");
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert_eq!(in_vec, out_vec, "dropout with p=0 must be identity");
    }

    #[test]
    fn alpha_dropout_eval_mode_is_identity() {
        let mut alpha = PyAlphaDropout::new(Some(0.5), Some(false)).expect("alpha");
        alpha.eval();
        let input = make_tensor(vec![0.5, -0.5, 1.5, -1.5], &[4]);
        let out = alpha.forward(&input).expect("forward");

        let in_vec = input.tensor.to_vec().expect("in vec");
        let out_vec = out.tensor.to_vec().expect("out vec");
        assert_eq!(
            in_vec, out_vec,
            "alpha dropout in eval mode must be identity"
        );
    }

    #[test]
    fn alpha_dropout_train_preserves_statistics() {
        // Alpha dropout's self-normalizing property is defined for SELU-normalized
        // activations, so feed it a standardized input (mean 0, variance 1) and
        // confirm the output keeps mean ~0 and variance ~1.
        let alpha = PyAlphaDropout::new(Some(0.5), Some(false)).expect("alpha");

        let raw = Tensor::randn(&[10000])
            .expect("randn")
            .to_vec()
            .expect("raw vec");
        let n = raw.len() as f32;
        let mean = raw.iter().sum::<f32>() / n;
        let var = raw.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = var.sqrt().max(1e-6);
        let standardized: Vec<f32> = raw.iter().map(|x| (x - mean) / std).collect();
        let input = make_tensor(standardized, &[10000]);

        let out = alpha.forward(&input).expect("forward");
        let v = out.tensor.to_vec().expect("out vec");
        let out_mean = v.iter().sum::<f32>() / n;
        let out_var = v.iter().map(|x| (x - out_mean).powi(2)).sum::<f32>() / n;

        assert!(
            out_mean.abs() < 0.1,
            "alpha dropout should preserve ~zero mean, got {}",
            out_mean
        );
        assert!(
            (out_var - 1.0).abs() < 0.15,
            "alpha dropout should preserve ~unit variance, got {}",
            out_var
        );
    }
}
