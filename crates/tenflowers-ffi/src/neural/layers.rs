//! Neural network layer implementations
//!
//! This module provides Python bindings for various neural network layers
//! including dense layers, parameter management, and sequential models.

use super::hooks::{HookManager, PyHookHandle};
use crate::tensor_ops::PyTensor;
use pyo3::prelude::*;
// use std::collections::HashMap; // Unused for now
use std::sync::Arc;
use tenflowers_core::Tensor;
use tenflowers_neural::layers::{Dense, Layer};

/// Python binding for Parameter (trainable tensor)
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyParameter {
    tensor: Arc<Tensor<f32>>,
    requires_grad: bool,
}

#[pymethods]
impl PyParameter {
    #[new]
    pub fn new(tensor: PyTensor, requires_grad: Option<bool>) -> Self {
        Self {
            tensor: tensor.tensor,
            requires_grad: requires_grad.unwrap_or(true),
        }
    }

    /// Get the underlying tensor
    pub fn tensor(&self) -> PyTensor {
        PyTensor {
            tensor: self.tensor.clone(),
            requires_grad: self.requires_grad,
            is_pinned: false,
        }
    }

    /// Get tensor shape
    pub fn shape(&self) -> Vec<usize> {
        self.tensor.shape().dims().to_vec()
    }

    /// Get tensor size
    pub fn size(&self) -> usize {
        self.tensor.size()
    }

    /// Check if gradients are required
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set gradient requirement
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// Clone the parameter
    pub fn clone_param(&self) -> Self {
        self.clone()
    }

    /// Get data type information
    pub fn dtype(&self) -> String {
        "f32".to_string()
    }

    /// Get device information
    #[allow(unexpected_cfgs)]
    pub fn device(&self) -> String {
        match self.tensor.device() {
            tenflowers_core::Device::Cpu => "cpu".to_string(),
            #[cfg(feature = "gpu")]
            tenflowers_core::Device::Gpu(id) => format!("gpu:{}", id),
            #[cfg(feature = "gpu")]
            tenflowers_core::Device::Rocm(id) => format!("rocm:{}", id),
        }
    }

    /// Zero out gradients (placeholder)
    pub fn zero_grad(&mut self) {
        // In a full implementation, this would zero the gradient buffer
        // For now, this is a placeholder
    }

    /// String representation
    pub fn __repr__(&self) -> String {
        format!(
            "PyParameter(shape={:?}, requires_grad={})",
            self.shape(),
            self.requires_grad
        )
    }
}

/// Python binding for Dense layer
#[pyclass]
#[derive(Debug)]
pub struct PyDense {
    layer: Dense<f32>,
    hook_manager: HookManager,
    training: bool,
}

impl Clone for PyDense {
    fn clone(&self) -> Self {
        Self {
            layer: self.layer.clone(),
            hook_manager: HookManager::new(),
            training: self.training,
        }
    }
}

#[pymethods]
impl PyDense {
    #[new]
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        use_bias: Option<bool>,
        activation: Option<String>,
    ) -> Self {
        let use_bias = use_bias.unwrap_or(true);
        let mut layer = Dense::new_xavier(input_dim, output_dim, use_bias);

        if let Some(act) = activation {
            layer = layer.with_activation(act);
        }

        Self {
            layer,
            hook_manager: HookManager::new(),
            training: false, // Start in eval mode by default
        }
    }

    /// Forward pass through the layer
    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // Execute pre-forward hooks
        self.hook_manager.execute_forward_hooks(input, input)?;

        match self.layer.forward(&input.tensor) {
            Ok(output) => {
                let output_tensor = PyTensor {
                    tensor: Arc::new(output),
                    requires_grad: input.requires_grad,
                    is_pinned: false,
                };

                // Execute post-forward hooks
                self.hook_manager
                    .execute_forward_hooks(input, &output_tensor)?;

                Ok(output_tensor)
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Forward pass failed: {}",
                e
            ))),
        }
    }

    /// Get layer parameters
    pub fn parameters(&self) -> Vec<PyTensor> {
        let mut params = Vec::new();

        // Get weights
        if let Some(weights) = self.layer.weights() {
            params.push(PyTensor {
                tensor: Arc::new(weights.clone()),
                requires_grad: true,
                is_pinned: false,
            });
        }

        // Get bias if it exists
        if let Some(bias) = self.layer.bias() {
            params.push(PyTensor {
                tensor: Arc::new(bias.clone()),
                requires_grad: true,
                is_pinned: false,
            });
        }

        params
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        if let Some(weights) = self.layer.weights() {
            count += weights.size();
        }
        if let Some(bias) = self.layer.bias() {
            count += bias.size();
        }
        count
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Check if layer is in training mode
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.layer.input_dim()
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.layer.output_dim()
    }

    /// Check if layer uses bias
    pub fn has_bias(&self) -> bool {
        self.layer.bias().is_some()
    }

    /// Get activation function name
    pub fn activation(&self) -> Option<String> {
        self.layer.activation_name()
    }

    /// Register a forward hook
    pub fn register_forward_hook(&self, hook: PyObject) -> PyResult<PyHookHandle> {
        self.hook_manager.register_forward_hook(hook)
    }

    /// Register a backward hook
    pub fn register_backward_hook(&self, hook: PyObject) -> PyResult<PyHookHandle> {
        self.hook_manager.register_backward_hook(hook)
    }

    /// Remove a hook by handle
    pub fn remove_hook(&self, handle: &PyHookHandle) -> PyResult<()> {
        self.hook_manager.remove_hook(handle)
    }

    /// Clear all hooks
    pub fn clear_hooks(&self) {
        self.hook_manager.clear_hooks();
    }

    /// Get hook count
    pub fn hook_count(&self) -> (usize, usize) {
        self.hook_manager.hook_count()
    }

    /// String representation
    pub fn __repr__(&self) -> String {
        format!(
            "PyDense(in_features={}, out_features={}, bias={})",
            self.input_dim(),
            self.output_dim(),
            self.has_bias()
        )
    }

    /// String representation
    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl PyDense {
    /// Execute forward hooks (internal method)
    fn execute_forward_hooks(&self, input: &PyTensor, output: &PyTensor) -> PyResult<()> {
        self.hook_manager.execute_forward_hooks(input, output)
    }

    /// Execute backward hooks (internal method)
    fn execute_backward_hooks(&self, grad_input: &PyTensor) -> PyResult<()> {
        self.hook_manager.execute_backward_hooks(grad_input)
    }
}

/// PyTorch-style Sequential model for neural networks
#[pyclass]
#[derive(Debug)]
pub struct PySequential {
    layers: Vec<PyDense>,
    hook_manager: HookManager,
}

impl Clone for PySequential {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
            hook_manager: HookManager::new(),
        }
    }
}

#[pymethods]
impl PySequential {
    #[new]
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            hook_manager: HookManager::new(),
        }
    }

    /// Add a dense layer to the sequential model
    pub fn add(&mut self, layer: PyDense) {
        self.layers.push(layer);
    }

    /// Forward pass through the model
    pub fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // Execute pre-forward hooks
        self.hook_manager.execute_forward_hooks(input, input)?;

        let mut current_input = input.clone();

        // Chain through all layers
        for layer in &self.layers {
            current_input = layer.forward(&current_input)?;
        }

        // Execute post-forward hooks
        self.hook_manager
            .execute_forward_hooks(input, &current_input)?;

        Ok(current_input)
    }

    /// Get the number of layers in the model
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if the model is empty
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Get all model parameters
    pub fn parameters(&self) -> Vec<PyTensor> {
        let mut all_params = Vec::new();
        for layer in &self.layers {
            all_params.extend(layer.parameters());
        }
        all_params
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(|layer| layer.num_parameters()).sum()
    }

    /// Set training mode for all layers
    pub fn train(&mut self, training: Option<bool>) {
        let training_mode = training.unwrap_or(true);
        for layer in &mut self.layers {
            layer.set_training(training_mode);
        }
    }

    /// Set evaluation mode for all layers
    pub fn eval(&mut self) {
        self.train(Some(false));
    }

    /// Check if model is in training mode
    pub fn is_training(&self) -> bool {
        self.layers
            .first()
            .map(|l| l.is_training())
            .unwrap_or(false)
    }

    /// Get layer at index
    pub fn get_layer(&self, index: usize) -> PyResult<PyDense> {
        if index < self.layers.len() {
            Ok(self.layers[index].clone())
        } else {
            Err(pyo3::exceptions::PyIndexError::new_err(
                "Layer index out of range",
            ))
        }
    }

    /// Insert layer at index
    pub fn insert(&mut self, index: usize, layer: PyDense) -> PyResult<()> {
        if index <= self.layers.len() {
            self.layers.insert(index, layer);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyIndexError::new_err(
                "Insert index out of range",
            ))
        }
    }

    /// Remove layer at index
    pub fn remove(&mut self, index: usize) -> PyResult<PyDense> {
        if index < self.layers.len() {
            Ok(self.layers.remove(index))
        } else {
            Err(pyo3::exceptions::PyIndexError::new_err(
                "Remove index out of range",
            ))
        }
    }

    /// Clear all layers
    pub fn clear(&mut self) {
        self.layers.clear();
    }

    /// String representation of the model
    pub fn __str__(&self) -> String {
        if self.layers.is_empty() {
            "Sequential()".to_string()
        } else {
            let layer_strs: Vec<String> = self
                .layers
                .iter()
                .enumerate()
                .map(|(i, layer)| format!("  ({}): {}", i, layer.__repr__()))
                .collect();
            format!("Sequential(\n{}\n)", layer_strs.join("\n"))
        }
    }

    /// String representation
    pub fn __repr__(&self) -> String {
        self.__str__()
    }

    /// Register a forward hook
    pub fn register_forward_hook(&self, hook: PyObject) -> PyResult<PyHookHandle> {
        self.hook_manager.register_forward_hook(hook)
    }

    /// Register a backward hook
    pub fn register_backward_hook(&self, hook: PyObject) -> PyResult<PyHookHandle> {
        self.hook_manager.register_backward_hook(hook)
    }

    /// Remove a hook by handle
    pub fn remove_hook(&self, handle: &PyHookHandle) -> PyResult<()> {
        self.hook_manager.remove_hook(handle)
    }

    /// Clear all hooks
    pub fn clear_hooks(&self) {
        self.hook_manager.clear_hooks();
    }

    /// Get hook count
    pub fn hook_count(&self) -> (usize, usize) {
        self.hook_manager.hook_count()
    }
}

impl Default for PySequential {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a linear layer (alias for Dense)
#[pyfunction]
pub fn linear(input_dim: usize, output_dim: usize, bias: Option<bool>) -> PyDense {
    PyDense::new(input_dim, output_dim, bias, None)
}

/// Create a ReLU dense layer
#[pyfunction]
pub fn relu_linear(input_dim: usize, output_dim: usize, bias: Option<bool>) -> PyDense {
    PyDense::new(input_dim, output_dim, bias, Some("relu".to_string()))
}

/// Create a sigmoid dense layer
#[pyfunction]
pub fn sigmoid_linear(input_dim: usize, output_dim: usize, bias: Option<bool>) -> PyDense {
    PyDense::new(input_dim, output_dim, bias, Some("sigmoid".to_string()))
}

/// Create a tanh dense layer
#[pyfunction]
pub fn tanh_linear(input_dim: usize, output_dim: usize, bias: Option<bool>) -> PyDense {
    PyDense::new(input_dim, output_dim, bias, Some("tanh".to_string()))
}
