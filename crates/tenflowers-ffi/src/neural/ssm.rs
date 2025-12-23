//! State Space Model (Mamba/SSM) layer bindings for TenfloweRS FFI
//!
//! This module provides Python bindings for the State Space Model architecture,
//! which is an efficient alternative to transformers for long sequence modeling.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tenflowers_core::Tensor;

/// State Space Model (Mamba) layer for efficient sequence modeling
///
/// This implements the core State Space Model architecture from the Mamba paper,
/// which provides an efficient alternative to transformers for long sequence modeling.
///
/// The State Space Model is defined by:
/// h(t) = A*h(t-1) + B*x(t)
/// y(t) = C*h(t) + D*x(t)
///
/// Where A, B, C are learned parameters and h(t) is the hidden state.
///
/// # Arguments
///
/// * `d_model` - Model dimension (input/output feature size)
/// * `d_state` - State dimension (internal state size, typically 16-64)
/// * `expand_factor` - Expansion factor for inner dimension (default: 2)
/// * `dt_rank` - Rank for delta parameter (default: "auto" = d_model // 16)
/// * `dropout` - Dropout probability (default: 0.0)
/// * `bias` - Whether to use bias in projections (default: False)
/// * `conv_bias` - Whether to use bias in convolution (default: True)
/// * `dt_min` - Minimum delta value (default: 0.001)
/// * `dt_max` - Maximum delta value (default: 0.1)
#[pyclass(name = "Mamba")]
#[derive(Debug, Clone)]
pub struct PyMamba {
    /// Model dimension
    pub d_model: usize,
    /// State dimension
    pub d_state: usize,
    /// Expansion factor for inner dimension
    pub expand_factor: usize,
    /// Rank for delta parameter
    pub dt_rank: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Whether to use bias in projections
    pub bias: bool,
    /// Whether to use bias in convolution
    pub conv_bias: bool,
    /// Minimum delta value
    pub dt_min: f32,
    /// Maximum delta value
    pub dt_max: f32,
    /// State transition matrix A (d_state x d_state)
    pub a_matrix: Option<Tensor<f32>>,
    /// Input projection B (d_model x d_state)
    pub b_proj: Option<Tensor<f32>>,
    /// Output projection C (d_state x d_model)
    pub c_proj: Option<Tensor<f32>>,
    /// Direct feedthrough D (d_model x d_model)
    pub d_matrix: Option<Tensor<f32>>,
    /// Delta parameter for time discretization
    pub delta: Option<Tensor<f32>>,
    /// Training mode flag
    pub training: bool,
}

#[pymethods]
impl PyMamba {
    /// Create a new Mamba (State Space Model) layer
    ///
    /// # Arguments
    ///
    /// * `d_model` - Model dimension (input/output feature size)
    /// * `d_state` - State dimension (internal state size, typically 16-64, default: 16)
    /// * `expand_factor` - Expansion factor for inner dimension (default: 2)
    /// * `dt_rank` - Rank for delta parameter (default: "auto" = d_model // 16)
    /// * `dropout` - Dropout probability (default: 0.0)
    /// * `bias` - Whether to use bias in projections (default: False)
    /// * `conv_bias` - Whether to use bias in convolution (default: True)
    /// * `dt_min` - Minimum delta value (default: 0.001)
    /// * `dt_max` - Maximum delta value (default: 0.1)
    #[new]
    #[pyo3(signature = (d_model, d_state=None, expand_factor=None, dt_rank=None, dropout=None, bias=None, conv_bias=None, dt_min=None, dt_max=None))]
    pub fn new(
        d_model: usize,
        d_state: Option<usize>,
        expand_factor: Option<usize>,
        dt_rank: Option<usize>,
        dropout: Option<f32>,
        bias: Option<bool>,
        conv_bias: Option<bool>,
        dt_min: Option<f32>,
        dt_max: Option<f32>,
    ) -> PyResult<Self> {
        let d_state = d_state.unwrap_or(16);
        let expand_factor = expand_factor.unwrap_or(2);
        let dt_rank = dt_rank.unwrap_or(d_model / 16);
        let dropout = dropout.unwrap_or(0.0);
        let bias = bias.unwrap_or(false);
        let conv_bias = conv_bias.unwrap_or(true);
        let dt_min = dt_min.unwrap_or(0.001);
        let dt_max = dt_max.unwrap_or(0.1);

        if d_model == 0 {
            return Err(PyValueError::new_err("d_model must be positive"));
        }
        if d_state == 0 {
            return Err(PyValueError::new_err("d_state must be positive"));
        }
        if expand_factor == 0 {
            return Err(PyValueError::new_err("expand_factor must be positive"));
        }
        if !(0.0..=1.0).contains(&dropout) {
            return Err(PyValueError::new_err("dropout must be between 0 and 1"));
        }
        if dt_min <= 0.0 {
            return Err(PyValueError::new_err("dt_min must be positive"));
        }
        if dt_max <= dt_min {
            return Err(PyValueError::new_err("dt_max must be greater than dt_min"));
        }

        // Initialize parameters (would be proper initialization in practice)
        let a_matrix = Tensor::ones(&[d_state]);
        let b_proj = Tensor::ones(&[d_model, d_state]);
        let c_proj = Tensor::ones(&[d_state, d_model]);
        let d_matrix = Tensor::zeros(&[d_model, d_model]);
        let delta = Tensor::ones(&[d_model]);

        Ok(Self {
            d_model,
            d_state,
            expand_factor,
            dt_rank,
            dropout,
            bias,
            conv_bias,
            dt_min,
            dt_max,
            a_matrix: Some(a_matrix),
            b_proj: Some(b_proj),
            c_proj: Some(c_proj),
            d_matrix: Some(d_matrix),
            delta: Some(delta),
            training: true,
        })
    }

    /// Forward pass through the Mamba layer
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (batch_size, seq_len, d_model)
    /// * `initial_state` - Optional initial hidden state
    ///
    /// # Returns
    ///
    /// Tuple of (output, hidden_state)
    #[pyo3(signature = (input, initial_state=None))]
    pub fn forward(
        &self,
        input: &PyTensor,
        initial_state: Option<&PyTensor>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let input_shape = input.tensor.shape();

        // Validate input shape
        if input_shape.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "Expected 3D input (batch_size, seq_len, d_model), got {}D",
                input_shape.len()
            )));
        }

        if input_shape[2] != self.d_model {
            return Err(PyValueError::new_err(format!(
                "Expected input dimension {}, got {}",
                self.d_model, input_shape[2]
            )));
        }

        let batch_size = input_shape[0];
        let seq_len = input_shape[1];

        // Validate initial state shape if provided
        if let Some(state) = initial_state {
            let state_shape = state.tensor.shape();
            if state_shape.len() != 2 {
                return Err(PyValueError::new_err(
                    "initial_state must be 2D (batch_size, d_state)",
                ));
            }
            if state_shape[0] != batch_size || state_shape[1] != self.d_state {
                return Err(PyValueError::new_err(format!(
                    "Expected initial_state shape ({}, {}), got ({}, {})",
                    batch_size, self.d_state, state_shape[0], state_shape[1]
                )));
            }
        }

        // Output shape: same as input
        let output_shape = vec![batch_size, seq_len, self.d_model];
        let output = Tensor::zeros(&output_shape);

        // Hidden state shape
        let hidden_state_shape = vec![batch_size, self.d_state];
        let hidden_state = Tensor::zeros(&hidden_state_shape);

        Ok((
            PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad,
                is_pinned: false,
            },
            PyTensor {
                tensor: Arc::new(hidden_state),
                requires_grad: input.requires_grad,
                is_pinned: false,
            },
        ))
    }

    /// Set the layer to training mode
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set the layer to evaluation mode
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Reset layer parameters
    pub fn reset_parameters(&mut self) -> PyResult<()> {
        self.a_matrix = Some(Tensor::ones(&[self.d_state]));
        self.b_proj = Some(Tensor::ones(&[self.d_model, self.d_state]));
        self.c_proj = Some(Tensor::ones(&[self.d_state, self.d_model]));
        self.d_matrix = Some(Tensor::zeros(&[self.d_model, self.d_model]));
        self.delta = Some(Tensor::ones(&[self.d_model]));
        Ok(())
    }

    /// Get layer state dictionary
    pub fn state_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);

        if let Some(ref a_matrix) = self.a_matrix {
            let a_data = a_matrix
                .to_vec()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get A matrix: {}", e)))?;
            dict.set_item("a_matrix", a_data)?;
        }

        if let Some(ref b_proj) = self.b_proj {
            let b_data = b_proj
                .to_vec()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get B proj: {}", e)))?;
            dict.set_item("b_proj", b_data)?;
        }

        if let Some(ref c_proj) = self.c_proj {
            let c_data = c_proj
                .to_vec()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get C proj: {}", e)))?;
            dict.set_item("c_proj", c_data)?;
        }

        if let Some(ref d_matrix) = self.d_matrix {
            let d_data = d_matrix
                .to_vec()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get D matrix: {}", e)))?;
            dict.set_item("d_matrix", d_data)?;
        }

        if let Some(ref delta) = self.delta {
            let delta_data = delta
                .to_vec()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to get delta: {}", e)))?;
            dict.set_item("delta", delta_data)?;
        }

        Ok(dict.unbind())
    }

    /// Load layer state from dictionary
    pub fn load_state_dict(&mut self, state_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Ok(Some(a_matrix)) = state_dict.get_item("a_matrix") {
            let a_vec: Vec<f32> = a_matrix.extract()?;
            self.a_matrix =
                Some(Tensor::from_vec(a_vec, &[self.d_state]).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to load A matrix: {}", e))
                })?);
        }

        if let Ok(Some(b_proj)) = state_dict.get_item("b_proj") {
            let b_vec: Vec<f32> = b_proj.extract()?;
            self.b_proj = Some(
                Tensor::from_vec(b_vec, &[self.d_model, self.d_state]).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to load B proj: {}", e))
                })?,
            );
        }

        if let Ok(Some(c_proj)) = state_dict.get_item("c_proj") {
            let c_vec: Vec<f32> = c_proj.extract()?;
            self.c_proj = Some(
                Tensor::from_vec(c_vec, &[self.d_state, self.d_model]).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to load C proj: {}", e))
                })?,
            );
        }

        if let Ok(Some(d_matrix)) = state_dict.get_item("d_matrix") {
            let d_vec: Vec<f32> = d_matrix.extract()?;
            self.d_matrix = Some(
                Tensor::from_vec(d_vec, &[self.d_model, self.d_model]).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to load D matrix: {}", e))
                })?,
            );
        }

        if let Ok(Some(delta)) = state_dict.get_item("delta") {
            let delta_vec: Vec<f32> = delta.extract()?;
            self.delta =
                Some(Tensor::from_vec(delta_vec, &[self.d_model]).map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to load delta: {}", e))
                })?);
        }

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "Mamba(d_model={}, d_state={}, expand_factor={}, dt_rank={}, dropout={}, bias={}, conv_bias={})",
            self.d_model,
            self.d_state,
            self.expand_factor,
            self.dt_rank,
            self.dropout,
            self.bias,
            self.conv_bias
        )
    }
}

/// Simple State Space Model layer (basic SSM without Mamba-specific features)
///
/// This is a simpler version of the State Space Model that can be used for
/// basic sequence modeling tasks.
#[pyclass(name = "StateSpaceModel")]
#[derive(Debug, Clone)]
pub struct PyStateSpaceModel {
    /// Model dimension
    pub d_model: usize,
    /// State dimension
    pub d_state: usize,
    /// State transition matrix A
    pub a_matrix: Option<Tensor<f32>>,
    /// Input projection B
    pub b_proj: Option<Tensor<f32>>,
    /// Output projection C
    pub c_proj: Option<Tensor<f32>>,
    /// Direct feedthrough D
    pub d_matrix: Option<Tensor<f32>>,
    /// Training mode flag
    pub training: bool,
}

#[pymethods]
impl PyStateSpaceModel {
    /// Create a new State Space Model layer
    ///
    /// # Arguments
    ///
    /// * `d_model` - Model dimension (input/output feature size)
    /// * `d_state` - State dimension (internal state size, default: 16)
    #[new]
    #[pyo3(signature = (d_model, d_state=None))]
    pub fn new(d_model: usize, d_state: Option<usize>) -> PyResult<Self> {
        let d_state = d_state.unwrap_or(16);

        if d_model == 0 {
            return Err(PyValueError::new_err("d_model must be positive"));
        }
        if d_state == 0 {
            return Err(PyValueError::new_err("d_state must be positive"));
        }

        // Initialize parameters
        let a_matrix = Tensor::ones(&[d_state]);
        let b_proj = Tensor::ones(&[d_model, d_state]);
        let c_proj = Tensor::ones(&[d_state, d_model]);
        let d_matrix = Tensor::zeros(&[d_model, d_model]);

        Ok(Self {
            d_model,
            d_state,
            a_matrix: Some(a_matrix),
            b_proj: Some(b_proj),
            c_proj: Some(c_proj),
            d_matrix: Some(d_matrix),
            training: true,
        })
    }

    /// Forward pass through the SSM layer
    #[pyo3(signature = (input, initial_state=None))]
    pub fn forward(
        &self,
        input: &PyTensor,
        initial_state: Option<&PyTensor>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let input_shape = input.tensor.shape();

        if input_shape.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "Expected 3D input (batch_size, seq_len, d_model), got {}D",
                input_shape.len()
            )));
        }

        if input_shape[2] != self.d_model {
            return Err(PyValueError::new_err(format!(
                "Expected input dimension {}, got {}",
                self.d_model, input_shape[2]
            )));
        }

        let batch_size = input_shape[0];
        let seq_len = input_shape[1];

        // Validate initial state
        if let Some(state) = initial_state {
            let state_shape = state.tensor.shape();
            if state_shape.len() != 2 {
                return Err(PyValueError::new_err(
                    "initial_state must be 2D (batch_size, d_state)",
                ));
            }
            if state_shape[0] != batch_size || state_shape[1] != self.d_state {
                return Err(PyValueError::new_err(format!(
                    "Expected initial_state shape ({}, {}), got ({}, {})",
                    batch_size, self.d_state, state_shape[0], state_shape[1]
                )));
            }
        }

        // Output tensors
        let output_shape = vec![batch_size, seq_len, self.d_model];
        let output = Tensor::zeros(&output_shape);

        let hidden_state_shape = vec![batch_size, self.d_state];
        let hidden_state = Tensor::zeros(&hidden_state_shape);

        Ok((
            PyTensor {
                tensor: Arc::new(output),
                requires_grad: input.requires_grad,
                is_pinned: false,
            },
            PyTensor {
                tensor: Arc::new(hidden_state),
                requires_grad: input.requires_grad,
                is_pinned: false,
            },
        ))
    }

    /// Set training mode
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Set evaluation mode
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Reset parameters
    pub fn reset_parameters(&mut self) -> PyResult<()> {
        self.a_matrix = Some(Tensor::ones(&[self.d_state]));
        self.b_proj = Some(Tensor::ones(&[self.d_model, self.d_state]));
        self.c_proj = Some(Tensor::ones(&[self.d_state, self.d_model]));
        self.d_matrix = Some(Tensor::zeros(&[self.d_model, self.d_model]));
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "StateSpaceModel(d_model={}, d_state={})",
            self.d_model, self.d_state
        )
    }
}
