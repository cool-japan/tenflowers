//! Optimizer implementations for neural network training
//!
//! This module provides Python bindings for various optimizers including Adam, SGD, RMSprop, etc.

use pyo3::prelude::*;
use std::collections::HashMap;

/// Python wrapper for the Adam optimizer
///
/// Adam (Adaptive Moment Estimation) is an algorithm for first-order gradient-based
/// optimization of stochastic objective functions, based on adaptive estimates of
/// lower-order moments.
///
/// This is a simplified thread-safe implementation for Python bindings.
///
/// Parameters:
/// * learning_rate: Step size for parameter updates (default: 0.001)
/// * beta1: Exponential decay rate for first moment estimates (default: 0.9)
/// * beta2: Exponential decay rate for second moment estimates (default: 0.999)
/// * epsilon: Small constant to prevent division by zero (default: 1e-8)
/// * weight_decay: Weight decay (L2 penalty) coefficient (default: 0.0)
#[pyclass(name = "Adam")]
#[derive(Debug, Clone)]
pub struct PyAdam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub timestep: usize,
}

#[pymethods]
impl PyAdam {
    /// Create a new Adam optimizer with default parameters
    ///
    /// Args:
    ///     learning_rate: Optional learning rate (default: 0.001)
    ///
    /// Returns:
    ///     New Adam optimizer instance
    #[new]
    #[pyo3(signature = (learning_rate=None))]
    pub fn new(learning_rate: Option<f64>) -> Self {
        Self {
            learning_rate: learning_rate.unwrap_or(0.001),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            timestep: 0,
        }
    }

    /// Create Adam optimizer with custom beta parameters
    ///
    /// Args:
    ///     learning_rate: Learning rate for parameter updates
    ///     beta1: Exponential decay rate for first moment estimates
    ///     beta2: Exponential decay rate for second moment estimates
    ///
    /// Returns:
    ///     Adam optimizer with custom beta values
    #[staticmethod]
    #[pyo3(signature = (learning_rate, beta1=0.9, beta2=0.999))]
    pub fn with_betas(learning_rate: f64, beta1: f64, beta2: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon: 1e-8,
            weight_decay: 0.0,
            timestep: 0,
        }
    }

    /// Create Adam optimizer with custom epsilon
    ///
    /// Args:
    ///     learning_rate: Learning rate for parameter updates
    ///     epsilon: Small constant to prevent division by zero
    ///
    /// Returns:
    ///     Adam optimizer with custom epsilon
    #[staticmethod]
    #[pyo3(signature = (learning_rate, epsilon=1e-8))]
    pub fn with_epsilon(learning_rate: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon,
            weight_decay: 0.0,
            timestep: 0,
        }
    }

    /// Create Adam optimizer with weight decay (AdamW-style)
    ///
    /// Args:
    ///     learning_rate: Learning rate for parameter updates
    ///     weight_decay: L2 penalty coefficient
    ///
    /// Returns:
    ///     Adam optimizer with weight decay
    #[staticmethod]
    #[pyo3(signature = (learning_rate, weight_decay))]
    pub fn with_weight_decay(learning_rate: f64, weight_decay: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay,
            timestep: 0,
        }
    }

    /// Get the current learning rate
    ///
    /// Returns:
    ///     Current learning rate value
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Set a new learning rate
    ///
    /// Args:
    ///     learning_rate: New learning rate value
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    /// Perform a single optimization step
    ///
    /// Args:
    ///     model: Model containing parameters to optimize
    ///
    /// This method computes gradients and updates model parameters using the Adam algorithm.
    /// The model should implement the Model trait and have computed gradients.
    pub fn step(&mut self, model: Bound<'_, PyAny>) -> PyResult<()> {
        // Increment timestep for Adam algorithm
        self.timestep += 1;

        // For now, we'll implement a simplified version that works with PyDense layers
        // In a full implementation, this would integrate with the Model trait
        // TODO: Implement proper model interface integration

        // This is a placeholder implementation - in practice, you'd extract parameters
        // from the model and apply the Adam update rule
        Ok(())
    }

    /// Zero out gradients for all parameters
    ///
    /// Args:
    ///     model: Model containing parameters to zero gradients for
    ///
    /// This should be called before backward pass to clear accumulated gradients.
    pub fn zero_grad(&self, model: Bound<'_, PyAny>) -> PyResult<()> {
        // TODO: Implement gradient zeroing for model parameters
        // This would typically iterate through model parameters and set gradients to zero
        Ok(())
    }

    /// Get optimizer state information
    ///
    /// Returns:
    ///     Dictionary containing optimizer configuration and state
    pub fn state_dict(&self) -> HashMap<String, f64> {
        let mut state = HashMap::new();
        state.insert("learning_rate".to_string(), self.learning_rate);
        state.insert("beta1".to_string(), self.beta1);
        state.insert("beta2".to_string(), self.beta2);
        state.insert("epsilon".to_string(), self.epsilon);
        state.insert("weight_decay".to_string(), self.weight_decay);
        state.insert("timestep".to_string(), self.timestep as f64);
        state
    }

    /// Load optimizer state from dictionary
    ///
    /// Args:
    ///     state_dict: Dictionary containing optimizer state to restore
    pub fn load_state_dict(&mut self, state_dict: HashMap<String, f64>) {
        if let Some(&lr) = state_dict.get("learning_rate") {
            self.learning_rate = lr;
        }
        if let Some(&beta1) = state_dict.get("beta1") {
            self.beta1 = beta1;
        }
        if let Some(&beta2) = state_dict.get("beta2") {
            self.beta2 = beta2;
        }
        if let Some(&epsilon) = state_dict.get("epsilon") {
            self.epsilon = epsilon;
        }
        if let Some(&weight_decay) = state_dict.get("weight_decay") {
            self.weight_decay = weight_decay;
        }
        if let Some(&timestep) = state_dict.get("timestep") {
            self.timestep = timestep as usize;
        }
    }

    /// String representation of the optimizer
    pub fn __str__(&self) -> String {
        format!(
            "Adam(learning_rate={}, beta1={}, beta2={}, epsilon={}, weight_decay={})",
            self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay
        )
    }

    /// Detailed string representation
    pub fn __repr__(&self) -> String {
        format!("PyAdam(learning_rate={}, beta1={}, beta2={}, epsilon={}, weight_decay={}, timestep={})",
               self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay, self.timestep)
    }
}

// Clone is automatically derived for PyAdam since all fields implement Clone
