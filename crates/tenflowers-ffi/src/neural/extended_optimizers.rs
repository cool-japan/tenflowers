//! Extended optimizer implementations for neural network training
//!
//! This module provides Python bindings for additional optimizers beyond the basic ones,
//! including AdaBelief, RAdam, Nadam, AdaGrad, AdaDelta, LAMB, Lion, and others.

use pyo3::prelude::*;
use std::collections::HashMap;

/// Python wrapper for the AdaBelief optimizer
///
/// AdaBelief adapts the step size according to the "belief" in observed gradients,
/// providing better convergence than Adam in many cases.
///
/// Reference: "AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients"
///
/// Parameters:
/// * learning_rate: Step size for parameter updates (default: 0.001)
/// * beta1: Exponential decay rate for first moment estimates (default: 0.9)
/// * beta2: Exponential decay rate for second moment estimates (default: 0.999)
/// * epsilon: Small constant for numerical stability (default: 1e-16)
/// * weight_decay: Weight decay (L2 penalty) coefficient (default: 0.0)
/// * amsgrad: Whether to use AMSGrad variant (default: true)
#[pyclass(name = "AdaBelief")]
#[derive(Debug, Clone)]
pub struct PyAdaBelief {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,
    pub timestep: usize,
}

#[pymethods]
impl PyAdaBelief {
    /// Create a new AdaBelief optimizer with default parameters
    #[new]
    #[pyo3(signature = (learning_rate=None))]
    pub fn new(learning_rate: Option<f64>) -> Self {
        Self {
            learning_rate: learning_rate.unwrap_or(0.001),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-16,
            weight_decay: 0.0,
            amsgrad: true,
            timestep: 0,
        }
    }

    /// Create AdaBelief optimizer with custom beta parameters
    #[staticmethod]
    #[pyo3(signature = (learning_rate, beta1=0.9, beta2=0.999))]
    pub fn with_betas(learning_rate: f64, beta1: f64, beta2: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon: 1e-16,
            weight_decay: 0.0,
            amsgrad: true,
            timestep: 0,
        }
    }

    /// Create AdaBelief optimizer with custom epsilon
    #[staticmethod]
    #[pyo3(signature = (learning_rate, epsilon=1e-16))]
    pub fn with_epsilon(learning_rate: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon,
            weight_decay: 0.0,
            amsgrad: true,
            timestep: 0,
        }
    }

    /// Create AdaBelief optimizer with AMSGrad variant
    #[staticmethod]
    #[pyo3(signature = (learning_rate, amsgrad=true))]
    pub fn with_amsgrad(learning_rate: f64, amsgrad: bool) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-16,
            weight_decay: 0.0,
            amsgrad,
            timestep: 0,
        }
    }

    /// Get the current learning rate
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Set a new learning rate
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    /// Perform a single optimization step
    pub fn step(&mut self, _model: Bound<'_, PyAny>) -> PyResult<()> {
        self.timestep += 1;
        // TODO: Implement proper model interface integration
        Ok(())
    }

    /// Zero out gradients for all parameters
    pub fn zero_grad(&self, _model: Bound<'_, PyAny>) -> PyResult<()> {
        // TODO: Implement gradient zeroing
        Ok(())
    }

    /// Get optimizer state information
    pub fn state_dict(&self) -> HashMap<String, f64> {
        let mut state = HashMap::new();
        state.insert("learning_rate".to_string(), self.learning_rate);
        state.insert("beta1".to_string(), self.beta1);
        state.insert("beta2".to_string(), self.beta2);
        state.insert("epsilon".to_string(), self.epsilon);
        state.insert("weight_decay".to_string(), self.weight_decay);
        state.insert("amsgrad".to_string(), if self.amsgrad { 1.0 } else { 0.0 });
        state.insert("timestep".to_string(), self.timestep as f64);
        state
    }

    /// Load optimizer state from dictionary
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
        if let Some(&amsgrad) = state_dict.get("amsgrad") {
            self.amsgrad = amsgrad > 0.5;
        }
        if let Some(&timestep) = state_dict.get("timestep") {
            self.timestep = timestep as usize;
        }
    }

    /// String representation
    pub fn __str__(&self) -> String {
        format!(
            "AdaBelief(learning_rate={}, beta1={}, beta2={}, epsilon={}, amsgrad={})",
            self.learning_rate, self.beta1, self.beta2, self.epsilon, self.amsgrad
        )
    }

    /// Detailed string representation
    pub fn __repr__(&self) -> String {
        format!(
            "PyAdaBelief(learning_rate={}, beta1={}, beta2={}, epsilon={}, weight_decay={}, amsgrad={}, timestep={})",
            self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay, self.amsgrad, self.timestep
        )
    }
}

/// Python wrapper for the RAdam (Rectified Adam) optimizer
///
/// RAdam provides an automated, dynamic adjustment to the adaptive learning rate
/// based on the variance, addressing bad convergence in the early training stage.
///
/// Reference: "On the Variance of the Adaptive Learning Rate and Beyond"
///
/// Parameters:
/// * learning_rate: Step size for parameter updates (default: 0.001)
/// * beta1: Exponential decay rate for first moment estimates (default: 0.9)
/// * beta2: Exponential decay rate for second moment estimates (default: 0.999)
/// * epsilon: Small constant for numerical stability (default: 1e-8)
/// * weight_decay: Weight decay (L2 penalty) coefficient (default: 0.0)
#[pyclass(name = "RAdam")]
#[derive(Debug, Clone)]
pub struct PyRAdam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub timestep: usize,
}

#[pymethods]
impl PyRAdam {
    /// Create a new RAdam optimizer with default parameters
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

    /// Create RAdam optimizer with custom beta parameters
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

    /// Get the current learning rate
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Set a new learning rate
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    /// Perform a single optimization step
    pub fn step(&mut self, _model: Bound<'_, PyAny>) -> PyResult<()> {
        self.timestep += 1;
        // TODO: Implement proper model interface integration
        Ok(())
    }

    /// Zero out gradients for all parameters
    pub fn zero_grad(&self, _model: Bound<'_, PyAny>) -> PyResult<()> {
        // TODO: Implement gradient zeroing
        Ok(())
    }

    /// Get optimizer state information
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

    /// String representation
    pub fn __str__(&self) -> String {
        format!(
            "RAdam(learning_rate={}, beta1={}, beta2={}, epsilon={})",
            self.learning_rate, self.beta1, self.beta2, self.epsilon
        )
    }

    /// Detailed string representation
    pub fn __repr__(&self) -> String {
        format!(
            "PyRAdam(learning_rate={}, beta1={}, beta2={}, epsilon={}, weight_decay={}, timestep={})",
            self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay, self.timestep
        )
    }
}

/// Python wrapper for the Nadam optimizer
///
/// Nadam combines Adam with Nesterov momentum for improved convergence.
///
/// Reference: "Incorporating Nesterov Momentum into Adam"
///
/// Parameters:
/// * learning_rate: Step size for parameter updates (default: 0.001)
/// * beta1: Exponential decay rate for first moment estimates (default: 0.9)
/// * beta2: Exponential decay rate for second moment estimates (default: 0.999)
/// * epsilon: Small constant for numerical stability (default: 1e-8)
/// * weight_decay: Weight decay (L2 penalty) coefficient (default: 0.0)
#[pyclass(name = "Nadam")]
#[derive(Debug, Clone)]
pub struct PyNadam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub timestep: usize,
}

#[pymethods]
impl PyNadam {
    /// Create a new Nadam optimizer with default parameters
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

    /// Create Nadam optimizer with custom beta parameters
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

    /// Get the current learning rate
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Set a new learning rate
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    /// Perform a single optimization step
    pub fn step(&mut self, _model: Bound<'_, PyAny>) -> PyResult<()> {
        self.timestep += 1;
        // TODO: Implement proper model interface integration
        Ok(())
    }

    /// Zero out gradients for all parameters
    pub fn zero_grad(&self, _model: Bound<'_, PyAny>) -> PyResult<()> {
        // TODO: Implement gradient zeroing
        Ok(())
    }

    /// Get optimizer state information
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

    /// String representation
    pub fn __str__(&self) -> String {
        format!(
            "Nadam(learning_rate={}, beta1={}, beta2={}, epsilon={})",
            self.learning_rate, self.beta1, self.beta2, self.epsilon
        )
    }

    /// Detailed string representation
    pub fn __repr__(&self) -> String {
        format!(
            "PyNadam(learning_rate={}, beta1={}, beta2={}, epsilon={}, weight_decay={}, timestep={})",
            self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay, self.timestep
        )
    }
}

/// Python wrapper for the AdaGrad optimizer
///
/// AdaGrad adapts the learning rate to parameters, performing smaller updates for
/// frequently occurring features and larger updates for infrequent features.
///
/// Reference: "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"
///
/// Parameters:
/// * learning_rate: Step size for parameter updates (default: 0.01)
/// * epsilon: Small constant for numerical stability (default: 1e-10)
/// * weight_decay: Weight decay (L2 penalty) coefficient (default: 0.0)
/// * lr_decay: Learning rate decay (default: 0.0)
#[pyclass(name = "AdaGrad")]
#[derive(Debug, Clone)]
pub struct PyAdaGrad {
    pub learning_rate: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub lr_decay: f64,
    pub timestep: usize,
}

#[pymethods]
impl PyAdaGrad {
    /// Create a new AdaGrad optimizer with default parameters
    #[new]
    #[pyo3(signature = (learning_rate=None))]
    pub fn new(learning_rate: Option<f64>) -> Self {
        Self {
            learning_rate: learning_rate.unwrap_or(0.01),
            epsilon: 1e-10,
            weight_decay: 0.0,
            lr_decay: 0.0,
            timestep: 0,
        }
    }

    /// Create AdaGrad optimizer with custom epsilon
    #[staticmethod]
    #[pyo3(signature = (learning_rate, epsilon=1e-10))]
    pub fn with_epsilon(learning_rate: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            epsilon,
            weight_decay: 0.0,
            lr_decay: 0.0,
            timestep: 0,
        }
    }

    /// Create AdaGrad optimizer with learning rate decay
    #[staticmethod]
    #[pyo3(signature = (learning_rate, lr_decay))]
    pub fn with_lr_decay(learning_rate: f64, lr_decay: f64) -> Self {
        Self {
            learning_rate,
            epsilon: 1e-10,
            weight_decay: 0.0,
            lr_decay,
            timestep: 0,
        }
    }

    /// Get the current learning rate
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate / (1.0 + self.timestep as f64 * self.lr_decay)
    }

    /// Set a new learning rate
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    /// Perform a single optimization step
    pub fn step(&mut self, _model: Bound<'_, PyAny>) -> PyResult<()> {
        self.timestep += 1;
        // TODO: Implement proper model interface integration
        Ok(())
    }

    /// Zero out gradients for all parameters
    pub fn zero_grad(&self, _model: Bound<'_, PyAny>) -> PyResult<()> {
        // TODO: Implement gradient zeroing
        Ok(())
    }

    /// Get optimizer state information
    pub fn state_dict(&self) -> HashMap<String, f64> {
        let mut state = HashMap::new();
        state.insert("learning_rate".to_string(), self.learning_rate);
        state.insert("epsilon".to_string(), self.epsilon);
        state.insert("weight_decay".to_string(), self.weight_decay);
        state.insert("lr_decay".to_string(), self.lr_decay);
        state.insert("timestep".to_string(), self.timestep as f64);
        state
    }

    /// Load optimizer state from dictionary
    pub fn load_state_dict(&mut self, state_dict: HashMap<String, f64>) {
        if let Some(&lr) = state_dict.get("learning_rate") {
            self.learning_rate = lr;
        }
        if let Some(&epsilon) = state_dict.get("epsilon") {
            self.epsilon = epsilon;
        }
        if let Some(&weight_decay) = state_dict.get("weight_decay") {
            self.weight_decay = weight_decay;
        }
        if let Some(&lr_decay) = state_dict.get("lr_decay") {
            self.lr_decay = lr_decay;
        }
        if let Some(&timestep) = state_dict.get("timestep") {
            self.timestep = timestep as usize;
        }
    }

    /// String representation
    pub fn __str__(&self) -> String {
        format!(
            "AdaGrad(learning_rate={}, epsilon={}, lr_decay={})",
            self.learning_rate, self.epsilon, self.lr_decay
        )
    }

    /// Detailed string representation
    pub fn __repr__(&self) -> String {
        format!(
            "PyAdaGrad(learning_rate={}, epsilon={}, weight_decay={}, lr_decay={}, timestep={})",
            self.learning_rate, self.epsilon, self.weight_decay, self.lr_decay, self.timestep
        )
    }
}

/// Python wrapper for the AdaDelta optimizer
///
/// AdaDelta is an extension of AdaGrad that seeks to reduce its aggressive,
/// monotonically decreasing learning rate by restricting the accumulation window.
///
/// Reference: "ADADELTA: An Adaptive Learning Rate Method"
///
/// Parameters:
/// * rho: Coefficient for running average of squared gradients (default: 0.9)
/// * epsilon: Small constant for numerical stability (default: 1e-6)
/// * weight_decay: Weight decay (L2 penalty) coefficient (default: 0.0)
#[pyclass(name = "AdaDelta")]
#[derive(Debug, Clone)]
pub struct PyAdaDelta {
    pub rho: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub timestep: usize,
}

#[pymethods]
impl PyAdaDelta {
    /// Create a new AdaDelta optimizer with default parameters
    #[new]
    #[pyo3(signature = (rho=None))]
    pub fn new(rho: Option<f64>) -> Self {
        Self {
            rho: rho.unwrap_or(0.9),
            epsilon: 1e-6,
            weight_decay: 0.0,
            timestep: 0,
        }
    }

    /// Create AdaDelta optimizer with custom epsilon
    #[staticmethod]
    #[pyo3(signature = (rho, epsilon=1e-6))]
    pub fn with_epsilon(rho: f64, epsilon: f64) -> Self {
        Self {
            rho,
            epsilon,
            weight_decay: 0.0,
            timestep: 0,
        }
    }

    /// Create AdaDelta optimizer with weight decay
    #[staticmethod]
    #[pyo3(signature = (rho, weight_decay))]
    pub fn with_weight_decay(rho: f64, weight_decay: f64) -> Self {
        Self {
            rho,
            epsilon: 1e-6,
            weight_decay,
            timestep: 0,
        }
    }

    /// Get the rho parameter
    pub fn get_rho(&self) -> f64 {
        self.rho
    }

    /// Set a new rho parameter
    pub fn set_rho(&mut self, rho: f64) {
        self.rho = rho;
    }

    /// Perform a single optimization step
    pub fn step(&mut self, _model: Bound<'_, PyAny>) -> PyResult<()> {
        self.timestep += 1;
        // TODO: Implement proper model interface integration
        Ok(())
    }

    /// Zero out gradients for all parameters
    pub fn zero_grad(&self, _model: Bound<'_, PyAny>) -> PyResult<()> {
        // TODO: Implement gradient zeroing
        Ok(())
    }

    /// Get optimizer state information
    pub fn state_dict(&self) -> HashMap<String, f64> {
        let mut state = HashMap::new();
        state.insert("rho".to_string(), self.rho);
        state.insert("epsilon".to_string(), self.epsilon);
        state.insert("weight_decay".to_string(), self.weight_decay);
        state.insert("timestep".to_string(), self.timestep as f64);
        state
    }

    /// Load optimizer state from dictionary
    pub fn load_state_dict(&mut self, state_dict: HashMap<String, f64>) {
        if let Some(&rho) = state_dict.get("rho") {
            self.rho = rho;
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

    /// String representation
    pub fn __str__(&self) -> String {
        format!("AdaDelta(rho={}, epsilon={})", self.rho, self.epsilon)
    }

    /// Detailed string representation
    pub fn __repr__(&self) -> String {
        format!(
            "PyAdaDelta(rho={}, epsilon={}, weight_decay={}, timestep={})",
            self.rho, self.epsilon, self.weight_decay, self.timestep
        )
    }
}
