// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

use super::layers::PyParameter;
use super::optimizer_bridge::collect_parameters;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;
use tenflowers_core::Tensor;
/// The per-parameter inputs a single `.step()` iteration needs: the
/// parameter's current value and its gradient, both flattened to `Vec<f32>`
/// of equal length, plus the parameter's shape (needed to reconstruct a
/// `Tensor` for [`write_back`] afterward).
///
/// A named struct rather than a `(Vec<f32>, Vec<f32>, Vec<usize>)` tuple
/// specifically to keep `param_value_and_grad`'s return type simple enough
/// for `clippy::type_complexity` (each field also reads far more clearly at
/// every one of this module's five destructuring call sites than a
/// positional tuple would).
struct ParamStepInputs {
    value: Vec<f32>,
    grad: Vec<f32>,
    shape: Vec<usize>,
}
/// Extract a parameter's current value and gradient as flat `Vec<f32>` buffers of
/// equal length, along with the parameter's shape.
///
/// Returns `Ok(None)` (not an error) when the parameter has no gradient recorded
/// for this step, so callers can `continue` past it.
fn param_value_and_grad(param: &PyParameter) -> PyResult<Option<ParamStepInputs>> {
    let grad_tensor = match param.grad() {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };
    let value_tensor = param.to_tensor()?;
    let shape: Vec<usize> = value_tensor.tensor.shape().iter().copied().collect();
    let value_data = value_tensor
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to read parameter data: {}", e)))?;
    let grad_data = grad_tensor
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to read gradient data: {}", e)))?;
    if value_data.len() != grad_data.len() {
        return Err(PyRuntimeError::new_err(format!(
            "parameter/gradient length mismatch: parameter has {} elements, gradient has {}",
            value_data.len(),
            grad_data.len()
        )));
    }
    Ok(Some(ParamStepInputs {
        value: value_data,
        grad: grad_data,
        shape,
    }))
}
/// Write a flat `Vec<f32>` back into a parameter as its new value.
fn write_back(param: &PyParameter, data: Vec<f32>, shape: &[usize]) -> PyResult<()> {
    let tensor = Tensor::from_vec(data, shape).map_err(|e| {
        PyRuntimeError::new_err(format!("failed to construct updated tensor: {}", e))
    })?;
    param.set_data(tensor)
}
/// Zero the gradient of every parameter reachable from `model.parameters()`.
fn zero_grad_all(model: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<()> {
    let params = collect_parameters(model)?;
    for param in &params {
        param.borrow(py).zero_grad()?;
    }
    Ok(())
}
/// Per-parameter state tracked by [`PyAdaBelief`]: first moment `m`, "belief"
/// second moment `s`, and (only used when `amsgrad` is enabled) the running
/// maximum of the bias-corrected `s` seen so far.
#[derive(Debug, Clone)]
struct AdaBeliefState {
    m: Vec<f32>,
    s: Vec<f32>,
    s_max: Vec<f32>,
}
impl AdaBeliefState {
    fn zeros(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            s: vec![0.0; n],
            s_max: vec![0.0; n],
        }
    }
}
/// Python wrapper for the AdaBelief optimizer
///
/// AdaBelief adapts the step size according to the "belief" in observed gradients,
/// providing better convergence than Adam in many cases.
///
/// Reference: "AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients"
///
/// Update rule (per parameter element, at step `t`):
/// ```text
/// m = beta1*m + (1-beta1)*grad
/// s = beta2*s + (1-beta2)*(grad-m)^2 + epsilon
/// m_hat = m / (1 - beta1^t)
/// s_hat = s / (1 - beta2^t)
/// if amsgrad: s_used = max(s_max, s_hat); s_max = s_used
/// else:       s_used = s_hat
/// w -= lr * m_hat / (sqrt(s_used) + epsilon)
/// ```
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
    state: HashMap<usize, AdaBeliefState>,
}
#[pymethods]
impl PyAdaBelief {
    /// Create a new AdaBelief optimizer with default parameters
    #[new]
    #[pyo3(signature = (learning_rate = None))]
    pub fn new(learning_rate: Option<f64>) -> Self {
        Self {
            learning_rate: learning_rate.unwrap_or(0.001),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-16,
            weight_decay: 0.0,
            amsgrad: true,
            timestep: 0,
            state: HashMap::new(),
        }
    }
    /// Create AdaBelief optimizer with custom beta parameters
    #[staticmethod]
    #[pyo3(signature = (learning_rate, beta1 = 0.9, beta2 = 0.999))]
    pub fn with_betas(learning_rate: f64, beta1: f64, beta2: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon: 1e-16,
            weight_decay: 0.0,
            amsgrad: true,
            timestep: 0,
            state: HashMap::new(),
        }
    }
    /// Create AdaBelief optimizer with custom epsilon
    #[staticmethod]
    #[pyo3(signature = (learning_rate, epsilon = 1e-16))]
    pub fn with_epsilon(learning_rate: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon,
            weight_decay: 0.0,
            amsgrad: true,
            timestep: 0,
            state: HashMap::new(),
        }
    }
    /// Create AdaBelief optimizer with AMSGrad variant
    #[staticmethod]
    #[pyo3(signature = (learning_rate, amsgrad = true))]
    pub fn with_amsgrad(learning_rate: f64, amsgrad: bool) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-16,
            weight_decay: 0.0,
            amsgrad,
            timestep: 0,
            state: HashMap::new(),
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
    pub fn step(&mut self, model: Bound<'_, PyAny>) -> PyResult<()> {
        let py = model.py();
        let params = collect_parameters(&model)?;
        self.timestep += 1;
        let t = self.timestep as i32;
        let lr = self.learning_rate as f32;
        let beta1 = self.beta1 as f32;
        let beta2 = self.beta2 as f32;
        let epsilon = self.epsilon as f32;
        let weight_decay = self.weight_decay as f32;
        let bias_correction1 = 1.0 - self.beta1.powi(t) as f32;
        let bias_correction2 = 1.0 - self.beta2.powi(t) as f32;
        for param in &params {
            let param_ref = param.borrow(py);
            let Some(ParamStepInputs {
                mut value,
                mut grad,
                shape,
            }) = param_value_and_grad(&param_ref)?
            else {
                continue;
            };
            if weight_decay != 0.0 {
                for (g, w) in grad.iter_mut().zip(value.iter()) {
                    *g += weight_decay * *w;
                }
            }
            let n = value.len();
            let param_state = self
                .state
                .entry(param_ref.id())
                .or_insert_with(|| AdaBeliefState::zeros(n));
            for i in 0..n {
                let g = grad[i];
                param_state.m[i] = beta1 * param_state.m[i] + (1.0 - beta1) * g;
                let diff = g - param_state.m[i];
                param_state.s[i] = beta2 * param_state.s[i] + (1.0 - beta2) * diff * diff + epsilon;
                let m_hat = param_state.m[i] / bias_correction1;
                let s_hat = param_state.s[i] / bias_correction2;
                let s_used = if self.amsgrad {
                    let m = param_state.s_max[i].max(s_hat);
                    param_state.s_max[i] = m;
                    m
                } else {
                    s_hat
                };
                value[i] -= lr * m_hat / (s_used.sqrt() + epsilon);
            }
            drop(param_ref);
            write_back(&param.borrow(py), value, &shape)?;
        }
        Ok(())
    }
    /// Zero out gradients for all parameters
    pub fn zero_grad(&self, model: Bound<'_, PyAny>) -> PyResult<()> {
        let py = model.py();
        zero_grad_all(&model, py)
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
            self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay,
            self.amsgrad, self.timestep
        )
    }
}
/// Per-parameter state tracked by [`PyRAdam`] and [`PyNadam`]: standard Adam
/// first/second raw moments.
#[derive(Debug, Clone)]
struct AdamMomentState {
    m: Vec<f32>,
    v: Vec<f32>,
}
impl AdamMomentState {
    fn zeros(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }
}
/// Python wrapper for the RAdam (Rectified Adam) optimizer
///
/// RAdam provides an automated, dynamic adjustment to the adaptive learning rate
/// based on the variance, addressing bad convergence in the early training stage.
///
/// Reference: "On the Variance of the Adaptive Learning Rate and Beyond"
///
/// Update rule (per parameter element, at step `t`):
/// ```text
/// m = beta1*m + (1-beta1)*grad
/// v = beta2*v + (1-beta2)*grad^2
/// m_hat = m / (1 - beta1^t)
/// v_hat = v / (1 - beta2^t)
/// rho_inf = 2/(1-beta2) - 1
/// rho_t   = rho_inf - 2*t*beta2^t/(1-beta2^t)
/// if rho_t > 4:
///     r_t = sqrt( ((rho_t-4)*(rho_t-2)*rho_inf) / ((rho_inf-4)*(rho_inf-2)*rho_t) )
///     w -= lr * r_t * m_hat / (sqrt(v_hat) + epsilon)
/// else:
///     w -= lr * m_hat
/// ```
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
    state: HashMap<usize, AdamMomentState>,
}
#[pymethods]
impl PyRAdam {
    /// Create a new RAdam optimizer with default parameters
    #[new]
    #[pyo3(signature = (learning_rate = None))]
    pub fn new(learning_rate: Option<f64>) -> Self {
        Self {
            learning_rate: learning_rate.unwrap_or(0.001),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            timestep: 0,
            state: HashMap::new(),
        }
    }
    /// Create RAdam optimizer with custom beta parameters
    #[staticmethod]
    #[pyo3(signature = (learning_rate, beta1 = 0.9, beta2 = 0.999))]
    pub fn with_betas(learning_rate: f64, beta1: f64, beta2: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon: 1e-8,
            weight_decay: 0.0,
            timestep: 0,
            state: HashMap::new(),
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
    pub fn step(&mut self, model: Bound<'_, PyAny>) -> PyResult<()> {
        let py = model.py();
        let params = collect_parameters(&model)?;
        self.timestep += 1;
        let t = self.timestep as i32;
        let lr = self.learning_rate as f32;
        let beta1 = self.beta1 as f32;
        let beta2 = self.beta2 as f32;
        let epsilon = self.epsilon as f32;
        let weight_decay = self.weight_decay as f32;
        let bias_correction1 = 1.0 - self.beta1.powi(t) as f32;
        let beta2_pow_t = self.beta2.powi(t);
        let bias_correction2 = 1.0 - beta2_pow_t as f32;
        let rho_inf = 2.0 / (1.0 - self.beta2) - 1.0;
        let rho_t = rho_inf - 2.0 * (t as f64) * beta2_pow_t / (1.0 - beta2_pow_t);
        let rectified_r_t = if rho_t > 4.0 {
            Some(
                (((rho_t - 4.0) * (rho_t - 2.0) * rho_inf)
                    / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                    .sqrt() as f32,
            )
        } else {
            None
        };
        for param in &params {
            let param_ref = param.borrow(py);
            let Some(ParamStepInputs {
                mut value,
                mut grad,
                shape,
            }) = param_value_and_grad(&param_ref)?
            else {
                continue;
            };
            if weight_decay != 0.0 {
                for (g, w) in grad.iter_mut().zip(value.iter()) {
                    *g += weight_decay * *w;
                }
            }
            let n = value.len();
            let param_state = self
                .state
                .entry(param_ref.id())
                .or_insert_with(|| AdamMomentState::zeros(n));
            for i in 0..n {
                let g = grad[i];
                param_state.m[i] = beta1 * param_state.m[i] + (1.0 - beta1) * g;
                param_state.v[i] = beta2 * param_state.v[i] + (1.0 - beta2) * g * g;
                let m_hat = param_state.m[i] / bias_correction1;
                if let Some(r_t) = rectified_r_t {
                    let v_hat = param_state.v[i] / bias_correction2;
                    value[i] -= lr * r_t * m_hat / (v_hat.sqrt() + epsilon);
                } else {
                    value[i] -= lr * m_hat;
                }
            }
            drop(param_ref);
            write_back(&param.borrow(py), value, &shape)?;
        }
        Ok(())
    }
    /// Zero out gradients for all parameters
    pub fn zero_grad(&self, model: Bound<'_, PyAny>) -> PyResult<()> {
        let py = model.py();
        zero_grad_all(&model, py)
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
            self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay,
            self.timestep
        )
    }
}
/// Python wrapper for the Nadam optimizer
///
/// Nadam combines Adam with Nesterov momentum for improved convergence.
///
/// Reference: Dozat, "Incorporating Nesterov Momentum into Adam" (ICLR 2016 workshop)
///
/// Update rule (per parameter element, at step `t`, using a constant `beta1`
/// schedule rather than the paper's optional momentum-decay schedule):
/// ```text
/// m = beta1*m + (1-beta1)*grad
/// v = beta2*v + (1-beta2)*grad^2
/// m_hat_next = beta1*m / (1 - beta1^(t+1))     // one-step Nesterov lookahead
/// g_hat      = (1-beta1)*grad / (1 - beta1^t)  // immediate-gradient bias correction
/// m_hat = m_hat_next + g_hat
/// v_hat = v / (1 - beta2^t)
/// w -= lr * m_hat / (sqrt(v_hat) + epsilon)
/// ```
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
    state: HashMap<usize, AdamMomentState>,
}
#[pymethods]
impl PyNadam {
    /// Create a new Nadam optimizer with default parameters
    #[new]
    #[pyo3(signature = (learning_rate = None))]
    pub fn new(learning_rate: Option<f64>) -> Self {
        Self {
            learning_rate: learning_rate.unwrap_or(0.001),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            timestep: 0,
            state: HashMap::new(),
        }
    }
    /// Create Nadam optimizer with custom beta parameters
    #[staticmethod]
    #[pyo3(signature = (learning_rate, beta1 = 0.9, beta2 = 0.999))]
    pub fn with_betas(learning_rate: f64, beta1: f64, beta2: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon: 1e-8,
            weight_decay: 0.0,
            timestep: 0,
            state: HashMap::new(),
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
    pub fn step(&mut self, model: Bound<'_, PyAny>) -> PyResult<()> {
        let py = model.py();
        let params = collect_parameters(&model)?;
        self.timestep += 1;
        let t = self.timestep as i32;
        let lr = self.learning_rate as f32;
        let beta1 = self.beta1 as f32;
        let beta2 = self.beta2 as f32;
        let epsilon = self.epsilon as f32;
        let weight_decay = self.weight_decay as f32;
        let bias_correction1_t = 1.0 - self.beta1.powi(t) as f32;
        let bias_correction1_t1 = 1.0 - self.beta1.powi(t + 1) as f32;
        let bias_correction2 = 1.0 - self.beta2.powi(t) as f32;
        for param in &params {
            let param_ref = param.borrow(py);
            let Some(ParamStepInputs {
                mut value,
                mut grad,
                shape,
            }) = param_value_and_grad(&param_ref)?
            else {
                continue;
            };
            if weight_decay != 0.0 {
                for (g, w) in grad.iter_mut().zip(value.iter()) {
                    *g += weight_decay * *w;
                }
            }
            let n = value.len();
            let param_state = self
                .state
                .entry(param_ref.id())
                .or_insert_with(|| AdamMomentState::zeros(n));
            for i in 0..n {
                let g = grad[i];
                param_state.m[i] = beta1 * param_state.m[i] + (1.0 - beta1) * g;
                param_state.v[i] = beta2 * param_state.v[i] + (1.0 - beta2) * g * g;
                let m_hat_next = beta1 * param_state.m[i] / bias_correction1_t1;
                let g_hat = (1.0 - beta1) * g / bias_correction1_t;
                let m_hat = m_hat_next + g_hat;
                let v_hat = param_state.v[i] / bias_correction2;
                value[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
            }
            drop(param_ref);
            write_back(&param.borrow(py), value, &shape)?;
        }
        Ok(())
    }
    /// Zero out gradients for all parameters
    pub fn zero_grad(&self, model: Bound<'_, PyAny>) -> PyResult<()> {
        let py = model.py();
        zero_grad_all(&model, py)
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
            self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay,
            self.timestep
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
/// Update rule (per parameter element, at step `t`):
/// ```text
/// accum += grad^2
/// effective_lr = learning_rate / (1 + t*lr_decay)
/// w -= effective_lr * grad / (sqrt(accum) + epsilon)
/// ```
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
    state: HashMap<usize, Vec<f32>>,
}
#[pymethods]
impl PyAdaGrad {
    /// Create a new AdaGrad optimizer with default parameters
    #[new]
    #[pyo3(signature = (learning_rate = None))]
    pub fn new(learning_rate: Option<f64>) -> Self {
        Self {
            learning_rate: learning_rate.unwrap_or(0.01),
            epsilon: 1e-10,
            weight_decay: 0.0,
            lr_decay: 0.0,
            timestep: 0,
            state: HashMap::new(),
        }
    }
    /// Create AdaGrad optimizer with custom epsilon
    #[staticmethod]
    #[pyo3(signature = (learning_rate, epsilon = 1e-10))]
    pub fn with_epsilon(learning_rate: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            epsilon,
            weight_decay: 0.0,
            lr_decay: 0.0,
            timestep: 0,
            state: HashMap::new(),
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
            state: HashMap::new(),
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
    pub fn step(&mut self, model: Bound<'_, PyAny>) -> PyResult<()> {
        let py = model.py();
        let params = collect_parameters(&model)?;
        self.timestep += 1;
        let weight_decay = self.weight_decay as f32;
        let epsilon = self.epsilon as f32;
        let effective_lr =
            (self.learning_rate / (1.0 + self.timestep as f64 * self.lr_decay)) as f32;
        for param in &params {
            let param_ref = param.borrow(py);
            let Some(ParamStepInputs {
                mut value,
                mut grad,
                shape,
            }) = param_value_and_grad(&param_ref)?
            else {
                continue;
            };
            if weight_decay != 0.0 {
                for (g, w) in grad.iter_mut().zip(value.iter()) {
                    *g += weight_decay * *w;
                }
            }
            let n = value.len();
            let accum = self
                .state
                .entry(param_ref.id())
                .or_insert_with(|| vec![0.0; n]);
            for i in 0..n {
                let g = grad[i];
                accum[i] += g * g;
                value[i] -= effective_lr * g / (accum[i].sqrt() + epsilon);
            }
            drop(param_ref);
            write_back(&param.borrow(py), value, &shape)?;
        }
        Ok(())
    }
    /// Zero out gradients for all parameters
    pub fn zero_grad(&self, model: Bound<'_, PyAny>) -> PyResult<()> {
        let py = model.py();
        zero_grad_all(&model, py)
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
/// Per-parameter state tracked by [`PyAdaDelta`]: running average of squared
/// gradients, and running average of squared parameter updates.
#[derive(Debug, Clone)]
struct AdaDeltaState {
    accum_grad: Vec<f32>,
    accum_update: Vec<f32>,
}
impl AdaDeltaState {
    fn zeros(n: usize) -> Self {
        Self {
            accum_grad: vec![0.0; n],
            accum_update: vec![0.0; n],
        }
    }
}
/// Python wrapper for the AdaDelta optimizer
///
/// AdaDelta is an extension of AdaGrad that seeks to reduce its aggressive,
/// monotonically decreasing learning rate by restricting the accumulation window.
///
/// Reference: "ADADELTA: An Adaptive Learning Rate Method"
///
/// Update rule (per parameter element; note there is no explicit learning rate —
/// the step size is entirely derived from `rho` and the two running averages):
/// ```text
/// accum_grad = rho*accum_grad + (1-rho)*grad^2
/// update = grad * sqrt(accum_update + epsilon) / sqrt(accum_grad + epsilon)
/// accum_update = rho*accum_update + (1-rho)*update^2
/// w -= update
/// ```
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
    state: HashMap<usize, AdaDeltaState>,
}
#[pymethods]
impl PyAdaDelta {
    /// Create a new AdaDelta optimizer with default parameters
    #[new]
    #[pyo3(signature = (rho = None))]
    pub fn new(rho: Option<f64>) -> Self {
        Self {
            rho: rho.unwrap_or(0.9),
            epsilon: 1e-6,
            weight_decay: 0.0,
            timestep: 0,
            state: HashMap::new(),
        }
    }
    /// Create AdaDelta optimizer with custom epsilon
    #[staticmethod]
    #[pyo3(signature = (rho, epsilon = 1e-6))]
    pub fn with_epsilon(rho: f64, epsilon: f64) -> Self {
        Self {
            rho,
            epsilon,
            weight_decay: 0.0,
            timestep: 0,
            state: HashMap::new(),
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
            state: HashMap::new(),
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
    pub fn step(&mut self, model: Bound<'_, PyAny>) -> PyResult<()> {
        let py = model.py();
        let params = collect_parameters(&model)?;
        self.timestep += 1;
        let rho = self.rho as f32;
        let epsilon = self.epsilon as f32;
        let weight_decay = self.weight_decay as f32;
        for param in &params {
            let param_ref = param.borrow(py);
            let Some(ParamStepInputs {
                mut value,
                mut grad,
                shape,
            }) = param_value_and_grad(&param_ref)?
            else {
                continue;
            };
            if weight_decay != 0.0 {
                for (g, w) in grad.iter_mut().zip(value.iter()) {
                    *g += weight_decay * *w;
                }
            }
            let n = value.len();
            let param_state = self
                .state
                .entry(param_ref.id())
                .or_insert_with(|| AdaDeltaState::zeros(n));
            for i in 0..n {
                let g = grad[i];
                param_state.accum_grad[i] = rho * param_state.accum_grad[i] + (1.0 - rho) * g * g;
                let update = g * (param_state.accum_update[i] + epsilon).sqrt()
                    / (param_state.accum_grad[i] + epsilon).sqrt();
                param_state.accum_update[i] =
                    rho * param_state.accum_update[i] + (1.0 - rho) * update * update;
                value[i] -= update;
            }
            drop(param_ref);
            write_back(&param.borrow(py), value, &shape)?;
        }
        Ok(())
    }
    /// Zero out gradients for all parameters
    pub fn zero_grad(&self, model: Bound<'_, PyAny>) -> PyResult<()> {
        let py = model.py();
        zero_grad_all(&model, py)
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

#[cfg(test)]
mod tests;
