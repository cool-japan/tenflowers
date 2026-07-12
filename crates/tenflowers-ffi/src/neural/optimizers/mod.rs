//! Optimizer implementations for neural network training
//!
//! This module provides Python bindings for various optimizers including Adam, SGD, RMSprop, etc.
//!
//! # Update rules
//!
//! Every optimizer below mirrors PyTorch's semantics as closely as this crate's
//! primitives allow:
//!
//! * [`PySGD::step`] — `w <- w - lr * (grad + weight_decay * w)`, with optional
//!   classic heavy-ball momentum: `v <- momentum * v + (grad + weight_decay * w);
//!   w <- w - lr * v`.
//! * [`PyAdam::step`] — bias-corrected first/second moment estimates (Kingma &
//!   Ba, 2014): `m <- beta1*m + (1-beta1)*g; v <- beta2*v + (1-beta2)*g^2;
//!   m_hat <- m/(1-beta1^t); v_hat <- v/(1-beta2^t);
//!   w <- w - lr * m_hat / (sqrt(v_hat) + epsilon)`, with `weight_decay` folded
//!   into the gradient *before* the moment update (classic Adam+L2, not
//!   decoupled — see [`PyAdamW::step`] for the decoupled variant).
//! * [`PyRMSprop::step`] — `cache <- alpha*cache + (1-alpha)*g^2;
//!   w <- w - lr * g / (sqrt(cache) + epsilon)`.
//! * [`PyAdamW::step`] — identical moment estimates to [`PyAdam::step`], but
//!   with **decoupled** weight decay applied as a separate multiplicative
//!   shrink of the weight itself (`w <- w - lr * weight_decay * w`), never
//!   folded into the gradient — this decoupling (Loshchilov & Hutter, 2019) is
//!   the entire point of AdamW over plain Adam + L2 regularization.
//!
//! # Per-parameter optimizer state
//!
//! Adam/AdamW/RMSprop need per-parameter buffers (moment estimates / squared-
//! gradient caches) that must persist across `.step()` calls. Each such
//! optimizer stores a `state: HashMap<usize, ParamState>` keyed by
//! [`super::layers::PyParameter::id`] (stable across `set_data` updates — see
//! that method's doc), lazily initialized to zero-tensors matching the
//! parameter's shape the first time that `id` is seen, and never reset except
//! by dropping/replacing the optimizer itself. SGD's momentum buffer follows
//! the same pattern when `momentum` is configured.
//!
//! # Parameters with no gradient are silently skipped, not an error
//!
//! [`crate::neural::collect_parameters`] can return a parameter whose
//! `.grad()` has never been populated — e.g. `requires_grad=False`, or the
//! parameter simply did not participate in the particular loss this
//! `.step()` follows (common in multi-head / multi-loss models where not
//! every parameter feeds every loss). Mirroring PyTorch's own
//! `Optimizer.step()` (which skips any `p` with `p.grad is None` rather than
//! raising), every `step()` below treats a `.grad()` error as "nothing to do
//! for this parameter this round" and moves on, rather than aborting the
//! whole step. This is the least-surprising choice: a caller building a model
//! with, say, an auxiliary head that is only sometimes in the loss should not
//! have every other parameter's update blocked by that head's occasionally-
//! absent gradient.

use pyo3::prelude::*;
use std::collections::HashMap;
use tenflowers_core::Tensor;

use super::layers::PyParameter;

/// Read `param`'s current value and gradient as raw `Tensor<f32>`s.
///
/// Returns `Ok(None)` (not `Err`) when `param` has no gradient yet — see the
/// module-level "Parameters with no gradient are silently skipped" doc for
/// why this is deliberately not an error a `step()` propagates. Returns `Err`
/// only for a genuine failure unrelated to "no gradient" (e.g.
/// [`PyParameter::to_tensor`]'s lock-poisoned case), which a `step()` should
/// still propagate rather than silently swallow.
fn read_value_and_grad(param: &PyParameter) -> PyResult<Option<(Tensor<f32>, Tensor<f32>)>> {
    let grad = match param.grad() {
        Ok(grad) => grad,
        Err(_) => return Ok(None),
    };
    let value = param.to_tensor()?;
    Ok(Some(((*value.tensor).clone(), (*grad.tensor).clone())))
}

/// Per-parameter state for [`PySGD`]: the momentum buffer, present only when
/// the optimizer was constructed with `momentum` configured.
#[derive(Debug, Clone)]
struct SgdParamState {
    /// Velocity buffer `v` in `v <- momentum * v + grad_with_decay`.
    velocity: Tensor<f32>,
}

/// Per-parameter state for [`PyAdam`] / [`PyAdamW`]: first and second raw
/// moment estimates.
#[derive(Debug, Clone)]
struct AdamParamState {
    /// First moment estimate `m`.
    m: Tensor<f32>,
    /// Second (raw, uncentered) moment estimate `v`.
    v: Tensor<f32>,
}

/// Per-parameter state for [`PyRMSprop`]: the squared-gradient running
/// average.
#[derive(Debug, Clone)]
struct RmspropParamState {
    /// Squared-gradient exponential moving average.
    cache: Tensor<f32>,
}

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
    /// Per-parameter first/second moment estimates, keyed by
    /// [`PyParameter::id`]. See the module-level "Per-parameter optimizer
    /// state" doc.
    state: HashMap<usize, AdamParamState>,
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
            state: HashMap::new(),
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
            state: HashMap::new(),
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
            state: HashMap::new(),
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
            state: HashMap::new(),
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
    /// Applies the bias-corrected Adam update rule (see the module-level
    /// doc) to every parameter [`crate::neural::collect_parameters`] finds on
    /// `model` that currently has a gradient; parameters with no gradient are
    /// silently skipped (see the module-level "Parameters with no gradient"
    /// doc).
    ///
    /// # Errors
    ///
    /// Propagates any error from [`crate::neural::collect_parameters`] (e.g.
    /// `model` has no `.parameters()` method) or from
    /// [`PyParameter::set_data`] (e.g. a poisoned lock).
    pub fn step(&mut self, model: Bound<'_, PyAny>) -> PyResult<()> {
        self.timestep += 1;
        let t = self.timestep as f32;
        let beta1 = self.beta1 as f32;
        let beta2 = self.beta2 as f32;
        let epsilon = self.epsilon as f32;
        let lr = self.learning_rate as f32;
        let weight_decay = self.weight_decay as f32;

        let py = model.py();
        let params = crate::neural::collect_parameters(&model)?;

        for param in &params {
            let param_ref = param.borrow(py);
            let Some((w, mut g)) = read_value_and_grad(&param_ref)? else {
                continue;
            };

            // Classic (non-decoupled) Adam weight decay: fold into the
            // gradient before the moment update.
            if weight_decay != 0.0 {
                let decay_term = w.scalar_mul(weight_decay).map_err(to_py_err)?;
                g = g.add(&decay_term).map_err(to_py_err)?;
            }

            let id = param_ref.id();
            let entry = self.state.entry(id).or_insert_with(|| AdamParamState {
                m: Tensor::<f32>::zeros(w.shape().dims()),
                v: Tensor::<f32>::zeros(w.shape().dims()),
            });

            // m <- beta1*m + (1-beta1)*g
            let m_scaled = entry.m.scalar_mul(beta1).map_err(to_py_err)?;
            let g_scaled = g.scalar_mul(1.0 - beta1).map_err(to_py_err)?;
            entry.m = m_scaled.add(&g_scaled).map_err(to_py_err)?;

            // v <- beta2*v + (1-beta2)*g^2
            let g_sq = g.mul(&g).map_err(to_py_err)?;
            let v_scaled = entry.v.scalar_mul(beta2).map_err(to_py_err)?;
            let g_sq_scaled = g_sq.scalar_mul(1.0 - beta2).map_err(to_py_err)?;
            entry.v = v_scaled.add(&g_sq_scaled).map_err(to_py_err)?;

            // Bias correction.
            let bias_correction1 = 1.0 - beta1.powf(t);
            let bias_correction2 = 1.0 - beta2.powf(t);
            let m_hat = entry
                .m
                .scalar_mul(1.0 / bias_correction1)
                .map_err(to_py_err)?;
            let v_hat = entry
                .v
                .scalar_mul(1.0 / bias_correction2)
                .map_err(to_py_err)?;

            // w <- w - lr * m_hat / (sqrt(v_hat) + epsilon)
            let v_hat_sqrt = v_hat.sqrt().map_err(to_py_err)?;
            let epsilon_tensor = Tensor::<f32>::full(v_hat_sqrt.shape().dims(), epsilon);
            let denom = v_hat_sqrt.add(&epsilon_tensor).map_err(to_py_err)?;
            let step_dir = m_hat.div(&denom).map_err(to_py_err)?;
            let update = step_dir.scalar_mul(lr).map_err(to_py_err)?;
            let new_w = w.sub(&update).map_err(to_py_err)?;

            param_ref.set_data(new_w)?;
        }

        Ok(())
    }

    /// Zero out gradients for all parameters
    ///
    /// Args:
    ///     model: Model containing parameters to zero gradients for
    ///
    /// This should be called before backward pass to clear accumulated gradients.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`crate::neural::collect_parameters`].
    pub fn zero_grad(&self, model: Bound<'_, PyAny>) -> PyResult<()> {
        zero_grad_all(model)
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

/// Python wrapper for the SGD (Stochastic Gradient Descent) optimizer
///
/// SGD is a simple but effective optimizer that updates parameters using gradients
/// with optional momentum for accelerated convergence.
///
/// Parameters:
/// * learning_rate: Step size for parameter updates (default: 0.01)
/// * momentum: Momentum factor for accelerated convergence (default: None)
/// * weight_decay: Weight decay (L2 penalty) coefficient (default: 0.0)
#[pyclass(name = "SGD")]
#[derive(Debug, Clone)]
pub struct PySGD {
    pub learning_rate: f64,
    pub momentum: Option<f64>,
    pub weight_decay: f64,
    /// Per-parameter momentum buffers, keyed by [`PyParameter::id`]. Only
    /// populated (and only consulted) when `momentum` is `Some`. See the
    /// module-level "Per-parameter optimizer state" doc.
    state: HashMap<usize, SgdParamState>,
}

#[pymethods]
impl PySGD {
    /// Create a new SGD optimizer with default parameters
    ///
    /// Args:
    ///     learning_rate: Optional learning rate (default: 0.01)
    ///
    /// Returns:
    ///     New SGD optimizer instance
    #[new]
    #[pyo3(signature = (learning_rate=None))]
    pub fn new(learning_rate: Option<f64>) -> Self {
        Self {
            learning_rate: learning_rate.unwrap_or(0.01),
            momentum: None,
            weight_decay: 0.0,
            state: HashMap::new(),
        }
    }

    /// Create SGD optimizer with momentum
    ///
    /// Args:
    ///     learning_rate: Learning rate for parameter updates
    ///     momentum: Momentum factor (typically 0.9)
    ///
    /// Returns:
    ///     SGD optimizer with momentum
    #[staticmethod]
    #[pyo3(signature = (learning_rate, momentum=0.9))]
    pub fn with_momentum(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum: Some(momentum),
            weight_decay: 0.0,
            state: HashMap::new(),
        }
    }

    /// Create SGD optimizer with weight decay
    ///
    /// Args:
    ///     learning_rate: Learning rate for parameter updates
    ///     weight_decay: L2 penalty coefficient
    ///
    /// Returns:
    ///     SGD optimizer with weight decay
    #[staticmethod]
    #[pyo3(signature = (learning_rate, weight_decay))]
    pub fn with_weight_decay(learning_rate: f64, weight_decay: f64) -> Self {
        Self {
            learning_rate,
            momentum: None,
            weight_decay,
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
    ///
    /// Applies `w <- w - lr * (grad + weight_decay * w)`, or — when
    /// `momentum` is configured — classic heavy-ball momentum:
    /// `v <- momentum * v + (grad + weight_decay * w); w <- w - lr * v`. See
    /// the module-level doc for the full rule and the "Parameters with no
    /// gradient" skip semantics.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`crate::neural::collect_parameters`] or
    /// [`PyParameter::set_data`].
    pub fn step(&mut self, model: Bound<'_, PyAny>) -> PyResult<()> {
        let lr = self.learning_rate as f32;
        let weight_decay = self.weight_decay as f32;
        let momentum = self.momentum.map(|m| m as f32);

        let py = model.py();
        let params = crate::neural::collect_parameters(&model)?;

        for param in &params {
            let param_ref = param.borrow(py);
            let Some((w, g)) = read_value_and_grad(&param_ref)? else {
                continue;
            };

            let mut grad_with_decay = g;
            if weight_decay != 0.0 {
                let decay_term = w.scalar_mul(weight_decay).map_err(to_py_err)?;
                grad_with_decay = grad_with_decay.add(&decay_term).map_err(to_py_err)?;
            }

            let new_w = if let Some(momentum) = momentum {
                let id = param_ref.id();
                let entry = self.state.entry(id).or_insert_with(|| SgdParamState {
                    velocity: Tensor::<f32>::zeros(w.shape().dims()),
                });

                // v <- momentum * v + grad_with_decay
                let v_scaled = entry.velocity.scalar_mul(momentum).map_err(to_py_err)?;
                entry.velocity = v_scaled.add(&grad_with_decay).map_err(to_py_err)?;

                let update = entry.velocity.scalar_mul(lr).map_err(to_py_err)?;
                w.sub(&update).map_err(to_py_err)?
            } else {
                let update = grad_with_decay.scalar_mul(lr).map_err(to_py_err)?;
                w.sub(&update).map_err(to_py_err)?
            };

            param_ref.set_data(new_w)?;
        }

        Ok(())
    }

    /// Zero out gradients for all parameters
    ///
    /// # Errors
    ///
    /// Propagates any error from [`crate::neural::collect_parameters`].
    pub fn zero_grad(&self, model: Bound<'_, PyAny>) -> PyResult<()> {
        zero_grad_all(model)
    }

    /// Get optimizer state information
    pub fn state_dict(&self) -> HashMap<String, f64> {
        let mut state = HashMap::new();
        state.insert("learning_rate".to_string(), self.learning_rate);
        if let Some(momentum) = self.momentum {
            state.insert("momentum".to_string(), momentum);
        }
        state.insert("weight_decay".to_string(), self.weight_decay);
        state
    }

    /// Load optimizer state from dictionary
    pub fn load_state_dict(&mut self, state_dict: HashMap<String, f64>) {
        if let Some(&lr) = state_dict.get("learning_rate") {
            self.learning_rate = lr;
        }
        if let Some(&momentum) = state_dict.get("momentum") {
            self.momentum = Some(momentum);
        }
        if let Some(&weight_decay) = state_dict.get("weight_decay") {
            self.weight_decay = weight_decay;
        }
    }

    /// String representation
    pub fn __str__(&self) -> String {
        format!(
            "SGD(learning_rate={}, momentum={:?}, weight_decay={})",
            self.learning_rate, self.momentum, self.weight_decay
        )
    }

    /// Detailed string representation
    pub fn __repr__(&self) -> String {
        format!(
            "PySGD(learning_rate={}, momentum={:?}, weight_decay={})",
            self.learning_rate, self.momentum, self.weight_decay
        )
    }
}

/// Python wrapper for the RMSprop optimizer
///
/// RMSprop is an adaptive learning rate optimizer that divides the learning rate
/// by an exponentially decaying average of squared gradients.
///
/// Parameters:
/// * learning_rate: Step size for parameter updates (default: 0.001)
/// * alpha: Smoothing constant for moving average (default: 0.99)
/// * epsilon: Small constant to prevent division by zero (default: 1e-8)
/// * weight_decay: Weight decay (L2 penalty) coefficient (default: 0.0)
#[pyclass(name = "RMSprop")]
#[derive(Debug, Clone)]
pub struct PyRMSprop {
    pub learning_rate: f64,
    pub alpha: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    /// Per-parameter squared-gradient caches, keyed by [`PyParameter::id`].
    /// See the module-level "Per-parameter optimizer state" doc.
    state: HashMap<usize, RmspropParamState>,
}

#[pymethods]
impl PyRMSprop {
    /// Create a new RMSprop optimizer with default parameters
    ///
    /// Args:
    ///     learning_rate: Optional learning rate (default: 0.001)
    ///
    /// Returns:
    ///     New RMSprop optimizer instance
    #[new]
    #[pyo3(signature = (learning_rate=None))]
    pub fn new(learning_rate: Option<f64>) -> Self {
        Self {
            learning_rate: learning_rate.unwrap_or(0.001),
            alpha: 0.99,
            epsilon: 1e-8,
            weight_decay: 0.0,
            state: HashMap::new(),
        }
    }

    /// Create RMSprop optimizer with custom alpha
    ///
    /// Args:
    ///     learning_rate: Learning rate for parameter updates
    ///     alpha: Smoothing constant (typically 0.9-0.99)
    ///
    /// Returns:
    ///     RMSprop optimizer with custom alpha
    #[staticmethod]
    #[pyo3(signature = (learning_rate, alpha=0.99))]
    pub fn with_alpha(learning_rate: f64, alpha: f64) -> Self {
        Self {
            learning_rate,
            alpha,
            epsilon: 1e-8,
            weight_decay: 0.0,
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
    ///
    /// Applies `cache <- alpha*cache + (1-alpha)*grad^2;
    /// w <- w - lr * grad / (sqrt(cache) + epsilon)`. See the module-level
    /// doc for the full rule and the "Parameters with no gradient" skip
    /// semantics.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`crate::neural::collect_parameters`] or
    /// [`PyParameter::set_data`].
    pub fn step(&mut self, model: Bound<'_, PyAny>) -> PyResult<()> {
        let lr = self.learning_rate as f32;
        let alpha = self.alpha as f32;
        let epsilon = self.epsilon as f32;
        let weight_decay = self.weight_decay as f32;

        let py = model.py();
        let params = crate::neural::collect_parameters(&model)?;

        for param in &params {
            let param_ref = param.borrow(py);
            let Some((w, mut g)) = read_value_and_grad(&param_ref)? else {
                continue;
            };

            if weight_decay != 0.0 {
                let decay_term = w.scalar_mul(weight_decay).map_err(to_py_err)?;
                g = g.add(&decay_term).map_err(to_py_err)?;
            }

            let id = param_ref.id();
            let entry = self.state.entry(id).or_insert_with(|| RmspropParamState {
                cache: Tensor::<f32>::zeros(w.shape().dims()),
            });

            // cache <- alpha*cache + (1-alpha)*g^2
            let g_sq = g.mul(&g).map_err(to_py_err)?;
            let cache_scaled = entry.cache.scalar_mul(alpha).map_err(to_py_err)?;
            let g_sq_scaled = g_sq.scalar_mul(1.0 - alpha).map_err(to_py_err)?;
            entry.cache = cache_scaled.add(&g_sq_scaled).map_err(to_py_err)?;

            // w <- w - lr * g / (sqrt(cache) + epsilon)
            let cache_sqrt = entry.cache.sqrt().map_err(to_py_err)?;
            let epsilon_tensor = Tensor::<f32>::full(cache_sqrt.shape().dims(), epsilon);
            let denom = cache_sqrt.add(&epsilon_tensor).map_err(to_py_err)?;
            let step_dir = g.div(&denom).map_err(to_py_err)?;
            let update = step_dir.scalar_mul(lr).map_err(to_py_err)?;
            let new_w = w.sub(&update).map_err(to_py_err)?;

            param_ref.set_data(new_w)?;
        }

        Ok(())
    }

    /// Zero out gradients for all parameters
    ///
    /// # Errors
    ///
    /// Propagates any error from [`crate::neural::collect_parameters`].
    pub fn zero_grad(&self, model: Bound<'_, PyAny>) -> PyResult<()> {
        zero_grad_all(model)
    }

    /// Get optimizer state information
    pub fn state_dict(&self) -> HashMap<String, f64> {
        let mut state = HashMap::new();
        state.insert("learning_rate".to_string(), self.learning_rate);
        state.insert("alpha".to_string(), self.alpha);
        state.insert("epsilon".to_string(), self.epsilon);
        state.insert("weight_decay".to_string(), self.weight_decay);
        state
    }

    /// Load optimizer state from dictionary
    pub fn load_state_dict(&mut self, state_dict: HashMap<String, f64>) {
        if let Some(&lr) = state_dict.get("learning_rate") {
            self.learning_rate = lr;
        }
        if let Some(&alpha) = state_dict.get("alpha") {
            self.alpha = alpha;
        }
        if let Some(&epsilon) = state_dict.get("epsilon") {
            self.epsilon = epsilon;
        }
        if let Some(&weight_decay) = state_dict.get("weight_decay") {
            self.weight_decay = weight_decay;
        }
    }

    /// String representation
    pub fn __str__(&self) -> String {
        format!(
            "RMSprop(learning_rate={}, alpha={}, epsilon={})",
            self.learning_rate, self.alpha, self.epsilon
        )
    }

    /// Detailed string representation
    pub fn __repr__(&self) -> String {
        format!(
            "PyRMSprop(learning_rate={}, alpha={}, epsilon={}, weight_decay={})",
            self.learning_rate, self.alpha, self.epsilon, self.weight_decay
        )
    }
}

/// Python wrapper for the AdamW optimizer
///
/// AdamW is a variant of Adam with decoupled weight decay regularization,
/// which improves training stability and generalization.
///
/// Parameters:
/// * learning_rate: Step size for parameter updates (default: 0.001)
/// * beta1: Exponential decay rate for first moment estimates (default: 0.9)
/// * beta2: Exponential decay rate for second moment estimates (default: 0.999)
/// * epsilon: Small constant to prevent division by zero (default: 1e-8)
/// * weight_decay: Weight decay coefficient (default: 0.01)
#[pyclass(name = "AdamW")]
#[derive(Debug, Clone)]
pub struct PyAdamW {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub timestep: usize,
    /// Per-parameter first/second moment estimates, keyed by
    /// [`PyParameter::id`]. See the module-level "Per-parameter optimizer
    /// state" doc.
    state: HashMap<usize, AdamParamState>,
}

#[pymethods]
impl PyAdamW {
    /// Create a new AdamW optimizer with default parameters
    ///
    /// Args:
    ///     learning_rate: Optional learning rate (default: 0.001)
    ///
    /// Returns:
    ///     New AdamW optimizer instance
    #[new]
    #[pyo3(signature = (learning_rate=None))]
    pub fn new(learning_rate: Option<f64>) -> Self {
        Self {
            learning_rate: learning_rate.unwrap_or(0.001),
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01, // Default weight decay for AdamW
            timestep: 0,
            state: HashMap::new(),
        }
    }

    /// Create AdamW optimizer with custom beta parameters
    ///
    /// Args:
    ///     learning_rate: Learning rate for parameter updates
    ///     beta1: Exponential decay rate for first moment estimates
    ///     beta2: Exponential decay rate for second moment estimates
    ///
    /// Returns:
    ///     AdamW optimizer with custom beta values
    #[staticmethod]
    #[pyo3(signature = (learning_rate, beta1=0.9, beta2=0.999))]
    pub fn with_betas(learning_rate: f64, beta1: f64, beta2: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon: 1e-8,
            weight_decay: 0.01,
            timestep: 0,
            state: HashMap::new(),
        }
    }

    /// Create AdamW optimizer with custom weight decay
    ///
    /// Args:
    ///     learning_rate: Learning rate for parameter updates
    ///     weight_decay: Weight decay coefficient
    ///
    /// Returns:
    ///     AdamW optimizer with custom weight decay
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

    /// Clear the tape and reset timestep
    pub fn clear(&mut self) {
        self.timestep = 0;
    }

    /// Perform a single optimization step
    ///
    /// Identical moment-estimate bookkeeping to [`PyAdam::step`], but weight
    /// decay is applied as a **decoupled** shrink of the weight itself
    /// (`w <- w - lr * weight_decay * w`, applied separately from the
    /// gradient-based update) rather than folded into the gradient before
    /// the moment update — see the module-level doc for why this distinction
    /// is the entire point of AdamW. See also the "Parameters with no
    /// gradient" skip semantics documented at module level.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`crate::neural::collect_parameters`] or
    /// [`PyParameter::set_data`].
    pub fn step(&mut self, model: Bound<'_, PyAny>) -> PyResult<()> {
        self.timestep += 1;
        let t = self.timestep as f32;
        let beta1 = self.beta1 as f32;
        let beta2 = self.beta2 as f32;
        let epsilon = self.epsilon as f32;
        let lr = self.learning_rate as f32;
        let weight_decay = self.weight_decay as f32;

        let py = model.py();
        let params = crate::neural::collect_parameters(&model)?;

        for param in &params {
            let param_ref = param.borrow(py);
            let Some((w, g)) = read_value_and_grad(&param_ref)? else {
                continue;
            };

            // Decoupled weight decay: shrink the weight directly, NOT folded
            // into the gradient. This must happen on the *original* weight
            // value before the gradient-based update below, matching the
            // reference AdamW algorithm (Loshchilov & Hutter, 2019, Algorithm
            // 2): both terms are subtracted from the same `w_{t-1}` in the
            // same step, not composed sequentially.
            let decoupled_decay = if weight_decay != 0.0 {
                w.scalar_mul(lr * weight_decay).map_err(to_py_err)?
            } else {
                Tensor::<f32>::zeros(w.shape().dims())
            };

            let id = param_ref.id();
            let entry = self.state.entry(id).or_insert_with(|| AdamParamState {
                m: Tensor::<f32>::zeros(w.shape().dims()),
                v: Tensor::<f32>::zeros(w.shape().dims()),
            });

            // m <- beta1*m + (1-beta1)*g   (note: g here is the RAW
            // gradient, unlike PyAdam::step — AdamW never folds weight_decay
            // into the gradient at all).
            let m_scaled = entry.m.scalar_mul(beta1).map_err(to_py_err)?;
            let g_scaled = g.scalar_mul(1.0 - beta1).map_err(to_py_err)?;
            entry.m = m_scaled.add(&g_scaled).map_err(to_py_err)?;

            // v <- beta2*v + (1-beta2)*g^2
            let g_sq = g.mul(&g).map_err(to_py_err)?;
            let v_scaled = entry.v.scalar_mul(beta2).map_err(to_py_err)?;
            let g_sq_scaled = g_sq.scalar_mul(1.0 - beta2).map_err(to_py_err)?;
            entry.v = v_scaled.add(&g_sq_scaled).map_err(to_py_err)?;

            // Bias correction.
            let bias_correction1 = 1.0 - beta1.powf(t);
            let bias_correction2 = 1.0 - beta2.powf(t);
            let m_hat = entry
                .m
                .scalar_mul(1.0 / bias_correction1)
                .map_err(to_py_err)?;
            let v_hat = entry
                .v
                .scalar_mul(1.0 / bias_correction2)
                .map_err(to_py_err)?;

            // gradient-based update = lr * m_hat / (sqrt(v_hat) + epsilon)
            let v_hat_sqrt = v_hat.sqrt().map_err(to_py_err)?;
            let epsilon_tensor = Tensor::<f32>::full(v_hat_sqrt.shape().dims(), epsilon);
            let denom = v_hat_sqrt.add(&epsilon_tensor).map_err(to_py_err)?;
            let step_dir = m_hat.div(&denom).map_err(to_py_err)?;
            let grad_update = step_dir.scalar_mul(lr).map_err(to_py_err)?;

            // w <- w - grad_update - decoupled_decay  (both subtracted from
            // the same original w, per Algorithm 2 — see comment above).
            let new_w = w
                .sub(&grad_update)
                .map_err(to_py_err)?
                .sub(&decoupled_decay)
                .map_err(to_py_err)?;

            param_ref.set_data(new_w)?;
        }

        Ok(())
    }

    /// Zero out gradients for all parameters
    ///
    /// # Errors
    ///
    /// Propagates any error from [`crate::neural::collect_parameters`].
    pub fn zero_grad(&self, model: Bound<'_, PyAny>) -> PyResult<()> {
        zero_grad_all(model)
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
            "AdamW(learning_rate={}, beta1={}, beta2={}, weight_decay={})",
            self.learning_rate, self.beta1, self.beta2, self.weight_decay
        )
    }

    /// Detailed string representation
    pub fn __repr__(&self) -> String {
        format!("PyAdamW(learning_rate={}, beta1={}, beta2={}, epsilon={}, weight_decay={}, timestep={})",
               self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay, self.timestep)
    }
}

/// Shared `zero_grad` body for every optimizer in this module:
/// [`crate::neural::collect_parameters`] then [`PyParameter::zero_grad`] on
/// each.
///
/// # Errors
///
/// Propagates any error from [`crate::neural::collect_parameters`]. Clearing
/// an individual parameter's gradient
/// ([`crate::implicit_autograd::clear_grad_by_id`], via
/// [`PyParameter::zero_grad`]) is currently infallible, but that method
/// returns `PyResult` for forward-compatibility (see its own doc), so its
/// `Result` is still propagated with `?` here rather than discarded.
fn zero_grad_all(model: Bound<'_, PyAny>) -> PyResult<()> {
    let py = model.py();
    let params = crate::neural::collect_parameters(&model)?;
    for param in &params {
        param.borrow(py).zero_grad()?;
    }
    Ok(())
}

/// Convert a [`tenflowers_core::TensorError`] (this crate's own arithmetic
/// `Result` error type) into a `PyErr`, for use with `.map_err` after any
/// `Tensor<f32>` arithmetic call in this module.
fn to_py_err(err: tenflowers_core::TensorError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("optimizer update failed: {}", err))
}

#[cfg(test)]
mod tests;
