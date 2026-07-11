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
fn read_value_and_grad(
    param: &PyParameter,
) -> PyResult<Option<(Tensor<f32>, Tensor<f32>)>> {
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
            let m_hat = entry.m.scalar_mul(1.0 / bias_correction1).map_err(to_py_err)?;
            let v_hat = entry.v.scalar_mul(1.0 / bias_correction2).map_err(to_py_err)?;

            // w <- w - lr * m_hat / (sqrt(v_hat) + epsilon)
            let v_hat_sqrt = v_hat.sqrt().map_err(to_py_err)?;
            let epsilon_tensor = Tensor::<f32>::full(v_hat_sqrt.shape().dims(), epsilon);
            let denom = v_hat_sqrt.add(&epsilon_tensor).map_err(to_py_err)?;
            let step_dir = m_hat.div(&denom).map_err(to_py_err)?;
            let update = step_dir.scalar_mul(lr).map_err(to_py_err)?;
            let _new_w = w.sub(&update).map_err(to_py_err)?;

            // TEMPORARY NEGATIVE-CONTROL EDIT (orchestrator spot-check, will
            // be reverted immediately): write back the UNCHANGED weight
            // instead of the real update, to confirm the new convergence
            // test actually detects a no-op optimizer.
            param_ref.set_data(w)?;
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
            let m_hat = entry.m.scalar_mul(1.0 / bias_correction1).map_err(to_py_err)?;
            let v_hat = entry.v.scalar_mul(1.0 / bias_correction2).map_err(to_py_err)?;

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
mod tests {
    use super::*;
    use crate::implicit_autograd::{
        mark_leaf_param, record_and_link_binary, record_and_link_unary, run_backward,
        BinaryOpKind, UnaryOpKind,
    };
    use crate::tensor_ops::PyTensor;
    use std::sync::Arc;

    /// Reset all thread-local implicit-autograd state so tests in this
    /// module do not observe leaves/gradients left behind by a previous test
    /// on the same (possibly nextest-reused) OS thread. Deliberately
    /// duplicated from `implicit_autograd::tests::reset_for_test` rather than
    /// exposed as a shared `pub(crate)` helper: that function lives inside a
    /// private `#[cfg(test)] mod tests` and is not reachable from here, and
    /// introducing cross-module test-only coupling for four call sites is
    /// not worth it. This calls the same four fully-`pub` free functions
    /// (`crate::implicit_autograd::{mark_leaf_param, run_backward, ...}` are
    /// all `pub`, but the thread-local tables themselves are private to that
    /// module) indirectly: rather than reaching into
    /// `TRACKED_REGISTRY`/`LEAVES`/`GRAD_STORE`/`IDENTITY_ANCHORS` directly
    /// (impossible — they are private `thread_local!` statics), each test
    /// below constructs a **fresh** `PyParameter` (and therefore a fresh,
    /// process-unique `id`, since `PyParameter::id` is the address of its own
    /// freshly-allocated `RwLock`), so no test can observe another test's
    /// leaf/gradient state even without an explicit reset: distinct `id`s
    /// cannot collide in `GRAD_STORE`, and each test only ever calls
    /// `run_backward` on its own self-contained graph.
    fn fresh_parameter(data: Vec<f32>, shape: &[usize]) -> PyParameter {
        let tensor = Tensor::from_vec(data, shape).expect("test: tensor construction");
        PyParameter::new(
            PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: true,
                is_pinned: false,
            },
            Some(true),
        )
    }

    /// Drive a `PyParameter` through a real forward + backward pass so its
    /// `.grad()` becomes populated, exactly mirroring what `PyDense::forward`
    /// does today for its `weight_arc`/`bias_arc` fields (see
    /// `layers.rs::PyDense::forward`) — except keyed by `PyParameter::id`
    /// (via `mark_leaf_param`) rather than by `tensor_key` (via `mark_leaf`),
    /// since that is the path `PyParameter::grad()` actually reads from
    /// (`get_grad_by_id(self.id)`, not `get_grad`/`tensor_key` — see
    /// `layers.rs::PyParameter::grad`'s own doc).
    ///
    /// Builds `loss = sum(param * param)`, so `d(loss)/d(param) = 2 * param`
    /// — a simple, hand-verifiable closed form used by the SGD numeric-check
    /// test below.
    fn forward_and_backward_square_sum(param: &PyParameter) -> PyResult<()> {
        let snapshot = param.to_tensor()?;
        mark_leaf_param(&snapshot, param.id());

        let raw = tenflowers_core::ops::mul(&snapshot.tensor, &snapshot.tensor)
            .expect("test: mul must succeed");
        let squared = PyTensor {
            tensor: Arc::new(raw),
            requires_grad: true,
            is_pinned: false,
        };
        record_and_link_binary(BinaryOpKind::Mul, &snapshot, &snapshot, &squared)?;

        let summed = tenflowers_core::ops::sum(&squared.tensor, None, false)
            .expect("test: sum must succeed");
        let loss = PyTensor {
            tensor: Arc::new(summed),
            requires_grad: true,
            is_pinned: false,
        };
        record_and_link_unary(
            UnaryOpKind::Sum {
                axes: None,
                keepdims: false,
            },
            &squared,
            &loss,
        )?;

        run_backward(&loss)
    }

    /// Build a minimal Python model object exposing `.parameters() ->
    /// list[PyParameter]`, exactly mirroring
    /// `optimizer_bridge.rs`'s own `FakeModel` test fixture (that file's
    /// `extracts_parameters_from_a_model_exposing_real_parameter_objects`
    /// test) — `collect_parameters` works via duck typing, so a plain
    /// pure-Python class with no `tenflowers` base class is exactly the
    /// right fixture, and `PyDense::parameters()` cannot be used here yet:
    /// as of this writing it still returns `Vec<PyTensor>`, not
    /// `Vec<PyParameter>` (see `optimizer_bridge.rs`'s "Design tension"
    /// doc), so `collect_parameters` would reject it with a `PyTypeError`.
    fn make_single_param_model<'py>(
        py: Python<'py>,
        param: Py<PyParameter>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let code = std::ffi::CString::new(
            r#"
class FakeModel:
    def __init__(self, params):
        self._params = params
    def parameters(self):
        return list(self._params)
"#,
        )
        .expect("test source has no NUL bytes");
        let module = PyModule::from_code(
            py,
            code.as_c_str(),
            c"optimizers_test_module.py",
            c"optimizers_test_module",
        )?;
        let make_model = module.getattr("FakeModel")?;
        let params_list = pyo3::types::PyList::new(py, [param])?;
        make_model.call1((params_list,))
    }

    #[test]
    fn sgd_step_moves_weight_in_expected_direction_and_magnitude() {
        // Ensure the interpreter is initialized regardless of which other
        // tests (if any) have run before this one in the same process —
        // this test module must not rely on test execution order/another
        // module's test happening to initialize it first. Matches the
        // established idiom already used by e.g. `layers.rs`,
        // `normalization.rs`, `embedding.rs`'s own `#[test]` functions.
        Python::initialize();
        Python::attach(|py| {
            // loss = sum(w * w)  =>  d(loss)/dw = 2w. At w = [3.0, -2.0],
            // grad = [6.0, -4.0] exactly — a simple, hand-verifiable case.
            let param = fresh_parameter(vec![3.0, -2.0], &[2]);
            forward_and_backward_square_sum(&param).expect("test: backward must succeed");

            let grad_before = param.grad().expect("test: grad must be populated");
            let grad_data = grad_before
                .tensor
                .to_vec()
                .expect("test: grad data readable");
            assert_eq!(
                grad_data,
                vec![6.0, -4.0],
                "sanity check: d(sum(w*w))/dw = 2w"
            );

            let old_w = param.to_tensor().expect("test: to_tensor").tensor.to_vec()
                .expect("test: old weight readable");
            assert_eq!(old_w, vec![3.0, -2.0]);

            let py_param = Py::new(py, param).expect("test: Py::new");
            let model = make_single_param_model(py, py_param.clone_ref(py))
                .expect("test: model construction");

            let lr = 0.1_f64;
            let mut sgd = PySGD::new(Some(lr));
            sgd.step(model).expect("test: sgd.step must succeed");

            let new_w = py_param
                .borrow(py)
                .to_tensor()
                .expect("test: to_tensor after step")
                .tensor
                .to_vec()
                .expect("test: new weight readable");

            // Plain (no momentum, no weight decay) SGD: new_w = old_w - lr*grad.
            let expected = vec![
                3.0 - lr as f32 * 6.0,
                -2.0 - lr as f32 * (-4.0),
            ];
            assert_eq!(
                new_w, expected,
                "SGD update must exactly match w - lr*grad for plain SGD"
            );
            assert_ne!(new_w, old_w, "weight must actually change after step()");
        });
    }

    #[test]
    fn sgd_step_with_momentum_persists_velocity_across_two_steps() {
        Python::initialize();
        Python::attach(|py| {
            let param = fresh_parameter(vec![1.0, 1.0], &[2]);
            let py_param = Py::new(py, param).expect("test: Py::new");

            let lr = 0.1_f64;
            let momentum = 0.9_f64;
            let mut sgd = PySGD::with_momentum(lr, momentum);

            // First step.
            {
                let param_ref = py_param.borrow(py);
                forward_and_backward_square_sum(&param_ref).expect("test: first backward");
            }
            let model1 = make_single_param_model(py, py_param.clone_ref(py))
                .expect("test: model1 construction");
            sgd.step(model1).expect("test: first sgd.step");
            let w_after_1 = py_param
                .borrow(py)
                .to_tensor()
                .expect("test: to_tensor 1")
                .tensor
                .to_vec()
                .expect("test: w1 readable");
            // v1 = 0.9*0 + grad(2.0,2.0) = (2.0,2.0); w1 = 1.0 - 0.1*2.0 = 0.8
            assert!(
                (w_after_1[0] - 0.8).abs() < 1e-5,
                "expected w ~= 0.8 after first momentum step, got {:?}",
                w_after_1
            );

            // Second step: gradient recomputed at the NEW weight, velocity
            // buffer must carry over from the first step (this is exactly
            // what proves per-parameter state persists across `.step()`
            // calls, not just that a single step works).
            {
                let param_ref = py_param.borrow(py);
                forward_and_backward_square_sum(&param_ref).expect("test: second backward");
            }
            let model2 = make_single_param_model(py, py_param.clone_ref(py))
                .expect("test: model2 construction");
            sgd.step(model2).expect("test: second sgd.step");
            let w_after_2 = py_param
                .borrow(py)
                .to_tensor()
                .expect("test: to_tensor 2")
                .tensor
                .to_vec()
                .expect("test: w2 readable");

            // grad2 = 2*w_after_1 = 1.6; v2 = 0.9*2.0 + 1.6 = 3.4;
            // w2 = 0.8 - 0.1*3.4 = 0.46
            let expected_v2 = 0.9 * 2.0 + 2.0 * w_after_1[0];
            let expected_w2 = w_after_1[0] - 0.1 * expected_v2;
            assert!(
                (w_after_2[0] - expected_w2).abs() < 1e-4,
                "expected w ~= {} after second momentum step (velocity must persist), got {:?}",
                expected_w2,
                w_after_2
            );
        });
    }

    #[test]
    fn adam_step_changes_weight_and_second_step_works_with_persisted_state() {
        Python::initialize();
        Python::attach(|py| {
            let param = fresh_parameter(vec![2.0], &[1]);
            let py_param = Py::new(py, param).expect("test: Py::new");
            let mut adam = PyAdam::new(Some(0.1));

            let old_w = py_param
                .borrow(py)
                .to_tensor()
                .expect("test: to_tensor")
                .tensor
                .to_vec()
                .expect("test: old w readable");

            {
                let param_ref = py_param.borrow(py);
                forward_and_backward_square_sum(&param_ref).expect("test: first backward");
            }
            let model1 = make_single_param_model(py, py_param.clone_ref(py))
                .expect("test: model1 construction");
            adam.step(model1).expect("test: first adam.step");

            let w1 = py_param
                .borrow(py)
                .to_tensor()
                .expect("test: to_tensor 1")
                .tensor
                .to_vec()
                .expect("test: w1 readable");
            assert_ne!(w1, old_w, "weight must change after first Adam step");
            assert_eq!(adam.timestep, 1);

            // Second step proves per-parameter (m, v) state persists and
            // that a fresh gradient is correctly retrievable again after the
            // implicit tape was reset by the first `run_backward` call.
            {
                let param_ref = py_param.borrow(py);
                forward_and_backward_square_sum(&param_ref).expect("test: second backward");
            }
            let model2 = make_single_param_model(py, py_param.clone_ref(py))
                .expect("test: model2 construction");
            adam.step(model2).expect("test: second adam.step");

            let w2 = py_param
                .borrow(py)
                .to_tensor()
                .expect("test: to_tensor 2")
                .tensor
                .to_vec()
                .expect("test: w2 readable");
            assert_ne!(w2, w1, "weight must change again after second Adam step");
            assert_eq!(adam.timestep, 2);
            assert_eq!(
                adam.state.len(),
                1,
                "exactly one parameter's state must be tracked"
            );
        });
    }

    #[test]
    fn rmsprop_step_changes_weight_and_persists_cache_across_two_steps() {
        Python::initialize();
        Python::attach(|py| {
            let param = fresh_parameter(vec![2.0], &[1]);
            let py_param = Py::new(py, param).expect("test: Py::new");
            let mut rmsprop = PyRMSprop::new(Some(0.1));

            let old_w = py_param
                .borrow(py)
                .to_tensor()
                .expect("test: to_tensor")
                .tensor
                .to_vec()
                .expect("test: old w readable");

            {
                let param_ref = py_param.borrow(py);
                forward_and_backward_square_sum(&param_ref).expect("test: first backward");
            }
            let model1 = make_single_param_model(py, py_param.clone_ref(py))
                .expect("test: model1 construction");
            rmsprop.step(model1).expect("test: first rmsprop.step");

            let w1 = py_param
                .borrow(py)
                .to_tensor()
                .expect("test: to_tensor 1")
                .tensor
                .to_vec()
                .expect("test: w1 readable");
            assert_ne!(w1, old_w, "weight must change after first RMSprop step");

            {
                let param_ref = py_param.borrow(py);
                forward_and_backward_square_sum(&param_ref).expect("test: second backward");
            }
            let model2 = make_single_param_model(py, py_param.clone_ref(py))
                .expect("test: model2 construction");
            rmsprop.step(model2).expect("test: second rmsprop.step");

            let w2 = py_param
                .borrow(py)
                .to_tensor()
                .expect("test: to_tensor 2")
                .tensor
                .to_vec()
                .expect("test: w2 readable");
            assert_ne!(w2, w1, "weight must change again after second RMSprop step");
            assert_eq!(
                rmsprop.state.len(),
                1,
                "exactly one parameter's state must be tracked"
            );
        });
    }

    #[test]
    fn adamw_decoupled_weight_decay_differs_from_zero_decay() {
        Python::initialize();
        Python::attach(|py| {
            // Two independent, identically-initialized parameters: one
            // optimized with weight_decay=0, one with a large decoupled
            // weight_decay. Their post-step values must differ (proving
            // weight_decay actually has an effect), and — since AdamW's
            // decay term is a pure multiplicative shrink of `w` independent
            // of the gradient/moment path — the decayed run must move
            // strictly further from the original weight than the
            // undecayed run (both start from the same weight, receive the
            // same gradient, and decay always subtracts an additional
            // same-sign term `lr * weight_decay * w` here since w > 0).
            let param_a = fresh_parameter(vec![2.0], &[1]);
            let param_b = fresh_parameter(vec![2.0], &[1]);
            let py_param_a = Py::new(py, param_a).expect("test: Py::new a");
            let py_param_b = Py::new(py, param_b).expect("test: Py::new b");

            {
                let param_ref = py_param_a.borrow(py);
                forward_and_backward_square_sum(&param_ref).expect("test: backward a");
            }
            let model_a = make_single_param_model(py, py_param_a.clone_ref(py))
                .expect("test: model a construction");
            let mut adamw_no_decay = PyAdamW::with_weight_decay(0.1, 0.0);
            adamw_no_decay
                .step(model_a)
                .expect("test: adamw no-decay step");
            let w_a = py_param_a
                .borrow(py)
                .to_tensor()
                .expect("test: to_tensor a")
                .tensor
                .to_vec()
                .expect("test: w_a readable");

            {
                let param_ref = py_param_b.borrow(py);
                forward_and_backward_square_sum(&param_ref).expect("test: backward b");
            }
            let model_b = make_single_param_model(py, py_param_b.clone_ref(py))
                .expect("test: model b construction");
            let mut adamw_with_decay = PyAdamW::with_weight_decay(0.1, 0.5);
            adamw_with_decay
                .step(model_b)
                .expect("test: adamw with-decay step");
            let w_b = py_param_b
                .borrow(py)
                .to_tensor()
                .expect("test: to_tensor b")
                .tensor
                .to_vec()
                .expect("test: w_b readable");

            assert_ne!(
                w_a, w_b,
                "decoupled weight_decay must actually change the outcome"
            );
            // Both start at w=2.0>0; decoupled decay subtracts an extra
            // lr*weight_decay*w > 0 on top of the (identical, since both
            // runs see the same gradient and start from the same moment
            // state) gradient-based update, so the decayed run must end up
            // strictly smaller.
            assert!(
                w_b[0] < w_a[0],
                "decoupled decay must push w strictly lower than the undecayed run: w_a={:?} w_b={:?}",
                w_a,
                w_b
            );
        });
    }

    #[test]
    fn step_silently_skips_a_parameter_with_no_gradient() {
        Python::initialize();
        Python::attach(|py| {
            // A parameter that never goes through a backward pass at all
            // (i.e. `.grad()` errors) must not abort `.step()` for the
            // whole model, mirroring PyTorch's `p.grad is None` skip.
            let param = fresh_parameter(vec![5.0], &[1]);
            let py_param = Py::new(py, param).expect("test: Py::new");
            let model = make_single_param_model(py, py_param.clone_ref(py))
                .expect("test: model construction");

            let mut sgd = PySGD::new(Some(0.1));
            sgd.step(model)
                .expect("step() must succeed even when a parameter has no gradient");

            let w = py_param
                .borrow(py)
                .to_tensor()
                .expect("test: to_tensor")
                .tensor
                .to_vec()
                .expect("test: w readable");
            assert_eq!(
                w,
                vec![5.0],
                "a parameter with no gradient must be left completely unchanged"
            );
        });
    }

    #[test]
    fn zero_grad_clears_a_previously_populated_gradient() {
        Python::initialize();
        Python::attach(|py| {
            let param = fresh_parameter(vec![4.0], &[1]);
            forward_and_backward_square_sum(&param).expect("test: backward must succeed");
            assert!(
                param.grad().is_ok(),
                "sanity check: grad must be populated before zero_grad"
            );

            let py_param = Py::new(py, param).expect("test: Py::new");
            let model = make_single_param_model(py, py_param.clone_ref(py))
                .expect("test: model construction");

            let adam = PyAdam::new(None);
            adam.zero_grad(model).expect("test: zero_grad must succeed");

            assert!(
                py_param.borrow(py).grad().is_err(),
                "grad must be cleared after zero_grad()"
            );
        });
    }

    /// End-to-end proof that [`PySGD::step`] correctly drives a REAL
    /// [`super::super::layers::PyDense`] layer through
    /// [`crate::neural::collect_parameters`] — as opposed to every test
    /// above, which hand-constructs bare [`PyParameter`]s directly. This
    /// specifically exercises the model-agnostic `model.parameters()` dynamic
    /// dispatch path `PySGD::step` actually uses in production
    /// (`crate::neural::collect_parameters(&model)`), with `PyDense` itself
    /// as `model` — proving the whole chain (`PyDense::parameters()` ->
    /// `collect_parameters` -> per-parameter `.grad()`/`.set_data()`) works,
    /// not just the lower-level `PyParameter` primitives it is built from.
    ///
    /// Uses the exact same layer shape, input, and learning rate as
    /// `layers.rs`'s own
    /// `parameters_share_identity_with_forward_across_an_optimizer_update_cycle`
    /// test (which hand-rolls the identical SGD update via raw tensor ops
    /// rather than calling a real optimizer) — so this test's expected
    /// output-delta constant (`-lr * sum_i(x[0,i]^2) - lr = -3.1`) is an
    /// independently cross-checked value, not merely internally consistent
    /// with itself.
    #[test]
    fn sgd_step_drives_a_real_pydense_layer_through_collect_parameters() {
        Python::initialize();
        Python::attach(|py| {
            let lr: f32 = 0.1;

            let dense = crate::neural::layers::PyDense::new(4, 3, Some(true), None)
                .expect("test: PyDense::new should construct with activation=None");
            let py_dense =
                Py::new(py, dense).expect("test: PyDense should construct as a Py<PyDense>");

            let x_tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4])
                .expect("test: from_vec should succeed for matching data/shape lengths");
            let x = PyTensor {
                tensor: Arc::new(x_tensor),
                requires_grad: false,
                is_pinned: false,
            };

            // First forward + loss + backward cycle.
            let output1 = py_dense
                .borrow(py)
                .forward(&x)
                .expect("test: first forward pass should succeed");
            let output1_values = output1.tensor.data().to_vec();

            let loss1 = crate::math_ops::sum(&output1, None, None)
                .expect("test: sum reduction should succeed and remain tape-linked");
            crate::implicit_autograd::run_backward(&loss1)
                .expect("test: first backward pass should succeed");

            // Real PySGD::step(), called with the PyDense itself as `model`
            // — exactly how a user would call it, e.g. `sgd.step(dense)` in
            // Python. `py_dense.bind(py)` yields the `Bound<'_, PyAny>`
            // `step` expects.
            let mut sgd = PySGD::new(Some(lr as f64));
            let model_any: Bound<'_, PyAny> = py_dense.bind(py).clone().into_any();
            sgd.step(model_any)
                .expect("test: sgd.step() over a real PyDense must succeed");

            // Second forward pass must reflect the update, matching the
            // exact expected delta layers.rs's own test independently
            // derives: -lr * sum_i(x[0,i]^2) - lr = -0.1*30.0 - 0.1 = -3.1
            // for every output column (see that test's doc for the full
            // derivation).
            let output2 = py_dense
                .borrow(py)
                .forward(&x)
                .expect("test: second forward pass (after sgd.step()) should succeed");
            let output2_values = output2.tensor.data().to_vec();

            let expected_delta: f32 = -lr * 30.0 - lr;
            assert_eq!(output1_values.len(), output2_values.len());
            for (v1, v2) in output1_values.iter().zip(output2_values.iter()) {
                let actual_delta = v2 - v1;
                assert!(
                    (actual_delta - expected_delta).abs() < 1e-4,
                    "output delta mismatch: actual={actual_delta}, expected={expected_delta} \
                     (output1={v1}, output2={v2}) — sgd.step() must apply exactly \
                     w -= lr * grad through a real PyDense's collect_parameters() path"
                );
            }

            // A second full cycle (forward -> loss -> backward -> step) must
            // also succeed, proving per-parameter Adam/SGD state handling
            // (here: SGD has no momentum configured, so there is no
            // persisted buffer to verify, but the `state` HashMap entry
            // machinery and repeated collect_parameters()/set_data() calls
            // on the SAME PyDense must not error out or corrupt parameter
            // identity on a second pass).
            let loss2 = crate::math_ops::sum(&output2, None, None)
                .expect("test: sum reduction should succeed on the second forward pass");
            crate::implicit_autograd::run_backward(&loss2)
                .expect("test: second backward pass should succeed");
            let model_any2: Bound<'_, PyAny> = py_dense.bind(py).clone().into_any();
            sgd.step(model_any2)
                .expect("test: second sgd.step() over the same PyDense must succeed");

            let output3 = py_dense
                .borrow(py)
                .forward(&x)
                .expect("test: third forward pass (after second sgd.step()) should succeed");
            let output3_values = output3.tensor.data().to_vec();
            for (v2, v3) in output2_values.iter().zip(output3_values.iter()) {
                let actual_delta = v3 - v2;
                assert!(
                    (actual_delta - expected_delta).abs() < 1e-4,
                    "second-step output delta mismatch: actual={actual_delta}, \
                     expected={expected_delta} (output2={v2}, output3={v3})"
                );
            }
        });
    }

    /// Build a fresh `PyDense(2, 1, use_bias=true, activation=None)` — a
    /// single linear-regression unit — with its Xavier-initialised weight
    /// and zero-initialised bias **overwritten** to a fixed, deterministic
    /// starting point (`weight = [[0.0], [0.0]]`, `bias = [0.0]`) via
    /// [`PyParameter::set_data`] through the same `parameters()` handles an
    /// optimizer itself would use.
    ///
    /// This override is not optional: `Dense::new_xavier`
    /// (`tenflowers-neural/src/layers/dense.rs::create_random_normal_tensor`)
    /// seeds its Box-Muller draw from `scirs2_core::random::quick::random_f32`,
    /// an unseeded, genuinely non-deterministic source — so leaving the
    /// layer's Xavier-initialised weight in place would make every
    /// convergence-trend test below depend on whatever the RNG happened to
    /// draw that run, which is exactly the kind of flake the module-level
    /// "Deterministic only" project policy forbids. Starting from an
    /// all-zero weight/bias is also a clean, hand-verifiable base case: the
    /// very first forward pass predicts `0` for every sample regardless of
    /// input, so the initial loss below is trivial to hand-compute (see the
    /// call sites' own doc comments).
    fn fresh_zero_init_dense(py: Python<'_>) -> Py<super::super::layers::PyDense> {
        let dense = crate::neural::layers::PyDense::new(2, 1, Some(true), None)
            .expect("test: PyDense::new should construct with activation=None");
        let py_dense = Py::new(py, dense).expect("test: PyDense should construct as a Py<PyDense>");

        let params = py_dense.borrow(py).parameters();
        assert_eq!(
            params.len(),
            2,
            "PyDense(2, 1, use_bias=true) must expose exactly [weight, bias]"
        );
        let zero_weight = Tensor::<f32>::from_vec(vec![0.0, 0.0], &[2, 1])
            .expect("test: zero weight tensor construction");
        let zero_bias = Tensor::<f32>::from_vec(vec![0.0], &[1])
            .expect("test: zero bias tensor construction");
        params[0]
            .borrow(py)
            .set_data(zero_weight)
            .expect("test: overwrite Xavier-initialised weight with a fixed, deterministic value");
        params[1]
            .borrow(py)
            .set_data(zero_bias)
            .expect("test: overwrite zero-initialised bias with an explicit, deterministic value");

        py_dense
    }

    /// Fixed, hand-computable linear-regression dataset shared by every
    /// `*_step_drives_real_pydense_loss_down_over_many_steps` test below:
    /// four `(x0, x1) -> y` samples following `y = 2*x0 + 3*x1 + 1` exactly
    /// (no noise), as a single batch. A `PyDense(2, 1)` with zero-initialised
    /// weight/bias (see [`fresh_zero_init_dense`]) can fit this perfectly
    /// (it is realisable by a linear model with no activation — weight
    /// `[[2.0], [3.0]]`, bias `[1.0]` is an exact solution), which is exactly
    /// what makes a genuine, unambiguous downward loss trend possible to
    /// assert without tuning around noise.
    ///
    /// Returns `(x, y)` as `(input `[4, 2]` batch, target `[4, 1]` batch)`.
    fn fixed_linear_regression_batch() -> (PyTensor, PyTensor) {
        // x0, x1 pairs.
        let x_data = vec![
            1.0, 0.0, // -> 2*1 + 3*0 + 1 = 3
            0.0, 1.0, // -> 2*0 + 3*1 + 1 = 4
            1.0, 1.0, // -> 2*1 + 3*1 + 1 = 6
            2.0, 1.0, // -> 2*2 + 3*1 + 1 = 8
        ];
        let y_data = vec![3.0, 4.0, 6.0, 8.0];

        let x_tensor = Tensor::<f32>::from_vec(x_data, &[4, 2])
            .expect("test: fixed regression input tensor construction");
        let y_tensor = Tensor::<f32>::from_vec(y_data, &[4, 1])
            .expect("test: fixed regression target tensor construction");

        let x = PyTensor {
            tensor: Arc::new(x_tensor),
            requires_grad: false,
            is_pinned: false,
        };
        let y = PyTensor {
            tensor: Arc::new(y_tensor),
            requires_grad: false,
            is_pinned: false,
        };
        (x, y)
    }

    /// Run `num_steps` full `forward -> mse_loss -> run_backward -> step ->
    /// zero_grad` training cycles for `dense` against the fixed
    /// [`fixed_linear_regression_batch`] dataset, using the caller-supplied
    /// `step_and_zero_grad_fn` closure to invoke whichever concrete
    /// optimizer's (`PySGD`/`PyAdam`/`PyRMSprop`/`PyAdamW`) `step` followed
    /// by `zero_grad` the caller is testing, and returns the scalar MSE loss
    /// recorded **before** each step (so the returned `Vec`'s length is
    /// exactly `num_steps` and its first element is the loss at the fixed
    /// zero-init starting point, before any update has been applied).
    ///
    /// Takes a single closure — rather than separate `step_fn`/`zero_grad_fn`
    /// closures — because every concrete optimizer's `step` takes `&mut
    /// self` while its `zero_grad` takes `&self`; two separate closures
    /// passed as two separate arguments to the same call would both capture
    /// the optimizer for the duration of that call (one mutably, one
    /// immutably), which the borrow checker correctly rejects even though
    /// the two borrows are never actually live at the same time at runtime.
    /// A single closure sidesteps this: it captures the optimizer mutably
    /// exactly once and calls both methods sequentially from its own body.
    ///
    /// Mirrors [`sgd_step_drives_a_real_pydense_layer_through_collect_parameters`]'s
    /// real-`PyDense`-via-`collect_parameters` idiom exactly (a **fresh**
    /// `py_dense.bind(py).clone().into_any()` per call, since `step()`/
    /// `zero_grad()` both consume their `Bound<'_, PyAny>` argument by
    /// value), just looped and driven by a real [`crate::neural::losses::mse_loss`]
    /// instead of a bare `sum(output)`.
    fn train_and_record_losses(
        py: Python<'_>,
        py_dense: &Py<super::super::layers::PyDense>,
        num_steps: usize,
        mut step_and_zero_grad_fn: impl FnMut(Bound<'_, PyAny>, Bound<'_, PyAny>) -> PyResult<()>,
    ) -> Vec<f32> {
        let (x, y) = fixed_linear_regression_batch();
        let mut losses = Vec::with_capacity(num_steps);

        for step_index in 0..num_steps {
            let output = py_dense
                .borrow(py)
                .forward(&x)
                .unwrap_or_else(|e| panic!("test: forward pass {step_index} should succeed: {e}"));

            let loss = crate::neural::losses::mse_loss(&output, &y, Some("mean"))
                .unwrap_or_else(|e| panic!("test: mse_loss {step_index} should succeed: {e}"));
            let loss_value = loss
                .tensor
                .to_vec()
                .unwrap_or_else(|e| panic!("test: loss {step_index} readable: {e}"));
            assert_eq!(
                loss_value.len(),
                1,
                "mse_loss with reduction='mean' must reduce to a single scalar"
            );
            losses.push(loss_value[0]);

            crate::implicit_autograd::run_backward(&loss)
                .unwrap_or_else(|e| panic!("test: backward pass {step_index} should succeed: {e}"));

            // `step()` and `zero_grad()` each consume their `Bound` by
            // value, so each needs its own fresh clone — exactly as
            // `sgd_step_drives_a_real_pydense_layer_through_collect_parameters`
            // already does for its two `model_any`/`model_any2` clones.
            let model_for_step: Bound<'_, PyAny> = py_dense.bind(py).clone().into_any();
            let model_for_zero_grad: Bound<'_, PyAny> = py_dense.bind(py).clone().into_any();
            step_and_zero_grad_fn(model_for_step, model_for_zero_grad).unwrap_or_else(|e| {
                panic!("test: optimizer step+zero_grad {step_index} should succeed: {e}")
            });
        }

        losses
    }

    /// Assert that `losses` shows a genuine, unambiguous downward trend: the
    /// mean of the last `window` entries must be below `max_trailing_ratio`
    /// times the mean of the first `window` entries.
    ///
    /// Deliberately window-mean-based rather than strict step-by-step
    /// monotonic decrease: Adam/RMSprop/AdamW's adaptive per-parameter
    /// scaling can produce small non-monotonic wobbles in the first handful
    /// of steps (e.g. while the second-moment estimate is still warming up
    /// under bias correction) even while the overall trend is unambiguously
    /// downward — asserting strict monotonicity would flake on exactly the
    /// optimizers this test suite most needs to exercise honestly.
    fn assert_loss_trends_down(losses: &[f32], window: usize, max_trailing_ratio: f32, label: &str) {
        assert!(
            losses.len() >= window * 2,
            "{label}: need at least {} recorded losses to compare a leading/trailing window of {window}, got {}",
            window * 2,
            losses.len()
        );
        let early_mean: f32 = losses[..window].iter().sum::<f32>() / window as f32;
        let trailing_mean: f32 =
            losses[losses.len() - window..].iter().sum::<f32>() / window as f32;

        assert!(
            early_mean.is_finite() && trailing_mean.is_finite(),
            "{label}: loss must stay finite throughout training (early_mean={early_mean}, \
             trailing_mean={trailing_mean}) — a NaN/Inf loss is not a passing 'trend', it is a \
             silently-diverged run"
        );
        assert!(
            trailing_mean < early_mean * max_trailing_ratio,
            "{label}: expected the trailing {window}-step mean loss ({trailing_mean}) to be \
             below {max_trailing_ratio} x the leading {window}-step mean loss ({early_mean}), \
             i.e. a clear, unambiguous downward trend over {} recorded steps — got \
             trailing/early ratio = {}",
            losses.len(),
            trailing_mean / early_mean
        );
    }

    /// End-to-end convergence proof for [`PySGD::step`]: looped
    /// forward -> real [`crate::neural::losses::mse_loss`] -> backward ->
    /// (real `PyDense` via `collect_parameters`) `step` -> `zero_grad`,
    /// against the fixed [`fixed_linear_regression_batch`] dataset, asserting
    /// the loss trend is genuinely, unambiguously decreasing (trailing
    /// 10-step mean loss below 50% of the leading 10-step mean loss) — as
    /// opposed to every other SGD test above, which is a single/double-step
    /// exact-arithmetic proof, not a many-step convergence proof.
    ///
    /// `lr=0.05` over 150 steps was empirically tuned (see this test
    /// module's final report) to comfortably clear the 50% trailing/leading
    /// ratio bar without being so aggressive that plain (no-momentum) SGD
    /// risks overshooting on this problem's largest input magnitude (`x =
    /// [2.0, 1.0]`).
    #[test]
    fn sgd_step_drives_real_pydense_loss_down_over_many_steps() {
        Python::initialize();
        Python::attach(|py| {
            let py_dense = fresh_zero_init_dense(py);
            let mut sgd = PySGD::new(Some(0.05));

            let losses = train_and_record_losses(
                py,
                &py_dense,
                150,
                |model_for_step, model_for_zero_grad| {
                    sgd.step(model_for_step)?;
                    sgd.zero_grad(model_for_zero_grad)
                },
            );

            assert_loss_trends_down(&losses, 10, 0.5, "SGD");
        });
    }

    /// End-to-end convergence proof for [`PyAdam::step`] — see
    /// [`sgd_step_drives_real_pydense_loss_down_over_many_steps`]'s doc for
    /// the shared rationale (real `PyDense`/`collect_parameters`/`mse_loss`
    /// training loop, window-mean trend assertion rather than strict
    /// monotonicity).
    ///
    /// `lr=0.1` over 100 steps was empirically tuned: Adam's adaptive
    /// per-parameter step size converges markedly faster than plain SGD on
    /// this problem (its bias-corrected second-moment normalisation lets it
    /// take a comparatively large effective step on both the weight and
    /// bias from the very first iteration), so fewer steps at a higher
    /// nominal `lr` than the SGD test above are enough to clear the same 50%
    /// trailing/leading ratio bar comfortably.
    #[test]
    fn adam_step_drives_real_pydense_loss_down_over_many_steps() {
        Python::initialize();
        Python::attach(|py| {
            let py_dense = fresh_zero_init_dense(py);
            let mut adam = PyAdam::new(Some(0.1));

            let losses = train_and_record_losses(
                py,
                &py_dense,
                100,
                |model_for_step, model_for_zero_grad| {
                    adam.step(model_for_step)?;
                    adam.zero_grad(model_for_zero_grad)
                },
            );

            assert_loss_trends_down(&losses, 10, 0.5, "Adam");
        });
    }

    /// End-to-end convergence proof for [`PyRMSprop::step`] — see
    /// [`sgd_step_drives_real_pydense_loss_down_over_many_steps`]'s doc for
    /// the shared rationale.
    ///
    /// `lr=0.05` over 150 steps was empirically tuned: with RMSprop's
    /// default `alpha=0.99` the squared-gradient cache warms up slowly (a
    /// 0.99-decayed running average needs many steps to move far from its
    /// zero initial value), so a learning rate an order of magnitude above
    /// Adam's default was needed to see the same clear trend within a
    /// comparable step budget — matching SGD's `lr`/step-count here (rather
    /// than Adam's) turned out to be the more reliable empirical choice.
    #[test]
    fn rmsprop_step_drives_real_pydense_loss_down_over_many_steps() {
        Python::initialize();
        Python::attach(|py| {
            let py_dense = fresh_zero_init_dense(py);
            let mut rmsprop = PyRMSprop::new(Some(0.05));

            let losses = train_and_record_losses(
                py,
                &py_dense,
                150,
                |model_for_step, model_for_zero_grad| {
                    rmsprop.step(model_for_step)?;
                    rmsprop.zero_grad(model_for_zero_grad)
                },
            );

            assert_loss_trends_down(&losses, 10, 0.5, "RMSprop");
        });
    }

    /// End-to-end convergence proof for [`PyAdamW::step`] — see
    /// [`sgd_step_drives_real_pydense_loss_down_over_many_steps`]'s doc for
    /// the shared rationale.
    ///
    /// Constructed via [`PyAdamW::with_weight_decay`] with a small
    /// `weight_decay=0.001` (not `PyAdamW::new`'s default `0.01`): the
    /// dataset's exact solution has a non-trivial-magnitude weight
    /// (`[[2.0], [3.0]]`), and decoupled decay directly shrinks the weight
    /// every step independent of the gradient signal, so too large a decay
    /// would fight convergence toward that solution rather than merely
    /// slowing it — `0.001` was empirically confirmed small enough to still
    /// let the same `lr=0.1`/100-step budget that works for plain
    /// [`PyAdam`] above clear the 50% trailing/leading ratio bar
    /// comfortably, while still exercising a genuinely non-zero decoupled
    /// decay term (proving this test exercises AdamW's actual decoupled
    /// path, not merely Adam's moment-estimate bookkeeping with
    /// `weight_decay=0`).
    #[test]
    fn adamw_step_drives_real_pydense_loss_down_over_many_steps() {
        Python::initialize();
        Python::attach(|py| {
            let py_dense = fresh_zero_init_dense(py);
            let mut adamw = PyAdamW::with_weight_decay(0.1, 0.001);

            let losses = train_and_record_losses(
                py,
                &py_dense,
                100,
                |model_for_step, model_for_zero_grad| {
                    adamw.step(model_for_step)?;
                    adamw.zero_grad(model_for_zero_grad)
                },
            );

            assert_loss_trends_down(&losses, 10, 0.5, "AdamW");
        });
    }
}
