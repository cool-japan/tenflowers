//! Learning rate schedulers module for TenfloweRS FFI
//!
//! This module provides learning rate scheduling implementations for adaptive
//! learning rate adjustment during training.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::f32::consts::PI;

/// Step Learning Rate Scheduler
///
/// Decays the learning rate by gamma every step_size epochs.
#[pyclass(name = "StepLR")]
#[derive(Debug, Clone)]
pub struct PyStepLR {
    /// Initial learning rate
    pub initial_lr: f32,
    /// Current learning rate
    pub current_lr: f32,
    /// Period of learning rate decay
    pub step_size: usize,
    /// Multiplicative factor of learning rate decay
    pub gamma: f32,
    /// Current epoch
    pub last_epoch: usize,
}

#[pymethods]
impl PyStepLR {
    /// Create a new StepLR scheduler
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `step_size` - Period of learning rate decay
    /// * `gamma` - Multiplicative factor of learning rate decay (default: 0.1)
    #[new]
    #[pyo3(signature = (initial_lr, step_size, gamma=None))]
    pub fn new(initial_lr: f32, step_size: usize, gamma: Option<f32>) -> PyResult<Self> {
        let gamma = gamma.unwrap_or(0.1);

        if initial_lr <= 0.0 {
            return Err(PyValueError::new_err("initial_lr must be positive"));
        }
        if step_size == 0 {
            return Err(PyValueError::new_err("step_size must be positive"));
        }
        if gamma <= 0.0 || gamma >= 1.0 {
            return Err(PyValueError::new_err(
                "gamma must be between 0 and 1 (exclusive)",
            ));
        }

        Ok(PyStepLR {
            initial_lr,
            current_lr: initial_lr,
            step_size,
            gamma,
            last_epoch: 0,
        })
    }

    /// Compute learning rate for the current epoch
    pub fn step(&mut self) -> f32 {
        self.last_epoch += 1;

        // lr = initial_lr * gamma^(epoch // step_size)
        let decay_epochs = self.last_epoch / self.step_size;
        self.current_lr = self.initial_lr * self.gamma.powi(decay_epochs as i32);

        self.current_lr
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f32 {
        self.current_lr
    }

    /// Get last epoch
    pub fn get_last_epoch(&self) -> usize {
        self.last_epoch
    }

    fn __repr__(&self) -> String {
        format!(
            "StepLR(initial_lr={}, step_size={}, gamma={}, last_epoch={})",
            self.initial_lr, self.step_size, self.gamma, self.last_epoch
        )
    }
}

/// Exponential Learning Rate Scheduler
///
/// Decays the learning rate exponentially by gamma every epoch.
#[pyclass(name = "ExponentialLR")]
#[derive(Debug, Clone)]
pub struct PyExponentialLR {
    /// Initial learning rate
    pub initial_lr: f32,
    /// Current learning rate
    pub current_lr: f32,
    /// Multiplicative factor of learning rate decay
    pub gamma: f32,
    /// Current epoch
    pub last_epoch: usize,
}

#[pymethods]
impl PyExponentialLR {
    /// Create a new ExponentialLR scheduler
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `gamma` - Multiplicative factor of learning rate decay
    #[new]
    pub fn new(initial_lr: f32, gamma: f32) -> PyResult<Self> {
        if initial_lr <= 0.0 {
            return Err(PyValueError::new_err("initial_lr must be positive"));
        }
        if gamma <= 0.0 || gamma >= 1.0 {
            return Err(PyValueError::new_err(
                "gamma must be between 0 and 1 (exclusive)",
            ));
        }

        Ok(PyExponentialLR {
            initial_lr,
            current_lr: initial_lr,
            gamma,
            last_epoch: 0,
        })
    }

    /// Compute learning rate for the current epoch
    pub fn step(&mut self) -> f32 {
        self.last_epoch += 1;

        // lr = initial_lr * gamma^epoch
        self.current_lr = self.initial_lr * self.gamma.powi(self.last_epoch as i32);

        self.current_lr
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f32 {
        self.current_lr
    }

    /// Get last epoch
    pub fn get_last_epoch(&self) -> usize {
        self.last_epoch
    }

    fn __repr__(&self) -> String {
        format!(
            "ExponentialLR(initial_lr={}, gamma={}, last_epoch={})",
            self.initial_lr, self.gamma, self.last_epoch
        )
    }
}

/// Cosine Annealing Learning Rate Scheduler
///
/// Sets the learning rate using a cosine annealing schedule.
#[pyclass(name = "CosineAnnealingLR")]
#[derive(Debug, Clone)]
pub struct PyCosineAnnealingLR {
    /// Initial learning rate
    pub initial_lr: f32,
    /// Current learning rate
    pub current_lr: f32,
    /// Maximum number of epochs
    pub t_max: usize,
    /// Minimum learning rate
    pub eta_min: f32,
    /// Current epoch
    pub last_epoch: usize,
}

#[pymethods]
impl PyCosineAnnealingLR {
    /// Create a new CosineAnnealingLR scheduler
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `t_max` - Maximum number of epochs
    /// * `eta_min` - Minimum learning rate (default: 0.0)
    #[new]
    #[pyo3(signature = (initial_lr, t_max, eta_min=None))]
    pub fn new(initial_lr: f32, t_max: usize, eta_min: Option<f32>) -> PyResult<Self> {
        let eta_min = eta_min.unwrap_or(0.0);

        if initial_lr <= 0.0 {
            return Err(PyValueError::new_err("initial_lr must be positive"));
        }
        if t_max == 0 {
            return Err(PyValueError::new_err("t_max must be positive"));
        }
        if eta_min < 0.0 {
            return Err(PyValueError::new_err("eta_min must be non-negative"));
        }
        if eta_min >= initial_lr {
            return Err(PyValueError::new_err(
                "eta_min must be less than initial_lr",
            ));
        }

        Ok(PyCosineAnnealingLR {
            initial_lr,
            current_lr: initial_lr,
            t_max,
            eta_min,
            last_epoch: 0,
        })
    }

    /// Compute learning rate for the current epoch
    pub fn step(&mut self) -> f32 {
        self.last_epoch += 1;

        // lr = eta_min + (initial_lr - eta_min) * (1 + cos(pi * epoch / t_max)) / 2
        let epoch = (self.last_epoch % self.t_max) as f32;
        let cosine_factor = (1.0 + (PI * epoch / self.t_max as f32).cos()) / 2.0;
        self.current_lr = self.eta_min + (self.initial_lr - self.eta_min) * cosine_factor;

        self.current_lr
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f32 {
        self.current_lr
    }

    /// Get last epoch
    pub fn get_last_epoch(&self) -> usize {
        self.last_epoch
    }

    fn __repr__(&self) -> String {
        format!(
            "CosineAnnealingLR(initial_lr={}, t_max={}, eta_min={}, last_epoch={})",
            self.initial_lr, self.t_max, self.eta_min, self.last_epoch
        )
    }
}

/// Reduce Learning Rate on Plateau Scheduler
///
/// Reduces learning rate when a metric has stopped improving.
#[pyclass(name = "ReduceLROnPlateau")]
#[derive(Debug, Clone)]
pub struct PyReduceLROnPlateau {
    /// Initial learning rate
    pub initial_lr: f32,
    /// Current learning rate
    pub current_lr: f32,
    /// Mode: 'min' or 'max'
    pub mode: String,
    /// Factor by which to reduce learning rate
    pub factor: f32,
    /// Number of epochs with no improvement after which learning rate will be reduced
    pub patience: usize,
    /// Threshold for measuring new optimum
    pub threshold: f32,
    /// Minimum learning rate
    pub min_lr: f32,
    /// Number of epochs to wait before resuming normal operation after lr reduction
    pub cooldown: usize,
    /// Best metric value seen so far
    pub best: f32,
    /// Number of epochs with no improvement
    pub num_bad_epochs: usize,
    /// Epochs since last lr reduction
    pub cooldown_counter: usize,
}

#[pymethods]
impl PyReduceLROnPlateau {
    /// Create a new ReduceLROnPlateau scheduler
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `mode` - 'min' or 'max' (default: 'min')
    /// * `factor` - Factor by which to reduce lr (default: 0.1)
    /// * `patience` - Number of epochs with no improvement (default: 10)
    /// * `threshold` - Threshold for measuring new optimum (default: 1e-4)
    /// * `min_lr` - Minimum learning rate (default: 0.0)
    /// * `cooldown` - Number of epochs to wait after lr reduction (default: 0)
    #[new]
    #[pyo3(signature = (initial_lr, mode=None, factor=None, patience=None, threshold=None, min_lr=None, cooldown=None))]
    pub fn new(
        initial_lr: f32,
        mode: Option<String>,
        factor: Option<f32>,
        patience: Option<usize>,
        threshold: Option<f32>,
        min_lr: Option<f32>,
        cooldown: Option<usize>,
    ) -> PyResult<Self> {
        let mode = mode.unwrap_or_else(|| "min".to_string());
        let factor = factor.unwrap_or(0.1);
        let patience = patience.unwrap_or(10);
        let threshold = threshold.unwrap_or(1e-4);
        let min_lr = min_lr.unwrap_or(0.0);
        let cooldown = cooldown.unwrap_or(0);

        if initial_lr <= 0.0 {
            return Err(PyValueError::new_err("initial_lr must be positive"));
        }
        if mode != "min" && mode != "max" {
            return Err(PyValueError::new_err("mode must be 'min' or 'max'"));
        }
        if factor <= 0.0 || factor >= 1.0 {
            return Err(PyValueError::new_err(
                "factor must be between 0 and 1 (exclusive)",
            ));
        }
        if threshold < 0.0 {
            return Err(PyValueError::new_err("threshold must be non-negative"));
        }
        if min_lr < 0.0 {
            return Err(PyValueError::new_err("min_lr must be non-negative"));
        }
        if min_lr >= initial_lr {
            return Err(PyValueError::new_err("min_lr must be less than initial_lr"));
        }

        let best = if mode == "min" {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };

        Ok(PyReduceLROnPlateau {
            initial_lr,
            current_lr: initial_lr,
            mode,
            factor,
            patience,
            threshold,
            min_lr,
            cooldown,
            best,
            num_bad_epochs: 0,
            cooldown_counter: 0,
        })
    }

    /// Update the learning rate based on metric
    ///
    /// # Arguments
    ///
    /// * `metric` - Current metric value
    pub fn step(&mut self, metric: f32) -> PyResult<f32> {
        // Check if in cooldown
        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            return Ok(self.current_lr);
        }

        // Check if metric has improved
        let is_better = if self.mode == "min" {
            metric < self.best - self.threshold
        } else {
            metric > self.best + self.threshold
        };

        if is_better {
            self.best = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
        }

        // Reduce learning rate if patience exceeded
        if self.num_bad_epochs >= self.patience {
            let new_lr = (self.current_lr * self.factor).max(self.min_lr);

            if new_lr < self.current_lr {
                self.current_lr = new_lr;
                self.cooldown_counter = self.cooldown;
                self.num_bad_epochs = 0;
            }
        }

        Ok(self.current_lr)
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f32 {
        self.current_lr
    }

    /// Get best metric value
    pub fn get_best(&self) -> f32 {
        self.best
    }

    fn __repr__(&self) -> String {
        format!(
            "ReduceLROnPlateau(mode='{}', factor={}, patience={}, current_lr={})",
            self.mode, self.factor, self.patience, self.current_lr
        )
    }
}

/// Cosine Annealing with Warm Restarts Scheduler
///
/// Sets the learning rate using a cosine annealing schedule with periodic restarts.
#[pyclass(name = "CosineAnnealingWarmRestarts")]
#[derive(Debug, Clone)]
pub struct PyCosineAnnealingWarmRestarts {
    /// Initial learning rate
    pub initial_lr: f32,
    /// Current learning rate
    pub current_lr: f32,
    /// Number of epochs for the first restart
    pub t_0: usize,
    /// Factor by which to increase t_i after a restart
    pub t_mult: usize,
    /// Minimum learning rate
    pub eta_min: f32,
    /// Current epoch
    pub last_epoch: usize,
    /// Current cycle
    pub t_i: usize,
    /// Epoch within current cycle
    pub t_cur: usize,
}

#[pymethods]
impl PyCosineAnnealingWarmRestarts {
    /// Create a new CosineAnnealingWarmRestarts scheduler
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `t_0` - Number of epochs for the first restart
    /// * `t_mult` - Factor to increase t_i after a restart (default: 1)
    /// * `eta_min` - Minimum learning rate (default: 0.0)
    #[new]
    #[pyo3(signature = (initial_lr, t_0, t_mult=None, eta_min=None))]
    pub fn new(
        initial_lr: f32,
        t_0: usize,
        t_mult: Option<usize>,
        eta_min: Option<f32>,
    ) -> PyResult<Self> {
        let t_mult = t_mult.unwrap_or(1);
        let eta_min = eta_min.unwrap_or(0.0);

        if initial_lr <= 0.0 {
            return Err(PyValueError::new_err("initial_lr must be positive"));
        }
        if t_0 == 0 {
            return Err(PyValueError::new_err("t_0 must be positive"));
        }
        if t_mult == 0 {
            return Err(PyValueError::new_err("t_mult must be positive"));
        }
        if eta_min < 0.0 {
            return Err(PyValueError::new_err("eta_min must be non-negative"));
        }
        if eta_min >= initial_lr {
            return Err(PyValueError::new_err(
                "eta_min must be less than initial_lr",
            ));
        }

        Ok(PyCosineAnnealingWarmRestarts {
            initial_lr,
            current_lr: initial_lr,
            t_0,
            t_mult,
            eta_min,
            last_epoch: 0,
            t_i: t_0,
            t_cur: 0,
        })
    }

    /// Compute learning rate for the current epoch
    pub fn step(&mut self) -> f32 {
        self.last_epoch += 1;
        self.t_cur += 1;

        // Check if we need to restart
        if self.t_cur >= self.t_i {
            self.t_cur = 0;
            self.t_i *= self.t_mult;
        }

        // Cosine annealing within current cycle
        let cosine_factor = (1.0 + (PI * self.t_cur as f32 / self.t_i as f32).cos()) / 2.0;
        self.current_lr = self.eta_min + (self.initial_lr - self.eta_min) * cosine_factor;

        self.current_lr
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f32 {
        self.current_lr
    }

    /// Get last epoch
    pub fn get_last_epoch(&self) -> usize {
        self.last_epoch
    }

    fn __repr__(&self) -> String {
        format!(
            "CosineAnnealingWarmRestarts(initial_lr={}, t_0={}, t_mult={}, eta_min={}, last_epoch={})",
            self.initial_lr, self.t_0, self.t_mult, self.eta_min, self.last_epoch
        )
    }
}

/// Linear Learning Rate Scheduler
///
/// Linearly increases or decreases the learning rate between two boundaries over epochs.
#[pyclass(name = "LinearLR")]
#[derive(Debug, Clone)]
pub struct PyLinearLR {
    /// Initial learning rate
    pub initial_lr: f32,
    /// Current learning rate
    pub current_lr: f32,
    /// Final learning rate
    pub end_lr: f32,
    /// Total number of epochs
    pub total_epochs: usize,
    /// Current epoch
    pub last_epoch: usize,
}

#[pymethods]
impl PyLinearLR {
    /// Create a new LinearLR scheduler
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `end_lr` - Final learning rate
    /// * `total_epochs` - Total number of epochs for the schedule
    #[new]
    pub fn new(initial_lr: f32, end_lr: f32, total_epochs: usize) -> PyResult<Self> {
        if initial_lr <= 0.0 {
            return Err(PyValueError::new_err("initial_lr must be positive"));
        }
        if end_lr <= 0.0 {
            return Err(PyValueError::new_err("end_lr must be positive"));
        }
        if total_epochs == 0 {
            return Err(PyValueError::new_err("total_epochs must be positive"));
        }

        Ok(PyLinearLR {
            initial_lr,
            current_lr: initial_lr,
            end_lr,
            total_epochs,
            last_epoch: 0,
        })
    }

    /// Compute learning rate for the current epoch
    pub fn step(&mut self) -> f32 {
        self.last_epoch += 1;

        if self.last_epoch >= self.total_epochs {
            self.current_lr = self.end_lr;
        } else {
            // Linear interpolation
            let progress = self.last_epoch as f32 / self.total_epochs as f32;
            self.current_lr = self.initial_lr + (self.end_lr - self.initial_lr) * progress;
        }

        self.current_lr
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f32 {
        self.current_lr
    }

    /// Get last epoch
    pub fn get_last_epoch(&self) -> usize {
        self.last_epoch
    }

    fn __repr__(&self) -> String {
        format!(
            "LinearLR(initial_lr={}, end_lr={}, total_epochs={}, last_epoch={})",
            self.initial_lr, self.end_lr, self.total_epochs, self.last_epoch
        )
    }
}
