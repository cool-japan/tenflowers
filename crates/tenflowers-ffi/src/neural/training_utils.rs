//! Training utilities module for TenfloweRS FFI
//!
//! This module provides advanced training utilities including early stopping,
//! learning rate warmup, and training callbacks for better training control.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::VecDeque;

/// Early Stopping Monitor
///
/// Monitors training metrics and stops training when no improvement is seen.
#[pyclass(name = "EarlyStopping")]
#[derive(Debug, Clone)]
pub struct PyEarlyStopping {
    /// Metric mode: 'min' or 'max'
    pub mode: String,
    /// Patience: number of epochs with no improvement after which training will be stopped
    pub patience: usize,
    /// Minimum change to qualify as an improvement
    pub min_delta: f32,
    /// Best metric value seen so far
    pub best_value: f32,
    /// Counter for epochs with no improvement
    pub wait: usize,
    /// Whether training should stop
    pub stopped: bool,
    /// Whether to restore best weights
    pub restore_best_weights: bool,
}

#[pymethods]
impl PyEarlyStopping {
    /// Create a new early stopping monitor
    ///
    /// # Arguments
    ///
    /// * `patience` - Number of epochs with no improvement to wait before stopping
    /// * `mode` - 'min' for metrics to minimize (loss), 'max' for metrics to maximize (accuracy)
    /// * `min_delta` - Minimum change to qualify as improvement (default: 0.0)
    /// * `restore_best_weights` - Whether to restore best weights when stopping (default: True)
    #[new]
    #[pyo3(signature = (patience, mode=None, min_delta=None, restore_best_weights=None))]
    pub fn new(
        patience: usize,
        mode: Option<String>,
        min_delta: Option<f32>,
        restore_best_weights: Option<bool>,
    ) -> PyResult<Self> {
        let mode = mode.unwrap_or_else(|| "min".to_string());
        let min_delta = min_delta.unwrap_or(0.0);
        let restore_best_weights = restore_best_weights.unwrap_or(true);

        if patience == 0 {
            return Err(PyValueError::new_err("patience must be positive"));
        }
        if mode != "min" && mode != "max" {
            return Err(PyValueError::new_err("mode must be 'min' or 'max'"));
        }
        if min_delta < 0.0 {
            return Err(PyValueError::new_err("min_delta must be non-negative"));
        }

        let best_value = if mode == "min" {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };

        Ok(PyEarlyStopping {
            mode,
            patience,
            min_delta,
            best_value,
            wait: 0,
            stopped: false,
            restore_best_weights,
        })
    }

    /// Update the early stopping monitor with a new metric value
    ///
    /// # Arguments
    ///
    /// * `value` - Current metric value
    ///
    /// # Returns
    ///
    /// True if training should stop, False otherwise
    pub fn step(&mut self, value: f32) -> bool {
        let is_improvement = if self.mode == "min" {
            value < self.best_value - self.min_delta
        } else {
            value > self.best_value + self.min_delta
        };

        if is_improvement {
            self.best_value = value;
            self.wait = 0;
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                self.stopped = true;
                return true;
            }
        }

        false
    }

    /// Check if training should stop
    pub fn should_stop(&self) -> bool {
        self.stopped
    }

    /// Get the best metric value seen so far
    pub fn get_best_value(&self) -> f32 {
        self.best_value
    }

    /// Reset the early stopping monitor
    pub fn reset(&mut self) {
        self.best_value = if self.mode == "min" {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };
        self.wait = 0;
        self.stopped = false;
    }

    fn __repr__(&self) -> String {
        format!(
            "EarlyStopping(mode='{}', patience={}, min_delta={}, best_value={}, wait={})",
            self.mode, self.patience, self.min_delta, self.best_value, self.wait
        )
    }
}

/// Learning Rate Warmup
///
/// Gradually increases learning rate from a low value to the target learning rate.
#[pyclass(name = "LRWarmup")]
#[derive(Debug, Clone)]
pub struct PyLRWarmup {
    /// Target learning rate
    pub target_lr: f32,
    /// Number of warmup steps
    pub warmup_steps: usize,
    /// Initial learning rate
    pub initial_lr: f32,
    /// Current step
    pub current_step: usize,
}

#[pymethods]
impl PyLRWarmup {
    /// Create a new learning rate warmup scheduler
    ///
    /// # Arguments
    ///
    /// * `target_lr` - Target learning rate to reach
    /// * `warmup_steps` - Number of steps for warmup
    /// * `initial_lr` - Initial learning rate (default: 0.0)
    #[new]
    #[pyo3(signature = (target_lr, warmup_steps, initial_lr=None))]
    pub fn new(target_lr: f32, warmup_steps: usize, initial_lr: Option<f32>) -> PyResult<Self> {
        let initial_lr = initial_lr.unwrap_or(0.0);

        if target_lr <= 0.0 {
            return Err(PyValueError::new_err("target_lr must be positive"));
        }
        if warmup_steps == 0 {
            return Err(PyValueError::new_err("warmup_steps must be positive"));
        }
        if initial_lr < 0.0 {
            return Err(PyValueError::new_err("initial_lr must be non-negative"));
        }
        if initial_lr >= target_lr {
            return Err(PyValueError::new_err(
                "initial_lr must be less than target_lr",
            ));
        }

        Ok(PyLRWarmup {
            target_lr,
            warmup_steps,
            initial_lr,
            current_step: 0,
        })
    }

    /// Get learning rate for current step and increment step counter
    pub fn step(&mut self) -> f32 {
        if self.current_step >= self.warmup_steps {
            return self.target_lr;
        }

        let lr = self.initial_lr
            + (self.target_lr - self.initial_lr) * (self.current_step as f32)
                / (self.warmup_steps as f32);

        self.current_step += 1;
        lr
    }

    /// Get current step
    pub fn get_step(&self) -> usize {
        self.current_step
    }

    /// Check if warmup is complete
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.warmup_steps
    }

    fn __repr__(&self) -> String {
        format!(
            "LRWarmup(target_lr={}, warmup_steps={}, current_step={})",
            self.target_lr, self.warmup_steps, self.current_step
        )
    }
}

/// Training Metrics Tracker
///
/// Tracks and computes statistics for training metrics over time.
#[pyclass(name = "MetricsTracker")]
#[derive(Debug, Clone)]
pub struct PyMetricsTracker {
    /// Window size for moving average
    pub window_size: usize,
    /// Metric history
    pub history: Vec<f32>,
    /// Moving average window
    pub moving_window: VecDeque<f32>,
}

#[pymethods]
impl PyMetricsTracker {
    /// Create a new metrics tracker
    ///
    /// # Arguments
    ///
    /// * `window_size` - Size of window for moving average (default: 10)
    #[new]
    #[pyo3(signature = (window_size=None))]
    pub fn new(window_size: Option<usize>) -> PyResult<Self> {
        let window_size = window_size.unwrap_or(10);

        if window_size == 0 {
            return Err(PyValueError::new_err("window_size must be positive"));
        }

        Ok(PyMetricsTracker {
            window_size,
            history: Vec::new(),
            moving_window: VecDeque::new(),
        })
    }

    /// Add a new metric value
    pub fn update(&mut self, value: f32) {
        self.history.push(value);

        self.moving_window.push_back(value);
        if self.moving_window.len() > self.window_size {
            self.moving_window.pop_front();
        }
    }

    /// Get moving average of recent values
    pub fn get_moving_average(&self) -> Option<f32> {
        if self.moving_window.is_empty() {
            return None;
        }

        let sum: f32 = self.moving_window.iter().sum();
        Some(sum / self.moving_window.len() as f32)
    }

    /// Get overall average
    pub fn get_average(&self) -> Option<f32> {
        if self.history.is_empty() {
            return None;
        }

        let sum: f32 = self.history.iter().sum();
        Some(sum / self.history.len() as f32)
    }

    /// Get minimum value
    pub fn get_min(&self) -> Option<f32> {
        self.history
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Get maximum value
    pub fn get_max(&self) -> Option<f32> {
        self.history
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Get latest value
    pub fn get_latest(&self) -> Option<f32> {
        self.history.last().copied()
    }

    /// Get all history
    pub fn get_history(&self) -> Vec<f32> {
        self.history.clone()
    }

    /// Reset the tracker
    pub fn reset(&mut self) {
        self.history.clear();
        self.moving_window.clear();
    }

    fn __repr__(&self) -> String {
        format!(
            "MetricsTracker(window_size={}, history_len={}, moving_avg={:?})",
            self.window_size,
            self.history.len(),
            self.get_moving_average()
        )
    }
}

/// Training Progress Tracker
///
/// Tracks overall training progress including epochs, steps, and time.
#[pyclass(name = "ProgressTracker")]
#[derive(Debug, Clone)]
pub struct PyProgressTracker {
    /// Total number of epochs
    pub total_epochs: usize,
    /// Current epoch
    pub current_epoch: usize,
    /// Total steps per epoch
    pub steps_per_epoch: usize,
    /// Current step in epoch
    pub current_step: usize,
    /// Total steps completed
    pub total_steps: usize,
}

#[pymethods]
impl PyProgressTracker {
    /// Create a new progress tracker
    ///
    /// # Arguments
    ///
    /// * `total_epochs` - Total number of epochs
    /// * `steps_per_epoch` - Number of steps per epoch
    #[new]
    pub fn new(total_epochs: usize, steps_per_epoch: usize) -> PyResult<Self> {
        if total_epochs == 0 {
            return Err(PyValueError::new_err("total_epochs must be positive"));
        }
        if steps_per_epoch == 0 {
            return Err(PyValueError::new_err("steps_per_epoch must be positive"));
        }

        Ok(PyProgressTracker {
            total_epochs,
            current_epoch: 0,
            steps_per_epoch,
            current_step: 0,
            total_steps: 0,
        })
    }

    /// Step forward in training
    pub fn step(&mut self) {
        self.current_step += 1;
        self.total_steps += 1;

        if self.current_step >= self.steps_per_epoch {
            self.current_step = 0;
            self.current_epoch += 1;
        }
    }

    /// Get current progress as fraction (0.0 to 1.0)
    pub fn get_progress(&self) -> f32 {
        let total_expected_steps = self.total_epochs * self.steps_per_epoch;
        self.total_steps as f32 / total_expected_steps as f32
    }

    /// Get current epoch progress as fraction (0.0 to 1.0)
    pub fn get_epoch_progress(&self) -> f32 {
        self.current_step as f32 / self.steps_per_epoch as f32
    }

    /// Check if training is complete
    pub fn is_complete(&self) -> bool {
        self.current_epoch >= self.total_epochs
    }

    /// Get estimated remaining steps
    pub fn get_remaining_steps(&self) -> usize {
        let total_expected = self.total_epochs * self.steps_per_epoch;
        total_expected.saturating_sub(self.total_steps)
    }

    fn __repr__(&self) -> String {
        format!(
            "ProgressTracker(epoch={}/{}, step={}/{}, progress={:.1}%)",
            self.current_epoch,
            self.total_epochs,
            self.current_step,
            self.steps_per_epoch,
            self.get_progress() * 100.0
        )
    }
}
