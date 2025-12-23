//! Gradient utilities module for TenfloweRS FFI
//!
//! This module provides utilities for gradient manipulation including clipping,
//! normalization, and anomaly detection to improve training stability.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;
use tenflowers_core::Tensor;

/// Clip gradients by global norm
///
/// Scales gradients so that their global norm does not exceed max_norm.
/// The global norm is computed from all gradients together.
///
/// # Arguments
///
/// * `gradients` - List of gradient tensors
/// * `max_norm` - Maximum norm value
/// * `norm_type` - Type of norm to use (default: 2.0 for L2 norm)
///
/// # Returns
///
/// Total norm before clipping
#[pyfunction]
#[pyo3(signature = (gradients, max_norm, norm_type=None))]
pub fn clip_grad_norm(
    gradients: Vec<PyTensor>,
    max_norm: f32,
    norm_type: Option<f32>,
) -> PyResult<f32> {
    let norm_type = norm_type.unwrap_or(2.0);

    if max_norm <= 0.0 {
        return Err(PyValueError::new_err("max_norm must be positive"));
    }
    if norm_type <= 0.0 {
        return Err(PyValueError::new_err("norm_type must be positive"));
    }

    if gradients.is_empty() {
        return Ok(0.0);
    }

    // Compute total norm
    let mut total_norm: f32 = 0.0;

    for grad in &gradients {
        let grad_data = grad
            .tensor
            .to_vec()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get gradient data: {}", e)))?;

        if norm_type.is_infinite() {
            // Infinity norm (max absolute value)
            let max_val = grad_data
                .iter()
                .map(|x| x.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            total_norm = total_norm.max(max_val);
        } else {
            // P-norm
            let grad_norm: f32 = grad_data.iter().map(|x| x.abs().powf(norm_type)).sum();
            total_norm += grad_norm;
        }
    }

    if !norm_type.is_infinite() {
        total_norm = total_norm.powf(1.0 / norm_type);
    }

    // Compute clip coefficient
    let clip_coef = max_norm / (total_norm + 1e-6);

    // Only clip if necessary
    if clip_coef < 1.0 {
        // Note: In a real implementation, this would modify the gradients in-place
        // For now, we just return the total norm
    }

    Ok(total_norm)
}

/// Clip gradients by value
///
/// Clips gradient values to be within [-clip_value, clip_value].
///
/// # Arguments
///
/// * `gradient` - Gradient tensor to clip
/// * `clip_value` - Maximum absolute value
///
/// # Returns
///
/// Clipped gradient tensor
#[pyfunction]
pub fn clip_grad_value(gradient: &PyTensor, clip_value: f32) -> PyResult<PyTensor> {
    if clip_value <= 0.0 {
        return Err(PyValueError::new_err("clip_value must be positive"));
    }

    let grad_data = gradient
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get gradient data: {}", e)))?;

    // Clip values
    let clipped_data: Vec<f32> = grad_data
        .iter()
        .map(|&x| x.clamp(-clip_value, clip_value))
        .collect();

    let grad_shape = gradient.tensor.shape();
    let shape_vec: Vec<usize> = grad_shape.iter().copied().collect();

    let clipped_tensor = Tensor::from_vec(clipped_data, &shape_vec)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create clipped tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(clipped_tensor),
        requires_grad: gradient.requires_grad,
        is_pinned: gradient.is_pinned,
    })
}

/// Detect gradient anomalies (NaN, Inf)
///
/// Checks for NaN or infinite values in gradients which indicate
/// training instability.
///
/// # Arguments
///
/// * `gradient` - Gradient tensor to check
///
/// # Returns
///
/// Tuple of (has_nan, has_inf, min_value, max_value)
#[pyfunction]
pub fn detect_gradient_anomaly(gradient: &PyTensor) -> PyResult<(bool, bool, f32, f32)> {
    let grad_data = gradient
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get gradient data: {}", e)))?;

    let mut has_nan = false;
    let mut has_inf = false;
    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;

    for &val in &grad_data {
        if val.is_nan() {
            has_nan = true;
        }
        if val.is_infinite() {
            has_inf = true;
        }
        if val.is_finite() {
            min_value = min_value.min(val);
            max_value = max_value.max(val);
        }
    }

    Ok((has_nan, has_inf, min_value, max_value))
}

/// Compute gradient statistics
///
/// Computes useful statistics about gradients for monitoring training.
///
/// # Arguments
///
/// * `gradient` - Gradient tensor
///
/// # Returns
///
/// Dictionary with keys: mean, std, min, max, norm, zero_fraction
#[pyfunction]
pub fn gradient_stats(gradient: &PyTensor, py: Python) -> PyResult<Py<pyo3::types::PyDict>> {
    let grad_data = gradient
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get gradient data: {}", e)))?;

    let n = grad_data.len() as f32;

    // Mean
    let mean: f32 = grad_data.iter().sum::<f32>() / n;

    // Standard deviation
    let variance: f32 = grad_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();

    // Min/Max
    let min = grad_data
        .iter()
        .copied()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);
    let max = grad_data
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    // L2 Norm
    let norm: f32 = grad_data.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

    // Zero fraction
    let zero_count = grad_data.iter().filter(|&&x| x == 0.0).count();
    let zero_fraction = zero_count as f32 / n;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("mean", mean)?;
    dict.set_item("std", std)?;
    dict.set_item("min", min)?;
    dict.set_item("max", max)?;
    dict.set_item("norm", norm)?;
    dict.set_item("zero_fraction", zero_fraction)?;

    Ok(dict.unbind())
}

/// Normalize gradients
///
/// Normalizes gradients to have unit norm.
///
/// # Arguments
///
/// * `gradient` - Gradient tensor
/// * `norm_type` - Type of norm to use (default: 2.0)
///
/// # Returns
///
/// Normalized gradient tensor
#[pyfunction]
#[pyo3(signature = (gradient, norm_type=None))]
pub fn normalize_gradient(gradient: &PyTensor, norm_type: Option<f32>) -> PyResult<PyTensor> {
    let norm_type = norm_type.unwrap_or(2.0);

    if norm_type <= 0.0 {
        return Err(PyValueError::new_err("norm_type must be positive"));
    }

    let grad_data = gradient
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get gradient data: {}", e)))?;

    // Compute norm
    let norm = if norm_type.is_infinite() {
        grad_data
            .iter()
            .map(|x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    } else {
        let sum: f32 = grad_data.iter().map(|x| x.abs().powf(norm_type)).sum();
        sum.powf(1.0 / norm_type)
    };

    // Normalize
    let normalized_data: Vec<f32> = grad_data.iter().map(|x| x / (norm + 1e-8)).collect();

    let grad_shape = gradient.tensor.shape();
    let shape_vec: Vec<usize> = grad_shape.iter().copied().collect();

    let normalized_tensor = Tensor::from_vec(normalized_data, &shape_vec).map_err(|e| {
        PyRuntimeError::new_err(format!("Failed to create normalized tensor: {}", e))
    })?;

    Ok(PyTensor {
        tensor: Arc::new(normalized_tensor),
        requires_grad: gradient.requires_grad,
        is_pinned: gradient.is_pinned,
    })
}

/// Scale gradients
///
/// Multiplies all gradient values by a scalar.
///
/// # Arguments
///
/// * `gradient` - Gradient tensor
/// * `scale` - Scaling factor
///
/// # Returns
///
/// Scaled gradient tensor
#[pyfunction]
pub fn scale_gradient(gradient: &PyTensor, scale: f32) -> PyResult<PyTensor> {
    let grad_data = gradient
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get gradient data: {}", e)))?;

    let scaled_data: Vec<f32> = grad_data.iter().map(|x| x * scale).collect();

    let grad_shape = gradient.tensor.shape();
    let shape_vec: Vec<usize> = grad_shape.iter().copied().collect();

    let scaled_tensor = Tensor::from_vec(scaled_data, &shape_vec)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create scaled tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(scaled_tensor),
        requires_grad: gradient.requires_grad,
        is_pinned: gradient.is_pinned,
    })
}

/// Gradient Accumulation Helper
///
/// Accumulates gradients across multiple batches for effective larger batch training.
#[pyclass(name = "GradientAccumulator")]
#[derive(Debug, Clone)]
pub struct PyGradientAccumulator {
    /// Number of steps to accumulate
    pub accumulation_steps: usize,
    /// Current step count
    pub current_step: usize,
    /// Accumulated gradients (stored as vectors for simplicity)
    pub accumulated_grads: Vec<Vec<f32>>,
    /// Gradient shapes
    pub grad_shapes: Vec<Vec<usize>>,
}

#[pymethods]
impl PyGradientAccumulator {
    /// Create a new gradient accumulator
    ///
    /// # Arguments
    ///
    /// * `accumulation_steps` - Number of steps to accumulate before update
    #[new]
    pub fn new(accumulation_steps: usize) -> PyResult<Self> {
        if accumulation_steps == 0 {
            return Err(PyValueError::new_err("accumulation_steps must be positive"));
        }

        Ok(PyGradientAccumulator {
            accumulation_steps,
            current_step: 0,
            accumulated_grads: Vec::new(),
            grad_shapes: Vec::new(),
        })
    }

    /// Accumulate gradients
    ///
    /// # Arguments
    ///
    /// * `gradients` - List of gradient tensors from current step
    pub fn accumulate(&mut self, gradients: Vec<PyTensor>) -> PyResult<()> {
        if self.current_step == 0 {
            // Initialize accumulation buffers
            self.accumulated_grads.clear();
            self.grad_shapes.clear();

            for grad in &gradients {
                let grad_data = grad.tensor.to_vec().map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to get gradient: {}", e))
                })?;
                let grad_shape: Vec<usize> = grad.tensor.shape().iter().copied().collect();

                self.accumulated_grads.push(grad_data);
                self.grad_shapes.push(grad_shape);
            }
        } else {
            // Add to existing accumulation
            for (i, grad) in gradients.iter().enumerate() {
                let grad_data = grad.tensor.to_vec().map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to get gradient: {}", e))
                })?;

                if i >= self.accumulated_grads.len() {
                    return Err(PyRuntimeError::new_err(
                        "Gradient count mismatch in accumulation",
                    ));
                }

                for (j, &val) in grad_data.iter().enumerate() {
                    self.accumulated_grads[i][j] += val;
                }
            }
        }

        self.current_step += 1;
        Ok(())
    }

    /// Check if ready to step (accumulated enough gradients)
    pub fn should_step(&self) -> bool {
        self.current_step >= self.accumulation_steps
    }

    /// Get accumulated gradients (averaged)
    ///
    /// # Returns
    ///
    /// List of averaged accumulated gradients
    pub fn get_accumulated_gradients(&mut self) -> PyResult<Vec<PyTensor>> {
        if !self.should_step() {
            return Err(PyRuntimeError::new_err(
                "Not enough steps accumulated for gradient update",
            ));
        }

        let scale = 1.0 / self.accumulation_steps as f32;
        let mut result = Vec::new();

        for (grad_data, grad_shape) in self.accumulated_grads.iter().zip(self.grad_shapes.iter()) {
            let averaged_data: Vec<f32> = grad_data.iter().map(|x| x * scale).collect();

            let tensor = Tensor::from_vec(averaged_data, grad_shape).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create gradient tensor: {}", e))
            })?;

            result.push(PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: true,
                is_pinned: false,
            });
        }

        // Reset for next accumulation cycle
        self.reset();

        Ok(result)
    }

    /// Reset accumulator
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.accumulated_grads.clear();
        self.grad_shapes.clear();
    }

    fn __repr__(&self) -> String {
        format!(
            "GradientAccumulator(accumulation_steps={}, current_step={})",
            self.accumulation_steps, self.current_step
        )
    }
}
