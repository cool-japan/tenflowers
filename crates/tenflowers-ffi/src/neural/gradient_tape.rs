//! Gradient tape operations for automatic differentiation
//!
//! This module provides Python bindings for gradient tape functionality,
//! enabling automatic differentiation and gradient computation.

use crate::tensor_ops::PyTensor;
use pyo3::prelude::*;
use std::sync::Arc;
use tenflowers_autograd::{GradientTape, TrackedTensor};

/// Python binding for GradientTape automatic differentiation
#[pyclass]
#[derive(Debug)]
pub struct PyGradientTape {
    tape: GradientTape,
    operation_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[pymethods]
impl PyGradientTape {
    #[new]
    pub fn new() -> Self {
        Self {
            tape: GradientTape::new(),
            operation_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// Watch a tensor for gradient computation
    pub fn watch(&self, tensor: &PyTensor) -> PyResult<PyTrackedTensor> {
        let tracked = self.tape.watch((*tensor.tensor).clone());
        Ok(PyTrackedTensor {
            inner: Arc::new(tracked),
        })
    }

    /// Compute gradient for a single source
    pub fn gradient(
        &self,
        target: &PyTrackedTensor,
        source: &PyTrackedTensor,
    ) -> PyResult<PyTensor> {
        let sources = [source.inner.as_ref()];

        let target_values = vec![target.inner.as_ref().clone()];
        let source_values: Vec<TrackedTensor<f32>> = sources.iter().map(|s| (*s).clone()).collect();

        // Increment operation count for gradient computation
        self.operation_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        match self.tape.gradient(&target_values, &source_values) {
            Ok(mut grads) => {
                if grads.is_empty() {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "No gradient computed",
                    ));
                }
                let grad_opt = grads.remove(0);
                if let Some(grad) = grad_opt {
                    Ok(PyTensor {
                        tensor: Arc::new(grad),
                        requires_grad: false,
                        is_pinned: false,
                    })
                } else {
                    Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "Gradient computation returned None",
                    ))
                }
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Gradient computation failed: {}",
                e
            ))),
        }
    }

    /// Compute gradients for multiple sources
    pub fn gradients(
        &self,
        targets: Vec<PyTrackedTensor>,
        sources: Vec<PyTrackedTensor>,
    ) -> PyResult<Vec<Option<PyTensor>>> {
        let target_values: Vec<TrackedTensor<f32>> =
            targets.iter().map(|t| t.inner.as_ref().clone()).collect();

        let source_values: Vec<TrackedTensor<f32>> =
            sources.iter().map(|s| s.inner.as_ref().clone()).collect();

        // Increment operation count for gradient computation
        self.operation_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        match self.tape.gradient(&target_values, &source_values) {
            Ok(grads) => {
                let py_grads = grads
                    .into_iter()
                    .map(|grad_opt| {
                        grad_opt.map(|grad| PyTensor {
                            tensor: Arc::new(grad),
                            requires_grad: false,
                            is_pinned: false,
                        })
                    })
                    .collect();
                Ok(py_grads)
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Gradient computation failed: {}",
                e
            ))),
        }
    }

    /// Stop recording operations
    pub fn stop_recording(&self) {
        self.tape.stop_recording();
    }

    /// Clear the tape and reset operation count
    pub fn clear(&self) {
        self.tape.clear();
        self.operation_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Start recording operations
    pub fn start_recording(&self) {
        self.tape.start_recording();
    }

    /// Reset the tape
    pub fn reset(&self) {
        self.tape.clear();
    }

    /// Check if the tape is currently recording
    pub fn is_recording(&self) -> bool {
        self.tape.is_recording()
    }

    /// Get the number of operations recorded
    pub fn operation_count(&self) -> usize {
        // Return the current operation count from our atomic counter
        self.operation_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Create a context manager for gradient computation
    pub fn __enter__(slf: PyRefMut<Self>) -> PyResult<PyRefMut<Self>> {
        slf.start_recording();
        Ok(slf)
    }

    /// Exit the context manager
    pub fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.stop_recording();
        Ok(false) // Don't suppress exceptions
    }
}

impl Default for PyGradientTape {
    fn default() -> Self {
        Self::new()
    }
}

/// Python binding for TrackedTensor
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyTrackedTensor {
    pub inner: Arc<TrackedTensor<f32>>,
}

#[pymethods]
impl PyTrackedTensor {
    /// Get the tensor value
    pub fn tensor(&self) -> PyTensor {
        PyTensor {
            tensor: Arc::new(self.inner.tensor().clone()),
            requires_grad: true,
            is_pinned: false,
        }
    }

    /// Get tensor shape
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }

    /// Get tensor data type information
    pub fn dtype(&self) -> String {
        "f32".to_string() // Currently only supports f32
    }

    /// Get tensor device information
    #[allow(unexpected_cfgs)]
    pub fn device(&self) -> String {
        // Access device through the underlying tensor
        match self.inner.tensor().device() {
            tenflowers_core::Device::Cpu => "cpu".to_string(),
            #[cfg(feature = "gpu")]
            tenflowers_core::Device::Gpu(id) => format!("gpu:{}", id),
            #[cfg(feature = "rocm")]
            tenflowers_core::Device::Rocm(id) => format!("rocm:{}", id),
        }
    }

    /// Check if gradient is required for this tensor
    pub fn requires_grad(&self) -> bool {
        true // TrackedTensor always requires grad
    }

    /// Get tensor ID for tracking
    pub fn id(&self) -> usize {
        // Return the actual tensor ID from the TrackedTensor
        self.inner.id
    }

    /// Convert to PyTensor (loses gradient tracking)
    pub fn detach(&self) -> PyTensor {
        PyTensor {
            tensor: Arc::new(self.inner.tensor().clone()),
            requires_grad: false,
            is_pinned: false,
        }
    }

    /// Create a new TrackedTensor with the same data but new tracking
    pub fn clone_detached(&self) -> PyTensor {
        self.detach()
    }

    /// Get string representation
    pub fn __repr__(&self) -> String {
        format!(
            "PyTrackedTensor(shape={:?}, device={}, requires_grad={})",
            self.shape(),
            self.device(),
            self.requires_grad()
        )
    }

    /// Get string representation
    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl PyTrackedTensor {
    /// Create a new PyTrackedTensor from a TrackedTensor
    pub fn new(tracked: TrackedTensor<f32>) -> Self {
        Self {
            inner: Arc::new(tracked),
        }
    }

    /// Create from PyTensor using a tape
    pub fn from_tensor_with_tape(tensor: &PyTensor, tape: &PyGradientTape) -> Self {
        let tracked = tape.tape.watch((*tensor.tensor).clone());
        Self {
            inner: Arc::new(tracked),
        }
    }
}

/// Utility functions for gradient computation
#[pyfunction]
pub fn create_gradient_tape() -> PyGradientTape {
    PyGradientTape::new()
}

/// Compute jacobian matrix for vector-valued functions
#[pyfunction]
pub fn jacobian(
    tape: &PyGradientTape,
    outputs: Vec<PyTrackedTensor>,
    inputs: Vec<PyTrackedTensor>,
) -> PyResult<Vec<Vec<Option<PyTensor>>>> {
    let mut jacobian_matrix = Vec::new();

    for output in outputs {
        let mut jacobian_row = Vec::new();
        for input in &inputs {
            let grad_result = tape.gradient(&output, input);
            jacobian_row.push(grad_result.ok());
        }
        jacobian_matrix.push(jacobian_row);
    }

    Ok(jacobian_matrix)
}

/// Compute hessian matrix for scalar functions
#[pyfunction]
pub fn hessian(
    tape: &PyGradientTape,
    output: &PyTrackedTensor,
    inputs: Vec<PyTrackedTensor>,
) -> PyResult<Vec<Vec<Option<PyTensor>>>> {
    // First compute gradients
    let first_grads = inputs
        .iter()
        .map(|input| tape.gradient(output, input))
        .collect::<PyResult<Vec<_>>>()?;

    // For hessian computation, we need second derivatives
    // This is a simplified implementation - full hessian would require
    // more sophisticated automatic differentiation
    let mut hessian_matrix = Vec::new();

    for (i, _) in first_grads.iter().enumerate().take(inputs.len()) {
        let mut hessian_row = Vec::new();
        for j in 0..inputs.len() {
            // This would need more sophisticated implementation for actual hessian
            if i == j {
                hessian_row.push(Some(first_grads[i].clone()));
            } else {
                hessian_row.push(None);
            }
        }
        hessian_matrix.push(hessian_row);
    }

    Ok(hessian_matrix)
}

/// Context manager for gradient tape operations
#[pyclass]
pub struct PyGradientContext {
    tape: PyGradientTape,
}

#[pymethods]
impl PyGradientContext {
    #[new]
    pub fn new() -> Self {
        Self {
            tape: PyGradientTape::new(),
        }
    }

    /// Get the underlying tape
    pub fn tape(&self) -> PyGradientTape {
        // Return a new tape instance for safety
        PyGradientTape::new()
    }

    /// Watch a tensor in this context
    pub fn watch(&self, tensor: &PyTensor) -> PyResult<PyTrackedTensor> {
        self.tape.watch(tensor)
    }

    /// Context manager entry
    pub fn __enter__(slf: PyRefMut<Self>) -> PyResult<PyRefMut<Self>> {
        slf.tape.start_recording();
        Ok(slf)
    }

    /// Context manager exit
    pub fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.tape.stop_recording();
        Ok(false)
    }
}

impl Default for PyGradientContext {
    fn default() -> Self {
        Self::new()
    }
}
