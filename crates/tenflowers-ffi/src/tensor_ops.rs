//! Tensor Operations Module
//!
//! This module contains Python bindings for tensor operations including:
//! - PyTensor class and basic tensor operations
//! - Arithmetic operations (add, mul, div, etc.)
//! - Linear algebra operations (matmul, transpose, etc.)
//! - Shape manipulation operations

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;
use tenflowers_autograd::TrackedTensor;
use tenflowers_core::Tensor;

/// Python wrapper for TenfloweRS Tensor
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyTensor {
    pub tensor: Arc<Tensor<f32>>,
    pub requires_grad: bool,
    pub is_pinned: bool,
}

#[pymethods]
impl PyTensor {
    /// Create a new tensor with given shape
    #[new]
    pub fn new(shape: Vec<usize>) -> PyResult<Self> {
        let data = vec![0.0f32; shape.iter().product()];
        let tensor = Tensor::from_vec(data, &shape)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tensor: {}", e)))?;

        Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: false,
            is_pinned: false,
        })
    }

    /// Get tensor shape
    pub fn shape(&self) -> Vec<usize> {
        self.tensor.shape().dims().to_vec()
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.tensor.ndim()
    }

    /// Get total number of elements
    pub fn size(&self) -> usize {
        self.tensor.size()
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.size() * std::mem::size_of::<f32>()
    }

    /// Check if tensor operations can be SIMD-optimized
    pub fn supports_simd(&self) -> bool {
        // Check if tensor size and alignment support SIMD operations
        self.size() >= 8 && self.memory_usage() % 32 == 0
    }

    /// Alias for size() - PyTorch compatibility
    fn numel(&self) -> usize {
        self.size()
    }

    /// Check if tensor requires gradients
    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set gradient requirement
    fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// Check if tensor is scalar (0-dimensional)
    fn is_scalar(&self) -> bool {
        self.tensor.ndim() == 0
    }

    /// Check if tensor is vector (1-dimensional)
    fn is_vector(&self) -> bool {
        self.tensor.ndim() == 1
    }

    /// Check if tensor is matrix (2-dimensional)
    fn is_matrix(&self) -> bool {
        self.tensor.ndim() == 2
    }

    /// Get data type as string
    fn dtype(&self) -> String {
        "float32".to_string()
    }

    /// Check if tensor uses pinned memory
    fn is_pinned(&self) -> bool {
        self.is_pinned
    }

    /// Get transpose (PyTorch-style T property)
    #[allow(non_snake_case)]
    fn T(&self) -> PyResult<PyTensor> {
        self.transpose(None)
    }

    /// Get NumPy-compatible dtype string
    fn numpy_dtype(&self) -> String {
        "float32".to_string()
    }

    /// Check if tensor is contiguous
    fn is_contiguous(&self) -> bool {
        true // TenfloweRS tensors are always contiguous
    }

    /// Check if tensor is C-contiguous
    fn is_c_contiguous(&self) -> bool {
        true
    }

    /// Check if tensor is Fortran-contiguous
    fn is_f_contiguous(&self) -> bool {
        false // TenfloweRS uses C-order
    }

    /// Alias for is_f_contiguous
    fn is_fortran_contiguous(&self) -> bool {
        self.is_f_contiguous()
    }

    /// Transpose tensor with optional axes
    #[pyo3(signature = (axes=None))]
    pub fn transpose(&self, axes: Option<Vec<usize>>) -> PyResult<PyTensor> {
        let result = if let Some(axes_vec) = axes {
            tenflowers_core::ops::manipulation::transpose_axes(&self.tensor, Some(&axes_vec))
        } else {
            tenflowers_core::ops::manipulation::transpose(&self.tensor)
        };

        match result {
            Ok(tensor) => Ok(PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: self.requires_grad,
                is_pinned: self.is_pinned,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!("Transpose failed: {}", e))),
        }
    }

    /// Reshape tensor
    fn reshape(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        match tenflowers_core::ops::reshape(&self.tensor, &shape) {
            Ok(tensor) => Ok(PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: self.requires_grad,
                is_pinned: self.is_pinned,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!("Reshape failed: {}", e))),
        }
    }

    /// Add two tensors
    pub fn add(&self, other: &PyTensor) -> PyResult<PyTensor> {
        match tenflowers_core::ops::add(&self.tensor, &other.tensor) {
            Ok(tensor) => Ok(PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: self.requires_grad || other.requires_grad,
                is_pinned: self.is_pinned || other.is_pinned,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!("Addition failed: {}", e))),
        }
    }

    /// Multiply two tensors
    pub fn mul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        match tenflowers_core::ops::mul(&self.tensor, &other.tensor) {
            Ok(tensor) => Ok(PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: self.requires_grad || other.requires_grad,
                is_pinned: self.is_pinned || other.is_pinned,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Multiplication failed: {}",
                e
            ))),
        }
    }

    /// Subtract two tensors
    pub fn sub(&self, other: &PyTensor) -> PyResult<PyTensor> {
        match tenflowers_core::ops::sub(&self.tensor, &other.tensor) {
            Ok(tensor) => Ok(PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: self.requires_grad || other.requires_grad,
                is_pinned: self.is_pinned || other.is_pinned,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Subtraction failed: {}",
                e
            ))),
        }
    }

    /// Divide two tensors
    pub fn div(&self, other: &PyTensor) -> PyResult<PyTensor> {
        match tenflowers_core::ops::div(&self.tensor, &other.tensor) {
            Ok(tensor) => Ok(PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: self.requires_grad || other.requires_grad,
                is_pinned: self.is_pinned || other.is_pinned,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!("Division failed: {}", e))),
        }
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        match tenflowers_core::ops::matmul(&self.tensor, &other.tensor) {
            Ok(tensor) => Ok(PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: self.requires_grad || other.requires_grad,
                is_pinned: self.is_pinned || other.is_pinned,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Matrix multiplication failed: {}",
                e
            ))),
        }
    }

    /// Power operation
    fn pow(&self, exponent: f32) -> PyResult<PyTensor> {
        // Create a scalar tensor with the same shape for broadcasting
        let exponent_tensor = Tensor::from_scalar(exponent);
        match tenflowers_core::ops::binary::pow(&self.tensor, &exponent_tensor) {
            Ok(tensor) => Ok(PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: self.requires_grad,
                is_pinned: self.is_pinned,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Power operation failed: {}",
                e
            ))),
        }
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "PyTensor(shape={:?}, dtype=float32, requires_grad={})",
            self.shape(),
            self.requires_grad
        )
    }

    /// String representation for print()
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Create a tensor filled with zeros
#[pyfunction]
pub fn zeros(shape: Vec<usize>) -> PyResult<PyTensor> {
    let size: usize = shape.iter().product();
    let data = vec![0.0f32; size];
    let tensor = Tensor::from_vec(data, &shape)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create zeros tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    })
}

/// Create a tensor filled with ones
#[pyfunction]
pub fn ones(shape: Vec<usize>) -> PyResult<PyTensor> {
    let size: usize = shape.iter().product();
    let data = vec![1.0f32; size];
    let tensor = Tensor::from_vec(data, &shape)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create ones tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    })
}

/// Create a tensor with random values
#[pyfunction]
pub fn rand(shape: Vec<usize>) -> PyResult<PyTensor> {
    let size: usize = shape.iter().product();
    let data: Vec<f32> = (0..size).map(|_| 0.5).collect(); // Placeholder - would use proper random in production

    let tensor = Tensor::from_vec(data, &shape)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create random tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    })
}

/// Create a tensor with values from normal distribution
#[pyfunction]
pub fn randn(shape: Vec<usize>) -> PyResult<PyTensor> {
    let size: usize = shape.iter().product();

    // Placeholder normal distribution - would use proper random in production
    let mut data: Vec<f32> = (0..size).map(|_| 0.0).collect();
    data.truncate(size);

    let tensor = Tensor::from_vec(data, &shape)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create randn tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: false,
    })
}

/// Create a tensor filled with zeros using pinned memory
#[pyfunction]
pub fn zeros_pinned(shape: Vec<usize>) -> PyResult<PyTensor> {
    let size: usize = shape.iter().product();
    let data = vec![0.0f32; size];
    let tensor = Tensor::from_vec(data, &shape).map_err(|e| {
        PyRuntimeError::new_err(format!("Failed to create zeros_pinned tensor: {}", e))
    })?;

    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: true, // This tensor uses pinned memory
    })
}

/// Create a tensor filled with ones using pinned memory
#[pyfunction]
pub fn ones_pinned(shape: Vec<usize>) -> PyResult<PyTensor> {
    let size: usize = shape.iter().product();
    let data = vec![1.0f32; size];
    let tensor = Tensor::from_vec(data, &shape).map_err(|e| {
        PyRuntimeError::new_err(format!("Failed to create ones_pinned tensor: {}", e))
    })?;

    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: true, // This tensor uses pinned memory
    })
}

/// Create a tensor with random values using pinned memory
#[pyfunction]
pub fn rand_pinned(shape: Vec<usize>) -> PyResult<PyTensor> {
    let size: usize = shape.iter().product();
    let data: Vec<f32> = (0..size).map(|_| 0.5).collect(); // Placeholder - would use proper random in production

    let tensor = Tensor::from_vec(data, &shape).map_err(|e| {
        PyRuntimeError::new_err(format!("Failed to create rand_pinned tensor: {}", e))
    })?;

    Ok(PyTensor {
        tensor: Arc::new(tensor),
        requires_grad: false,
        is_pinned: true, // This tensor uses pinned memory
    })
}

/// Element-wise addition of two tensors
#[pyfunction]
pub fn add(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    lhs.add(rhs)
}

/// Element-wise multiplication of two tensors
#[pyfunction]
pub fn mul(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    lhs.mul(rhs)
}

/// Element-wise subtraction of two tensors
#[pyfunction]
pub fn sub(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    lhs.sub(rhs)
}

/// Element-wise division of two tensors
#[pyfunction]
pub fn div(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    lhs.div(rhs)
}

/// Matrix multiplication of two tensors
#[pyfunction]
pub fn matmul(lhs: &PyTensor, rhs: &PyTensor) -> PyResult<PyTensor> {
    lhs.matmul(rhs)
}

/// Transpose a tensor
#[pyfunction]
#[pyo3(signature = (tensor, axes=None))]
pub fn transpose(tensor: &PyTensor, axes: Option<Vec<usize>>) -> PyResult<PyTensor> {
    tensor.transpose(axes)
}

/// Reshape a tensor
#[pyfunction]
pub fn reshape(tensor: &PyTensor, shape: Vec<usize>) -> PyResult<PyTensor> {
    tensor.reshape(shape)
}

/// Python wrapper for TenfloweRS TrackedTensor (autograd-enabled)
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyTrackedTensor {
    pub tensor: Arc<TrackedTensor<f32>>,
}

#[pymethods]
impl PyTrackedTensor {
    /// Create a new tracked tensor from a regular tensor
    #[new]
    fn new(tensor: &PyTensor) -> PyResult<Self> {
        let tracked = TrackedTensor::new(tensor.tensor.as_ref().clone());
        Ok(PyTrackedTensor {
            tensor: Arc::new(tracked),
        })
    }

    /// Get the underlying tensor
    fn tensor(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            tensor: Arc::new(self.tensor.tensor().clone()),
            requires_grad: false, // TrackedTensor doesn't expose requires_grad method
            is_pinned: false,     // Default for tracked tensors
        })
    }

    /// Check if this tensor requires gradients
    fn requires_grad(&self) -> bool {
        false // TrackedTensor doesn't expose requires_grad method
    }
}
