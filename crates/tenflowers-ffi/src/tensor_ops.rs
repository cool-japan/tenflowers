//! Tensor Operations Module
//!
//! This module contains Python bindings for tensor operations including:
//! - PyTensor class and basic tensor operations
//! - Arithmetic operations (add, mul, div, etc.)
//! - Linear algebra operations (matmul, transpose, etc.)
//! - Shape manipulation operations

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
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

    /// The device this tensor resides on (always CPU in the current implementation).
    #[getter]
    pub fn device(&self) -> crate::device::PyDevice {
        crate::device::PyDevice::cpu()
    }

    /// String representation.
    ///
    /// Includes shape, dtype (derived from the inner tensor type — always `float32`
    /// for the current `Tensor<f32>` implementation), device, and requires_grad.
    fn __repr__(&self) -> String {
        format!(
            "PyTensor(shape={:?}, dtype={}, device={}, requires_grad={})",
            self.shape(),
            self.dtype(),
            self.device().__str__(),
            self.requires_grad
        )
    }

    /// String representation for print().
    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Length along the first dimension (Python `len()` support).
    ///
    /// Raises `TypeError` for scalar (rank-0) tensors, matching NumPy / PyTorch behaviour.
    fn __len__(&self) -> PyResult<usize> {
        let shape = self.shape();
        if shape.is_empty() {
            Err(PyTypeError::new_err(
                "len() of a scalar tensor (rank 0) is not defined",
            ))
        } else {
            Ok(shape[0])
        }
    }

    /// Iterator over slices along the first dimension.
    ///
    /// Raises `TypeError` for scalar (rank-0) tensors.
    fn __iter__(&self) -> PyResult<PyTensorIter> {
        let shape = self.shape();
        if shape.is_empty() {
            return Err(PyTypeError::new_err(
                "cannot iterate over a scalar tensor (rank 0)",
            ));
        }
        Ok(PyTensorIter {
            source: self.clone(),
            index: 0,
            len: shape[0],
        })
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

/// Iterator over first-dimension slices of a `PyTensor`.
///
/// Yielded items are rank-(N-1) tensors obtained by slicing one row.
#[pyclass]
pub struct PyTensorIter {
    source: PyTensor,
    index: usize,
    len: usize,
}

#[pymethods]
impl PyTensorIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<PyTensor>> {
        if self.index >= self.len {
            return Ok(None);
        }

        let full_shape = self.source.shape();
        // Compute the shape of one slice: drop the first dimension.
        let slice_shape: Vec<usize> = full_shape[1..].to_vec();
        let slice_numel: usize = if slice_shape.is_empty() {
            1
        } else {
            slice_shape.iter().product()
        };

        let start = self.index * slice_numel;
        let end = start + slice_numel;

        // Obtain the raw data vector from the tensor.
        let all_data =
            self.source.tensor.to_vec().map_err(|e| {
                PyRuntimeError::new_err(format!("iterator: failed to get data: {}", e))
            })?;

        if end > all_data.len() {
            return Err(PyRuntimeError::new_err(
                "iterator: slice index out of range (data length mismatch)",
            ));
        }

        let slice_data: Vec<f32> = all_data[start..end].to_vec();

        // When slice_shape is empty (source was 1-D), wrap in a length-1 vector tensor.
        let out_shape: Vec<usize> = if slice_shape.is_empty() {
            vec![1]
        } else {
            slice_shape
        };

        let t = Tensor::from_vec(slice_data, &out_shape)
            .map_err(|e| PyRuntimeError::new_err(format!("iterator: reshape failed: {}", e)))?;

        self.index += 1;

        Ok(Some(PyTensor {
            tensor: Arc::new(t),
            requires_grad: self.source.requires_grad,
            is_pinned: self.source.is_pinned,
        }))
    }
}
