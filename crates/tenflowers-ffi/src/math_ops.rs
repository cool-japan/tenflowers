//! Mathematical Operations Module
//!
//! This module contains Python bindings for mathematical operations including:
//! - Basic math functions (exp, log, sqrt, etc.)
//! - Trigonometric functions (sin, cos, tan, etc.)
//! - Statistical functions (mean, std, var, etc.)
//! - Reduction operations (sum, max, min, etc.)

use crate::tensor_ops::PyTensor;
use ::std::sync::Arc;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use tenflowers_core::Tensor;

/// Exponential function
#[pyfunction]
pub fn exp(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::exp(&input.tensor) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Exp failed: {}", e))),
    }
}

/// Natural logarithm function
#[pyfunction]
pub fn log(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::log(&input.tensor) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Log failed: {}", e))),
    }
}

/// Square root function
#[pyfunction]
pub fn sqrt(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::sqrt(&input.tensor) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Sqrt failed: {}", e))),
    }
}

/// Absolute value function
#[pyfunction]
pub fn abs(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::numpy_compat::absolute(&input.tensor) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Abs failed: {}", e))),
    }
}

/// Negation function
#[pyfunction]
pub fn neg(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::numpy_compat::negative(&input.tensor) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Neg failed: {}", e))),
    }
}

/// Sine function
#[pyfunction]
pub fn sin(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::sin(&input.tensor) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Sin failed: {}", e))),
    }
}

/// Cosine function
#[pyfunction]
pub fn cos(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::cos(&input.tensor) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Cos failed: {}", e))),
    }
}

/// Tangent function
#[pyfunction]
pub fn tan(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::tan(&input.tensor) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Tan failed: {}", e))),
    }
}

/// Sum reduction
#[pyfunction]
pub fn sum(input: &PyTensor, dim: Option<Vec<i32>>, keepdim: Option<bool>) -> PyResult<PyTensor> {
    let keep_dims = keepdim.unwrap_or(false);
    let axes = dim.as_deref();

    match tenflowers_core::ops::sum(&input.tensor, axes, keep_dims) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Sum failed: {}", e))),
    }
}

/// Mean reduction
#[pyfunction]
pub fn mean(input: &PyTensor, dim: Option<Vec<i32>>, keepdim: Option<bool>) -> PyResult<PyTensor> {
    let keep_dims = keepdim.unwrap_or(false);
    let axes = dim.as_deref();

    match tenflowers_core::ops::mean(&input.tensor, axes, keep_dims) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Mean failed: {}", e))),
    }
}

/// Maximum reduction
#[pyfunction]
pub fn max(input: &PyTensor, dim: Option<i32>, keepdim: Option<bool>) -> PyResult<PyTensor> {
    let keep_dims = keepdim.unwrap_or(false);

    let result = if let Some(axis) = dim {
        tenflowers_core::ops::max(&input.tensor, Some(&[axis]), keep_dims)
    } else {
        tenflowers_core::ops::max(&input.tensor, None, keep_dims)
    };

    match result {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Max failed: {}", e))),
    }
}

/// Minimum reduction
#[pyfunction]
pub fn min(input: &PyTensor, dim: Option<i32>, keepdim: Option<bool>) -> PyResult<PyTensor> {
    let keep_dims = keepdim.unwrap_or(false);

    let result = if let Some(axis) = dim {
        tenflowers_core::ops::min(&input.tensor, Some(&[axis]), keep_dims)
    } else {
        tenflowers_core::ops::min(&input.tensor, None, keep_dims)
    };

    match result {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Min failed: {}", e))),
    }
}

/// Variance calculation
#[pyfunction]
pub fn var(
    input: &PyTensor,
    dim: Option<Vec<i32>>,
    keepdim: Option<bool>,
    unbiased: Option<bool>,
) -> PyResult<PyTensor> {
    let keep_dims = keepdim.unwrap_or(false);
    let is_unbiased = unbiased.unwrap_or(true);
    let axes = dim.as_deref();
    let ddof = if is_unbiased { 1 } else { 0 };

    match tenflowers_core::ops::reduction::variance(&input.tensor, axes, keep_dims, ddof) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Var failed: {}", e))),
    }
}

/// Standard deviation calculation
#[pyfunction]
pub fn standard_deviation(
    input: &PyTensor,
    dim: Option<Vec<i32>>,
    keepdim: Option<bool>,
    unbiased: Option<bool>,
) -> PyResult<PyTensor> {
    let keep_dims = keepdim.unwrap_or(false);
    let is_unbiased = unbiased.unwrap_or(true);
    let axes = dim.as_deref();
    let ddof = if is_unbiased { 1 } else { 0 };

    // Compute variance first, then take square root
    match tenflowers_core::ops::reduction::variance(&input.tensor, axes, keep_dims, ddof) {
        Ok(var_tensor) => match tenflowers_core::ops::sqrt(&var_tensor) {
            Ok(tensor) => Ok(PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!("Sqrt failed: {}", e))),
        },
        Err(e) => Err(PyRuntimeError::new_err(format!("Var failed: {}", e))),
    }
}

/// Alias for standard_deviation function to match PyTorch API
#[pyfunction]
pub fn std(
    input: &PyTensor,
    dim: Option<Vec<i32>>,
    keepdim: Option<bool>,
    unbiased: Option<bool>,
) -> PyResult<PyTensor> {
    standard_deviation(input, dim, keepdim, unbiased)
}

/// Clamp (clip) function
#[pyfunction]
pub fn clamp(input: &PyTensor, min_val: Option<f32>, max_val: Option<f32>) -> PyResult<PyTensor> {
    if min_val.is_none() && max_val.is_none() {
        return Err(PyValueError::new_err(
            "At least one of min_val or max_val must be provided",
        ));
    }

    let min = min_val.unwrap_or(f32::NEG_INFINITY);
    let max = max_val.unwrap_or(f32::INFINITY);

    match input.tensor.clamp(min, max) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Clamp failed: {}", e))),
    }
}

/// Element-wise comparison functions
#[pyfunction]
pub fn eq(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::eq(&input.tensor, &other.tensor) {
        Ok(bool_tensor) => {
            // Convert boolean tensor to f32 tensor (0.0 for false, 1.0 for true)
            let data: Vec<f32> = bool_tensor
                .as_slice()
                .ok_or_else(|| PyRuntimeError::new_err("Cannot access tensor data on GPU tensor"))?
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            let f32_tensor = Tensor::from_vec(data, bool_tensor.shape().dims())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tensor: {}", e)))?;
            Ok(PyTensor {
                tensor: Arc::new(f32_tensor),
                requires_grad: false, // Comparison operations don't require gradients
                is_pinned: false,
            })
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Eq failed: {}", e))),
    }
}

#[pyfunction]
pub fn ne(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::ne(&input.tensor, &other.tensor) {
        Ok(bool_tensor) => {
            // Convert boolean tensor to f32 tensor (0.0 for false, 1.0 for true)
            let data: Vec<f32> = bool_tensor
                .as_slice()
                .ok_or_else(|| PyRuntimeError::new_err("Cannot access tensor data on GPU tensor"))?
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            let f32_tensor = Tensor::from_vec(data, bool_tensor.shape().dims())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tensor: {}", e)))?;
            Ok(PyTensor {
                tensor: Arc::new(f32_tensor),
                requires_grad: false,
                is_pinned: false,
            })
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Ne failed: {}", e))),
    }
}

#[pyfunction]
pub fn lt(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::lt(&input.tensor, &other.tensor) {
        Ok(bool_tensor) => {
            // Convert boolean tensor to f32 tensor (0.0 for false, 1.0 for true)
            let data: Vec<f32> = bool_tensor
                .as_slice()
                .ok_or_else(|| PyRuntimeError::new_err("Cannot access tensor data on GPU tensor"))?
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            let f32_tensor = Tensor::from_vec(data, bool_tensor.shape().dims())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tensor: {}", e)))?;
            Ok(PyTensor {
                tensor: Arc::new(f32_tensor),
                requires_grad: false,
                is_pinned: false,
            })
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Lt failed: {}", e))),
    }
}

#[pyfunction]
pub fn le(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::le(&input.tensor, &other.tensor) {
        Ok(bool_tensor) => {
            // Convert boolean tensor to f32 tensor (0.0 for false, 1.0 for true)
            let data: Vec<f32> = bool_tensor
                .as_slice()
                .ok_or_else(|| PyRuntimeError::new_err("Cannot access tensor data on GPU tensor"))?
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            let f32_tensor = Tensor::from_vec(data, bool_tensor.shape().dims())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tensor: {}", e)))?;
            Ok(PyTensor {
                tensor: Arc::new(f32_tensor),
                requires_grad: false,
                is_pinned: false,
            })
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Le failed: {}", e))),
    }
}

#[pyfunction]
pub fn gt(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::gt(&input.tensor, &other.tensor) {
        Ok(bool_tensor) => {
            // Convert boolean tensor to f32 tensor (0.0 for false, 1.0 for true)
            let data: Vec<f32> = bool_tensor
                .as_slice()
                .ok_or_else(|| PyRuntimeError::new_err("Cannot access tensor data on GPU tensor"))?
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            let f32_tensor = Tensor::from_vec(data, bool_tensor.shape().dims())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tensor: {}", e)))?;
            Ok(PyTensor {
                tensor: Arc::new(f32_tensor),
                requires_grad: false,
                is_pinned: false,
            })
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Gt failed: {}", e))),
    }
}

#[pyfunction]
pub fn ge(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::ge(&input.tensor, &other.tensor) {
        Ok(bool_tensor) => {
            // Convert boolean tensor to f32 tensor (0.0 for false, 1.0 for true)
            let data: Vec<f32> = bool_tensor
                .as_slice()
                .ok_or_else(|| PyRuntimeError::new_err("Cannot access tensor data on GPU tensor"))?
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            let f32_tensor = Tensor::from_vec(data, bool_tensor.shape().dims())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tensor: {}", e)))?;
            Ok(PyTensor {
                tensor: Arc::new(f32_tensor),
                requires_grad: false,
                is_pinned: false,
            })
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Ge failed: {}", e))),
    }
}

/// Argmax function
#[pyfunction]
pub fn argmax(input: &PyTensor, dim: Option<i32>, keepdim: Option<bool>) -> PyResult<PyTensor> {
    let keep_dims = keepdim.unwrap_or(false);

    match tenflowers_core::ops::argmax(&input.tensor, dim, keep_dims) {
        Ok(index_tensor) => {
            // Convert usize indices to f32
            let data: Vec<f32> = index_tensor
                .as_slice()
                .ok_or_else(|| PyRuntimeError::new_err("Cannot access tensor data on GPU tensor"))?
                .iter()
                .map(|&idx| idx as f32)
                .collect();
            let f32_tensor = Tensor::from_vec(data, index_tensor.shape().dims())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tensor: {}", e)))?;
            Ok(PyTensor {
                tensor: Arc::new(f32_tensor),
                requires_grad: false, // Argmax returns indices, not differentiable
                is_pinned: false,
            })
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Argmax failed: {}", e))),
    }
}

/// Argmin function
#[pyfunction]
pub fn argmin(input: &PyTensor, dim: Option<i32>, keepdim: Option<bool>) -> PyResult<PyTensor> {
    let keep_dims = keepdim.unwrap_or(false);

    match tenflowers_core::ops::argmin(&input.tensor, dim, keep_dims) {
        Ok(index_tensor) => {
            // Convert usize indices to f32
            let data: Vec<f32> = index_tensor
                .as_slice()
                .ok_or_else(|| PyRuntimeError::new_err("Cannot access tensor data on GPU tensor"))?
                .iter()
                .map(|&idx| idx as f32)
                .collect();
            let f32_tensor = Tensor::from_vec(data, index_tensor.shape().dims())
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create tensor: {}", e)))?;
            Ok(PyTensor {
                tensor: Arc::new(f32_tensor),
                requires_grad: false, // Argmin returns indices, not differentiable
                is_pinned: false,
            })
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Argmin failed: {}", e))),
    }
}

/// Concatenate tensors along a dimension
#[pyfunction]
pub fn cat(tensors: &Bound<'_, PyList>, dim: Option<i32>) -> PyResult<PyTensor> {
    let tensor_vec: Vec<PyRef<PyTensor>> = tensors
        .iter()
        .map(|item| item.extract::<PyRef<PyTensor>>().map_err(Into::into))
        .collect::<PyResult<Vec<_>>>()?;
    let axis = dim.unwrap_or(0);

    if tensor_vec.is_empty() {
        return Err(PyValueError::new_err(
            "Cannot concatenate empty list of tensors",
        ));
    }

    let tensor_refs: Vec<&Tensor<f32>> = tensor_vec.iter().map(|t| &*t.tensor).collect();
    let requires_grad = tensor_vec.iter().any(|t| t.requires_grad);

    match tenflowers_core::ops::concat(&tensor_refs, axis as usize) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad,
            is_pinned: false,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Cat failed: {e}"))),
    }
}

/// Stack tensors along a new dimension
#[pyfunction]
pub fn stack(tensors: &Bound<'_, PyList>, dim: Option<i32>) -> PyResult<PyTensor> {
    let tensor_vec: Vec<PyRef<PyTensor>> = tensors
        .iter()
        .map(|item| item.extract::<PyRef<PyTensor>>().map_err(Into::into))
        .collect::<PyResult<Vec<_>>>()?;
    let axis = dim.unwrap_or(0);

    if tensor_vec.is_empty() {
        return Err(PyValueError::new_err("Cannot stack empty list of tensors"));
    }

    let tensor_refs: Vec<&Tensor<f32>> = tensor_vec.iter().map(|t| &*t.tensor).collect();
    let requires_grad = tensor_vec.iter().any(|t| t.requires_grad);

    match tenflowers_core::ops::stack(&tensor_refs, axis as usize) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad,
            is_pinned: false,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Stack failed: {e}"))),
    }
}

/// Split tensor into chunks
#[pyfunction]
pub fn split(
    input: &PyTensor,
    split_size_or_sections: Vec<usize>,
    dim: Option<i32>,
) -> PyResult<Vec<PyTensor>> {
    let axis = dim.unwrap_or(0);

    let num_splits = split_size_or_sections.first().cloned().unwrap_or(1);
    match tenflowers_core::ops::split(&input.tensor, num_splits, axis as usize) {
        Ok(tensors) => {
            let result: Vec<PyTensor> = tensors
                .into_iter()
                .map(|tensor| PyTensor {
                    tensor: Arc::new(tensor),
                    requires_grad: input.requires_grad,
                    is_pinned: input.is_pinned,
                })
                .collect();
            Ok(result)
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Split failed: {e}"))),
    }
}

/// Squeeze dimensions of size 1
#[pyfunction]
pub fn squeeze(input: &PyTensor, dim: Option<Vec<usize>>) -> PyResult<PyTensor> {
    let result = if let Some(axes) = dim {
        tenflowers_core::ops::manipulation::squeeze(&input.tensor, Some(&axes))
    } else {
        tenflowers_core::ops::manipulation::squeeze(&input.tensor, None)
    };

    match result {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Squeeze failed: {e}"))),
    }
}

/// Add singleton dimensions
#[pyfunction]
pub fn unsqueeze(input: &PyTensor, dim: i32) -> PyResult<PyTensor> {
    let axes = vec![dim as usize];
    match tenflowers_core::ops::unsqueeze(&input.tensor, &axes) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Unsqueeze failed: {e}"))),
    }
}

/// Flatten tensor
#[pyfunction]
pub fn flatten(
    input: &PyTensor,
    start_dim: Option<i32>,
    end_dim: Option<i32>,
) -> PyResult<PyTensor> {
    let start = start_dim.unwrap_or(0) as usize;
    let end = end_dim.unwrap_or(-1);

    let actual_end = if end == -1 {
        input.tensor.ndim() - 1
    } else {
        end as usize
    };

    match tenflowers_core::ops::flatten(&input.tensor) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Flatten failed: {e}"))),
    }
}
