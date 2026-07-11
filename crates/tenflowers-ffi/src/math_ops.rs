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
#[pyo3(signature = (input, dim=None, keepdim=None))]
pub fn sum(input: &PyTensor, dim: Option<Vec<i32>>, keepdim: Option<bool>) -> PyResult<PyTensor> {
    let keep_dims = keepdim.unwrap_or(false);
    let axes = dim.as_deref();

    match tenflowers_core::ops::sum(&input.tensor, axes, keep_dims) {
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            };
            crate::implicit_autograd::record_and_link_unary(
                crate::implicit_autograd::UnaryOpKind::Sum {
                    axes: dim,
                    keepdims: keep_dims,
                },
                input,
                &result,
            )?;
            Ok(result)
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Sum failed: {}", e))),
    }
}

/// Mean reduction
#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=None))]
pub fn mean(input: &PyTensor, dim: Option<Vec<i32>>, keepdim: Option<bool>) -> PyResult<PyTensor> {
    let keep_dims = keepdim.unwrap_or(false);
    let axes = dim.as_deref();

    match tenflowers_core::ops::mean(&input.tensor, axes, keep_dims) {
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            };
            crate::implicit_autograd::record_and_link_unary(
                crate::implicit_autograd::UnaryOpKind::Mean {
                    axes: dim,
                    keepdims: keep_dims,
                },
                input,
                &result,
            )?;
            Ok(result)
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Mean failed: {}", e))),
    }
}

/// Maximum reduction
#[pyfunction]
#[pyo3(signature = (input, dim=None, keepdim=None))]
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
#[pyo3(signature = (input, dim=None, keepdim=None))]
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
#[pyo3(signature = (input, dim=None, keepdim=None, unbiased=None))]
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
#[pyo3(signature = (input, dim=None, keepdim=None, unbiased=None))]
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
#[pyo3(signature = (input, dim=None, keepdim=None, unbiased=None))]
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
#[pyo3(signature = (input, min_val=None, max_val=None))]
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
#[pyo3(signature = (input, dim=None, keepdim=None))]
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
#[pyo3(signature = (input, dim=None, keepdim=None))]
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
///
/// `dim` follows Python's negative-indexing convention: `-1` means "the last
/// axis", normalized as `ndim + dim` against the *input* tensors' rank
/// (concat does not introduce a new dimension, so the valid range is
/// `0..ndim`) — mirroring `TrackedTensor::concat`'s own normalization.
#[pyfunction]
#[pyo3(signature = (tensors, dim=None))]
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

    let ndim = tensor_vec[0].tensor.ndim() as i32;
    let actual_axis = if axis < 0 {
        (ndim + axis) as usize
    } else {
        axis as usize
    };

    let tensor_refs: Vec<&Tensor<f32>> = tensor_vec.iter().map(|t| &*t.tensor).collect();
    let requires_grad = tensor_vec.iter().any(|t| t.requires_grad);

    match tenflowers_core::ops::concat(&tensor_refs, actual_axis) {
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad,
                is_pinned: false,
            };
            let refs: Vec<&PyTensor> = tensor_vec.iter().map(|t| &**t).collect();
            crate::implicit_autograd::record_and_link_variadic(
                crate::implicit_autograd::VariadicOpKind::Concat {
                    axis: actual_axis as i32,
                },
                &refs,
                &result,
            )?;
            Ok(result)
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Cat failed: {e}"))),
    }
}

/// Stack tensors along a new dimension
///
/// `dim` follows Python's negative-indexing convention. Unlike [`cat`],
/// stack inserts a *new* dimension, so the valid (and negative-normalized)
/// range is `0..=ndim` (one more than concat's `0..ndim`) — mirroring
/// `TrackedTensor::stack`'s own normalization.
#[pyfunction]
#[pyo3(signature = (tensors, dim=None))]
pub fn stack(tensors: &Bound<'_, PyList>, dim: Option<i32>) -> PyResult<PyTensor> {
    let tensor_vec: Vec<PyRef<PyTensor>> = tensors
        .iter()
        .map(|item| item.extract::<PyRef<PyTensor>>().map_err(Into::into))
        .collect::<PyResult<Vec<_>>>()?;
    let axis = dim.unwrap_or(0);

    if tensor_vec.is_empty() {
        return Err(PyValueError::new_err("Cannot stack empty list of tensors"));
    }

    let ndim = tensor_vec[0].tensor.ndim() as i32;
    let actual_axis = if axis < 0 {
        (ndim + 1 + axis) as usize
    } else {
        axis as usize
    };

    let tensor_refs: Vec<&Tensor<f32>> = tensor_vec.iter().map(|t| &*t.tensor).collect();
    let requires_grad = tensor_vec.iter().any(|t| t.requires_grad);

    match tenflowers_core::ops::stack(&tensor_refs, actual_axis) {
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad,
                is_pinned: false,
            };
            let refs: Vec<&PyTensor> = tensor_vec.iter().map(|t| &**t).collect();
            crate::implicit_autograd::record_and_link_variadic(
                crate::implicit_autograd::VariadicOpKind::Stack {
                    axis: actual_axis as i32,
                },
                &refs,
                &result,
            )?;
            Ok(result)
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Stack failed: {e}"))),
    }
}

/// Split tensor into chunks
#[pyfunction]
#[pyo3(signature = (input, split_size_or_sections, dim=None))]
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
#[pyo3(signature = (input, dim=None))]
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
#[pyo3(signature = (input, start_dim=None, end_dim=None))]
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

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyList;

    // Test isolation note: see the identical reasoning documented in
    // `crate::tensor_ops::tests` — `implicit_autograd`'s `reset_for_test` is
    // a private helper unreachable from this module, but every test here
    // builds its own self-contained graph and calls `.backward()` exactly
    // once synchronously, and `run_backward` clears the shared
    // tape/registry/leaves immediately upon returning, so no two tests can
    // observe each other's in-flight state even if the test harness reuses
    // OS threads across tests.

    fn make_tensor(data: Vec<f32>, shape: &[usize]) -> PyTensor {
        let tensor =
            tenflowers_core::Tensor::from_vec(data, shape).expect("tensor construction must succeed");
        PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: false,
            is_pinned: false,
        }
    }

    #[test]
    fn cat_links_onto_tape_and_grad_is_correct() {
        pyo3::Python::initialize();
        let mut a = make_tensor(vec![1.0, 2.0], &[2]);
        let mut b = make_tensor(vec![3.0, 4.0], &[2]);
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        let c = pyo3::Python::attach(|py| -> PyResult<PyTensor> {
            let list = PyList::new(py, [a.clone(), b.clone()])?;
            cat(&list, Some(0))
        })
        .expect("cat must succeed");
        assert_eq!(c.shape(), vec![4]);
        let c_data = c.tensor.to_vec().expect("cat output readable");
        assert_eq!(c_data, vec![1.0, 2.0, 3.0, 4.0]);

        let scalar = sum(&c, None, None).expect("sum must succeed");
        scalar.backward().expect("backward must succeed");

        let grad_a = a.grad().expect("a's grad must be populated");
        let grad_a_data = grad_a.tensor.to_vec().expect("grad readable");
        assert_eq!(grad_a_data, vec![1.0, 1.0]);

        let grad_b = b.grad().expect("b's grad must be populated");
        let grad_b_data = grad_b.tensor.to_vec().expect("grad readable");
        assert_eq!(grad_b_data, vec![1.0, 1.0]);
    }

    /// Proves the negative-axis wraparound fix: `dim=-1` on a 2-D tensor
    /// must normalize to the same axis as the equivalent explicit
    /// `dim=1` (last axis), producing identical forward output — before the
    /// fix, `axis as usize` on a negative `i32` silently wrapped to a huge,
    /// invalid `usize` instead.
    #[test]
    fn cat_negative_axis_matches_explicit_positive_axis() {
        pyo3::Python::initialize();
        let a2d = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b2d = make_tensor(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

        let via_negative = pyo3::Python::attach(|py| -> PyResult<PyTensor> {
            let list = PyList::new(py, [a2d.clone(), b2d.clone()])?;
            cat(&list, Some(-1))
        })
        .expect("cat with dim=-1 must succeed");
        let via_positive = pyo3::Python::attach(|py| -> PyResult<PyTensor> {
            let list = PyList::new(py, [a2d.clone(), b2d.clone()])?;
            cat(&list, Some(1))
        })
        .expect("cat with dim=1 must succeed");

        assert_eq!(via_negative.shape(), via_positive.shape());
        assert_eq!(
            via_negative.tensor.to_vec().expect("readable"),
            via_positive.tensor.to_vec().expect("readable"),
            "dim=-1 must be equivalent to dim=1 (the last axis) for a 2-D input"
        );
    }

    #[test]
    fn stack_links_onto_tape_and_grad_is_correct() {
        pyo3::Python::initialize();
        let mut a = make_tensor(vec![1.0, 2.0], &[2]);
        let mut b = make_tensor(vec![3.0, 4.0], &[2]);
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        let c = pyo3::Python::attach(|py| -> PyResult<PyTensor> {
            let list = PyList::new(py, [a.clone(), b.clone()])?;
            stack(&list, Some(0))
        })
        .expect("stack must succeed");
        assert_eq!(c.shape(), vec![2, 2]);

        let scalar = sum(&c, None, None).expect("sum must succeed");
        scalar.backward().expect("backward must succeed");

        let grad_a = a.grad().expect("a's grad must be populated");
        assert_eq!(grad_a.shape(), vec![2]);
        let grad_a_data = grad_a.tensor.to_vec().expect("grad readable");
        assert_eq!(grad_a_data, vec![1.0, 1.0]);

        let grad_b = b.grad().expect("b's grad must be populated");
        assert_eq!(grad_b.shape(), vec![2]);
        let grad_b_data = grad_b.tensor.to_vec().expect("grad readable");
        assert_eq!(grad_b_data, vec![1.0, 1.0]);
    }

    /// Proves the negative-axis fix for stack, which has a *different*
    /// normalization formula than cat (valid range `0..=ndim`, since stack
    /// inserts a new dimension). For two shape-`[2]` inputs `a=[1,2]`,
    /// `b=[3,4]`: `dim=-1` normalizes to `1 + 1 + (-1) = 1`, i.e. the *last*
    /// axis of the resulting `[2, 2]` tensor, which interleaves values
    /// (`[1,3,2,4]`) — this must match `dim=1` exactly, and must DIFFER from
    /// `dim=0` (which stacks whole rows: `[1,2,3,4]`).
    #[test]
    fn stack_negative_axis_matches_positive_and_differs_from_axis_zero() {
        pyo3::Python::initialize();
        let a = make_tensor(vec![1.0, 2.0], &[2]);
        let b = make_tensor(vec![3.0, 4.0], &[2]);

        let via_negative = pyo3::Python::attach(|py| -> PyResult<PyTensor> {
            let list = PyList::new(py, [a.clone(), b.clone()])?;
            stack(&list, Some(-1))
        })
        .expect("stack with dim=-1 must succeed");
        let via_positive_one = pyo3::Python::attach(|py| -> PyResult<PyTensor> {
            let list = PyList::new(py, [a.clone(), b.clone()])?;
            stack(&list, Some(1))
        })
        .expect("stack with dim=1 must succeed");
        let via_axis_zero = pyo3::Python::attach(|py| -> PyResult<PyTensor> {
            let list = PyList::new(py, [a.clone(), b.clone()])?;
            stack(&list, Some(0))
        })
        .expect("stack with dim=0 must succeed");

        let negative_data = via_negative.tensor.to_vec().expect("readable");
        let positive_one_data = via_positive_one.tensor.to_vec().expect("readable");
        let axis_zero_data = via_axis_zero.tensor.to_vec().expect("readable");

        assert_eq!(
            negative_data, positive_one_data,
            "dim=-1 must be equivalent to dim=1 (the last axis) for shape-[2] inputs"
        );
        assert_eq!(
            positive_one_data,
            vec![1.0, 3.0, 2.0, 4.0],
            "dim=1 must interleave values from a and b"
        );
        assert_eq!(
            axis_zero_data,
            vec![1.0, 2.0, 3.0, 4.0],
            "dim=0 must stack whole rows (a then b), not interleave"
        );
        assert_ne!(
            negative_data, axis_zero_data,
            "dim=-1 (last axis) must produce different data than dim=0 (first axis)"
        );
    }
}
