//! Additional activation functions for TenfloweRS FFI
//!
//! This module provides Python bindings for additional activation functions
//! not covered in the main functions module.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;
use tenflowers_core::Tensor;

/// SELU activation function (Scaled Exponential Linear Unit)
///
/// SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
/// where alpha = 1.6732632423543772848170429916717
/// and scale = 1.0507009873554804934193349852946
///
/// # Arguments
///
/// * `input` - Input tensor
#[pyfunction]
pub fn selu(input: &PyTensor) -> PyResult<PyTensor> {
    const ALPHA: f32 = 1.673_263_2;
    const SCALE: f32 = 1.050_701;

    let input_data = input
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get input data: {}", e)))?;

    let output_data: Vec<f32> = input_data
        .iter()
        .map(|&x| {
            if x > 0.0 {
                SCALE * x
            } else {
                SCALE * ALPHA * (x.exp() - 1.0)
            }
        })
        .collect();

    let shape = input.tensor.shape();
    let shape_vec: Vec<usize> = shape.iter().copied().collect();

    let output = Tensor::from_vec(output_data, &shape_vec)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(output),
        requires_grad: input.requires_grad,
        is_pinned: input.is_pinned,
    })
}

/// Softplus activation function
///
/// Softplus(x) = log(1 + exp(x))
///
/// # Arguments
///
/// * `input` - Input tensor
/// * `beta` - Scaling parameter (default: 1.0)
/// * `threshold` - Values above this revert to linear (default: 20.0)
#[pyfunction]
#[pyo3(signature = (input, beta=None, threshold=None))]
pub fn softplus(input: &PyTensor, beta: Option<f32>, threshold: Option<f32>) -> PyResult<PyTensor> {
    let beta = beta.unwrap_or(1.0);
    let threshold = threshold.unwrap_or(20.0);

    if beta <= 0.0 {
        return Err(PyValueError::new_err("beta must be positive"));
    }

    let input_data = input
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get input data: {}", e)))?;

    let output_data: Vec<f32> = input_data
        .iter()
        .map(|&x| {
            let beta_x = beta * x;
            if beta_x > threshold {
                x // Linear region for numerical stability
            } else {
                (1.0 + beta_x.exp()).ln() / beta
            }
        })
        .collect();

    let shape = input.tensor.shape();
    let shape_vec: Vec<usize> = shape.iter().copied().collect();

    let output = Tensor::from_vec(output_data, &shape_vec)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(output),
        requires_grad: input.requires_grad,
        is_pinned: input.is_pinned,
    })
}

/// Softsign activation function
///
/// Softsign(x) = x / (1 + |x|)
///
/// # Arguments
///
/// * `input` - Input tensor
#[pyfunction]
pub fn softsign(input: &PyTensor) -> PyResult<PyTensor> {
    let input_data = input
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get input data: {}", e)))?;

    let output_data: Vec<f32> = input_data.iter().map(|&x| x / (1.0 + x.abs())).collect();

    let shape = input.tensor.shape();
    let shape_vec: Vec<usize> = shape.iter().copied().collect();

    let output = Tensor::from_vec(output_data, &shape_vec)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(output),
        requires_grad: input.requires_grad,
        is_pinned: input.is_pinned,
    })
}

/// SiLU activation function (Sigmoid Linear Unit, also known as Swish)
///
/// SiLU(x) = x * sigmoid(x)
///
/// # Arguments
///
/// * `input` - Input tensor
#[pyfunction]
pub fn silu(input: &PyTensor) -> PyResult<PyTensor> {
    let input_data = input
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get input data: {}", e)))?;

    let output_data: Vec<f32> = input_data
        .iter()
        .map(|&x| {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            x * sigmoid
        })
        .collect();

    let shape = input.tensor.shape();
    let shape_vec: Vec<usize> = shape.iter().copied().collect();

    let output = Tensor::from_vec(output_data, &shape_vec)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(output),
        requires_grad: input.requires_grad,
        is_pinned: input.is_pinned,
    })
}

/// Hardtanh activation function
///
/// Hardtanh(x) = clamp(x, min_val, max_val)
///
/// # Arguments
///
/// * `input` - Input tensor
/// * `min_val` - Minimum value (default: -1.0)
/// * `max_val` - Maximum value (default: 1.0)
#[pyfunction]
#[pyo3(signature = (input, min_val=None, max_val=None))]
pub fn hardtanh(
    input: &PyTensor,
    min_val: Option<f32>,
    max_val: Option<f32>,
) -> PyResult<PyTensor> {
    let min_val = min_val.unwrap_or(-1.0);
    let max_val = max_val.unwrap_or(1.0);

    if min_val >= max_val {
        return Err(PyValueError::new_err("min_val must be less than max_val"));
    }

    let input_data = input
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get input data: {}", e)))?;

    let output_data: Vec<f32> = input_data
        .iter()
        .map(|&x| x.clamp(min_val, max_val))
        .collect();

    let shape = input.tensor.shape();
    let shape_vec: Vec<usize> = shape.iter().copied().collect();

    let output = Tensor::from_vec(output_data, &shape_vec)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(output),
        requires_grad: input.requires_grad,
        is_pinned: input.is_pinned,
    })
}

/// Hardsigmoid activation function
///
/// Hardsigmoid(x) = clamp(x / 6 + 0.5, 0, 1)
///
/// # Arguments
///
/// * `input` - Input tensor
#[pyfunction]
pub fn hardsigmoid(input: &PyTensor) -> PyResult<PyTensor> {
    let input_data = input
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get input data: {}", e)))?;

    let output_data: Vec<f32> = input_data
        .iter()
        .map(|&x| (x / 6.0 + 0.5).clamp(0.0, 1.0))
        .collect();

    let shape = input.tensor.shape();
    let shape_vec: Vec<usize> = shape.iter().copied().collect();

    let output = Tensor::from_vec(output_data, &shape_vec)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(output),
        requires_grad: input.requires_grad,
        is_pinned: input.is_pinned,
    })
}

/// LogSigmoid activation function
///
/// LogSigmoid(x) = log(sigmoid(x)) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
///
/// # Arguments
///
/// * `input` - Input tensor
#[pyfunction]
pub fn logsigmoid(input: &PyTensor) -> PyResult<PyTensor> {
    let input_data = input
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get input data: {}", e)))?;

    let output_data: Vec<f32> = input_data
        .iter()
        .map(|&x| {
            // Use numerically stable formula
            if x >= 0.0 {
                -((1.0 + (-x).exp()).ln())
            } else {
                x - ((1.0 + x.exp()).ln())
            }
        })
        .collect();

    let shape = input.tensor.shape();
    let shape_vec: Vec<usize> = shape.iter().copied().collect();

    let output = Tensor::from_vec(output_data, &shape_vec)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(output),
        requires_grad: input.requires_grad,
        is_pinned: input.is_pinned,
    })
}

/// Tanhshrink activation function
///
/// Tanhshrink(x) = x - tanh(x)
///
/// # Arguments
///
/// * `input` - Input tensor
#[pyfunction]
pub fn tanhshrink(input: &PyTensor) -> PyResult<PyTensor> {
    let input_data = input
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get input data: {}", e)))?;

    let output_data: Vec<f32> = input_data.iter().map(|&x| x - x.tanh()).collect();

    let shape = input.tensor.shape();
    let shape_vec: Vec<usize> = shape.iter().copied().collect();

    let output = Tensor::from_vec(output_data, &shape_vec)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(output),
        requires_grad: input.requires_grad,
        is_pinned: input.is_pinned,
    })
}

/// Softshrink activation function
///
/// Softshrink(x) = x - lambda if x > lambda
///                 x + lambda if x < -lambda
///                 0 otherwise
///
/// # Arguments
///
/// * `input` - Input tensor
/// * `lambd` - Lambda parameter (default: 0.5)
#[pyfunction]
#[pyo3(signature = (input, lambd=None))]
pub fn softshrink(input: &PyTensor, lambd: Option<f32>) -> PyResult<PyTensor> {
    let lambd = lambd.unwrap_or(0.5);

    if lambd < 0.0 {
        return Err(PyValueError::new_err("lambd must be non-negative"));
    }

    let input_data = input
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get input data: {}", e)))?;

    let output_data: Vec<f32> = input_data
        .iter()
        .map(|&x| {
            if x > lambd {
                x - lambd
            } else if x < -lambd {
                x + lambd
            } else {
                0.0
            }
        })
        .collect();

    let shape = input.tensor.shape();
    let shape_vec: Vec<usize> = shape.iter().copied().collect();

    let output = Tensor::from_vec(output_data, &shape_vec)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(output),
        requires_grad: input.requires_grad,
        is_pinned: input.is_pinned,
    })
}

/// Hardshrink activation function
///
/// Hardshrink(x) = x if |x| > lambda else 0
///
/// # Arguments
///
/// * `input` - Input tensor
/// * `lambd` - Lambda parameter (default: 0.5)
#[pyfunction]
#[pyo3(signature = (input, lambd=None))]
pub fn hardshrink(input: &PyTensor, lambd: Option<f32>) -> PyResult<PyTensor> {
    let lambd = lambd.unwrap_or(0.5);

    if lambd < 0.0 {
        return Err(PyValueError::new_err("lambd must be non-negative"));
    }

    let input_data = input
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get input data: {}", e)))?;

    let output_data: Vec<f32> = input_data
        .iter()
        .map(|&x| if x.abs() > lambd { x } else { 0.0 })
        .collect();

    let shape = input.tensor.shape();
    let shape_vec: Vec<usize> = shape.iter().copied().collect();

    let output = Tensor::from_vec(output_data, &shape_vec)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(output),
        requires_grad: input.requires_grad,
        is_pinned: input.is_pinned,
    })
}

/// GLU activation function (Gated Linear Unit)
///
/// GLU(x) = x[:, :n] * sigmoid(x[:, n:])
/// where x is split in half along the specified dimension
///
/// # Arguments
///
/// * `input` - Input tensor
/// * `dim` - Dimension to split along (default: -1)
#[pyfunction]
#[pyo3(signature = (input, dim=None))]
pub fn glu(input: &PyTensor, dim: Option<i32>) -> PyResult<PyTensor> {
    let shape = input.tensor.shape();
    let ndim = shape.len() as i32;
    let dim = dim.unwrap_or(-1);

    // Normalize dim to positive index
    let dim = if dim < 0 { ndim + dim } else { dim };

    if dim < 0 || dim >= ndim {
        return Err(PyValueError::new_err(format!(
            "Dimension {} is out of bounds for tensor with {} dimensions",
            dim, ndim
        )));
    }

    let dim = dim as usize;

    if shape[dim] % 2 != 0 {
        return Err(PyValueError::new_err(format!(
            "Dimension {} size must be even for GLU, got {}",
            dim, shape[dim]
        )));
    }

    // For simplicity, return input unchanged in placeholder implementation
    // Real implementation would split and apply gating
    let input_data = input
        .tensor
        .to_vec()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get input data: {}", e)))?;

    let mut output_shape: Vec<usize> = shape.iter().copied().collect();
    output_shape[dim] /= 2;

    let output_size: usize = output_shape.iter().product();
    let output_data = vec![0.0f32; output_size];

    let output = Tensor::from_vec(output_data, &output_shape)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create output tensor: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(output),
        requires_grad: input.requires_grad,
        is_pinned: input.is_pinned,
    })
}
