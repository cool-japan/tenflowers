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
///
/// The input is split in half along `dim`; the first half is multiplied
/// element-wise by the sigmoid of the second half.  The output has half the
/// size of the input along `dim`.
///
/// # Arguments
///
/// * `input` - Input tensor; size along `dim` must be even.
/// * `dim`   - Dimension to split along (default: -1, i.e. last dimension).
#[pyfunction]
#[pyo3(signature = (input, dim=None))]
pub fn glu(input: &PyTensor, dim: Option<i32>) -> PyResult<PyTensor> {
    let shape = input.tensor.shape();
    let ndim = shape.len() as i32;
    let raw_dim = dim.unwrap_or(-1);

    // Normalize dim to a non-negative index.
    let norm_dim = if raw_dim < 0 { ndim + raw_dim } else { raw_dim };

    if norm_dim < 0 || norm_dim >= ndim {
        return Err(PyValueError::new_err(format!(
            "Dimension {} is out of bounds for tensor with {} dimensions",
            raw_dim, ndim
        )));
    }

    let split_axis = norm_dim as usize;

    if shape[split_axis] % 2 != 0 {
        return Err(PyValueError::new_err(format!(
            "Dimension {} size must be even for GLU, got {}",
            split_axis, shape[split_axis]
        )));
    }

    // Split input into two equal halves along split_axis.
    // tenflowers_core::ops::split divides the axis into `num_splits` equal chunks.
    let halves = tenflowers_core::ops::split(&input.tensor, 2, split_axis)
        .map_err(|e| PyRuntimeError::new_err(format!("GLU split failed: {}", e)))?;

    // halves[0] = a  (linear pass-through)
    // halves[1] = b  (gate, passed through sigmoid)
    let a = &halves[0];
    let b = &halves[1];

    // gate = sigmoid(b)
    let gate = b
        .sigmoid()
        .map_err(|e| PyRuntimeError::new_err(format!("GLU sigmoid failed: {}", e)))?;

    // output = a * gate
    let output = a
        .mul(&gate)
        .map_err(|e| PyRuntimeError::new_err(format!("GLU element-wise mul failed: {}", e)))?;

    Ok(PyTensor {
        tensor: Arc::new(output),
        requires_grad: input.requires_grad,
        is_pinned: input.is_pinned,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    #[test]
    fn test_glu_output_shape_and_range() {
        // Input shape [4, 8]: GLU splits along dim=-1 (axis 1, size 8 → 4).
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.6).collect();
        let tensor =
            Tensor::from_vec(data, &[4, 8]).expect("test: tensor construction should succeed");
        let py_tensor = PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: false,
            is_pinned: false,
        };

        let result = glu(&py_tensor, Some(-1)).expect("test: GLU should succeed");

        // Output shape: feature dim halved
        let out_shape = result.tensor.shape();
        assert_eq!(
            out_shape.dims(),
            &[4, 4],
            "GLU must halve the split dimension"
        );

        // Gate values are sigmoid outputs in (0, 1), so |output| <= |input_first_half|.
        // More concretely: result values are finite and within a bounded range.
        let out_data = result.tensor.to_vec().expect("test: to_vec should succeed");
        assert_eq!(out_data.len(), 16, "output must have 4*4 elements");
        for &v in &out_data {
            assert!(v.is_finite(), "GLU output must be finite, got {v}");
        }
    }

    #[test]
    fn test_glu_gate_saturates_correctly() {
        // With very large positive second-half values the gate → 1 and output ≈ first_half.
        // first_half = [1.0, 2.0], second_half = [100.0, 100.0]
        let data = vec![1.0f32, 2.0, 100.0, 100.0];
        let tensor =
            Tensor::from_vec(data, &[1, 4]).expect("test: tensor construction should succeed");
        let py_tensor = PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: false,
            is_pinned: false,
        };

        let result = glu(&py_tensor, Some(-1)).expect("test: GLU should succeed");
        let out_data = result.tensor.to_vec().expect("test: to_vec should succeed");

        // sigmoid(100) ≈ 1.0, so output should be very close to [1.0, 2.0]
        assert!(
            (out_data[0] - 1.0).abs() < 1e-3,
            "output[0] ≈ 1.0, got {}",
            out_data[0]
        );
        assert!(
            (out_data[1] - 2.0).abs() < 1e-3,
            "output[1] ≈ 2.0, got {}",
            out_data[1]
        );
    }

    #[test]
    fn test_glu_gate_suppresses_correctly() {
        // With very large negative second-half values the gate → 0 and output ≈ 0.
        let data = vec![5.0f32, 5.0, -100.0, -100.0];
        let tensor =
            Tensor::from_vec(data, &[1, 4]).expect("test: tensor construction should succeed");
        let py_tensor = PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: false,
            is_pinned: false,
        };

        let result = glu(&py_tensor, Some(-1)).expect("test: GLU should succeed");
        let out_data = result.tensor.to_vec().expect("test: to_vec should succeed");

        // sigmoid(-100) ≈ 0.0, so output should be near 0
        assert!(
            out_data[0].abs() < 1e-3,
            "output[0] ≈ 0.0, got {}",
            out_data[0]
        );
        assert!(
            out_data[1].abs() < 1e-3,
            "output[1] ≈ 0.0, got {}",
            out_data[1]
        );
    }

    #[test]
    fn test_glu_odd_dim_rejected() {
        // An odd-sized split dimension must be rejected.
        let data = vec![1.0f32; 3];
        let tensor =
            Tensor::from_vec(data, &[3]).expect("test: tensor construction should succeed");
        let py_tensor = PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: false,
            is_pinned: false,
        };
        assert!(
            glu(&py_tensor, Some(0)).is_err(),
            "odd split dim must error"
        );
    }
}
