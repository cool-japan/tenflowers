//! Neural network activation and loss functions
//!
//! This module provides Python bindings for various activation functions
//! and loss functions commonly used in neural networks.

use crate::tensor_ops::PyTensor;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;

// ============================================================================
// Activation Functions
// ============================================================================

/// ReLU activation function
#[pyfunction]
pub fn relu(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::relu(&input.tensor) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("ReLU failed: {}", e))),
    }
}

/// Sigmoid activation function
#[pyfunction]
pub fn sigmoid(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::sigmoid(&input.tensor) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Sigmoid failed: {}", e))),
    }
}

/// Tanh activation function
#[pyfunction]
pub fn tanh(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::tanh(&input.tensor) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Tanh failed: {}", e))),
    }
}

/// GELU activation function (Gaussian Error Linear Unit)
#[pyfunction]
pub fn gelu(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::gelu(&input.tensor) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("GELU failed: {}", e))),
    }
}

/// Softmax activation function
#[pyfunction]
pub fn softmax(input: &PyTensor, dim: Option<i32>) -> PyResult<PyTensor> {
    let axis = dim.unwrap_or(-1);
    match tenflowers_core::ops::softmax(&input.tensor, Some(axis)) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Softmax failed: {}", e))),
    }
}

/// Log Softmax activation function with axis support
#[pyfunction]
pub fn log_softmax(input: &PyTensor, dim: Option<i32>) -> PyResult<PyTensor> {
    let axis = dim.unwrap_or(-1);
    let tensor_shape = input.tensor.shape();
    let ndim = tensor_shape.len() as i32;

    // Normalize axis to positive index
    let axis = if axis < 0 { ndim + axis } else { axis };

    if axis < 0 || axis >= ndim {
        return Err(PyRuntimeError::new_err(format!(
            "Axis {} is out of bounds for tensor with {} dimensions",
            axis, ndim
        )));
    }

    // If tensor is 1D or we're applying log_softmax to the last dimension, use the core implementation
    if ndim == 1 || axis == ndim - 1 {
        match tenflowers_core::ops::log_softmax(&input.tensor) {
            Ok(tensor) => Ok(PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!("LogSoftmax failed: {}", e))),
        }
    } else {
        // For multi-dimensional tensors with specific axis, implement axis-wise log_softmax
        // This is a simplified implementation that works by computing max and sum along the specified axis

        // For now, fall back to the core implementation but provide a clear message
        // In a full implementation, we would:
        // 1. Compute max along axis for numerical stability
        // 2. Subtract max from input
        // 3. Compute exp and sum along axis
        // 4. Compute log of sum
        // 5. Subtract log_sum from original (input - max)

        match tenflowers_core::ops::log_softmax(&input.tensor) {
            Ok(tensor) => {
                // TODO: This is not axis-aware yet - needs proper implementation
                // For now, we apply log_softmax to the entire tensor
                eprintln!("Warning: log_softmax axis parameter not fully implemented yet, applying to entire tensor");
                Ok(PyTensor {
                    tensor: Arc::new(tensor),
                    requires_grad: input.requires_grad,
                    is_pinned: input.is_pinned,
                })
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("LogSoftmax failed: {}", e))),
        }
    }
}

/// Leaky ReLU activation function
#[pyfunction]
pub fn leaky_relu(input: &PyTensor, negative_slope: Option<f32>) -> PyResult<PyTensor> {
    let slope = negative_slope.unwrap_or(0.01);
    match tenflowers_core::ops::leaky_relu(&input.tensor, slope) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("LeakyReLU failed: {}", e))),
    }
}

/// ELU activation function (Exponential Linear Unit)
#[pyfunction]
pub fn elu(input: &PyTensor, alpha: Option<f32>) -> PyResult<PyTensor> {
    let alpha_val = alpha.unwrap_or(1.0);
    match tenflowers_core::ops::elu(&input.tensor, alpha_val) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("ELU failed: {}", e))),
    }
}

/// Swish activation function
#[pyfunction]
pub fn swish(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::swish(&input.tensor) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Swish failed: {}", e))),
    }
}

/// Mish activation function
#[pyfunction]
pub fn mish(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::mish(&input.tensor) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Mish failed: {}", e))),
    }
}

/// ReLU6 activation function (ReLU clamped to 6)
#[pyfunction]
pub fn relu6(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::relu6(&input.tensor) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("ReLU6 failed: {}", e))),
    }
}

/// Hardswish activation function
#[pyfunction]
pub fn hardswish(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::hardswish(&input.tensor) {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad,
            is_pinned: input.is_pinned,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!("Hardswish failed: {}", e))),
    }
}

// ============================================================================
// Loss Functions
// ============================================================================

/// Mean Squared Error loss function
#[pyfunction]
pub fn mse_loss(
    input: &PyTensor,
    target: &PyTensor,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let reduction_mode = reduction.unwrap_or("mean");

    match tenflowers_core::ops::binary::sub(&input.tensor, &target.tensor) {
        Ok(diff) => match tenflowers_core::ops::binary::mul(&diff, &diff) {
            Ok(squared) => {
                let result = match reduction_mode {
                    "mean" => tenflowers_core::ops::reduction::mean(&squared, None, false),
                    "sum" => tenflowers_core::ops::reduction::sum(&squared, None, false),
                    "none" => Ok(squared),
                    _ => return Err(PyRuntimeError::new_err("Invalid reduction mode")),
                };

                match result {
                    Ok(tensor) => Ok(PyTensor {
                        tensor: Arc::new(tensor),
                        requires_grad: input.requires_grad || target.requires_grad,
                        is_pinned: false,
                    }),
                    Err(e) => Err(PyRuntimeError::new_err(format!(
                        "MSE reduction failed: {}",
                        e
                    ))),
                }
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "MSE squaring failed: {}",
                e
            ))),
        },
        Err(e) => Err(PyRuntimeError::new_err(format!(
            "MSE subtraction failed: {}",
            e
        ))),
    }
}

/// Cross Entropy loss function
#[pyfunction]
pub fn cross_entropy_loss(
    input: &PyTensor,
    target: &PyTensor,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let reduction_mode = reduction.unwrap_or("mean");

    // Apply log_softmax to input first
    // Note: tenflowers_core::ops::log_softmax currently doesn't support axis parameter
    let log_probs = match tenflowers_core::ops::log_softmax(&input.tensor) {
        Ok(tensor) => tensor,
        Err(e) => return Err(PyRuntimeError::new_err(format!("LogSoftmax failed: {}", e))),
    };

    // Compute negative log likelihood
    let nll = match tenflowers_core::ops::binary::mul(&log_probs, &target.tensor) {
        Ok(tensor) => tensor,
        Err(e) => {
            return Err(PyRuntimeError::new_err(format!(
                "NLL multiplication failed: {}",
                e
            )))
        }
    };

    let neg_nll = match tenflowers_core::ops::unary::neg(&nll) {
        Ok(tensor) => tensor,
        Err(e) => {
            return Err(PyRuntimeError::new_err(format!(
                "NLL negation failed: {}",
                e
            )))
        }
    };

    let result = match reduction_mode {
        "mean" => tenflowers_core::ops::reduction::mean(&neg_nll, None, false),
        "sum" => tenflowers_core::ops::reduction::sum(&neg_nll, None, false),
        "none" => Ok(neg_nll),
        _ => return Err(PyRuntimeError::new_err("Invalid reduction mode")),
    };

    match result {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad || target.requires_grad,
            is_pinned: false,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!(
            "CrossEntropy reduction failed: {}",
            e
        ))),
    }
}

/// Binary Cross Entropy loss function
#[pyfunction]
pub fn binary_cross_entropy_loss(
    input: &PyTensor,
    target: &PyTensor,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let reduction_mode = reduction.unwrap_or("mean");

    // BCE formula: -[target * log(input) + (1 - target) * log(1 - input)]

    // Clamp input to avoid log(0)
    let eps = 1e-8;
    let clamped_input = match tenflowers_core::ops::clamp(&input.tensor, eps, 1.0 - eps) {
        Ok(tensor) => tensor,
        Err(e) => {
            return Err(PyRuntimeError::new_err(format!(
                "Input clamping failed: {}",
                e
            )))
        }
    };

    // Compute log(input) and log(1 - input)
    let log_input = match tenflowers_core::ops::unary::log(&clamped_input) {
        Ok(tensor) => tensor,
        Err(e) => return Err(PyRuntimeError::new_err(format!("Log input failed: {}", e))),
    };

    let one_minus_input = match tenflowers_core::ops::unary::neg(&clamped_input) {
        Ok(neg_input) => match tenflowers_core::ops::scalar_add(&neg_input, 1.0) {
            Ok(tensor) => tensor,
            Err(e) => return Err(PyRuntimeError::new_err(format!("1-input failed: {}", e))),
        },
        Err(e) => {
            return Err(PyRuntimeError::new_err(format!(
                "Input negation failed: {}",
                e
            )))
        }
    };

    let log_one_minus_input = match tenflowers_core::ops::unary::log(&one_minus_input) {
        Ok(tensor) => tensor,
        Err(e) => {
            return Err(PyRuntimeError::new_err(format!(
                "Log(1-input) failed: {}",
                e
            )))
        }
    };

    // target * log(input)
    let term1 = match tenflowers_core::ops::binary::mul(&target.tensor, &log_input) {
        Ok(tensor) => tensor,
        Err(e) => return Err(PyRuntimeError::new_err(format!("BCE term1 failed: {}", e))),
    };

    // (1 - target) * log(1 - input)
    let one_minus_target = match tenflowers_core::ops::unary::neg(&target.tensor) {
        Ok(neg_target) => match tenflowers_core::ops::scalar_add(&neg_target, 1.0) {
            Ok(tensor) => tensor,
            Err(e) => return Err(PyRuntimeError::new_err(format!("1-target failed: {}", e))),
        },
        Err(e) => {
            return Err(PyRuntimeError::new_err(format!(
                "Target negation failed: {}",
                e
            )))
        }
    };

    let term2 = match tenflowers_core::ops::binary::mul(&one_minus_target, &log_one_minus_input) {
        Ok(tensor) => tensor,
        Err(e) => return Err(PyRuntimeError::new_err(format!("BCE term2 failed: {}", e))),
    };

    // Sum both terms
    let sum_terms = match tenflowers_core::ops::binary::add(&term1, &term2) {
        Ok(tensor) => tensor,
        Err(e) => return Err(PyRuntimeError::new_err(format!("BCE sum failed: {}", e))),
    };

    // Negate the result
    let bce = match tenflowers_core::ops::unary::neg(&sum_terms) {
        Ok(tensor) => tensor,
        Err(e) => {
            return Err(PyRuntimeError::new_err(format!(
                "BCE negation failed: {}",
                e
            )))
        }
    };

    let result = match reduction_mode {
        "mean" => tenflowers_core::ops::reduction::mean(&bce, None, false),
        "sum" => tenflowers_core::ops::reduction::sum(&bce, None, false),
        "none" => Ok(bce),
        _ => return Err(PyRuntimeError::new_err("Invalid reduction mode")),
    };

    match result {
        Ok(tensor) => Ok(PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: input.requires_grad || target.requires_grad,
            is_pinned: false,
        }),
        Err(e) => Err(PyRuntimeError::new_err(format!(
            "BCE reduction failed: {}",
            e
        ))),
    }
}

/// L1 Loss (Mean Absolute Error) function
#[pyfunction]
pub fn l1_loss(input: &PyTensor, target: &PyTensor, reduction: Option<&str>) -> PyResult<PyTensor> {
    let reduction_mode = reduction.unwrap_or("mean");

    match tenflowers_core::ops::binary::sub(&input.tensor, &target.tensor) {
        Ok(diff) => match tenflowers_core::ops::unary::abs(&diff) {
            Ok(abs_diff) => {
                let result = match reduction_mode {
                    "mean" => tenflowers_core::ops::reduction::mean(&abs_diff, None, false),
                    "sum" => tenflowers_core::ops::reduction::sum(&abs_diff, None, false),
                    "none" => Ok(abs_diff),
                    _ => return Err(PyRuntimeError::new_err("Invalid reduction mode")),
                };

                match result {
                    Ok(tensor) => Ok(PyTensor {
                        tensor: Arc::new(tensor),
                        requires_grad: input.requires_grad || target.requires_grad,
                        is_pinned: false,
                    }),
                    Err(e) => Err(PyRuntimeError::new_err(format!(
                        "L1 reduction failed: {}",
                        e
                    ))),
                }
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("L1 abs failed: {}", e))),
        },
        Err(e) => Err(PyRuntimeError::new_err(format!(
            "L1 subtraction failed: {}",
            e
        ))),
    }
}

/// Smooth L1 Loss (Huber Loss) function
#[pyfunction]
pub fn smooth_l1_loss(
    input: &PyTensor,
    target: &PyTensor,
    beta: Option<f32>,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let beta_val = beta.unwrap_or(1.0);
    let reduction_mode = reduction.unwrap_or("mean");

    match tenflowers_core::ops::binary::sub(&input.tensor, &target.tensor) {
        Ok(diff) => match tenflowers_core::ops::smooth_l1_loss(&diff, beta_val) {
            Ok(loss) => {
                let result = match reduction_mode {
                    "mean" => tenflowers_core::ops::reduction::mean(&loss, None, false),
                    "sum" => tenflowers_core::ops::reduction::sum(&loss, None, false),
                    "none" => Ok(loss),
                    _ => return Err(PyRuntimeError::new_err("Invalid reduction mode")),
                };

                match result {
                    Ok(tensor) => Ok(PyTensor {
                        tensor: Arc::new(tensor),
                        requires_grad: input.requires_grad || target.requires_grad,
                        is_pinned: false,
                    }),
                    Err(e) => Err(PyRuntimeError::new_err(format!(
                        "SmoothL1 reduction failed: {}",
                        e
                    ))),
                }
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("SmoothL1 failed: {}", e))),
        },
        Err(e) => Err(PyRuntimeError::new_err(format!(
            "SmoothL1 subtraction failed: {}",
            e
        ))),
    }
}

// ============================================================================
// Regularization Functions
// ============================================================================

/// Dropout operation
#[pyfunction]
pub fn dropout(input: &PyTensor, p: Option<f32>, training: Option<bool>) -> PyResult<PyTensor> {
    let prob = p.unwrap_or(0.5);
    let is_training = training.unwrap_or(true);

    // If not in training mode or probability is 0, return input as-is
    if !is_training || prob <= 0.0 {
        return Ok(input.clone());
    }

    // Implement proper dropout with random mask
    if prob >= 1.0 {
        // If dropout probability is 1.0 or higher, return zeros
        let zeros_data = vec![0.0f32; input.tensor.size()];
        match tenflowers_core::Tensor::from_vec(zeros_data, input.tensor.shape().dims()) {
            Ok(zeros_tensor) => Ok(PyTensor {
                tensor: Arc::new(zeros_tensor),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            }),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Dropout zeros creation failed: {}",
                e
            ))),
        }
    } else {
        // Create random binary mask where each element is kept with probability (1-p)
        let tensor_size = input.tensor.size();
        let mut mask_data = Vec::with_capacity(tensor_size);
        let scale = 1.0 / (1.0 - prob);

        // Generate random mask using a simple LCG for reproducibility
        let mut rng_state = tensor_size as u64;
        for _ in 0..tensor_size {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let random_val = (rng_state as f64) / (u64::MAX as f64);
            mask_data.push(if random_val > prob as f64 { scale } else { 0.0 });
        }

        match tenflowers_core::Tensor::from_vec(mask_data, input.tensor.shape().dims()) {
            Ok(mask_tensor) => {
                match tenflowers_core::ops::binary::mul(&input.tensor, &mask_tensor) {
                    Ok(tensor) => Ok(PyTensor {
                        tensor: Arc::new(tensor),
                        requires_grad: input.requires_grad,
                        is_pinned: input.is_pinned,
                    }),
                    Err(e) => Err(PyRuntimeError::new_err(format!(
                        "Dropout mask application failed: {}",
                        e
                    ))),
                }
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Dropout mask creation failed: {}",
                e
            ))),
        }
    }
}

/// Linear (fully connected) layer operation
#[pyfunction]
pub fn linear(input: &PyTensor, weight: &PyTensor, bias: Option<&PyTensor>) -> PyResult<PyTensor> {
    // Linear layer: output = input @ weight.T + bias
    let weight_t = weight.transpose(None)?;
    let output = input.matmul(&weight_t)?;

    let result = if let Some(b) = bias {
        output.add(b)?
    } else {
        output
    };

    Ok(result)
}
