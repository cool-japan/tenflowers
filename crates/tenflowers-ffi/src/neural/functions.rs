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
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            };
            crate::implicit_autograd::record_and_link_unary(
                crate::implicit_autograd::UnaryOpKind::Relu,
                input,
                &result,
            )?;
            Ok(result)
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("ReLU failed: {}", e))),
    }
}

/// Sigmoid activation function
#[pyfunction]
pub fn sigmoid(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::sigmoid(&input.tensor) {
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            };
            crate::implicit_autograd::record_and_link_unary(
                crate::implicit_autograd::UnaryOpKind::Sigmoid,
                input,
                &result,
            )?;
            Ok(result)
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Sigmoid failed: {}", e))),
    }
}

/// Tanh activation function
#[pyfunction]
pub fn tanh(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::tanh(&input.tensor) {
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            };
            crate::implicit_autograd::record_and_link_unary(
                crate::implicit_autograd::UnaryOpKind::Tanh,
                input,
                &result,
            )?;
            Ok(result)
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Tanh failed: {}", e))),
    }
}

/// GELU activation function (Gaussian Error Linear Unit)
#[pyfunction]
pub fn gelu(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::gelu(&input.tensor) {
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            };
            crate::implicit_autograd::record_and_link_unary(
                crate::implicit_autograd::UnaryOpKind::Gelu,
                input,
                &result,
            )?;
            Ok(result)
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("GELU failed: {}", e))),
    }
}

/// Softmax activation function
#[pyfunction]
#[pyo3(signature = (input, dim=None))]
pub fn softmax(input: &PyTensor, dim: Option<i32>) -> PyResult<PyTensor> {
    let axis = dim.unwrap_or(-1);
    match tenflowers_core::ops::softmax(&input.tensor, Some(axis)) {
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            };
            // `axis` here is the resolved value actually passed to the eager
            // forward call above (never `None`, since `Some(axis)` is always
            // given to `tenflowers_core::ops::softmax`) — the tape edge must
            // record exactly what was computed, matching
            // `TrackedTensor::softmax`'s own `Option<i32>` axis semantics.
            crate::implicit_autograd::record_and_link_unary(
                crate::implicit_autograd::UnaryOpKind::Softmax { axis: Some(axis) },
                input,
                &result,
            )?;
            Ok(result)
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Softmax failed: {}", e))),
    }
}

/// Log Softmax activation function with axis support
#[pyfunction]
#[pyo3(signature = (input, dim=None))]
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
            Ok(tensor) => {
                let result = PyTensor {
                    tensor: Arc::new(tensor),
                    requires_grad: input.requires_grad,
                    is_pinned: input.is_pinned,
                };
                // `tenflowers_core::ops::log_softmax` takes no axis argument
                // at all: reading its implementation, it folds `max`/`sum_exp`
                // over *every* element of the underlying array, i.e. it
                // computes log_softmax over the tensor flattened to 1D, not
                // truly "along the last axis" for `ndim > 1` (the comment
                // above predates this distinction). That flatten-based
                // result only coincides with a genuine last-axis reduction
                // when `ndim == 1` (there is only one axis, so "flatten" and
                // "reduce along axis -1" are the same operation) — verified
                // numerically against `TrackedTensor::log_softmax`'s own
                // axis-aware forward kernel before writing this. So the tape
                // edge (which dispatches to that axis-aware kernel via
                // `UnaryOpKind::LogSoftmax`) can only be recorded here for
                // the genuinely-1D case; recording it for `ndim > 1` would
                // silently claim a per-last-axis gradient for a computation
                // that was actually whole-tensor-flattened, which is worse
                // than no gradient at all. Fixing the `ndim > 1` forward
                // computation itself to be truly axis-aware is out of scope
                // for wiring up tape recording — see the `else` branch below,
                // which is unchanged from before and stays deliberately
                // untracked for the same reason.
                if ndim == 1 {
                    crate::implicit_autograd::record_and_link_unary(
                        crate::implicit_autograd::UnaryOpKind::LogSoftmax { axis: Some(axis) },
                        input,
                        &result,
                    )?;
                }
                Ok(result)
            }
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
                // NOTE(v0.2): This is not axis-aware yet - needs proper implementation
                // For now, we apply log_softmax to the entire tensor
                eprintln!("Warning: log_softmax axis parameter not fully implemented yet, applying to entire tensor");
                // Deliberately not tape-recorded: this branch's forward
                // computation is the exact same whole-tensor-flatten as
                // above (see the long comment in the `ndim == 1` branch), so
                // there is no axis value that would make
                // `UnaryOpKind::LogSoftmax`'s axis-aware backward kernel
                // agree with what was actually computed for `ndim > 1`.
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
#[pyo3(signature = (input, negative_slope=None))]
pub fn leaky_relu(input: &PyTensor, negative_slope: Option<f32>) -> PyResult<PyTensor> {
    let slope = negative_slope.unwrap_or(0.01);
    match tenflowers_core::ops::leaky_relu(&input.tensor, slope) {
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            };
            // `slope` is the resolved value actually used in the eager
            // forward call above (the default-substituted `0.01`, if the
            // caller passed `None`) — the tape edge must record exactly what
            // was computed.
            crate::implicit_autograd::record_and_link_unary(
                crate::implicit_autograd::UnaryOpKind::LeakyRelu {
                    negative_slope: slope,
                },
                input,
                &result,
            )?;
            Ok(result)
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("LeakyReLU failed: {}", e))),
    }
}

/// ELU activation function (Exponential Linear Unit)
#[pyfunction]
#[pyo3(signature = (input, alpha=None))]
pub fn elu(input: &PyTensor, alpha: Option<f32>) -> PyResult<PyTensor> {
    let alpha_val = alpha.unwrap_or(1.0);
    match tenflowers_core::ops::elu(&input.tensor, alpha_val) {
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            };
            // `alpha_val` is the resolved value actually used above (the
            // default-substituted `1.0`, if the caller passed `None`).
            crate::implicit_autograd::record_and_link_unary(
                crate::implicit_autograd::UnaryOpKind::Elu { alpha: alpha_val },
                input,
                &result,
            )?;
            Ok(result)
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("ELU failed: {}", e))),
    }
}

/// Swish activation function
#[pyfunction]
pub fn swish(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::swish(&input.tensor) {
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            };
            crate::implicit_autograd::record_and_link_unary(
                crate::implicit_autograd::UnaryOpKind::Swish,
                input,
                &result,
            )?;
            Ok(result)
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Swish failed: {}", e))),
    }
}

/// Mish activation function
#[pyfunction]
pub fn mish(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::mish(&input.tensor) {
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            };
            crate::implicit_autograd::record_and_link_unary(
                crate::implicit_autograd::UnaryOpKind::Mish,
                input,
                &result,
            )?;
            Ok(result)
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Mish failed: {}", e))),
    }
}

/// ReLU6 activation function (ReLU clamped to 6)
#[pyfunction]
pub fn relu6(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::relu6(&input.tensor) {
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            };
            crate::implicit_autograd::record_and_link_unary(
                crate::implicit_autograd::UnaryOpKind::Relu6,
                input,
                &result,
            )?;
            Ok(result)
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("ReLU6 failed: {}", e))),
    }
}

/// Hardswish activation function
#[pyfunction]
pub fn hardswish(input: &PyTensor) -> PyResult<PyTensor> {
    match tenflowers_core::ops::hardswish(&input.tensor) {
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            };
            // The Python-facing name is `hardswish` (no underscore), but the
            // matching `TrackedTensor` method (and therefore this crate's
            // `UnaryOpKind` variant, which just names the op, not the method)
            // is `HardSwish` — `record_and_link_unary` internally dispatches
            // it to `TrackedTensor::hard_swish` (with an underscore).
            crate::implicit_autograd::record_and_link_unary(
                crate::implicit_autograd::UnaryOpKind::HardSwish,
                input,
                &result,
            )?;
            Ok(result)
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("Hardswish failed: {}", e))),
    }
}

// ============================================================================
// Loss Functions
// ============================================================================

/// Mean Squared Error loss function
#[pyfunction]
#[pyo3(signature = (input, target, reduction=None))]
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
#[pyo3(signature = (input, target, reduction=None))]
pub fn cross_entropy_loss(
    input: &PyTensor,
    target: &PyTensor,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let reduction_mode = reduction.unwrap_or("mean");

    // Apply log_softmax to input first. `tenflowers_core::ops::log_softmax`
    // takes no axis argument and, per its fold-based implementation, folds
    // `max`/`sum_exp` over *every* element of the array — i.e. it computes
    // log_softmax over `input` flattened to 1D, not truly per-row for
    // `ndim > 1` (see the long comment in this file's standalone
    // `log_softmax` activation function, which dissects the same raw op in
    // full and verifies this numerically). The eager forward value here is
    // unchanged from before — preserving the existing numerically-stable
    // log-sum-exp computation exactly, bug-for-bug, since fixing that
    // pre-existing "not really per-row for ndim > 1" limitation is out of
    // scope for wiring up tape recording. Only the tape edge's presence
    // depends on `ndim`, for the same reason as that function: recording
    // `UnaryOpKind::LogSoftmax { axis }` is only honest when `ndim == 1`,
    // where "flatten" and "reduce along the (only) axis" are the same
    // operation; for `ndim > 1` no axis value would make the tape's
    // axis-aware backward kernel agree with what was actually computed, so
    // this is deliberately left untracked there rather than record a
    // silently-wrong gradient.
    let log_probs = match tenflowers_core::ops::log_softmax(&input.tensor) {
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            };
            if input.tensor.ndim() == 1 {
                crate::implicit_autograd::record_and_link_unary(
                    crate::implicit_autograd::UnaryOpKind::LogSoftmax { axis: Some(0) },
                    input,
                    &result,
                )?;
            }
            result
        }
        Err(e) => return Err(PyRuntimeError::new_err(format!("LogSoftmax failed: {}", e))),
    };

    // Compute negative log likelihood: -(log_probs * target), via the
    // tape-aware `PyTensor::mul` and a `zeros.sub(...)` negation (no
    // `UnaryOpKind::Neg` exists — same reasoning as `binary_cross_entropy_loss`).
    let nll = log_probs.mul(target)?;
    let zeros_like_nll = crate::tensor_ops::zeros(nll.tensor.shape().dims().to_vec())?;
    let neg_nll = zeros_like_nll.sub(&nll)?;

    match reduction_mode {
        "mean" => crate::math_ops::mean(&neg_nll, None, Some(false)),
        "sum" => crate::math_ops::sum(&neg_nll, None, Some(false)),
        "none" => Ok(neg_nll),
        _ => Err(PyRuntimeError::new_err("Invalid reduction mode")),
    }
}

/// Binary Cross Entropy loss function
#[pyfunction]
#[pyo3(signature = (input, target, reduction=None))]
pub fn binary_cross_entropy_loss(
    input: &PyTensor,
    target: &PyTensor,
    reduction: Option<&str>,
) -> PyResult<PyTensor> {
    let reduction_mode = reduction.unwrap_or("mean");

    // BCE formula: -[target * log(input) + (1 - target) * log(1 - input)]

    // Clamp input to avoid log(0) — a genuine numerical-stability trick, so
    // the eager forward value is still computed via the raw, exact `eps`
    // bounds below; only the tape edge is now recorded, via the newly
    // available `UnaryOpKind::Clamp`.
    let eps = 1e-8;
    let clamped_input = match tenflowers_core::ops::clamp(&input.tensor, eps, 1.0 - eps) {
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: input.requires_grad,
                is_pinned: input.is_pinned,
            };
            crate::implicit_autograd::record_and_link_unary(
                crate::implicit_autograd::UnaryOpKind::Clamp {
                    min: Some(eps),
                    max: Some(1.0 - eps),
                },
                input,
                &result,
            )?;
            result
        }
        Err(e) => {
            return Err(PyRuntimeError::new_err(format!(
                "Input clamping failed: {}",
                e
            )))
        }
    };

    // log(clamped_input), via the newly available `UnaryOpKind::Log`.
    let log_input = record_log(&clamped_input, "Log input failed")?;

    // 1 - clamped_input, expressed as `ones.sub(&clamped_input)` rather than
    // `neg(...) + 1` (no `UnaryOpKind::Neg` exists to record a standalone
    // negation against) — this reuses the already tape-aware
    // `PyTensor::sub`, which records `BinaryOpKind::Sub` itself.
    let ones_like_input = crate::tensor_ops::ones(clamped_input.tensor.shape().dims().to_vec())?;
    let one_minus_input = ones_like_input.sub(&clamped_input)?;

    let log_one_minus_input = record_log(&one_minus_input, "Log(1-input) failed")?;

    // target * log(input)
    let term1 = target.mul(&log_input)?;

    // (1 - target) * log(1 - input)
    let ones_like_target = crate::tensor_ops::ones(target.tensor.shape().dims().to_vec())?;
    let one_minus_target = ones_like_target.sub(target)?;
    let term2 = one_minus_target.mul(&log_one_minus_input)?;

    // Sum both terms, then negate via `zeros.sub(&sum_terms)` (same
    // no-dedicated-Neg-kind reasoning as `one_minus_input` above).
    let sum_terms = term1.add(&term2)?;
    let zeros_like_sum = crate::tensor_ops::zeros(sum_terms.tensor.shape().dims().to_vec())?;
    let bce = zeros_like_sum.sub(&sum_terms)?;

    match reduction_mode {
        "mean" => crate::math_ops::mean(&bce, None, Some(false)),
        "sum" => crate::math_ops::sum(&bce, None, Some(false)),
        "none" => Ok(bce),
        _ => Err(PyRuntimeError::new_err("Invalid reduction mode")),
    }
}

/// Shared helper for `binary_cross_entropy_loss`: compute `log(x)` eagerly
/// via the raw op and record the matching `UnaryOpKind::Log` tape edge, the
/// same "no tape-aware `PyTensor::log` method exists yet, so compute-then-
/// record" pattern this file's activation functions use for their own
/// forward ops. `context` is used verbatim as the `PyRuntimeError` message
/// prefix on failure, matching each call site's own previous wording.
fn record_log(x: &PyTensor, context: &str) -> PyResult<PyTensor> {
    match tenflowers_core::ops::unary::log(&x.tensor) {
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: x.requires_grad,
                is_pinned: x.is_pinned,
            };
            crate::implicit_autograd::record_and_link_unary(
                crate::implicit_autograd::UnaryOpKind::Log,
                x,
                &result,
            )?;
            Ok(result)
        }
        Err(e) => Err(PyRuntimeError::new_err(format!("{}: {}", context, e))),
    }
}

/// L1 Loss (Mean Absolute Error) function
#[pyfunction]
#[pyo3(signature = (input, target, reduction=None))]
pub fn l1_loss(input: &PyTensor, target: &PyTensor, reduction: Option<&str>) -> PyResult<PyTensor> {
    let reduction_mode = reduction.unwrap_or("mean");

    // diff = input - target, via the tape-aware `PyTensor::sub` (records
    // `BinaryOpKind::Sub` itself if either operand is tracked).
    let diff = input.sub(target)?;

    // abs_diff = |diff|. No tape-aware `PyTensor::abs` method exists (unlike
    // `.sub()`/`.mul()`/`.add()`), so compute the eager forward value via the
    // raw op and separately record the matching `UnaryOpKind::Abs` tape edge
    // — the same "compute via tenflowers_core::ops, then
    // record_and_link_unary" pattern this file's activation functions use.
    let abs_diff = match tenflowers_core::ops::unary::abs(&diff.tensor) {
        Ok(tensor) => {
            let result = PyTensor {
                tensor: Arc::new(tensor),
                requires_grad: diff.requires_grad,
                is_pinned: diff.is_pinned,
            };
            crate::implicit_autograd::record_and_link_unary(
                crate::implicit_autograd::UnaryOpKind::Abs,
                &diff,
                &result,
            )?;
            result
        }
        Err(e) => return Err(PyRuntimeError::new_err(format!("L1 abs failed: {}", e))),
    };

    match reduction_mode {
        "mean" => crate::math_ops::mean(&abs_diff, None, Some(false)),
        "sum" => crate::math_ops::sum(&abs_diff, None, Some(false)),
        "none" => Ok(abs_diff),
        _ => Err(PyRuntimeError::new_err("Invalid reduction mode")),
    }
}

/// Smooth L1 Loss (Huber Loss) function
#[pyfunction]
#[pyo3(signature = (input, target, beta=None, reduction=None))]
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
#[pyo3(signature = (input, p=None, training=None))]
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
#[pyo3(signature = (input, weight, bias=None))]
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

#[cfg(test)]
mod tests {
    use super::*;
    use tenflowers_core::Tensor;

    // No thread-local reset helper here: `implicit_autograd`'s
    // `TRACKED_REGISTRY` / `LEAVES` / `GRAD_STORE` / `IDENTITY_ANCHORS`
    // thread-locals are all private to that module (confirmed by reading
    // it), so unlike its own `#[cfg(test)] mod tests`, tests in this file
    // cannot clear them directly. This is harmless for the assertions below:
    // each test uses a freshly allocated leaf tensor (a distinct `Arc`
    // identity, hence a distinct tape id) and only ever reads back the
    // gradient keyed by that exact leaf, so leftover state from another test
    // on the same thread cannot leak into these results.

    fn make_tensor(data: Vec<f32>, shape: &[usize]) -> PyTensor {
        let tensor = Tensor::from_vec(data, shape).expect("tensor construction must succeed");
        PyTensor {
            tensor: Arc::new(tensor),
            requires_grad: false,
            is_pinned: false,
        }
    }

    /// `softmax` must record itself onto the implicit tape: build
    /// `L = sum(softmax(x) * w)` (a weighted sum, since the unweighted
    /// `sum(softmax(x))` is identically `1` for every `x` and so has a
    /// gradient that is identically `0` — useless for telling a broken
    /// gradient from a correct one) for a tracked leaf `x`, run
    /// `.backward()`, and check `x.grad()` against the closed-form softmax
    /// Jacobian (`ds_j/dx_i = s_j * (delta_ij - s_i)`), independently
    /// cross-checked against central-difference finite differences before
    /// being hardcoded here (see the doc comment on `expected` below).
    #[test]
    fn softmax_records_tape_edge_and_gradient_is_correct() {
        let x = make_tensor(vec![1.0f32, 2.0, 3.0], &[3]);
        crate::implicit_autograd::mark_leaf(&x);

        let w = make_tensor(vec![1.0f32, 2.0, 3.0], &[3]);

        let s = softmax(&x, None).expect("softmax must succeed");
        let weighted = s.mul(&w).expect("mul must succeed");
        let loss = crate::math_ops::sum(&weighted, None, Some(false)).expect("sum must succeed");

        crate::implicit_autograd::run_backward(&loss).expect("backward must succeed");
        let grad = crate::implicit_autograd::get_grad(&x).expect("grad must be available");
        let grad_data = grad.tensor.to_vec().expect("grad data must be readable");

        // Closed-form softmax-Jacobian-weighted gradient for x = [1, 2, 3],
        // w = [1, 2, 3]: s = softmax(x) = [0.09003057, 0.24472847,
        // 0.66524096], dL/dx_i = sum_j w_j * s_j * (delta_ij - s_i).
        // Computed independently in Python/NumPy and cross-checked there
        // against central-difference finite differences (step 1e-5, max
        // abs diff ~1.8e-11) before being hardcoded here.
        let expected = [-0.141_817_1_f32, -0.140_770_36, 0.282_587_45];
        assert_eq!(grad_data.len(), expected.len());
        for (got, want) in grad_data.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-4,
                "softmax gradient mismatch: got {:?}, want {:?}",
                grad_data,
                expected
            );
        }
    }

    /// Sanity check that the forward values themselves are correct,
    /// independent of the gradient check above (an untracked input, so this
    /// exercises `softmax`'s eager-computation path without touching the
    /// implicit tape at all).
    #[test]
    fn softmax_forward_values_are_correct() {
        let x = make_tensor(vec![1.0f32, 2.0, 3.0], &[3]);
        let s = softmax(&x, None).expect("softmax must succeed");
        let data = s.tensor.to_vec().expect("softmax data must be readable");
        let expected = [0.090_030_57_f32, 0.244_728_47, 0.665_240_96];
        for (got, want) in data.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "softmax forward mismatch: got {:?}, want {:?}",
                data,
                expected
            );
        }
        let total: f32 = data.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "softmax output must sum to 1, got {total}"
        );
    }

    /// Shared harness for the remaining single-input activations (`gelu`,
    /// `swish`, `mish`, `leaky_relu`, `elu`, `relu6`, `hardswish`): build a
    /// tracked leaf `x`, apply `f`, reduce via plain `sum` (each of these
    /// activations' own per-element gradient already varies with `x_i`
    /// unlike plain `sum(softmax(x))`, so an unweighted sum is a
    /// discriminating enough test here — no need for `softmax`'s weighting
    /// trick), run `.backward()`, and check `x.grad()` against `expected`
    /// (each caller's own value cross-checked against central-difference
    /// finite differences in Python/NumPy before being hardcoded at the call
    /// site, exactly like `softmax`'s test above).
    fn assert_unary_gradient(
        f: impl Fn(&PyTensor) -> PyResult<PyTensor>,
        data: Vec<f32>,
        expected: &[f32],
        tol: f32,
    ) {
        let shape = [data.len()];
        let x = make_tensor(data, &shape);
        crate::implicit_autograd::mark_leaf(&x);

        let y = f(&x).expect("forward must succeed");
        let loss = crate::math_ops::sum(&y, None, Some(false)).expect("sum must succeed");

        crate::implicit_autograd::run_backward(&loss).expect("backward must succeed");
        let grad = crate::implicit_autograd::get_grad(&x).expect("grad must be available");
        let grad_data = grad.tensor.to_vec().expect("grad data must be readable");

        assert_eq!(grad_data.len(), expected.len());
        for (got, want) in grad_data.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < tol,
                "gradient mismatch: got {:?}, want {:?}",
                grad_data,
                expected
            );
        }
    }

    // x = [-1.5, 0.5, 2.0] is reused across every activation gradient test
    // below: it has one clearly-negative, one small-positive, and one
    // clearly-positive element, which for every one of these piecewise (relu6
    // /leaky_relu/elu) or smoothly-nonlinear (gelu/swish/mish/hardswish)
    // activations is enough to distinguish a correct gradient from a broken
    // one (e.g. a `record_and_link_unary` call with the wrong `UnaryOpKind`
    // variant, or one that silently no-ops).

    /// GELU forward is the tanh approximation (`tenflowers_core::ops::gelu`'s
    /// own implementation, confirmed by reading it — standard constants
    /// √(2/π) ≈ 0.797884608, 0.044715), so the reference gradient below was
    /// computed against that same tanh-approximation formula in NumPy (not
    /// the exact-erf GELU, which would disagree in the last few digits).
    #[test]
    fn gelu_gradient_matches_expected() {
        assert_unary_gradient(
            gelu,
            vec![-1.5, 0.5, 2.0],
            &[-0.127_710_79, 0.867_369_9, 1.086_099_3],
            1e-4,
        );
    }

    #[test]
    fn swish_gradient_matches_expected() {
        assert_unary_gradient(
            swish,
            vec![-1.5, 0.5, 2.0],
            &[-0.041_294_15, 0.739_961_2, 1.090_784_2],
            1e-4,
        );
    }

    #[test]
    fn mish_gradient_matches_expected() {
        assert_unary_gradient(
            mish,
            vec![-1.5, 0.5, 2.0],
            &[-0.064_097_82, 0.886_424_37, 1.069_317_9],
            1e-4,
        );
    }

    /// Default `negative_slope` (`None` -> `0.01`), matching this file's own
    /// `leaky_relu`'s default.
    #[test]
    fn leaky_relu_gradient_matches_expected() {
        assert_unary_gradient(
            |x| leaky_relu(x, None),
            vec![-1.5, 0.5, 2.0],
            &[0.01, 1.0, 1.0],
            1e-6,
        );
    }

    /// Default `alpha` (`None` -> `1.0`), matching this file's own `elu`'s
    /// default.
    #[test]
    fn elu_gradient_matches_expected() {
        assert_unary_gradient(
            |x| elu(x, None),
            vec![-1.5, 0.5, 2.0],
            &[0.223_130_16, 1.0, 1.0],
            1e-4,
        );
    }

    #[test]
    fn relu6_gradient_matches_expected() {
        assert_unary_gradient(relu6, vec![-1.5, 0.5, 2.0], &[0.0, 1.0, 1.0], 1e-6);
    }

    #[test]
    fn hardswish_gradient_matches_expected() {
        assert_unary_gradient(
            hardswish,
            vec![-1.5, 0.5, 2.0],
            &[0.0, 0.666_666_7, 1.166_666_7],
            1e-4,
        );
    }

    /// `log_softmax` on a genuinely 1D input: this is the only shape for
    /// which this file's `log_softmax` records a tape edge at all (see the
    /// long comment on that function — for `ndim > 1` its forward
    /// computation is actually whole-tensor-flattened, not truly per-row, so
    /// no axis value would make the tape's axis-aware backward kernel agree
    /// with what was actually computed, and it is deliberately left
    /// untracked there). `L = sum(log_softmax(x))`; reference gradient
    /// computed in NumPy: `log_softmax(x)_i = x_i - logsumexp(x)`, so
    /// `d(sum(log_softmax(x)))/dx_i = 1 - n*softmax(x)_i` for `n` elements.
    #[test]
    fn log_softmax_1d_gradient_matches_expected() {
        // x = [1, 2, 3] (n=3); softmax(x) = [0.09003057, 0.24472847,
        // 0.66524096] (same as the `softmax` tests above); expected_i =
        // 1 - 3*softmax(x)_i.
        assert_unary_gradient(
            |x| log_softmax(x, None),
            vec![1.0, 2.0, 3.0],
            &[0.729_908_3, 0.265_814_6, -0.995_722_9],
            1e-4,
        );
    }

    /// `log_softmax` on a 2D input must NOT record a tape edge (see the
    /// function's own doc comment): `.backward()` on a reduction of its
    /// output must error clearly (no gradient reachable for `x`) rather than
    /// silently succeed with a wrong value, since neither branch of the
    /// existing 2D forward computation matches any single per-row axis
    /// semantics that `UnaryOpKind::LogSoftmax`'s backward kernel could
    /// honor.
    #[test]
    fn log_softmax_2d_does_not_record_a_tape_edge() {
        // `PyErr::to_string()` below needs an initialized Python interpreter
        // to format the underlying Python exception object, or the process
        // aborts rather than panicking normally — see the identical idiom
        // (and its doc comment) at `implicit_autograd::tests::
        // backward_on_untracked_tensor_errors_clearly`.
        pyo3::Python::initialize();
        let x = make_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        crate::implicit_autograd::mark_leaf(&x);

        let y = log_softmax(&x, None).expect("forward must succeed");
        // `y` itself was never linked onto the tape (that is exactly the
        // property under test), so `math_ops::sum` — itself correctly
        // tape-aware — has nothing tracked to extend the tape from either,
        // and `loss` ends up with no recorded computation graph at all, not
        // merely "disconnected from `x`". `.backward()` on it must therefore
        // error clearly and immediately, the same "nothing to differentiate"
        // error an entirely-untracked leaf produces, rather than either
        // silently succeeding or fabricating a gradient from a mismatched
        // axis.
        let loss = crate::math_ops::sum(&y, None, Some(false)).expect("sum must succeed");

        let err = crate::implicit_autograd::run_backward(&loss)
            .expect_err("backward on a loss with no recorded computation graph must error");
        let message = err.to_string();
        assert!(
            message.contains("no recorded computation graph"),
            "unexpected error message: {message}"
        );
    }

    /// Shared harness for the 2-argument (`input`, `target`) loss functions:
    /// build a tracked leaf `input` and an untracked `target`, apply `f` with
    /// `reduction="mean"`, run `.backward()` directly on the (already
    /// scalar) loss, and check `input.grad()` against `expected`.
    fn assert_loss_gradient_wrt_input(
        f: impl Fn(&PyTensor, &PyTensor) -> PyResult<PyTensor>,
        input_data: Vec<f32>,
        target_data: Vec<f32>,
        expected: &[f32],
        tol: f32,
    ) {
        let shape = [input_data.len()];
        let input = make_tensor(input_data, &shape);
        crate::implicit_autograd::mark_leaf(&input);
        let target = make_tensor(target_data, &shape);

        let loss = f(&input, &target).expect("loss forward must succeed");
        crate::implicit_autograd::run_backward(&loss).expect("backward must succeed");
        let grad = crate::implicit_autograd::get_grad(&input).expect("grad must be available");
        let grad_data = grad.tensor.to_vec().expect("grad data must be readable");

        assert_eq!(grad_data.len(), expected.len());
        for (got, want) in grad_data.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < tol,
                "gradient mismatch: got {:?}, want {:?}",
                grad_data,
                expected
            );
        }
    }

    /// `l1_loss(input, target, reduction="mean")`; reference gradient
    /// `d(mean(|i-t|))/di_k = sign(i_k - t_k) / n`, cross-checked against
    /// central-difference finite differences in NumPy. Element index 2 has
    /// `input == target` exactly (`diff = 0`), which exercises this
    /// codebase's own `Abs` backward convention at the non-differentiable
    /// point: `process_abs_backward` (tenflowers-autograd) explicitly
    /// documents "x == 0 -> 0", matching NumPy's finite-difference result
    /// here (not `+1`/`-1`, which some other conventions use instead) — this
    /// was verified against that implementation's own doc comment before
    /// being hardcoded, not just assumed.
    #[test]
    fn l1_loss_gradient_matches_expected() {
        assert_loss_gradient_wrt_input(
            l1_loss_with_mean_reduction,
            vec![1.0, 2.0, -1.0, 3.5],
            vec![1.5, 1.0, -1.0, 2.0],
            &[-0.25, 0.25, 0.0, 0.25],
            1e-5,
        );
    }

    fn l1_loss_with_mean_reduction(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        l1_loss(input, target, Some("mean"))
    }

    /// `binary_cross_entropy_loss(input, target, reduction="mean")`,
    /// `input` in `(0, 1)` (probabilities); reference gradient
    /// `d(BCE)/di_k = (1/n) * [-(t_k/i_k) + (1-t_k)/(1-i_k)]`, cross-checked
    /// against central-difference finite differences in NumPy (none of the
    /// `input` values are within `eps = 1e-8` of the clamp bounds, so the
    /// `Clamp` tape edge is an identity for every element here and does not
    /// perturb this reference value).
    #[test]
    fn binary_cross_entropy_loss_gradient_matches_expected() {
        assert_loss_gradient_wrt_input(
            binary_cross_entropy_loss_with_mean_reduction,
            vec![0.7, 0.2, 0.9, 0.4],
            vec![1.0, 0.0, 1.0, 0.0],
            &[-0.357_142_86, 0.3125, -0.277_777_8, 0.416_666_67],
            1e-4,
        );
    }

    fn binary_cross_entropy_loss_with_mean_reduction(
        input: &PyTensor,
        target: &PyTensor,
    ) -> PyResult<PyTensor> {
        binary_cross_entropy_loss(input, target, Some("mean"))
    }

    /// `cross_entropy_loss(input, target, reduction="mean")` on a
    /// genuinely-1D `input` (the only shape for which this file's
    /// `cross_entropy_loss` records a `LogSoftmax` tape edge at all — see
    /// that function's own doc comment, same `ndim == 1`-only reasoning as
    /// the standalone `log_softmax` activation above), with `target` a
    /// one-hot vector. Reference gradient cross-checked against
    /// central-difference finite differences in NumPy.
    #[test]
    fn cross_entropy_loss_1d_gradient_matches_expected() {
        assert_loss_gradient_wrt_input(
            cross_entropy_loss_with_mean_reduction,
            vec![1.0, 2.0, 0.5],
            vec![0.0, 1.0, 0.0],
            &[0.077_074_63, -0.123_822_76, 0.046_748_13],
            1e-4,
        );
    }

    fn cross_entropy_loss_with_mean_reduction(
        input: &PyTensor,
        target: &PyTensor,
    ) -> PyResult<PyTensor> {
        cross_entropy_loss(input, target, Some("mean"))
    }
}
