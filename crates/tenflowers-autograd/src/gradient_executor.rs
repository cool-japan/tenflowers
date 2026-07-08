//! Real, `GradientTape`-backed [`GradientExecutor`] implementation, registered
//! with `tenflowers-core` via [`install_gradient_executor`] (called from
//! `tenflowers_autograd::init()`).
//!
//! Supports a small, deliberately-curated set of operations that have both a
//! genuine forward `TrackedTensor` method and a genuine (non-placeholder)
//! backward implementation in `tape::gradient_computation`: binary {add, sub,
//! mul, div, matmul, pow}, unary {relu, sigmoid, tanh}, plus a small composed-op
//! convention `"stage1->stage2->..."` (unary stages only) used by
//! `tenflowers_core::gradient_validation_framework`'s ChainRule check to obtain a
//! genuinely tape-composed (not manually chain-multiplied) gradient.

use crate::{GradientTape, TrackedTensor};
use tenflowers_core::gradient_executor::{register_gradient_executor, GradientExecutor};
use tenflowers_core::{Result, Tensor, TensorError};

fn apply_unary(stage: &str, t: &TrackedTensor<f64>) -> Result<TrackedTensor<f64>> {
    match stage {
        "relu" => t.relu(),
        "sigmoid" => t.sigmoid(),
        "tanh" => t.tanh(),
        other => Err(TensorError::not_implemented_simple(format!(
            "AutogradGradientExecutor: unary stage '{other}' is not supported in composed \
             chains"
        ))),
    }
}

fn compute_unary_gradient(op: &str, inputs: &[Tensor<f64>]) -> Result<Vec<Tensor<f64>>> {
    let [x] = inputs else {
        return Err(TensorError::invalid_argument(format!(
            "operation '{op}' expects exactly 1 input, got {}",
            inputs.len()
        )));
    };
    let tape = GradientTape::new();
    let xt = tape.watch(x.clone());
    let y = apply_unary(op, &xt)?;
    let grads = tape.gradient(&[y], &[xt])?;
    let g = grads.into_iter().next().flatten().ok_or_else(|| {
        TensorError::invalid_operation_simple(format!(
            "no gradient was recorded for operation '{op}'"
        ))
    })?;
    Ok(vec![g])
}

fn compute_binary_gradient(op: &str, inputs: &[Tensor<f64>]) -> Result<Vec<Tensor<f64>>> {
    let [a, b] = inputs else {
        return Err(TensorError::invalid_argument(format!(
            "operation '{op}' expects exactly 2 inputs, got {}",
            inputs.len()
        )));
    };
    let tape = GradientTape::new();
    let at = tape.watch(a.clone());
    let bt = tape.watch(b.clone());
    let y = match op {
        "add" => at.add(&bt),
        "sub" => at.sub(&bt),
        "mul" => at.mul(&bt),
        "div" => at.div(&bt),
        "matmul" => at.matmul(&bt),
        "pow" => at.pow(&bt),
        other => {
            return Err(TensorError::not_implemented_simple(format!(
                "operation '{other}' is not a supported binary operation"
            )))
        }
    }?;
    let grads = tape.gradient(&[y], &[at, bt])?;
    if grads.len() != 2 {
        return Err(TensorError::invalid_operation_simple(format!(
            "expected 2 gradients for operation '{op}', got {}",
            grads.len()
        )));
    }
    let mut out = Vec::with_capacity(2);
    for (idx, g) in grads.into_iter().enumerate() {
        out.push(g.ok_or_else(|| {
            TensorError::invalid_operation_simple(format!(
                "no gradient was recorded for input {idx} of operation '{op}'"
            ))
        })?);
    }
    Ok(out)
}

fn compute_composed_unary_gradient(op: &str, inputs: &[Tensor<f64>]) -> Result<Vec<Tensor<f64>>> {
    let [x] = inputs else {
        return Err(TensorError::invalid_argument(format!(
            "composed operation '{op}' expects exactly 1 input, got {}",
            inputs.len()
        )));
    };
    let stages: Vec<&str> = op.split("->").collect();
    if stages.is_empty() {
        return Err(TensorError::invalid_argument(format!(
            "composed operation '{op}' has no stages"
        )));
    }

    let tape = GradientTape::new();
    let xt = tape.watch(x.clone());
    let mut current = xt.clone();
    for stage in &stages {
        current = apply_unary(stage, &current)?;
    }
    let grads = tape.gradient(&[current], &[xt])?;
    let g = grads.into_iter().next().flatten().ok_or_else(|| {
        TensorError::invalid_operation_simple(format!(
            "no gradient was recorded for composed operation '{op}'"
        ))
    })?;
    Ok(vec![g])
}

struct AutogradGradientExecutor;

impl GradientExecutor for AutogradGradientExecutor {
    fn compute_gradient(&self, op: &str, inputs: &[Tensor<f64>]) -> Result<Vec<Tensor<f64>>> {
        if op.contains("->") {
            return compute_composed_unary_gradient(op, inputs);
        }
        match op {
            "add" | "sub" | "mul" | "div" | "matmul" | "pow" => compute_binary_gradient(op, inputs),
            "relu" | "sigmoid" | "tanh" => compute_unary_gradient(op, inputs),
            other => Err(TensorError::not_implemented_simple(format!(
                "AutogradGradientExecutor: no forward/backward mapping registered for \
                 operation '{other}'. Supported operations: add, sub, mul, div, matmul, pow, \
                 relu, sigmoid, tanh, and '->'-composed chains of the unary ops."
            ))),
        }
    }
}

/// Register this crate's real, [`GradientTape`]-backed
/// [`tenflowers_core::gradient_executor::GradientExecutor`] with
/// `tenflowers-core`. Idempotent: only the first call takes effect.
pub(crate) fn install_gradient_executor() {
    register_gradient_executor(Box::new(AutogradGradientExecutor));
}
