//! Integration tests for `gradient_validation_framework`'s Finiteness,
//! ZeroForConstants, Linearity, and ChainRule checks with a real (test-local)
//! `GradientExecutor` registered.
//!
//! These live in `tests/` (a separate process/binary per Rust's default test
//! harness) rather than as inline `#[cfg(test)]` unit tests in `src/`, because
//! `GradientExecutor` registration is process-global (`OnceLock`): registering a
//! mock here must never be visible to the crate's own unit tests (in
//! `src/gradient_validation_framework.rs`), which rely on no executor being
//! registered to exercise the "honest, unverified" code path.

use tenflowers_core::gradient_executor::{register_gradient_executor, GradientExecutor};
use tenflowers_core::gradient_validation_framework::{
    GradientProperty, GradientTestCase, GradientValidator,
};
use tenflowers_core::{DType, Result, Shape, Tensor, TensorError};

/// A small, self-contained analytical `GradientExecutor` used only by this test
/// binary. It computes real, closed-form derivatives (not fabricated results)
/// for a handful of elementwise operations, independent of `tenflowers-autograd`
/// (`tenflowers-core` cannot depend on `tenflowers-autograd`, so this crate's own
/// tests need their own minimal executor).
struct MockGradientExecutor;

fn two(inputs: &[Tensor<f64>]) -> Result<(&Tensor<f64>, &Tensor<f64>)> {
    if inputs.len() != 2 {
        return Err(TensorError::invalid_argument(format!(
            "expected 2 inputs, got {}",
            inputs.len()
        )));
    }
    Ok((&inputs[0], &inputs[1]))
}

fn unary_forward(stage: &str, x: &Tensor<f64>) -> Result<Tensor<f64>> {
    let data: Vec<f64> = match stage {
        "sigmoid" => x.data().iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect(),
        "tanh" => x.data().iter().map(|&v| v.tanh()).collect(),
        other => {
            return Err(TensorError::not_implemented_simple(format!(
                "MockGradientExecutor: unsupported stage '{other}'"
            )))
        }
    };
    Tensor::from_vec(data, x.shape().dims())
}

fn unary_local_grad(stage: &str, x: &Tensor<f64>) -> Result<Vec<f64>> {
    match stage {
        "sigmoid" => Ok(x
            .data()
            .iter()
            .map(|&v| {
                let s = 1.0 / (1.0 + (-v).exp());
                s * (1.0 - s)
            })
            .collect()),
        "tanh" => Ok(x
            .data()
            .iter()
            .map(|&v| {
                let t = v.tanh();
                1.0 - t * t
            })
            .collect()),
        other => Err(TensorError::not_implemented_simple(format!(
            "MockGradientExecutor: unsupported stage '{other}'"
        ))),
    }
}

impl GradientExecutor for MockGradientExecutor {
    fn compute_gradient(&self, op: &str, inputs: &[Tensor<f64>]) -> Result<Vec<Tensor<f64>>> {
        if op.contains("->") {
            let stages: Vec<&str> = op.split("->").collect();
            let x = inputs.first().ok_or_else(|| {
                TensorError::invalid_argument("composed op requires 1 input".to_string())
            })?;

            let mut values = vec![x.clone()];
            for stage in &stages {
                let prev = values.last().ok_or_else(|| {
                    TensorError::invalid_operation_simple("no previous stage value".to_string())
                })?;
                values.push(unary_forward(stage, prev)?);
            }

            let mut grad: Vec<f64> = vec![1.0; x.data().len()];
            for (i, stage) in stages.iter().enumerate().rev() {
                let stage_input = &values[i];
                let local = unary_local_grad(stage, stage_input)?;
                grad = grad.iter().zip(local.iter()).map(|(g, l)| g * l).collect();
            }
            return Ok(vec![Tensor::from_vec(grad, x.shape().dims())?]);
        }

        match op {
            "add" => {
                let (a, b) = two(inputs)?;
                Ok(vec![
                    Tensor::from_vec(vec![1.0; a.data().len()], a.shape().dims())?,
                    Tensor::from_vec(vec![1.0; b.data().len()], b.shape().dims())?,
                ])
            }
            "mul" => {
                let (a, b) = two(inputs)?;
                Ok(vec![b.clone(), a.clone()])
            }
            "div" => {
                let (a, b) = two(inputs)?;
                let grad_a: Vec<f64> = b.data().iter().map(|v| 1.0 / v).collect();
                let grad_b: Vec<f64> = a
                    .data()
                    .iter()
                    .zip(b.data().iter())
                    .map(|(av, bv)| -av / (bv * bv))
                    .collect();
                Ok(vec![
                    Tensor::from_vec(grad_a, a.shape().dims())?,
                    Tensor::from_vec(grad_b, b.shape().dims())?,
                ])
            }
            "sigmoid" | "tanh" => {
                let x = inputs.first().ok_or_else(|| {
                    TensorError::invalid_argument(format!("{op} requires 1 input"))
                })?;
                let grad = unary_local_grad(op, x)?;
                Ok(vec![Tensor::from_vec(grad, x.shape().dims())?])
            }
            other => Err(TensorError::not_implemented_simple(format!(
                "MockGradientExecutor: unsupported operation '{other}'"
            ))),
        }
    }
}

fn ensure_registered() {
    register_gradient_executor(Box::new(MockGradientExecutor));
}

#[test]
fn finiteness_passes_for_normal_add_gradient() {
    ensure_registered();
    let validator = GradientValidator::new();
    let test_case = GradientTestCase::new(
        "add",
        DType::Float64,
        vec![Shape::from_slice(&[2, 3]), Shape::from_slice(&[2, 3])],
    );
    let result = validator.validate_test_case(test_case);
    let prop = result
        .property_results
        .get(&GradientProperty::Finiteness)
        .expect("Finiteness should have been checked");
    assert!(prop.passed, "expected Finiteness to pass: {}", prop.details);
}

#[test]
fn finiteness_fails_for_genuine_non_finite_gradient() {
    ensure_registered();
    let validator = GradientValidator::new();
    // synthesize_inputs makes the *second* operand start at exactly zero, so
    // "div" naturally produces a real (not fabricated) Inf/NaN gradient here.
    let test_case = GradientTestCase::new("div", DType::Float64, vec![Shape::from_slice(&[3])]);
    let result = validator.validate_test_case(test_case);
    let prop = result
        .property_results
        .get(&GradientProperty::Finiteness)
        .expect("Finiteness should have been checked");
    assert!(
        !prop.passed,
        "expected Finiteness to fail for div-by-zero gradient"
    );
    assert!(
        prop.details.contains("non-finite"),
        "details should explain the non-finite failure: {}",
        prop.details
    );
}

#[test]
fn zero_for_constants_passes_with_registered_executor() {
    ensure_registered();
    let validator = GradientValidator::new();
    let test_case = GradientTestCase::new("mul", DType::Float64, vec![Shape::from_slice(&[2, 2])])
        .with_property(GradientProperty::ZeroForConstants);
    let result = validator.validate_test_case(test_case);
    let prop = result
        .property_results
        .get(&GradientProperty::ZeroForConstants)
        .expect("ZeroForConstants should have been checked");
    assert!(
        prop.passed,
        "expected ZeroForConstants to pass: {}",
        prop.details
    );
}

#[test]
fn linearity_passes_with_registered_executor() {
    ensure_registered();
    let validator = GradientValidator::new();
    let test_case = GradientTestCase::new("add", DType::Float64, vec![Shape::from_slice(&[2, 3])])
        .with_property(GradientProperty::Linearity);
    let result = validator.validate_test_case(test_case);
    let prop = result
        .property_results
        .get(&GradientProperty::Linearity)
        .expect("Linearity should have been checked");
    assert!(prop.passed, "expected Linearity to pass: {}", prop.details);
}

#[test]
fn chain_rule_passes_with_registered_executor() {
    ensure_registered();
    let validator = GradientValidator::new();
    let test_case =
        GradientTestCase::new("sigmoid", DType::Float64, vec![Shape::from_slice(&[2, 3])])
            .with_property(GradientProperty::ChainRule);
    let result = validator.validate_test_case(test_case);
    let prop = result
        .property_results
        .get(&GradientProperty::ChainRule)
        .expect("ChainRule should have been checked");
    assert!(prop.passed, "expected ChainRule to pass: {}", prop.details);
}
