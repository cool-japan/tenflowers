//! End-to-end integration test proving
//! `tenflowers_core::gradient_validation_framework`'s Finiteness,
//! ZeroForConstants, Linearity, and ChainRule checks work with this crate's
//! real, `GradientTape`-backed `GradientExecutor`, once
//! `tenflowers_autograd::init()` has been called.

use tenflowers_core::gradient_validation_framework::{
    GradientProperty, GradientTestCase, GradientValidator,
};
use tenflowers_core::{DType, Shape};

#[test]
fn finiteness_passes_for_normal_add_gradient() {
    tenflowers_autograd::init();
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
fn finiteness_detects_real_non_finite_gradient_for_div_by_zero() {
    tenflowers_autograd::init();
    let validator = GradientValidator::new();
    let test_case = GradientTestCase::new("div", DType::Float64, vec![Shape::from_slice(&[3])]);
    let result = validator.validate_test_case(test_case);
    let prop = result
        .property_results
        .get(&GradientProperty::Finiteness)
        .expect("Finiteness should have been checked");
    assert!(
        !prop.passed,
        "expected Finiteness to fail for a genuine div-by-zero gradient: {}",
        prop.details
    );
}

#[test]
fn shape_consistency_end_to_end() {
    tenflowers_autograd::init();
    let validator = GradientValidator::new();
    let ok_case = GradientTestCase::new(
        "matmul",
        DType::Float64,
        vec![Shape::from_slice(&[2, 3]), Shape::from_slice(&[3, 4])],
    );
    let ok_result = validator.validate_test_case(ok_case);
    let ok_prop = ok_result
        .property_results
        .get(&GradientProperty::ShapeConsistency)
        .expect("ShapeConsistency should have been checked");
    assert!(ok_prop.passed, "expected pass: {}", ok_prop.details);

    let bad_case = GradientTestCase::new(
        "matmul",
        DType::Float64,
        vec![Shape::from_slice(&[2, 3]), Shape::from_slice(&[4, 5])],
    );
    let bad_result = validator.validate_test_case(bad_case);
    let bad_prop = bad_result
        .property_results
        .get(&GradientProperty::ShapeConsistency)
        .expect("ShapeConsistency should have been checked");
    assert!(
        !bad_prop.passed,
        "expected failure for incompatible matmul shapes"
    );
}

#[test]
fn zero_for_constants_passes_with_real_executor() {
    tenflowers_autograd::init();
    let validator = GradientValidator::new();
    let test_case = GradientTestCase::new("mul", DType::Float64, vec![Shape::from_slice(&[2, 2])])
        .with_property(GradientProperty::ZeroForConstants);
    let result = validator.validate_test_case(test_case);
    let prop = result
        .property_results
        .get(&GradientProperty::ZeroForConstants)
        .expect("ZeroForConstants should have been checked");
    assert!(prop.passed, "expected pass: {}", prop.details);
}

#[test]
fn linearity_passes_with_real_executor() {
    tenflowers_autograd::init();
    let validator = GradientValidator::new();
    let test_case = GradientTestCase::new("add", DType::Float64, vec![Shape::from_slice(&[2, 3])])
        .with_property(GradientProperty::Linearity);
    let result = validator.validate_test_case(test_case);
    let prop = result
        .property_results
        .get(&GradientProperty::Linearity)
        .expect("Linearity should have been checked");
    assert!(prop.passed, "expected pass: {}", prop.details);
}

#[test]
fn chain_rule_passes_with_real_executor() {
    tenflowers_autograd::init();
    let validator = GradientValidator::new();
    let test_case =
        GradientTestCase::new("sigmoid", DType::Float64, vec![Shape::from_slice(&[2, 3])])
            .with_property(GradientProperty::ChainRule);
    let result = validator.validate_test_case(test_case);
    let prop = result
        .property_results
        .get(&GradientProperty::ChainRule)
        .expect("ChainRule should have been checked");
    assert!(prop.passed, "expected pass: {}", prop.details);
}
