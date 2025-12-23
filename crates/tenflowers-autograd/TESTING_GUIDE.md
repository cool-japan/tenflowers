# TenfloweRS Autograd Testing Guide

**Version**: 0.1.0-alpha.2
**Last Updated**: December 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Testing Philosophy](#testing-philosophy)
3. [Unit Testing](#unit-testing)
4. [Integration Testing](#integration-testing)
5. [Property-Based Testing](#property-based-testing)
6. [Numerical Validation](#numerical-validation)
7. [Performance Testing](#performance-testing)
8. [Regression Testing](#regression-testing)
9. [Test Organization](#test-organization)
10. [Best Practices](#best-practices)

---

## Overview

Comprehensive testing is critical for automatic differentiation systems. Incorrect gradients can lead to:
- Silent training failures
- Poor model convergence
- Incorrect optimization results
- Wasted compute resources

This guide provides systematic approaches to ensure gradient correctness.

### Test Coverage Goals

- **Unit tests**: >95% code coverage
- **Integration tests**: All major workflows
- **Numerical validation**: All operations
- **Property tests**: Critical operations
- **Performance tests**: All optimization paths

---

## Testing Philosophy

### The Gradient Testing Pyramid

```
         /\
        /  \    Property-Based Tests (10%)
       /____\   Comprehensive random inputs
      /      \
     /        \  Numerical Validation (20%)
    /__________\ Finite difference checks
   /            \
  /              \ Integration Tests (30%)
 /________________\ End-to-end workflows
/                  \
/____________________\ Unit Tests (40%)
      Basic operations
```

### Key Principles

1. **Test correctness first, performance second**
2. **Use numerical validation for all operations**
3. **Test edge cases thoroughly**
4. **Maintain fast test suites**
5. **Make tests reproducible**

---

## Unit Testing

Unit tests verify individual gradient operations in isolation.

### Basic Gradient Test Template

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;
    use tenflowers_autograd::GradientTape;
    use tenflowers_core::Tensor;

    #[test]
    fn test_operation_gradient() {
        // Setup
        let tape = GradientTape::new();
        let x = tape.watch(Tensor::from_array(
            array![1.0f32, 2.0, 3.0].into_dyn()
        ));

        // Forward pass
        let y = x.custom_operation()?;

        // Backward pass
        let grads = tape.gradient(&[y], &[x])?;
        let grad = grads[0].as_ref().unwrap();

        // Assertions
        assert_eq!(grad.shape().dims(), &[3]);
        assert_approx_eq(grad.as_slice().unwrap(), &[expected1, expected2, expected3]);
    }
}

fn assert_approx_eq(actual: &[f32], expected: &[f32]) {
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-5, "Expected {}, got {}", e, a);
    }
}
```

### Testing Common Operations

#### 1. Arithmetic Operations

```rust
#[test]
fn test_add_gradient() {
    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![1.0f32, 2.0].into_dyn()));
    let y = tape.watch(Tensor::from_array(array![3.0f32, 4.0].into_dyn()));

    // z = x + y
    let z = x.add(&y)?;

    let grads = tape.gradient(&[z], &[x, y])?;

    // ∂z/∂x = 1, ∂z/∂y = 1
    assert_eq!(grads[0].as_ref().unwrap().as_slice().unwrap(), &[1.0, 1.0]);
    assert_eq!(grads[1].as_ref().unwrap().as_slice().unwrap(), &[1.0, 1.0]);
}

#[test]
fn test_mul_gradient() {
    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![2.0f32, 3.0].into_dyn()));
    let y = tape.watch(Tensor::from_array(array![4.0f32, 5.0].into_dyn()));

    // z = x * y
    let z = x.mul(&y)?;

    let grads = tape.gradient(&[z], &[x, y])?;

    // ∂z/∂x = y, ∂z/∂y = x
    assert_eq!(grads[0].as_ref().unwrap().as_slice().unwrap(), &[4.0, 5.0]);
    assert_eq!(grads[1].as_ref().unwrap().as_slice().unwrap(), &[2.0, 3.0]);
}
```

#### 2. Matrix Operations

```rust
#[test]
fn test_matmul_gradient() {
    let tape = GradientTape::new();

    // A: 2x3, B: 3x2 → C: 2x2
    let a = tape.watch(Tensor::from_array(
        array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn()
    ));
    let b = tape.watch(Tensor::from_array(
        array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn()
    ));

    let c = a.matmul(&b)?;

    let grads = tape.gradient(&[c], &[a, b])?;

    // ∂C/∂A = dC/dC @ B^T
    // ∂C/∂B = A^T @ dC/dC
    assert_eq!(grads[0].as_ref().unwrap().shape().dims(), &[2, 3]);
    assert_eq!(grads[1].as_ref().unwrap().shape().dims(), &[3, 2]);
}
```

#### 3. Activation Functions

```rust
#[test]
fn test_relu_gradient() {
    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(
        array![-1.0f32, 0.0, 1.0, 2.0].into_dyn()
    ));

    let y = x.relu()?;

    let grads = tape.gradient(&[y], &[x])?;
    let grad = grads[0].as_ref().unwrap();

    // ReLU gradient: 0 for x<0, 1 for x>0
    assert_eq!(grad.as_slice().unwrap(), &[0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn test_sigmoid_gradient() {
    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![0.0f32].into_dyn()));

    let y = x.sigmoid()?;

    let grads = tape.gradient(&[y], &[x])?;
    let grad = grads[0].as_ref().unwrap().as_slice().unwrap()[0];

    // σ'(0) = σ(0) * (1 - σ(0)) = 0.5 * 0.5 = 0.25
    assert!((grad - 0.25).abs() < 1e-5);
}
```

### Testing Edge Cases

```rust
#[test]
fn test_zero_gradient() {
    let tape = GradientTape::new();
    let x = tape.watch(Tensor::zeros(&[3]));

    let y = x.mul(&x)?;  // y = x^2

    let grads = tape.gradient(&[y], &[x])?;

    // Gradient at x=0 should be 0
    assert_eq!(grads[0].as_ref().unwrap().as_slice().unwrap(), &[0.0, 0.0, 0.0]);
}

#[test]
fn test_large_values() {
    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![1e10f32].into_dyn()));

    let y = x.mul(&x)?;

    let grads = tape.gradient(&[y], &[x])?;

    // Should not overflow or produce NaN
    assert!(grads[0].as_ref().unwrap().as_slice().unwrap()[0].is_finite());
}

#[test]
fn test_broadcasting() {
    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![1.0f32, 2.0].into_dyn()));  // [2]
    let y = tape.watch(Tensor::from_array(array![[1.0f32], [2.0]].into_dyn()));  // [2, 1]

    let z = x.add(&y)?;  // Broadcasting

    let grads = tape.gradient(&[z], &[x, y])?;

    // Check shapes after broadcasting
    assert_eq!(grads[0].as_ref().unwrap().shape().dims(), &[2]);
    assert_eq!(grads[1].as_ref().unwrap().shape().dims(), &[2, 1]);
}
```

---

## Integration Testing

Integration tests verify complete workflows.

### Training Loop Test

```rust
#[test]
fn test_complete_training_loop() {
    // Setup
    let mut params = Tensor::ones(&[10]);
    let target = Tensor::zeros(&[10]);
    let learning_rate = 0.01;

    let initial_loss = compute_loss(&params, &target);

    // Train for a few iterations
    for _ in 0..100 {
        let tape = GradientTape::new();
        let params_tracked = tape.watch(params.clone());

        // Forward pass
        let loss = mse_loss(&params_tracked, &target)?;

        // Backward pass
        let grads = tape.gradient(&[loss], &[params_tracked])?;
        let grad = grads[0].as_ref().unwrap();

        // Update parameters
        params = params.sub(&grad.mul(&Tensor::scalar(learning_rate))?)?;
    }

    let final_loss = compute_loss(&params, &target);

    // Loss should decrease
    assert!(final_loss < initial_loss * 0.5);
}
```

### Multi-Layer Network Test

```rust
#[test]
fn test_multilayer_gradients() {
    let tape = GradientTape::new();

    // Input
    let x = tape.watch(Tensor::ones(&[2, 3]));

    // Layer 1
    let w1 = tape.watch(Tensor::randn(&[3, 4]));
    let h1 = x.matmul(&w1)?.relu()?;

    // Layer 2
    let w2 = tape.watch(Tensor::randn(&[4, 2]));
    let h2 = h1.matmul(&w2)?.relu()?;

    // Output
    let w3 = tape.watch(Tensor::randn(&[2, 1]));
    let out = h2.matmul(&w3)?;

    // Compute gradients for all parameters
    let grads = tape.gradient(&[out], &[x, w1, w2, w3])?;

    // All gradients should be computed
    assert!(grads.iter().all(|g| g.is_some()));

    // All gradients should have correct shapes
    assert_eq!(grads[0].as_ref().unwrap().shape().dims(), &[2, 3]);
    assert_eq!(grads[1].as_ref().unwrap().shape().dims(), &[3, 4]);
    assert_eq!(grads[2].as_ref().unwrap().shape().dims(), &[4, 2]);
    assert_eq!(grads[3].as_ref().unwrap().shape().dims(), &[2, 1]);
}
```

---

## Property-Based Testing

Property tests verify that gradients satisfy mathematical properties.

### Linearity Property

```rust
#[test]
fn test_gradient_linearity() {
    // Property: ∇(αf + βg) = α∇f + β∇g
    let alpha = 2.0f32;
    let beta = 3.0f32;

    let tape1 = GradientTape::new();
    let x1 = tape1.watch(Tensor::ones(&[5]));
    let f = x1.mul(&x1)?;
    let grad_f = tape1.gradient(&[f], &[x1])?[0].clone().unwrap();

    let tape2 = GradientTape::new();
    let x2 = tape2.watch(Tensor::ones(&[5]));
    let g = x2.pow(&Tensor::scalar(3.0))?;
    let grad_g = tape2.gradient(&[g], &[x2])?[0].clone().unwrap();

    let tape3 = GradientTape::new();
    let x3 = tape3.watch(Tensor::ones(&[5]));
    let combined = x3.mul(&x3)?.mul(&Tensor::scalar(alpha))?
        .add(&x3.pow(&Tensor::scalar(3.0))?.mul(&Tensor::scalar(beta))?)?;
    let grad_combined = tape3.gradient(&[combined], &[x3])?[0].clone().unwrap();

    // Check: ∇(αf + βg) = α∇f + β∇g
    let expected = grad_f.mul(&Tensor::scalar(alpha))?
        .add(&grad_g.mul(&Tensor::scalar(beta))?)?;

    assert_tensors_approx_eq(&grad_combined, &expected, 1e-4);
}
```

### Chain Rule Property

```rust
#[test]
fn test_chain_rule() {
    // Property: ∂(g∘f)/∂x = (∂g/∂f) * (∂f/∂x)
    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![2.0f32].into_dyn()));

    // f(x) = x^2
    let f = x.mul(&x)?;

    // g(f) = sin(f) (approximated)
    let g = f.mul(&Tensor::scalar(0.5))?;  // Simplified

    let grad = tape.gradient(&[g], &[x])?[0].clone().unwrap();

    // Verify chain rule manually
    // ∂g/∂x = (∂g/∂f) * (∂f/∂x) = 0.5 * 2x = x
    let expected = 2.0f32;  // x = 2
    assert!((grad.as_slice().unwrap()[0] - expected).abs() < 1e-5);
}
```

---

## Numerical Validation

Always validate gradients using numerical differentiation.

### Basic Numerical Validation

```rust
use tenflowers_autograd::numerical_checker::NumericalChecker;

#[test]
fn test_numerical_validation() {
    let mut checker = NumericalChecker::default();

    let x = Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn());

    // Function to test
    let f = |tensor: &Tensor<f32>| -> Result<Tensor<f32>> {
        tensor.mul(tensor)  // f(x) = x^2
    };

    // Compute numerical gradient
    let numerical_grad = checker.compute_numerical_gradient(&x, f, 1e-6)?;

    // Compute analytical gradient
    let tape = GradientTape::new();
    let x_tracked = tape.watch(x.clone());
    let y = x_tracked.mul(&x_tracked)?;
    let analytical_grad = tape.gradient(&[y], &[x_tracked])?[0].clone().unwrap();

    // Compare
    let result = checker.compare_gradients(&analytical_grad, &numerical_grad)?;

    assert!(result.is_valid, "Gradient check failed: max error = {}", result.max_error);
    assert!(result.max_error < 1e-4);
}
```

### Comprehensive Numerical Test

```rust
#[test]
fn test_all_operations_numerically() {
    let operations = vec![
        ("add", |x: &TrackedTensor<f32>, y: &TrackedTensor<f32>| x.add(y)),
        ("mul", |x: &TrackedTensor<f32>, y: &TrackedTensor<f32>| x.mul(y)),
        ("sub", |x: &TrackedTensor<f32>, y: &TrackedTensor<f32>| x.sub(y)),
        ("div", |x: &TrackedTensor<f32>, y: &TrackedTensor<f32>| x.div(y)),
    ];

    for (name, op) in operations {
        println!("Testing {}...", name);

        let mut checker = NumericalChecker::default();

        // Test at multiple random points
        for _ in 0..10 {
            let x = Tensor::randn(&[3]);
            let y = Tensor::randn(&[3]);

            let f = |x: &Tensor<f32>| -> Result<Tensor<f32>> {
                let tape = GradientTape::new();
                let x_t = tape.watch(x.clone());
                let y_t = tape.watch(y.clone());
                let result = op(&x_t, &y_t)?;
                Ok(result.tensor)
            };

            let numerical = checker.compute_numerical_gradient(&x, f, 1e-6)?;

            let tape = GradientTape::new();
            let x_t = tape.watch(x.clone());
            let y_t = tape.watch(y.clone());
            let result = op(&x_t, &y_t)?;
            let analytical = tape.gradient(&[result], &[x_t])?[0].clone().unwrap();

            let comparison = checker.compare_gradients(&analytical, &numerical)?;

            assert!(comparison.is_valid, "{} gradient check failed", name);
        }

        println!("✓ {} passed", name);
    }
}
```

---

## Performance Testing

### Benchmark Template

```rust
use tenflowers_autograd::{PerformanceBenchmark, BenchmarkConfig};

#[test]
fn benchmark_backward_pass() {
    let config = BenchmarkConfig {
        iterations: 100,
        warmup_iterations: 10,
        ..Default::default()
    };

    let mut benchmark = PerformanceBenchmark::new(config);

    benchmark.start_benchmark("backward_pass")?;

    for _ in 0..config.iterations {
        let tape = GradientTape::new();
        let x = tape.watch(Tensor::randn(&[100, 100]));
        let y = x.matmul(&x)?;
        let grads = tape.gradient(&[y], &[x])?;
    }

    let result = benchmark.end_benchmark("backward_pass")?;

    println!("Backward pass performance:");
    println!("  Mean: {:.2}ms", result.mean_time_ms);
    println!("  Std: {:.2}ms", result.std_time_ms);

    // Regression check
    assert!(result.mean_time_ms < 100.0, "Performance regression detected");
}
```

---

## Regression Testing

### Gradient Value Regression

```rust
#[test]
fn test_gradient_regression() {
    // Known good gradients from previous version
    let expected_gradients = load_reference_gradients("test_case_1.json");

    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn()));
    let y = x.mul(&x)?;

    let grads = tape.gradient(&[y], &[x])?;
    let grad = grads[0].as_ref().unwrap();

    // Compare against reference
    assert_tensors_exact_eq(grad, &expected_gradients);
}
```

---

## Test Organization

### Directory Structure

```
tests/
├── unit/
│   ├── arithmetic_ops.rs
│   ├── matrix_ops.rs
│   ├── activation_functions.rs
│   └── reduction_ops.rs
├── integration/
│   ├── training_loop.rs
│   ├── multilayer_networks.rs
│   └── optimization.rs
├── numerical/
│   ├── gradient_validation.rs
│   └── second_order.rs
├── performance/
│   ├── backward_pass_bench.rs
│   └── memory_usage.rs
└── regression/
    ├── gradient_values.rs
    └── performance_baselines.rs
```

---

## Best Practices

### DO ✅

1. **Test every gradient operation**
2. **Use numerical validation**
3. **Test edge cases** (zeros, infinities, NaN)
4. **Make tests deterministic** (use fixed seeds)
5. **Test at multiple scales** (small and large values)
6. **Verify gradient shapes**
7. **Check for memory leaks**
8. **Benchmark critical paths**
9. **Document test failures**
10. **Keep tests fast** (<1s per test)

### DON'T ❌

1. **Don't skip numerical validation**
2. **Don't use loose tolerances** (>1e-3)
3. **Don't ignore warnings**
4. **Don't test only happy paths**
5. **Don't commit failing tests**
6. **Don't make tests order-dependent**
7. **Don't use random values without seeds**
8. **Don't skip performance tests**
9. **Don't test too many things in one test**
10. **Don't ignore flaky tests**

---

## Test Checklist

Before merging new gradient operations:

- [ ] Unit tests for basic functionality
- [ ] Numerical gradient validation
- [ ] Edge case tests (zeros, large values, etc.)
- [ ] Shape mismatch tests
- [ ] Broadcasting tests
- [ ] Integration test in training loop
- [ ] Performance benchmark
- [ ] Memory usage test
- [ ] Documentation updated
- [ ] All tests passing

---

## Debugging Failed Tests

### Gradient Mismatch

```rust
if let Err(e) = checker.compare_gradients(&analytical, &numerical) {
    println!("Analytical: {:?}", analytical.as_slice());
    println!("Numerical: {:?}", numerical.as_slice());
    println!("Difference: {:?}", compute_diff(&analytical, &numerical));
    panic!("Gradient mismatch: {}", e);
}
```

### Memory Leaks

```rust
let mut profiler = GradientMemoryProfiler::new();
profiler.start_profiling();

// Run test
for _ in 0..1000 {
    let tape = GradientTape::new();
    let grads = compute_gradients(&tape)?;
}

// Check for leaks
let leaks = profiler.detect_leaks()?;
assert_eq!(leaks.num_suspicious, 0, "Memory leaks detected");
```

---

**Last Updated**: December 2025
**Contributors**: TenfloweRS Team
**License**: See main repository LICENSE
