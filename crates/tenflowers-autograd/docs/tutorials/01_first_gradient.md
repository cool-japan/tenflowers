# Tutorial 1: Your First Gradient

**Level**: Beginner
**Duration**: 15 minutes
**Prerequisites**: Basic Rust knowledge

---

## Learning Objectives

By the end of this tutorial, you will:
- Understand what automatic differentiation is
- Create your first gradient tape
- Compute gradients of simple functions
- Verify gradient correctness

---

## What is Automatic Differentiation?

Automatic differentiation (AD) is a technique for computing derivatives of functions defined by computer programs. It's essential for:
- Training neural networks (backpropagation)
- Optimization problems (finding minima/maxima)
- Sensitivity analysis (how outputs change with inputs)
- Scientific computing (solving differential equations)

**Key insight**: AD computes *exact* derivatives (no approximation), efficiently (similar cost to computing the function itself).

---

## Setup

Add dependencies to your `Cargo.toml`:

```toml
[dependencies]
tenflowers-autograd = "0.1.0-beta.1"
tenflowers-core = "0.1.0-beta.1"
scirs2-core = "0.1.0-rc.1"
```

Create a new file `examples/first_gradient.rs`:

```rust
use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;
use scirs2_core::ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Your code will go here
    Ok(())
}
```

---

## Step 1: Computing Your First Gradient

Let's compute the gradient of a simple function: **f(x) = x²**

### Mathematical Background

For f(x) = x², the derivative is:
```
f'(x) = 2x
```

At x=3:
```
f'(3) = 2 × 3 = 6
```

### Implementation

```rust
use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;
use scirs2_core::ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Create a gradient tape
    let tape = GradientTape::new();

    // Step 2: Create input tensor and "watch" it
    let x = tape.watch(Tensor::from_array(
        array![3.0f32].into_dyn()
    ));

    // Step 3: Compute function y = x²
    let y = x.mul(&x)?;

    // Step 4: Compute gradient dy/dx
    let grads = tape.gradient(&[y], &[x])?;
    let dy_dx = grads[0].as_ref().unwrap();

    println!("x = {}", 3.0);
    println!("y = x² = {}", y.tensor().as_slice()?[0]);
    println!("dy/dx = 2x = {}", dy_dx.as_slice()?[0]);
    println!("Expected: dy/dx = 6.0");

    Ok(())
}
```

### Understanding the Code

1. **`GradientTape::new()`**: Creates a tape that records operations
2. **`tape.watch(tensor)`**: Marks a tensor for gradient computation
3. **`x.mul(&x)`**: Computes x² and records the operation on the tape
4. **`tape.gradient(&[y], &[x])`**: Computes ∂y/∂x using backpropagation

### Run the Program

```bash
cargo run --example first_gradient
```

**Expected output**:
```
x = 3
y = x² = 9
dy/dx = 2x = 6
Expected: dy/dx = 6.0
```

✅ Success! You've computed your first gradient.

---

## Step 2: Multi-Dimensional Gradients

Let's compute gradients for vectors.

### Function

```
f(x) = ||x||² = x₁² + x₂² + x₃²
```

### Gradient

```
∇f = [∂f/∂x₁, ∂f/∂x₂, ∂f/∂x₃] = [2x₁, 2x₂, 2x₃]
```

### Implementation

```rust
fn vector_gradient() -> Result<(), Box<dyn std::error::Error>> {
    let tape = GradientTape::new();

    // Vector input
    let x = tape.watch(Tensor::from_array(
        array![1.0f32, 2.0, 3.0].into_dyn()
    ));

    // Compute f(x) = ||x||² = x₁² + x₂² + x₃²
    let x_squared = x.mul(&x)?;
    let sum = x_squared.sum(Some(&[0]), false)?;

    // Compute gradient
    let grads = tape.gradient(&[sum], &[x])?;
    let gradient = grads[0].as_ref().unwrap();

    println!("x = {:?}", x.tensor().as_slice()?);
    println!("f(x) = ||x||² = {}", sum.tensor().as_slice()?[0]);
    println!("∇f = 2x = {:?}", gradient.as_slice()?);
    println!("Expected: [2.0, 4.0, 6.0]");

    Ok(())
}
```

**Output**:
```
x = [1.0, 2.0, 3.0]
f(x) = ||x||² = 14.0
∇f = 2x = [2.0, 4.0, 6.0]
Expected: [2.0, 4.0, 6.0]
```

---

## Step 3: Chain Rule in Action

Compute gradients of composite functions.

### Function

```
f(x) = sin(x²)
```

### Derivative (Chain Rule)

```
f'(x) = cos(x²) × 2x
```

At x=1:
```
f'(1) = cos(1) × 2 ≈ 1.081
```

### Implementation

```rust
fn chain_rule_example() -> Result<(), Box<dyn std::error::Error>> {
    let tape = GradientTape::new();

    let x = tape.watch(Tensor::from_array(
        array![1.0f32].into_dyn()
    ));

    // Compute f(x) = sin(x²)
    let x_squared = x.mul(&x)?;
    let y = x_squared.sin()?;

    // Compute gradient
    let grads = tape.gradient(&[y], &[x])?;
    let dy_dx = grads[0].as_ref().unwrap();

    println!("x = {}", 1.0);
    println!("f(x) = sin(x²) = {}", y.tensor().as_slice()?[0]);
    println!("f'(x) = cos(x²) × 2x = {}", dy_dx.as_slice()?[0]);
    println!("Expected: cos(1) × 2 ≈ 1.081");

    Ok(())
}
```

---

## Step 4: Multiple Inputs

Compute partial derivatives with respect to multiple variables.

### Function

```
f(x, y) = x² + xy + y²
```

### Partial Derivatives

```
∂f/∂x = 2x + y
∂f/∂y = x + 2y
```

At (x=2, y=3):
```
∂f/∂x = 2(2) + 3 = 7
∂f/∂y = 2 + 2(3) = 8
```

### Implementation

```rust
fn multiple_inputs() -> Result<(), Box<dyn std::error::Error>> {
    let tape = GradientTape::new();

    let x = tape.watch(Tensor::from_array(array![2.0f32].into_dyn()));
    let y = tape.watch(Tensor::from_array(array![3.0f32].into_dyn()));

    // Compute f(x, y) = x² + xy + y²
    let x_squared = x.mul(&x)?;
    let xy = x.mul(&y)?;
    let y_squared = y.mul(&y)?;
    let result = x_squared.add(&xy)?.add(&y_squared)?;

    // Compute partial derivatives
    let grads = tape.gradient(&[result], &[x.clone(), y.clone()])?;
    let df_dx = grads[0].as_ref().unwrap();
    let df_dy = grads[1].as_ref().unwrap();

    println!("f(2, 3) = {}", result.tensor().as_slice()?[0]);
    println!("∂f/∂x = 2x + y = {}", df_dx.as_slice()?[0]);
    println!("∂f/∂y = x + 2y = {}", df_dy.as_slice()?[0]);
    println!("Expected: ∂f/∂x = 7, ∂f/∂y = 8");

    Ok(())
}
```

---

## Step 5: Verifying Gradient Correctness

Always verify your gradients using numerical methods!

### Numerical Gradient (Finite Differences)

```
f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
```

where h is a small number (e.g., 1e-5).

### Implementation

```rust
use tenflowers_autograd::numerical_checker::NumericalChecker;

fn verify_gradient() -> Result<(), Box<dyn std::error::Error>> {
    let mut checker = NumericalChecker::default();

    // Define function
    let f = |x: &Tensor<f32>| -> Result<Tensor<f32>> {
        x.mul(x)  // f(x) = x²
    };

    let x = Tensor::from_array(array![3.0f32].into_dyn());

    // Compute numerical gradient
    let numerical_grad = checker.compute_numerical_gradient(&x, f, 1e-5)?;

    // Compute analytical gradient
    let tape = GradientTape::new();
    let x_tracked = tape.watch(x.clone());
    let y = x_tracked.mul(&x_tracked)?;
    let analytical_grad = tape.gradient(&[y], &[x_tracked])?[0]
        .as_ref().unwrap().clone();

    // Compare
    let result = checker.compare_gradients(&analytical_grad, &numerical_grad)?;

    if result.is_valid {
        println!("✅ Gradient check passed!");
        println!("   Max error: {:.2e}", result.max_error);
    } else {
        println!("❌ Gradient check failed!");
        println!("   Max error: {:.2e}", result.max_error);
    }

    Ok(())
}
```

**Output**:
```
✅ Gradient check passed!
   Max error: 1.23e-07
```

---

## Exercises

Test your understanding with these exercises:

### Exercise 1: Cubic Function

Compute the gradient of f(x) = x³ at x=2.

**Expected**: f'(2) = 3(2²) = 12

<details>
<summary>Solution</summary>

```rust
fn exercise1() -> Result<(), Box<dyn std::error::Error>> {
    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![2.0f32].into_dyn()));

    // f(x) = x³
    let x_squared = x.mul(&x)?;
    let x_cubed = x_squared.mul(&x)?;

    let grads = tape.gradient(&[x_cubed], &[x])?;
    let dy_dx = grads[0].as_ref().unwrap();

    println!("f'(2) = {}", dy_dx.as_slice()?[0]);
    assert!((dy_dx.as_slice()?[0] - 12.0).abs() < 1e-5);

    Ok(())
}
```

</details>

### Exercise 2: Exponential Function

Compute the gradient of f(x) = eˣ at x=0.

**Expected**: f'(0) = e⁰ = 1

<details>
<summary>Solution</summary>

```rust
fn exercise2() -> Result<(), Box<dyn std::error::Error>> {
    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![0.0f32].into_dyn()));

    let y = x.exp()?;

    let grads = tape.gradient(&[y], &[x])?;
    let dy_dx = grads[0].as_ref().unwrap();

    println!("f'(0) = {}", dy_dx.as_slice()?[0]);
    assert!((dy_dx.as_slice()?[0] - 1.0).abs() < 1e-5);

    Ok(())
}
```

</details>

### Exercise 3: Product Rule

Compute the gradient of f(x) = x × sin(x) at x=π/4.

**Expected**: f'(x) = sin(x) + x×cos(x)

<details>
<summary>Solution</summary>

```rust
fn exercise3() -> Result<(), Box<dyn std::error::Error>> {
    use std::f32::consts::PI;

    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![PI / 4.0].into_dyn()));

    let sin_x = x.sin()?;
    let y = x.mul(&sin_x)?;

    let grads = tape.gradient(&[y], &[x])?;
    let dy_dx = grads[0].as_ref().unwrap();

    println!("f'(π/4) = {}", dy_dx.as_slice()?[0]);

    // Expected: sin(π/4) + (π/4)×cos(π/4)
    let expected = (PI / 4.0).sin() + (PI / 4.0) * (PI / 4.0).cos();
    assert!((dy_dx.as_slice()?[0] - expected).abs() < 1e-5);

    Ok(())
}
```

</details>

---

## Common Mistakes

### Mistake 1: Not Watching Tensors

```rust
// ❌ WRONG: x not watched
let x = Tensor::from_array(array![1.0f32].into_dyn());
let y = x.mul(&x)?;
let grads = tape.gradient(&[y], &[x])?;  // ERROR: x not tracked

// ✅ CORRECT: Watch x first
let x = tape.watch(Tensor::from_array(array![1.0f32].into_dyn()));
let y = x.mul(&x)?;
let grads = tape.gradient(&[y], &[x])?;  // OK
```

### Mistake 2: Reusing Tape

```rust
// ❌ WRONG: Tape consumed by gradient()
let tape = GradientTape::new();
let grads1 = tape.gradient(&[y1], &[x])?;
let grads2 = tape.gradient(&[y2], &[x])?;  // ERROR: tape moved

// ✅ CORRECT: Create new tape
let tape1 = GradientTape::new();
let grads1 = tape1.gradient(&[y1], &[x])?;

let tape2 = GradientTape::new();
let grads2 = tape2.gradient(&[y2], &[x])?;
```

### Mistake 3: Wrong Gradient Target

```rust
// ❌ WRONG: Gradients of x with respect to y (backwards!)
let grads = tape.gradient(&[x], &[y])?;

// ✅ CORRECT: Gradients of y with respect to x
let grads = tape.gradient(&[y], &[x])?;
```

---

## Summary

In this tutorial, you learned:
- ✅ How to create a gradient tape
- ✅ How to watch tensors for gradient computation
- ✅ How to compute gradients using `tape.gradient()`
- ✅ How to verify gradients with numerical methods
- ✅ Common mistakes to avoid

### Key Concepts

- **GradientTape**: Records operations for automatic differentiation
- **watch()**: Marks tensors for gradient tracking
- **gradient()**: Computes gradients via backpropagation
- **Numerical validation**: Always verify custom gradients

### Next Steps

Continue to [Tutorial 2: Training a Neural Network](./02_neural_network.md) to learn how to use gradients for training models.

---

## Complete Example

Here's the complete, runnable example:

```rust
use tenflowers_autograd::{GradientTape, numerical_checker::NumericalChecker};
use tenflowers_core::Tensor;
use scirs2_core::ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tutorial 1: Your First Gradient ===\n");

    // Example 1: Simple gradient
    simple_gradient()?;

    // Example 2: Vector gradient
    vector_gradient()?;

    // Example 3: Chain rule
    chain_rule_example()?;

    // Example 4: Multiple inputs
    multiple_inputs()?;

    // Example 5: Verification
    verify_gradient()?;

    println!("\n✅ All examples completed successfully!");

    Ok(())
}

fn simple_gradient() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 1: f(x) = x²");
    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![3.0f32].into_dyn()));
    let y = x.mul(&x)?;
    let grads = tape.gradient(&[y], &[x])?;
    println!("  dy/dx at x=3: {}\n", grads[0].as_ref().unwrap().as_slice()?[0]);
    Ok(())
}

fn vector_gradient() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 2: f(x) = ||x||²");
    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn()));
    let x_squared = x.mul(&x)?;
    let sum = x_squared.sum(Some(&[0]), false)?;
    let grads = tape.gradient(&[sum], &[x])?;
    println!("  ∇f = {:?}\n", grads[0].as_ref().unwrap().as_slice()?);
    Ok(())
}

fn chain_rule_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 3: f(x) = sin(x²)");
    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![1.0f32].into_dyn()));
    let x_squared = x.mul(&x)?;
    let y = x_squared.sin()?;
    let grads = tape.gradient(&[y], &[x])?;
    println!("  f'(1) = {}\n", grads[0].as_ref().unwrap().as_slice()?[0]);
    Ok(())
}

fn multiple_inputs() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 4: f(x,y) = x² + xy + y²");
    let tape = GradientTape::new();
    let x = tape.watch(Tensor::from_array(array![2.0f32].into_dyn()));
    let y = tape.watch(Tensor::from_array(array![3.0f32].into_dyn()));
    let result = x.mul(&x)?.add(&x.mul(&y)?)?.add(&y.mul(&y)?)?;
    let grads = tape.gradient(&[result], &[x.clone(), y.clone()])?;
    println!("  ∂f/∂x = {}", grads[0].as_ref().unwrap().as_slice()?[0]);
    println!("  ∂f/∂y = {}\n", grads[1].as_ref().unwrap().as_slice()?[0]);
    Ok(())
}

fn verify_gradient() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 5: Gradient Verification");
    let mut checker = NumericalChecker::default();
    let f = |x: &Tensor<f32>| -> Result<Tensor<f32>> { x.mul(x) };
    let x = Tensor::from_array(array![3.0f32].into_dyn());
    let numerical_grad = checker.compute_numerical_gradient(&x, f, 1e-5)?;
    let tape = GradientTape::new();
    let x_tracked = tape.watch(x.clone());
    let y = x_tracked.mul(&x_tracked)?;
    let analytical_grad = tape.gradient(&[y], &[x_tracked])?[0].as_ref().unwrap().clone();
    let result = checker.compare_gradients(&analytical_grad, &numerical_grad)?;
    if result.is_valid {
        println!("  ✅ Gradient check passed! (error: {:.2e})\n", result.max_error);
    }
    Ok(())
}
```

Run with:
```bash
cargo run --example first_gradient
```

---

**Next**: [Tutorial 2: Training a Neural Network](./02_neural_network.md)

**Last Updated**: February 6, 2026
**Author**: COOLJAPAN OU (Team Kitasan)
