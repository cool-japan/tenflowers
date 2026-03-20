# Gradient Computation: Mathematical Foundations

**Target Audience**: Intermediate to advanced users
**Prerequisites**: Basic calculus, linear algebra
**Estimated Reading Time**: 45 minutes

---

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Forward-Mode Automatic Differentiation](#forward-mode-automatic-differentiation)
4. [Reverse-Mode Automatic Differentiation](#reverse-mode-automatic-differentiation)
5. [Computational Complexity](#computational-complexity)
6. [Jacobian and Hessian Computation](#jacobian-and-hessian-computation)
7. [Numerical Stability](#numerical-stability)
8. [Advanced Topics](#advanced-topics)

---

## Introduction

Automatic differentiation (AD) is a family of techniques for efficiently and accurately computing derivatives of functions specified by computer programs. Unlike symbolic differentiation (which can lead to expression swell) or numerical differentiation (which suffers from truncation and rounding errors), automatic differentiation computes derivatives to machine precision with computational cost proportional to the cost of computing the function itself.

### Why Automatic Differentiation?

**Symbolic Differentiation**:
```
f(x) = exp(x) * sin(x) * cos(x)
f'(x) = exp(x) * (sin(x) * cos(x) + cos²(x) - sin²(x))
```
- ✅ Exact symbolic expressions
- ❌ Expression swell (exponential growth)
- ❌ Not applicable to control flow

**Numerical Differentiation** (finite differences):
```
f'(x) ≈ (f(x + h) - f(x)) / h
```
- ✅ Easy to implement
- ❌ Truncation error O(h)
- ❌ Cancellation error (subtractive)
- ❌ Requires many function evaluations

**Automatic Differentiation**:
```rust
let tape = GradientTape::new();
let x = tape.watch(x_value);
let y = x.exp()?.mul(&x.sin()?)?.mul(&x.cos()?)?;
let dy_dx = tape.gradient(&[y], &[x])?;
```
- ✅ Machine precision (no approximation error)
- ✅ Efficient (same order as function evaluation)
- ✅ Handles control flow
- ✅ Composable and modular

---

## Mathematical Foundations

### The Chain Rule

The foundation of automatic differentiation is the multivariable chain rule:

```
Given: z = f(g(x))
Then:  dz/dx = (df/dg) × (dg/dx)
```

For multivariate functions with intermediate variables:
```
z = f(w₁, w₂, ..., wₙ)
where each wᵢ = gᵢ(v₁, v₂, ..., vₘ)
```

The chain rule becomes:
```
∂z/∂vⱼ = Σᵢ (∂z/∂wᵢ) × (∂wᵢ/∂vⱼ)
```

### Computational Graph

Every computer program can be represented as a directed acyclic graph (DAG) where:
- **Nodes** represent variables or intermediate values
- **Edges** represent data dependencies
- **Labels** on edges represent partial derivatives

Example: `z = (x + y) × (x - y)`

```
    x ────┐        ┌──── z
           ├─ t₁ ──┤
    y ────┘        │
                   │
    x ────┐        │
           ├─ t₂ ──┘
    y ────┘

where: t₁ = x + y
       t₂ = x - y
       z  = t₁ × t₂
```

Edge labels (partial derivatives):
```
∂t₁/∂x = 1,  ∂t₁/∂y = 1
∂t₂/∂x = 1,  ∂t₂/∂y = -1
∂z/∂t₁ = t₂, ∂z/∂t₂ = t₁
```

### Jacobian Matrix

For a function f: ℝⁿ → ℝᵐ, the Jacobian matrix J ∈ ℝᵐˣⁿ is:

```
J = [∂fᵢ/∂xⱼ] = ⎡ ∂f₁/∂x₁  ∂f₁/∂x₂  ...  ∂f₁/∂xₙ ⎤
                ⎢ ∂f₂/∂x₁  ∂f₂/∂x₂  ...  ∂f₂/∂xₙ ⎥
                ⎢    ⋮        ⋮      ⋱      ⋮    ⎥
                ⎣ ∂fₘ/∂x₁  ∂fₘ/∂x₂  ...  ∂fₘ/∂xₙ ⎦
```

**Interpretation**:
- Each row i: gradient of output fᵢ with respect to all inputs
- Each column j: sensitivities of all outputs with respect to input xⱼ

---

## Forward-Mode Automatic Differentiation

### Dual Numbers

Forward-mode AD uses dual numbers to carry both function values and derivatives:

```
x̂ = x + ẋε  where ε² = 0
```

**Arithmetic rules**:
```
(a + bε) + (c + dε) = (a + c) + (b + d)ε
(a + bε) × (c + dε) = ac + (ad + bc)ε
```

**Elementary functions**:
```
exp(a + bε) = exp(a) + b·exp(a)ε
sin(a + bε) = sin(a) + b·cos(a)ε
```

### Forward Accumulation

For function composition z = f(g(x)):

1. **Initialize**: Set tangent seed ẋ (typically 1 for derivatives, 0 otherwise)
2. **Propagate**: Compute value and tangent forward through the graph
3. **Result**: Final tangent is the derivative

**Example**: Compute dz/dx where z = sin(x²)

```rust
// Step 1: Initialize with tangent seed
let x̂ = DualNumber { value: x, tangent: 1.0 };

// Step 2: Forward propagation
let t̂ = x̂ * x̂;  // t = x², dt/dx = 2x
// t̂.value = x², t̂.tangent = 2x

let ẑ = sin(t̂);  // z = sin(t), dz/dt = cos(t)
// ẑ.value = sin(x²), ẑ.tangent = 2x·cos(x²)

// Step 3: Extract derivative
let dz_dx = ẑ.tangent;  // = 2x·cos(x²)
```

### Computational Cost

For f: ℝⁿ → ℝᵐ:
- **Computing the Jacobian column-by-column**: O(n) forward passes
- **Each forward pass**: ~1.5-3× cost of evaluating f
- **Total**: O(n × cost(f)) to compute full Jacobian

**Efficient when**: Few inputs, many outputs (n ≪ m)

---

## Reverse-Mode Automatic Differentiation

### Adjoint Variables

Reverse-mode AD uses adjoint (or backpropagated) variables:

```
Adjoint of v: v̄ = ∂z/∂v
```

where z is the final output (typically a scalar loss).

### Reverse Accumulation (Backpropagation)

For function composition z = f(g(x)):

1. **Forward pass**: Compute and store all intermediate values
2. **Initialize**: Set adjoint seed z̄ = 1
3. **Propagate backward**: Compute adjoints using chain rule
4. **Result**: Adjoint of each input is its gradient

**Example**: Compute dz/dx where z = sin(x²)

```rust
// Forward pass (store intermediates)
let x = 3.0;
let t = x * x;      // t = 9.0
let z = sin(t);     // z = sin(9.0)

// Backward pass
let z̄ = 1.0;        // Initial seed
let t̄ = z̄ * cos(t); // dt̄ = dz/dt = cos(9.0)
let x̄ = t̄ * 2*x;    // dx̄ = dt̄ × dt/dx = cos(9.0) × 6
// Result: dz/dx = 6·cos(9.0)
```

### Implementation in TenfloweRS

```rust
pub struct GradientTape {
    // Stores computational graph
    nodes: Vec<TapeNode>,
    // Maps tensor IDs to their adjoints
    gradients: HashMap<TensorId, Tensor<T>>,
}

impl GradientTape {
    pub fn gradient<T>(
        mut self,
        targets: &[TrackedTensor<T>],
        sources: &[TrackedTensor<T>],
    ) -> Result<Vec<Option<Tensor<T>>>> {
        // Initialize adjoint of output
        for target in targets {
            self.set_gradient(target.id(), Tensor::ones(target.shape()));
        }

        // Traverse graph in reverse topological order
        for node in self.nodes.iter().rev() {
            let grad_output = self.get_gradient(node.output_id)?;

            // Compute gradients for inputs using chain rule
            let grad_inputs = node.operation.backward(
                &grad_output,
                &node.inputs,
                &node.output,
            )?;

            // Accumulate gradients
            for (input_id, grad_input) in node.input_ids.iter().zip(grad_inputs) {
                self.accumulate_gradient(*input_id, grad_input)?;
            }
        }

        // Extract gradients for requested sources
        sources.iter()
            .map(|s| Ok(self.get_gradient(s.id())?))
            .collect()
    }
}
```

### Computational Cost

For f: ℝⁿ → ℝᵐ:
- **Computing the Jacobian row-by-row**: O(m) reverse passes
- **Each reverse pass**: ~1.5-3× cost of evaluating f
- **Total**: O(m × cost(f)) to compute full Jacobian

**Efficient when**: Many inputs, few outputs (n ≫ m)

**Neural networks**: Typically n = millions (parameters), m = 1 (loss) → reverse-mode wins!

---

## Computational Complexity

### Complexity Comparison

| Method | Full Jacobian | Single Gradient | Memory |
|--------|---------------|-----------------|--------|
| Finite Differences | O(n·cost(f)) | O(n·cost(f)) | O(1) |
| Forward-Mode AD | O(n·cost(f)) | O(cost(f)) | O(1) |
| Reverse-Mode AD | O(m·cost(f)) | O(cost(f)) | O(size(graph)) |

where:
- n = input dimension
- m = output dimension
- cost(f) = computational cost of evaluating f

### Optimal Strategy

For computing ∇f where f: ℝⁿ → ℝ:
- **Reverse-mode**: O(cost(f)) - optimal!
- **Forward-mode**: O(n·cost(f)) - n times slower

For computing Jacobian J where f: ℝⁿ → ℝᵐ:
- **If n < m**: Forward-mode with n passes
- **If m < n**: Reverse-mode with m passes
- **If n ≈ m**: Either mode, or hybrid

**Hybrid approach** (best of both):
```rust
// Compute cheapest subset in each mode
let hybrid = HybridScheduler::new();
let strategy = hybrid.determine_strategy(n, m)?;

match strategy {
    DifferentiationMode::Forward => {
        for j in 0..n {
            // Compute column j of Jacobian
        }
    }
    DifferentiationMode::Reverse => {
        for i in 0..m {
            // Compute row i of Jacobian
        }
    }
    DifferentiationMode::Hybrid { forward_dims, reverse_dims } => {
        // Mix both modes optimally
    }
}
```

---

## Jacobian and Hessian Computation

### Jacobian Matrix

For f: ℝⁿ → ℝᵐ, we have three approaches:

**1. Forward-Mode (column-by-column)**:
```rust
let mut jacobian = Array2::zeros((m, n));
for j in 0..n {
    let mut seed = Array1::zeros(n);
    seed[j] = 1.0;  // Unit vector eⱼ

    let result = forward_ad(&f, &x, &seed)?;
    jacobian.column_mut(j).assign(&result);
}
// Cost: O(n × cost(f))
```

**2. Reverse-Mode (row-by-row)**:
```rust
let mut jacobian = Array2::zeros((m, n));
for i in 0..m {
    let mut seed = Array1::zeros(m);
    seed[i] = 1.0;  // Unit vector eᵢ

    let result = reverse_ad(&f, &x, &seed)?;
    jacobian.row_mut(i).assign(&result);
}
// Cost: O(m × cost(f))
```

**3. Hybrid Mode (optimal)**:
```rust
// Choose based on aspect ratio
if n < m {
    // Use forward mode
    jacobian_forward(f, x, n)
} else {
    // Use reverse mode
    jacobian_reverse(f, x, m)
}
// Cost: O(min(n, m) × cost(f))
```

### Hessian Matrix

For f: ℝⁿ → ℝ, the Hessian H ∈ ℝⁿˣⁿ is:

```
H = [∂²f/∂xᵢ∂xⱼ] = ⎡ ∂²f/∂x₁²   ∂²f/∂x₁∂x₂  ...  ∂²f/∂x₁∂xₙ ⎤
                     ⎢ ∂²f/∂x₂∂x₁ ∂²f/∂x₂²    ...  ∂²f/∂x₂∂xₙ ⎥
                     ⎢     ⋮          ⋮       ⋱        ⋮      ⎥
                     ⎣ ∂²f/∂xₙ∂x₁ ∂²f/∂xₙ∂x₂  ...  ∂²f/∂xₙ²   ⎦
```

**Properties**:
- Symmetric (if f is twice continuously differentiable)
- Encodes local curvature
- Eigenvalues indicate optimization landscape

**Computation methods**:

**1. Forward-over-Reverse** (most efficient for scalar f):
```rust
// Outer loop: Reverse-mode for gradient
let tape1 = GradientTape::new();
let x_tracked = tape1.watch(x.clone());
let y = f(&x_tracked)?;
let grad = tape1.gradient(&[y], &[x_tracked])?[0].clone();

// Inner loop: Forward-mode for each Hessian column
let mut hessian = Array2::zeros((n, n));
for j in 0..n {
    let tape2 = GradientTape::new();
    let x_tracked = tape2.watch(x.clone());
    let grad_tracked = tape2.watch(grad.clone());

    // Compute directional derivative of gradient
    let hess_col = tape2.forward_gradient(&x_tracked, &grad_tracked)?;
    hessian.column_mut(j).assign(&hess_col.as_slice()?);
}
// Cost: O(n × cost(f))
```

**2. Reverse-over-Reverse** (good for batching):
```rust
// Compute each row of Hessian using reverse-mode twice
for i in 0..n {
    let hess_row = compute_hessian_row_reverse(f, x, i)?;
    hessian.row_mut(i).assign(&hess_row);
}
// Cost: O(n² × cost(f)) - less efficient
```

**3. Hessian-Vector Product** (most efficient for optimization):
```rust
// Compute H·v without forming H explicitly
pub fn hessian_vector_product<T>(
    tape: &GradientTape,
    loss: &TrackedTensor<T>,
    params: &TrackedTensor<T>,
    vector: &Tensor<T>,
) -> Result<Tensor<T>> {
    // First: Compute gradient g = ∇f
    let grad = tape.gradient(&[loss.clone()], &[params.clone()])?[0].clone();

    // Second: Compute directional derivative (∇g)·v = H·v
    let tape2 = GradientTape::new();
    let grad_tracked = tape2.watch(grad);
    let hvp = tape2.directional_derivative(&grad_tracked, params, vector)?;

    Ok(hvp)
}
// Cost: O(cost(f)) - optimal!
```

### Laplacian

The Laplacian ∇²f is the trace of the Hessian:

```
∇²f = Tr(H) = Σᵢ ∂²f/∂xᵢ²
```

**Efficient computation** (without full Hessian):
```rust
pub fn compute_laplacian<T>(
    f: impl Fn(&Tensor<T>) -> Result<Tensor<T>>,
    x: &Tensor<T>,
) -> Result<T> {
    let mut laplacian = T::zero();

    // Compute diagonal entries only
    for i in 0..x.size() {
        let tape = GradientTape::new();
        let x_tracked = tape.watch(x.clone());
        let y = f(&x_tracked)?;

        // First derivative
        let grad = tape.gradient(&[y], &[x_tracked])?[0].clone();

        // Second derivative (i-th diagonal entry)
        let tape2 = GradientTape::new();
        let grad_tracked = tape2.watch(grad);
        let hess_ii = tape2.gradient(&[grad_tracked], &[x_tracked])?[0].as_slice()?[i];

        laplacian += hess_ii;
    }

    Ok(laplacian)
}
// Cost: O(n × cost(f))
```

---

## Numerical Stability

### Sources of Instability

**1. Gradient Underflow** (mixed precision):
```
Small gradients in FP16: |g| < 2^(-24) → underflow to zero
Solution: Loss scaling
```

**2. Gradient Overflow** (exploding gradients):
```
Large gradients: |g| > 2^16 → overflow to inf
Solution: Gradient clipping
```

**3. Cancellation Errors**:
```
x = 1e8 + 1e-8
y = 1e8
z = x - y  // Should be 1e-8, but might be 0 due to rounding
```

**4. Vanishing Gradients**:
```
For sigmoid: σ'(x) = σ(x)(1-σ(x))
At x=10: σ'(10) ≈ 0.000045 → gradients vanish in deep networks
```

### Stabilization Techniques

**1. Loss Scaling** (for FP16):
```rust
let scale = 2^16;
let scaled_loss = loss * scale;
let scaled_grads = tape.gradient(&[scaled_loss], &params)?;
let grads = scaled_grads.map(|g| g / scale);
```

**2. Gradient Clipping**:
```rust
// Global norm clipping
let global_norm = compute_global_norm(&grads);
if global_norm > threshold {
    grads = grads.map(|g| g * (threshold / global_norm));
}

// Value clipping
let grads = grads.map(|g| g.clamp(-clip_value, clip_value));
```

**3. Numerical Stability in Operations**:

**Log-Sum-Exp** (numerically stable):
```rust
// Naive: log(Σ exp(xᵢ)) - prone to overflow
let naive = x.iter().map(|xi| xi.exp()).sum::<f64>().ln();

// Stable: log(Σ exp(xᵢ)) = a + log(Σ exp(xᵢ - a)) where a = max(x)
let a = x.max();
let stable = a + x.iter()
    .map(|xi| (xi - a).exp())
    .sum::<f64>()
    .ln();
```

**Softmax** (numerically stable):
```rust
// Naive: exp(xᵢ) / Σⱼ exp(xⱼ)
let naive_softmax = |x: &[f64]| {
    let sum_exp = x.iter().map(|xi| xi.exp()).sum::<f64>();
    x.iter().map(|xi| xi.exp() / sum_exp).collect()
};

// Stable: exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))
let stable_softmax = |x: &[f64]| {
    let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_shifted: Vec<_> = x.iter().map(|xi| (xi - max_x).exp()).collect();
    let sum_exp = exp_shifted.iter().sum::<f64>();
    exp_shifted.iter().map(|e| e / sum_exp).collect()
};
```

---

## Advanced Topics

### Checkpointing (Memory-Time Tradeoff)

**Problem**: Storing all intermediate activations for backward pass requires O(depth) memory.

**Solution**: Checkpointing trades computation for memory:

```
┌─────────┐
│Forward 1│ → Save checkpoint
├─────────┤
│Forward 2│ → Discard activations
├─────────┤
│Forward 3│ → Save checkpoint
├─────────┤
│Backward│  → Recompute from checkpoint 2
└─────────┘
```

**Optimal checkpointing** (Griewank's algorithm):
- For depth d and c checkpoints: O(d/c) memory, O(d log d) time
- TenfloweRS: Configurable strategies

```rust
let config = CheckpointConfig {
    strategy: CheckpointStrategy::Auto,
    memory_budget_mb: 4096,
    ..Default::default()
};

let tape = GradientTape::with_checkpoint_config(config);
```

### Custom Gradient Functions

Sometimes analytical gradients are more efficient than automatic ones:

```rust
struct EfficientOp;

impl CustomGradientFunction<f32> for EfficientOp {
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>> {
        // Complex forward computation
        expensive_computation(inputs[0])
    }

    fn backward(
        &self,
        grad_output: &Tensor<f32>,
        inputs: &[&Tensor<f32>],
        output: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>> {
        // Use output to avoid recomputation
        // Analytical gradient: more efficient
        Ok(vec![efficient_gradient(grad_output, inputs[0], output)?])
    }

    fn name(&self) -> &str {
        "EfficientOp"
    }
}
```

### Implicit Differentiation

For functions defined implicitly by equations:

```
F(x, y) = 0  where y = f(x)
```

**Implicit Function Theorem**:
```
dy/dx = -(∂F/∂x) / (∂F/∂y)
```

**Use case**: Optimization layers, equilibrium models

```rust
pub fn implicit_gradient<T>(
    residual: impl Fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
    x: &Tensor<T>,
    y: &Tensor<T>,  // Solution to F(x, y) = 0
) -> Result<Tensor<T>> {
    // Compute ∂F/∂x and ∂F/∂y
    let tape = GradientTape::new();
    let x_tracked = tape.watch(x.clone());
    let y_tracked = tape.watch(y.clone());

    let f = residual(&x_tracked, &y_tracked)?;

    let grads = tape.gradient(&[f], &[&x_tracked, &y_tracked])?;
    let df_dx = grads[0].clone();
    let df_dy = grads[1].clone();

    // Solve: (∂F/∂y) × (dy/dx) = -(∂F/∂x)
    let dy_dx = solve_linear_system(&df_dy, &df_dx.neg())?;

    Ok(dy_dx)
}
```

### Sparse Gradients

Many operations produce sparse gradients:

```rust
// Embedding lookup: gradient only at selected indices
pub fn embedding_backward(
    grad_output: &Tensor<f32>,  // [batch, embed_dim]
    indices: &Tensor<i64>,      // [batch]
    vocab_size: usize,
) -> Result<Tensor<f32>> {
    // Sparse gradient: [vocab_size, embed_dim]
    // Only rows corresponding to indices are non-zero
    let mut grad_embedding = Tensor::zeros(&[vocab_size, grad_output.shape()[1]]);

    for (i, &idx) in indices.as_slice()?.iter().enumerate() {
        // Accumulate gradient at this index
        grad_embedding.index_add(idx as usize, &grad_output.select(i, 0)?)?;
    }

    Ok(grad_embedding)
}
```

**Sparse storage**: Only store non-zero entries for efficiency.

---

## Summary

### Key Takeaways

1. **Automatic differentiation** computes exact derivatives efficiently
2. **Forward-mode**: Efficient for few inputs, many outputs (n ≪ m)
3. **Reverse-mode**: Efficient for many inputs, few outputs (n ≫ m)
4. **Neural networks**: Always use reverse-mode (millions of parameters → 1 loss)
5. **Hessian computation**: Forward-over-reverse is most efficient
6. **Numerical stability**: Use loss scaling, gradient clipping, stable numerics
7. **Memory optimization**: Checkpointing trades time for space
8. **Custom gradients**: Analytical gradients can be more efficient

### Further Reading

- [Automatic Differentiation in Machine Learning: a Survey (Baydin et al., 2018)](https://arxiv.org/abs/1502.05767)
- [Evaluating Derivatives (Griewank & Walther, 2008)](https://epubs.siam.org/doi/book/10.1137/1.9780898717761)
- [The Simple Essence of Automatic Differentiation (Elliot, 2018)](http://conal.net/papers/essence-of-ad/)

### Related Documentation

- [Automatic Differentiation Theory](./autodiff_theory.md)
- [Computational Graphs](./computation_graphs.md)
- [Complete Autograd Guide](../../AUTOGRAD_GUIDE.md)
- [Performance Optimization Guide](../../PERFORMANCE_GUIDE.md)

---

**Last Updated**: February 6, 2026
**Author**: COOLJAPAN OU (Team KitaSan)
