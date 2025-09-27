# TenfloweRS Autograd

Automatic differentiation engine for TenfloweRS, providing both tape-based (eager) and graph-based (static) automatic differentiation capabilities.

> Alpha Notice (0.1.0-alpha.1 · 2025-09-27)
> Reverse-mode eager tape core is functional; forward-mode & higher-order support are partial. Gradient coverage and performance instrumentation will expand rapidly pre-beta.

## Overview

`tenflowers-autograd` implements:
- **Tape-based Autograd**: Dynamic computation graph for eager execution mode
- **Forward-mode AD**: Dual number-based forward automatic differentiation
- **Static Graph Integration**: Seamless integration with SciRS2-autograd for static graphs
- **Higher-order Derivatives**: Support for computing Hessians and beyond
- **Mixed-mode Differentiation**: Combine forward and reverse mode for efficiency

## Features

- **Gradient Tape**: PyTorch-like dynamic autograd with operation recording
- **Tracked Tensors**: Automatic gradient tracking for participating tensors
- **Flexible API**: Both functional and object-oriented interfaces
- **Memory Efficient**: Automatic cleanup of intermediate values
- **GPU Support**: Gradient computations on GPU tensors
- **Custom Gradients**: Define custom backward passes for operations

## Usage

### Basic Gradient Computation

```rust
use tenflowers_autograd::{GradientTape, TensorAutograd};
use tenflowers_core::{Tensor, Device};

// Create a gradient tape context
let tape = GradientTape::new();

// Create tracked tensors
let x = tape.variable(Tensor::from_vec(vec![2.0, 3.0], &[2], Device::Cpu)?);
let w = tape.variable(Tensor::from_vec(vec![1.0, 0.5], &[2], Device::Cpu)?);

// Perform computations (automatically tracked)
let y = x.mul(&w)?;  // y = x * w
let z = y.sum()?;    // z = sum(y)

// Compute gradients
let grads = tape.gradient(&z, &[&x, &w])?;
// grads[0] = dz/dx = w = [1.0, 0.5]
// grads[1] = dz/dw = x = [2.0, 3.0]
```

### Forward Mode Automatic Differentiation

```rust
use tenflowers_autograd::{ForwardADContext, DualTensor};

// Create forward AD context
let mut ctx = ForwardADContext::new();

// Create dual tensors (value + derivative)
let x = DualTensor::new(
    Tensor::scalar(2.0, Device::Cpu)?,
    Tensor::scalar(1.0, Device::Cpu)?  // dx/dx = 1
);

// Compute function and derivative simultaneously
let y = ctx.sin(&x)?;     // y = sin(x), dy/dx = cos(x)
let z = ctx.mul(&y, &x)?;  // z = y * x, dz/dx = ...

println!("f(x) = {}", z.value());
println!("f'(x) = {}", z.tangent());
```

### Integration with Static Graphs

```rust
use tenflowers_autograd::TensorAutograd;
use scirs2_autograd::{Graph, Variable};

// Build static computation graph
let mut graph = Graph::new();
let x = graph.placeholder("x", &[None, 784]);
let w = graph.variable("w", Tensor::randn(&[784, 10], Device::Cpu)?);

// Define forward pass
let logits = graph.matmul(&x, &w)?;
let loss = graph.softmax_cross_entropy(&logits, &labels)?;

// Compute gradients using integrated autograd
let grads = graph.gradients(&loss, &[&w])?;

// Use gradients for optimization
optimizer.apply_gradients(&[(w, grads[0])])?;
```

### Higher-order Derivatives

```rust
use tenflowers_autograd::{GradientTape, TensorAutograd};

// Enable higher-order derivatives
let tape = GradientTape::new().persistent();

let x = tape.variable(Tensor::scalar(2.0, Device::Cpu)?);

// f(x) = x^3
let y = x.pow(3)?;

// First derivative: f'(x) = 3x^2
let grad = tape.gradient(&y, &[&x])?[0];

// Second derivative: f''(x) = 6x
let grad2 = tape.gradient(&grad, &[&x])?[0];
```

### Custom Gradient Functions

```rust
use tenflowers_autograd::{CustomOp, GradientTape};

// Define custom operation with gradient
struct ClipGradient;

impl CustomOp for ClipGradient {
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>> {
        // Forward pass: identity
        Ok(inputs[0].clone())
    }
    
    fn backward(&self, grad_output: &Tensor<f32>, inputs: &[&Tensor<f32>]) -> Result<Vec<Tensor<f32>>> {
        // Backward pass: clip gradients to [-1, 1]
        let clipped = grad_output.clamp(-1.0, 1.0)?;
        Ok(vec![clipped])
    }
}

// Use in computation
let tape = GradientTape::new();
let x = tape.variable(tensor);
let y = tape.custom_op(&ClipGradient, &[&x])?;
```

## Architecture

### Core Components

- **GradientTape**: Records operations and manages backward pass
- **TrackedTensor**: Wrapper that enables gradient tracking
- **TapeNode**: Computation graph nodes with operation metadata
- **Operation**: Enumeration of differentiable operations
- **ForwardADContext**: Manages forward-mode differentiation

### Design Principles

1. **Zero-cost Abstractions**: Minimal overhead when gradients aren't needed
2. **Type Safety**: Compile-time guarantees for gradient computations
3. **Lazy Evaluation**: Gradients computed only when requested
4. **Memory Management**: Automatic cleanup of intermediate values

### Integration Points

- **SciRS2-Autograd**: For static graph construction and optimization
- **TenfloweRS-Core**: All tensor operations are differentiable
- **TenfloweRS-Neural**: Automatic gradient computation for layers

## Performance Considerations

- Tape recording has minimal overhead (~5% for most operations)
- Forward-mode AD is efficient for functions with few inputs
- Reverse-mode AD (tape) is efficient for functions with few outputs
- Gradient checkpointing available for memory-constrained scenarios
- Custom CUDA kernels for fused gradient operations

## Supported Operations

Currently supported differentiable operations:
- Arithmetic: `add`, `sub`, `mul`, `div`, `pow`, `neg`
- Matrix: `matmul`, `transpose`, `reshape`
- Reductions: `sum`, `mean`, `max` (with indices)
- Activations: `relu`, `sigmoid`, `tanh`, `softmax`
- Neural: `conv2d`, `max_pool2d`, `batch_norm`
- More operations continuously being added

## Future Enhancements

See TODO.md for detailed roadmap. Key areas:
- Automatic mixed precision training
- Distributed gradient aggregation
- JIT compilation of gradient kernels
- Gradient compression techniques
- Advanced optimization algorithms

### Current Alpha Limitations
- Incomplete gradient definitions for some advanced tensor manipulation & sparse ops
- Higher-order gradients not guaranteed for all activation combinations
- Mixed precision path experimental; no stability guarantees yet
- Graph-mode differentiation integration pending optimizer passes

### Short-Term Priorities
1. Coverage audit & gap tests for missing backward ops
2. Gradient checkpointing ergonomics API
3. Memory profiler diff reports between passes
4. Forward+reverse hybrid scheduling heuristics
5. Deterministic mode (seeded) for reproducibility

## Contributing

We welcome contributions! Priority areas:
- Implementing gradients for more operations
- Optimizing existing gradient computations
- Adding gradient correctness tests
- Improving documentation and examples

## License

Dual-licensed under MIT OR Apache-2.0