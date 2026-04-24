# TenfloweRS Autograd

Automatic differentiation engine for TenfloweRS, providing both tape-based (eager) and graph-based (static) automatic differentiation capabilities.

> Stable (v0.1.1 -- 2026-04-24) | 455 tests passing | 0 clippy warnings

## Overview

`tenflowers-autograd` implements:
- **Tape-based Autograd**: Dynamic computation graph for eager execution mode
- **Forward-mode AD**: Dual number-based forward automatic differentiation
- **Reverse-mode AD**: Gradient tape with full backward pass support
- **Higher-order Derivatives**: Support for computing Hessians and beyond
- **Gradient Accumulation**: Accumulate gradients across micro-batches
- **Checkpointing**: Memory-efficient gradient checkpointing for large models
- **In-place Operations**: Gradient-aware in-place tensor modifications
- **Jacobian Checks**: Numerical Jacobian verification for gradient correctness
- **Interpretability**: Gradient-based attribution and saliency analysis
- **Second-order Utilities**: Hessian-vector products and Fisher information

## Features

- **Gradient Tape**: PyTorch-like dynamic autograd with operation recording
- **Tracked Tensors**: Automatic gradient tracking for participating tensors
- **Flexible API**: Both functional and object-oriented interfaces
- **Memory Efficient**: Automatic cleanup of intermediate values with checkpointing support
- **GPU Support**: Gradient computations on GPU tensors
- **Custom Gradients**: Define custom backward passes for operations
- **Forward Gradients**: Efficient forward-mode for low-input-dimension functions
- **Gradient Utils**: Clipping, scaling, and diagnostic utilities

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
- **GradientAccumulator**: Accumulates gradients across steps
- **CheckpointManager**: Memory-efficient recomputation strategy

### Design Principles

1. **Zero-cost Abstractions**: Minimal overhead when gradients are not needed
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
- In-place operations reduce memory allocations during backward pass

## Supported Operations

Differentiable operations:
- Arithmetic: `add`, `sub`, `mul`, `div`, `pow`, `neg`
- Matrix: `matmul`, `transpose`, `reshape`
- Reductions: `sum`, `mean`, `max` (with indices)
- Activations: `relu`, `sigmoid`, `tanh`, `softmax`, `gelu`, `mish`
- Neural: `conv2d`, `max_pool2d`, `batch_norm`
- Advanced: `logsumexp`, `layer_norm`, `group_norm`

## Feature Flags

- `default`: Standard reverse-mode autograd
- `gpu`: GPU-accelerated gradient computations
- `parallel`: Parallel gradient accumulation
- `jit`: JIT compilation of gradient kernels

## License

Licensed under Apache-2.0
