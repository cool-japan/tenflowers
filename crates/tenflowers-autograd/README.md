# TenfloweRS Autograd

Automatic differentiation engine for TenfloweRS, providing both tape-based (eager) and graph-based (static) automatic differentiation capabilities.

> Stable (v0.1.2 -- 2026-07-08) | 521 tests passing (5 skipped) | 0 clippy warnings

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
use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;

// Create a gradient tape context
let tape = GradientTape::new();

// Watch tensors so operations against them are recorded
let x = tape.watch(Tensor::from_vec(vec![2.0f32, 3.0], &[2])?);
let w = tape.watch(Tensor::from_vec(vec![1.0f32, 0.5], &[2])?);

// Perform computations (automatically tracked)
let y = x.mul(&w)?;               // y = x * w
let z = y.sum(None, false)?;      // z = sum(y) over all axes

// Compute gradients (targets and sources are slices of TrackedTensor)
let grads = tape.gradient(&[z], &[x, w])?;
// grads[0] = Some(dz/dx) = Some(w) = Some([1.0, 0.5])
// grads[1] = Some(dz/dw) = Some(x) = Some([2.0, 3.0])
```

### Forward Mode Automatic Differentiation

```rust
use tenflowers_autograd::{ForwardMode, forward_ops};
use tenflowers_core::Tensor;

// ForwardMode drives dual-tensor id assignment
let mut mode: ForwardMode<f32> = ForwardMode::new();

// `variable` seeds a unit tangent (dx/dx = 1) in a fresh direction;
// `constant` carries no tangent at all
let x = mode.variable(Tensor::from_vec(vec![2.0f32], &[1])?)?;
let two = mode.constant(Tensor::from_vec(vec![2.0f32], &[1])?);

// Compute function and derivative simultaneously: y = x * 2, z = relu(y)
let y = forward_ops::mul(&x, &two)?;
let z = forward_ops::relu(&y)?;

println!("f(x) = {:?}", z.primal);
println!("f'(x) = {:?}", z.tangent(x.id));
```

### Higher-order Derivatives

> Note: reverse-mode `GradientTape` does not yet support persistent-tape
> higher-order autodiff (there is no `.persistent()` method). For second-order
> quantities, use the finite-difference helpers in [`second_order`], which are
> real and tested today:

```rust
use tenflowers_autograd::second_order::hessian_vector_product_fd;
use tenflowers_core::Tensor;

let x = Tensor::from_vec(vec![1.0f32, 2.0], &[2])?;
let v = Tensor::from_vec(vec![1.0f32, 0.0], &[2])?;

// Hessian-vector product Hv for f(x) = x^2 (gradient_fn returns ∇f(x) = 2x)
let hv = hessian_vector_product_fd(
    |x_val| Ok(vec![x_val.mul(&Tensor::from_scalar(2.0f32))?]),
    &x,
    &v,
    Some(1e-5),
)?;
```

[`second_order`]: https://docs.rs/tenflowers-autograd/latest/tenflowers_autograd/second_order/index.html

### Custom Gradient Functions

```rust
use tenflowers_autograd::{CustomGradientFunction, CustomGradientOp, GradientTape};
use tenflowers_core::{Result, Tensor};

// Define a custom operation with gradient
struct ClipGradient;

impl CustomGradientFunction<f32> for ClipGradient {
    fn forward(&self, inputs: &[&Tensor<f32>]) -> Result<Tensor<f32>> {
        // Forward pass: identity
        Ok(inputs[0].clone())
    }

    fn backward(
        &self,
        grad_output: &Tensor<f32>,
        _inputs: &[&Tensor<f32>],
        _output: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>> {
        // Backward pass: clip gradients to [-1, 1]
        Ok(vec![grad_output.clamp(-1.0, 1.0)?])
    }

    fn name(&self) -> &str {
        "ClipGradient"
    }
}

// Wrap and run the forward pass through the tape
let tape = GradientTape::new();
let x = tape.watch(tensor);
let op = CustomGradientOp::new(ClipGradient);
let y = op.apply(&tape, &[&x])?;
// Note: `CustomGradientOp::apply` currently runs the custom forward pass and
// records the output on the tape, but does not yet wire the custom `backward`
// into `tape.gradient()`'s traversal — full backward-pass integration is
// tracked in TODO.md.
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
- Advanced: `layer_norm`, `group_norm`, `instance_norm`, `einsum`

## Feature Flags

- `default`: Standard reverse-mode autograd (no optional features enabled)
- `gpu`: GPU-accelerated gradient computations
- `rocm`: AMD ROCm GPU backend (via `tenflowers-core/rocm`)
- `parallel`: Parallel gradient accumulation
- `jit`: JIT compilation of gradient kernels
- `distributed`: Enables the `distributed_replication` module re-export (cross-datacenter replication scaffolding — see module docs for what is and isn't implemented yet)

## License

Licensed under Apache-2.0
