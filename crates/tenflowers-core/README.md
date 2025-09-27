# TenfloweRS Core

The foundational crate of TenfloweRS, providing core tensor operations, device management, and the computational infrastructure for machine learning in Rust.

> Alpha Release (0.1.0-alpha.1 · 2025-09-27)
> This crate participates in the first public alpha. Expect API refinements and additional safety/shape checks before 0.1.0 stable. Pin the exact pre-release version if used in downstream experiments.

## Overview

`tenflowers-core` implements:
- Multi-dimensional tensor operations with CPU and GPU support
- Device abstraction for heterogeneous computing
- Efficient memory management and zero-copy operations where possible
- Integration with the NumRS2/SciRS2 ecosystem
- Computation graph construction for static optimization
- Session-based execution model inspired by TensorFlow

## Features

- **Device Management**: Seamless CPU/GPU tensor operations with automatic device placement
- **Data Types**: Support for `f32`, `f64`, `i32`, `i64`, `u8`, and more
- **Operations**: Comprehensive set of tensor operations including:
  - Arithmetic: element-wise and broadcasting operations
  - Linear Algebra: matrix multiplication, decompositions, eigenvalues
  - Neural Network: convolutions, pooling, activations
  - Reductions: sum, mean, max, argmax along axes
  - Manipulation: reshape, transpose, concatenate, slice
- **GPU Acceleration**: WGPU-based compute shaders for cross-platform GPU support
- **BLAS Integration**: Optional acceleration via OpenBLAS/MKL/Accelerate

## Usage

### Basic Tensor Operations

```rust
use tenflowers_core::{Tensor, Device, DType};

// Create tensors
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], Device::Cpu)?;
let b = Tensor::ones(&[2, 2], DType::F32, Device::Cpu)?;

// Arithmetic operations
let c = &a + &b;  // Element-wise addition
let d = a.matmul(&b)?;  // Matrix multiplication

// Reductions
let sum = c.sum(None)?;  // Sum all elements
let mean = c.mean(Some(&[0]))?;  // Mean along axis 0
```

### GPU Operations

```rust
#[cfg(feature = "gpu")]
{
    let gpu_device = Device::Gpu(0);
    let a_gpu = a.to_device(&gpu_device)?;
    let b_gpu = b.to_device(&gpu_device)?;
    
    // Operations automatically dispatch to GPU kernels
    let c_gpu = a_gpu.matmul(&b_gpu)?;
    
    // Transfer back to CPU if needed
    let c_cpu = c_gpu.to_device(&Device::Cpu)?;
}
```

### Computation Graphs

```rust
use tenflowers_core::{Graph, Session};

// Build a computation graph
let mut graph = Graph::new();
let x = graph.placeholder("x", DType::F32, Some(&[None, 784]));
let w = graph.variable("w", Tensor::randn(&[784, 10], DType::F32, Device::Cpu)?);
let b = graph.variable("b", Tensor::zeros(&[10], DType::F32, Device::Cpu)?);

let logits = graph.matmul(&x, &w)?;
let output = graph.add(&logits, &b)?;

// Execute with session
let mut session = Session::new(&graph);
let result = session.run(
    &[output],
    &[("x", input_tensor)],
)?;
```

## Architecture

### Core Components

- **Tensor**: The fundamental data structure, wrapping device-specific storage
- **Device**: Abstraction over CPU and GPU devices with placement strategies
- **TensorStorage**: Internal storage handling CPU (ndarray) and GPU buffers
- **Operations**: Modular operation system with device-specific implementations
- **Graph/Session**: Static graph construction and optimized execution

### Integration with NumRS2/SciRS2

This crate is designed to work seamlessly with the broader Rust scientific computing ecosystem:

```rust
use numrs2::array::Array2;
use tenflowers_core::Tensor;

// Convert from NumRS2 arrays
let array = Array2::from_shape_vec((3, 3), vec![1.0; 9])?;
let tensor = Tensor::from_numrs2(array, Device::Cpu)?;

// Convert to NumRS2 arrays
let array_back: Array2<f32> = tensor.to_numrs2()?;
```

## Feature Flags

- `gpu`: Enable GPU support via WGPU (default: disabled)
- `blas-openblas`: Use OpenBLAS for accelerated linear algebra
- `blas-mkl`: Use Intel MKL for accelerated linear algebra
- `blas-accelerate`: Use Apple Accelerate framework (macOS)
- `f16`: Enable half-precision floating point support
- `serialize`: Enable serialization support via serde

## Performance Considerations

- Tensors use reference counting for efficient memory management
- Operations are lazily evaluated when using computation graphs
- GPU operations are asynchronous and batched for efficiency
- Broadcasting follows NumPy semantics for compatibility
- Zero-copy views are used where possible (slicing, transposition)

### Current Alpha Limitations
- Some reduction / advanced linear algebra ops fall back to scalar paths
- Graph execution optimizer passes not yet enabled
- GPU kernel coverage is partial; several ops dispatch to CPU
- Error messages for shape mismatches in composite ops will be improved in beta

### Near-Term Roadmap (toward 0.1.0-beta)
1. Unified kernel dispatch registry
2. Expanded BLAS feature auto-detection & benchmark gating
3. Shape inference + validation layer consolidation
4. Memory pool diagnostics & leak detection tooling
5. Initial ONNX tensor I/O utilities

## Dependencies

Core dependencies:
- `ndarray`: CPU tensor storage and operations
- `num-traits`: Numeric trait bounds
- `rayon`: Parallel CPU operations
- `wgpu` (optional): GPU compute support
- `ndarray-linalg` (optional): BLAS/LAPACK integration

## Contributing

Contributions are welcome! Priority areas:
- Implementing missing operations (see TODO.md)
- Optimizing existing operations
- Adding more GPU kernels
- Improving error messages and documentation

## License

Dual-licensed under MIT OR Apache-2.0