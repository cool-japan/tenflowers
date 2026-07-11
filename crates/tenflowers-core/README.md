# TenfloweRS Core

The foundational crate of TenfloweRS, providing core tensor operations, device management, and the computational infrastructure for machine learning in Rust.

> Stable (v0.2.0 -- 2026-07-11) | 1174 tests passing (26 skipped, `--all-features`) | 0 clippy warnings

## Overview

`tenflowers-core` implements:
- Multi-dimensional tensor operations with CPU and GPU support
- Device abstraction for heterogeneous computing (CPU, WGPU, CUDA, Metal, ROCm)
- Efficient memory management and zero-copy operations where possible
- Integration with the NumRS2/SciRS2 ecosystem
- Operation registry with shape inference and kernel fusion
- Autocast, sparse tensors, fused ops, and advanced math functions

## Features

- **Device Management**: Seamless CPU/GPU tensor operations with automatic device placement
- **Data Types**: Support for `f32`, `f64`, `i32`, `i64`, `u8`, and more
- **Operations**: Comprehensive set of tensor operations including:
  - Arithmetic: element-wise and broadcasting operations
  - Linear Algebra: matrix multiplication, decompositions, eigenvalues
  - Neural Network: convolutions, pooling, activations
  - Reductions: sum, mean, max, argmax along axes (including real `StdDev`, `L1Norm`, `L2Norm`; segment reductions now support N-D data, not just 1-D)
  - Manipulation: reshape, transpose, concatenate, slice (strided slicing now uses correct row-major linear-index accumulation for every rank/shape combination, fixing a stride-direction bug that previously silently returned wrong elements for non-square, non-1D strided slices)
  - Advanced Math: logsumexp, GELU, Mish, Swish, and more
- **GPU Acceleration**: WGPU-based compute shaders for cross-platform GPU support; `device::get_gpu_adapter_capabilities` exposes a real, unprocessed `wgpu::Adapter` capability snapshot (no vendor-specific guessing)
- **Operation Registry**: Extensible dispatch registry with shape inference
- **Kernel Fusion**: Automatic fusion of eligible operation sequences
- **Autocast**: Automatic dtype promotion for mixed-precision workflows
- **Sparse Tensors**: COO and CSR sparse tensor support
- **Fused Ops**: Pre-fused compound operations for performance
- **BLAS Integration**: Optional acceleration via OxiBLAS
- **LAPACK f64 Ops**: `ops::lapack_f64` — real LAPACK-backed `inverse_f64`, `determinant_f64`, `svd_f64`, `solve_f64` via `scirs2-linalg`
- **Graph Optimization**: `session::SessionConfig::enable_graph_optimization` (default `true`) wires constant folding, algebraic simplification, CSE, strength reduction, scheduling, and dead-code elimination into real session execution, with fetch-history-aware output protection across repeated `run()` calls with different fetch lists
- **ONNX Graph Interop**: `onnx_interop::{convert, lowering, proto}` — real protobuf import/export for `tenflowers-core`'s own graph representation (`OnnxModel`/`OnnxGraph`/`OnnxNode`/`OnnxTensor` `to_protobuf`/`from_protobuf`), plus `lowering::{OnnxOpMapping, StandardOpMapping, lower_graph}` mapping parsed ONNX nodes (`Add, Sub, Mul, Div, Relu, Sigmoid, Tanh, MatMul, Reshape, Transpose, Identity, Concat, Softmax, Flatten, Gemm`) to real tenflowers-core ops; `OnnxImporter`/`OnnxExporter` file/byte round-trips are real behind the `onnx` feature (this is distinct from `tenflowers-neural`'s separate `Sequential`-model ONNX subsystem)
- **Gradient Executor Bridge**: `gradient_executor` module — a `GradientExecutor` trait (`register_gradient_executor`/`get_gradient_executor`) letting `gradient_validation_framework` call into a real forward+backward implementation supplied by `tenflowers-autograd` without a circular crate dependency; without a registered executor, validation now reports an honest "unverified" status instead of fabricating `passed: true`

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

By default (`SessionConfig::enable_graph_optimization == true`), `Session::run` applies constant folding, algebraic simplification, common-subexpression elimination, strength reduction, scheduling, and dead-code elimination before executing the graph; nodes fetched by any prior `run()` call remain protected from later optimization passes even under a different fetch list.

## Architecture

### Core Components

- **Tensor**: The fundamental data structure, wrapping device-specific storage
- **Device**: Abstraction over CPU and GPU devices with placement strategies
- **TensorStorage**: Internal storage handling CPU (ndarray) and GPU buffers
- **Operations**: Modular operation system with device-specific implementations
- **Graph/Session**: Static graph construction and optimized execution
- **DispatchRegistry**: Extensible operation dispatch with kernel selection
- **ShapeInferenceRegistry**: Automatic output shape computation

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

- `std` (default): Standard library support
- `parallel` (default): Parallel CPU operations via Rayon
- `gpu`: Enable GPU support via WGPU
- `cuda`: CUDA backend support
- `metal`: Metal backend support (macOS)
- `rocm`: ROCm backend support (AMD GPUs)
- `blas-oxiblas`: Use OxiBLAS for accelerated linear algebra
- `simd`: SIMD vectorization optimizations
- `serialize`: Enable serialization support via serde
- `onnx`: Real protobuf-backed ONNX graph import/export (`onnx_interop::{OnnxImporter, OnnxExporter}`)
- `wasm`: WebAssembly target support; `wasm_optimization::tensor` runtime-probes `WebAssembly.validate` for SIMD support and `typeof SharedArrayBuffer` for shared-memory support on `wasm32` targets (honest `false` off-`wasm32`, not a hardcoded guess)

## Performance Considerations

- Tensors use reference counting for efficient memory management
- Operations are lazily evaluated when using computation graphs
- GPU operations are asynchronous and batched for efficiency
- Broadcasting follows NumPy semantics for compatibility
- Zero-copy views are used where possible (slicing, transposition)
- Kernel fusion reduces memory bandwidth pressure for eligible op sequences

## Dependencies

Core dependencies:
- `ndarray`: CPU tensor storage and operations
- `num-traits`: Numeric trait bounds
- `rayon`: Parallel CPU operations
- `wgpu` (optional): GPU compute support

## License

Licensed under Apache-2.0
