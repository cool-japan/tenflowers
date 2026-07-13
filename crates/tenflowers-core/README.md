# TenfloweRS Core

The foundational crate of TenfloweRS, providing core tensor operations, device management, and the computational infrastructure for machine learning in Rust.

> Stable (v0.2.0 -- 2026-07-13) | 1174 tests passing (26 skipped, `--all-features`) | 0 clippy warnings

## Overview

`tenflowers-core` implements:
- Multi-dimensional tensor operations with CPU and GPU support
- Device abstraction for heterogeneous computing (CPU, WGPU, CUDA, Metal, ROCm)
- Efficient memory management and zero-copy operations where possible
- Built on the SciRS2 numerical ecosystem (`scirs2-core`, `scirs2-linalg`)
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
- **Automatic Fallback**: `fallback::{get_fallback_config, set_fallback_config}` for GPU-to-CPU operation recovery, backed by a thread-safe `OnceLock<RwLock<FallbackConfig>>` (replacing a prior `unsafe static mut` whose writes raced unsynchronized against concurrent reads)
- **BLAS Integration**: Optional acceleration via OxiBLAS
- **LAPACK f64 Ops**: `ops::lapack_f64` — real LAPACK-backed `inverse_f64`, `determinant_f64`, `svd_f64`, `solve_f64` via `scirs2-linalg`
- **Graph Optimization**: `session::SessionConfig::enable_graph_optimization` (default `true`) wires constant folding, algebraic simplification, CSE, strength reduction, scheduling, and dead-code elimination into real session execution, with fetch-history-aware output protection across repeated `run()` calls with different fetch lists
- **ONNX Graph Interop**: `onnx_interop::{convert, lowering, proto}` — real protobuf import/export for `tenflowers-core`'s own graph representation (`OnnxModel`/`OnnxGraph`/`OnnxNode`/`OnnxTensor` `to_protobuf`/`from_protobuf`), plus `lowering::{OnnxOpMapping, StandardOpMapping, lower_graph}` mapping parsed ONNX nodes (`Add, Sub, Mul, Div, Relu, Sigmoid, Tanh, MatMul, Reshape, Transpose, Identity, Concat, Softmax, Flatten, Gemm`) to real tenflowers-core ops; `OnnxImporter`/`OnnxExporter` file/byte round-trips are real behind the `onnx` feature (this is distinct from `tenflowers-neural`'s separate `Sequential`-model ONNX subsystem)
- **Gradient Executor Bridge**: `gradient_executor` module — a `GradientExecutor` trait (`register_gradient_executor`/`get_gradient_executor`) letting `gradient_validation_framework` call into a real forward+backward implementation supplied by `tenflowers-autograd` without a circular crate dependency; without a registered executor, validation now reports an honest "unverified" status instead of fabricating `passed: true`

## Usage

### Basic Tensor Operations

```rust
use tenflowers_core::{Tensor, Device};

// Create tensors (from_vec / ones always start on Device::Cpu; use
// `.to_device(...)` below to move onto a GPU when the `gpu` feature is on)
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
let b: Tensor<f32> = Tensor::ones(&[2, 2]);

// Arithmetic operations
let c = &a + &b;  // Element-wise addition
let d = a.matmul(&b)?;  // Matrix multiplication

// Reductions (axes, keepdims)
let sum = c.sum(None, false)?;  // Sum all elements
let mean = c.mean(Some(&[0]), false)?;  // Mean along axis 0
```

### GPU Operations

```rust
#[cfg(feature = "gpu")]
{
    let gpu_device = Device::Gpu(0);
    let a_gpu = a.to_device(gpu_device)?;
    let b_gpu = b.to_device(gpu_device)?;

    // Operations automatically dispatch to GPU kernels
    let c_gpu = a_gpu.matmul(&b_gpu)?;

    // Transfer back to CPU if needed
    let c_cpu = c_gpu.to_device(Device::Cpu)?;
}
```

### Computation Graphs

`Graph` currently exposes a low-level node/edge builder rather than a fluent
expression API — there is no `graph.matmul(...)`/`graph.add(...)` sugar yet,
so graphs are assembled via `add_node`/`add_edge` and executed through a
`Session` built by `create_session`:

```rust
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tenflowers_core::{
    create_session, DType, Device, FeedDict, FetchSpec, Graph, NodeType, Session, Shape, Tensor,
};

// Build a computation graph: output = x + y
let mut graph = Graph::new();
let x_id = graph.add_node(
    "x".to_string(),
    NodeType::Placeholder { dtype: DType::Float32, shape: Shape::new(vec![2]) },
    Device::Cpu,
    HashMap::new(),
)?;
let y_id = graph.add_node(
    "y".to_string(),
    NodeType::Placeholder { dtype: DType::Float32, shape: Shape::new(vec![2]) },
    Device::Cpu,
    HashMap::new(),
)?;
let add_id = graph.add_node(
    "output".to_string(),
    NodeType::Operation("Add".to_string()),
    Device::Cpu,
    HashMap::new(),
)?;
graph.add_edge(x_id, add_id, 0, 0, DType::Float32, Shape::new(vec![2]), false)?;
graph.add_edge(y_id, add_id, 0, 1, DType::Float32, Shape::new(vec![2]), false)?;

// Execute with a session (feed dict and results are always Tensor<f32>)
let mut session = create_session(Arc::new(RwLock::new(graph)), None, None);
let mut feed_dict = FeedDict::new();
feed_dict.insert("x".to_string(), Tensor::from_vec(vec![1.0, 2.0], &[2])?);
feed_dict.insert("y".to_string(), Tensor::from_vec(vec![3.0, 4.0], &[2])?);
let results = session.run(&[FetchSpec::Name("output".to_string())], &feed_dict)?;
```

By default (`SessionConfig::enable_graph_optimization == true`), a session's
`run()` call applies constant folding, algebraic simplification,
common-subexpression elimination, strength reduction, scheduling, and
dead-code elimination before executing the graph; nodes fetched by any prior
`run()` call remain protected from later optimization passes even under a
different fetch list. `NodeType::Variable` nodes (stateful values such as
learned weights) are also supported, initialized either from an
`AttributeValue::Tensor(...)` under the `"initializer"` attribute key passed
to `add_node`, or with zeros by default.

## Architecture

### Core Components

| Item | Kind | Description |
|---|---|---|
| `Tensor<T>` | struct | Core N-dimensional array, generic over element type, wrapping CPU (`ndarray`) or GPU storage |
| `Device` | enum | `Cpu` / `Gpu(usize)` (feature `gpu`) / `Rocm(usize)` (feature `rocm`) placement target |
| `DType` | enum | Runtime dtype tag (`Float32`, `Float64`, `Int32`, ... `Complex64`) used by `Graph`, ONNX interop, and quantization |
| `TensorStorage<T>` | enum | Internal CPU (`ArrayD<T>`) vs. GPU-buffer storage backing a `Tensor` |
| `Graph` | struct | Low-level computation graph (`add_node`/`add_edge`/`NodeType`) |
| `Session` / `DefaultSession` | trait / struct | Graph execution (`run`, `partial_run`); build via `create_session` |
| `DispatchRegistry<T>` | struct | Extensible per-dtype operation dispatch with kernel selection |
| `ShapeInferenceRegistry` | struct | Automatic output-shape computation per operation |
| `fallback::{get_fallback_config, set_fallback_config}` | fn | Thread-safe global GPU-to-CPU fallback configuration |
| `gradient_executor::{GradientExecutor, register_gradient_executor}` | trait / fn | Bridge letting `tenflowers-autograd` supply real backward passes |

### Integration with SciRS2

`tenflowers-core` is built directly on `scirs2-core`'s `ndarray`-based array
type for CPU storage (there is no separate NumRS2 adapter crate/method —
`Tensor` interoperates with `scirs2_core::ndarray` directly):

```rust
use scirs2_core::ndarray::{ArrayD, IxDyn};
use tenflowers_core::Tensor;

// Build a tensor from an existing scirs2-core/ndarray array
let array = ArrayD::from_shape_vec(IxDyn(&[3, 3]), vec![1.0f32; 9])?;
let tensor = Tensor::from_array(array);

// Read back a flat, contiguous view of the underlying data
if let Some(flat) = tensor.as_slice() {
    println!("{flat:?}");
}
```

## Feature Flags

- `std` (default): Standard library support
- `parallel` (default): Parallel CPU operations via Rayon
- `gpu`: Enable GPU support via WGPU
- `cuda`/`nccl`: gate the CUDA/NCCL module surface; Linux/Windows only, and no CUDA SDK is actually linked (`cudarc`/`cuda-sys` are not dependencies), so these operations return honest `NotImplemented` errors rather than real CUDA execution
- `cudnn`: gates the cuDNN module surface; no `libcudnn.so` is linked, so all operations return honest `NotImplemented` errors
- `opencl`: routes to the `wgpu` backend; present for API-surface completeness, no OpenCL SDK involved
- `metal`: Metal backend support (macOS)
- `rocm`: gates the ROCm/HIP module surface; no `libamdhip64.so` is linked, so all operations return honest `NotImplemented` errors
- `blas`/`blas-openblas`/`blas-mkl`/`blas-accelerate`: BLAS acceleration via `ndarray-linalg`
- `blas-oxiblas`: Use OxiBLAS (`oxiblas-ndarray`) for accelerated linear algebra — the Pure-Rust BLAS option
- `simd`: reserved for future explicit SIMD opt-in; runtime-detected CPU SIMD (via `std::arch` intrinsics) is always active in the `simd` module regardless of this flag — enabling it currently only gates optional `dispatch_registry` SIMD kernel paths and benchmark helpers
- `serialize`: Enable serialization support via serde
- `compression`: Enable `oxiarc-archive`-backed compression
- `onnx`: Real protobuf-backed ONNX graph import/export (`onnx_interop::{OnnxImporter, OnnxExporter}`)
- `wasm`: WebAssembly target support; `wasm_optimization::tensor` runtime-probes `WebAssembly.validate` for SIMD support and `typeof SharedArrayBuffer` for shared-memory support on `wasm32` targets (honest `false` off-`wasm32`, not a hardcoded guess)
- `autograd`: reserved for future use; currently gates a single internal test only and does not affect `gradient_executor` or other production functionality (which work regardless of this flag)

## Performance Considerations

- Tensors use reference counting for efficient memory management
- Operations are lazily evaluated when using computation graphs
- GPU operations are asynchronous and batched for efficiency
- Broadcasting follows NumPy semantics for compatibility
- Zero-copy views are used where possible (slicing, transposition)
- Kernel fusion reduces memory bandwidth pressure for eligible op sequences

## Dependencies

Core dependencies:
- `scirs2-core` / `scirs2-linalg`: numerical primitives (`ndarray`-based CPU storage, random, linear algebra), num-trait bounds, and parallel/SIMD utilities, re-exported throughout this crate rather than depending on `ndarray`/`num-traits` directly
- `rayon` (optional, via `parallel` feature): parallel CPU operations
- `wgpu` (optional, via `gpu` feature): GPU compute support
- `bytemuck`, `half`, `num-complex`: POD casting and half-precision/complex numeric types
- `oxifft`: FFT (Pure-Rust, replacing `rustfft`)
- `oxicode`: serialization codec (Pure-Rust, replacing `bincode`)
- `oxiblas-ndarray` (optional, via `blas-oxiblas` feature): Pure-Rust BLAS acceleration
- `oxiarc-archive` (optional, via `compression` feature): compression (Pure-Rust, replacing `zip`/`flate2`/`zstd`)

## License

Licensed under Apache-2.0
