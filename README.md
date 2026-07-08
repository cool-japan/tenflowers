# TenfloweRS

A pure Rust implementation of TensorFlow, providing a full-featured machine learning framework with Rust's safety and performance.

[![Version](https://img.shields.io/badge/version-0.1.2-blue)](https://github.com/cool-japan/tenflowers)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange)](https://www.rust-lang.org)
[![Tests](https://img.shields.io/badge/tests-14289%2B%20passing-brightgreen)](https://github.com/cool-japan/tenflowers)
[![Security](https://img.shields.io/badge/advisories-3-yellow)](https://github.com/cool-japan/tenflowers)

> **v0.1.2 (2026-07-08)**
>
> TenfloweRS v0.1.2 adds unified error handling, structured logging, and platform introspection
> to the meta-crate; session-based Python profiling and implicit PyTorch-style autograd in the
> FFI crate; advanced HDF5/Parquet readers, a from-scratch pure-Rust Blosc codec, and real
> Symphonia-backed audio decoding in the dataset crate; real ONNX protobuf import/export,
> N-D segment reductions, GPU-einsum CPU fallbacks, real wgpu device-capability queries, and
> graph-optimizer wiring into `Session` in the core crate; plus a wide honesty-hardening sweep
> replacing fabricated results with real computation or honest errors across every crate.
> 14,289+ tests passing across 6 crates, zero clippy warnings, zero rustdoc warnings.

## Overview

TenfloweRS is a native Rust machine learning framework inspired by TensorFlow, designed to bring the power of deep learning to the Rust ecosystem. It leverages Rust's memory safety, zero-cost abstractions, and excellent performance while maintaining compatibility with the broader ML ecosystem through ONNX support.

## Design Principles

TenfloweRS adapts TensorFlow's proven architecture to Rust's strengths:

1. **Memory Safety First**: All operations are memory-safe by design, eliminating segfaults and data races
2. **Zero-Cost Abstractions**: High-level APIs compile down to efficient machine code
3. **Explicit over Implicit**: Clear ownership and error handling following Rust conventions
4. **Modular Architecture**: Organized as a workspace of focused, reusable crates
5. **Cross-Platform**: Native support for Windows, macOS, and Linux with unified GPU abstraction
6. **Pure Rust**: No C/Fortran dependencies in the default build -- the entire stack is 100% Rust

## TensorFlow to TenfloweRS Mapping

| TensorFlow Concept | TenfloweRS Implementation |
|-------------------|---------------------------|
| `tf.Tensor` | `Tensor<T>` with static typing |
| `tf.Operation` | `Op` trait with registered kernels |
| `tf.Graph` | `Graph` struct with ownership semantics |
| `tf.Session` | `Session` trait for graph execution |
| `tf.GradientTape` | `GradientTape` for automatic differentiation |
| `tf.keras.Layer` | `Layer` trait with builder pattern |
| `tf.data.Dataset` | Iterator-based `Dataset` trait |
| `tf.device` | `Device` enum with placement control |

## Key Features

- **Dual Execution Modes**: Both eager execution (PyTorch-style) and static computation graphs (TensorFlow-style)
- **Pure Rust Implementation**: No C/C++ dependencies in the core, ensuring memory safety
- **GPU Support**: Cross-platform GPU acceleration via WGPU (Metal, Vulkan, DirectX)
- **Rust Scientific Stack**: Built on NumRS2 and SciRS2 for numerical computing
- **Python Bindings**: PyO3-based FFI crate with 48 passing tests
- **ONNX Support**: Import and export models for cross-framework compatibility
- **Performance**: SIMD vectorization, optional BLAS integration, and parallel execution
- **150+ Research Domains**: From transformers and diffusion models to quantum ML and protein structure prediction
- **Production Ready**: 14,289+ tests passing, 3 known advisories (all upstream-blocked, none directly exploitable), comprehensive docs

## Project Status

**Current Version: 0.1.2** (Released 2026-07-08)

### v0.1.2 Quality Metrics

- **Tests:** 14,289+ passing, 39 skipped (last verified full-workspace run; 0.1.2 adds real ONNX import/export, N-D segment reductions, GPU-einsum CPU fallbacks, real device-capability queries, graph-optimizer wiring, a pure-Rust Blosc codec, real audio decoding, and implicit PyTorch-style autograd in the FFI crate)
- **Code:** 1,515+ Rust files, ~677K SLoC (~805K total Rust lines)
- **Security:** 3 known advisories, all transitive and tracked (RUSTSEC-2026-0204 `crossbeam-epoch` via `scirs2-core`; RUSTSEC-2024-0384 `instant` via `hdf5`; RUSTSEC-2024-0436 `paste` via `rav1e`/`parquet`/`metal` — none directly exploitable; fixes pending upstream). The prior pyo3 advisories (RUSTSEC-2026-0176/0177) were resolved this release via the pyo3 0.28 → 0.29 upgrade.
- **Clippy:** 0 warnings, 0 errors (verified)
- **Rustdoc:** Builds clean with `-D warnings` (verified)
- **Format:** `cargo fmt` clean (verified)

### Published Crates

| Crate | Tests | Status | Description |
|-------|-------|--------|-------------|
| tenflowers-core | 1,171 | Stable | Core tensor operations and GPU support |
| tenflowers-autograd | 521 | Stable | Automatic differentiation engine |
| tenflowers-neural | 11,596 | Stable | Neural network layers, models, and 150+ research domains |
| tenflowers-dataset | 660 | Stable | Data loading and preprocessing |
| tenflowers-ffi | 185 | Stable | Python bindings via PyO3 |
| tenflowers | 156 | Stable | Unified API and prelude |

### What Is Included

- Core tensor operations fully tested and validated
- Automatic differentiation engine with comprehensive gradient support
- Neural network layers (Dense, Conv2D, BatchNorm, Dropout, Attention, RNN, GNN, Transformers, and many more)
- Training utilities (optimizers including SGD, Adam, AdamW, LAMB, Lion, Muon; loss functions; training loops; LR schedulers)
- Data loading pipeline with multi-format support
- GPU acceleration via WGPU (cross-platform)
- SciRS2/NumRS2 ecosystem integration
- Python bindings with PyO3 (185 tests passing), including implicit PyTorch-style `.backward()`/`.grad()` autograd
- Security hardening (3 known transitive advisories — upstream fixes pending)
- Comprehensive documentation

### tenflowers-neural Feature Coverage

The neural crate alone has 11,596 tests covering:

**Core architectures:** attention mechanisms (multi-head, flash, ALiBi, RoPE), RNN (LSTM, GRU, bidirectional), transformers (encoder, decoder, efficient variants including RetNet, Mamba-2, GQA), CNN, graph neural networks (GCN, GAT, GraphSAGE, GIN, and advanced variants)

**Generative models:** normalizing flows, diffusion models, GANs, VAEs, energy-based models, neural rendering (3D Gaussian splatting, NeRF)

**Reinforcement learning:** policy gradient, actor-critic, PPO, SAC, multi-agent RL, safe RL, inverse RL, reward shaping, world models

**Scientific ML:** physics-informed neural networks (PINNs), neural ODEs/SDEs, operator learning (FNO, DeepONet, WNO, GNO), differentiable physics, simulation-based inference

**Domain-specific:** molecular GNN, protein structure prediction, drug discovery, medical imaging, audio models, speech recognition, video understanding, geospatial ML, climate ML, satellite ML, digital pathology, bio ML

**Advanced methods:** Bayesian deep learning, federated learning, meta-learning, NAS, knowledge distillation, quantum ML, geometric deep learning, causal inference, optimal transport, topological ML, continual learning, active learning, conformal prediction, and many more

## Installation

Add TenfloweRS to your `Cargo.toml`:

```toml
[dependencies]
tenflowers-core = "0.1.2"
tenflowers-neural = "0.1.2"
```

For GPU support:
```toml
[dependencies]
tenflowers-core = { version = "0.1.2", features = ["gpu"] }
```

For the unified API:
```toml
[dependencies]
tenflowers = "0.1.2"
```

## Quick Start

### Basic Tensor Operations
```rust,ignore
use tenflowers_core::{Tensor, Device, Context};

// Create a context for eager execution
let ctx = Context::new()?;

// Create tensors
let a = Tensor::<f32>::ones(&[2, 3]);
let b = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

// Operations execute immediately in eager mode
let c = a.add(&b)?;
let d = c.matmul(&b.transpose()?)?;

// Move to GPU
let gpu_tensor = a.to(Device::Gpu(0))?;

// Automatic differentiation
let tape = GradientTape::new();
let x = Tensor::variable(vec![1.0, 2.0, 3.0], &[3]);
let y = tape.watch(x.clone());
let z = y.pow(2.0)?;
let grads = tape.gradient(&z, &[&x])?;
```

### Graph Mode (TensorFlow 1.x style)
```rust,ignore
use tenflowers_core::{Graph, Session, Placeholder};

// Build a computation graph
let graph = Graph::new();
let a = graph.placeholder::<f32>("input_a", &[None, 784])?;
let w = graph.variable("weights", &[784, 10])?;
let b = graph.variable("bias", &[10])?;
let y = a.matmul(&w)?.add(&b)?;

// Create a session and run
let session = Session::new(&graph)?;
session.run(
    &[("input_a", input_tensor)],
    &["output"],
    &mut outputs
)?;
```

### Building a Neural Network
```rust,ignore
use tenflowers_neural::{Sequential, Dense, Conv2D, Model};
use tenflowers_core::Tensor;

// Define a CNN for image classification
let mut model = Sequential::new(vec![
    Box::new(Conv2D::new(32, (3, 3)).with_activation("relu")),
    Box::new(Conv2D::new(64, (3, 3)).with_activation("relu")),
    Box::new(layers::GlobalAveragePooling2D::new()),
    Box::new(Dense::new(128, true).with_activation("relu")),
    Box::new(layers::Dropout::new(0.5)),
    Box::new(Dense::new(10, true).with_activation("softmax")),
]);

// Compile the model
model.compile(
    optimizer::Adam::new(0.001),
    loss::SparseCategoricalCrossentropy::new(),
    vec![metrics::Accuracy::new()]
)?;

// Train the model
model.fit(
    &train_dataset,
    epochs: 10,
    batch_size: 32,
    validation_data: Some(&val_dataset),
)?;
```

### Data Pipeline
```rust,ignore
use tenflowers_dataset::{Dataset, DataLoader};

// Create a dataset from tensors
let dataset = Dataset::from_tensor_slices((images, labels))?
    .shuffle(1000)
    .batch(32)
    .prefetch(2);

// Iterate through batches
for (batch_images, batch_labels) in dataset.iter() {
    // Training step
}
```

## Architecture

TenfloweRS follows a modular architecture inspired by TensorFlow:

```
tenflowers/
├── tenflowers-core/      # Core tensor operations and device management
│   ├── tensor/           # Tensor implementation with device support
│   ├── ops/              # Operation registry and implementations
│   ├── kernels/          # CPU and GPU kernel implementations
│   ├── graph/            # Computation graph representation
│   └── device/           # Device abstraction and management
├── tenflowers-autograd/  # Automatic differentiation engine
│   ├── tape/             # GradientTape for eager mode
│   ├── graph_grad/       # Graph-based backpropagation
│   └── ops/              # Gradient definitions for operations
├── tenflowers-neural/    # Neural network layers, models, and research domains
│   ├── layers/           # Layer implementations (attention, RNN, GNN, etc.)
│   ├── optimizers/       # Training optimizers (SGD, Adam, LAMB, Lion, Muon)
│   ├── rl/               # Reinforcement learning
│   ├── federated/        # Federated learning
│   ├── diffusion/        # Diffusion models
│   ├── graph_neural_ode/ # Neural ODE on graphs
│   └── ...               # 150+ research domain modules
├── tenflowers-dataset/   # Data loading and preprocessing
│   ├── sources/          # Data source implementations
│   ├── transforms/       # Data transformation ops
│   └── iterators/        # Efficient iteration strategies
├── tenflowers-ffi/       # Python bindings via PyO3
│   └── src/              # Python-facing API
└── tenflowers/           # Unified API crate and prelude
```

### Core Components

#### 1. Tensor System
- Reference-counted tensors with device placement
- Lazy allocation and memory pooling
- Zero-copy views and slicing
- Automatic broadcasting

#### 2. Operation Framework
- Extensible operation registry
- Multi-dispatch for device/dtype specialization
- Shape inference at graph construction time
- Automatic gradient registration

#### 3. Execution Engines
- **Eager Mode**: Operations execute immediately
- **Graph Mode**: Build once, run multiple times with optimization

#### 4. Device Management
- Unified API for CPU, GPU, and custom devices
- Automatic device placement with hints
- Cross-device memory transfers
- Multi-GPU support with collective operations

## Building from Source

```bash
# Clone the repository
git clone https://github.com/cool-japan/tenflowers
cd tenflowers

# Build all crates
cargo build --workspace

# Run tests (requires cargo-nextest)
cargo nextest run --workspace

# Build with GPU support
cargo build --workspace --features gpu

# Build with BLAS acceleration (pure Rust)
cargo build --workspace --features blas-oxiblas

# Check for warnings (must pass -- no warnings policy)
cargo check --workspace
cargo clippy --workspace -- -D warnings

# Build documentation
cargo doc --workspace --no-deps
```

## Examples

Check out the [examples](examples/) directory for usage examples:

- `mnist_eager.rs` - MNIST classification with eager execution

## Performance

TenfloweRS is designed for high performance:

- **CPU**: SIMD vectorization, optional BLAS integration (OxiBLAS), Rayon parallelization
- **GPU**: WGPU compute shaders, memory pooling, kernel fusion
- **Memory**: Zero-copy operations, buffer reuse, lazy allocation

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas where we need help:
- GPU kernel development and optimization
- Performance benchmarking
- Documentation and examples
- Testing edge cases
- Python API expansion (tenflowers-ffi)

### Development Process
1. Open an issue to discuss your contribution
2. Follow the no-warnings policy (clippy must pass with `-D warnings`)
3. Write tests including gradient checks where applicable
4. Ensure zero `unwrap()` usage in production code
5. Submit a PR with clear description

## Roadmap

### v0.1.2 (Released 2026-07-08)
- Meta-crate: `error`, `logging`, `utils`, `platform`, `version_check` modules
- FFI: session-based `PyProfiler` / `PyProfileReport` profiling API; new `implicit_autograd` module giving `PyTensor` PyTorch-style eager `.backward()`/`.grad()` on top of the existing `GradientTape` engine; standalone `gradient_parity` finite-difference gradient checker
- Dataset: `hdf5_advanced` and `parquet_advanced` modules with chunked/filtered readers; from-scratch pure-Rust `formats::blosc` Blosc decoder (all 5 inner codecs + byte/bit-shuffle) wired into Zarr; real Symphonia-backed `formats::audio` decoding (WAV/MP3/FLAC); `formats::tfrecord_advanced` `SequenceExample` reader with real masked-CRC32 verification; new GPU image transforms (affine/perspective/elastic/histogram-equalize)
- Core: real ONNX protobuf import/export (`onnx_interop`) for the core graph representation, covering `Add/Sub/Mul/Div/Relu/Sigmoid/Tanh/MatMul/Reshape/Transpose/Identity/Concat/Softmax/Flatten/Gemm`; `session::SessionConfig::enable_graph_optimization` (default on) wires constant-folding/CSE/algebraic-simplification/strength-reduction/DCE/scheduling into real `Session` execution; N-D (`[N,d1,d2,...]`) segment reductions for all of `segment_max/min/prod/any/all`; GPU-einsum batched-matmul/transpose/diagonal/outer/trace now correctly delegate to CPU instead of honest-erroring; real `device::GpuAdapterCapabilities` from the live `wgpu::Adapter` (no more fabricated vendor/capability guessing); real LAPACK-backed `ops::lapack_f64` (inverse/determinant/SVD/solve)
- CUDA/ROCm/OpenCL feature flags documented with inline explanations
- Dependencies: numrs2 0.4.0, scirs2 0.6.0, oxicode 0.2.4, oxiarc-archive 0.3.4, oxifft 0.3.2, wgpu 30.0, pyo3 0.29, arrow/parquet 59.0; new oxiarc-lz4/oxiarc-deflate/oxiarc-snappy (Blosc inner codecs)
- Lock-poisoning `.expect()` calls replaced with `Result` propagation across `CheckpointManager`, `CrossDatacenterReplicator`, and `DeterministicContext`
- NCCL/Gloo/MPI/thread collective backends return honest `NotImplemented` errors (previously fabricated/simulated data); `DataParallelTrainer::train_step` now computes real gradients via finite differences instead of simulating the backward pass
- Two real GPU-path crash bugs fixed (`Tensor::from_storage` panic on GPU storage; hardcoded `Device::Gpu(0)` regardless of actual buffer device); a Miri-confirmed alignment UB fixed in `tenflowers-dataset`'s `MemoryPool`
- Security: resolved RUSTSEC-2026-0176/0177 (pyo3, via the 0.29 upgrade); 3 new/tracked transitive advisories (crossbeam-epoch, instant, paste — see Security section of CHANGELOG.md)
- 14,289+ tests, 39 skipped, 0 clippy warnings, 0 rustdoc warnings, 3 known transitive advisories (upstream-blocked, none directly exploitable)

### v0.1.1 (Released 2026-04-24)
- Core tensor operations and autograd
- 150+ neural network research domains
- GPU support via WGPU
- Python bindings via PyO3
- 13,484 tests, 41 skipped, 0 warnings, 0 vulnerabilities

### v0.2.0 (Planned)
- Expanded GPU kernel coverage (native GPU compute for currently CPU-fallback ops)
- Performance benchmarking suite with CI gates
- Wider ONNX operator coverage beyond the current core subset; TensorFlow SavedModel protobuf import
- Multi-GPU orchestration improvements; real NCCL/Gloo/MPI collective-communications backend
- Zarr Blosc *encoder* (decoder landed in 0.1.2)
- API stability improvements toward 1.0

### v1.0.0 (Future)
- Stable public API with semantic versioning guarantees
- Comprehensive ONNX compatibility
- Production deployment tooling
- WASM compilation target

## Comparison with TensorFlow

| Feature | TensorFlow | TenfloweRS |
|---------|------------|-------------|
| Language | C++ with Python API | Pure Rust with Python bindings |
| Memory Safety | Manual management | Guaranteed by Rust |
| Execution | Eager + Graph | Eager + Graph |
| GPU Support | CUDA, ROCm | WGPU (cross-platform) |
| Autodiff | Tape + Graph | Tape + Graph |
| Deployment | TFLite, TF.js | Native, WASM (planned) |
| Ecosystem | Mature, extensive | Growing, Rust-focused |

## Sponsorship

TenFlowers is developed and maintained by **COOLJAPAN OU (Team KitaSan)**.

If you find TenFlowers useful, please consider sponsoring the project to support continued development of the Pure Rust ecosystem.

[![Sponsor](https://img.shields.io/badge/Sponsor-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps us:
- Maintain and improve the COOLJAPAN ecosystem
- Keep the entire ecosystem (OxiBLAS, OxiFFT, SciRS2, etc.) 100% Pure Rust
- Provide long-term support and security updates

## License

This project is licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE)).

## Acknowledgments

TenfloweRS builds upon the excellent Rust scientific computing ecosystem:
- [NumRS2](https://github.com/cool-japan/numrs2) for n-dimensional arrays
- [SciRS2](https://github.com/cool-japan/scirs2) for scientific algorithms
- [OxiBLAS](https://github.com/cool-japan/oxiblas) for pure Rust BLAS
- [OxiFFT](https://github.com/cool-japan/oxifft) for pure Rust FFT
- [WGPU](https://github.com/gfx-rs/wgpu) for GPU compute

Special thanks to the TensorFlow team for the inspiration and architectural patterns.

## Community

- GitHub Issues: [Bug reports and feature requests](https://github.com/cool-japan/tenflowers/issues)
- Discussions: [Community forum](https://github.com/cool-japan/tenflowers/discussions)

---

**Note**: TenfloweRS is not affiliated with Google's TensorFlow. It is an independent project bringing ML capabilities to Rust.
