# TenfloweRS

A pure Rust implementation of TensorFlow, providing a full-featured machine learning framework with Rust's safety and performance.

[![Version](https://img.shields.io/badge/version-0.1.1-blue)](https://github.com/cool-japan/tenflowers)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange)](https://www.rust-lang.org)
[![Tests](https://img.shields.io/badge/tests-13484%20passing-brightgreen)](https://github.com/cool-japan/tenflowers)
[![Security](https://img.shields.io/badge/vulnerabilities-0-brightgreen)](https://github.com/cool-japan/tenflowers)

> **v0.1.1 (2026-04-24)**
>
> TenfloweRS v0.1.1 is the latest release, with 13,484 tests passing across 6 crates,
> zero clippy warnings, zero security vulnerabilities, and comprehensive documentation.

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
- **Production Ready**: 13,484 tests passing, 0 security vulnerabilities, comprehensive docs

## Project Status

**Current Version: 0.1.1** (Released 2026-04-24)

First release with full-featured ML capabilities across all 6 crates.

### v0.1.1 Quality Metrics

- **Tests:** 13,484 passing, 41 skipped (100% pass rate)
- **Code:** 1,465 Rust files, ~643K SLoC (~772K total Rust lines)
- **Security:** 0 vulnerabilities
- **Clippy:** 0 warnings, 0 errors
- **Rustdoc:** Builds clean with `-D warnings`
- **TODO markers:** 3 remaining (stubs in autograd/neural)

### Published Crates

| Crate | Tests | Status | Description |
|-------|-------|--------|-------------|
| tenflowers-core | 936 | Stable | Core tensor operations and GPU support |
| tenflowers-autograd | 455 | Stable | Automatic differentiation engine |
| tenflowers-neural | 11,537 | Stable | Neural network layers, models, and 150+ research domains |
| tenflowers-dataset | 504 | Stable | Data loading and preprocessing |
| tenflowers-ffi | 48 | Stable | Python bindings via PyO3 |
| tenflowers | 13 (doc) | Stable | Unified API and prelude |

### What Is Included

- Core tensor operations fully tested and validated
- Automatic differentiation engine with comprehensive gradient support
- Neural network layers (Dense, Conv2D, BatchNorm, Dropout, Attention, RNN, GNN, Transformers, and many more)
- Training utilities (optimizers including SGD, Adam, AdamW, LAMB, Lion, Muon; loss functions; training loops; LR schedulers)
- Data loading pipeline with multi-format support
- GPU acceleration via WGPU (cross-platform)
- SciRS2/NumRS2 ecosystem integration
- Python bindings with PyO3 (48 tests passing)
- Security hardening (zero vulnerabilities)
- Comprehensive documentation

### tenflowers-neural Feature Coverage

The neural crate alone has 11,537 tests covering:

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
tenflowers-core = "0.1.1"
tenflowers-neural = "0.1.1"
```

For GPU support:
```toml
[dependencies]
tenflowers-core = { version = "0.1.1", features = ["gpu"] }
```

For the unified API:
```toml
[dependencies]
tenflowers = "0.1.1"
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

### v0.1.1 (Released 2026-04-24)
- Core tensor operations and autograd
- 150+ neural network research domains
- GPU support via WGPU
- Python bindings via PyO3
- 13,484 tests, 41 skipped, 0 warnings, 0 vulnerabilities

### v0.2.0 (Planned)
- Graph optimization passes (constant folding, operator fusion, dead code elimination)
- Expanded GPU kernel coverage
- Performance benchmarking suite with CI gates
- ONNX import/export finalization
- Multi-GPU orchestration improvements
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
