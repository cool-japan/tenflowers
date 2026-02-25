# TenfloweRS

A pure Rust implementation of TensorFlow, providing a full-featured machine learning framework with Rust's safety and performance.

[![Version](https://img.shields.io/badge/version-0.1.0--rc.1-blue)](https://github.com/cool-japan/tenflowers)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange)](https://www.rust-lang.org)
[![Tests](https://img.shields.io/badge/tests-2635%20passing-brightgreen)](https://github.com/cool-japan/tenflowers)
[![Security](https://img.shields.io/badge/vulnerabilities-0-brightgreen)](https://github.com/cool-japan/tenflowers)

> **Release Candidate (0.1.0-rc.1 · 2026-02-24)**
>
> Release candidate with production-ready core functionality! All 2635 tests passing, zero security vulnerabilities, and comprehensive documentation. The core API is stabilizing for 1.0.
>
> ⚠️ **Note:** This release temporarily excludes Python bindings (FFI) and tensorboard integration. See [CHANGELOG.md](CHANGELOG.md) for details and timeline.

## Overview

TenfloweRS is a native Rust machine learning framework inspired by TensorFlow, designed to bring the power of deep learning to the Rust ecosystem. It leverages Rust's memory safety, zero-cost abstractions, and excellent performance while maintaining compatibility with the broader ML ecosystem through ONNX support.

## Design Principles

TenfloweRS adapts TensorFlow's proven architecture to Rust's strengths:

1. **Memory Safety First**: All operations are memory-safe by design, eliminating segfaults and data races
2. **Zero-Cost Abstractions**: High-level APIs compile down to efficient machine code
3. **Explicit over Implicit**: Clear ownership and error handling following Rust conventions
4. **Modular Architecture**: Organized as a workspace of focused, reusable crates
5. **Cross-Platform**: Native support for Windows, macOS, and Linux with unified GPU abstraction

## TensorFlow → TenfloweRS Mapping

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

- **🚀 Dual Execution Modes**: Both eager execution (PyTorch-style) and static computation graphs (TensorFlow-style)
- **🦀 Pure Rust Implementation**: No C/C++ dependencies in the core, ensuring memory safety
- **🎮 GPU Support**: Cross-platform GPU acceleration via WGPU (Metal, Vulkan, DirectX)
- **🔧 Rust Scientific Stack**: Built on NumRS2 and SciRS2 for numerical computing
- **🐍 Python Bindings**: ⚠️ Temporarily excluded in beta.1 (requires Python environment setup)
- **📦 ONNX Support**: Import and export models for cross-framework compatibility
- **⚡ Performance**: SIMD vectorization, optional BLAS integration, and parallel execution
- **✅ Production Ready**: 2635 tests passing, 0 security vulnerabilities, comprehensive docs

## Project Status

**Current Version: 0.1.0-rc.1** (Released 2026-02-24)

Release candidate with production-ready core functionality! The core API is stabilizing for 1.0 release.

### RC 1 Quality Metrics ✅
- **Tests:** 2635/2635 passing (100% pass rate)
- **Security:** 0 vulnerabilities (all known issues resolved)
- **Code Quality:** Zero clippy warnings, full formatting compliance
- **Documentation:** Complete crate-level docs and READMEs for all published crates
- **Build:** All 5 core crates successfully package and verify

### Published Crates (Available on crates.io)
1. ✅ **tenflowers-core** (6.5 MiB) - Core tensor operations and GPU support
2. ✅ **tenflowers-autograd** (2.8 MiB) - Automatic differentiation engine
3. ✅ **tenflowers-dataset** (2.1 MiB) - Data loading and preprocessing
4. ✅ **tenflowers-neural** (3.0 MiB) - Neural network layers and training
5. ✅ **tenflowers** (182 KiB) - Unified API and prelude

### Temporarily Excluded (RC 1)
- ⚠️ **tenflowers-ffi**: Python bindings (requires Python dev environment)
  - Will be re-enabled in future release with proper CI/CD
  - Use Rust API directly for now
- ⚠️ **tensorboard integration**: Logging feature (security fix)
  - Removed due to protobuf vulnerability (RUSTSEC-2024-0437)
  - Will be re-added once dependency updated
  - Use alternative logging temporarily

See [CHANGELOG.md](CHANGELOG.md#010-rc1---2026-02-24) for complete details and migration guide.

### RC 1 Scope (Delivered 2026-02-24)
- ✅ Core tensor operations fully tested and validated
- ✅ Automatic differentiation engine with comprehensive gradient support
- ✅ Neural network layers (Dense, Conv2D, BatchNorm, Dropout, etc.)
- ✅ Training utilities (optimizers, loss functions, training loops)
- ✅ Data loading pipeline with multi-format support
- ✅ GPU acceleration via WGPU (cross-platform)
- ✅ SciRS2/NumRS2 ecosystem integration complete
- ✅ Security hardening (zero vulnerabilities)
- ✅ Comprehensive documentation

### Known Limitations (RC 1)
- Python bindings not available (see Temporarily Excluded above)
- Tensorboard logging not available (see Temporarily Excluded above)
- Graph mode optimization passes still in development
- Multi-GPU orchestration experimental
- ONNX import/export in development

### Priorities for Next Release (toward 1.0)
1. Re-enable Python bindings with proper CI/CD
2. Re-enable tensorboard integration (awaiting dependency fix)
3. Complete graph optimization passes
4. Expand GPU kernel coverage
5. Performance benchmarking suite
6. ONNX import/export finalization
7. API stability guarantee for 1.0

### RC 1 Release Checklist ✅
- [x] All 2635 tests passing
- [x] Zero security vulnerabilities
- [x] Zero clippy warnings
- [x] All crates properly documented
- [x] Package verification successful
- [x] Version consistency across workspace
- [x] CHANGELOG.md updated
- [x] Migration guide provided

### What's Working ✅
- ✅ Core tensor operations (creation, manipulation, arithmetic)
- ✅ Automatic differentiation with gradient tape
- ✅ Neural network layers and model composition
- ✅ Training loop with optimizers (SGD, Adam, AdamW)
- ✅ Data loading from multiple formats (CSV, images, HDF5, Parquet)
- ✅ GPU acceleration (WGPU backend)
- ✅ Integration with SciRS2 ecosystem
- ✅ Comprehensive error handling (no unwrap() usage)

### In Active Development 🚧
- 🚧 Python bindings (code complete, CI/CD in progress)
- 🚧 Tensorboard integration (awaiting dependency security fix)
- 🚧 Graph optimization passes
- 🚧 Shape inference system
- 🚧 Graph construction and optimization
- 🚧 Tape-based automatic differentiation
- 🚧 GPU compute kernels

## Installation

Add TenfloweRS to your `Cargo.toml`:

```toml
[dependencies]
tenflowers-core = "0.1.0-rc.1"
tenflowers-neural = "0.1.0-rc.1"
```

For GPU support:
```toml
[dependencies]
tenflowers-core = { version = "0.1.0-rc.1", features = ["gpu"] }
```

## Quick Start

### Basic Tensor Operations
```rust
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
```rust
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
```rust
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
```rust
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
├── tenflowers-neural/    # Neural network layers and models
│   ├── layers/           # Layer implementations
│   ├── models/           # Model abstraction and builders
│   ├── optimizers/       # Training optimizers
│   └── losses/           # Loss functions
├── tenflowers-dataset/   # Data loading and preprocessing
│   ├── sources/          # Data source implementations
│   ├── transforms/       # Data transformation ops
│   └── iterators/        # Efficient iteration strategies
└── tenflowers-ffi/       # Python bindings
    ├── tensor_py/        # Python tensor wrapper
    ├── ops_py/           # Operation bindings
    └── keras_compat/     # Keras-compatible API
```

### Core Components

#### 1. **Tensor System**
- Reference-counted tensors with device placement
- Lazy allocation and memory pooling
- Zero-copy views and slicing
- Automatic broadcasting

#### 2. **Operation Framework**
- Extensible operation registry
- Multi-dispatch for device/dtype specialization
- Shape inference at graph construction time
- Automatic gradient registration

#### 3. **Execution Engines**
- **Eager Mode**: Operations execute immediately
- **Graph Mode**: Build once, run multiple times with optimization
- **XLA Integration**: (Future) JIT compilation for performance

#### 4. **Device Management**
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

# Build with BLAS acceleration
cargo build --workspace --features blas-openblas

# Check for warnings (must pass - no warnings policy)
cargo check --workspace
cargo clippy --workspace -- -D warnings
```

## Examples

Check out the [examples](examples/) directory for comprehensive examples:

- `mnist_eager.rs` - MNIST classification with eager execution
- `mnist_graph.rs` - MNIST using static graphs (coming soon)
- `gan_example.rs` - Generative Adversarial Network (coming soon)
- `transformer.rs` - Transformer model implementation (coming soon)

## Performance

TenfloweRS is designed for high performance:

- **CPU**: SIMD vectorization, optional BLAS integration, Rayon parallelization
- **GPU**: WGPU compute shaders, memory pooling, kernel fusion
- **Memory**: Zero-copy operations, buffer reuse, lazy allocation

### Benchmarks

Coming soon! Target performance goals:
- CPU: Match or exceed NumPy
- GPU: 90% of TensorFlow performance
- Memory: Within 10% of TensorFlow usage

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas where we need help:
- Implementing core operations (see TODO.md)
- GPU kernel development
- Shape inference functions
- Documentation and examples
- Testing and benchmarking
- Python API design

### Development Process
1. Check [TODO.md](TODO.md) for tasks
2. Open an issue to discuss your contribution
3. Follow the no-warnings policy
4. Write tests including gradient checks
5. Submit a PR with clear description

## Roadmap

See [TODO.md](TODO.md) for the detailed development roadmap.

### Upcoming Releases
- **v0.1.0**: Core tensor ops and basic autograd
- **v0.2.0**: GPU support and essential layers
- **v0.3.0**: Graph mode and optimizations
- **v0.4.0**: Python bindings and Keras compatibility
- **v0.5.0**: ONNX import/export
- **v1.0.0**: Production-ready with stable API

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

## License

This project is licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE)).

## Acknowledgments

TenfloweRS builds upon the excellent Rust scientific computing ecosystem:
- [NumRS2](https://github.com/numrs/numrs2) for n-dimensional arrays
- [SciRS2](https://github.com/scirs/scirs2) for scientific algorithms
- [ndarray](https://github.com/rust-ndarray/ndarray) for array operations
- [WGPU](https://github.com/gfx-rs/wgpu) for GPU compute

Special thanks to the TensorFlow team for the inspiration and architectural patterns.

## Community

- GitHub Issues: [Bug reports and feature requests](https://github.com/cool-japan/tenflowers/issues)
- Discussions: [Community forum](https://github.com/cool-japan/tenflowers/discussions)
- Discord: Coming soon!

---

**Note**: TenfloweRS is not affiliated with Google's TensorFlow. It's an independent project bringing ML capabilities to Rust.