# TenfloweRS

[![Crates.io](https://img.shields.io/crates/v/tenflowers.svg)](https://crates.io/crates/tenflowers)
[![Documentation](https://docs.rs/tenflowers/badge.svg)](https://docs.rs/tenflowers)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../README.md#license)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)

A pure Rust implementation of TensorFlow, providing a comprehensive deep learning framework with Rust's safety and performance guarantees.

## Overview

TenfloweRS is the main convenience crate that re-exports all TenfloweRS subcrates, providing a unified API for deep learning in Rust. Built on the robust [SciRS2](https://github.com/cool-japan/scirs) ecosystem, it offers:

- **Production-Ready**: Full-featured neural networks, training, and deployment
- **High Performance**: GPU acceleration, SIMD optimization, mixed precision
- **Type Safety**: Rust's type system prevents common ML bugs at compile time
- **Cross-Platform**: CPU, GPU (CUDA, Metal, Vulkan), and WebGPU support
- **Ecosystem Integration**: Seamless integration with SciRS2, NumRS2, and OptiRS

## Quick Start

Add TenfloweRS to your `Cargo.toml`:

```toml
[dependencies]
tenflowers = "0.1.0"
```

### Basic Example

```rust
use tenflowers::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create tensors
    let a = Tensor::<f32>::zeros(&[2, 3]);
    let b = Tensor::<f32>::ones(&[2, 3]);

    // Arithmetic operations
    let c = ops::add(&a, &b)?;

    // Matrix multiplication
    let x = Tensor::<f32>::ones(&[2, 3]);
    let y = Tensor::<f32>::ones(&[3, 4]);
    let z = ops::matmul(&x, &y)?;

    Ok(())
}
```

### Build a Neural Network

```rust
use tenflowers::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple feedforward network
    let mut model = Sequential::new();
    model.add(Dense::new(784, 128)?);
    model.add_activation(ActivationFunction::ReLU);
    model.add(Dense::new(128, 10)?);
    model.add_activation(ActivationFunction::Softmax);

    // Forward pass
    let input = Tensor::zeros(&[32, 784]);
    let output = model.forward(&input)?;

    Ok(())
}
```

### Train a Model

```rust
use tenflowers::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Sequential::new();
    model.add(Dense::new(10, 64)?);
    model.add(Dense::new(64, 3)?);

    let x_train = Tensor::zeros(&[100, 10]);
    let y_train = Tensor::zeros(&[100, 3]);

    // Quick training
    let results = quick_train(
        model,
        &x_train,
        &y_train,
        Box::new(SGD::new(0.01)),
        categorical_cross_entropy,
        10,  // epochs
        32,  // batch_size
    )?;

    Ok(())
}
```

## Features

TenfloweRS provides several optional features:

### Default Features
- `std`: Standard library support
- `parallel`: Parallel execution via Rayon

### GPU Acceleration
- `gpu`: GPU acceleration via WGPU (Metal, Vulkan, DirectX, WebGPU)
- `cuda`: CUDA support (Linux/Windows only)
- `cudnn`: cuDNN support (requires CUDA)
- `opencl`: OpenCL support
- `metal`: Metal support (macOS only)
- `rocm`: ROCm support (AMD GPUs)
- `nccl`: NCCL for distributed GPU training

### BLAS Acceleration
- `blas`: Generic BLAS support
- `blas-oxiblas`: OxiBLAS acceleration (pure Rust)
- `blas-accelerate`: Apple Accelerate framework (macOS only)

### Performance and Optimization
- `simd`: SIMD vectorization optimizations

### Serialization and I/O
- `serialize`: Serialization support (JSON, MessagePack)
- `compression`: Compression support for checkpoints
- `onnx`: ONNX model import/export

### Platform Support
- `wasm`: WebAssembly support

### Development
- `autograd`: Automatic differentiation support
- `benchmark`: Benchmarking utilities

### Language Bindings
- `python`: Python bindings via PyO3 (requires Python environment)

### Convenience
- `full`: Enable most features (gpu, blas-oxiblas, simd, serialize, compression, onnx, autograd)

### Enable GPU Support

```toml
[dependencies]
tenflowers = { version = "0.1.0", features = ["gpu"] }
```

### Enable All Features

```toml
[dependencies]
tenflowers = { version = "0.1.0", features = ["full"] }
```

## Architecture

TenfloweRS is organized into focused subcrates:

- **[tenflowers-core](../crates/tenflowers-core)**: Core tensor operations and device management (675 tests)
- **[tenflowers-autograd](../crates/tenflowers-autograd)**: Automatic differentiation engine (334 tests)
- **[tenflowers-neural](../crates/tenflowers-neural)**: Neural network layers, models, and 150+ ML domains (11,407 tests)
- **[tenflowers-dataset](../crates/tenflowers-dataset)**: Data loading and preprocessing (472 tests)
- **[tenflowers-ffi](../crates/tenflowers-ffi)**: Python and C bindings (48 tests)

This meta crate re-exports all public APIs for convenience, including the `tensor!` macro and `prelude` module.

## SciRS2 Integration

TenfloweRS is built on the SciRS2 scientific computing ecosystem:

```
TenfloweRS (Deep Learning Framework - TensorFlow-compatible API)
    builds upon
OptiRS (ML Optimization Specialization)
    builds upon
SciRS2 (Scientific Computing Foundation)
    builds upon
ndarray, num-traits, etc. (Core Rust Scientific Stack)
```

This architecture provides:
- Advanced numerical operations via `scirs2-core`
- Automatic differentiation via `scirs2-autograd`
- Neural network abstractions via `scirs2-neural`
- Optimized algorithms via `optirs`

## Documentation

- [API Documentation](https://docs.rs/tenflowers)
- [Architecture Guide](../ARCHITECTURE.md)
- [Performance Tuning](../PERFORMANCE_TUNING.md)
- [Capabilities Overview](../CAPABILITIES.md)

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).

## Status

TenfloweRS v0.1.0 is the first release (2026-03-20). All 11,477 tests passing across the workspace, 0 clippy warnings, 0 TODO markers. The project comprises ~765K SLoC of Rust across 1,446 files in 6 published crates.

## Links

- [GitHub Repository](https://github.com/cool-japan/tenflowers)
- [Issue Tracker](https://github.com/cool-japan/tenflowers/issues)
- [SciRS2 Project](https://github.com/cool-japan/scirs)
- [NumRS2 Project](https://github.com/cool-japan/numrs)
- [OptiRS Project](https://github.com/cool-japan/optirs)
