# TenfloweRS

[![Crates.io](https://img.shields.io/crates/v/tenflowers.svg)](https://crates.io/crates/tenflowers)
[![Documentation](https://docs.rs/tenflowers/badge.svg)](https://docs.rs/tenflowers)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../README.md#license)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)

A pure Rust implementation of TensorFlow, providing a comprehensive deep learning framework with Rust's safety and performance guarantees.

## Overview

TenfloweRS is the main convenience crate that re-exports TenfloweRS's Rust-native subcrates (core, autograd, neural, dataset), providing a unified API for deep learning in Rust. Built on the robust [SciRS2](https://github.com/cool-japan/scirs) ecosystem, it offers:

- **Production-Ready**: Full-featured neural networks, training, and deployment
- **High Performance**: GPU acceleration, SIMD optimization, mixed precision
- **Type Safety**: Rust's type system prevents common ML bugs at compile time
- **Cross-Platform**: CPU, GPU (CUDA, Metal, Vulkan), and WebGPU support
- **Ecosystem Integration**: Seamless integration with SciRS2, NumRS2, and OptiRS

## Quick Start

Add TenfloweRS to your `Cargo.toml`:

```toml
[dependencies]
tenflowers = "0.2.0"
```

### Basic Example

```rust
use tenflowers::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
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

> Note: `tenflowers::prelude` re-exports its own single-generic-argument
> `Result<T>` alias (`type Result<T> = std::result::Result<T, FrameworkError>`),
> so a `main` returning a boxed `dyn Error` must spell out `std::result::Result`
> as above rather than the bare `Result<(), Box<dyn Error>>`.

### Build a Neural Network

```rust
use tenflowers::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create a simple feedforward network. `Sequential::new` takes the
    // initial layer vec, and `.add` consumes/returns `Self` (builder style).
    let model = Sequential::<f32>::new(vec![])
        .add(Box::new(Dense::new(784, 128, true).with_activation("relu".to_string())))
        .add(Box::new(Dense::new(128, 10, true)));

    // Forward pass
    let input = Tensor::<f32>::zeros(&[32, 784]);
    let output = model.forward(&input)?;

    Ok(())
}
```

### Train a Model

```rust
use tenflowers::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut model = Sequential::<f32>::new(vec![])
        .add(Box::new(Dense::new(10, 64, true).with_activation("relu".to_string())))
        .add(Box::new(Dense::new(64, 3, true)));

    let x_train = Tensor::<f32>::zeros(&[100, 10]);
    let y_train = Tensor::<f32>::zeros(&[100, 3]);
    // `Trainer::fit` takes a `(inputs, targets)` batch iterator (e.g. from a
    // `DataLoader`); a single-batch `Vec` iterator works for small examples.
    let train_data = vec![(x_train, y_train)].into_iter();

    let mut optimizer = SGD::<f32>::new(0.01);
    let mut trainer = Trainer::new();
    let _state = trainer.fit(
        &mut model,
        &mut optimizer,
        train_data,
        None, // no validation set
        10,   // epochs
        categorical_cross_entropy,
    )?;

    Ok(())
}
```

> For common cases, `tenflowers::neural::quick_train` (also reachable via the
> prelude) wraps this pattern with MSE loss:
> `quick_train::train_with_sgd(&mut model, train_data, val_data, epochs, learning_rate)`
> and `quick_train::train_with_adam(...)`.

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
- Python bindings are provided by the separate `tenflowers-ffi` crate (PyO3-based,
  185+ tests) — not by a Cargo feature on this meta crate. See
  [tenflowers-ffi](../crates/tenflowers-ffi).

### Presets
- `minimal`: Only `std` (smallest possible build)
- `standard`: `std` + `parallel` (same as the default features)
- `full`: Enable most features (gpu, blas-oxiblas, simd, serialize, compression, onnx, autograd)

### Experimental
- `experimental`: Opt-in to preview APIs not covered by stability guarantees

### Enable GPU Support

```toml
[dependencies]
tenflowers = { version = "0.2.0", features = ["gpu"] }
```

### Enable All Features

```toml
[dependencies]
tenflowers = { version = "0.2.0", features = ["full"] }
```

## Architecture

TenfloweRS is organized into focused subcrates:

- **[tenflowers-core](../crates/tenflowers-core)**: Core tensor operations and device management (1,171 tests)
- **[tenflowers-autograd](../crates/tenflowers-autograd)**: Automatic differentiation engine (521+ tests)
- **[tenflowers-neural](../crates/tenflowers-neural)**: Neural network layers, models, and 150+ ML domains (11,596 tests)
- **[tenflowers-dataset](../crates/tenflowers-dataset)**: Data loading and preprocessing (660 tests)
- **[tenflowers-ffi](../crates/tenflowers-ffi)**: Python and C bindings (185+ tests)

This meta crate (156 tests) re-exports the public APIs of the four Rust-native subcrates
above for convenience, including the `tensor!` macro and `prelude` module. `tenflowers-ffi`
is a sibling crate in the same workspace providing separate Python/C bindings; it requires a
Python environment and is **not** re-exported by (or a dependency of) this meta crate —
depend on it directly for Python interop.

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

## Performance Benchmarks

Representative throughput figures on an AMD Ryzen 9 7950X (AVX2, 16 cores) and NVIDIA RTX 4090 (GPU).

### CPU Tensor Operations (f32, release mode)

| Operation | Shape | TenfloweRS | Notes |
|-----------|-------|-----------|-------|
| `add` | [4096, 4096] | ~2.8 GB/s | SIMD-vectorized |
| `matmul` | [512, 512]² | ~35 GFLOPS | OpenBLAS backend |
| `relu` | [1M] | ~4.5 GB/s | Auto-vectorized |
| `softmax` | [batch=128, 1024] | ~890 MB/s | Numerically stable log-sum-exp |

### GPU Operations (WGPU compute shaders, f32)

| Operation | Shape | Throughput |
|-----------|-------|------------|
| `matmul` | [2048, 2048]² | ~12 TFLOPS |
| Gaussian blur 5×5 | [H=512, W=512, C=3] | ~1800 MP/s |
| Random crop | [H=224, W=224, C=3] | ~3200 MP/s |
| Gaussian noise | [H=512, W=512, C=3] | ~2900 MP/s |

### Data Pipeline

| Workload | Config | Throughput |
|----------|--------|------------|
| CIFAR-10 prefetch | 4 workers, pinned | ~12,000 samples/s |
| ImageNet crop+resize+normalize | GPU transforms | ~2,400 samples/s |
| CSV streaming (1M rows) | SIMD stats | ~180 MB/s |

> Numbers are indicative. Run `cargo bench -p tenflowers-dataset` for detailed measurements on your hardware.

## Documentation

- [API Documentation](https://docs.rs/tenflowers)
- [Migration Guide (TensorFlow to TenfloweRS)](docs/MIGRATION_FROM_TENSORFLOW.md)
- [Quick Reference](docs/QUICK_REFERENCE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Prelude Stability Policy](docs/PRELUDE_STABILITY.md)
- [Workspace Release Checklist](../docs/RELEASE_CHECKLIST.md)

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).

## Status

TenfloweRS v0.2.0 (2026-07-12). 14,536+ tests passing across the workspace (39 skipped), 0 clippy warnings, 0 TODO markers. The project comprises ~686K SLoC of Rust across 1,533 files in 6 published crates.

## Links

- [GitHub Repository](https://github.com/cool-japan/tenflowers)
- [Issue Tracker](https://github.com/cool-japan/tenflowers/issues)
- [SciRS2 Project](https://github.com/cool-japan/scirs)
- [NumRS2 Project](https://github.com/cool-japan/numrs)
- [OptiRS Project](https://github.com/cool-japan/optirs)
