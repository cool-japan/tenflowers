# Changelog

All notable changes to TenfloweRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### In Progress
- GPU kernel implementations for core operations
- Shape inference system completion
- Graph mode execution engine
- Python bindings expansion
- ONNX import/export support

## [0.1.0-alpha.1] - 2025-09-27

### Added

#### Core Infrastructure
- **Tensor System**: Generic tensor type with device abstraction
  - Reference-counted buffer management
  - Zero-copy views and slicing
  - Strided layout support
  - Automatic broadcasting
- **Device Management**: Unified CPU/GPU abstraction
  - CPU backend via ndarray
  - GPU backend via WGPU (experimental)
  - Cross-device tensor transfers
  - Device placement strategies
- **Operation Registry**: Extensible operation system
  - Trait-based operation definitions
  - Kernel dispatch by device/dtype
  - Macro-based registration
  - Basic shape inference

#### Tensor Operations
- **Basic Ops**: Add, Sub, Mul, Div, Pow, Neg
- **Reductions**: Sum, Mean, Max, Min, ArgMax, ArgMin
- **Manipulation**: Reshape, Transpose, Concat, Stack, Squeeze
- **Linear Algebra**: MatMul (CPU only)
- **Activation**: ReLU, Sigmoid, Tanh (stubs)

#### Automatic Differentiation
- **GradientTape**: Reverse-mode automatic differentiation
  - Tape-based operation recording
  - Basic operation gradients (Add, Mul, MatMul, ReLU)
  - Multiple gradient computation
  - Persistent tape support
- **Integration**: Seamless integration with scirs2-autograd

#### Neural Network Module
- **Layers**: Layer trait with builder pattern
  - Dense/Linear layers
  - Conv2D (stub)
  - BatchNorm (stub)
  - Dropout
- **Models**: Sequential and Model traits
- **Optimizers**: SGD, Adam (simplified for f64)
- **Loss Functions**: MSE, CrossEntropy (stubs)

#### Data Pipeline
- **Dataset Trait**: Flexible data loading abstraction
- **TensorDataset**: In-memory tensor dataset
- **Transformations**: Basic preprocessing pipeline

#### FFI
- **Python Bindings**: Initial PyO3 integration
  - PyTensor wrapper
  - Basic tensor operations
  - NumPy interop foundation

### Known Limitations
- Most operations return "Not Implemented" errors
- GPU support is experimental and incomplete
- Limited operation coverage
- No graph mode execution yet
- Minimal Python API
- f32 support limited in some modules

### [0.1.0-alpha.1] 
- Complete GPU kernel implementations
- Expand operation coverage (Conv2D, pooling, normalization)
- Implement graph mode execution
- Add DataLoader with parallel loading
- Improve Python API coverage
- Graph optimization passes
- Mixed precision training
- Distributed training support
- ONNX import/export
- Performance optimizations
- Production-ready Python API
- TorchScript-like JIT compilation
- Quantization support
- Model zoo with pretrained models
- Comprehensive documentation

## Roadmap

### [1.0.0] - Target: 2026
- Stable API guarantee
- Performance parity with TensorFlow/PyTorch
- Full operation coverage
- Production deployment tools
- Extensive ecosystem integrations

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0-alpha.1 | 2025-09-27 | Initial alpha release with core infrastructure |
