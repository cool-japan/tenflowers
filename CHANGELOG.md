# Changelog

All notable changes to TenfloweRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### In Progress
- Additional GPU kernel implementations for advanced operations
- Complete shape inference system
- Graph mode execution engine enhancements
- Python bindings expansion
- ONNX import/export support

## [0.1.0-alpha.2] - 2025-12-23

### Added

#### Documentation Improvements
- **Comprehensive Crate Documentation**: Added extensive crate-level documentation to all crates
  - `tenflowers-core`: Complete API overview with examples for tensor operations, GPU acceleration, mixed precision, and performance monitoring
  - `tenflowers-dataset`: Full guide to data loading, transformations, and advanced features
  - `tenflowers-ffi`: Python bindings documentation with NumPy integration examples
  - All crates now include Quick Start guides and architecture overviews
- **Enhanced README**: Updated with alpha.2 information and current capabilities
- **API Documentation**: Improved rustdoc comments throughout the codebase

#### Performance Features
- **CUDA Support**: Enhanced GPU backend with CUDA optimization paths
- **Memory Optimization**: Improved memory management and buffer pooling
- **SIMD Enhancements**: Additional SIMD-accelerated operations
- **Profiling Tools**: Built-in performance benchmarking and monitoring utilities

#### Core Enhancements
- **Deterministic Execution**: Added deterministic mode for reproducible results
- **Quantization**: Expanded quantization support for model deployment
- **Mixed Precision**: Improved mixed precision training capabilities
- **Checkpointing**: Enhanced model checkpointing and restoration
- **Error Handling**: Improved error messages and shape validation

#### Neural Network Module
- **Layer Expansion**: Additional neural network layer implementations
- **Optimizer Improvements**: Enhanced optimizer implementations
- **Training Utilities**: Improved training loop abstractions

#### Dataset Module
- **Data Quality Tools**: Built-in data quality analysis and drift detection
- **Advanced Sampling**: Stratified and importance sampling strategies
- **Performance**: NUMA-aware scheduling and zero-copy operations
- **Distributed Loading**: Distributed and sharded data loading support

### Improved
- **SciRS2 Integration**: Complete migration to SciRS2 ecosystem primitives
  - All operations now use `scirs2_core::ndarray` instead of direct `ndarray`
  - Random number generation via `scirs2_core::random`
  - Numeric traits via `scirs2_core::num_traits`
- **Type System**: Enhanced data type support (f16, bf16, etc.)
- **Shape Inference**: Improved shape validation and broadcasting
- **GPU Memory**: Better GPU memory management and metrics
- **Documentation**: Comprehensive rustdoc throughout all modules

### Fixed
- **Compilation Issues**: Resolved various compilation warnings and errors
- **Type Safety**: Fixed trait bound issues across generic implementations
- **Memory Leaks**: Fixed memory management issues in GPU operations
- **API Consistency**: Standardized API patterns across crates

### Changed
- **Version**: Updated to 0.1.0-alpha.2 across all crates
- **Build System**: Improved workspace configuration
- **Testing**: Enhanced test coverage and infrastructure

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
| 0.1.0-alpha.2 | 2025-12-23 | Documentation overhaul, CUDA enhancements, SciRS2 integration complete |
| 0.1.0-alpha.1 | 2025-09-27 | Initial alpha release with core infrastructure |
