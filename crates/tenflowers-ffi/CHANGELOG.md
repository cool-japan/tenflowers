# Changelog

All notable changes to the TenfloweRS FFI (Python bindings) crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-20

### Added

#### Python Bindings via PyO3
- Core tensor operations: creation, arithmetic, shape manipulation, and computation
- Gradient tape integration with PyTorch-style automatic differentiation
- Neural network layers: Dense, Conv1D/2D, pooling, Sequential
- Normalization layers: BatchNorm, LayerNorm, GroupNorm, InstanceNorm
- Recurrent layers: LSTM, GRU, RNN
- Activation functions: ReLU, Sigmoid, Tanh, GELU, Swish, Mish, and more
- Loss functions: MSE, CrossEntropy, BCE, and variants
- NumPy interoperability with seamless tensor <-> ndarray conversion
- Device management for CPU/GPU
- Memory profiling utilities and optimization tools
- Benchmarking framework with TensorFlow baseline comparison
- Hook system for forward/backward debugging and monitoring
- Large model support infrastructure for 1B+ parameter models

#### Type Stubs for IDE Support
- Complete type hints (`python/tenflowers/__init__.pyi`, 700+ lines)
- Full IDE autocomplete and type checking support for all exposed classes and functions
- Numpy typing integration with `numpy.typing`
- PEP 561 compliant with `py.typed` marker file
- Compatible with mypy, pyright, pylance, and other type checkers

#### Extended Optimizers
- SGD, Adam, AdamW, RMSprop (core set)
- AdaBelief with AMSGrad variant and weight decay
- RAdam (Rectified Adam) with automatic variance warmup
- Nadam (Nesterov-accelerated Adam) with momentum schedule
- AdaGrad with per-parameter learning rate adaptation
- AdaDelta with decaying gradient accumulation
- Learning rate schedulers
- All optimizers support parameter dict management, state persistence, and LR adjustment

#### DType System
- Comprehensive dtype abstraction: float32, float64, float16, bfloat16
- Integer types: int8, int16, int32, int64, uint8, uint16, uint32, uint64
- Boolean type support
- Type property queries: `is_floating_point()`, `is_integer()`, `is_signed()`, `is_supported()`
- Type promotion (`result_type()`) and casting validation (`can_cast_to()`, `is_safe_cast()`)
- NumPy-compatible dtype constants exported to Python

#### Mamba/State Space Models Support
- PyMamba: Selective State Space Model with configurable d_model, d_state, expansion factor, dt_rank, dropout, and bias
- PyStateSpaceModel: General SSM with learnable A/B/C/D matrices and step-by-step state evolution
- Forward pass with optional initial state, returning output and final hidden state
- State dictionary serialization for both model types

#### C API Scaffolding
- Opaque type definitions for memory safety
- Result enum for error handling
- Function declarations for tensor operations, gradient tape, device management, and memory management
- Cross-platform compatibility and automated header generation script

#### Integration Test Suite
- 14 comprehensive end-to-end tests covering tensor operations, gradient flow, model workflows, all optimizer types, normalization layers, conv/pooling layers, recurrent layers, Mamba/SSM, NumPy interop, DType validation, and training workflow simulation
- Gradient parity testing harness with numerical gradient comparison (finite differences)
- Detailed test reporting with pass/fail summary

#### Performance Benchmark Suite
- 12 benchmark categories: tensor creation, arithmetic, matrix operations, reductions, shape manipulation, dense layer forward, conv layer, normalization, recurrent layers, optimizer step speed, NumPy comparison, and memory usage
- Warmup iterations for stable measurements
- Throughput metrics (operations per second) with sorted results

#### Utility Functions
- Tensor inspection: `tensor_info()`, `tensor_summary()`, `same_shape()`, `is_scalar()`, `is_vector()`, `is_matrix()`, `numel()`
- Shape operations: `validate_shapes()`, `broadcast_shape()`, `is_broadcastable()`, `normalize_dimension()`
- Memory utilities: `tensor_memory_bytes()`, `tensor_memory_str()`, `format_bytes()`
- Array generation: `arange()`, `linspace()`
- Device information: `get_device_info()`, `is_gpu_available()`, `version()`

#### Documentation and Examples
- Comprehensive README with installation instructions, quick start guide, and architecture diagram
- Python examples: basic tensors, neural network components, MNIST training workflow
- C API documentation with usage patterns

### Known Limitations
- Only `float32` is fully supported; `float16`, `bfloat16`, and other types are prepared but not yet implemented
- GPU features require proper hardware and driver support
- C API provides scaffolding only; not yet ready for production distribution
- No published Python wheels; build from source required

---

Developed by COOLJAPAN OU (Team KitaSan)
