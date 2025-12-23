# Changelog

All notable changes to the TenfloweRS FFI (Python bindings) crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha.2] - 2025-01-XX

### Added

#### Python Type Stubs and IDE Support
- **Type Stubs** (`python/tenflowers/__init__.pyi` - 700+ lines)
  - Complete type hints for all exposed classes and functions
  - Full IDE autocomplete and type checking support
  - Covers tensors, layers, optimizers, utilities, and custom exceptions
  - Numpy typing integration with `numpy.typing`
  - PEP 561 compliant with `py.typed` marker file
  - Supports mypy, pyright, pylance, and other type checkers

#### Testing and Quality Assurance
- **Integration Test Suite** (`tests/integration_test.py` - 500+ lines)
  - 14 comprehensive end-to-end tests covering:
    - Basic tensor operations and arithmetic
    - Gradient flow and automatic differentiation
    - Dense and Sequential model workflows
    - All optimizer types (9 optimizers)
    - Normalization layers (BatchNorm, LayerNorm, GroupNorm, InstanceNorm)
    - Convolutional and pooling layers
    - Recurrent layers (LSTM, GRU, RNN)
    - State Space Models (Mamba, SSM)
    - Utility functions and helper methods
    - NumPy interoperability
    - DType system validation
    - Complete training workflow simulation
  - Detailed test reporting with pass/fail summary
  - Executable test runner with clear output

- **Performance Benchmark Suite** (`tests/performance_benchmark.py` - 600+ lines)
  - Comprehensive performance benchmarks across 12 categories:
    - Tensor creation operations (zeros, ones, rand, randn)
    - Arithmetic operations (add, mul, sub, div)
    - Matrix operations (matmul) at multiple sizes
    - Reduction operations (sum, mean, max, min)
    - Shape manipulation (reshape, transpose, squeeze)
    - Dense layer forward passes
    - Convolutional layer operations
    - Normalization layer throughput
    - Recurrent layer performance
    - Optimizer step speed
    - NumPy vs TenfloweRS comparison
    - Memory usage patterns
  - Warmup iterations for stable measurements
  - Throughput metrics (operations per second)
  - Detailed summary with sorted results
  - Comparison mode for relative performance analysis

#### Extended Optimizers (`src/neural/extended_optimizers.rs`)
- **PyAdaBelief**: AdaBelief optimizer with adaptive belief in gradient direction
  - Supports AMSGrad variant for stability
  - Weight decay regularization
  - Configurable beta parameters and epsilon
- **PyRAdam**: Rectified Adam optimizer with variance warmup
  - Automatic warmup period calculation
  - Variance rectification for stable early training
  - Degenerate to SGD when warmup needed
- **PyNadam**: Nesterov-accelerated Adam optimizer
  - Combines Adam with Nesterov momentum
  - Momentum schedule with decay
  - Bias correction for both moments
- **PyAdaGrad**: Adaptive gradient optimizer
  - Per-parameter learning rate adaptation
  - Accumulates squared gradients
  - Suitable for sparse gradients
- **PyAdaDelta**: Extension of AdaGrad with decaying gradient accumulation
  - RMSprop-like decay of gradient history
  - No explicit learning rate required
  - Delta-based parameter updates

All optimizers support:
- Parameter dictionary management (`get_param_dict`, `set_param_dict`)
- State persistence and loading
- Learning rate adjustment (`get_lr`, `set_lr`)
- Zero gradient functionality

#### Mamba/State Space Models (`src/neural/ssm.rs`)
- **PyMamba**: Selective State Space Model (Mamba architecture)
  - Configurable model dimension (`d_model`)
  - State dimension (`d_state`) for memory capacity
  - Expansion factor for hidden dimension scaling
  - Delta rank (`dt_rank`) for time step projection
  - Dropout regularization
  - Optional bias in linear projections
  - State dictionary serialization
  - Forward pass with optional initial state
  - Returns output and final hidden state

- **PyStateSpaceModel**: General state space model implementation
  - Configurable input/state/output dimensions
  - Learnable state transition matrix (A)
  - Input-to-state projection (B)
  - State-to-output projection (C)
  - Optional feedthrough matrix (D)
  - Step-by-step state evolution
  - Sequence processing with state tracking

#### Data Type System (`src/dtype.rs`)
- **PyDType**: Comprehensive dtype abstraction for type system
  - Support for floating point types: `float32`, `float64`, `float16`, `bfloat16`
  - Support for integer types: `int8`, `int16`, `int32`, `int64`
  - Support for unsigned types: `uint8`, `uint16`, `uint32`, `uint64`
  - Boolean type support
  - Type property queries:
    - `is_floating_point()`: Check if dtype is floating point
    - `is_integer()`: Check if dtype is integer
    - `is_signed()`: Check if dtype is signed
    - `is_supported()`: Check if dtype is currently supported (only float32 for now)
  - Type promotion and casting:
    - `result_type()`: Compute result type for mixed-dtype operations
    - `can_cast_to()`: Check if cast is possible
    - `is_safe_cast()`: Verify precision-preserving casts
  - String representation and parsing
  - NumPy compatibility

Module-level dtype functions:
- `result_type_py()`: Python-accessible type promotion
- `is_safe_cast_py()`: Python-accessible safe casting check
- `can_cast_py()`: Python-accessible cast validation

Dtype constants exported to Python:
- `float32`, `float64`, `float16`, `bfloat16`
- `int8`, `int16`, `int32`, `int64`
- `uint8`, `uint16`, `uint32`, `uint64`
- `bool_dtype`

#### Utility Functions (`src/utils.rs`)
- **Tensor Inspection**:
  - `tensor_info()`: Returns dict with shape, ndim, numel, dtype, requires_grad, is_pinned
  - `same_shape()`: Check if two tensors have identical shapes
  - `is_scalar()`: Check if tensor is 0-dimensional
  - `is_vector()`: Check if tensor is 1-dimensional
  - `is_matrix()`: Check if tensor is 2-dimensional
  - `numel()`: Get total number of elements
  - `tensor_summary()`: Human-readable tensor description
  - `print_tensor_info()`: Print tensor information to console

- **Shape Operations**:
  - `validate_shapes()`: Validate shapes for binary operations
  - `all_same_shape()`: Check if all tensors have same shape
  - `broadcast_shape()`: Compute broadcast shape (NumPy rules)
  - `is_broadcastable()`: Check if shapes are broadcastable
  - `validate_dimension()`: Validate dimension index
  - `normalize_dimension()`: Convert negative dimension to positive

- **Memory Utilities**:
  - `tensor_memory_bytes()`: Calculate memory usage in bytes
  - `tensor_memory_str()`: Format memory as human-readable string
  - `format_bytes()`: Format byte count (B/KB/MB/GB/TB)

- **Array Generation**:
  - `arange()`: Generate range of values (NumPy-like)
  - `linspace()`: Generate linearly spaced values

- **Device Information**:
  - `get_device_info()`: Get device information string
  - `is_gpu_available()`: Check GPU availability
  - `version()`: Get TenfloweRS version

#### Documentation and Examples
- **README_FFI.md**: Comprehensive documentation
  - Feature overview with categorized capabilities
  - Installation instructions (PyPI and from source)
  - Quick start guide with code examples
  - Neural network building tutorial
  - Advanced layers usage (Mamba, attention, transformers)
  - Data type system documentation
  - Architecture diagram
  - Performance notes
  - Compatibility matrix
  - Development workflow

- **Python Examples**:
  - `examples/01_basic_tensors.py`: Basic tensor operations
    - Tensor creation (zeros, ones, rand, randn, from_numpy)
    - Basic operations (add, mul, sub, matmul, transpose)
    - Shape manipulation (reshape, squeeze, unsqueeze, cat)
    - Reductions (sum, mean, max, min)
    - Mathematical functions (exp, log, sin, cos, sqrt, abs)
    - DType system demonstration

  - `examples/02_neural_networks.py`: Neural network components
    - Activation functions (ReLU, GELU, Swish, Mish, etc.)
    - Core layers (Dense, Conv1D/2D, pooling)
    - Normalization (BatchNorm, LayerNorm, GroupNorm, InstanceNorm)
    - Recurrent layers (LSTM, GRU, RNN)
    - Attention mechanisms (Multi-head attention, scaled dot-product)
    - Mamba/SSM layers
    - Transformer components (encoder/decoder layers, positional encoding)
    - Optimizers (SGD, Adam, AdamW, AdaBelief, RAdam, etc.)
    - Learning rate schedulers
    - Loss functions

  - `examples/03_training_mnist.py`: Complete training example
    - SimpleMNISTModel class definition
    - Training loop with gradient tape
    - Optimizer usage and learning rate scheduling
    - Loss computation and metrics tracking
    - Model evaluation

#### Testing Infrastructure
- **scripts/generate_c_header.py**: C header generation
  - Automated C/C++ header file generation
  - Opaque type definitions for safety
  - Result enum for error handling
  - Function declarations for:
    - Tensor operations (creation, arithmetic, shape manipulation)
    - Gradient tape operations
    - Device management
    - Memory management
  - Cross-platform compatibility
  - Proper error handling patterns

- **tests/gradient_parity_test.py**: Gradient validation harness
  - `GradientParityTester` class for comprehensive testing
  - Numerical gradient computation (finite differences)
  - Autograd gradient extraction from gradient tape
  - Gradient comparison with configurable tolerance
  - Operation-level testing:
    - Unary operations (exp, log, sin, cos, sqrt, abs, neg)
    - Binary operations (add, sub, mul, div, pow)
    - Matrix operations (matmul)
    - Reduction operations (sum, mean, max)
    - Activation functions (relu, sigmoid, tanh)
  - Comprehensive test suite
  - Verbose logging and debugging output
  - Statistical gradient verification

### Changed

#### Module Organization
- **src/neural/mod.rs**:
  - Added `extended_optimizers` module
  - Added `ssm` (State Space Models) module
  - Re-exported new optimizer types: `PyAdaBelief`, `PyRAdam`, `PyNadam`, `PyAdaGrad`, `PyAdaDelta`
  - Re-exported SSM types: `PyMamba`, `PyStateSpaceModel`
  - Registered extended optimizers in `register_neural_functions()`
  - Registered SSM layers in `register_neural_functions()`

- **src/lib.rs**:
  - Added `dtype` module declaration
  - Added `utils` module declaration
  - Registered dtype system with Python module:
    - All dtype constants (float32, float64, int32, etc.)
    - Type promotion function (`result_type`)
    - Safe casting validation (`is_safe_cast_py`)
    - Cast validation (`can_cast`)
  - Registered 22 utility functions for tensor manipulation
  - Improved module organization and documentation

### Fixed

#### Dependency Issues
- **tenflowers-autograd/src/amp_policy.rs**:
  - Fixed `MixedPrecisionConfig` initialization with missing fields:
    - Added `opt_level: OptimizationLevel::O1`
    - Added `scale_growth_factor: 2.0`
    - Added `scale_backoff_factor: 0.5`
    - Added `keep_master_weights: true`
    - Added `enable_gradient_clipping: false`
    - Added `gradient_clip_norm: 1.0`
  - Added `OptimizationLevel` import from `tenflowers_core::mixed_precision`
  - Resolved compilation error blocking FFI crate build

### Technical Details

#### Architecture Enhancements
1. **Optimizer Extensibility**: Extended optimizer collection provides comprehensive training algorithm support matching PyTorch/TensorFlow capabilities

2. **State Space Models**: Modern sequence modeling architecture (Mamba) with selective state space design for efficient long-range dependencies

3. **Type System**: Future-proof dtype abstraction supporting mixed precision training (fp16/bf16) with proper type promotion rules

4. **Utility Layer**: Comprehensive helper functions reducing boilerplate code in Python, improving developer experience

5. **Testing Infrastructure**: Gradient parity testing ensures autograd correctness through numerical validation

#### Performance Considerations
- Zero-copy NumPy interop maintained
- SIMD-optimized operations (via tenflowers-core)
- GPU acceleration ready (via WebGPU/Metal/CUDA)
- Efficient memory management with buffer pooling
- Multi-threaded operations using Rayon

#### API Stability
- All new APIs are exposed through PyO3 with proper error handling
- Type safety maintained through Rust's strong type system
- Memory safety guaranteed by Rust ownership model
- No unsafe code in FFI layer (except PyO3 internals)

### Migration Notes

For users upgrading from 0.1.0-alpha.1:

1. **New Optimizers**: Five new optimizer classes available:
   ```python
   import tenflowers as tf

   # New optimizers
   optimizer = tf.PyAdaBelief(learning_rate=0.001)
   optimizer = tf.PyRAdam(learning_rate=0.001)
   optimizer = tf.PyNadam(learning_rate=0.001)
   optimizer = tf.PyAdaGrad(learning_rate=0.01)
   optimizer = tf.PyAdaDelta(learning_rate=1.0)
   ```

2. **Mamba/SSM Layers**: State space models now available:
   ```python
   # Mamba layer
   mamba = tf.PyMamba(d_model=256, d_state=16)
   output, hidden = mamba.forward(input_seq)

   # General SSM
   ssm = tf.PyStateSpaceModel(input_dim=128, state_dim=32, output_dim=128)
   output, state = ssm.forward(input_seq)
   ```

3. **DType System**: Access dtype constants and type operations:
   ```python
   # Dtype constants
   print(tf.float32, tf.bfloat16, tf.int32)

   # Type promotion
   result_dtype = tf.result_type(tf.float32, tf.float64)

   # Safe casting
   is_safe = tf.is_safe_cast_py(tf.float32, tf.float64)
   ```

4. **Utility Functions**: New helper functions available:
   ```python
   # Tensor inspection
   info = tf.tensor_info(tensor)
   summary = tf.tensor_summary(tensor, "my_tensor")

   # Shape operations
   broadcast_result = tf.broadcast_shape([3, 1, 4], [1, 5, 4])

   # Memory utilities
   memory = tf.tensor_memory_str(tensor)

   # Device info
   device_info = tf.get_device_info()
   ```

### Known Issues

1. **Limited Dtype Support**: Currently only `float32` is fully supported; `float16`, `bfloat16`, and other types are prepared but not yet implemented
2. **GPU Requirements**: GPU features require proper hardware and driver support
3. **Memory Profiling**: Some advanced profiling features may not be available on all platforms

### Future Plans

- Complete float16/bfloat16 implementation
- Add ONNX export support
- Implement stable ABI for C/C++ interop
- Add more comprehensive examples (vision, NLP, RL)
- Improve error messages and diagnostics
- Add quantization support (int8, int4)

## [0.1.0-alpha.1] - 2025-01-XX

### Added
- Initial Python bindings via PyO3
- Core tensor operations (creation, arithmetic, shape manipulation)
- Basic neural network layers (Dense, Conv1D/2D, pooling)
- Normalization layers (BatchNorm, LayerNorm, GroupNorm, InstanceNorm)
- Recurrent layers (LSTM, GRU, RNN)
- Basic optimizers (SGD, Adam, AdamW, RMSprop)
- Learning rate schedulers
- Activation functions (ReLU, Sigmoid, Tanh, GELU, etc.)
- Loss functions (MSE, CrossEntropy, BCE, etc.)
- Gradient tape for automatic differentiation
- NumPy interoperability
- Device management (CPU/GPU)
- Memory profiling utilities
- Benchmarking framework
- Visualization tools

### Initial Release Notes
First alpha release of TenfloweRS Python bindings, providing core functionality for tensor operations and neural network training.

---

## Version History

- **0.1.0-alpha.2** (Current): Extended optimizers, Mamba/SSM, dtype system, utilities
- **0.1.0-alpha.1**: Initial release with core functionality
