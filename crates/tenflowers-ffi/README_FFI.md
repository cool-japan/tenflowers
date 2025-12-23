# TenfloweRS FFI - Python Bindings

[![Build Status](https://github.com/cool-japan/tenflowers/workflows/Build%20Wheels/badge.svg)](https://github.com/cool-japan/tenflowers/actions)
[![PyPI version](https://badge.fury.io/py/tenflowers.svg)](https://pypi.org/project/tenflowers/)
[![Python](https://img.shields.io/pypi/pyversions/tenflowers.svg)](https://pypi.org/project/tenflowers/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](../LICENSE)

Python bindings for TenfloweRS - a pure Rust implementation of TensorFlow with high-performance tensor operations and automatic differentiation.

## Features

### Core Tensor Operations
- **Tensor Creation**: `zeros`, `ones`, `rand`, `randn`, `from_numpy`
- **Arithmetic**: `add`, `sub`, `mul`, `div`, `matmul`
- **Shape Manipulation**: `reshape`, `transpose`, `squeeze`, `unsqueeze`
- **Reductions**: `sum`, `mean`, `max`, `min`, `var`, `std`
- **Mathematical**: `exp`, `log`, `sqrt`, `sin`, `cos`, `tan`, `abs`

### Neural Network Layers
- **Dense Layers**: Fully connected layers with Xavier/He initialization
- **Convolutional**: Conv1D, Conv2D with various padding modes
- **Normalization**: BatchNorm, LayerNorm, GroupNorm, InstanceNorm
- **Recurrent**: LSTM, GRU, RNN with multi-layer support
- **Attention**: Multi-head attention, scaled dot-product attention
- **State Space Models**: Mamba/SSM for efficient sequence modeling
- **Pooling**: MaxPool2D, AvgPool2D
- **Regularization**: Dropout, Dropout2D, AlphaDropout

### Activation Functions
- Basic: ReLU, Sigmoid, Tanh, Softmax, LogSoftmax
- Advanced: GELU, Swish, Mish, ELU, SELU, LeakyReLU
- Specialized: Hardswish, Hardsigmoid, GLU variants

### Optimizers
- **Basic**: SGD (with momentum), Adam, AdamW, RMSprop
- **Advanced**: AdaBelief, RAdam, Nadam, AdaGrad, AdaDelta
- **Features**: Learning rate scheduling, weight decay, AMSGrad

### Learning Rate Schedulers
- StepLR, ExponentialLR, CosineAnnealingLR
- ReduceLROnPlateau, LinearLR, CosineAnnealingWarmRestarts

### Loss Functions
- MSE Loss, L1 Loss, Smooth L1 Loss
- Binary Cross Entropy, Cross Entropy, KL Divergence
- Hinge Embedding Loss, Cosine Embedding Loss

### Training Utilities
- Gradient Tape for automatic differentiation
- Early Stopping, LR Warmup
- Metrics Tracking, Progress Tracking
- Gradient clipping and normalization

### Data Type System
- **Floating Point**: float32, float64, float16, bfloat16
- **Integer**: int8, int16, int32, int64, uint8, uint16, uint32, uint64
- **Type Promotion**: Automatic type promotion for mixed operations
- **Safe Casting**: Validation for precision-preserving casts

## Installation

### From PyPI (when released)

```bash
pip install tenflowers
```

### From Source

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone repository
git clone https://github.com/cool-japan/tenflowers.git
cd tenflowers/crates/tenflowers-ffi

# Install with maturin
pip install maturin
maturin develop --release

# Or build wheel
maturin build --release
pip install target/wheels/tenflowers-*.whl
```

## Quick Start

### Basic Tensor Operations

```python
import tenflowers as tf
import numpy as np

# Create tensors
a = tf.zeros([3, 3])
b = tf.ones([3, 3])
c = tf.rand([3, 3])

# Arithmetic operations
d = tf.add(a, b)
e = tf.mul(b, c)

# Matrix multiplication
x = tf.ones([2, 3])
y = tf.ones([3, 4])
z = tf.matmul(x, y)  # Shape: [2, 4]

# NumPy interop
np_array = np.array([[1.0, 2.0], [3.0, 4.0]])
tensor = tf.tensor_from_numpy(np_array)
back_to_numpy = tf.tensor_to_numpy(tensor)
```

### Building Neural Networks

```python
import tenflowers as tf

# Create a simple feedforward network
class MLP:
    def __init__(self):
        self.layer1 = tf.PyDense(input_dim=784, output_dim=128)
        self.layer2 = tf.PyDense(input_dim=128, output_dim=10)
        self.bn = tf.PyBatchNorm1d(num_features=128)
        self.dropout = tf.PyDropout(p=0.2)

    def forward(self, x):
        x = self.layer1.forward(x)
        x = self.bn.forward(x)
        x = tf.relu(x)
        x = self.dropout.forward(x)
        x = self.layer2.forward(x)
        return x

# Create model
model = MLP()

# Forward pass
input_data = tf.randn([32, 784])
output = model.forward(input_data)
```

### Using Optimizers

```python
import tenflowers as tf

# Create optimizer
optimizer = tf.PyAdam(learning_rate=0.001)

# Or use variants
adamw = tf.PyAdamW(learning_rate=0.001)
sgd = tf.PySGD.with_momentum(learning_rate=0.01, momentum=0.9)
adabelief = tf.PyAdaBelief(learning_rate=0.001)
radam = tf.PyRAdam(learning_rate=0.001)

# Learning rate scheduling
scheduler = tf.PyStepLR(initial_lr=0.1, step_size=10, gamma=0.5)
```

### Advanced Layers

```python
import tenflowers as tf

# Multi-head attention
mha = tf.PyMultiheadAttention(embed_dim=512, num_heads=8)
query = tf.randn([10, 2, 512])  # (seq_len, batch, embed_dim)
key = tf.randn([10, 2, 512])
value = tf.randn([10, 2, 512])
output, weights = mha.forward(query, key, value)

# Mamba/State Space Model
mamba = tf.PyMamba(d_model=256, d_state=16)
input_seq = tf.randn([2, 10, 256])  # (batch, seq_len, d_model)
output, hidden = mamba.forward(input_seq)

# Transformer layers
encoder_layer = tf.PyTransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048
)
```

### Data Type System

```python
import tenflowers as tf

# Access dtype constants
print(f"Float32: {tf.float32}")
print(f"BFloat16: {tf.bfloat16}")

# Check dtype properties
assert tf.float32.is_floating_point()
assert tf.int32.is_integer()

# Type promotion
result_type = tf.result_type(tf.float32, tf.float64)
print(f"Result type: {result_type}")  # float64

# Safe casting
is_safe = tf.is_safe_cast_py(tf.float32, tf.float64)
print(f"Safe cast: {is_safe}")  # True
```

## Examples

Comprehensive examples are available in the `examples/` directory:

- `01_basic_tensors.py` - Basic tensor operations
- `02_neural_networks.py` - Neural network layers and components
- `03_training_mnist.py` - Complete training example

Run examples:

```bash
python examples/01_basic_tensors.py
python examples/02_neural_networks.py
python examples/03_training_mnist.py
```

## Development

### Building from Source

```bash
# Install development dependencies
pip install maturin pytest numpy

# Build in development mode
maturin develop

# Run tests
pytest tests/

# Run gradient parity tests
python tests/gradient_parity_test.py --verbose
```

### C/C++ Header Generation

Generate C header files for native interop:

```bash
python scripts/generate_c_header.py --output include/tenflowers.h
```

### Running Benchmarks

```bash
# Install benchmark dependencies
pip install pytest-benchmark

# Run benchmarks
pytest tests/ --benchmark-only
```

## Architecture

The FFI layer is organized into focused modules:

```
tenflowers-ffi/
├── src/
│   ├── lib.rs                  # Main module registration
│   ├── dtype.rs               # Data type abstraction
│   ├── tensor_ops.rs          # Core tensor operations
│   ├── math_ops.rs            # Mathematical operations
│   ├── error_mapping.rs       # Error handling
│   ├── serialization.rs       # Model save/load
│   ├── metrics.rs             # Evaluation metrics
│   └── neural/                # Neural network components
│       ├── mod.rs
│       ├── layers.rs          # Core layers
│       ├── optimizers.rs      # Basic optimizers
│       ├── extended_optimizers.rs  # Advanced optimizers
│       ├── activations.rs     # Activation functions
│       ├── losses.rs          # Loss functions
│       ├── attention.rs       # Attention mechanisms
│       ├── ssm.rs            # Mamba/SSM layers
│       ├── conv_layers.rs    # Convolutional layers
│       ├── recurrent.rs      # RNN/LSTM/GRU
│       ├── normalization.rs  # Normalization layers
│       ├── regularization.rs # Dropout, etc.
│       ├── schedulers.rs     # LR schedulers
│       ├── transformer.rs    # Transformer components
│       └── gradient_tape.rs  # Autograd
├── examples/                  # Python examples
├── tests/                     # Test suite
└── scripts/                   # Utility scripts
```

## Performance

TenfloweRS is designed for high performance:

- **Zero-copy NumPy interop** where possible
- **SIMD-optimized operations** for CPU
- **GPU acceleration** via WebGPU/Metal/CUDA
- **Efficient memory management** with buffer pooling
- **Multi-threaded operations** using Rayon

## Compatibility

- **Python**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Platforms**: Linux (manylinux), macOS (universal2), Windows
- **Architectures**: x86_64, aarch64

## Contributing

Contributions are welcome! Please see the main repository's CONTRIBUTING.md for guidelines.

## License

TenfloweRS is dual-licensed under:

- MIT License ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

## Citation

If you use TenfloweRS in your research, please cite:

```bibtex
@software{tenflowers2024,
  title = {TenfloweRS: Pure Rust TensorFlow-compatible Deep Learning Framework},
  author = {TenfloweRS Team},
  year = {2024},
  url = {https://github.com/cool-japan/tenflowers}
}
```

## Acknowledgments

TenfloweRS builds upon the excellent Rust scientific computing ecosystem:

- [scirs2](https://github.com/cool-japan/scirs) - Scientific computing foundation
- [ndarray](https://github.com/rust-ndarray/ndarray) - N-dimensional arrays
- [PyO3](https://github.com/PyO3/pyo3) - Rust-Python bindings
- [rayon](https://github.com/rayon-rs/rayon) - Data parallelism

## Support

- **Issues**: [GitHub Issues](https://github.com/cool-japan/tenflowers/issues)
- **Discussions**: [GitHub Discussions](https://github.com/cool-japan/tenflowers/discussions)
- **Documentation**: [Read the Docs](https://tenflowers.readthedocs.io)
