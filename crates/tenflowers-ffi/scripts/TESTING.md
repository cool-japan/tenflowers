# TenfloweRS FFI Testing and Benchmarking Guide

This guide explains how to run tests, benchmarks, and validation for the TenfloweRS Python bindings.

## Table of Contents

- [Installation](#installation)
- [Unit Tests](#unit-tests)
- [Integration Tests](#integration-tests)
- [Performance Benchmarks](#performance-benchmarks)
- [Gradient Parity Tests](#gradient-parity-tests)
- [Type Checking](#type-checking)
- [Continuous Integration](#continuous-integration)

## Installation

### Build and Install in Development Mode

```bash
# From the tenflowers-ffi directory
cd crates/tenflowers-ffi

# Build and install in development mode (editable install)
maturin develop --release

# Or for faster development iteration (debug build)
maturin develop
```

### Install from Wheel

```bash
# Build wheel
maturin build --release

# Install wheel
pip install target/wheels/tenflowers-*.whl
```

## Unit Tests

### Rust Unit Tests

Run Rust unit tests with nextest:

```bash
# Run all tests
cargo nextest run --all-features

# Run only FFI crate tests
cargo nextest run --package tenflowers-ffi --all-features

# Run specific test
cargo nextest run --package tenflowers-ffi test_py_dense_layer

# Run with verbose output
cargo nextest run --package tenflowers-ffi --all-features --nocapture
```

### Python Unit Tests

After installing the package, run Python unit tests:

```bash
# Run all Python tests
python -m pytest crates/tenflowers-ffi/tests/

# Run specific test file
python -m pytest crates/tenflowers-ffi/tests/gradient_parity_test.py

# Run with verbose output
python -m pytest crates/tenflowers-ffi/tests/ -v

# Run with coverage
python -m pytest crates/tenflowers-ffi/tests/ --cov=tenflowers
```

## Integration Tests

The integration test suite validates end-to-end functionality across all components.

### Running Integration Tests

```bash
# Make sure the package is installed first
maturin develop --release

# Run integration tests
python crates/tenflowers-ffi/tests/integration_test.py

# Or run directly
cd crates/tenflowers-ffi/tests
./integration_test.py
```

### Integration Test Coverage

The integration tests cover:

1. **Basic Tensor Operations** - Creation, arithmetic, shape manipulation
2. **Gradient Flow** - Automatic differentiation with gradient tape
3. **Dense Layer Forward** - Fully connected layer operations
4. **Sequential Model** - Multi-layer model composition
5. **Optimizer Step** - Parameter updates and optimization
6. **Multiple Optimizers** - All 9 optimizer types
7. **Normalization Layers** - BatchNorm, LayerNorm, GroupNorm, InstanceNorm
8. **Conv and Pooling Layers** - Conv2D, MaxPool2D, AvgPool2D
9. **Recurrent Layers** - LSTM, GRU, RNN
10. **SSM Layers** - Mamba and StateSpaceModel
11. **Utility Functions** - Helper functions and tensor utilities
12. **NumPy Interop** - Bidirectional numpy conversion
13. **DType System** - Type system and type promotion
14. **End-to-End Training** - Complete training workflow

### Expected Output

```
==============================================================
TenfloweRS FFI Integration Test Suite
Version: 0.1.0-alpha.2
==============================================================

Running: Basic Tensor Operations
✓ PASSED: Basic Tensor Operations

Running: Gradient Flow
✓ PASSED: Gradient Flow

...

==============================================================
TEST SUMMARY
==============================================================
Total tests: 14
Passed: 14
Failed: 0
Success rate: 100.0%
```

## Performance Benchmarks

The performance benchmark suite measures throughput and latency across operations.

### Running Benchmarks

```bash
# Make sure the package is installed first
maturin develop --release

# Run full benchmark suite
python crates/tenflowers-ffi/tests/performance_benchmark.py

# Or run directly
cd crates/tenflowers-ffi/tests
./performance_benchmark.py
```

### Benchmark Categories

1. **Tensor Creation** - zeros, ones, rand, randn
2. **Arithmetic Operations** - add, mul, sub, div
3. **Matrix Operations** - matmul at various sizes
4. **Reduction Operations** - sum, mean, max, min
5. **Shape Manipulation** - reshape, transpose, squeeze
6. **Dense Layers** - Forward pass throughput
7. **Convolutional Layers** - Conv2D performance
8. **Normalization Layers** - BatchNorm, LayerNorm
9. **Recurrent Layers** - LSTM, GRU throughput
10. **Optimizer Steps** - Optimizer update speed
11. **NumPy Comparison** - TenfloweRS vs NumPy
12. **Memory Patterns** - Memory allocation/deallocation

### Benchmark Output

```
==============================================================
TenfloweRS FFI Performance Benchmark Suite
Version: 0.1.0-alpha.2
==============================================================

Benchmarking: zeros(1000x1000) x100
  Warmup iterations: 3
  Benchmark iterations: 10
  ✓ 45.23ms (2211.50 ops/sec)

...

==============================================================
BENCHMARK SUMMARY
==============================================================
Total benchmarks: 35

Results (sorted by throughput):
1. add(1000x1000) x1000
   Time: 12.345ms | Throughput: 81000.00 ops/sec
2. mul(1000x1000) x1000
   Time: 13.456ms | Throughput: 74300.00 ops/sec
...
```

### Customizing Benchmarks

You can modify benchmark parameters in the script:

```python
# In performance_benchmark.py

# Adjust number of iterations
suite = BenchmarkSuite(warmup_iters=5, bench_iters=20)

# Adjust tensor sizes
size = 2000  # Larger matrices
ops = 500    # More iterations
```

## Gradient Parity Tests

Gradient parity tests validate autograd correctness using numerical gradients.

### Running Gradient Tests

```bash
# Make sure the package is installed first
maturin develop --release

# Run gradient parity tests
python crates/tenflowers-ffi/tests/gradient_parity_test.py

# Or run directly
cd crates/tenflowers-ffi/tests
./gradient_parity_test.py
```

### What It Tests

The gradient parity test compares analytical gradients (from autograd) with numerical gradients (finite differences) for:

- Unary operations: exp, log, sin, cos, sqrt, abs, neg
- Binary operations: add, sub, mul, div, pow
- Matrix operations: matmul
- Reduction operations: sum, mean, max
- Activation functions: relu, sigmoid, tanh

### Expected Output

```
Testing operation: exp
  Input shape: (3, 3)
  Numerical gradient computation...
  Autograd gradient extraction...
  Gradient comparison...
  ✓ Gradients match within tolerance

...

Gradient Parity Test Results:
  Total operations tested: 16
  Passed: 16
  Failed: 0
  Success rate: 100.0%
```

## Type Checking

TenfloweRS provides complete type stubs for static type checking.

### Running Type Checkers

```bash
# Install type checking tools
pip install mypy pyright

# Run mypy
mypy your_code.py

# Run pyright
pyright your_code.py
```

### Example Typed Code

```python
import tenflowers as tf
from tenflowers import Tensor, Dense, Adam

def train_step(
    model: Dense,
    optimizer: Adam,
    x: Tensor,
    y: Tensor
) -> Tensor:
    """Type-checked training step."""
    with tf.GradientTape() as tape:
        predictions = model.forward(x)
        loss = tf.sum(tf.mul(tf.sub(predictions, y), tf.sub(predictions, y)))

    params = model.parameters()
    for param in params:
        grad = tape.gradient(loss, param.data())
        if grad is not None:
            param.data().backward(grad)

    optimizer.step()
    optimizer.zero_grad()

    return loss
```

### IDE Support

The type stubs enable:
- **Autocomplete** - Full method and parameter completion
- **Inline Documentation** - Docstrings in hover tooltips
- **Type Checking** - Catch type errors before runtime
- **Refactoring** - Safe renaming and code transformations

Supported IDEs:
- VSCode (with Pylance)
- PyCharm
- Vim/Neovim (with language servers)
- Emacs (with language servers)
- Any IDE with Python LSP support

## Continuous Integration

### GitHub Actions Workflow

The project includes a comprehensive CI workflow (`.github/workflows/build-wheels.yml`) that:

1. **Builds wheels** for multiple platforms:
   - Linux (x86_64, aarch64) - manylinux2014
   - macOS (x86_64, aarch64, universal2)
   - Windows (x64, x86)

2. **Tests wheel installation** across:
   - Python 3.9, 3.10, 3.11, 3.12
   - Multiple operating systems
   - Basic functionality tests

3. **Publishes to PyPI** (on tags)

4. **Publishes to TestPyPI** (on main branch)

### Running Locally

To simulate CI locally:

```bash
# Build all wheels
maturin build --release

# Run clippy (zero warnings required)
cargo clippy --package tenflowers-ffi --all-features -- -D warnings

# Run formatter
cargo fmt --package tenflowers-ffi --check

# Run tests
cargo nextest run --package tenflowers-ffi --all-features

# Build documentation
cargo doc --package tenflowers-ffi --no-deps
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure to build and install the package first
   ```bash
   maturin develop --release
   ```

2. **Test Failures**: Ensure all dependencies are installed
   ```bash
   pip install numpy pytest
   ```

3. **Performance Issues**: Use release build for benchmarks
   ```bash
   maturin develop --release  # Not just `maturin develop`
   ```

4. **GPU Tests**: GPU features require proper hardware and drivers
   ```bash
   # Check GPU availability
   python -c "import tenflowers as tf; print(tf.is_gpu_available())"
   ```

### Debug Mode

For debugging, use debug build and enable logging:

```bash
# Debug build
maturin develop

# Enable Rust logging
export RUST_LOG=debug

# Run tests with backtrace
RUST_BACKTRACE=1 cargo nextest run --package tenflowers-ffi
```

## Performance Tips

1. **Always use release builds** for benchmarking
2. **Warmup iterations** ensure stable measurements
3. **Multiple iterations** reduce variance
4. **Pin CPU affinity** for consistent results on multi-core systems
5. **Close background applications** to reduce noise
6. **Use consistent power settings** (disable CPU throttling)

## Contributing

When adding new features:

1. **Add unit tests** in Rust (`src/test_module.rs`)
2. **Add Python tests** in `tests/` directory
3. **Update integration tests** if adding new APIs
4. **Add benchmarks** for performance-critical code
5. **Update type stubs** (`python/tenflowers/__init__.pyi`)
6. **Update CHANGELOG** with your changes

## Resources

- [PyO3 Documentation](https://pyo3.rs/)
- [Maturin Documentation](https://www.maturin.rs/)
- [NumPy C API](https://numpy.org/doc/stable/reference/c-api/)
- [Python Type Hints](https://peps.python.org/pep-0484/)
- [Benchmark Best Practices](https://pyperf.readthedocs.io/)

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions and share ideas
- **Documentation**: Check README and API docs
