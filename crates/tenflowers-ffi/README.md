# TenfloweRS FFI

Foreign Function Interface for TenfloweRS, providing Python bindings and C API for seamless integration with other languages and frameworks.

> v0.2.0 (2026-07-13) | 337 Rust tests + 55 Python (pytest) + 13 Python (integration_test.py) passing | 0 clippy warnings
> Python bindings are functional. Build from source via maturin.

## Overview

`tenflowers-ffi` implements:
- **Python Bindings**: PyO3-based Python API for tensor operations, neural network layers, and optimizers
- **Eager Autograd (PyTorch-style)**: `PyTensor.set_requires_grad()` / `.backward()` / `.grad()` — a thread-local, auto-activating implicit gradient tape (`implicit_autograd`) layered on the existing explicit `GradientTape` engine, so `x.backward(); x.grad()` works without ever constructing a tape object by hand. As of the 2026-07-13 rewrite the implicit tape is genuinely wired through the entire layer/optimizer/loss surface, not just a minimal Dense/Sequential/MSE path:
  - **Layers** (all `#[pyclass]`, real `forward()` participating in the tape): `Dense` and `PyParameter` (the autograd leaf-node type returned by `layer.parameters()`), `Conv1D`/`Conv2D`/`Conv3D` + `MaxPool2D`/`AvgPool2D`, `Embedding`/`EmbeddingBag`, `BatchNorm1d`/`LayerNorm`/`GroupNorm`/`InstanceNorm1d`, `MultiheadAttention`, `TransformerEncoderLayer`/`TransformerDecoderLayer`/`PositionalEncoding`, `LSTM`/`GRU`/`RNN` + single-step `LSTMCell`/`GRUCell`. Caveat: `Conv2D`'s backward/autograd wiring is confirmed only for unit dilation, `groups == 1`, and no explicit integer padding (default `padding=(0,0)`); other configurations still forward correctly but are not confirmed tape-linked (documented on the struct itself in `neural/conv_layers/mod.rs`).
  - **Optimizers** (9 total, all real `#[pyclass]` with `step(&mut self, model)` performing genuine gradient-based updates, not no-ops): `SGD`, `Adam`, `RMSprop`, `AdamW` (`neural/optimizers/mod.rs`); `AdaBelief`, `RAdam`, `Nadam`, `AdaGrad`, `AdaDelta` (`neural/extended_optimizers/mod.rs`)
  - **Losses** (free `#[pyfunction]`s, not classes — call as `tf.mse_loss(y_pred, y_true)`): `mse_loss`, `binary_cross_entropy`, `cross_entropy`, `l1_loss`, `smooth_l1_loss`, `kl_div_loss`, `hinge_embedding_loss`, `cosine_embedding_loss` (`neural/losses.rs`)
  - `PyTensor::transpose`/`reshape`/`slice` now correctly record themselves onto the implicit tape (`tensor_ops.rs`); previously `transpose`/`reshape` silently dropped out of `.backward()` on a tracked tensor since no `UnaryOpKind::Transpose`/`Reshape` variant existed.
  - Real multi-step training-loop convergence (not just "runs without crashing") is proven in `tests/test_training_convergence.py` for a lone `Dense` layer + `SGD`, a 3-layer `Sequential` MLP + `Adam`, and `Conv2D` + `Adam`.
- **C API**: C FFI bindings for cross-language compatibility
- **NumPy Integration**: Tensor conversion with NumPy arrays
- **Visualization**: Gradient flow analysis and visualization utilities
- **DType Promotion**: Automatic dtype promotion across the Python/Rust boundary
- **Eager Execution Optimizer**: Python-side eager execution optimization
- **Device Abstraction**: `PyDevice` class with `Device.cpu()`, `Device.gpu(id)`, `Device.rocm(id)` constructors
- **Structured Error Mapping**: Exhaustive `TensorError` → `TenflowersError` mapping covering all 23 variants
- **Rich Tensor Repr**: `PyTensor.__repr__` shows actual dtype; `__len__`, `.ndim`, `.numel()` properties added
- **C Header Generation**: `build.rs` regenerates `tenflowers.h` when `TENFLOWERS_REGENERATE_C_HEADER=1` is set
- **Gradient Parity Checking**: standalone finite-difference gradient checker (`gradient_parity`) for validating analytical gradients against numerical ones
- **Profiling**: session-based `PyProfiler` for per-op timing/memory tracking
- **Stable API Surface**: `stable_api` module cataloguing the stability guarantees of the public Python/C surface

## Features

- **Zero-Copy Interop**: Efficient data exchange with Python/NumPy where possible
- **Pythonic API**: Familiar interface for Python users
- **PyTorch-Familiar Autograd**: eager `.backward()` / `.grad()` on `PyTensor`, in addition to the explicit `GradientTape` API — no need to learn a TensorFlow-style tape-first workflow just to get a gradient. Genuinely wired through every layer type (Dense, Conv1D/2D/3D, Embedding, all 4 normalization layers, MultiheadAttention, Transformer encoder/decoder, LSTM/GRU/RNN) and all 9 optimizers (SGD, Adam, RMSprop, AdamW, AdaBelief, RAdam, Nadam, AdaGrad, AdaDelta), with real multi-step convergence proven in `tests/test_training_convergence.py`
- **Type Safety**: Automatic type conversions with safety checks
- **Error Handling**: Exhaustive Rust→Python exception mapping (all 23 `TensorError` variants)
- **GPU Support**: Tensor operations on GPU from Python via `PyDevice`
- **Gradient Flow Analysis**: Inspect and visualize gradient propagation
- **Debug Support**: `__repr__` with real dtype, `__len__`, `.ndim`, `.numel()` on `PyTensor`

## Python API Usage

### Installation

```bash
# Build from source
pip install maturin
maturin develop --release
```

### Basic Tensor Operations

```python
import tenflowers as tf

# Create tensors
a = tf.Tensor([1, 2, 3, 4], shape=[2, 2])
b = tf.ones([2, 2])

# Basic operations
c = a + b
d = a @ b  # Matrix multiplication
e = tf.relu(a)

# NumPy interoperability
import numpy as np
np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
tensor = tf.from_numpy(np_array)
back_to_np = tensor.numpy()

# GPU operations
if tf.cuda.is_available():
    gpu_tensor = a.to("cuda:0")
    result = gpu_tensor @ gpu_tensor.T
    cpu_result = result.to("cpu")
```

### Neural Network Example

```python
import tenflowers as tf
import tenflowers.nn as nn
import tenflowers.optim as optim

# Define a model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = tf.relu(self.fc1(x))
        return self.fc2(x)

# Training
model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch_x, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Autograd Integration

Eager, PyTorch-style autograd is available directly on `PyTensor` via a
thread-local implicit gradient tape — no explicit tape object required:

```python
import tenflowers as tf

# Enable gradient tracking on each leaf tensor
x = tf.ones([2])
x.set_requires_grad(True)
y = tf.ones([2])
y.set_requires_grad(True)

# Compute function: z = sum(x * x + y * y)
z = tf.sum(tf.add(tf.mul(x, x), tf.mul(y, y)))

# Compute gradients
z.backward()
print(x.grad())  # dz/dx = 2x -> [2.0, 2.0]
print(y.grad())  # dz/dy = 2y -> [2.0, 2.0]
```

The 2026-07-13 rewrite wires this same implicit tape through real layers and
optimizers, so a genuine training loop converges end to end — the pattern
below mirrors `tests/test_training_convergence.py::test_dense_layer_training_converges`,
which asserts the loss drops by at least two orders of magnitude over 50 steps
on a linear-regression target:

```python
import tenflowers as tf

layer = tf.PyDense(4, 1, activation=None)
for param in layer.parameters():
    param.set_requires_grad(True)

optimizer = tf.SGD(learning_rate=0.1)

x_tensor = tf.tensor_from_numpy(X)  # X: np.ndarray, shape (batch, 4)
y_tensor = tf.tensor_from_numpy(y)  # y: np.ndarray, shape (batch, 1)

for step in range(50):
    y_pred = layer.forward(x_tensor)
    loss = tf.mse_loss(y_pred, y_tensor)

    loss.backward()
    optimizer.step(layer)
    optimizer.zero_grad(layer)
```

This same `.backward()` / `optimizer.step()` pattern works for every layer
and optimizer listed under Overview above (`Sequential` MLPs, `Conv2D`,
`LSTM`/`GRU`/`RNN`, `MultiheadAttention`, Transformer encoder/decoder layers,
and all 9 optimizers) — not just `Dense` + `SGD`.

#### Implicit tape vs. explicit `GradientTape`

The eager API above (`x.set_requires_grad()` / `.backward()` / `.grad()`) is
distinct from the explicit, TensorFlow-style `PyGradientTape` API
(`tenflowers.PyGradientTape` / `.watch()` / `.gradient()`,
`neural::gradient_tape`). Both are backed by the same underlying Rust
`GradientTape` engine, but **the explicit `PyGradientTape` API does not work
end-to-end**: `PyGradientTape.watch()` only clones a tensor's current value
onto the tape at call time, and the free functions used to build computations
(`tf.add`, `tf.mul`, etc.) are not tape-aware for this explicit-tape style —
they never record operation-graph edges onto it, so `tape.gradient()` cannot
trace a real computation chain. This is unrelated to, and does not affect,
the implicit `.backward()`/`.grad()` API documented above, which is what got
fixed this release and is confirmed fully working through the entire
layer/optimizer surface. Real tape-based autograd through plain tensor ops
via the explicit-tape style remains a separate, not-yet-implemented future
project.

## C API Usage

### Basic Example

```c
#include <tenflowers.h>

int main() {
    // Initialize TenfloweRS
    tf_init();

    // Create tensors
    size_t shape[] = {2, 3};
    TF_Tensor* a = tf_zeros(shape, 2, TF_FLOAT32);
    TF_Tensor* b = tf_ones(shape, 2, TF_FLOAT32);

    // Perform operations
    TF_Tensor* c = tf_add(a, b);

    // Get data pointer
    float* data = (float*)tf_data_ptr(c);

    // Cleanup
    tf_free_tensor(a);
    tf_free_tensor(b);
    tf_free_tensor(c);
    tf_cleanup();

    return 0;
}
```

## Architecture

### Python Bindings Structure

- **Core Module**: Tensor operations and basic functionality
- **NN Module**: Neural network layers and utilities
- **Optim Module**: Optimization algorithms
- **Autograd Module**: Automatic differentiation
- **Utils Module**: Data loading, metrics, visualization

### Memory Management

- **Reference Counting**: Automatic memory management in Python
- **Buffer Protocol**: NumPy integration via buffer protocol
- **GPU Memory**: GPU memory management coordinated with Python GC

### Type System

- **Automatic Conversion**: Python types to Rust types
- **DType Mapping**: NumPy dtypes to TenfloweRS dtypes
- **Shape Inference**: Automatic shape broadcasting

## Building and Distribution

### Building the Python Package

```bash
# Development build
maturin develop

# Release build
maturin build --release

# Build wheels for distribution
maturin build --release --compatibility manylinux2014
```

## Feature Flags

- `extension-module`: Build as Python extension module (required for maturin)
- `gpu`: GPU support for Python tensor operations
- `python-tests`: Include Python-side test utilities

## Integration Examples

### With scikit-learn

```python
from sklearn.preprocessing import StandardScaler
import tenflowers as tf

# Use sklearn for preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to TenfloweRS
X_tensor = tf.from_numpy(X_scaled)
```

### With Pandas

```python
import pandas as pd
import tenflowers as tf

# Load data with pandas
df = pd.read_csv("data.csv")

# Convert to tensor
features = tf.from_numpy(df[feature_cols].values)
labels = tf.from_numpy(df["target"].values)
```

### With Matplotlib

```python
import matplotlib.pyplot as plt
import tenflowers as tf

# Visualize tensor data
tensor = tf.randn([28, 28])
plt.imshow(tensor.numpy(), cmap='gray')
plt.show()
```

## Extending the FFI

### Adding New Functions

```rust
#[pyfunction]
fn custom_operation(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    // Implementation
}

#[pymodule]
fn tenflowers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(custom_operation, m)?)?;
    Ok(())
}
```

## License

Licensed under Apache-2.0
