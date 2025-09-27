# TenfloweRS FFI

Foreign Function Interface for TenfloweRS, providing Python bindings and C API for seamless integration with other languages and frameworks.

> Alpha Notice (0.1.0-alpha.1 · 2025-09-27)
> Python bindings are in-progress: the API surface shown below reflects intended design; many functions still map to provisional Rust implementations. Wheels are not yet published—build from source via maturin.

## Overview

`tenflowers-ffi` implements:
- **Python Bindings**: Complete PyO3-based Python API
- **C API**: Stable C interface for maximum compatibility
- **NumPy Integration**: Zero-copy tensor conversion
- **ONNX Support**: Model import/export capabilities
- **Language Bindings**: Foundation for Julia, R, and other languages
- **Backwards Compatibility**: Stable API with versioning

## Features

- **Zero-Copy Interop**: Efficient data exchange with Python/NumPy
- **Pythonic API**: Familiar interface for Python users
- **Type Safety**: Automatic type conversions with safety checks
- **Error Handling**: Proper exception propagation
- **GPU Support**: Tensor operations on GPU from Python
- **Async Support**: Python async/await integration

## Python API Usage

### Installation

```bash
# Install from PyPI
pip install tenflowers

# Or build from source
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

```python
import tenflowers as tf

# Enable gradient tracking
x = tf.tensor([2.0], requires_grad=True)
y = tf.tensor([3.0], requires_grad=True)

# Compute function
z = x * x + y * y

# Compute gradients
z.backward()
print(f"dz/dx = {x.grad}")  # 4.0
print(f"dz/dy = {y.grad}")  # 6.0

# Higher-order derivatives
x = tf.tensor([1.0], requires_grad=True)
y = x ** 3
grad = tf.autograd.grad(y, x, create_graph=True)[0]
grad2 = tf.autograd.grad(grad, x)[0]
print(f"d²y/dx² = {grad2}")  # 6.0
```

### Dataset and DataLoader

```python
import tenflowers as tf
from tenflowers.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create dataset and loader
dataset = MyDataset(train_data, train_labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Iterate through batches
for batch_data, batch_labels in dataloader:
    # Process batch
    pass
```

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

### Error Handling

```c
TF_Status* status = tf_new_status();
TF_Tensor* result = tf_matmul(a, b, status);

if (tf_get_status_code(status) != TF_OK) {
    printf("Error: %s\n", tf_get_status_message(status));
}

tf_delete_status(status);
```

## Architecture

### Python Bindings Structure

- **Core Module**: Tensor operations and basic functionality
- **NN Module**: Neural network layers and utilities
- **Optim Module**: Optimization algorithms
- **Autograd Module**: Automatic differentiation
- **Utils Module**: Data loading, metrics, etc.

### Memory Management

- **Reference Counting**: Automatic memory management in Python
- **Buffer Protocol**: Zero-copy NumPy integration
- **GPU Memory**: CUDA memory management with Python GC

### Type System

- **Automatic Conversion**: Python types to Rust types
- **DType Mapping**: NumPy dtypes to TenfloweRS dtypes
- **Shape Inference**: Automatic shape broadcasting

## Performance Considerations

### Python Overhead

- Minimize Python/Rust boundary crossings
- Use batch operations instead of loops
- Leverage NumPy for preprocessing when possible

### Memory Efficiency

```python
# Good: In-place operations
tensor.add_(1.0)  # In-place add

# Good: View operations
view = tensor.view(new_shape)  # No copy

# Good: Batch operations
result = tf.stack([t1, t2, t3])  # Single operation

# Bad: Python loops
result = []
for t in tensors:
    result.append(t + 1)  # Many small operations
```

### GPU Best Practices

```python
# Keep operations on same device
gpu_tensor = tensor.to("cuda:0")
# All subsequent operations stay on GPU
result = gpu_tensor @ gpu_tensor.T

# Batch transfers
tensors = [t.to("cuda:0") for t in cpu_tensors]  # Bad
gpu_batch = tf.stack(cpu_tensors).to("cuda:0")   # Good
```

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

### Custom Types

```rust
#[pyclass]
struct CustomLayer {
    // fields
}

#[pymethods]
impl CustomLayer {
    #[new]
    fn new(args: Args) -> Self {
        // Constructor
    }
    
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // Forward pass
    }
}
```

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

### Creating Conda Package

```yaml
# conda recipe/meta.yaml
package:
  name: tenflowers
  version: "0.1.0"

requirements:
  build:
    - rust
    - maturin
  run:
    - python
    - numpy
```

## Contributing

Priority areas for contribution:
- Expanding Python API coverage
- Improving NumPy compatibility
- Adding more language bindings
- Performance optimizations
- Documentation and examples

### Current Alpha Limitations
- Autograd coverage in Python layer incomplete (some ops lack grad)
- No distributed / multi-GPU orchestration exposed yet
- Mixed precision & checkpoint APIs not yet bound
- Error messages from Rust sometimes surface as generic Python exceptions
- Wheels / manylinux / macOS universal2 build pipeline pending CI setup

### Short-Term FFI Roadmap
1. Minimal gradient parity test matrix (Python vs Rust)
2. Exception mapping refactor for richer error classes
3. Initial ONNX load (inference) exposure
4. Packaging: GitHub Actions wheel build + auditwheel / delocate
5. Tutorial notebooks + docstring completeness audit

## License

Dual-licensed under MIT OR Apache-2.0