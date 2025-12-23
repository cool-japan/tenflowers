## TenfloweRS C API Documentation

This document describes the C API for TenfloweRS, providing a stable interface for C/C++ applications.

### Overview

The TenfloweRS C API provides:
- Tensor operations (creation, manipulation, computation)
- Automatic differentiation support
- Neural network layers
- Optimizers
- Device management (CPU/GPU)
- Error handling

### Getting Started

#### Installation

```bash
# Build the library
cargo build --release --manifest-path crates/tenflowers-ffi/Cargo.toml

# Headers are in crates/tenflowers-ffi/include/
# Library is at target/release/libtenflowers.so (Linux)
#              target/release/libtenflowers.dylib (macOS)
#              target/release/tenflowers.dll (Windows)
```

#### Compiling Your C Code

```bash
# Linux
gcc -o myapp myapp.c -I/path/to/include -L/path/to/lib -ltenflowers -lm

# macOS
gcc -o myapp myapp.c -I/path/to/include -L/path/to/lib -ltenflowers

# Run
LD_LIBRARY_PATH=/path/to/lib ./myapp  # Linux
DYLD_LIBRARY_PATH=/path/to/lib ./myapp  # macOS
```

### Core Types

#### TF_Tensor

Represents a multi-dimensional array.

```c
typedef struct TF_Tensor TF_Tensor;
```

#### TF_Status

Represents operation status and error information.

```c
typedef struct TF_Status TF_Status;

// Status codes
typedef enum {
    TF_OK = 0,
    TF_ERROR = 1,
    TF_INVALID_ARGUMENT = 2,
    TF_OUT_OF_MEMORY = 3,
    TF_SHAPE_MISMATCH = 4,
    TF_DEVICE_ERROR = 5
} TF_StatusCode;
```

#### TF_DataType

Data types supported by tensors.

```c
typedef enum {
    TF_FLOAT32,
    TF_FLOAT64,
    TF_INT32,
    TF_INT64,
    TF_UINT8,
    TF_BOOL
} TF_DataType;
```

### Tensor Creation Functions

#### TF_Zeros

Create a tensor filled with zeros.

```c
TF_Tensor* TF_Zeros(
    const size_t* shape,
    size_t ndim,
    TF_DataType dtype,
    TF_Status* status
);
```

**Parameters:**
- `shape`: Array of dimensions
- `ndim`: Number of dimensions
- `dtype`: Data type
- `status`: Status object for error reporting

**Returns:** Newly created tensor, or NULL on error

**Example:**
```c
size_t shape[] = {3, 3};
TF_Tensor* t = TF_Zeros(shape, 2, TF_FLOAT32, status);
```

#### TF_Ones

Create a tensor filled with ones.

```c
TF_Tensor* TF_Ones(
    const size_t* shape,
    size_t ndim,
    TF_DataType dtype,
    TF_Status* status
);
```

#### TF_Rand

Create a tensor with random values (uniform distribution).

```c
TF_Tensor* TF_Rand(
    const size_t* shape,
    size_t ndim,
    TF_DataType dtype,
    TF_Status* status
);
```

#### TF_RandN

Create a tensor with random values (normal distribution).

```c
TF_Tensor* TF_RandN(
    const size_t* shape,
    size_t ndim,
    TF_DataType dtype,
    TF_Status* status
);
```

### Tensor Operations

#### TF_Add

Element-wise addition.

```c
TF_Tensor* TF_Add(
    const TF_Tensor* a,
    const TF_Tensor* b,
    TF_Status* status
);
```

#### TF_Mul

Element-wise multiplication.

```c
TF_Tensor* TF_Mul(
    const TF_Tensor* a,
    const TF_Tensor* b,
    TF_Status* status
);
```

#### TF_MatMul

Matrix multiplication.

```c
TF_Tensor* TF_MatMul(
    const TF_Tensor* a,
    const TF_Tensor* b,
    TF_Status* status
);
```

#### TF_Transpose

Transpose tensor.

```c
TF_Tensor* TF_Transpose(
    const TF_Tensor* tensor,
    const int* axes,
    size_t num_axes,
    TF_Status* status
);
```

#### TF_Reshape

Reshape tensor.

```c
TF_Tensor* TF_Reshape(
    const TF_Tensor* tensor,
    const size_t* new_shape,
    size_t ndim,
    TF_Status* status
);
```

### Tensor Inspection

#### TF_TensorData

Get pointer to tensor data.

```c
void* TF_TensorData(const TF_Tensor* tensor);
```

#### TF_TensorShape

Get tensor shape.

```c
void TF_TensorShape(
    const TF_Tensor* tensor,
    size_t* shape,
    size_t* ndim
);
```

#### TF_TensorNumDims

Get number of dimensions.

```c
size_t TF_TensorNumDims(const TF_Tensor* tensor);
```

#### TF_TensorDim

Get size of specific dimension.

```c
size_t TF_TensorDim(const TF_Tensor* tensor, size_t axis);
```

#### TF_TensorNumElements

Get total number of elements.

```c
size_t TF_TensorNumElements(const TF_Tensor* tensor);
```

### Status Management

#### TF_NewStatus

Create a new status object.

```c
TF_Status* TF_NewStatus(void);
```

#### TF_DeleteStatus

Delete a status object.

```c
void TF_DeleteStatus(TF_Status* status);
```

#### TF_GetStatusCode

Get status code.

```c
TF_StatusCode TF_GetStatusCode(const TF_Status* status);
```

#### TF_GetStatusMessage

Get error message.

```c
const char* TF_GetStatusMessage(const TF_Status* status);
```

### Memory Management

#### TF_DeleteTensor

Delete a tensor and free its memory.

```c
void TF_DeleteTensor(TF_Tensor* tensor);
```

**Important:** Always call this when done with a tensor to avoid memory leaks.

### Device Management

#### TF_SetDevice

Set default device for tensor operations.

```c
void TF_SetDevice(const char* device, TF_Status* status);
```

**Example:**
```c
TF_SetDevice("cpu", status);      // Use CPU
TF_SetDevice("gpu:0", status);    // Use first GPU
```

#### TF_GetDevice

Get current default device.

```c
const char* TF_GetDevice(void);
```

### Error Handling

Always check status after operations:

```c
TF_Status* status = TF_NewStatus();

TF_Tensor* result = TF_Add(a, b, status);

if (TF_GetStatusCode(status) != TF_OK) {
    fprintf(stderr, "Error: %s\n", TF_GetStatusMessage(status));
    // Handle error
}

TF_DeleteStatus(status);
```

### Complete Example

```c
#include "tenflowers.h"
#include <stdio.h>

int main() {
    // Create status object
    TF_Status* status = TF_NewStatus();

    // Create tensors
    size_t shape[] = {2, 3};
    TF_Tensor* a = TF_Ones(shape, 2, TF_FLOAT32, status);
    TF_Tensor* b = TF_Ones(shape, 2, TF_FLOAT32, status);

    // Perform operation
    TF_Tensor* result = TF_Add(a, b, status);

    if (TF_GetStatusCode(status) == TF_OK) {
        // Get result data
        float* data = (float*)TF_TensorData(result);

        // Print results
        printf("Result: ");
        for (size_t i = 0; i < 6; i++) {
            printf("%.1f ", data[i]);
        }
        printf("\n");
    } else {
        fprintf(stderr, "Error: %s\n", TF_GetStatusMessage(status));
    }

    // Cleanup
    TF_DeleteTensor(a);
    TF_DeleteTensor(b);
    TF_DeleteTensor(result);
    TF_DeleteStatus(status);

    return 0;
}
```

### Thread Safety

The C API is designed to be thread-safe for most operations:
- Tensors are immutable by default
- Each thread should have its own status object
- GPU operations are synchronized internally

### ABI Stability

The C API follows semantic versioning:
- **Major version**: Breaking changes to API/ABI
- **Minor version**: New features, backward compatible
- **Patch version**: Bug fixes, fully compatible

Current version: **0.1.0-alpha.2**

**Note:** Alpha versions may have breaking changes. Stable ABI will be guaranteed from version 1.0.0 onwards.

### Performance Tips

1. **Minimize copies**: Use reshape and transpose (views) instead of creating new tensors
2. **Batch operations**: Combine multiple operations when possible
3. **GPU usage**: Transfer data to GPU once, perform multiple operations, then transfer back
4. **Memory reuse**: Reuse tensors when possible instead of creating new ones

### Debugging

Enable debug logging:

```c
TF_SetLogLevel(TF_LOG_DEBUG);  // Enable debug logging
```

Check tensor properties:

```c
size_t ndim = TF_TensorNumDims(tensor);
size_t nelems = TF_TensorNumElements(tensor);

printf("Tensor: %zu dimensions, %zu elements\n", ndim, nelems);
```

### Further Reading

- [Python API Documentation](https://tenflowers.readthedocs.io)
- [Rust API Documentation](https://docs.rs/tenflowers)
- [Examples](../examples/)
- [GitHub Repository](https://github.com/cool-japan/tenflowers)
