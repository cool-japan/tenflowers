# TenfloweRS FFI Scripts

This directory contains utility scripts for the TenfloweRS FFI layer.

## C Header Generator

**Script**: `generate_c_header.py`

Generates C header files from the Rust FFI bindings, making it easier to use TenfloweRS from C/C++ code.

### Usage

```bash
# Generate header with default name (tenflowers.h)
python scripts/generate_c_header.py

# Generate header with custom name
python scripts/generate_c_header.py --output include/tenflowers.h
```

### Features

- Generates complete C header with:
  - Version information
  - Opaque type definitions
  - Result/error types
  - Data type enumerations
  - Device type enumerations
  - Tensor creation functions
  - Tensor operation functions
  - Gradient tape functions
  - Memory management functions
  - Error handling functions
  - Device management functions
  - Utility functions

### Example C Usage

```c
#include "tenflowers.h"
#include <stdio.h>

int main() {
    // Create a tensor
    TenflowersTensor* tensor = NULL;
    size_t shape[] = {3, 3};

    TenflowersResult result = tenflowers_tensor_zeros(
        shape, 2,
        TENFLOWERS_DTYPE_FLOAT32,
        TENFLOWERS_DEVICE_CPU,
        &tensor
    );

    if (result != TENFLOWERS_OK) {
        fprintf(stderr, "Failed to create tensor\n");
        return 1;
    }

    // Use the tensor...

    // Clean up
    tenflowers_tensor_free(tensor);

    return 0;
}
```

## Future Scripts

Additional scripts will be added for:
- Gradient parity testing (Python vs Rust)
- Benchmark generation
- API documentation generation
- Performance profiling
