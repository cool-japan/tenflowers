# FFI Error Mapping

Maps `tenflowers_core::TensorError` variants to `TenflowersError` variants and Python exceptions.

This table is the single authoritative reference for the error taxonomy crossing the
Rust → Python FFI boundary. The mapping is implemented in
`crates/tenflowers-ffi/src/error_mapping.rs` by `TenflowersError::from_core_error`.

## Version

Applies to TenfloweRS **≥ 0.1.1**. Earlier versions used `Display` stringification only.

## Complete Mapping Table

| # | `TensorError` Variant | `TenflowersError` Variant | Python Exception | Rationale |
|---|----------------------|---------------------------|-----------------|-----------|
| 1 | `ShapeMismatch { operation, expected, got }` | `ShapeMismatch { operation, expected, got }` | `ShapeError(ValueError)` | Structural fields preserved 1-to-1 |
| 2 | `InvalidShape { operation, reason, shape }` | `ShapeMismatch { operation, expected="valid shape", got="<shape>: <reason>" }` | `ShapeError(ValueError)` | Normalised into shape-mismatch taxonomy |
| 3 | `InvalidAxis { operation, axis, ndim }` | `ShapeMismatch { operation, expected="axis < ndim", got=axis }` | `ShapeError(ValueError)` | Axis violation is a shape contract violation |
| 4 | `DeviceMismatch { operation, device1, device2 }` | `DevicePlacement { operation, details="device1 vs device2" }` | `DeviceError(RuntimeError)` | Device mismatch is a placement issue |
| 5 | `UnsupportedDevice { operation, device }` | `DevicePlacement { operation, details="unsupported: <device>" }` | `DeviceError(RuntimeError)` | Device not available at all |
| 6 | `GpuError { operation, details }` *(cfg gpu)* | `DevicePlacement { operation, details }` | `DeviceError(RuntimeError)` | GPU driver / API failure |
| 7 | `DeviceError { operation, details }` | `DevicePlacement { operation, details }` | `DeviceError(RuntimeError)` | Generic device failure |
| 8 | `GradientNotEnabled { operation }` | `GradientComputation { operation, details="gradient not enabled" }` | `GradientError(RuntimeError)` | Autograd tape must be active |
| 9 | `NumericalError { operation, details }` | `NumericalInstability { operation, details }` | `NumericalError(RuntimeError)` | NaN / Inf / overflow detected |
| 10 | `AllocationError { operation, details }` | `MemoryAllocation { operation, details }` | `MemoryError(Exception)` | Host / device memory exhausted |
| 11 | `ResourceExhausted { operation, resource }` | `MemoryAllocation { operation, details=resource }` | `MemoryError(Exception)` | Generic resource ceiling hit |
| 12 | `InvalidArgument { operation, reason }` | `InvalidOperation { operation, details=reason }` | `TensorOpError(RuntimeError)` | Bad parameter to an op |
| 13 | `UnsupportedOperation { operation, reason }` | `InvalidOperation { operation, details=reason }` | `TensorOpError(RuntimeError)` | Op not available |
| 14 | `ComputeError { operation, details }` | `InvalidOperation { operation, details }` | `TensorOpError(RuntimeError)` | Numeric compute failure |
| 15 | `BlasError { operation, details }` *(cfg blas)* | `InvalidOperation { operation, details }` | `TensorOpError(RuntimeError)` | BLAS/LAPACK routine failed |
| 16 | `InvalidOperation { operation, reason }` | `InvalidOperation { operation, details=reason }` | `TensorOpError(RuntimeError)` | Semantically invalid op |
| 17 | `SerializationError { operation, details }` | `Serialization { operation, details }` | `SerializationError(RuntimeError)` | Save / encode failed |
| 18 | `IoError { operation, details }` | `DataLoad { operation, details }` | `DataLoadError(RuntimeError)` | File / network I/O failure |
| 19 | `NotImplemented { operation, details }` | `NotImplemented { operation, details }` | `PyNotImplementedError` | Feature not yet built |
| 20 | `BenchmarkError { operation, details }` | `Generic { operation, details }` | `PyRuntimeError` | Benchmark instrumentation failure |
| 21 | `Timeout { operation, duration_ms }` | `Generic { operation, details="timeout after <N>ms" }` | `PyRuntimeError` | Op exceeded time budget |
| 22 | `CacheError { operation, details }` | `Generic { operation, details }` | `PyRuntimeError` | Internal cache management error |
| 23 | `Other { operation, details }` | `Generic { operation, details }` | `PyRuntimeError` | Catch-all variant |

## Feature-Gated Variants

Two `TensorError` variants only exist when the corresponding Cargo feature is enabled:

- `GpuError` — requires `--features gpu`
- `BlasError` — requires `--features blas`

The match arms in `from_core_error` are guarded with `#[cfg(feature = "gpu")]` and
`#[cfg(feature = "blas")]` respectively. When those features are disabled the variants
simply do not exist in the enum and no arm is needed.

## Python Exception Hierarchy

```
BaseException
└── Exception
    ├── RuntimeError
    │   ├── DeviceError          # tenflowers.DeviceError
    │   ├── GradientError        # tenflowers.GradientError
    │   ├── NumericalError       # tenflowers.NumericalError
    │   ├── TensorOpError        # tenflowers.TensorOpError
    │   ├── OptimizerError       # tenflowers.OptimizerError
    │   ├── SerializationError   # tenflowers.SerializationError
    │   ├── DataLoadError        # tenflowers.DataLoadError
    │   ├── GraphCompileError    # tenflowers.GraphCompileError
    │   └── CheckpointError      # tenflowers.CheckpointError
    ├── ValueError
    │   ├── ShapeError           # tenflowers.ShapeError
    │   └── LayerConfigError     # tenflowers.LayerConfigError
    ├── IndexError               # (standard) for InvalidDimension / IndexOutOfBounds
    ├── TypeError                # (standard) for TypeConversion
    ├── NotImplementedError      # (standard) for NotImplemented
    └── MemoryError (Exception)  # tenflowers.MemoryError
```

## Usage in Python

```python
import tenflowers as tf

try:
    a = tf.zeros([2, 3])
    b = tf.zeros([4, 5])
    c = tf.matmul(a, b)
except tf.ShapeError as e:
    print(f"Shape mismatch: {e}")
except tf.DeviceError as e:
    print(f"Device error: {e}")
except tf.TensorOpError as e:
    print(f"Operation failed: {e}")
except Exception as e:
    print(f"Unexpected: {e}")
```

## Design Principles

1. **Structural preservation**: Variant fields (`operation`, `expected`, `got`, etc.) are
   passed through rather than stringified, enabling programmatic inspection.
2. **Specificity**: The most specific exception class is used so callers can write
   fine-grained `except` clauses.
3. **Feature gating**: `#[cfg]` guards mirror the core crate's gates exactly.
4. **Exhaustiveness**: The `match` in `from_core_error` is non-exhaustive-free — if
   `TensorError` gains a new variant, the build fails, which is the desired behaviour.
