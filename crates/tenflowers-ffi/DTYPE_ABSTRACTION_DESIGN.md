# Dtype Abstraction Design for TenfloweRS FFI

**Version:** 0.1.0-alpha.2
**Status:** Design Document
**Date:** 2025-11-10

---

## Overview

This document outlines the design for comprehensive dtype (data type) abstraction in TenfloweRS FFI, enabling support for multiple data types including half-precision formats (f16, bf16) and integer types.

## Current State

### Supported Dtypes
- **f32** (float32): Primary supported type
- **Limited**: f64, i32, i64, u8, bool (partial support)

### Limitations
- No half-precision support (f16, bf16)
- Inconsistent dtype handling across operations
- Limited dtype promotion/casting logic
- No explicit dtype specification in most Python APIs

## Design Goals

1. **Comprehensive Type Support**: Support all common ML dtypes
2. **Explicit Control**: Users can explicitly specify and control dtypes
3. **Automatic Promotion**: Intelligent automatic dtype promotion when needed
4. **Performance**: Efficient dtype conversion and operations
5. **Memory Efficiency**: Support for reduced precision (f16, bf16)
6. **GPU Optimization**: Leverage hardware-accelerated half-precision ops

## Proposed Dtype Hierarchy

### Core Data Types

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    // Floating point types
    Float16,      // IEEE 754 half precision (f16)
    BFloat16,     // Brain float 16 (bf16)
    Float32,      // IEEE 754 single precision (f32)
    Float64,      // IEEE 754 double precision (f64)

    // Integer types
    Int8,         // Signed 8-bit integer
    Int16,        // Signed 16-bit integer
    Int32,        // Signed 32-bit integer
    Int64,        // Signed 64-bit integer

    // Unsigned integer types
    UInt8,        // Unsigned 8-bit integer
    UInt16,       // Unsigned 16-bit integer
    UInt32,       // Unsigned 32-bit integer
    UInt64,       // Unsigned 64-bit integer

    // Special types
    Bool,         // Boolean
    Complex64,    // Complex float32 (future)
    Complex128,   // Complex float64 (future)
}
```

### Python Dtype Mapping

```python
# Python API
import tenflowers as tf

# Dtype constants
tf.float16    # or tf.half
tf.bfloat16   # or tf.bf16
tf.float32    # or tf.float
tf.float64    # or tf.double
tf.int8
tf.int16
tf.int32      # or tf.int
tf.int64      # or tf.long
tf.uint8
tf.bool

# NumPy compatibility
tf.from_numpy(np_array)  # Auto-detect dtype
tf.from_numpy(np_array, dtype=tf.float16)  # Explicit dtype

# Explicit tensor creation
x = tf.zeros([2, 3], dtype=tf.float16)
y = tf.ones([2, 3], dtype=tf.bfloat16)
z = tf.rand([2, 3], dtype=tf.float32)
```

## Implementation Plan

### Phase 1: Core Infrastructure (High Priority)

#### 1.1 Dtype Trait and Type System

```rust
// src/dtype.rs

/// Trait for dtype support
pub trait DTypeSupport: Sized + Copy + Default {
    fn dtype() -> DType;
    fn from_f64(val: f64) -> Self;
    fn to_f64(self) -> f64;
    fn is_floating_point() -> bool;
    fn is_integer() -> bool;
}

impl DTypeSupport for f32 {
    fn dtype() -> DType { DType::Float32 }
    fn from_f64(val: f64) -> Self { val as f32 }
    fn to_f64(self) -> f64 { self as f64 }
    fn is_floating_point() -> bool { true }
    fn is_integer() -> bool { false }
}

// Similar implementations for f64, i32, i64, etc.
```

#### 1.2 Half-Precision Support

```rust
// Use half crate for f16 and bf16
use half::{f16, bf16};

impl DTypeSupport for f16 {
    fn dtype() -> DType { DType::Float16 }
    fn from_f64(val: f64) -> Self { f16::from_f64(val) }
    fn to_f64(self) -> f64 { self.to_f64() }
    fn is_floating_point() -> bool { true }
    fn is_integer() -> bool { false }
}

impl DTypeSupport for bf16 {
    fn dtype() -> DType { DType::BFloat16 }
    fn from_f64(val: f64) -> Self { bf16::from_f64(val) }
    fn to_f64(self) -> f64 { self.to_f64() }
    fn is_floating_point() -> bool { true }
    fn is_integer() -> bool { false }
}
```

#### 1.3 Python Dtype Bindings

```rust
// src/dtype_bindings.rs

#[pyclass]
#[derive(Clone, Copy)]
pub struct PyDType {
    inner: DType,
}

#[pymethods]
impl PyDType {
    #[getter]
    fn name(&self) -> &str {
        match self.inner {
            DType::Float16 => "float16",
            DType::BFloat16 => "bfloat16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::Bool => "bool",
            // ...
        }
    }

    fn __repr__(&self) -> String {
        format!("dtype('{}')", self.name())
    }

    fn __eq__(&self, other: &PyDType) -> bool {
        self.inner == other.inner
    }
}

// Export dtype constants to Python
#[pymodule]
fn tenflowers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("float16", PyDType { inner: DType::Float16 })?;
    m.add("bfloat16", PyDType { inner: DType::BFloat16 })?;
    m.add("float32", PyDType { inner: DType::Float32 })?;
    m.add("float64", PyDType { inner: DType::Float64 })?;
    // ...
    Ok(())
}
```

### Phase 2: Dtype Promotion and Casting

#### 2.1 Promotion Rules

```rust
// src/dtype_promotion.rs

/// Determine result dtype for binary operations
pub fn promote_dtypes(lhs: DType, rhs: DType) -> DType {
    use DType::*;

    match (lhs, rhs) {
        // Same dtype -> same dtype
        (a, b) if a == b => a,

        // Float promotion hierarchy
        (Float64, _) | (_, Float64) => Float64,
        (Float32, _) | (_, Float32) => Float32,
        (BFloat16, Float16) | (Float16, BFloat16) => Float32,
        (BFloat16, _) | (_, BFloat16) => BFloat16,
        (Float16, _) | (_, Float16) => Float16,

        // Integer promotion
        (Int64, _) | (_, Int64) => Int64,
        (Int32, _) | (_, Int32) => Int32,
        (Int16, _) | (_, Int16) => Int16,
        (Int8, _) | (_, Int8) => Int8,

        // Mixed float-int -> float
        (f, i) if f.is_floating_point() && i.is_integer() => f,
        (i, f) if i.is_integer() && f.is_floating_point() => f,

        // Default: promote to float32
        _ => Float32,
    }
}

/// Cast tensor to different dtype
pub fn cast_dtype<T, U>(tensor: &Tensor<T>, target_dtype: DType) -> Result<Tensor<U>>
where
    T: DTypeSupport,
    U: DTypeSupport,
{
    // Implementation
}
```

#### 2.2 Automatic Promotion in Operations

```rust
// Update tensor operations to support dtype promotion

impl Tensor<f32> {
    pub fn add_promoted<T>(&self, other: &Tensor<T>) -> Result<Tensor<f32>>
    where
        T: DTypeSupport,
    {
        let promoted_dtype = promote_dtypes(self.dtype(), other.dtype());
        // Cast both tensors to promoted dtype and perform operation
    }
}
```

### Phase 3: API Enhancement

#### 3.1 Explicit Dtype Specification

```python
# All tensor creation functions accept dtype parameter

import tenflowers as tf

# Explicit dtype
x = tf.zeros([2, 3], dtype=tf.float16)
y = tf.ones([2, 3], dtype=tf.bfloat16)
z = tf.rand([2, 3], dtype=tf.float32)

# Default dtype (can be configured)
a = tf.zeros([2, 3])  # Uses default dtype (float32)

# Set default dtype
tf.set_default_dtype(tf.float16)
b = tf.zeros([2, 3])  # Now uses float16 by default
```

#### 3.2 Dtype Conversion Methods

```python
# Tensor dtype conversion

x = tf.ones([2, 3], dtype=tf.float32)

# Explicit conversion
y = x.to(dtype=tf.float16)
z = x.half()       # Shorthand for to(tf.float16)
w = x.bfloat16()   # Shorthand for to(tf.bfloat16)
a = x.float()      # Shorthand for to(tf.float32)
b = x.double()     # Shorthand for to(tf.float64)

# In-place conversion (if supported)
x.to_(dtype=tf.float16)
```

### Phase 4: Performance Optimization

#### 4.1 GPU Half-Precision Kernels

```rust
// Leverage GPU tensor cores for f16/bf16

#[cfg(feature = "gpu")]
impl GpuOps for Tensor<f16> {
    fn matmul_gpu(&self, other: &Tensor<f16>) -> Result<Tensor<f16>> {
        // Use tensor cores for accelerated half-precision matmul
        // Potential 2-8x speedup on modern GPUs
    }
}
```

#### 4.2 Mixed Precision Training Support

```python
# Automatic Mixed Precision (AMP)

import tenflowers as tf
from tenflowers.amp import autocast, GradScaler

model = MyModel()
optimizer = tf.Adam(model.parameters())
scaler = GradScaler()

for epoch in range(epochs):
    for batch in dataloader:
        with autocast():  # Automatically uses float16 where beneficial
            output = model(batch)
            loss = criterion(output, targets)

        # Scale loss and gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

## Feature Gating

### Cargo Features

```toml
[features]
default = ["f32"]

# Basic dtypes
f32 = []
f64 = []
i32 = []
i64 = []

# Half-precision (requires half crate)
f16 = ["half"]
bf16 = ["half"]

# All dtypes
all-dtypes = ["f32", "f64", "i32", "i64", "f16", "bf16"]

[dependencies]
half = { version = "2.4", optional = true }
```

### Runtime Feature Detection

```rust
// Check if dtype is supported at runtime

pub fn is_dtype_supported(dtype: DType) -> bool {
    match dtype {
        DType::Float32 => true,
        DType::Float64 => cfg!(feature = "f64"),
        DType::Float16 => cfg!(feature = "f16"),
        DType::BFloat16 => cfg!(feature = "bf16"),
        // ...
    }
}
```

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_promotion() {
        assert_eq!(
            promote_dtypes(DType::Float32, DType::Int32),
            DType::Float32
        );
        assert_eq!(
            promote_dtypes(DType::Float16, DType::BFloat16),
            DType::Float32
        );
    }

    #[test]
    #[cfg(feature = "f16")]
    fn test_f16_operations() {
        let a = Tensor::<f16>::zeros(&[2, 2]);
        let b = Tensor::<f16>::ones(&[2, 2]);
        let c = a.add(&b).unwrap();
        assert_eq!(c.dtype(), DType::Float16);
    }
}
```

### Python Integration Tests

```python
def test_dtype_specification():
    """Test explicit dtype specification."""
    import tenflowers as tf

    x = tf.zeros([2, 3], dtype=tf.float16)
    assert x.dtype == tf.float16

    y = tf.ones([2, 3], dtype=tf.bfloat16)
    assert y.dtype == tf.bfloat16


def test_dtype_promotion():
    """Test automatic dtype promotion."""
    import tenflowers as tf

    x = tf.ones([2, 2], dtype=tf.float16)
    y = tf.ones([2, 2], dtype=tf.float32)

    z = tf.add(x, y)
    assert z.dtype == tf.float32  # Promoted to float32


def test_dtype_conversion():
    """Test dtype conversion methods."""
    import tenflowers as tf

    x = tf.ones([2, 3], dtype=tf.float32)

    y = x.half()
    assert y.dtype == tf.float16

    z = x.bfloat16()
    assert z.dtype == tf.bfloat16
```

## Migration Path

### Backward Compatibility

1. **Default behavior unchanged**: All existing code continues to work with f32
2. **Gradual adoption**: Users can opt-in to new dtypes as needed
3. **Feature flags**: f16/bf16 behind feature flags initially
4. **Deprecation warnings**: Clear migration path for any breaking changes

### Implementation Phases

**Phase 1 (Alpha.3):**
- Core dtype infrastructure
- f16 and bf16 support (feature-gated)
- Basic dtype promotion
- Python dtype constants

**Phase 2 (Beta.1):**
- Comprehensive dtype promotion rules
- Dtype conversion methods
- NumPy dtype compatibility
- Extended test coverage

**Phase 3 (Beta.2):**
- GPU half-precision kernels
- Mixed precision training support
- Performance optimizations
- Production-ready dtype system

## Performance Considerations

### Memory Savings

- **f16**: 50% memory reduction vs f32
- **bf16**: 50% memory reduction vs f32
- **Enables larger models**: 2x model size on same hardware

### Computational Performance

- **GPU Tensor Cores**: 2-8x speedup for f16/bf16 matmul on modern GPUs
- **CPU SIMD**: Potential speedups with hardware f16 support
- **Bandwidth**: Reduced memory bandwidth requirements

### Tradeoffs

- **Precision loss**: Reduced numeric precision (manage with careful algorithm design)
- **Range limitations**: f16 has limited range (±65504)
- **Conversion overhead**: Cost of dtype conversions

## Open Questions

1. **Default dtype**: Should we allow changing default dtype globally?
2. **Implicit conversion**: How aggressive should automatic promotion be?
3. **Integer operations**: Full support for integer dtypes or limited?
4. **Complex numbers**: Timeline for complex dtype support?

## References

- PyTorch dtype system: https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype
- TensorFlow dtypes: https://www.tensorflow.org/api_docs/python/tf/dtypes
- Half crate: https://docs.rs/half/
- Mixed precision training: https://arxiv.org/abs/1710.03740

---

**Status**: Design document complete, ready for implementation in phases.
