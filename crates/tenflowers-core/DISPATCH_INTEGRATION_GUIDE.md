# Dispatch Registry Integration Guide

**Version:** 0.1.0-alpha.2
**Audience:** TenfloweRS Core Contributors
**Status:** Active Reference Document

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Registration Patterns](#registration-patterns)
5. [Kernel Implementation](#kernel-implementation)
6. [Best Practices](#best-practices)
7. [Testing](#testing)
8. [Common Pitfalls](#common-pitfalls)

## Overview

The TenfloweRS dispatch registry provides a unified system for registering and executing tensor operations across multiple backends (CPU, SIMD, GPU, BLAS, etc.). It eliminates per-module logic duplication and provides automatic backend selection based on device, dtype, and availability.

### Key Benefits

- **Unified Interface**: Single dispatch path for all operations
- **Automatic Backend Selection**: Choose optimal implementation at runtime
- **Feature Gating**: Conditional compilation for optional backends
- **Type Safety**: Type-specific registries prevent runtime errors
- **Extensibility**: Easy to add new operations and backends

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Operation Call                             │
│           (e.g., tensor.abs(), add(a, b))                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Dispatch Registry Query                         │
│         get_registry::<T>().dispatch_unary("abs", x)        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Backend Selection Logic                         │
│  1. Check device → preferred backend                        │
│  2. Filter available backends                               │
│  3. Select highest priority                                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Kernel Execution                                │
│         Backend-specific implementation                      │
│    (CPU, SIMD, GPU, BLAS, CUDA, Metal, etc.)               │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Basic Unary Operation Registration

```rust
use crate::dispatch_registry::{
    BackendType, KernelImplementation, OperationDescriptor, F32_REGISTRY
};
use crate::{DType, Tensor, Result};

// Step 1: Define CPU kernel
fn sqrt_f32_cpu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let data = x.data();
    let result: Vec<f32> = data.iter().map(|v| v.sqrt()).collect();
    let array = scirs2_autograd::ndarray::ArrayD::from_shape_vec(
        x.shape().dims(),
        result
    )?;
    Ok(Tensor::from_array(array))
}

// Step 2: Register operation
pub fn register_sqrt() {
    let desc = OperationDescriptor::new("sqrt", "unary")
        .with_dtypes(vec![DType::Float32])
        .with_broadcast();

    F32_REGISTRY.register_operation(desc).unwrap();

    // Step 3: Register CPU kernel
    F32_REGISTRY.register_kernel(
        "sqrt",
        KernelImplementation::unary(BackendType::Cpu, sqrt_f32_cpu)
    ).unwrap();
}

// Step 4: Use in operation
pub fn sqrt(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    F32_REGISTRY.dispatch_unary("sqrt", x)
}
```

### 2. Basic Binary Operation Registration

```rust
// Step 1: Define CPU kernel
fn add_f32_cpu(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.shape() != b.shape() {
        return Err(TensorError::shape_mismatch("add", a.shape(), b.shape()));
    }
    let a_data = a.data();
    let b_data = b.data();
    let result: Vec<f32> = a_data.iter()
        .zip(b_data.iter())
        .map(|(x, y)| x + y)
        .collect();
    let array = scirs2_autograd::ndarray::ArrayD::from_shape_vec(
        a.shape().dims(),
        result
    )?;
    Ok(Tensor::from_array(array))
}

// Step 2: Register operation
pub fn register_add() {
    let desc = OperationDescriptor::new("add", "binary")
        .with_dtypes(vec![DType::Float32])
        .with_broadcast();

    F32_REGISTRY.register_operation(desc).unwrap();

    // Step 3: Register CPU kernel
    F32_REGISTRY.register_kernel(
        "add",
        KernelImplementation::binary(BackendType::Cpu, add_f32_cpu)
    ).unwrap();
}

// Step 4: Use in operation
pub fn add(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    F32_REGISTRY.dispatch_binary("add", a, b)
}
```

## Architecture

### Type-Specific Registries

TenfloweRS uses type-specific registries to ensure type safety:

```rust
// Global registries (lazy_static)
pub static ref F32_REGISTRY: DispatchRegistry<f32>;
pub static ref F64_REGISTRY: DispatchRegistry<f64>;
pub static ref I32_REGISTRY: DispatchRegistry<i32>;

// Access via helper function
let registry = get_registry::<f32>().unwrap();
```

### Backend Types

```rust
pub enum BackendType {
    Cpu,                    // Always available
    #[cfg(feature = "simd")]
    SimdCpu,                // SIMD-optimized CPU
    #[cfg(feature = "blas")]
    Blas,                   // BLAS library
    #[cfg(feature = "gpu")]
    Gpu,                    // WebGPU
    #[cfg(feature = "cuda")]
    Cuda,                   // NVIDIA CUDA
    #[cfg(feature = "metal")]
    Metal,                  // Apple Metal
    #[cfg(feature = "rocm")]
    Rocm,                   // AMD ROCm
}
```

### Backend Priority

Backends are selected by priority (higher = preferred):

| Backend | Priority | Typical Use Case |
|---------|----------|------------------|
| Cpu     | 0        | Fallback, small tensors |
| SimdCpu | 10       | Medium tensors, CPU-only |
| Blas    | 20       | Linear algebra on CPU |
| Gpu     | 30       | General GPU (WebGPU) |
| Cuda    | 40       | NVIDIA GPUs |
| Rocm    | 40       | AMD GPUs |
| Metal   | 50       | Apple Silicon |

## Registration Patterns

### Pattern 1: Multi-Backend Registration

Register an operation for multiple backends:

```rust
pub fn register_matmul() {
    let desc = OperationDescriptor::new("matmul", "linalg")
        .with_dtypes(vec![DType::Float32])
        .with_rank_range(Some(2), None); // At least 2D

    F32_REGISTRY.register_operation(desc).unwrap();

    // CPU implementation
    F32_REGISTRY.register_kernel(
        "matmul",
        KernelImplementation::binary(BackendType::Cpu, matmul_f32_cpu)
    ).unwrap();

    // BLAS implementation (if available)
    #[cfg(feature = "blas")]
    F32_REGISTRY.register_kernel(
        "matmul",
        KernelImplementation::binary(BackendType::Blas, matmul_f32_blas)
    ).unwrap();

    // GPU implementation (if available)
    #[cfg(feature = "gpu")]
    F32_REGISTRY.register_kernel(
        "matmul",
        KernelImplementation::binary(BackendType::Gpu, matmul_f32_gpu)
    ).unwrap();
}
```

### Pattern 2: Multi-Type Registration

Register an operation for multiple data types:

```rust
pub fn register_abs_all_types() {
    // F32
    {
        let desc = OperationDescriptor::new("abs", "unary")
            .with_dtypes(vec![DType::Float32]);
        F32_REGISTRY.register_operation(desc).unwrap();
        F32_REGISTRY.register_kernel(
            "abs",
            KernelImplementation::unary(BackendType::Cpu, abs_f32_cpu)
        ).unwrap();
    }

    // F64
    {
        let desc = OperationDescriptor::new("abs", "unary")
            .with_dtypes(vec![DType::Float64]);
        F64_REGISTRY.register_operation(desc).unwrap();
        F64_REGISTRY.register_kernel(
            "abs",
            KernelImplementation::unary(BackendType::Cpu, abs_f64_cpu)
        ).unwrap();
    }

    // I32
    {
        let desc = OperationDescriptor::new("abs", "unary")
            .with_dtypes(vec![DType::Int32]);
        I32_REGISTRY.register_operation(desc).unwrap();
        I32_REGISTRY.register_kernel(
            "abs",
            KernelImplementation::unary(BackendType::Cpu, abs_i32_cpu)
        ).unwrap();
    }
}
```

### Pattern 3: Lazy Registration with Macros

Use macros to simplify registration:

```rust
// In your module
pub fn init_operations() {
    register_operation!(F32_REGISTRY, "abs", "unary",
        dtypes: [DType::Float32]);

    register_unary_kernel!(F32_REGISTRY, "abs",
        BackendType::Cpu, abs_f32_cpu);

    #[cfg(feature = "simd")]
    register_unary_kernel!(F32_REGISTRY, "abs",
        BackendType::SimdCpu, abs_f32_simd);
}
```

## Kernel Implementation

### CPU Kernels

CPU kernels should be simple, correct, and serve as reference implementations:

```rust
fn operation_cpu(inputs...) -> Result<Tensor<T>> {
    // 1. Validate inputs (shapes, constraints)
    // 2. Extract data
    // 3. Perform computation
    // 4. Package result

    let data = input.data();
    let result: Vec<T> = data.iter()
        .map(|v| /* operation */)
        .collect();

    let array = ArrayD::from_shape_vec(input.shape().dims(), result)?;
    Ok(Tensor::from_array(array))
}
```

### SIMD Kernels

SIMD kernels should use scirs2_core SIMD abstractions:

```rust
#[cfg(feature = "simd")]
fn operation_simd(input: &Tensor<f32>) -> Result<Tensor<f32>> {
    use scirs2_core::simd::{SimdArray, SimdOps};

    // Use SIMD operations from scirs2_core
    // If not available, fallback to CPU

    operation_cpu(input) // Fallback for now
}
```

### GPU Kernels

GPU kernels should use WebGPU compute shaders:

```rust
#[cfg(feature = "gpu")]
fn operation_gpu(input: &Tensor<f32>) -> Result<Tensor<f32>> {
    use crate::gpu::{GpuContext, execute_kernel};

    // 1. Get GPU context
    let context = GpuContext::get_or_create()?;

    // 2. Create shader if needed
    let shader = context.get_or_create_shader("operation", SHADER_SOURCE)?;

    // 3. Execute kernel
    execute_kernel(&context, &shader, input)
}
```

### BLAS Kernels

BLAS kernels should leverage optimized libraries:

```rust
#[cfg(feature = "blas")]
fn matmul_blas(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    use crate::ops::lapack::matmul_blas;

    // Delegate to BLAS implementation
    matmul_blas(a, b)
}
```

## Best Practices

### 1. Always Provide CPU Fallback

Every operation MUST have a CPU implementation:

```rust
✅ GOOD:
F32_REGISTRY.register_kernel("op",
    KernelImplementation::unary(BackendType::Cpu, op_cpu)).unwrap();
#[cfg(feature = "gpu")]
F32_REGISTRY.register_kernel("op",
    KernelImplementation::unary(BackendType::Gpu, op_gpu)).unwrap();

❌ BAD:
#[cfg(feature = "gpu")]
F32_REGISTRY.register_kernel("op",
    KernelImplementation::unary(BackendType::Gpu, op_gpu)).unwrap();
// No CPU fallback!
```

### 2. Use Shape Error Taxonomy

Use standardized error messages:

```rust
use crate::shape_error_taxonomy::{ShapeErrorBuilder, ShapeErrorCategory};

✅ GOOD:
return Err(ShapeErrorBuilder::new("matmul", ShapeErrorCategory::MatMulIncompatible)
    .expected(&format!("(..., m, k) and (..., k, n)"))
    .got(&format!("({:?}) and ({:?})", a.shape(), b.shape()))
    .detail(&format!("Inner dimensions must match: {} != {}", k1, k2))
    .build());

❌ BAD:
return Err(TensorError::invalid_argument("matmul shapes don't match"));
```

### 3. Validate Inputs Early

Check preconditions before computation:

```rust
fn operation(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    // ✅ Validate early
    if a.shape() != b.shape() {
        return Err(ShapeErrorBuilder::new("op",
            ShapeErrorCategory::ElementwiseMismatch)
            .expected(&format!("{:?}", a.shape()))
            .got(&format!("{:?}", b.shape()))
            .build());
    }

    // Now dispatch
    F32_REGISTRY.dispatch_binary("op", a, b)
}
```

### 4. Register at Module Initialization

Register operations when the module loads:

```rust
// In ops/mod.rs or specific operation module
pub fn register_all_operations() {
    register_unary_ops();
    register_binary_ops();
    register_reduction_ops();
    // ...
}

// Call this in lib.rs or at first use
lazy_static! {
    static ref INIT: () = {
        crate::ops::register_all_operations();
    };
}
```

### 5. Document Operation Constraints

Use OperationDescriptor to document constraints:

```rust
let desc = OperationDescriptor::new("conv2d", "nn")
    .with_dtypes(vec![DType::Float32, DType::Float64])
    .with_rank_range(Some(4), Some(4))  // Exactly 4D
    .with_broadcast()
    .with_inplace();  // If in-place is possible
```

## Testing

### Unit Tests for Kernels

Test each kernel implementation:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_abs_f32_cpu() {
        let input = Tensor::from_array(
            array![-1.0f32, 2.0, -3.0].into_dyn()
        );
        let result = abs_f32_cpu(&input).unwrap();
        assert_eq!(result.data(), &[1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn test_abs_dispatch() {
        crate::ops::register_all_operations();

        let input = Tensor::from_array(
            array![-1.0f32, 2.0, -3.0].into_dyn()
        );
        let result = F32_REGISTRY.dispatch_unary("abs", &input).unwrap();
        assert_eq!(result.data(), &[1.0f32, 2.0, 3.0]);
    }
}
```

### Cross-Backend Consistency Tests

Ensure all backends produce identical results:

```rust
#[test]
#[cfg(all(feature = "simd", feature = "gpu"))]
fn test_cross_backend_consistency() {
    let input = Tensor::from_array(/* ... */);

    let cpu_result = F32_REGISTRY.dispatch_unary_on_backend(
        "op", &input, BackendType::Cpu
    ).unwrap();

    let simd_result = F32_REGISTRY.dispatch_unary_on_backend(
        "op", &input, BackendType::SimdCpu
    ).unwrap();

    let gpu_result = F32_REGISTRY.dispatch_unary_on_backend(
        "op", &input, BackendType::Gpu
    ).unwrap();

    // Allow small numerical differences
    assert_tensors_close(&cpu_result, &simd_result, 1e-6);
    assert_tensors_close(&cpu_result, &gpu_result, 1e-5);
}
```

### Performance Tests

Benchmark different backends:

```rust
#[bench]
fn bench_abs_cpu(b: &mut Bencher) {
    let input = Tensor::<f32>::randn(&[1000, 1000]);
    b.iter(|| {
        F32_REGISTRY.dispatch_unary_on_backend(
            "abs", &input, BackendType::Cpu
        )
    });
}
```

## Common Pitfalls

### Pitfall 1: Forgetting to Register

```rust
❌ BAD:
pub fn sqrt(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    F32_REGISTRY.dispatch_unary("sqrt", x)  // Not registered!
}

✅ GOOD:
lazy_static! {
    static ref INIT: () = { register_sqrt(); };
}

pub fn sqrt(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let _ = *INIT;  // Ensure registration
    F32_REGISTRY.dispatch_unary("sqrt", x)
}
```

### Pitfall 2: Wrong Registry for Type

```rust
❌ BAD:
pub fn abs(x: &Tensor<f64>) -> Result<Tensor<f64>> {
    F32_REGISTRY.dispatch_unary("abs", x)  // Wrong registry!
}

✅ GOOD:
pub fn abs(x: &Tensor<f64>) -> Result<Tensor<f64>> {
    F64_REGISTRY.dispatch_unary("abs", x)
}
```

### Pitfall 3: Duplicate Registration

```rust
❌ BAD:
register_operation!(F32_REGISTRY, "add", "binary");
register_operation!(F32_REGISTRY, "add", "binary");  // Error!

✅ GOOD:
register_operation!(F32_REGISTRY, "add", "binary");
// Only register once
```

### Pitfall 4: Missing Feature Gates

```rust
❌ BAD:
F32_REGISTRY.register_kernel("op",
    KernelImplementation::unary(BackendType::Gpu, op_gpu)).unwrap();
// Compilation error if GPU feature not enabled!

✅ GOOD:
#[cfg(feature = "gpu")]
F32_REGISTRY.register_kernel("op",
    KernelImplementation::unary(BackendType::Gpu, op_gpu)).unwrap();
```

## Migration Checklist

When migrating an existing operation to use the dispatch registry:

- [ ] Create CPU kernel implementation
- [ ] Register operation with OperationDescriptor
- [ ] Register CPU kernel
- [ ] Add SIMD kernel (if applicable)
- [ ] Add GPU kernel (if applicable)
- [ ] Add BLAS kernel (if applicable)
- [ ] Update public API to use dispatch
- [ ] Add unit tests for each backend
- [ ] Add cross-backend consistency test
- [ ] Add performance benchmark
- [ ] Update documentation
- [ ] Remove old dispatch code

## References

- [dispatch_registry.rs](/src/dispatch_registry.rs) - Core registry implementation
- [dispatch_registry_examples.rs](/src/dispatch_registry_examples.rs) - Example registrations
- [shape_error_taxonomy.rs](/src/shape_error_taxonomy.rs) - Error message standards
- [GPU Kernel Priorities](GPU_KERNEL_PRIORITIES.md) - GPU development roadmap

---

**Questions?** Ask in #tenflowers-dev or file an issue
**Contributions:** Please follow this guide when adding new operations
**Last Updated:** 2025-11-10
