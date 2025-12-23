# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TenfloweRS is a pure Rust implementation of TensorFlow, providing a machine learning framework with Rust's safety and performance. It's designed to integrate with the SciRS2/NumRS2 ecosystem and offers both eager execution and static computation graph modes.

## Build and Test Commands

```bash
# Run tests with nextest (required - install with: cargo install cargo-nextest)
cargo nextest run --workspace

# Build all crates
cargo build --workspace

# Build with GPU support
cargo build --workspace --features gpu

# Build with BLAS acceleration
cargo build --workspace --features blas-openblas

# Check for warnings (IMPORTANT: No warnings policy - must pass before any work is complete)
cargo check --workspace
cargo clippy --workspace -- -D warnings

# Run specific test
cargo nextest run test_tensor_creation

# Note: FFI tests require Python development libraries
# Run tests excluding FFI if Python environment is not set up:
cargo test --workspace --exclude tenflowers-ffi
```

## Architecture

### Workspace Structure
- **tenflowers-core**: Core tensor operations, device management, data types
  - `tensor.rs`: Main Tensor type with CPU/GPU storage
  - `device.rs`: Device abstraction (CPU, GPU variants)
  - `ops/`: Tensor operations (arithmetic, linalg, nn)
- **tenflowers-autograd**: Automatic differentiation engine
  - Integrates with scirs2-autograd for static graphs
  - Provides tape-based eager mode gradients
- **tenflowers-neural**: High-level neural network APIs
  - `layers/`: Neural network layers (Dense, Conv2D, BatchNorm, etc.)
  - `models/`: Model abstractions (Sequential, Model trait)
  - `optimizers/`: Training optimizers (SGD, Adam)
- **tenflowers-dataset**: Data loading and preprocessing
- **tenflowers-ffi**: Python bindings via PyO3

### Key Design Patterns

1. **Device Management**: All tensors have an associated Device (CPU or GPU). Operations dispatch based on device placement.

2. **Storage Abstraction**: TensorStorage enum abstracts over CPU (ndarray) and GPU buffers:
   ```rust
   pub(crate) enum TensorStorage<T> {
       Cpu(ArrayD<T>),
       #[cfg(feature = "gpu")]
       Gpu(GpuBuffer<T>),
   }
   ```

3. **Trait-Based Extensibility**: Core traits like `Layer`, `Dataset`, `Differentiable` allow extending functionality.

4. **Feature Flags**: Conditional compilation for optional dependencies (GPU, BLAS, etc.)

### Dependencies and Integration

- Uses NumRS2/SciRS2 ecosystem (numrs2-array, scirs2-autograd, etc.)
- GPU support via WGPU for cross-platform compute
- Python bindings use PyO3
- All operations must maintain compatibility with the broader SciRS2 ecosystem

## Critical Dependencies

TenfloweRS **must** use SciRS2 as its foundation (see SCIRS2_INTEGRATION_POLICY.md):
- `scirs2-core` - Core scientific primitives (required) - **provides ndarray, random, num_traits**
- `scirs2-autograd` - Automatic differentiation (required)
- `scirs2-neural` - Neural network abstractions (required)
- `optirs` - Advanced optimizers from OptiRS project (required)
- Additional SciRS2 crates added based on compilation evidence

SciRS2 is located at `../scirs/` relative to this project.
OptiRS is located at `../optirs/` relative to this project.

### Architectural Hierarchy
```
TenfloweRS (Deep Learning Framework - TensorFlow-compatible API)
    ↓ builds upon
OptiRS (ML Optimization Specialization)
    ↓ builds upon
SciRS2 (Scientific Computing Foundation)
    ↓ builds upon
ndarray, num-traits, etc. (Core Rust Scientific Stack)
```

### FULL USE OF SciRS2-Core

TenfloweRS must make **FULL USE** of scirs2-core's extensive capabilities:

#### Core Array Operations (replaces ndarray)
```rust
// PRIMARY: Use scirs2-core's ndarray re-exports (full functionality)
use scirs2_core::ndarray::{Array, ArrayView, ArrayViewMut, Axis, Ix1, Ix2, IxDyn};
use scirs2_core::ndarray::{array, Array1, Array2, Array3, Array4};

// ALTERNATIVE: scirs2_core::ndarray_ext provides additional extensions
use scirs2_core::ndarray_ext::stats;         // Statistical functions
use scirs2_core::ndarray_ext::matrix;        // Matrix operations
use scirs2_core::ndarray_ext::manipulation;  // Array manipulation
```

#### Random Number Generation (replaces rand)
```rust
use scirs2_core::random::{Random, rng, DistributionExt};
use scirs2_core::random::{QuasiMonteCarloSequence, SecureRandom};
use scirs2_core::random::{ImportanceSampling, VarianceReduction};
```

#### Performance Optimization Features
```rust
// SIMD acceleration
use scirs2_core::simd::{SimdArray, SimdOps, auto_vectorize};
use scirs2_core::simd_ops::{simd_dot_product, simd_matrix_multiply};

// Parallel processing
use scirs2_core::parallel::{ParallelExecutor, ChunkStrategy, LoadBalancer};
use scirs2_core::parallel_ops::{par_chunks, par_join, par_scope};

// GPU acceleration
use scirs2_core::gpu::{GpuContext, GpuBuffer, GpuKernel, CudaBackend, MetalBackend};
use scirs2_core::tensor_cores::{TensorCore, MixedPrecision, AutoTuning};
```

#### Memory Management & Efficiency
```rust
// Memory-efficient operations
use scirs2_core::memory_efficient::{MemoryMappedArray, LazyArray, ChunkedArray};
use scirs2_core::memory_efficient::{ZeroCopyOps, AdaptiveChunking, DiskBackedArray};

// Memory management
use scirs2_core::memory::{BufferPool, GlobalBufferPool, ChunkProcessor};
use scirs2_core::memory::{LeakDetector, MemoryMetricsCollector};
```

#### Advanced Scientific Computing
```rust
// Complex numbers and numeric conversions
use scirs2_core::types::{ComplexOps, ComplexExt, NumericConversion};

// Validation and error handling
use scirs2_core::validation::{check_finite, check_in_bounds, ValidationSchema};
use scirs2_core::error::{CoreError, Result};
```

#### Production-Ready Features
```rust
// Performance profiling
use scirs2_core::profiling::{Profiler, profiling_memory_tracker};
use scirs2_core::benchmarking::{BenchmarkSuite, BenchmarkRunner};

// Metrics and monitoring
use scirs2_core::metrics::{MetricRegistry, Counter, Gauge, Histogram, Timer};
use scirs2_core::observability::{audit, tracing};
```

### Mandatory Usage Guidelines

1. **NEVER** import `ndarray` directly - use `scirs2_core::ndarray` for full functionality
2. **NEVER** import `rand` directly - use `scirs2_core::random`
3. **NEVER** import `num_traits` directly - use `scirs2_core::num_traits`
4. **ALWAYS** use scirs2-core's SIMD operations for performance-critical code (if available)
5. **ALWAYS** use scirs2-core's GPU abstractions for hardware acceleration (if available)
6. **ALWAYS** use scirs2-core's memory management for large data operations (if available)
7. **ALWAYS** use scirs2-core's profiling and benchmarking tools (if available)
8. **ALWAYS** use scirs2-core's error types and result handling
9. **ALWAYS** use scirs2-autograd for automatic differentiation
10. **ALWAYS** use optirs for optimization algorithms instead of implementing from scratch

## Development Guidelines

1. **No Warnings Policy**: All code must compile without warnings. Run `cargo check` and `cargo clippy` before marking any task complete.

2. **Implementation Status**: Many functions contain `todo!()`. When implementing:
   - Check existing patterns in the codebase
   - Ensure trait bounds are complete (especially `Default` for generic types)
   - Maintain consistency with SciRS2 APIs

3. **Testing**: Write tests for new implementations. Integration tests go in `tests/`, unit tests as `#[cfg(test)]` modules.

4. **GPU Development**: GPU kernels use WGPU compute shaders. See `tenflowers-core/src/gpu/` for patterns.

5. **Variable Naming**: Always use `snake_case` for variables, functions, and methods

6. **Type Naming**: Use `PascalCase` for structs, enums, traits

7. **Constants**: Use `SCREAMING_SNAKE_CASE`

8. **Workspace Dependencies**: Use `workspace = true` in Cargo.toml

9. **Latest Crates**: Always use the latest version available on crates.io

## Key Implementation Patterns

1. **Error Handling**: Use `scirs2_core::error::CoreError` and `scirs2_core::Result` when available
2. **Array Operations**: Use `scirs2_core::ndarray` for all ndarray functionality
3. **Random Numbers**: Use `scirs2_core::random` exclusively
4. **Numeric Traits**: Use `scirs2_core::num_traits` exclusively
5. **Parallelization**: Use `scirs2_core::parallel` and `parallel_ops`
6. **SIMD Optimization**: Use `scirs2_core::simd` and `simd_ops`
7. **Memory Efficiency**: Use `scirs2_core::memory_efficient` for large data
8. **Profiling**: Use `scirs2_core::profiling` and `benchmarking`
9. **Metrics**: Use `scirs2_core::metrics` for monitoring

## Common Workflows

### Importing Core Types - FULL SciRS2 Usage
```rust
// PRIMARY: Use scirs2_core::ndarray for all array operations
use scirs2_core::ndarray::{Array, Array1, Array2, ArrayView, Ix1, Ix2, IxDyn, array};

// Additional array extensions
use scirs2_core::ndarray_ext::{stats, matrix, manipulation};

// Random number generation
use scirs2_core::random::{Random, rng, DistributionExt};

// Performance features
use scirs2_core::simd::SimdArray;
use scirs2_core::parallel_ops::{par_chunks, par_join};

// Error handling
use scirs2_core::error::{CoreError, Result};

// Profiling and metrics
use scirs2_core::profiling::Profiler;
use scirs2_core::metrics::{Counter, Timer};
```

### Creating Tensors with SciRS2
```rust
use tenflowers_core::{Tensor, Device};
use scirs2_core::ndarray::{Array2, ArrayView2, array};
use scirs2_core::random::Random;

// Create tensor from array
let array = Array2::zeros((3, 3));
let tensor = Tensor::from_array(array, Device::cpu());

// Using array! macro (from scirs2_core::ndarray)
let data = array![[1.0, 2.0], [3.0, 4.0]];
let tensor = Tensor::from_array(data, Device::cpu());

// Random initialization
let mut rng = Random::new();
let random_array = rng.uniform((5, 5), 0.0, 1.0);
let random_tensor = Tensor::from_array(random_array, Device::cpu());
```

### GPU Operations with SciRS2
```rust
use scirs2_core::gpu::{GpuContext, GpuBuffer};
use tenflowers_core::gpu::TensorGpuOps;

async fn gpu_matmul() -> Result<()> {
    // Use scirs2-core's GPU abstractions
    let context = GpuContext::new()?;
    let a_buffer = GpuBuffer::from_slice(&context, &data_a)?;
    let b_buffer = GpuBuffer::from_slice(&context, &data_b)?;

    // Perform GPU matmul
    let result = TensorGpuOps::matmul_gpu(a_buffer, b_buffer).await?;

    Ok(())
}
```

## Migration Checklist - Ensure Full SciRS2 Usage

When reviewing or writing TenfloweRS code, verify:

### ✅ Arrays and Numerical Operations
- [ ] NO direct `use ndarray::{...}`
- [ ] NO direct `Array`, `Array1`, `Array2` from ndarray
- [ ] YES `use scirs2_core::ndarray::{Array, Array1, Array2, array, ...}` for all ndarray functionality
- [ ] YES `use scirs2_core::ndarray_ext::{stats, matrix, manipulation}` for additional extensions

### ✅ Random Number Generation
- [ ] NO direct `use rand::{...}`
- [ ] NO direct `use rand_distr::{...}`
- [ ] YES `use scirs2_core::random::{Random, rng, ...}`
- [ ] YES use scirs2-core's distribution extensions

### ✅ Performance Optimization
- [ ] YES use `scirs2_core::simd` for vectorized operations
- [ ] YES use `scirs2_core::parallel_ops` for parallelization
- [ ] YES use `scirs2_core::gpu` for GPU acceleration
- [ ] YES use `scirs2_core::memory_efficient` for large datasets

### ✅ Machine Learning Components
- [ ] YES use `scirs2_autograd` for automatic differentiation
- [ ] YES use `scirs2_neural` for neural network abstractions
- [ ] YES use `optirs` for optimization algorithms
- [ ] NO implementing optimizers from scratch

### Common Anti-Patterns to Avoid
```rust
// ❌ WRONG - Direct dependencies
use ndarray::{Array2, array};
use rand::Rng;
use rand_distr::Normal;
use num_traits::Float;

// ✅ CORRECT - Full SciRS2 usage
use scirs2_core::ndarray::{Array2, array};
use scirs2_core::random::{Random, Rng};
use scirs2_core::random::distributions::Normal;
use scirs2_core::num_traits::Float;

// Key Points:
// - scirs2_core::ndarray - Full ndarray re-export WITH array! macro
// - scirs2_core::num_traits - All numeric traits
// - scirs2_core::random - All random number generation
// - Use scirs2_core for ALL scientific computing primitives
```

## Current Focus Areas

The project is in early alpha (0.1.0-alpha.1) with these priorities:
1. Implement basic tensor operations (add, mul, matmul)
2. Complete autograd integration with scirs2-autograd
3. Implement core neural network layers
4. Add GPU compute kernels for operations

See TODO.md for detailed task list and priorities.

## Important Note

**Remember**: TenfloweRS is built on top of the SciRS2 ecosystem, not as a standalone project. It must leverage the full power of SciRS2 to provide a robust deep learning framework. All tensor operations, automatic differentiation, and neural network abstractions should be built upon the solid foundation provided by SciRS2-core, SciRS2-autograd, and SciRS2-neural.

### Correct Import Patterns Summary

```rust
// PRIMARY: Use scirs2_core for all scientific computing
use scirs2_core::ndarray::{Array, Array1, Array2, array};
use scirs2_core::num_traits::{Float, Zero, One};
use scirs2_core::random::{Random, Rng};

// Additional extensions available
use scirs2_core::ndarray_ext::{stats, matrix, manipulation};

// NEVER use direct dependencies:
// use ndarray::{...}     // ❌ Violates SciRS2 policy
// use num_traits::{...}  // ❌ Violates SciRS2 policy
// use rand::{...}        // ❌ Violates SciRS2 policy
```

**Key Points**:
- `scirs2_core::ndarray` - Full ndarray re-export with array! macro
- `scirs2_core::num_traits` - All numeric traits
- `scirs2_core::random` - All random number generation
- ALL scientific computing primitives come from scirs2_core

## SciRS2 Migration Status

✅ **COMPLETED** - Full SciRS2 integration successfully achieved (September 2025)

### Migration Results

| Component | Status | Count |
|-----------|--------|-------|
| **Direct ndarray imports** | ✅ **ELIMINATED** | 0 remaining |
| **Direct rand imports** | ✅ **ELIMINATED** | 0 remaining |
| **Direct num_traits imports** | ✅ **ELIMINATED** | 0 remaining |
| **SciRS2 imports total** | ✅ **ACTIVE** | All files use SciRS2 |
| **scirs2_core::ndarray** | ✅ **WIDESPREAD** | Full ndarray functionality |
| **scirs2_core::num_traits** | ✅ **COMPREHENSIVE** | All numeric operations |
| **scirs2_core::random** | ✅ **COMPREHENSIVE** | All RNG operations |

### Key Achievements

1. **Complete API Migration**: All direct usage of `ndarray`, `rand`, and `num_traits` eliminated
2. **Comprehensive Coverage**: All scientific computing through scirs2_core
3. **Proper Architecture**: Full adherence to the architectural hierarchy
4. **Random Number Generation**: Complete migration to `scirs2_core::random`
5. **Array Operations**: Using `scirs2_core::ndarray` with array! macro support
6. **Numeric Traits**: Using `scirs2_core::num_traits` for all numeric operations
7. **Performance Features**: Access to SIMD, GPU, and parallel optimizations via SciRS2
8. **Policy Compliance**: 100% adherence to SCIRS2_INTEGRATION_POLICY.md

### Validation Metrics

- ✅ **Zero Policy Violations**: No direct ndarray or rand imports found
- ✅ **Comprehensive Adoption**: SciRS2 used in all major components
- ✅ **Correct Patterns**: Proper use of scirs2_autograd vs scirs2_core
- ✅ **Feature Complete**: Random, SIMD, GPU capabilities all available
- ✅ **Future Ready**: Built for scientific computing workflows

TenfloweRS now serves as an **exemplary implementation** of SciRS2 integration and demonstrates best practices for building on the SciRS2 ecosystem.