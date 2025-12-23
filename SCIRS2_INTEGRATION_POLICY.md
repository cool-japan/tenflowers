# SciRS2 Integration Policy for TenfloweRS

## 🚨 CRITICAL ARCHITECTURAL REQUIREMENT

**TenfloweRS MUST use SciRS2 as its scientific computing foundation.** This document establishes the policy for proper, minimal, and effective integration of SciRS2 crates into TenfloweRS.

## Policy Version
- **Version**: 2.0.0 (Enhanced - Full SciRS2 Ecosystem Alignment)
- **Based on**: SciRS2 Ecosystem Policy v3.0.0
- **Effective Date**: TenfloweRS v0.1.0-alpha.2
- **Last Updated**: 2025-10-04
- **Status**: Active - Full Compliance Required

## Core Integration Principles

### 1. **Foundation, Not Dependency Bloat**
- TenfloweRS extends SciRS2's capabilities with deep learning framework specialization
- Use SciRS2 crates **only when actually needed** by TenfloweRS functionality
- **DO NOT** add SciRS2 crates "just in case" - add them when code requires them

### 2. **Evidence-Based Integration**
- Each SciRS2 crate must have **clear justification** based on TenfloweRS features
- Document **specific use cases** for each integrated SciRS2 crate
- Remove unused SciRS2 dependencies during code reviews

### 3. **Architectural Hierarchy**
```
TenfloweRS (Deep Learning Framework - TensorFlow-compatible API)
    ↓ builds upon
OptiRS (ML Optimization Specialization)
    ↓ builds upon
SciRS2 (Scientific Computing Foundation)
    ↓ builds upon
ndarray, num-traits, rand, etc. (Core Rust Scientific Stack)
```

### 4. **Layered Abstraction Architecture**
Following SciRS2's core principle: **Only scirs2-core can use external dependencies directly**. All TenfloweRS crates must access external scientific libraries through SciRS2-Core abstractions.

## Dependency Abstraction Policy

### **Core Principle: No Direct External Dependencies**

**Applies to:** All TenfloweRS crates
- `tenflowers-core`, `tenflowers-autograd`, `tenflowers-neural`, `tenflowers-dataset`, `tenflowers-ffi`
- All tests, examples, benchmarks in all crates
- All integration tests and documentation examples

### **Prohibited Direct Dependencies in Cargo.toml**

The following dependencies are **FORBIDDEN** in TenfloweRS crates' `[dependencies]` sections:

```toml
# ❌ FORBIDDEN - Use scirs2-core instead
rand = { workspace = true }              # Use scirs2_core::random
rand_distr = { workspace = true }        # Use scirs2_core::random
rand_core = { workspace = true }         # Use scirs2_core::random
rand_chacha = { workspace = true }       # Use scirs2_core::random
rand_pcg = { workspace = true }          # Use scirs2_core::random
ndarray = { workspace = true }           # Use scirs2_core::ndarray
ndarray-rand = { workspace = true }      # Use scirs2_core::ndarray (array feature)
ndarray-stats = { workspace = true }     # Use scirs2_core::ndarray (array feature)
ndarray-npy = { workspace = true }       # Use scirs2_core::ndarray (array feature)
ndarray-linalg = { workspace = true }    # Use scirs2_core::ndarray (array feature)
num-traits = { workspace = true }        # Use scirs2_core::numeric
num-complex = { workspace = true }       # Use scirs2_core::numeric
num-integer = { workspace = true }       # Use scirs2_core::numeric
nalgebra = { workspace = true }          # Use scirs2_core::linalg
```

### **Required SciRS2-Core Dependency**

```toml
# ✅ REQUIRED in all TenfloweRS crates
[dependencies]
scirs2-core = { workspace = true, features = ["array", "random"] }
# All external dependencies accessed through scirs2-core
```

### **Prohibited Direct Imports in Code**

```rust
// ❌ FORBIDDEN in TenfloweRS code
use rand::*;
use rand::Rng;
use rand::seq::SliceRandom;
use rand_distr::{Beta, Normal, StudentT};
use ndarray::*;
use ndarray::{Array, Array1, Array2};
use ndarray::{array, s};
use num_complex::Complex;
use num_traits::*;
// etc.
```

### **Required SciRS2-Core Abstractions**

```rust
// ✅ REQUIRED in TenfloweRS code

// === Random Number Generation ===
use scirs2_core::random::*;           // Complete rand + rand_distr functionality
// Includes: thread_rng, Rng, SliceRandom, etc.
// All distributions: Beta, Cauchy, ChiSquared, Normal, StudentT, Weibull, etc.

// === Array Operations ===
use scirs2_core::ndarray::*;          // Full ndarray functionality with array! macro
// Includes: Array, Array1, Array2, ArrayView, array!, s!, azip! macros

// === Array Extensions ===
use scirs2_core::ndarray_ext::*;      // Additional array utilities
// Includes: stats, matrix, manipulation functions

// === Numerical Traits ===
use scirs2_core::num_traits::*;       // num-traits, num-complex, num-integer
use scirs2_core::numeric::*;          // Numeric utilities
// Includes: Float, Zero, One, Num, Complex, etc.

// === Advanced Types ===
use scirs2_core::array::*;            // Scientific array types
use scirs2_core::linalg::*;           // Linear algebra (nalgebra when needed)
```

### **Complete Dependency Mapping**

| External Crate | SciRS2-Core Module | Note |
|----------------|-------------------|------|
| `rand` | `scirs2_core::random` | Full functionality |
| `rand_distr` | `scirs2_core::random` | All distributions |
| `rand_core` | `scirs2_core::random` | Core traits |
| `rand_chacha` | `scirs2_core::random` | ChaCha RNG |
| `rand_pcg` | `scirs2_core::random` | PCG RNG |
| `ndarray` | `scirs2_core::ndarray` | Full with array! macro |
| `ndarray-rand` | `scirs2_core::ndarray` | Via ndarray re-export |
| `ndarray-stats` | `scirs2_core::ndarray_ext::stats` | Stats extensions |
| `ndarray-npy` | `scirs2_core::ndarray` | Via ndarray re-export |
| `ndarray-linalg` | `scirs2_core::ndarray` | Via ndarray re-export |
| `num-traits` | `scirs2_core::num_traits` | All numeric traits |
| `num-complex` | `scirs2_core::num_traits` | Complex numbers |
| `num-integer` | `scirs2_core::num_traits` | Integer traits |
| `nalgebra` | `scirs2_core::linalg` | When needed |

### **Benefits of This Architecture**

1. **Consistent APIs**: All TenfloweRS crates use the same interfaces
2. **Version Control**: Only SciRS2-core manages external dependency versions
3. **Type Safety**: Prevents mixing external types with SciRS2 types
4. **Maintainability**: Changes to external APIs only affect core
5. **Performance**: Core can optimize all external library usage
6. **Documentation**: Single source of truth for API documentation

## Required SciRS2 Crates Analysis

### **ESSENTIAL (Always Required)**

#### `scirs2-core` - FOUNDATION
- **Use Cases**: Core scientific primitives, ScientificNumber trait, random number generation, array utilities
- **TenfloweRS Modules**: All modules use core utilities, tensor operations, device management
- **Status**: ✅ REQUIRED - Foundation crate

#### `scirs2-autograd` - AUTOMATIC DIFFERENTIATION
- **Use Cases**: Gradient computation, backpropagation, computational graph, **array! macro access**
- **TenfloweRS Modules**: `tenflowers-autograd/`, gradient tape, automatic differentiation, test modules throughout
- **Status**: ✅ REQUIRED - Core autograd functionality and primary source for ndarray types
- **Special Note**: The `array!` macro is accessed via `scirs2_autograd::ndarray::array` for tests
- **Important**: `scirs2_autograd::ndarray` provides full ndarray re-export with array! macro

#### `scirs2-neural` - NEURAL NETWORKS
- **Use Cases**: Neural network layers, activation functions, loss functions, model abstractions
- **TenfloweRS Modules**: `tenflowers-neural/`, layers implementation, model building
- **Status**: ✅ REQUIRED - Core neural network functionality

#### `optirs` - OPTIMIZERS (from OptiRS project)
- **Use Cases**: Advanced ML optimizers (SGD, Adam, RMSprop, etc.), learning rate schedules, hardware-accelerated optimization
- **TenfloweRS Modules**: `tenflowers-neural/optimizers/`, training loops
- **Status**: ✅ REQUIRED - Training optimization algorithms (using local OptiRS project)

### **HIGHLY LIKELY REQUIRED**

#### `scirs2-linalg` - LINEAR ALGEBRA
- **Use Cases**: Matrix operations, tensor operations, BLAS/LAPACK integration
- **TenfloweRS Modules**: `tenflowers-core/ops/`, tensor arithmetic, matmul operations
- **Status**: 🔶 INVESTIGATE - Check if tensor ops need beyond ndarray

#### `scirs2-datasets` - DATA HANDLING
- **Use Cases**: Dataset loading, batching, preprocessing pipelines
- **TenfloweRS Modules**: `tenflowers-dataset/`, data loaders, preprocessing
- **Status**: 🔶 INVESTIGATE - If TenfloweRS uses SciRS2's dataset utilities

#### `scirs2-metrics` - PERFORMANCE MONITORING
- **Use Cases**: Training metrics, loss tracking, accuracy measurement
- **TenfloweRS Modules**: Model evaluation, training monitoring
- **Status**: 🔶 INVESTIGATE - For training metrics and evaluation

#### `scirs2-transform` - MATHEMATICAL TRANSFORMS
- **Use Cases**: Data transformations, normalization, standardization
- **TenfloweRS Modules**: `tenflowers-dataset/`, preprocessing pipelines
- **Status**: 🔶 INVESTIGATE - For data preprocessing operations

### **CONDITIONALLY REQUIRED**

#### `scirs2-vision` - COMPUTER VISION
- **Use Cases**: Image preprocessing, augmentation, vision-specific layers
- **TenfloweRS Modules**: Vision models, CNN implementations
- **Status**: ⚠️ CONDITIONAL - Only if TenfloweRS implements vision-specific features

#### `scirs2-text` - TEXT PROCESSING
- **Use Cases**: Text tokenization, embedding layers, NLP preprocessing
- **TenfloweRS Modules**: NLP models, text processing pipelines
- **Status**: ⚠️ CONDITIONAL - Only if TenfloweRS implements NLP features

#### `scirs2-signal` - SIGNAL PROCESSING
- **Use Cases**: Audio processing, 1D convolutions, signal transformations
- **TenfloweRS Modules**: Audio models, signal processing layers
- **Status**: ⚠️ CONDITIONAL - Only if TenfloweRS handles signal data

#### `scirs2-series` - TIME SERIES
- **Use Cases**: RNN/LSTM support, sequence modeling, temporal data
- **TenfloweRS Modules**: Recurrent layers, sequence models
- **Status**: ⚠️ CONDITIONAL - Only if TenfloweRS implements RNN/LSTM

#### `scirs2-sparse` - SPARSE MATRICES
- **Use Cases**: Sparse tensor operations, graph neural networks
- **TenfloweRS Modules**: Sparse tensor support, GNN layers
- **Status**: ⚠️ CONDITIONAL - Only if sparse tensor support is added

#### `scirs2-fft` - FAST FOURIER TRANSFORM
- **Use Cases**: Frequency domain operations, spectral layers
- **TenfloweRS Modules**: FFT-based layers, spectral operations
- **Status**: ⚠️ CONDITIONAL - Only for frequency domain operations

#### `scirs2-stats` - STATISTICAL ANALYSIS
- **Use Cases**: Statistical layers, probabilistic models, distributions
- **TenfloweRS Modules**: Probabilistic layers, VAE/GAN support
- **Status**: ⚠️ CONDITIONAL - For statistical modeling features

### **LIKELY NOT REQUIRED**

#### `scirs2-cluster` - CLUSTERING
- **Status**: ❌ UNLIKELY - Unless clustering layers are implemented

#### `scirs2-graph` - GRAPH ALGORITHMS
- **Status**: ❌ UNLIKELY - Unless GNN support is added

#### `scirs2-spatial` - SPATIAL DATA
- **Status**: ❌ UNLIKELY - Unless spatial transformers are added

#### `scirs2-ndimage` - IMAGE PROCESSING
- **Status**: ❌ UNLIKELY - Basic tensor ops likely sufficient

#### `scirs2-interpolate` - INTERPOLATION
- **Status**: ❌ UNLIKELY - Unless interpolation layers needed

#### `scirs2-integrate` - NUMERICAL INTEGRATION
- **Status**: ❌ UNLIKELY - Not typical in deep learning

#### `scirs2-special` - SPECIAL FUNCTIONS
- **Status**: ❌ UNLIKELY - Unless specialized activations needed

#### `scirs2-io` - INPUT/OUTPUT
- **Status**: ❌ UNLIKELY - Basic I/O likely sufficient

#### `scirs2-optimize` - OPTIMIZATION
- **Status**: ❌ UNLIKELY - Using optirs for neural optimizers instead

## Integration Guidelines

### **Adding New SciRS2 Dependencies**

1. **Document Justification**
   ```markdown
   ## SciRS2 Crate Addition Request

   **Crate**: scirs2-[name]
   **Requestor**: [Developer Name]
   **Date**: [Date]

   **Justification**:
   - Specific TenfloweRS feature requiring this crate
   - Code modules that will use it
   - Alternatives considered and why SciRS2 is preferred

   **Impact Assessment**:
   - Compilation time impact
   - Binary size impact
   - Maintenance burden
   ```

2. **Code Review Requirements**
   - Demonstrate actual usage in TenfloweRS code
   - Show integration examples
   - Verify no equivalent functionality exists in already-included crates

3. **Documentation Requirements**
   - Update this policy document
   - Document usage patterns in relevant module docs
   - Add examples to integration tests

### **Removing SciRS2 Dependencies**

1. **Regular Audits** (quarterly)
   - Review all SciRS2 dependencies for actual usage
   - Remove unused imports and dependencies
   - Update documentation

2. **Deprecation Process**
   - Mark as deprecated with removal timeline
   - Provide migration guide if functionality moves
   - Remove after deprecation period

### **Best Practices**

1. **Import Granularity**
   ```rust
   // ✅ GOOD - Specific imports
   use scirs2_core::random::Random;
   use scirs2_autograd::tape::Tape;
   use scirs2_autograd::ndarray::{Array, Array2}; // Array types from autograd
   use scirs2_neural::layers::Dense;

   // ❌ BAD - Broad imports
   use scirs2_core::*;
   use scirs2_neural::*;
   ```

2. **Array Import Pattern**
   ```rust
   // ✅ CORRECT - Primary choice: scirs2_autograd for full ndarray functionality
   use scirs2_autograd::ndarray::{Array, Array1, Array2, array};

   // ✅ ALSO CORRECT - Alternative: scirs2_core::ndarray_ext for basic types
   use scirs2_core::ndarray_ext::{Array, ArrayView, ArrayViewMut};
   use scirs2_core::ndarray_ext::stats;   // Statistical operations
   use scirs2_core::ndarray_ext::matrix;  // Matrix operations
   // Note: scirs2_core::ndarray_ext does NOT provide array! macro

   // ❌ WRONG - Don't use ndarray directly
   use ndarray::{Array, array};  // Violates SciRS2 integration policy

   // Example usage in tests:
   #[cfg(test)]
   mod tests {
       use super::*;
       // Use scirs2_autograd::ndarray when you need array! macro
       use scirs2_autograd::ndarray::{array, Array1};

       #[test]
       fn test_tensor_ops() {
           let data = array![1.0, 2.0, 3.0];  // Requires scirs2_autograd::ndarray
           let arr: Array1<f64> = Array1::zeros(10);
           // test implementation
       }
   }
   ```

3. **Feature Gates**
   ```rust
   // ✅ GOOD - Optional features
   #[cfg(feature = "vision")]
   use scirs2_vision::transforms::ImageTransform;

   #[cfg(feature = "nlp")]
   use scirs2_text::tokenizers::Tokenizer;
   ```

4. **Device Compatibility**
   ```rust
   // ✅ GOOD - Maintain device abstraction
   use scirs2_core::ScientificNumber;
   // Ensure SciRS2 types work with CPU/GPU device abstraction
   ```

## Enforcement

### **Automated Checks**
- CI pipeline checks for unused SciRS2 dependencies
- Documentation tests verify integration examples work
- Dependency graph analysis in builds
- Cargo nextest for comprehensive testing

### **Manual Reviews**
- All SciRS2 integration changes require team review
- Quarterly dependency audits
- Annual architecture review

### **Violation Response**
1. **Warning**: Document why integration is needed
2. **Correction**: Remove unjustified dependencies
3. **Training**: Educate team on integration policy

## Future Considerations

### **SciRS2 Version Management**
- Track SciRS2 release cycle (currently at beta.2)
- Test TenfloweRS against SciRS2 beta releases
- Coordinate breaking change migrations
- Follow workspace version management

### **Performance Monitoring**
- Benchmark impact of SciRS2 integration
- Monitor compilation times
- Track binary size impact
- GPU kernel performance with SciRS2 types

### **Community Alignment**
- Coordinate with SciRS2 team on roadmap
- Contribute improvements back to SciRS2
- Maintain architectural consistency with NumRS2/SciRS2 ecosystem

## TenfloweRS-Specific Considerations

### **Tensor Operations**
- Use SciRS2 array types as foundation for tensors
- Extend with GPU support via WGPU
- Maintain compatibility with SciRS2 autograd

### **Neural Network Layers**
- Build on top of scirs2-neural abstractions
- Extend with TensorFlow-compatible layers
- Maintain similar API patterns

### **Python FFI**
- Ensure SciRS2 types can be exposed via PyO3
- Maintain NumPy compatibility through SciRS2

## Conclusion

This policy ensures TenfloweRS properly leverages SciRS2's scientific computing foundation while building a comprehensive deep learning framework. **TenfloweRS must use SciRS2, but intelligently and purposefully.**

---

**Document Version**: 2.0.0
**Last Updated**: 2025-10-04
**Next Review**: Q1 2026
**Owner**: TenfloweRS Architecture Team

## Quick Reference

### Current Recommended Integration (Minimal Start)
```toml
# Essential SciRS2 dependencies for TenfloweRS (using RC.1)
scirs2-core = "0.1.0-rc.1"      # Always required - foundation
scirs2-autograd = "0.1.0-rc.1"  # Primary source for ndarray types with array! macro
scirs2-neural = "0.1.0-rc.1"    # Neural network abstractions
optirs = "0.1.0-beta.1"         # Training optimizers from OptiRS project

# Add these only when needed:
# scirs2-linalg = "0.1.0-rc.1"    # If advanced linalg beyond ndarray
# scirs2-datasets = "0.1.0-rc.1"  # If using SciRS2 data utilities
# scirs2-metrics = "0.1.0-rc.1"   # If using SciRS2 metrics
# scirs2-transform = "0.1.0-rc.1" # If data transformations needed
```

### Correct Import Patterns for Arrays

```rust
// OPTION 1: When you need full ndarray functionality including array! macro:
use scirs2_autograd::ndarray::{Array, Array1, Array2, array};

// OPTION 2: When you only need basic array types (no array! macro):
use scirs2_core::ndarray_ext::{Array, ArrayView, ArrayViewMut};
use scirs2_core::ndarray_ext::{stats, matrix, manipulation};

// NEVER use ndarray directly:
// use ndarray::{...}  // ❌ Violates SciRS2 policy
```

**Key Points**:
- `scirs2_autograd::ndarray` - Full ndarray re-export with array! macro
- `scirs2_core::ndarray_ext` - Basic types and operations, NO array! macro
- Choose based on your needs (array! macro requirement)

**Remember**: Start minimal, add based on evidence, document everything!

---

## 🔧 PRACTICAL IMPLEMENTATION GUIDE

### **Migration Patterns from TenfloweRS Experience**

Based on the successful migration of TenfloweRS to SciRS2 beta.2, here are practical patterns for implementing the integration policy:

#### **Examples and Demo Code**
```rust
// ✅ MIGRATED PATTERN - Examples using SciRS2
use scirs2_autograd::ndarray::{ArrayD, Array1, Array2, IxDyn};
use tenflowers_core::{Tensor, Device};

fn demo_tensor_creation() -> Result<(), Box<dyn std::error::Error>> {
    // Create tensor using scirs2_autograd::ndarray
    let tensor_1d = Tensor::from_array(ArrayD::from_shape_vec(
        IxDyn(&[3]),
        vec![1.0f32, 2.0, 3.0],
    )?);

    let tensor_2d = Tensor::from_array(ArrayD::from_shape_vec(
        IxDyn(&[2, 3]),
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
    )?);

    Ok(())
}

// ❌ OLD PATTERN - Direct ndarray usage (migrated away from)
// use ndarray::{ArrayD, IxDyn};  // Don't do this anymore
```

#### **Random Number Generation Migration Status**
```rust
// 🔶 PARTIAL MIGRATION - rand usage analysis
use rand::Rng;  // Still needed in some cases

// Current status of scirs2_core::random API:
// ✅ Basic functionality available
// ⚠️  API not yet complete enough for full replacement
// 📋 TODO: Complete migration when scirs2_core::random API matures

// Example current usage (retention of rand for now):
fn dropout_layer_example() {
    let mut rng = rand::thread_rng();
    let random_val: f64 = rng.random(); // Basic random generation
    // Will migrate to scirs2_core::random when API is complete
}
```

#### **Testing Pattern Updates**
```rust
// ✅ CORRECT - Test imports using SciRS2
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::{Array1, array};  // Full functionality

    #[test]
    fn test_tensor_operations() -> Result<(), Box<dyn std::error::Error>> {
        let data = array![1.0, 2.0, 3.0];  // array! macro available
        let tensor = Tensor::from_array(data.into_dyn());
        assert_eq!(tensor.shape(), &[3]);
        Ok(())
    }
}
```

### **Common Migration Issues and Solutions**

#### **Issue 1: IxDyn Import Missing**
```rust
// ❌ COMPILATION ERROR
use scirs2_autograd::ndarray::ArrayD;
// ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), data)  // ndarray::IxDyn not available

// ✅ SOLUTION
use scirs2_autograd::ndarray::{ArrayD, IxDyn};
// ArrayD::from_shape_vec(IxDyn(&[3]), data)  // IxDyn available directly
```

#### **Issue 2: Array Macro Access**
```rust
// ❌ NO ARRAY MACRO - Using scirs2_core::ndarray_ext
use scirs2_core::ndarray_ext::{Array1, Array2};
// let data = array![1.0, 2.0];  // Error: array! macro not available

// ✅ SOLUTION - Use scirs2_autograd::ndarray for full functionality
use scirs2_autograd::ndarray::{Array1, Array2, array};
let data = array![1.0, 2.0];  // Works correctly
```

#### **Issue 3: Mixed Import Sources**
```rust
// ❌ INCONSISTENT - Mixed sources
use ndarray::Array1;                    // Direct ndarray
use scirs2_autograd::ndarray::Array2;   // SciRS2

// ✅ CONSISTENT - Single source
use scirs2_autograd::ndarray::{Array1, Array2, ArrayD, IxDyn};
```

### **Migration Checklist for New Code**

When writing new TenfloweRS code, ensure:

- [ ] **No direct ndarray imports** - Use `scirs2_autograd::ndarray` or `scirs2_core::ndarray_ext`
- [ ] **Choose appropriate import source**:
  - `scirs2_autograd::ndarray` - When you need `array!` macro or full ndarray functionality
  - `scirs2_core::ndarray_ext` - When you only need basic array types (no macro)
- [ ] **Import all needed types together** - Include `IxDyn`, `ArrayD`, etc. in single import
- [ ] **Update test imports** - Ensure test modules use SciRS2 imports consistently
- [ ] **Document migration reasoning** - Why this import choice was made

### **File Types and Import Patterns**

#### **Examples and Demos** (User-facing code)
```rust
// Priority: Use scirs2_autograd::ndarray for full functionality
use scirs2_autograd::ndarray::{ArrayD, Array1, Array2, IxDyn, array};
```

#### **Internal Library Code** (Performance-critical)
```rust
// Choice depends on needs:
// Option 1: Full functionality
use scirs2_autograd::ndarray::{ArrayD, ArrayView, ArrayViewMut};

// Option 2: Basic types only (when array! not needed)
use scirs2_core::ndarray_ext::{Array, ArrayView, ArrayViewMut};
```

#### **Test Code** (Needs array! macro)
```rust
#[cfg(test)]
mod tests {
    use scirs2_autograd::ndarray::{array, Array1, Array2};  // array! macro essential
}
```

### **Performance Considerations**

1. **Compilation Time**: Using `scirs2_autograd::ndarray` includes autograd dependencies
2. **Runtime Impact**: Both options have equivalent runtime performance
3. **Memory Usage**: No significant difference between import patterns
4. **Recommendation**: Use `scirs2_autograd::ndarray` by default unless you specifically want to avoid autograd dependencies

### **Future Migration Tasks**

#### **High Priority**
- [ ] Complete `scirs2_core::random` API assessment
- [ ] Migrate remaining `rand` usage when API is ready
- [ ] Add SciRS2 SIMD optimizations where beneficial

#### **Medium Priority**
- [ ] Evaluate `scirs2_linalg` for advanced matrix operations
- [ ] Assess `scirs2_datasets` for data loading utilities
- [ ] Consider `scirs2_metrics` for training metrics

#### **Low Priority**
- [ ] Specialized modules (vision, text, signal) based on TenfloweRS feature expansion

### **Validation Commands**

Regular checks to ensure SciRS2 compliance:

```bash
# Check for direct ndarray imports (should be empty)
grep -r "^use ndarray::" --include="*.rs" crates/

# Check for direct rand imports (document remaining cases)
grep -r "^use rand::" --include="*.rs" crates/

# Verify tests pass with SciRS2 integration
cargo nextest run --workspace

# Ensure zero warnings policy maintained
cargo clippy --workspace -- -D warnings
```

---

**Migration Status**: TenfloweRS successfully migrated all direct `ndarray` imports to SciRS2 patterns. Random number generation migration pending completion of `scirs2_core::random` API.