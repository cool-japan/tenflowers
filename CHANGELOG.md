# Changelog

All notable changes to TenfloweRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### In Progress
- Additional GPU kernel implementations for advanced operations
- Complete shape inference system
- Graph mode execution engine enhancements
- Re-enable Python bindings (requires Python environment setup)
- Re-enable tensorboard integration (awaiting protobuf security fix)
- ONNX import/export support

## [0.1.0-rc.1] - 2026-02-08

### Summary
Release Candidate 1 with updated dependencies and stability improvements. This release focuses on keeping dependencies up-to-date and maintaining compatibility with the latest SciRS2 ecosystem.

**Release Status:** ✅ Release Candidate (5 crates ready)
- **Tests:** Maintained 100% pass rate
- **Security:** 0 vulnerabilities
- **Quality:** Zero clippy warnings, full formatting compliance
- **Dependencies:** Updated to latest compatible versions

### Changed

#### Dependency Updates
- **SciRS2 Ecosystem Updates**: All scirs2-* dependencies updated to 0.1.5
  - scirs2-core: 0.1.4 → 0.1.5
  - scirs2-autograd: 0.1.4 → 0.1.5
  - scirs2-neural: 0.1.4 → 0.1.5
  - scirs2-linalg: 0.1.4 → 0.1.5
  - scirs2-numpy: 0.1.4 → 0.1.5
- **Compatibility**: Verified compatibility with latest SciRS2 releases
- **Build System**: All workspace builds verified after dependency updates

### Fixed
- **Workspace Consistency**: Ensured all subcrates properly use workspace dependencies
- **Version Alignment**: All internal version references updated to rc.1

## [0.1.0-beta.1] - 2026-02-06

### Summary
First beta release with comprehensive quality assurance and security hardening. This release focuses on stability, security, and production readiness for the core functionality.

**Release Status:** ✅ Production-ready core (5 crates published)
- **Tests:** 2357/2357 passing (100% pass rate)
- **Security:** 0 vulnerabilities (all known issues resolved)
- **Quality:** Zero clippy warnings, full formatting compliance
- **Documentation:** Complete crate-level docs and READMEs

### Added

#### Quality Assurance
- **Comprehensive Testing**: All 2357 tests passing across workspace
  - Core tensor operations: 100% coverage
  - Autograd engine: Full gradient validation
  - Neural network layers: Complete integration tests
  - Dataset loading: Multi-format support verified
- **Security Hardening**: Zero security vulnerabilities
  - All dependencies audited with cargo-audit
  - Known vulnerabilities resolved (see Removed section)
- **Code Quality**: Zero warnings policy enforced
  - All clippy warnings resolved
  - Complete formatting compliance
  - No `unwrap()` usage (safe error handling throughout)

#### Documentation
- **Crate-level Documentation**: All published crates have comprehensive docs
  - tenflowers-core (6.5 MiB): Core tensor operations and GPU support
  - tenflowers-autograd (2.8 MiB): Automatic differentiation engine
  - tenflowers-dataset (2.1 MiB): Data loading and preprocessing
  - tenflowers-neural (3.0 MiB): Neural network layers and training
  - tenflowers (182 KiB): Unified API and prelude
- **README Files**: All subcrates include usage examples and feature documentation
- **Version Consistency**: All internal dependencies aligned to beta.1

### Changed

#### Version Updates
- **Workspace Version**: 0.1.0-alpha.2 → 0.1.0-beta.1
- **Internal Dependencies**: All subcrates updated to reference beta.1
- **API Stability**: Moving toward stable API for 1.0 release

#### Dependency Management
- **Dependency Reduction**: 684 → 668 crates (-16 dependencies)
- **Security Focus**: Removed vulnerable and unmaintained packages
- **SciRS2 Integration**: Continued pure Rust ecosystem alignment

### Removed

#### Security Fixes (Temporary)
- **Tensorboard Integration** ⚠️ TEMPORARY REMOVAL
  - **Reason:** Removed tensorboard-rs due to RUSTSEC-2024-0437 (protobuf 2.28.0 crash vulnerability)
  - **Impact:** Users needing tensorboard should use alternative logging temporarily
  - **Timeline:** Will be re-added once tensorboard-rs updates to protobuf >=3.7.2
  - **Workaround:** Use standard logging or wait for next release

- **Related Packages Removed** (due to tensorboard removal):
  - protobuf v2.28.0 (security vulnerability)
  - tensorboard-rs v0.5.9
  - tensorboard-proto v0.5.7
  - Image processing dependencies (adler, deflate, miniz_oxide, jpeg-decoder, png, tiff, gif)

#### Python FFI ⚠️ TEMPORARY EXCLUSION
- **tenflowers-ffi** (publish = false for this release)
  - **Reason:** Requires Python development environment setup
  - **Impact:** Python bindings not available in this release
  - **Timeline:** Will be re-enabled in future release with proper CI/CD
  - **Status:** Code remains in repository but crate not published
  - **Workaround:** Use Rust API directly or wait for next release

### Fixed

#### Security
- **RUSTSEC-2024-0437**: Fixed protobuf crash vulnerability by removing tensorboard-rs
- **Dependency Audit**: All remaining dependencies verified safe
  - Only 2 acceptable warnings (unmaintained transitive dependencies)
  - instant v0.1.13 (from hdf5, low risk)
  - paste v1.0.15 (from SciRS2 ecosystem, low risk)

#### Build & Package
- **Package Verification**: All 5 crates successfully package and verify
- **Internal Dependencies**: Fixed version mismatches between crates
- **Feature Flags**: Cleaned up feature dependencies (removed python from "full" feature)
- **FFI Exports**: Properly excluded from main crate to prevent build errors

#### Code Quality
- **Formatting**: All code formatted to project standards
- **Clippy Warnings**: Zero warnings with strict checking (-D warnings)
- **Documentation**: All public APIs documented
- **Tests**: All test suites passing (excluding optional FFI)

### Migration Guide

#### From alpha.2 to beta.1

**Breaking Changes:**
1. **Tensorboard feature removed** (temporarily)
   ```toml
   # BEFORE (alpha.2)
   [features]
   tensorboard = ["tensorboard-rs"]

   # AFTER (beta.1)
   # Feature removed - use alternative logging
   ```

2. **Python bindings not available** (temporarily)
   ```toml
   # BEFORE (alpha.2)
   [dependencies]
   tenflowers = { version = "0.1.0-alpha.2", features = ["python"] }

   # AFTER (beta.1)
   # Python feature not available - use Rust API
   [dependencies]
   tenflowers = "0.1.0-beta.1"
   ```

**No Other Breaking Changes:**
- Core API remains compatible
- All tensor operations unchanged
- Autograd functionality preserved
- Neural network APIs stable
- Dataset loading unchanged

### Known Issues

**Transitive Dependencies:**
- 2 unmaintained dependencies (acceptable risk):
  - `instant` v0.1.13: Transitive from hdf5, low severity
  - `paste` v1.0.15: Transitive from SciRS2, low severity
- These are dependency-of-dependency issues and will be resolved in future releases

**Platform-Specific:**
- ARM64 target feature warning (fp-armv8): Minor deprecation, won't block builds

### Performance

**Benchmarks:** (from test suite execution)
- Test suite: 2357 tests in 67.671s (~35 tests/second)
- No performance regressions from alpha.2
- GPU operations maintain performance characteristics

### Crates Published

| Crate | Size | Compressed | Description |
|-------|------|------------|-------------|
| tenflowers-core | 6.5 MiB | 1.0 MiB | Core tensor operations and GPU support |
| tenflowers-autograd | 2.8 MiB | 517 KiB | Automatic differentiation engine |
| tenflowers-dataset | 2.1 MiB | 408 KiB | Data loading and preprocessing |
| tenflowers-neural | 3.0 MiB | 534 KiB | Neural network layers and training |
| tenflowers | 182 KiB | 48 KiB | Unified API and prelude |

**Not Published:**
- tenflowers-ffi: Marked as `publish = false` (see Removed section)

### Installation

```toml
[dependencies]
tenflowers = "0.1.0-beta.1"

# Optional features
tenflowers = { version = "0.1.0-beta.1", features = ["gpu", "simd"] }
```

**Note:** Python bindings not available in this release. Use Rust API directly.

### Contributors

This release was prepared with comprehensive testing and quality assurance by the COOLJAPAN OU (Team Kitasan) development team.

## [0.1.0-alpha.2] - 2025-12-23

### Added

#### Documentation Improvements
- **Comprehensive Crate Documentation**: Added extensive crate-level documentation to all crates
  - `tenflowers-core`: Complete API overview with examples for tensor operations, GPU acceleration, mixed precision, and performance monitoring
  - `tenflowers-dataset`: Full guide to data loading, transformations, and advanced features
  - `tenflowers-ffi`: Python bindings documentation with NumPy integration examples
  - All crates now include Quick Start guides and architecture overviews
- **Enhanced README**: Updated with alpha.2 information and current capabilities
- **API Documentation**: Improved rustdoc comments throughout the codebase

#### Performance Features
- **CUDA Support**: Enhanced GPU backend with CUDA optimization paths
- **Memory Optimization**: Improved memory management and buffer pooling
- **SIMD Enhancements**: Additional SIMD-accelerated operations
- **Profiling Tools**: Built-in performance benchmarking and monitoring utilities

#### Core Enhancements
- **Deterministic Execution**: Added deterministic mode for reproducible results
- **Quantization**: Expanded quantization support for model deployment
- **Mixed Precision**: Improved mixed precision training capabilities
- **Checkpointing**: Enhanced model checkpointing and restoration
- **Error Handling**: Improved error messages and shape validation

#### Neural Network Module
- **Layer Expansion**: Additional neural network layer implementations
- **Optimizer Improvements**: Enhanced optimizer implementations
- **Training Utilities**: Improved training loop abstractions

#### Dataset Module
- **Data Quality Tools**: Built-in data quality analysis and drift detection
- **Advanced Sampling**: Stratified and importance sampling strategies
- **Performance**: NUMA-aware scheduling and zero-copy operations
- **Distributed Loading**: Distributed and sharded data loading support

### Improved
- **SciRS2 Integration**: Complete migration to SciRS2 ecosystem primitives
  - All operations now use `scirs2_core::ndarray` instead of direct `ndarray`
  - Random number generation via `scirs2_core::random`
  - Numeric traits via `scirs2_core::num_traits`
- **Type System**: Enhanced data type support (f16, bf16, etc.)
- **Shape Inference**: Improved shape validation and broadcasting
- **GPU Memory**: Better GPU memory management and metrics
- **Documentation**: Comprehensive rustdoc throughout all modules

### Fixed
- **Compilation Issues**: Resolved various compilation warnings and errors
- **Type Safety**: Fixed trait bound issues across generic implementations
- **Memory Leaks**: Fixed memory management issues in GPU operations
- **API Consistency**: Standardized API patterns across crates

### Changed
- **Version**: Updated to 0.1.0-alpha.2 across all crates
- **Build System**: Improved workspace configuration
- **Testing**: Enhanced test coverage and infrastructure

## [0.1.0-alpha.1] - 2025-09-27

### Added

#### Core Infrastructure
- **Tensor System**: Generic tensor type with device abstraction
  - Reference-counted buffer management
  - Zero-copy views and slicing
  - Strided layout support
  - Automatic broadcasting
- **Device Management**: Unified CPU/GPU abstraction
  - CPU backend via ndarray
  - GPU backend via WGPU (experimental)
  - Cross-device tensor transfers
  - Device placement strategies
- **Operation Registry**: Extensible operation system
  - Trait-based operation definitions
  - Kernel dispatch by device/dtype
  - Macro-based registration
  - Basic shape inference

#### Tensor Operations
- **Basic Ops**: Add, Sub, Mul, Div, Pow, Neg
- **Reductions**: Sum, Mean, Max, Min, ArgMax, ArgMin
- **Manipulation**: Reshape, Transpose, Concat, Stack, Squeeze
- **Linear Algebra**: MatMul (CPU only)
- **Activation**: ReLU, Sigmoid, Tanh (stubs)

#### Automatic Differentiation
- **GradientTape**: Reverse-mode automatic differentiation
  - Tape-based operation recording
  - Basic operation gradients (Add, Mul, MatMul, ReLU)
  - Multiple gradient computation
  - Persistent tape support
- **Integration**: Seamless integration with scirs2-autograd

#### Neural Network Module
- **Layers**: Layer trait with builder pattern
  - Dense/Linear layers
  - Conv2D (stub)
  - BatchNorm (stub)
  - Dropout
- **Models**: Sequential and Model traits
- **Optimizers**: SGD, Adam (simplified for f64)
- **Loss Functions**: MSE, CrossEntropy (stubs)

#### Data Pipeline
- **Dataset Trait**: Flexible data loading abstraction
- **TensorDataset**: In-memory tensor dataset
- **Transformations**: Basic preprocessing pipeline

#### FFI
- **Python Bindings**: Initial PyO3 integration
  - PyTensor wrapper
  - Basic tensor operations
  - NumPy interop foundation

### Known Limitations
- Most operations return "Not Implemented" errors
- GPU support is experimental and incomplete
- Limited operation coverage
- No graph mode execution yet
- Minimal Python API
- f32 support limited in some modules

### [0.1.0-alpha.1] 
- Complete GPU kernel implementations
- Expand operation coverage (Conv2D, pooling, normalization)
- Implement graph mode execution
- Add DataLoader with parallel loading
- Improve Python API coverage
- Graph optimization passes
- Mixed precision training
- Distributed training support
- ONNX import/export
- Performance optimizations
- Production-ready Python API
- TorchScript-like JIT compilation
- Quantization support
- Model zoo with pretrained models
- Comprehensive documentation

## Roadmap

### [1.0.0] - Target: 2026
- Stable API guarantee
- Performance parity with TensorFlow/PyTorch
- Full operation coverage
- Production deployment tools
- Extensive ecosystem integrations

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0-beta.1 | 2026-02-06 | First beta: 2357 tests passing, 0 vulnerabilities, production-ready core |
| 0.1.0-alpha.2 | 2025-12-23 | Documentation overhaul, CUDA enhancements, SciRS2 integration complete |
| 0.1.0-alpha.1 | 2025-09-27 | Initial alpha release with core infrastructure |
