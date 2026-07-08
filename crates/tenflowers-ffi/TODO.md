# TenfloweRS FFI TODO & Roadmap (v0.2.0)

Initial release capabilities and forward development plan.

Last updated: 2026-07-08

## v0.1.2 — Eager Autograd & Masking/Recurrent Correctness (2026-07-07)

- [x] **Eager Autograd**: new `implicit_autograd` module — a thread-local,
  auto-activating `GradientTape` plus allocation-address-keyed side-tables
  (`TRACKED_REGISTRY`/`LEAVES`/`GRAD_STORE`/`IDENTITY_ANCHORS`) layered on the
  existing explicit `GradientTape` engine. New `PyTensor` methods
  `set_requires_grad`/`backward()`/`grad()` give the Python API a PyTorch-style
  `x.backward(); x.grad()` workflow without requiring an explicit tape object.
  `add`/`sub`/`mul`/`div`/`matmul`/`sum`/`mean`/`relu`/`sigmoid`/`tanh` now
  record onto the implicit tape automatically.
- [x] **Attention masking fix**: `key_padding_mask` previously reached
  shape validation in `PyMultiheadAttention`/transformer encoder-decoder layers
  but was silently dropped before softmax; masks are now genuinely combined
  into one additive bias before scoring (regression-tested with
  hand-computed attention-weight assertions).
- [x] **Recurrent-layer correctness**: GRU forward now threads a
  caller-supplied initial hidden state through every layer instead of
  ignoring it; RNN forward now honors its `nonlinearity` (`"tanh"`/`"relu"`)
  parameter for real; `MaxPool2D`/`AvgPool2D` gained a real `ceil_mode`
  implementation with PyTorch-matching boundary-divisor correction, and
  dilated max-pooling now genuinely dilates the sampling window.
- [x] **46 `#[pyo3(signature = (...))]` fixes** across `math_ops.rs`,
  `neural/functions.rs`, `neural/layers.rs`, `neural/recurrent.rs`,
  `neural/transformer.rs`, `neural/embedding.rs`, `neural/conv_layers.rs`,
  `visualization/mod.rs`, `profiling.rs`, `memory_optimizer.rs` — functions
  whose `Option<T>` parameters had Rust-side defaults but no pyo3 default
  previously forced callers to pass every argument explicitly.
- [x] `arange()`/`linspace()` (`utils.rs`) now return a real `PyTensor`
  instead of a plain Python list, matching `zeros`/`ones`/`rand`.
- Added `gradient_parity` module (standalone finite-difference gradient
  checker), `profiling` module (`PyProfiler` session-based op timing),
  `stable_api` module (stable C/Python API surface catalogue), a Criterion
  benchmark harness (`benches/bindings_bench.rs`), and 3 new examples
  (`examples/basic_ops.rs`, `examples/gradient_example.rs`,
  `examples/optimizer_example.rs`).

## v0.1.2 — Honesty Hardening (2026-06-22)

- Removed production lock-poison panics via signature-preserving recovery; a
  poisoned lock no longer aborts the process.
- 21 Python neural-layer `forward` methods that silently returned
  `Tensor::zeros` now compute real results, wired to `tenflowers_core::ops` /
  `tenflowers_neural`: Conv1D/2D, Max/AvgPool2D, Batch/Layer/Group/InstanceNorm,
  Embedding/EmbeddingBag, LSTM/GRU/RNN (+ cells), MultiheadAttention/SDPA,
  Transformer encoder/decoder, PositionalEncoding, Dropout/Dropout2D.
- AlphaDropout/FeatureAlphaDropout now implement the real SELU-preserving
  formula (were no-ops/zeros).

## 1. Current Capabilities

### Python Bindings (PyO3)
- **Core Tensor Operations**: Comprehensive tensor creation, manipulation, and computation
- **Eager Autograd**: PyTorch-style `x.backward()` / `x.grad()` via the implicit,
  auto-activating `implicit_autograd` tape, in addition to the explicit
  `GradientTape` API
- **Gradient Tape Integration**: Full autograd support with PyTorch-style gradient tape
- **Neural Network Layers**: Dense and Sequential layer implementations with training support
- **Numpy Interoperability**: Seamless tensor <-> ndarray conversion for f32 data types
- **Memory Optimization**: Memory alignment, prefetch utilities, and fragmentation analysis

### Development & Profiling Tools
- **Hook System**: Forward/backward hooks comparable to PyTorch for debugging and monitoring
- **Benchmarking Suite**: Comprehensive performance benchmarking against TensorFlow baselines
- **Memory Profiler**: Advanced memory usage tracking and optimization recommendations
- **Visualization**: Basic tensor visualization and debugging capabilities
- **Performance Analysis**: TensorFlow baseline comparison with memory and throughput metrics

### Advanced Features
- **Large Model Support**: Infrastructure for 1B+ parameter models with parameter sharding
- **Memory Management**: Smart memory pooling, garbage collection, and compaction strategies
- **Eager Execution Optimizer**: Sub-millisecond overhead optimization for eager operations
- **Multi-GPU Support**: Basic multi-GPU tensor operations and device management

### C API Foundation
- **Type System**: C API scaffolding with fundamental types and tensor creation
- **Memory Management**: C-compatible memory management and tensor lifecycle
- **Function Bindings**: Core tensor operation bindings for C/C++ integration
- **Safety**: Memory-safe C API design with proper error handling

### SciRS2 Integration
- **Complete Migration**: 100% usage of SciRS2 ecosystem for underlying implementations
- **Foundation**: Built on scirs2-core, scirs2-autograd for scientific computing primitives
- **Ecosystem**: Seamless integration with broader SciRS2/NumRS2 scientific computing stack

## 2. Current Gaps & Limitations

### Distribution & Packaging
- **No Published Wheels**: No packaging pipeline for Python wheel distribution
- **Build System**: Missing CI/CD for manylinux, macOS universal2, Windows builds
- **Package Management**: No automated package publishing or version management
- **Installation**: No standardized installation process for end users

### API Coverage & Completeness
- **Limited Dtype Support**: Restricted to f32, missing f16/bf16/i32 support
- **Device Coverage**: Limited device abstraction and multi-device support
- **Neural Network APIs**: Incomplete coverage for advanced layers and optimizers
- **Exception Mapping**: Non-standardized error taxonomy and Python exception mapping

### C API Development
- **Not Packaged**: C API not yet ready for distribution or external use
- **Limited Functionality**: Basic scaffolding only, missing comprehensive operation coverage
- **Header Generation**: No automated C header generation or distribution
- **Versioning**: No stable ABI or versioning policy established

### Testing & Validation
- **Python Test Coverage**: Limited Python-side test coverage and validation
- **Performance Validation**: Missing comprehensive performance regression testing
- **Cross-Platform Testing**: Limited testing across different platforms and Python versions

### Honest-error deferrals inherited from core/neural (post-2026-06-22 sweep)
The Python `forward` paths now compute real results on CPU. Capabilities that
surface through these bindings but rely on unfinished backends fail loudly
(no longer faked):
- **GPU compute kernels**: Metal MPS GPU→host readback, GPU einsum correctness,
  and real device-capability queries return honest errors (CPU paths are real).
- **NCCL collective ops**: require the `libnccl` runtime → honest error.
- **TensorFlow / ONNX protobuf import-export**: no protobuf parser wired →
  honest error.

## 3. Near-Term Roadmap

### Priority 1: Distribution & Packaging
1. **Wheel Build CI**: GitHub Actions workflow for manylinux, macOS universal2, Windows
2. **Package Publishing**: Automated PyPI publishing with proper metadata and versioning
3. **Auditwheel/Maturin**: Proper wheel auditing and Python package configuration
4. **Installation Testing**: Cross-platform installation testing and validation

### Priority 2: API Enhancement
5. **Exception Mapping**: Unified error taxonomy (Rust -> Python exception classes)
6. **Dtype Abstraction**: f32 CPU/GPU support, roadmap for f16/bf16 gating
7. **Extended API Surface**: Full optimizer bindings, normalization layers, Mamba/SSM exposure
8. **Device Management**: Enhanced device abstraction and multi-device support

### Priority 3: Testing & Validation
9. **Gradient Parity Harness**: Python vs Rust reference testing framework
10. **Performance Regression**: Comprehensive performance testing and validation
11. **Cross-Platform Testing**: Multi-platform CI testing and validation
12. **Python Test Suite**: Enhanced Python-side test coverage and validation

### Priority 4: C API Development
13. **C Header Export**: Automated C header generation and distribution
14. **Version Symbols**: Stable ABI versioning and symbol management
15. **Extended C API**: Comprehensive operation coverage and functionality
16. **C API Documentation**: Complete C API documentation and examples

## 4. Mid-Term Roadmap

### Advanced Language Bindings
- **Multi-Language Support**: C++, Swift, and other language binding exploration
- **Stable ABI**: Comprehensive stable ABI design and semantic versioning guidelines
- **Plugin System**: Binary extension plugin system for external operators
- **Foreign Bindings**: Integration with other ML framework ecosystems

### Python Ecosystem Integration
- **Async Dataloader**: Python <-> Rust dataset bridge for asynchronous data loading
- **Multi-GPU Python**: Advanced multi-GPU and distributed training Python APIs
- **Jupyter Integration**: Enhanced Jupyter notebook support and visualization
- **Scientific Python**: Deep integration with NumPy, SciPy, scikit-learn ecosystem

### Production & Deployment
- **Production Optimization**: Production-grade performance optimization for bindings
- **Deployment Tools**: Containerization, packaging, and deployment utilities
- **Cloud Integration**: Cloud platform integration and optimization
- **Edge Deployment**: Mobile and edge device deployment optimization

## 5. Active TODO Items

### Immediate Development Tasks
- [x] **CI Wheel Workflow**: GitHub Actions for multi-platform wheel building (COMPLETED 2026-06-10 — .github/workflows/build-wheels.yml enabled; Linux x86_64/aarch64 + macOS Intel/ARM/universal2 + Windows x86_64 + sdist + PyPI publish)
- [x] **Error Mapping Spec**: Design Rust -> Python exception mapping system (done 2026-04-19: see docs/FFI_ERROR_MAPPING.md and error_mapping.rs)
- [x] **Gradient Parity Harness**: Python vs Rust gradient validation framework (COMPLETED 2026-06-10 — gradient_parity.rs: GradientParityChecker, check_scalar_function, numeric_jacobian, gradients_are_close, 12 tests passing)
- [x] **Extended Optimizer Bindings**: Complete optimizer suite Python exposure (COMPLETED 2026-06-10 — neural/extended_optimizers.rs: PyAdamW, PySGD, PyRMSprop, PyAdagrad, PyLion)
- [x] **Layer Export List**: Normalization + SSM Python API implementation (COMPLETED 2026-06-10 — neural/normalization.rs + neural/ssm.rs exposed in Python module)

### Packaging & Distribution
- [x] **Dtype/Device Abstraction**: PyDevice class with Device.cpu()/gpu(id)/rocm(id) and PyDeviceKind (done 2026-04-20: device.rs)
- [x] **C Header Generator**: Automated header generation script (done 2026-04-19: build.rs with TENFLOWERS_REGENERATE_C_HEADER=1 env-var opt-in, c-header-generate feature)
- [x] **Package Metadata**: PyPI package metadata and documentation (done 2026-04-19: added Python 3.13 classifier, Apache-2.0, OS Independent, Changelog URL, updated dev deps)
- [x] **Installation Testing**: Cross-platform installation validation (COMPLETED 2026-06-10 — scripts/check_install.sh: maturin build + venv install + smoke test)
- [x] **Version Management**: Automated version bumping and release management (COMPLETED prior — scripts/bump_version.sh; workspace semver bump + doc-version strings)

### API & Testing Enhancement
- [x] **Python Test Suite**: Comprehensive Python-side testing framework (done 2026-04-19: tests/conftest.py with shared fixtures, markers registered, duplicate test deduped)
- [x] **Performance Benchmarks**: Python binding performance regression testing (COMPLETED 2026-06-10 — benches/bindings_bench.rs: Criterion benchmarks for gradient_parity, tensor_creation, arithmetic, diff_methods)
- [x] **Documentation**: Complete Python API documentation and tutorials (COMPLETED 2026-06-10 — comprehensive `///` and `//!` doc comments added to lib.rs, tensor_ops.rs, neural/mod.rs, neural/layers.rs, neural/gradient_tape.rs; lib.rs //! expanded with full Python tutorial covering all major APIs)
- [x] **Example Gallery**: Comprehensive example gallery and tutorials (COMPLETED 2026-06-10 — examples/basic_ops.rs, gradient_example.rs, optimizer_example.rs)
- [x] **API Stabilization**: Prepare FFI APIs for stable release (COMPLETED 2026-06-10 — stable_api.rs: StableApiVersion, ApiStability, ApiEntry, ApiSurface, stable_api_surface(), 70+ entries catalogued)

### Infrastructure & Quality
- [x] **Memory Safety**: Enhanced memory safety validation and testing (done 2026-04-19: scripts/run_miri.sh + docs/MEMORY_SAFETY.md created)
- [x] **Error Handling**: Exhaustive TensorError → TenflowersError mapping, 23+ variants, 23+ tests (done 2026-04-20: error_mapping.rs)
- [x] **Profiling Integration**: Advanced profiling tool integration (COMPLETED 2026-06-10 — profiling.rs: PyProfiler, PyProfileRecord, PyProfileReport with session lifecycle, record(), top_ops(), 22 unit tests)
- [x] **Debug Support**: PyTensor.__repr__ shows actual dtype; __len__, .ndim, .numel() properties added (done 2026-04-20)

## 6. Advanced Research Areas

### Language Innovation
- **WebAssembly**: WASM bindings for browser-based ML applications
- **GPU Languages**: CUDA Python, OpenCL bindings, and GPU language support
- **DSL Integration**: Domain-specific language integration and code generation
- **JIT Compilation**: Just-in-time compilation for Python operations

### Performance Research
- **Zero-Copy Bindings**: Advanced zero-copy data transfer between languages
- **Memory Management**: Intelligent memory management across language boundaries
- **Async Programming**: Advanced asynchronous programming model integration
- **Hardware Optimization**: Hardware-specific optimization for bindings

### Ecosystem Integration
- **MLOps Integration**: Production MLOps pipeline integration and tooling
- **Cloud Native**: Cloud-native deployment and scaling for language bindings
- **Edge Computing**: Edge device optimization and deployment strategies
- **Research Frameworks**: Integration with cutting-edge research frameworks

## 7. Deferred Items

### Advanced Features
- **Full Multi-Language**: Complete multi-language binding suite beyond Python/C
- **Advanced Plugin System**: Complex plugin architecture for external extensions
- **Research Integration**: Deep integration with academic research frameworks
- **Custom Hardware**: Specialized hardware backend language binding support

### Infrastructure
- **Production Services**: Complete production service integration and deployment
- **Enterprise Features**: Enterprise-grade features like authentication, monitoring
- **Compliance**: Security, compliance, and auditing capabilities
- **Advanced Tooling**: Sophisticated development and debugging tooling

---

Copyright 2025-2026 COOLJAPAN OU (Team KitaSan)
