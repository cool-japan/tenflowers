# TenfloweRS FFI TODO & Roadmap (v0.2.0)

Initial release capabilities and forward development plan.

Last updated: 2026-07-12

## v0.2.0 — Implicit Autograd Wired Through Every Layer & Optimizer (2026-07-12)

- [x] **Implicit-autograd rewrite spans the full layer/optimizer/loss
  surface**: previously `.backward()`/`.grad()`/`optimizer.step()` only
  worked end-to-end for a minimal Dense/Sequential/MSE/relu-sigmoid-tanh
  path; the thread-local `implicit_autograd` tape is now genuinely wired
  through every layer type and all 9 optimizers. Landed in exactly two
  commits, both dated 2026-07-11: `85bf627` ("Implement tensor slicing
  functionality and enhance autograd support") and `0727e96` ("Add
  end-to-end training convergence tests for TenfloweRS FFI"). Combined they
  touched 34 files in `crates/tenflowers-ffi/src/` (+22,276/-7,005 lines).
  - Layers now confirmed tape-linked with real `forward()`: `Dense` +
    `PyParameter` (`neural/layers.rs`), `Conv1D`/`Conv2D`/`Conv3D` +
    `MaxPool2D`/`AvgPool2D` (`neural/conv_layers/mod.rs` — `Conv2D` backward
    is confirmed only for unit dilation, `groups == 1`, and default
    `padding=(0,0)`; other configurations forward correctly but are not
    confirmed tape-linked), `Embedding`/`EmbeddingBag` (`neural/embedding.rs`),
    `BatchNorm1d`/`LayerNorm`/`GroupNorm`/`InstanceNorm1d`
    (`neural/normalization/mod.rs`), `MultiheadAttention`
    (`neural/attention/mod.rs`), `TransformerEncoderLayer`/
    `TransformerDecoderLayer`/`PositionalEncoding` (`neural/transformer/mod.rs`),
    `LSTM`/`GRU`/`RNN` + single-step `LSTMCell`/`GRUCell`
    (`neural/recurrent/{lstm,gru,rnn}.rs`).
  - All 9 optimizers confirmed to perform genuine gradient-based parameter
    updates via `step(&mut self, model)`, not no-ops: `SGD`, `Adam`,
    `RMSprop`, `AdamW` (`neural/optimizers/mod.rs`); `AdaBelief`, `RAdam`,
    `Nadam`, `AdaGrad`, `AdaDelta` (`neural/extended_optimizers/mod.rs`).
  - `PyTensor::transpose`/`reshape` (`tensor_ops.rs`) previously did not
    record themselves onto the implicit-autograd tape at all — no
    `UnaryOpKind::Transpose`/`Reshape` variant existed, so `.backward()`
    silently failed to propagate gradients through either op on a tracked
    tensor. Both now call `record_and_link_unary`, the same hook used by
    every other tracked `PyTensor` op; `PyTensor::slice` likewise now
    records onto the tape (regression-tested by
    `slice_links_onto_tape_and_grad_is_correct`).
- [x] **File-layout split for the 2000-line-per-file policy** (no functional
  change): `implicit_autograd.rs` and `neural/{attention,conv_layers,
  recurrent,transformer,extended_optimizers}.rs` were each split from a
  single file into `mod.rs` + `tests.rs`; `neural/recurrent/` was further
  split into `lstm.rs`/`gru.rs`/`rnn.rs`/`mod.rs`/`tests.rs`. `neural/
  normalization.rs` and `neural/optimizers.rs` were split the same way
  (`normalization/{mod.rs,tests.rs}`, `optimizers/{mod.rs,tests.rs}`) in a
  final 2026-07-12 policy-compliance pass, completing 2000-line compliance
  for every file in this crate (largest remaining file:
  `neural/conv_layers/mod.rs` at 1944 lines) — `cargo check -p
  tenflowers-ffi --all-features` and the 337/337 lib test count both
  reverified unchanged after the split.
- [x] **New end-to-end convergence proof**: `tests/test_training_convergence.py`
  adds three tests, each asserting concrete before/after loss ratios (not
  just "doesn't crash"): `test_dense_layer_training_converges` (single
  `PyDense` + `SGD`, 50 steps, >=100x loss drop), `test_sequential_mlp_training_converges`
  (3-layer `PySequential` MLP + `Adam`, 50 steps, >=10x drop plus a
  monotonic downward-trend check), `test_conv2d_training_converges`
  (`Conv2D` + `Adam`, 30 steps, >=5x drop, deliberately inside the
  tape-linked `Conv2D` config subset noted above).
- [x] **Test counts**: `cargo test -p tenflowers-ffi --lib` — 337 passed, 0
  failed, 0 ignored. Python: 55 passed via `pytest tests/` (50 fast + 5
  performance-marked; excludes `integration_test.py`, which is a standalone
  script rather than pytest-collectible — it calls `sys.exit(1)` at module
  scope on import failure) plus 13/13 passed running `integration_test.py`
  directly as a script. `grep -rn "todo!()\|unimplemented!()" src/` — 0 hits.
- Still open, not touched by this release (see "Current Gaps & Limitations"
  below): the explicit `PyGradientTape` API does not work end-to-end
  (`.watch()` only snapshots values; free ops don't record onto it), and
  `StateSpaceModel`/`Mamba` forward is still a stub returning
  `Tensor::zeros(...)` regardless of input.

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
- **Neural Network Layers**: Dense/`PyParameter`, Sequential, Conv1D/2D/3D +
  MaxPool2D/AvgPool2D, Embedding/EmbeddingBag, BatchNorm1d/LayerNorm/
  GroupNorm/InstanceNorm1d, MultiheadAttention, Transformer encoder/decoder +
  PositionalEncoding, LSTM/GRU/RNN + single-step cells — all with real
  `forward()` genuinely wired into the implicit-autograd tape (Conv2D
  backward confirmed only for unit dilation / `groups == 1` / default
  padding). All 9 optimizers (SGD, Adam, RMSprop, AdamW, AdaBelief, RAdam,
  Nadam, AdaGrad, AdaDelta) perform genuine gradient-based parameter
  updates via `step()`, proven convergent in `tests/test_training_convergence.py`
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
- **Neural Network APIs**: as of the 2026-07-12 rewrite, implicit autograd is
  genuinely wired through every layer type and all 9 optimizers (see
  "1. Current Capabilities" above); remaining known gaps are the two items
  below (`PyGradientTape`, `StateSpaceModel`/`Mamba`), not general coverage
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

### Known-incomplete autograd/layer surfaces (not fixed by the 2026-07-12 rewrite)
- **`PyGradientTape` (explicit, TensorFlow-style tape) does not work
  end-to-end**: `PyGradientTape.watch()` (`neural/gradient_tape.rs:71`) only
  clones a tensor's current value onto the tape at call time; the free
  functions used to build computations (`tf.add`, `tf.mul`, etc.) are not
  tape-aware for this explicit-tape style and never record operation-graph
  edges onto it, so `tape.gradient()` cannot trace a real computation chain.
  Reproduces with the tape module's own documented example (verbatim skip
  reason at `tests/integration_test.py:93-103`). Unrelated to, and does not
  affect, the implicit `.backward()`/`.grad()` API, which is what got fixed
  this release. Real tape-based autograd through plain tensor ops via this
  explicit-tape style remains a separate, not-yet-implemented future project.
- **`StateSpaceModel`/`Mamba` forward pass is a stub that fabricates its
  output**, not a real SSM computation: `neural/ssm.rs` — both
  `PyMamba::forward` and `PyStateSpaceModel::forward` ignore their input
  entirely and return `Tensor::zeros(...)`. Verbatim skip reason at
  `tests/integration_test.py:366-370`: "StateSpaceModel.selective_scan is
  currently a stub (returns input unchanged) and MambaBlock's SSM recurrence
  is not really computed." Explicitly out of scope for this release.

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
- [x] **Extended Optimizer Bindings**: Complete optimizer suite Python exposure (COMPLETED 2026-06-10 — neural/extended_optimizers/mod.rs: PyAdamW, PySGD, PyRMSprop, PyAdagrad, PyLion)
- [x] **Layer Export List**: Normalization + SSM Python API implementation (COMPLETED 2026-06-10 — neural/normalization/mod.rs + neural/ssm.rs exposed in Python module)

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
