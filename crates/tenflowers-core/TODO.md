# TenfloweRS Core TODO & Roadmap (v0.1.2)

**Version:** 0.1.2  
**Date:** 2026-07-08

v0.1.1 focus: core tensor engine capabilities and forward development plan.

## v0.1.2 — Honesty Hardening (2026-06-22)

- Removed production lock-poison panics (`.lock()/.read()/.write().unwrap()`)
  via signature-preserving recovery; a poisoned lock no longer aborts.
- Serialization checksum was hardcoded `Ok(0)` and never verified on read →
  real FNV-1a with verification on deserialize.
- `Tensor::randn` was a deterministic shifted-uniform (mean ≈ −0.5) → real
  standard-normal N(0,1); added seeded `randn_with_seed`.
- Correctness fixes: `gather` (whole-row), `slice` (non-contiguous),
  `segment_sum`/`segment_mean` (feature dim).
- Graph scheduling pass no longer falsely reports "changed".
- GPU honesty: Metal MPS matmul/conv2d/reductions + layer/group-norm +
  flash-attention (ran a kernel, returned un-read-back zeros) → honest errors;
  5 wrong-semantics GPU einsum ops + 7 `unreachable!()` einsum fallbacks →
  honest errors; 8 fabricated "measured" device-capability values → honest
  errors; `Tensor::to` GPU→GPU and ROCm `unreachable!()` → honest errors.
- RNG-state checkpoint capture → honest error (scirs2_core RNG has no
  serializable state).

## v0.1.2 — Later 2026-07-07 cycle additions

- `onnx_interop::{convert, lowering, proto}` landed: real protobuf
  import/export (`OnnxModel::to_protobuf`/`from_protobuf`) for
  `tenflowers-core`'s own graph representation, plus `lowering::{
  OnnxOpMapping, StandardOpMapping, lower_graph}` mapping parsed ONNX nodes
  to real ops. `OnnxImporter`/`OnnxExporter` file/byte round-trips are now
  real behind the `onnx` feature (previously always `NotImplemented`).
- `gradient_executor` module landed: `GradientExecutor` trait bridges
  `gradient_validation_framework` to a real `tenflowers-autograd`-backed
  implementation via `register_gradient_executor`/`get_gradient_executor`,
  without a circular crate dependency; unregistered state now reports
  honest "unverified" instead of fabricated `passed: true`.
- `session::SessionConfig::enable_graph_optimization` (default `true`)
  landed: wires constant folding, algebraic simplification, CSE, strength
  reduction, scheduling, and dead-code elimination into real session
  execution, with a `protected_output_roots` accumulator guaranteeing
  previously-fetched nodes survive later optimization passes.
- `ops::lapack_f64` landed: real LAPACK-backed `inverse_f64`,
  `determinant_f64`, `svd_f64`, `solve_f64` via `scirs2-linalg`.
- `device::{GpuAdapterCapabilities, get_gpu_adapter_capabilities}` landed:
  real, unprocessed `wgpu::Adapter` capability snapshot, replacing
  previously-hardcoded/fabricated GPU vendor and compute-capability
  detection in `gpu::advanced_kernel_manager`.
- CPU reductions gained real `StdDev`, `L1Norm`, `L2Norm` (previously
  `not_implemented` beyond Sum/Mean/Max/Min/Variance); segment reduction
  (`segment_max`/`segment_min`/`segment_sum`/etc.) now supports N-D `data`
  (`[N, d1, d2, ...]` → `[num_segments, d1, d2, ...]`), closing the
  "segment_max/min/prod/any/all for D>1" gap noted below.
- Two GPU-path crash bugs fixed: `gpu::ultra_fusion_integration::
  create_result_tensor` now uses `Tensor::from_gpu_buffer` instead of
  `from_storage` (which panics unconditionally for GPU storage);
  `from_gpu_buffer`'s device id is now derived from `buffer.device_enum()`
  instead of hardcoded `Device::Gpu(0)`; `ops::linalg::inv`'s GPU path
  no longer has a panicking `.expect(...)`.
- A wave of GPU ops that previously errored "not yet implemented" now do a
  genuine device→host readback and delegate to the CPU implementation:
  `ifft` (1D/2D/3D), `ops::matmul::{batch_matmul, dot}`,
  `ops::random::multinomial_f32`, `ops::einsum::gpu`'s wrappers (batched
  matmul, transpose, diagonal, outer product, trace).
- Graph optimization passes split from a single 1264-line
  `graph/optimization/passes.rs` into
  `graph/optimization/passes/{algebraic,constant_folding,cse,dead_code,
  mod,pass_support,scheduling,strength_reduction,tests}.rs` via SplitRS;
  `ConstantFoldingPass` now genuinely evaluates and replaces foldable
  constant subexpressions (previously only marked nodes as foldable
  without computing/substituting a result).

**Known gap, not yet resolved**: four new memory/perf diagnostic modules
landed on disk this cycle — `allocation_timeline.rs`, `memory_pressure.rs`,
`per_op_tracker.rs`, and top-level `pool_diagnostics.rs` (distinct from the
pre-existing `memory::pool_diagnostics`) — but none has a `pub mod` (or any
`mod`) declaration in `lib.rs`, so they are currently unreachable from the
public API. TODO: wire `allocation_timeline`/`memory_pressure`/
`per_op_tracker`/`pool_diagnostics` into `lib.rs`, resolving naming/overlap
with the existing `memory::pool_diagnostics` submodule before exposing it.

## 1. Current Capabilities

### Tensor Engine Foundation
- **Eager Execution**: Complete tensor operations (creation, arithmetic, reduction, manipulation)
- **Matrix Operations**: Blocked multiplication + outer product specialization + optional BLAS acceleration
- **SIMD Optimization**: 8-element chunking, unchecked fast paths, comprehensive mathematical functions
- **Memory Management**: Reference counting, buffer reuse metrics, allocation tracing capabilities
- **Error Handling**: Zero-warning baseline, consolidated error patterns across operations

### Modular Architecture
- **16/21 Large Files Refactored**: Successfully modularized major components
- Systematic approach established: analyze -> modularize -> test -> backup

### GPU & Performance
- **GPU Support**: Partial WGSL compute kernels with safe CPU fallbacks
- **Cross-Platform**: WGSL/WebGPU backend enabling portable GPU acceleration across vendors
- **Operation Fusion**: Elementwise fusion across 16 operations to reduce kernel launches and memory traffic
- **SciRS2 Foundation**: Built on scirs2-core primitives for GPU buffers and device abstraction
- **Shape Diagnostics**: Shape error taxonomy with standardized categories and fix suggestions
- **Documentation**: GPU usage examples and integration guide

## 2. Current Gaps & Limitations

### Core Infrastructure
- **GPU Coverage**: Many operations still fall back to CPU (the CPU result is real); partial kernel coverage across the operation set. A growing subset of previously-unimplemented GPU ops (`ifft`, `batch_matmul`, `dot`, `multinomial_f32`, several `einsum::gpu` wrappers) now do a genuine device→host readback + CPU delegation rather than erroring, closing the gap for those specific ops.
- **Error Taxonomy**: Error categories and fix suggestions need further standardization across modules
- **Mixed Precision**: Casting policies and numerical stability are still maturing
- **Graph Execution**: `session::SessionConfig::enable_graph_optimization` now wires a real optimization pipeline (constant folding, algebraic simplification, CSE, strength reduction, scheduling, dead-code elimination) into session execution — the "no full graph execution/optimization mode" gap is resolved. Remaining work is expanding pass coverage/tuning, not landing the mode itself.

### Honest-error deferrals (post-2026-06-22 sweep; fail loudly, not faked)
- **Metal MPS GPU→host readback**: matmul, conv2d, reductions, layer/group-norm, flash-attention return an honest error on GPU (CPU paths are real).
- **GPU einsum correctness**: previously-wrong GPU einsum ops and their `unreachable!()` fallbacks now error instead of returning bad data. (Several other `einsum::gpu` wrappers — batched matmul, transpose, diagonal, outer product, trace — now delegate to the real CPU `einsum` instead of erroring; the fallback/honest-error path remains for wrong-semantics cases not yet reworked.)
- **Device-capability queries**: real bandwidth / tensor-core / cache values are still not queried (were fabricated) → honest error. Note: `device::get_gpu_adapter_capabilities` (new this cycle) provides a real, unprocessed `wgpu::Adapter` snapshot (`AdapterInfo`/`Limits`/`Features` — workgroup size, shared-memory size, buffer-size caps, optional feature flags), but `wgpu` itself has no API for raw memory-bandwidth, tensor-core-presence, or cache-size numbers, so that narrower gap remains.
- **`Tensor::to` GPU→GPU** and **ROCm** unsupported paths → honest error.
- **RNG-state checkpoint capture**: scirs2_core RNG exposes no serializable state → honest error.
- **`segment_max/min/prod/any/all`** for feature width D>1: **resolved this cycle** — segment reduction now supports N-D `data` (`[N, d1, d2, ...]` → `[num_segments, d1, d2, ...]`), not just the 1-D case.

## 3. Post-v0.1.0 Roadmap

### Priority 1: GPU Kernel Coverage
1. Expand WGSL compute kernel coverage to reduce CPU fallbacks
2. Standardize kernel dispatch and device abstraction across operations
3. Backend portability validation across Vulkan/Metal/DX12 via WebGPU

### Priority 2: Operation Graph & Scheduling
4. Operation fusion expansion beyond elementwise paths
5. Lazy/deferred execution prototype for kernel batching
6. Operation graph/IR abstraction layer
7. Advanced operation scheduling optimization

### Priority 3: Performance & Memory
8. Mixed precision stability improvements + automatic casting policies
9. Advanced linear algebra: batched factorizations, sparse primitives
10. Host <-> device memory spill strategy

### Priority 4: Advanced Features
11. ~~Full graph execution mode with optimization~~ — landed this cycle via `session::SessionConfig::enable_graph_optimization`; remaining follow-on work is pass-coverage expansion (see Priority 2)
12. Multi-GPU orchestration foundations
13. Quantization pipeline infrastructure
14. Performance regression gates in CI (criterion-based)

## 4. Remaining Large File Refactoring (5/21 files)

Priority files for future modular extraction (none is over the 2000-line hard limit yet, but all are refactoring candidates):
- `gpu/advanced_kernel_manager.rs` (~1800 lines) — kernel lifecycle and dispatch management (grew this cycle with `GpuAdapterCapabilities`-based vendor detection)
- `gpu/rocm_kernels.rs` (~1220 lines) — ROCm/HIP compute kernel definitions
- `ops/benchmark.rs` (~1180 lines) — operation benchmarking harness
- `async_gpu_optimizations.rs` (~1180 lines) — asynchronous GPU execution paths
- `memory/ultra_cache_optimizer.rs` (~1170 lines) — cache optimization strategies

Note: `graph/optimization/passes.rs` (formerly ~1264 lines, a 6th former candidate) was split this cycle via SplitRS into `graph/optimization/passes/{algebraic,constant_folding,cse,dead_code,mod,pass_support,scheduling,strength_reduction,tests}.rs`.

---

Copyright 2025-2026 COOLJAPAN OU (Team KitaSan)
