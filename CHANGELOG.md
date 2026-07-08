# Changelog

All notable changes to TenfloweRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### Added

### Changed

### Fixed

### Security

## [0.1.2] - 2026-07-08

### Added

#### Meta-Crate (`tenflowers`)
- `tenflowers::error` module: `FrameworkError` unified error enum wrapping all subcrate errors, plus `Result<T>` type alias
- `tenflowers::logging` module: `LogLevel` enum, `set_log_level`, `current_log_level`, `set_log_backend`, `init_from_env` (reads `TENFLOWERS_LOG`), `emit`; `log_info!`, `log_warn!`, `log_error!`, `log_debug!`, `log_trace!` macros
- `tenflowers::utils` module: common utility functions (`softmax`, `sigmoid`, `bytes_to_human_readable`, etc.)
- `tenflowers::platform` module: `current_platform`, `detect_simd_capabilities` for runtime CPU introspection
- `tenflowers::version_check` module: `check_version_consistency`, `assert_versions_consistent` for subcrate version alignment
- `ml_compat` integration test suite (549 lines) and `feature_flags_comprehensive` test suite (568 lines)
- Examples: `basic_example`, `logging_example`, `version_check_example`

#### FFI Crate (`tenflowers-ffi`)
- `profiling` module: session-based `PyProfiler` / `PyProfileRecord` / `PyProfileReport`; records op name, duration (ns), memory bytes, device; thread-safe via `Arc<Mutex<_>>`
- `stable_api` module: stable C/Python API surface (396 lines)
- Criterion benchmark harness (`benches/bindings_bench.rs`) added to `tenflowers-ffi`
- `implicit_autograd` module: PyTorch-style eager `.backward()` / `.grad()` for `PyTensor`, layered on top of the existing explicit `GradientTape` engine. A thread-local, auto-activating `GradientTape` (`IMPLICIT_TAPE`) plus three side-tables (`TRACKED_REGISTRY`, `LEAVES`, `GRAD_STORE`) key every tracked tensor by its `Arc<Tensor<f32>>` allocation address rather than adding a field to `PyTensor` (which would have touched ~190 struct-literal call sites); a fourth table, `IDENTITY_ANCHORS`, keeps that address's `Arc` allocation alive for as long as any table still references it, closing an address-reuse hazard where a freed tensor's stale gradient could otherwise be attributed to a new, unrelated tensor allocated at the same address (caught during development and covered by a dedicated regression test). New `PyTensor` methods: `set_requires_grad` (now begins implicit tracking via `mark_leaf`), `backward()`, `grad()`. `PyTensor::add`/`sub`/`mul`/`div`/`matmul` and `sum`/`mean`/`relu`/`sigmoid`/`tanh` now record themselves onto the implicit tape via `record_and_link_binary`/`record_and_link_unary` — no new gradient math, just wiring into the existing tape/`TrackedTensor` engine
- `gradient_parity` module: standalone, pure-Rust (no PyO3) finite-difference gradient checker — `GradientParityChecker` (configurable `atol`/`rtol`/`h`, central or forward differences), `GradientParityResult` (`max_abs_error`, `max_rel_error`, `rms_error`, `passed`, per-element errors, `format_report()`), plus free functions `gradients_are_close` and `numeric_jacobian`; used by the new `examples/gradient_example.rs` and the Criterion bench harness
- Examples: `examples/basic_ops.rs` (tensor creation/arithmetic via `tenflowers_core::ops` directly), `examples/gradient_example.rs` (`GradientParityChecker` demo), `examples/optimizer_example.rs` (manual SGD and single-step Adam)
- `neural::attention::combine_attention_masks`-backed masking is now real end-to-end in `PyMultiheadAttention::forward` and the transformer encoder/decoder layers: `key_padding_mask` previously reached the mask-shape validator but was silently dropped before the actual softmax, so a "masked" key still received real attention weight; masks are now genuinely combined into one additive bias before scoring (regression-tested with hand-computed attention-weight assertions)
- `PyMultiheadAttention`/transformer layers: GRU forward now threads a caller-supplied initial hidden state through every layer instead of ignoring it; `RNN` forward now honors its `nonlinearity` ("tanh" | "relu") parameter for real (previously stored and validated but never consulted); `PyMaxPool2D`/`PyAvgPool2D` gained an integer-exact `ceil_mode` implementation (`pooling_output_len`) with a boundary-divisor correction (`avg_pool2d_boundary_correct`) matching PyTorch's "exclude ceil-mode overhang from the average" convention, and dilated max-pooling now genuinely dilates the sampling window
- 46 new `#[pyo3(signature = (...))]` annotations across `math_ops.rs`, `neural/functions.rs`, `neural/layers.rs`, `neural/recurrent.rs`, `neural/transformer.rs`, `neural/embedding.rs`, `neural/conv_layers.rs`, `visualization/mod.rs`, `profiling.rs`, `memory_optimizer.rs`, fixing functions whose `Option<T>` parameters had Rust-side `.unwrap_or(default)` fallback logic but no pyo3 default, so callers were forced to pass every argument explicitly (e.g. `tf.sum(x)` without `dim`/`keepdim` previously raised `TypeError: missing required argument`)
- `arange()` / `linspace()` (in `utils.rs`) now return a real `PyTensor` instead of a plain Python list, matching the `zeros`/`ones`/`rand` return convention

#### Dataset Crate (`tenflowers-dataset`)
- `formats::hdf5_advanced`: `Hdf5AttributeValue`, `CompressionKind`, `DatasetInfo`, `Hdf5ChunkedReader`, `Hdf5AttributeReader`, `Hdf5TreeWalker`, `Hdf5SliceReader`
- `formats::parquet_advanced`: `FilterPredicate`, `ColumnDtype`, `ColumnInfo`, `ParquetSchemaInfo`, `RowGroupBatch`, `ParquetRowGroupReader`, `read_columns`, `read_filtered`, `inspect_schema`
- `sha2` 0.11.0 dependency for download-integrity verification (SHA-256, pure Rust)
- `thiserror` dependency for structured error types
- `formats::blosc`: from-scratch, pure-Rust Blosc chunk **decoder** for Zarr arrays (encoder not implemented). Implements the full C-Blosc container format (verified directly against upstream `blosc/blosc.c`/`blosc/blosc.h`/`README_CHUNK_FORMAT.rst`) plus all five inner codecs — `BloscLz` (own hand-written, bounds-checked port of `blosclz_decompress`), `Lz4` (via the new `oxiarc-lz4`), `Snappy` (via the new `oxiarc-snappy`), `Zlib` (via the new `oxiarc-deflate`), and `Zstd` (via `oxiarc-archive`'s zstd, with a documented pre-existing `oxiarc-zstd` 0.3.3 panic on some short frames contained via `catch_unwind`) — and both byte-shuffle and bit-shuffle pre-filters (`unshuffle`/`bitunshuffle`, ported from `blosc/shuffle-generic.h`/`bitshuffle-generic.c`). Public surface: `pub fn decompress(compressed: &[u8]) -> Result<Vec<u8>>`. Validated against real chunks produced by the reference C-Blosc library. Wired into `formats::zarr`'s `"blosc"` compressor dispatch (previously a stub that returned the still-compressed bytes unchanged)
- `formats::tfrecord_advanced`: `TfRecordSequenceReader`/`SequenceExample`/`SequenceStep` for `SequenceExample`-shaped records (context features + per-step feature lists); `TfRecordRawReader`/`RawTfRecord` low-level streaming reader; `masked_crc32`/`verify_masked_crc32` implementing TensorFlow's real masked-CRC record-integrity check; `parse_feature_proto` with correct `BytesList`/`FloatList`/`Int64List` varint-length-delimited decoding
- `gpu_transforms::affine` / `::perspective` / `::elastic` / `::equalize`: new GPU-accelerated image transforms — 2×3 affine warp (`GpuAffineTransform` + composable `affine_translation`/`affine_scale`/`affine_rotation`/`affine_shear`/`affine_compose` builders), 3×3 homographic/perspective warp (`GpuPerspectiveTransform`, `homography_from_quad`), SimoNikolenko-style elastic distortion (`GpuElasticDistortion`), and two-pass per-channel histogram equalization (`GpuHistogramEqualize`) — each backed by a new WGSL shader (`shaders/{affine,perspective,elastic_distortion,histogram_equalize}.wgsl`) with a non-GPU-feature stub fallback
- `simd_transforms::functional`: SIMD-friendly (auto-vectorized, no unsafe intrinsics) preprocessing free functions — `simd_normalize_chw`, `simd_standardize`, `simd_scale_offset_inplace`, `simd_clamp_inplace`, `simd_hwc_to_chw`/`simd_chw_to_hwc`, `simd_u8_to_f32_normalized`, `simd_rgb_to_grayscale`, `simd_mean_variance`
- `transform_arena`: `TransformArena`, a preallocated, `Rc`/`RefCell`-based arena for reusable transform intermediate buffers, with `acquire`/`stats`/`reset`/`reuse_ratio` and an RAII `ArenaGuard`
- `arrow_advanced::ArrowPredicate::In(column, values)` — previously fell through to a generic "not yet implemented" error; now ORs together per-value equality masks

#### Core Crate (`tenflowers-core`)
- `onnx_interop::{convert, lowering, proto}`: real ONNX protobuf import/export for `tenflowers-core`'s own graph representation (distinct from `tenflowers-neural`'s separate `Sequential`-model ONNX subsystem — see Fixed, below). `convert.rs` implements `OnnxModel::to_protobuf`/`from_protobuf` (and matching methods on `OnnxGraph`/`OnnxNode`/`OnnxTensor`/etc.) against a hand-rolled `proto::ModelProto` schema, rejecting unrecognized ONNX attribute/dtype codes with a descriptive `Err` rather than the silent `Float(0.0)`/`Float32` coercion bugs present in the reference `tenflowers-neural` implementation it improves on. `lowering.rs` adds `OnnxOpMapping`/`TenfloweRSOperation` trait implementations (`StandardOpMapping`, `lower_graph`) mapping parsed ONNX nodes (`Add, Sub, Mul, Div, Relu, Sigmoid, Tanh, MatMul, Reshape, Transpose, Identity, Concat`, plus `Softmax, Flatten, Gemm`) to real tenflowers-core ops — operates on in-memory structs only, so it is available even without the `onnx` feature. `OnnxImporter::import_from_{file,bytes}` / `OnnxExporter::export_to_{file,bytes}` now perform real protobuf decode/encode behind the `onnx` feature (previously always returned `NotImplemented`, regardless of the feature flag)
- `gradient_executor` module: new `GradientExecutor` trait — a dependency-inversion bridge letting `tenflowers-core`'s `gradient_validation_framework` call into a real forward+backward implementation living in `tenflowers-autograd` (which depends on `tenflowers-core`, not the reverse) via `register_gradient_executor`/`get_gradient_executor`, without introducing a circular crate dependency. `tenflowers-autograd` registers a real, `GradientTape`-backed implementation (`AutogradGradientExecutor`, supporting `add/sub/mul/div/matmul/pow`, `relu/sigmoid/tanh`, and `"->"`-composed unary chains) via a new `tenflowers_autograd::init()` entry point; without a registered executor, the validation framework now reports an honest "unverified" status instead of fabricating `passed: true`
- `session::SessionConfig::enable_graph_optimization` (default `true`): wires the existing graph-optimization passes (constant folding, algebraic simplification, CSE, strength reduction, scheduling, dead-code elimination) into real session execution for the first time. A new `protected_output_roots: HashSet<NodeId>` accumulator guarantees every node a session has ever been asked to fetch survives optimization on later calls with a different fetch list (e.g. CSE will not merge away a previously-fetched node)
- `ops::lapack_f64`: real LAPACK-backed f64 linear-algebra ops via `scirs2-linalg` — `inverse_f64`, `determinant_f64`, `svd_f64`, `solve_f64` (matrix inverse, determinant, SVD, and linear-system solve)
- `device::{GpuAdapterCapabilities, get_gpu_adapter_capabilities}`: real, unprocessed `wgpu::Adapter` capability snapshot (`AdapterInfo`/`Limits`/`Features`) with no vendor-specific guessing; used by `gpu::advanced_kernel_manager` to replace previously-hardcoded/fabricated GPU vendor and compute-capability detection
- `gpu::gpu_device_available()`: real runtime probe for whether a *usable* GPU device can actually be created — deliberately stronger than `wgpu::Instance::request_adapter()` succeeding, since that alone returns `Ok` in headless/sandboxed environments (e.g. via GL/EGL) where the subsequent `request_device` then fails with "Parent device is lost"; performs the same adapter + `request_device` creation used by the rest of the crate's GPU paths and caches the result for the process lifetime via `GpuContext::global` (see Fixed for callers migrated onto it)
- CPU reduction ops gained real `StdDev`, `L1Norm`, `L2Norm` implementations (previously fell through to a generic `not_implemented` error for any op beyond Sum/Mean/Max/Min/Variance)
- Segment reduction (`segment_max`/`segment_min`/`segment_sum`/etc.) now supports N-D `data` (`[N, d1, d2, ...]` → `[num_segments, d1, d2, ...]`), not just 1-D

> **Known gap**: four new memory/perf diagnostic modules landed on disk this cycle — `allocation_timeline.rs`, `memory_pressure.rs`, `per_op_tracker.rs`, and top-level `pool_diagnostics.rs` (distinct from the pre-existing `memory::pool_diagnostics`) — but none of them has a `pub mod` (or any `mod`) declaration in `lib.rs`, so they are currently unreachable from `tenflowers-core`'s public API. Not advertised as Added above; tracked for a follow-up wiring pass.

### Changed

- `numrs2` updated 0.3.3 → 0.4.0
- `scirs2-core / -autograd / -neural / -linalg / -numpy` updated 0.4.2 → 0.5.0 → **0.6.0**
- `oxicode` updated 0.2 → 0.2.4
- `oxiarc-archive` updated 0.2.6 → 0.3.3 → 0.3.4 → **0.3.5**
- `oxifft` updated 0.2.0 → 0.3.2
- `wgpu` updated 29.0 → **30.0**
- `pyo3` updated 0.28 → **0.29** (plus new `pyo3-build-config = "0.29"` dependency); comment added noting it must track `scirs2-numpy`'s own pyo3 requirement, since both link against the same native `python` library
- `arrow` / `parquet` updated 58.1 → **59.0**
- `prost` / `prost-build` / `prost-types` updated 0.14.3 → **0.14.4**
- `symphonia` updated 0.5 → **0.6**; `rubato` updated 2.0 → **3.0**
- New `oxiarc-lz4`, `oxiarc-deflate`, `oxiarc-snappy` dependencies (0.3.4 → **0.3.5**) — direct deps needed for Blosc's inner codecs, which use raw block/stream formats that the `oxiarc-archive` umbrella crate does not expose directly (it only exposes the framed/container forms of lz4 and snappy)
- `tenflowers-ffi`'s `pyo3-build-config` build-dependency now uses `{ workspace = true }` instead of a locally-pinned version (COOLJAPAN workspace-policy compliance)
- `tenflowers-autograd` gained a new `distributed = ["tokio"]` feature flag, gating a `pub mod distributed_replication` re-export of the existing cross-datacenter replication module
- `tenflowers-dataset`'s `compression` feature now also pulls in `oxiarc-lz4`/`oxiarc-deflate`/`oxiarc-snappy` (previously only `oxiarc-archive`)
- `CheckpointManager` API (`tenflowers-autograd`): `should_checkpoint`, `save_checkpoint`, `restore_checkpoint`, `remove_checkpoint`, `clear_checkpoints`, `memory_usage`, `checkpoint_count` now return `Result<T>` instead of panicking on lock poisoning
- NCCL backend (`tenflowers-neural`): `all_reduce`, `all_gather`, `broadcast`, `send`, `recv` now return honest `NotImplemented` errors (were silently returning cloned/fabricated data); `init_gpu_devices` reads `CUDA_VISIBLE_DEVICES` instead of hardcoding 4 GPUs
- Gloo, MPI, and thread-based communication backends (`tenflowers-neural`) get the same honest-error treatment as NCCL: `GlooBackend`/`MpiBackend`'s `all_reduce`/`all_gather`/`broadcast`/`send_f32`/`recv_f32` and `ThreadBackend`'s `simulate_all_reduce`/`send_f32`/`recv_f32` now return `NotImplemented` instead of silently scaling, echoing, or cloning the caller's own local tensor as if it were a genuine cross-rank result (no backend in this crate links a real collective-communications runtime). `ThreadBackend`'s dead `mpsc`-channel scaffolding — whose receiver half was dropped immediately in `create_group`, so sends always silently no-op'd — was removed rather than left in place
- `DataParallelTrainer::train_step` (`tenflowers-neural`) previously only simulated the backward pass (marked gradient buckets "ready" without computing anything) and zeroed gradients as a placeholder optimizer step; it now computes real per-parameter gradients via central finite differences against the actual mini-batch loss, all-reduces them, and calls the real optimizer step — parameters now genuinely learn
- `deployment::pruning::engine` (`tenflowers-neural`): every pruning strategy previously computed statistics from synthetic simulated data and returned a fabricated, unmodified `Sequential::new(vec![])`; `Magnitude`/`Random` pruning now genuinely zero real weight tensors based on actual magnitude quantiles or a seeded RNG, while `Structured`/`Gradual`/`LotteryTicket` honestly report `NotImplemented` with a specific rationale instead of fabricating results
- `layers::moe` (`tenflowers-neural`): `TopKRouter` now performs real top-k expert selection with a genuine Switch-Transformer/GShard-style load-balancing loss, and `MixtureOfExperts::forward` now evaluates and combines every selected expert (previously routed all tokens to expert 0 unconditionally)
- `serialization::weight_loader` (`tenflowers-neural`): implemented a real, versioned JSON weight save/load format (previously every format silently no-op'd — `load_*` returned an empty weight map with a warning, `save_*` wrote nothing while returning `Ok`); `Binary`/`SafeTensors`/`NumPy` now honestly report `NotImplemented` naming the missing dependency instead of silently claiming success
- `tensorflow_compat::SavedModelLoader` (`tenflowers-neural`): `load_from_pb`/`parse_pbtxt_content` no longer fabricate a `SavedModel` with hardcoded metadata regardless of input; `convert_operation_to_layer` now derives real layer dimensions from the operation's actual weight tensor instead of hardcoded placeholder dimensions, honestly erroring when no matching weight variable exists
- Graph optimization passes (`tenflowers-core`) split from a single 1264-line `graph/optimization/passes.rs` into `graph/optimization/passes/{algebraic,constant_folding,cse,dead_code,mod,pass_support,scheduling,strength_reduction,tests}.rs` via SplitRS; `ConstantFoldingPass` now genuinely evaluates and replaces foldable constant subexpressions (previously only marked nodes as foldable without ever computing or substituting a result)

### Fixed

- Lock-poisoning `.expect()` calls replaced with `map_err` + proper `Result` propagation in `CheckpointManager`, `CrossDatacenterReplicator`, `DeterministicContext`, and `tenflowers-core`'s `Context` (`get_context`/`set_context`, `Context::set_attribute`/`get_attribute`); `set_context` and `set_attribute` now return `Result<()>` and `get_attribute` now returns `Result<Option<String>>` (breaking change for direct callers)
- `gradient_visualization_example.rs`: gradient-tape `Mutex` locks no longer panic on poison; propagated via `?`
- `numerical_gradient_validation_example.rs`: `property_test` signature now requires explicit analytical gradient (`f_grad`) argument
- `CrossDatacenterReplicator`/`DatacenterConnection` (`tenflowers-autograd`): network-simulation methods (`broadcast_prepare`/`broadcast_commit`, `aggregate_parameters*`, `get_current_step`, `get_datacenter_steps`, `collect_all_parameters`, `get_health`, `DatacenterConnection::new`, bandwidth/congestion/RTT helpers) previously fabricated successful cross-datacenter coordination (fake `PrepareResult`, pass-through "aggregation", hardcoded step counters, always-healthy status) with no real network transport; all now return honest `NotImplemented` errors describing the missing transport, and several gained `Result` return types as a result (breaking change for direct callers)
- `numerical_checker::property_test` (`tenflowers-autograd`): gained a required `f_grad` closure parameter and now genuinely compares the analytical gradient against it — previously the "analytical" value was just a clone of the numerical estimate, so every property test trivially passed regardless of correctness
- `kernel_fusion::FusableOp::Mish` (`tenflowers-autograd`): was computed as a `tanh` approximation (numerically wrong: `mish(2.0)` ≈ 1.944 vs. `tanh(2.0)` ≈ 0.964) — now calls the real `mish` op; `BatchNorm`/`LayerNorm`/`GroupNorm`/`Conv2D`/`Linear`/`Scale`/`Bias` fused ops previously silently passed input through unchanged as a fake identity — now return an honest `Err` instead
- Conv2D/Conv3D backward gradients (`tenflowers-autograd`): `compute_conv2d_input_gradient`/`compute_conv2d_weight_gradient`/`compute_conv3d_input_gradient`/`compute_conv3d_weight_gradient` were zero-tensor placeholders (and the tape's Conv2D backward path additionally used an outright-wrong identity/zero shortcut) — all now real, correct transposed-cross-correlation implementations, validated by new finite-difference gradient-check tests
- `neural_integration::trainer::compute_accuracy` (`tenflowers-autograd`): was a hardcoded constant (`0.95`) regardless of predictions — now a real argmax-based (multi-class) or threshold-based (binary) accuracy computation
- `MPSNeuralOps` training-forward path (`tenflowers-core`, Metal): every layer type (`Dense`, `Convolution`, `BatchNorm`, `LayerNorm`) either filled a correctly-shaped tensor with zeros or silently passed the input through while claiming to have normalized/activated it; now returns an honest, layer-specific `Err` naming the missing Metal MPS kernel instead of fabricating output
- `WasmBundleOptimizer::optimize_for_edge`/`optimize_for_minimal_size` (`tenflowers-core`): previously fabricated "bytes saved" reports from hardcoded per-category constants (e.g. always "150KB dead code eliminated") that never varied with any real input; now return an honest `Err` explaining that no compiled `.wasm` binary is ever available to analyze
- Two GPU-path crash bugs (`tenflowers-core`): `Tensor::from_storage` panics unconditionally for GPU-resident storage — `gpu::ultra_fusion_integration::create_result_tensor` built `TensorStorage::Gpu` and handed it to `from_storage` directly, guaranteeing a crash on every real invocation; now uses `Tensor::from_gpu_buffer`, the correct constructor for GPU storage. Separately, `Tensor::from_gpu_buffer`/`tensor::creation`'s GPU path hardcoded `Device::Gpu(0)` regardless of the buffer's actual device id — now derived from `buffer.device_enum()`; `ops::linalg::inv`'s GPU path also removed an `.expect("GPU linalg context must be initialized")` that could panic
- A wave of GPU-path operations that previously errored with "not yet implemented" now perform a genuine device→host readback and delegate to the (already-correct) CPU implementation instead: `ifft` (1D/2D/3D), `ops::matmul::{batch_matmul, dot}`, `ops::random::multinomial_f32`, `ops::einsum::gpu`'s wrappers (batched matmul, transpose, diagonal, outer product, trace — all now delegate to the exact, independently-tested `einsum` CPU entry point)
- `ops::activation::functions::mish`'s GPU path removed an `unsafe { std::mem::transmute::<&GpuBuffer<T>, &GpuBuffer<f32>> }` type-pun that backed a code path which always errored internally and was dead code; replaced with the same safe host round-trip used for other unsupported-on-GPU activations
- `ops::fft::inplace` (`tenflowers-core`): all six in-place FFT entry points — `fft_inplace`/`ifft_inplace`/`fft2_inplace`/`ifft2_inplace`/`fft3_inplace`/`ifft3_inplace` — were complete no-op placeholders (`Ok(())` returned immediately, input parameter unused); now genuinely compute the transform via `oxifft::Plan` and write the result back into `*input` on the CPU path, with a GPU path that round-trips through the host and delegates to the same real CPU implementation
- `Device::try_gpu` (`tenflowers-core`) unconditionally returned `Ok(Device::Gpu(gpu_id))` regardless of whether a GPU could actually be created; it and the `gpu_adapter_available`/`gpu_available` test helpers in `advanced_kernel_manager.rs` and `gpu_reduction_integration.rs` now probe genuine device creation (adapter *and* `request_device`, not just adapter enumeration) via the new `gpu::gpu_device_available()` (see Added), returning an honest `Err`/`false` instead of a false positive. `dispatch_init::register_gpu_reductions` was registering only GPU kernels for `sum`/`mean`, so a CPU tensor would incorrectly dispatch to the GPU kernel and fail with "device lost" on hosts with the `gpu` feature enabled but no real device present — real CPU kernels (`sum_f32_cpu`/`mean_f32_cpu`) are now registered too. `AsyncBinaryOperationExecutor::new` now degrades to an honest CPU-only executor when no GPU context can be created instead of propagating a hard error
- `memory_pool::MemoryBlock`/`MemoryPool` (`tenflowers-dataset`): `MemoryBlock::new` hardcoded 1-byte alignment for every allocation regardless of the type that would later reinterpret it — confirmed by Miri as undefined behavior for any `T` needing more than 1-byte alignment. New `MemoryPool::allocate_aligned(size, align)` and a private `allocate_exact(size, align)` (bypassing the size-class recycling pool, which could otherwise hand back a larger-than-requested block and produce a mismatched alloc/dealloc `Layout` — a second, independent flavor of UB) fix both issues; `with_pool_capacity::<T>()` now correctly requests `mem::align_of::<T>()`
- `arrow_advanced` (`tenflowers-dataset`): predicate evaluation (`evaluate_equals`/`evaluate_greater_than`/`evaluate_less_than`) previously rejected any comparison between a 1-row scalar and an N-row column outright under arrow 59.0's `compare_op` ("Cannot compare arrays of different lengths"), making the predicate subsystem unusable on any batch with more than one row; now wraps the scalar in `arrow::array::Scalar::new(...)` and works correctly, with consistent type coverage across all three comparison functions
- `arrow::ArrowArrayExt::to_tensor` (`tenflowers-dataset`): was an unconditional `Err("not implemented for this array type")` stub; now delegates to a shared, real Arrow-array-to-Tensor conversion path
- `formats::audio::get_audio_info` (`tenflowers-dataset`): previously fabricated `channels = 1`, `duration = 1.0`, etc.; now probes/decodes the container via Symphonia to derive real `sample_rate`/`channels`/`num_samples`/`duration`
- `formats::zarr` (`tenflowers-dataset`): LZ4 and Zstd chunk decompression previously silently returned the still-compressed bytes unchanged; now decompress for real via `oxiarc_archive::{Lz4Reader, ZstdReader}`, and an unrecognized codec now returns an honest `Err` instead of passing data through unchanged
- `transforms::noise::AddNoise::transform` (`tenflowers-dataset`): always added zero noise regardless of configuration (computed then discarded a real noise tensor); now generates genuine Gaussian noise via a Box-Muller transform
- `real_datasets::download::verify_checksum` (`tenflowers-dataset`): was a no-op that always reported success; now computes a real SHA-256 or CRC-32 digest (algorithm inferred from a `"algo:hex"` prefix or digest length) and honestly errors if it can't be determined
- `numa_scheduler` (`tenflowers-dataset`): Linux-affinity code was gated on `target_os = "linux"` alone despite depending on `libc`, which is only linked under the `numa` feature — now correctly gated on both, fixing a build failure on Linux without `--features numa`
- ONNX protobuf handling (`tenflowers-neural::serialization::onnx`, a subsystem distinct from `tenflowers-core::onnx_interop` above): `OnnxLoader::load_from_file`/`load_from_bytes` now perform real prost-based protobuf decoding behind the `onnx` feature (previously hardcoded `Err`s); `convert_weights` now does real byte-level reinterpretation of ONNX initializer data into `Tensor<T>` via `bytemuck`, validating shape/dtype/length instead of silently building nothing; `OnnxValueInfo::from_protobuf`/`OnnxTensor::from_protobuf`/`OnnxAttribute::from_protobuf` (`tenflowers-neural::onnx::data`) now reject unrecognized protobuf type codes with an error instead of silently defaulting to `Float32`/`Float(0.0)`
- Two independent hardcoded Python version strings: `tenflowers-ffi`'s compiled-module `__version__` setattr, and a separate literal `__version__ = "0.1.0"` in `python/tenflowers/__init__.py` that unconditionally overrode it — both now derive from `env!("CARGO_PKG_VERSION")` / the Rust-side version at runtime
- `python/tenflowers/__init__.py`: dtype constants (`float32`, `float64`, `int8`..`uint64`, `bool`, etc.) are set on the compiled module via `setattr` rather than `add_function`/`add_class`, so pyo3's auto-generated `__all__` never listed them and `from tenflowers import *` silently skipped them; now explicitly re-imported and re-exported
- Remaining hardcoded `"0.1.1"` version-string literals replaced with `env!("CARGO_PKG_VERSION")` in `tenflowers-neural`'s `onnx::model`, `serialization::schema` (test fixture), and `deployment::mobile` (`OnnxMobileExporter` producer metadata)

### Security

- **RUSTSEC-2026-0176** and **RUSTSEC-2026-0177** (`pyo3` 0.28.3): resolved by this release's `pyo3` 0.28 → 0.29 upgrade (see Changed) — the `scirs2-numpy` transitive pin that previously blocked upgrading past 0.28.3 was lifted once `scirs2-numpy` itself moved to 0.6.0.
- **RUSTSEC-2026-0204** (`crossbeam-epoch` 0.9.18, published 2026-07-06): invalid pointer dereference in the `fmt::Pointer` impl for `Atomic`/`Shared` when the underlying pointer is invalid. Pulled in transitively via `scirs2-core 0.6.0` → `crossbeam-deque` → `crossbeam-epoch`; fix requires upgrading to `crossbeam-epoch` ≥ 0.9.20, which is not a direct dependency of this workspace. Tracked for resolution once upstream `scirs2-core`/`crossbeam` lift the pin.
- **RUSTSEC-2024-0384** (`instant` 0.1.13): Unmaintained crate (transitive via `hdf5`). Not directly exploitable; tracked for future cleanup.
- **RUSTSEC-2024-0436** (`paste` 1.0.15): Unmaintained crate (transitive via `rav1e`, `parquet`, `metal`). Not directly exploitable; tracked for future cleanup.

## [0.1.1] - 2026-04-24

### Added

#### Meta-Crate Ergonomics (`tenflowers`)
- `tensor![]` declarative macro for shape-inferred tensor creation (1-D, 2-D, nested, explicit dtype)
- Type aliases: `Tensor1D<T>`, `Tensor2D<T>`, `Tensor3D<T>`, `Tensor4D<T>`, `Vector`, `Matrix`, `BatchTensor`
- Expanded prelude: `MultiHeadAttention`, `RMSNorm`, `TransformerEncoder/Decoder`, `GRU`, `LSTM`, `RNN`, `Optimizer`, `RandomSampler`, `DType`
- `tenflowers::interop::ndarray` — `from_ndarray` / `to_ndarray` conversion utilities
- `tenflowers::io` — `save_tensor` / `load_tensor` convenience wrappers
- `tenflowers::onnx` — ONNX re-exports under `onnx` feature gate
- `deprecated_use!` macro for structured deprecation notices
- Feature presets: `experimental`, `minimal`, `standard`

#### Documentation
- Getting Started tutorial and PyTorch↔TenfloweRS API mapping table (35 rows) in README
- Mermaid architecture diagram (crate dependency graph with feature-gate annotations)
- `docs/QUICK_REFERENCE.md` — 10-section cheat sheet
- `tenflowers/docs/MIGRATION_FROM_TENSORFLOW.md` — 5 side-by-side TF→Rust scenarios
- `tenflowers/docs/TROUBLESHOOTING.md` — 10 symptom-fix triples
- `tenflowers/docs/PRELUDE_STABILITY.md` — semver stability policy for prelude

#### Release Tooling
- `scripts/publish_meta.sh` — dry-run publish ordering script
- `scripts/bump_version.sh` — workspace version sync script
- `docs/RELEASE_CHECKLIST.md` — pre-release validation checklist
- `scripts/run_miri.sh` + `docs/MEMORY_SAFETY.md` — Miri testing policy

#### Dataset Crate (`tenflowers-dataset`)
- `PidAdaptiveController` — PID-controlled prefetch depth driven by cache hit-rate signal
- Criterion throughput benchmark harness (`benches/throughput.rs`)
- Drift metrics: PSI, KS two-sample statistic, Jensen-Shannon divergence, `DriftReport`
- `PipelineInspector` with per-step latency, shape-in/out, and `PipelineInspectionReport`
- `SchemaValidator::validate_full` with per-field `FieldDiff` (TypeMismatch, Widening, MissingRequired, UnexpectedExtra)
- Expanded module docs and API stabilization

#### FFI Crate (`tenflowers-ffi`)
- Structured `TensorError → TenflowersError` variant mapping (all 23 core variants, exhaustive match)
- `into_py_err()` — maps `TenflowersError` to typed Python exceptions (`ValueError`, `RuntimeError`, `IndexError`, `MemoryError`, `NotImplementedError`)
- `PyDevice` / `PyDeviceKind` — GPU/ROCm/CPU device as Python classes with `Device.cpu()`, `Device.gpu(id)`, `Device.rocm(id)`
- `PyTensor.__repr__` fix: dtype now reflects actual tensor dtype (was hardcoded `float32`)
- `PyTensor.__len__`, `.ndim`, `.numel()` properties
- `build.rs` for opt-in cbindgen header regeneration (`TENFLOWERS_REGENERATE_C_HEADER=1`)
- `tests/conftest.py` — shared pytest fixtures and marker registration
- `docs/FFI_ERROR_MAPPING.md` — full error mapping reference table

### Fixed

- **wgpu v29 API compatibility** — 57 sites updated across `tenflowers-core` and `tenflowers-dataset`:
  - `InstanceDescriptor::default()` replaced with `InstanceDescriptor::new_without_display_handle()`
  - `bind_group_layouts: &[&layout]` updated to `&[Some(&layout)]` per wgpu 29 API
- `data_quality.rs` — 9 `&str`/`String` type mismatches in drift metric constructors

### Improved

- Autograd module docs expanded to ~180 lines: mixed-precision, checkpointing, higher-order grads, custom ops
- Neural module docs finalized (removed "IN PROGRESS" markers)
- 3 new autograd examples: `mixed_precision.rs`, `gradient_checkpointing.rs`, `higher_order_grads.rs`
- FFI package metadata: classifiers, project URLs, optional dependency groups in `pyproject.toml`

## [0.1.0] - 2026-03-20

### Summary

First release of TenfloweRS — a research-grade, pure-Rust machine learning framework built on the SciRS2 ecosystem.

**Release Status:** Production-ready (6 crates)
- **Tests:** 12,949 passing across the workspace (0 failures, 0 warnings)
- **Code:** ~790K lines of Rust across 1,446+ source files
- **Security:** 0 vulnerabilities
- **Quality:** Zero clippy warnings, full formatting compliance, no `unwrap()` usage

### Added

#### Workspace Crates
- **tenflowers-core**: Core tensor operations, GPU abstraction, operation registry, shape inference, kernel fusion, autocast, sparse tensors, fused ops
- **tenflowers-autograd**: Reverse-mode automatic differentiation, gradient accumulation, checkpointing, in-place ops, forward-mode gradients, Jacobian checks, interpretability utilities
- **tenflowers-dataset**: Data loading and preprocessing, distributed streaming, dataset core, cache telemetry
- **tenflowers-neural**: Comprehensive neural network layers, training utilities, and research-grade algorithm modules (see below)
- **tenflowers-ffi**: C FFI and Python bindings via PyO3 (`publish = false` — requires Python environment)
- **tenflowers**: Unified API and prelude, user-facing macros and re-exports

#### Neural Network Modules (tenflowers-neural)

- **Attention mechanisms**: Flash Attention, ALiBi, RoPE, transformer decoder, TCN
- **Optimizers**: LAMB, Lion, Muon, learning-rate schedulers, LR finder
- **Generative models**: Diffusion (DDPM/advanced), VAE, normalizing flows, continuous normalizing flows, flow matching
- **Graph neural networks**: GNN advanced layers, graph-level pooling, temporal GNN, graph signal processing, graph matching, graph transformer, graph foundation models, graph generation, GraphODE, molecular GNN
- **Geometric deep learning**: EGNN, SE(3)-Transformer, VNN, IPA (AlphaFold2-style)
- **Quantum ML**: QAOA, quantum kernels (IQP/ZZ feature maps), zero-noise extrapolation, probabilistic error cancellation, measurement error mitigation, quantum Boltzmann machines
- **Federated learning**: Byzantine-robust aggregation (Krum, FLAME, Bulyan), personalized FL (pFedMe, APFL, FedBN), FedMA, clustered FL (IFCA)
- **Operator learning**: FNO, WNO, GNO, PINO, UNO
- **Bio ML**: scVAE (ZINB), Leiden clustering, DNA conv nets, survival analysis (CoxPH, DeepSurv, Kaplan-Meier), pathway enrichment, multi-omics factor analysis
- **AutoML**: Dataset meta-features, landmarking, algorithm selection, SMAC optimizer, portfolio selection, efficient NAS predictor
- **Molecular GNN**: DimeNet, AttentiveFP, MolBERT, JunctionTreeVAE, GraphVAE, reaction yield prediction
- **Audio models**: HuBERT, Data2Vec-Audio, SoundStream codec, RVQ, beat tracking, chord recognition, FastSpeech2, HiFi-GAN vocoder
- **Sparse learning**: BigBird, sparse sliding window attention, Group Lasso, N:M structured pruning, LISTA, predictive coding, basis pursuit
- **Geospatial ML**: H3/Quadkey grid encoders, spatial GCN, spatial attention, kriging, ST-GCN, diffusion convolution, Moran's I
- **Neural SDE**: VP/VE SDE, score matching, rough paths, NeuralRDE
- **Simulation-based inference**: Flow-SBI, ABC-SMC, NRE
- **Structured prediction**: Neural CRF, span parsing, biaffine dependency, SRL
- **Efficient transformers**: RetNet, Mamba-2, GQA, KV-cache management
- **Neural rendering**: 3D Gaussian splatting, NRC, ReSTIR, DeformNeRF
- **Riemannian geometry**: Poincaré ball, Ollivier-Ricci, Ricci flow
- **World models**: TD-MPC2, GWM tokenizer, GPT imagination loop
- **Symbolic math**: ATP tactics, neural tactic selector, equation database
- **Knowledge graph**: Temporal KG (TeRo/TntComplEx), hyper-relational KG, KG+LLM, rule induction; advanced knowledge distillation
- **Robotics**: RRT*/NeuralRRT/PRM, Ferrari-Canny grasp, whole-body control, DANN sim2real
- **Video understanding**: VideoSwin-V2, TimeSformer, VideoMAE, VOS memory, ConsistencyModel
- **Compression**: Hyperprior model, RD optimizer, movement pruning, mixed-precision search
- **Online learning**: LinUCB, Thompson sampling, ADWIN, LODA, OGD/FTRL
- **Generation**: Speculative decoding, RegexFSM/CFG constrained generation, RAG, BLEU/ROUGE metrics
- **Multimodal foundation**: UnifiedIO, PaLI, visual grounding, symbolic visual reasoning
- **Optimal transport**: Unbalanced/partial OT, JDOT, online sliced-Wasserstein, tree-Wasserstein
- **Additional modules**: active_inference, active_learning, adaptive_computation, adversarial, anomaly_detection, architecture_distillation, audio_generation, bayesian, bayesian_dl, bayesian_opt, bio_ml, causal_discovery_advanced, causal_discovery_ts, causal_inference, causal_representation, causal_rl, causal_ts, checkpoint_advanced, climate_ml, concept_learning, conformal_prediction, continual_learning, contrastive, cooperative_game_theory, cross_modal_retrieval, curriculum_learning, data_augmentation, depth_estimation, differentiable_physics, digital_pathology, distillation, document_understanding, domain_adaptation, drug_discovery, edge_optimization, embodied_ai, emotion_recognition, energy_models, ensemble, evolutionary_computation, financial_ml, functional_data_analysis, hierarchical_time_series, hparam, hyperdimensional, hypernetworks, hyperparameter_optimization, image_generation_advanced, implicit_neural_repr, influence_functions, information_theory, inverse_rl, knowledge_distillation_advanced, kolmogorov_arnold, learning_to_learn, lifelong_learning, llm_serving, lm_evaluation, lora_adapters, marl, materials_ml, mean_field_games, mechanistic_interpretability, medical_imaging, memory_networks, meta_learning, mixture_density_networks, mixture_of_depths, mixture_of_experts_advanced, mixture_of_modalities, model_merging, monte_carlo, multi_fidelity, multi_objective, multi_task, multimodal, music_generation, nas, network_science, neural_collapse, neural_combinatorial, neural_compression, neural_ode, neural_process, neuro_symbolic, neuromorphic, nlp_components, nn_verification, object_tracking, optimal_control, pinn, point_processes, pomdp_planning, privacy_ml, probabilistic, probabilistic_circuits, protein_lm, protein_structure, quantum_ml, recommendation_systems, reward_learning, reward_shaping, rl, safe_rl, safety_alignment, satellite_ml, scene_graph, self_play, self_supervised, signal, simulation_ml, sparse_mixture_experts, spectral, speech_recognition, ssm, state_space_models, statistical_testing, synthetic_data, tabular_learning, tensor_networks, test_time_adaptation, test_time_compute, text_generation_pipelines, time_series, tokenizer, topological_ml, training_dynamics, trajectory_prediction, uncertainty_quantification, variational_inference, vision_transformer, zero_shot_learning, and more

### Crates Published

| Crate | Description |
|-------|-------------|
| tenflowers-core | Core tensor operations, GPU abstraction, autocast, sparse, fused ops |
| tenflowers-autograd | Automatic differentiation, checkpointing, gradient accumulation |
| tenflowers-dataset | Data loading, preprocessing, distributed streaming |
| tenflowers-neural | Research-grade neural network layers and ML algorithms (300+ modules) |
| tenflowers | Unified public API and prelude |

**Not published:** tenflowers-ffi (`publish = false` — requires Python development environment)

### Notes

- **Tensorboard integration** is excluded from this release due to a known upstream vulnerability (RUSTSEC-2024-0437 in protobuf 2.x). It will be restored once the upstream fix is available.
- **SciRS2 dependencies**: All scirs2-* crates at 0.3.0+

### Installation

```toml
[dependencies]
tenflowers = "0.1.0"

# Optional features
tenflowers = { version = "0.1.0", features = ["gpu", "simd"] }
```

### Contributors

Developed by COOLJAPAN OU (Team KitaSan).
Contact: contact@cooljapan.tech

[0.1.2]: https://github.com/cool-japan/tenflowers/releases/tag/v0.1.2
