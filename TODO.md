# TenfloweRS TODO & Roadmap (v0.2.1 · 2026-07-13)

## v0.2.0 — Implicit Autograd Rewrite (2026-07-13)

A complete rewrite of the Python-facing, PyTorch-style implicit autograd
system in `tenflowers-ffi`. Previously `.backward()`/`.grad()`/
`optimizer.step()` only worked end-to-end for a minimal Dense/Sequential/MSE
path; every other layer, loss, and optimizer either raised or silently
produced wrong gradients. Full detail lives in CHANGELOG.md's
`[0.2.0] - 2026-07-13` entry, summarized here.

### Every layer type now has real, tape-backed backward support
`Dense`, `Conv1D`/`Conv2D`/`Conv3D` (+ pooling), `Embedding`/`EmbeddingBag`,
`BatchNorm1d`, `LayerNorm`, `GroupNorm`, `InstanceNorm1d`,
`MultiheadAttention`, `TransformerEncoderLayer`/`TransformerDecoderLayer`,
and `LSTM`/`GRU`/`RNN` (plus their single-step cell variants) all now
genuinely wire into the gradient tape — with two narrow, honestly-documented
exceptions (not glossed over, see "Known Limitations" below for the current
tracked status):
- **Conv1D/Conv2D/Conv3D and MaxPool2D/AvgPool2D**: tape-recording only fires
  for unit dilation, `groups == 1`, and no explicit padding, gated by a
  `tape_recordable` check (`crates/tenflowers-ffi/src/neural/conv_layers/
  mod.rs:784`). Dilated/grouped/padded convolutions still compute a correct
  forward value but skip gradient recording.
- **EmbeddingBag**: `sum`/`mean` reduction modes are tape-wired; `mode="max"`
  is not — its own source doc states plainly "no tape-aware max reduction
  exists" anywhere in the crate (`crates/tenflowers-ffi/src/neural/
  embedding.rs`).
- `PyTensor::slice` now records itself on the tape, so slicing a tracked
  tensor participates correctly in `.backward()`.

### All 9 optimizers now do real gradient-based parameter updates
`SGD`, `Adam`, `RMSprop`, `AdamW`, `AdaBelief`, `RAdam`, `Nadam`, `AdaGrad`,
`AdaDelta` all read `.grad()` and write back to parameters via a new
`optimizer_bridge` module's `collect_parameters()` helper, replacing the
previous no-op/partial `step()`. The actual step() math lives in
`neural/optimizers.rs` and the newly-split-out `neural/extended_optimizers/`
module.

### Every loss function is genuinely backward-connected to the tape
`neural::losses` was substantially rewritten so loss values participate in
`.backward()` rather than being a dead end.

### Real end-to-end training convergence, proven with actual loss traces
`crates/tenflowers-ffi/tests/test_training_convergence.py` — three tests
proving finite, decreasing loss over real training steps:
`test_dense_layer_training_converges` (PyDense + SGD),
`test_sequential_mlp_training_converges` (3-layer Sequential MLP + Adam),
`test_conv2d_training_converges` (Conv2D + Adam, deliberately using only a
unit-dilation/groups=1/no-padding configuration to stay inside the wired
boundary above).

### Autograd crate (`tenflowers-autograd`) correctness fixes
- **Softmax/LogSoftmax backward** were computing wrong gradients; rewritten
  to recompute the forward output on the tape and apply the correct
  Jacobian-vector product.
- **BatchNorm backward** (`ops::normalization_ops`): eval-mode (inference)
  `grad_gamma`/`grad_beta` were hardcoded to zero instead of derived from the
  running statistics; a missing 3-D (NCL) shape case fell through to an
  incorrect channel-last default.
- **LayerNorm backward**: `gamma` was applied as a single post-reduction
  factor instead of being folded in before the per-axis reduction sum — only
  correct when `gamma` is uniform across the normalized axis, silently wrong
  otherwise.
- **GroupNorm backward**: the same gamma-before-reduction bug as LayerNorm,
  compounded by `gamma` varying per-channel *within* a group — measured up to
  **760% relative error** for non-uniform gamma prior to this fix.
- **Slice/Gather backward** were stubs; now produce real gradients.
- A row-major-vs-Fortran-order stride bug in `slice_with_stride`
  (`tenflowers-core::ops::manipulation::indexing`): the linear-index
  computation used a forward-order running-product stride formula instead of
  the correct row-major (reverse-order) one, silently producing wrong
  elements for any non-square, non-1-D sliced array — caught by an LSTM/GRU
  gate-slicing finite-difference gradient test.
- A `PyParameter` tape-registry lifecycle bug where a dropped parameter could
  poison a later parameter reusing the same allocation address, silently
  losing its gradients.
- A `mark_leaf_param` bug where any second-or-later `forward()` call on the
  same parameter without an intervening `backward()` silently broke that
  parameter's tape registration, dropping its gradient without error.

### New gradient-check test suites
`activation_gaps_gradient_test`, `conv1d_gradient_test`,
`conv3d_gradient_test`, `group_instance_norm_gradient_check`,
`normalization_gradient_check`, `slice_concat_stack_split_gather_gradient_test`
(all in `tenflowers-autograd`), plus per-module `tests.rs` suites for
`neural::{attention,conv_layers,recurrent,transformer,extended_optimizers}`
in `tenflowers-ffi`. `Conv1D` gained a dedicated backward implementation
(`ops/convolution_ops/conv1d.rs`, `conv1d_utils.rs`).

### Module splits (COOLJAPAN 2000-line refactor policy)
`crates/tenflowers-ffi/src/implicit_autograd.rs` and
`neural/{attention,conv_layers,recurrent,transformer}.rs` split into
`mod.rs` + `tests.rs` submodule directories.

### Security
RUSTSEC-2026-0204 (`crossbeam-epoch`), tracked as open in the prior release
(fix noted then as requiring `crossbeam-epoch >= 0.9.20`), is now resolved —
`cargo tree -i crossbeam-epoch` confirms `crossbeam-epoch v0.9.20` in the
lockfile via the `scirs2-core` 0.6.0 transitive chain, exactly the version
the prior release's own tracking note said would fix it. 2 advisories remain
open, both transitive and unchanged from last release: RUSTSEC-2024-0384
(`instant`, unmaintained, via `hdf5`) and RUSTSEC-2024-0436 (`paste`,
unmaintained, via `rav1e`/`parquet`/`metal`).

### Verified metrics (2026-07-13 full-workspace run)
- **Tests**: 14,536 passing, 39 skipped, 0 failures (`cargo nextest run
  --workspace --all-features`); 14,093 passing, 14 skipped, 0 failures with
  default features. Skipped count is unchanged from the prior 2026-07-07 run
  (39), so the +247 newly-passing tests are net-new additions, not
  previously-skipped tests newly passing.
- **Code size**: 685,753 SLoC Rust (`tokei .`), 1,635 total files (1,533 Rust
  files / 823,712 total Rust lines)
- **Warnings**: 0 compilation warnings, 0 clippy warnings
- **Security**: 0 direct vulnerabilities, 2 known transitive advisories (see
  "Security" above)
- **Publishability**: `cargo publish --dry-run` succeeds for all 5
  publishable crates (`tenflowers-ffi` is `publish = false`)

## v0.1.2 — Honesty Hardening (2026-06-23)

A workspace-wide sweep to remove silent fabrication: code that compiled cleanly
and returned plausible-but-fake values now either computes the real result or
returns an HONEST error. Doc-only summary; see each crate's TODO for specifics.

### Lock-poison panics removed (all crates)
- Eradicated ~500+ production lock-poison panics (`.lock()/.read()/.write()`
  followed by `.unwrap()/.expect()`) via signature-preserving recovery:
  fallible fns use `map_err(... lock poisoned)?`; infallible fns use
  `unwrap_or_else(|p| p.into_inner())`. A poisoned lock no longer aborts the
  process. Remaining such calls are test-only.

### Fakes replaced with real computation
- **tenflowers-ffi**: 21 Python neural-layer `forward` methods that silently
  returned `Tensor::zeros` now compute real results (Conv1D/2D, Max/AvgPool2D,
  Batch/Layer/Group/InstanceNorm, Embedding/EmbeddingBag, LSTM/GRU/RNN + cells,
  MultiheadAttention/SDPA, Transformer enc/dec, PositionalEncoding,
  Dropout/Dropout2D); AlphaDropout/FeatureAlphaDropout use the real
  SELU-preserving formula.
- **tenflowers-core**: serialization checksum (was hardcoded `Ok(0)`, never
  verified) → real FNV-1a with verification; `Tensor::randn` (was a shifted
  uniform, mean ≈ −0.5) → real N(0,1) (+ seeded `randn_with_seed`);
  `gather`/`slice`/`segment_sum`/`segment_mean` correctness fixes; graph
  scheduling pass no longer falsely reports "changed".
- **tenflowers-autograd**: `relu_mask` (was all-ones) → real 0/1 mask; fused
  Mish (was `tanh`) → real Mish; gradient cache (stored empty data) → real
  round-trip; memory stats (hardcoded) → real profiler values.
- **tenflowers-neural**: MultiHeadAttention head-reshape bug fixed (the
  `[0,2,1,3]` permutation), unblocking the Transformer encoder/decoder;
  `data_parallel` (was a no-op "simulate_backward") → real gradients;
  `ultra_dense*` (zeros-init) → real random init; fabricated attention/layer
  metrics → real values or honest "not measured".
- **tenflowers-dataset**: download `verify_checksum` (was `Ok(true)`) → real
  SHA-256; zarr lz4/zstd chunks now decompress via oxiarc; benchmark/CPU
  metrics (hardcoded/random) → real `/proc` measurements or honest sentinels.

### Capabilities now returning honest errors (were faked)
See "Known Limitations" below for the full list. These previously returned
fabricated success; they now fail loudly until real backends are wired.

## v0.1.2 — Continued Hardening & Feature Landing (2026-06-24 -> 2026-07-07)

The section above covers the sweep as of 2026-06-23. A large amount of
additional work landed in the two weeks since; full detail lives in
CHANGELOG.md's `[0.1.2] - 2026-07-08` entry, summarized here.

### Former "Known Limitations" gaps closed for real (see "Completed" below)
All six items tracked under "Completed (planned 2026-07-02, verified done
2026-07-07)" were finished and verified against source; see that section for
the file-level detail. In short: N-D segment reductions, graph-optimizer
wiring into `Session`, GPU-einsum CPU fallbacks, real wgpu device-capability
queries, a from-scratch Zarr Blosc codec, and real Symphonia-backed audio
decoding are now all real, tested code.

### More fakes replaced with real computation
- **tenflowers-core**: real ONNX protobuf import/export (`onnx_interop::{convert,
  lowering, proto}`) for the core graph representation — `OnnxImporter`/
  `OnnxExporter` previously always returned `NotImplemented` regardless of the
  `onnx` feature flag; now genuinely decode/encode, and unrecognized ONNX
  attribute/dtype codes are rejected with a descriptive `Err` instead of
  silently coercing to `Float(0.0)`/`Float32`. New `gradient_executor` module
  lets `gradient_validation_framework` call into a real `tenflowers-autograd`
  forward+backward implementation (`AutogradGradientExecutor`) instead of
  fabricating `passed: true` when no executor is registered. Real LAPACK-backed
  `ops::lapack_f64` (inverse/determinant/SVD/solve). Real `StdDev`/`L1Norm`/
  `L2Norm` CPU reductions (previously fell through to a generic error).
- **tenflowers-neural**: `layers::moe::TopKRouter` now does real top-k routing
  with a genuine load-balancing loss (previously routed every token to expert 0
  unconditionally); `deployment::pruning::engine` now genuinely zeros real
  weight tensors for Magnitude/Random strategies (previously fabricated stats
  from synthetic data and returned an unmodified empty `Sequential`);
  `serialization::weight_loader` implements a real versioned JSON save/load
  format (previously every format silently no-op'd); `tensorflow_compat::
  SavedModelLoader` no longer fabricates a `SavedModel` with hardcoded metadata;
  Conv2D/Conv3D backward gradients are now real transposed-cross-correlation
  implementations (previously zero-tensor placeholders, with one path an
  outright-wrong identity/zero shortcut); `numerical_checker::property_test`
  now genuinely compares against a required `f_grad` closure instead of
  trivially passing against a clone of its own numerical estimate.
- **tenflowers-autograd**: `CrossDatacenterReplicator`/`DatacenterConnection`
  network-simulation methods now honestly return `NotImplemented` instead of
  fabricating successful cross-datacenter coordination (fake prepare/aggregate
  results, hardcoded step counters, always-healthy status).
- **tenflowers-dataset**: `formats::audio::get_audio_info` now probes/decodes
  via Symphonia for real `sample_rate`/`channels`/`num_samples`/`duration`
  (previously fabricated `channels = 1`, `duration = 1.0`); `arrow::
  ArrowArrayExt::to_tensor` now does a real conversion (previously an
  unconditional `Err` stub); `transforms::noise::AddNoise` now adds genuine
  Box-Muller Gaussian noise (previously computed-then-discarded); a
  Miri-confirmed alignment UB in `memory_pool::MemoryBlock`/`MemoryPool` fixed
  via new `allocate_aligned`/`allocate_exact`.
- **tenflowers-ffi**: new `implicit_autograd` module gives `PyTensor` real
  PyTorch-style eager `.backward()`/`.grad()` on top of the existing
  `GradientTape` engine; `key_padding_mask` in `PyMultiheadAttention`/
  transformer layers is now genuinely combined into the attention bias
  (previously silently dropped before softmax, so "masked" keys still received
  real attention weight); GRU forward now honors a caller-supplied initial
  hidden state (previously ignored); `RNN`'s `nonlinearity` parameter is now
  consulted (previously stored and validated but never used); dilated
  max-pooling now genuinely dilates the sampling window.
- **NCCL/Gloo/MPI/thread collective backends** (`tenflowers-neural`): all
  return honest `NotImplemented` instead of silently scaling, echoing, or
  cloning the caller's own local tensor as if it were a genuine cross-rank
  result; `DataParallelTrainer::train_step` now computes real per-parameter
  gradients via central finite differences and calls the real optimizer step
  (previously simulated the backward pass and zeroed gradients as a
  placeholder) — parameters now genuinely learn.

### Dependency updates
numrs2 0.4.0; scirs2 (core/autograd/neural/linalg/numpy) 0.4.2 -> 0.6.0;
oxicode 0.2.4; oxiarc-archive 0.3.4 (+ new oxiarc-lz4/oxiarc-deflate/
oxiarc-snappy for Blosc's inner codecs); oxifft 0.3.2; wgpu 30.0; pyo3 0.29;
arrow/parquet 59.0; prost 0.14.4; symphonia 0.6; rubato 3.0.

### Security
RUSTSEC-2026-0176/0177 (pyo3 0.28.3) resolved via the pyo3 0.29 upgrade once
the blocking `scirs2-numpy` pin lifted. Newly tracked (all transitive, none
directly exploitable): RUSTSEC-2026-0204 (`crossbeam-epoch` 0.9.18, via
`scirs2-core` 0.6.0 -> `crossbeam-deque`; fix needs `crossbeam-epoch` >= 0.9.20,
not a direct dependency); RUSTSEC-2024-0384 (`instant`, transitive via `hdf5`);
RUSTSEC-2024-0436 (`paste`, transitive via `rav1e`/`parquet`/`metal`).

### Verified metrics (2026-07-07 full-workspace run)
- **Tests**: 14,289 passing, 39 skipped, 0 failures (`cargo nextest run
  --workspace --all-features`, 102s)
- **Per-crate**: tenflowers (meta-crate) 156, tenflowers-core 1,171,
  tenflowers-autograd 521, tenflowers-neural 11,596, tenflowers-dataset 660,
  tenflowers-ffi 185
- **Code size**: ~677K SLoC Rust (`tokei .`), 1,616 total files (1,515 Rust
  files / 805,252 Rust lines)
- **Warnings**: 0 compilation warnings, 0 clippy warnings

## Current Capabilities

### Project Status
- **Tests**: 14,536 passing, 39 skipped across all crates (verified 2026-07-13
  full-workspace `cargo nextest run --workspace --all-features` run); 14,093
  passing, 14 skipped with default features
- **Code Size**: 685,753 SLoC Rust code (823,712 total Rust lines, 1,635 total
  files / 1,533 Rust files, per `tokei .`)
- **Warnings**: 0 compilation warnings, 0 clippy warnings
- **Vulnerabilities**: 0 direct vulnerabilities; 2 known transitive advisories,
  both upstream-blocked and neither directly exploitable (see "Security"
  under "v0.2.0 — Implicit Autograd Rewrite" above)
- **SciRS2 Integration**: Full migration to SciRS2 ecosystem (0.6.0)

### TenfloweRS-Core (Tensor Engine)
- Complete eager tensor engine with arithmetic, reduction, manipulation, matrix multiplication
- Blocked matrix multiplication + outer product specialization + optional BLAS acceleration
- Comprehensive SIMD optimization (8-element chunking, unchecked fast paths, math functions)
- GPU support via WGSL compute kernels with safe CPU fallbacks (cross-platform WebGPU)
- Memory management with reference counting, buffer reuse metrics, allocation tracing

### TenfloweRS-Autograd (Automatic Differentiation)
- Full reverse-mode gradient tape with recording + backward traversal
- Optimized gradient computation (hashmap lookup reductions, allocation optimization)
- Integrated memory profiling on gradient path with usage tracking
- Basic GPU gradient support for selected core operations
- Experimental mixed precision gradient path with loss scaling

### TenfloweRS-Neural (Neural Networks)
- Core layers: Dense, Conv1D/2D, Embedding (basic/positional/sparse), Residual
- Normalization: BatchNorm, LayerNorm, GroupNorm, SyncBatchNorm (training/inference)
- Advanced layers: Mamba/SSM blocks, attention scaffolding, transformer building blocks
- Extended activations: ReLU, GELU, Swish, LeakyReLU, ELU, SELU, Hardswish, GLU variants
- Optimizers: SGD, Momentum, Adam, AdamW, RMSProp, AdaBelief with AMSGrad support
- Training pipeline with gradient accumulation, metric tracking, hook system
- Pretrained model architectures: ResNet, EfficientNet, ViT, BERT, GPT families

### TenfloweRS-Dataset (Data Loading)
- Dataset trait + composable transform pipeline with builder pattern
- SIMD transforms (stats, color conversion, histogram) with runtime fallback
- GPU-accelerated transforms (crop, rotate, jitter, blur, noise, resize, flip)
- Predictive smart cache with pattern-based prefetch and memory pool management
- Formats: JSON/JSONL, Text, Parquet, HDF5, Audio, TFRecord, WebDataset, Zarr, CSV, Image
- Memory-mapped file dataset for large file zero-copy access

### TenfloweRS-FFI (Language Bindings)
- Python bindings via PyO3 with real, tape-backed `.backward()`/`.grad()` for
  every major layer type (Dense, Conv1D/2D/3D+pooling, Embedding/
  EmbeddingBag, all normalization layers, MultiheadAttention, Transformer
  enc/dec layers, LSTM/GRU/RNN+cells) — see "Known Limitations" for the two
  narrow tape-recording exceptions (dilated/grouped/padded conv+pool,
  EmbeddingBag max-mode)
- All 9 optimizers (SGD, Adam, RMSprop, AdamW, AdaBelief, RAdam, Nadam,
  AdaGrad, AdaDelta) perform real gradient-based parameter updates
- All loss functions genuinely backward-connected to the tape
- NumPy tensor conversion (f32), memory optimization utilities
- C API scaffolding (types, tensor creation)
- Hook system (forward/backward), benchmarking, visualization

## Known Limitations (v0.2.0, updated 2026-07-13)

### Resolved since the 2026-06-22 sweep (see "Completed" section below for detail)
The following were listed as honest-error deferrals as of 2026-06-23 and are
now real, tested implementations rather than gaps:
- **GPU einsum correctness**: batched-matmul/transpose/diagonal/outer/trace now
  correctly delegate to the CPU `einsum` implementation via GPU->host readback
  (previously wrong-semantics ops + `unreachable!()` fallbacks).
- **Real device-capability queries**: `device::GpuAdapterCapabilities` /
  `get_gpu_adapter_capabilities` now report a real, unprocessed `wgpu::Adapter`
  snapshot (`AdapterInfo`/`Limits`/`Features`) — no more fabricated
  vendor/bandwidth/tensor-core guessing.
- **Audio file decoding**: real Symphonia-backed decode (WAV/MP3/FLAC) wired
  into `formats::audio::load_audio_file`/`get_audio_info` (previously
  returned a fabricated 440 Hz sine wave / hardcoded metadata). OGG/Vorbis
  remains honestly unsupported (symphonia feature not enabled).
- **Zarr `blosc` codec**: a from-scratch, pure-Rust `formats::blosc` decoder
  now handles all 5 inner codecs (BloscLz/Lz4/Snappy/Zlib/Zstd) plus
  byte/bit-shuffle, wired into the Zarr `"blosc"` compressor dispatch. The
  Blosc *encoder* is not implemented (decode-only).
- **`segment_max/min/prod/any/all`** for feature width D>1: all five ops now
  produce correct `[num_segments, d1, d2, ...]` output for N-D input via a
  shared `SegmentLayout` row-based reducer (previously only the 1-D case and
  `segment_sum`/`segment_mean` were correct for D>1).
- **Graph optimizer passes**: now enabled by default in real `Session`
  execution via `SessionConfig::enable_graph_optimization` (constant folding,
  CSE, algebraic simplification, strength reduction, DCE, scheduling); see
  "Other limitations" below — this does *not* mean every planned rewrite is
  landed, see the `v0.3.0` roadmap below for further optimizer work.
- **ONNX protobuf import/export**: real prost-based decode/encode is now wired
  for both `tenflowers-core::onnx_interop` (core op subset: Add/Sub/Mul/Div/
  Relu/Sigmoid/Tanh/MatMul/Reshape/Transpose/Identity/Concat/Softmax/Flatten/
  Gemm) and `tenflowers-neural::serialization::onnx` (behind the `onnx`
  feature), replacing what were previously hardcoded `NotImplemented`/`Err`
  returns regardless of the feature flag.

### FFI implicit-autograd gaps (new in v0.2.0, narrow and honestly documented)
The v0.2.0 autograd rewrite (see section above) wired real tape-backed
gradients into every layer type, with two specific, currently-open
exceptions — forward computation is correct in both cases, only gradient
recording is skipped:
- **Conv1D/Conv2D/Conv3D and MaxPool2D/AvgPool2D tape-recording**: only fires
  for unit dilation, `groups == 1`, and no explicit padding (the
  `tape_recordable` gate in `crates/tenflowers-ffi/src/neural/conv_layers/
  mod.rs:784`). Dilated, grouped, or explicitly-padded convolutions/pools
  still forward correctly but do not record onto the gradient tape.
- **`EmbeddingBag` `mode="max"`**: not tape-wired — `sum`/`mean` modes are.
  Documented directly in the source
  (`crates/tenflowers-ffi/src/neural/embedding.rs`): "no tape-aware max
  reduction exists" anywhere in the crate.

### Honest-error deferrals still open (fail loudly instead of faking)
- **GPU compute kernels (Metal MPS)**: GPU→host readback for the Metal MPS
  training-forward path (matmul/conv2d/reductions/layer+group-norm/
  flash-attention) still returns an honest, layer-specific `Err` naming the
  missing kernel rather than fabricating output — this is not yet a real
  implementation, only honestly erroring instead of returning un-read-back
  zeros. GPU FFT/linalg already have real CPU fallbacks; `Tensor::to` GPU→GPU
  and ROCm `unreachable!()` paths error rather than silently misbehaving.
- **NCCL/Gloo/MPI/thread collective ops**: all backends (`tenflowers-neural`)
  now consistently return honest `NotImplemented` — none links a real
  cross-process collective-communications runtime yet (NCCL specifically
  requires the `libnccl` runtime).
- **TensorFlow SavedModel protobuf import**: `tensorflow_compat::
  SavedModelLoader::load_from_pb`/`load_from_pbtxt`/`parse_pbtxt_content`
  no longer fabricate a `SavedModel` with hardcoded metadata, but binary/text
  TensorFlow protobuf decoding itself is still not implemented — these
  honestly error rather than parse. (This is distinct from ONNX, which is
  now real — see above.)
- **Checkpoint bundle (`.index`/`.data-*` BundleEntry protobuf)** parsing for
  TensorFlow checkpoints: not implemented, honest error.

### Other limitations
- Many GPU operations still fall back to CPU (the CPU result is real); GPU
  memory management inconsistent.
- Higher-order gradients unreliable for composite activation chains.
- RNG-state checkpoint capture is not supported (scirs2_core RNG exposes no
  serializable state) — returns an honest error.
- Four new memory/perf diagnostic modules (`allocation_timeline.rs`,
  `memory_pressure.rs`, `per_op_tracker.rs`, top-level `pool_diagnostics.rs`)
  landed on disk in `tenflowers-core` but have no `pub mod` declaration in
  `lib.rs` yet, so they are currently unreachable from the public API —
  tracked for a follow-up wiring pass.

### Build/feature notes
- `tenflowers-dataset --no-default-features` build: **FIXED (2026-06-22)** —
  added proper `#[cfg(feature = …)]` gates (`serialize`/`numa`/`tfrecord`/
  `parquet`); default, `--no-default-features`, and `--all-features` all build
  clean. (A few `examples/` still hard-assume default features — minor, separate.)
- `--all-features` pulls `openblas-src` only via tenflowers-core's own opt-in
  `blas-openblas` feature (`ndarray-linalg/openblas-system`). This is the
  standard `--all-features` mutually-exclusive-backend artifact: **default builds
  and the recommended `blas-oxiblas` (pure-Rust) backend are openblas-free**, so
  the oxiblas policy holds. For an openblas-free guarantee in CI, use
  `--features blas-oxiblas` rather than bare `--all-features`.

## Roadmap

**Note on the old "v0.2.0 — Attention & Training Polish" roadmap entry**: this
project previously roadmapped v0.2.0 under that title with the item list
below. v0.2.0 shipped 2026-07-13 with substantially different actual content
(the FFI implicit-autograd rewrite — see the section near the top of this
file). Verified by direct source check (2026-07-13) that most of the
originally-roadmapped items were, in fact, *already implemented* prior to
this release (they did not ship as part of v0.2.0's work, they simply
predate it and this roadmap was stale about their status):
- **Multi-head + scaled dot-product attention**: already implemented
  (`crates/tenflowers-neural/src/layers/attention/multi_head.rs`'s
  `MultiHeadAttention`, `layers/attention/utils.rs`'s
  `scaled_dot_product_attention`, plus `layers/transformer.rs`).
- **Learning rate schedulers** (step/exponential w/ staircase, cosine,
  warmup, one-cycle): already implemented
  (`crates/tenflowers-neural/src/optimizers/schedulers.rs`'s
  `CosineAnnealingScheduler`/`OneCycleLrScheduler`/`WarmupScheduler`/
  `ExponentialDecayScheduler`, `crates/tenflowers-neural/src/scheduler.rs`'s
  `CosineAnnealingLR`).
- **Gradient clipping utilities**: already implemented
  (`crates/tenflowers-neural/src/optimizers/gradient_clipping.rs`'s
  `clip_gradients_by_value`/`_by_norm`/`_by_global_norm`/`_adaptive`).
  Anomaly-detection hooks specifically were not verified as implemented —
  moved to "still pending" below.
- **Activation checkpointing**: already implemented
  (`crates/tenflowers-autograd/src/checkpointing.rs`, `efficient_memory.rs`).
- **Deterministic mode**: already implemented
  (`crates/tenflowers-core/src/deterministic.rs`,
  `crates/tenflowers-autograd/src/deterministic.rs`).
- **Mixed precision**: already implemented
  (`crates/tenflowers-autograd/src/amp_policy.rs`,
  `crates/tenflowers-neural/src/mixed_precision.rs`). Dynamic loss-scaling
  *refinement* specifically was not re-verified as complete — moved to
  "still pending" below out of caution.

These are cut from the pending list entirely (they are already-shipped
background capability, not upcoming work). The remaining items from that old
list that are still genuinely pending/unverified are folded into v0.3.0
below rather than kept under a stale "v0.2.0" heading.

### v0.3.0 — Scale & Distributed
- Multi-GPU data-parallel execution with optimizer state sync
- Graph optimizer passes: constant-fold/CSE/algebraic-simplify/strength-reduction/DCE/scheduling
  landed and enabled by default in v0.1.2 (see "Completed" below); remaining v0.3.0 work is
  further fusion passes and real NCCL/Gloo/MPI collective backends for multi-GPU sync
- ONNX export/import for the core operator subset landed in v0.1.2 (see "Completed" below,
  item and Continued Hardening section); remaining v0.3.0 work is wider operator coverage
- Unified dispatch registry (CPU/GPU) with backend feature gating (carried over from the
  stale "v0.2.0" roadmap entry, still pending as of 2026-07-13)
- Consolidated shape inference + standardized error taxonomy (carried over, still pending)
- GPU memory diagnostics: allocation tracing, pool diagnostics, usage reporting (carried over,
  still pending — note the four diagnostic *modules* already exist on disk in tenflowers-core
  per "Other limitations" above but are not yet wired into the public API; this roadmap item
  is that wiring pass plus the reporting layer on top)
- Elementwise fusion MVP for performance improvement (carried over, still pending)
- Gradient anomaly-detection hooks (carried over from the gradient-clipping roadmap item,
  still pending — the clipping utilities themselves already shipped, see note above)
- Dynamic loss-scaling refinement for the mixed precision policy (carried over, not
  re-verified as complete, kept pending out of caution)
- Streaming data loaders with deterministic sharding for distributed training (carried over,
  still pending)
- Python wheel builds (manylinux, macOS universal2, Windows) (carried over, still pending)
- Sequence parallel / model parallel experiments
- Parameter grouping & weight decay configurability
- Pretrained model export/import (JSON weights + binary format)
- Arrow zero-copy integration for dataset pipelines
- Stable ABI & semantic versioning for FFI

### v1.0.0 — Production & Advanced Features
- Model/pipeline parallel & gradient compression
- Auto kernel fusion + JIT compilation prototype
- Advanced linear algebra (batched factorizations, sparse primitives)
- Memory offload / spill strategy (host ↔ device)
- INT8/INT4 quantization toolchain
- Plugin system for external operators
- Comprehensive documentation (Getting Started, Performance, Safety, Migration)

## Future Directions

- Federated learning and distributed sharding beyond standard data-parallel
- Advanced sparsity & compression research implementations
- Additional language bindings (C++, Swift)
- Cloud-native deployment & scaling infrastructure
- Federated data loaders with privacy preservation

## Completed (planned 2026-07-02, verified done 2026-07-07)

An `/ultra` pass turning documented "Known Limitations" into real implementations.
Six file-disjoint items, implemented in parallel. All six verified against real
source on 2026-07-07 (not just changelog claims) — each has a working, tested
implementation described below. See git history / diff for the line-level detail.

- [x] **A: segment_max/min/prod/any/all correct for feature width D>1** (planned 2026-07-02, done)
  - **Goal:** `[N,d1,d2,...]` input produces `[num_segments,d1,d2,...]` output for all five ops (previously only segment_sum/segment_mean were correct for D>1); 1-D case unchanged.
  - **Design:** Extended a `SegmentLayout` struct (crates/tenflowers-core/src/ops/reduction/segment/sum_mean.rs) with a generic per-cell row-based reducer; segment_max/min (minmax.rs) and segment_prod/any/all (prod_any_all.rs) now route through it instead of the old buggy whole-array flatten+zip. unsorted_segment_* inherit the fix by delegation.
  - **Files:** crates/tenflowers-core/src/ops/reduction/segment/{minmax.rs,prod_any_all.rs,sum_mean.rs,unsorted.rs,tests.rs}
  - **Verified:** `SegmentLayout` (sum_mean.rs) confirmed present with row-chunked `reduce_rows`/`accumulate`/`combine_row` methods; `segment_max`/`segment_min`/`segment_prod`/`segment_any`/`segment_all` all confirmed defined and routed through it; 41 `#[test]` functions in segment/tests.rs.
  - **Tests:** D>1 cases for all 5 ops, empty segments, i32/f64, negative values, shape-mismatch, unsorted D>1, 1-D regression parity.

- [x] **B: Enable graph optimizer in Session executor + finish stubbed passes** (planned 2026-07-02, done)
  - **Goal:** GraphOptimizer (constant-fold/CSE/algebraic-simplify/DCE/scheduling) runs before execution in Session; previously-stubbed rewrites (nested constant folding, output-aware DCE, x+x->2x, Square/DivToMul strength reduction) work; semantics-preserving (opt-on == opt-off numerically).
  - **Design:** Fixed ConstantFoldingPass to not `?`-propagate on not-yet-foldable nested nodes (fixpoint iteration handles it); real output-marking added to DeadCodeEliminationPass; ConvertToMultiply (x+x->2x) and sound StrengthReduction rewrites (Square, DivToMul) implemented; `GraphOptimizer::add_default_passes().optimize()` wired into `Session::build_execution_plan` (crates/tenflowers-core/src/session.rs) before `compute_topological_order()`, gated by `SessionConfig::enable_graph_optimization` (default `true`).
  - **Files:** crates/tenflowers-core/src/graph/optimization/passes/{algebraic,constant_folding,cse,dead_code,mod,pass_support,scheduling,strength_reduction,tests}.rs (split from a single 1264-line passes.rs via SplitRS), crates/tenflowers-core/src/session.rs
  - **Verified:** `SessionConfig::enable_graph_optimization: bool` confirmed present (session.rs:44), defaults to `true` (session.rs:56), consulted at session.rs:238 before execution; `test_graph_optimization_preserves_results_and_reduces_node_count` confirmed present, comparing optimizer-on vs. optimizer-off numeric results plus a `protected_output_roots` mechanism guaranteeing previously-fetched nodes survive later optimization passes with a different fetch list.
  - **Tests:** per-pass units + end-to-end Session test comparing optimized vs unoptimized numeric output equality.

- [x] **C: GPU einsum batched-matmul/transpose/diagonal/outer/trace -> correct CPU fallback** (planned 2026-07-02, done)
  - **Goal:** these 5 GPU einsum patterns return correct results via GPU->CPU readback + delegate to the correct CPU einsum, instead of honest-erroring.
  - **Design:** In crates/tenflowers-core/src/ops/einsum/gpu.rs, the both-GPU-branch honest errors (and the diagonal/outer/trace non-GPU-else-arm errors) were replaced with `to_cpu()` readback + `crate::ops::einsum::einsum(...)` delegate.
  - **Files:** crates/tenflowers-core/src/ops/einsum/gpu.rs, patterns.rs, einsum_ops.rs (comments only)
  - **Verified:** all 5 patterns confirmed in gpu.rs — batched matmul (`"bij,bjk->bik"`), transpose (`"ij->ji"`), diagonal (`"ii->i"`), outer product (`"i,j->ij"`), and trace (`"ii->"`) each call `.to_cpu()?` followed by `crate::ops::einsum::einsum(...)`.
  - **Tests:** CPU-correctness tests for all 5 patterns; guarded GPU end-to-end tests under `#[cfg(feature="gpu")]`.

- [x] **D: Real wgpu device-capability queries (queryable subset)** (planned 2026-07-02, done)
  - **Goal:** AdvancedKernelManager returns real vendor/max_threads_per_block/shared_memory_size/supports_fp16 from the live wgpu adapter instead of fabricating/honest-erroring; fields wgpu genuinely can't expose stay honest errors.
  - **Design:** The `wgpu::Adapter` is threaded via a new `GpuAdapterCapabilities` struct and `get_gpu_adapter_capabilities(device_id)` function (crates/tenflowers-core/src/device/context.rs) into `advanced_kernel_manager.rs`'s device-capability detection.
  - **Files:** crates/tenflowers-core/src/gpu/advanced_kernel_manager.rs, crates/tenflowers-core/src/device/context.rs
  - **Verified:** `pub struct GpuAdapterCapabilities` and `pub fn get_gpu_adapter_capabilities(device_id: usize) -> Result<GpuAdapterCapabilities>` both confirmed present in device/context.rs; `advanced_kernel_manager.rs` confirmed referencing `GpuAdapterCapabilities` fetched via `crate::device::get_gpu_adapter_capabilities`.
  - **Tests:** pure-helper unit tests (no GPU needed) + guarded GPU integration test.

- [x] **E: Zarr blosc codec (header+block table+inner codecs+shuffle/bitshuffle)** (planned 2026-07-02, done)
  - **Goal:** Zarr chunks compressed with blosc decode correctly (blosclz/lz4/snappy/zlib/zstd inner codecs, byte-shuffle and bit-shuffle filters), replacing the previous honest error.
  - **Design:** Implemented as a module directory (not a single file) — crates/tenflowers-dataset/src/formats/blosc/{mod.rs,blosclz.rs,filters.rs,tests.rs} (1,278 lines total) — parsing the 16-byte blosc header + block offset table, dispatching per-block by flags to the inner codec (zstd/snappy via oxiarc_archive, lz4 via new oxiarc-lz4, zlib via new oxiarc-deflate, blosclz hand-written), then inverse byte/bit-shuffle by typesize. Wired into `zarr.rs`'s `decompress_chunk_data`'s `"blosc"` arm.
  - **Files:** crates/tenflowers-dataset/src/formats/{zarr.rs,blosc/ (mod.rs,blosclz.rs,filters.rs,tests.rs)}, crates/tenflowers-dataset/Cargo.toml, /Users/kitasan/work/tenflowers/Cargo.toml (workspace pins for oxiarc-lz4/oxiarc-deflate/oxiarc-snappy, all 0.3.4)
  - **Verified:** `pub fn decompress(compressed: &[u8]) -> Result<Vec<u8>>` confirmed present in formats/blosc/mod.rs; zarr.rs confirmed dispatching `"blosc" => super::blosc::decompress(compressed_data)`. Encoder is not implemented (decode-only), which is documented as intentional scope in CHANGELOG.md, not a gap.
  - **Tests:** crafted blosc buffers per inner codec + memcpy + shuffle/bitshuffle round-trips (tests.rs is 52,599 bytes); real success tests replace the former test_decompress_chunk_blosc_is_honest_error.

- [x] **F: Audio decode via already-present symphonia+rubato** (planned 2026-07-02, done)
  - **Goal:** load_audio_file decodes real WAV/MP3/FLAC samples via symphonia; get_audio_info reports real sample_rate/channels/duration/num_samples.
  - **Design:** symphonia probe/decode + rubato resampling wired into crates/tenflowers-dataset/src/formats/audio.rs's `load_audio_file` and `get_audio_info`. OGG stays honestly unsupported (symphonia ogg/vorbis features not enabled) — documented, not faked.
  - **Files:** crates/tenflowers-dataset/src/formats/audio.rs
  - **Verified:** `fn get_audio_info` and `fn load_audio_file` both confirmed present and using real `symphonia::default::get_probe()`/`get_codecs()` decode calls (not a stub); symphonia updated 0.5 -> 0.6, rubato 2.0 -> 3.0 in this cycle.
  - **Tests:** hand-crafted WAV round-trip in std::env::temp_dir(); test_load_audio_file_is_honest_error and test_get_audio_info_does_not_fabricate_duration updated to assert real decoded values.

---

Copyright 2025-2026 COOLJAPAN OU (Team KitaSan) · contact@cooljapan.tech
