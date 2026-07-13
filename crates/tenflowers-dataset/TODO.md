# TenfloweRS Dataset TODO & Roadmap (v0.2.1)

v0.1.1 focus: data loading and preprocessing capabilities and forward development plan.

## v0.1.2 — Honesty Hardening (2026-06-22), extended through 2026-07-07

- Removed production lock-poison panics via signature-preserving recovery; a
  poisoned lock no longer aborts the process.
- Download `verify_checksum` returned `Ok(true)` (integrity never checked) →
  real SHA-256/CRC-32 verification (algorithm inferred from an `"algo:hex"`
  prefix or digest length).
- Audio file "loading" returned a synthetic 440 Hz sine wave → real
  Symphonia-based decoding; `get_audio_info` now probes/decodes the
  container for real `sample_rate`/`channels`/`num_samples`/`duration`
  instead of fabricating them.
- Zarr lz4/zstd chunks now really decompress via oxiarc; the `blosc` codec
  is no longer an honest-error stub — `formats::blosc` is a new from-scratch
  pure-Rust Blosc chunk decoder (BloscLZ own port, LZ4/Snappy/Zlib/Zstd via
  oxiarc, byte/bit-shuffle filters) wired into the zarr `"blosc"` dispatch.
- `memory_pool::MemoryBlock`/`MemoryPool`: Miri-confirmed UB from hardcoded
  1-byte allocation alignment fixed via new `allocate_aligned`/
  `allocate_exact`; `with_pool_capacity::<T>()` now requests
  `mem::align_of::<T>()` correctly.
- `arrow_advanced::ArrowPredicate::In` and `arrow::ArrowArrayExt::to_tensor`
  were stubs (`not yet implemented` / generic error) → real implementations
  (OR-of-equality-masks; real Arrow-array-to-Tensor conversion).
- `transforms::noise::AddNoise::transform` always added zero noise → real
  Box-Muller Gaussian noise.
- `numa_scheduler` build-gating bug fixed (Linux-affinity code depended on
  `libc`, which is only linked under the `numa` feature, but was gated on
  `target_os = "linux"` alone).
- New `formats::tfrecord_advanced` (`SequenceExample` reader, masked-CRC
  integrity check), `gpu_transforms::{affine,perspective,elastic,equalize}`
  (WGSL-backed GPU image transforms), `simd_transforms::functional` (SIMD
  preprocessing functions), and `transform_arena::TransformArena`
  (reusable-buffer arena).
- Benchmark/CPU metrics were hardcoded or random → real `/proc` measurements
  or honest sentinels.

## v0.2.0 — Test Coverage Hardening (2026-07-13)

No production-code behavior changed this cycle. Test coverage was expanded
for previously-uncovered paths:
- `zero_copy::MemoryMappedFileDataset::from_file` (the crate's one real
  `mmap()`-backed `unsafe` block) had zero test coverage anywhere in the
  crate — a regression test now exercises the real file-backed mmap path
  end-to-end (via `std::env::temp_dir()`, not a hardcoded path). Skipped
  under Miri, which does not model file-backed memory mappings at all
  (confirmed: `error: unsupported operation: Miri does not support
  file-backed memory mappings`, not an isolation-flag issue).
- `simd_transforms::{normalization, statistics, image_processing,
  element_wise}` gained end-to-end tests through the public `Transform`
  API, exercising the scalar fallback path (the only path Miri can
  interpret, and the same path any non-AVX2 x86_64 host or non-x86_64 host
  such as aarch64 runs in practice); these files previously had zero test
  coverage.
- Verified 2026-07-11: `cargo nextest run -p tenflowers-dataset
  --all-features` → **698 tests run: 698 passed, 0 skipped** (up from 660).

## 1. Current Capabilities

### Core Data Pipeline
- **Dataset Trait**: Comprehensive dataset abstraction with builder pattern support
- **Transform Pipeline**: Composable data transformation system with method chaining
- **Memory Management**: Smart caching with predictive prefetch and memory pool management
- **Performance**: SIMD-accelerated transforms with runtime CPU fallback for compatibility

### Format Support & I/O
- **Structured Data**: JSON/JSONL with nested array flattening and configurable field mapping
- **Text Processing**: Advanced NLP-focused dataset with vocabulary management and tokenization
- **Scientific Formats**: Parquet, HDF5, Audio, TFRecord support behind feature gates
- **Image Processing**: Comprehensive image format support with GPU-accelerated transforms
- **Web Formats**: WebDataset, Zarr, CSV with streaming and batch processing capabilities
- **Memory-Mapped**: Large file zero-copy access for efficient data loading

### GPU Acceleration & Optimization
- **GPU Transforms**: Selected GPU image/data transforms (crop, rotate, jitter, blur, noise, resize, flip)
- **SIMD Operations**: Color conversion, statistics computation, histogram analysis with vectorization
- **Caching Strategy**: Predictive smart cache with pattern-based prefetch algorithms
- **Memory Efficiency**: Buffer reuse, streaming optimization, and memory pool management

### Advanced Features
- **Text Processing**: Vocabulary building, tokenization strategies (word/character/subword), label extraction
- **Statistics & Analysis**: Histogram computation, dataset statistics, and data quality metrics
- **Streaming Support**: Lazy loading for large datasets with efficient memory usage
- **Data Validation**: Schema validation and error handling with comprehensive diagnostics

### SciRS2 Integration
- **Complete Migration**: 100% usage of scirs2-core for scientific computing primitives
- **Foundation**: Built on scirs2-autograd for array operations with array! macro support
- **Ecosystem**: Seamless integration with broader SciRS2/NumRS2 scientific computing stack

## 2. Current Gaps & Limitations

### Distributed & Streaming
- **Streaming Loaders**: ✅ COMPLETED - Comprehensive distributed streaming with deterministic partitioning
- **Distributed Coordination**: ✅ COMPLETED - Multi-worker coordinator with health monitoring and load balancing
- **Partition Strategy**: ✅ COMPLETED - Advanced partitioning (RoundRobin, Contiguous, Hash, Range, Stratified, Adaptive)

### Format Integration
- **Arrow Integration**: Deep Apache Arrow integration incomplete, limited zero-copy operations
- **Unified Reader**: No unified format abstraction layer for cross-format iteration
- **Schema Validation**: Limited schema/validation diagnostics consistency across formats
- **Advanced HDF5**: ✅ COMPLETED - `hdf5_advanced.rs` (chunked/attribute/tree/slice readers, feature-gated)

### Error Handling & Diagnostics
- **Error Taxonomy**: Limited error taxonomy alignment with core crate patterns
- **Diagnostics**: Inconsistent error messaging and validation across different formats
- **Debug Tools**: Limited debugging and profiling tools for data pipeline optimization

### Performance & Optimization
- **Adaptive Caching**: No auto-tuning cache policies for different access patterns
- **Throughput Analysis**: Limited benchmarking harness for ingest and transform performance
- **Memory Optimization**: Room for improvement in memory usage patterns and allocation strategies

### Honest-error deferrals (post-2026-06-22 sweep; fail loudly, not faked)
- [x] **Audio file decoding**: ~~no pure-Rust codec wired → honest error~~ RESOLVED — now
  decodes for real via Symphonia (WAV/FLAC/MP3/OGG), with resampling and
  normalization; `get_audio_info` probes/decodes real `sample_rate`/
  `channels`/`num_samples`/`duration` instead of fabricating them.
- [x] **Zarr `blosc` codec**: ~~returns an honest error~~ RESOLVED — new
  `formats::blosc` pure-Rust decoder (BloscLZ own port, LZ4/Snappy/Zlib/Zstd
  via oxiarc, byte/bit-shuffle filters) wired into the zarr dispatch;
  lz4/zstd/gzip continue to decompress for real via oxiarc.
- [ ] **MessagePack serialization**: `msgpack` feature flag is default-on
  and pulls in `rmp-serde`, but no serializer/deserializer is implemented
  anywhere in the crate — the dependency is currently unused.

### Pre-existing build issue (RESOLVED 2026-07-07)
- [x] Building with `--no-default-features` previously failed due to a
  feature-gating gap in `numa_scheduler.rs` (Linux-affinity code depended on
  `libc`, only linked under the `numa` feature, but was gated on
  `target_os = "linux"` alone) — now correctly gated on both; verified
  `cargo check -p tenflowers-dataset --no-default-features` builds clean.

## 3. Near-Term Roadmap

### Priority 1: Distributed & Streaming ✅ COMPLETED
1. ✅ **Streaming Loaders**: Deterministic partitioning specification for distributed training
   - Implemented `StreamingShardLoader` with 6 partition strategies
   - Deterministic shuffling with seeded RNG
   - Prefetching and buffering support
2. ✅ **Shard-Aware Loaders**: Deterministic data sharding with consistent partitioning across workers
   - Hash-based, range-based, stratified, and adaptive partitioning
   - Reproducible data distribution across runs
3. ✅ **Distributed Coordinator**: Multi-worker dataset prefetch and coordination system
   - `StreamCoordinator` with worker health monitoring
   - Dynamic load balancing based on throughput metrics
   - Checkpoint coordination across workers
4. ✅ **Partition Strategy**: Advanced partitioning algorithms for balanced data distribution
   - 6 strategies: RoundRobin, Contiguous, HashBased, RangeBased, Stratified, Adaptive
   - Configurable rebalancing thresholds
   - Custom partition strategy support

### Priority 2: Format & Integration
5. **Unified Format Reader**: Abstraction layer for cross-format iteration and processing
6. **Arrow Zero-Copy Integration**: Comprehensive Apache Arrow integration with zero-copy where possible
7. **Schema Validator**: Unified schema validation system across all supported formats
8. **Advanced Format Features**: Enhanced HDF5, Parquet, and TFRecord feature support

### Priority 3: Performance & Quality
9. **Adaptive Prefetch Policy**: Auto-tuning prefetch algorithms based on access patterns
10. **Cache Telemetry Metrics**: Comprehensive cache performance monitoring and optimization
11. **Throughput Benchmark Harness**: Performance measurement and regression detection system
12. **Memory Optimization**: Advanced memory usage optimization and allocation strategies

### Priority 4: Error Handling & Diagnostics
13. **Error Taxonomy Mapping**: Align error handling patterns with core crate standards
14. **Diagnostics Enhancement**: Improved error messaging and validation across formats
15. **Debug Tools**: Comprehensive debugging and profiling tools for data pipeline analysis
16. **Data Quality Metrics**: Advanced data quality assessment and drift detection

## 4. Mid-Term Roadmap

### Advanced Data Processing
- **On-the-fly Augmentation**: GPU kernel fusion for real-time data augmentation
- **Columnar Statistics**: Persistent cache service for columnar data statistics
- **Data Quality & Drift**: Advanced data quality monitoring and drift detection hooks
- **Real-time Processing**: Streaming data processing with low-latency requirements

### Distributed Systems
- **Multi-Node Coordination**: Advanced multi-node dataset coordination and management
- **Federated Data**: Federated data loading with privacy preservation techniques
- **Cloud Integration**: Cloud-native data loading with object storage optimization
- **Edge Processing**: Edge device data processing and optimization

### Format & Ecosystem
- **Custom Format API**: Pluggable custom format implementation system
- **Data Versioning**: Data versioning and lineage tracking capabilities
- **Metadata Management**: Advanced metadata management and search capabilities
- **Format Conversion**: Automated format conversion and optimization tools

## 5. Active TODO Items

### Immediate Development Tasks
- [x] **Shard Loader Spec**: Design deterministic partitioning specification (COMPLETED)
  - Implemented `distributed_streaming.rs` module with comprehensive features
  - 25+ test cases covering all functionality
  - Example code in `examples/distributed_streaming_example.rs`
  - Documentation in `docs/distributed_streaming.md`
- [x] **Unified Reader Trait**: Draft format abstraction layer design (COMPLETED 2026-04-19)
- [x] **Arrow Zero-Copy Prototype**: Implement initial Apache Arrow integration (COMPLETED 2026-04-19)
- [x] **Cache Telemetry System**: Metrics collection for cache performance (COMPLETED 2026-04-19)
- [x] **Error Taxonomy Mapping**: Align error patterns with core crate standards (COMPLETED 2026-04-19)

### Performance & Optimization
- [x] **Adaptive Prefetch Policy**: Auto-tuning cache policy implementation (COMPLETED 2026-04-19 — PidAdaptiveController with PID+anti-windup in adaptive_prefetch.rs)
- [x] **Throughput Benchmark Setup**: Performance harness for data pipeline analysis (COMPLETED 2026-04-19 — benches/throughput.rs with Criterion, raw_get/dataloader_workers/transform_chain groups)
- [x] **Memory Usage Optimization**: Enhanced memory allocation and usage patterns (COMPLETED 2026-06-10 — TransformArena in transform_arena.rs; Rc<Cell<T>> interior mutability, O(1) acquire, reuse tracking, 12 tests passing)
- [x] **SIMD Optimization**: Advanced SIMD acceleration for transform operations (COMPLETED 2026-06-10 — simd_transforms/ module with normalization, element_wise, statistics, image_processing, convolution, matrix_ops, functional sub-modules; zero clippy warnings)
- [x] **GPU Transform Expansion**: Additional GPU-accelerated data transforms (COMPLETED 2026-06-10 — affine.rs/affine.wgsl 2×3 matrix warp; perspective.rs/perspective.wgsl 3×3 homography warp; elastic.rs/elastic_distortion.wgsl SimoNikolenko elastic deformation with CPU Gaussian smoothing; equalize.rs/histogram_equalize.wgsl two-pass histogram equalization with GPU atomic histogram build + CPU CDF + GPU remap; 34 tests all passing)

### Integration & Quality
- [x] **Schema Validation**: Unified validation system across formats (COMPLETED 2026-04-19 — FieldDiff/ValidationReport/validate_full/strict/lenient in schema_validator.rs)
- [x] **Advanced HDF5 Features**: Enhanced HDF5 support and optimization (COMPLETED 2026-06-10 — hdf5_advanced.rs: Hdf5ChunkedReader/Hdf5AttributeReader/Hdf5TreeWalker/DatasetInfo/Hdf5SliceReader; full #[cfg(feature="hdf5")] gating + stubs; 5 tests passing)
- [x] **Format Integration**: Improved Parquet, TFRecord, and other format support (COMPLETED 2026-06-10 — parquet_advanced.rs: read_columns/ParquetRowGroupReader/read_filtered/FilterPredicate/inspect_schema; tfrecord_advanced.rs: TfRecordRawReader/TfRecordSequenceReader/masked_crc32/parse_feature_proto; 29 tests passing)
- [x] **Documentation**: Comprehensive data loading concepts and usage guide (COMPLETED 2026-04-19 — expanded //! docs in lib.rs with PipelineInspector/DriftMetrics/PID/SchemaValidation sections + runnable doctest)
- [x] **API Stabilization**: Prepare dataset APIs for stable release (COMPLETED 2026-04-19 — doc comments on all new public items, verified re-exports)

### Infrastructure Tasks
- [x] **Distributed Coordination**: Multi-worker dataset coordinator implementation (COMPLETED 2026-04-19 — StreamCoordinator in distributed_streaming/, StreamingShardLoader in distributed_sharding.rs, 6 partition strategies; see section 3 Priority 1)
- [x] **Streaming Enhancement**: Advanced streaming capabilities and optimization (COMPLETED 2026-04-19 — streaming_optimized.rs, stream_prefetch_optimizer.rs, distributed_streaming/ with full coordinator; see section 3 Priority 1)
- [x] **Debug Tools**: Data pipeline debugging and profiling tool development (COMPLETED 2026-04-19 — InspectablePipeline/InspectionEvent/PipelineInspectionReport in debug_tools.rs)
- [x] **Quality Metrics**: Data quality assessment and monitoring implementation (COMPLETED 2026-04-19 — PSI/KS/JSD functions + DriftReport + compute_drift in data_quality.rs)

## 6. Advanced Research Areas

### Data Processing Innovation
- **AutoML Data**: Automated data preprocessing and feature engineering
- **Neural Data Processing**: Learning-based data preprocessing and augmentation
- **Federated Analytics**: Privacy-preserving data analysis and processing
- **Edge AI Data**: Optimized data processing for edge AI applications

### Performance Research
- **Zero-Copy Processing**: Advanced zero-copy data processing techniques
- **Compression Optimization**: Intelligent data compression for storage and transfer
- **Cache Intelligence**: AI-driven cache optimization and prefetch strategies
- **Hardware Acceleration**: Specialized hardware acceleration for data processing

### Ecosystem Integration
- **Cloud Native**: Advanced cloud-native data processing and optimization
- **Streaming Systems**: Integration with real-time streaming data systems
- **Data Mesh**: Data mesh architecture and decentralized data management
- **MLOps Integration**: Production MLOps pipeline integration and optimization

## 7. Deferred Items

### Advanced Features
- **Full Distributed Engine**: Complete distributed data processing system
- **Advanced Privacy**: Differential privacy and secure multi-party computation
- **Real-time Analytics**: Real-time data analytics and processing capabilities
- **Custom Hardware**: Specialized hardware backend integration

### Infrastructure
- **Production Serving**: Production data serving and optimization infrastructure
- **Monitoring**: Advanced data pipeline monitoring and observability
- **Governance**: Data governance, compliance, and auditing capabilities
- **Research Integration**: Integration with cutting-edge data research frameworks

---

**v0.1.2 Status** (2026-07-07): Production-ready data loading capabilities with comprehensive format support (including a new pure-Rust Zarr Blosc decoder and TFRecord `SequenceExample` support), GPU-accelerated transforms (affine/perspective/elastic/histogram-equalize), SIMD preprocessing, a Miri-verified memory pool, and SciRS2 integration — 660 tests passing with `--all-features`.

**v0.2.0 Status** (2026-07-13): No production-code changes this cycle; test coverage expanded for previously-uncovered `mmap`/SIMD-transform paths — 698 tests passing with `--all-features`. Forward development focuses on distributed loading and advanced format integration.
