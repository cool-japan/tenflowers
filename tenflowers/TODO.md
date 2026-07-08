# TenfloweRS Meta Crate TODO

This file tracks tasks specific to the meta crate (`tenflowers`), which serves as a convenience re-export layer for all TenfloweRS subcrates.

## Current Status

- ✅ Meta crate structure created
- ✅ Basic re-exports from all subcrates
- ✅ Prelude module with common imports
- ✅ Feature flag propagation
- ✅ Documentation with examples
- ✅ README and TODO files

### v0.1.2 Release Status (2026-07-07)

All items below remain complete as of the v0.1.2 release. Per `CHANGELOG.md`'s
`[0.1.2] - 2026-07-07` entry, the meta crate's `error`, `logging`, `utils`,
`platform`, and `version_check` modules plus the `ml_compat` and
`feature_flags_comprehensive` integration test suites (all built 2026-06-10,
tracked below) shipped unchanged in this release — no new meta-crate API
surface landed since the last update to this file. The bulk of v0.1.2's work
landed in the subcrates (`tenflowers-core`, `-autograd`, `-neural`, `-dataset`,
`-ffi`); see their own `TODO.md`/`CHANGELOG.md` entries for detail. Workspace-wide
regression: **14,289 tests passing, 39 skipped, 0 failures, 0 warnings**
(`cargo nextest run --workspace --all-features`, verified 2026-07-07).

## High Priority

### Documentation & Examples
- [x] Add comprehensive doc examples for prelude usage patterns (done 2026-04-19: 4 narrative prelude examples in lib.rs: minimal-model, MNIST train loop, regression Adam, inference)
- [x] Create migration guide from TensorFlow to TenfloweRS (done 2026-04-20: docs/MIGRATION_FROM_TENSORFLOW.md)
- [x] Add doctests for all prelude re-exports (done 2026-04-19: 23 doctests pass; lib.rs no_run blocks + macros.rs runnable doctest + type_aliases.rs doctests)
- [x] Create "Getting Started" tutorial in README (done 2026-04-20: README.md)
- [x] Add comparison table with PyTorch/TensorFlow APIs (done 2026-04-20: README.md)

### API Surface
- [x] Review prelude exports - ensure most common types are included (done 2026-04-19: prelude includes MultiHeadAttention, RMSNorm, TransformerEncoder/Decoder, GRU, LSTM, RNN, all optimizers + loss fns)
- [x] Add convenience macros (e.g., `tensor![]` for tensor creation) (done 2026-04-19: tenflowers/src/macros.rs with tensor! macro + tensor_macro tests)
- [x] Consider adding `nn` module alias for `neural` (done 2026-04-19: pub mod nn in lib.rs)
- [x] Add `data` module alias for `dataset` (done 2026-04-19: pub mod data in lib.rs)
- [x] Ensure all feature flags are properly tested (COMPLETED 2026-06-10 — tests/feature_flags_comprehensive.rs: 48 tests covering default features, gpu/simd/serialize/onnx/experimental guards, version_check, logging macros, error/Result types, prelude completeness)

### Testing
- [x] Add integration tests using prelude (done 2026-04-19: tenflowers/tests/integration_test.rs, prelude_surface.rs, tensor_macro.rs, type_aliases.rs, feature_flags_test.rs)
- [x] Test feature flag combinations (done 2026-06-10: feature_flags_comprehensive.rs extended with 15 new combination/matrix/preset tests: std-only, std+parallel, std+simd, std+serialize, std+parallel+simd, std+parallel+gpu, std+experimental, minimal preset, standard preset, full preset, plus 5 Cargo.toml matrix coherence tests)
- [x] Add CI tests for meta crate (done 2026-06-10: .github/workflows/meta-crate-ci.yml — check-default, check-minimal, test, version-check, feature-combinations, clippy, cross-platform jobs)
- [x] Test compile-time with minimal features (done 2026-06-10: meta-crate-ci.yml feature-combinations matrix includes "", "std", "std,parallel,simd", "full" + --no-default-features check)
- [x] Verify all examples compile with meta crate (done 2026-04-19: cargo check + doctests pass)

## Medium Priority

### Developer Experience
- [x] Add error type unification if needed (done 2026-06-10: tenflowers/src/error.rs — FrameworkError wrapping TensorError + ArenaError + Other; pub use error::Result at crate root)
- [x] Consider adding re-export for common Result types (done 2026-06-10: pub use error::Result in lib.rs and prelude)
- [x] Add version checking between subcrates (done 2026-06-10: tenflowers/src/version_check.rs — SubcrateVersion, subcrate_versions(), check_version_consistency(), assert_versions_consistent())
- [x] Create unified logging/tracing interface (done 2026-06-10: tenflowers/src/logging.rs — LogLevel, set_log_level(), set_log_backend(), init_from_env(), log_info!/log_warn!/log_error!/log_debug!/log_trace! macros)
- [x] Add common utility functions module (done 2026-06-10: tenflowers/src/utils.rs — clamp, softmax, sigmoid, log_softmax, lerp, next_power_of_two, bytes_to_human_readable, num_elements, shape_to_strides, flat_index)

### Documentation
- [x] Add architecture diagram to README (done 2026-04-20: README.md)
- [x] Create "Quick Reference" guide (done 2026-04-20: docs/QUICK_REFERENCE.md)
- [x] Add performance comparison charts (done 2026-06-10: added Performance Benchmarks section to tenflowers/README.md — CPU/GPU throughput tables for key tensor/data-pipeline ops)
- [x] Document feature flag combinations (done 2026-04-20: README.md Feature Flags section)
- [x] Add troubleshooting section (done 2026-04-20: docs/TROUBLESHOOTING.md)

### Tooling
- [x] Add publish script for meta crate
- [x] Ensure version bumps are synchronized
- [x] Add changelog automation (done 2026-06-10: scripts/generate_changelog.sh — extracts feat/fix/perf/refactor/docs/test/deps sections from git log since last vX.Y.Z tag)
- [x] Create release checklist
- [x] Add deprecation warnings for API changes

## Low Priority

### Future Enhancements
- [x] Consider adding commonly used type aliases (done 2026-04-19: tenflowers/src/type_aliases.rs with Tensor1D/2D/3D/4D, Vector, Matrix, Scalar aliases for f32/f64/generic)
- [x] Add experimental features flag
- [x] Create "batteries-included" preset features
- [x] Add platform-specific optimizations (COMPLETED 2026-06-10 — src/platform.rs: Platform enum (X86_64/Aarch64/Wasm32/Other), current_platform(), SimdCapabilities struct, detect_simd_capabilities(), 10 unit tests)
- [x] Consider stability guarantees for prelude (done 2026-04-19: docs/PRELUDE_STABILITY.md created; prelude module doc references it)

### Ecosystem Integration
- [x] Ensure compatibility with common Rust ML crates (done 2026-06-10: tests/ml_compat.rs with 19 tests: bytemuck Pod cast_slice roundtrips for f32/f64/u8, num_traits::Float generic function interop, DType exhaustive match coverage for all 17 variants, numeric stability for add/sub/mul/softmax/sigmoid)
- [x] Add interop examples with other frameworks (COMPLETED 2026-06-10 — examples/basic_example.rs (tensor creation/arithmetic/matmul via prelude), version_check_example.rs (subcrate_versions/assert_versions_consistent), logging_example.rs (log levels, custom backend, macros))
- [x] Create conversion utilities for ndarray (done 2026-04-19: tenflowers/src/interop/ndarray.rs with from_ndarray/to_ndarray/round_trip/from_slice_with_shape + 7 unit tests)
- [x] Add serialization format helpers (done 2026-04-19: tenflowers/src/io.rs with save_tensor/load_tensor wrappers, serialize feature gated, stub for no-feature builds)
- [x] Consider ONNX import/export helpers (done 2026-04-19: #[cfg(feature="onnx")] pub mod onnx re-exports tenflowers_neural::onnx surface in tenflowers/src/lib.rs)

## Completed

- ✅ Created meta crate directory structure
- ✅ Set up Cargo.toml with proper workspace dependencies
- ✅ Created lib.rs with re-exports
- ✅ Added prelude module
- ✅ Added common module for utilities
- ✅ Created comprehensive README
- ✅ Added version information
- ✅ Set up feature flags
- ✅ Added workspace integration

## Notes

### Design Principles

1. **Minimal Re-export Policy**: Only re-export the most commonly used types and functions
2. **Feature Parity**: All features from subcrates should be accessible
3. **Documentation First**: Every public item should have examples
4. **Zero Overhead**: Meta crate should add no runtime cost
5. **Stability**: Prelude should be stable even if internal crates change

### Dependencies to Watch

- All subcrates must use compatible versions
- Feature flags must be properly propagated
- SciRS2 ecosystem version alignment

### Future Considerations

- May need to add macro re-exports
- Consider stabilizing prelude separately from subcrates
- Think about backwards compatibility strategy
- Plan for 1.0 API stabilization
