# TenfloweRS Meta Crate TODO

This file tracks tasks specific to the meta crate (`tenflowers`), which serves as a convenience re-export layer for all TenfloweRS subcrates.

## Current Status

- ✅ Meta crate structure created
- ✅ Basic re-exports from all subcrates
- ✅ Prelude module with common imports
- ✅ Feature flag propagation
- ✅ Documentation with examples
- ✅ README and TODO files

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
- [ ] Ensure all feature flags are properly tested

### Testing
- [x] Add integration tests using prelude (done 2026-04-19: tenflowers/tests/integration_test.rs, prelude_surface.rs, tensor_macro.rs, type_aliases.rs, feature_flags_test.rs)
- [ ] Test feature flag combinations
- [ ] Add CI tests for meta crate
- [ ] Test compile-time with minimal features
- [x] Verify all examples compile with meta crate (done 2026-04-19: cargo check + doctests pass)

## Medium Priority

### Developer Experience
- [ ] Add error type unification if needed
- [ ] Consider adding re-export for common Result types
- [ ] Add version checking between subcrates
- [ ] Create unified logging/tracing interface
- [ ] Add common utility functions module

### Documentation
- [x] Add architecture diagram to README (done 2026-04-20: README.md)
- [x] Create "Quick Reference" guide (done 2026-04-20: docs/QUICK_REFERENCE.md)
- [ ] Add performance comparison charts
- [x] Document feature flag combinations (done 2026-04-20: README.md Feature Flags section)
- [x] Add troubleshooting section (done 2026-04-20: docs/TROUBLESHOOTING.md)

### Tooling
- [x] Add publish script for meta crate
- [x] Ensure version bumps are synchronized
- [ ] Add changelog automation
- [x] Create release checklist
- [x] Add deprecation warnings for API changes

## Low Priority

### Future Enhancements
- [x] Consider adding commonly used type aliases (done 2026-04-19: tenflowers/src/type_aliases.rs with Tensor1D/2D/3D/4D, Vector, Matrix, Scalar aliases for f32/f64/generic)
- [x] Add experimental features flag
- [x] Create "batteries-included" preset features
- [ ] Add platform-specific optimizations
- [x] Consider stability guarantees for prelude (done 2026-04-19: docs/PRELUDE_STABILITY.md created; prelude module doc references it)

### Ecosystem Integration
- [ ] Ensure compatibility with common Rust ML crates
- [ ] Add interop examples with other frameworks
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
