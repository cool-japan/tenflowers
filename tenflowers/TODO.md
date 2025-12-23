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
- [ ] Add comprehensive doc examples for prelude usage patterns
- [ ] Create migration guide from TensorFlow to TenfloweRS
- [ ] Add doctests for all prelude re-exports
- [ ] Create "Getting Started" tutorial in README
- [ ] Add comparison table with PyTorch/TensorFlow APIs

### API Surface
- [ ] Review prelude exports - ensure most common types are included
- [ ] Add convenience macros (e.g., `tensor![]` for tensor creation)
- [ ] Consider adding `nn` module alias for `neural`
- [ ] Add `data` module alias for `dataset`
- [ ] Ensure all feature flags are properly tested

### Testing
- [ ] Add integration tests using prelude
- [ ] Test feature flag combinations
- [ ] Add CI tests for meta crate
- [ ] Test compile-time with minimal features
- [ ] Verify all examples compile with meta crate

## Medium Priority

### Developer Experience
- [ ] Add error type unification if needed
- [ ] Consider adding re-export for common Result types
- [ ] Add version checking between subcrates
- [ ] Create unified logging/tracing interface
- [ ] Add common utility functions module

### Documentation
- [ ] Add architecture diagram to README
- [ ] Create "Quick Reference" guide
- [ ] Add performance comparison charts
- [ ] Document feature flag combinations
- [ ] Add troubleshooting section

### Tooling
- [ ] Add publish script for meta crate
- [ ] Ensure version bumps are synchronized
- [ ] Add changelog automation
- [ ] Create release checklist
- [ ] Add deprecation warnings for API changes

## Low Priority

### Future Enhancements
- [ ] Consider adding commonly used type aliases
- [ ] Add experimental features flag
- [ ] Create "batteries-included" preset features
- [ ] Add platform-specific optimizations
- [ ] Consider stability guarantees for prelude

### Ecosystem Integration
- [ ] Ensure compatibility with common Rust ML crates
- [ ] Add interop examples with other frameworks
- [ ] Create conversion utilities for ndarray
- [ ] Add serialization format helpers
- [ ] Consider ONNX import/export helpers

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
