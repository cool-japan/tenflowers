# TenfloweRS Autograd API Stabilization Plan

## Status: Alpha → Beta Preparation

**Current Version**: 0.1.0-alpha.2
**Target Version**: 0.1.0-beta.1
**Target Date**: Q1 2025

---

## Executive Summary

This document outlines the plan to stabilize the TenfloweRS Autograd API for beta release. The primary goals are:

1. **API Consistency**: Ensure all public APIs follow consistent naming and design patterns
2. **Backwards Compatibility**: Define stability guarantees and deprecation policy
3. **Documentation**: Complete API documentation with examples
4. **Testing**: Comprehensive test coverage for all public APIs
5. **Performance**: Benchmark and optimize critical paths

---

## Table of Contents

1. [Current API Status](#current-api-status)
2. [Breaking Changes](#breaking-changes)
3. [API Improvements](#api-improvements)
4. [Stability Guarantees](#stability-guarantees)
5. [Migration Guide](#migration-guide)
6. [Release Checklist](#release-checklist)

---

## Current API Status

### Core APIs (Stable)

These APIs are considered stable and will not change in beta:

#### GradientTape

```rust
pub struct GradientTape {
    pub fn new() -> Self;
    pub fn watch<T>(&self, tensor: Tensor<T>) -> TrackedTensor<T>;
    pub fn gradient<T>(
        self,
        targets: &[TrackedTensor<T>],
        sources: &[TrackedTensor<T>],
    ) -> Result<Vec<Tensor<T>>>;
}
```

**Status**: ✅ Stable
**Changes**: None planned

#### TrackedTensor

```rust
pub struct TrackedTensor<T> {
    pub fn tensor(&self) -> &Tensor<T>;
    // All tensor operations available
}
```

**Status**: ✅ Stable
**Changes**: None planned

### Configuration APIs (Needs Review)

#### CheckpointConfig

```rust
pub struct CheckpointConfig {
    pub strategy: CheckpointStrategy,
    pub checkpoint_every_n: usize,
    pub min_compute_cost: usize,
    pub memory_budget_mb: Option<usize>,
}
```

**Status**: ⚠️ Needs Review
**Issues**:
- Field visibility (should some be private?)
- Builder pattern vs. struct literal
- Validation logic

**Proposed Changes**:

```rust
pub struct CheckpointConfig {
    strategy: CheckpointStrategy,  // Private
    checkpoint_every_n: usize,
    min_compute_cost: usize,
    memory_budget_mb: Option<usize>,
}

impl CheckpointConfig {
    pub fn new(strategy: CheckpointStrategy) -> Self;
    pub fn with_interval(mut self, n: usize) -> Self;
    pub fn with_compute_threshold(mut self, cost: usize) -> Self;
    pub fn with_memory_budget(mut self, mb: usize) -> Self;

    // Getters
    pub fn strategy(&self) -> CheckpointStrategy;
    pub fn interval(&self) -> usize;
    // ...
}
```

#### AmpConfig

**Status**: ⚠️ Needs Review
**Issues**:
- Too many public fields
- Complex configuration

**Proposed Changes**: Use builder pattern with validation

### Utility APIs (Stable)

#### gradient_ops Module

```rust
pub fn clip_by_global_norm(
    gradients: &[Tensor<f32>],
    max_norm: f32,
) -> Result<(Vec<Tensor<f32>>, f32)>;

pub fn clip_by_value(
    gradients: &[Tensor<f32>],
    clip_value: f32,
) -> Result<Vec<Tensor<f32>>>;

pub fn compute_gradient_statistics(
    gradients: &[Tensor<f32>]
) -> Result<GradientStatistics>;
```

**Status**: ✅ Stable (New)
**Changes**: None planned

### Advanced APIs (Experimental)

These APIs are marked as experimental and may change:

- `HybridScheduler` - May change scheduling heuristics
- `GradientFlowVisualizer` - Output format may change
- `PerformanceBenchmark` - Metrics may be added/removed
- `device_placement` module - API under review

---

## Breaking Changes

### Planned Breaking Changes for Beta

#### 1. Configuration Structs → Builder Pattern

**Current**:
```rust
let config = CheckpointConfig {
    strategy: CheckpointStrategy::Selective,
    checkpoint_every_n: 2,
    min_compute_cost: 100,
    memory_budget_mb: Some(1024),
};
```

**Beta**:
```rust
let config = CheckpointConfig::new(CheckpointStrategy::Selective)
    .with_interval(2)
    .with_compute_threshold(100)
    .with_memory_budget(1024);
```

**Rationale**:
- Better validation
- Forward compatibility
- Clearer API

**Migration**: Provided via deprecation warnings in alpha.2

#### 2. Remove Redundant Methods

**Removing**:
```rust
impl GradientTape {
    pub fn gradient_with_options(...);  // Redundant
}
```

**Use Instead**:
```rust
let tape = GradientTape::with_checkpoint_config(config);
tape.gradient(...);
```

**Rationale**: Reduces API surface, clearer intent

#### 3. Rename for Consistency

| Current | Beta | Reason |
|---------|------|--------|
| `NumericalChecker::check_gradient_central` | `NumericalChecker::validate_central_difference` | Clarity |
| `GradientMemoryProfiler::get_memory_delta` | `GradientMemoryProfiler::memory_delta` | Rust conventions |
| `PerformanceBenchmark::start_benchmark` | `PerformanceBenchmark::begin` | Brevity |

---

## API Improvements

### 1. Enhanced Error Messages

**Current**:
```rust
Err(TensorError::invalid_input("Invalid shape"))
```

**Beta**:
```rust
Err(TensorError::InvalidGradientShape {
    expected: vec![3, 4],
    actual: vec![3, 5],
    context: "gradient computation for matmul",
})
```

### 2. Const Generics Where Appropriate

**Future** (post-beta, when stable):
```rust
pub fn gradient<const N: usize>(
    self,
    targets: &[TrackedTensor<T>; N],
    sources: &[TrackedTensor<T>; N],
) -> Result<[Tensor<T>; N]>;
```

### 3. Trait-Based Configuration

**Beta Addition**:
```rust
pub trait GradientConfig {
    fn apply_to_tape(&self, tape: &mut GradientTape);
}

impl GradientConfig for CheckpointConfig { /* ... */ }
impl GradientConfig for AmpConfig { /* ... */ }
impl GradientConfig for DeterministicConfig { /* ... */ }

// Combined configuration
let tape = GradientTape::new()
    .with_config(&checkpoint_config)
    .with_config(&amp_config)
    .with_config(&deterministic_config);
```

### 4. Gradient Operation Chaining

**Beta Addition**:
```rust
use tenflowers_autograd::gradient_ops::GradientPipeline;

let pipeline = GradientPipeline::new()
    .clip_by_norm(1.0)
    .add_noise(0.01, Some(42))
    .scale(0.1);

let processed_grads = pipeline.apply(&raw_grads)?;
```

---

## Stability Guarantees

### Beta Guarantees

1. **Core APIs**: No breaking changes without major version bump
2. **Configuration**: Builder patterns locked, new options additive only
3. **Deprecation Policy**:
   - Deprecated in beta.1
   - Removal warnings in beta.2
   - Removed in 1.0
4. **Semver**: Strict semantic versioning adherence

### API Levels

| Level | Description | Examples | Stability |
|-------|-------------|----------|-----------|
| **Stable** | Core functionality | `GradientTape`, `TrackedTensor` | No breaking changes |
| **Standard** | Common features | Configuration builders | Minor breaking changes allowed |
| **Experimental** | Advanced features | Visualization, JIT | May change significantly |
| **Internal** | Not public | Tape internals | No guarantees |

### Feature Flags

```toml
[features]
default = ["std"]
std = []  # Standard library
experimental = []  # Experimental APIs
unstable = []  # Unstable/internal APIs exposed for testing
```

**Policy**: Experimental features may have breaking changes in minor versions.

---

## Migration Guide

### Alpha.2 → Beta.1

#### Configuration Changes

**Before**:
```rust
let config = CheckpointConfig {
    strategy: CheckpointStrategy::Selective,
    checkpoint_every_n: 2,
    min_compute_cost: 100,
    memory_budget_mb: Some(1024),
};
let tape = GradientTape::with_checkpoint_config(config);
```

**After**:
```rust
let tape = GradientTape::new()
    .with_checkpointing(
        CheckpointConfig::selective()
            .interval(2)
            .compute_threshold(100)
            .memory_budget(1024)
    );
```

#### Gradient Operations

**Before**:
```rust
// Manual implementation
let mut total_norm_sq = 0.0;
for grad in &grads {
    // ... compute norm
}
let clipped = // ... manual clipping
```

**After**:
```rust
use tenflowers_autograd::gradient_ops::clip_by_global_norm;
let (clipped, norm) = clip_by_global_norm(&grads, 1.0)?;
```

#### Error Handling

**Before**:
```rust
.map_err(|e| format!("Gradient error: {}", e))?
```

**After**:
```rust
.with_context(|| "Computing gradients for layer 3")?
```

---

## Release Checklist

### Documentation

- [ ] All public APIs have rustdoc comments
- [ ] All public APIs have `# Examples` section
- [ ] All public APIs have `# Panics` section (if applicable)
- [ ] All public APIs have `# Errors` section (if applicable)
- [ ] AUTOGRAD_GUIDE.md complete and reviewed
- [ ] API_STABILIZATION.md (this document) reviewed

### Testing

- [ ] Unit tests for all public functions (>90% coverage)
- [ ] Integration tests for common workflows
- [ ] Property-based tests for gradient correctness
- [ ] Benchmark suite for performance tracking
- [ ] Example programs compile and run

### Code Quality

- [ ] Zero clippy warnings with `-D warnings`
- [ ] Zero rustc warnings
- [ ] No `todo!()` in public API paths
- [ ] No `unimplemented!()` in public API paths
- [ ] All `unsafe` code documented and justified

### Performance

- [ ] Benchmarks run and tracked
- [ ] No performance regressions >5%
- [ ] Memory usage profiled
- [ ] No memory leaks detected

### Compatibility

- [ ] MSRV (Minimum Supported Rust Version) documented
- [ ] Platform support documented (x86_64, aarch64, etc.)
- [ ] Feature combinations tested
- [ ] Optional dependencies tested

### Migration

- [ ] Migration guide complete
- [ ] Deprecation warnings in place
- [ ] Example code updated
- [ ] CHANGELOG.md updated

---

## Timeline

### Phase 1: API Review (Week 1-2)

- [ ] Review all public APIs
- [ ] Identify breaking changes
- [ ] Design builder patterns
- [ ] Update documentation

### Phase 2: Implementation (Week 3-4)

- [ ] Implement builder patterns
- [ ] Add deprecation warnings
- [ ] Improve error messages
- [ ] Add gradient operation utilities

### Phase 3: Testing (Week 5-6)

- [ ] Write comprehensive tests
- [ ] Run benchmark suite
- [ ] Profile memory usage
- [ ] Fix issues

### Phase 4: Documentation (Week 7-8)

- [ ] Complete rustdoc
- [ ] Update examples
- [ ] Write migration guide
- [ ] Review AUTOGRAD_GUIDE.md

### Phase 5: Beta Release (Week 9)

- [ ] Tag beta.1 release
- [ ] Publish to crates.io
- [ ] Announce release
- [ ] Gather feedback

---

## Post-Beta Plans

### Beta → 1.0

1. **Gather Feedback**: 2-3 months of beta testing
2. **Address Issues**: Fix bugs, improve docs
3. **Finalize APIs**: Lock all standard APIs
4. **Performance**: Final optimization pass
5. **1.0 Release**: Stable API guarantee

### Future Enhancements (1.x)

These can be added without breaking changes:

- [ ] Additional gradient operations
- [ ] More checkpoint strategies
- [ ] Enhanced profiling
- [ ] Better error messages
- [ ] Performance improvements

### 2.0 Considerations

Breaking changes to consider for 2.0:

- Const generics for array sizes
- Associated type constructors
- Async gradient computation
- Generic device abstraction

---

## Appendix: API Audit

### Complete Public API Surface

#### Core Types

```rust
pub struct GradientTape;
pub struct TrackedTensor<T>;
pub struct CustomGradientFunction<T>;
```

#### Configuration

```rust
pub struct CheckpointConfig;
pub struct AmpConfig;
pub struct DeterministicConfig;
pub struct SchedulerConfig;
pub struct BenchmarkConfig;
pub struct MemoryProfileConfig;
pub struct CheckerConfig;
```

#### Enums

```rust
pub enum CheckpointStrategy;
pub enum PrecisionMode;
pub enum LossScaler;
pub enum DeterministicMode;
pub enum SchedulingStrategy;
pub enum FiniteDifferenceMethod;
```

#### Utilities

```rust
pub mod gradient_ops {
    pub fn clip_by_global_norm(...);
    pub fn clip_by_value(...);
    pub fn compute_gradient_statistics(...);
    pub fn zero_gradients(...);
    pub fn scale_gradients(...);
    pub fn accumulate_gradients(...);
    pub fn average_gradients(...);
    pub fn has_invalid_gradients(...);
    pub fn add_gradient_noise(...);
    pub struct NamedGradientAccumulator;
}
```

#### Profiling & Debugging

```rust
pub struct NumericalChecker;
pub struct GradientMemoryProfiler;
pub struct PerformanceBenchmark;
pub struct GradientFlowVisualizer;
pub struct CoverageMatrix;
```

### Deprecated APIs

These will be removed in 1.0:

```rust
#[deprecated(since = "0.1.0-beta.1", note = "Use CheckpointConfig::selective()")]
pub fn checkpoint_config_selective(...);
```

---

## Conclusion

This stabilization plan provides a clear path from alpha to beta and eventually to 1.0. The focus is on:

1. **User Experience**: Consistent, intuitive APIs
2. **Reliability**: Comprehensive testing and validation
3. **Documentation**: Clear examples and migration guides
4. **Performance**: Benchmarked and optimized
5. **Compatibility**: Clear stability guarantees

**Next Steps**:
1. Review this document with team
2. Begin Phase 1: API Review
3. Create tracking issues for each task
4. Set beta.1 target date

**Questions/Feedback**: Please open an issue on GitHub with the `api-stability` label.
