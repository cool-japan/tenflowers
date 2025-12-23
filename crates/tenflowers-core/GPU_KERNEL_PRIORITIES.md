# GPU Kernel Expansion Priorities for TenfloweRS Core

**Version:** 0.1.0-alpha.2
**Last Updated:** 2025-11-10
**Status:** Active Planning Document

## Executive Summary

This document defines the prioritized roadmap for expanding GPU kernel coverage in TenfloweRS Core. Current GPU coverage is partial, with many operations falling back to CPU execution. This roadmap establishes clear priorities based on performance impact, usage frequency, and implementation complexity.

## Current GPU Kernel Coverage

### ✅ Implemented (High-Quality)
- Basic arithmetic: `add`, `sub`, `mul`, `div`
- Matrix multiplication: `matmul`, `batch_matmul` (WGSL compute shaders)
- Basic reductions: `sum` (partial)
- Memory operations: allocation, deallocation, transfer

### ⚠️ Partial/CPU Fallback
- Advanced reductions: `mean`, `min`, `max`, `argmin`, `argmax`
- Cumulative operations: `cumsum`, `cumprod`
- Statistical operations: `variance`, `std`, `median`
- Segment reductions: `segment_sum`, `segment_mean`
- Boolean reductions: `all`, `any`

### ❌ CPU-Only (No GPU Implementation)
- Topk operations
- Advanced sorting operations
- Sparse operations
- Complex number operations
- Some activation functions (GELU variants, Mish, etc.)

## Priority Tiers

### Tier 1: Critical Path Operations (Weeks 1-2)

**Impact:** Directly affect training/inference performance for most models
**Target:** 80% of common use cases

1. **Reduction Operations (High Priority)**
   - `sum` - Full axis-generic implementation with tree reduction
   - `mean` - Building on sum kernel
   - `max` / `min` - Tree reduction with identity element handling
   - `argmax` / `argmin` - Track indices during reduction

   **Rationale:** Used in nearly every neural network layer (loss computation, normalization, pooling)

   **Implementation Strategy:**
   - Generic reduction kernel template in WGSL
   - Workgroup-level tree reduction (shared memory)
   - Multi-stage reduction for large tensors
   - Template specialization for different operations

2. **Activation Functions (High Priority)**
   - `gelu` - Critical for transformers
   - `silu` (Swish) - Used in modern architectures
   - `mish` - Smooth activation alternative
   - `hardswish` - Mobile-optimized variant

   **Rationale:** Executed billions of times in deep networks

   **Implementation Strategy:**
   - Fused activation kernels with element-wise operations
   - Use fast approximations where acceptable
   - Leverage WGSL math functions

3. **Normalization Operations (High Priority)**
   - `layer_norm` - Essential for transformers
   - `batch_norm` - Convolutional networks
   - `group_norm` - Modern CNN alternative

   **Rationale:** Normalization is compute-intensive and benefits greatly from GPU

   **Implementation Strategy:**
   - Two-pass algorithm (mean + variance)
   - Fused operations to minimize memory bandwidth
   - Welford's online algorithm for numerical stability

### Tier 2: Performance Optimization (Weeks 3-4)

**Impact:** Significant speedups for specific workloads
**Target:** 95% of use cases

4. **Broadcasting Operations**
   - Optimize broadcast-add, broadcast-mul patterns
   - Fused broadcast + activation
   - Multi-dimensional broadcasting kernels

   **Rationale:** Constant overhead in many operations

   **Implementation Strategy:**
   - Specialized kernels for common broadcast patterns
   - Stride-based indexing in WGSL
   - Coalesced memory access patterns

5. **Cumulative Operations**
   - `cumsum` - Parallel prefix scan (Blelloch scan)
   - `cumprod` - Similar to cumsum

   **Rationale:** Used in attention mechanisms, recurrent patterns

   **Implementation Strategy:**
   - Up-sweep/down-sweep parallel scan
   - Block-level scan with cross-block communication
   - Hillis-Steele for small arrays

6. **Pooling Operations**
   - `max_pool2d` - Already implemented, optimize further
   - `avg_pool2d` - Similar to max pooling
   - `adaptive_avg_pool2d` - Dynamic sizing

   **Rationale:** Core CNN primitive

   **Implementation Strategy:**
   - Tile-based processing
   - Coalesced memory loads
   - Shared memory for filter regions

### Tier 3: Advanced Features (Weeks 5-6)

**Impact:** Enabling advanced use cases
**Target:** 99% of use cases

7. **Sorting and Selection**
   - `topk` - Critical for beam search, ranking
   - `argsort` - General sorting primitive

   **Rationale:** Essential for NLP/ranking tasks

   **Implementation Strategy:**
   - Bitonic sort for small k
   - Radix sort for large arrays
   - Approximate topk for very large tensors

8. **Statistical Operations**
   - `variance` - Building on mean
   - `std` - Square root of variance
   - `quantile` / `percentile` - Distribution analysis

   **Rationale:** Data preprocessing, analysis pipelines

   **Implementation Strategy:**
   - Two-pass algorithm with Welford's method
   - Histogram-based quantile approximation

9. **Advanced Indexing**
   - `gather` - Neural architecture search, dynamic graphs
   - `scatter` - Sparse updates, graph neural networks
   - `masked_select` - Conditional operations

   **Rationale:** Dynamic computation graphs, GNNs

   **Implementation Strategy:**
   - Atomic operations for scatter
   - Coalesced access patterns for gather
   - Warp-level primitives

### Tier 4: Specialized Workloads (Weeks 7-8)

**Impact:** Niche but important use cases
**Target:** >99% of use cases

10. **Segment Operations**
    - `segment_sum` - Graph neural networks
    - `segment_mean` - Pooling over variable-length sequences
    - `segment_max` - Dynamic graphs

    **Rationale:** GNN, point cloud processing

    **Implementation Strategy:**
    - Segment descriptor preprocessing
    - Warp-level aggregation
    - Atomic operations for cross-segment boundaries

11. **Sparse Operations**
    - Sparse matrix multiplication
    - Sparse tensor conversion
    - Sparse gradient updates

    **Rationale:** Large-scale models, efficient training

    **Implementation Strategy:**
    - COO/CSR format support
    - Specialized sparse kernels
    - Dense-sparse hybrid operations

12. **Complex Number Operations**
    - FFT optimizations
    - Complex arithmetic
    - Spectral operations

    **Rationale:** Signal processing, quantum ML

    **Implementation Strategy:**
    - Use WebGPU complex number emulation
    - Fused real/imaginary operations
    - Optimize memory layout

## Implementation Guidelines

### Kernel Development Best Practices

1. **Memory Bandwidth Optimization**
   - Use shared memory for frequently accessed data
   - Coalesce global memory accesses
   - Minimize bank conflicts
   - Vectorized loads/stores where possible

2. **Compute Efficiency**
   - Maximize occupancy (threads per SM)
   - Minimize thread divergence
   - Use warp-level primitives
   - Balance workload across workgroups

3. **Numerical Stability**
   - Use Kahan summation for reductions
   - Welford's algorithm for variance
   - Careful handling of infinities and NaNs
   - Consistent rounding modes

4. **Testing and Validation**
   - Correctness tests against CPU reference
   - Numerical precision tests (ULP tolerance)
   - Performance benchmarks vs. PyTorch/TensorFlow
   - Edge case testing (empty tensors, large sizes, etc.)

### WebGPU/WGSL Considerations

1. **Cross-Platform Compatibility**
   - Test on multiple GPU vendors (NVIDIA, AMD, Intel, Apple)
   - Handle workgroup size limits (max 256 on some platforms)
   - Feature detection and graceful fallback

2. **WGSL Limitations**
   - No recursion
   - Limited shared memory (32KB typical)
   - No dynamic dispatch in shaders
   - Template via string manipulation

3. **Performance Tuning**
   - Profile with WebGPU timing queries
   - Experiment with workgroup sizes (32, 64, 128, 256)
   - Balance compute vs. memory operations
   - Consider async compute overlap

## Success Metrics

### Performance Targets

| Operation Category | Target Speedup | Baseline |
|-------------------|----------------|----------|
| Elementwise       | 5-10x          | CPU SIMD |
| Reductions        | 10-50x         | CPU parallel |
| MatMul           | 50-200x        | CPU BLAS |
| Normalization    | 10-30x         | CPU parallel |
| Pooling          | 5-15x          | CPU optimized |

### Coverage Targets

- **Alpha.2 (Current)**: 30% GPU coverage
- **Beta.1**: 60% GPU coverage (Tier 1 complete)
- **Beta.2**: 80% GPU coverage (Tier 1-2 complete)
- **RC.1**: 95% GPU coverage (Tier 1-3 complete)
- **v1.0**: 99% GPU coverage (All tiers complete)

### Quality Gates

- ✅ All GPU kernels have CPU fallback
- ✅ Numerical accuracy within 1e-4 of CPU (float32)
- ✅ Performance within 2x of PyTorch/TensorFlow equivalent
- ✅ Zero memory leaks in extended tests
- ✅ Cross-platform validation (at least 2 GPU vendors)

## Resource Allocation

### Development Time Estimates

| Tier | Weeks | Kernels | Complexity |
|------|-------|---------|------------|
| 1    | 2     | ~15     | Medium-High |
| 2    | 2     | ~10     | Medium |
| 3    | 2     | ~12     | High |
| 4    | 2     | ~8      | High |
| **Total** | **8** | **~45** | - |

### Team Recommendations

- 1 Senior GPU engineer (kernel development)
- 1 Systems engineer (testing/integration)
- 1 Performance engineer (profiling/optimization)

## Risk Mitigation

### Technical Risks

1. **WebGPU Limitations**
   - Risk: Some operations may not be efficiently expressible in WGSL
   - Mitigation: Maintain CPU fallback, explore compute shader workarounds

2. **Platform Fragmentation**
   - Risk: Different GPU vendors have different performance characteristics
   - Mitigation: Adaptive dispatch, vendor-specific tuning

3. **Numerical Instability**
   - Risk: GPU floating-point behavior differs from CPU
   - Mitigation: Comprehensive numerical testing, stable algorithms

### Schedule Risks

1. **Dependency Blocking**
   - Risk: WGPU/WebGPU API changes
   - Mitigation: Pin dependencies, maintain compatibility layer

2. **Complexity Underestimation**
   - Risk: Some kernels may be more complex than anticipated
   - Mitigation: Buffer time, prioritize ruthlessly

## Next Steps

1. **Week 1**: Implement generic reduction kernel template
2. **Week 1-2**: Complete Tier 1 reduction operations
3. **Week 2**: Begin activation function implementations
4. **Week 3**: Start Tier 2 broadcasting optimizations
5. **Week 4**: Performance profiling and optimization pass
6. **Week 5-6**: Tier 3 advanced features
7. **Week 7-8**: Tier 4 specialized workloads
8. **Week 8**: Final testing, documentation, release prep

## References

- [WebGPU Compute Shader Best Practices](https://gpuweb.github.io/gpuweb/)
- [CUDA Reduction Optimization](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [Parallel Prefix Sum (Scan) with CUDA](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back)
- [TenfloweRS GPU Module](/src/gpu/)
- [TenfloweRS Memory Tracing](/src/gpu/memory_tracing.rs)

---

**Approval Required:** Tech Lead, Performance Team Lead
**Review Cycle:** Bi-weekly progress check-ins
**Document Status:** Living document - updated as priorities shift
