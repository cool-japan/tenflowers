# TenfloweRS Performance Benchmarking Guide

This document provides comprehensive information about benchmarking TenfloweRS performance and comparing it with other frameworks like PyTorch and TensorFlow.

## Available Benchmarking Tools

### 1. Comprehensive Benchmark Tool (`tools/comprehensive_benchmark.py`)

A Python script that provides comprehensive performance comparison between TenfloweRS and PyTorch.

**Features:**
- Side-by-side performance comparison
- Detailed reports in Markdown format
- JSON export for further analysis
- Optional visualization charts
- Support for both CPU and GPU benchmarking

**Usage:**
```bash
# Basic CPU benchmark comparison
python tools/comprehensive_benchmark.py --device cpu

# GPU benchmark (if CUDA available)
python tools/comprehensive_benchmark.py --device cuda

# Create visualizations
python tools/comprehensive_benchmark.py --device cpu --visualize

# Skip TenfloweRS benchmarks (PyTorch only)
python tools/comprehensive_benchmark.py --skip-tenflowers

# Custom output directory
python tools/comprehensive_benchmark.py --output-dir my_benchmark_results
```

**Output Files:**
- `comparison_report_{device}.md` - Human-readable performance report
- `comparison_results_{device}.json` - Raw benchmark data
- `performance_comparison.png` - Performance visualization (if --visualize used)

### 2. PyTorch Comparison Tool (`tools/pytorch_comparison.py`)

Standalone PyTorch benchmarking tool for generating reference performance data.

**Usage:**
```bash
# Run PyTorch benchmarks
python tools/pytorch_comparison.py --device cpu --all

# Specific operation types
python tools/pytorch_comparison.py --binary-ops --unary-ops --matmul

# Export results
python tools/pytorch_comparison.py --output-json pytorch_results.json --output-report pytorch_report.md
```

### 3. TenfloweRS Native Benchmarks (`examples/run_benchmarks`)

Rust-based benchmarking for TenfloweRS operations.

**Usage:**
```bash
# Run TenfloweRS benchmarks
cargo run --example run_benchmarks --release

# With GPU support (if available)
cargo run --example run_benchmarks --release --features gpu
```

## Benchmark Categories

### Binary Operations
- Addition, Multiplication, Subtraction, Division
- Various tensor shapes: 100x100, 500x500, 1000x1000
- Element-wise operations performance

### Unary Operations  
- ReLU, Sigmoid, Tanh activations
- Mathematical functions (exp, log, sqrt)
- Vector and matrix operations

### Matrix Multiplication
- GEMM operations: 64x64x64 to 1024x1024x1024
- FLOPS calculation and analysis
- Memory bandwidth vs compute bound analysis

### Neural Network Operations
- Convolution operations (Conv1D, Conv2D, Conv3D)
- Pooling operations (MaxPool, AvgPool, AdaptivePool)
- Normalization layers (BatchNorm, LayerNorm, GroupNorm)

## Performance Metrics

### Timing Metrics
- **Execution Time**: Wall-clock time for operation completion
- **Throughput**: Elements processed per second
- **FLOPS**: Floating-point operations per second (for compute-bound ops)

### Memory Metrics
- **Memory Usage**: Peak memory consumption during operation
- **Memory Bandwidth**: Effective memory bandwidth utilization
- **Cache Efficiency**: L1/L2/L3 cache hit ratios (where available)

### Comparison Metrics
- **Speedup**: TenfloweRS time / PyTorch time
- **Efficiency**: Performance relative to theoretical peak
- **Scaling**: Performance across different tensor sizes

## Understanding Results

### Interpreting Speedup
- **Speedup > 1.0**: TenfloweRS is faster
- **Speedup < 1.0**: PyTorch is faster  
- **Speedup ≈ 1.0**: Similar performance

### Performance Categories
- **Memory Bound**: Limited by memory bandwidth (element-wise ops)
- **Compute Bound**: Limited by computational throughput (matrix ops)
- **Latency Bound**: Limited by operation overhead (small tensors)

### Expected Performance Characteristics

#### TenfloweRS Strengths
- Zero-cost Rust abstractions
- Compile-time optimizations
- Memory safety without runtime overhead
- Fine-grained control over memory layout

#### TenfloweRS Growth Areas
- Kernel optimization maturity
- SIMD vectorization coverage
- GPU optimization depth
- Operation fusion opportunities

#### PyTorch Advantages
- Highly optimized kernels (Intel MKL, cuDNN)
- Mature ecosystem with years of optimization
- Extensive SIMD and GPU acceleration
- Advanced automatic differentiation optimizations

## Optimization Strategies

### For CPU Performance
1. **SIMD Vectorization**: Leverage ARM NEON, x86 AVX instructions
2. **Cache Optimization**: Improve memory access patterns
3. **Kernel Fusion**: Combine element-wise operations
4. **Threading**: Better utilize multi-core processors

### For GPU Performance  
1. **Memory Coalescing**: Optimize GPU memory access patterns
2. **Occupancy**: Maximize GPU core utilization
3. **Shared Memory**: Use GPU shared memory effectively
4. **Async Execution**: Overlap computation and memory transfers

### For Algorithmic Performance
1. **Operation Fusion**: Reduce memory round-trips
2. **Graph Optimization**: Eliminate redundant computations
3. **Precision**: Use appropriate data types (FP16 vs FP32)
4. **Batching**: Amortize overhead across operations

## Continuous Integration

### Automated Benchmarking
Add to CI/CD pipeline:
```yaml
# Example GitHub Actions workflow
- name: Run Performance Benchmarks
  run: |
    python tools/comprehensive_benchmark.py --device cpu
    # Compare against baseline performance
    python tools/performance_regression_check.py
```

### Performance Regression Detection
- Track performance over time
- Alert on significant regressions
- Maintain performance baseline database

## Contributing Benchmarks

### Adding New Operations
1. Implement benchmark in `tools/comprehensive_benchmark.py`
2. Add corresponding PyTorch reference implementation
3. Include performance expectations and analysis
4. Add tests for benchmark correctness

### Adding New Platforms
1. Test on target platform (ARM, x86, GPU variants)
2. Document platform-specific optimizations
3. Include platform in CI matrix
4. Add platform-specific performance expectations

## Troubleshooting

### Common Issues
- **CUDA not available**: Use `--device cpu` flag
- **Memory errors**: Reduce tensor sizes for memory-constrained systems
- **Compilation errors**: Ensure Rust toolchain and PyTorch are properly installed

### Performance Debugging
- Use profiling tools: `perf`, `vtune`, `nsight`
- Enable detailed timing in benchmarks
- Check for thermal throttling on mobile devices
- Verify proper CPU/GPU affinity

## Performance Targets

### Short-term Goals
- Match PyTorch CPU performance within 2x for most operations
- Achieve 90% of theoretical peak memory bandwidth
- Optimize critical path operations (MatMul, Conv2D)

### Long-term Goals  
- Exceed PyTorch performance for Rust-optimized use cases
- Achieve best-in-class memory efficiency
- Provide superior compile-time optimization
- Enable novel optimization techniques unique to Rust

## References

- [PyTorch Benchmarking Best Practices](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Intel oneAPI Performance Libraries](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
- [NVIDIA cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html)