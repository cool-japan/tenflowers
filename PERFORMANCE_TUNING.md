# TenfloweRS Performance Tuning Guide

A comprehensive guide to optimizing TenfloweRS applications for maximum performance across different hardware configurations and use cases.

## Table of Contents

- [Quick Start](#quick-start)
- [Performance Fundamentals](#performance-fundamentals)
- [CPU Optimization](#cpu-optimization)
- [GPU Optimization](#gpu-optimization)
- [Memory Optimization](#memory-optimization)
- [Model Architecture Optimization](#model-architecture-optimization)
- [Training Optimization](#training-optimization)
- [Inference Optimization](#inference-optimization)
- [Profiling and Debugging](#profiling-and-debugging)
- [Common Performance Patterns](#common-performance-patterns)
- [Hardware-Specific Optimizations](#hardware-specific-optimizations)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Essential Performance Checklist

Before diving into specific optimizations, ensure these fundamentals are in place:

```rust
// ✅ DO: Use appropriate data types
let tensor = Tensor::zeros(&[1024, 1024], DType::F32, Device::Cpu)?;

// ❌ DON'T: Use unnecessary precision
let tensor = Tensor::zeros(&[1024, 1024], DType::F64, Device::Cpu)?;

// ✅ DO: Batch operations
let batch_size = 32;
let input = Tensor::randn(&[batch_size, 784], DType::F32, device)?;

// ❌ DON'T: Process single samples
for i in 0..1000 {
    let single_input = Tensor::randn(&[1, 784], DType::F32, device)?;
}

// ✅ DO: Reuse memory where possible
let mut buffer = Tensor::zeros(&[1024, 1024], DType::F32, device)?;
for epoch in 0..100 {
    buffer.copy_from(&new_data)?;
    let result = model.forward(&buffer)?;
}
```

### Performance Compilation Flags

```bash
# Release build with optimizations
cargo build --release

# CPU-specific optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Link-time optimization
RUSTFLAGS="-C lto=fat" cargo build --release

# Combined optimizations
RUSTFLAGS="-C target-cpu=native -C lto=fat" cargo build --release
```

---

## Performance Fundamentals

### Understanding Performance Bottlenecks

TenfloweRS applications typically encounter these bottlenecks:

1. **Memory Bandwidth**: Element-wise operations (add, mul, ReLU)
2. **Compute Bound**: Matrix multiplications, convolutions
3. **Latency Bound**: Small tensor operations, device transfers
4. **Memory Capacity**: Large model parameters, activations

### Performance Metrics

```rust
use tenflowers_core::{PerformanceMonitor, Tensor};

// Enable performance monitoring
let monitor = PerformanceMonitor::new();
monitor.start();

// Your operation
let result = tensor_a.matmul(&tensor_b)?;

// Get performance metrics
let stats = monitor.stop();
println!("Operation time: {:.2}ms", stats.duration_ms);
println!("Memory usage: {:.2}MB", stats.memory_usage_mb);
println!("FLOPS: {:.2}GF/s", stats.flops_per_second / 1e9);
```

### Device Selection Strategy

```rust
use tenflowers_core::{Device, Tensor};

fn select_optimal_device(tensor_size: usize) -> Device {
    match tensor_size {
        // Small tensors: CPU is often faster due to GPU overhead
        size if size < 1000 => Device::Cpu,
        // Medium tensors: GPU if available
        size if size < 1_000_000 => {
            if Device::Gpu(0).is_available() {
                Device::Gpu(0)
            } else {
                Device::Cpu
            }
        }
        // Large tensors: Prefer GPU
        _ => Device::Gpu(0),
    }
}
```

---

## CPU Optimization

### SIMD Vectorization

Enable SIMD instructions for element-wise operations:

```rust
// Enable target-specific CPU features
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Vectorized operations are automatically used when available
let a = Tensor::from_vec(vec![1.0f32; 1024], &[1024]);
let b = Tensor::from_vec(vec![2.0f32; 1024], &[1024]);
let result = &a + &b;  // Uses AVX/SSE when available
```

### Compilation Flags for CPU

```bash
# Enable AVX2 (most modern CPUs)
RUSTFLAGS="-C target-feature=+avx2" cargo build --release

# Enable all native CPU features
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Enable specific features
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release
```

### Threading and Parallelism

```rust
use tenflowers_core::{ThreadingConfig, Tensor};

// Configure threading for optimal performance
let config = ThreadingConfig::new()
    .with_num_threads(num_cpus::get())
    .with_thread_affinity(true);

// Use parallel operations
let tensors = vec![tensor1, tensor2, tensor3, tensor4];
let results: Vec<_> = tensors.par_iter()
    .map(|t| t.sum(None))
    .collect();
```

### Cache Optimization

```rust
// ✅ DO: Use cache-friendly access patterns
fn cache_friendly_matmul(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    // Block-wise multiplication for better cache locality
    let block_size = 64;
    // Implementation would use blocking for cache efficiency
    a.matmul_blocked(b, block_size)
}

// ❌ DON'T: Use cache-unfriendly patterns
fn cache_unfriendly_access(tensor: &Tensor<f32>) {
    // Strided access patterns hurt cache performance
    for i in (0..tensor.len()).step_by(1000) {
        let value = tensor.get_unchecked(i);
    }
}
```

### BLAS Integration

```rust
// Enable BLAS for linear algebra operations
use tenflowers_core::{BlasConfig, Tensor};

let config = BlasConfig::new()
    .with_provider(BlasProvider::OpenBlas)
    .with_num_threads(8);

// Matrix operations automatically use BLAS
let result = a.matmul(&b)?;  // Uses optimized BLAS GEMM
```

---

## GPU Optimization

### GPU Memory Management

```rust
use tenflowers_core::{GpuMemoryPool, Device, Tensor};

// Use memory pools for frequent allocations
let pool = GpuMemoryPool::new(Device::Gpu(0));

// Allocate from pool
let tensor = pool.allocate_tensor(&[1024, 1024], DType::F32)?;

// Memory is automatically returned to pool when dropped
```

### Kernel Optimization

```rust
// ✅ DO: Use kernel fusion for related operations
let fused_result = tensor_a.add(&tensor_b)?.relu()?;  // Fused add+relu

// ❌ DON'T: Use separate kernels unnecessarily
let intermediate = tensor_a.add(&tensor_b)?;  // Kernel 1
let result = intermediate.relu()?;            // Kernel 2
```

### Batch Size Optimization

```rust
// Find optimal batch size for your GPU
fn find_optimal_batch_size(model: &Model, input_size: &[usize]) -> usize {
    let mut batch_size = 1;
    let mut best_throughput = 0.0;
    let mut optimal_batch = 1;
    
    while batch_size <= 512 {
        let mut test_input = input_size.to_vec();
        test_input[0] = batch_size;
        
        let input = Tensor::randn(&test_input, DType::F32, Device::Gpu(0))?;
        
        let start = std::time::Instant::now();
        let _ = model.forward(&input)?;
        let duration = start.elapsed();
        
        let throughput = batch_size as f64 / duration.as_secs_f64();
        
        if throughput > best_throughput {
            best_throughput = throughput;
            optimal_batch = batch_size;
        }
        
        batch_size *= 2;
    }
    
    optimal_batch
}
```

### Asynchronous Operations

```rust
use tenflowers_core::{AsyncStream, Device};

// Use async streams for overlapping computation
let stream = AsyncStream::new(Device::Gpu(0))?;

// Queue operations asynchronously
let future1 = stream.matmul_async(&a, &b);
let future2 = stream.add_async(&c, &d);

// Execute other work while GPU computes
do_cpu_work();

// Wait for results
let result1 = future1.await?;
let result2 = future2.await?;
```

### Memory Coalescing

```rust
// ✅ DO: Use coalesced memory access patterns
let tensor = Tensor::zeros(&[1024, 1024], DType::F32, Device::Gpu(0))?;
let result = tensor.transpose()?;  // Optimized transpose with coalescing

// ❌ DON'T: Use non-coalesced access
for i in 0..1024 {
    for j in (0..1024).step_by(32) {  // Strided access
        let value = tensor.get(&[i, j])?;
    }
}
```

---

## Memory Optimization

### Memory Pool Management

```rust
use tenflowers_core::{MemoryPool, PoolConfig};

// Configure memory pool
let config = PoolConfig::new()
    .with_initial_size(1024 * 1024 * 1024)  // 1GB
    .with_growth_factor(1.5)
    .with_max_size(4 * 1024 * 1024 * 1024); // 4GB

let pool = MemoryPool::new(config);

// Use pooled allocations
let tensor = pool.allocate_tensor(&[1024, 1024], DType::F32)?;
```

### In-Place Operations

```rust
// ✅ DO: Use in-place operations when possible
let mut tensor = Tensor::randn(&[1024, 1024], DType::F32, device)?;
tensor.add_(&other)?;      // In-place addition
tensor.relu_()?;           // In-place activation

// ❌ DON'T: Create unnecessary copies
let tensor = Tensor::randn(&[1024, 1024], DType::F32, device)?;
let result1 = tensor.add(&other)?;  // Creates new tensor
let result2 = result1.relu()?;      // Creates another tensor
```

### Memory Reuse Patterns

```rust
// Buffer reuse for training loops
struct TrainingBuffers {
    gradients: Vec<Tensor<f32>>,
    activations: Vec<Tensor<f32>>,
    workspace: Tensor<f32>,
}

impl TrainingBuffers {
    fn new(model: &Model) -> Self {
        let gradients = model.parameters().iter()
            .map(|p| Tensor::zeros_like(p))
            .collect();
        
        let activations = vec![
            Tensor::zeros(&[batch_size, hidden_size], DType::F32, device)?,
            // ... other activation buffers
        ];
        
        let workspace = Tensor::zeros(&[max_workspace_size], DType::F32, device)?;
        
        Self { gradients, activations, workspace }
    }
    
    fn reset(&mut self) {
        for grad in &mut self.gradients {
            grad.zero_()?;
        }
    }
}
```

### Gradient Checkpointing

```rust
use tenflowers_autograd::{checkpoint, GradientTape};

// Use checkpointing for memory-efficient training
let tape = GradientTape::new();

// Checkpoint expensive layers
let layer1_output = layer1.forward(&input)?;
let checkpointed_output = checkpoint(layer1_output, || {
    layer2.forward(&layer1_output)
})?;

// Memory is freed and recomputed during backward pass
```

---

## Model Architecture Optimization

### Layer Fusion

```rust
// ✅ DO: Use fused operations
struct FusedConvBnRelu {
    conv: Conv2D,
    bn: BatchNorm,
}

impl FusedConvBnRelu {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // Fused computation reduces memory bandwidth
        let conv_out = self.conv.forward(input)?;
        let bn_out = self.bn.forward(&conv_out)?;
        Ok(bn_out.relu()?)
    }
}

// ❌ DON'T: Use separate layers for fusable operations
let conv_out = conv.forward(&input)?;
let bn_out = bn.forward(&conv_out)?;
let relu_out = relu.forward(&bn_out)?;
```

### Activation Function Choice

```rust
// Performance ranking for activation functions (fastest to slowest)
// 1. ReLU - simple thresholding
let relu = ReLU::new();

// 2. Leaky ReLU - similar to ReLU with small computation
let leaky_relu = LeakyReLU::new(0.01);

// 3. GELU - more expensive but often worth it
let gelu = GELU::new();

// 4. Swish/SiLU - requires sigmoid computation
let swish = Swish::new();

// 5. Mish - most expensive (tanh + softplus)
let mish = Mish::new();
```

### Efficient Attention

```rust
// Use efficient attention implementations
struct EfficientAttention {
    attention: MultiHeadAttention,
    use_flash_attention: bool,
}

impl EfficientAttention {
    fn forward(&self, query: &Tensor<f32>, key: &Tensor<f32>, value: &Tensor<f32>) -> Result<Tensor<f32>> {
        if self.use_flash_attention && query.shape()[1] > 512 {
            // Flash attention for long sequences
            self.attention.forward_flash(query, key, value)
        } else {
            // Standard attention for short sequences
            self.attention.forward(query, key, value, None)
        }
    }
}
```

---

## Training Optimization

### Gradient Accumulation

```rust
// Simulate large batch sizes with gradient accumulation
let accumulation_steps = 4;
let effective_batch_size = batch_size * accumulation_steps;

for step in 0..accumulation_steps {
    let batch = get_batch(step)?;
    let loss = model.forward(&batch.input)?.cross_entropy(&batch.target)?;
    let scaled_loss = loss / accumulation_steps as f32;
    
    // Accumulate gradients
    scaled_loss.backward()?;
    
    // Don't update parameters yet
}

// Update parameters after accumulation
optimizer.step(&mut model)?;
model.zero_grad()?;
```

### Mixed Precision Training

```rust
use tenflowers_core::{MixedPrecisionConfig, AutocastContext, GradScaler};

// Configure mixed precision
let config = MixedPrecisionConfig::new()
    .with_enabled(true)
    .with_target_dtype(DType::F16);

let autocast = AutocastContext::new(config);
let scaler = GradScaler::new();

// Training loop with mixed precision
for batch in data_loader {
    // Forward pass in FP16
    autocast.enable();
    let predictions = model.forward(&batch.input)?;
    let loss = loss_fn(&predictions, &batch.target)?;
    autocast.disable();
    
    // Scale loss for FP16 training
    let scaled_loss = scaler.scale(&loss)?;
    
    // Backward pass
    scaled_loss.backward()?;
    
    // Unscale gradients before clipping
    scaler.unscale(&mut model)?;
    
    // Gradient clipping
    clip_gradients_by_norm(&mut model, 1.0)?;
    
    // Update parameters
    scaler.step(&mut optimizer, &mut model)?;
    scaler.update()?;
}
```

### Efficient Data Loading

```rust
use tenflowers_dataset::{DataLoader, Dataset};

// Optimize data loading
let data_loader = DataLoader::new(dataset)
    .batch_size(64)
    .num_workers(4)          // Parallel data loading
    .prefetch_factor(2)      // Prefetch batches
    .pin_memory(true)        // Pin memory for faster GPU transfer
    .drop_last(true);        // Drop incomplete batches

// Use iterator efficiently
for batch in data_loader {
    // Process batch
}
```

### Learning Rate Scheduling

```rust
use tenflowers_neural::{OneCycleLR, WarmupCosineDecayLR};

// Use effective learning rate schedules
let scheduler = OneCycleLR::new(
    max_lr: 0.01,
    total_steps: total_training_steps,
    pct_start: 0.1,  // 10% warmup
    anneal_strategy: "cos",
    cycle_momentum: true,
);

// Or use warmup with cosine decay
let warmup_scheduler = WarmupCosineDecayLR::new(
    initial_lr: 0.0,
    max_lr: 0.01,
    warmup_steps: 1000,
    total_steps: total_training_steps,
);
```

---

## Inference Optimization

### Model Quantization

```rust
use tenflowers_core::{QuantizationConfig, quantize_model};

// Quantize model for faster inference
let config = QuantizationConfig::new()
    .with_mode(QuantizationMode::Static)
    .with_dtype(DType::INT8)
    .with_calibration_dataset(&calibration_data);

let quantized_model = quantize_model(model, config)?;

// Use quantized model for inference
let output = quantized_model.forward(&input)?;
```

### Graph Optimization

```rust
use tenflowers_core::{GraphOptimizer, OptimizationLevel};

// Optimize computation graph
let optimizer = GraphOptimizer::new()
    .with_level(OptimizationLevel::Aggressive)
    .with_constant_folding(true)
    .with_dead_code_elimination(true)
    .with_kernel_fusion(true);

let optimized_model = optimizer.optimize(&model)?;
```

### Batch Processing

```rust
// Process multiple inputs in batches
fn batch_inference(model: &Model, inputs: Vec<Tensor<f32>>) -> Result<Vec<Tensor<f32>>> {
    let batch_size = 32;
    let mut results = Vec::with_capacity(inputs.len());
    
    for chunk in inputs.chunks(batch_size) {
        let batch = Tensor::stack(chunk, 0)?;
        let batch_output = model.forward(&batch)?;
        
        // Split batch output back to individual results
        let individual_outputs = batch_output.chunk(chunk.len(), 0)?;
        results.extend(individual_outputs);
    }
    
    Ok(results)
}
```

---

## Profiling and Debugging

### Performance Profiling

```rust
use tenflowers_core::{Profiler, ProfilerConfig};

// Enable detailed profiling
let profiler = Profiler::new(ProfilerConfig {
    track_memory: true,
    track_timing: true,
    track_gpu_utilization: true,
    output_format: ProfilerFormat::Chrome,
});

profiler.start();

// Your code here
let result = model.forward(&input)?;

// Generate profiling report
profiler.stop();
profiler.save_report("profile_report.json")?;
```

### Memory Debugging

```rust
use tenflowers_core::{MemoryDebugger, MemoryTracker};

// Track memory allocations
let tracker = MemoryTracker::new();
tracker.start_tracking();

// Your code
let tensor = Tensor::randn(&[1024, 1024], DType::F32, device)?;

// Check for memory leaks
let report = tracker.stop_tracking();
println!("Peak memory usage: {}MB", report.peak_usage_mb);
println!("Current usage: {}MB", report.current_usage_mb);
if report.potential_leaks > 0 {
    println!("Potential leaks detected: {}", report.potential_leaks);
}
```

### Operation Timing

```rust
use std::time::Instant;

// Time individual operations
fn time_operation<F, R>(op: F, name: &str) -> R
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = op();
    let duration = start.elapsed();
    println!("{}: {:.2}ms", name, duration.as_secs_f64() * 1000.0);
    result
}

// Usage
let result = time_operation(|| {
    tensor_a.matmul(&tensor_b)
}, "Matrix multiplication");
```

---

## Common Performance Patterns

### Efficient Model Initialization

```rust
// ✅ DO: Initialize parameters efficiently
struct EfficientModel {
    layers: Vec<Box<dyn Layer<f32>>>,
    parameter_buffer: Tensor<f32>,
}

impl EfficientModel {
    fn new() -> Self {
        // Pre-allocate all parameters in single buffer
        let total_params = calculate_total_parameters();
        let parameter_buffer = Tensor::zeros(&[total_params], DType::F32, device)?;
        
        // Initialize layers with views into buffer
        let layers = create_layers_with_buffer_views(&parameter_buffer);
        
        Self { layers, parameter_buffer }
    }
}

// ❌ DON'T: Allocate parameters separately
struct IneffientModel {
    layer1: Dense,  // Separate allocation
    layer2: Dense,  // Separate allocation
    layer3: Dense,  // Separate allocation
}
```

### Workspace Reuse

```rust
// Reuse workspace tensors across operations
struct WorkspaceManager {
    workspaces: HashMap<String, Tensor<f32>>,
}

impl WorkspaceManager {
    fn get_workspace(&mut self, name: &str, size: &[usize]) -> Result<&mut Tensor<f32>> {
        if !self.workspaces.contains_key(name) {
            let workspace = Tensor::zeros(size, DType::F32, device)?;
            self.workspaces.insert(name.to_string(), workspace);
        }
        Ok(self.workspaces.get_mut(name).unwrap())
    }
}
```

### Efficient Error Handling

```rust
// ✅ DO: Use Result types efficiently
fn efficient_computation(input: &Tensor<f32>) -> Result<Tensor<f32>> {
    let intermediate = input.matmul(&weights)?;
    let output = intermediate.add(&bias)?;
    Ok(output.relu()?)
}

// ❌ DON'T: Use unwrap() in performance-critical code
fn inefficient_computation(input: &Tensor<f32>) -> Tensor<f32> {
    let intermediate = input.matmul(&weights).unwrap();  // Potential panic
    let output = intermediate.add(&bias).unwrap();
    output.relu().unwrap()
}
```

---

## Hardware-Specific Optimizations

### Intel CPU Optimizations

```bash
# Enable Intel MKL
export BLAS_BACKEND=mkl

# Intel-specific compiler flags
RUSTFLAGS="-C target-feature=+avx2,+avx512f" cargo build --release

# Intel oneAPI optimizations
source /opt/intel/oneapi/setvars.sh
```

### AMD CPU Optimizations

```bash
# Enable AMD BLIS
export BLAS_BACKEND=blis

# AMD-specific compiler flags
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release
```

### Apple Silicon (M1/M2) Optimizations

```bash
# Enable Apple Accelerate framework
export BLAS_BACKEND=accelerate

# ARM-specific optimizations
RUSTFLAGS="-C target-feature=+neon" cargo build --release
```

### NVIDIA GPU Optimizations

```rust
// Configure CUDA optimizations
let cuda_config = CudaConfig::new()
    .with_allow_tf32(true)
    .with_benchmark_cudnn(true)
    .with_deterministic(false);  // Allows faster algorithms

// Use Tensor Cores when available
let mixed_precision = MixedPrecisionConfig::new()
    .with_enabled(true)
    .with_target_dtype(DType::F16)
    .with_use_tensor_cores(true);
```

---

## Troubleshooting

### Common Performance Issues

#### Issue: Slow GPU Operations

```rust
// Check GPU utilization
let gpu_stats = Device::Gpu(0).get_utilization()?;
if gpu_stats.utilization < 80.0 {
    println!("GPU underutilized: {}%", gpu_stats.utilization);
    // Increase batch size or use more complex operations
}

// Check memory bandwidth
if gpu_stats.memory_bandwidth_utilization < 70.0 {
    println!("Memory bandwidth underutilized");
    // Use more memory-intensive operations or increase parallelism
}
```

#### Issue: High Memory Usage

```rust
// Monitor memory usage
let memory_info = Device::Gpu(0).memory_info()?;
let usage_percent = (memory_info.used as f64 / memory_info.total as f64) * 100.0;

if usage_percent > 90.0 {
    println!("High memory usage: {:.1}%", usage_percent);
    // Reduce batch size or use gradient checkpointing
}
```

#### Issue: CPU Bottlenecks

```rust
// Check CPU utilization
let cpu_usage = psutil::cpu_percent()?;
if cpu_usage > 95.0 {
    println!("CPU bottleneck detected");
    // Optimize data loading or use more efficient algorithms
}
```

### Performance Debugging Tools

```rust
// Enable detailed logging
env_logger::init_from_env(env_logger::Env::new().default_filter_or("debug"));

// Use performance counters
let counters = PerformanceCounters::new();
counters.enable(&["instructions", "cache_misses", "branch_mispredicts"]);

// Your code here
let result = model.forward(&input)?;

let stats = counters.read();
println!("Instructions: {}", stats.instructions);
println!("Cache misses: {}", stats.cache_misses);
```

### Optimization Verification

```rust
// Verify optimizations are working
fn verify_optimization(original_model: &Model, optimized_model: &Model, input: &Tensor<f32>) -> Result<()> {
    let start = Instant::now();
    let original_output = original_model.forward(input)?;
    let original_time = start.elapsed();
    
    let start = Instant::now();
    let optimized_output = optimized_model.forward(input)?;
    let optimized_time = start.elapsed();
    
    // Check correctness
    let diff = (&original_output - &optimized_output)?.abs()?.max(None)?;
    assert!(diff.get(&[]).unwrap() < &1e-5, "Optimization changed results");
    
    // Check performance improvement
    let speedup = original_time.as_secs_f64() / optimized_time.as_secs_f64();
    println!("Speedup: {:.2}x", speedup);
    
    Ok(())
}
```

---

## Best Practices Summary

### Do's ✅

1. **Use appropriate data types** - F32 for most operations, F16 for memory-constrained scenarios
2. **Batch operations** - Process multiple samples together
3. **Profile regularly** - Identify bottlenecks with proper tooling
4. **Reuse memory** - Use in-place operations and buffer pools
5. **Optimize for your hardware** - Use target-specific compiler flags
6. **Use established patterns** - Follow proven optimization techniques

### Don'ts ❌

1. **Don't use unnecessary precision** - F64 is rarely needed
2. **Don't ignore memory layout** - Cache-friendly access patterns matter
3. **Don't over-optimize prematurely** - Profile first, then optimize
4. **Don't forget to measure** - Always verify optimization effectiveness
5. **Don't ignore compiler warnings** - They often indicate performance issues
6. **Don't use debug builds for benchmarking** - Always use release builds

---

## Further Reading

- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Intel oneAPI Performance Libraries](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
- [ARM Neon Intrinsics](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon)

---

This performance tuning guide provides a comprehensive foundation for optimizing TenfloweRS applications. Remember that performance optimization is an iterative process - measure, optimize, and verify improvements systematically.