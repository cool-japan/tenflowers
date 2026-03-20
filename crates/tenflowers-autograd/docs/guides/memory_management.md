# Memory Management Guide

**Target Audience**: All users (essential for production deployment)
**Prerequisites**: Basic understanding of gradient computation
**Estimated Reading Time**: 60 minutes

---

## Table of Contents

1. [Introduction](#introduction)
2. [Memory Architecture](#memory-architecture)
3. [Gradient Tape Memory](#gradient-tape-memory)
4. [Activation Checkpointing](#activation-checkpointing)
5. [Memory Profiling](#memory-profiling)
6. [Optimization Strategies](#optimization-strategies)
7. [Common Memory Issues](#common-memory-issues)
8. [Best Practices](#best-practices)
9. [Case Studies](#case-studies)

---

## Introduction

Memory management is critical for training deep learning models. Inefficient memory usage can:
- **Limit model size**: Cannot fit large models in available memory
- **Reduce batch size**: Smaller batches lead to slower training and worse generalization
- **Cause OOM errors**: Out-of-memory crashes during training
- **Waste resources**: Underutilizing available memory capacity

This guide provides comprehensive strategies for optimizing memory usage in TenfloweRS Autograd.

### Memory Budget Breakdown

Typical memory usage during training:

```
┌─────────────────────────────────────────────┐
│ Model Parameters (θ)         │ 20-30%     │
├─────────────────────────────────────────────┤
│ Optimizer State (m, v)       │ 40-50%     │  For Adam: 2× parameters
├─────────────────────────────────────────────┤
│ Gradients (∇θ)               │ 20-30%     │  Same size as parameters
├─────────────────────────────────────────────┤
│ Activations (intermediate)   │ 50-70%     │  Depends on depth & batch size
├─────────────────────────────────────────────┤
│ Gradient Tape (metadata)     │ 5-10%      │  Computational graph
└─────────────────────────────────────────────┘

Total: ~135-190% of parameter size (without optimizations)
```

**With optimizations** (checkpointing, mixed precision):
```
Total: ~40-60% of parameter size (60-70% savings!)
```

---

## Memory Architecture

### Memory Hierarchy

```
CPU Memory (Host)
├── RAM (16-256 GB)
│   ├── Fast access (~100 GB/s)
│   ├── Large capacity
│   └── Can swap to disk (slow)
└── L3 Cache (8-64 MB)
    └── Very fast (~400 GB/s)

GPU Memory (Device)
├── VRAM (8-80 GB)
│   ├── Very fast (~1000 GB/s)
│   ├── Limited capacity
│   └── Cannot swap (OOM = crash)
└── L2 Cache (4-40 MB)
    └── Extremely fast (~2000 GB/s)
```

**Key insight**: GPU memory is fast but limited. We must optimize for GPU VRAM.

### Memory Allocation Patterns

**1. Static Allocation** (parameters, buffers):
```rust
// Allocated once, persists entire training
let parameters = model.parameters();  // ~1-10 GB
let optimizer_state = optimizer.state();  // ~2-20 GB
```

**2. Dynamic Allocation** (activations, gradients):
```rust
// Allocated per batch, freed after backward
for batch in dataloader {
    let tape = GradientTape::new();  // Allocates tape
    let activations = model.forward(&batch)?;  // Allocates activations
    let grads = tape.gradient(&[loss], &params)?;  // Allocates gradients
    // tape, activations, grads freed here
}
```

**3. Temporary Allocation** (intermediate computations):
```rust
// Very short-lived allocations
let temp = x.mul(&y)?;  // Allocated
let result = temp.add(&z)?;  // temp can be freed
```

---

## Gradient Tape Memory

### What the Tape Stores

The `GradientTape` stores:

1. **TapeNodes** (graph structure):
   ```rust
   struct TapeNode {
       operation: Operation,      // ~8 bytes (enum)
       input_ids: Vec<TensorId>,  // ~24 bytes
       output_id: TensorId,       // ~8 bytes
       // Total: ~40 bytes per node
   }
   ```

2. **Intermediate Tensors** (activations):
   ```rust
   // For each layer:
   // - Layer input: [batch, input_dim]
   // - Layer output: [batch, output_dim]
   // Memory: batch × (input_dim + output_dim) × 4 bytes (FP32)
   ```

3. **Operation Metadata**:
   ```rust
   // Shapes, strides, hyperparameters needed for backward
   // Typically small: ~100 bytes per operation
   ```

### Memory Growth

**Without checkpointing**:
```
Memory = O(depth × batch_size × feature_dim)
```

For a 50-layer network with batch_size=32, feature_dim=1024:
```
Memory ≈ 50 × 32 × 1024 × 4 bytes = 6.5 MB per forward pass
```

Seems small, but scales poorly!

**With 200 layers**:
```
Memory ≈ 200 × 32 × 1024 × 4 bytes = 26 MB per forward pass
```

**With vision transformers (large feature maps)**:
```
Memory ≈ depth × batch × height × width × channels × 4 bytes
       ≈ 12 × 16 × 224 × 224 × 768 × 4 bytes
       ≈ 2.3 GB per forward pass
```

This is why checkpointing is essential!

---

## Activation Checkpointing

### The Memory-Compute Tradeoff

**Without checkpointing**:
- ✅ Fast backward pass (all activations cached)
- ❌ High memory usage O(depth)
- ❌ Limits batch size and model depth

**With checkpointing**:
- ✅ Low memory usage O(√depth) with optimal strategy
- ✅ Enables larger models and batch sizes
- ❌ Slower backward pass (recomputation needed)

### Checkpointing Strategies

#### 1. No Checkpointing (Baseline)

```rust
let tape = GradientTape::new();  // Default: no checkpointing

// All activations stored
for i in 0..num_layers {
    hidden = layer_forward(&hidden)?;
    // activation[i] stored in tape
}

let grads = tape.gradient(&[loss], &params)?;
// Backward pass: uses all stored activations
```

**Memory**: O(depth)
**Time**: 1× (baseline)

#### 2. Full Checkpointing

```rust
let config = CheckpointConfig {
    strategy: CheckpointStrategy::Full,
    ..Default::default()
};
let tape = GradientTape::with_checkpoint_config(config);

// No activations stored
for i in 0..num_layers {
    hidden = layer_forward(&hidden)?;
    // activation[i] NOT stored
}

let grads = tape.gradient(&[loss], &params)?;
// Backward pass: recomputes all activations
```

**Memory**: O(1) - minimal
**Time**: 2× (one extra forward pass)

**Use when**: Extremely memory constrained

#### 3. Selective Checkpointing

```rust
let config = CheckpointConfig {
    strategy: CheckpointStrategy::Selective,
    // Only checkpoint operations with high computation cost
    min_compute_cost: 1000,  // FLOPs threshold
    ..Default::default()
};

let tape = GradientTape::with_checkpoint_config(config);
```

**Memory**: O(depth × p) where p ≈ 0.3-0.5 (30-50% of operations)
**Time**: ~1.2-1.5× (partial recomputation)

**Heuristics** (automatically applied):
- ✅ **Checkpoint**: Matrix multiplications, convolutions (expensive)
- ❌ **Store**: Activations (ReLU, etc.), normalization (cheap to recompute)

Example:
```rust
// Expensive operation (CHECKPOINT)
let y = x.matmul(&w)?;  // O(n³) FLOPs
// Cheap to recompute

// Cheap operation (STORE)
let y = y.relu()?;  // O(n) FLOPs
// Stored for backward pass
```

#### 4. Block Checkpointing

```rust
let config = CheckpointConfig {
    strategy: CheckpointStrategy::Block,
    checkpoint_every_n: 5,  // Checkpoint every 5 layers
    ..Default::default()
};

let tape = GradientTape::with_checkpoint_config(config);
```

**Memory**: O(depth / n)
**Time**: ~1.5× (on average, recompute n/2 layers per checkpoint)

**Optimal n** (Griewank's algorithm):
```
n* ≈ √(2 × depth)

For 100 layers: n* ≈ 14
For 1000 layers: n* ≈ 45
```

**Tradeoff curve**:
```
Memory reduction vs. Time overhead:
n=2:  90% memory reduction, 1.5× time
n=5:  80% memory reduction, 1.3× time
n=10: 70% memory reduction, 1.2× time
n=20: 50% memory reduction, 1.1× time
```

#### 5. Automatic Checkpointing

```rust
let config = CheckpointConfig {
    strategy: CheckpointStrategy::Auto,
    memory_budget_mb: 4096,  // 4 GB limit
    ..Default::default()
};

let tape = GradientTape::with_checkpoint_config(config);

// Dynamically adjusts checkpointing based on memory usage
```

**Algorithm**:
1. Profile memory usage during first few batches
2. Estimate checkpoint intervals to fit memory budget
3. Adjust dynamically if memory pressure detected

**Advantages**:
- ✅ No manual tuning required
- ✅ Adapts to different model architectures
- ✅ Maximizes batch size within memory budget

---

## Memory Profiling

### Built-in Memory Profiler

```rust
use tenflowers_autograd::GradientMemoryProfiler;

let mut profiler = GradientMemoryProfiler::new();
profiler.start_profiling();

// Record checkpoints at key points
profiler.record_checkpoint("before_forward")?;

let output = model.forward(&input)?;
profiler.record_checkpoint("after_forward")?;

let grads = tape.gradient(&[output], &params)?;
profiler.record_checkpoint("after_backward")?;

// Analyze memory usage
let forward_memory = profiler.get_memory_delta(
    "before_forward",
    "after_forward"
)?;

println!("Forward pass memory: {:.2} MB", forward_memory / 1_048_576.0);

// Check for memory leaks
let leak_report = profiler.detect_leaks()?;
if leak_report.num_suspicious > 0 {
    println!("⚠️  Warning: {} potential memory leaks detected",
        leak_report.num_suspicious);
    for leak in &leak_report.suspicious_allocations {
        println!("  - {} bytes at {:?}", leak.size, leak.location);
    }
}
```

### Memory Reports

```rust
let report = profiler.generate_report()?;

println!("Memory Report:");
println!("  Peak usage: {:.2} GB", report.peak_memory_gb);
println!("  Current usage: {:.2} GB", report.current_memory_gb);
println!("  Allocations: {}", report.num_allocations);
println!("  Deallocations: {}", report.num_deallocations);
println!("  Net allocations: {}", report.num_allocations - report.num_deallocations);

// Per-operation breakdown
for (op_name, stats) in &report.operation_stats {
    println!("  {}: {:.2} MB", op_name, stats.total_memory_mb);
}
```

### Memory Debugging

**Identify memory bottlenecks**:

```rust
// Profile each layer individually
for (i, layer) in model.layers.iter().enumerate() {
    profiler.record_checkpoint(&format!("layer_{}_before", i))?;

    let output = layer.forward(&input)?;

    profiler.record_checkpoint(&format!("layer_{}_after", i))?;

    let layer_memory = profiler.get_memory_delta(
        &format!("layer_{}_before", i),
        &format!("layer_{}_after", i),
    )?;

    println!("Layer {}: {:.2} MB", i, layer_memory / 1_048_576.0);
}

// Identify which layers use most memory
let bottleneck_layers = profiler.find_memory_bottlenecks(threshold_mb: 100.0)?;
for (layer_name, memory_mb) in bottleneck_layers {
    println!("🔴 Bottleneck: {} uses {:.2} MB", layer_name, memory_mb);
}
```

---

## Optimization Strategies

### Strategy 1: Mixed Precision Training

**Halve memory usage** by using FP16 instead of FP32:

```rust
use tenflowers_autograd::{AMPConfig, AMPPolicy};

let amp_config = AMPConfig {
    enabled: true,
    target_dtype: DType::Float16,
    initial_scale: 65536.0,
    ..Default::default()
};

let mut amp_policy = AMPPolicy::new(amp_config);

// Parameters: 4 bytes → 2 bytes per element (50% reduction)
// Activations: 4 bytes → 2 bytes per element (50% reduction)
// Total memory: ~50% reduction
```

**Memory savings**:
- Parameters: 50% (e.g., 10 GB → 5 GB)
- Activations: 50% (e.g., 20 GB → 10 GB)
- Gradients: 50% (e.g., 10 GB → 5 GB)
- **Total: ~50% memory reduction**

**Considerations**:
- Use BFloat16 for better numerical stability
- Keep normalization layers in FP32
- Use loss scaling to prevent underflow

### Strategy 2: Gradient Accumulation

**Simulate large batch sizes** without memory cost:

```rust
use tenflowers_autograd::GradientAccumulator;

let mut accumulator = GradientAccumulator::new();
let micro_batch_size = 8;   // Fits in memory
let effective_batch_size = 32;  // Desired batch size
let accumulation_steps = effective_batch_size / micro_batch_size;

for step in 0..accumulation_steps {
    let tape = GradientTape::new();

    // Process micro-batch
    let micro_batch = get_micro_batch(step, micro_batch_size)?;
    let loss = model.forward(&micro_batch)?;
    let grads = tape.gradient(&[loss], &params)?;

    // Accumulate gradients
    accumulator.accumulate(&grads)?;

    // tape freed here → memory released
}

// Average accumulated gradients
let final_grads = accumulator.get_averaged_gradients()?;
optimizer.step(&final_grads)?;

accumulator.clear();
```

**Memory**: O(micro_batch_size)
**Effective batch**: As large as needed
**Tradeoff**: Slightly slower (more forward/backward passes)

### Strategy 3: In-Place Operations

**Reuse memory buffers** instead of allocating new ones:

```rust
// ❌ BAD: Creates new tensor every iteration
for _ in 0..1000 {
    gradient = gradient.mul(&decay)?;  // New allocation
}
// Memory: 1000× gradient size

// ✅ GOOD: Reuses same buffer
for _ in 0..1000 {
    gradient.mul_inplace(&decay)?;  // No allocation
}
// Memory: 1× gradient size (1000× reduction!)
```

**Caution**: Only use when original value is not needed later.

**Safe usage**:
```rust
// Check if tensor is used later
if !tensor.is_used_later() {
    tensor.add_inplace(&delta)?;  // Safe
} else {
    tensor = tensor.add(&delta)?;  // Allocate new
}
```

### Strategy 4: Gradient Checkpointing (Revisited)

**Reduce activation memory** by 50-90%:

```rust
let config = CheckpointConfig {
    strategy: CheckpointStrategy::Auto,
    memory_budget_mb: 8192,  // 8 GB target
    ..Default::default()
};

let tape = GradientTape::with_checkpoint_config(config);

// Automatically checkpoints to fit within budget
```

**Memory reduction examples**:
- ResNet-50: 40% reduction (no significant slowdown)
- GPT-2: 60% reduction (~30% slowdown)
- Vision Transformer: 70% reduction (~50% slowdown)

### Strategy 5: Gradient Compression

**For distributed training**, compress gradients before communication:

```rust
use tenflowers_autograd::{GradientCompressor, CompressionConfig, CompressionMethod};

let config = CompressionConfig {
    method: CompressionMethod::TopK { k: 1000 },
    error_feedback: true,  // Accumulate compression error
    ..Default::default()
};

let mut compressor = GradientCompressor::with_config(config);

// Compress gradients (keep only top-K largest values)
let compressed = compressor.compress(&gradient, "param_name")?;

// Communication: ~90% less data
// Memory: Only store compressed form

// Decompress on receiver
let decompressed = compressor.decompress(&compressed)?;
```

**Memory savings**:
- 80-95% for communication buffers
- Minimal accuracy impact with error feedback

---

## Common Memory Issues

### Issue 1: Out of Memory (OOM)

**Symptoms**:
```
CUDA out of memory. Tried to allocate 512.00 MiB
(GPU 0; 15.90 GiB total capacity; 14.23 GiB already allocated)
```

**Root causes**:
1. Batch size too large
2. Model too deep (too many activations)
3. Large intermediate tensors (e.g., attention maps)
4. Memory leak (gradients not released)

**Solutions**:

**Solution 1: Reduce batch size**
```rust
// Before: batch_size = 64
let batch_size = 32;  // Try halving
// Or use gradient accumulation
```

**Solution 2: Enable checkpointing**
```rust
let config = CheckpointConfig {
    strategy: CheckpointStrategy::Auto,
    memory_budget_mb: current_free_memory_mb * 0.9,
    ..Default::default()
};
```

**Solution 3: Use mixed precision**
```rust
let amp_config = AMPConfig::default();
let amp_policy = AMPPolicy::new(amp_config);
// 50% memory reduction
```

**Solution 4: Clear cache between batches**
```rust
for batch in dataloader {
    // Training step
    optimizer.step(&grads)?;

    // Clear unused memory
    drop(tape);
    drop(grads);
    std::mem::drop(activations);

    // Force cleanup (if needed)
    if batch_idx % 10 == 0 {
        unsafe {
            // GPU-specific cleanup
            // cuda::synchronize()?;
            // cuda::empty_cache()?;
        }
    }
}
```

### Issue 2: Memory Leaks

**Symptoms**:
- Memory usage grows over time
- Eventually crashes with OOM
- Happens after many iterations

**Detection**:
```rust
let profiler = GradientMemoryProfiler::new();
profiler.start_profiling();

for epoch in 0..num_epochs {
    for batch in dataloader {
        // Training step
        train_step(&batch)?;

        // Check for memory growth
        let current_memory = profiler.get_current_usage_mb();
        if current_memory > expected_memory_mb * 1.5 {
            println!("⚠️  Memory leak detected!");
            let leak_report = profiler.detect_leaks()?;
            println!("{:?}", leak_report);
            break;
        }
    }
}
```

**Common causes**:

**1. Accumulating gradients without clearing**:
```rust
// ❌ BAD
let mut all_gradients = Vec::new();
for batch in dataloader {
    let grads = tape.gradient(&[loss], &params)?;
    all_gradients.push(grads);  // LEAK: never cleared
}

// ✅ GOOD
for batch in dataloader {
    let grads = tape.gradient(&[loss], &params)?;
    optimizer.step(&grads)?;
    // grads freed here
}
```

**2. Retaining computation graph**:
```rust
// ❌ BAD
let tape = GradientTape::new();
for batch in dataloader {
    let output = model.forward(&batch)?;
    // tape keeps growing!
}

// ✅ GOOD
for batch in dataloader {
    let tape = GradientTape::new();  // Fresh tape each iteration
    let output = model.forward(&batch)?;
    let grads = tape.gradient(&[output], &params)?;
    // tape freed here
}
```

**3. Circular references** (rare in Rust, but possible with `Rc`):
```rust
// Use `Weak` pointers to break cycles
use std::rc::{Rc, Weak};

struct Node {
    parent: Option<Weak<Node>>,  // Weak reference
    children: Vec<Rc<Node>>,     // Strong references
}
```

### Issue 3: Fragmentation

**Symptoms**:
- "Out of memory" even though total allocated < total available
- Happens after many allocations/deallocations

**Cause**: Memory fragmentation
```
Free memory: [   ][     ][  ][ ][    ]
Requested:   [          ]  (cannot fit!)
```

**Solutions**:

**1. Use memory pools**:
```rust
use tenflowers_autograd::GradientMemoryPool;

let mut pool = GradientMemoryPool::new();

// Pre-allocate common sizes
pool.preallocate_buffers(&[
    (1024, 1024),  // 1M elements
    (2048, 2048),  // 4M elements
    (4096, 4096),  // 16M elements
])?;

// Allocate from pool (reuses buffers)
for batch in dataloader {
    let buffer = pool.allocate(batch.size())?;
    // Use buffer
    pool.deallocate(buffer)?;  // Returns to pool
}
```

**2. Batch similar-sized operations**:
```rust
// ❌ BAD: Alternating sizes
for (small, large) in zip(small_batches, large_batches) {
    process(small)?;  // Allocate small
    process(large)?;  // Allocate large (fragments)
}

// ✅ GOOD: Process same sizes together
for batch in small_batches {
    process(batch)?;
}
for batch in large_batches {
    process(batch)?;
}
```

---

## Best Practices

### Memory Optimization Checklist

- [ ] Profile memory usage before optimizing
- [ ] Enable mixed precision training (FP16/BF16)
- [ ] Use gradient accumulation for large effective batch sizes
- [ ] Enable automatic activation checkpointing
- [ ] Clear gradients and tape after each batch
- [ ] Use in-place operations where safe
- [ ] Preallocate buffers for common sizes
- [ ] Monitor memory usage during training
- [ ] Set memory budget limits
- [ ] Test with different batch sizes

### Memory-Efficient Training Loop

```rust
use tenflowers_autograd::*;

fn memory_efficient_training(
    model: &mut Model,
    dataloader: &DataLoader,
    optimizer: &mut Optimizer,
) -> Result<()> {
    // Configure memory optimizations
    let amp_config = AMPConfig::default();
    let mut amp_policy = AMPPolicy::new(amp_config);

    let checkpoint_config = CheckpointConfig {
        strategy: CheckpointStrategy::Auto,
        memory_budget_mb: 8192,
        ..Default::default()
    };

    let profiler = GradientMemoryProfiler::new();
    profiler.start_profiling();

    // Gradient accumulation setup
    let micro_batch_size = 8;
    let effective_batch_size = 32;
    let accumulation_steps = effective_batch_size / micro_batch_size;
    let mut accumulator = GradientAccumulator::new();

    for epoch in 0..num_epochs {
        for (batch_idx, batch) in dataloader.enumerate() {
            // Process micro-batches
            for step in 0..accumulation_steps {
                // Fresh tape with checkpointing
                let tape = GradientTape::with_checkpoint_config(
                    checkpoint_config.clone()
                );

                // Forward pass with mixed precision
                let micro_batch = batch.slice(step, micro_batch_size)?;
                let output = model.forward(&micro_batch)?;
                let loss = compute_loss(&output, &micro_batch.targets())?;

                // Scale loss for mixed precision
                let scaled_loss = amp_policy.scale_loss(&loss)?;

                // Backward pass
                let mut grads = tape.gradient(&[scaled_loss], &model.parameters())?;

                // Unscale gradients
                if amp_policy.unscale_and_check(&mut grads)? {
                    accumulator.accumulate(&grads)?;
                }

                // tape and grads freed here
            }

            // Apply accumulated gradients
            let final_grads = accumulator.get_averaged_gradients()?;
            optimizer.step(&final_grads)?;
            accumulator.clear();

            // Periodic memory check
            if batch_idx % 100 == 0 {
                let memory_mb = profiler.get_current_usage_mb();
                println!("Batch {}: {:.2} MB", batch_idx, memory_mb);

                if memory_mb > 10_000.0 {  // 10 GB warning
                    println!("⚠️  High memory usage detected!");
                }
            }
        }
    }

    Ok(())
}
```

---

## Case Studies

### Case Study 1: Training GPT-3 (175B Parameters)

**Challenge**: 175 billion parameters don't fit in single GPU (80 GB).

**Solution**:
1. **Mixed precision**: FP16 → 350 GB → 175 GB
2. **Activation checkpointing**: 175 GB → 70 GB (60% reduction)
3. **Gradient checkpointing**: 70 GB → 50 GB
4. **Model parallelism**: Distribute across 8× GPUs → 6.25 GB per GPU
5. **ZeRO optimizer**: Partition optimizer state → 3 GB per GPU

**Result**: Successfully trained on 8× 80GB A100 GPUs.

### Case Study 2: Image Segmentation (High Resolution)

**Challenge**: 4K images (3840×2160) with U-Net → OOM at batch_size=4.

**Before**:
- Resolution: 3840×2160
- Batch size: 4
- Memory: 24 GB
- Result: OOM

**Optimization**:
1. **Patch-based training**: Divide into 512×512 patches
2. **Gradient accumulation**: Accumulate over 4 patches
3. **Mixed precision**: FP16
4. **Selective checkpointing**: Encoder only

**After**:
- Effective batch size: 4 (via accumulation)
- Memory: 8 GB (67% reduction)
- Training time: ~20% slower (acceptable)
- Result: Successful training

---

## Summary

### Key Takeaways

1. **Profile first**: Measure memory usage before optimizing
2. **Mixed precision**: 50% memory reduction with minimal accuracy impact
3. **Checkpointing**: Trade 30-50% time for 50-90% memory reduction
4. **Gradient accumulation**: Achieve large batch sizes without memory cost
5. **Clear resources**: Drop tapes and gradients after each batch
6. **Monitor continuously**: Watch for leaks and fragmentation

### Memory Optimization Priority

**Priority 1** (Essential):
- Enable mixed precision (FP16/BF16)
- Use automatic checkpointing
- Clear gradients after each batch

**Priority 2** (Important):
- Use gradient accumulation
- Profile memory usage
- Set memory budgets

**Priority 3** (Advanced):
- Optimize in-place operations
- Use memory pools
- Compress gradients (distributed training)

### Further Reading

- [Performance Optimization Guide](../../PERFORMANCE_GUIDE.md)
- [Gradient Computation Concepts](../concepts/gradient_computation.md)
- [Mixed Precision Training](./mixed_precision.md)
- [Checkpointing Strategies](./checkpointing.md)

---

**Last Updated**: February 6, 2026
**Author**: COOLJAPAN OU (Team KitaSan)
