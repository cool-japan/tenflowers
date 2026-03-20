# Quick Reference Guide

**Quick lookup for common operations and patterns.**

---

## Table of Contents

1. [Imports](#imports)
2. [Creating a Gradient Tape](#creating-a-gradient-tape)
3. [Computing Gradients](#computing-gradients)
4. [Common Operations](#common-operations)
5. [Gradient Utilities](#gradient-utilities)
6. [Configuration](#configuration)
7. [Debugging](#debugging)
8. [Error Handling](#error-handling)

---

## Imports

### Essential Imports

```rust
use tenflowers_autograd::{GradientTape, TrackedTensor};
use tenflowers_core::Tensor;
use scirs2_core::ndarray::array;
```

### Advanced Features

```rust
// Mixed precision
use tenflowers_autograd::{AMPConfig, AMPPolicy};

// Checkpointing
use tenflowers_autograd::{CheckpointConfig, CheckpointStrategy};

// Numerical validation
use tenflowers_autograd::numerical_checker::NumericalChecker;

// Profiling
use tenflowers_autograd::{GradientMemoryProfiler, PerformanceBenchmark};

// Gradient utilities
use tenflowers_autograd::{clip_by_global_norm, scale_gradients};
```

---

## Creating a Gradient Tape

### Basic Tape

```rust
let tape = GradientTape::new();
```

### With Checkpointing

```rust
let config = CheckpointConfig {
    strategy: CheckpointStrategy::Auto,
    memory_budget_mb: Some(4096),
    ..Default::default()
};
let tape = GradientTape::with_checkpoint_config(config);
```

### With Mixed Precision

```rust
let amp_config = AMPConfig::default();
let tape = GradientTape::with_amp_config(amp_config);
```

### With Deterministic Mode

```rust
let det_config = DeterministicConfig {
    mode: DeterministicMode::Strict,
    global_seed: Some(42),
    ..Default::default()
};
let tape = GradientTape::with_deterministic_config(det_config);
```

---

## Computing Gradients

### Single Input, Single Output

```rust
let tape = GradientTape::new();
let x = tape.watch(Tensor::ones(&[3]));
let y = x.mul(&x)?;
let grads = tape.gradient(&[y], &[x])?;
let dy_dx = grads[0].as_ref().unwrap();
```

### Multiple Inputs

```rust
let tape = GradientTape::new();
let x = tape.watch(Tensor::ones(&[3]));
let y = tape.watch(Tensor::ones(&[3]));
let z = x.mul(&y)?.sum(Some(&[0]), false)?;

let grads = tape.gradient(&[z], &[x.clone(), y.clone()])?;
let dz_dx = grads[0].as_ref().unwrap();
let dz_dy = grads[1].as_ref().unwrap();
```

### Higher-Order Derivatives

```rust
// Second derivative
let tape1 = GradientTape::new();
let x = tape1.watch(Tensor::scalar(2.0f32));
let y = x.pow(&Tensor::scalar(3.0))?;
let dy_dx = tape1.gradient(&[y], &[x])?[0].as_ref().unwrap().clone();

let tape2 = GradientTape::new();
let dy_dx_tracked = tape2.watch(dy_dx);
let d2y_dx2 = tape2.gradient(&[dy_dx_tracked], &[x])?[0].as_ref().unwrap();
```

---

## Common Operations

### Arithmetic Operations

```rust
// Addition
let z = x.add(&y)?;

// Subtraction
let z = x.sub(&y)?;

// Multiplication (element-wise)
let z = x.mul(&y)?;

// Division
let z = x.div(&y)?;

// Power
let z = x.pow(&exponent)?;

// Negation
let z = x.neg()?;
```

### Matrix Operations

```rust
// Matrix multiplication
let z = x.matmul(&y)?;

// Transpose
let z = x.transpose(&[1, 0])?;

// Reshape
let z = x.reshape(&new_shape)?;
```

### Activation Functions

```rust
// ReLU
let z = x.relu()?;

// Sigmoid
let z = x.sigmoid()?;

// Tanh
let z = x.tanh()?;

// GELU
let z = x.gelu()?;

// Softmax
let z = x.softmax(axis)?;
```

### Reduction Operations

```rust
// Sum
let z = x.sum(Some(&[0]), false)?;  // Sum along axis 0
let z = x.sum(None, false)?;         // Sum all elements

// Mean
let z = x.mean(Some(&[0]))?;

// Max
let z = x.max(Some(&[0]))?;

// Min
let z = x.min(Some(&[0]))?;
```

### Mathematical Functions

```rust
// Exponential
let z = x.exp()?;

// Logarithm
let z = x.log()?;

// Square root
let z = x.sqrt()?;

// Sine
let z = x.sin()?;

// Cosine
let z = x.cos()?;
```

---

## Gradient Utilities

### Gradient Clipping

```rust
use tenflowers_autograd::clip_by_global_norm;

// Clip by global norm
let clipped_grads = clip_by_global_norm(&grads, max_norm)?;

// Clip by value
let clipped_grads = clip_by_value(&grads, clip_value)?;
```

### Gradient Scaling

```rust
use tenflowers_autograd::scale_gradients;

let scaled_grads = scale_gradients(&grads, scale_factor)?;
```

### Gradient Accumulation

```rust
use tenflowers_autograd::GradientAccumulator;

let mut accumulator = GradientAccumulator::new();

for step in 0..accumulation_steps {
    let grads = compute_gradients()?;
    accumulator.accumulate(&grads)?;
}

let avg_grads = accumulator.get_averaged_gradients()?;
```

### Numerical Validation

```rust
use tenflowers_autograd::numerical_checker::NumericalChecker;

let checker = NumericalChecker::default();
let result = checker.check_gradient_central(
    &mut tape,
    &x,
    |tape, x| my_function(tape, x),
)?;

if result.is_valid {
    println!("✅ Gradient check passed!");
} else {
    println!("❌ Gradient check failed! Error: {:.2e}", result.max_error);
}
```

---

## Configuration

### Checkpointing Strategies

```rust
// No checkpointing (default)
CheckpointStrategy::None

// Full checkpointing (recompute everything)
CheckpointStrategy::Full

// Selective checkpointing (only expensive ops)
CheckpointStrategy::Selective

// Block checkpointing (every N layers)
CheckpointStrategy::Block

// Automatic (based on memory budget)
CheckpointStrategy::Auto
```

### Mixed Precision Configuration

```rust
let amp_config = AMPConfig {
    enabled: true,
    initial_scale: 65536.0,
    min_scale: 1.0,
    max_scale: 16777216.0,
    growth_interval: 2000,
    growth_factor: 2.0,
    backoff_factor: 0.5,
    target_dtype: DType::Float16,  // or DType::BFloat16
    fp32_operations: vec![
        "batch_norm".to_string(),
        "layer_norm".to_string(),
        "softmax".to_string(),
    ],
    ..Default::default()
};
```

### Deterministic Configuration

```rust
let det_config = DeterministicConfig {
    mode: DeterministicMode::Strict,
    global_seed: Some(42),
    operation_seeds: true,
    cudnn_deterministic: true,
};
```

---

## Debugging

### Memory Profiling

```rust
use tenflowers_autograd::GradientMemoryProfiler;

let mut profiler = GradientMemoryProfiler::new();
profiler.start_profiling();

profiler.record_checkpoint("before_forward")?;
let output = model.forward(&input)?;
profiler.record_checkpoint("after_forward")?;

let forward_memory = profiler.get_memory_delta(
    "before_forward",
    "after_forward"
)?;
println!("Forward memory: {:.2} MB", forward_memory / 1_048_576.0);

// Detect leaks
let leak_report = profiler.detect_leaks()?;
if leak_report.num_suspicious > 0 {
    println!("⚠️  {} potential leaks detected", leak_report.num_suspicious);
}
```

### Performance Benchmarking

```rust
use tenflowers_autograd::PerformanceBenchmark;

let mut benchmark = PerformanceBenchmark::new(Default::default());

benchmark.start_benchmark("backward_pass")?;
let grads = tape.gradient(&[loss], &params)?;
let result = benchmark.end_benchmark("backward_pass")?;

println!("Mean time: {:.2}ms", result.mean_time_ms);
println!("Throughput: {:.0} ops/sec", result.throughput_ops_per_sec);
```

### Gradient Flow Visualization

```rust
use tenflowers_autograd::GradientFlowVisualizer;

let mut visualizer = GradientFlowVisualizer::new();
visualizer.analyze_flow(&tape, &loss, &params)?;

if let Some(analysis) = visualizer.flow_analysis() {
    println!("Health score: {:.1}/100", analysis.health_score);

    for issue in &analysis.issues {
        println!("⚠️  {:?}: {}", issue.issue_type, issue.description);
    }
}
```

---

## Error Handling

### Proper Error Propagation

```rust
// ✅ GOOD: Propagate errors
fn training_step() -> Result<f32> {
    let tape = GradientTape::new();
    let output = model.forward(&input)?;
    let loss = compute_loss(&output)?;
    let grads = tape.gradient(&[loss], &params)?;
    Ok(loss.to_scalar()?)
}

// ❌ BAD: Unwrap (will panic!)
let grads = tape.gradient(&[loss], &params).unwrap();
```

### Handling Optional Gradients

```rust
let grads = tape.gradient(&[loss], &params)?;

for (i, grad_opt) in grads.iter().enumerate() {
    match grad_opt {
        Some(grad) => {
            println!("Gradient {}: {:?}", i, grad.shape());
        }
        None => {
            println!("Gradient {}: None (unused variable)", i);
        }
    }
}
```

---

## Common Patterns

### Training Loop

```rust
fn train(
    model: &mut Model,
    dataloader: &DataLoader,
    optimizer: &mut Optimizer,
    num_epochs: usize,
) -> Result<()> {
    for epoch in 0..num_epochs {
        for (batch_idx, batch) in dataloader.enumerate() {
            // Create fresh tape
            let tape = GradientTape::new();

            // Forward pass
            let output = model.forward(&batch.inputs)?;
            let loss = compute_loss(&output, &batch.targets)?;

            // Backward pass
            let grads = tape.gradient(&[loss], &model.parameters())?;

            // Update parameters
            optimizer.step(&grads)?;

            // Log
            if batch_idx % 100 == 0 {
                println!("Epoch {}, Batch {}: Loss = {:.4}",
                    epoch, batch_idx, loss.to_scalar()?);
            }
        }
    }
    Ok(())
}
```

### With Gradient Accumulation

```rust
let accumulation_steps = 4;
let mut accumulator = GradientAccumulator::new();

for (batch_idx, batch) in dataloader.enumerate() {
    let tape = GradientTape::new();
    let loss = model.forward(&batch)?;
    let grads = tape.gradient(&[loss], &params)?;

    accumulator.accumulate(&grads)?;

    if (batch_idx + 1) % accumulation_steps == 0 {
        let avg_grads = accumulator.get_averaged_gradients()?;
        optimizer.step(&avg_grads)?;
        accumulator.clear();
    }
}
```

### With Mixed Precision

```rust
let amp_config = AMPConfig::default();
let mut amp_policy = AMPPolicy::new(amp_config);

for batch in dataloader {
    let tape = GradientTape::new();

    // Forward
    let loss = model.forward(&batch)?;
    let scaled_loss = amp_policy.scale_loss(&loss)?;

    // Backward
    let mut grads = tape.gradient(&[scaled_loss], &params)?;

    // Unscale and check
    if amp_policy.unscale_and_check(&mut grads)? {
        optimizer.step(&grads)?;
        amp_policy.update_scale(false);
    } else {
        println!("⚠️  Overflow detected, skipping step");
        amp_policy.update_scale(true);
    }
}
```

---

## Common Mistakes

### ❌ Not Watching Tensors

```rust
// Wrong
let x = Tensor::ones(&[3]);
let y = x.mul(&x)?;
let grads = tape.gradient(&[y], &[x])?;  // ERROR

// Correct
let x = tape.watch(Tensor::ones(&[3]));
let y = x.mul(&x)?;
let grads = tape.gradient(&[y], &[x])?;  // OK
```

### ❌ Reusing Tape

```rust
// Wrong
let tape = GradientTape::new();
let grads1 = tape.gradient(&[y1], &[x])?;
let grads2 = tape.gradient(&[y2], &[x])?;  // ERROR: tape moved

// Correct
let tape1 = GradientTape::new();
let grads1 = tape1.gradient(&[y1], &[x])?;

let tape2 = GradientTape::new();
let grads2 = tape2.gradient(&[y2], &[x])?;
```

### ❌ Accumulating without Clearing

```rust
// Wrong (memory leak)
let mut all_grads = Vec::new();
for batch in dataloader {
    let grads = compute_gradients()?;
    all_grads.push(grads);  // Never cleared!
}

// Correct
for batch in dataloader {
    let grads = compute_gradients()?;
    optimizer.step(&grads)?;
    // grads freed here
}
```

---

## Performance Tips

### ✅ Do

- Use mixed precision (FP16/BF16) for 2x speedup
- Enable automatic checkpointing for deep networks
- Batch operations for better GPU utilization
- Clear gradients after each batch
- Use gradient accumulation for large effective batch sizes

### ❌ Don't

- Create new tensors unnecessarily
- Use small batch sizes (underutilizes GPU)
- Keep references to old gradients
- Ignore memory warnings
- Skip numerical validation in development

---

## Keyboard Shortcuts (for examples)

```bash
# Run basic example
cargo run --example first_gradient

# Run with features
cargo run --example mixed_precision --features gpu

# Run benchmarks
cargo bench

# Generate documentation
cargo doc --open

# Run tests
cargo test --all-features
```

---

## Links

### Documentation

- [Quick Start Guide](../../QUICK_START.md)
- [Complete Autograd Guide](../../AUTOGRAD_GUIDE.md)
- [Performance Guide](../../PERFORMANCE_GUIDE.md)
- [API Documentation](../api/gradient_tape.md)

### Examples

- [First Gradient](../../examples/numerical_gradient_validation_example.rs)
- [Second-Order Derivatives](../../examples/second_order_derivatives_example.rs)
- [Mixed Precision](../../examples/mixed_precision_example.rs)
- [Custom Gradients](../../examples/custom_gradient_operations.rs)

---

**Last Updated**: February 6, 2026
**Author**: COOLJAPAN OU (Team KitaSan)
