# TenfloweRS Autograd: Comprehensive Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Getting Started](#getting-started)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Performance Optimization](#performance-optimization)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

---

## Introduction

TenfloweRS Autograd is a production-ready automatic differentiation library for Rust, providing efficient gradient computation for machine learning and scientific computing applications. Built on the SciRS2 ecosystem, it offers both tape-based reverse mode and forward mode differentiation with advanced memory management and performance optimization features.

### Key Features

- **Dual-Mode Differentiation**: Both reverse-mode (backpropagation) and forward-mode AD
- **Hybrid Scheduler**: Automatic selection of optimal differentiation strategy
- **Memory Efficiency**: Gradient checkpointing with multiple strategies
- **Mixed Precision**: FP16/BFloat16 support with dynamic loss scaling
- **Deterministic Training**: Reproducible results with seed management
- **Performance Tools**: Built-in profiling, benchmarking, and optimization
- **Production Ready**: Comprehensive error handling, debugging, and monitoring

### When to Use TenfloweRS Autograd

✅ **Use for:**
- Neural network training
- Scientific computing with gradient-based optimization
- Inverse problems and parameter estimation
- Sensitivity analysis and uncertainty quantification
- Research requiring reproducible gradients

❌ **Consider alternatives for:**
- Static computation graphs (consider TensorFlow-style graph mode)
- Symbolic differentiation (consider SymPy/SymEngine)
- Simple numerical derivatives (finite differences may suffice)

---

## Core Concepts

### 1. Gradient Tape

The **GradientTape** is the central abstraction for automatic differentiation. It records operations on tensors to build a computation graph, which is then traversed in reverse to compute gradients.

```rust
use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;

// Create a tape
let tape = GradientTape::new();

// Operations are recorded automatically
let x = tape.watch(Tensor::ones(&[2, 2]));
let y = x.mul(&x)?;  // Recorded: y = x²

// Compute gradients
let grads = tape.gradient(&[y], &[x])?;  // dy/dx = 2x
```

**Key Properties:**
- **Eager execution**: Operations execute immediately while recording
- **Dynamic graphs**: Graph structure can vary per iteration
- **Single-use**: Each tape is consumed after computing gradients
- **Thread-safe**: Multiple tapes can coexist independently

### 2. Tracked Tensors

**TrackedTensor** wraps regular tensors to enable gradient tracking:

```rust
// Untracked tensor - no gradient computation
let a = Tensor::ones(&[3, 3]);

// Tracked tensor - participates in gradient computation
let b = tape.watch(a);
```

**Characteristics:**
- Lightweight wrapper around `Tensor<T>`
- Carries a unique identifier for tape association
- Automatically propagates through operations
- Can be converted back to regular tensors

### 3. Differentiation Modes

#### Reverse Mode (Backpropagation)

**Best for:** Many inputs → Few outputs (typical neural networks)

```rust
// 1000 parameters → 1 loss value
// Reverse mode: O(1) backward pass computes all gradients
let loss = model.forward(&inputs)?;
let grads = tape.gradient(&[loss], &parameters)?;  // Efficient!
```

**Complexity:** O(output_dim) for computing all gradients

#### Forward Mode

**Best for:** Few inputs → Many outputs (Jacobian computation)

```rust
// 1 input → 1000 outputs
// Forward mode: O(1) forward pass computes all derivatives
let outputs = compute_function(&input)?;
let jacobian = tape.forward_gradient(&input, &outputs)?;  // Efficient!
```

**Complexity:** O(input_dim) for computing all derivatives

#### Hybrid Mode (Automatic Selection)

The **HybridScheduler** automatically selects the optimal mode:

```rust
let config = SchedulerConfig {
    strategy: SchedulingStrategy::Auto,
    forward_mode_threshold: 10.0,
    reverse_mode_threshold: 0.1,
    enable_mixed_mode: true,
};

let tape = GradientTape::with_scheduler_config(config);
// Automatically uses forward or reverse mode based on graph structure
```

### 4. Higher-Order Derivatives

Compute derivatives of derivatives (Hessians, etc.):

```rust
// First-order gradient
let tape1 = GradientTape::new();
let x = tape1.watch(tensor);
let y = x.pow(&Tensor::scalar(3.0))?;  // y = x³
let dy_dx = tape1.gradient(&[y], &[x])?;  // dy/dx = 3x²

// Second-order gradient
let tape2 = GradientTape::new();
let dy_dx_tracked = tape2.watch(dy_dx);
let d2y_dx2 = tape2.gradient(&[dy_dx_tracked], &[x])?;  // d²y/dx² = 6x
```

---

## Getting Started

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
tenflowers-autograd = "0.1.0-alpha.2"
tenflowers-core = "0.1.0-alpha.2"
scirs2-autograd = "0.1.0-rc.1"  # Core dependency
```

### Minimal Example

```rust
use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;
use scirs2_autograd::ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create gradient tape
    let tape = GradientTape::new();

    // Create input tensor
    let x = tape.watch(Tensor::from_array(
        array![[1.0f32, 2.0], [3.0, 4.0]].into_dyn()
    ));

    // Forward pass: y = x²
    let y = x.mul(&x)?;

    // Sum for scalar loss
    let loss = y.sum(None, false)?;

    // Compute gradient: dy/dx = 2x
    let grads = tape.gradient(&[loss], &[x])?;

    println!("Gradient: {:?}", grads[0]);
    Ok(())
}
```

**Expected output:**
```
Gradient: [[2.0, 4.0], [6.0, 8.0]]  // 2x
```

---

## Basic Usage

### Computing Gradients

#### Single Input, Single Output

```rust
let tape = GradientTape::new();
let x = tape.watch(Tensor::from_scalar(3.0f32));

// y = x² + 2x + 1
let x_squared = x.mul(&x)?;
let two_x = x.mul(&Tensor::scalar(2.0))?;
let y = x_squared.add(&two_x)?.add(&Tensor::scalar(1.0))?;

// dy/dx = 2x + 2
let grad = tape.gradient(&[y], &[x])?;
println!("dy/dx at x=3: {}", grad[0].to_scalar()?);  // 8.0 = 2(3) + 2
```

#### Multiple Inputs

```rust
let tape = GradientTape::new();
let x = tape.watch(Tensor::ones(&[2, 2]));
let y = tape.watch(Tensor::ones(&[2, 2]));

// z = x² + y²
let z = x.mul(&x)?.add(&y.mul(&y)?)?;
let loss = z.sum(None, false)?;

// Compute both gradients simultaneously
let grads = tape.gradient(&[loss], &[x.clone(), y.clone()])?;
let dx = &grads[0];  // dz/dx = 2x
let dy = &grads[1];  // dz/dy = 2y
```

#### Multiple Outputs

```rust
let tape = GradientTape::new();
let x = tape.watch(Tensor::ones(&[3]));

let y1 = x.mul(&Tensor::scalar(2.0))?;
let y2 = x.mul(&x)?;

// Compute gradients for both outputs
let grad_y1 = tape.gradient(&[y1], &[x.clone()])?;  // dy1/dx = 2
let grad_y2 = tape.gradient(&[y2], &[x.clone()])?;  // dy2/dx = 2x
```

### Working with Neural Networks

```rust
use tenflowers_neural::{Layer, Dense};

fn training_step(
    model: &Dense<f32>,
    inputs: &Tensor<f32>,
    targets: &Tensor<f32>,
    learning_rate: f32,
) -> Result<f32> {
    let tape = GradientTape::new();

    // Forward pass with tracked weights
    let weights_tracked = tape.watch(model.weights().clone());
    let predictions = forward_with_tracked_weights(&weights_tracked, inputs)?;

    // Compute loss
    let loss = mse_loss(&predictions, targets)?;
    let loss_value = loss.to_scalar()?;

    // Backward pass
    let grads = tape.gradient(&[loss], &[weights_tracked])?;

    // Update weights
    let updated_weights = model.weights()
        .sub(&grads[0].mul(&Tensor::scalar(learning_rate))?)?;
    model.set_weights(updated_weights)?;

    Ok(loss_value)
}
```

---

## Advanced Features

### 1. Gradient Checkpointing

Trade computation for memory by recomputing activations during backward pass:

```rust
use tenflowers_autograd::{CheckpointConfig, CheckpointStrategy};

// Configure selective checkpointing
let config = CheckpointConfig {
    strategy: CheckpointStrategy::Selective,
    checkpoint_every_n: 3,  // Checkpoint every 3 layers
    min_compute_cost: 100,  // Only checkpoint expensive ops
    memory_budget_mb: Some(1024),  // 1GB memory limit
};

let tape = GradientTape::with_checkpoint_config(config);

// Build deep network - activations recomputed as needed
for i in 0..50 {
    hidden = layer_forward(&hidden)?;
    // Only every 3rd layer's activations are stored
}
```

**Strategies:**

- **None**: Store all activations (fastest, most memory)
- **Selective**: Store only expensive operations
- **Block**: Checkpoint every N operations
- **Full**: Recompute everything (slowest, least memory)
- **Auto**: Automatically decide based on memory budget

**Memory Savings:** 40-60% typical, up to 90% for very deep networks

### 2. Mixed Precision Training

Accelerate training with FP16/BFloat16:

```rust
use tenflowers_autograd::{AmpConfig, PrecisionMode, LossScaler};

let amp_config = AmpConfig {
    enabled: true,
    precision_mode: PrecisionMode::Float16,
    loss_scaler: LossScaler::Dynamic {
        init_scale: 65536.0,
        growth_factor: 2.0,
        backoff_factor: 0.5,
        growth_interval: 2000,
    },
    autocast_ops: vec![
        "matmul".to_string(),
        "conv2d".to_string(),
    ],
    keep_fp32_ops: vec![
        "softmax".to_string(),
        "layer_norm".to_string(),
    ],
};

let tape = GradientTape::with_amp_config(amp_config);

// Operations automatically cast to FP16 where safe
// Loss scaling prevents underflow in gradients
```

**Benefits:**
- 2x faster training on modern GPUs
- 50% memory reduction
- Minimal accuracy impact with proper loss scaling

**Considerations:**
- Use BFloat16 for better numerical stability
- Keep normalization operations in FP32
- Monitor for NaN/Inf during training

### 3. Deterministic Training

Ensure reproducible results:

```rust
use tenflowers_autograd::{DeterministicConfig, DeterministicMode};

let det_config = DeterministicConfig {
    mode: DeterministicMode::Strict,
    global_seed: Some(42),
    operation_seeds: true,
    cudnn_deterministic: true,
};

let tape = GradientTape::with_deterministic_config(det_config);

// All random operations use deterministic seeding
// Same seed → identical results every time
```

**Use Cases:**
- Debugging gradient computation bugs
- Reproducing research results
- Unit testing with expected outputs
- Benchmarking and performance comparisons

### 4. Performance Benchmarking

Measure and track gradient computation performance:

```rust
use tenflowers_autograd::{PerformanceBenchmark, BenchmarkConfig};

let config = BenchmarkConfig {
    num_warmup: 10,
    num_iterations: 100,
    measure_throughput: true,
    measure_memory: true,
    statistical_analysis: true,
};

let mut benchmark = PerformanceBenchmark::new(config);

benchmark.start_benchmark("backward_pass")?;
// Run backward pass
let grads = tape.gradient(&[loss], &parameters)?;
let result = benchmark.end_benchmark("backward_pass")?;

println!("Mean time: {:.2}ms", result.mean_time_ms);
println!("Throughput: {:.0} ops/sec", result.throughput_ops_per_sec);
```

### 5. Memory Profiling

Track memory usage and detect leaks:

```rust
use tenflowers_autograd::GradientMemoryProfiler;

let mut profiler = GradientMemoryProfiler::new();
profiler.start_profiling();

profiler.record_checkpoint("before_forward")?;
let output = model.forward(&input)?;
profiler.record_checkpoint("after_forward")?;

let grads = tape.gradient(&[output], &[input])?;
profiler.record_checkpoint("after_backward")?;

// Get memory delta
let forward_memory = profiler.get_memory_delta(
    "before_forward",
    "after_forward"
)?;
println!("Forward pass used: {:.2}MB", forward_memory.unwrap());

// Check for leaks
let leak_report = profiler.detect_leaks()?;
if leak_report.num_suspicious > 0 {
    println!("Warning: {} potential leaks detected", leak_report.num_suspicious);
}
```

---

## Performance Optimization

### Optimization Checklist

- [ ] Use gradient checkpointing for deep networks
- [ ] Enable mixed precision (AMP) on compatible hardware
- [ ] Profile to identify bottlenecks
- [ ] Use hybrid scheduler for complex graphs
- [ ] Batch operations when possible
- [ ] Minimize CPU↔GPU transfers
- [ ] Reuse tensors to reduce allocations
- [ ] Consider in-place operations where safe

### Common Bottlenecks

#### 1. Memory Bandwidth

**Problem:** Excessive memory allocations/deallocations

**Solutions:**
```rust
// Bad: Create new tensors every iteration
for _ in 0..1000 {
    let temp = x.mul(&y)?;  // Allocates each time
}

// Good: Reuse buffers
let mut temp = Tensor::zeros(&shape);
for _ in 0..1000 {
    x.mul_into(&y, &mut temp)?;  // In-place
}
```

#### 2. Small Batch Sizes

**Problem:** Underutilizing GPU parallelism

**Solutions:**
- Increase batch size (2x batch = ~1.5-1.8x speedup)
- Use gradient accumulation for effective larger batches
- Consider per-sample gradient computation only when necessary

#### 3. Tape Overhead

**Problem:** Recording overhead for simple operations

**Solutions:**
```rust
// Bad: Track everything
let x = tape.watch(input);
let y = x.add(&Tensor::scalar(1.0))?;  // Records even trivial ops

// Good: Track only what needs gradients
let x = tape.watch(input);
let y = input.add(&Tensor::scalar(1.0))?;  // Not tracked
let z = tape.watch(y);  // Only track when needed
```

### Performance Patterns

#### Pattern 1: Gradient Accumulation

```rust
// Simulate larger batch sizes
let accumulation_steps = 4;
let mut accumulated_grads = vec![Tensor::zeros(&shape); num_params];

for step in 0..accumulation_steps {
    let tape = GradientTape::new();
    let output = model.forward(&batch)?;
    let loss = compute_loss(&output)?;
    let grads = tape.gradient(&[loss], &parameters)?;

    // Accumulate gradients
    for (accum, grad) in accumulated_grads.iter_mut().zip(grads.iter()) {
        *accum = accum.add(grad)?;
    }
}

// Average and apply
for grad in &mut accumulated_grads {
    *grad = grad.div(&Tensor::scalar(accumulation_steps as f32))?;
}
optimizer.step(&accumulated_grads)?;
```

#### Pattern 2: Conditional Gradients

```rust
// Only compute gradients when needed (e.g., validation)
fn forward(
    model: &Model,
    input: &Tensor<f32>,
    training: bool,
) -> Result<Tensor<f32>> {
    if training {
        let tape = GradientTape::new();
        let input_tracked = tape.watch(input.clone());
        model.forward(&input_tracked)
    } else {
        model.forward(input)  // No tracking overhead
    }
}
```

---

## Best Practices

### 1. Tape Management

✅ **Do:**
```rust
// Create new tape for each iteration
for _ in 0..epochs {
    let tape = GradientTape::new();
    // ... training step ...
}  // Tape dropped, memory released
```

❌ **Don't:**
```rust
// Reuse tape (not possible - consumed by gradient())
let tape = GradientTape::new();
for _ in 0..epochs {
    let grads = tape.gradient(&[loss], &params)?;  // ERROR: tape moved
}
```

### 2. Gradient Validation

Always validate gradients during development:

```rust
#[cfg(test)]
mod tests {
    use tenflowers_autograd::NumericalChecker;

    #[test]
    fn test_custom_operation_gradient() {
        let checker = NumericalChecker::default();
        let mut tape = GradientTape::new();
        let x = tape.watch(Tensor::ones(&[5]));

        // Test your custom operation
        let result = checker.check_gradient_central(
            &mut tape,
            &x,
            |tape, x| custom_op(tape, x),
        )?;

        assert!(result.is_valid, "Gradient check failed: {}", result.max_error);
    }
}
```

### 3. Error Handling

Use proper error handling:

```rust
// Good: Propagate errors
fn training_step(model: &Model) -> Result<f32> {
    let tape = GradientTape::new();
    let output = model.forward(&input)?;  // Propagate
    let loss = compute_loss(&output)?;    // Propagate
    let grads = tape.gradient(&[loss], &params)?;  // Propagate
    Ok(loss.to_scalar()?)
}

// Bad: Unwrap in production code
let grads = tape.gradient(&[loss], &params).unwrap();  // Will panic!
```

### 4. Resource Cleanup

```rust
// Explicit cleanup for long-running processes
{
    let tape = GradientTape::new();
    let grads = tape.gradient(&[loss], &params)?;
    optimizer.step(&grads)?;
}  // Tape and intermediate tensors dropped here

// Force cleanup if needed
std::mem::drop(large_tensor);
```

---

## Troubleshooting

### Common Issues

#### Issue 1: NaN/Inf in Gradients

**Symptoms:** Gradients become NaN or Inf during training

**Causes & Solutions:**

1. **Exploding gradients:**
   ```rust
   use tenflowers_autograd::gradient_clipping;

   let grads = tape.gradient(&[loss], &params)?;
   let clipped = gradient_clipping::clip_by_global_norm(&grads, 1.0)?;
   ```

2. **Numerical instability:**
   ```rust
   // Use log-space computations
   let log_probs = logits.log_softmax()?;
   let loss = -log_probs.sum()?;  // More stable than exp
   ```

3. **Mixed precision issues:**
   ```rust
   // Check loss scale
   if loss.to_scalar()? > 1e10 || loss.to_scalar()? < 1e-10 {
       println!("Warning: Loss scale may need adjustment");
   }
   ```

#### Issue 2: Out of Memory

**Solutions:**

1. **Enable checkpointing:**
   ```rust
   let config = CheckpointConfig {
       strategy: CheckpointStrategy::Auto,
       memory_budget_mb: Some(4096),  // 4GB limit
       ..Default::default()
   };
   ```

2. **Reduce batch size:**
   ```rust
   // Use gradient accumulation for effective larger batches
   let effective_batch = 64;
   let actual_batch = 16;
   let accumulation_steps = effective_batch / actual_batch;
   ```

3. **Profile memory:**
   ```rust
   let profiler = GradientMemoryProfiler::new();
   // Identify which operations use most memory
   ```

#### Issue 3: Slow Gradient Computation

**Diagnosis:**

```rust
use tenflowers_autograd::PerformanceBenchmark;

let mut benchmark = PerformanceBenchmark::new(Default::default());
benchmark.start_benchmark("backward")?;
let grads = tape.gradient(&[loss], &params)?;
let result = benchmark.end_benchmark("backward")?;

println!("Backward pass: {:.2}ms", result.mean_time_ms);
```

**Optimizations:**
- Enable mixed precision
- Use hybrid scheduler
- Batch operations
- Reduce tape overhead

#### Issue 4: Non-Deterministic Results

**Solution:**

```rust
use tenflowers_autograd::DeterministicConfig;

let config = DeterministicConfig {
    mode: DeterministicMode::Strict,
    global_seed: Some(42),
    operation_seeds: true,
    cudnn_deterministic: true,
};

let tape = GradientTape::with_deterministic_config(config);
```

---

## API Reference

### Core Types

#### `GradientTape`

Main automatic differentiation engine.

```rust
impl GradientTape {
    // Creation
    pub fn new() -> Self;
    pub fn with_checkpoint_config(config: CheckpointConfig) -> Self;
    pub fn with_amp_config(config: AmpConfig) -> Self;
    pub fn with_scheduler_config(config: SchedulerConfig) -> Self;
    pub fn with_deterministic_config(config: DeterministicConfig) -> Self;

    // Watching tensors
    pub fn watch<T>(&self, tensor: Tensor<T>) -> TrackedTensor<T>;

    // Computing gradients
    pub fn gradient<T>(
        self,
        targets: &[TrackedTensor<T>],
        sources: &[TrackedTensor<T>],
    ) -> Result<Vec<Tensor<T>>>;

    // Forward mode
    pub fn forward_gradient<T>(
        &self,
        source: &TrackedTensor<T>,
        target: &TrackedTensor<T>,
    ) -> Result<Tensor<T>>;
}
```

#### `TrackedTensor<T>`

Wrapper for gradient-enabled tensors.

```rust
impl<T> TrackedTensor<T> {
    // Access underlying tensor
    pub fn tensor(&self) -> &Tensor<T>;

    // Operations (same as Tensor)
    pub fn add(&self, other: &TrackedTensor<T>) -> Result<TrackedTensor<T>>;
    pub fn mul(&self, other: &TrackedTensor<T>) -> Result<TrackedTensor<T>>;
    pub fn matmul(&self, other: &TrackedTensor<T>) -> Result<TrackedTensor<T>>;
    // ... all tensor operations ...
}
```

### Configuration Types

#### `CheckpointConfig`

```rust
pub struct CheckpointConfig {
    pub strategy: CheckpointStrategy,
    pub checkpoint_every_n: usize,
    pub min_compute_cost: usize,
    pub memory_budget_mb: Option<usize>,
}
```

#### `AmpConfig`

```rust
pub struct AmpConfig {
    pub enabled: bool,
    pub precision_mode: PrecisionMode,
    pub loss_scaler: LossScaler,
    pub autocast_ops: Vec<String>,
    pub keep_fp32_ops: Vec<String>,
}
```

#### `DeterministicConfig`

```rust
pub struct DeterministicConfig {
    pub mode: DeterministicMode,
    pub global_seed: Option<u64>,
    pub operation_seeds: bool,
    pub cudnn_deterministic: bool,
}
```

### Utility Functions

#### Gradient Clipping

```rust
pub mod gradient_clipping {
    pub fn clip_by_global_norm<T>(
        gradients: &[Tensor<T>],
        max_norm: T,
    ) -> Result<Vec<Tensor<T>>>;

    pub fn clip_by_value<T>(
        gradients: &[Tensor<T>],
        clip_value: T,
    ) -> Result<Vec<Tensor<T>>>;
}
```

#### Numerical Validation

```rust
pub struct NumericalChecker {
    pub fn check_gradient_central<F>(
        &self,
        tape: &mut GradientTape,
        input: &Tensor<f32>,
        function: F,
    ) -> Result<ValidationResult>
    where
        F: Fn(&mut GradientTape, &Tensor<f32>) -> Result<Tensor<f32>>;
}
```

---

## Conclusion

TenfloweRS Autograd provides a comprehensive automatic differentiation system with:

- ✅ Production-ready gradient computation
- ✅ Advanced memory and performance optimization
- ✅ Comprehensive debugging and profiling tools
- ✅ Full SciRS2 ecosystem integration

For more examples, see the `examples/` directory. For API details, run `cargo doc --open`.

### Additional Resources

- [Examples Directory](./examples/)
- [API Documentation](https://docs.rs/tenflowers-autograd)
- [GitHub Repository](https://github.com/cool-japan/tenflowers)
- [SciRS2 Documentation](https://github.com/cool-japan/scirs)

### Contributing

Contributions welcome! See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

### License

See [LICENSE](../../LICENSE) for details.
