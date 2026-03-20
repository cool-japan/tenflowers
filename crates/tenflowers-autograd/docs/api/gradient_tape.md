# GradientTape API Reference

**Module**: `tenflowers_autograd::tape`
**Main Type**: `GradientTape`

---

## Overview

`GradientTape` is the core automatic differentiation engine in TenfloweRS. It records operations on tensors to build a computational graph, which is then traversed in reverse to compute gradients via backpropagation.

### Key Features

- **Eager execution**: Operations execute immediately while being recorded
- **Dynamic graphs**: Graph structure can vary between iterations
- **Memory efficient**: Automatic cleanup of intermediate values
- **Configurable**: Supports checkpointing, mixed precision, deterministic mode
- **Type-safe**: Generic over numeric types (f32, f64, etc.)

---

## Type Definition

```rust
pub struct GradientTape {
    // Private fields
}
```

---

## Constructors

### `new()`

Creates a new gradient tape with default configuration.

```rust
pub fn new() -> Self
```

**Example**:
```rust
let tape = GradientTape::new();
```

**Configuration**: Uses default settings (no checkpointing, no mixed precision).

---

### `with_checkpoint_config()`

Creates a gradient tape with custom activation checkpointing configuration.

```rust
pub fn with_checkpoint_config(config: CheckpointConfig) -> Self
```

**Parameters**:
- `config`: Checkpointing strategy and parameters

**Example**:
```rust
let config = CheckpointConfig {
    strategy: CheckpointStrategy::Auto,
    memory_budget_mb: Some(4096),
    ..Default::default()
};
let tape = GradientTape::with_checkpoint_config(config);
```

**Use Case**: Memory-constrained environments, deep networks.

---

### `with_amp_config()`

Creates a gradient tape with automatic mixed precision (AMP) configuration.

```rust
pub fn with_amp_config(config: AMPConfig) -> Self
```

**Parameters**:
- `config`: Mixed precision settings (FP16/BF16, loss scaling, etc.)

**Example**:
```rust
let amp_config = AMPConfig {
    enabled: true,
    target_dtype: DType::Float16,
    initial_scale: 65536.0,
    ..Default::default()
};
let tape = GradientTape::with_amp_config(amp_config);
```

**Use Case**: GPU training acceleration, memory reduction.

---

### `with_deterministic_config()`

Creates a gradient tape with deterministic execution configuration.

```rust
pub fn with_deterministic_config(config: DeterministicConfig) -> Self
```

**Parameters**:
- `config`: Determinism settings (seeds, reproducibility)

**Example**:
```rust
let det_config = DeterministicConfig {
    mode: DeterministicMode::Strict,
    global_seed: Some(42),
    operation_seeds: true,
    ..Default::default()
};
let tape = GradientTape::with_deterministic_config(det_config);
```

**Use Case**: Reproducible experiments, debugging, testing.

---

### `with_scheduler_config()`

Creates a gradient tape with hybrid differentiation scheduler.

```rust
pub fn with_scheduler_config(config: SchedulerConfig) -> Self
```

**Parameters**:
- `config`: Strategy for choosing forward vs. reverse mode

**Example**:
```rust
let config = SchedulerConfig {
    strategy: SchedulingStrategy::Auto,
    forward_mode_threshold: 10.0,
    reverse_mode_threshold: 0.1,
    ..Default::default()
};
let tape = GradientTape::with_scheduler_config(config);
```

**Use Case**: Optimal differentiation for complex graphs with varying dimensions.

---

## Core Methods

### `watch()`

Marks a tensor for gradient computation by wrapping it in a `TrackedTensor`.

```rust
pub fn watch<T>(&self, tensor: Tensor<T>) -> TrackedTensor<T>
where
    T: Clone + Send + Sync + 'static
```

**Parameters**:
- `tensor`: Input tensor to track

**Returns**: `TrackedTensor<T>` - Gradient-enabled tensor wrapper

**Example**:
```rust
let tape = GradientTape::new();
let x = tape.watch(Tensor::ones(&[2, 2]));
// x is now tracked for gradients
```

**Note**: Only watched tensors can be used as gradient sources.

---

### `gradient()`

Computes gradients of target tensors with respect to source tensors.

```rust
pub fn gradient<T>(
    self,
    targets: &[TrackedTensor<T>],
    sources: &[TrackedTensor<T>],
) -> Result<Vec<Option<Tensor<T>>>>
where
    T: Clone + Send + Sync + 'static
```

**Parameters**:
- `targets`: Output tensors (typically loss)
- `sources`: Input tensors (typically parameters)

**Returns**: `Result<Vec<Option<Tensor<T>>>>` - Gradients for each source
  - `Some(gradient)` if gradient exists
  - `None` if no gradient (e.g., unused variable)

**Consumes**: The tape (single-use)

**Example**:
```rust
let tape = GradientTape::new();
let x = tape.watch(Tensor::ones(&[3]));
let y = x.mul(&x)?;  // y = x²

let grads = tape.gradient(&[y], &[x])?;
let dy_dx = grads[0].as_ref().unwrap();  // dy/dx = 2x
```

**Algorithm**:
1. Initialize adjoint of each target to 1
2. Traverse computational graph in reverse topological order
3. For each operation, compute gradients using chain rule
4. Accumulate gradients at each source

**Complexity**: O(num_operations × operation_cost)

---

### `forward_gradient()` (Forward-Mode AD)

Computes directional derivatives using forward-mode automatic differentiation.

```rust
pub fn forward_gradient<T>(
    &self,
    source: &TrackedTensor<T>,
    target: &TrackedTensor<T>,
) -> Result<Tensor<T>>
where
    T: Clone + Send + Sync + 'static
```

**Parameters**:
- `source`: Input tensor
- `target`: Output tensor

**Returns**: `Result<Tensor<T>>` - Forward-mode gradient (Jacobian-vector product)

**Example**:
```rust
let tape = GradientTape::new();
let x = tape.watch(Tensor::ones(&[3]));
let y = some_function(&x)?;

let forward_grad = tape.forward_gradient(&x, &y)?;
```

**Use Case**: Few inputs, many outputs (n ≪ m). More efficient than reverse-mode for computing Jacobian columns.

**Complexity**: O(num_operations × operation_cost) per column

---

### `nth_derivative()`

Computes nth-order derivatives by nesting gradient computations.

```rust
pub fn nth_derivative<T>(
    &self,
    target: &TrackedTensor<T>,
    source: &TrackedTensor<T>,
    order: usize,
) -> Result<Tensor<T>>
where
    T: Clone + Send + Sync + 'static
```

**Parameters**:
- `target`: Output tensor
- `source`: Input tensor
- `order`: Derivative order (1 = first derivative, 2 = second, etc.)

**Returns**: `Result<Tensor<T>>` - nth-order derivative

**Example**:
```rust
let tape = GradientTape::new();
let x = tape.watch(Tensor::scalar(2.0f32));
let y = x.pow(&Tensor::scalar(4.0))?;  // y = x⁴

// First derivative: dy/dx = 4x³ = 4(2³) = 32
let first = tape.nth_derivative(&y, &x, 1)?;

// Second derivative: d²y/dx² = 12x² = 12(2²) = 48
let second = tape.nth_derivative(&y, &x, 2)?;

// Third derivative: d³y/dx³ = 24x = 24(2) = 48
let third = tape.nth_derivative(&y, &x, 3)?;
```

**Complexity**: O(order × num_operations × operation_cost)

---

### `hessian()`

Computes the Hessian matrix (second-order derivatives).

```rust
pub fn hessian<T>(
    &self,
    target: &TrackedTensor<T>,
    sources: &[TrackedTensor<T>],
) -> Result<Tensor<T>>
where
    T: Clone + Send + Sync + 'static
```

**Parameters**:
- `target`: Scalar output (loss)
- `sources`: Input tensors (parameters)

**Returns**: `Result<Tensor<T>>` - Hessian matrix [n × n]

**Example**:
```rust
let tape = GradientTape::new();
let x = tape.watch(Tensor::from_vec(vec![1.0, 2.0], &[2])?);
let y = quadratic_function(&x)?;  // Scalar output

let hessian = tape.hessian(&y, &[x])?;
// Hessian is [2 × 2] symmetric matrix
```

**Use Case**: Second-order optimization (Newton's method), curvature analysis.

**Complexity**: O(n × num_operations × operation_cost) using forward-over-reverse

---

### `jacobian()`

Computes the Jacobian matrix (all first-order partial derivatives).

```rust
pub fn jacobian<T>(
    &self,
    targets: &[TrackedTensor<T>],
    sources: &[TrackedTensor<T>],
) -> Result<Tensor<T>>
where
    T: Clone + Send + Sync + 'static
```

**Parameters**:
- `targets`: Output tensors (m elements)
- `sources`: Input tensors (n elements)

**Returns**: `Result<Tensor<T>>` - Jacobian matrix [m × n]

**Example**:
```rust
let tape = GradientTape::new();
let x = tape.watch(Tensor::from_vec(vec![1.0, 2.0], &[2])?);
let y = vector_function(&x)?;  // Returns [3] vector

let jacobian = tape.jacobian(&[y], &[x])?;
// Jacobian is [3 × 2] matrix
```

**Complexity**:
- If m < n: O(m × cost) using reverse-mode
- If n < m: O(n × cost) using forward-mode
- Automatically chooses optimal strategy

---

## Utility Methods

### `set_gradient()`

Manually sets the gradient (adjoint) for a tensor.

```rust
pub fn set_gradient<T>(&mut self, tensor_id: TensorId, gradient: Tensor<T>)
where
    T: Clone + Send + Sync + 'static
```

**Parameters**:
- `tensor_id`: Unique identifier of the tensor
- `gradient`: Gradient value to set

**Use Case**: Custom gradient computation, implicit differentiation.

---

### `accumulate_gradient()`

Accumulates (adds) a gradient to existing gradient for a tensor.

```rust
pub fn accumulate_gradient<T>(
    &mut self,
    tensor_id: TensorId,
    gradient: Tensor<T>,
) -> Result<()>
where
    T: Clone + Send + Sync + 'static
```

**Parameters**:
- `tensor_id`: Unique identifier of the tensor
- `gradient`: Gradient to add

**Use Case**: Gradient accumulation, multi-branch gradients.

---

### `get_gradient()`

Retrieves the current gradient for a tensor.

```rust
pub fn get_gradient<T>(&self, tensor_id: TensorId) -> Result<Option<Tensor<T>>>
where
    T: Clone + Send + Sync + 'static
```

**Parameters**:
- `tensor_id`: Unique identifier of the tensor

**Returns**: `Result<Option<Tensor<T>>>` - Current gradient if available

---

### `clear_gradients()`

Clears all stored gradients.

```rust
pub fn clear_gradients(&mut self)
```

**Use Case**: Memory cleanup, resetting state.

---

## Configuration Types

### `CheckpointConfig`

Configuration for activation checkpointing.

```rust
pub struct CheckpointConfig {
    pub strategy: CheckpointStrategy,
    pub checkpoint_every_n: usize,
    pub min_compute_cost: usize,
    pub memory_budget_mb: Option<usize>,
}
```

**Fields**:
- `strategy`: Checkpointing strategy (None, Full, Selective, Block, Auto)
- `checkpoint_every_n`: Interval for block checkpointing
- `min_compute_cost`: Minimum FLOPs to checkpoint (for Selective)
- `memory_budget_mb`: Memory budget constraint (for Auto)

---

### `CheckpointStrategy`

Checkpointing strategies.

```rust
pub enum CheckpointStrategy {
    None,                    // Store all activations
    Full,                    // Store no activations (recompute all)
    Selective,               // Checkpoint expensive ops only
    Block,                   // Checkpoint every N layers
    Auto,                    // Automatically determine based on budget
}
```

---

### `AMPConfig`

Configuration for automatic mixed precision.

```rust
pub struct AMPConfig {
    pub enabled: bool,
    pub initial_scale: f32,
    pub min_scale: f32,
    pub max_scale: f32,
    pub growth_interval: usize,
    pub growth_factor: f32,
    pub backoff_factor: f32,
    pub target_dtype: DType,
    pub fp32_operations: Vec<String>,
}
```

**Fields**:
- `enabled`: Enable mixed precision
- `initial_scale`: Initial loss scaling factor
- `min_scale`: Minimum scale (safety floor)
- `max_scale`: Maximum scale (safety ceiling)
- `growth_interval`: Steps before increasing scale
- `growth_factor`: Multiplier for scale growth
- `backoff_factor`: Multiplier for scale backoff (on overflow)
- `target_dtype`: Target precision (Float16 or BFloat16)
- `fp32_operations`: Operations that must use FP32

---

### `DeterministicConfig`

Configuration for deterministic execution.

```rust
pub struct DeterministicConfig {
    pub mode: DeterministicMode,
    pub global_seed: Option<u64>,
    pub operation_seeds: bool,
    pub cudnn_deterministic: bool,
}
```

**Fields**:
- `mode`: Determinism level (Strict, Relaxed, Best Effort)
- `global_seed`: Global random seed
- `operation_seeds`: Enable per-operation seeding
- `cudnn_deterministic`: Force deterministic cuDNN operations

---

## Examples

### Basic Usage

```rust
use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create tape
    let tape = GradientTape::new();

    // Watch inputs
    let x = tape.watch(Tensor::ones(&[3]));
    let w = tape.watch(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?);

    // Forward pass
    let y = x.mul(&w)?;
    let loss = y.sum(Some(&[0]), false)?;

    // Backward pass
    let grads = tape.gradient(&[loss], &[x.clone(), w.clone()])?;

    println!("∂loss/∂x: {:?}", grads[0].as_ref().unwrap());
    println!("∂loss/∂w: {:?}", grads[1].as_ref().unwrap());

    Ok(())
}
```

### With Checkpointing

```rust
let config = CheckpointConfig {
    strategy: CheckpointStrategy::Auto,
    memory_budget_mb: Some(4096),
    ..Default::default()
};
let tape = GradientTape::with_checkpoint_config(config);

// Deep network forward pass
for layer in model.layers() {
    hidden = layer.forward(&hidden)?;
}

// Backward pass with automatic checkpointing
let grads = tape.gradient(&[loss], &parameters)?;
```

### With Mixed Precision

```rust
let amp_config = AMPConfig::default();
let tape = GradientTape::with_amp_config(amp_config);
let mut amp_policy = AMPPolicy::new(amp_config);

// Forward with FP16
let output = model.forward(&input)?;
let loss = compute_loss(&output)?;

// Scale loss
let scaled_loss = amp_policy.scale_loss(&loss)?;

// Backward
let mut grads = tape.gradient(&[scaled_loss], &parameters)?;

// Unscale and check
if amp_policy.unscale_and_check(&mut grads)? {
    optimizer.step(&grads)?;
}
```

---

## Performance Considerations

### Memory Usage

- **Tape overhead**: ~40 bytes per operation
- **Activations**: Depends on model depth and batch size
- **Gradients**: Same size as parameters

**Optimization**: Use checkpointing to reduce activation memory by 50-90%.

### Computational Overhead

- **Recording**: ~5-10% overhead vs. no tracking
- **Backward pass**: ~1-2× forward pass cost
- **Total**: ~2-3× cost of forward-only inference

**Optimization**: Use mixed precision for 2× speedup.

---

## Thread Safety

`GradientTape` is **not thread-safe**. Each thread should have its own tape instance.

```rust
// ✅ GOOD: Separate tapes per thread
use std::thread;

let handles: Vec<_> = (0..num_threads)
    .map(|_| {
        thread::spawn(|| {
            let tape = GradientTape::new();
            // Use tape in this thread
        })
    })
    .collect();

// ❌ BAD: Shared tape across threads
let tape = Arc::new(Mutex::new(GradientTape::new()));
// Will cause data races!
```

---

## Limitations

1. **Single-use**: Tape is consumed by `gradient()` call
2. **Scalar outputs**: `gradient()` requires scalar or reduced outputs
3. **Dynamic graphs only**: No graph optimization passes (yet)
4. **CPU/GPU sync**: May require synchronization between devices

---

## See Also

- [`TrackedTensor`](./tracked_tensor.md) - Gradient-enabled tensor wrapper
- [`Operations`](./operations.md) - Supported differentiable operations
- [Gradient Computation Concepts](../concepts/gradient_computation.md)
- [Performance Optimization Guide](../../PERFORMANCE_GUIDE.md)

---

**Last Updated**: February 6, 2026
**Author**: COOLJAPAN OU (Team KitaSan)
