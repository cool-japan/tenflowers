# TenfloweRS Autograd Quick Start Guide

Get started with TenfloweRS Autograd in 5 minutes!

> **📚 For in-depth information**, see the comprehensive [AUTOGRAD_GUIDE.md](./AUTOGRAD_GUIDE.md)

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
tenflowers-autograd = "0.1.0-alpha.2"
tenflowers-core = "0.1.0-alpha.2"
scirs2-autograd = "0.1.0-rc.1"
```

---

## Your First Gradient

```rust
use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;
use scirs2_autograd::ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create gradient tape
    let tape = GradientTape::new();

    // Watch input tensor
    let x = tape.watch(Tensor::from_array(
        array![1.0f32, 2.0, 3.0].into_dyn()
    ));

    // Forward pass: y = x²
    let y = x.mul(&x)?;

    // Compute gradients: dy/dx = 2x
    let grads = tape.gradient(&[y], &[x])?;

    println!("Gradients: {:?}", grads[0].as_ref().unwrap().as_slice().unwrap());
    // Output: [2.0, 4.0, 6.0]

    Ok(())
}
```

---

## Common Patterns

### Pattern 1: Simple Function Gradient

```rust
// Compute gradient of f(x) = x³ + 2x² + x
let tape = GradientTape::new();
let x = tape.watch(Tensor::from_array(array![2.0f32].into_dyn()));

let x2 = x.mul(&x)?;
let x3 = x2.mul(&x)?;

let term1 = x3;
let term2 = x2.mul(&Tensor::scalar(2.0))?;
let term3 = x;

let y = term1.add(&term2)?.add(&term3)?;

let grad = tape.gradient(&[y], &[x])?[0].clone().unwrap();

// f'(x) = 3x² + 4x + 1
// At x=2: f'(2) = 3(4) + 4(2) + 1 = 21
println!("Gradient at x=2: {}", grad.as_slice().unwrap()[0]);
```

### Pattern 2: Multi-Input Gradient

```rust
let tape = GradientTape::new();
let x = tape.watch(Tensor::from_array(array![1.0f32, 2.0].into_dyn()));
let y = tape.watch(Tensor::from_array(array![3.0f32, 4.0].into_dyn()));

// z = x² + y²
let z = x.mul(&x)?.add(&y.mul(&y)?)?;

let grads = tape.gradient(&[z], &[x, y])?;

println!("∂z/∂x: {:?}", grads[0].as_ref().unwrap().as_slice().unwrap());  // [2.0, 4.0]
println!("∂z/∂y: {:?}", grads[1].as_ref().unwrap().as_slice().unwrap());  // [6.0, 8.0]
```

### Pattern 3: Neural Network Layer

```rust
let tape = GradientTape::new();

// Input: batch_size=2, features=3
let x = tape.watch(Tensor::from_array(
    array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn()
));

// Weights: 3 inputs → 2 outputs
let w = tape.watch(Tensor::from_array(
    array![[0.1f32, 0.2], [0.3, 0.4], [0.5, 0.6]].into_dyn()
));

// Forward: y = xW
let y = x.matmul(&w)?;

// Compute gradients
let grads = tape.gradient(&[y], &[x, w])?;

println!("Input gradient shape: {:?}", grads[0].as_ref().unwrap().shape().dims());
println!("Weight gradient shape: {:?}", grads[1].as_ref().unwrap().shape().dims());
```

---

## Training Loop Template

```rust
use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;

fn train(
    mut params: Tensor<f32>,
    data: &[(Tensor<f32>, Tensor<f32>)],
    epochs: usize,
) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
    let learning_rate = 0.01f32;

    for epoch in 0..epochs {
        let mut total_loss = 0.0f32;

        for (inputs, targets) in data {
            // Create fresh tape for each batch
            let tape = GradientTape::new();
            let params_tracked = tape.watch(params.clone());

            // Forward pass
            let predictions = model_forward(&params_tracked, inputs)?;
            let loss = mse_loss(&predictions, targets)?;

            // Track loss value
            total_loss += loss.tensor.as_slice().unwrap()[0];

            // Backward pass
            let grads = tape.gradient(&[loss], &[params_tracked])?;
            let grad = grads[0].as_ref().unwrap();

            // Update parameters: params -= learning_rate * grad
            params = params.sub(&grad.mul(&Tensor::scalar(learning_rate))?)?;

        }  // tape dropped here, memory freed

        println!("Epoch {}: Loss = {:.4}", epoch, total_loss / data.len() as f32);
    }

    Ok(params)
}

fn model_forward(
    params: &TrackedTensor<f32>,
    inputs: &Tensor<f32>,
) -> Result<TrackedTensor<f32>> {
    // Your model implementation
    // e.g., linear: y = xW + b
    params.mul(&inputs)
}

fn mse_loss(
    predictions: &TrackedTensor<f32>,
    targets: &Tensor<f32>,
) -> Result<TrackedTensor<f32>> {
    let diff = predictions.sub(&targets)?;
    let squared = diff.mul(&diff)?;
    squared.mean()
}
```

---

## Advanced Features

> **💡 Tip**: For comprehensive performance optimization strategies, see [PERFORMANCE_GUIDE.md](./PERFORMANCE_GUIDE.md)

### Mixed Precision Training

```rust
use tenflowers_autograd::{AMPConfig, AMPPolicy};

let amp_config = AMPConfig {
    enabled: true,
    initial_scale: 65536.0,
    ..Default::default()
};

let mut amp_policy = AMPPolicy::new(amp_config);

// In training loop
let scaled_loss = amp_policy.scale_loss(&loss)?;
let mut grads = tape.gradient(&[scaled_loss], &params)?;

if amp_policy.unscale_and_check(&mut grads)? {
    // Gradients are valid, update parameters
    optimizer.step(&grads)?;
    amp_policy.update_scale(false);
}
```

> **📖 See also**: [examples/mixed_precision_example.rs](./examples/mixed_precision_example.rs) for complete training workflows

### Gradient Clipping

```rust
use tenflowers_autograd::clip_by_global_norm;

// Compute gradients
let grads = tape.gradient(&[loss], &params)?;

// Extract tensors from Option<Tensor>
let grad_tensors: Vec<Tensor<f32>> = grads
    .into_iter()
    .filter_map(|g| g)
    .collect();

// Clip to max norm
let clipped_grads = clip_by_global_norm(&grad_tensors, 1.0)?;

// Use clipped gradients
optimizer.step(&clipped_grads)?;
```

### Gradient Accumulation

```rust
use tenflowers_autograd::GradientAccumulator;

let mut accumulator = GradientAccumulator::new();
let accumulation_steps = 4;

for step in 0..accumulation_steps {
    let tape = GradientTape::new();
    let loss = model.forward(&micro_batch)?;
    let grads = tape.gradient(&[loss], &params)?;

    accumulator.accumulate(&grads)?;
}

// Get averaged gradients
let avg_grads = accumulator.get_averaged_gradients()?;
optimizer.step(&avg_grads)?;

accumulator.clear();
```

---

## Debugging Gradients

> **💡 Tip**: For comprehensive testing strategies, see [TESTING_GUIDE.md](./TESTING_GUIDE.md)

### Check Gradient Correctness

```rust
use tenflowers_autograd::numerical_checker::NumericalChecker;

let mut checker = NumericalChecker::default();

let x = Tensor::from_array(array![1.0f32, 2.0, 3.0].into_dyn());

// Your function
let f = |tensor: &Tensor<f32>| -> Result<Tensor<f32>> {
    tensor.mul(tensor)
};

// Compute numerical gradient
let numerical = checker.compute_numerical_gradient(&x, f, 1e-6)?;

// Compute analytical gradient
let tape = GradientTape::new();
let x_tracked = tape.watch(x.clone());
let y = x_tracked.mul(&x_tracked)?;
let analytical = tape.gradient(&[y], &[x_tracked])?[0].clone().unwrap();

// Compare
let result = checker.compare_gradients(&analytical, &numerical)?;

if result.is_valid {
    println!("✓ Gradient check passed!");
} else {
    println!("✗ Gradient check failed!");
    println!("  Max error: {:.2e}", result.max_error);
}
```

> **📖 See also**: [examples/numerical_gradient_validation_example.rs](./examples/numerical_gradient_validation_example.rs) for 6 comprehensive validation examples

### Visualize Gradient Flow

```rust
use tenflowers_autograd::GradientFlowVisualizer;

let mut visualizer = GradientFlowVisualizer::new();
visualizer.analyze_flow(&tape, &loss, &params)?;

let health = visualizer.get_health_summary()?;
println!("Gradient health: {}", health);

if let Some(analysis) = visualizer.flow_analysis() {
    println!("Health score: {:.1}/100", analysis.health_score);

    for issue in &analysis.issues {
        println!("⚠️  {:?}: {}", issue.issue_type, issue.description);
    }
}
```

---

## Performance Tips

### 1. Reuse Tensors

```rust
// ❌ Slow: Creates new tensors
for _ in 0..1000 {
    gradient = gradient.mul(&decay)?;
}

// ✅ Fast: Reuses memory
for _ in 0..1000 {
    gradient.mul_inplace(&decay)?;
}
```

### 2. Use Larger Batches

```rust
// ❌ Slow: Small batch (underutilizes GPU)
let batch_size = 4;

// ✅ Fast: Larger batch (better GPU utilization)
let batch_size = 32;

// ✅ Memory-constrained: Use gradient accumulation
let micro_batch_size = 8;
let accumulation_steps = 4;
// Effective batch size = 32
```

### 3. Enable Mixed Precision

```rust
// 2x faster on modern GPUs, 50% memory reduction
let amp_config = AMPConfig {
    enabled: true,
    initial_scale: 65536.0,
    ..Default::default()
};
```

---

## Common Errors

### Error: "Gradient is None"

```rust
// ❌ Problem: Tensor not watched
let x = Tensor::ones(&[3]);
let y = x.mul(&x)?;
let grads = tape.gradient(&[y], &[x])?;  // Error: x not tracked

// ✅ Solution: Watch the tensor
let tape = GradientTape::new();
let x = tape.watch(Tensor::ones(&[3]));
let y = x.mul(&x)?;
let grads = tape.gradient(&[y], &[x])?;  // OK
```

### Error: "Shape Mismatch"

```rust
// ❌ Problem: Incompatible shapes
let x = Tensor::ones(&[2, 3]);
let y = Tensor::ones(&[3, 4]);
let z = x.add(&y)?;  // Error: can't add [2,3] and [3,4]

// ✅ Solution: Use compatible operations
let z = x.matmul(&y)?;  // OK: [2,3] @ [3,4] = [2,4]
```

### Error: "Out of Memory"

```rust
// ❌ Problem: Too large batch or deep network
let batch_size = 128;
let model = VeryDeepNetwork::new(200_layers);

// ✅ Solution 1: Reduce batch size
let batch_size = 32;

// ✅ Solution 2: Use gradient accumulation
let micro_batch = 32;
let accumulation_steps = 4;

// ✅ Solution 3: Enable checkpointing
let policy = ActivationCheckpointPolicy::default()
    .with_memory_budget_mb(4096);
```

---

## Next Steps

### Essential Reading

1. **📖 Full Guide**: [AUTOGRAD_GUIDE.md](./AUTOGRAD_GUIDE.md) - Comprehensive autograd documentation
2. **⚡ Performance**: [PERFORMANCE_GUIDE.md](./PERFORMANCE_GUIDE.md) - Memory and compute optimization
3. **✅ Testing**: [TESTING_GUIDE.md](./TESTING_GUIDE.md) - Gradient validation strategies
4. **📊 Implementation Status**: [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) - Complete feature overview

### Key Examples

- **[numerical_gradient_validation_example.rs](./examples/numerical_gradient_validation_example.rs)** - 6 validation techniques
- **[second_order_derivatives_example.rs](./examples/second_order_derivatives_example.rs)** - Hessian, Jacobian, Newton's method
- **[mixed_precision_example.rs](./examples/mixed_precision_example.rs)** - AMP training workflows
- **[checkpointing_example.rs](./examples/checkpointing_example.rs)** - Memory optimization strategies
- **[advanced_optimization_techniques.rs](./examples/advanced_optimization_techniques.rs)** - Complete training patterns

---

## Quick Reference

### Core Operations

```rust
// Arithmetic
y = x.add(&other)?;
y = x.sub(&other)?;
y = x.mul(&other)?;
y = x.div(&other)?;

// Matrix
y = x.matmul(&other)?;
y = x.transpose()?;

// Activation
y = x.relu()?;
y = x.sigmoid()?;
y = x.tanh()?;

// Reduction
y = x.sum(None, false)?;
y = x.mean()?;
y = x.max()?;

// Shape
y = x.reshape(&new_shape)?;
y = x.squeeze()?;
y = x.unsqueeze(axis)?;
```

### Utilities

```rust
// Gradient clipping
grads = clip_by_global_norm(&grads, max_norm)?;
grads = clip_by_value(&grads, clip_value)?;

// Gradient scaling
grads = scale_gradients(&grads, scale_factor)?;

// Numerical validation
checker.compare_gradients(&analytical, &numerical)?;
```

---

## Help & Support

- **Issues**: [GitHub Issues](https://github.com/cool-japan/tenflowers/issues)
- **Documentation**: [API Docs](https://docs.rs/tenflowers-autograd)
- **Examples**: [examples/](./examples/)

---

**Happy Training! 🚀**
