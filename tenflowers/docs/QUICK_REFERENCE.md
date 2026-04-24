# TenfloweRS Quick Reference

> Generated for v0.1.0. APIs marked with `[planned]` are scheduled for a future release.

## Tensor Creation

```rust,no_run
use tenflowers::prelude::*;

let zeros = Tensor::<f32>::zeros(&[2, 3]);
let ones  = Tensor::<f32>::ones(&[4]);
let randn = Tensor::<f32>::randn(&[3, 3]);
let from_data = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
```

Construct tensors from shapes or flat `Vec<T>`. Shape is always `&[usize]` (row-major).

## Math Operations

```rust,no_run
use tenflowers::prelude::*;

let a = Tensor::<f32>::ones(&[2, 3]);
let b = Tensor::<f32>::ones(&[2, 3]);
let c = a.add(&b).unwrap();
let d = a.matmul(&b.transpose().unwrap()).unwrap();
let e = a.pow(2.0).unwrap();
```

Element-wise ops (`add`, `sub`, `mul`, `div`), matrix multiply (`matmul`), and scalar ops (`pow`, `exp`, `log`) all return `Result<Tensor<T>>`.

## Autograd

```rust,no_run
use tenflowers_autograd::GradientTape;
use tenflowers_core::Tensor;

let tape = GradientTape::new();
let x = Tensor::<f32>::from_vec(vec![2.0, 3.0], &[2]).unwrap();
let y = tape.watch(x.clone());
let z = y.pow(2.0).unwrap();
let grads = tape.gradient(&z, &[&x]).unwrap();
```

Wrap operations in a `GradientTape` scope; call `gradient` to compute derivatives. The tape records only inside its scope.

## Neural Layers

```rust,no_run
use tenflowers_neural::{Sequential, Dense, Dropout, layers::BatchNorm};

let model = Sequential::new(vec![
    Box::new(Dense::new(128, true).with_activation("relu")),
    Box::new(BatchNorm::new(128)),
    Box::new(Dropout::new(0.3)),
    Box::new(Dense::new(10, true).with_activation("softmax")),
]);
```

Compose layers into a `Sequential`. Each layer implements the `Layer` trait. Add activations inline with `.with_activation("relu")`.

## Optimizers

```rust,no_run
use tenflowers_neural::optimizers::{Adam, SGD};

let adam = Adam::new(0.001);
let sgd  = SGD::new(0.01).with_momentum(0.9);
```

Use `Adam` for most tasks. `SGD` with momentum is preferred for fine-tuning. All optimizers implement `Optimizer`.

## Data Loading

```rust,no_run
use tenflowers_dataset::{Dataset, DataLoader};

let loader = DataLoader::new(dataset)
    .batch_size(32)
    .shuffle(true)
    .num_workers(4);

for (inputs, labels) in loader.iter() {
    // training step
}
```

Build a `DataLoader` from any type implementing the `Dataset` trait. Set batch size, shuffling, and worker count at construction time.

## Training Loop

```rust,no_run
use tenflowers_neural::{Sequential, Dense, optimizers::Adam, loss::CrossEntropyLoss};
use tenflowers_autograd::GradientTape;

let mut model = Sequential::new(vec![
    Box::new(Dense::new(64, true).with_activation("relu")),
    Box::new(Dense::new(10, true)),
]);
let mut optimizer = Adam::new(0.001);
let loss_fn = CrossEntropyLoss::new();

for epoch in 0..10 {
    for (x, y) in train_loader.iter() {
        let tape = GradientTape::new();
        let logits = model.forward(&x).unwrap();
        let loss   = loss_fn.compute(&logits, &y).unwrap();
        let grads  = tape.gradient(&loss, model.parameters()).unwrap();
        optimizer.update(model.parameters_mut(), &grads).unwrap();
    }
}
```

The standard loop: create tape, forward pass, compute loss, differentiate, update weights.

## Save & Load

```rust,no_run
// Feature-gated: requires `serialize` feature
#[cfg(feature = "serialize")]
{
    tenflowers::io::save_model("model.oxicode", &model).unwrap();
    let loaded = tenflowers::io::load_model::<Sequential>("model.oxicode").unwrap();
}
```

Model serialization uses `oxicode` (Pure Rust) by default. Enable the `serialize` feature in `Cargo.toml`. [planned in tenflowers 0.1.1]

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Rust standard library (required) |
| `parallel` | yes | Rayon parallelism for data loading and ops |
| `gpu` | no | Generic GPU support via WGPU |
| `cuda` | no | NVIDIA CUDA backend |
| `metal` | no | Apple Metal backend |
| `blas-oxiblas` | no | Pure Rust BLAS via OxiBLAS (recommended) |
| `simd` | no | SIMD vectorization (AVX2/NEON) |
| `serialize` | no | Model and tensor serialization via oxicode |
| `onnx` | no | ONNX model import/export |
| `full` | no | All non-GPU features (blas-oxiblas, simd, serialize, onnx) |

See the main [README](../../README.md#feature-flags) for the complete feature matrix and mutual-exclusion rules.

## Device Placement

```rust,no_run
use tenflowers_core::{Device};

// CPU is the default
let t_cpu = Tensor::<f32>::ones(&[2, 2]);

// Move to GPU (requires `gpu` feature)
#[cfg(feature = "gpu")]
let t_gpu = t_cpu.to(Device::Gpu(0)).unwrap();
```

All tensors default to `Device::Cpu`. Use `.to(device)` to move across devices. Operations require operands on the same device.

## Cross References

- [Migration from TensorFlow](MIGRATION_FROM_TENSORFLOW.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [README](../../README.md)
