# TenfloweRS Troubleshooting Guide

A reference for the most common build, runtime, and correctness issues encountered with
TenfloweRS. Each entry provides the symptom, root cause, and a concrete fix.

---

## 1. Link error when `blas-openblas` is enabled

**Symptom**
```
error: linking with `cc` failed: exit status 1
  = note: ld: library not found for -lopenblas
```

**Root cause**
`blas-openblas` requires a system-installed OpenBLAS shared library (`libopenblas.so` /
`libopenblas.dylib`). When the library is absent or mislocated, the linker fails.

**Fix**
Use `blas-oxiblas` instead — TenfloweRS's recommended Pure Rust BLAS backend with no
system library dependency:

```toml
[dependencies]
tenflowers-core = { version = "0.1.0", features = ["blas-oxiblas"] }
```

`blas-openblas` is retained for advanced users who require identical numerical results to
reference BLAS on a pre-configured system. See the COOLJAPAN Pure Rust Policy.

---

## 2. `gpu` feature enabled but no GPU found at runtime

**Symptom**
```
TensorError::UnsupportedDevice("No GPU adapter found")
```
or a panic on `Device::Gpu(0)` when iterating adapters.

**Root cause**
The `gpu` feature enables the WGPU backend but the runtime host may lack a compatible
GPU adapter (e.g. headless CI, CPU-only cloud instances, or missing drivers).

**Fix**
Add an explicit availability check and fall back to CPU:

```rust,no_run
use tenflowers_core::Device;

let device = if Device::is_gpu_available() {
    Device::Gpu(0)
} else {
    eprintln!("No GPU available, falling back to CPU.");
    Device::Cpu
};
```

For CI or environments without a GPU, build without the `gpu` feature to avoid the overhead
of adapter enumeration entirely.

---

## 3. DType mismatch error during loss computation

**Symptom**
```
TensorError::DeviceMismatch  -- or --
thread 'main' panicked at 'dtype mismatch: expected f32, found f64'
```

**Root cause**
TenfloweRS operations require both operands to share the same dtype. Mixing `f32` weights
with `f64` inputs (or vice versa) produces a dtype-mismatch error at the loss call site.

**Fix**
Cast tensors to a common dtype before passing them to the model:

```rust,no_run
// Cast input to f32 to match model weights
let input_f32 = input_f64.cast::<f32>().unwrap();
let logits = model.forward(&input_f32).unwrap();
```

Or construct all tensors with an explicit dtype from the start:

```rust,no_run
let x = Tensor::<f32>::from_vec(data, &[batch, features]).unwrap();
```

---

## 4. Rank mismatch in a Sequential chain

**Symptom**
```
TensorError::ShapeMismatch { expected: [batch, 128], actual: [batch, 32, 4, 4] }
```

**Root cause**
A `Dense` layer received a 4-D tensor (e.g. the output of a `Conv2D`) without an intervening
`Flatten` or `GlobalAveragePooling2D`. Each layer's expected input rank must match its actual
input.

**Fix**
Insert a `Flatten` between the convolutional stack and the first `Dense` layer:

```rust,no_run
use tenflowers_neural::layers::Flatten;

let model = Sequential::new(vec![
    Box::new(Conv2D::new(64, (3, 3)).with_activation("relu")),
    Box::new(Flatten::new()),          // <-- required
    Box::new(Dense::new(128, true).with_activation("relu")),
    Box::new(Dense::new(10, true)),
]);
```

Add `.assert_input_shape()` calls during debugging to catch the mismatch at construction
time rather than at the first forward pass.

---

## 5. Python wheel fails to import: GLIBC not found

**Symptom**
```
ImportError: /lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.29' not found
```

**Root cause**
The published wheel was compiled against a newer glibc than the host system provides.
PyO3-based wheels must target `manylinux_2_17` (glibc 2.17) or `manylinux_2_28` for broad
compatibility.

**Fix**
Build the wheel inside the official `manylinux_2_17` container:

```bash
docker run --rm -v $(pwd):/io \
  ghcr.io/rust-cross/maturin:latest \
  build --release --compatibility manylinux_2_17
```

Or, if you are installing from source:

```bash
pip install maturin
maturin develop --release
```

---

## 6. `cargo build` takes more than 5 minutes on the first build

**Symptom**
First-time workspace build stalls. `tenflowers-neural` has 150+ modules; full workspace
compile can exceed 5 minutes on low-core machines.

**Root cause**
The workspace compiles all 6 crates and ~150 neural domain modules simultaneously, which
saturates memory on machines with fewer than 8 logical cores.

**Fix**
Build crates incrementally, starting with the lightest:

```bash
cargo build -p tenflowers-core
cargo build -p tenflowers-autograd
cargo build -p tenflowers-dataset
cargo build -p tenflowers-neural    # the long step
cargo build -p tenflowers
```

Use `cargo nextest run -p tenflowers-core` for fast targeted tests while `neural` compiles
in the background. On CI, enable sccache or use a distributed build cache.

---

## 7. `features` not found during `cargo add`

**Symptom**
```
error: package `tenflowers` not found in registry
```
or features appear empty when installing from a local path checkout.

**Root cause**
If the crate is not yet published to crates.io (e.g. you are working on a pre-release branch),
`cargo add` cannot locate it by name.

**Fix**
Use a path dependency directly in your `Cargo.toml`:

```toml
[dependencies]
tenflowers = { path = "../path/to/tenflowers/tenflowers", features = ["gpu"] }
```

For workspace builds, add `tenflowers` to the workspace members in the root `Cargo.toml`
and depend on it with `version.workspace = true`.

---

## 8. NaN loss after a few training steps

**Symptom**
Loss prints `NaN` or `inf` after one to three gradient updates.

**Root cause**
Common causes:
- Learning rate too high (exploding gradients).
- Missing gradient clipping.
- Division by zero in a softmax or log-softmax with degenerate inputs.
- Uninitialized or saturated weights.

**Fix**

1. Lower the learning rate by a factor of 10:
   ```rust,no_run
   let optimizer = Adam::new(1e-4);  // instead of 1e-3
   ```

2. Add gradient clipping before the optimizer step:
   ```rust,no_run
   let clipped_grads = clip_gradients(&grads, /* max_norm */ 1.0).unwrap();
   optimizer.update(model.parameters_mut(), &clipped_grads).unwrap();
   ```

3. Check inputs for NaN before the forward pass:
   ```rust,no_run
   assert!(x_batch.is_finite().all(), "NaN in input batch");
   ```

4. Verify weight initialization — default `Dense` uses Glorot/Xavier uniform, which is
   stable for most architectures. For very deep networks, consider He initialization.

---

## 9. `cargo check` passes but `cargo test` OOM-crashes

**Symptom**
`cargo test --workspace` causes the linker or `rustc` to be killed (OOM) on machines with
less than 16 GB RAM.

**Root cause**
Linking the full workspace binary (all 6 crates, all features, all test binaries) can require
8+ GB of RAM in debug mode.

**Fix**
Use `cargo nextest run` which compiles tests individually and avoids one monolithic link:

```bash
cargo nextest run --workspace --all-features
```

Or limit the scope to one crate at a time:

```bash
cargo test -p tenflowers-core
cargo test -p tenflowers-autograd
```

---

## 10. `experimental` module not found at compile time

**Symptom**
```
error[E0433]: failed to resolve: use of undeclared crate or module `experimental`
```

**Root cause**
The `experimental` module is feature-gated. It is not compiled into the default build.

**Fix**
Enable the `experimental` feature in `Cargo.toml`:

```toml
[dependencies]
tenflowers = { version = "0.1.0", features = ["experimental"] }
```

Note: symbols in `tenflowers::experimental` are explicitly unstable and may change in any
minor release. They are excluded from the prelude stability guarantees.

---

Back to [README](../../README.md) | [Quick Reference](QUICK_REFERENCE.md) | [Migration Guide](MIGRATION_FROM_TENSORFLOW.md)
