# TenfloweRS Prelude — Stability Policy

The `tenflowers::prelude` module provides a curated set of re-exports that are
intended to cover the most common use cases for building and training models with
TenfloweRS.

## What is in the Prelude

The prelude re-exports a deliberately compact set of symbols from all TenfloweRS
subcrates:

- **Core types**: `Tensor`, `Device`, `DType`, `ops` module, `Shape`
- **Autograd**: `GradientTape`, `TrackedTensor`
- **Layers**: `Dense`, `Conv2D`, `BatchNorm`, `Dropout`, `MaxPool2D`
- **Attention / Transformer**: `MultiHeadAttention`, `RMSNorm`, `TransformerEncoder`, `TransformerDecoder`
- **Recurrent**: `RNN`, `GRU`, `LSTM`
- **Models**: `Sequential`, `Model` trait
- **Activations**: `ActivationFunction`
- **Optimizers**: `Adam`, `AdamW`, `SGD`, `Optimizer` trait, `ParameterGroup`
- **Loss functions**: `mse`, `binary_cross_entropy`, `categorical_cross_entropy`
- **Training**: `Trainer`, `quick_train`, `EarlyStopping`, `ModelCheckpoint`
- **Dataset**: `CsvDataset`, `CsvDatasetBuilder`, `DataLoader`, `DataLoaderBuilder`,
  `ImageFolderDataset`, `ImageFolderDatasetBuilder`, `RandomSampler`, `Dataset` trait
- **Error handling**: `Result`, `TensorError`
- **Type aliases**: `Tensor1D`, `Tensor2D`, `Tensor3D`, `Tensor4D`, `Vector`, `Matrix`,
  `BatchTensor`, and the `f32`/`f64` shorthand variants

## Stability Guarantees

### Additions (minor version)

New symbols **may be added** to the prelude in any minor version (`0.x.y → 0.x.(y+1)`
or `0.x → 0.(x+1)`). Additions are always backward-compatible since existing code
does not import the new symbols.

If an added symbol name clashes with a user's own definition when using glob
imports (`use tenflowers::prelude::*`), the user can resolve the ambiguity by
importing the prelude items explicitly instead of via glob.

### Removals (major version only)

Removing a symbol from the prelude is a **breaking change** and requires a major
version bump (`0.x → 1.0` or `1.x → 2.0`). No symbol present in the prelude at
a given major version will be removed before the next major version.

### Renames

Renaming a public item in the prelude is treated as a removal + addition and
therefore also requires a major version bump. A deprecation cycle (adding the old
name as a deprecated re-export alongside the new name) will be used where possible
before removal.

### Feature-gated symbols

Symbols that are only available under specific Cargo feature flags (e.g. `gpu`,
`cuda`, `onnx`, `serialize`) are **not covered** by the stability guarantees above
for builds that do not enable the corresponding feature. They follow the stability
of the underlying feature flag.

An `experimental` Cargo feature (when present) gates items explicitly marked
unstable. Items behind `experimental` may change or be removed in any version.

## 0.x Semantics

TenfloweRS is currently in the `0.x` version series. During this phase:

- The above guarantees are followed on a **best-effort basis**.
- Strict semver stability formally begins at version `1.0`.
- Users building on `0.x` are encouraged to pin minor versions in `Cargo.toml`
  (e.g. `tenflowers = "=0.1.0"`) if API stability is critical for their
  application.

## Out-of-Scope

The following are **not** covered by the prelude stability policy:

- Internal modules not re-exported by the prelude (e.g. `tenflowers::core::*`,
  `tenflowers::neural::*`).
- Items behind `#[doc(hidden)]`.
- Trait method signatures and implementations internal to subcrates.
- The order of items returned by iterators or methods not explicitly documented
  as ordered.

## Contact

Please open an issue on <https://github.com/cool-japan/tenflowers> if you
encounter an unexpected breaking change in the prelude.
