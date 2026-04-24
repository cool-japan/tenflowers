# Migration from TensorFlow to TenfloweRS

This guide helps TensorFlow users migrate their code to TenfloweRS. Each section presents
a side-by-side comparison of a common pattern. See the
[TensorFlow to TenfloweRS Mapping](../../README.md#tensorflow-to-tenflowers-mapping) table
in the README for a quick symbol-level reference.

**Key conceptual differences to keep in mind:**

- **Ownership vs garbage collection**: Rust's ownership model replaces Python's GC. Tensors
  are moved or borrowed explicitly.
- **Row-major (NCHW) vs channel-last (NHWC)**: TenfloweRS defaults to NCHW. TensorFlow
  (Keras) defaults to NHWC. Transpose your data or use `permute` when importing TF checkpoints.
- **Tape scope**: TenfloweRS `GradientTape` is an explicit Rust value with a defined lifetime,
  not a Python context manager. Create a new tape for each forward pass.
- **`Dataset` trait vs `tf.data`**: TenfloweRS `Dataset<T>` is a Rust trait; compose
  transforms with method chaining rather than graph-building.
- **Error handling**: All fallible operations return `Result<T>`, propagated with `?`. There
  are no silent Python exceptions to catch.

---

## 1. Simple MNIST Classification

**TensorFlow (Python)**
```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

**TenfloweRS (Rust)**
```rust,no_run
use tenflowers_neural::{Sequential, Dense, layers::Flatten, optimizers::Adam};
use tenflowers_neural::loss::SparseCategoricalCrossentropy;
use tenflowers_neural::metrics::Accuracy;

let mut model = Sequential::new(vec![
    Box::new(Flatten::new()),
    Box::new(Dense::new(128, true).with_activation("relu")),
    Box::new(Dense::new(10, true).with_activation("softmax")),
]);
model.compile(
    Adam::new(0.001),
    SparseCategoricalCrossentropy::new(),
    vec![Box::new(Accuracy::new())],
).unwrap();
model.fit(&train_dataset, 5, 32, None).unwrap();
```

---

## 2. CNN Image Classification

**TensorFlow (Python)**
```python
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

**TenfloweRS (Rust)**
```rust,no_run
use tenflowers_neural::{Sequential, Dense, Conv2D};
use tenflowers_neural::layers::{MaxPool2D, GlobalAveragePooling2D};
use tenflowers_neural::optimizers::Adam;
use tenflowers_neural::loss::SparseCategoricalCrossentropy;

// Note: TenfloweRS uses NCHW; TensorFlow Keras defaults to NHWC.
// Input shape is (batch, channels, height, width) = (N, 3, 32, 32).
let mut model = Sequential::new(vec![
    Box::new(Conv2D::new(32, (3, 3)).with_activation("relu")),
    Box::new(MaxPool2D::new((2, 2))),
    Box::new(Conv2D::new(64, (3, 3)).with_activation("relu")),
    Box::new(GlobalAveragePooling2D::new()),
    Box::new(Dense::new(10, true).with_activation("softmax")),
]);
model.compile(
    Adam::new(0.001),
    SparseCategoricalCrossentropy::new(),
    vec![],
).unwrap();
```

---

## 3. Transformer Encoder Layer

**TensorFlow (Python)**
```python
from tensorflow.keras.layers import MultiHeadAttention, Dense, LayerNormalization

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn  = keras.Sequential([Dense(dff, activation='relu'), Dense(d_model)])
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
    def call(self, x, training=False):
        attn_out = self.attn(x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)
```

**TenfloweRS (Rust)**
```rust,no_run
use tenflowers_neural::{Sequential, Dense};
use tenflowers_neural::attention::MultiHeadAttention;
use tenflowers_neural::layers::{LayerNorm, Residual};

// TransformerEncoder bundles self-attention + FFN + layer norms.
use tenflowers_neural::transformers::TransformerEncoderLayer;

let encoder = TransformerEncoderLayer::new(
    /* d_model */ 512,
    /* num_heads */ 8,
    /* d_ff */ 2048,
    /* dropout */ 0.1,
);
// Forward pass:
// let output = encoder.forward(&input_tensor).unwrap();
```

---

## 4. Custom Training Loop

**TensorFlow (Python)**
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

for epoch in range(5):
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss   = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

**TenfloweRS (Rust)**
```rust,no_run
use tenflowers_autograd::GradientTape;
use tenflowers_neural::optimizers::Adam;
use tenflowers_neural::loss::SparseCategoricalCrossentropy;

let mut optimizer = Adam::new(1e-3);
let loss_fn = SparseCategoricalCrossentropy::new();

for _epoch in 0..5 {
    for (x_batch, y_batch) in train_loader.iter() {
        let tape   = GradientTape::new();
        let logits = model.forward(&x_batch).unwrap();
        let loss   = loss_fn.compute(&logits, &y_batch).unwrap();
        let grads  = tape.gradient(&loss, model.parameters()).unwrap();
        optimizer.update(model.parameters_mut(), &grads).unwrap();
    }
}
```

Differences: a fresh `GradientTape` is created each iteration (no `with` context manager).
The tape records operations while it is alive. `gradient` consumes the tape.

---

## 5. tf.data Pipeline to DataLoader

**TensorFlow (Python)**
```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

for x_batch, y_batch in dataset:
    # training step
    pass
```

**TenfloweRS (Rust)**
```rust,no_run
use tenflowers_dataset::{TensorSliceDataset, DataLoader};
use tenflowers_core::Tensor;

// images: Tensor<f32>, labels: Tensor<i64>
let dataset = TensorSliceDataset::new(images, labels).unwrap();
let loader = DataLoader::new(dataset)
    .shuffle(true)
    .batch_size(32)
    .prefetch(4)
    .num_workers(2);

for (x_batch, y_batch) in loader.iter() {
    // training step
}
```

Key differences:
- `shuffle(buffer_size)` vs `shuffle(true)` — TenfloweRS shuffles the full index permutation by default.
- `prefetch(tf.data.AUTOTUNE)` vs `.prefetch(n)` — TenfloweRS uses a fixed prefetch depth
  (adaptive prefetch policy available with the `adaptive-prefetch` feature).
- TenfloweRS uses Rayon threads via `.num_workers(n)`, not Python multiprocessing.

---

## Key Differences Summary

1. **Ownership vs GC**: Python objects are reference-counted with a GC; Rust tensors follow
   move semantics. When sharing a tensor, use `.clone()` or pass by reference (`&tensor`).

2. **NCHW vs NHWC**: TenfloweRS defaults to channel-first (`N, C, H, W`). TensorFlow/Keras
   defaults to channel-last (`N, H, W, C`). Use `.permute(&[0, 3, 1, 2])` when converting
   TF checkpoints.

3. **Explicit gradient tape**: In TF, `with tf.GradientTape() as tape:` is a context manager
   that auto-closes. In TenfloweRS, `GradientTape::new()` is a plain Rust struct; the tape
   records operations until it is passed to `.gradient()` or dropped.

4. **`Dataset` trait vs `tf.data.Dataset`**: `tf.data` builds a lazy graph; TenfloweRS
   `Dataset<T>` is an iterator-based Rust trait. Transforms are chained with method calls at
   construction time, not deferred graph operations.

5. **All errors are `Result<T>`**: TensorFlow raises Python exceptions; TenfloweRS returns
   `Result<T, TensorError>`. Propagate errors with `?` or handle them explicitly with `match`.

---

Back to [README](../../README.md) | [Quick Reference](QUICK_REFERENCE.md) | [Troubleshooting](TROUBLESHOOTING.md)
