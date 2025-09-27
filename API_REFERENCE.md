# TenfloweRS API Reference

A comprehensive guide to the TenfloweRS API with detailed examples and usage patterns.

## Table of Contents

- [Core Module (`tenflowers-core`)](#core-module)
  - [Tensor Operations](#tensor-operations)
  - [Device Management](#device-management)
  - [Data Types](#data-types)
  - [Graph Operations](#graph-operations)
- [Neural Network Module (`tenflowers-neural`)](#neural-network-module)
  - [Layers](#layers)
  - [Models](#models)
  - [Optimizers](#optimizers)
  - [Loss Functions](#loss-functions)
  - [Metrics](#metrics)
- [Autograd Module (`tenflowers-autograd`)](#autograd-module)
  - [Gradient Computation](#gradient-computation)
  - [Gradient Tape](#gradient-tape)
  - [Higher-Order Derivatives](#higher-order-derivatives)
- [Dataset Module (`tenflowers-dataset`)](#dataset-module)
  - [Data Loading](#data-loading)
  - [Transformations](#transformations)
  - [Synthetic Data](#synthetic-data)
- [FFI Module (`tenflowers-ffi`)](#ffi-module)
  - [Python Bindings](#python-bindings)
  - [C API](#c-api)

---

## Core Module

The `tenflowers-core` module provides the fundamental tensor operations and device management.

### Tensor Operations

#### Basic Creation

```rust
use tenflowers_core::{Tensor, Device, DType};

// Create from data
let data = vec![1.0, 2.0, 3.0, 4.0];
let tensor = Tensor::from_vec(data, &[2, 2]);

// Create with specific device
let gpu_tensor = Tensor::zeros(&[3, 3], DType::F32, Device::Gpu(0))?;

// Create from ranges
let range_tensor = Tensor::arange(0.0, 10.0, 1.0)?;
let linspace_tensor = Tensor::linspace(0.0, 1.0, 100)?;

// Random tensors
let random_tensor = Tensor::randn(&[128, 784], DType::F32, Device::Cpu)?;
let uniform_tensor = Tensor::rand(&[64, 32], DType::F32, Device::Cpu)?;
```

#### Arithmetic Operations

```rust
use tenflowers_core::Tensor;

let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]);

// Element-wise operations
let sum = &a + &b;          // [5.0, 7.0, 9.0]
let diff = &a - &b;         // [-3.0, -3.0, -3.0]
let product = &a * &b;      // [4.0, 10.0, 18.0]
let quotient = &a / &b;     // [0.25, 0.4, 0.5]
let power = a.pow(&b)?;     // [1.0, 32.0, 729.0]

// Broadcasting
let scalar = Tensor::from_scalar(2.0);
let scaled = &a * &scalar;  // [2.0, 4.0, 6.0]
```

#### Linear Algebra

```rust
use tenflowers_core::Tensor;

// Matrix multiplication
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
let product = a.matmul(&b)?;

// Dot product
let vec1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
let vec2 = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]);
let dot = vec1.dot(&vec2)?;

// Advanced linear algebra
let matrix = Tensor::from_vec(vec![4.0, 2.0, 2.0, 1.0], &[2, 2]);
let inverse = matrix.inverse()?;
let determinant = matrix.det()?;
let eigenvalues = matrix.eigenvalues()?;

// Decompositions
let (u, s, v) = matrix.svd()?;
let (q, r) = matrix.qr()?;
let cholesky = matrix.cholesky()?;
```

#### Reduction Operations

```rust
use tenflowers_core::Tensor;

let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

// Global reductions
let sum = tensor.sum(None)?;           // Sum all elements
let mean = tensor.mean(None)?;         // Mean of all elements
let max = tensor.max(None)?;           // Maximum value
let min = tensor.min(None)?;           // Minimum value

// Axis-specific reductions
let sum_axis0 = tensor.sum(Some(&[0]))?;  // Sum along axis 0
let mean_axis1 = tensor.mean(Some(&[1]))?; // Mean along axis 1

// Argument operations
let argmax = tensor.argmax(Some(0))?;  // Index of maximum along axis 0
let argmin = tensor.argmin(Some(1))?;  // Index of minimum along axis 1

// Statistical operations
let variance = tensor.var(Some(&[0]))?;
let std_dev = tensor.std(Some(&[0]))?;
```

#### Tensor Manipulation

```rust
use tenflowers_core::Tensor;

let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

// Reshape
let reshaped = tensor.reshape(&[3, 2])?;
let flattened = tensor.reshape(&[-1])?;  // Flatten to 1D

// Transpose
let transposed = tensor.transpose()?;
let perm_transposed = tensor.transpose_axes(&[1, 0])?;

// Squeeze/Unsqueeze
let squeezed = tensor.squeeze(Some(&[0]))?;
let unsqueezed = tensor.unsqueeze(1)?;

// Slicing
let slice = tensor.slice(&[
    Some((0, 2)),  // First two rows
    Some((1, 3)),  // Columns 1-2
])?;

// Concatenation
let other = Tensor::ones(&[2, 3], DType::F32, Device::Cpu)?;
let concatenated = Tensor::concat(&[&tensor, &other], 0)?;

// Stacking
let stacked = Tensor::stack(&[&tensor, &other], 0)?;
```

#### Comparison and Logical Operations

```rust
use tenflowers_core::Tensor;

let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
let b = Tensor::from_vec(vec![2.0, 2.0, 1.0], &[3]);

// Comparison operations
let eq = a.eq(&b)?;      // [false, true, false]
let ne = a.ne(&b)?;      // [true, false, true]
let lt = a.lt(&b)?;      // [true, false, false]
let le = a.le(&b)?;      // [true, true, false]
let gt = a.gt(&b)?;      // [false, false, true]
let ge = a.ge(&b)?;      // [false, true, true]

// Logical operations (on boolean tensors)
let bool_a = a.gt(&Tensor::from_scalar(1.5))?;
let bool_b = b.lt(&Tensor::from_scalar(2.5))?;
let and_result = bool_a.and(&bool_b)?;
let or_result = bool_a.or(&bool_b)?;
let not_result = bool_a.not()?;
```

### Device Management

#### Device Placement

```rust
use tenflowers_core::{Device, Tensor};

// CPU device
let cpu_device = Device::Cpu;
let cpu_tensor = Tensor::zeros(&[10, 10], DType::F32, cpu_device)?;

// GPU device
let gpu_device = Device::Gpu(0);  // First GPU
let gpu_tensor = Tensor::zeros(&[10, 10], DType::F32, gpu_device)?;

// Device transfer
let gpu_tensor = cpu_tensor.to_device(&gpu_device)?;
let cpu_tensor = gpu_tensor.to_device(&cpu_device)?;

// Check device
println!("Tensor is on GPU: {}", tensor.device().is_gpu());
println!("Tensor is on CPU: {}", tensor.device().is_cpu());
```

#### Device Context

```rust
use tenflowers_core::{Device, DeviceContext};

// Create device context
let mut ctx = DeviceContext::new(Device::Gpu(0))?;

// Set default device
ctx.set_default_device(Device::Gpu(0));

// Get device information
let device_count = ctx.device_count()?;
let memory_info = ctx.memory_info()?;
```

### Data Types

#### Supported Types

```rust
use tenflowers_core::{DType, Tensor};

// Floating point
let f32_tensor = Tensor::zeros(&[5], DType::F32, Device::Cpu)?;
let f64_tensor = Tensor::zeros(&[5], DType::F64, Device::Cpu)?;
let f16_tensor = Tensor::zeros(&[5], DType::F16, Device::Cpu)?;  // Half precision

// Integer types
let i32_tensor = Tensor::zeros(&[5], DType::I32, Device::Cpu)?;
let i64_tensor = Tensor::zeros(&[5], DType::I64, Device::Cpu)?;
let u8_tensor = Tensor::zeros(&[5], DType::U8, Device::Cpu)?;

// Complex types
let c64_tensor = Tensor::zeros(&[5], DType::C64, Device::Cpu)?;
let c128_tensor = Tensor::zeros(&[5], DType::C128, Device::Cpu)?;

// Type conversion
let converted = f32_tensor.to_dtype(DType::F64)?;
```

#### Mixed Precision

```rust
use tenflowers_core::{MixedPrecisionConfig, AutocastContext};

// Configure mixed precision
let config = MixedPrecisionConfig::new()
    .with_enabled(true)
    .with_target_dtype(DType::F16);

// Use autocast context
let autocast = AutocastContext::new(config);
autocast.enable();

// Operations will automatically use FP16 where appropriate
let result = tensor_a.matmul(&tensor_b)?;

autocast.disable();
```

### Graph Operations

#### Building Computation Graphs

```rust
use tenflowers_core::{Graph, Session, Placeholder};

// Create a new graph
let mut graph = Graph::new();

// Add placeholders
let x = graph.placeholder("x", DType::F32, Some(&[None, 784]))?;
let y = graph.placeholder("y", DType::F32, Some(&[None, 10]))?;

// Add variables
let w = graph.variable("w", Tensor::randn(&[784, 10], DType::F32, Device::Cpu)?)?;
let b = graph.variable("b", Tensor::zeros(&[10], DType::F32, Device::Cpu)?)?;

// Build computation
let logits = graph.matmul(&x, &w)?;
let output = graph.add(&logits, &b)?;
let loss = graph.cross_entropy(&output, &y)?;

// Create session
let mut session = Session::new(&graph);

// Execute graph
let input_data = Tensor::randn(&[32, 784], DType::F32, Device::Cpu)?;
let target_data = Tensor::zeros(&[32, 10], DType::F32, Device::Cpu)?;

let results = session.run(
    &[&output, &loss],
    &[("x", &input_data), ("y", &target_data)]
)?;
```

#### Graph Optimization

```rust
use tenflowers_core::{Graph, GraphOptimizer};

let mut graph = Graph::new();
// ... build graph ...

// Apply optimizations
let optimizer = GraphOptimizer::new()
    .with_constant_folding(true)
    .with_dead_code_elimination(true)
    .with_common_subexpression_elimination(true)
    .with_operation_fusion(true);

let optimized_graph = optimizer.optimize(&graph)?;
```

---

## Neural Network Module

The `tenflowers-neural` module provides high-level neural network APIs.

### Layers

#### Dense (Fully Connected) Layer

```rust
use tenflowers_neural::{Dense, Layer};
use tenflowers_core::Tensor;

// Create dense layer
let dense = Dense::new(784, 128, true)  // input_size, output_size, use_bias
    .with_activation("relu".to_string());

// Forward pass
let input = Tensor::randn(&[32, 784], DType::F32, Device::Cpu)?;
let output = dense.forward(&input)?;

// Access parameters
let weights = dense.weight();
let bias = dense.bias();

// Set training mode
let mut dense = dense;
dense.set_training(true);
```

#### Convolutional Layers

```rust
use tenflowers_neural::{Conv2D, Conv1D, Conv3D};

// 2D Convolution
let conv2d = Conv2D::new(
    3,                    // input channels
    32,                   // output channels
    3,                    // kernel size
    1,                    // stride
    "same".to_string(),   // padding
    true                  // use bias
);

// 1D Convolution
let conv1d = Conv1D::new(
    128,                  // input channels
    256,                  // output channels
    3,                    // kernel size
    1,                    // stride
    "valid".to_string(),  // padding
    1,                    // dilation
    1                     // groups
);

// 3D Convolution
let conv3d = Conv3D::new(
    1,                    // input channels
    16,                   // output channels
    (3, 3, 3),           // kernel size
    (1, 1, 1),           // stride
    "same".to_string(),   // padding
    true                  // use bias
);
```

#### Recurrent Layers

```rust
use tenflowers_neural::{LSTM, GRU, RNN};

// LSTM Layer
let lstm = LSTM::new(
    128,                  // input_size
    256,                  // hidden_size
    2,                    // num_layers
    true,                 // bias
    0.1,                  // dropout
    false,                // bidirectional
    true                  // batch_first
);

// GRU Layer
let gru = GRU::new(
    128,                  // input_size
    256,                  // hidden_size
    2,                    // num_layers
    true,                 // bias
    0.1,                  // dropout
    false,                // bidirectional
    true                  // batch_first
);

// Forward pass with sequence
let input = Tensor::randn(&[32, 10, 128], DType::F32, Device::Cpu)?;  // batch, seq, features
let (output, hidden) = lstm.forward(&input, None)?;
```

#### Attention Layers

```rust
use tenflowers_neural::{MultiHeadAttention, TransformerEncoder, TransformerDecoder};

// Multi-Head Attention
let attention = MultiHeadAttention::new(
    512,                  // embed_dim
    8,                    // num_heads
    0.1,                  // dropout
    true                  // batch_first
);

// Transformer Encoder
let encoder = TransformerEncoder::new(
    512,                  // d_model
    8,                    // num_heads
    2048,                 // d_ff
    0.1,                  // dropout
    "relu".to_string()    // activation
);

// Transformer Decoder
let decoder = TransformerDecoder::new(
    512,                  // d_model
    8,                    // num_heads
    2048,                 // d_ff
    0.1,                  // dropout
    "relu".to_string()    // activation
);

// Forward pass
let src = Tensor::randn(&[32, 10, 512], DType::F32, Device::Cpu)?;
let tgt = Tensor::randn(&[32, 8, 512], DType::F32, Device::Cpu)?;

let encoder_output = encoder.forward(&src, None)?;
let decoder_output = decoder.forward(&tgt, &encoder_output, None, None)?;
```

#### Normalization Layers

```rust
use tenflowers_neural::{BatchNorm, LayerNorm, GroupNorm};

// Batch Normalization
let batch_norm = BatchNorm::new(128)
    .with_momentum(0.1)
    .with_epsilon(1e-5);

// Layer Normalization
let layer_norm = LayerNorm::new(512)
    .with_epsilon(1e-5);

// Group Normalization
let group_norm = GroupNorm::new(8, 128)  // num_groups, num_channels
    .with_epsilon(1e-5);

// Forward pass
let input = Tensor::randn(&[32, 128], DType::F32, Device::Cpu)?;
let normalized = batch_norm.forward(&input)?;
```

#### Pooling Layers

```rust
use tenflowers_neural::{MaxPool2D, AvgPool2D, GlobalMaxPool2D, AdaptiveMaxPool2D};

// Max Pooling
let max_pool = MaxPool2D::new(2, 2);  // kernel_size, stride

// Average Pooling
let avg_pool = AvgPool2D::new(2, 2);

// Global Pooling
let global_pool = GlobalMaxPool2D::new();

// Adaptive Pooling
let adaptive_pool = AdaptiveMaxPool2D::new((7, 7));  // output_size

// Forward pass
let input = Tensor::randn(&[32, 64, 28, 28], DType::F32, Device::Cpu)?;
let pooled = max_pool.forward(&input)?;
```

#### Activation Functions

```rust
use tenflowers_neural::*;

// Common activations
let relu = ReLU::new();
let sigmoid = Sigmoid::new();
let tanh = Tanh::new();
let softmax = Softmax::new(1);  // dim

// Advanced activations
let gelu = GELU::new();
let swish = Swish::new();
let mish = Mish::new();
let leaky_relu = LeakyReLU::new(0.01);

// Learnable activations
let prelu = PReLU::new(128);  // num_parameters
let elu = ELU::new(1.0);      // alpha

// Forward pass
let input = Tensor::randn(&[32, 128], DType::F32, Device::Cpu)?;
let activated = relu.forward(&input)?;
```

### Models

#### Sequential Model

```rust
use tenflowers_neural::{Sequential, Dense, ReLU, Dropout};

// Create sequential model
let mut model = Sequential::new(vec![
    Box::new(Dense::new(784, 512, true)),
    Box::new(ReLU::new()),
    Box::new(Dropout::new(0.5)),
    Box::new(Dense::new(512, 256, true)),
    Box::new(ReLU::new()),
    Box::new(Dropout::new(0.3)),
    Box::new(Dense::new(256, 10, true)),
]);

// Forward pass
let input = Tensor::randn(&[32, 784], DType::F32, Device::Cpu)?;
let output = model.forward(&input)?;

// Training mode
model.set_training(true);

// Get parameters
let params = model.parameters();
```

#### Functional Model

```rust
use tenflowers_neural::{FunctionalModel, FunctionalModelBuilder};

// Create functional model
let mut builder = FunctionalModelBuilder::new();

// Add inputs
let input1 = builder.add_input("input1", &[None, 784]);
let input2 = builder.add_input("input2", &[None, 128]);

// Add layers
let dense1 = builder.add_layer("dense1", Box::new(Dense::new(784, 256, true)));
let dense2 = builder.add_layer("dense2", Box::new(Dense::new(128, 256, true)));

// Connect layers
let hidden1 = builder.connect(&input1, &dense1);
let hidden2 = builder.connect(&input2, &dense2);

// Combine branches
let combined = builder.add_operation("add", |inputs| {
    &inputs[0] + &inputs[1]
});
let combined_output = builder.connect_multiple(&[&hidden1, &hidden2], &combined);

// Output layer
let output_layer = builder.add_layer("output", Box::new(Dense::new(256, 10, true)));
let output = builder.connect(&combined_output, &output_layer);

// Build model
let model = builder.build(&[&output])?;
```

#### Model Serialization

```rust
use tenflowers_neural::{Sequential, ModelSerialization};

// Save model
let model = Sequential::new(vec![...]);
model.save("model.json")?;

// Load model
let loaded_model = Sequential::<f32>::load("model.json")?;

// Save/load state dict
let state_dict = model.state_dict();
let mut new_model = Sequential::new(vec![...]);
new_model.load_state_dict(state_dict)?;
```

### Optimizers

#### Basic Optimizers

```rust
use tenflowers_neural::{SGD, Adam, AdamW, RMSprop};

// SGD with momentum
let sgd = SGD::new(0.01)
    .with_momentum(0.9)
    .with_weight_decay(1e-4);

// Adam optimizer
let adam = Adam::new(0.001)
    .with_beta1(0.9)
    .with_beta2(0.999)
    .with_epsilon(1e-8);

// AdamW (decoupled weight decay)
let adamw = AdamW::new(0.001)
    .with_weight_decay(0.01);

// RMSprop
let rmsprop = RMSprop::new(0.01)
    .with_alpha(0.99)
    .with_epsilon(1e-8);
```

#### Advanced Optimizers

```rust
use tenflowers_neural::{Nadam, RAdam, LAMB, Lookahead};

// Nadam (Adam with Nesterov momentum)
let nadam = Nadam::new(0.001);

// RAdam (Rectified Adam)
let radam = RAdam::new(0.001);

// LAMB (Layer-wise Adaptive Moments)
let lamb = LAMB::new(0.001);

// Lookahead wrapper
let base_optimizer = Adam::new(0.001);
let lookahead = Lookahead::new(base_optimizer, 0.5, 5);
```

#### Gradient Clipping

```rust
use tenflowers_neural::{clip_gradients_by_norm, clip_gradients_by_value};

// Gradient clipping by norm
let mut model = Sequential::new(vec![...]);
clip_gradients_by_norm(&mut model, 1.0)?;

// Gradient clipping by value
clip_gradients_by_value(&mut model, -1.0, 1.0)?;
```

#### Learning Rate Scheduling

```rust
use tenflowers_neural::{StepLR, ExponentialLR, CosineAnnealingLR, OneCycleLR};

// Step LR
let step_lr = StepLR::new(0.1, 10, 0.1);  // initial_lr, step_size, gamma

// Exponential LR
let exp_lr = ExponentialLR::new(0.1, 0.95);  // initial_lr, gamma

// Cosine Annealing
let cosine_lr = CosineAnnealingLR::new(0.1, 0.001, 100);  // max_lr, min_lr, T_max

// One Cycle LR
let one_cycle = OneCycleLR::new(0.1, 100, 0.3);  // max_lr, total_steps, pct_start

// Use with optimizer
let mut optimizer = Adam::new(0.001);
for epoch in 0..100 {
    let lr = step_lr.get_lr(epoch);
    optimizer.set_learning_rate(lr);
    // ... training loop
}
```

### Loss Functions

#### Classification Losses

```rust
use tenflowers_neural::{
    categorical_cross_entropy, sparse_categorical_cross_entropy,
    binary_cross_entropy, focal_loss, hinge_loss
};

// Categorical Cross Entropy
let predictions = Tensor::randn(&[32, 10], DType::F32, Device::Cpu)?;
let targets = Tensor::randn(&[32, 10], DType::F32, Device::Cpu)?;
let loss = categorical_cross_entropy(&predictions, &targets)?;

// Sparse Categorical Cross Entropy
let sparse_targets = Tensor::from_vec(vec![1, 3, 5, 2], &[4]);
let sparse_loss = sparse_categorical_cross_entropy(&predictions, &sparse_targets)?;

// Binary Cross Entropy
let binary_pred = Tensor::randn(&[32, 1], DType::F32, Device::Cpu)?;
let binary_target = Tensor::randn(&[32, 1], DType::F32, Device::Cpu)?;
let binary_loss = binary_cross_entropy(&binary_pred, &binary_target)?;

// Focal Loss (for imbalanced datasets)
let focal_loss = focal_loss(&predictions, &targets, 2.0, 0.25)?;

// Hinge Loss
let hinge_loss = hinge_loss(&predictions, &targets)?;
```

#### Regression Losses

```rust
use tenflowers_neural::{mse_loss, l1_loss, huber_loss, smooth_l1_loss};

let predictions = Tensor::randn(&[32, 1], DType::F32, Device::Cpu)?;
let targets = Tensor::randn(&[32, 1], DType::F32, Device::Cpu)?;

// Mean Squared Error
let mse = mse_loss(&predictions, &targets)?;

// Mean Absolute Error
let mae = l1_loss(&predictions, &targets)?;

// Huber Loss
let huber = huber_loss(&predictions, &targets, 1.0)?;

// Smooth L1 Loss
let smooth_l1 = smooth_l1_loss(&predictions, &targets, 1.0)?;
```

### Metrics

#### Classification Metrics

```rust
use tenflowers_neural::{accuracy, precision, recall, f1_score, confusion_matrix};

let predictions = Tensor::randn(&[100, 10], DType::F32, Device::Cpu)?;
let targets = Tensor::randn(&[100, 10], DType::F32, Device::Cpu)?;

// Accuracy
let acc = accuracy(&predictions, &targets)?;

// Precision
let prec = precision(&predictions, &targets, 0.5)?;

// Recall
let rec = recall(&predictions, &targets, 0.5)?;

// F1 Score
let f1 = f1_score(&predictions, &targets, 0.5)?;

// Confusion Matrix
let cm = confusion_matrix(&predictions, &targets, 0.5)?;
```

#### Regression Metrics

```rust
use tenflowers_neural::{r2_score, explained_variance};

let predictions = Tensor::randn(&[100, 1], DType::F32, Device::Cpu)?;
let targets = Tensor::randn(&[100, 1], DType::F32, Device::Cpu)?;

// R-squared
let r2 = r2_score(&predictions, &targets)?;

// Explained Variance
let ev = explained_variance(&predictions, &targets)?;
```

---

## Autograd Module

The `tenflowers-autograd` module provides automatic differentiation capabilities.

### Gradient Computation

#### Basic Gradient Computation

```rust
use tenflowers_autograd::{GradientTape, TrackedTensor};
use tenflowers_core::Tensor;

// Create gradient tape
let tape = GradientTape::new();

// Create tracked tensors
let x = Tensor::from_vec(vec![2.0, 3.0], &[2]);
let x_tracked = tape.watch(x);

// Compute function
let y = x_tracked.pow(&Tensor::from_scalar(3.0))?;  // y = x^3
let z = y.sum(None)?;                               // z = sum(y)

// Compute gradient
let grad = tape.gradient(&z, &[&x_tracked])?;
// grad[0] = [12.0, 27.0]  // dz/dx = 3x^2
```

#### Multi-Variable Gradients

```rust
use tenflowers_autograd::{GradientTape, TrackedTensor};

let tape = GradientTape::new();

let x = Tensor::from_vec(vec![1.0, 2.0], &[2]);
let y = Tensor::from_vec(vec![3.0, 4.0], &[2]);

let x_tracked = tape.watch(x);
let y_tracked = tape.watch(y);

// f(x, y) = x^2 + y^2 + 2*x*y
let x_squared = x_tracked.pow(&Tensor::from_scalar(2.0))?;
let y_squared = y_tracked.pow(&Tensor::from_scalar(2.0))?;
let xy = &x_tracked * &y_tracked;
let two_xy = &xy * &Tensor::from_scalar(2.0);

let f = &x_squared + &y_squared + &two_xy;

// Compute gradients
let grads = tape.gradient(&f.sum(None)?, &[&x_tracked, &y_tracked])?;
// grads[0] = df/dx = 2x + 2y
// grads[1] = df/dy = 2y + 2x
```

#### Neural Network Gradients

```rust
use tenflowers_autograd::GradientTape;
use tenflowers_neural::{Dense, categorical_cross_entropy};

let tape = GradientTape::new();

// Create a simple network
let layer1 = Dense::new(784, 128, true);
let layer2 = Dense::new(128, 10, true);

// Forward pass with gradient tracking
let input = Tensor::randn(&[32, 784], DType::F32, Device::Cpu)?;
let input_tracked = tape.watch(input);

let hidden = layer1.forward(&input_tracked)?;
let output = layer2.forward(&hidden)?;

// Compute loss
let targets = Tensor::randn(&[32, 10], DType::F32, Device::Cpu)?;
let loss = categorical_cross_entropy(&output, &targets)?;

// Compute gradients with respect to parameters
let layer1_weights = tape.watch(layer1.weight().clone());
let layer2_weights = tape.watch(layer2.weight().clone());

let grads = tape.gradient(&loss, &[&layer1_weights, &layer2_weights])?;
```

### Gradient Tape

#### Persistent Tape

```rust
use tenflowers_autograd::GradientTape;

// Create persistent tape for multiple gradient computations
let tape = GradientTape::new().persistent();

let x = Tensor::from_vec(vec![2.0], &[1]);
let x_tracked = tape.watch(x);

let y = x_tracked.pow(&Tensor::from_scalar(3.0))?;
let z = &y * &Tensor::from_scalar(2.0);

// Compute multiple gradients
let dy_dx = tape.gradient(&y, &[&x_tracked])?;
let dz_dx = tape.gradient(&z, &[&x_tracked])?;
```

#### Gradient Accumulation

```rust
use tenflowers_autograd::GradientTape;

let tape = GradientTape::new();

let x = Tensor::from_vec(vec![1.0, 2.0], &[2]);
let x_tracked = tape.watch(x);

// Multiple computations that accumulate gradients
let y1 = x_tracked.pow(&Tensor::from_scalar(2.0))?;
let y2 = x_tracked.pow(&Tensor::from_scalar(3.0))?;
let total = &y1 + &y2;

// Gradient includes contributions from both terms
let grad = tape.gradient(&total.sum(None)?, &[&x_tracked])?;
```

### Higher-Order Derivatives

#### Second-Order Derivatives (Hessian)

```rust
use tenflowers_autograd::{GradientTape, hessian};

let x = Tensor::from_vec(vec![1.0, 2.0], &[2]);

// Compute Hessian matrix
let hessian_matrix = hessian(&x, |x| {
    // f(x) = x1^2 + 2*x1*x2 + x2^2
    let x1 = x.slice(&[Some((0, 1))])?;
    let x2 = x.slice(&[Some((1, 2))])?;
    let term1 = x1.pow(&Tensor::from_scalar(2.0))?;
    let term2 = &x1 * &x2 * &Tensor::from_scalar(2.0);
    let term3 = x2.pow(&Tensor::from_scalar(2.0))?;
    Ok((&term1 + &term2 + &term3).sum(None)?)
})?;
```

#### Third-Order Derivatives

```rust
use tenflowers_autograd::{GradientTape, third_order_gradient};

let x = Tensor::from_vec(vec![2.0], &[1]);

// Compute third-order derivative
let third_deriv = third_order_gradient(&x, |x| {
    // f(x) = x^4
    x.pow(&Tensor::from_scalar(4.0))
})?;
// Result: 24x = 48.0
```

#### Mixed Partial Derivatives

```rust
use tenflowers_autograd::{GradientTape, mixed_partial_derivative};

let x = Tensor::from_vec(vec![1.0], &[1]);
let y = Tensor::from_vec(vec![2.0], &[1]);

// Compute ∂²f/∂x∂y
let mixed_partial = mixed_partial_derivative(&x, &y, |x, y| {
    // f(x, y) = x^2 * y^3
    let x_squared = x.pow(&Tensor::from_scalar(2.0))?;
    let y_cubed = y.pow(&Tensor::from_scalar(3.0))?;
    Ok(&x_squared * &y_cubed)
})?;
```

---

## Dataset Module

The `tenflowers-dataset` module provides data loading and preprocessing utilities.

### Data Loading

#### TensorDataset

```rust
use tenflowers_dataset::{TensorDataset, Dataset};
use tenflowers_core::Tensor;

// Create dataset from tensors
let features = Tensor::randn(&[1000, 784], DType::F32, Device::Cpu)?;
let labels = Tensor::randn(&[1000, 10], DType::F32, Device::Cpu)?;
let dataset = TensorDataset::new(features, labels);

// Access samples
let (sample_features, sample_labels) = dataset.get(0)?;
println!("Dataset length: {}", dataset.len());

// Iterate through dataset
for (features, labels) in dataset.iter().take(10) {
    // Process batch
}
```

#### DataLoader

```rust
use tenflowers_dataset::{DataLoader, TensorDataset};

let dataset = TensorDataset::new(features, labels);

// Create data loader
let data_loader = DataLoader::new(dataset)
    .batch_size(32)
    .shuffle(true)
    .num_workers(4)
    .drop_last(true);

// Training loop
for epoch in 0..10 {
    for (batch_features, batch_labels) in data_loader.iter() {
        // Training step
        let predictions = model.forward(&batch_features)?;
        let loss = loss_fn(&predictions, &batch_labels)?;
        // ... backward pass
    }
}
```

#### File-Based Datasets

```rust
use tenflowers_dataset::{CsvDataset, ImageDataset, AudioDataset};

// CSV Dataset
let csv_dataset = CsvDataset::new("data.csv")
    .with_header(true)
    .with_target_column("label")
    .with_delimiter(',');

// Image Dataset
let image_dataset = ImageDataset::new("images/")
    .with_extensions(&[".jpg", ".png"])
    .with_transform(|img| {
        // Resize to 224x224
        img.resize(224, 224)
    });

// Audio Dataset
let audio_dataset = AudioDataset::new("audio/")
    .with_sample_rate(44100)
    .with_transform(|audio| {
        // Convert to spectrogram
        audio.to_spectrogram(1024, 512)
    });
```

### Transformations

#### Data Augmentation

```rust
use tenflowers_dataset::{Transform, Compose, RandomCrop, RandomFlip, Normalize};

// Create transform pipeline
let transform = Compose::new(vec![
    Box::new(RandomCrop::new(224, 224)),
    Box::new(RandomFlip::horizontal()),
    Box::new(Normalize::new(
        vec![0.485, 0.456, 0.406],  // mean
        vec![0.229, 0.224, 0.225],  // std
    )),
]);

// Apply to dataset
let augmented_dataset = dataset.map(|sample| {
    transform.apply(sample)
});
```

#### Custom Transforms

```rust
use tenflowers_dataset::{Transform, Dataset};

struct CustomTransform {
    scale: f32,
}

impl Transform for CustomTransform {
    fn apply(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        Ok(input * self.scale)
    }
}

// Use custom transform
let custom_transform = CustomTransform { scale: 2.0 };
let transformed_dataset = dataset.map(|sample| {
    custom_transform.apply(sample)
});
```

### Synthetic Data

#### Synthetic Datasets

```rust
use tenflowers_dataset::{SyntheticDataset, DatasetGenerator};

// Generate synthetic classification data
let synthetic_data = SyntheticDataset::classification(
    1000,        // samples
    784,         // features
    10,          // classes
    0.1,         // noise
    42           // random seed
);

// Generate synthetic regression data
let regression_data = SyntheticDataset::regression(
    1000,        // samples
    10,          // features
    1,           // targets
    0.1,         // noise
    42           // random seed
);

// Custom synthetic data
let custom_data = DatasetGenerator::new()
    .with_distribution("normal")
    .with_parameters(&[0.0, 1.0])  // mean, std
    .generate(1000, 784);
```

---

## FFI Module

The `tenflowers-ffi` module provides Python bindings and C API.

### Python Bindings

#### Basic Tensor Operations

```python
import tenflowers as tf

# Create tensors
a = tf.tensor([1.0, 2.0, 3.0])
b = tf.zeros([3, 3])
c = tf.randn([10, 10])

# Operations
result = a + b
matmul_result = tf.matmul(a, b)
```

#### Neural Networks

```python
import tenflowers.nn as nn

# Create model
model = nn.Sequential([
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)
])

# Forward pass
x = tf.randn([32, 784])
y = model(x)

# Training
optimizer = tf.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(100):
    # Forward pass
    predictions = model(x)
    loss = loss_fn(predictions, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

#### Autograd

```python
import tenflowers as tf

# Enable gradients
x = tf.tensor([2.0, 3.0], requires_grad=True)

# Compute function
y = x ** 3
z = y.sum()

# Compute gradients
z.backward()
print(x.grad)  # [12.0, 27.0]
```

### C API

#### Basic Usage

```c
#include "tenflowers.h"

int main() {
    // Initialize TenfloweRS
    tensor_init();
    
    // Create tensors
    float data[] = {1.0, 2.0, 3.0, 4.0};
    size_t shape[] = {2, 2};
    TensorHandle* tensor = tensor_from_data(data, shape, 2, DTYPE_F32);
    
    // Operations
    TensorHandle* result = tensor_add(tensor, tensor);
    
    // Get data
    float* result_data = tensor_data(result);
    
    // Cleanup
    tensor_destroy(tensor);
    tensor_destroy(result);
    tensor_cleanup();
    
    return 0;
}
```

#### Error Handling

```c
#include "tenflowers.h"

int main() {
    TensorStatus status;
    
    // Operation with error checking
    TensorHandle* result = tensor_matmul(a, b, &status);
    if (status != TENSOR_OK) {
        const char* error_msg = tensor_last_error();
        printf("Error: %s\n", error_msg);
        tensor_free_error();
        return 1;
    }
    
    return 0;
}
```

---

## Best Practices

### Performance Optimization

1. **Use appropriate data types**: Use `f32` for most operations, `f16` for memory-constrained scenarios
2. **Batch operations**: Process multiple samples together for better GPU utilization
3. **Memory management**: Reuse tensors where possible, avoid unnecessary copies
4. **Device placement**: Keep related tensors on the same device to avoid transfers

### Memory Management

```rust
// Good: Reuse tensors
let mut buffer = Tensor::zeros(&[1000, 784], DType::F32, Device::Cpu)?;
for epoch in 0..epochs {
    // Reuse buffer instead of creating new tensors
    buffer.copy_from(&new_data)?;
    let result = model.forward(&buffer)?;
}

// Good: In-place operations where possible
tensor.add_(&other)?;  // In-place addition
```

### Error Handling

```rust
use tenflowers_core::{TensorError, Result};

// Always handle errors explicitly
match tensor.matmul(&other) {
    Ok(result) => {
        // Process result
    }
    Err(TensorError::ShapeMismatch { expected, actual }) => {
        eprintln!("Shape mismatch: expected {:?}, got {:?}", expected, actual);
    }
    Err(e) => {
        eprintln!("Tensor operation failed: {}", e);
    }
}
```

### Training Best Practices

```rust
// Use mixed precision for large models
let config = MixedPrecisionConfig::new().with_enabled(true);
let autocast = AutocastContext::new(config);

// Implement gradient accumulation for large batches
let accumulation_steps = 4;
for (i, batch) in data_loader.enumerate() {
    let loss = model.train_step(&batch) / accumulation_steps as f32;
    
    if (i + 1) % accumulation_steps == 0 {
        optimizer.step(&model)?;
        model.zero_grad();
    }
}
```

---

This API reference provides comprehensive examples for all major components of TenfloweRS. For more specific use cases and advanced features, refer to the individual module documentation and examples in the repository.