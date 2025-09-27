# TenfloweRS Neural

High-level neural network APIs for TenfloweRS, providing layers, models, optimizers, and training utilities for deep learning in Rust.

> Alpha Notice (0.1.0-alpha.1 · 2025-09-27)
> Core layer/optimizer abstractions are stable; some advanced architectures (Transformer variants, large model tooling, distributed training) are skeletal or experimental in this release.

## Overview

`tenflowers-neural` implements:
- **Neural Network Layers**: Dense, Conv2D, LSTM, Transformer, and more
- **Model Abstractions**: Sequential and functional model APIs
- **Optimizers**: SGD, Adam, AdamW, RMSprop with advanced features
- **Loss Functions**: Common losses for classification and regression
- **Metrics**: Training and evaluation metrics
- **Learning Rate Schedulers**: Various scheduling strategies
- **Training Utilities**: Model checkpointing, early stopping, callbacks

## Features

- **Layer Composition**: Build complex models from modular layers
- **Automatic Differentiation**: Seamless integration with tenflowers-autograd
- **Mixed Precision Training**: FP16 training with loss scaling
- **Distributed Training**: Data and model parallelism support
- **ONNX Export**: Export trained models to ONNX format
- **Pretrained Models**: Common architectures with pretrained weights

## Usage

### Building a Simple Neural Network

```rust
use tenflowers_neural::{Sequential, Dense, Activation};
use tenflowers_core::{Device, DType};

// Create a sequential model
let mut model = Sequential::new();

// Add layers
model.add(Dense::new(784, 128)?);
model.add(Activation::relu());
model.add(Dense::new(128, 64)?);
model.add(Activation::relu());
model.add(Dense::new(64, 10)?);
model.add(Activation::softmax());

// Compile with optimizer and loss
model.compile(
    Adam::new(0.001),
    Loss::CrossEntropy,
    vec![Metric::Accuracy],
)?;

// Train the model
model.fit(
    &train_data,
    &train_labels,
    FitConfig {
        batch_size: 32,
        epochs: 10,
        validation_data: Some((&val_data, &val_labels)),
        callbacks: vec![
            Callback::EarlyStopping { patience: 3 },
            Callback::ModelCheckpoint { path: "model.pt" },
        ],
    },
)?;
```

### Building a Convolutional Neural Network

```rust
use tenflowers_neural::{Conv2D, MaxPool2D, BatchNorm, Dropout};

let mut model = Sequential::new();

// Convolutional layers
model.add(Conv2D::new(3, 32, [3, 3], [1, 1], Padding::Same)?);
model.add(BatchNorm::new(32)?);
model.add(Activation::relu());
model.add(MaxPool2D::new([2, 2], [2, 2])?);

model.add(Conv2D::new(32, 64, [3, 3], [1, 1], Padding::Same)?);
model.add(BatchNorm::new(64)?);
model.add(Activation::relu());
model.add(MaxPool2D::new([2, 2], [2, 2])?);

// Dense layers
model.add(Flatten::new());
model.add(Dense::new(7 * 7 * 64, 128)?);
model.add(Dropout::new(0.5));
model.add(Activation::relu());
model.add(Dense::new(128, 10)?);
```

### Building a Transformer Model

```rust
use tenflowers_neural::{MultiHeadAttention, LayerNorm, FeedForward};

// Create a transformer encoder layer
struct TransformerEncoder {
    attention: MultiHeadAttention,
    norm1: LayerNorm,
    feedforward: FeedForward,
    norm2: LayerNorm,
}

impl TransformerEncoder {
    fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Result<Self> {
        Ok(Self {
            attention: MultiHeadAttention::new(d_model, num_heads)?,
            norm1: LayerNorm::new(d_model)?,
            feedforward: FeedForward::new(d_model, d_ff)?,
            norm2: LayerNorm::new(d_model)?,
        })
    }
    
    fn forward(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        // Multi-head attention with residual connection
        let attn_out = self.attention.forward(x, x, x, None)?;
        let x = self.norm1.forward(&(x + &attn_out)?)?;
        
        // Feed-forward with residual connection
        let ff_out = self.feedforward.forward(&x)?;
        let x = self.norm2.forward(&(&x + &ff_out)?)?;
        
        Ok(x)
    }
}
```

### Custom Layers

```rust
use tenflowers_neural::Layer;
use tenflowers_core::{Tensor, Result};

struct CustomLayer {
    weight: Tensor<f32>,
    bias: Tensor<f32>,
    training: bool,
}

impl Layer<f32> for CustomLayer {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let output = input.matmul(&self.weight)?;
        let output = output.add(&self.bias)?;
        
        if self.training {
            // Apply training-specific behavior
        }
        
        Ok(output)
    }
    
    fn parameters(&self) -> Vec<&Tensor<f32>> {
        vec![&self.weight, &self.bias]
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
        vec![&mut self.weight, &mut self.bias]
    }
    
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}
```

### Advanced Optimizers

```rust
use tenflowers_neural::{Adam, AdamW, CosineAnnealingLR};

// Adam with weight decay
let optimizer = AdamW::builder()
    .learning_rate(0.001)
    .weight_decay(0.01)
    .beta1(0.9)
    .beta2(0.999)
    .epsilon(1e-8)
    .build()?;

// Learning rate scheduling
let scheduler = CosineAnnealingLR::new(
    initial_lr: 0.1,
    min_lr: 0.0001,
    T_max: 100,
);

// Training loop with scheduler
for epoch in 0..num_epochs {
    let lr = scheduler.get_lr(epoch);
    optimizer.set_learning_rate(lr);
    
    for (batch_x, batch_y) in train_loader {
        let loss = model.train_step(&batch_x, &batch_y, &optimizer)?;
    }
}
```

### Mixed Precision Training

```rust
use tenflowers_neural::{MixedPrecisionTrainer, GradScaler};

let scaler = GradScaler::new();
let trainer = MixedPrecisionTrainer::new(model, optimizer, scaler);

// Training with automatic mixed precision
trainer.train_step(&input, &target, |logits, target| {
    // Compute loss in FP32
    loss_fn(logits.to_f32()?, target)
})?;
```

## Architecture

### Core Components

- **Layer Trait**: Common interface for all neural network layers
- **Model Trait**: Training and inference capabilities
- **Optimizer Trait**: Parameter update algorithms
- **Loss Functions**: Various objective functions
- **Metrics**: Performance measurement utilities

### Layer Types

**Basic Layers**:
- Dense: Fully connected layer
- Conv2D/Conv3D: Convolutional layers
- LSTM/GRU: Recurrent layers
- MultiHeadAttention: Transformer attention

**Normalization**:
- BatchNorm: Batch normalization
- LayerNorm: Layer normalization
- GroupNorm: Group normalization

**Regularization**:
- Dropout: Standard and variational dropout
- L1/L2 regularization
- Spectral normalization

**Activation Functions**:
- ReLU, GELU, SiLU, Tanh, Sigmoid
- Learnable: PReLU, ELU
- Custom activation support

### Optimizer Features

- **Gradient Clipping**: By value or norm
- **Gradient Accumulation**: For large batch training
- **Parameter Groups**: Different LR for different layers
- **State Checkpointing**: Resume training from checkpoint
- **Distributed**: Gradient aggregation across devices

## Performance Optimizations

- **Kernel Fusion**: Fused operations for common patterns
- **Graph Optimization**: Layer fusion and constant folding
- **Memory Efficiency**: Gradient checkpointing, inplace ops
- **Multi-GPU**: Data and model parallelism
- **Quantization**: INT8 inference support

### Current Alpha Limitations
- Limited pretrained weight bundles not yet published
- Distributed / multi-GPU trainers are feature-gated prototypes
- Mixed precision path requires manual opt-in; scaling heuristics evolving
- Some exotic layers (e.g., advanced attention variants) unoptimized

### Roadmap Focus (next milestones)
1. Exportable model serialization format (intermediate before ONNX)
2. Gradient accumulation + micro-batch scheduler polish
3. Checkpoint versioning & integrity verification
4. Expanded metrics (AUC, F1, perplexity) with streaming reducers
5. ONNX export subset for inference graphs
6. Automated regression benchmark harness per layer type

## Pretrained Models

Available models with ImageNet weights:
- ResNet: ResNet18, ResNet34, ResNet50, ResNet101
- EfficientNet: B0-B7 variants
- Vision Transformer: ViT-B/16, ViT-L/16
- BERT: Base and Large variants
- GPT-2: Small, Medium, Large

```rust
use tenflowers_neural::models::{ResNet50, Pretrained};

// Load pretrained ResNet50
let model = ResNet50::pretrained(ImageNetWeights)?;

// Fine-tune on custom dataset
model.freeze_backbone();
model.replace_head(num_classes)?;
```

## Integration with TenfloweRS Ecosystem

- **Autograd**: Automatic gradient computation for all layers
- **Dataset**: Efficient data loading and augmentation
- **Core**: Low-level tensor operations
- **FFI**: Export models for Python inference

## Contributing

Priority areas for contribution:
- Implementing missing layers (see TODO.md)
- Adding more pretrained models
- Optimizing existing implementations
- Writing comprehensive tests
- Improving documentation

## License

Dual-licensed under MIT OR Apache-2.0