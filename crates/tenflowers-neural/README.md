# TenfloweRS Neural

High-level neural network APIs for TenfloweRS, providing layers, models, optimizers, training utilities, and domain-specific architectures for deep learning in Rust.

> Stable (v0.2.0 -- 2026-07-11) | 11,596 tests passing (8 skipped, `--all-features`) | 0 clippy warnings

## Overview

`tenflowers-neural` is the largest crate in the TenfloweRS ecosystem, implementing a comprehensive set of neural network components spanning 150+ domains:

- **Neural Network Layers**: Dense, Conv2D, LSTM, GRU, Transformer, GNN, TCN, and more
- **Attention Mechanisms**: Multi-head, Flash Attention, ALiBi, RoPE, GQA
- **Model Abstractions**: Sequential and functional model APIs
- **Optimizers**: SGD, Adam, AdamW, LAMB, Lion, Muon, with LR schedulers
- **Loss Functions**: Common losses for classification, regression, and contrastive learning
- **Training Utilities**: Checkpointing, early stopping, callbacks, mixed precision
- **Deployment**: Quantization, magnitude/random pruning, compression (structured/gradual/lottery-ticket pruning strategies honestly report not-yet-implemented rather than fabricating results)

## Domain Coverage

TenfloweRS Neural covers an extensive range of ML domains:

**Core Architectures**: Transformers (encoder/decoder), Vision Transformers, State Space Models (Mamba, S4), Efficient Transformers (RetNet, GQA), Kolmogorov-Arnold Networks, Hypernetworks

**Computer Vision**: CNN layers, depth estimation, object tracking, image generation, neural rendering (NeRF, 3D Gaussian splatting), scene graphs

**Natural Language Processing**: Tokenizers, text generation pipelines, LLM serving, language model evaluation, document understanding

**Audio and Speech**: Speech recognition, audio generation, music generation, emotion recognition, audio models (HuBERT, Data2Vec)

**Reinforcement Learning**: DQN, PPO, SAC, multi-agent RL (MARL), safe RL, inverse RL, reward learning, self-play, world models

**Generative Models**: VAE, normalizing flows, diffusion models, energy-based models, GAN variants, neural compression

**Graph Neural Networks**: GNN (basic and advanced), graph transformers, temporal GNN, molecular GNN, graph neural ODE, graph signal processing, graph generation, graph matching, graph foundation models

**Scientific ML**: Physics-informed neural networks (PINN), neural ODE/SDE, operator learning (FNO, DeepONet, WNO, GNO), differentiable physics, simulation-based inference

**Probabilistic Methods**: Bayesian deep learning, Bayesian optimization, variational inference, probabilistic circuits, conformal prediction, Monte Carlo methods, mixture density networks

**Causal Inference**: Causal discovery, causal representation learning, causal RL, causal time series analysis

**Meta-Learning and AutoML**: Meta-learning, NAS, hyperparameter optimization, AutoML pipelines, learning to learn, curriculum learning

**Structured and Specialized**: Knowledge graphs, knowledge distillation, federated learning, continual learning, lifelong learning, online learning, active learning, zero-shot learning, self-supervised learning, contrastive learning

**Domain-Specific**: Protein structure/language models, molecular GNN, drug discovery, bio ML, medical imaging, digital pathology, climate ML, geospatial ML, satellite ML, materials ML, financial ML, robotics, trajectory prediction

**Optimization and Theory**: Optimal transport, Riemannian geometry, topological ML, information theory, tensor decomposition, tensor networks, cooperative game theory, mean field games, optimal control

**Efficiency and Deployment**: Model compression, edge optimization, mixture of experts, sparse learning, quantization, pruning, model merging, LoRA adapters, test-time compute/adaptation

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

## Architecture

### Core Components

- **Layer Trait**: Common interface for all neural network layers
- **Model Trait**: Training and inference capabilities
- **Optimizer Trait**: Parameter update algorithms (SGD, Adam, AdamW, LAMB, Lion, Muon)
- **Loss Functions**: Various objective functions
- **Metrics**: Performance measurement utilities
- **LR Schedulers**: Cosine annealing, warmup, step decay, and more

### Layer Types

**Basic Layers**: Dense, Conv2D/Conv3D, LSTM/GRU (bidirectional), Transformer encoder/decoder, TCN

**Attention**: Multi-head attention, Flash Attention, ALiBi, RoPE, GQA, KV-cache

**Normalization**: BatchNorm, LayerNorm, GroupNorm

**Regularization**: Dropout (standard and variational), L1/L2 regularization, spectral normalization

**Activation Functions**: ReLU, GELU, SiLU, Mish, Tanh, Sigmoid, PReLU, ELU

### Optimizer Features

- **Gradient Clipping**: By value or norm
- **Gradient Accumulation**: For large effective batch training
- **Parameter Groups**: Different LR for different layers
- **State Checkpointing**: Resume training from checkpoint
- **LR Finding**: Automatic learning rate range test

## Feature Flags

- `default`: Standard neural network functionality
- `gpu`: GPU-accelerated layer operations
- `serialize`: Model serialization and checkpointing
- `onnx`: ONNX model import/export (real `prost`-based protobuf decode/encode)
- `gloo`: Distributed training communication
- `gzip`: Gzip compression for weight/checkpoint serialization (via `oxiarc-archive`)

## Integration with TenfloweRS Ecosystem

- **Autograd**: Automatic gradient computation for all layers
- **Dataset**: Efficient data loading and augmentation
- **Core**: Low-level tensor operations
- **FFI**: Export models for Python inference

## License

Licensed under Apache-2.0
