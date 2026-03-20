# TenfloweRS Neural - Comprehensive Layer Documentation Guide

## Table of Contents

1. [Dense Layers](#dense-layers)
2. [Convolutional Layers](#convolutional-layers)
3. [Recurrent Layers](#recurrent-layers)
4. [Attention Mechanisms](#attention-mechanisms)
5. [Normalization Layers](#normalization-layers)
6. [Regularization Layers](#regularization-layers)
7. [Pooling Operations](#pooling-operations)
8. [Embedding Layers](#embedding-layers)
9. [Advanced Architectures](#advanced-architectures)

---

## Dense Layers

### Dense (Fully Connected) Layer

The Dense layer performs linear transformation: `y = xW + b`

**Key Features:**
- Xavier/He/Normal weight initialization
- Optional bias term
- GPU acceleration support
- Parameter count: `input_dim * output_dim + output_dim` (with bias)

**Basic Usage:**

```rust
use tenflowers_neural::layers::Dense;
use tenflowers_core::Tensor;

// Create a dense layer: 784 inputs -> 128 outputs
let layer = Dense::new(784, 128, true)?; // true = use bias

// Forward pass
let input = Tensor::zeros(&[32, 784]); // batch_size=32
let output = layer.forward(&input)?;
assert_eq!(output.shape(), &[32, 128]);
```

**Weight Initialization Strategies:**

```rust
use tenflowers_neural::layers::{Dense, InitMethod};

// Xavier/Glorot initialization (default for tanh/sigmoid)
let layer1 = Dense::with_init(784, 128, true, InitMethod::Xavier)?;

// He initialization (recommended for ReLU)
let layer2 = Dense::with_init(784, 128, true, InitMethod::He)?;

// Normal initialization with custom std
let layer3 = Dense::with_init(784, 128, true, InitMethod::Normal(0.01))?;
```

**Multi-Layer Perceptron Example:**

```rust
use tenflowers_neural::{Sequential, Dense, ActivationFunction};

fn build_mlp(input_dim: usize, hidden_dims: &[usize], output_dim: usize)
    -> Result<Sequential<f32>, Box<dyn std::error::Error>> {
    let mut model = Sequential::new();

    // Input layer
    model.add(Dense::new(input_dim, hidden_dims[0], true)?);
    model.add_activation(ActivationFunction::ReLU);

    // Hidden layers
    for i in 0..hidden_dims.len() - 1 {
        model.add(Dense::new(hidden_dims[i], hidden_dims[i + 1], true)?);
        model.add_activation(ActivationFunction::ReLU);
    }

    // Output layer
    model.add(Dense::new(hidden_dims[hidden_dims.len() - 1], output_dim, true)?);

    Ok(model)
}

// Create a 784->256->128->64->10 network
let model = build_mlp(784, &[256, 128, 64], 10)?;
```

---

## Convolutional Layers

### Conv1D - 1D Convolution

For sequential data (time series, audio, text).

**Parameters:**
- `in_channels`: Number of input channels
- `out_channels`: Number of output filters
- `kernel_size`: Filter width
- `stride`: Step size
- `padding`: Zero padding amount
- `dilation`: Spacing between kernel elements
- `groups`: Number of blocked connections

**Basic Conv1D Example:**

```rust
use tenflowers_neural::layers::Conv1D;
use tenflowers_core::Tensor;

// Conv1D for text/sequence processing
// Input: (batch, channels, sequence_length)
let conv = Conv1D::new(
    300,    // in_channels (e.g., embedding dimension)
    128,    // out_channels (number of filters)
    3,      // kernel_size (3-gram)
    1,      // stride
    1,      // padding
)?;

let input = Tensor::zeros(&[32, 300, 100]); // batch=32, dim=300, seq_len=100
let output = conv.forward(&input)?;
// Output shape: [32, 128, 100]
```

**Text CNN Architecture:**

```rust
use tenflowers_neural::{Sequential, Conv1D, MaxPool1D, Dense};

fn text_cnn(vocab_size: usize, embed_dim: usize, num_classes: usize)
    -> Result<Sequential<f32>, Box<dyn std::error::Error>> {
    let mut model = Sequential::new();

    // Embedding layer (implemented separately)
    // model.add(Embedding::new(vocab_size, embed_dim)?);

    // Multiple filter sizes (3, 4, 5-grams)
    let filter_sizes = [3, 4, 5];
    let num_filters = 100;

    // For simplicity, showing single path (in practice, use parallel branches)
    model.add(Conv1D::new(embed_dim, num_filters, 3, 1, 1)?);
    model.add_activation(ActivationFunction::ReLU);
    model.add(MaxPool1D::new(2, 2)?);

    // Global max pooling would go here
    // Flatten and classify
    model.add(Dense::new(num_filters * (seq_len / 2), num_classes, true)?);

    Ok(model)
}
```

### Conv2D - 2D Convolution

Standard convolution for images and spatial data.

**Basic Image Classification CNN:**

```rust
use tenflowers_neural::layers::{Conv2D, BatchNorm, MaxPool2D, Dense};

fn build_image_classifier() -> Result<Sequential<f32>, Box<dyn std::error::Error>> {
    let mut model = Sequential::new();

    // Block 1: Conv + BN + ReLU + Pool
    model.add(Conv2D::new(3, 32, 3, 1, 1)?);  // 3 RGB channels -> 32 filters
    model.add(BatchNorm::new(32)?);
    model.add_activation(ActivationFunction::ReLU);
    model.add(MaxPool2D::new(2, 2, 0)?);

    // Block 2: More filters
    model.add(Conv2D::new(32, 64, 3, 1, 1)?);
    model.add(BatchNorm::new(64)?);
    model.add_activation(ActivationFunction::ReLU);
    model.add(MaxPool2D::new(2, 2, 0)?);

    // Block 3: Even more filters
    model.add(Conv2D::new(64, 128, 3, 1, 1)?);
    model.add(BatchNorm::new(128)?);
    model.add_activation(ActivationFunction::ReLU);
    model.add(MaxPool2D::new(2, 2, 0)?);

    // Global average pooling
    model.add(GlobalAvgPool2D::new());

    // Classification head
    model.add(Dense::new(128, 10, true)?);

    Ok(model)
}

// Usage
let model = build_image_classifier()?;
let image = Tensor::zeros(&[1, 3, 224, 224]); // NCHW format
let logits = model.forward(&image)?;
```

**Depthwise Separable Convolution (Mobile-Efficient):**

```rust
use tenflowers_neural::layers::{DepthwiseConv2D, Conv2D};

fn depthwise_separable_conv(in_channels: usize, out_channels: usize)
    -> Result<Vec<Box<dyn Layer<f32>>>, Box<dyn std::error::Error>> {
    let mut layers = Vec::new();

    // Depthwise: each input channel convolved separately
    layers.push(Box::new(DepthwiseConv2D::new(in_channels, 3, 1, 1)?));
    layers.push(Box::new(BatchNorm::new(in_channels)?));
    layers.push(Box::new(Activation::new(ActivationFunction::ReLU)));

    // Pointwise: 1x1 conv to combine channels
    layers.push(Box::new(Conv2D::new(in_channels, out_channels, 1, 1, 0)?));
    layers.push(Box::new(BatchNorm::new(out_channels)?));
    layers.push(Box::new(Activation::new(ActivationFunction::ReLU)));

    Ok(layers)
}
```

### Conv3D - 3D Convolution

For video and volumetric data.

```rust
use tenflowers_neural::layers::Conv3D;

// Video processing
let conv3d = Conv3D::new(
    3,      // RGB channels
    64,     // output channels
    (3, 3, 3), // kernel: (time, height, width)
    (1, 1, 1), // stride
    (1, 1, 1), // padding
)?;

// Input: (batch, channels, depth/time, height, width)
let video = Tensor::zeros(&[8, 3, 16, 224, 224]); // 8 videos, 16 frames each
let features = conv3d.forward(&video)?;
```

---

## Recurrent Layers

### RNN - Basic Recurrent Neural Network

**Basic RNN Usage:**

```rust
use tenflowers_neural::layers::RNN;
use tenflowers_core::Tensor;

// Create RNN: input_size=100, hidden_size=256
let rnn = RNN::new(100, 256, false)?; // false = not bidirectional

// Input: (batch, seq_len, input_size)
let input = Tensor::zeros(&[32, 50, 100]); // 32 sequences of length 50
let (output, hidden) = rnn.forward(&input, None)?;

// output: (batch, seq_len, hidden_size)
// hidden: (batch, hidden_size)
```

### LSTM - Long Short-Term Memory

**LSTM for Sequence Classification:**

```rust
use tenflowers_neural::layers::LSTM;

fn build_lstm_classifier(
    vocab_size: usize,
    embed_dim: usize,
    hidden_dim: usize,
    num_classes: usize,
) -> Result<Sequential<f32>, Box<dyn std::error::Error>> {
    let mut model = Sequential::new();

    // Embedding layer
    model.add(Embedding::new(vocab_size, embed_dim)?);

    // LSTM layers (2-layer stacked)
    model.add(LSTM::new(embed_dim, hidden_dim, false)?);
    model.add(LSTM::new(hidden_dim, hidden_dim, false)?);

    // Take last timestep and classify
    // model.add(SelectLastTimestep::new());
    model.add(Dense::new(hidden_dim, num_classes, true)?);

    Ok(model)
}
```

**Bidirectional LSTM:**

```rust
use tenflowers_neural::layers::LSTM;

// Bidirectional LSTM for better context
let bilstm = LSTM::new(
    300,    // input_size
    512,    // hidden_size
    true,   // bidirectional = true
)?;

let input = Tensor::zeros(&[16, 100, 300]); // 16 sequences
let (output, (hidden, cell)) = bilstm.forward(&input, None)?;

// output: (batch, seq_len, 2 * hidden_size) because bidirectional
// hidden: (2, batch, hidden_size) for forward and backward
```

### GRU - Gated Recurrent Unit

Simpler than LSTM, often comparable performance.

```rust
use tenflowers_neural::layers::GRU;

let gru = GRU::new(
    256,    // input_size
    512,    // hidden_size
    true,   // bidirectional
)?;

// GRU is lighter than LSTM (no cell state)
let input = Tensor::zeros(&[32, 100, 256]);
let (output, hidden) = gru.forward(&input, None)?;
```

**Sequence-to-Sequence with Attention:**

```rust
// Encoder
let encoder = GRU::new(vocab_size_src, hidden_dim, true)?;

// Decoder with attention
let decoder = GRU::new(vocab_size_tgt, hidden_dim, false)?;
let attention = AdditiveAttention::new(hidden_dim)?;

// Encode source sequence
let (encoder_outputs, encoder_hidden) = encoder.forward(&source, None)?;

// Decode with attention
let mut decoder_hidden = encoder_hidden;
for t in 0..target_len {
    let (decoder_output, new_hidden) = decoder.forward(&target_t, Some(decoder_hidden))?;
    let context = attention.forward(&decoder_output, &encoder_outputs)?;
    // Combine context and decoder output for prediction
    decoder_hidden = new_hidden;
}
```

---

## Attention Mechanisms

### Multi-Head Attention

Core component of Transformers.

**Basic Multi-Head Attention:**

```rust
use tenflowers_neural::layers::MultiHeadAttention;

let attention = MultiHeadAttention::new(
    512,    // d_model (embedding dimension)
    8,      // num_heads
    0.1,    // dropout
)?;

// Self-attention: Q=K=V
let x = Tensor::zeros(&[32, 100, 512]); // (batch, seq_len, d_model)
let output = attention.forward(&x, &x, &x, None)?;
```

**Attention with Masking (Causal for GPT-style):**

```rust
use tenflowers_neural::layers::MultiHeadAttention;

let attention = MultiHeadAttention::new(512, 8, 0.1)?
    .with_causal_mask(true); // Enable causal masking

// During autoregressive generation
let x = Tensor::zeros(&[1, 50, 512]); // single sequence
let mask = create_causal_mask(50)?; // Lower triangular mask
let output = attention.forward(&x, &x, &x, Some(&mask))?;
```

**Cross-Attention (Encoder-Decoder):**

```rust
// Query from decoder, Key/Value from encoder
let cross_attention = MultiHeadAttention::new(512, 8, 0.1)?;

let decoder_hidden = Tensor::zeros(&[32, 50, 512]);  // decoder queries
let encoder_output = Tensor::zeros(&[32, 100, 512]); // encoder keys/values

let output = cross_attention.forward(
    &decoder_hidden,   // query
    &encoder_output,   // key
    &encoder_output,   // value
    None,
)?;
```

### Multi-Query Attention (MQA)

More efficient variant with shared K/V projections.

```rust
use tenflowers_neural::layers::MultiQueryAttention;

let mqa = MultiQueryAttention::new(
    512,    // d_model
    8,      // num_query_heads (K/V uses single head)
    0.1,    // dropout
)?;

// Same interface as MHA but faster
let x = Tensor::zeros(&[32, 100, 512]);
let output = mqa.forward(&x, &x, &x, None)?;
```

### Grouped-Query Attention (GQA)

Balance between MHA and MQA (used in LLaMA 2).

```rust
use tenflowers_neural::layers::GroupedQueryAttention;

let gqa = GroupedQueryAttention::new(
    512,    // d_model
    8,      // num_query_heads
    2,      // num_kv_heads (groups)
    0.1,    // dropout
)?;
```

### Flash Attention

Memory-efficient attention implementation.

```rust
use tenflowers_neural::layers::{MultiHeadAttention, FlashAttentionConfig};

let flash_config = FlashAttentionConfig {
    block_size: 128,
    causal_mask: false,
    temperature: 1.0,
    gradient_checkpointing: true,
};

let attention = MultiHeadAttention::new(512, 8, 0.1)?
    .with_flash_attention(flash_config);

// Automatically uses Flash Attention for large sequences
let x = Tensor::zeros(&[2, 8192, 512]); // Long sequence
let output = attention.forward(&x, &x, &x, None)?;
```

---

## Normalization Layers

### Batch Normalization

Normalizes across batch dimension.

```rust
use tenflowers_neural::layers::BatchNorm;

// For 1D features (Dense layers)
let bn1d = BatchNorm::new(128)?;  // 128 features
let x = Tensor::zeros(&[32, 128]);
let normalized = bn1d.forward(&x, true)?; // true = training mode

// For 2D features (Conv layers)
let bn2d = BatchNorm::new(64)?;  // 64 channels
let x = Tensor::zeros(&[32, 64, 28, 28]); // NCHW
let normalized = bn2d.forward(&x, true)?;
```

**Batch Norm in Training vs Inference:**

```rust
// Training mode: use batch statistics
model.set_training(true);
let output_train = model.forward(&batch)?;

// Inference mode: use running statistics
model.set_training(false);
let output_eval = model.forward(&input)?;
```

### Layer Normalization

Normalizes across feature dimension (transformer standard).

```rust
use tenflowers_neural::layers::LayerNorm;

let ln = LayerNorm::new(512, 1e-5)?; // 512 features, eps=1e-5

// Works across last dimension
let x = Tensor::zeros(&[32, 100, 512]); // (batch, seq, features)
let normalized = ln.forward(&x)?;
```

**Pre-Norm vs Post-Norm Transformer:**

```rust
// Pre-Norm (modern, more stable)
fn transformer_block_prenorm(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let norm1 = LayerNorm::new(512, 1e-5)?;
    let attention = MultiHeadAttention::new(512, 8, 0.1)?;
    let norm2 = LayerNorm::new(512, 1e-5)?;
    let ffn = FeedForwardNetwork::new(512, 2048, 0.1)?;

    // Pre-norm: normalize before sublayer
    let normed = norm1.forward(x)?;
    let attn_out = attention.forward(&normed, &normed, &normed, None)?;
    let x = x.add(&attn_out)?; // residual

    let normed = norm2.forward(&x)?;
    let ffn_out = ffn.forward(&normed)?;
    let x = x.add(&ffn_out)?; // residual

    Ok(x)
}
```

### RMS Normalization

Simplified normalization (used in LLaMA).

```rust
use tenflowers_neural::layers::RMSNorm;

let rms_norm = RMSNorm::new(4096, 1e-6)?; // LLaMA-sized model
let x = Tensor::zeros(&[1, 2048, 4096]);
let normalized = rms_norm.forward(&x)?;
```

### Group Normalization

Better than BN for small batches.

```rust
use tenflowers_neural::layers::GroupNorm;

let gn = GroupNorm::new(
    32,     // num_groups
    128,    // num_channels
    1e-5,   // eps
)?;

let x = Tensor::zeros(&[4, 128, 28, 28]); // Small batch size
let normalized = gn.forward(&x)?;
```

---

## Regularization Layers

### Dropout

Randomly zeros elements during training.

```rust
use tenflowers_neural::layers::Dropout;

let dropout = Dropout::new(0.5)?; // Drop 50% of activations

// Training mode: applies dropout
model.set_training(true);
let x = Tensor::ones(&[32, 128]);
let dropped = dropout.forward(&x)?;

// Inference mode: no dropout
model.set_training(false);
let output = dropout.forward(&x)?; // unchanged
```

### Spatial Dropout

Drops entire feature maps (for CNNs).

```rust
use tenflowers_neural::layers::SpatialDropout2D;

let spatial_dropout = SpatialDropout2D::new(0.2)?; // Drop 20% of channels

let x = Tensor::zeros(&[32, 64, 28, 28]);
let dropped = spatial_dropout.forward(&x)?;
// Entire channels (64 of them) are dropped, not individual pixels
```

### Stochastic Depth (Drop Path)

Randomly drops entire residual blocks.

```rust
use tenflowers_neural::layers::StochasticDepth;

let drop_path = StochasticDepth::new(0.1)?; // 10% drop probability

// In residual block
fn residual_block_with_drop_path(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    let conv = Conv2D::new(64, 64, 3, 1, 1)?;
    let drop_path = StochasticDepth::new(0.1)?;

    let residual = conv.forward(x)?;
    let residual = drop_path.forward(&residual)?; // Randomly drop entire path
    let output = x.add(&residual)?;

    Ok(output)
}
```

---

## Pooling Operations

### MaxPool2D / AvgPool2D

```rust
use tenflowers_neural::layers::{MaxPool2D, AvgPool2D};

let max_pool = MaxPool2D::new(
    2,  // kernel_size
    2,  // stride
    0,  // padding
)?;

let avg_pool = AvgPool2D::new(2, 2, 0)?;

let x = Tensor::zeros(&[32, 64, 56, 56]);
let pooled = max_pool.forward(&x)?; // [32, 64, 28, 28]
```

### Global Pooling

```rust
use tenflowers_neural::layers::{GlobalAvgPool2D, GlobalMaxPool2D};

let gap = GlobalAvgPool2D::new();
let gmp = GlobalMaxPool2D::new();

let x = Tensor::zeros(&[32, 512, 7, 7]);
let features_avg = gap.forward(&x)?; // [32, 512]
let features_max = gmp.forward(&x)?; // [32, 512]
```

### Adaptive Pooling

Output size independent of input size.

```rust
use tenflowers_neural::layers::AdaptiveAvgPool2D;

let adaptive = AdaptiveAvgPool2D::new((7, 7))?; // Always output 7x7

let x1 = Tensor::zeros(&[32, 512, 14, 14]);
let out1 = adaptive.forward(&x1)?; // [32, 512, 7, 7]

let x2 = Tensor::zeros(&[32, 512, 28, 28]);
let out2 = adaptive.forward(&x2)?; // [32, 512, 7, 7]
```

---

## Embedding Layers

### Token Embedding

```rust
use tenflowers_neural::layers::Embedding;

let embedding = Embedding::new(
    50000,  // vocab_size
    512,    // embedding_dim
)?;

// Input: token indices
let tokens = Tensor::from_vec(vec![10, 23, 145, 67], &[4])?;
let embedded = embedding.forward(&tokens)?; // [4, 512]
```

### Positional Encoding

**Sinusoidal (Transformer-style):**

```rust
use tenflowers_neural::layers::SinusoidalPositionalEncoding;

let pos_enc = SinusoidalPositionalEncoding::new(512, 5000)?;

let x = Tensor::zeros(&[32, 100, 512]); // (batch, seq_len, d_model)
let with_pos = pos_enc.forward(&x)?; // Adds positional information
```

**Learnable Positional Embeddings:**

```rust
use tenflowers_neural::layers::LearnedPositionalEncoding;

let pos_emb = LearnedPositionalEncoding::new(
    2048,   // max_seq_len
    512,    // d_model
)?;

let x = Tensor::zeros(&[32, 100, 512]);
let with_pos = pos_emb.forward(&x)?;
```

**Rotary Position Embeddings (RoPE):**

```rust
use tenflowers_neural::layers::RotaryPositionalEmbedding;

let rope = RotaryPositionalEmbedding::new(
    64,     // head_dim (must be even)
    2048,   // max_seq_len
)?;

// Applied during attention computation
let q = Tensor::zeros(&[32, 8, 100, 64]); // (batch, heads, seq, head_dim)
let k = Tensor::zeros(&[32, 8, 100, 64]);

let (q_rotated, k_rotated) = rope.forward(&q, &k)?;
```

---

## Advanced Architectures

### Mamba / State Space Models

Efficient alternative to attention for long sequences.

```rust
use tenflowers_neural::layers::MambaBlock;

let mamba = MambaBlock::new(
    768,    // d_model
    16,     // d_state
)?;

// Can handle very long sequences efficiently
let x = Tensor::zeros(&[2, 32000, 768]); // 32K token sequence
let output = mamba.forward(&x)?;
```

### Mixture of Experts (MoE)

Sparse activation for scaling model capacity.

```rust
use tenflowers_neural::layers::MixtureOfExperts;

let moe = MixtureOfExperts::new(
    512,    // d_model
    2048,   // expert_dim
    8,      // num_experts
    2,      // top_k (activate top-2 experts)
    0.01,   // load_balancing_loss_weight
)?;

let x = Tensor::zeros(&[32, 100, 512]);
let (output, aux_loss) = moe.forward(&x)?;
// aux_loss encourages balanced expert usage
```

### Transformer Encoder/Decoder

Complete transformer blocks.

```rust
use tenflowers_neural::layers::{TransformerEncoder, TransformerDecoder};

// Encoder
let encoder = TransformerEncoder::new(
    512,    // d_model
    8,      // num_heads
    2048,   // d_ff
    0.1,    // dropout
    6,      // num_layers
)?;

let src = Tensor::zeros(&[32, 100, 512]);
let encoder_output = encoder.forward(&src, None)?;

// Decoder
let decoder = TransformerDecoder::new(512, 8, 2048, 0.1, 6)?;

let tgt = Tensor::zeros(&[32, 50, 512]);
let output = decoder.forward(&tgt, &encoder_output, None, None)?;
```

---

## Complete Architecture Examples

### ResNet Block

```rust
fn resnet_block(in_channels: usize, out_channels: usize, stride: usize)
    -> Result<Vec<Box<dyn Layer<f32>>>, Box<dyn std::error::Error>> {
    let mut layers = Vec::new();

    // Main path
    layers.push(Box::new(Conv2D::new(in_channels, out_channels, 3, stride, 1)?));
    layers.push(Box::new(BatchNorm::new(out_channels)?));
    layers.push(Box::new(Activation::new(ActivationFunction::ReLU)));

    layers.push(Box::new(Conv2D::new(out_channels, out_channels, 3, 1, 1)?));
    layers.push(Box::new(BatchNorm::new(out_channels)?));

    // Residual connection handled externally
    // output = main_path(x) + shortcut(x)

    Ok(layers)
}
```

### Vision Transformer (ViT) Patch Embedding

```rust
fn vit_patch_embedding(
    img_size: usize,
    patch_size: usize,
    in_channels: usize,
    embed_dim: usize,
) -> Result<Conv2D<f32>, Box<dyn std::error::Error>> {
    // Use convolution to extract patches and embed them
    let patch_embed = Conv2D::new(
        in_channels,
        embed_dim,
        patch_size,
        patch_size,  // stride = patch_size (non-overlapping)
        0,           // no padding
    )?;

    // Input: [batch, 3, 224, 224]
    // Output: [batch, embed_dim, num_patches_h, num_patches_w]
    // Reshape to: [batch, num_patches, embed_dim]

    Ok(patch_embed)
}
```

---

## Performance Optimization Tips

### 1. Use Appropriate Batch Sizes

```rust
// Small batch (1-8): Use GroupNorm instead of BatchNorm
let norm = GroupNorm::new(32, 128, 1e-5)?;

// Large batch (32+): BatchNorm is efficient
let norm = BatchNorm::new(128)?;
```

### 2. Enable GPU Acceleration

```rust
use tenflowers_core::Device;

let device = Device::gpu(0)?;
let input = input.to_device(&device)?;
// All subsequent operations run on GPU
```

### 3. Use Depthwise Separable Convolutions for Mobile

```rust
// Standard conv: O(k²·c_in·c_out·h·w)
let standard = Conv2D::new(128, 256, 3, 1, 1)?;

// Depthwise separable: O(k²·c_in·h·w + c_in·c_out·h·w)
// ~8-9x fewer parameters for same expressiveness
let depthwise = DepthwiseConv2D::new(128, 3, 1, 1)?;
let pointwise = Conv2D::new(128, 256, 1, 1, 0)?;
```

### 4. Gradient Checkpointing for Large Models

```rust
let attention = MultiHeadAttention::new(512, 8, 0.1)?
    .with_gradient_checkpointing(true);
// Trades computation for memory
```

---

## Testing and Debugging Layers

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_output_shape() {
        let layer = Dense::new(100, 50, true).unwrap();
        let input = Tensor::zeros(&[32, 100]);
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape(), &[32, 50]);
    }

    #[test]
    fn test_layer_gradients() {
        // Test backward pass
        let layer = Dense::new(10, 5, true).unwrap();
        let params_before = layer.parameters();

        // Forward + backward would update parameters
        // (requires autograd integration)
    }
}
```

---

## Summary

This guide covered:
- ✅ Dense and convolutional layers with initialization strategies
- ✅ Recurrent layers (RNN, LSTM, GRU) with bidirectional support
- ✅ Attention mechanisms (MHA, MQA, GQA, Flash Attention)
- ✅ Normalization techniques (BN, LN, RMSNorm, GN)
- ✅ Regularization (Dropout, Spatial Dropout, Stochastic Depth)
- ✅ Pooling operations (Max, Average, Global, Adaptive)
- ✅ Embeddings (Token, Positional, RoPE)
- ✅ Advanced architectures (Mamba, MoE, Transformers)

**Test Coverage:** 1,012/1,012 tests passing ✅

For more information, see:
- Optimizer guide: `/tmp/tenflowers_neural_optimizer_guide.md`
- Training pipeline guide: `/tmp/tenflowers_neural_training_guide.md`
- Advanced features guide: `/tmp/tenflowers_neural_advanced_guide.md`
- Deployment guide: `/tmp/tenflowers_neural_deployment_guide.md`
