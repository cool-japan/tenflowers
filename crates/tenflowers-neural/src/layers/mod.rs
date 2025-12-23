//! Neural network layer implementations.
//!
//! This module provides a comprehensive collection of neural network layers that can be
//! composed to build complex models. All layers implement the [`Layer`] trait, providing
//! a consistent interface for forward and backward propagation.
//!
//! # Layer Categories
//!
//! ## Core Layers
//!
//! - [`Dense`]: Fully-connected (linear) layer with optional bias
//! - [`Conv1D`], [`Conv2D`], [`Conv3D`]: Convolutional layers for 1D, 2D, and 3D data
//! - [`ConvTranspose2D`]: Transposed convolution for upsampling
//!
//! ## Activation Layers
//!
//! - [`Activation`]: Standard activation functions (ReLU, Tanh, Sigmoid, etc.)
//! - [`PReLU`]: Parametric ReLU with learnable parameters
//! - [`SwiGLU`], [`GeGLU`]: Gated linear units for transformers
//!
//! ## Attention Mechanisms
//!
//! - [`MultiHeadAttention`]: Standard multi-head attention with Flash Attention support
//! - [`MultiQueryAttention`]: Efficient multi-query attention
//! - [`TransformerEncoder`], [`TransformerDecoder`]: Complete transformer blocks
//! - [`FeedForwardNetwork`]: Position-wise feed-forward network
//!
//! ## Normalization
//!
//! - [`BatchNorm`]: Batch normalization with running statistics
//! - [`LayerNorm`]: Layer normalization for transformers
//! - [`RMSNorm`]: Root mean square normalization (LLaMA style)
//! - [`GroupNorm`]: Group normalization for small batches
//! - [`InstanceNorm`]: Instance normalization for style transfer
//!
//! ## Recurrent Layers
//!
//! - [`RNN`]: Basic recurrent neural network
//! - [`LSTM`]: Long short-term memory with forget gates
//! - [`GRU`]: Gated recurrent unit
//!
//! ## Regularization
//!
//! - [`Dropout`]: Standard dropout for regularization
//! - [`SpatialDropout2D`]: Spatial dropout for convolutional layers
//! - [`StochasticDepth`]: Stochastic depth (drop path)
//!
//! ## Pooling Operations
//!
//! - [`MaxPool2D`], [`AvgPool2D`]: Standard 2D pooling
//! - [`GlobalMaxPool2D`], [`GlobalAvgPool2D`]: Global pooling
//! - [`AdaptiveAvgPool2D`]: Adaptive pooling to target size
//!
//! ## Embeddings
//!
//! - [`Embedding`]: Token embedding layer
//! - [`SinusoidalPositionalEncoding`]: Fixed positional encodings (Transformer)
//! - [`LearnedPositionalEncoding`]: Learnable positional encodings
//! - [`RotaryPositionalEmbedding`]: Rotary position embeddings (RoPE)
//!
//! ## Advanced Architectures
//!
//! - [`MambaBlock`]: Mamba state-space model block
//! - [`StateSpaceModel`]: Generic state-space model (S4, S5)
//! - [`MixtureOfExperts`]: Sparse mixture of experts layer
//! - [`GraphConv`]: Graph convolution for graph neural networks
//!
//! # Usage Examples
//!
//! ## Building a Simple Network
//!
//! ```rust,no_run
//! use tenflowers_neural::layers::{Dense, Activation};
//! use tenflowers_neural::ActivationFunction;
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let layer1 = Dense::new(784, 128)?;
//! let activation = Activation::new(ActivationFunction::ReLU);
//! let layer2 = Dense::new(128, 10)?;
//!
//! let input = Tensor::zeros(&[32, 784]);
//! let hidden = layer1.forward(&input)?;
//! let activated = activation.forward(&hidden)?;
//! let output = layer2.forward(&activated)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Convolutional Network
//!
//! ```rust,no_run
//! use tenflowers_neural::layers::{Conv2D, BatchNorm, MaxPool2D};
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let conv = Conv2D::new(3, 64, 3, 1, 1)?;  // in_channels, out_channels, kernel, stride, padding
//! let bn = BatchNorm::new(64)?;
//! let pool = MaxPool2D::new(2, 2, 0)?;      // kernel_size, stride, padding
//!
//! let input = Tensor::zeros(&[32, 3, 224, 224]); // NCHW format
//! let features = conv.forward(&input)?;
//! let normalized = bn.forward(&features, true)?; // training mode
//! let pooled = pool.forward(&normalized)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Transformer Block
//!
//! ```rust,no_run
//! use tenflowers_neural::layers::{MultiHeadAttention, LayerNorm, FeedForwardNetwork};
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let attention = MultiHeadAttention::new(512, 8, 0.1)?; // d_model, n_heads, dropout
//! let norm1 = LayerNorm::new(512, 1e-5)?;
//! let ffn = FeedForwardNetwork::new(512, 2048, 0.1)?;
//! let norm2 = LayerNorm::new(512, 1e-5)?;
//!
//! let x = Tensor::zeros(&[32, 128, 512]); // batch, seq_len, d_model
//! let attn_out = attention.forward(&x, &x, &x, None)?;
//! let x = norm1.forward(&(x.clone() + attn_out))?;
//! let ffn_out = ffn.forward(&x)?;
//! let output = norm2.forward(&(x + ffn_out))?;
//! # Ok(())
//! # }
//! ```
//!
//! ## State-Space Model (Mamba)
//!
//! ```rust,no_run
//! use tenflowers_neural::layers::MambaBlock;
//! use tenflowers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mamba = MambaBlock::new(512, 16)?; // d_model, d_state
//!
//! let input = Tensor::zeros(&[32, 1024, 512]); // batch, seq_len, d_model
//! let output = mamba.forward(&input)?;
//! # Ok(())
//! # }
//! ```

pub mod activation;
pub mod attention;
pub mod augmentation;
pub mod conv;
pub mod dense;
pub mod dropout;
pub mod embedding;
pub mod gnn;
pub mod moe;
pub mod normalization;
pub mod pooling;
pub mod rnn;
pub mod state_space;
pub mod stochastic_depth;
pub mod ultra_conv_simple;
pub mod ultra_dense_simple;
pub mod ultra_layer_manager_minimal;

pub use activation::{
    Activation, AdaptivePiecewiseLinear, AdaptivePolynomial, AdaptiveSwish, PReLU,
    ParametricSoftplus, SwiGLU as SwiGLUActivation,
};
pub use attention::{
    analyze_attention_patterns, apply_attention_mask, apply_rotary_position_embedding,
    create_causal_mask, create_padding_mask, scaled_dot_product_attention,
    sinusoidal_positional_encoding, AttentionStats, FeedForwardNetwork, GeGLU, KVCache,
    MultiHeadAttention, MultiQueryAttention, SwiGLU, TransformerDecoder, TransformerEncoder,
};
pub use augmentation::{CutMix, LabelSmoothing, Mixup};
pub use conv::{Conv1D, Conv2D, Conv3D, ConvTranspose2D, DepthwiseConv2D, SeparableConv2D};
pub use dense::Dense;
pub use dropout::{Dropout, SpatialDropout2D};
pub use embedding::{
    Embedding, EmbeddingRegularization, LearnedPositionalEncoding, RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding, SparseEmbedding, SparseEmbeddingGrad,
};
pub use gnn::{AggregatorType, GraphAttention, GraphConv, GraphSAGE};
pub use moe::{Expert, MixtureOfExperts, RoutingStats, TopKRouter};
pub use normalization::{
    BatchNorm, GroupNorm, InstanceNorm, LayerNorm, RMSNorm, SpectralNorm, WeightNorm,
};
pub use pooling::{
    AdaptiveAvgPool2D, AdaptiveMaxPool2D, AvgPool2D, AvgPool3D, FractionalAvgPool2D,
    FractionalMaxPool2D, GlobalAvgPool2D, GlobalAvgPool3D, GlobalMaxPool2D, GlobalMaxPool3D,
    MaxPool2D, MaxPool3D, ROIAlign2D, ROIPool2D,
};
pub use rnn::{
    BahdanauAttention, HierarchicalAttention, LuongAttention, LuongAttentionType,
    ResetGateVariation, GRU, LSTM, RNN,
};
pub use state_space::{MambaBlock, StateSpaceModel};
pub use stochastic_depth::{StochasticDepth, StochasticDepthNoResidual};
pub use ultra_conv_simple::{ultra_conv2d, ConvPerformanceMetrics, UltraConv2D, UltraConvConfig};
pub use ultra_dense_simple::{
    ultra_dense, ultra_dense_no_bias, DensePerformanceMetrics, UltraDense, UltraDenseConfig,
    UltraDenseExt,
};
pub use ultra_layer_manager_minimal::{
    create_ultra_layer_manager, global_ultra_layer_manager, LayerExecutionResult, LayerId,
    LayerMetrics, OptimizationReport, UltraLayerManager, UltraLayerManagerConfig,
    UltraPerformanceReport,
};

use tenflowers_core::{Result, Tensor};

/// Represents different types of neural network layers for ONNX export
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerType {
    Dense,
    Conv1D,
    Conv2D,
    Conv3D,
    ConvTranspose2D,
    MaxPool2D,
    AvgPool2D,
    GlobalMaxPool2D,
    GlobalAvgPool2D,
    BatchNorm,
    LayerNorm,
    RMSNorm,
    GroupNorm,
    Dropout,
    LSTM,
    GRU,
    RNN,
    MultiHeadAttention,
    TransformerEncoder,
    TransformerDecoder,
    Embedding,
    Activation,
    MixtureOfExperts,
    StateSpaceModel,
    MambaBlock,
    GraphConv,
    GraphSAGE,
    GraphAttention,
    Unknown,
}

impl LayerType {
    /// Convert layer type to ONNX operation type
    pub fn to_onnx_op_type(&self) -> &'static str {
        match self {
            LayerType::Dense => "MatMul",
            LayerType::Conv1D => "Conv",
            LayerType::Conv2D => "Conv",
            LayerType::Conv3D => "Conv",
            LayerType::ConvTranspose2D => "ConvTranspose",
            LayerType::MaxPool2D => "MaxPool",
            LayerType::AvgPool2D => "AveragePool",
            LayerType::GlobalMaxPool2D => "GlobalMaxPool",
            LayerType::GlobalAvgPool2D => "GlobalAveragePool",
            LayerType::BatchNorm => "BatchNormalization",
            LayerType::LayerNorm => "LayerNormalization",
            LayerType::RMSNorm => "LayerNormalization", // ONNX doesn't have native RMSNorm, use LayerNorm
            LayerType::GroupNorm => "GroupNormalization",
            LayerType::Dropout => "Dropout",
            LayerType::LSTM => "LSTM",
            LayerType::GRU => "GRU",
            LayerType::RNN => "RNN",
            LayerType::MultiHeadAttention => "MultiHeadAttention",
            LayerType::TransformerEncoder => "Transformer",
            LayerType::TransformerDecoder => "Transformer",
            LayerType::Embedding => "Gather",
            LayerType::Activation => "Relu", // Default, would need specific handling
            LayerType::MixtureOfExperts => "Identity", // Custom operation, needs special handling
            LayerType::StateSpaceModel => "Identity", // Custom State Space operation, needs special handling
            LayerType::MambaBlock => "Identity", // Custom Mamba operation, needs special handling
            LayerType::GraphConv => "Identity", // Custom Graph Convolution operation, needs special handling
            LayerType::GraphSAGE => "Identity", // Custom GraphSAGE operation, needs special handling
            LayerType::GraphAttention => "Identity", // Custom Graph Attention operation, needs special handling
            LayerType::Unknown => "Identity",
        }
    }
}

pub trait Layer<T> {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>>;
    fn parameters(&self) -> Vec<&Tensor<T>>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>>;
    fn set_training(&mut self, training: bool);
    fn clone_box(&self) -> Box<dyn Layer<T>>;

    /// Returns the type of this layer for ONNX export and introspection
    fn layer_type(&self) -> LayerType {
        LayerType::Unknown // Default implementation
    }

    /// Set weight tensor for layers that support weights
    /// Default implementation returns an error
    fn set_weight(&mut self, _weight: Tensor<T>) -> Result<()> {
        Err(tenflowers_core::TensorError::unsupported_operation_simple(
            "This layer type does not support weight setting".to_string(),
        ))
    }

    /// Set bias tensor for layers that support bias
    /// Default implementation returns an error
    fn set_bias(&mut self, _bias: Option<Tensor<T>>) -> Result<()> {
        Err(tenflowers_core::TensorError::unsupported_operation_simple(
            "This layer type does not support bias setting".to_string(),
        ))
    }
}
