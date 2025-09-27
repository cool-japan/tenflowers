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
    FeedForwardNetwork, GeGLU, KVCache, MultiHeadAttention, MultiQueryAttention, SwiGLU,
    TransformerDecoder, TransformerEncoder,
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
pub use rnn::{ResetGateVariation, GRU, LSTM, RNN};
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
