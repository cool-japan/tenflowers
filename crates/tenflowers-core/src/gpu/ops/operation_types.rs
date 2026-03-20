//! GPU Operation Type Definitions
//!
//! This module contains all the enum definitions for different types of GPU operations.
//! These types are used to dispatch to appropriate GPU kernels and configure operation
//! parameters for WGPU compute shaders.

/// Reduction operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    Sum,
    Mean,
    Max,
    Min,
    Product,
    InfNanDetection,
    ArgMax,
    ArgMin,
    All,
    Any,
    Prod,
    Variance,
    TopK,
}

/// Activation operation types for GPU kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationOp {
    ReLU,
    Sigmoid,
    Tanh,
    Swish,
    HardSwish,
    GELU,
    LeakyReLU,
    ELU,
    Softmax,
    Mish,
}

/// Comparison operation types for GPU kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Logical operation types for GPU kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalOp {
    And,
    Or,
    Xor,
    Not,
}

/// Pooling operation types for GPU kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingOp {
    MaxPool2D,
    AvgPool2D,
    MaxPool3D,
    AvgPool3D,
    GlobalMaxPool,
    GlobalAvgPool,
    GlobalMaxPool3D,
    GlobalAvgPool3D,
    AdaptiveAvgPool2D,
    AdaptiveMaxPool2D,
    FractionalMaxPool2D,
    FractionalAvgPool2D,
}
