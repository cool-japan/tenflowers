//! Operation types that can be recorded in the computation tape
//!
//! This module defines all the operations that the automatic differentiation system
//! can track and compute gradients for, organized by category.

use super::TensorId;

/// Operation types that can be recorded in the tape
#[derive(Debug, Clone)]
pub enum Operation {
    // Basic arithmetic operations
    Add {
        lhs: TensorId,
        rhs: TensorId,
    },
    Sub {
        lhs: TensorId,
        rhs: TensorId,
    },
    Mul {
        lhs: TensorId,
        rhs: TensorId,
    },
    Div {
        lhs: TensorId,
        rhs: TensorId,
    },
    Pow {
        lhs: TensorId,
        rhs: TensorId,
    },
    MatMul {
        lhs: TensorId,
        rhs: TensorId,
    },

    // Activation functions
    Relu {
        input: TensorId,
    },
    Sigmoid {
        input: TensorId,
    },
    Tanh {
        input: TensorId,
    },
    Gelu {
        input: TensorId,
    },
    Swish {
        input: TensorId,
    },
    Mish {
        input: TensorId,
    },
    LeakyRelu {
        input: TensorId,
        alpha: f32,
    },
    Elu {
        input: TensorId,
        alpha: f32,
    },
    Prelu {
        input: TensorId,
        alpha: TensorId,
    },
    Softmax {
        input: TensorId,
        axis: Option<i32>,
    },

    // Reduction operations
    Sum {
        input: TensorId,
        axes: Option<Vec<i32>>,
        keepdims: bool,
    },
    Mean {
        input: TensorId,
        axes: Option<Vec<i32>>,
        keepdims: bool,
    },
    Max {
        input: TensorId,
        axes: Option<Vec<i32>>,
        keepdims: bool,
    },
    Min {
        input: TensorId,
        axes: Option<Vec<i32>>,
        keepdims: bool,
    },
    Var {
        input: TensorId,
        axes: Option<Vec<i32>>,
        keepdims: bool,
        correction: usize,
    },
    Std {
        input: TensorId,
        axes: Option<Vec<i32>>,
        keepdims: bool,
        correction: usize,
    },

    // Unary operations
    Neg {
        input: TensorId,
    },
    Identity {
        input: TensorId,
    },

    // Constants
    Constant,

    // Tensor manipulation operations
    Reshape {
        input: TensorId,
        original_shape: Vec<usize>,
        new_shape: Vec<usize>,
    },
    Transpose {
        input: TensorId,
        axes: Option<Vec<usize>>,
    },
    Squeeze {
        input: TensorId,
        axes: Option<Vec<usize>>,
        original_shape: Vec<usize>,
    },
    Unsqueeze {
        input: TensorId,
        axes: Vec<usize>,
    },
    Slice {
        input: TensorId,
        slice_specs: Vec<crate::grad_ops::SliceSpec>,
        input_shape: Vec<usize>,
    },
    Concat {
        inputs: Vec<TensorId>,
        axis: i32,
        input_shapes: Vec<Vec<usize>>,
    },
    Stack {
        inputs: Vec<TensorId>,
        axis: i32,
    },
    Split {
        input: TensorId,
        sizes: Vec<usize>,
        axis: i32,
    },
    Einsum {
        inputs: Vec<TensorId>,
        equation: String,
        input_shapes: Vec<Vec<usize>>,
    },

    // Convolution operations
    Conv2D {
        input: TensorId,
        weight: TensorId,
        bias: Option<TensorId>,
        stride: (usize, usize),
        padding: String,
    },
    Conv3D {
        input: TensorId,
        weight: TensorId,
        bias: Option<TensorId>,
        stride: (usize, usize, usize),
        padding: String,
    },
    ConvTranspose2D {
        input: TensorId,
        weight: TensorId,
        bias: Option<TensorId>,
        stride: (usize, usize),
        padding: String,
        output_padding: (usize, usize),
    },
    DepthwiseConv2D {
        input: TensorId,
        weight: TensorId,
        bias: Option<TensorId>,
        stride: (usize, usize),
        padding: String,
        groups: usize,
    },
    GroupedConv2D {
        input: TensorId,
        weight: TensorId,
        bias: Option<TensorId>,
        stride: (usize, usize),
        padding: String,
        groups: usize,
    },

    // Pooling operations
    MaxPool2D {
        input: TensorId,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: String,
    },
    AvgPool2D {
        input: TensorId,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: String,
    },
    GlobalAvgPool2D {
        input: TensorId,
    },
    GlobalMaxPool2D {
        input: TensorId,
    },
    AdaptiveAvgPool2D {
        input: TensorId,
        output_size: (usize, usize),
    },
    AdaptiveMaxPool2D {
        input: TensorId,
        output_size: (usize, usize),
    },

    // Normalization operations
    BatchNorm {
        input: TensorId,
        gamma: TensorId,
        beta: TensorId,
        running_mean: TensorId,
        running_var: TensorId,
        epsilon: f32,
        training: bool,
    },
    LayerNorm {
        input: TensorId,
        gamma: TensorId,
        beta: TensorId,
        normalized_shape: Vec<usize>,
        epsilon: f32,
    },
    GroupNorm {
        input: TensorId,
        gamma: TensorId,
        beta: TensorId,
        num_groups: usize,
        epsilon: f32,
    },
    InstanceNorm {
        input: TensorId,
        gamma: TensorId,
        beta: TensorId,
        epsilon: f32,
    },

    // Indexing and masking operations
    BooleanMask {
        input: TensorId,
        mask: TensorId,
    },
    Where {
        condition: TensorId,
        x: TensorId,
        z: TensorId,
    },
    IntegerArrayIndexing {
        input: TensorId,
        indices: TensorId,
        axis: usize,
    },

    // FFT operations
    Fft {
        input: TensorId,
        n: Option<usize>,
        axis: i32,
        norm: Option<String>,
    },
    Ifft {
        input: TensorId,
        n: Option<usize>,
        axis: i32,
        norm: Option<String>,
    },
    Rfft {
        input: TensorId,
        n: Option<usize>,
        axis: i32,
        norm: Option<String>,
    },
    Fft2 {
        input: TensorId,
        s: Option<(usize, usize)>,
        axes: (i32, i32),
        norm: Option<String>,
    },
    Ifft2 {
        input: TensorId,
        s: Option<(usize, usize)>,
        axes: (i32, i32),
        norm: Option<String>,
    },
    Fft3 {
        input: TensorId,
        s: Option<(usize, usize, usize)>,
        axes: (i32, i32, i32),
        norm: Option<String>,
    },
    Ifft3 {
        input: TensorId,
        s: Option<(usize, usize, usize)>,
        axes: (i32, i32, i32),
        norm: Option<String>,
    },

    // Linear algebra operations
    Eig {
        input: TensorId,
    },
    Svd {
        input: TensorId,
    },
    Inv {
        input: TensorId,
    },
    Det {
        input: TensorId,
    },
    Cholesky {
        input: TensorId,
    },
    Lu {
        input: TensorId,
    },
    Pinv {
        input: TensorId,
    },

    // Special mathematical functions
    Gamma {
        input: TensorId,
    },
    Lgamma {
        input: TensorId,
    },
    Digamma {
        input: TensorId,
    },
    Erf {
        input: TensorId,
    },
    Erfc {
        input: TensorId,
    },
    BesselJ0 {
        input: TensorId,
    },
    BesselJ1 {
        input: TensorId,
    },
    Beta {
        a: TensorId,
        b: TensorId,
    },

    // Advanced operations
    StopGradient {
        input: TensorId,
    },

    // Fused operations for tape optimization
    FusedAddReLU {
        lhs: TensorId,
        rhs: TensorId,
    },
    FusedDense {
        input: TensorId,
        weight: TensorId,
        bias: Option<TensorId>,
    },
    FusedConvBatchNorm {
        input: TensorId,
        weight: TensorId,
        bias: Option<TensorId>,
        gamma: TensorId,
        beta: TensorId,
        running_mean: TensorId,
        running_var: TensorId,
        stride: (usize, usize),
        padding: String,
        epsilon: f32,
        training: bool,
    },
}
