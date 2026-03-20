//! Core data structures for the automatic differentiation tape
//!
//! This module defines the fundamental structures used to build and maintain
//! the computation graph for gradient tracking.

use super::{Operation, TensorId};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};
use tenflowers_core::Tensor;

/// Extract parent tensor IDs from an operation
pub fn extract_parent_ids(operation: &Operation) -> Vec<TensorId> {
    match operation {
        // Binary operations
        Operation::Add { lhs, rhs }
        | Operation::Sub { lhs, rhs }
        | Operation::Mul { lhs, rhs }
        | Operation::Div { lhs, rhs }
        | Operation::Pow { lhs, rhs }
        | Operation::MatMul { lhs, rhs }
        | Operation::Beta { a: lhs, b: rhs } => vec![*lhs, *rhs],

        // Single input operations
        Operation::Relu { input }
        | Operation::Sigmoid { input }
        | Operation::Tanh { input }
        | Operation::Gelu { input }
        | Operation::Swish { input }
        | Operation::Mish { input }
        | Operation::Neg { input }
        | Operation::Identity { input }
        | Operation::Eig { input }
        | Operation::Svd { input }
        | Operation::Inv { input }
        | Operation::Det { input }
        | Operation::Cholesky { input }
        | Operation::Lu { input }
        | Operation::Pinv { input }
        | Operation::Gamma { input }
        | Operation::Lgamma { input }
        | Operation::Digamma { input }
        | Operation::Erf { input }
        | Operation::Erfc { input }
        | Operation::BesselJ0 { input }
        | Operation::BesselJ1 { input }
        | Operation::StopGradient { input }
        | Operation::GlobalAvgPool2D { input }
        | Operation::GlobalMaxPool2D { input } => vec![*input],

        // Single input with parameters
        Operation::LeakyRelu { input, .. }
        | Operation::Elu { input, .. }
        | Operation::Softmax { input, .. }
        | Operation::Sum { input, .. }
        | Operation::Mean { input, .. }
        | Operation::Max { input, .. }
        | Operation::Min { input, .. }
        | Operation::Var { input, .. }
        | Operation::Std { input, .. }
        | Operation::Reshape { input, .. }
        | Operation::Transpose { input, .. }
        | Operation::Squeeze { input, .. }
        | Operation::Unsqueeze { input, .. }
        | Operation::Slice { input, .. }
        | Operation::Split { input, .. }
        | Operation::MaxPool2D { input, .. }
        | Operation::AvgPool2D { input, .. }
        | Operation::AdaptiveAvgPool2D { input, .. }
        | Operation::AdaptiveMaxPool2D { input, .. }
        | Operation::Fft { input, .. }
        | Operation::Ifft { input, .. }
        | Operation::Rfft { input, .. }
        | Operation::Fft2 { input, .. }
        | Operation::Ifft2 { input, .. }
        | Operation::Fft3 { input, .. }
        | Operation::Ifft3 { input, .. } => vec![*input],

        // Operations with two inputs where one might be a parameter tensor
        Operation::Prelu { input, alpha } => vec![*input, *alpha],

        // Multi-input operations
        Operation::Concat { inputs, .. }
        | Operation::Stack { inputs, .. }
        | Operation::Einsum { inputs, .. } => inputs.clone(),

        // Convolution operations
        Operation::Conv2D {
            input,
            weight,
            bias,
            ..
        }
        | Operation::Conv3D {
            input,
            weight,
            bias,
            ..
        }
        | Operation::ConvTranspose2D {
            input,
            weight,
            bias,
            ..
        }
        | Operation::DepthwiseConv2D {
            input,
            weight,
            bias,
            ..
        }
        | Operation::GroupedConv2D {
            input,
            weight,
            bias,
            ..
        } => {
            let mut parents = vec![*input, *weight];
            if let Some(bias_id) = bias {
                parents.push(*bias_id);
            }
            parents
        }

        // Normalization operations
        Operation::BatchNorm {
            input,
            gamma,
            beta,
            running_mean,
            running_var,
            ..
        } => {
            vec![*input, *gamma, *beta, *running_mean, *running_var]
        }
        Operation::LayerNorm {
            input, gamma, beta, ..
        }
        | Operation::GroupNorm {
            input, gamma, beta, ..
        }
        | Operation::InstanceNorm {
            input, gamma, beta, ..
        } => {
            vec![*input, *gamma, *beta]
        }

        // Indexing and masking operations
        Operation::BooleanMask { input, mask } => vec![*input, *mask],
        Operation::Where { condition, x, z } => vec![*condition, *x, *z],
        Operation::IntegerArrayIndexing { input, indices, .. } => vec![*input, *indices],

        // Fused operations
        Operation::FusedAddReLU { lhs, rhs } => vec![*lhs, *rhs],
        Operation::FusedDense {
            input,
            weight,
            bias,
        } => {
            let mut parents = vec![*input, *weight];
            if let Some(bias_id) = bias {
                parents.push(*bias_id);
            }
            parents
        }
        Operation::FusedConvBatchNorm {
            input,
            weight,
            bias,
            gamma,
            beta,
            running_mean,
            running_var,
            ..
        } => {
            let mut parents = vec![*input, *weight, *gamma, *beta, *running_mean, *running_var];
            if let Some(bias_id) = bias {
                parents.push(*bias_id);
            }
            parents
        }

        // Constants have no parents
        Operation::Constant => vec![],
    }
}

/// Node in the computation graph
#[derive(Debug, Clone)]
pub struct TapeNode {
    pub id: TensorId,
    pub operation: Operation,
    pub output_shape: Vec<usize>,
    pub requires_grad: bool,
    pub parents: Vec<TensorId>,
}

/// Reference to a tensor being tracked by the tape
#[derive(Debug, Clone)]
pub struct TrackedTensor<T> {
    pub tensor: Tensor<T>,
    pub id: TensorId,
    pub(crate) tape: Weak<Mutex<GradientTapeInner>>,
}

impl<T> TrackedTensor<T> {
    /// Get a reference to the underlying tensor
    pub fn tensor(&self) -> &Tensor<T> {
        &self.tensor
    }

    /// Get the shape of the tracked tensor
    pub fn shape(&self) -> &tenflowers_core::Shape {
        self.tensor.shape()
    }

    /// Create a new TrackedTensor without tracking (for temporary computations)
    pub fn new(tensor: Tensor<T>) -> Self {
        Self {
            tensor,
            id: 0,
            tape: Weak::new(),
        }
    }
}

/// Internal state of the gradient tape
#[derive(Debug)]
pub(crate) struct GradientTapeInner {
    pub(crate) nodes: Vec<TapeNode>,
    pub(crate) tensor_values: HashMap<TensorId, Box<dyn std::any::Any + Send + Sync>>,
    pub(crate) next_id: TensorId,
    pub(crate) is_recording: bool,
}

impl Drop for GradientTapeInner {
    fn drop(&mut self) {
        // Clear tensor_values and nodes safely
        self.tensor_values.clear();
        self.nodes.clear();
    }
}

impl GradientTapeInner {
    /// Record an operation in the tape
    pub fn record_op<T>(
        &mut self,
        operation: Operation,
        result: Tensor<T>,
        tape_ref: &Arc<Mutex<GradientTapeInner>>,
    ) -> TrackedTensor<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        let tensor_id = self.next_id;
        self.next_id += 1;

        // Create tape node
        let node = TapeNode {
            id: tensor_id,
            operation: operation.clone(),
            output_shape: result.shape().dims().to_vec(),
            requires_grad: true,
            parents: extract_parent_ids(&operation),
        };

        // Store the node
        if self.is_recording {
            self.nodes.push(node);
            // Store tensor value for gradient computation
            self.tensor_values
                .insert(tensor_id, Box::new(result.clone()));
        }

        TrackedTensor {
            tensor: result,
            id: tensor_id,
            tape: Arc::downgrade(tape_ref),
        }
    }
}

/// Gradient tape for automatic differentiation
#[derive(Debug, Clone)]
pub struct GradientTape {
    pub(crate) inner: Arc<Mutex<GradientTapeInner>>,
}

impl Default for GradientTape {
    fn default() -> Self {
        Self::new()
    }
}
