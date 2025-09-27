//! GPU Activation Operations
//!
//! This module provides GPU-accelerated activation functions commonly used
//! in neural networks, including ReLU, sigmoid, tanh, GELU, and others.

use super::super::*;
use super::basic_ops::execute_unary_op;
use super::operation_types::ActivationOp;
use crate::{Result, TensorError};

/// Execute an activation operation on GPU
pub fn execute_activation_op<T>(input: &GpuBuffer<T>, op: ActivationOp) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // Delegate to unary operation implementations
    let unary_op = match op {
        ActivationOp::ReLU => unary_ops::UnaryOp::ReLU,
        ActivationOp::Sigmoid => unary_ops::UnaryOp::Sigmoid,
        ActivationOp::Tanh => unary_ops::UnaryOp::Tanh,
        _ => {
            return Err(TensorError::InvalidOperation {
                operation: "activation".to_string(),
                reason: format!("GPU activation operation {:?} not implemented", op),
                context: None,
            })
        }
    };
    execute_unary_op(input, unary_op)
}
