//! Common Pattern Optimizations for Einstein Summation
//!
//! This module contains optimizations for frequently used einsum patterns
//! including batch operations, transpose operations, and trace computations.

use super::utils::{batch_transpose, cache_friendly_trace, compute_outer_product};
use crate::{Result, Tensor};
use num_traits::{One, Zero};

#[cfg(feature = "gpu")]
use super::gpu::{
    gpu_einsum_batched_matmul, gpu_einsum_diagonal, gpu_einsum_matmul, gpu_einsum_outer_product,
    gpu_einsum_trace, gpu_einsum_transpose, gpu_einsum_vector_dot,
};

/// Try to optimize common einsum patterns
pub fn try_optimize_common_patterns<T>(
    equation: &str,
    operands: &[&Tensor<T>],
) -> Option<Result<Tensor<T>>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // Check if this is a GPU operation that we can optimize
    #[cfg(feature = "gpu")]
    let is_gpu = operands
        .iter()
        .any(|op| matches!(&op.storage, crate::tensor::TensorStorage::Gpu(_)));
    #[cfg(not(feature = "gpu"))]
    let is_gpu = false;

    if is_gpu {
        return try_optimize_gpu_patterns(equation, operands);
    }

    match equation {
        // Batch matrix multiplication patterns
        "bij,bjk->bik" | "bik,bkj->bij" if operands.len() == 2 => {
            Some(crate::ops::matmul(operands[0], operands[1]))
        }
        // Trace operation with cache-friendly access
        "ii->" if operands.len() == 1 => Some(cache_friendly_trace(operands[0])),
        // Inner product (vectorized)
        "i,i->" if operands.len() == 2 => Some(crate::ops::dot(operands[0], operands[1])),
        // Outer product
        "i,j->ij" if operands.len() == 2 => Some(compute_outer_product(operands[0], operands[1])),
        // Batch transpose
        "bij->bji" if operands.len() == 1 => Some(batch_transpose(operands[0])),
        _ => None,
    }
}

/// Try to optimize GPU einsum patterns
#[cfg(feature = "gpu")]
pub fn try_optimize_gpu_patterns<T>(
    equation: &str,
    operands: &[&Tensor<T>],
) -> Option<Result<Tensor<T>>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match equation {
        // Matrix multiplication: "ij,jk->ik"
        "ij,jk->ik" if operands.len() == 2 => Some(gpu_einsum_matmul(operands[0], operands[1])),
        // Batched matrix multiplication: "bij,bjk->bik"
        "bij,bjk->bik" if operands.len() == 2 => {
            Some(gpu_einsum_batched_matmul(operands[0], operands[1]))
        }
        // Transpose: "ij->ji"
        "ij->ji" if operands.len() == 1 => Some(gpu_einsum_transpose(operands[0])),
        // Diagonal extraction: "ii->i"
        "ii->i" if operands.len() == 1 => Some(gpu_einsum_diagonal(operands[0])),
        // Element-wise multiplication: "ij,ij->ij"
        eq if operands.len() == 2 && eq.starts_with("ij,ij->ij") => {
            Some(operands[0].mul(operands[1]))
        }
        // Sum of element-wise multiplication: "ij,ij->"
        eq if operands.len() == 2 && eq.starts_with("ij,ij->") && eq.ends_with("->") => {
            let elementwise = operands[0].mul(operands[1]);
            Some(elementwise.and_then(|t| crate::ops::sum(&t, None, false)))
        }
        // Outer product: "i,j->ij"
        "i,j->ij" if operands.len() == 2 => {
            Some(gpu_einsum_outer_product(operands[0], operands[1]))
        }
        // Vector dot product: "i,i->"
        "i,i->" if operands.len() == 2 => Some(gpu_einsum_vector_dot(operands[0], operands[1])),
        // Trace: "ii->"
        "ii->" if operands.len() == 1 => Some(gpu_einsum_trace(operands[0])),
        _ => None,
    }
}

#[cfg(not(feature = "gpu"))]
pub fn try_optimize_gpu_patterns<T>(
    _equation: &str,
    _operands: &[&Tensor<T>],
) -> Option<Result<Tensor<T>>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    None
}
