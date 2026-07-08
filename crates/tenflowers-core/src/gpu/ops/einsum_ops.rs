//! GPU Einsum Operations
//!
//! This module provides GPU-accelerated Einstein summation (einsum) operations
//! for complex tensor contractions, matrix multiplications, and linear algebra.

use super::super::*;
use crate::{Result, TensorError};

/// Execute einsum matrix multiplication on GPU
pub fn execute_einsum_matmul<T>(
    lhs: &GpuBuffer<T>,
    rhs: &GpuBuffer<T>,
    _lhs_shape: &[usize],
    _rhs_shape: &[usize],
    output_len: usize,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // Einsum matrix multiplication follows standard matrix multiplication patterns.
    // Delegate to the existing GPU matrix multiplication implementation; this
    // handles einsum notations like "ij,jk->ik" (standard 2D matmul). Shapes are
    // validated by the caller, which also precomputes `output_len`.
    crate::gpu::ops::binary_ops::execute_binary_op(
        lhs,
        rhs,
        crate::gpu::binary_ops::BinaryOp::MatMul,
        output_len,
    )
}

/// Execute einsum batched matrix multiplication on GPU
pub fn execute_einsum_batched_matmul<T>(
    _lhs: &GpuBuffer<T>,
    _rhs: &GpuBuffer<T>,
    _lhs_shape: &[usize],
    _rhs_shape: &[usize],
    _output_len: usize,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // HONEST ERROR: batched matmul for "bij,bjk->bik" requires per-batch GEMM,
    // but the previous implementation simply delegated to a single 2D
    // `BinaryOp::MatMul` over the flattened buffers, ignoring the batch dimension
    // entirely and producing silently-wrong results. Return an honest error so
    // callers fall back to the CPU path instead of getting wrong numbers.
    Err(TensorError::unsupported_operation_simple(
        "GPU einsum batched matmul not correctly implemented".to_string(),
    ))
}

/// Execute einsum transpose operation on GPU
pub fn execute_einsum_transpose<T>(_input: &GpuBuffer<T>, _axes: &[usize]) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // HONEST ERROR: a correct einsum transpose ("ij->ji") needs the real input
    // shape to permute strides. The previous implementation passed a 1D
    // `&[input_len]` placeholder shape into `execute_transpose`, which cannot
    // produce a correct 2D transpose and yields silently-wrong data/shape.
    // Return an honest error so callers fall back to the CPU transpose.
    Err(TensorError::unsupported_operation_simple(
        "GPU einsum transpose not correctly implemented".to_string(),
    ))
}

/// Execute einsum diagonal operation on GPU
pub fn execute_einsum_diagonal<T>(
    _input: &GpuBuffer<T>,
    _input_shape: &[usize],
    _axes: &[usize],
    _output_len: usize,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // HONEST ERROR: extracting a matrix diagonal ("ii->i") requires gathering the
    // strided elements at indices 0, n+1, 2*(n+1), ... . The previous
    // implementation instead copied the *first* `output_len` contiguous elements
    // (the first matrix row), which is silently wrong for any n > 1. Return an
    // honest error so callers fall back instead of receiving the wrong diagonal.
    Err(TensorError::unsupported_operation_simple(
        "GPU einsum diagonal not correctly implemented".to_string(),
    ))
}

/// Execute einsum outer product on GPU
pub fn execute_einsum_outer_product<T>(
    _lhs: &GpuBuffer<T>,
    _rhs: &GpuBuffer<T>,
    _lhs_shape: &[usize],
    _rhs_shape: &[usize],
    _output_len: usize,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // HONEST ERROR: the outer product "i,j->ij" must produce output[i,j] = lhs[i] *
    // rhs[j] (an M*N matrix). The previous implementation called the element-wise
    // `BinaryOp::Mul`, which only computes lhs[k] * rhs[k] for matching indices and
    // does NOT broadcast into the outer grid - silently wrong shape and values.
    // Return an honest error so callers fall back instead of getting wrong numbers.
    Err(TensorError::unsupported_operation_simple(
        "GPU einsum outer product not correctly implemented".to_string(),
    ))
}

/// Execute einsum vector dot product on GPU
pub fn execute_einsum_vector_dot<T>(
    lhs: &GpuBuffer<T>,
    rhs: &GpuBuffer<T>,
    lhs_shape: &[usize],
    _rhs_shape: &[usize],
    output_len: usize,
) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // Einsum vector dot product like "i,i->" (sum of element-wise multiplication).
    // This is correct: multiply element-wise, then sum all elements to a scalar.

    // Step 1: Element-wise multiplication.
    let temp_result = crate::gpu::ops::binary_ops::execute_binary_op(
        lhs,
        rhs,
        crate::gpu::binary_ops::BinaryOp::Mul,
        lhs.len(), // Same size as input vectors
    )?;

    // Step 2: Sum reduction over all elements.
    let result = crate::gpu::ops::reduction_ops::execute_axis_reduction_op(
        &temp_result,
        super::operation_types::ReductionOp::Sum,
        lhs_shape,
        None,  // Sum over all axes
        false, // keep_dims
        output_len,
    )?;

    Ok(result)
}

/// Execute einsum trace operation on GPU
pub fn execute_einsum_trace<T>(_input: &GpuBuffer<T>, _axes: &[usize]) -> Result<GpuBuffer<T>>
where
    T: bytemuck::Pod + bytemuck::Zeroable + Clone + Send + Sync + 'static,
{
    // HONEST ERROR: the trace "ii->" must sum ONLY the diagonal elements
    // (indices 0, n+1, 2*(n+1), ...). The previous implementation ran a plain
    // `ReductionOp::Sum` over the whole flattened buffer, summing every element of
    // the matrix - a silently-wrong result (sum of all entries, not the trace).
    // Return an honest error so callers fall back instead of getting wrong numbers.
    Err(TensorError::unsupported_operation_simple(
        "GPU einsum trace not correctly implemented".to_string(),
    ))
}
