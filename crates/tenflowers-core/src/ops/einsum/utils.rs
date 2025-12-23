//! Utility Functions for Einstein Summation
//!
//! This module contains utility functions for einsum operations including
//! optimal path computation, cost estimation, and tensor manipulation utilities.

use crate::{Result, Tensor, TensorError};
use scirs2_core::numeric::{One, Zero};
use std::collections::HashSet;

/// Compute optimal contraction path for multi-operand einsum
pub(super) fn compute_optimal_path(
    input_subscripts: &[String],
    _output_subscript: &str,
) -> Result<Vec<(usize, usize)>> {
    let n = input_subscripts.len();
    if n <= 2 {
        // For small cases, simple path is fine
        let mut path = Vec::new();
        for i in 1..n {
            path.push((0, i));
        }
        return Ok(path);
    }

    // Enhanced greedy algorithm that considers intermediate result sizes
    // This is a heuristic improvement over simple left-to-right contraction
    let mut remaining_indices: Vec<usize> = (0..n).collect();
    let mut path = Vec::new();

    while remaining_indices.len() > 1 {
        let mut best_pair = (0, 1);
        let mut best_score = f64::INFINITY;

        // Find the pair that minimizes the estimated cost
        for i in 0..remaining_indices.len() {
            for j in (i + 1)..remaining_indices.len() {
                let idx1 = remaining_indices[i];
                let idx2 = remaining_indices[j];

                // Estimate cost based on string length (proxy for tensor size)
                let cost =
                    estimate_contraction_cost(&input_subscripts[idx1], &input_subscripts[idx2]);

                if cost < best_score {
                    best_score = cost;
                    best_pair = (i, j);
                }
            }
        }

        let left_idx = remaining_indices[best_pair.0];
        let right_idx = remaining_indices[best_pair.1];
        path.push((left_idx, right_idx));

        // Remove the contracted indices and add a placeholder for the result
        remaining_indices.remove(best_pair.1); // Remove higher index first
        remaining_indices.remove(best_pair.0);
        remaining_indices.push(n + path.len() - 1); // Placeholder index for result
    }

    Ok(path)
}

/// Estimate the computational cost of contracting two subscripts
pub(super) fn estimate_contraction_cost(sub1: &str, sub2: &str) -> f64 {
    // Simple heuristic: longer subscripts generally mean larger tensors
    // In a real implementation, this would use actual tensor shapes
    let total_indices = sub1.len() + sub2.len();
    let unique_indices = sub1
        .chars()
        .chain(sub2.chars())
        .collect::<HashSet<_>>()
        .len();

    // Prefer contractions that eliminate more indices (reduce dimensionality)
    let eliminated_indices = total_indices - unique_indices;

    // Cost increases with total size but decreases with eliminated indices
    total_indices as f64 * 10.0 - eliminated_indices as f64 * 5.0
}

/// Convert flat index to multi-dimensional index
pub(super) fn flat_to_multi_index(flat_idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut multi_idx = Vec::with_capacity(shape.len());
    let mut remaining = flat_idx;

    for &dim_size in shape.iter().rev() {
        multi_idx.push(remaining % dim_size);
        remaining /= dim_size;
    }

    multi_idx.reverse();
    multi_idx
}

/// Determine optimal loop ordering for cache efficiency
#[allow(dead_code)]
pub(super) fn optimal_loop_ordering(shapes: &[&[usize]]) -> Vec<usize> {
    // Simple heuristic: process innermost dimensions first for better cache locality
    let max_dims = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut ordering = Vec::with_capacity(max_dims);

    // Reverse order to process innermost dimensions first (row-major order)
    for i in (0..max_dims).rev() {
        ordering.push(i);
    }

    ordering
}

/// Cache-friendly trace extraction with optimal memory access
pub fn cache_friendly_trace<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + std::ops::Add<Output = T> + Send + Sync + 'static,
{
    use crate::tensor::TensorStorage;

    match &tensor.storage {
        TensorStorage::Cpu(arr) => {
            let shape = tensor.shape().dims();
            if shape.len() != 2 || shape[0] != shape[1] {
                return Err(TensorError::invalid_argument(
                    "Trace requires square 2D matrix".to_string(),
                ));
            }

            let n = shape[0];
            let mut trace = T::zero();

            // Cache-friendly diagonal access with predictable stride
            let stride = n + 1; // Diagonal elements are at indices 0, n+1, 2(n+1), ...
            for i in 0..n {
                let idx = i * stride;
                if let Some(val) = arr.as_slice().and_then(|s| s.get(idx)) {
                    trace = trace + val.clone();
                }
            }

            Tensor::from_vec(vec![trace], &[])
        }
        #[cfg(feature = "gpu")]
        _ => {
            // Fall back to regular trace extraction
            extract_trace(tensor)
        }
    }
}

/// Extract trace of a matrix (sum of diagonal elements)
#[allow(dead_code)]
pub(super) fn extract_trace<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + std::ops::Add<Output = T> + Send + Sync + 'static,
{
    let shape = tensor.shape().dims();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(TensorError::invalid_argument(
            "Trace requires square 2D matrix".to_string(),
        ));
    }

    let n = shape[0];
    let mut trace = T::zero();

    for i in 0..n {
        if let Some(val) = tensor.get(&[i, i]) {
            trace = trace + val;
        }
    }

    Tensor::from_vec(vec![trace], &[])
}

/// Compute outer product of two 1D tensors
pub fn compute_outer_product<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    let a_shape = a.shape().dims();
    let b_shape = b.shape().dims();

    if a_shape.len() != 1 || b_shape.len() != 1 {
        return Err(TensorError::invalid_argument(
            "Outer product requires 1D tensors".to_string(),
        ));
    }

    let m = a_shape[0];
    let n = b_shape[0];

    // Optimized access to tensor data for better performance
    let a_vec = a.to_vec().map_err(|e| {
        TensorError::invalid_argument(format!("Failed to access tensor A data: {e}"))
    })?;
    let b_vec = b.to_vec().map_err(|e| {
        TensorError::invalid_argument(format!("Failed to access tensor B data: {e}"))
    })?;

    // Pre-allocate result vector with exact capacity
    let mut result_data = Vec::with_capacity(m * n);

    // Cache-friendly nested loop with direct vector access
    for a_val in a_vec.iter().take(m) {
        for b_val in b_vec.iter().take(n) {
            result_data.push(*a_val * *b_val);
        }
    }

    Tensor::from_vec(result_data, &[m, n])
}

/// Batch transpose operation
pub fn batch_transpose<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    let shape = tensor.shape().dims();
    if shape.len() != 3 {
        return Err(TensorError::invalid_argument(
            "Batch transpose requires 3D tensor".to_string(),
        ));
    }

    // Transpose the last two dimensions of each batch
    let permutation = vec![0, 2, 1]; // (B, H, W) -> (B, W, H)
    crate::ops::manipulation::transpose_axes(tensor, Some(&permutation))
}

/// Get intermediate subscript for multi-operand einsum
#[allow(dead_code)]
pub(super) fn get_intermediate_subscript(left: &str, right: &str) -> Result<String> {
    let mut chars = Vec::new();
    let left_chars: Vec<char> = left.chars().collect();
    let right_chars: Vec<char> = right.chars().collect();

    // Add all unique characters from both subscripts
    for &c in &left_chars {
        if !chars.contains(&c) {
            chars.push(c);
        }
    }
    for &c in &right_chars {
        if !chars.contains(&c) {
            chars.push(c);
        }
    }

    chars.sort();
    Ok(chars.into_iter().collect())
}

/// Get result subscript for tensor
#[allow(dead_code)]
pub(super) fn get_result_subscript<T>(tensor: &Tensor<T>) -> Result<String> {
    let rank = tensor.shape().rank();
    let chars: String = (0..rank).map(|i| char::from(b'a' + (i as u8))).collect();
    Ok(chars)
}
