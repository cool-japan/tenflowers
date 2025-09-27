//! Cache-Friendly Operations for Einstein Summation
//!
//! This module contains cache-optimized implementations for tensor operations
//! with intelligent memory access patterns and parallel processing.

use super::utils::flat_to_multi_index;
use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use num_traits::{One, Zero};

/// Execute a contraction path with cache-friendly optimizations
pub fn execute_contraction_path<T>(
    operands: &[&Tensor<T>],
    path: &[(usize, usize)],
) -> Result<Tensor<T>>
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
    // Cache-optimized contraction execution
    let mut intermediate_tensors = operands.iter().map(|&t| t.clone()).collect::<Vec<_>>();

    for &(left_idx, right_idx) in path {
        if left_idx >= intermediate_tensors.len() || right_idx >= intermediate_tensors.len() {
            return Err(TensorError::invalid_argument(
                "Invalid contraction path indices".to_string(),
            ));
        }

        let left = &intermediate_tensors[left_idx];
        let right = &intermediate_tensors[right_idx];

        // Use cache-optimized contraction
        let contracted = cache_optimized_contraction(left, right)?;

        // Remove the contracted tensors (remove higher index first)
        if right_idx > left_idx {
            intermediate_tensors.remove(right_idx);
            intermediate_tensors.remove(left_idx);
        } else {
            intermediate_tensors.remove(left_idx);
            intermediate_tensors.remove(right_idx);
        }

        // Add the result
        intermediate_tensors.push(contracted);
    }

    if intermediate_tensors.len() != 1 {
        return Err(TensorError::invalid_argument(
            "Invalid contraction path: should result in single tensor".to_string(),
        ));
    }

    Ok(intermediate_tensors.into_iter().next().unwrap())
}

/// Cache-optimized tensor contraction with intelligent loop ordering
pub fn cache_optimized_contraction<T>(left: &Tensor<T>, right: &Tensor<T>) -> Result<Tensor<T>>
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
    let _left_shape = left.shape().dims();
    let _right_shape = right.shape().dims();

    // Use cache-friendly access patterns for CPU tensors
    match (&left.storage, &right.storage) {
        (TensorStorage::Cpu(_), TensorStorage::Cpu(_)) => {
            // Determine optimal loop ordering based on memory layout
            cache_friendly_cpu_contraction(left, right)
        }
        #[cfg(feature = "gpu")]
        _ => {
            // For GPU tensors or mixed storage, use existing implementation
            left.mul(right)
        }
    }
}

/// Cache-friendly CPU tensor contraction with optimized loop ordering
pub(super) fn cache_friendly_cpu_contraction<T>(
    left: &Tensor<T>,
    right: &Tensor<T>,
) -> Result<Tensor<T>>
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
    let left_shape = left.shape().dims();
    let right_shape = right.shape().dims();

    // For same-shape tensors, use cache-friendly element-wise multiplication
    if left_shape == right_shape {
        return cache_friendly_elementwise_mul(left, right);
    }

    // For different shapes, analyze memory layout and use optimal ordering
    left.mul(right) // Fall back to broadcasting for now
}

/// Cache-friendly element-wise multiplication with optimal memory access patterns
pub(super) fn cache_friendly_elementwise_mul<T>(
    left: &Tensor<T>,
    right: &Tensor<T>,
) -> Result<Tensor<T>>
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
    match (&left.storage, &right.storage) {
        (TensorStorage::Cpu(left_arr), TensorStorage::Cpu(right_arr)) => {
            let shape = left.shape().dims();

            // For large tensors, use cache-friendly blocked processing
            if shape.iter().product::<usize>() > 16384 {
                // 16KB threshold
                cache_friendly_blocked_multiply(left_arr, right_arr, shape)
            } else {
                // For small tensors, use regular multiplication
                left.mul(right)
            }
        }
        #[cfg(feature = "gpu")]
        _ => left.mul(right),
    }
}

/// Cache-friendly blocked multiplication for large tensors with parallel processing
pub(super) fn cache_friendly_blocked_multiply<T>(
    left_arr: &scirs2_autograd::ndarray::ArrayD<T>,
    right_arr: &scirs2_autograd::ndarray::ArrayD<T>,
    shape: &[usize],
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static,
{
    let total_elements = shape.iter().product::<usize>();

    // Use cache-friendly block size (typically L1 cache size / element size)
    let block_size = 1024.min(total_elements);

    // For large tensors, use parallel processing
    if total_elements > 65536 {
        // 64KB threshold for parallel processing
        parallel_blocked_multiply(left_arr, right_arr, shape, block_size)
    } else {
        // Sequential processing for smaller tensors
        sequential_blocked_multiply(left_arr, right_arr, shape, block_size)
    }
}

/// Sequential blocked multiplication for smaller tensors
pub(super) fn sequential_blocked_multiply<T>(
    left_arr: &scirs2_autograd::ndarray::ArrayD<T>,
    right_arr: &scirs2_autograd::ndarray::ArrayD<T>,
    shape: &[usize],
    block_size: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static,
{
    let total_elements = shape.iter().product::<usize>();
    let mut result_data = Vec::with_capacity(total_elements);

    // Process in blocks to improve cache locality
    for block_start in (0..total_elements).step_by(block_size) {
        let block_end = (block_start + block_size).min(total_elements);

        // Convert flat indices to multi-dimensional indices and process
        for flat_idx in block_start..block_end {
            let multi_idx = flat_to_multi_index(flat_idx, shape);

            // Access elements in cache-friendly order
            let left_val = left_arr
                .get(multi_idx.as_slice())
                .unwrap_or(&T::zero())
                .clone();
            let right_val = right_arr
                .get(multi_idx.as_slice())
                .unwrap_or(&T::zero())
                .clone();

            result_data.push(left_val * right_val);
        }
    }

    Tensor::from_vec(result_data, shape)
}

/// Parallel blocked multiplication for large tensors
pub(super) fn parallel_blocked_multiply<T>(
    left_arr: &scirs2_autograd::ndarray::ArrayD<T>,
    right_arr: &scirs2_autograd::ndarray::ArrayD<T>,
    shape: &[usize],
    block_size: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + Zero
        + One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static,
{
    let total_elements = shape.iter().product::<usize>();

    // Calculate number of threads
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(8); // Limit to avoid thread overhead

    if num_threads <= 1 {
        return sequential_blocked_multiply(left_arr, right_arr, shape, block_size);
    }

    // Divide work among threads
    let elements_per_thread = (total_elements + num_threads - 1) / num_threads;
    let mut result_data = vec![T::zero(); total_elements];

    // Use scoped threads to process chunks in parallel
    std::thread::scope(|s| {
        let mut handles = Vec::new();

        for thread_id in 0..num_threads {
            let start_idx = thread_id * elements_per_thread;
            let end_idx = (start_idx + elements_per_thread).min(total_elements);

            if start_idx >= total_elements {
                break;
            }

            let handle = s.spawn(move || {
                let mut chunk_results = Vec::new();

                // Process this thread's range of elements
                for flat_idx in start_idx..end_idx {
                    let multi_idx = flat_to_multi_index(flat_idx, shape);

                    // Access elements in cache-friendly order
                    let left_val = left_arr
                        .get(multi_idx.as_slice())
                        .unwrap_or(&T::zero())
                        .clone();
                    let right_val = right_arr
                        .get(multi_idx.as_slice())
                        .unwrap_or(&T::zero())
                        .clone();

                    chunk_results.push((flat_idx, left_val * right_val));
                }

                chunk_results
            });

            handles.push(handle);
        }

        // Collect results from all threads
        for handle in handles {
            let chunk_results = handle.join().unwrap();
            for (idx, value) in chunk_results {
                result_data[idx] = value;
            }
        }
    });

    Tensor::from_vec(result_data, shape)
}
