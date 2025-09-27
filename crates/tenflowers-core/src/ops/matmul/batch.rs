//! Batch Matrix Multiplication Operations
//!
//! This module handles batch processing of matrix multiplications,
//! including utilities for extracting and storing 2D slices from higher-dimensional tensors.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor};
use num_traits::Zero;
use scirs2_autograd::ndarray::{Array2, ArrayD, IxDyn};

use super::optimized::matmul_2d_optimized;
use super::shapes::compute_broadcast_indices;

/// Batch matrix multiplication implementation
pub fn matmul_batch<T>(
    a: &TensorStorage<T>,
    b: &TensorStorage<T>,
    result_shape: &[usize],
) -> Result<Tensor<T>>
where
    T: Clone
        + Zero
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + Default
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    match (a, b) {
        (TensorStorage::Cpu(a_arr), TensorStorage::Cpu(b_arr)) => {
            let total_batches: usize = result_shape[..result_shape.len() - 2].iter().product();

            // Create result tensor
            let mut result_arr = ArrayD::zeros(IxDyn(result_shape));

            // Process each batch
            for batch_idx in 0..total_batches {
                // Convert linear batch index to multi-dimensional indices
                let mut batch_indices = Vec::new();
                let mut temp_idx = batch_idx;
                for &dim in result_shape[..result_shape.len() - 2].iter().rev() {
                    batch_indices.push(temp_idx % dim);
                    temp_idx /= dim;
                }
                batch_indices.reverse();

                // Extract 2D slices for this batch
                let a_slice = extract_2d_slice(a_arr, &batch_indices, a_arr.shape(), result_shape)?;
                let b_slice = extract_2d_slice(b_arr, &batch_indices, b_arr.shape(), result_shape)?;

                // Perform 2D matrix multiplication
                let result_slice = matmul_2d_optimized(a_slice.view(), b_slice.view());

                // Store result back into batch
                store_2d_slice(&mut result_arr, &batch_indices, &result_slice);
            }

            Ok(Tensor::from_array(result_arr))
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(_), TensorStorage::Gpu(_)) => {
            // Delegate to GPU batch implementation
            super::gpu::matmul_batch_gpu(a, b, result_shape)
        }
        #[cfg(feature = "gpu")]
        _ => Err(crate::TensorError::invalid_operation_simple(
            "Device mismatch: both tensors must be on the same device".to_string(),
        )),
    }
}

/// Extract a 2D slice from a higher-dimensional tensor for a specific batch
pub fn extract_2d_slice<T: Clone + Zero>(
    arr: &ArrayD<T>,
    batch_indices: &[usize],
    arr_shape: &[usize],
    _result_shape: &[usize],
) -> Result<Array2<T>> {
    let arr_batch_shape = &arr_shape[..arr_shape.len() - 2];
    let broadcast_indices = compute_broadcast_indices(batch_indices, arr_batch_shape);

    let rows = arr_shape[arr_shape.len() - 2];
    let cols = arr_shape[arr_shape.len() - 1];

    let mut slice = Array2::zeros((rows, cols));

    for i in 0..rows {
        for j in 0..cols {
            let mut full_indices = broadcast_indices.clone();
            full_indices.push(i);
            full_indices.push(j);

            // Convert to IxDyn for indexing
            let idx = IxDyn(&full_indices);
            if let Some(val) = arr.get(idx) {
                slice[[i, j]] = val.clone();
            }
        }
    }

    Ok(slice)
}

/// Store a 2D matrix back into a specific batch of a higher-dimensional tensor
pub fn store_2d_slice<T: Clone>(arr: &mut ArrayD<T>, batch_indices: &[usize], mat: &Array2<T>) {
    let (rows, cols) = mat.dim();

    for i in 0..rows {
        for j in 0..cols {
            let mut full_indices = batch_indices.to_vec();
            full_indices.push(i);
            full_indices.push(j);

            let idx = IxDyn(&full_indices);
            if let Some(dest) = arr.get_mut(idx) {
                *dest = mat[[i, j]].clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_extract_2d_slice() {
        // Create a 3D tensor: 2x2x2
        let arr = ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();

        let batch_indices = vec![0];
        let slice = extract_2d_slice(&arr, &batch_indices, &[2, 2, 2], &[2, 2, 2]).unwrap();

        let expected = array![[1.0, 2.0], [3.0, 4.0]];
        assert_eq!(slice, expected);
    }

    #[test]
    fn test_store_2d_slice() {
        let mut arr = ArrayD::zeros(IxDyn(&[2, 2, 2]));
        let mat = array![[1.0, 2.0], [3.0, 4.0]];
        let batch_indices = vec![0];

        store_2d_slice(&mut arr, &batch_indices, &mat);

        assert_eq!(arr[[0, 0, 0]], 1.0);
        assert_eq!(arr[[0, 0, 1]], 2.0);
        assert_eq!(arr[[0, 1, 0]], 3.0);
        assert_eq!(arr[[0, 1, 1]], 4.0);
    }
}
