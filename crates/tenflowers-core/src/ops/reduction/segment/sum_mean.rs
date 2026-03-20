//! Segment sum and segment mean operations

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use rayon::prelude::*;

/// GPU stub for segment sum
#[cfg(feature = "gpu")]
pub(super) fn segment_sum_gpu<T>(
    data: &Tensor<T>,
    data_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    ids_gpu: &crate::gpu::buffer::GpuBuffer<i32>,
    num_segments: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + std::ops::Add<Output = T>
        + scirs2_core::num_traits::Zero
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // NOTE(v0.2): GPU implementation needs proper device/queue management
    Err(TensorError::unsupported_operation_simple(
        "GPU reduction operation not yet implemented".to_string(),
    ))
}

/// GPU stub for segment mean
#[cfg(feature = "gpu")]
pub(super) fn segment_mean_gpu<T>(
    data: &Tensor<T>,
    data_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    ids_gpu: &crate::gpu::buffer::GpuBuffer<i32>,
    num_segments: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    // NOTE(v0.2): GPU implementation needs proper device/queue management
    Err(TensorError::unsupported_operation_simple(
        "GPU reduction operation not yet implemented".to_string(),
    ))
}

/// Segmented sum operation for ragged tensor support
///
/// Computes the sum of elements within each segment defined by segment_ids.
///
/// # Arguments
/// * `data` - Input tensor containing the data to be reduced
/// * `segment_ids` - Tensor of non-negative integers that define segments. Must be sorted.
/// * `num_segments` - Total number of segments (maximum segment_id + 1)
///
/// # Returns
/// A tensor of shape `[num_segments]` containing the sum for each segment
pub fn segment_sum<T>(
    data: &Tensor<T>,
    segment_ids: &Tensor<i32>,
    num_segments: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + std::ops::Add<Output = T>
        + scirs2_core::num_traits::Zero
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    if data.shape().dims()[0] != segment_ids.shape().dims()[0] {
        return Err(TensorError::shape_mismatch(
            "segment_reduction",
            "data and segment_ids must have same first dimension",
            &format!(
                "data: {:?}, segment_ids: {:?}",
                data.shape().dims(),
                segment_ids.shape().dims()
            ),
        ));
    }

    match (&data.storage, &segment_ids.storage) {
        (TensorStorage::Cpu(data_arr), TensorStorage::Cpu(ids_arr)) => {
            let data_flat = data_arr
                .view()
                .into_shape_with_order([data_arr.len()])
                .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;
            let ids_flat = ids_arr
                .view()
                .into_shape_with_order([ids_arr.len()])
                .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;

            let mut result = vec![T::zero(); num_segments];

            if data_flat.len() > 1000 {
                let chunk_size = std::cmp::max(1, data_flat.len() / rayon::current_num_threads());
                let data_slice = data_flat.as_slice().expect("tensor should be contiguous");
                let ids_slice = ids_flat.as_slice().expect("tensor should be contiguous");
                let chunks: Vec<_> = data_slice
                    .chunks(chunk_size)
                    .zip(ids_slice.chunks(chunk_size))
                    .collect();

                let partial_results: Vec<Vec<T>> = chunks
                    .par_iter()
                    .map(|(data_chunk, ids_chunk)| {
                        let mut local_result = vec![T::zero(); num_segments];

                        for (data_val, &segment_id) in data_chunk.iter().zip(ids_chunk.iter()) {
                            if segment_id >= 0 && (segment_id as usize) < num_segments {
                                let idx = segment_id as usize;
                                local_result[idx] = local_result[idx] + *data_val;
                            }
                        }
                        local_result
                    })
                    .collect();

                for partial in partial_results {
                    for (i, val) in partial.into_iter().enumerate() {
                        result[i] = result[i] + val;
                    }
                }
            } else {
                for (data_val, &segment_id) in data_flat.iter().zip(ids_flat.iter()) {
                    if segment_id >= 0 && (segment_id as usize) < num_segments {
                        let idx = segment_id as usize;
                        result[idx] = result[idx] + *data_val;
                    }
                }
            }

            Tensor::from_vec(result, &[num_segments])
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(data_gpu), TensorStorage::Gpu(ids_gpu)) => {
            segment_sum_gpu(data, data_gpu, ids_gpu, num_segments)
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::unsupported_operation_simple(
            "Mixed CPU/GPU segment operations not supported".to_string(),
        )),
    }
}

/// Segmented mean operation for ragged tensor support
///
/// Computes the mean of elements within each segment defined by segment_ids.
///
/// # Arguments
/// * `data` - Input tensor containing the data to be reduced
/// * `segment_ids` - Tensor of non-negative integers that define segments. Must be sorted.
/// * `num_segments` - Total number of segments (maximum segment_id + 1)
///
/// # Returns
/// A tensor of shape `[num_segments]` containing the mean for each segment
pub fn segment_mean<T>(
    data: &Tensor<T>,
    segment_ids: &Tensor<i32>,
    num_segments: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + scirs2_core::num_traits::Zero
        + scirs2_core::num_traits::FromPrimitive
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    if data.shape().dims()[0] != segment_ids.shape().dims()[0] {
        return Err(TensorError::shape_mismatch(
            "segment_reduction",
            "data and segment_ids must have same first dimension",
            &format!(
                "data: {:?}, segment_ids: {:?}",
                data.shape().dims(),
                segment_ids.shape().dims()
            ),
        ));
    }

    match (&data.storage, &segment_ids.storage) {
        (TensorStorage::Cpu(data_arr), TensorStorage::Cpu(ids_arr)) => {
            let data_flat = data_arr
                .view()
                .into_shape_with_order([data_arr.len()])
                .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;
            let ids_flat = ids_arr
                .view()
                .into_shape_with_order([ids_arr.len()])
                .map_err(|e| TensorError::invalid_shape_simple(e.to_string()))?;

            let mut result = vec![T::zero(); num_segments];
            let mut counts = vec![0usize; num_segments];

            if data_flat.len() > 1000 {
                let chunk_size = std::cmp::max(1, data_flat.len() / rayon::current_num_threads());
                let data_slice = data_flat.as_slice().expect("tensor should be contiguous");
                let ids_slice = ids_flat.as_slice().expect("tensor should be contiguous");
                let chunks: Vec<_> = data_slice
                    .chunks(chunk_size)
                    .zip(ids_slice.chunks(chunk_size))
                    .collect();

                let partial_results: Vec<(Vec<T>, Vec<usize>)> = chunks
                    .par_iter()
                    .map(|(data_chunk, ids_chunk)| {
                        let mut local_result = vec![T::zero(); num_segments];
                        let mut local_counts = vec![0usize; num_segments];

                        for (data_val, &segment_id) in data_chunk.iter().zip(ids_chunk.iter()) {
                            if segment_id >= 0 && (segment_id as usize) < num_segments {
                                let idx = segment_id as usize;
                                local_result[idx] = local_result[idx] + *data_val;
                                local_counts[idx] += 1;
                            }
                        }
                        (local_result, local_counts)
                    })
                    .collect();

                for (partial_result, partial_counts) in partial_results {
                    for (i, (val, count)) in
                        partial_result.into_iter().zip(partial_counts).enumerate()
                    {
                        result[i] = result[i] + val;
                        counts[i] += count;
                    }
                }
            } else {
                for (data_val, &segment_id) in data_flat.iter().zip(ids_flat.iter()) {
                    if segment_id >= 0 && (segment_id as usize) < num_segments {
                        let idx = segment_id as usize;
                        result[idx] = result[idx] + *data_val;
                        counts[idx] += 1;
                    }
                }
            }

            for (i, count) in counts.iter().enumerate() {
                if *count > 0 {
                    if let Some(count_t) = T::from_usize(*count) {
                        result[i] = result[i] / count_t;
                    }
                }
            }

            Tensor::from_vec(result, &[num_segments])
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(data_gpu), TensorStorage::Gpu(ids_gpu)) => {
            segment_mean_gpu(data, data_gpu, ids_gpu, num_segments)
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::unsupported_operation_simple(
            "Mixed CPU/GPU segment operations not supported".to_string(),
        )),
    }
}
