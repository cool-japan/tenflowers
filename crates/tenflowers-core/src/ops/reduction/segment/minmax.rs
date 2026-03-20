//! Segment min, segment max operations

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use rayon::prelude::*;

/// GPU stub for segment max
#[cfg(feature = "gpu")]
pub(super) fn segment_max_gpu<T>(
    data: &Tensor<T>,
    data_gpu: &crate::gpu::buffer::GpuBuffer<T>,
    ids_gpu: &crate::gpu::buffer::GpuBuffer<i32>,
    num_segments: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + PartialOrd
        + scirs2_core::num_traits::Bounded
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

/// Segmented max operation for ragged tensor support
///
/// Computes the maximum of elements within each segment defined by segment_ids.
///
/// # Arguments
/// * `data` - Input tensor containing the data to be reduced
/// * `segment_ids` - Tensor of non-negative integers that define segments. Must be sorted.
/// * `num_segments` - Total number of segments (maximum segment_id + 1)
///
/// # Returns
/// A tensor of shape `[num_segments]` containing the max for each segment
pub fn segment_max<T>(
    data: &Tensor<T>,
    segment_ids: &Tensor<i32>,
    num_segments: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + PartialOrd
        + scirs2_core::num_traits::Bounded
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

            let mut result = vec![T::min_value(); num_segments];
            let mut segment_initialized = vec![false; num_segments];

            if data_flat.len() > 1000 {
                let chunk_size = std::cmp::max(1, data_flat.len() / rayon::current_num_threads());
                let data_slice = data_flat.as_slice().expect("tensor should be contiguous");
                let ids_slice = ids_flat.as_slice().expect("tensor should be contiguous");
                let chunks: Vec<_> = data_slice
                    .chunks(chunk_size)
                    .zip(ids_slice.chunks(chunk_size))
                    .collect();

                let partial_results: Vec<(Vec<T>, Vec<bool>)> = chunks
                    .par_iter()
                    .map(|(data_chunk, ids_chunk)| {
                        let mut local_result = vec![T::min_value(); num_segments];
                        let mut local_initialized = vec![false; num_segments];

                        for (data_val, &segment_id) in data_chunk.iter().zip(ids_chunk.iter()) {
                            if segment_id >= 0 && (segment_id as usize) < num_segments {
                                let idx = segment_id as usize;
                                if !local_initialized[idx] {
                                    local_result[idx] = *data_val;
                                    local_initialized[idx] = true;
                                } else if *data_val > local_result[idx] {
                                    local_result[idx] = *data_val;
                                }
                            }
                        }
                        (local_result, local_initialized)
                    })
                    .collect();

                for (partial_result, partial_initialized) in partial_results {
                    for (i, (val, initialized)) in partial_result
                        .into_iter()
                        .zip(partial_initialized)
                        .enumerate()
                    {
                        if initialized {
                            if !segment_initialized[i] {
                                result[i] = val;
                                segment_initialized[i] = true;
                            } else if val > result[i] {
                                result[i] = val;
                            }
                        }
                    }
                }
            } else {
                for (data_val, &segment_id) in data_flat.iter().zip(ids_flat.iter()) {
                    if segment_id >= 0 && (segment_id as usize) < num_segments {
                        let idx = segment_id as usize;
                        if !segment_initialized[idx] {
                            result[idx] = *data_val;
                            segment_initialized[idx] = true;
                        } else if *data_val > result[idx] {
                            result[idx] = *data_val;
                        }
                    }
                }
            }

            Tensor::from_vec(result, &[num_segments])
        }
        #[cfg(feature = "gpu")]
        (TensorStorage::Gpu(data_gpu), TensorStorage::Gpu(ids_gpu)) => {
            segment_max_gpu(data, data_gpu, ids_gpu, num_segments)
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::unsupported_operation_simple(
            "Mixed CPU/GPU segment operations not supported".to_string(),
        )),
    }
}

/// Segmented min operation for ragged tensor support
///
/// Computes the minimum of elements within each segment defined by segment_ids.
pub fn segment_min<T>(
    data: &Tensor<T>,
    segment_ids: &Tensor<i32>,
    num_segments: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + PartialOrd
        + scirs2_core::num_traits::Bounded
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

            let mut result = vec![T::max_value(); num_segments];
            let mut segment_initialized = vec![false; num_segments];

            for (data_val, &segment_id) in data_flat.iter().zip(ids_flat.iter()) {
                if segment_id >= 0 && (segment_id as usize) < num_segments {
                    let idx = segment_id as usize;
                    if !segment_initialized[idx] {
                        result[idx] = *data_val;
                        segment_initialized[idx] = true;
                    } else if *data_val < result[idx] {
                        result[idx] = *data_val;
                    }
                }
            }

            Tensor::from_vec(result, &[num_segments])
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::unsupported_operation_simple(
            "GPU segment_min not yet implemented".to_string(),
        )),
    }
}
