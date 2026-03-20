//! Segment product, any, and all operations

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};

/// Segmented product operation
///
/// Computes the product of elements within each segment defined by segment_ids.
pub fn segment_prod<T>(
    data: &Tensor<T>,
    segment_ids: &Tensor<i32>,
    num_segments: usize,
) -> Result<Tensor<T>>
where
    T: Clone
        + Default
        + std::ops::Mul<Output = T>
        + scirs2_core::num_traits::One
        + Send
        + Sync
        + 'static
        + bytemuck::Pod
        + bytemuck::Zeroable,
{
    if data.shape().dims()[0] != segment_ids.shape().dims()[0] {
        return Err(TensorError::shape_mismatch(
            "segment_prod",
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

            let mut result = vec![T::one(); num_segments];
            let mut segment_initialized = vec![false; num_segments];

            for (data_val, &segment_id) in data_flat.iter().zip(ids_flat.iter()) {
                if segment_id >= 0 && (segment_id as usize) < num_segments {
                    let idx = segment_id as usize;
                    if !segment_initialized[idx] {
                        result[idx] = *data_val;
                        segment_initialized[idx] = true;
                    } else {
                        result[idx] = result[idx] * *data_val;
                    }
                }
            }

            Tensor::from_vec(result, &[num_segments])
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::unsupported_operation_simple(
            "GPU segment_prod not yet implemented".to_string(),
        )),
    }
}

/// Segmented any operation (logical OR within each segment)
///
/// Returns true for a segment if any element in that segment is non-zero.
pub fn segment_any(
    data: &Tensor<u8>,
    segment_ids: &Tensor<i32>,
    num_segments: usize,
) -> Result<Tensor<u8>> {
    if data.shape().dims()[0] != segment_ids.shape().dims()[0] {
        return Err(TensorError::shape_mismatch(
            "segment_any",
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

            let mut result = vec![0u8; num_segments];

            for (&data_val, &segment_id) in data_flat.iter().zip(ids_flat.iter()) {
                if segment_id >= 0 && (segment_id as usize) < num_segments {
                    let idx = segment_id as usize;
                    if data_val != 0 {
                        result[idx] = 1;
                    }
                }
            }

            Tensor::from_vec(result, &[num_segments])
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::unsupported_operation_simple(
            "GPU segment_any not yet implemented".to_string(),
        )),
    }
}

/// Segmented all operation (logical AND within each segment)
///
/// Returns true for a segment if all elements in that segment are non-zero.
pub fn segment_all(
    data: &Tensor<u8>,
    segment_ids: &Tensor<i32>,
    num_segments: usize,
) -> Result<Tensor<u8>> {
    if data.shape().dims()[0] != segment_ids.shape().dims()[0] {
        return Err(TensorError::shape_mismatch(
            "segment_all",
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

            // Initialize to 1 (true); any zero element will set to 0
            let mut result = vec![1u8; num_segments];
            let mut segment_seen = vec![false; num_segments];

            for (&data_val, &segment_id) in data_flat.iter().zip(ids_flat.iter()) {
                if segment_id >= 0 && (segment_id as usize) < num_segments {
                    let idx = segment_id as usize;
                    segment_seen[idx] = true;
                    if data_val == 0 {
                        result[idx] = 0;
                    }
                }
            }

            // Segments with no data default to 1 (vacuously true) — keep as 1
            Tensor::from_vec(result, &[num_segments])
        }
        #[cfg(feature = "gpu")]
        _ => Err(TensorError::unsupported_operation_simple(
            "GPU segment_all not yet implemented".to_string(),
        )),
    }
}
