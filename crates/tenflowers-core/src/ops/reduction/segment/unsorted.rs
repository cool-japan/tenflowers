//! Unsorted segment operations
//!
//! These functions handle segment_ids that are not necessarily sorted,
//! and include unsorted_segment_sum, unsorted_segment_mean, etc.

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};

/// Unsorted segment sum: like segment_sum but segment_ids need not be sorted
///
/// # Arguments
/// * `data` - Input tensor
/// * `segment_ids` - Integer tensor with segment assignments (unsorted is allowed)
/// * `num_segments` - Total number of output segments
///
/// # Returns
/// A tensor of shape `[num_segments]` with the sum for each segment
pub fn unsorted_segment_sum<T>(
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
    // unsorted_segment_sum is identical in implementation to segment_sum (order-independent)
    super::sum_mean::segment_sum(data, segment_ids, num_segments)
}

/// Unsorted segment mean
pub fn unsorted_segment_mean<T>(
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
    super::sum_mean::segment_mean(data, segment_ids, num_segments)
}

/// Unsorted segment max
pub fn unsorted_segment_max<T>(
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
    super::minmax::segment_max(data, segment_ids, num_segments)
}

/// Unsorted segment min
pub fn unsorted_segment_min<T>(
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
    super::minmax::segment_min(data, segment_ids, num_segments)
}

/// Unsorted segment product
pub fn unsorted_segment_prod<T>(
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
    super::prod_any_all::segment_prod(data, segment_ids, num_segments)
}
