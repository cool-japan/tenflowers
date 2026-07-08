//! Segment min, segment max operations

use super::sum_mean::SegmentLayout;
use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};

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
/// For a 1-D `data` of shape `[N]`, a tensor of shape `[num_segments]`.
/// For an N-D `data` of shape `[N, d1, d2, ...]`, a tensor of shape
/// `[num_segments, d1, d2, ...]` whose row `s` is the element-wise max of all
/// input rows `i` with `segment_ids[i] == s`. A segment with no members is
/// filled with `T::min_value()`.
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

    let layout = SegmentLayout::new(data.shape().dims(), num_segments);

    match (&data.storage, &segment_ids.storage) {
        (TensorStorage::Cpu(data_arr), TensorStorage::Cpu(ids_arr)) => {
            // Materialise data in logical row-major order so non-contiguous
            // inputs are handled correctly; rows are `feature_width` wide.
            let data_flat = data_arr.iter().cloned().collect::<Vec<T>>();
            let ids = ids_arr.iter().copied().collect::<Vec<i32>>();

            let (result, _initialized) =
                layout.reduce_rows(&data_flat, &ids, T::min_value(), |a, b| {
                    if b > a {
                        b.clone()
                    } else {
                        a.clone()
                    }
                });

            Tensor::from_vec(result, &layout.out_shape)
        }
        #[cfg(feature = "gpu")]
        _ => {
            let cpu_data = data.to_cpu()?;
            let cpu_ids = segment_ids.to_cpu()?;
            segment_max(&cpu_data, &cpu_ids, num_segments)
        }
    }
}

/// Segmented min operation for ragged tensor support
///
/// Computes the minimum of elements within each segment defined by segment_ids.
///
/// # Returns
/// For a 1-D `data` of shape `[N]`, a tensor of shape `[num_segments]`.
/// For an N-D `data` of shape `[N, d1, d2, ...]`, a tensor of shape
/// `[num_segments, d1, d2, ...]` whose row `s` is the element-wise min of all
/// input rows `i` with `segment_ids[i] == s`. A segment with no members is
/// filled with `T::max_value()`.
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

    let layout = SegmentLayout::new(data.shape().dims(), num_segments);

    match (&data.storage, &segment_ids.storage) {
        (TensorStorage::Cpu(data_arr), TensorStorage::Cpu(ids_arr)) => {
            let data_flat = data_arr.iter().cloned().collect::<Vec<T>>();
            let ids = ids_arr.iter().copied().collect::<Vec<i32>>();

            let (result, _initialized) =
                layout.reduce_rows(&data_flat, &ids, T::max_value(), |a, b| {
                    if b < a {
                        b.clone()
                    } else {
                        a.clone()
                    }
                });

            Tensor::from_vec(result, &layout.out_shape)
        }
        #[cfg(feature = "gpu")]
        _ => {
            let cpu_data = data.to_cpu()?;
            let cpu_ids = segment_ids.to_cpu()?;
            segment_min(&cpu_data, &cpu_ids, num_segments)
        }
    }
}

// End-to-end tests for the GPU-resident code paths of `segment_max` and
// `segment_min`. These build real GPU-resident tensors and call the actual
// public functions, verifying that the device->host readback + CPU-delegate
// fallback produces numerically correct results (not just "doesn't panic").
// Mirrors `ops::einsum::gpu::gpu_delegate_tests`.
//
// A GPU adapter is not guaranteed to be present in every environment that
// builds with `--features gpu` (e.g. a headless CI runner). `Tensor::to(Device::Gpu(0))`
// surfaces adapter/device creation failures as an honest `Err` rather than
// panicking, so each test attempts the transfer and skips its assertions -
// without failing the suite - if no adapter is available.
#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use super::*;
    use crate::Device;

    #[test]
    fn gpu_segment_max_matches_cpu_reference() {
        let data_cpu = Tensor::<f32>::from_vec(vec![1.0, 5.0, 3.0, 2.0, 8.0, 0.0], &[6])
            .expect("test: from_vec should succeed");
        let ids_cpu = Tensor::<i32>::from_vec(vec![0, 0, 1, 1, 2, 2], &[6])
            .expect("test: from_vec should succeed");

        let (data_gpu, ids_gpu) = match (data_cpu.to(Device::Gpu(0)), ids_cpu.to(Device::Gpu(0))) {
            (Ok(d), Ok(i)) => (d, i),
            _ => return, // No GPU adapter available in this environment; skip.
        };

        let expected =
            segment_max(&data_cpu, &ids_cpu, 3).expect("test: CPU segment_max should succeed");
        let actual = segment_max(&data_gpu, &ids_gpu, 3)
            .expect("test: GPU segment_max should succeed with a real adapter");

        assert_eq!(actual.shape().dims(), expected.shape().dims());
        assert_eq!(
            actual.to_vec().expect("test: to_vec should succeed"),
            expected.to_vec().expect("test: to_vec should succeed")
        );
    }

    #[test]
    fn gpu_segment_min_matches_cpu_reference() {
        let data_cpu = Tensor::<f32>::from_vec(vec![1.0, 5.0, 3.0, 2.0, 8.0, 0.0], &[6])
            .expect("test: from_vec should succeed");
        let ids_cpu = Tensor::<i32>::from_vec(vec![0, 0, 1, 1, 2, 2], &[6])
            .expect("test: from_vec should succeed");

        let (data_gpu, ids_gpu) = match (data_cpu.to(Device::Gpu(0)), ids_cpu.to(Device::Gpu(0))) {
            (Ok(d), Ok(i)) => (d, i),
            _ => return, // No GPU adapter available in this environment; skip.
        };

        let expected =
            segment_min(&data_cpu, &ids_cpu, 3).expect("test: CPU segment_min should succeed");
        let actual = segment_min(&data_gpu, &ids_gpu, 3)
            .expect("test: GPU segment_min should succeed with a real adapter");

        assert_eq!(actual.shape().dims(), expected.shape().dims());
        assert_eq!(
            actual.to_vec().expect("test: to_vec should succeed"),
            expected.to_vec().expect("test: to_vec should succeed")
        );
    }
}
