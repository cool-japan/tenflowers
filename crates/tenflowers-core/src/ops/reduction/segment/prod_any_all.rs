//! Segment product, any, and all operations

use super::sum_mean::SegmentLayout;
use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};

/// Segmented product operation
///
/// Computes the product of elements within each segment defined by segment_ids.
///
/// # Returns
/// For a 1-D `data` of shape `[N]`, a tensor of shape `[num_segments]`.
/// For an N-D `data` of shape `[N, d1, d2, ...]`, a tensor of shape
/// `[num_segments, d1, d2, ...]` whose row `s` is the element-wise product of
/// all input rows `i` with `segment_ids[i] == s`. A segment with no members
/// is filled with `T::one()`.
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

    let layout = SegmentLayout::new(data.shape().dims(), num_segments);

    match (&data.storage, &segment_ids.storage) {
        (TensorStorage::Cpu(data_arr), TensorStorage::Cpu(ids_arr)) => {
            let data_flat = data_arr.iter().cloned().collect::<Vec<T>>();
            let ids = ids_arr.iter().copied().collect::<Vec<i32>>();

            let (result, _initialized) =
                layout.reduce_rows(&data_flat, &ids, T::one(), |a, b| a.clone() * b.clone());

            Tensor::from_vec(result, &layout.out_shape)
        }
        #[cfg(feature = "gpu")]
        _ => {
            let cpu_data = data.to_cpu()?;
            let cpu_ids = segment_ids.to_cpu()?;
            segment_prod(&cpu_data, &cpu_ids, num_segments)
        }
    }
}

/// Segmented any operation (logical OR within each segment)
///
/// Returns true (`1`) for a segment if any element in that segment is
/// non-zero.
///
/// # Returns
/// For a 1-D `data` of shape `[N]`, a tensor of shape `[num_segments]`.
/// For an N-D `data` of shape `[N, d1, d2, ...]`, a tensor of shape
/// `[num_segments, d1, d2, ...]` whose row `s` is the element-wise logical OR
/// of all input rows `i` with `segment_ids[i] == s`. A segment with no
/// members is filled with `0` (false).
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

    let layout = SegmentLayout::new(data.shape().dims(), num_segments);

    match (&data.storage, &segment_ids.storage) {
        (TensorStorage::Cpu(data_arr), TensorStorage::Cpu(ids_arr)) => {
            let data_flat = data_arr.iter().copied().collect::<Vec<u8>>();
            let ids = ids_arr.iter().copied().collect::<Vec<i32>>();

            let (result, _initialized) =
                layout.reduce_rows(&data_flat, &ids, 0u8, |a, b| u8::from(*a != 0 || *b != 0));

            Tensor::from_vec(result, &layout.out_shape)
        }
        #[cfg(feature = "gpu")]
        _ => {
            let cpu_data = data.to_cpu()?;
            let cpu_ids = segment_ids.to_cpu()?;
            segment_any(&cpu_data, &cpu_ids, num_segments)
        }
    }
}

/// Segmented all operation (logical AND within each segment)
///
/// Returns true (`1`) for a segment if all elements in that segment are
/// non-zero.
///
/// # Returns
/// For a 1-D `data` of shape `[N]`, a tensor of shape `[num_segments]`.
/// For an N-D `data` of shape `[N, d1, d2, ...]`, a tensor of shape
/// `[num_segments, d1, d2, ...]` whose row `s` is the element-wise logical
/// AND of all input rows `i` with `segment_ids[i] == s`. A segment with no
/// members is filled with `1` (vacuously true).
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

    let layout = SegmentLayout::new(data.shape().dims(), num_segments);

    match (&data.storage, &segment_ids.storage) {
        (TensorStorage::Cpu(data_arr), TensorStorage::Cpu(ids_arr)) => {
            let data_flat = data_arr.iter().copied().collect::<Vec<u8>>();
            let ids = ids_arr.iter().copied().collect::<Vec<i32>>();

            // Segments with no members default to `1` (vacuously true).
            let (result, _initialized) =
                layout.reduce_rows(&data_flat, &ids, 1u8, |a, b| u8::from(*a != 0 && *b != 0));

            Tensor::from_vec(result, &layout.out_shape)
        }
        #[cfg(feature = "gpu")]
        _ => {
            let cpu_data = data.to_cpu()?;
            let cpu_ids = segment_ids.to_cpu()?;
            segment_all(&cpu_data, &cpu_ids, num_segments)
        }
    }
}

// End-to-end tests for the GPU-resident code paths of `segment_prod`,
// `segment_any`, and `segment_all`. These build real GPU-resident tensors and
// call the actual public functions, verifying that the device->host readback
// + CPU-delegate fallback produces numerically correct results (not just
// "doesn't panic"). Mirrors `ops::einsum::gpu::gpu_delegate_tests`.
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
    fn gpu_segment_prod_matches_cpu_reference() {
        let data_cpu = Tensor::<f32>::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[4])
            .expect("test: from_vec should succeed");
        let ids_cpu =
            Tensor::<i32>::from_vec(vec![0, 0, 1, 1], &[4]).expect("test: from_vec should succeed");
        let (data_gpu, ids_gpu) = match (data_cpu.to(Device::Gpu(0)), ids_cpu.to(Device::Gpu(0))) {
            (Ok(d), Ok(i)) => (d, i),
            _ => return, // No GPU adapter available in this environment; skip.
        };
        let expected =
            segment_prod(&data_cpu, &ids_cpu, 2).expect("test: CPU segment_prod should succeed");
        let actual = segment_prod(&data_gpu, &ids_gpu, 2)
            .expect("test: GPU segment_prod should succeed with a real adapter");
        assert_eq!(actual.shape().dims(), expected.shape().dims());
        assert_eq!(
            actual.to_vec().expect("test: to_vec should succeed"),
            expected.to_vec().expect("test: to_vec should succeed")
        );
    }

    #[test]
    fn gpu_segment_any_matches_cpu_reference() {
        let data_cpu =
            Tensor::<u8>::from_vec(vec![0, 1, 0, 0], &[4]).expect("test: from_vec should succeed");
        let ids_cpu =
            Tensor::<i32>::from_vec(vec![0, 0, 1, 1], &[4]).expect("test: from_vec should succeed");
        let (data_gpu, ids_gpu) = match (data_cpu.to(Device::Gpu(0)), ids_cpu.to(Device::Gpu(0))) {
            (Ok(d), Ok(i)) => (d, i),
            _ => return, // No GPU adapter available in this environment; skip.
        };
        let expected =
            segment_any(&data_cpu, &ids_cpu, 2).expect("test: CPU segment_any should succeed");
        let actual = segment_any(&data_gpu, &ids_gpu, 2)
            .expect("test: GPU segment_any should succeed with a real adapter");
        assert_eq!(
            actual.to_vec().expect("test: to_vec should succeed"),
            expected.to_vec().expect("test: to_vec should succeed")
        );
    }

    #[test]
    fn gpu_segment_all_matches_cpu_reference() {
        // Segment 0 = [1, 1] (all-nonzero -> true); segment 1 = [1, 0] (has a
        // zero -> false). This distinguishes `all`'s semantics from `any`'s:
        // an analogous `any` test on this same data would yield [1, 1].
        let data_cpu =
            Tensor::<u8>::from_vec(vec![1, 1, 1, 0], &[4]).expect("test: from_vec should succeed");
        let ids_cpu =
            Tensor::<i32>::from_vec(vec![0, 0, 1, 1], &[4]).expect("test: from_vec should succeed");
        let (data_gpu, ids_gpu) = match (data_cpu.to(Device::Gpu(0)), ids_cpu.to(Device::Gpu(0))) {
            (Ok(d), Ok(i)) => (d, i),
            _ => return, // No GPU adapter available in this environment; skip.
        };
        let expected =
            segment_all(&data_cpu, &ids_cpu, 2).expect("test: CPU segment_all should succeed");
        let actual = segment_all(&data_gpu, &ids_gpu, 2)
            .expect("test: GPU segment_all should succeed with a real adapter");
        assert_eq!(
            actual.to_vec().expect("test: to_vec should succeed"),
            expected.to_vec().expect("test: to_vec should succeed")
        );
        assert_eq!(
            expected.to_vec().expect("test: to_vec should succeed"),
            vec![1u8, 0u8]
        );
    }
}
