//! Segment sum and segment mean operations

use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use rayon::prelude::*;

/// Geometry of a segmented reduction over the leading axis of a tensor.
///
/// A `[N, d1, d2, ...]` input is treated as `N` rows, each `feature_width`
/// elements wide (`feature_width = d1 * d2 * ...`). Reductions accumulate whole
/// rows into per-segment buckets, producing `[num_segments, d1, d2, ...]`.
/// For a 1-D `[N]` input, `feature_width == 1` and the output is `[num_segments]`,
/// preserving the original 1-D contract.
pub(super) struct SegmentLayout {
    pub(super) num_segments: usize,
    pub(super) feature_width: usize,
    pub(super) out_shape: Vec<usize>,
}

impl SegmentLayout {
    pub(super) fn new(data_dims: &[usize], num_segments: usize) -> Self {
        let feature_width: usize = data_dims.iter().skip(1).product();
        let mut out_shape = Vec::with_capacity(data_dims.len());
        out_shape.push(num_segments);
        out_shape.extend_from_slice(&data_dims[1..]);
        Self {
            num_segments,
            feature_width,
            out_shape,
        }
    }

    /// Number of rows (leading dimension) implied by `feature_width`.
    fn row_count(&self, data_len: usize) -> usize {
        data_len.checked_div(self.feature_width).unwrap_or(0)
    }

    /// Add `row`'s feature vector into segment `seg` of `acc`.
    fn add_row<T>(&self, acc: &mut [T], seg: usize, row: usize, data: &[T])
    where
        T: Clone + std::ops::Add<Output = T>,
    {
        let dst = seg * self.feature_width;
        let src = row * self.feature_width;
        for offset in 0..self.feature_width {
            acc[dst + offset] = acc[dst + offset].clone() + data[src + offset].clone();
        }
    }

    /// Accumulate row sums into `[num_segments * feature_width]`.
    fn accumulate<T>(&self, data: &[T], ids: &[i32]) -> Vec<T>
    where
        T: Clone
            + Default
            + std::ops::Add<Output = T>
            + scirs2_core::num_traits::Zero
            + Send
            + Sync,
    {
        let rows = self.row_count(data.len());
        let acc_len = self.num_segments * self.feature_width;

        // Parallel tree-reduction for large inputs; serial otherwise.
        if rows > 1000 {
            let chunk_size = std::cmp::max(1, rows / rayon::current_num_threads());
            (0..rows)
                .into_par_iter()
                .chunks(chunk_size)
                .map(|row_chunk| {
                    let mut local = vec![T::zero(); acc_len];
                    for row in row_chunk {
                        let seg = ids[row];
                        if seg >= 0 && (seg as usize) < self.num_segments {
                            self.add_row(&mut local, seg as usize, row, data);
                        }
                    }
                    local
                })
                .reduce(
                    || vec![T::zero(); acc_len],
                    |mut a, b| {
                        for (slot, val) in a.iter_mut().zip(b) {
                            *slot = slot.clone() + val;
                        }
                        a
                    },
                )
        } else {
            let mut acc = vec![T::zero(); acc_len];
            for (row, &seg) in ids.iter().enumerate().take(rows) {
                if seg >= 0 && (seg as usize) < self.num_segments {
                    self.add_row(&mut acc, seg as usize, row, data);
                }
            }
            acc
        }
    }

    /// Accumulate row sums together with the per-segment row counts.
    fn accumulate_with_counts<T>(&self, data: &[T], ids: &[i32]) -> (Vec<T>, Vec<usize>)
    where
        T: Clone + Default + std::ops::Add<Output = T> + scirs2_core::num_traits::Zero,
    {
        let rows = self.row_count(data.len());
        let acc_len = self.num_segments * self.feature_width;
        let mut acc = vec![T::zero(); acc_len];
        let mut counts = vec![0usize; self.num_segments];

        for (row, &seg) in ids.iter().enumerate().take(rows) {
            if seg >= 0 && (seg as usize) < self.num_segments {
                let seg = seg as usize;
                self.add_row(&mut acc, seg, row, data);
                counts[seg] += 1;
            }
        }

        (acc, counts)
    }

    /// Fold `row`'s feature vector into segment `seg` of `acc` using `combine`.
    ///
    /// The first row seen for a given `(segment, feature offset)` cell is
    /// copied in directly (bypassing `combine`); every later row for that
    /// cell folds in via `combine(&acc[cell], &data[cell])`. This lets a
    /// cell's pre-fill value (see [`SegmentLayout::reduce_rows`]) double as
    /// the value returned for a cell that is never touched (e.g. every cell
    /// of an empty segment), while still correctly seeding cells with values
    /// that would not combine sensibly against an arbitrary sentinel (e.g.
    /// float NaN compared against `T::min_value()`).
    fn combine_row<T, F>(
        &self,
        acc: &mut [T],
        initialized: &mut [bool],
        seg: usize,
        row: usize,
        data: &[T],
        combine: &F,
    ) where
        T: Clone,
        F: Fn(&T, &T) -> T,
    {
        let dst = seg * self.feature_width;
        let src = row * self.feature_width;
        for offset in 0..self.feature_width {
            let cell = dst + offset;
            if initialized[cell] {
                acc[cell] = combine(&acc[cell], &data[src + offset]);
            } else {
                acc[cell] = data[src + offset].clone();
                initialized[cell] = true;
            }
        }
    }

    /// Row-wise segmented fold shared by `segment_max`, `segment_min`,
    /// `segment_prod`, `segment_any`, and `segment_all`.
    ///
    /// For each row `r` in `0..row_count`, looks up `seg = ids[r]` and folds
    /// `data`'s `r`-th feature row into segment `seg`'s `feature_width`-wide
    /// bucket via `combine` (see [`SegmentLayout::combine_row`]). Returns
    /// `(values, initialized)`, both of length `num_segments * feature_width`;
    /// `initialized[i]` is `true` iff cell `i` was touched by at least one
    /// row. A cell that is never touched (e.g. every cell of an empty
    /// segment) keeps its pre-fill `identity` value, so callers can use
    /// `identity` to encode the desired empty-segment fill (e.g.
    /// `T::min_value()` for max, `T::one()` for product).
    ///
    /// Mirrors [`SegmentLayout::accumulate`]'s row-chunked parallel strategy
    /// for large inputs (`rows > 1000`): each chunk folds into its own
    /// `(values, initialized)` pair, and chunks are merged pairwise via
    /// [`merge_partial_rows`].
    pub(super) fn reduce_rows<T, F>(
        &self,
        data: &[T],
        ids: &[i32],
        identity: T,
        combine: F,
    ) -> (Vec<T>, Vec<bool>)
    where
        T: Clone + Send + Sync,
        F: Fn(&T, &T) -> T + Sync + Send,
    {
        let rows = self.row_count(data.len());
        let acc_len = self.num_segments * self.feature_width;

        if rows > 1000 {
            let chunk_size = std::cmp::max(1, rows / rayon::current_num_threads());
            (0..rows)
                .into_par_iter()
                .chunks(chunk_size)
                .map(|row_chunk| {
                    let mut local_values = vec![identity.clone(); acc_len];
                    let mut local_initialized = vec![false; acc_len];
                    for row in row_chunk {
                        let seg = ids[row];
                        if seg >= 0 && (seg as usize) < self.num_segments {
                            self.combine_row(
                                &mut local_values,
                                &mut local_initialized,
                                seg as usize,
                                row,
                                data,
                                &combine,
                            );
                        }
                    }
                    (local_values, local_initialized)
                })
                .reduce(
                    || (vec![identity.clone(); acc_len], vec![false; acc_len]),
                    |mut a, b| {
                        merge_partial_rows(&mut a.0, &mut a.1, b, &combine);
                        a
                    },
                )
        } else {
            let mut values = vec![identity.clone(); acc_len];
            let mut initialized = vec![false; acc_len];
            for (row, &seg) in ids.iter().enumerate().take(rows) {
                if seg >= 0 && (seg as usize) < self.num_segments {
                    self.combine_row(
                        &mut values,
                        &mut initialized,
                        seg as usize,
                        row,
                        data,
                        &combine,
                    );
                }
            }
            (values, initialized)
        }
    }
}

/// Merge one of [`SegmentLayout::reduce_rows`]'s row-chunked partial results
/// (`src`) into another (`dst`), preserving first-touch semantics per cell:
/// a cell `src` touched but `dst` did not is copied in directly; a cell both
/// touched is folded via `combine`; a cell neither touched is left at
/// `dst`'s identity pre-fill.
fn merge_partial_rows<T, F>(
    dst_values: &mut [T],
    dst_initialized: &mut [bool],
    src: (Vec<T>, Vec<bool>),
    combine: &F,
) where
    T: Clone,
    F: Fn(&T, &T) -> T,
{
    let (src_values, src_initialized) = src;
    for i in 0..dst_values.len() {
        if src_initialized[i] {
            if dst_initialized[i] {
                dst_values[i] = combine(&dst_values[i], &src_values[i]);
            } else {
                dst_values[i] = src_values[i].clone();
                dst_initialized[i] = true;
            }
        }
    }
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
/// For a 1-D `data` of shape `[N]`, a tensor of shape `[num_segments]`.
/// For an N-D `data` of shape `[N, d1, d2, ...]`, a tensor of shape
/// `[num_segments, d1, d2, ...]` whose row `s` is the element-wise sum of all
/// input rows `i` with `segment_ids[i] == s`.
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

    let layout = SegmentLayout::new(data.shape().dims(), num_segments);

    match (&data.storage, &segment_ids.storage) {
        (TensorStorage::Cpu(data_arr), TensorStorage::Cpu(ids_arr)) => {
            // Materialise data in logical row-major order so non-contiguous
            // inputs are handled correctly; rows are `feature_width` wide.
            let data_flat = data_arr.iter().cloned().collect::<Vec<T>>();
            let ids = ids_arr.iter().copied().collect::<Vec<i32>>();

            let result = layout.accumulate(&data_flat, &ids);

            Tensor::from_vec(result, &layout.out_shape)
        }
        #[cfg(feature = "gpu")]
        _ => {
            let cpu_data = data.to_cpu()?;
            let cpu_ids = segment_ids.to_cpu()?;
            segment_sum(&cpu_data, &cpu_ids, num_segments)
        }
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
/// For a 1-D `data` of shape `[N]`, a tensor of shape `[num_segments]`.
/// For an N-D `data` of shape `[N, d1, d2, ...]`, a tensor of shape
/// `[num_segments, d1, d2, ...]` whose row `s` is the element-wise mean of all
/// input rows `i` with `segment_ids[i] == s`.
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

    let layout = SegmentLayout::new(data.shape().dims(), num_segments);

    match (&data.storage, &segment_ids.storage) {
        (TensorStorage::Cpu(data_arr), TensorStorage::Cpu(ids_arr)) => {
            let data_flat = data_arr.iter().cloned().collect::<Vec<T>>();
            let ids = ids_arr.iter().copied().collect::<Vec<i32>>();

            let (mut result, counts) = layout.accumulate_with_counts(&data_flat, &ids);

            // Divide each segment's feature row by that segment's row count.
            for (segment, &count) in counts.iter().enumerate() {
                if count > 0 {
                    if let Some(count_t) = T::from_usize(count) {
                        let base = segment * layout.feature_width;
                        for offset in 0..layout.feature_width {
                            let cell = base + offset;
                            result[cell] = result[cell].clone() / count_t.clone();
                        }
                    }
                }
            }

            Tensor::from_vec(result, &layout.out_shape)
        }
        #[cfg(feature = "gpu")]
        _ => {
            let cpu_data = data.to_cpu()?;
            let cpu_ids = segment_ids.to_cpu()?;
            segment_mean(&cpu_data, &cpu_ids, num_segments)
        }
    }
}

// End-to-end tests for the GPU-resident code paths of `segment_sum` and
// `segment_mean`. These build real GPU-resident tensors and call the actual
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
    fn gpu_segment_sum_matches_cpu_reference() {
        let data_cpu = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6])
            .expect("test: from_vec should succeed");
        let ids_cpu = Tensor::<i32>::from_vec(vec![0, 0, 1, 1, 2, 2], &[6])
            .expect("test: from_vec should succeed");

        let (data_gpu, ids_gpu) = match (data_cpu.to(Device::Gpu(0)), ids_cpu.to(Device::Gpu(0))) {
            (Ok(d), Ok(i)) => (d, i),
            _ => return, // No GPU adapter available in this environment; skip.
        };

        let expected =
            segment_sum(&data_cpu, &ids_cpu, 3).expect("test: CPU segment_sum should succeed");
        let actual = segment_sum(&data_gpu, &ids_gpu, 3)
            .expect("test: GPU segment_sum should succeed with a real adapter");

        assert_eq!(actual.shape().dims(), expected.shape().dims());
        assert_eq!(
            actual.to_vec().expect("test: to_vec should succeed"),
            expected.to_vec().expect("test: to_vec should succeed")
        );
    }

    #[test]
    fn gpu_segment_mean_matches_cpu_reference() {
        // Mean divides evenly ([2,4]->3, [10,20]->15) so there is no
        // float-tolerance ambiguity; still compare with an epsilon to stay
        // robust to any future change in how the GPU readback path composes
        // its arithmetic.
        let data_cpu = Tensor::<f32>::from_vec(vec![2.0, 4.0, 10.0, 20.0], &[4])
            .expect("test: from_vec should succeed");
        let ids_cpu =
            Tensor::<i32>::from_vec(vec![0, 0, 1, 1], &[4]).expect("test: from_vec should succeed");

        let (data_gpu, ids_gpu) = match (data_cpu.to(Device::Gpu(0)), ids_cpu.to(Device::Gpu(0))) {
            (Ok(d), Ok(i)) => (d, i),
            _ => return, // No GPU adapter available in this environment; skip.
        };

        let expected =
            segment_mean(&data_cpu, &ids_cpu, 2).expect("test: CPU segment_mean should succeed");
        let actual = segment_mean(&data_gpu, &ids_gpu, 2)
            .expect("test: GPU segment_mean should succeed with a real adapter");

        assert_eq!(actual.shape().dims(), expected.shape().dims());
        let actual_vals = actual.to_vec().expect("test: to_vec should succeed");
        let expected_vals = expected.to_vec().expect("test: to_vec should succeed");
        assert_eq!(expected_vals, vec![3.0, 15.0]);
        for (a, e) in actual_vals.iter().zip(expected_vals.iter()) {
            assert!(
                (a - e).abs() < 1e-6,
                "GPU segment_mean {a} does not match CPU reference {e} within tolerance"
            );
        }
    }
}
