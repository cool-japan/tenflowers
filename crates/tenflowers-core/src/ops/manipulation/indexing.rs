//! Indexing Operations Module
//!
//! This module contains tensor indexing and selection operations:
//! - slice: Extract tensor slices using ranges
//! - slice_with_stride: Advanced slicing with stride support
//! - gather: Gather elements using index arrays
//! - scatter: Scatter updates into tensor positions
//! - select: Select tensor slices using index arrays
//! - where_op: Conditional element selection
//!
//! All operations support both CPU and GPU execution when available.

#[cfg(feature = "gpu")]
use crate::gpu::buffer::GpuBuffer;
use crate::strided::{SliceParams, StridedLayout};
use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use scirs2_core::numeric::Zero;

// Import common helper functions
use super::common::{broadcast_indices, calculate_strides, coords_to_flat, flat_to_coords};

/// Slice a tensor along specified ranges
pub fn slice<T>(tensor: &Tensor<T>, ranges: &[std::ops::Range<usize>]) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    let shape = tensor.shape();

    if ranges.len() != shape.rank() {
        return Err(TensorError::invalid_argument(format!(
            "Slice ranges length {} does not match tensor rank {}",
            ranges.len(),
            shape.rank()
        )));
    }

    // Validate ranges
    for (i, range) in ranges.iter().enumerate() {
        if range.start > range.end || range.end > shape.dims()[i] {
            return Err(TensorError::invalid_argument(format!(
                "Invalid slice range {range:?} for dimension {i} of size {}",
                shape.dims()[i]
            )));
        }
    }

    match &tensor.storage {
        TensorStorage::Cpu(array) => {
            // Output shape: width of each requested range.
            let out_shape: Vec<usize> = ranges.iter().map(|r| r.end - r.start).collect();

            let mut result = ArrayD::<T>::zeros(IxDyn(&out_shape));

            // Walk every coordinate of the output, map it back to a source
            // coordinate via the range starts, and copy using ndarray's
            // logical (stride-aware) indexing. This is correct for BOTH
            // contiguous and non-contiguous (e.g. transposed/permuted) inputs
            // because we never assume a standard memory layout.
            fn copy_recursive<T: Clone>(
                src: &ArrayD<T>,
                dst: &mut ArrayD<T>,
                ranges: &[std::ops::Range<usize>],
                depth: usize,
                src_coords: &mut Vec<usize>,
                dst_coords: &mut Vec<usize>,
            ) {
                if depth == ranges.len() {
                    if let Some(val) = src.get(IxDyn(src_coords)) {
                        if let Some(slot) = dst.get_mut(IxDyn(dst_coords)) {
                            *slot = val.clone();
                        }
                    }
                    return;
                }

                for (dst_idx, src_idx) in ranges[depth].clone().enumerate() {
                    src_coords.push(src_idx);
                    dst_coords.push(dst_idx);
                    copy_recursive(src, dst, ranges, depth + 1, src_coords, dst_coords);
                    src_coords.pop();
                    dst_coords.pop();
                }
            }

            let mut src_coords = Vec::with_capacity(ranges.len());
            let mut dst_coords = Vec::with_capacity(ranges.len());
            copy_recursive(
                array,
                &mut result,
                ranges,
                0,
                &mut src_coords,
                &mut dst_coords,
            );

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => {
            gpu_slice_dispatch(gpu_buffer, tensor.shape().dims(), ranges)
        }
    }
}

/// Slice a tensor along specified ranges with stride support
pub fn slice_with_stride<T>(tensor: &Tensor<T>, slice_params: &[SliceParams]) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    let shape = tensor.shape();

    if slice_params.len() != shape.rank() {
        return Err(TensorError::invalid_argument(format!(
            "Slice params length {} does not match tensor rank {}",
            slice_params.len(),
            shape.rank()
        )));
    }

    // Create a strided layout and perform the slice
    let original_layout = StridedLayout::new(shape.dims().to_vec());
    let sliced_layout = original_layout.slice_with_stride(slice_params)?;

    match &tensor.storage {
        TensorStorage::Cpu(array) => {
            // For CPU tensors, we need to materialize the strided view
            let out_shape = sliced_layout.shape().to_vec();
            let mut result = ArrayD::<T>::zeros(IxDyn(&out_shape));

            if let Some(result_slice) = result.as_slice_mut() {
                let mut result_idx = 0;

                // Iterate through the strided layout to copy elements
                for indices in sliced_layout.indices_iter() {
                    // Map back to original indices
                    let mut original_indices = Vec::new();
                    for (dim, &index) in indices.iter().enumerate() {
                        let (start, _end, step) = slice_params[dim].normalize(shape.dims()[dim])?;
                        let original_idx = start + (index * step.unsigned_abs());
                        original_indices.push(original_idx);
                    }

                    // Get the linear index in the original array
                    let linear_idx: usize = original_indices
                        .iter()
                        .zip(shape.dims())
                        .scan(1, |acc, (&idx, &dim)| {
                            let stride = *acc;
                            *acc *= dim;
                            Some(idx * stride)
                        })
                        .sum();

                    // Copy the element
                    if let Some(val) = array.as_slice().and_then(|s| s.get(linear_idx)) {
                        result_slice[result_idx] = val.clone();
                        result_idx += 1;
                    }
                }
            }

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_buffer) => {
            // If every dimension normalizes to a unit step, this is a plain
            // range slice and the existing GPU range-slice kernel handles it
            // directly.
            let mut ranges = Vec::with_capacity(slice_params.len());
            let mut all_unit_step = true;
            for (i, param) in slice_params.iter().enumerate() {
                let size = shape.dims()[i];
                let (start, end, step) = param.normalize(size)?;
                if step != 1 {
                    all_unit_step = false;
                    break;
                }
                ranges.push(start..end);
            }

            if all_unit_step {
                gpu_slice_dispatch(gpu_buffer, tensor.shape().dims(), &ranges)
            } else {
                // Non-unit step (including negative strides): no native GPU
                // strided-slice kernel exists yet. Read the tensor back to
                // the host (a real device->host transfer) and delegate to
                // the CPU implementation above, which is known-correct
                // (built on `StridedLayout::slice_with_stride`, which
                // already supports arbitrary starts/ends/steps).
                let cpu_tensor = tensor.to_cpu()?;
                slice_with_stride(&cpu_tensor, slice_params)
            }
        }
    }
}

/// Gather operation - gather slices from params according to indices
pub fn gather<T>(params: &Tensor<T>, indices: &Tensor<i32>, axis: usize) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    let params_shape = params.shape();
    let indices_shape = indices.shape();

    if axis >= params_shape.rank() {
        return Err(TensorError::invalid_argument(format!(
            "Axis {axis} out of range for tensor of rank {}",
            params_shape.rank()
        )));
    }

    // Calculate output shape: params.shape with axis dimension replaced by indices.shape
    let mut out_shape = params_shape.dims().to_vec();
    out_shape.remove(axis);
    for &dim in indices_shape.dims().iter().rev() {
        out_shape.insert(axis, dim);
    }

    match (&params.storage, &indices.storage) {
        (TensorStorage::Cpu(params_arr), TensorStorage::Cpu(indices_arr)) => {
            if indices_shape.dims().is_empty() {
                // Scalar index: select a single slice along `axis` and drop it.
                if let Some(&idx) = indices_arr.iter().next() {
                    if idx < 0 || idx as usize >= params_shape.dims()[axis] {
                        return Err(TensorError::invalid_argument(format!(
                            "Index {idx} out of bounds for axis {axis} of size {}",
                            params_shape.dims()[axis]
                        )));
                    }
                    // Extract the slice at the given index
                    let mut ranges: Vec<_> = (0..params_shape.rank())
                        .map(|i| 0..params_shape.dims()[i])
                        .collect();
                    ranges[axis] = idx as usize..(idx as usize + 1);
                    let sliced = slice(params, &ranges)?;
                    return super::shape::squeeze(&sliced, Some(&[axis]));
                }
            }

            // Handle multi-dimensional indices.
            //
            // General `gather` along `axis`: the output replaces the single
            // `axis` dimension of `params` with the full shape of `indices`.
            // Concretely, for output coordinates split as
            //   (pre_axis..., index_coords..., post_axis...)
            // the `index_coords` portion selects an entry of `indices`, whose
            // value `g` is then used as the `axis` coordinate into `params`:
            //   params[pre_axis..., g, post_axis...].
            //
            // We iterate over EVERY output element (out_total = product of
            // out_shape), not just over the indices, so the full `[..., D]`
            // feature width of whole-row gathers is written correctly.
            let indices_dims = indices_shape.dims();
            let indices_rank = indices_dims.len();
            let params_dims = params_shape.dims();

            // Read params/indices logically so non-contiguous inputs are safe.
            let params_data = params_arr.iter().cloned().collect::<Vec<T>>();
            let params_strides = calculate_strides(params_dims);
            let indices_strides = calculate_strides(indices_dims);
            let indices_data = indices_arr.iter().copied().collect::<Vec<i32>>();

            let mut result = ArrayD::<T>::zeros(IxDyn(&out_shape));
            let out_total: usize = out_shape.iter().product();

            for out_flat in 0..out_total {
                let out_coords = flat_to_coords(out_flat, &out_shape);

                // The index block occupies positions [axis, axis + indices_rank)
                // within the output coordinates.
                let index_coords = &out_coords[axis..axis + indices_rank];
                let indices_flat = coords_to_flat(index_coords, &indices_strides);
                let gathered = indices_data[indices_flat];

                if gathered < 0 || gathered as usize >= params_dims[axis] {
                    return Err(TensorError::invalid_argument(format!(
                        "Index {gathered} out of bounds for axis {axis} of size {}",
                        params_dims[axis]
                    )));
                }

                // Build params coordinates: pre-axis dims, the gathered index,
                // then post-axis dims (which follow the index block in output).
                let mut params_coords = Vec::with_capacity(params_dims.len());
                params_coords.extend_from_slice(&out_coords[..axis]);
                params_coords.push(gathered as usize);
                params_coords.extend_from_slice(&out_coords[axis + indices_rank..]);

                let params_flat = coords_to_flat(&params_coords, &params_strides);
                if let Some(slot) = result.get_mut(IxDyn(&out_coords)) {
                    *slot = params_data[params_flat].clone();
                }
            }

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        _ => gpu_gather_dispatch(params, indices, axis),
    }
}

/// Scatter operation - scatter updates into a tensor at specified indices
pub fn scatter<T>(
    tensor: &Tensor<T>,
    indices: &Tensor<i32>,
    updates: &Tensor<T>,
    axis: usize,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    if axis >= tensor.shape().rank() {
        return Err(TensorError::invalid_argument(format!(
            "Axis {axis} out of range for tensor of rank {}",
            tensor.shape().rank()
        )));
    }

    // Validate shapes
    let expected_updates_shape: Vec<_> = tensor
        .shape()
        .dims()
        .iter()
        .enumerate()
        .map(|(i, &dim)| {
            if i == axis {
                indices.shape().dims()[0]
            } else {
                dim
            }
        })
        .collect();

    if updates.shape().dims() != expected_updates_shape {
        return Err(TensorError::invalid_argument(format!(
            "Updates shape {:?} does not match expected shape {:?}",
            updates.shape().dims(),
            expected_updates_shape
        )));
    }

    match (&tensor.storage, &indices.storage, &updates.storage) {
        (
            TensorStorage::Cpu(tensor_arr),
            TensorStorage::Cpu(indices_arr),
            TensorStorage::Cpu(updates_arr),
        ) => {
            let mut result = tensor_arr.clone();
            let indices_slice = indices_arr.as_slice().ok_or_else(|| {
                TensorError::invalid_argument("Indices must be contiguous ".to_string())
            })?;

            // Simple implementation for 1D scatter along axis
            if tensor.shape().rank() == 1 && axis == 0 {
                for (i, &idx) in indices_slice.iter().enumerate() {
                    if idx < 0 || idx as usize >= tensor.shape().dims()[0] {
                        return Err(TensorError::invalid_argument(format!(
                            "Index {idx} out of bounds "
                        )));
                    }
                    result[idx as usize] = updates_arr[[i]].clone();
                }
            } else {
                // For higher dimensions, we need to iterate through all positions
                // and scatter along the specified axis
                let mut update_indices = vec![0; updates.shape().rank()];
                let update_shape = updates.shape().dims();

                loop {
                    // Get the index to scatter to
                    let scatter_idx = indices_slice[update_indices[axis]] as usize;
                    if scatter_idx >= tensor.shape().dims()[axis] {
                        return Err(TensorError::invalid_argument(format!(
                            "Index {scatter_idx} out of bounds for axis {axis} of size {}",
                            tensor.shape().dims()[axis]
                        )));
                    }

                    // Build the target indices
                    let mut target_indices = update_indices.clone();
                    target_indices[axis] = scatter_idx;

                    // Copy the value
                    result[IxDyn(&target_indices)] = updates_arr[IxDyn(&update_indices)].clone();

                    // Increment indices
                    let mut carry = true;
                    for i in (0..update_shape.len()).rev() {
                        if carry {
                            update_indices[i] += 1;
                            if update_indices[i] < update_shape[i] {
                                carry = false;
                            } else {
                                update_indices[i] = 0;
                            }
                        }
                    }
                    if carry {
                        break;
                    }
                }
            }

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        _ => gpu_scatter_dispatch(tensor, indices, updates, axis),
    }
}

/// Where operation - select elements from x or y depending on condition
pub fn where_op<T>(condition: &Tensor<bool>, x: &Tensor<T>, y: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // Check shapes are broadcastable
    let xy_broadcast_shape = x.shape().broadcast_shape(y.shape()).ok_or_else(|| {
        TensorError::invalid_argument(format!(
            "Cannot broadcast shapes {} and {} for where operation ",
            x.shape(),
            y.shape()
        ))
    })?;

    let broadcast_shape = condition
        .shape()
        .broadcast_shape(&xy_broadcast_shape)
        .ok_or_else(|| {
            TensorError::invalid_argument(format!(
                "Condition shape {} cannot be broadcast to {xy_broadcast_shape}",
                condition.shape()
            ))
        })?;

    match (&condition.storage, &x.storage, &y.storage) {
        (TensorStorage::Cpu(cond_arr), TensorStorage::Cpu(x_arr), TensorStorage::Cpu(y_arr)) => {
            let mut result = ArrayD::<T>::zeros(IxDyn(broadcast_shape.dims()));

            // Get the shapes for broadcasting
            let cond_shape = condition.shape().dims();
            let x_shape = x.shape().dims();
            let y_shape = y.shape().dims();
            let out_shape = broadcast_shape.dims();

            // Iterate through all positions in the output
            let mut out_indices = vec![0; out_shape.len()];

            loop {
                // Calculate broadcast indices for each input
                let cond_indices = broadcast_indices(&out_indices, cond_shape, out_shape);
                let x_indices = broadcast_indices(&out_indices, x_shape, out_shape);
                let y_indices = broadcast_indices(&out_indices, y_shape, out_shape);

                // Select value based on condition
                result[IxDyn(&out_indices)] = if cond_arr[IxDyn(&cond_indices)] {
                    x_arr[IxDyn(&x_indices)].clone()
                } else {
                    y_arr[IxDyn(&y_indices)].clone()
                };

                // Increment output indices
                let mut carry = true;
                for i in (0..out_shape.len()).rev() {
                    if carry {
                        out_indices[i] += 1;
                        if out_indices[i] < out_shape[i] {
                            carry = false;
                        } else {
                            out_indices[i] = 0;
                        }
                    }
                }
                if carry {
                    break;
                }
            }

            Ok(Tensor::from_array(result))
        }
        #[cfg(feature = "gpu")]
        _ => gpu_where_dispatch(condition, x, y),
    }
}

/// Select operation - select slices from a tensor along an axis using an index array
pub fn select<T>(tensor: &Tensor<T>, index: &Tensor<i32>, axis: usize) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    gather(tensor, index, axis)
}

// GPU dispatch functions
#[cfg(feature = "gpu")]
fn gpu_slice_dispatch<T>(
    gpu_buffer: &crate::gpu::buffer::GpuBuffer<T>,
    input_shape: &[usize],
    ranges: &[std::ops::Range<usize>],
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    // Currently, we only support f32 for GPU operations
    let type_name = std::any::type_name::<T>();

    if type_name == "f32" {
        // Cast to f32 buffer for the actual GPU operation
        let gpu_buffer_f32 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<T>,
                &crate::gpu::buffer::GpuBuffer<f32>,
            >(gpu_buffer)
        };

        // Calculate slice parameters
        let slice_starts: Vec<usize> = ranges.iter().map(|r| r.start).collect();
        let slice_ends: Vec<usize> = ranges.iter().map(|r| r.end).collect();
        let slice_steps: Vec<usize> = vec![1; ranges.len()]; // Default step of 1 for each dimension
        let output_shape: Vec<usize> = ranges.iter().map(|r| r.end - r.start).collect();
        let output_len: usize = output_shape.iter().product();

        let result_buffer = crate::gpu::ops::execute_slice(
            gpu_buffer_f32,
            &slice_starts,
            &slice_ends,
            &slice_steps,
            input_shape,
            output_len,
        )?;

        // Cast result back to T
        let result_buffer_t = unsafe {
            std::mem::transmute::<
                crate::gpu::buffer::GpuBuffer<f32>,
                crate::gpu::buffer::GpuBuffer<T>,
            >(result_buffer)
        };

        Ok(Tensor::from_gpu_buffer(
            result_buffer_t,
            crate::Shape::from_slice(&output_shape),
        ))
    } else {
        Err(TensorError::unsupported_operation_simple(format!(
            "GPU slice only supports f32, got {}",
            std::any::type_name::<T>()
        )))
    }
}

#[cfg(feature = "gpu")]
fn gpu_gather_dispatch<T>(
    params: &Tensor<T>,
    indices: &Tensor<i32>,
    axis: usize,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    let type_name = std::any::type_name::<T>();

    if type_name == "f32" {
        let params_gpu_buffer = match &params.storage {
            TensorStorage::Gpu(buf) => unsafe {
                std::mem::transmute::<
                    &crate::gpu::buffer::GpuBuffer<T>,
                    &crate::gpu::buffer::GpuBuffer<f32>,
                >(buf)
            },
            _ => {
                return Err(TensorError::device_error_simple(
                    "Expected GPU tensor ".to_string(),
                ))
            }
        };

        let indices_gpu_buffer = match &indices.storage {
            TensorStorage::Gpu(buf) => buf,
            _ => {
                return Err(TensorError::device_error_simple(
                    "Expected GPU tensor ".to_string(),
                ))
            }
        };

        // Calculate output shape and length
        let mut out_shape = params.shape().dims().to_vec();
        out_shape.remove(axis);
        for &dim in indices.shape().dims().iter().rev() {
            out_shape.insert(axis, dim);
        }
        let output_len: usize = out_shape.iter().product();

        // Cast indices buffer to u32 if needed
        let indices_gpu_buffer_u32 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<i32>,
                &crate::gpu::buffer::GpuBuffer<u32>,
            >(indices_gpu_buffer)
        };

        let result_buffer = crate::gpu::ops::execute_gather(
            params_gpu_buffer,
            indices_gpu_buffer_u32,
            axis,
            params.shape().dims(),
            indices.shape().dims(),
            output_len,
        )?;

        let result_buffer_t = unsafe {
            std::mem::transmute::<
                crate::gpu::buffer::GpuBuffer<f32>,
                crate::gpu::buffer::GpuBuffer<T>,
            >(result_buffer)
        };

        Ok(Tensor::from_gpu_buffer(
            result_buffer_t,
            crate::Shape::from_slice(&out_shape),
        ))
    } else {
        Err(TensorError::unsupported_operation_simple(format!(
            "GPU gather only supports f32, got {}",
            std::any::type_name::<T>()
        )))
    }
}

#[cfg(feature = "gpu")]
fn gpu_scatter_dispatch<T>(
    tensor: &Tensor<T>,
    indices: &Tensor<i32>,
    updates: &Tensor<T>,
    axis: usize,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    let type_name = std::any::type_name::<T>();

    if type_name == "f32" {
        let tensor_gpu_buffer = match &tensor.storage {
            TensorStorage::Gpu(buf) => unsafe {
                std::mem::transmute::<
                    &crate::gpu::buffer::GpuBuffer<T>,
                    &crate::gpu::buffer::GpuBuffer<f32>,
                >(buf)
            },
            _ => {
                return Err(TensorError::device_error_simple(
                    "Expected GPU tensor ".to_string(),
                ))
            }
        };

        let indices_gpu_buffer = match &indices.storage {
            TensorStorage::Gpu(buf) => buf,
            _ => {
                return Err(TensorError::device_error_simple(
                    "Expected GPU tensor ".to_string(),
                ))
            }
        };

        let updates_gpu_buffer = match &updates.storage {
            TensorStorage::Gpu(buf) => unsafe {
                std::mem::transmute::<
                    &crate::gpu::buffer::GpuBuffer<T>,
                    &crate::gpu::buffer::GpuBuffer<f32>,
                >(buf)
            },
            _ => {
                return Err(TensorError::device_error_simple(
                    "Expected GPU tensor ".to_string(),
                ))
            }
        };

        // Cast indices buffer to u32 if needed
        let indices_gpu_buffer_u32 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<i32>,
                &crate::gpu::buffer::GpuBuffer<u32>,
            >(indices_gpu_buffer)
        };

        let result_buffer = crate::gpu::ops::execute_scatter(
            tensor_gpu_buffer,
            indices_gpu_buffer_u32,
            updates_gpu_buffer,
            axis,
            tensor.shape().dims(),
            indices.shape().dims(),
            updates.shape().dims(),
        )?;

        let result_buffer_t = unsafe {
            std::mem::transmute::<
                crate::gpu::buffer::GpuBuffer<f32>,
                crate::gpu::buffer::GpuBuffer<T>,
            >(result_buffer)
        };

        Ok(Tensor::from_gpu_buffer(
            result_buffer_t,
            tensor.shape().clone(),
        ))
    } else {
        Err(TensorError::unsupported_operation_simple(format!(
            "GPU scatter only supports f32, got {}",
            std::any::type_name::<T>()
        )))
    }
}

#[cfg(feature = "gpu")]
fn gpu_where_dispatch<T>(
    condition: &Tensor<bool>,
    x: &Tensor<T>,
    y: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static + bytemuck::Pod + bytemuck::Zeroable,
{
    let type_name = std::any::type_name::<T>();

    if type_name == "f32" {
        let condition_gpu_buffer = match &condition.storage {
            TensorStorage::Gpu(buf) => buf,
            _ => {
                return Err(TensorError::device_error_simple(
                    "Expected GPU tensor ".to_string(),
                ))
            }
        };

        let x_gpu_buffer = match &x.storage {
            TensorStorage::Gpu(buf) => unsafe {
                std::mem::transmute::<
                    &crate::gpu::buffer::GpuBuffer<T>,
                    &crate::gpu::buffer::GpuBuffer<f32>,
                >(buf)
            },
            _ => {
                return Err(TensorError::device_error_simple(
                    "Expected GPU tensor ".to_string(),
                ))
            }
        };

        let y_gpu_buffer = match &y.storage {
            TensorStorage::Gpu(buf) => unsafe {
                std::mem::transmute::<
                    &crate::gpu::buffer::GpuBuffer<T>,
                    &crate::gpu::buffer::GpuBuffer<f32>,
                >(buf)
            },
            _ => {
                return Err(TensorError::device_error_simple(
                    "Expected GPU tensor ".to_string(),
                ))
            }
        };

        // Calculate broadcast shape
        let xy_broadcast_shape = x.shape().broadcast_shape(y.shape()).ok_or_else(|| {
            TensorError::invalid_argument(format!(
                "Cannot broadcast shapes {} and {} for where operation ",
                x.shape(),
                y.shape()
            ))
        })?;

        let broadcast_shape = condition
            .shape()
            .broadcast_shape(&xy_broadcast_shape)
            .ok_or_else(|| {
                TensorError::invalid_argument(format!(
                    "Condition shape {} cannot be broadcast to {xy_broadcast_shape}",
                    condition.shape()
                ))
            })?;

        // Cast condition buffer from bool to u32
        let condition_gpu_buffer_u32 = unsafe {
            std::mem::transmute::<
                &crate::gpu::buffer::GpuBuffer<bool>,
                &crate::gpu::buffer::GpuBuffer<u32>,
            >(condition_gpu_buffer)
        };

        let output_len = broadcast_shape.size();
        let result_buffer = crate::gpu::ops::execute_where(
            condition_gpu_buffer_u32,
            x_gpu_buffer,
            y_gpu_buffer,
            output_len,
        )?;

        let result_buffer_t = unsafe {
            std::mem::transmute::<
                crate::gpu::buffer::GpuBuffer<f32>,
                crate::gpu::buffer::GpuBuffer<T>,
            >(result_buffer)
        };

        Ok(Tensor::from_gpu_buffer(result_buffer_t, broadcast_shape))
    } else {
        Err(TensorError::unsupported_operation_simple(format!(
            "GPU where only supports f32, got {}",
            std::any::type_name::<T>()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::manipulation::transpose::transpose;

    /// Whole-row gather (embedding lookup): gathering N indices over a `[V, D]`
    /// table must yield the correct `[N, D]` rows. The OLD implementation only
    /// wrote N of the N*D outputs and read wrong offsets, returning garbage.
    #[test]
    fn test_gather_whole_rows_feature_width_gt_1() {
        // Table [3, 4] with distinct, recognisable rows.
        let table = Tensor::<f32>::from_vec(
            vec![
                10.0, 11.0, 12.0, 13.0, // row 0
                20.0, 21.0, 22.0, 23.0, // row 1
                30.0, 31.0, 32.0, 33.0, // row 2
            ],
            &[3, 4],
        )
        .expect("table creation should succeed");

        let indices =
            Tensor::<i32>::from_vec(vec![2, 0, 1], &[3]).expect("indices creation should succeed");

        let gathered = gather(&table, &indices, 0).expect("gather should succeed");

        assert_eq!(gathered.shape().dims(), &[3, 4]);
        let got = gathered
            .as_slice()
            .expect("gather output must be contiguous");
        let expected = [
            30.0, 31.0, 32.0, 33.0, // row 2
            10.0, 11.0, 12.0, 13.0, // row 0
            20.0, 21.0, 22.0, 23.0, // row 1
        ];
        assert_eq!(got, &expected);
    }

    /// Slicing a NON-contiguous (transposed) tensor must return the real
    /// elements. The OLD implementation relied on `as_slice()` (None for
    /// non-contiguous), silently producing zeros.
    #[test]
    fn test_slice_non_contiguous_transposed() {
        // Source [2, 3]:
        //   [[1, 2, 3],
        //    [4, 5, 6]]
        let src = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .expect("source creation should succeed");

        // Transpose -> [3, 2], non-contiguous:
        //   [[1, 4],
        //    [2, 5],
        //    [3, 6]]
        let transposed = transpose(&src).expect("transpose should succeed");
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        assert!(
            !transposed.is_contiguous(),
            "transposed tensor should be non-contiguous for this test to be meaningful"
        );

        // Slice rows 1..3, all columns -> [[2, 5], [3, 6]].
        let sliced = slice(&transposed, &[1..3, 0..2]).expect("slice should succeed");
        assert_eq!(sliced.shape().dims(), &[2, 2]);

        let got = sliced.as_slice().expect("slice output must be contiguous");
        assert_eq!(got, &[2.0, 5.0, 3.0, 6.0]);
    }

    /// A contiguous slice must keep working (no regression from the rewrite).
    #[test]
    fn test_slice_contiguous_subrange() {
        let src = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .expect("source creation should succeed");
        let sliced = slice(&src, &[0..2, 1..3]).expect("slice should succeed");
        assert_eq!(sliced.shape().dims(), &[2, 2]);
        let got = sliced.as_slice().expect("slice output must be contiguous");
        assert_eq!(got, &[2.0, 3.0, 5.0, 6.0]);
    }
}

// GPU-resident correctness test for the readback+delegate fix in
// `slice_with_stride()`'s GPU arm: a non-unit step used to hit a hard
// "not yet implemented" error; it must now delegate to the CPU
// implementation and return the real strided result. Skips gracefully
// (without failing the suite) if no GPU adapter is available.
#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use super::*;
    use crate::Device;

    #[test]
    fn gpu_slice_with_stride_non_unit_step_matches_cpu_reference() {
        let src = Tensor::<f32>::from_vec(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[10],
        )
        .expect("test: from_vec should succeed");

        let src_gpu = match src.to(Device::Gpu(0)) {
            Ok(t) => t,
            Err(_) => return, // No GPU adapter available in this environment; skip.
        };

        // Step of 2 over the whole range: expect [0, 2, 4, 6, 8].
        let params = vec![SliceParams::with_step(Some(0), Some(10), Some(2))];
        let result = slice_with_stride(&src_gpu, &params)
            .expect("test: gpu slice_with_stride should succeed with a real adapter");
        assert_eq!(result.shape().dims(), &[5]);
        let data = result.to_vec().expect("test: to_vec should succeed");
        assert_eq!(data, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    }
}
