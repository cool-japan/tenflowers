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
use num_traits::Zero;
use scirs2_autograd::ndarray::{ArrayD, IxDyn};

// Import common helper functions
use super::common::{broadcast_indices, calculate_strides, coords_to_flat, flat_to_coords};

/// Slice a tensor along specified ranges
pub fn slice<T>(tensor: &Tensor<T>, ranges: &[std::ops::Range<usize>]) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
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
            // Use ndarray's select method to extract the slice
            let out_shape: Vec<usize> = ranges.iter().map(|r| r.end - r.start).collect();

            // Create the indices for the slice
            let mut result = ArrayD::<T>::zeros(IxDyn(&out_shape));

            // Copy the sliced region
            // Note: This is not the most efficient approach but works without unsafe
            if let Some(result_slice) = result.as_slice_mut() {
                let mut idx = 0;
                let strides = tensor
                    .shape()
                    .dims()
                    .iter()
                    .rev()
                    .scan(1, |acc, &dim| {
                        let stride = *acc;
                        *acc *= dim;
                        Some(stride)
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>();

                fn copy_recursive<T: Clone>(
                    src: &ArrayD<T>,
                    dst: &mut [T],
                    ranges: &[std::ops::Range<usize>],
                    strides: &[usize],
                    current_idx: &mut usize,
                    depth: usize,
                    src_indices: &mut Vec<usize>,
                ) {
                    if depth == ranges.len() {
                        let linear_idx: usize = src_indices
                            .iter()
                            .zip(strides)
                            .map(|(idx, stride)| idx * stride)
                            .sum();
                        if let Some(val) = src.as_slice().and_then(|s| s.get(linear_idx)) {
                            dst[*current_idx] = val.clone();
                            *current_idx += 1;
                        }
                        return;
                    }

                    for i in ranges[depth].clone() {
                        src_indices.push(i);
                        copy_recursive(
                            src,
                            dst,
                            ranges,
                            strides,
                            current_idx,
                            depth + 1,
                            src_indices,
                        );
                        src_indices.pop();
                    }
                }

                let mut src_indices = Vec::new();
                copy_recursive(
                    array,
                    result_slice,
                    ranges,
                    &strides,
                    &mut idx,
                    0,
                    &mut src_indices,
                );
            }

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
    T: Clone + Default + Zero + Send + Sync + 'static,
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
            // For GPU tensors, we currently fall back to the regular slice
            // A future implementation could create GPU strided kernels
            // For now, convert slice params to ranges if possible
            let ranges: Result<Vec<_>> = slice_params
                .iter()
                .enumerate()
                .map(|(i, param)| {
                    let size = shape.dims()[i];
                    let (start, end, step) = param.normalize(size)?;
                    if step == 1 {
                        Ok(start..end)
                    } else {
                        Err(TensorError::invalid_argument(
                            "GPU strided slicing not yet implemented for non-unit steps"
                                .to_string(),
                        ))
                    }
                })
                .collect();

            match ranges {
                Ok(ranges) => gpu_slice_dispatch(gpu_buffer, tensor.shape().dims(), &ranges),
                Err(e) => Err(e),
            }
        }
    }
}

/// Gather operation - gather slices from params according to indices
pub fn gather<T>(params: &Tensor<T>, indices: &Tensor<i32>, axis: usize) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
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
        (TensorStorage::Cpu(_params_arr), TensorStorage::Cpu(indices_arr)) => {
            let mut _result = ArrayD::<T>::zeros(IxDyn(&out_shape));

            // For simplicity, we'll implement the basic gather operation
            // In a full implementation, this would be optimized
            if indices_shape.dims().is_empty() {
                // Scalar index
                if let Some(&idx) = indices_arr.as_slice().unwrap().first() {
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

            // Handle multi-dimensional indices
            let indices_slice = indices_arr.as_slice().ok_or_else(|| {
                TensorError::invalid_argument("Indices tensor is not contiguous ".to_string())
            })?;

            // Create output tensor
            let mut result = ArrayD::<T>::zeros(IxDyn(&out_shape));
            let result_slice = result.as_slice_mut().ok_or_else(|| {
                TensorError::invalid_argument("Result tensor is not contiguous ".to_string())
            })?;

            // Calculate strides for params and result tensors
            let params_strides = calculate_strides(params_shape.dims());
            let _result_strides = calculate_strides(&out_shape);

            // Gather elements
            for (result_idx, &index) in indices_slice.iter().enumerate() {
                if index < 0 || index as usize >= params_shape.dims()[axis] {
                    return Err(TensorError::invalid_argument(format!(
                        "Index {index} out of bounds for axis {axis} of size {}",
                        params_shape.dims()[axis]
                    )));
                }

                // Convert flat result index to multi-dimensional coordinates
                let result_coords = flat_to_coords(result_idx, &out_shape);

                // Build corresponding coordinates in params tensor
                let mut params_coords = result_coords.clone();

                // Insert the gathered index at the axis position
                if axis < params_coords.len() {
                    params_coords[axis] = index as usize;
                } else {
                    params_coords.insert(axis, index as usize);
                }

                // Get the linear index in params tensor
                let params_linear_idx = coords_to_flat(&params_coords, &params_strides);

                // Copy the value
                if let Some(params_slice) = params.as_slice() {
                    if params_linear_idx < params_slice.len() {
                        result_slice[result_idx] = params_slice[params_linear_idx].clone();
                    }
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
    T: Clone + Default + Zero + Send + Sync + 'static,
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
    T: Clone + Default + Zero + Send + Sync + 'static,
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
    T: Clone + Default + Zero + Send + Sync + 'static,
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
    T: Clone + Default + Zero + Send + Sync + 'static,
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
    T: Clone + Default + Zero + Send + Sync + 'static,
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
    T: Clone + Default + Zero + Send + Sync + 'static,
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
    T: Clone + Default + Zero + Send + Sync + 'static,
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
