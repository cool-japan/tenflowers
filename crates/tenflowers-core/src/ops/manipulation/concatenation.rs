//! Concatenation and related tensor manipulation operations
//!
//! This module contains functions for concatenating, stacking, splitting, tiling,
//! and repeating tensors along various dimensions.

#[cfg(feature = "gpu")]
use crate::gpu::buffer::GpuBuffer;
use crate::tensor::TensorStorage;
use crate::{Result, Tensor, TensorError};
use num_traits::Zero;
use scirs2_autograd::ndarray::{ArrayD, IxDyn};

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

/// Concatenate tensors along a specified axis
pub fn concat<T>(tensors: &[&Tensor<T>], axis: usize) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    if tensors.is_empty() {
        return Err(TensorError::invalid_argument(
            "Cannot concatenate empty tensor list ".to_string(),
        ));
    }

    // Check device compatibility
    let device = tensors[0].device();
    for tensor in &tensors[1..] {
        if tensor.device() != device {
            return Err(TensorError::device_mismatch(
                "concat",
                &device.to_string(),
                &tensor.device().to_string(),
            ));
        }
    }

    // Infer result shape
    let shapes: Vec<_> = tensors.iter().map(|t| t.shape()).collect();
    let _result_shape = crate::ops::shape_inference::infer_concat(&shapes, axis as i32)?;

    // Perform concatenation based on device
    match &tensors[0].storage {
        TensorStorage::Cpu(_) => {
            // Extract CPU arrays
            let arrays: Result<Vec<_>> = tensors
                .iter()
                .map(|t| match &t.storage {
                    TensorStorage::Cpu(arr) => Ok(arr.view()),
                    #[cfg(feature = "gpu")]
                    _ => unreachable!("Device mismatch should have been caught "),
                })
                .collect();
            let arrays = arrays?;

            // Use ndarray's concatenate
            let concatenated = scirs2_autograd::ndarray::concatenate(
                scirs2_autograd::ndarray::Axis(axis),
                &arrays,
            )
            .map_err(|e| TensorError::invalid_argument(format!("Concatenation failed: {e}")))?;

            Ok(Tensor::from_array(concatenated))
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(gpu_a) => {
            // Check if we're concatenating exactly two tensors and have proper GPU support
            if tensors.len() != 2 {
                return Err(TensorError::unsupported_operation_simple(
                    "GPU concatenate currently only supports exactly 2 tensors".to_string(),
                ));
            }

            // Check that the second tensor is also on GPU
            let gpu_b = match &tensors[1].storage {
                #[cfg(feature = "gpu")]
                TensorStorage::Gpu(gpu_b) => gpu_b,
                _ => {
                    return Err(TensorError::device_error_simple(
                        "All tensors must be on the same device for GPU concatenation".to_string(),
                    ))
                }
            };

            // Check type compatibility
            let type_name = std::any::type_name::<T>();
            match type_name {
                "f32" | "f64" | "i32" | "i64" => {
                    // TODO: Implement GPU concatenation
                    // For now, fallback to CPU implementation
                    Err(TensorError::unsupported_operation_simple(
                        "GPU concatenation not yet implemented".to_string()
                    ))
                },
                _ => {
                    Err(TensorError::unsupported_operation_simple(
                        format!("GPU concatenate not implemented for type {}. Supported types: f32, f64, i32, i64", type_name)
                    ))
                }
            }
        }
    }
}

/// Add a dimension of size 1 at the specified axis
pub fn expand_dims<T>(tensor: &Tensor<T>, axis: usize) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    let mut new_shape = tensor.shape().dims().to_vec();

    if axis > new_shape.len() {
        return Err(TensorError::invalid_argument(format!(
            "Axis {axis} out of range for tensor of rank {}",
            tensor.shape().rank()
        )));
    }

    new_shape.insert(axis, 1);

    match &tensor.storage {
        TensorStorage::Cpu(array) => {
            let expanded = array
                .clone()
                .into_shape_with_order(IxDyn(&new_shape))
                .map_err(|e| TensorError::invalid_argument(format!("Expand dims failed: {e}")))?;
            Ok(Tensor::from_array(expanded))
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => {
            // GPU operations require T: Pod + Zeroable which isn't guaranteed for generic T
            Err(TensorError::unsupported_operation_simple(
                "GPU expand_dims not implemented for this type. Only f32 is currently supported."
                    .to_string(),
            ))
        }
    }
}

/// Stack tensors along a new axis
pub fn stack<T>(tensors: &[&Tensor<T>], axis: usize) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    if tensors.is_empty() {
        return Err(TensorError::invalid_argument(
            "Cannot stack empty tensor list ".to_string(),
        ));
    }

    // All tensors must have the same shape
    let shape = tensors[0].shape();
    for tensor in &tensors[1..] {
        if tensor.shape() != shape {
            return Err(TensorError::invalid_argument(format!(
                "All tensors must have the same shape for stacking. Got {} and {}",
                shape,
                tensor.shape()
            )));
        }
    }

    // Check axis validity
    if axis > shape.rank() {
        return Err(TensorError::invalid_argument(format!(
            "Axis {axis} out of range for stacking tensors of rank {}",
            shape.rank()
        )));
    }

    // Expand dimensions and concatenate
    let expanded: Result<Vec<_>> = tensors.iter().map(|t| expand_dims(t, axis)).collect();
    let expanded = expanded?;
    let expanded_refs: Vec<_> = expanded.iter().collect();

    concat(&expanded_refs, axis)
}

/// Split a tensor into multiple tensors along an axis
pub fn split<T>(tensor: &Tensor<T>, num_splits: usize, axis: usize) -> Result<Vec<Tensor<T>>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    let shape = tensor.shape();

    if axis >= shape.rank() {
        return Err(TensorError::invalid_argument(format!(
            "Axis {axis} out of range for tensor of rank {}",
            shape.rank()
        )));
    }

    let axis_size = shape.dims()[axis];
    if axis_size % num_splits != 0 {
        return Err(TensorError::invalid_argument(format!(
            "Axis size {axis_size} is not divisible by num_splits {num_splits}"
        )));
    }

    let split_size = axis_size / num_splits;
    let mut results = Vec::with_capacity(num_splits);

    for i in 0..num_splits {
        let mut ranges: Vec<_> = (0..shape.rank()).map(|j| 0..shape.dims()[j]).collect();
        ranges[axis] = (i * split_size)..((i + 1) * split_size);

        results.push(slice(tensor, &ranges)?);
    }

    Ok(results)
}

/// Tile operation - construct a tensor by repeating the input multiple times
pub fn tile<T>(tensor: &Tensor<T>, multiples: &[usize]) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    if multiples.len() != tensor.shape().rank() {
        return Err(TensorError::invalid_argument(format!(
            "Multiples length {} does not match tensor rank {}",
            multiples.len(),
            tensor.shape().rank()
        )));
    }

    // Calculate output shape
    let out_shape: Vec<_> = tensor
        .shape()
        .dims()
        .iter()
        .zip(multiples)
        .map(|(&dim, &mult)| dim * mult)
        .collect();

    match &tensor.storage {
        TensorStorage::Cpu(array) => {
            // Create output array
            let mut result = ArrayD::<T>::zeros(IxDyn(&out_shape));

            // Get the input shape
            let in_shape = tensor.shape().dims();

            // Iterate through all positions in the output array
            let mut out_indices = vec![0; out_shape.len()];

            loop {
                // Calculate corresponding input indices
                let in_indices: Vec<_> = out_indices
                    .iter()
                    .zip(in_shape)
                    .map(|(&out_idx, &in_dim)| out_idx % in_dim)
                    .collect();

                // Copy the value
                result[IxDyn(&out_indices)] = array[IxDyn(&in_indices)].clone();

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
        TensorStorage::Gpu(_) => gpu_tile_dispatch(tensor, multiples),
    }
}

/// Repeat operation - repeat elements of a tensor
pub fn repeat<T>(tensor: &Tensor<T>, repeats: usize, axis: Option<usize>) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    if let Some(axis) = axis {
        if axis >= tensor.shape().rank() {
            return Err(TensorError::invalid_argument(format!(
                "Axis {} out of range for tensor of rank {}",
                axis,
                tensor.shape().rank()
            )));
        }
    }

    match &tensor.storage {
        TensorStorage::Cpu(array) => {
            if let Some(axis) = axis {
                // Repeat along a specific axis
                let mut out_shape = tensor.shape().dims().to_vec();
                out_shape[axis] *= repeats;

                let mut result = ArrayD::<T>::zeros(IxDyn(&out_shape));
                let _in_shape = tensor.shape().dims();

                // Iterate through all positions in the output
                let mut out_indices = vec![0; out_shape.len()];

                loop {
                    // Calculate corresponding input indices
                    let in_indices: Vec<_> = out_indices
                        .iter()
                        .enumerate()
                        .map(|(i, &out_idx)| {
                            if i == axis {
                                out_idx / repeats
                            } else {
                                out_idx
                            }
                        })
                        .collect();

                    // Copy the value
                    result[IxDyn(&out_indices)] = array[IxDyn(&in_indices)].clone();

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
            } else {
                // Repeat the entire tensor (flatten, repeat, reshape)
                let flat = array
                    .view()
                    .into_shape_with_order((array.len(),))
                    .map_err(|e| {
                        TensorError::invalid_argument(format!("Failed to flatten: {e}"))
                    })?;

                let mut repeated = Vec::with_capacity(flat.len() * repeats);
                for elem in flat.iter() {
                    for _ in 0..repeats {
                        repeated.push(elem.clone());
                    }
                }

                let result =
                    ArrayD::from_shape_vec(IxDyn(&[repeated.len()]), repeated).map_err(|e| {
                        TensorError::invalid_argument(format!(
                            "Failed to create repeated array: {e}"
                        ))
                    })?;

                Ok(Tensor::from_array(result))
            }
        }
        #[cfg(feature = "gpu")]
        TensorStorage::Gpu(_) => gpu_repeat_dispatch(tensor, repeats, axis),
    }
}

// GPU dispatch functions

/// GPU slice dispatch for supported types
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

/// GPU tile dispatch for supported types
#[cfg(feature = "gpu")]
fn gpu_tile_dispatch<T>(tensor: &Tensor<T>, multiples: &[usize]) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    let type_name = std::any::type_name::<T>();

    if type_name == "f32" {
        let gpu_buffer = match &tensor.storage {
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

        // Calculate output shape and length
        let out_shape: Vec<_> = tensor
            .shape()
            .dims()
            .iter()
            .zip(multiples)
            .map(|(&dim, &mult)| dim * mult)
            .collect();
        let output_len: usize = out_shape.iter().product();

        let result_buffer = crate::gpu::ops::execute_tile(
            gpu_buffer,
            multiples,
            tensor.shape().dims(),
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
            "GPU tile only supports f32, got {}",
            std::any::type_name::<T>()
        )))
    }
}

/// GPU repeat dispatch for supported types
#[cfg(feature = "gpu")]
fn gpu_repeat_dispatch<T>(
    tensor: &Tensor<T>,
    repeats: usize,
    axis: Option<usize>,
) -> Result<Tensor<T>>
where
    T: Clone + Default + Zero + Send + Sync + 'static,
{
    let type_name = std::any::type_name::<T>();

    if type_name == "f32" {
        let gpu_buffer = match &tensor.storage {
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

        // Calculate output shape and length
        let out_shape = if let Some(axis) = axis {
            let mut shape = tensor.shape().dims().to_vec();
            shape[axis] *= repeats;
            shape
        } else {
            vec![tensor.shape().size() * repeats]
        };
        let output_len: usize = out_shape.iter().product();

        let result_buffer = crate::gpu::ops::execute_repeat(
            gpu_buffer,
            &[repeats], // Convert single repeat value to slice
            axis,
            tensor.shape().dims(),
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
            "GPU repeat only supports f32, got {}",
            std::any::type_name::<T>()
        )))
    }
}
